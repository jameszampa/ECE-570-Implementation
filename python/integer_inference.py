"""
Code for integer only arithmetic inferencing for quantization aware trained model,
limited to Dense, Conv1D layers

Original Source Code:
https://github.com/google/gemmlowp/blob/master/fixedpoint/fixedpoint.h
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/common.h
"""


import numpy as np


# Returns the integer that represents the product of two fixed-point
# numbers, interpreting all integers as fixed-point values in the
# interval [-1, 1), rounding to the nearest value, and saturating
# -1 * -1 to the maximum value (since 1 is not in the half-open
# interval [-1, 1)).
#
# [The explanation below specializes to std::int32_t for example purpose.]
#
# The mapping between IntegerType and the interval [-1, 1) is unique and
# implied by IntegerType, which is assumed to be signed. For example,
# for IntegerType==std::int32_t, the mapping is
# real_value = integer_value / 2^31.

# So in this case, and leaving aside rounding and saturating, this
# function computes ((a / 2^31) * (b / 2^31)) * 2^31, which simplifies to
# (a * b) / 2^31.
#
# The 'doubling' part in the name of this function comes from the fact that
# this operation is very close to a "multiply-high" operation, keeping only
# the top half bits, except that that would be effectively computing
# (a * b) / 2^32, so here we are computing 2x that, since
# 1/2^31 = 2 * 1/2^32. The idea is to use all of the available 32 bits
# in the destination int32 value.
# Good to write C code for? YES
def SaturatingRoundingDoublingHighMul(a, b):
    overflow = (a == b) & (a == -2147483648)
    a_64 = np.int64(a)
    b_64 = np.int64(b)

    ab_64 = a_64 * b_64
    if ab_64 >= 0:
        nudge = np.int32(1 << 30)
    else:
        nudge = np.int32(1 - (1 << 30))
    ab_x2_high32 = np.int32((ab_64 + nudge) / (np.int64(1 << 31)))

    if overflow:
        return np.int32(2147483647)
    else:
        return ab_x2_high32


# Good to write C code for? YES
def MultiplyByQuantizedMultiplierGreaterThanOne(x, quant_mul, left_shift):
    return SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quant_mul)


# Good to write C code for? YES
def MultiplyByQuantizedMultiplierSmallerThanOne(x, quant_mul, right_shift):
    return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x, quant_mul), right_shift)


# Correctly-rounded-to-nearest division by a power-of-two.
# Also known as a rounding arithmetic right shift.
# Good to write C code for? YES
def RoundingDivideByPOT(x, exponent):
    if (exponent < 0) | (exponent > 31):
        raise ValueError('Inputs incorrect')
    mask = np.int32((1 << exponent) - 1)
    zero = np.int32(0)
    one = np.int32(1)
    remainder = x & mask
    if x < zero:
        maskiflessthan = x & zero
    else:
        maskiflessthan = x

    threshold = (mask >> 1) + (maskiflessthan & one)

    if remainder > threshold:
        maskifgreaterthan = remainder & threshold
    else:
        maskifgreaterthan = remainder

    return (x >> exponent) + (maskifgreaterthan & one)


# Integer math for FullyConnected or Dense layer
# Good to write C code for? YES
def FullyConnected(quantized_inputs, input_offset, quantized_weights, weight_offset, quantized_bias,
                   output_offset, M_0, right_shift, output_shape, num_skip_dyn, num_skip_static, total_exc):
    output_full_conn_arr = np.zeros(shape=output_shape, dtype=np.uint8)
    rows, cols = quantized_weights.shape
    for i in range(rows):
        acc = np.int32(0)
        for j in range(cols):
            input_val = np.int32(quantized_inputs[0][j])
            weight_val = np.int32(quantized_weights[i][j])
            if input_val - input_offset == 0:
                num_skip_dyn += 1
            if weight_val - weight_offset == 0:
                num_skip_static += 1
            total_exc += 1
            acc += (input_val - input_offset) * (weight_val - weight_offset)
        acc += quantized_bias[i]
        acc = MultiplyByQuantizedMultiplierSmallerThanOne(acc, M_0, right_shift)
        acc += output_offset  # activation offset
        acc = np.max([acc, np.int32(0)])
        acc = np.min([acc, np.int32(255)])
        output_full_conn_arr[0][i] = np.uint8(acc)
    return output_full_conn_arr, num_skip_dyn, num_skip_static, total_exc


# Integer math for Conv1D
# Good to write C code for? YES
def Conv(quantized_inputs, input_offset, quantized_weights, weight_offset, quantized_bias,
                   output_offset, M_0, right_shift, output_shape, num_skip_dyn, num_skip_static, total_exc):
    output_conv_arr = np.zeros(shape=output_shape, dtype=np.uint8)
    kernel_shape = quantized_weights.shape[2]
    rows = quantized_weights.shape[3]
    cols = quantized_inputs.shape[1]
    for i in range(rows):
        acc = np.int32(0)
        for j in range(cols):
            if j + 1 <= cols - 1:
                for k in range(kernel_shape):
                    input_val = np.int32(quantized_inputs[0, j - 1 + k, 0])
                    weight_val = np.int32(quantized_weights[0, 0, k, i])
                    if input_val - input_offset == 0:
                        num_skip_dyn += 1
                    if weight_val - weight_offset == 0:
                        num_skip_static += 1
                    total_exc += 1
                    acc += (input_val - input_offset) * (weight_val - weight_offset)
            acc += quantized_bias[i]
            acc = MultiplyByQuantizedMultiplierSmallerThanOne(acc, M_0, right_shift)
            acc += output_offset  # activation offset
            acc = np.max([acc, np.int32(0)])
            acc = np.min([acc, np.int32(255)])
            output_conv_arr[j][i] = np.uint8(acc)
    return output_conv_arr, num_skip_dyn, num_skip_static, total_exc


# Returns the product of a run-time integer value by a compile-time power
# of two, with either a positive exponent (equivalent to an arithmetic
# left shift, saturating) or a negative exponent (equivalent to an arithmetic
# right shift, rounding to nearest).
# Good to write C code for? NO
def SaturatingRoundingMultiplyByPOT(x, exponent):
    return RoundingDivideByPOT(x, -exponent)


# I don't know if we will need this
# Good to write C code for? NO
def SaturatingAdd(a, b):
    a32 = np.int32(a)
    b32 = np.int32(b)
    sum = np.int64(a32 + b32)
    sum = np.max([-32768, sum])
    sum = np.min([32767, sum])
    return np.int16(sum)


# Good to write C code for? NO
def exp_on_interval_between_negative_one_quarter_and_0_excl(x, num_int_bits):
    if x < float_to_q(-0.25, 5):
        raise ValueError('Must be > -0.25')
    elif x > 0:
        raise ValueError('Must be < 0')
    mask = (1 << (32 - num_int_bits - 1)) - 1
    mask_shift = np.int32((x & mask) << num_int_bits)
    rescaled_x = np.int32(mask_shift | (1 << 31))
    scaled_x = np.int32(rescaled_x + (1 << 28))
    x2 = SaturatingRoundingDoublingHighMul(scaled_x, scaled_x)
    x3 = SaturatingRoundingDoublingHighMul(x2, scaled_x)
    x4 = SaturatingRoundingDoublingHighMul(x2, x2)
    x4_over_4 = SaturatingRoundingMultiplyByPOT(x4, -2)
    constant_1_over_3 = float_to_q(1/3, 0)
    constant_term = float_to_q(np.exp(-1.0 / 8.0), 0)
    x4_over_24_plus_x3_over_6_plus_x2_over_2 = SaturatingRoundingMultiplyByPOT(SaturatingRoundingDoublingHighMul(x4_over_4 + x3, constant_1_over_3) + x2, -1)
    x4_over_24_plus_x3_over_6_plus_x2_over_2_mul_constant_term = SaturatingRoundingDoublingHighMul(constant_term, x4_over_24_plus_x3_over_6_plus_x2_over_2)
    result = SaturatingAdd(constant_term, x4_over_24_plus_x3_over_6_plus_x2_over_2_mul_constant_term)

    if result < 0:
        raise ValueError('Negative Output')
    return result


# Good to write C code for? NO
def CountLeadingZeros(x):
    count = 0
    for i in range(32):
        if (x >> (31 - i)) == 0:  # ignore sign? i think so
            count += 1
        else:
            break
    return count


# Returns (a+b)/2, rounded to the nearest integer.
# Equivalent to VRHADD in the ARM NEON instruction set.
# Good to write C code for? NO
def RoundingHalfSum(a, b):
    a64 = np.int64(a)
    b64 = np.int64(b)
    sum = a64 + b64
    if sum >= 0:
        sign = 1
    else:
        sign = -1
    return np.int32((sum + sign) / 2)


# returns 1 / (1 + x) for x (0, 1)
# Good to write C code for? NO
def one_over_one_plus_x_for_x_in_0_1(a, num_int_bits):
    if (change_to_float(a, num_int_bits) > 1) | (change_to_float(a, num_int_bits) < 0):
        raise ValueError('input not between 0 and 1')
    half_denominator = RoundingHalfSum(a, 2147483647)
    constant_48_over_17 = float_to_q(48.0/17.0, 2)
    constant_neg_32_over_17 = float_to_q(-32.0/17.0, 2)
    constant_one = float_to_q(1.0, 2)
    half_denominator_mul_constant_neg_32_over_17 = SaturatingRoundingDoublingHighMul(half_denominator, constant_neg_32_over_17)
    x = constant_48_over_17 + half_denominator_mul_constant_neg_32_over_17
    for i in range(3):
        half_denominator_times_x = SaturatingRoundingDoublingHighMul(half_denominator, x)
        one_minus_half_denominator_times_x = constant_one - half_denominator_times_x
        x = x + SaturatingRoundingDoublingHighMul(x, one_minus_half_denominator_times_x)
    return np.int32(x << 1)


# Good to write C code for? NO
def exp_on_negative_values(a, num_int_bits):
    # change to 0 bit int rep
    mask = (1 << 24) - 1
    one_quarter = 1 << 24
    a_mod_quarter_minus_one_quarter = (input_diff_rescaled & mask) - one_quarter
    result = exp_on_interval_between_negative_one_quarter_and_0_excl(a_mod_quarter_minus_one_quarter, 5)
    remainder = a_mod_quarter_minus_one_quarter - a
    result = SelectUsingMask(MaskIfZero(a), (1 << 31) - 1, result)
    return result
    # constant_one = 2147483647
    # constant_half = float_to_q(0.5, num_int_bits)
    # constant_1_over_6 = float_to_q(1 / 6, num_int_bits)
    # constant_1_over_24 = float_to_q(1 / 24, num_int_bits)
    # one_plus_x = constant_one + x
    # x2 = FixedPointMul(x, x, num_int_bits)
    # x2_over_2 = FixedPointMul(x2, constant_half, num_int_bits)
    # x3 = FixedPointMul(x2, x, num_int_bits)
    # x3_over_6 = FixedPointMul(x3, constant_1_over_6, num_int_bits)
    # x4 = FixedPointMul(x2, x2, num_int_bits)
    # x4_over_24 = FixedPointMul(x4, constant_1_over_24, num_int_bits)
    # return one_plus_x + x2_over_2 + x3_over_6 + x4_over_24


# For each input scalar, the corresponding bits of the result are set if the
# input scalar is non-zero.
# Good to write C code for? NO
def MaskIfNonZero(a):
    if a:
        return ~np.int32(0)
    else:
        return np.int32(0)


# For each input scalar, the corresponding bits of the result are set if the
# input scalar is zero.
# Good to write C code for? NO
def MaskIfZero(a):
    return MaskIfNonZero(~a)


# Each bit of the result is set to the corresponding bit of either then_val or
# else_val depending on whether the corresponding bit of if_mask is set.
# Equivalent to the VBSL instruction in ARM NEON.
# Good to write C code for? NO
def SelectUsingMask(if_mask, then_val, else_val):
    return (if_mask & then_val) ^ ((~if_mask) & else_val)
