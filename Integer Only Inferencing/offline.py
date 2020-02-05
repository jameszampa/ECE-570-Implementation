"""
Functions related to information computed prior to model deployment.
This file was mainly used for internal testing to verify my understanding of the quantization methods
"""
import numpy as np


def quantize_mult_smaller_one(real_mul):
    s = 0
    while real_mul < 0.5:
        real_mul *= 2
        s += 1
    q = np.int64(round(real_mul * (1 << 31)))
    if q == (1 << 31):
        q /= 2
        s -= 1
    return s, np.int32(q)


def reverse_M_0(real_mul, right_shift):
    while right_shift > 0:
        real_mul /= 2
        right_shift -= 1
    return real_mul


def quant_mult_16_bit(real_mul):
    s = 0
    while real_mul < 0.5:
        real_mul *= 2
        s += 1
    q = np.int32(round(real_mul * (1 << 15)))
    if q == (1 << 15):
        q /= 2
        s -= 1
    return s, np.int16(q)


def quantize_mult_greater_one(real_mul):
    s = 0
    while real_mul >= 1.0:
        real_mul /= 2
        s += 1
    q = np.int64(round(real_mul * (1 << 31)))
    if q == (1 << 31):
        q /= 2
        s += 1
    return s, np.int32(q)


def change_to_float(int_val, num_int_bits):
    n = 32 - num_int_bits - 1
    return int_val * (2 ** -n)


def float_to_q(float_val, num_int_bits):
    n = 32 - num_int_bits - 1
    return np.int32(round(float_val * (2 ** n)))


# Change original int32 M_0 to a int16 M_0
def change_32_M_0_to_16_M_0(og_M_0, right_shift):
    M = og_M_0 * (2 ** -31)
    og_S = reverse_M_0(M, right_shift)
    return quant_mult_16_bit(og_S)