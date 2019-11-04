"""
Functions related to information computed prior to model deployment
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


def quantize(detail, data):
    shape = detail['shape']
    dtype = detail['dtype']
    a, b = detail['quantization']

    return (data / a + b).astype(dtype).reshape(shape)


def dequantize(detail, data):
    a, b = detail['quantization']

    return (data - b) * a


def change_to_float(int_val, num_int_bits):
    n = 32 - num_int_bits - 1
    return int_val * (2 ** -n)


def float_to_q(float_val, num_int_bits):
    n = 32 - num_int_bits - 1
    return np.int32(round(float_val * (2 ** n)))
