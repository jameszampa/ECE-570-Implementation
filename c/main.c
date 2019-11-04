/* Copyright 2019 SiFive, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#include "common.h"
#include "conv_params.h"
#include "img.h"
#include "dense_params.h"
#include "pred_params.h"

void Conv(uint8_t output_conv_arr[imgSize][nFilters]);
void flattenConv(uint8_t arrIN[imgSize][nFilters], uint8_t arrOUT[1][imgSize*nFilters]);
int32_t maxOf(int32_t a, int32_t b);
int32_t minOf(int32_t a, int32_t b);
int32_t RoundingDivideByPOT(int32_t x, int exponent);
int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b);
int32_t MultiplyByQuantizedMultiplierGreaterThanOne(int32_t x, int32_t quant_mul, int left_shift);
int32_t MultiplyByQuantizedMultiplierSmallerThanOne(int32_t x, int32_t quant_mul, int right_shift);
void FullyConnectedDense(uint8_t quantized_inputs[1][imgSize*nFilters], uint8_t output_full_conn_dense_arr[1][n_dense_nodes]);
void FullyConnectedPred(uint8_t quantized_inputs[1][n_dense_nodes], uint8_t output_full_conn_pred_arr[1][n_pred_nodes]);
uint8_t argMaxPred(uint8_t quantized_input[1][n_pred_nodes]);

int32_t maxOf(int32_t a, int32_t b) {
	if (a > b) return a;
	else return b;
}

int32_t minOf(int32_t a, int32_t b) {
	if (a < b) return a;
	else return b;
}

int32_t RoundingDivideByPOT(int32_t x, int exponent) { //DONE
	if ((exponent < 0) | (exponent > 31)) perror("RoundingDivideByPOT: Inputs Incorrect\n"); //CHECK
	int32_t mask = (int32_t)((1 << exponent) - 1);
	int32_t zero = (int32_t)(0);
	int32_t one = (int32_t)(1);
	int32_t remainder = x & mask;

	int32_t maskiflessthan = x;
	if (x < zero) maskiflessthan &= zero;

	int32_t threshold = (mask >> 1) + (maskiflessthan & one);
	int32_t maskifgreaterthan = remainder;
	if (remainder > threshold) maskifgreaterthan &= threshold;

	return (x >> exponent) + (maskifgreaterthan & one);
}

int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) { //DONE
	bool overflow = (a == b) & (a == 0-2147483648);
	if (overflow) return (int32_t)(2147483647);

	int64_t a_64 = (int64_t) a;
	int64_t b_64 = (int64_t) b;
	int64_t ab_64 = a_64 * b_64;
	int64_t nudge;

	if (ab_64 >= 0) nudge = (int32_t)(1 << 30);
	else nudge = (int32_t)(1 - (1 << 30));

	int32_t ab_x2_high32 = (int32_t)((ab_64 + nudge) / (int64_t)((int64_t)(1) << 31));
	return ab_x2_high32;
}

int32_t MultiplyByQuantizedMultiplierGreaterThanOne(int32_t x, int32_t quant_mul, int left_shift) { //DONE
	return SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quant_mul);
}

int32_t MultiplyByQuantizedMultiplierSmallerThanOne(int32_t x, int32_t quant_mul, int right_shift) { //DONE
	return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x, quant_mul), right_shift);
}

void Conv(uint8_t output_conv_arr[imgSize][nFilters]) {
	int rows = convOUTshape[1];
	int cols = convOUTshape[0];

	int32_t acc;
	int32_t input_val;
	int32_t weight_val;
	for (int i = 0; i < rows; i++) {
		acc = 0;
		for (int j = 0; j < cols; j++) {
			if ((j + 1 <= cols - 1) & (j > 0)) {
				for (int k = 0; k < kernel_shape; k++) {
					input_val = (int32_t)(img[0][j - 1 + k][0]);
					weight_val = (int32_t)(quantized_weight_conv[0][0][k][i]);
					acc += (input_val - input_offset_conv) * (weight_val - weight_offset_conv);
				}
			}
			acc += quantized_bias_conv[i];
			acc = MultiplyByQuantizedMultiplierSmallerThanOne(acc, M_0_conv, right_shift_conv);
			acc += output_offset_conv; // activation offset
			acc = maxOf(acc, (int32_t)(0));
			acc = minOf(acc, (int32_t)(255));
			output_conv_arr[j][i] = (uint8_t)(acc); //RETURN ARRAY
			//printf("[%d][%d]: %d\n", j, i, output_conv_arr[j][i]);
		}
	}
}

void flattenConv(uint8_t arrIN[imgSize][nFilters], uint8_t arrOUT[1][imgSize*nFilters]) {
	for (int i = 0; i < imgSize; i++) {
		for (int j = 0; j < nFilters; j++) {
			arrOUT[0][i*nFilters + j] = arrIN[i][j];
			//printf("[%d]: %d\n", i*nFilters + j, arrOUT[i*nFilters + j]);
		}
	}
}

void FullyConnectedDense(uint8_t quantized_inputs[1][imgSize*nFilters], uint8_t output_full_conn_dense_arr[1][n_dense_nodes]) {
	int rows = fdcWshape[0];
	int cols = fdcWshape[1];

	int32_t acc;
	int32_t input_val;
	int32_t weight_val;
	for (int i = 0; i < rows; i++) {
		acc = 0;
		for (int j = 0; j < cols; j++) {
			input_val = (int32_t)(quantized_inputs[0][j]);
			weight_val = (int32_t)(quantized_weight_dense[i][j]);
			acc += (input_val - input_offset_dense) * (weight_val - weight_offset_dense);
		}
		acc += quantized_bias_dense[i];
		acc = MultiplyByQuantizedMultiplierSmallerThanOne(acc, M_0_dense, right_shift_dense);
		acc += output_offset_dense;  // activation offset
		acc = maxOf(acc, (int32_t)(0));
		acc = minOf(acc, (int32_t)(255));
		output_full_conn_dense_arr[0][i] = (uint8_t)(acc);
		// printf("[%d]: %d\n", i, output_full_conn_dense_arr[0][i]);
	}
}

void FullyConnectedPred(uint8_t quantized_inputs[1][n_dense_nodes], uint8_t output_full_conn_pred_arr[1][n_pred_nodes]) {
	int rows = fcpWshape[0];
	int cols = fcpWshape[1];

	int32_t acc;
	int32_t input_val;
	int32_t weight_val;
	for (int i = 0; i < rows; i++) {
		acc = 0;
		for (int j = 0; j < cols; j++) {
			input_val = (int32_t)(quantized_inputs[0][j]);
			weight_val = (int32_t)(quantized_weight_pred[i][j]);
			acc += (input_val - input_offset_pred) * (weight_val - weight_offset_pred);
		}
		acc += quantized_bias_pred[i];
		acc = MultiplyByQuantizedMultiplierSmallerThanOne(acc, M_0_pred, right_shift_pred);
		acc += output_offset_pred;  // activation offset
		acc = maxOf(acc, (int32_t)(0));
		acc = minOf(acc, (int32_t)(255));
		output_full_conn_pred_arr[0][i] = (uint8_t)(acc);
		//printf("[%d]: %d\n", i, output_full_conn_pred_arr[0][i]);
	}
}

uint8_t argMaxPred(uint8_t quantized_input[1][n_pred_nodes]) {
	uint8_t argMax      = 0;
	uint8_t argMaxIndex = 0;

	for (int i = 0; i < n_pred_nodes; i++) {
		if (quantized_input[0][i] > argMax) {
			argMax      = quantized_input[0][i];
			argMaxIndex = i;
		}
	}
	return argMaxIndex;
}

int main() {
	for (int i = 0; i < 10000; i++) {
		uint8_t output_conv_arr[imgSize][nFilters];
		Conv(output_conv_arr);

		uint8_t output_conv_arr_flat[1][imgSize*nFilters];
		flattenConv(output_conv_arr, output_conv_arr_flat);

		uint8_t output_full_conn_dense_arr[1][n_dense_nodes];
		FullyConnectedDense(output_conv_arr_flat, output_full_conn_dense_arr);

		uint8_t output_full_conn_pred_arr[1][n_pred_nodes];
		FullyConnectedPred(output_full_conn_dense_arr, output_full_conn_pred_arr);

		uint8_t final_pred = argMaxPred(output_full_conn_pred_arr);

		printf("Guess  : %d", final_pred);
		printf("Answer : %d", lbl);
	}
	return 0;
}

