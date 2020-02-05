#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <sys/time.h>
#include <dirent.h>
#include <string.h>

#include "common.h"
#include "conv_params.h"
#include "dense_params.h"
#include "pred_params.h"

void Conv(uint8_t img[1][imgSize][1], uint8_t output_conv_arr[imgSize][nFilters]);
void flattenConv(uint8_t arrIN[imgSize][nFilters], uint8_t arrOUT[1][imgSize*nFilters]);
int32_t maxOf(int32_t a, int32_t b);
int32_t minOf(int32_t a, int32_t b);
int32_t RoundingDivideByPOT(int32_t x, int exponent);
int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b);
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

int32_t RoundingDivideByPOT(int32_t x, int exponent) {
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

int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
	/*
	I altered the implementation of this function to use M_0 represented as an
	int16 value instead of a int32 value. This was done because the microcontroller
	which this algorithm is being implemented for does not support int64.
	*/
	int overflow = (a == b) & (a == 0-2147483648);
	if (overflow) return (int32_t)(2147483647);

	int32_t a_32 = (int32_t) a;
	int32_t b_32 = (int32_t) b;
	int32_t ab_32 = a_32 * b_32;
	int32_t nudge;

	if (ab_32 >= 0) nudge = (int32_t)(1 << 14);
	else nudge = (int32_t)(1 - (1 << 14));

	int32_t ab_x2_high16 = (int32_t)((ab_32 + nudge) / (int32_t)((int32_t)(1) << 15));
	return ab_x2_high16;
}

int32_t MultiplyByQuantizedMultiplierSmallerThanOne(int32_t x, int32_t quant_mul, int right_shift) {
	return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x, quant_mul), right_shift);
}

void Conv(uint8_t img[1][imgSize][1], uint8_t output_conv_arr[imgSize][nFilters]) {
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
			acc += output_offset_conv;
			acc = maxOf(acc, (int32_t)(0));
			acc = minOf(acc, (int32_t)(255));
			output_conv_arr[j][i] = (uint8_t)(acc);
		}
	}
}

void flattenConv(uint8_t arrIN[imgSize][nFilters], uint8_t arrOUT[1][imgSize*nFilters]) {
	for (int i = 0; i < imgSize; i++) {
		for (int j = 0; j < nFilters; j++) {
			arrOUT[0][i*nFilters + j] = arrIN[i][j];
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
	}
}

uint8_t argMaxPred(uint8_t quantized_input[1][n_pred_nodes]) {
	uint8_t argMax = 0;
	uint8_t argMaxIndex = 0;

	for (int i = 0; i < n_pred_nodes; i++) {
		if (quantized_input[0][i] > argMax) {
			argMax = quantized_input[0][i];
			argMaxIndex = i;
		}
	}
	return argMaxIndex;
}

int main() {
	printf("Beginning tests...\n");
	uint8_t output_conv_arr[imgSize][nFilters];
	uint8_t output_conv_arr_flat[1][imgSize*nFilters];
	uint8_t output_full_conn_dense_arr[1][n_dense_nodes];
	uint8_t output_full_conn_pred_arr[1][n_pred_nodes];
	uint8_t final_pred;
	struct timeval  tv1, tv2;
	double time_spent;
	uint8_t img[1][imgSize][1];
	int index = 0;
	int ch = 0;
	int lbl = 0;
	char str[32];
	char num[6];
	int acc = 0;
	double avg_time = 0;
	int num_imgs = 10000;
	bool display = false;

	
	for (int i = 0; i < num_imgs; i++) {
		ch = 0;
		sprintf(str, "test_imgs/img_");
		sprintf(num, "%d", index);
		strcat(str, num);
		FILE* fp = fopen(str, "r");
		for (int j = 0; j < 196; j++) {
			fscanf(fp, "%d, ", &ch);
			img[0][j][0] = (uint8_t)ch;
		}
		fscanf(fp, "%d, ", &lbl);
		fclose(fp);
		
		gettimeofday(&tv1, NULL);
		
		Conv(img, output_conv_arr);
		flattenConv(output_conv_arr, output_conv_arr_flat);
		FullyConnectedDense(output_conv_arr_flat, output_full_conn_dense_arr);
		FullyConnectedPred(output_full_conn_dense_arr, output_full_conn_pred_arr);
		final_pred = argMaxPred(output_full_conn_pred_arr);
		
		gettimeofday(&tv2, NULL);
		time_spent = (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec);
		avg_time += time_spent;
		
		if (final_pred == (uint8_t)lbl) {
			acc++;
		}
		
		if (display) {
			printf("Index: %d\nPred: %d\nAnswer: %d\n", i, final_pred, (uint8_t)lbl);
		}
		
		index++;
		
	}
	printf("ACC: %f\n", ((double)acc) / num_imgs);
	printf("LAT: %f\n", avg_time / num_imgs);
	return 0;
}

