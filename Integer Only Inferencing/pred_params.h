/*
 * pred_params.h
 *
 *  Created on: Oct 30, 2019
 *      Author: James
 */

#ifndef PRED_PARAMS_H_
#define PRED_PARAMS_H_

const uint8_t quantized_weight_pred[10][16] = {
	190, 186, 225, 187, 112, 202,  85,  57, 187,  37,  86, 122, 203,  79,  63, 233,
	60,  46,  88, 112, 157, 199, 223, 196, 102, 186, 218, 165,   8,  62, 205,  82,
	180,  91, 101, 202,  79,  73,  95, 143, 252, 167, 185, 181,  84,  35, 222, 121,
	213, 217,  33, 195, 123, 241,  94, 214,  66, 169,  99,  79, 128,  83,  95, 126,
	46, 131, 158, 104,  41,   5, 186, 228,  22,  74, 236,  59, 211, 242, 209,  90,
	223, 247, 179,  84,  42, 126, 229, 217, 193,  93, 225,  49,  54,  71,  83, 133,
	179, 166, 106,   1,  31,  86,  81,  94, 164, 171, 232, 255, 137, 190, 145, 173,
	50, 110,  38, 199, 210, 180, 192, 220, 249,  12,  68, 116, 201, 219, 130, 196,
	147, 214,  69, 196, 199, 179, 206,  65, 165, 229, 238,  69, 176, 118,  36, 193,
	177, 111, 175, 207, 218,  34, 210, 212,  88,  85, 129, 180,  99, 201,  56,  90
};

const int16_t quantized_bias_pred[10] = { -105, -182, 115, 31, -6, 195, -491, -205, -555, -196 };

const uint8_t weight_offset_pred = 153;

const uint8_t input_offset_pred = 0;

const uint8_t output_offset_pred = 181;

const int right_shift_pred = 8;

const int16_t M_0_pred = 20452;

const int n_pred_nodes = 10;

const int fcpINshape[2] = { 1, 16 };

const int fcpWshape[2] = { 10, 16 };

const int fcpBshape[1] = { 10 };

const int fcpOUTshape[2] = { 1, 10 };

#endif /* PRED_PARAMS_H_ */
