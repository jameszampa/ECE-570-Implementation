# Integer Only Inferencing
This folder contains all the code required for testing the same CNN used in the term paper. I did not include the python files for generating the .h files as I did not deem it necessary. This folder contains code used for my VIP Senior Design Project and was discussed in the implementation section of my term paper. I added some extra functionality to measure execution time and overall accuracy, but other than that it is the same piece of code that will be used for my machine learning benchmark for senior design.

## main.c
All functions not explicitly stated as Adapted from... were written by me with no reference code.
### RoundingDivideByPOT
This function was directly implemented in C

Adapted from: https://github.com/google/gemmlowp/blob/master/fixedpoint/fixedpoint.h
### SaturatingRoundingDoublingHighMul
This function was directly implemented in C with a slight variation to accomodate quantizing M_0 as a int16 value.

Adapted from: https://github.com/google/gemmlowp/blob/master/fixedpoint/fixedpoint.h
### MultiplyByQuantizedMultiplierSmallerThanOne
This function was directly implemented in C

Adapted from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/common.h
### Conv
This function was directly implemented in C with a simplified for loop and no consideration for padding options. My implementation is always full padding with a stride of 1

Adapted from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/conv.h
### FullyConnectedDense/Pred
This function was directly implemented in C

Adapted from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/fully_connected.h

## write_files.py
Written by me nothing special here. This is used to create a directory of quantized images for main.c to run on. This code needs to be executed before main.c.

## offline.py
Written by me from my understanding of quantization scheme outlined in https://arxiv.org/abs/1712.05877

## model.tflite
Contains all of the parameters required for using CNN including weights, biases, and quantization parameters.

# Quanitzation Aware Training

## implementation.py
Adapted from:
https://colab.research.google.com/gist/ohtaman/c1cf119c463fd94b0da50feea320ba1e/edgetpu-with-keras.ipynb#scrollTo=jWp9_I06ZjDo

The following website gave a walkthrough on setting up quantization aware training in TensorFlow. All of the code related to setting up TensorFlow graphs for quantization aware training where pulled from the website above. The two build_keras_model functions were changed so that the CNNs are of my own making.
