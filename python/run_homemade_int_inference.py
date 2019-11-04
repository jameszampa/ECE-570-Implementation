"""
Homemade implementation for inferencing using 'model.tflite' compares 'My Accuracy' with Tensorflow's implementation
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import offline
import integer_inference
from PIL import Image
import time

IMG_SIZE = (14, 14)

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_imgs_resize = []
test_imgs_resize = []

for img in train_images:
    res = np.array(Image.fromarray(img).resize(size=IMG_SIZE))
    train_imgs_resize.append(res)
train_imgs_resize = np.asarray(train_imgs_resize)

for img in test_images:
    res = np.array(Image.fromarray(img).resize(size=IMG_SIZE))
    test_imgs_resize.append(res)
test_imgs_resize = np.asarray(test_imgs_resize)

train_imgs_resize = train_imgs_resize / 255.0
test_imgs_resize = test_imgs_resize / 255.0

flat_train = []
flat_test = []

for i, img in enumerate(train_imgs_resize):
    flat_train.append(img.flatten())
flat_train = np.asarray(flat_train)

for i, img in enumerate(test_imgs_resize):
    flat_test.append(img.flatten())
flat_test = np.asarray(flat_test)

flat_train = flat_train[..., np.newaxis]
flat_test = flat_test[..., np.newaxis]

# load TFLite file
interpreter = tf.lite.Interpreter(model_path=f'model.tflite')
# Allocate memory.
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
inter_layer = interpreter.get_tensor_details()

# Conv1D offline parameters
# Hardcoded values for specific weights/biases ect. The application Netron was very helpful in understanding
# inputs and outputs to different layers. Netron gives a good overview of what a model looks like
weight_index = 4
bias_index = 6
output_index = 1
input_index = 7
quantized_weight_conv = interpreter.get_tensor(inter_layer[weight_index]['index'])
quantized_bias_conv = interpreter.get_tensor(inter_layer[bias_index]['index'])
weight_scale_conv, weight_offset_conv = inter_layer[weight_index]['quantization']
input_scale_conv, input_offset_conv = inter_layer[input_index]['quantization']
output_scale_conv, output_offset_conv = inter_layer[output_index]['quantization']
M_conv = (input_scale_conv * weight_scale_conv) / output_scale_conv
right_shift_conv, M_0_conv = offline.quantize_mult_smaller_one(M_conv)

# hidden dense layer offline parameters
weight_index = 10
bias_index = 8
output_index = 9
input_index = 0
quantized_weight_dense = interpreter.get_tensor(inter_layer[weight_index]['index'])
quantized_bias_dense = interpreter.get_tensor(inter_layer[bias_index]['index'])
weight_scale_dense, weight_offset_dense = inter_layer[weight_index]['quantization']
input_scale_dense, input_offset_dense = inter_layer[input_index]['quantization']
output_scale_dense, output_offset_dense = inter_layer[output_index]['quantization']
M_dense = (input_scale_dense * weight_scale_dense) / output_scale_dense
right_shift_dense, M_0_dense = offline.quantize_mult_smaller_one(M_dense)

# prediction layer offline parameters
weight_index = 14
bias_index = 12
output_index = 11
input_index = 9
quantized_weight_pred = interpreter.get_tensor(inter_layer[weight_index]['index'])
quantized_bias_pred = interpreter.get_tensor(inter_layer[bias_index]['index'])
weight_scale_pred, weight_offset_pred = inter_layer[weight_index]['quantization']
input_scale_pred, input_offset_pred = inter_layer[input_index]['quantization']
output_scale_pred, output_offset_pred = inter_layer[output_index]['quantization']
M_pred = (input_scale_pred * weight_scale_pred) / output_scale_pred
right_shift_pred, M_0_pred = offline.quantize_mult_smaller_one(M_pred)

avg_num_skip_dyn = 0
avg_num_skip_static = 0
total_exc = 0

avg_time_tf = 0
avg_time_home = 0

tensorflow_acc = 0
homemade_acc = 0

num_test_imgs = 10000

for i in range(num_test_imgs):
    # set up img to be infered on...
    quantized_input = offline.quantize(input_details[0], flat_test[i:i+1])
    interpreter.set_tensor(input_details[0]['index'], quantized_input)

    # let tensorflow do the math
    start = time.time()
    interpreter.invoke()
    end = time.time()
    avg_time_tf += end - start

    # Output from tf
    quantized_output_tf = interpreter.get_tensor(output_details[0]['index'])

    # Homemade inference time!
    num_skip_dyn = 0
    num_skip_static = 0
    total_exc = 0

    start = time.time()
    output_conv_arr, num_skip_dyn, num_skip_static, total_exc = (integer_inference.Conv(quantized_input, input_offset_conv, quantized_weight_conv,
                                              weight_offset_conv, quantized_bias_conv, output_offset_conv, M_0_conv,
                                              right_shift_conv, (IMG_SIZE[0] * IMG_SIZE[1], 8), num_skip_dyn, num_skip_static, total_exc))

    output_conv_arr = output_conv_arr.flatten()
    output_conv_arr = output_conv_arr[np.newaxis, ...]

    output_full_conn_arr, num_skip_dyn, num_skip_static, total_exc = (integer_inference.FullyConnected(output_conv_arr, input_offset_dense,
                                                             quantized_weight_dense, weight_offset_dense,
                                                             quantized_bias_dense, output_offset_dense, M_0_dense,
                                                             right_shift_dense, (1, 16), num_skip_dyn, num_skip_static, total_exc))

    output_full_conn_arr_2, num_skip_dyn, num_skip_static, total_exc = (integer_inference.FullyConnected(output_full_conn_arr, input_offset_pred,
                                                               quantized_weight_pred, weight_offset_pred,
                                                               quantized_bias_pred, output_offset_pred, M_0_pred,
                                                               right_shift_pred, (1, 10), num_skip_dyn, num_skip_static, total_exc))
    end = time.time()
    avg_time_home += end - start
    avg_num_skip_dyn += num_skip_dyn
    avg_num_skip_static += num_skip_static

    if test_labels[i] == np.argmax(quantized_output_tf):
        tensorflow_acc += 1
    if test_labels[i] == np.argmax(output_full_conn_arr_2):
        homemade_acc += 1

    if (i + 1) % 10 == 0:
        print('Interation ', i + 1, ':', num_test_imgs)
        print('Tensorflow - accuracy : ', tensorflow_acc / (i + 1))
        print('Tensorflow - latency  : ', avg_time_tf / (i + 1))
        print('Homemade   - accuracy : ', homemade_acc / (i + 1))
        print('Homemade   - latency  : ', avg_time_home / (i + 1))

print('Final Tensorflow - accuracy :', tensorflow_acc / num_test_imgs)
print('Final Tensorflow - latency  :', avg_time_tf / num_test_imgs)
print('Final Homemade   - accuracy :', homemade_acc / num_test_imgs)
print('Final Homemade   - latency  :', avg_time_home / num_test_imgs)
print('% of dynamic skippable excecutions     :', (avg_num_skip_dyn / num_test_imgs) / total_exc)
print('% of static skippable excecutions      :', (avg_num_skip_static / num_test_imgs) / total_exc)
