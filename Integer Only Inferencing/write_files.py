import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import sys
sys.path.append('..')
from implementation import quantize

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

# load TFLite file
interpreter = tf.lite.Interpreter(model_path=f'model.tflite')
# Allocate memory.
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
inter_layer = interpreter.get_tensor_details()

if not os.path.exists('test_imgs'):
    os.makedirs('test_imgs')

for i in range(10000):
    quantized_input = quantize(input_details[0], flat_test[i:i+1])
    path = 'test_imgs/img_'
    path += str(i)
    with open(path, 'w') as f:
        for j in range(196):
            f.write(str(quantized_input[0][j][0]) + ', ')
        f.write(str(test_labels[i]))

