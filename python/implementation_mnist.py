"""
Trains model in build_keras_model on MNIST, compares float accuracy with integer accuracy
adapted from:
https://colab.research.google.com/gist/ohtaman/c1cf119c463fd94b0da50feea320ba1e/edgetpu-with-keras.ipynb#scrollTo=jWp9_I06ZjDo
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from time import time


def quantize(detail, data):
    shape = detail['shape']
    dtype = detail['dtype']
    a, b = detail['quantization']

    return (data / a + b).astype(dtype).reshape(shape)


def build_keras_model():
    return keras.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        # keras.layers.MaxPool2D(pool_size=(2,2)),
        # keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        # keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        # keras.layers.BatchNormalization(),
        keras.layers.Flatten(),
        # keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])


def representative_dataset_gen():
    for i in range(1000):
        yield [train_images[i: i + 1]]


IMG_SIZE = (28, 28)

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = np.reshape(train_images, (train_images.shape[0], IMG_SIZE[0], IMG_SIZE[1], 1))
test_images = np.reshape(test_images, (test_images.shape[0], IMG_SIZE[0], IMG_SIZE[1], 1))

train_images = train_images.astype('float32') / 255.
test_images = test_images.astype('float32') / 255.

# Creates Quantization Aware Graph for determining quantization parameters
train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

tf.compat.v1.keras.backend.set_session(train_sess)

with train_graph.as_default():
    train_model = build_keras_model()
    train_model.summary()

    tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
    train_sess.run(tf.compat.v1.global_variables_initializer())

    train_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    train_model.fit(train_images, train_labels, batch_size=256)
    # save graph and checkpoints
    saver = tf.compat.v1.train.Saver()
    saver.save(train_sess, 'checkpoints')

float_acc = 0
float_lat = 0
print("Working on Float Model...")
with train_graph.as_default():
    for i in range(10000):
        start = time()
        with tf.device('/CPU:0'):
            pred = train_model.predict(test_images[i:i+1])
        end = time()
        float_lat += end - start
        if np.argmax(pred) == test_labels[i]:
            float_acc += 1

# eval
eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)

tf.compat.v1.keras.backend.set_session(eval_sess)

with eval_graph.as_default():
    keras.backend.set_learning_phase(0)
    eval_model = build_keras_model()
    tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
    eval_graph_def = eval_graph.as_graph_def()
    saver = tf.compat.v1.train.Saver()
    saver.restore(eval_sess, 'checkpoints')

    frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        eval_sess,
        eval_graph_def,
        [eval_model.output.op.name]
    )

    with open('frozen_model.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())

# convert to tflite from frozen graph
converter = tf.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file='frozen_model.pb',
    input_arrays=[train_model.layers[0].input.name.split(':')[0]],
    output_arrays=[train_model.layers[-1].output.name.split(':')[0]]
)

converter.representative_dataset = representative_dataset_gen
converter.inference_type = tf.uint8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (0., 255)}  # mean, std_dev
converter.default_ranges_stats = (0, 25)
tflite_model = converter.convert()
open('model.tflite', 'wb').write(tflite_model)

interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

quant_acc = 0
quant_lat = 0
print("Working on Quantized Model...")
for i in range(10000):
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    sample_input = quantize(input_detail, test_images[i:i+1])

    interpreter.set_tensor(input_detail['index'], sample_input)

    start = time()
    with tf.device('/CPU:0'):
        interpreter.invoke()
    end = time()
    quant_lat += end - start

    pred_quantized_model = interpreter.get_tensor(output_detail['index'])

    if np.argmax(pred_quantized_model) == test_labels[i]:
        quant_acc += 1

print('Floating Point Model Accuracy : ', float_acc / 10000)
print('Floating Point Model Latency  : ', float_lat / 10000)
print('Quantized Model Accuracy      : ', quant_acc / 10000)
print('Quantized Model Latency       : ', quant_lat / 10000)
