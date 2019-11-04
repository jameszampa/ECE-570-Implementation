"""
Trains model in build_keras_model on MNIST, compares float accuracy with integer accuracy

adapted from:
https://colab.research.google.com/gist/ohtaman/c1cf119c463fd94b0da50feea320ba1e/edgetpu-with-keras.ipynb#scrollTo=jWp9_I06ZjDo
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import offline
from PIL import Image

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


def build_keras_model():
    return keras.Sequential([
        keras.layers.Conv1D(filters=8, kernel_size=3, input_shape=(IMG_SIZE[0] * IMG_SIZE[1], 1), padding='same', activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])


def representative_dataset_gen():
    for i in range(1000):
        yield [flat_train[i: i + 1]]


train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

keras.backend.set_session(train_sess)

with train_graph.as_default():
    train_model = build_keras_model()

    tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
    train_sess.run(tf.global_variables_initializer())

    train_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    train_model.fit(flat_train, train_labels, epochs=1)
    train_model.summary()
    # save graph and checkpoints
    saver = tf.train.Saver()
    saver.save(train_sess, 'checkpoints')

original_predictions = []
original_acc = 0
with train_graph.as_default():
    print('Testing on Original')
    for i in range(9999):
        pred = train_model.predict(flat_test[i:i+1])
        if np.argmax(pred) == test_labels[i]:
            original_acc += 1
        original_predictions.append(np.argmax(pred))


# eval
eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)

keras.backend.set_session(eval_sess)

with eval_graph.as_default():
    keras.backend.set_learning_phase(0)
    eval_model = build_keras_model()
    tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
    eval_graph_def = eval_graph.as_graph_def()
    saver = tf.train.Saver()
    saver.restore(eval_sess, 'checkpoints')

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
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
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (0., 255)}  # mean, std_dev
converter.default_ranges_stats = (0, 25)
tflite_model = converter.convert()
open('model.tflite', 'wb').write(tflite_model)

interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

quant_acc = 0

print('Testing on Quantized')
for i in range(9999):
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    sample_input = offline.quantize(input_detail, flat_test[i:i+1])

    interpreter.set_tensor(input_detail['index'], sample_input)
    interpreter.invoke()

    pred_quantized_model = interpreter.get_tensor(output_detail['index'])

    if np.argmax(pred_quantized_model) == test_labels[i]:
        quant_acc += 1

print('Integer Acc: ',quant_acc / 9999)
print('Float Acc: ', original_acc / 9999)