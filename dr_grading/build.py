# Based on https://www.tensorflow.org/tutorials/images/transfer_learning

import tensorflow as tf
import os
import sys
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

# Must be a multiple of 32.
img_width = 224 # TODO: make adjustments
img_height = 224
img_channels = 3

training_img_path = 'input/training_set/'
training_truth_path = 'truth/'

train_ids = []
input_images = 101 # 414
for i in range(1, input_images):
    if(i < 10):
        train_ids.append('IDRiD_00' + str(i))
    elif(i < 100):
        train_ids.append('IDRiD_0' + str(i))
    else:
        train_ids.append('IDRiD_' + str(i))

X_train = np.zeros((len(train_ids), img_width, img_height,
        img_channels), dtype=np.float32) # dtype=np.uint8

print("Preprocess the training images")
for n, id in tqdm(enumerate(train_ids), total=len(train_ids)):
    input_img = cv2.imread(training_img_path + id + '.jpg')
    input_img = cv2.resize(input_img, (img_width, img_height))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
    input_img = input_img / 255.0 # Normalization
    X_train[n] = input_img
print("Done!")

# plt.imshow(X_train[0])
# plt.show()

csv_batch_size = input_images - 1
label_name = 'Retinopathy grade'

training_dataset = tf.data.experimental.make_csv_dataset(
        training_truth_path + 'training_labels.csv',
        batch_size=csv_batch_size,
        label_name = label_name,
        shuffle=False)

_,labels = next(iter(training_dataset))
print(labels)

X_train_tensor = tf.convert_to_tensor(X_train)
print(X_train_tensor)

IMG_SHAPE = (img_width, img_height, img_channels)

# Instantiate a MobileNet V2 / an InceptionV3 model pre-loaded with weights trained on ImageNet, don't include the classification layers at the top (include_top=False).
base_model = tf.keras.applications.inception_v3.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=IMG_SHAPE
)
'''
base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet')
'''

feature_batch = base_model(X_train_tensor)
print(feature_batch.shape)

# Freeze the convolutional base, this prevents the weights in a given layer from being updated during training. Change
# to True during fine-tuning.
base_model.trainable = False
base_model.summary()

'''
# Fine-tuning

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
'''

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(5, activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
])

# BinaryCrossentropy
# This is different from the tutorial
base_learning_rate = 0.0001
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
'''
# From tutorial (fine tuning section):
base_learning_rate = 0.0001
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])
'''
model.summary()

batch_size = 16
epochs = 15

# TODO: crop input images to reduce noise
# TODO: train on the full dataset, now no images with 0 grading exists

history = model.fit(X_train_tensor, labels, batch_size, epochs, validation_split=0.1)

# Plot training and validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training and validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


model.save('model.h5')
