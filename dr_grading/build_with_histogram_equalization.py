# Based on https://www.tensorflow.org/tutorials/images/transfer_learning

import tensorflow as tf
import os
import sys
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

# Must be a multiple of 32.
img_width = 125
img_height = 125
img_channels = 3

training_img_path = 'input/training_set/'
training_truth_path = 'truth/'

train_ids = []
input_images = 414 # 101
for i in range(1, input_images):
    if(i < 10):
        train_ids.append('IDRiD_00' + str(i))
    elif(i < 100):
        train_ids.append('IDRiD_0' + str(i))
    else:
        train_ids.append('IDRiD_' + str(i))

X_train = np.zeros((len(train_ids), img_width, img_height,
        img_channels), dtype=np.float32) # dtype=np.uint8

# Histogram equalization algorithm, from:
# https://towardsdatascience.com/histogram-equalization-a-simple-way-to-improve-the-contrast-of-your-image-bcd66596d815
def histogram_equalization(img_in):
# segregate color streams
    b,g,r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
# calculate cdf    
    cdf_b = np.cumsum(h_b)  
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)
    
# mask all pixels with value=0 and replace it with mean of the pixel values 
    cdf_m_b = np.ma.masked_equal(cdf_b,0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min())*255/(cdf_m_b.max()-cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b,0).astype('uint8')
  
    cdf_m_g = np.ma.masked_equal(cdf_g,0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min())*255/(cdf_m_g.max()-cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g,0).astype('uint8')
    cdf_m_r = np.ma.masked_equal(cdf_r,0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min())*255/(cdf_m_r.max()-cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r,0).astype('uint8')
# merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]
  
    img_out = cv2.merge((img_b, img_g, img_r))
# validation
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    equ = cv2.merge((equ_b, equ_g, equ_r))
    #print(equ)
    #cv2.imwrite('output_name.png', equ)
    return img_out

print("Preprocess the training images")
for n, id in tqdm(enumerate(train_ids), total=len(train_ids)):
    input_img = cv2.imread(training_img_path + id + '.jpg')
    input_img = cv2.resize(input_img, (img_width, img_height))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
    input_img = histogram_equalization(input_img)
    input_img = input_img / 255.0 # Normalization
    X_train[n] = input_img
print("Done!")


for i in range(10):
    plt.imshow(X_train[i])
    plt.show()

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
epochs = 30

#TODO: crop input images to reduce noise

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
