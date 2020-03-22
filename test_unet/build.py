from keras_unet.models import vanilla_unet

import tensorflow as tf
import os
import sys
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

img_width = 512 #TODO: make adjustmnets to your images, also in the U-Net
img_height = 512
img_channels = 3

train_path = 'training_set/'
test_path = 'testing_set/'

train_ids = []
for i in range(1, 55):
    if(i < 10):
        train_ids.append('IDRiD_0' + str(i))
    else:
        train_ids.append('IDRiD_' + str(i))

test_ids = []
for i in range(55, 82):
    test_ids.append('IDRiD_' + str(i))

X_train = np.zeros((len(train_ids), img_width, img_height,
        img_channels), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), 324, 324, 1),
        dtype=np.bool)

print("Preprocess the training images")
for n, id in tqdm(enumerate(train_ids), total=len(train_ids)):
    input_img_path = os.path.join(train_path, 'input_images_cropped/')
    ground_truth_img_path = os.path.join(train_path,
            'ground_truth_jpg_cropped/')

    input_img = cv2.imread(input_img_path + id + '.jpg')
    input_img = cv2.resize(input_img, (img_width, img_height))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
    X_train[n] = input_img

    ground_truth_img = cv2.imread(ground_truth_img_path + id + '_EX.jpg', cv2.IMREAD_GRAYSCALE)
    ground_truth_img = cv2.resize(ground_truth_img, (324, 324))
    ground_truth_img = ground_truth_img[..., np.newaxis]
    Y_train[n] = ground_truth_img
print("Done!")

print(np.max(Y_train))

model = vanilla_unet(input_shape=(img_width, img_height, img_channels))

model.compile(optimizer='adam', loss='binary_crossentropy',
        metrics=['accuracy'])
model.summary()
sys.exit()
model.fit(X_train, Y_train, validation_split=0.1, batch_size=8,
        epochs=1)

model.save("unet_model.h5")
