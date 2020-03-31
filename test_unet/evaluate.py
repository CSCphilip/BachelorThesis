# Based on https://www.youtube.com/watch?v=azM57JuQpQI
import tensorflow as tf
import os
import sys
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
import random

seed = 42
np.random.seed = seed

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
for i in range(55, 56): #To 82
    test_ids.append('IDRiD_' + str(i))

X_train = np.zeros((len(train_ids), img_width, img_height,
        img_channels), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), img_width, img_height, 1),
        dtype=np.bool)

print("Preprocess the training images")
for n, id in tqdm(enumerate(train_ids), total=len(train_ids)):
    input_img_path = os.path.join(train_path, 'input_images_cropped/')
    ground_truth_img_path = os.path.join(train_path,
            'ground_truth_jpg_cropped/')

    #TODO: no normalization, i.e. X_train / 255.0
    input_img = cv2.imread(input_img_path + id + '.jpg')
    input_img = cv2.resize(input_img, (img_width, img_height))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
    X_train[n] = input_img

    ground_truth_img = cv2.imread(ground_truth_img_path + id + '_EX.jpg', cv2.IMREAD_GRAYSCALE)
    ground_truth_img = cv2.resize(ground_truth_img, (img_width, img_height))
    ground_truth_img = ground_truth_img[..., np.newaxis]
    Y_train[n] = ground_truth_img
print("Done!")

X_test = np.zeros((len(test_ids), img_width, img_height,
        img_channels), dtype=np.uint8)
Y_test = np.zeros((len(test_ids), img_width, img_height, 1),
        dtype=np.bool)

print("Preprocess the testing images")
for n, id in tqdm(enumerate(test_ids), total=len(test_ids)):
    test_img_path = os.path.join(test_path, 'input_images_cropped/')
    ground_truth_img_path = os.path.join(test_path, 'ground_truth_jpg_cropped/')

    test_img = cv2.imread(test_img_path + id + '.jpg')
    test_img = cv2.resize(test_img, (img_width, img_height))
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    X_test[n] = test_img

    ground_truth_img = cv2.imread(ground_truth_img_path + id + '_EX.jpg', cv2.IMREAD_GRAYSCALE)
    ground_truth_img = cv2.resize(ground_truth_img, (img_width, img_height))
    ground_truth_img = ground_truth_img[..., np.newaxis]
    Y_test[n] = ground_truth_img

print("Done!")




# Load model
model = keras.models.load_model("unet_model.h5")

# Not used anywhere
idx = random.randint(0, len(X_train))

# preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
# preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

X_test = X_test / 255.0
preds_test = model.predict(X_test, verbose=1)

print(preds_test.shape)
print(preds_test[0])
print(np.max(preds_test))

# preds_train_t = (preds_train > 0.5).astype(np.uint8)
# preds_val_t = (preds_val > 0.5).astype(np.uint8)
# preds_test_t = (preds_test > 0.5).astype(np.uint8)
preds_test_t = preds_test #(preds_test > 0.3).astype(np.bool)

img_index = 0

# On train dataset
# fig = plt.figure()
# fig.add_subplot(1,3,1)
# plt.title("Input image")
# plt.imshow(X_train[img_index])
# fig.add_subplot(1,3,2)
# plt.title("Ground truth")
# plt.imshow(np.squeeze(Y_train[img_index]))
# fig.add_subplot(1,3,3)
# plt.title("Model output")
# plt.imshow(np.squeeze(preds_train_t[img_index]))
# plt.show()

# On test dataset
fig = plt.figure()
fig.add_subplot(1,3,1)
plt.title("Input image")
plt.imshow(X_test[img_index])
fig.add_subplot(1,3,2)
plt.title("Ground truth")
plt.imshow(np.squeeze(Y_test[img_index]))
fig.add_subplot(1,3,3)
plt.title("Model output")
plt.imshow(np.squeeze(preds_test_t[img_index]),cmap="gray")
plt.show()