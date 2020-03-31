import tensorflow as tf
from tensorflow import keras
import os
import sys
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

#Must be a multiple of 32. Old = 512
img_width = 224 #TODO: make adjustmnets
img_height = 224
img_channels = 3

training_img_path = 'input/training_set/'
training_truth_path = 'truth/'

train_ids = []
train_input_images = 101 #To 414
for i in range(1, train_input_images):
    if(i < 10):
        train_ids.append('IDRiD_00' + str(i))
    elif(i < 100):
        train_ids.append('IDRiD_0' + str(i))
    else:
        train_ids.append('IDRiD_' + str(i))

X_train = np.zeros((len(train_ids), img_width, img_height,
        img_channels), dtype=np.float32) #dtype=np.uint8

print("Preprocess the training images")
for n, id in tqdm(enumerate(train_ids), total=len(train_ids)):
    input_img = cv2.imread(training_img_path + id + '.jpg')
    input_img = cv2.resize(input_img, (img_width, img_height))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
    input_img = input_img / 255.0 # Normalization
    X_train[n] = input_img
print("Done!")

train_csv_batch_size = train_input_images - 1
label_name = 'Retinopathy grade'

training_dataset = tf.data.experimental.make_csv_dataset(
        training_truth_path + 'training_labels.csv',
        batch_size=train_csv_batch_size,
        label_name = label_name,
        shuffle=False)

_,train_labels = next(iter(training_dataset))
#print(train_labels)

X_train_tensor = tf.convert_to_tensor(X_train)
#print(X_train_tensor)

testing_img_path = 'input/testing_set/'
testing_truth_path = 'truth/'

test_ids = []
test_input_images = 104 #To 104
for i in range(1, test_input_images):
    if(i < 10):
        test_ids.append('IDRiD_00' + str(i))
    elif(i < 100):
        test_ids.append('IDRiD_0' + str(i))
    else:
        test_ids.append('IDRiD_' + str(i))

X_test = np.zeros((len(test_ids), img_width, img_height,
        img_channels), dtype=np.float32)

print("Preprocess the testing images")
for n, id in tqdm(enumerate(test_ids), total=len(test_ids)):
    input_img = cv2.imread(testing_img_path + id + '.jpg')
    input_img = cv2.resize(input_img, (img_width, img_height))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
    input_img = input_img / 255.0 # Normalization
    X_test[n] = input_img
print("Done!")

test_csv_batch_size = test_input_images - 1
testing_dataset = tf.data.experimental.make_csv_dataset(
        testing_truth_path + 'testing_labels.csv',
        batch_size=test_csv_batch_size,
        label_name = label_name,
        shuffle=False)

_,test_labels = next(iter(testing_dataset))
print(test_labels)

X_test_tensor = tf.convert_to_tensor(X_test)


# Load model
model = keras.models.load_model("model.h5")

pred_test_res = model.predict(X_test_tensor)

test_labels_array = test_labels.numpy()
print(test_labels_array)

right_predicted = 0
for i in range(len(pred_test_res)):
    print('IMG ' + str(i) + ':')
    print(pred_test_res[i])
    print('Predicted label: ' + str(np.argmax(pred_test_res[i])) + ' Actual label: ' + str(test_labels_array[i]))
    print(np.argmax(pred_test_res[i]) == test_labels_array[i])
    if(np.argmax(pred_test_res[i]) == test_labels_array[i]):
        right_predicted = right_predicted + 1
    print()
print('Test images accuracy: ' + str(right_predicted/len(pred_test_res)))

print()
pred_train_res = model.predict(X_train_tensor)
train_labels_array = train_labels.numpy()
print(train_labels_array)

right_predicted = 0
for i in range(len(pred_train_res)):
    print('IMG ' + str(i) + ':')
    print(pred_train_res[i])
    print('Predicted label: ' + str(np.argmax(pred_train_res[i])) + ' Actual label: ' + str(train_labels_array[i]))
    print(np.argmax(pred_train_res[i]) == train_labels_array[i])
    if(np.argmax(pred_train_res[i]) == train_labels_array[i]):
        right_predicted = right_predicted + 1
    print()
print('Train images accuracy: ' + str(right_predicted/len(pred_train_res)))
