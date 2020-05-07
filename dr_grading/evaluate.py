import tensorflow as tf
from tensorflow import keras
import os
import sys
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

#Must be a multiple of 32. Old = 512
img_width = 125 #TODO: make adjustmnets
img_height = 125
img_channels = 3

training_img_path = 'input/training_set_cropped/'
training_truth_path = 'truth/'

train_ids = []
train_input_images = 414 #To 414
for i in range(1, train_input_images):
    if(i < 10):
        train_ids.append('IDRiD_00' + str(i))
    elif(i < 100):
        train_ids.append('IDRiD_0' + str(i))
    else:
        train_ids.append('IDRiD_' + str(i))

X_train = np.zeros((len(train_ids), img_width, img_height,
        img_channels), dtype=np.float32) #dtype=np.uint8

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

testing_img_path = 'input/testing_set_cropped/'
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
    input_img = histogram_equalization(input_img)
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

X_test_tensor = tf.convert_to_tensor(X_test)


# Load model
model = keras.models.load_model("model.h5")

# Evaluate testing images
pred_test_res = model.predict(X_test_tensor)
test_labels_array = test_labels.numpy()
print(test_labels_array)

test_confusion_matrix = np.zeros((5,5), dtype=np.int32)

test_correct_predicted = 0
for i in range(len(pred_test_res)):
    if(np.argmax(pred_test_res[i]) == test_labels_array[i]):
        test_correct_predicted += 1
    test_confusion_matrix[np.argmax(pred_test_res[i])][test_labels_array[i]] += 1
print('Test images accuracy: ' + str(test_correct_predicted/len(pred_test_res)))
print(len(pred_test_res))
print(test_correct_predicted)

for row in range(5):
    for column in range(5):
        print(test_confusion_matrix[row][column], end = ' ')
    print()
print()

# Evaluate training images
pred_train_res = model.predict(X_train_tensor)
train_labels_array = train_labels.numpy()
print(train_labels_array)

train_confusion_matrix = np.zeros((5,5), dtype=np.int32)

train_correct_predicted = 0
for i in range(len(pred_train_res)):
    if(np.argmax(pred_train_res[i]) == train_labels_array[i]):
        train_correct_predicted += 1
    train_confusion_matrix[np.argmax(pred_train_res[i])][train_labels_array[i]] += 1
print('Train images accuracy: ' + str(train_correct_predicted/len(pred_train_res)))
print(len(pred_train_res))
print(train_correct_predicted)

for row in range(5):
    for column in range(5):
        print(train_confusion_matrix[row][column], end = ' ')
    print()
print()
