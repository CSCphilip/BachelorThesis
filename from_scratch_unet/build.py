# Based on https://www.youtube.com/watch?v=azM57JuQpQI
import tensorflow as tf
import os
import sys
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

img_width = 128 #TODO: make adjustmnets to your images, also in the U-Net
img_height = 128
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
Y_train = np.zeros((len(train_ids), img_width, img_height, 1),
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
    ground_truth_img = cv2.resize(ground_truth_img, (img_width, img_height))
    ground_truth_img = ground_truth_img[..., np.newaxis]
    Y_train[n] = ground_truth_img
print("Done!")

X_test = np.zeros((len(test_ids), img_width, img_height, img_channels), dtype=np.uint8)

print("Preprocess the testing images")
for n, id in tqdm(enumerate(test_ids), total=len(test_ids)):
    test_img_path = os.path.join(test_path, 'input_images_cropped/')

    test_img = cv2.imread(test_img_path + id + '.jpg')
    test_img = cv2.resize(test_img, (img_width, img_height))
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    X_test[n] = test_img
print("Done!")

# Testing input
i = 0
print(X_train)
print(Y_train)
print(X_train.shape)
print(Y_train.shape)
plt.imshow(X_test[i])
plt.show()
plt.imshow(np.reshape(Y_train[i], (img_width, img_height)),cmap='gray')
plt.show()

# Build the model
inputs = tf.keras.layers.Input((img_width, img_height, img_channels))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs) # Convert from integers to floating points

# Downsampling
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1) # Prevent overfitting
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(c5)

# Upsampling
u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2),
        padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2),
        padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7,c3])
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2),
        padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8,c2])
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2),
        padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9,c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu',
        kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy',
        metrics=['accuracy'])
model.summary()

# Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_HE_segmentation.h5',
        verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2,
                monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]


results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16,
        epochs=25, callbacks=callbacks)

model.save('he_model.h5')
