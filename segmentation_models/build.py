import os
import sys
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

import segmentation_models as sm

# Data Generator
class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=2048):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()

    def __load__(self, id_name):
        # Path
        image_path = os.path.join(self.path, 'training_set', id_name) + ".jpg"
        ground_truth_path = os.path.join(self.path, 'ground_truth_jpg', id_name) + "_EX.jpg"

        # Reading Image
        image = cv2.imread(image_path, 1)
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Reading Masks
        ground_truth = cv2.imread(ground_truth_path, 1)
        ground_truth = cv2.resize(ground_truth, (self.image_size, self.image_size))

        # Convert from BGR to RGB #TODO: uncomment
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        ground_truth = cv2.cvtColor(ground_truth,cv2.COLOR_BGR2RGB)

        # Normalizaing, Info: https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
        image = image/255.0
        #print('Image; Min: %.3f, Max: %.3f' % (image.min(), image.max()))
        ground_truth = ground_truth/255.0
        #print('Truth; Min: %.3f, Max: %.3f' % (ground_truth.min(), ground_truth.max()))

        return image, ground_truth

    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size

        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]

        image = []
        mask  = []

        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)

        image = np.array(image)
        mask  = np.array(mask)

        return image, mask

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

# Hyperparameters
image_size = 64 #128 #works fine with 512 #2048   #TODO: change later on
train_path = "dr/" #TODO: add test set and change val_data_size
epochs = 5
batch_size = 1

# Training Ids
train_ids = []
for i in range(1, 55): # To 55
    if(i < 10):
        train_ids.append('IDRiD_0' + str(i))
    else:
        train_ids.append('IDRiD_' + str(i))

# Validation Data Size
val_data_size = 10

valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]

gen = DataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)
x, y = gen.__getitem__(0)
print(x.shape, y.shape)

r = random.randint(0, len(x)-1)

f = plt.figure(num=None, figsize=(15,7))
f.add_subplot(1,2,1)
plt.imshow(x[r])
f.add_subplot(1,2,2)
plt.imshow(y[r], cmap="gray")
plt.show()

model = sm.Unet()

model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
valid_gen = DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)

train_steps = len(train_ids)//batch_size
valid_steps = len(valid_ids)//batch_size

model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps,
                    epochs=epochs)

model.save("unet.h5")
print("model saved")
