# Based on https://www.tensorflow.org/tutorials/keras/classification

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import matplotlib.pyplot as plt

# Load dataset
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def discover_and_visualize_data():
    print(train_images.shape)
    print(len(train_labels))
    print(train_labels[0])
    print(class_names[train_labels[0]])
    print(train_labels)
    print(test_images.shape)
    print(len(test_labels))

    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

def visualize_multiple():
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

#discover_and_visualize_data()
#visualize_multiple()

model = create_model()
#print(model.summary())

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Save the model in HDF5 format.
# We could change this to the SavedModel format. SavedModels
# are able to save custom objects like subclassed models and
# custom layers without requiring the orginal code. This is
# possible with HDF5 but it's a bit trickier.
model.save('saved_model.h5')
print("model saved")
