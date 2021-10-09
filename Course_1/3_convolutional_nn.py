# This script will predict whether if given photo is showing rock, paper or scissors.

# Last Edit: 09/10/2021 - Muhammet Ali Öztürk

# Importing requirements.
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

""" Summary
*- We predict given photo's category of 10 categories chosen. 
1- Get data.
2- Normalize data.
3- Initialize Neural Network. This time, convolutional model.
4- Compile model.
5- Train model.
6- Predict images.

*- MaxPooling2D = https://keras.io/api/layers/pooling_layers/max_pooling2d/
*- CNN Detailed Tutorial: https://www.youtube.com/watch?v=x_VrgWTKkiM&list=PLwlQ2xBkVGMEeWaSOacadu19OBk4jie8F&index=3
*- More CNN codes: https://developers.google.com/codelabs/tensorflow-3-convolutions#5

"""


# Function below will run automaticly whenever we run the script . We can run script using "python $(file_location)"

def predict_using_ml():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.reshape(60000, 28, 28, 1)
    # Normalize data:
    train_images = train_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0

    model = keras.Sequential([
        # 64 is number of different filters. (Check video for more about filters)
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),  # Compress the image and enhance filters with max pooling.
        keras.layers.Flatten(),  # Not giving input_shape parameter this time here.
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()  # You can call model.summary() to see the size and shape of the network.
    model.fit(train_images, train_labels, epochs=5)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    classifications = model.predict(test_images)

    # print(classifications[0]) (10 different probabilities, softmax will choose max of those)
    # print(test_labels[0]) (softmax have chosen last element (9th) of classifications[0] as we expect.
    print(f"ML prediction completed, test accuracy: {'%.2f'%(test_acc*100)}%.")


if __name__ == '__main__':
    predict_using_ml()

