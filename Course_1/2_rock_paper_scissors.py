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
3- Create model.
4- Compile model.
5- Train model.
6- Predict images.

*- Softmax Function: https://en.wikipedia.org/wiki/Softmax_function
*- Relu(X) = if x > 0 ? x : 0
"""


# Function below will run automaticly whenever we run the script . We can run script using "python $(file_location)"

def predict_using_ml():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # Normalize data:
    train_images = train_images / 255.0
    test_images = test_images / 255.0  # (255 is max RGB)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),  # Photos' size.
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    classifications = model.predict(test_images)

    # print(classifications[0]) (10 different probabilities, softmax will choose max of those)
    # print(test_labels[0]) (softmax have chosen last element (9th) of classifications[0] as we expect.
    print(f"ML prediction completed, test accuracy: {'%.2f'%(test_acc*100)}%.")


if __name__ == '__main__':
    predict_using_ml()

