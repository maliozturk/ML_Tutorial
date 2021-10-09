# This script will predict whether if given photo is showing rock, paper or scissors.

# Last Edit: 09/10/2021 - Muhammet Ali Öztürk

# Importing requirements.
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import os

""" Summary
*- We predict given photo's category of 3 categories (rock, paper or scissors). 
1- Get data (from local this time).
2- Normalize data.
3- Initialize Neural Network. This time, convolutional model.
4- Compile model.
5- Train model.
6- Predict images.

*- source: https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%208%20-%20Lesson%202%20-%20Notebook%20(RockPaperScissors).ipynb#scrollTo=ZABJp7T3VLCU

"""

rock_dir = os.path.join('Course_1/rps-datasets/rps/rock')
paper_dir = os.path.join('Course_1/rps-datasets/rps/paper')
scissors_dir = os.path.join('Course_1/rps-datasets/rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))
# Function below will run automaticly whenever we run the script . We can run script using "python $(file_location)"


def predict_using_ml():
    TRAINING_DIR = "Course_1/rps-datasets/rps/"
    training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    VALIDATION_DIR = "Course_1/rps-datasets/rps-test-set/"
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        class_mode='categorical',
        batch_size=126
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150, 150),
        class_mode='categorical',
        batch_size=126
    )

    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5), # DROPOUT IS ALSO USED.
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax') # Predict from 3 (rock, paper or scissors)
    ])

    model.summary()  # You can call model.summary() to see the size and shape of the network.
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data=validation_generator, verbose=1,
                        validation_steps=3)

    model.save("rps.h5")

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print(f"ML prediction completed, test accuracy: {'%.2f' % (val_acc * 100)}%, test loss: {'%.2f' % (val_acc * 100)}%.")
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()

    plt.show()


if __name__ == '__main__':
    predict_using_ml()

