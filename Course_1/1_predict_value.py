# This script will predict whether if given photo is showing rock, paper or scissors.

# Last Edit: 09/10/2021 - Muhammet Ali Öztürk

# Importing requirements.
import tensorflow as tf
import numpy as np
from tensorflow import keras


# Function below will run automaticly whenever we run the script . We can run script using "python $(file_location)"
""" Summary   
1- Give x values.
2- Give results. F(X) = -X + 1
3- Initialize sequential model with single layer and unit.
4- Provide your models' optimization method, here we are using stochastic gradient descent (sgd: 
   source: https://en.wikipedia.org/wiki/Stochastic_gradient_descent) as optimizer and mean squared error as
   loss function (https://en.wikipedia.org/wiki/Loss_function) (MSE: https://en.wikipedia.org/wiki/Mean_squared_error)
5- Train your model. Here we train for 500 different epochs.
6- Predict new inputs with trained model. """


def predict_using_ml(number):
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    ys = np.array([2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0], dtype=float)

    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=500)
    print(f"Predicting output for given number: {number}...\nResult: {model.predict(number)}")
    return number


if __name__ == '__main__':
    predict_using_ml([10.0])
    # Correct value should be close to -9.0

