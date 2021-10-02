from BikerEnv import State
from numpy.random import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import random

class ValueFunctionApproximator:
    def __init__(self, n=4) -> None:
        self.weights = random(size=n)

    def predict(self, s, a=None):
        pass

    def stochastic_gradient_descent(X, y_true, epochs, learning_rate=0.01):  # X-> train data, y_true -> test outcome data(count), epoch-> iterations

        number_of_features = X.shape[1]

        # numpy array with 1 row and columns equal to number of features
        w = np.ones(shape=(number_of_features))
        b = 0  # bias
        total_samples = y_true.shape[0]

        cost_list = []
        epoch_list = []

        for i in range(epochs):
            random_index = random.randint(0, total_samples - 1)  # random index from total samples

            sample_x = X.iloc[random_index]
            sample_y = y_true.iloc[random_index]

            sample_x_transp = X.iloc[random_index, :].values.reshape(1, X.shape[1])

            y_predicted = abs(np.dot(w, sample_x.T) + b) #abs value due to longtitude having a negative value a lot

            #gradient for weight and bias
            w_grad = -(2 / total_samples) * (sample_x_transp.T.dot(y_predicted - sample_y))
            b_grad = -(2 / total_samples) * (y_predicted - sample_y)

            w = w - learning_rate * w_grad  # update weight
            b = b - learning_rate * b_grad  # bias update

            mse = mean_squared_error(sample_y, y_predicted)

            if i % 50 == 0:  # every 50th iteration record the cost and epoch value
                cost_list.append(mse)
                epoch_list.append(i)

        return w, mse