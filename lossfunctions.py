import numpy as np

class Lossfunction:

    def __init__(self, name):
        self.name = name

    def loss(self, y_true, y_prediction):
        pass

    def prime(self, y_true, y_prediction):
        pass


class MSE(Lossfunction):

    def __init__(self):
        super().__init__('mse')

    def loss(y_true, y_prediction):
        return np.mean(np.power(y_true - y_prediction, 2))

    def prime(y_true, y_prediction):
        return 2 * (y_prediction - y_true) / np.size(y_true)