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
        super().__init__('MSE')

    def loss(y_true, y_prediction):
        return np.mean(np.power(y_true - y_prediction, 2))

    def prime(y_true, y_prediction):
        return 2 * (y_prediction - y_true) / np.size(y_true)
    
class BinaryCrossEntropy(Lossfunction):

    def __init__(self):
        super().__init__('BinaryCrossEntropy')

    def loss(y_true, y_prediction):
        # To avoid division by zero we clip the values as really small instead
        y_prediction = np.clip(y_prediction, 1e-12, 1 - 1e-12)
        return -np.mean(y_true * np.log(y_prediction) + (1 - y_true) * np.log(1 - y_prediction))

    def prime(y_true, y_prediction):
        # To avoid division by zero we clip the values as really small instead
        y_prediction = np.clip(y_prediction, 1e-12, 1 - 1e-12)
        return -(y_true / y_prediction) + (1 - y_true) / (1 - y_prediction)