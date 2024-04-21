import numpy as np

class Activation:

    def __init__(self, name):
        self.name = name

    def activate(self, z):
        pass

    def prime(self, z):
        pass

class Relu(Activation):

    def __init__(self):
        super().__init__('relu')

    def activate(self, z):
        return np.maximum(0, z)
    
    def prime(self, z):
        return np.where(z > 0, 1, 0)

class Sigmoid(Activation):

    def __init__(self):
        super().__init__('sigmoid')

    def activate(self, z):
        return 1 / (1 + np.exp(-z))
    
    def prime(self, z):
        sigmoid_z = 1 / (1 + np.exp(-z))
        return sigmoid_z * (1 - sigmoid_z)

class Tanh(Activation):

    def __init__(self):
        super().__init__('tanh')

    def activate(self, z):
        return np.tanh(z)
    
    def prime(self, z):
        tanh_z = np.tanh(z)
        return 1 - tanh_z**2

def softmax(z):
    exp_z = np.exp(z)
    softmax_scores = exp_z / np.sum(exp_z, axis=0)
    return softmax_scores