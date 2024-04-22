import numpy as np

class Regularizer:
    def __init__(self, lamdba):
        self.lamdba = lamdba

    def apply(self, weights):
        pass

class L1Regularizer(Regularizer):
    def apply(self, weights):
        return self.lamdba * np.sum(np.abs(weights))

class L2Regularizer(Regularizer):
    def apply(self, weights):
        return self.lamdba * np.sum(weights ** 2)
