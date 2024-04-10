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

def main():
    # Example usage:
    relu = Relu()
    sigmoid = Sigmoid()
    tanh = Tanh()

    # Example input vector
    x = np.array([-1.0, 0.0, 1.0])
    print(x.shape)

    # Apply activation functions to the input vector
    relu_result = relu.activate(x)
    sigmoid_result = sigmoid.activate(x)
    tanh_result = tanh.activate(x)

    print("ReLU:", relu_result)
    print("Sigmoid:", sigmoid_result)
    print("Tanh:", tanh_result)

    # Apply derivative functions to the input vector
    relu_prime = relu.derivative(x)
    sigmoid_prime = sigmoid.derivative(x)
    tanh_prime = tanh.derivative(x)

    print("ReLU prime:", relu_prime)
    print("Sigmoid prime:", sigmoid_prime)
    print("Tanh prime:", tanh_prime)

if __name__ == "__main__":
    main()

