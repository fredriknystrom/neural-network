import logging
import numpy as np

from nn.activations import Activation, Relu

class Layer():

    def __init__(self, m_input, n_output, activation : Activation):
        self.layer_type = None
        self.m_input = m_input
        self.n_output = n_output
        self.weights = None
        self.activation = activation()
        self.init_weights()
        self.init_bias()

    # Use this method in NN to set learning rate
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        
    """
    The weights is a (m, n) matrix where n is the number of output neurons and m is the number 
    of inputs. The input vector shape is a (m,1) where m is the number of inputs. Finally the bias 
    vector is of shape (n,1)
    """
    def init_weights(self):
        
        if self.activation.name == 'sigmoid':
            # Initialize weights for sigmoid activation
            self.weights = np.random.randn(self.m_input, self.n_output)
        elif self.activation.name == 'relu':
            # Initialize weights for ReLU activation
            # Use He initialization for ReLU
            variance = 2.0 / self.m_input
            self.weights = np.random.randn(self.m_input, self.n_output) * np.sqrt(variance)
        elif self.activation.name == 'tanh':
            # Initialize weights for tanh activation
            # Use Xavier initialization for tanh
            variance = 1.0 / self.m_input
            self.weights = np.random.randn(self.m_input, self.n_output) * np.sqrt(variance)
            
        else:
            raise ValueError("Unsupported activation function")
        logging.debug(f"init weights with shape: {self.weights.shape}, activation {self.activation.name}")

    def init_bias(self):
        self.bias = np.random.randn(self.n_output, 1)
        logging.debug(f"init bias shape: {self.bias.shape}")

    def forward(self, input_data):
        self.input = input_data
        # Calculate z from weights and previous layers activations
        self.z = np.dot(self.weights.T, self.input) + self.bias
        # Apply the activation function on z
        activation_result = self.activation.activate(self.z)
        logging.debug(f"forward: {self.weights.T.shape}x{self.input.shape} => ({self.weights.T.shape[0]},{self.input.shape[1]})")
        logging.debug(f"activation shape: {activation_result.shape}")
        return activation_result

    def backward(self, loss_gradient): # loss gradient with respect to output
        # Compute the gradient of the activation function with respect to the layer's input
        activation_gradient = self.activation.prime(self.z)
        logging.debug(f"activation_gradient: {activation_gradient.shape}")
        logging.debug(f"loss_gradient: {loss_gradient.shape}")
        # Combine the loss and activation gradient
        loss_activation_gradient = loss_gradient * activation_gradient
        logging.debug(f"loss_activation_gradient: {loss_activation_gradient.shape}")
        logging.debug(f"input: {self.input.shape}")
        # Update the weights and biases using gradients
        weights_gradient = np.dot(self.input, loss_activation_gradient.T)
        bias_gradient = np.sum(loss_activation_gradient, axis=0, keepdims=True)
        logging.debug(f"weights: {self.weights.shape}")
        logging.debug(f"weights_gradient: {weights_gradient.shape}")
        self.weights -= self.learning_rate * weights_gradient
        self.bias -= self.learning_rate * bias_gradient
        # Compute and return the gradient with respect to the input of this layer
        input_gradient = np.dot(self.weights, loss_activation_gradient)
        return input_gradient

    def __str__(self):
        return f"{self.layer_type}(n_input: {self.m_input}, n_output: {self.n_output}, activation: {self.activation.name})"

    def __repr__(self):
        return self.__str__()

class Dense(Layer):

    def __init__(self, m_input, n_output, activation : Activation):
        super().__init__(m_input, n_output, activation)
        self.layer_type = "Dense"