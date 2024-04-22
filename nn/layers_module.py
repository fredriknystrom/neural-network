import logging
import numpy as np

from nn.activations import Activation
from nn.regularizer import Regularizer

class Layer():

    def __init__(self, m_input, n_output, activation : Activation, regularization: Regularizer=None, weights=None, bias=None):
        self.layer_type = None
        self.m_input = m_input
        self.n_output = n_output
        self.activation = activation()
        self.regularization = regularization
        self.weights = weights
        self.bias = bias
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
        if self.weights is None:
            if self.activation.name in ['sigmoid', 'softmax']:
                self.weights = np.random.randn(self.m_input, self.n_output)
            elif self.activation.name in ['relu', 'linear']:
                # Use He initialization for ReLU
                variance = 2.0 / self.m_input
                self.weights = np.random.randn(self.m_input, self.n_output) * np.sqrt(variance)
            elif self.activation.name == 'tanh':
                # Use Xavier initialization for tanh
                variance = 1.0 / self.m_input
                self.weights = np.random.randn(self.m_input, self.n_output) * np.sqrt(variance)
                
            else:
                raise ValueError("Unsupported activation function")
        logging.info(f"init weights with shape: {self.weights.shape}, activation {self.activation.name}")

    def init_bias(self):
        if self.bias is None:
            self.bias = np.random.randn(self.n_output, 1)
        logging.info(f"init bias shape: {self.bias.shape}")

    def get_weights(self):
        return self.weights
    
    def get_biases(self):
        return self.bias

    def forward(self, input_data):
        self.input = input_data
        logging.debug(f"self.input shape: {self.input.shape}")
        logging.debug(f"self.weights.T: {self.weights.T.shape}")
        logging.debug(f"self.weights: {self.weights.shape}")
        logging.debug(f"self.bias: {self.bias.shape}")
        # Calculate z from weights and previous layers activations
        self.z = np.dot(self.weights.T, self.input) + self.bias
        # Apply the activation function on z
        activation_result = self.activation.activate(self.z)
        logging.debug(f"activation_result: {activation_result}")
        logging.debug(f"forward: {self.weights.T.shape}x{self.input.shape} => ({self.weights.T.shape[0]},{self.input.shape[1]})")
        logging.debug(f"activation shape: {activation_result.shape}")
        return activation_result

    def backward(self, loss_gradient): # loss gradient with respect to output
        # Compute the gradient of the activation function with respect to the layer's input
        activation_gradient = self.activation.prime(self.z)
        logging.debug(f"activation_gradient: {activation_gradient.shape}")
        logging.debug(f"activation_gradient: {activation_gradient}")
        logging.debug(f"loss_gradient: {loss_gradient}")
        logging.debug(f"loss_gradient: {loss_gradient.shape}")
        # Combine the loss and activation gradient
        loss_activation_gradient = loss_gradient * activation_gradient
        logging.debug(f"loss_activation_gradient: {loss_activation_gradient.shape}")
        logging.debug(f"loss_activation_gradient: {loss_activation_gradient}")
        logging.debug(f"input: {self.input.shape}")
        logging.debug(f"input: {self.input}")
        # Update the weights and biases using gradients
        weights_gradient = np.dot(self.input, loss_activation_gradient.T)
        logging.debug(f"loss_activation_gradient: {loss_activation_gradient.shape}")
        bias_gradient = np.sum(loss_activation_gradient, axis=1, keepdims=True)
        logging.debug(f"bias_gradient: {bias_gradient.shape}")
        logging.debug(f"weights: {self.weights.shape}")
        logging.debug(f"weights: {self.weights}")
        logging.debug(f"weights_gradient: {weights_gradient.shape}")
        logging.debug(f"weights_gradient: {weights_gradient}")
        self.weights -= self.learning_rate * weights_gradient
        self.bias -= self.learning_rate * bias_gradient
        # NOTE: I think bias gradient should be summed over axis one
        assert self.bias.shape == bias_gradient.shape, "should have same shape"
        logging.debug(f"self.bias: {self.bias.shape}")
        
        # Compute and return the gradient with respect to the input of this layer
        input_gradient = np.dot(self.weights, loss_activation_gradient)
        return input_gradient
    
    def compute_regularization_loss(self):
        if self.regularization:
            return self.regularization.apply(self.weights)
        return 0

    def __str__(self):
        return f"{self.layer_type}(n_input: {self.m_input}, n_output: {self.n_output}, activation: {self.activation.name})"

    def __repr__(self):
        return self.__str__()

class Dense(Layer):

    def __init__(self, m_input, n_output, activation : Activation, regularization=None, weights=None, bias=None ):
        super().__init__(m_input, n_output, activation, regularization, weights, bias)
        self.layer_type = "Dense"