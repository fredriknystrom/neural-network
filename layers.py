import numpy as np

from activations import Activation, Relu

class Layer():

    def __init__(self, m_input, n_output, activation : Activation):
        self.layer_type = None
        self.m_input = m_input
        self.n_output = n_output
        self.weights = None
        self.activation = activation()
        self.debug = False
        self.init_weights()
        self.init_bias()

    # Use this method in NN to set learning rate
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    # Use this method in NN to set to debug
    def set_debug(self, debug):
        self.debug = debug
        
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
        if self.debug:
            print(f"init weights with shape: {self.weights.shape}, activation {self.activation.name}")

    def init_bias(self):
        self.bias = np.random.randn(self.n_output, 1)
        if self.debug:
            print(f"init bias shape: {self.bias.shape}")

    def forward(self, input_data):
        self.input = input_data
        # Calculate z from weights and previous layers activations
        self.z = np.dot(self.weights.T, self.input) + self.bias
        # Apply the activation function on z
        activation_result = self.activation.activate(self.z)
        if self.debug:
            print(f"forward mul {self.weights.T.shape}x{self.input.shape} => ({self.weights.T.shape[0]},{self.input.shape[1]})")
            print(f"activation shape: {activation_result.shape}")
        return activation_result

    def backward(self, output_gradient): # output_gradient is the loss gradient with respect to output
        # Compute the gradient of the activation function with respect to the layer's input
        activation_gradient = self.activation.prime(self.z)

        if self.debug:
            print("\nComputing the weigths_gradient with:")
            print(f"output_gradient: {output_gradient.shape}")
            print(f"activation_gradient: {activation_gradient.shape}")
            print(f"input shape: {self.input.shape}")
            print(f"weights shape: {self.weights.shape}")

        loss_activation = output_gradient * activation_gradient
        weights_gradient = np.dot(self.input, loss_activation.T)
        bias_gradient = np.sum(loss_activation, axis=0, keepdims=True)
        # Update the weights and biases using gradient descent
        self.weights -= self.learning_rate * weights_gradient
        self.bias -= self.learning_rate * bias_gradient
        
        # Compute and return the gradient with respect to the input of this layer
        input_gradient = np.dot(self.weights, loss_activation)
        return input_gradient

    def __str__(self):
        return f"{self.layer_type}(n_input: {self.m_input}, n_output: {self.n_output}, activation: {self.activation.name})"

    def __repr__(self):
        return self.__str__()

class Dense(Layer):

    def __init__(self, m_input, n_output, activation : Activation):
        super().__init__(m_input, n_output, activation)
        self.layer_type = "Dense"