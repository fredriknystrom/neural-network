import logging
import matplotlib.pyplot as plt

from logger_config import setup_logging
setup_logging()

class NN():

    def __init__(self, layers, epochs, loss_function, learning_rate, regular=None, debug=False):
        self.layers = layers
        self.epochs = epochs
        self.loss_function = loss_function
        self.regular = regular
        self.debug = debug
        self.learning_rate = learning_rate
        self.check_layers_matching()
        self.set_learning_rate()
        self.set_debug()
        logging.info(self.__str__())
        

    def set_learning_rate(self):
        for layer in self.layers:
            layer.set_learning_rate(self.learning_rate)
    
    def set_debug(self):
        for layer in self.layers:
            layer.set_debug(self.debug)
        
    def check_layers_matching(self):
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]

            assert current_layer.n_output == next_layer.m_input, f"Output of {current_layer.info()} and input of {next_layer.info()} don't match"

    def train(self, X, Y, plot=False):
        self.errors = []
        for epoch in range(self.epochs):
            batch_error = 0
            for x, y in zip(X, Y):
                # Forward propagation
                if self.debug:
                    logging.debug("--- Forwarding start ---")
                output = x.reshape(-1, 1) # make sure matrix shape is (m,1)
                for layer in self.layers:
                    output = layer.forward(output)
                if self.debug:
                    logging.debug("--- Forwarding end ---\n")


                # Calculate error for input with our loss function
                error = self.loss_function.loss(y, output)
                # Add the error to the batch error
                batch_error += error
            
                # Back propagation
                if self.debug:
                    logging.debug("---- Backpropagation start ---")
                
                y = y.reshape(-1,1)
                loss_gradient = self.loss_function.prime(y, output)
                for layer in reversed(self.layers):
                    loss_gradient = layer.backward(loss_gradient)

                if self.debug:
                    logging.debug("--- Backpropagation end ---\n")

            avg_batch_error = batch_error / len(X)
            if self.debug:
                logging.debug(f"avg_batch_error: {avg_batch_error}")
            self.errors.append(avg_batch_error)
        
        if plot:
            self.plot_error()


    def predict():
        pass

    def __repr__(self):
        s = "\nNeural network layers:\n"
        for layer in self.layers:
            s += str(layer) + "\n"
        return s

    def plot_error(self):
        plt.plot(range(len(self.errors)), self.errors)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Training Error')
        plt.show()