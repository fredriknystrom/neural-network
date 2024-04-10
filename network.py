import logging
import matplotlib.pyplot as plt
import numpy as np

from logger_config import setup_logging, change_logging_level
setup_logging()

class NN():

    def __init__(self, layers, epochs, loss_function, learning_rate, regular=None, debug=False):
        self.layers = layers
        self.epochs = epochs
        self.loss_function = loss_function
        self.regular = regular
        self.learning_rate = learning_rate
        self.check_layers_matching()
        self.set_learning_rate()
        self.set_debug(debug)
        logging.info(self.__str__())
        

    def set_learning_rate(self):
        for layer in self.layers:
            layer.set_learning_rate(self.learning_rate)
    
    def set_debug(self, debug):
        if debug:
            change_logging_level(logging.DEBUG)
        
    def check_layers_matching(self):
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]

            assert current_layer.n_output == next_layer.m_input, f"Output of {current_layer.info()} and input of {next_layer.info()} don't match"

    def train(self, X, Y, batch_size=1, plot=False, ):
        batch_size = 2  # Define your batch size here
        num_samples = X.shape[0]
        batches = int(np.ceil(num_samples / batch_size))
        logging.debug(f"batch_size: {batch_size}")
        logging.debug(f"num_samples: {num_samples}")
        logging.debug(f"batches: {batches}")
        
        self.errors = []
        total_batches_processed = 0  # Initialize counter for total batches processed

        for epoch in range(self.epochs):
            # Shuffle the dataset at the beginning of each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            for batch_num in range(batches):
                logging.info(f"Epoch: {epoch}/{self.epochs} - Batch: {batch_num}/{batches}")
                batch_error = 0
                batch_loss_gradient = 0
                start_idx = batch_num * batch_size
                end_idx = start_idx + batch_size
                x_batch = X_shuffled[start_idx:end_idx]
                y_batch = Y_shuffled[start_idx:end_idx]
                logging.debug(f"x_batch: {x_batch}")
                logging.debug(f"y_batch: {y_batch}")
                for x, y in zip(x_batch, y_batch):
                    logging.debug(f"x: {x}")
                    logging.debug(f"y: {y}")
                    # Forward propagation for the batch
                    logging.debug("--- Forwarding start (batch) ---")
                    output = x.reshape(-1, 1)
                    for layer in self.layers:
                        output = layer.forward(output)
                    logging.debug("--- Forwarding end (batch) ---\n")

                    # Calculate error for the batch with our loss function
                    error = self.loss_function.loss(y, output)
                    batch_error += error
                    batch_loss_gradient += self.loss_function.prime(y_batch, output)

                # Back propagation for the batch
                logging.debug("---- Backpropagation start (batch) ---")
                avg_loss_gradient = batch_loss_gradient/len(x_batch)
                for layer in reversed(self.layers):
                    avg_loss_gradient = layer.backward(avg_loss_gradient)
                logging.debug("--- Backpropagation end (batch) ---\n")
                
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