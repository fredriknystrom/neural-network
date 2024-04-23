import logging
import time
import matplotlib.pyplot as plt
import numpy as np

from nn.logger_config import change_logging_level


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

            assert current_layer.n_output == next_layer.m_input, f"Output of {current_layer.info()} and input of {next_layer.info()} does not match"

    def compute_total_regularization_loss(self):
        total_reg_loss = 0
        for layer in self.layers:
            reg_loss = layer.compute_regularization_loss()
            total_reg_loss += reg_loss
        return total_reg_loss

    def train(self, X_train, Y_train, X_val, Y_val, batch_size=1, plot=False):
        start = time.time()
        self.loss = []
        self.val_loss = []
        self.accuracy = []
        num_samples = X_train.shape[0]
        batches = int(np.ceil(num_samples / batch_size))

        logging.debug(f"batch_size: {batch_size}")
        logging.debug(f"num_samples: {num_samples}")
        logging.debug(f"num_samples: {num_samples}")
        logging.debug(f"batches: {batches}")

        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_correct_predictions = 0

            # Shuffle dataset at the start of each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            for batch_num in range(batches):
                logging.info(f"Epoch: {epoch}/{self.epochs} - Batch: {batch_num}/{batches}")
                start_idx = batch_num * batch_size
                end_idx = start_idx + batch_size
                x_batch = X_shuffled[start_idx:end_idx]
                y_batch = Y_shuffled[start_idx:end_idx]

                batch_loss = 0
                batch_loss_gradient = 0
                
                for x, y in zip(x_batch, y_batch):
                    logging.debug(f"x shape: {x.shape}")
                    logging.debug(f"y shape: {y.shape}")
                    # Forward propagation for the batch
                    logging.debug(f"--- Forwarding start (batch {batch_num}) ---")
                    output = x.reshape(-1, 1)
                    for layer in self.layers:
                        output = layer.forward(output)
                    logging.debug(f"--- Forwarding end (batch {batch_num}) ---\n")
                    logging.debug(f"output.shape: {output.shape}")
                    logging.debug(f"output value: {output}")
                    logging.debug(f"y: {y}")
                    # Calculate error for the batch with our loss function
                    loss = self.loss_function.loss(y, output)
                    regularization_loss = self.compute_total_regularization_loss()
                    logging.debug(f"regularization_loss: {regularization_loss}")
                    logging.debug(f"loss: {loss}")
                    combined_loss = loss + regularization_loss
                    logging.debug(f"combined_loss: {combined_loss}")
                    batch_loss += combined_loss
                    batch_loss_gradient += self.loss_function.prime(y, output)

                    # Calculate predictions and update correct_count
                    predicted_class = np.argmax(output, axis=0)[0]
                    if predicted_class == y:
                        epoch_correct_predictions += 1

                epoch_loss += batch_loss
                # Back propagation for the batch
                logging.debug("---- Backpropagation start (batch) ---")
                logging.debug(f"batch_loss_gradient: {batch_loss_gradient}")
                avg_loss_gradient = batch_loss_gradient/len(x_batch)
                logging.debug(f"len(x_batch): {len(x_batch)}")
                logging.debug(f"avg_loss_gradient: {avg_loss_gradient}")
                for layer in reversed(self.layers):
                    avg_loss_gradient = layer.backward(avg_loss_gradient)
                logging.debug("--- Backpropagation end (batch) ---\n")
             

            avg_epoch_loss = epoch_loss/num_samples
            self.loss.append(avg_epoch_loss)
            epoch_accuracy = epoch_correct_predictions/num_samples
            self.accuracy.append(epoch_accuracy)
            avg_val_loss = self.evaluate(X_val, Y_val)
            self.val_loss.append(avg_val_loss)

            print(f"Correct pred: {epoch_correct_predictions}")
            logging.info(f"Epoch {epoch}  - accuracy: {epoch_accuracy} - loss: {avg_epoch_loss} - val_loss: {avg_val_loss}")
            print(f"Epoch {epoch} - accuracy: {epoch_accuracy} - loss: {avg_epoch_loss}")

        if plot:
            self.plot()
        logging.info(f"Total training time: {round(time.time() - start, 3)}s")

    def evaluate(self, test_X, test_Y):
        """ Evaluate the model on test data """
        #print('test_X', test_X.shape)
        #print('test_Y', test_Y.shape)
        #print(test_X[0].shape)
        # NOTE: reshape(-1,1) prob doesnt work for any dim here....
        predictions = np.array([self.predict(x.reshape(-1,1)) for x in test_X])
        loss = self.loss_function.loss(test_Y, predictions)
        return loss

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def get_weights(self):
        return [(layer.layer_type, layer.get_weights(), layer.get_biases()) for layer in self.layers]

    def __repr__(self):
        s = "\nNeural network layers:\n"
        for layer in self.layers:
            s += str(layer) + "\n"
        return s

    def plot(self):
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.loss, label='Loss')
        #plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy, label='Accuracy')
        #plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()