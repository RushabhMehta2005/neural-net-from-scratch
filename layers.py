import numpy as np

class Layer:
    """Base class for all layers."""
    def __init__(self):
        self.inputs = None
        self.outputs = None
    
    def forward_propagation(self, input):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    """Fully-connected layer."""
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        self.biases = np.random.rand(1, output_size)
    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * output_error
        return input_error
