import numpy as np
from layers import Layer


class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    tanh_x = tanh(x)
    return 1 - np.square(tanh_x)
