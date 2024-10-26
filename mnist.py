import numpy as np
from network import Network
from layers import Dense, Layer
from activation_layers import ActivationLayer
from activation_layers import tanh, tanh_prime, sigmoid, sigmoid_prime
from losses import mse, mse_prime



nn = Network()

nn.add(Dense(28 * 28, 128))
nn.add(ActivationLayer(tanh, tanh_prime))
nn.add(Dense(128, 64))
nn.add(ActivationLayer(sigmoid, sigmoid_prime))
nn.add(Dense(64, 10))
nn.add(ActivationLayer(tanh, tanh_prime))

nn.use(mse, mse_prime)
