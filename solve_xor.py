import numpy as np
from network import Network
from layers import Dense, Layer
from activation_layers import ActivationLayer, sigmoid, sigmoid_prime, tanh, tanh_prime
from losses import mse, mse_prime

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(Dense(2, 8))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(Dense(8, 1))
net.add(ActivationLayer(tanh, tanh))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=10000, learning_rate=0.01)


for x in x_train:
    output = net.predict(x)
    print(f"Input: {x} -> Predicted Output: {output}")
