import numpy as np

from ml_framework.network import Network
from ml_framework.Layers import Dense

class Xor_solver:
    def __init__(self):
        self.net = Network()
        self.net.sequential([
            Dense(2, 3, activation="tanh"),
            Dense(3, 1, activation="tanh"),
        ])
        self.net.show_architecture()
    
    def train(self):
        X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
        Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
        self.net.train(X,Y,epochs = 10000, lr = 0.01, verbose = False, digits = 3, metrics = ['recall','precision'])

    def test(self):
        test = np.reshape([[0,0]],(1,2,1))
        print(self.net.predict(test))

    def evaluate(self):
        X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
        Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
        self.net.evaluate(X,Y)


