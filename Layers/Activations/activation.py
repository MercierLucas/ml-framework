import numpy as np
from ml_framework.Layers.layer import Layer

class Activation(Layer):
    def __init__(self,activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        
    def forward(self, input):
        self.input = input
        return self.activation(input)
    
    def backward(self, grad):
        return np.multiply(grad,self.activation_prime(self.input))

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1-np.tanh(x)**2
        self.name = "Tanh"
        super().__init__(tanh, tanh_prime)
        
    def get_details(self):
        return "Tanh"

class Sigmoid(Activation):
    def __init__(self):
        sig = lambda x: 1/(1+np.exp(x))
        sig_prime = lambda x: sig(x)-(1-sig(x))
        super().__init__(sig, sig_prime)