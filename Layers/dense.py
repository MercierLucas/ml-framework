import numpy as np
from ml_framework.Layers.layer import Layer
from ml_framework.Layers.Activations import Tanh, Sigmoid

class Dense(Layer):
    def __init__(self, input_size, output_size, activation = "tanh"):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size,input_size)
        self.bias = np.random.randn(output_size,1)
        self.name = "Dense"
        
        self.available_activations = {
            'tanh' : Tanh(),
            'sigmoid' : Sigmoid()
        }

        self._setup_activation(activation)

    def _setup_activation(self, activation):
        assert activation in self.available_activations, f"Activation not found, please select one of {self.available_activations}"

        self.activation = self.available_activations[activation]
        
    def forward(self, input):
        self.input = input
        forward = np.dot(self.weights,input) + self.bias
        return self.activation.forward(forward)
    
    def get_details(self):
        return f"Dense ({self.input_size},{self.output_size})"
    
    def backward(self, grad, lr):
        grad = self.activation.backward(grad)
        weight_grad = np.dot(grad,self.input.T)
        self.weights -= weight_grad * lr
        self.bias -= lr * grad
        return np.dot(self.weights.T,grad)