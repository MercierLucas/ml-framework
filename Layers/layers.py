import numpy as np
from ml_framework.Layers.layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size,input_size)
        self.bias = np.random.randn(output_size,1)
        self.name = "Dense"
        
    def forward(self, input):
        self.input = input
        return np.dot(self.weights,input)+self.bias
    
    def get_details(self):
        return f"Dense ({self.input_size},{self.output_size})"
    
    def backward(self, grad, lr):
        weight_grad = np.dot(grad,self.input.T)
        self.weights -= weight_grad * lr
        self.bias -= lr * grad
        return np.dot(self.weights.T,grad)