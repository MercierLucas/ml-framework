import numpy as np

import time
from ml_framework.Losses.losses import mse,mse_prime
from ml_framework.Metrics import Metrics

class Network:
    def __init__(self):
        self.layers = []
        self.lr = 0
        self.epochs = 0
        
    def show_architecture(self):
        return " > ".join([l.get_details() for l in self.layers])
        
    def add(self, layer):
        self.layers.append(layer)
        
    def sequential(self, layers):
        self.layers = layers
        
    def _forward(self, x):
        output = x 
        for layer in self.layers:
            output = layer.forward(output)
        return output
        
    def train(self, X, Y, lr = 0.001, epochs = 100, verbose = False, metrics = ['recall'], digits = 6):
        full_error = []
        for i in range(epochs):
            error = 0
            for x,y in zip(X,Y):
                output = self._forward(x)
                error += mse(y,output)
                output = mse_prime(y,output)
                
                for layer in reversed(self.layers):
                    output = layer.backward(output,lr)
                    
            pred = self.predict(X)
            metric = Metrics(Y, pred)
            error = error/len(x)
            full_error.append(error)
                    
            if verbose:
                computed_metrics = metric.compute_many(metrics)
                report = f"[Epoch {i+1}] Error: {error:.{digits}f}"
                for m in computed_metrics:
                    report+= f" {m}:{computed_metrics[m]:.{digits}f}"
                        
                print(report)
                
        print(f"Training ended with error {error:.{digits}f}")
        return full_error
    
    def evaluate(self, X, Y, digits = 5):
        pred = self.predict(X)
        metrics = Metrics(Y, pred)
        metrics.report(digits = digits)
    
    def predict(self, X, threshold = .9):
        res = []
        for x in X:
            pred = self._forward(x)
            #if len(pred) > 1:
            #    pred = pred.argmax()
            res.append(pred)
        return np.array(res).squeeze()