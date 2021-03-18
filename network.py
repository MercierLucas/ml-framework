import time
from ml_framework.Losses.losses import mse,mse_prime

class Network:
    def __init__(self, loss="mse"):
        available_losses = ["mse"]
        assert loss in available_losses, "{loss} isn't a valid loss"

        self.layers = []
        self.lr = 0
        self.epochs = 0
        
    def show_architecture(self):
        print(" > ".join([l.get_details() for l in self.layers]))
        
    def add(self, layer):
        self.layers.append(layer)
        
    def sequential(self, layers):
        self.layers = layers
        
    def _forward(self, x):
        output = x 
        for layer in self.layers:
            output = layer.forward(output)
        return output
        
    def train(self, X, Y, lr = 0.001, epochs = 100, verbose=False):
        t = time.time()
        for i in range(epochs):
            error = 0
            for x,y in zip(X,Y):
                output = self._forward(x)
                error += mse(y,output)
                output = mse_prime(y,output)
                for layer in reversed(self.layers):
                    output = layer.backward(output,lr)
                    
            if verbose:      
                print(f"Error after epoch {(i+1)} : {error}")

        tot_time = time.time() - t
        print(f"Training ended in {tot_time:.2f} with error {error}")
    
    def predict(self, x):
        return self._forward(x)