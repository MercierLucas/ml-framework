class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.name = None
        
    def get_details(self):
        pass
        
    def forward(self, input):
        pass
    
    def backward(self, grad, lr):
        pass