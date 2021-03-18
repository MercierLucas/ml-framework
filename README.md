# Machine learning framework

This "framework" is for learning purpose only.


## Usage

```python
    from ml_framework.network import Network
    from ml_framework.Layers import Dense
    
    self.net = Network()
    self.net.sequential([
        Dense(2, 3, activation="tanh"),
        Dense(3, 1, activation="tanh"),
    ])
```