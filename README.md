# Machine learning framework

This "framework" is for learning purpose only.


## Usage

```python
    from ml_framework.network import Network
    from ml_framework.Layers import Dense

    net = Network()
    net.sequential([
        Dense(2, 3, activation="tanh"),
        Dense(3, 1, activation="tanh"),
    ])
```


## To implement

[ ] Dataloader and dataset utils
