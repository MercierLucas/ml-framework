import numpy as np


def to_categorical(Y : np.ndarray, num_classes : int = -1):
    """Encode as a one hot encoder"""

    encoded = []
    mapped_y = dict()
    index = 0
    
    if num_classes == -1:
        num_classes = len(list(set(Y)))
    
    for y in Y:
        if y in mapped_y:
            encoded.append(mapped_y[y])
            continue
        
        vec = np.zeros(num_classes)
        vec[index] = 1
        encoded.append(vec)
        mapped_y[y] = vec
        index += 1
            
    return np.array(encoded)

def from_categorical(Y : np.ndarray):
    """ Decode one hot encoder """
    res = []
    for y in Y:
        res.append(y.argmax())

    return np.array(res)
