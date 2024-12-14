from typing import List
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    '''
    Compute the softmax of vector x.
    '''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
