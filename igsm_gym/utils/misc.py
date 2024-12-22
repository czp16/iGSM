from typing import List, Optional
import numpy as np
import random

def softmax(x: np.ndarray) -> np.ndarray:
    '''
    Compute the softmax of vector x.
    '''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def random_select_and_remove(lst: List):
    if not lst:
        return None
    index = random.randrange(len(lst))
    lst[index], lst[-1] = lst[-1], lst[index]
    return lst.pop()