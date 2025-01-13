from typing import List, Dict, Optional, Sequence
import numpy as np
import random
# import torch
import os

def softmax(x: np.ndarray) -> np.ndarray:
    '''
    Compute the softmax of vector x.
    '''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def seed_all(seed=1029, others: Optional[list] = None):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # if others is not None:
    #     if hasattr(others, "seed"):
    #         others.seed(seed)
    #         return True
    #     try:
    #         for item in others:
    #             if hasattr(item, "seed"):
    #                 item.seed(seed)
    #     except:
    #         pass

def random_select_and_remove(lst: List):
    if not lst:
        return None
    index = random.randrange(len(lst))
    lst[index], lst[-1] = lst[-1], lst[index]
    return lst.pop()
