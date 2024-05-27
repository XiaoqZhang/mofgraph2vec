import random
import numpy as np

def set_seed(seed):
    """ Set seed for reproducibility. """
    random.seed(seed)
    np.random.seed(seed)