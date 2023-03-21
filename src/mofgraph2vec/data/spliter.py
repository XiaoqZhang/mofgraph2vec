import numpy as np
import pandas as pd
from typing import Sequence
from sklearn.model_selection import train_test_split
from numpy.typing import ArrayLike

def quantile_binning(values: ArrayLike, q: Sequence[float]) -> np.array:
    """Use :py:meth:`pandas.qcut` to bin the values based on quantiles."""
    values = pd.qcut(values, q, labels=np.arange(len(q) - 1)).astype(int)
    return values

def train_valid_test_split(df, train_frac, valid_frac, test_frac, stratification_col, seed):
    stratification = quantile_binning(df.loc[:, stratification_col].values.reshape(-1,), np.array([0, 0.25, 0.5, 0.75, 1]))
    train_valid_idx, test_idx = train_test_split(range(len(df)), test_size=test_frac, stratify=stratification, random_state=seed)
    train_idx, valid_idx = train_test_split(train_valid_idx, test_size=valid_frac, stratify=stratification[train_valid_idx], random_state=seed)

    return train_idx, valid_idx, test_idx
