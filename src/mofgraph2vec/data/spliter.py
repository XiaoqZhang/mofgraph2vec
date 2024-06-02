import numpy as np
import pandas as pd
from typing import Sequence
from sklearn.model_selection import train_test_split
from numpy.typing import ArrayLike

def quantile_binning(values: ArrayLike, q: Sequence[float]) -> np.array:
    """bin the values into quantiles

    Args:
        values (ArrayLike): the values to bin
        q (Sequence[float]): the quantiles

    Returns:
        np.array: the binned values
    """
    values = pd.qcut(values, q, labels=np.arange(len(q) - 1)).astype(int)
    return values

def train_test_stratified_split(df, train_frac, test_frac, stratification_col, seed):
    """split the dataframe into train and test sets with stratification

    Args:
        df (pandas.DataFrame): the dataframe to split
        train_frac (float): the fraction of the training set
        test_frac (float): the fraction of the test set
        stratification_col (str): the column to stratify the split
        seed (int): random seed

    Returns:
        List[int], List[int]: the list of train and test indices
    """
    stratification = quantile_binning(df.loc[:, stratification_col].values.reshape(-1,), np.array([0, 0.25, 0.5, 0.75, 1]))
    train_idx, test_idx = train_test_split(range(len(df)), test_size=test_frac, stratify=stratification, random_state=seed)

    return train_idx, test_idx
