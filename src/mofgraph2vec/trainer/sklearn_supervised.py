from loguru import logger
from omegaconf import DictConfig
from hydra.utils import instantiate

import numpy as np
from mofgraph2vec.utils.seed import set_seed
from mofgraph2vec.utils.loss import get_numpy_regression_metrics
from mofgraph2vec.data.datamodule import DataModuleFactory

def train(
    config: DictConfig
):  
    set_seed(config.seed)

    dm = DataModuleFactory(**config.data.nn, device="cpu")

    train_ds = dm.get_train_dataset()
    valid_ds = dm.get_valid_dataset()
    x_train = np.concatenate((train_ds.vectors.numpy(), valid_ds.vectors.numpy()), axis=0)
    y_train = np.concatenate((train_ds.labels, valid_ds.labels), axis=0)

    test_ds = dm.get_test_dataset()
    x_test = test_ds.vectors.numpy()
    y_test = test_ds.labels

    logger.info(f"Start fitting xgbt model. ")
    regressor = instantiate(config.model.sklearn)
    regressor.fit(x_train, y_train)
    metrics = regressor.test(x_test, y_test, dm.target_transform)

    return regressor, metrics