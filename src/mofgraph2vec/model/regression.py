from loguru import logger
from omegaconf import DictConfig
from hydra.utils import instantiate

import numpy as np
from sklearn.model_selection import cross_val_score
from mofgraph2vec.data.datamodule import DataModuleFactory

def run_regression(
    config: DictConfig
):
    config.doc2label_model.random_state = config.seed
    dm = DataModuleFactory(**config.doc2label_data, device="cpu")

    train_ds = dm.get_train_dataset()
    valid_ds = dm.get_valid_dataset()
    x_train = np.concatenate((train_ds.vectors, valid_ds.vectors), axis=0)
    y_train = np.concatenate((train_ds.labels, valid_ds.labels), axis=0)

    test_ds = dm.get_test_dataset()
    x_test = test_ds.vectors
    y_test = test_ds.labels

    logger.info(f"Start fitting xgbt model. ")
    regressor = instantiate(config.doc2label_model)
    regressor.fit(x_train, y_train)
    #scores = cross_val_score(regressor, x_train, y_train, cv=5, n_jobs=4)
    metrics = regressor.test(x_test, y_test, dm.transform, dm.target_transform)
    """
    metrics.update({
        "cv_mean": np.mean(scores),
        "cv_std": np.std(scores)
    })
    """

    return regressor, metrics