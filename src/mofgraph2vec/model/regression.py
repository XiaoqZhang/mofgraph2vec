from loguru import logger
from omegaconf import DictConfig
from hydra.utils import instantiate

import numpy as np
from sklearn.model_selection import cross_val_score
from mofgraph2vec.data.datamodule import DataModuleFactory
from xgboost import XGBRegressor
from mofgraph2vec.utils.loss import get_numpy_regression_metrics

def run_regression(
    config: DictConfig
):
    config.doc2label_model.random_state = config.seed
    dm = DataModuleFactory(**config.doc2label_data, device="cpu")

    train_ds = dm.get_train_dataset()
    valid_ds = dm.get_valid_dataset()
    x_train = np.concatenate((train_ds.vectors.numpy(), valid_ds.vectors.numpy()), axis=0)
    y_train = np.concatenate((train_ds.labels.numpy(), valid_ds.labels.numpy()), axis=0)

    test_ds = dm.get_test_dataset()
    x_test = test_ds.vectors.numpy()
    y_test = test_ds.labels.numpy()

    regressor = XGBRegressor(**config.doc2label_model)
    logger.info(f"Start fitting xgbt model. ")
    regressor.fit(x_train, y_train)
    #scores = cross_val_score(regressor, x_train, y_train, cv=5, n_jobs=4)
    
    """
    metrics.update({
        "cv_mean": np.mean(scores),
        "cv_std": np.std(scores)
    })
    """
    logger.info(f"Evaluating the model. ")
    pred = regressor.predict(x_test)
    if dm.target_transform is not None:
        true = dm.target_transform.inverse_transform(x_test)
        pred = dm.target_transform.inverse_transform(x_test)
    else: 
        true = y_test.flatten()
    
    metrics = get_numpy_regression_metrics(y_train, regressor.predict(x_train), "train")
    metrics.update(get_numpy_regression_metrics(true, pred, "test"))
    fig_data = [[x, y] for (x, y) in zip(true, pred)]

    return regressor, metrics, fig_data