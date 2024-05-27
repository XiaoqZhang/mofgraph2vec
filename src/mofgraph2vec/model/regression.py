from loguru import logger
from omegaconf import DictConfig, OmegaConf

import numpy as np
from mofgraph2vec.data.datamodule import DataModuleFactory
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from mofgraph2vec.utils.loss import get_numpy_regression_metrics

def run_regression(
    config: DictConfig
):
    """ Run regression model. """
    dm = DataModuleFactory(**config.doc2label_data)

    # Load the data
    train_ds = dm.get_train_dataset()
    x_train = train_ds.vectors
    y_train = train_ds.labels

    test_ds = dm.get_test_dataset()
    x_test = test_ds.vectors
    y_test = test_ds.labels

    # Train the supervised model
    if config.doc2label_model.grid_search:
        # convert the config to a dictionary and remove None values
        search_params = OmegaConf.to_container(config.doc2label_model.params)
        search_params = {k: v for k, v in search_params.items() if v is not None}
        regressor = GridSearchCV(
            XGBRegressor(),
            param_grid=search_params,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
        )
    else:
        regressor = XGBRegressor(
            **config.doc2label_model.params, 
            n_jobs=config.doc2label_model.n_jobs, 
            random_state=config.seed
        )
    logger.info(f"x shape: {x_train.shape}; y shape: {y_train.shape}")
    logger.info(f"Start fitting xgbt model. ")
    regressor.fit(x_train, y_train)
    if config.doc2label_model.grid_search:
        logger.info(f"Best params: {regressor.best_params_}")

    # Evaluate the model
    logger.info(f"Evaluating the model. ")
    train_pred = regressor.predict(x_train)
    test_pred = regressor.predict(x_test)
    if dm.target_transform is not None:
        test_true = dm.target_transform.inverse_transform(x_test)
        test_pred = dm.target_transform.inverse_transform(x_test)
        train_pred = dm.target_transform.inverse_transform(x_train)
    else: 
        test_true = y_test.flatten()
        train_true = y_train.flatten()
    
    metrics = get_numpy_regression_metrics(y_train, regressor.predict(x_train), "train")
    metrics.update(get_numpy_regression_metrics(test_true, test_pred, "test"))
    train_parity = {
        'train_true': train_true,
        'train_pred': train_pred
    }
    test_parity = {
        'test_true': test_true,
        'test_pred': test_pred
    }

    return regressor, metrics, train_parity, test_parity