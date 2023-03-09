import os
from loguru import logger
from omegaconf import DictConfig
from hydra.utils import instantiate

import numpy as np
import pandas as pd
import torch
from mofgraph2vec.data.datamodule import DataModuleFactory
from mofgraph2vec.utils.loss import get_numpy_regression_metrics
from mofgraph2vec.model.vecnn import VecModel
from mofgraph2vec.model.nn_lightning import VecLightningModule
from mofgraph2vec.utils.loss import get_numpy_regression_metrics

def run_regression(
        config: DictConfig
):
    config.doc2label_model.random_state = config.seed
    dm = DataModuleFactory(**config.doc2label_data)

    pl_model = VecLightningModule(instantiate(config.doc2label_model.nn), config.doc2label_model.loss, config.doc2label_model.lr)
    trainer = instantiate(config.trainer)

    trainer.tune(pl_model, datamodule=dm.get_datamodule())
    trainer.fit(pl_model, datamodule=dm.get_datamodule())

    train_predictions = trainer.predict(pl_model, dataloaders=dm.train_dataloader())
    train_true = np.concatenate([pred[0].view(-1).numpy() for pred in train_predictions])
    train_pred = np.concatenate([pred[1].view(-1).numpy() for pred in train_predictions])
    train_table = np.vstack((train_true, train_pred)).T

    test_predictions = trainer.predict(pl_model, dataloaders=dm.test_dataloader())
    test_true = np.concatenate([pred[0].view(-1).numpy() for pred in test_predictions])
    test_pred = np.concatenate([pred[1].view(-1).numpy() for pred in test_predictions])
    test_table = np.vstack((test_true, test_pred)).T

    metrics = get_numpy_regression_metrics(train_true, train_pred, "train")
    metrics.update(
        get_numpy_regression_metrics(test_true, test_pred, prefix="test")
    )

    # use the trained model to get new embeddings
    ori_feat = pd.read_csv(config.doc2label_data.embedding_path).set_index('type')
    new_feat = pl_model.model.get_embedding(torch.Tensor(ori_feat.values)).detach().numpy()
    ori_feat.loc[:] = new_feat
    ori_feat.to_csv(os.path.join(os.path.dirname(config.doc2label_data.embedding_path),
                                 "embedding-%s.csv" %config.doc2label_data.task[0]))

    return pl_model, metrics, test_table