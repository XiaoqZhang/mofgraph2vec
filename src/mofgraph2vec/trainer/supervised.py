from loguru import logger
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from mofgraph2vec.utils.seed import set_seed
from mofgraph2vec.data.datamodule import DataModuleFactory
from mofgraph2vec.model.nn_lightning import VecLightningModule

def train(
    config: DictConfig
):  
    set_seed(config.seed)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    dm = DataModuleFactory(**config.data.nn, device="cpu")
    datamodule = dm.get_datamodule()

    logger.info(f"Instantiate neural network model. ")
    vec_model = instantiate(config.model.nn).to(device)
    pl_model = VecLightningModule(vec_model, loss=config.model.loss, lr=config.model.lr)

    logger.info(f"config trainer: {config.trainer}")

    callbacks = []
    if config.model.patience >= 0:
        callbacks.append(EarlyStopping(monitor="valid_loss", patience=config.model.patience))
    if config.model.checkpoint:
        callbacks.append(ModelCheckpoint(monitor="valid_loss"))
    
    trainer = instantiate(config.trainer, callbacks=callbacks)

    if trainer.auto_lr_find:
        trainer.tune(pl_model, datamodule=datamodule)

    logger.info(f"Start fitting")
    trainer.fit(pl_model, datamodule=datamodule)

    logger.info(f"Start testing")
    test_metrics = trainer.test(pl_model, datamodule=datamodule)
    
    return pl_model, test_metrics[0]