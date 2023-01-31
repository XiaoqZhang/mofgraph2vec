import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from mofgraph2vec.utils.dict_helpers import get, put
from mofgraph2vec.data.datamodule import DataModuleFactory
from mofgraph2vec.model.nn_lightning import VecLightningModule

def train(
    config: DictConfig,
    sweep: bool = False
):
    with wandb.init(
        project=config.logger.project,
        entity=config.logger.entity,
        mode=config.logger.mode,
    ) as run:
        if sweep:
            sweep_config = run.config
            try:
                for key, value in sweep_config.items():
                    logger.debug(f"Overriding {key} with {value} for sweep.")
                    put(config, key, value)
                    assert get(config, key) == value
                logger.debug("Completed overriding config for sweep.")
            except Exception as e:
                logger.exception(f"Error {e} trying to set key {key}")
    
        dm = DataModuleFactory(**config.data.nn)

        train_ds = dm.get_train_dataset()
        valid_ds = dm.get_valid_dataset()
        test_ds = dm.get_test_dataset()
    
        train_loader = DataLoader(train_ds, batch_size=config.model.nn.batch_size)
        valid_loader = DataLoader(valid_ds, batch_size=config.model.nn.batch_size)
        test_loader = DataLoader(test_ds, batch_size=config.model.nn.batch_size)

        logger.info(f"Instantiate neural network model. ")
        pl_model = instantiate(config.model.nn)

        trainer = pl.Trainer(max_epochs=config.model.nn.max_epochs)

        logger.info(f"Start fitting")
        trainer.fit(pl_model, train_loader, valid_loader)

        logger.info(f"Start testing")
        test_metrics = trainer.test(pl_model, test_loader)
        wandb.log(test_metrics[0])
