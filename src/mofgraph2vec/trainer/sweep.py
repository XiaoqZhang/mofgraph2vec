"""Orchestrates the sweep of models based on hydra configs."""
from copy import deepcopy
from functools import partial

import wandb
from loguru import logger
from omegaconf import OmegaConf

from mofgraph2vec.trainer.unsupervised import train


def sweep(config):
    sweep_config = OmegaConf.to_container(config.sweep.config)
    train_wrapper_curried = partial(train, config=deepcopy(config), sweep=True)
    if config.sweep.id:
        logger.debug(f"Resuming sweep {config.sweep.id}")
        sweep_id = config.sweep.id
    else:
        logger.debug("No sweep id provided, creating new sweep")
        sweep_id = wandb.sweep(
            sweep_config, project=config.logger.project, entity=config.logger.entity
        )
    wandb.agent(sweep_id, function=train_wrapper_curried, count=config.sweep.count)