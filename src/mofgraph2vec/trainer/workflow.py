import os
import torch
import wandb
from omegaconf import DictConfig
from loguru import logger
from mofgraph2vec.utils.dict_helpers import get, put
from mofgraph2vec.utils.seed import set_seed
from mofgraph2vec.mof2doc.unsupervised import train as unsupervised_train
from mofgraph2vec.trainer.supervised import train as supervised_train

def train(config: DictConfig, sweep: bool=False):
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
    
        set_seed(config.seed)
    
        config.data.nn.embedding_path = os.path.join(wandb.run.dir, "../tmp/embedding.csv")
        config.model.nn.input_dim = config.model.gensim.vector_size
        logger.info(f"{config}")
        unsupervised_metrics = unsupervised_train(config, os.path.join(wandb.run.dir, "../tmp/"))
        model, supervised_metrics = supervised_train(config)
        torch.save(model.model.state_dict(), os.path.join(wandb.run.dir, "../tmp/best_model.pt"))

        wandb.log(unsupervised_metrics)
        wandb.log(supervised_metrics)