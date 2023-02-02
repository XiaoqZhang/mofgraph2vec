import hydra
from omegaconf import DictConfig
import wandb
from loguru import logger
from copy import deepcopy
from functools import partial

import wandb
from loguru import logger
from omegaconf import OmegaConf
from mofgraph2vec.trainer.supervised import train
from mofgraph2vec.utils.dict_helpers import get, put

@hydra.main(config_path="../../conf", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    with wandb.init(
        project=config.logger.project,
        entity=config.logger.entity,
        mode=config.logger.mode,
    ) as run:
        sweep_config = run.config
        logger.info(f"Sweep config: {sweep_config}")
        try:
            for key, value in sweep_config.items():
                logger.debug(f"Overriding {key} with {value} for sweep.")
                put(config, key, value)
                assert get(config, key) == value
            logger.debug("Completed overriding config for sweep.")
        except Exception as e:
            logger.exception(f"Error {e} trying to set key {key}")
    
    if config.supervised_sweep.id:
        logger.debug(f"Resuming sweep {config.sweep.id}")
        sweep_id = config.supervised_sweep.id
    else:
        logger.debug("No sweep id provided, creating new sweep")
        sweep_id = wandb.sweep(
            sweep_config, project=config.logger.project, entity=config.logger.entity
        )
    
    sweep_config = OmegaConf.to_container(config.supervised_sweep.config)
    train_wrapper_curried = partial(train, config=deepcopy(config))
    # wandb.init every run
    wandb.agent(sweep_id, function=train_wrapper_curried, count=config.supervised_sweep.count)
        

if __name__ == "__main__":
    main()