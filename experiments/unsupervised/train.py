import hydra
from omegaconf import DictConfig
import wandb
from loguru import logger

from mofgraph2vec.trainer.unsupervised import train
from mofgraph2vec.utils.dict_helpers import get, put

@hydra.main(config_path="../../conf", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    with wandb.init(
        project=config.logger.project,
        entity=config.logger.entity,
        mode=config.logger.mode,
    ) as run:
        if config.sweep:
            sweep_config = run.config
            try:
                for key, value in sweep_config.items():
                    logger.debug(f"Overriding {key} with {value} for sweep.")
                    put(config, key, value)
                    assert get(config, key) == value
                logger.debug("Completed overriding config for sweep.")
            except Exception as e:
                logger.exception(f"Error {e} trying to set key {key}")

        metrics = train(config, wandb.run.dir)
        wandb.log(metrics)

if __name__ == "__main__":
    main()