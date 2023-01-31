import hydra
from omegaconf import DictConfig
import wandb
import os
from loguru import logger

from mofgraph2vec.trainer.unsupervised import train as unsupervised_train
from mofgraph2vec.trainer.supervised import train as supervised_train
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
        
        config.data.nn.embedding_path = os.path.join(wandb.run.dir, "../tmp/embedding.csv")
        config.model.nn.input_dim = config.model.gensim.vector_size
        logger.info(f"{config}")
        unsupervised_metrics = unsupervised_train(config, wandb.run.dir)
        supervised_metrics = supervised_train(config)

        wandb.log(unsupervised_metrics)
        wandb.log(supervised_metrics)

if __name__ == "__main__":
    main()