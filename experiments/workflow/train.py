import hydra
from omegaconf import DictConfig
import wandb
import os
from loguru import logger

from mofgraph2vec.trainer.sklearn_workflow import train


@hydra.main(config_path="../../conf", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    train(config)

if __name__ == "__main__":
    main()