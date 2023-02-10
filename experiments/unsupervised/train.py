import hydra
from omegaconf import DictConfig
import wandb

from mofgraph2vec.embedding.embedding import run_embedding

@hydra.main(config_path="../../conf", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    with wandb.init(
        project=config.logger.project,
        entity=config.logger.entity,
        mode=config.logger.mode,
    ) as run:
        metrics = run_embedding(config, wandb.run.dir)
        wandb.log(metrics)

if __name__ == "__main__":
    main()