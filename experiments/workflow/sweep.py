import os
import hydra
from omegaconf import DictConfig

from mofgraph2vec.sweep.sweep import sweep

@hydra.main(config_path="../../conf", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
        sweep(config)
    

if __name__ == "__main__":
    main()
    os.system('say "Your sweep has finished"')