import os
import wandb
from omegaconf import DictConfig
from loguru import logger
import joblib
import pandas as pd
from mofgraph2vec.utils.dict_helpers import get, put
from mofgraph2vec.utils.seed import set_seed
from mofgraph2vec.embedding.embedding import run_embedding
from mofgraph2vec.model.regression import run_regression

def train(config: DictConfig, sweep: bool=False):
    """ Load the config and run the workflow. """

    assert config.mode in ["mof2vec", "doc2label", "workflow"]
    
    # Initialize wandb
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

        # Run the workflow
        # Case 1: Workflow
        if config.mode == "workflow":
            logger.info(f"Running workflow. ")
            config.doc2label_data.embedding_path = os.path.join(wandb.run.dir, "../tmp/embedding_dv.csv")
            run_embedding(config, os.path.join(wandb.run.dir, "../tmp/"), pretraining=config.doc2label_data.pretraining)
            model, supervised_metrics, train_parity, test_parity = run_regression(config)
            logger.info(f"Model performance: {supervised_metrics}")
            joblib.dump(model, os.path.join(wandb.run.dir, "../tmp/best_model.pkl"))
            pd.DataFrame(train_parity).to_csv(os.path.join(wandb.run.dir, "../tmp/train_parity.csv"))
            pd.DataFrame(test_parity).to_csv(os.path.join(wandb.run.dir, "../tmp/test_parity.csv"))

            to_log = {'task': config.doc2label_data.task[0]}
            to_log.update(supervised_metrics)
            wandb.log(to_log)

        # Case 2: MOF2Vec, only train the Doc2Vec model
        elif config.mode == "mof2vec":
            logger.info(f"Running MOF embedding. ")
            run_embedding(config, os.path.join(wandb.run.dir, "../tmp/"), pretraining=config.doc2label_data.pretraining)
        
        # Case 3: Doc2Label, load the trained MOF embeddings and train the downstream regression model
        elif config.mode == "doc2label":
            if ((config.doc2label_data.pretraining == False) and (config.doc2label_data.embedding_model_path is not None)):
                config.doc2label_data.embedding_path = os.path.join(wandb.run.dir, "../tmp/embedding_dv.csv")
                run_embedding(config, os.path.join(wandb.run.dir, "../tmp/"), pretraining=config.doc2label_data.pretraining)
            
            logger.info(f"Running regression. ")
            model, supervised_metrics, train_parity, test_parity = run_regression(config)
            logger.info(f"Model performance: {supervised_metrics}")
            joblib.dump(model, os.path.join(wandb.run.dir, "../tmp/best_model.pkl"))
            pd.DataFrame(train_parity).to_csv(os.path.join(wandb.run.dir, "../tmp/train_parity.csv"))
            pd.DataFrame(test_parity).to_csv(os.path.join(wandb.run.dir, "../tmp/test_parity.csv"))

            to_log = {'task': config.doc2label_data.task[0]}
            to_log.update(supervised_metrics)
            wandb.log(to_log)

