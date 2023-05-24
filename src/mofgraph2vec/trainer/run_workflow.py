import os
import wandb
from omegaconf import DictConfig
from loguru import logger
import joblib
from mofgraph2vec.utils.dict_helpers import get, put
from mofgraph2vec.utils.seed import set_seed
from mofgraph2vec.embedding.embedding import run_embedding

def train(config: DictConfig, sweep: bool=False):

    assert config.mode in ["mof2vec", "doc2label", "workflow"]

    if config.doc2label_model.get("nn") is None:
        from mofgraph2vec.model.regression import run_regression
    else:
        from mofgraph2vec.model.nn_regression import run_regression
    
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

        if config.mode == "workflow":
            logger.info(f"Running workflow. ")
            config.doc2label_data.embedding_path = os.path.join(wandb.run.dir, "../tmp/embedding_dv.csv")
            unsupervised_metrics = run_embedding(config, os.path.join(wandb.run.dir, "../tmp/"), pretraining=config.doc2label_data.pretraining)
            model, supervised_metrics, figure = run_regression(config)
            logger.info(f"Model performance: {supervised_metrics}")
            joblib.dump(model, os.path.join(wandb.run.dir, "../tmp/best_model.pkl"))

            table = wandb.Table(data=figure, columns = ["True", "Pred"])
            to_log = {
                'task': config.doc2label_data.task[0],
                'parity': wandb.plot.scatter(table, "True", "Pred")
            }
            to_log.update(unsupervised_metrics)
            to_log.update(supervised_metrics)
            wandb.log(to_log)

        elif config.mode == "mof2vec":
            logger.info(f"Running MOF embedding. ")
            unsupervised_metrics = run_embedding(config, os.path.join(wandb.run.dir, "../tmp/"), pretraining=config.doc2label_data.pretraining)
            wandb.log(unsupervised_metrics)
        
        elif config.mode == "doc2label":
            if ((config.doc2label_data.pretraining == False) and (config.doc2label_data.embedding_model_path is not None)):
                config.doc2label_data.embedding_path = os.path.join(wandb.run.dir, "../tmp/embedding_dv.csv")
                unsupervised_metrics = run_embedding(config, os.path.join(wandb.run.dir, "../tmp/"), pretraining=config.doc2label_data.pretraining)
            
            logger.info(f"Running regression. ")
            model, supervised_metrics, figure = run_regression(config)
            logger.info(f"Model performance: {supervised_metrics}")
            joblib.dump(model, os.path.join(wandb.run.dir, "../tmp/best_model.pkl"))

            table = wandb.Table(data=figure, columns = ["True", "Pred"])
            to_log = {
                'task': config.doc2label_data.task[0],
                'parity': wandb.plot.scatter(table, "True", "Pred")
            }
            to_log.update(supervised_metrics)
            wandb.log(to_log)

