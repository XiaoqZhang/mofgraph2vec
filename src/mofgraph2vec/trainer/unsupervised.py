import os
from typing import List, Optional, Tuple

import omegaconf
import wandb
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from mofgraph2vec.utils.dict_helpers import get, put
from mofgraph2vec.graph.mof2doc import MOF2doc
from gensim.models.doc2vec import Doc2Vec
from mofgraph2vec.utils.evaluation import evaluate_model
from mofgraph2vec.utils.saving import save_embedding

def train(
    config: DictConfig,
    sweep: bool = False
):

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

        doc = instantiate(config.data.data, seed=config.seed)
        train_documents = doc.get_documents()
        word_percentage = doc.distribution_analysis(config.model.gensim.min_count)
        wandb.log({"Percentage": word_percentage})

        logger.info(f"Instantiate model. ")
        if config.load_model:
            assert config.model_checkpoint is None
            logger.info(f"Load trained model from {config.model_checkpoint}. ")
            model = Doc2Vec.load(config.model_checkpoint)
        else:
            model = Doc2Vec(**config.model.gensim)
            model.build_vocab(train_documents)
            logger.info(f"Start training model. ")
            model.train(train_documents, total_examples=model.corpus_count, epochs=config.model.gensim.epochs)

            accuracy = evaluate_model(model, train_documents)
            wandb.log({"Accuracy": "%.4f" %accuracy})
            model.save(os.path.join(wandb.run.dir, "../tmp/model.pt"))


        logger.info(f"Saving embedded vectors. ")
        save_embedding(os.path.join(wandb.run.dir, "../tmp/embedding.csv"), model, train_documents, config.model.gensim.vector_size)
