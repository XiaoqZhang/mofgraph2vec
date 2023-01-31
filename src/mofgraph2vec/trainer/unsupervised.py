import os
from typing import List, Optional, Tuple

import omegaconf
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
    wandb_run_dir: str
):

    doc = instantiate(config.data.data, seed=config.seed)
    documents = doc.get_documents()
    word_percentage = doc.distribution_analysis(config.model.gensim.min_count)

    logger.info(f"Instantiate model. ")
    if config.load_model:
        assert config.model_checkpoint is None
        logger.info(f"Load trained model from {config.model_checkpoint}. ")
        model = Doc2Vec.load(config.model_checkpoint)
    else:
        def cross_validation():
            return mean, dev

    logger.info(f"Evaluating the model performance. ")
    accuracy = evaluate_model(model, train_documents)
    model.save(os.path.join(wandb_run_dir, "../tmp/model.pt"))

    logger.info(f"Saving embedded vectors. ")
    save_embedding(os.path.join(wandb_run_dir, "../tmp/embedding.csv"), model, train_documents, config.model.gensim.vector_size)

    return word_percentage, accuracy
