import os
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from gensim.models.doc2vec import Doc2Vec
from mofgraph2vec.utils.cv import cross_validation
from mofgraph2vec.utils.saving import save_embedding
from mofgraph2vec.utils.evaluation import evaluate_model
from mofgraph2vec.utils.seed import set_seed

def train(
    config: DictConfig,
    wandb_dir: str
):
    set_seed(config.seed)

    doc = instantiate(config.data.data, seed=config.seed)
    documents = doc.get_documents()
    word_percentage = doc.distribution_analysis(config.model.gensim.min_count)

    logger.info(f"Instantiate model. ")
    if config.load_model:
        assert config.model_checkpoint is None
        logger.info(f"Load trained model from {config.model_checkpoint}. ")
        model = Doc2Vec.load(config.model_checkpoint)
    else:
        model = Doc2Vec(**config.model.gensim)
        model.build_vocab(documents)
        if config.model.cv:
            cv_mean, cv_std = cross_validation(documents, model, k_foldes=5, epochs=config.model.gensim.epochs)

    model.train(documents, total_examples=model.corpus_count, epochs=config.model.gensim.epochs)
    logger.info(f"Evaluating the model performance. ")
    accuracy = evaluate_model(model, documents)
    model.save(os.path.join(wandb_dir, "embedding_model.pt"))

    logger.info(f"Saving embedded vectors. ")
    save_embedding(os.path.join(wandb_dir, "embedding.csv"), model, documents, config.model.gensim.vector_size)

    if config.model.cv:
        return {
            "percentage": word_percentage,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "accuracy": accuracy
        }
    else:
        return {
            "percentage": word_percentage,
            "accuracy": accuracy
        }

