import os
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from gensim.models.doc2vec import Doc2Vec
from mofgraph2vec.utils.saving import save_embedding

def run_embedding(
    config: DictConfig,
    log_dir: str,
    pretraining: bool
):
    """run the embedding pipeline

    Args:
        config (DictConfig): configuration for the gensim Doc2Vec model
        log_dir (str): the directory to save the embedding
        pretraining (bool): whether to pretrain a model or load a pretrained model
    """
    # Load MOF document data
    doc = instantiate(config.mof2vec_data.data, seed=config.seed)
    documents = doc.get_documents()
    logger.info(f"Learning MOF embedding with {len(documents)} training data. ")

    if pretraining == True:
    # Train the Doc2Vec model
        # Doc2Vec model instantiation
        logger.info(f"Instantiate model. ")
        model = Doc2Vec(**config.mof2vec_model.gensim, seed=config.seed)
        model.build_vocab(documents)

        model.train(
            documents, 
            total_examples=model.corpus_count, 
            epochs=config.mof2vec_model.gensim.epochs, 
        )
        logger.info(f"Evaluating the model performance. ")
        model.save(os.path.join(log_dir, "embedding_model.pt"))
    else:
    # Load the pretrained Doc2Vec model
        if config.doc2label_data.embedding_model_path is not None:
            logger.debug(f"Loading pretrained model from {config.doc2label_data.embedding_model_path}")
            model = Doc2Vec.load(config.doc2label_data.embedding_model_path)  

    # Log info
    logger.info(f"Saving embedded vectors. ")
    save_embedding(
        pretraining,
        log_dir, 
        model, 
        documents, 
        config.mof2vec_model.gensim.vector_size,
    )

