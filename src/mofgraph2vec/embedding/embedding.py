import os
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from gensim.models.doc2vec import Doc2Vec
from mofgraph2vec.utils.saving import save_embedding
from mofgraph2vec.utils.evaluation import evaluate_model
from mofgraph2vec.embedding.callbacks import AccuracyCallback

def run_embedding(
    config: DictConfig,
    log_dir: str
):
    # Load MOF document data
    doc = instantiate(config.mof2vec_data.data, seed=config.seed)
    documents, train_documents, test_documents = doc.get_documents()
    word_percentage = doc.distribution_analysis(config.mof2vec_model.gensim.min_count)
    logger.info(f"Learning MOF embedding with {len(train_documents)} training data and {len(test_documents)} test data. ")

    # Gensim model instantiation
    logger.info(f"Instantiate model. ")
    if config.mof2vec_model.load_checkpoint:
        assert config.mof2vec_model.model_checkpoint is None
        logger.info(f"Load trained model from {config.mof2vec_model.model_checkpoint}. ")
        model = Doc2Vec.load(config.mof2vec_model.model_checkpoint)
        model.build_vocab(documents)
    else:
        model = Doc2Vec(**config.mof2vec_model.gensim, seed=config.seed)
        model.build_vocab(documents)
        accuracy_callback = AccuracyCallback(log_dir, test_documents, config.mof2vec_model.evaluate_patience)

        # Model training
        model.train(
            documents, 
            total_examples=model.corpus_count, 
            epochs=config.mof2vec_model.gensim.epochs, 
            callbacks=[accuracy_callback]
        )
        logger.info(f"Evaluating the model performance. ")
        model.save(os.path.join(log_dir, "embedding_model.pt"))

        # Log info
        logger.info(f"Saving embedded vectors. ")
        save_embedding(
            os.path.join(log_dir, "embedding.csv"), 
            model, 
            documents, 
            config.mof2vec_model.gensim.vector_size
        )
    
    accuracy = evaluate_model(model, documents, config.mof2vec_model.evaluate_patience)

    return {
        "percentage": word_percentage,
        "accuracy": accuracy
    }
