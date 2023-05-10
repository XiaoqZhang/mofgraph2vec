import os
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from gensim.models.doc2vec import Doc2Vec
from mofgraph2vec.utils.saving import save_embedding
from sklearn.model_selection import train_test_split

def run_embedding(
    config: DictConfig,
    log_dir: str,
    pretraining: bool
):
    # Load MOF document data
    doc = instantiate(config.mof2vec_data.data, seed=config.seed)
    documents = doc.get_documents()
    _, valid_documents = train_test_split(documents, test_size=0.1, random_state=1234)
    word_percentage = doc.distribution_analysis(config.mof2vec_model.gensim.min_count)
    logger.info(f"Learning MOF embedding with {len(documents)} training data. ")

    if pretraining == True:
        # Gensim model instantiation
        logger.info(f"Instantiate model. ")
        if config.mof2vec_model.load_checkpoint:
            assert config.mof2vec_model.model_checkpoint is None
            logger.debug(f"Load trained model from {config.mof2vec_model.model_checkpoint}. ")
            model = Doc2Vec.load(config.mof2vec_model.model_checkpoint)
            model.build_vocab(documents)
        else:
            model = Doc2Vec(**config.mof2vec_model.gensim, seed=config.seed)
            model.build_vocab(documents)
            #accuracy_callback = AccuracyCallback(log_dir, valid_documents, config.mof2vec_model.evaluate_patience)

            # Model training
            model.train(
                documents, 
                total_examples=model.corpus_count, 
                epochs=config.mof2vec_model.gensim.epochs, 
                #callbacks=[accuracy_callback]
            )
            logger.info(f"Evaluating the model performance. ")
            model.save(os.path.join(log_dir, "embedding_model.pt"))
    else:
        if config.doc2label_data.embedding_model_path is not None:
            logger.debug(f"Loading pretrained model from {config.doc2label_data.embedding_model_path}")
            model = Doc2Vec.load(config.doc2label_data.embedding_model_path)
    
    # Get topology vectors
    if config.mof2vec_model.topology:
        logger.info(f"Calculating topology vectors. ")
        topo_vectors = doc.get_topovectors()
        topo_dim = topo_vectors[0].vectors.shape[0]
    else:
        topo_vectors = None
        topo_dim = None    

    # Log info
    logger.info(f"Saving embedded vectors. ")
    save_embedding(
        pretraining,
        log_dir, 
        model, 
        documents, 
        config.mof2vec_model.gensim.vector_size,
        topo_vectors,
        topo_dim
    )
    
    accuracy = 0 #evaluate_model(model, documents, config.mof2vec_model.evaluate_patience)

    return {
        "percentage": word_percentage,
        "accuracy": accuracy
    }

