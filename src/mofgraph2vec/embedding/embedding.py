import os
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
from mofgraph2vec.utils.saving import save_embedding
import numpy as np
import pandas as pd

def run_embedding(
    config: DictConfig,
    log_dir: str,
    pretraining: bool
):
    # Load MOF document data
    doc = instantiate(config.mof2vec_data.data, seed=config.seed)
    names, documents = doc.get_documents()
    #word_percentage = doc.distribution_analysis(config.mof2vec_model.gensim.min_count)
    logger.info(f"Learning MOF embedding with {len(documents)} training data. ")

    if pretraining == True:
        # Gensim model instantiation
        logger.info(f"Instantiate model. ")
        if config.mof2vec_model.load_checkpoint:
            assert config.mof2vec_model.model_checkpoint is None
            logger.debug(f"Load trained model from {config.mof2vec_model.model_checkpoint}. ")
            model = Word2Vec.load(config.mof2vec_model.model_checkpoint)
            model.build_vocab(documents)
        else:
            model = Word2Vec(**config.mof2vec_model.gensim, seed=config.seed)
            model.build_vocab(documents)

            # Model training
            model.train(
                documents, 
                total_examples=model.corpus_count, 
                epochs=config.mof2vec_model.gensim.epochs, 
            )
            logger.info(f"Evaluating the model performance. ")
            model.save(os.path.join(log_dir, "embedding_model.pt"))
    else:
        if config.doc2label_data.embedding_model_path is not None:
            logger.debug(f"Loading pretrained model from {config.doc2label_data.embedding_model_path}")
            model = Word2Vec.load(config.doc2label_data.embedding_model_path)
    
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
    model.wv.save(os.path.join(log_dir, "w2v.wordvectors"))
    out_dv = []
    for name, sentence in zip(names, documents):
        sen_vector = [model.wv[word] for word in sentence]
        #sen_vector_sum = np.array(sen_vector).sum(axis=0)
        sen_vector_mean = np.array(sen_vector).mean(axis=0)
        out_dv.append([name] + list(sen_vector_mean))

    column_names = ["type"]+["x_"+str(dim) for dim in range(model.vector_size)]    
    out_dv = pd.DataFrame(out_dv, columns=column_names)
    out_dv = out_dv.sort_values(["type"])
    out_dv.to_csv(
        os.path.join(log_dir, "embedding_dv.csv"), 
        index=None
    )

    
    return {
        "percentage": 0,
    }

