import os
from glob import glob
from tqdm import tqdm
from loguru import logger
from collections import Counter
import pandas as pd
import numpy as np
from mofgraph2vec.graph.cif2graph import MOFDataset
from mofgraph2vec.graph.mof2vec import WeisfeilerLehmanMachine
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile


def main():
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """
    cifs = glob("../../../data/cifs/*.cif")
    logger.info(f"Feature extraction started.")
    dataset_loader = MOFDataset(strategy="vesta")

    corpus = []
    documents = []
    graphs = []
    for cif in tqdm(cifs):
        name = cif.split("/")[-1].rstrip(".cif")
        graph, feature = dataset_loader.to_WL_machine(cif)
        machine = WeisfeilerLehmanMachine(graph, feature, 5)
        word = machine.extracted_features
        doc = TaggedDocument(words=word, tags=[name])   
        corpus.append(word)
        documents.append(doc)
        graphs.append(graph)
    

    word_count = 4
    corpus = [word for words in corpus for word in words]
    distribution = Counter(corpus)
    times_count = [distribution[word] for idx, word in enumerate(distribution)]
    percentage = np.sum(np.array(times_count)<word_count)/len(times_count)
    logger.info(f"Words that appear less than {word_count} is {percentage}")

    logger.info(f"Inistantiate model.")
    model_checkpoint = "../../../experiments/unsupervised/model.pt"
    if not os.path.exists(model_checkpoint):
        model = Doc2Vec(vector_size=64,
                        window=10,
                        min_count=word_count,
                        dm=0,
                        sample=0.9,
                        workers=4,
                        epochs=10,
                        alpha=0.001)
        model.build_vocab(documents)
        logger.info(f"Start training.")
        model.train(documents, total_examples=model.corpus_count, epochs=1)
    
        model.save(model_checkpoint)
    else:
        logger.info(f"Load trained model. ")
        model = Doc2Vec.load(model_checkpoint)

    ranks = []
    for doc_id in range(len(documents)):
        inferred_vector = model.infer_vector(documents[doc_id].words)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        rank = [docid for docid, sim in sims].index(documents[doc_id].tags[0])
        ranks.append(rank)
    
    logger.info(f"Ranks counting: {Counter(ranks)}")
    logger.info(f"The percentage of MOFs that are consistent: {Counter(ranks)[0]/len(documents)}")

    save_embedding("../../../data/vec/embedding.csv", model, cifs, 64)


if __name__ == "__main__":
    main()