import os
from glob import glob
from tqdm import tqdm
from loguru import logger
from collections import Counter
import pandas as pd
from mofgraph2vec.graph.cif2graph import MOFDataset
from mofgraph2vec.graph.mof2vec import WeisfeilerLehmanMachine
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


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
        machine = WeisfeilerLehmanMachine(graph, feature, 4)
        word = machine.extracted_features
        doc = TaggedDocument(words=word, tags=["g_" + name])    # to check: what does label do? 
        corpus.append(word)
        documents.append(doc)
        graphs.append(graph)
    
    corpus = [word for words in corpus for word in words]
    distribution = Counter(corpus)

    logger.info(f"Inistantiate model.")
    model = Doc2Vec(vector_size=64,
                    window=10,
                    min_count=3,
                    dm=0,
                    sample=0.9,
                    workers=4,
                    epochs=10,
                    alpha=0.001)
    model.build_vocab(documents)
    logger.info(f"Start training.")
    model.train(documents, total_examples=model.corpus_count, epochs=40)


    save_embedding("../../../data/vec/embedding.csv", model, cifs, 64)


def path2name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def save_embedding(output_path, model, files, dimensions):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    for f in files:
        identifier = path2name(f)
        out.append([identifier] + list(model.dv["g_"+identifier]))
    column_names = ["type"]+["x_"+str(dim) for dim in range(dimensions)]
    out = pd.DataFrame(out, columns=column_names)
    out = out.sort_values(["type"])
    out.to_csv(output_path, index=None)

if __name__ == "__main__":
    main()