import os
import pandas as pd
from typing import List, Optional
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

def path2name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def save_embedding(
        pretraining: bool,
        output_path: str, 
        model: Doc2Vec, 
        documents: List[TaggedDocument], 
        doc_dimensions: int, 
):
    """ Function to save the MOF embeddings. """
    output_dv = os.path.join(output_path, "embedding_dv.csv")
    out_dv = []
    for id in range(len(documents)):
        identifier = documents[id].tags[0]

        if pretraining == True:
            out_dv.append([identifier] + list(model.dv[identifier]))
        else: 
            out_dv.append([identifier] + list(model.infer_vector(documents[id].words, epochs=100)))

    column_names = ["type"]+["x_"+str(dim) for dim in range(doc_dimensions)]
    out_dv = pd.DataFrame(out_dv, columns=column_names)
    out_dv = out_dv.sort_values(["type"])
    out_dv.to_csv(output_dv, index=None)

