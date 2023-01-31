import os
import pandas as pd

def path2name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def save_embedding(output_path, model, documents, dimensions):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    for id in range(len(documents)):
        identifier = documents[id].tags[0]
        out.append([identifier] + list(model.dv[identifier]))
    column_names = ["type"]+["x_"+str(dim) for dim in range(dimensions)]
    out = pd.DataFrame(out, columns=column_names)
    out = out.sort_values(["type"])
    out.to_csv(output_path, index=None)