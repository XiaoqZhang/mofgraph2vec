import os
import pandas as pd
from typing import List, Optional
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from mofgraph2vec.featurize.topo2vec import TaggedVector

def path2name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def save_embedding(
        output_path: str, 
        model: Doc2Vec, 
        documents: List[TaggedDocument], 
        doc_dimensions: int, 
        topo_vectors: Optional[List[TaggedVector]], 
        topo_dimensions: Optional[int]
):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    output_dv = os.path.join(output_path, "embedding_dv.csv")
    out_dv = []
    for id in range(len(documents)):
        identifier = documents[id].tags[0]
        if topo_vectors is not None:
            for tagged_vec in topo_vectors:
                if tagged_vec.tags[0] == identifier:
                    topo_vec = tagged_vec.vectors
            out_dv.append([identifier] + list(model.dv[identifier]) + list(topo_vec))
        else:
            out_dv.append([identifier] + list(model.dv[identifier]))

    column_names = ["type"]+["x_"+str(dim) for dim in range(doc_dimensions)]
    if topo_vectors is not None:
        column_names += ["topo_"+str(dim) for dim in range(topo_dimensions)]
    out_dv = pd.DataFrame(out_dv, columns=column_names)
    out_dv = out_dv.sort_values(["type"])
    out_dv.to_csv(output_dv, index=None)


def _get_config_file(model_path, model_name):
    # Name of the file for storing hyperparameter details
    return os.path.join(model_path, model_name + ".config")


def _get_model_file(model_path, model_name):
    # Name of the file for storing network parameters
    return os.path.join(model_path, model_name + ".tar")


def load_model(model_path, model_name, net=None):
    """Loads a saved model from disk.

    Args:
        model_path: Path of the checkpoint directory
        model_name: Name of the model (str)
        net: (Optional) If given, the state dict is loaded into this model. Otherwise, a new model is created.
    """
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    assert os.path.isfile(
        config_file
    ), f'Could not find the config file "{config_file}". Are you sure this is the correct path and you have your model config stored here?'
    assert os.path.isfile(
        model_file
    ), f'Could not find the model file "{model_file}". Are you sure this is the correct path and you have your model stored here?'
    with open(config_file) as f:
        config_dict = json.load(f)
    if net is None:
        act_fn_name = config_dict["act_fn"].pop("name").lower()
        act_fn = act_fn_by_name[act_fn_name](**config_dict.pop("act_fn"))
        net = BaseNetwork(act_fn=act_fn, **config_dict)
    net.load_state_dict(torch.load(model_file, map_location=device))
    return net


def save_model(model, model_path, model_name):
    """Given a model, we save the state_dict and hyperparameters.

    Args:
        model: Network object to save parameters from
        model_path: Path of the checkpoint directory
        model_name: Name of the model (str)
    """
    config_dict = model.config
    os.makedirs(model_path, exist_ok=True)
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    with open(config_file, "w") as f:
        json.dump(config_dict, f)
    torch.save(model.state_dict(), model_file)