import os
import random
from glob import glob
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, List
from mofgraph2vec.featurize.cif2graph import MOFDataset
from mofgraph2vec.featurize.tokenize import WeisfeilerLehmanMachine
from gensim.models.doc2vec import TaggedDocument

class MOF2doc:
    def __init__(
        self,
        cif_path: List[str],
        embed_label: bool,
        label_path: str,
        descriptors_to_embed: List[str],
        category_to_embed: List[str],
        id_column: str,
        wl_step: int = 5,
        seed: Optional[int] = 1234,
        **kwarg
    ):     
        """convert MOF CIF files to documents for doc2vec

        Args:
            cif_path (List[str]): the list of cif file paths
            embed_label (bool): whether to embed the labels
            label_path (str): the path to the label .csv file
            descriptors_to_embed (List[str]): the list of descriptors to embed
            category_to_embed (List[str]): the list of categories to embed
            id_column (str): the column name of the id
            wl_step (int, optional): the order of neighbors to extract. Defaults to 5.
            seed (Optional[int], optional): random seed. Defaults to 1234.
        """
        self.files = []

        self.embed_label = embed_label
        if self.embed_label:
            self.df_label = pd.read_csv(label_path).set_index(id_column)
            self.descriptors_to_embed = ["binned_%s" %label for label in descriptors_to_embed]
            self.category_to_embed = category_to_embed
            for label in descriptors_to_embed:
                binned_values = pd.qcut(self.df_label.loc[:, label], q=10, labels=range(10))
                self.df_label["binned_%s" %label] = ["%s_%s" %(label, v) if not pd.isna(v) else "UNKNOWN" for v in binned_values]
        
        for pt in cif_path:
            files_in_pt = glob(os.path.join(pt, "*.cif"))
            self.files.append(files_in_pt)
        self.files = [file for folder in self.files for file in folder]
        if subsample is not None and subsample < 1:
            random.seed(seed)
            self.files: List[str] = random.sample(self.files, int(subsample*len(self.files)))

        self.wl_step = wl_step
        self.seed = seed

        self.embed_cif = embed_cif

    def get_documents(self):
        ds_loader = MOFDataset(strategy="vesta")

        self.documents = []
        for cif in tqdm(self.files):
            name = Path(cif).stem
        
            word = []

            graph, feature = ds_loader.to_WL_machine(cif)
            machine = WeisfeilerLehmanMachine(graph, feature, self.wl_step)
            word += machine.extracted_features

            # embed binned labels
            if self.embed_label:
                word += list([bin for bin in self.df_label.loc[name, self.descriptors_to_embed] if bin!="UNKNOWN"])
                word += list([bin for bin in self.df_label.loc[name, self.category_to_embed] if not bin is np.nan])

            doc = TaggedDocument(words=word, tags=[name])
            self.documents.append(doc)

        return self.documents
    