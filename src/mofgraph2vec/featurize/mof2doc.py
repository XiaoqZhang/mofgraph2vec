import os
import random
import re
from glob import glob
from tqdm import tqdm
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from typing import Optional, List
from mofgraph2vec.data.spliter import quantile_binning
from mofgraph2vec.featurize.cif2graph import MOFDataset
from mofgraph2vec.featurize.tokenize import WeisfeilerLehmanMachine
from gensim.models.doc2vec import TaggedDocument
from mofgraph2vec.featurize.topo2vec import TaggedVector
from pymatgen.core import Structure, Element
from loguru import logger

class MOF2doc:
    def __init__(
        self,
        cif_path: List[str],
        embed_property: bool,
        label_path: str,
        embed_label: List[str],
        id_column: str,
        wl_step: int = 5,
        n_components: int = 20,
        use_hash: bool = False,
        writing_style: str = "sentence",
        composition: bool = True,
        mode: str = "all",
        embed_cif: Optional[bool] = False,
        subsample: Optional[int] = None,
        seed: Optional[int] = 1234,
        **kwarg
    ):     
        self.files = []
        self.embed_property = embed_property
        if self.embed_property:
            self.df_label = pd.read_csv(label_path).set_index(id_column)
            self.embed_label = ["binned_%s" %label for label in embed_label]
            for label in embed_label:
                binned_values = quantile_binning(self.df_label.loc[:, label].values.reshape(-1,), np.arange(0, 1.01, 0.05))
                self.df_label["binned_%s" %label] = ["%s_%s" %(label, v) for v in binned_values]
        for pt in cif_path:
            files_in_pt = glob(os.path.join(pt, "*.cif"))
            self.files.append(files_in_pt)
        self.files = [file for folder in self.files for file in folder]
        if subsample is not None and subsample < 1:
            random.seed(seed)
            self.files: List[str] = random.sample(self.files, int(subsample*len(self.files)))

        self.wl_step = wl_step
        self.n_components = n_components
        self.hash = use_hash
        self.writing_style = writing_style
        self.composition = composition
        self.mode = mode
        self.seed = seed

        self.embed_cif = embed_cif

    def get_documents(self):
        ds_loader = MOFDataset(strategy="vesta")

        self.documents = []
        for cif in tqdm(self.files):
            name = Path(cif).stem

            if self.embed_cif:
                py_cif = Structure.from_file(cif)
                # composition
                com = str(py_cif.composition).split()
                opt = re.compile("([a-zA-Z]+)([0-9]+)")
                com = [opt.match(c).groups() for c in com]
                # lattice
                lattice = ([round(x, 2) for x in list(py_cif.lattice.abc)]
                            + [round(x, 2) for x in list(py_cif.lattice.angles)]
                            + [x for x in py_cif.lattice.pbc])
                # sites
                sites = [[[str(site.specie)] + [round(x, 2) for x in list(site.coords)]] for site in py_cif.sites]

                word = com + lattice + list(np.array(sites).flatten())


            else:          
                if self.composition:
                    com = str(Structure.from_file(cif).composition).split()
                    opt = re.compile("([a-zA-Z]+)([0-9]+)")
                    word = [x for c in com for x in list(opt.match(c).groups())]
                else:
                    word = []

                graph, feature, nodes_idx, linker_idx = ds_loader.to_WL_machine(cif)
                machine = WeisfeilerLehmanMachine(graph, feature, nodes_idx, linker_idx, self.wl_step, self.hash, self.writing_style, self.mode)
                word += machine.extracted_features

                feature_affinity = {}
                for i, ele in feature.items():
                    feature_affinity.update({i: "g%s" %Element(ele).group})
                    #feature_affinity.update({i: "%.2f" %Element(ele).electron_affinity})
                machine_affinity = WeisfeilerLehmanMachine(graph, feature_affinity, nodes_idx, linker_idx, 1, self.hash, self.writing_style, self.mode)
                word += machine_affinity.extracted_features
                
                feature_block = {}
                for i, ele in feature.items():
                    feature_block.update({i: Element(ele).block})
                machine_block = WeisfeilerLehmanMachine(graph, feature_block, nodes_idx, linker_idx, 1, self.hash, self.writing_style, self.mode)
                word += machine_block.extracted_features

                # embed edges
                word += list(np.unique(["%s=%s" %(feature_block[i], feature_block[j]) for i, j in graph.edges]))
                # embed binned labels
                if self.embed_property:
                    word += list(self.df_label.loc[name, self.embed_label].values)
            
            if name == "RSM0001":
                logger.info(f"{word}")
            
            doc = TaggedDocument(words=word, tags=[name])
            self.documents.append(doc)

        return self.documents
    
    def get_topovectors(self):
        from mofdscribe.featurizers.topology.ph_vect import PHVect

        topo_featurizer = PHVect(
            atom_types=(),
            compute_for_all_elements=True,
            dimensions=(1,2,3),
            min_size=20,
            n_components=self.n_components,
            apply_umap=False,
            random_state=self.seed
        )

        py_cifs = [Structure.from_file(c) for c in self.files]
        topo_vectors = topo_featurizer.fit_transform(py_cifs)
        names = [Path(cif).stem for cif in self.files]

        self.topo_vectors = []
        for name, vector in zip(names, topo_vectors):
            vec = TaggedVector(vectors=vector, tags=[name])
            self.topo_vectors.append(vec)

        return self.topo_vectors
    
    def distribution_analysis(self, threshold: int = 4) -> float:
        """
            Args:
                threshold: int, the maximum times that a word appear in the corpus
            Return:
                the percentage of words that appear less than {threshold} times
        """
        if not hasattr(self, "documents"):
            self.documents = self.get_documents()
        corpus = [doc.words for doc in self.documents]
        corpus = [word for words in corpus for word in words]
        distribution = Counter(corpus)
        times_count = [distribution[word] for idx, word in enumerate(distribution)]
        percentage = np.sum(np.array(times_count)<threshold)/len(times_count)
        return percentage
