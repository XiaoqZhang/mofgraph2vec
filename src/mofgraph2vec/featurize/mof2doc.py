import os
import random
from glob import glob
from tqdm import tqdm
from pathlib import Path
from collections import Counter
import numpy as np
from typing import Optional, List
from mofgraph2vec.featurize.cif2graph import MOFDataset
from mofgraph2vec.featurize.tokenize import WeisfeilerLehmanMachine
from gensim.models.doc2vec import TaggedDocument

class MOF2doc:
    def __init__(
        self,
        cif_path: List[str],
        wl_step: int = 5,
        subsample: Optional[int] = None,
        seed: Optional[int] = 1234,
        **kwarg
    ):     
        self.files = []
        for pt in cif_path:
            files_in_pt = glob(os.path.join(pt, "*.cif"))
            self.files.append(files_in_pt)
        self.files = [file for folder in self.files for file in folder]
        if subsample is not None and subsample < 1:
            self.files: List[str] = random.sample(self.files, int(subsample*len(self.files)))

        self.wl_step = wl_step
        self.seed = seed

    def get_documents(self):
        ds_loader = MOFDataset(strategy="vesta")

        self.documents = []
        for cif in tqdm(self.files):
            name = Path(cif).stem
            graph, feature = ds_loader.to_WL_machine(cif)
            machine = WeisfeilerLehmanMachine(graph, feature, self.wl_step)
            word = machine.extracted_features
            doc = TaggedDocument(words=word, tags=[name])

            self.documents.append(doc)

        return self.documents
    
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
