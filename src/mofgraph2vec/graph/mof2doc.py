import os
import random
from glob import glob
from tqdm import tqdm
from collections import Counter
import numpy as np
from typing import Optional, List
from mofgraph2vec.graph.cif2graph import MOFDataset
from mofgraph2vec.graph.tokenize import WeisfeilerLehmanMachine
from gensim.models.doc2vec import TaggedDocument

class MOF2doc:
    def __init__(
        self,
        cif_path: str,
        wl_step: int = 5,
        subsample: Optional[int] = None,
        seed: Optional[int] = 1234,
    ) -> None:
        random.seed(seed)

        self.files = glob(os.path.join(cif_path, "*.cif"))
        if subsample is not None and subsample < len(self.files):
            self.files: List[str] = random.sample(self.files, subsample)

        self.wl_step = wl_step

    def get_documents(self) -> List[TaggedDocument]:
        ds_loader = MOFDataset(strategy="vesta")

        self.documents = []
        for cif in tqdm(self.files):
            name = cif.split("/")[-1].rstrip(".cif")
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
