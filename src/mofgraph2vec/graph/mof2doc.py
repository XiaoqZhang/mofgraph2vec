import os
import random
from glob import glob
from tqdm import tqdm
from pathlib import Path
from collections import Counter
import numpy as np
from typing import Optional, List
from mofgraph2vec.graph.cif2graph import MOFDataset
from mofgraph2vec.graph.tokenize import WeisfeilerLehmanMachine
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split

class MOF2doc:
    def __init__(
        self,
        cif_path: List[str],
        wl_step: int = 5,
        subsample: Optional[int] = None,
        seed: Optional[int] = 1234,
    ) -> None:
        random.seed(seed)
        
        self.files = []
        for pt in cif_path:
            files_in_pt = glob(os.path.join(pt, "*.cif"))
            self.files.append(files_in_pt)
        self.files = [file for folder in self.files for file in folder]
        if subsample is not None and subsample < 1:
            self.files: List[str] = random.sample(self.files, int(subsample*len(self.files)))

        self.wl_step = wl_step

    def get_documents(self): #-> List[TaggedDocument]:
        ds_loader = MOFDataset(strategy="vesta")

        self.documents = []
        for cif in tqdm(self.files):
            name = Path(cif).stem
            graph, feature = ds_loader.to_WL_machine(cif)
            machine = WeisfeilerLehmanMachine(graph, feature, self.wl_step)
            word = machine.extracted_features
            doc = TaggedDocument(words=word, tags=[name])

            self.documents.append(doc)
        train_doc, test_doc = train_test_split(self.documents, train_size=0.9, test_size=0.1)
        #valid_documents = random.sample(self.documents, int(0.1*len(self.documents)))

        return self.documents, train_doc, test_doc
    
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
