from loguru import logger
from typing import Optional
import pandas as pd
import numpy as np
from mofgraph2vec.data.spliter import train_test_stratified_split
from mofgraph2vec.data.dataset import VecDataset

class DataModuleFactory:
    def __init__(
        self,
        task: str,
        MOF_id: str,
        label_path: str,
        embedding_path: str,
        train_frac: float=0.8,
        test_frac: float=0.2,
        seed: Optional[int] = 1234, 
        **kwargs
    ):
        """Data loader for MOF embeddingss

        Args:
            task (str): the name of the target column
            MOF_id (str): the name of the MOF id column
            label_path (str): the path to the .csv file
            embedding_path (str): the path to the cif files
            train_frac (float, optional): the training size of the downstream task. Defaults to 0.8.
            test_frac (float, optional): the test size of the downstream. Defaults to 0.2.
            seed (Optional[int], optional): random seed. Defaults to 1234.

        Raises:
            ValueError: Fractions must sum to 1.0
        """

        if not (train_frac + test_frac == 1.0):
            raise ValueError("Fractions must sum to 1.0")
        
        self.task = task
        self.MOF_id = MOF_id
        
        self.label_path = label_path
        self.embedding_path = embedding_path

        df_label = pd.read_csv(label_path)
        df_feat = pd.read_csv(embedding_path)
        logger.debug(f"Loading embedded features from {embedding_path}")
        embedded_mofs = list(df_feat["type"])
        df_label = df_label[df_label[self.MOF_id].isin(embedded_mofs)].set_index(self.MOF_id)
        df_label = df_label.dropna(subset=self.task)

        train_idx, test_idx = train_test_stratified_split(
            df_label, train_frac, test_frac, self.task, seed=seed
        )

        self.train_names = [df_label.iloc[i].name for i in train_idx]
        self.test_names = [df_label.iloc[i].name for i in test_idx]

        # fit transformers
        self.transform = None
        self.target_transform = None

        logger.info(
            f"Train: {len(self.train_names)} Test: {len(self.test_names)}"
        )

    def get_train_dataset(self, **kwargs):
        return VecDataset(
            target=self.task,
            MOF_id=self.MOF_id,  
            mofnames=self.train_names, 
            vector_file=self.embedding_path, 
            label_file=self.label_path, 
            transform=self.transform, 
            target_transform=self.target_transform,
        )

    def get_test_dataset(self, **kwargs):
        if self.test_names is None:
            return None
        return VecDataset(
            target=self.task, 
            MOF_id=self.MOF_id, 
            mofnames=self.test_names, 
            vector_file=self.embedding_path, 
            label_file=self.label_path, 
            transform=self.transform, 
            target_transform=self.target_transform,
        )
