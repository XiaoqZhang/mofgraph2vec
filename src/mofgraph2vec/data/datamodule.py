import os
import random
from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from mofgraph2vec.data.dataset import VecDataset


class DataModuleFactory:
    def __init__(
        self,
        task: str,
        MOF_id: str,
        label_path: str,
        embedding_path: str,
        train_frac: float=0.8,
        valid_frac: float=0.1,
        test_frac: float=0.1,
        batch_size: int=64,
        seed: int=2024
    ):
        if not (train_frac + valid_frac + test_frac == 1.0):
            raise ValueError("Fractions must sum to 1.0")

        self.train_frac = train_frac
        self.valid_frac = valid_frac
        self.test_frac = test_frac
        
        self.task = task
        self.MOF_id = MOF_id
        self.batch_size = batch_size
        
        self.label_path = label_path
        self.embedding_path = embedding_path

        df_label = pd.read_csv(label_path)
        df_feat = pd.read_csv(embedding_path)
        embedded_mofs = list(df_feat["type"])
        df_label = df_label[df_label[self.MOF_id].isin(embedded_mofs)].set_index(self.MOF_id)

        assert len(df_label) == len(embedded_mofs)

        train_valid_idx, test_idx = train_test_split(range(len(df_label)), test_size=test_frac, random_state=seed)
        train_idx, valid_idx = train_test_split(train_valid_idx, test_size=valid_frac, random_state=seed)

        self.train_names = [embedded_mofs[i] for i in train_idx]
        self.valid_names = [embedded_mofs[i] for i in valid_idx]
        self.test_names = [embedded_mofs[i] for i in test_idx]

        logger.info(
            f"Train: {len(self.train_names)} Valid: {len(self.valid_names)} Test: {len(self.test_names)}"
        )

    def get_train_dataset(self, **kwargs):
        return VecDataset(
            target=self.task, 
            mofnames=self.train_names, 
            vector_file=self.embedding_path, 
            label_file=self.label_path, 
            transform=None, 
            target_transform=None
        )


    def get_valid_dataset(self, **kwargs):
        if self.valid_names is None:
            return None
        return VecDataset(
            target=self.task, 
            mofnames=self.valid_names, 
            vector_file=self.embedding_path, 
            label_file=self.label_path, 
            transform=None, 
            target_transform=None
        )

    def get_test_dataset(self, **kwargs):
        if self.test_names is None:
            return None
        return VecDataset(
            target=self.task, 
            mofnames=self.test_names, 
            vector_file=self.embedding_path, 
            label_file=self.label_path, 
            transform=None, 
            target_transform=None
        )