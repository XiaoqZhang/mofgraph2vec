from loguru import logger
from typing import Optional
import torch
import pandas as pd
import numpy as np
from mofgraph2vec.data.spliter import train_valid_test_split
from torch.utils.data import DataLoader
from mofgraph2vec.data.dataset import VecDataset
from torch_geometric.data import LightningDataset
from sklearn.preprocessing import MinMaxScaler

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
        num_workers: Optional[int] = None,
        device: Optional[torch.device] = "cpu",
        seed: Optional[int] = 1234, 
        **kwargs
    ):
        self.device = device
        self.num_workers = num_workers

        if not (train_frac + valid_frac + test_frac == 1.0):
            raise ValueError("Fractions must sum to 1.0")
        
        self.task = task
        self.MOF_id = MOF_id
        self.batch_size = batch_size
        
        self.label_path = label_path
        self.embedding_path = embedding_path

        df_label = pd.read_csv(label_path)
        df_feat = pd.read_csv(embedding_path)
        embedded_mofs = list(df_feat["type"])
        df_label = df_label[df_label[self.MOF_id].isin(embedded_mofs)].set_index(self.MOF_id)
        df_label = df_label.dropna(subset=self.task)

        train_idx, valid_idx, test_idx = train_valid_test_split(
            df_label, train_frac, valid_frac, test_frac, self.task, seed=seed
        )
        train_valid_idx = np.concatenate([train_idx, valid_idx])

        self.train_names = [df_label.iloc[i].name for i in train_idx]
        self.valid_names = [df_label.iloc[i].name for i in valid_idx]
        self.test_names = [df_label.iloc[i].name for i in test_idx]

        # fit transformers
        train_valid_names = np.concatenate([self.train_names, self.valid_names])
        df_feat = df_feat.set_index("type")
        x_to_transform = df_feat[df_feat.index.isin(train_valid_names)].values
        #self.transform = MinMaxScaler().fit(x_to_transform)
        #self.target_transform = MinMaxScaler().fit(df_label.iloc[train_valid_idx][self.task].values.reshape(-1,1))
        self.transform = None
        self.target_transform = None

        logger.info(
            f"Train: {len(self.train_names)} Valid: {len(self.valid_names)} Test: {len(self.test_names)}"
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
            device=self.device
        )

    def get_valid_dataset(self, **kwargs):
        if self.valid_names is None:
            return None
        return VecDataset(
            target=self.task,
            MOF_id=self.MOF_id, 
            mofnames=self.valid_names, 
            vector_file=self.embedding_path, 
            label_file=self.label_path, 
            transform=self.transform, 
            target_transform=self.target_transform,
            device=self.device
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
            device=self.device
        )

    def get_datamodule(self) -> LightningDataset:
        return LightningDataset(
            train_dataset=self.get_train_dataset(),
            val_dataset=self.get_valid_dataset(),
            test_dataset=self.get_test_dataset(),
            batch_size=self.batch_size,
        )

    def train_dataloader(self):
        return DataLoader(self.get_train_dataset(), batch_size=self.batch_size)
    
    def valid_dataloader(self):
        return DataLoader(self.get_valid_dataset(), batch_size=self.batch_size)
     
    def test_dataloader(self):
        return DataLoader(self.get_test_dataset(), batch_size=self.batch_size)