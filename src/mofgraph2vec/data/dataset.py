import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class VecDataset(Dataset):
    def __init__(self, target, mofnames, vector_file, label_file, transform=None, target_transform=None):
        self.target = target
        df_vectors = pd.read_csv(vector_file).set_index("type")
        df_labels = pd.read_csv(label_file).set_index("cif.label")

        self.vectors = df_vectors.loc[mofnames].values
        self.labels = df_labels.loc[mofnames][self.target].values

        self.transform = transform
        self.target_transform = target_transform

        if self.transform is not None:
            self.vectors = transform(df_vectors)
        if self.target_transform is not None:
            self.labels = target_transform(df_labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.vectors[idx]
        y = self.labels[idx]
        return torch.Tensor(X), torch.Tensor([y])