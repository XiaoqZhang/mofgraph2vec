import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class VecDataset(Dataset):
    def __init__(self, target, mofnames, vector_file, label_file, transform=None, target_transform=None, device="cpu"):

        self.target = target
        df_vectors = pd.read_csv(vector_file).set_index("type")
        df_labels = pd.read_csv(label_file).set_index("cif.label")

        self.vectors = torch.from_numpy(df_vectors.loc[mofnames].values.astype(np.float32)).to(device)
        self.labels = torch.from_numpy(df_labels.loc[mofnames][self.target].values.astype(np.float32).reshape(-1,len(self.target))).to(device)

        self.transform = transform
        self.target_transform = target_transform

        if self.transform is not None:
            self.vectors = transform.transform(self.vectors)
        if self.target_transform is not None:
            self.labels = target_transform.transform(self.labels)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        X = self.vectors[idx]
        y = self.labels[idx]
        return X, y