import pandas as pd

class VecDataset():
    def __init__(self, target, MOF_id, mofnames, vector_file, label_file, transform=None, target_transform=None):

        self.target = target
        df_vectors = pd.read_csv(vector_file).set_index("type")
        df_labels = pd.read_csv(label_file).set_index(MOF_id)
        
        df_vectors.drop_duplicates(inplace=True)    # Drop duplicates in the embedding space

        self.vectors = df_vectors.loc[mofnames].values
        self.labels = df_labels.loc[mofnames][self.target].values.reshape(-1,len(self.target))

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