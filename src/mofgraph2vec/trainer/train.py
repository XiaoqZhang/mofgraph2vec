from loguru import logger
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from mofgraph2vec.data.dataset import VecDataset
from mofgraph2vec.data.datamodule import DataModuleFactory
from mofgraph2vec.model.vecnn import VecModel
from mofgraph2vec.model.lightningmodule import VecLightningModule

def train(
    task,
    id_column,
    label_path,
    embedding_path,
    train_frac,
    valid_frac,
    test_frac,
    batch_size,
    seed
):
    dm = DataModuleFactory(
        task=task,
        MOF_id=id_column,
        label_path=label_path,
        embedding_path=embedding_path,
        train_frac=train_frac,
        valid_frac=valid_frac,
        test_frac=test_frac,
        seed=seed
    )

    train_ds = dm.get_train_dataset()
    valid_ds = dm.get_valid_dataset()
    test_ds = dm.get_test_dataset()
    
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    pl_model = VecLightningModule(model="nn", loss="mse", lr=1e-3)

    trainer = pl.Trainer(limit_train_batches=100, max_epochs=100)

    logger.info(f"Start fitting")
    trainer.fit(pl_model, train_loader, valid_loader)

    logger.info(f"Start testing")
    trainer.test(pl_model, test_loader)
