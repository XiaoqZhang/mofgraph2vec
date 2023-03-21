import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from mofgraph2vec.utils.loss import get_regression_metrics
from typing import Optional

class VecLightningModule(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: str = "mse",
        lr: float = 1e-3,
        mc_iterations: Optional[int] = 5,
        **kwargs
    ):
        super().__init__()

        self.model = model
        if loss == "mse":
            self.loss = nn.MSELoss()
        self.lr = lr
        self.metrics = get_regression_metrics
        self.dropout = nn.Dropout()
        self.mc_iteration = mc_iterations
        self.save_hyperparameters()
    
    def forward(self, x):
        pred = self.model(x)
        return pred

    def training_step(self, batch, batch_index):
        x, y = batch
        y = y.reshape(-1,1)
        pred = self(x)
        loss = self.loss(pred, y)
        self.log("train_loss", loss, batch_size=len(y))
        self.log_dict(self.metrics(y, pred, "train"), batch_size=len(y))
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        y = y.reshape(-1,1)
        pred = self(x)
        loss = self.loss(pred, y)
        self.log("valid_loss", loss, batch_size=len(y))
        self.log_dict(self.metrics(y, pred, "valid"), batch_size=len(y))
        return loss

    def test_step(self, batch, batch_index):
        x, y = batch
        y = y.reshape(-1,1)
        pred = self(x)
        loss = self.loss(pred, y)
        self.log("test_loss", loss, batch_size=len(y))
        self.log_dict(self.metrics(y, pred, "test"), batch_size=len(y))
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y = y.reshape(-1,1)

        # enable Monte Carlo Dropout
        self.dropout.train()

        # take average of `self.mc_iteration` iterations
        pred = torch.vstack([self(x).unsqueeze(0) for _ in range(self.mc_iteration)]).mean(dim=0)
        #pred = torch.vstack([self.dropout(self(x)).unsqueeze(0) for _ in range(self.mc_iteration)]).mean(dim=0)
        #pred_std = torch.vstack([self.dropout(self(x)).unsqueeze(0) for _ in range(self.mc_iteration)]).std(dim=0)
        return y, pred#, pred_std

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
