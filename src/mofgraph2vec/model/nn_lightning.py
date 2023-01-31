from typing import Optional
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from mofgraph2vec.model.vecnn import VecModel
from mofgraph2vec.utils.loss import get_regression_metrics

class VecLightningModule(LightningModule):
    def __init__(
        self,
        model: str = "nn",
        input_dim: int = 64,
        output_dim: int = 1,
        fcnn_n_layers: int = 3,
        fcnn_hidden_size: int = 128, 
        fcnn_activation: str = "relu",
        dropout: Optional[float] = None,
        loss: str = "mse",
        lr: float = 1e-3,
        **kwargs
    ):
        super().__init__()

        if model == "nn":
            self.model = VecModel(
                input_dim,
                output_dim,
                fcnn_n_layers,
                fcnn_hidden_size,
                fcnn_activation,
                dropout
            )
        if loss == "mse":
            self.loss = nn.MSELoss()
        self.lr = lr
        self.metrics = get_regression_metrics

    def training_step(self, batch, batch_index):
        x, y = batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log("train_loss", loss, batch_size=len(y))
        self.log_dict(self.metrics(y, pred, "train"), batch_size=len(y))
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log("valid_loss", loss, batch_size=len(y))
        self.log_dict(self.metrics(y, pred, "valid"), batch_size=len(y))
        return loss

    def test_step(self, batch, batch_index):
        x, y = batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log("test_loss", loss, batch_size=len(y))
        self.log_dict(self.metrics(y, pred, "test"), batch_size=len(y))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer