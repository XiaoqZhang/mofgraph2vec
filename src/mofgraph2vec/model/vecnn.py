from typing import Optional
import torch.nn as nn
import torch.nn.functional as F

class VecModel(nn.Module):
    def __init__(
        self,
        input_dim: int=64,
        output_dim: int=1,
        fcnn_n_layers: int=3,
        fcnn_hidden_size: int=128,
        fcnn_activation: str="relu",
        dropout: Optional[float] = None
    ) -> None:
        super(VecModel, self).__init__()

        self.dropout = dropout

        self.embedding = nn.Linear(input_dim, fcnn_hidden_size)

        self.convs = nn.ModuleList(
            [
                nn.Linear(fcnn_hidden_size, fcnn_hidden_size)
                for _ in range(fcnn_n_layers)
            ]
        )

        if fcnn_activation == "relu":
            self.activation = nn.ReLU()
        
        self.output = nn.Linear(fcnn_hidden_size, output_dim)

    def forward(self, vec):
        out = self.embedding(vec)
        for conv in self.convs:
            if self.dropout is not None:
                out = F.dropout(out, p=self.dropout)
            out = conv(out)
            out = self.activation(out)
        out = self.output(out)

        return out
