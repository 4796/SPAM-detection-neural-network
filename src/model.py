"""PyTorch MLP for spam classification, parametrized for skorch + GridSearchCV."""

import torch.nn as nn

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class SpamMLP(nn.Module):
    """Feedforward network over TF-IDF vectors, ending in a single logit (binary classification)."""

    def __init__(
        self,
        input_dim: int,
        n_hidden_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        super().__init__()
        act_cls = _ACTIVATIONS[activation]

        layers = []
        in_dim = input_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_cls())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X).squeeze(-1)
