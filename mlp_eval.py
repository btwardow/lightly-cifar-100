import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import CIFAR100

import lightly
from lightly.utils import BenchmarkModule
from lightly.utils.benchmarking import knn_predict


class SSLEvaluator(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.out_features = n_classes  # for *head* compability
        if n_hidden is None or n_hidden == 0:
            # use linear classifier
            self.model = nn.Sequential(nn.Flatten(), nn.Dropout(p=p), nn.Linear(n_input, n_classes, bias=True))
        else:
            # use simple MLP classifier
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True),
            )

    def forward(self, x):
        logits = self.model(x)
        return logits


class MLP(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_size: int = 4096, output_dim: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
