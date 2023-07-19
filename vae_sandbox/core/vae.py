"""
This file contains the implementation of the Variational Autoencoder (VAE) model. 
"""

import pytorch_lightning as pl
import torch


class VAE(pl.LightningModule):
    def __init__(self):
        super(self).__init__()

    def encode(self, x: torch.Tensor):
        raise NotImplementedError

    def decode(self, z: torch.Tensor):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch: list[torch.Tensor], batch_idx: int):
        raise NotImplementedError
