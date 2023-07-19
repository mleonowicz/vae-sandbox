"""
This file contains the implementation of the Variational Autoencoder (VAE) model. 
"""

import pytorch_lightning as pl
from torch import Tensor
from torch import nn


class VAE(pl.LightningModule):
    def __init__(self, in_dim: tuple[int, int, int], latent_dim: int):
        """
        Initializes the VAE model.

        Parameters
        ----------
        in_dim : tuple[int, int, int]
            The input dimension of the model.
            The tuple should be of the form (channels, height, width).
        latent_dim : int
            The dimension of the latent space.
        """
        super().__init__()

        out_channels = [32, 64, 128, 256, 512]

        self.encoder = []

        in_channel = in_dim[0]
        for out_channel in out_channels:
            self.encoder += [
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
            ]
            in_channel = out_channel
        self.encoder.append(nn.Flatten())

        self.encoder = nn.Sequential(*self.encoder)

        # Calculating the shape of the encoder output
        encoder_output = self.encoder(Tensor(1, *in_dim)).shape[1]
        self.fc_mu = nn.Linear(encoder_output, latent_dim)
        self.fc_var = nn.Linear(encoder_output, latent_dim)

    def encode(self, x: Tensor) -> list[Tensor]:
        """
        Encodes the input tensor into a latent space distribution
        parameterized by a mean and variance,
        It is then used to sample from to produce the latent space vector.

        Parameters
        ----------
        x : Tensor
            The input tensor to encode.

        Returns
        -------
        list[Tensor]
            The mean and variance of the latent space distribution.
        """
        encoded = self.encoder(x)

        mu = self.fc_mu(encoded)
        var = self.fc_var(encoded)
        return [mu, var]

    def decode(self, z: Tensor):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch: list[Tensor], batch_idx: int):
        raise NotImplementedError
