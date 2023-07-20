"""
This file contains the implementation of the Variational Autoencoder (VAE) model.
"""

import pytorch_lightning as pl
import torch


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
        self.in_dim = in_dim
        self.latent_dim = latent_dim

        out_channels = [32, 64, 128]

        # ----- Encoder -----
        self.encoder = []

        in_channel = in_dim[0]
        for out_channel in out_channels:
            self.encoder += [
                torch.nn.Conv2d(
                    in_channel, out_channel, kernel_size=3, stride=2, padding=1
                ),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(),
            ]
            in_channel = out_channel
        self.encoder = torch.nn.Sequential(*self.encoder)

        # Calculating the shape of the encoder output
        self.eval()
        self.encoder_output_dim = self.encoder(torch.Tensor(1, *in_dim)).shape[1:]
        self.encoder_output_flatten = torch.flatten(
            torch.Tensor(*self.encoder_output_dim)
        ).shape[0]
        self.train()

        # ----- Reparametrization
        self.fc_mu = torch.nn.Linear(self.encoder_output_flatten, latent_dim)

        # The logvar is used instead of the variance to ensure that the
        # variance is always positive.
        # Otherwise the model would be able to learn to produce negative
        # variances which would be problematic.
        self.fc_logvar = torch.nn.Linear(self.encoder_output_flatten, latent_dim)

        # ----- Decoder -----
        self.decoder_input = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, self.encoder_output_flatten)
        )

        self.decoder = []
        in_channels = out_channels[::-1]
        for in_channel, out_channel in zip(in_channels, in_channels[1:]):
            self.decoder += [
                torch.nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(),
            ]
        self.decoder.append(
            torch.nn.ConvTranspose2d(
                in_channels[-1],
                in_dim[0],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )
        self.decoder = torch.nn.Sequential(*self.decoder)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input tensor into a latent space distribution
        parameterized by a mean and variance,
        It is then used to sample from to produce the latent space vector.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to encode.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The mean and variance of the latent space distribution.
        """
        encoded = self.encoder(x)
        flatten = torch.flatten(encoded, start_dim=1)

        mu = self.fc_mu(flatten)
        logvar = self.fc_logvar(flatten)
        return (mu, logvar)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Performs the reparametrization trick to sample from the latent space
        and enable backpropagation.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent space distribution.
        var : torch.Tensor
            Variance of the latent space distribution.

        Returns
        -------
        torch.Tensor
            The latent space vector.
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.normal(0, 1, size=logvar.shape)
        return mu + std * epsilon

    def decode(self, z: torch.Tensor):
        z = self.decoder_input(z)
        z = z.reshape(-1, *self.encoder_output_dim)
        z = self.decoder(z)
        return z

    def generate(self, num_of_samples: int) -> torch.Tensor:
        z = torch.normal(0, 1, size=(num_of_samples, self.latent_dim))
        samples = self.decode(z)
        return samples

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y = self.decode(z)
        return y

    def training_step(self, batch: list[torch.Tensor], batch_idx: int):
        raise NotImplementedError
