"""
This file contains the implementation of the Variational Autoencoder (VAE) model.
"""

import pytorch_lightning as pl
import torch


class VAE(pl.LightningModule):
    def __init__(
        self,
        in_dim: tuple[int, int, int],
        latent_dim: int,
        out_channels: list[int] = [32, 64, 128, 256, 512],
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ):
        """
        Initializes the VAE model.

        Parameters
        ----------
        in_dim : tuple[int, int, int]
            The input dimension of the model.
            The tuple should be of the form (channels, height, width).
        latent_dim : int
            The dimension of the latent space.
        out_channels : list[int]
            The number of output channels of the convolutional layers.
            The default is [32, 64, 128, 256, 512].
        kernel_size : int, optional
            The kernel size of the convolutional layers.
            The default is 3.
        stride : int, optional
            The stride of the convolutional layers.
            The default is 2.
        padding : int, optional
            The padding of the convolutional layers.
            The default is 1.
        """
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.save_hyperparameters()

        # ----- Encoder -----
        encoder = []

        in_channel = in_dim[0]
        for out_channel in self.out_channels:
            encoder += [
                torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(),
            ]
            in_channel = out_channel
        self.encoder = torch.nn.Sequential(*encoder)

        # Calculating the shape of the encoder output
        self.eval()
        x = torch.Tensor(1, *in_dim)

        conv_shapes = []
        for layer in self.encoder:
            x = layer(x)
            if type(layer).__name__ == "Conv2d":
                conv_shapes.append(x.shape[1:])

        self.encoder_output_dim = conv_shapes[-1]
        encoder_output_flatten = torch.flatten(
            torch.Tensor(*self.encoder_output_dim)
        ).shape[0]
        self.train()

        # ----- Reparametrization
        self.fc_mu = torch.nn.Linear(encoder_output_flatten, latent_dim)

        # The logvar is used instead of the variance to ensure that the
        # variance is always positive.
        # Otherwise the model would be able to learn to produce negative
        # variances which would be problematic.
        self.fc_logvar = torch.nn.Linear(encoder_output_flatten, latent_dim)

        # ----- Decoder -----
        self.decoder_input = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, encoder_output_flatten)
        )

        decoder = []
        in_channels = self.out_channels[::-1]
        conv_shapes = conv_shapes[::-1]
        for i, (in_channel, out_channel) in enumerate(
            zip(in_channels, in_channels[1:])
        ):
            # Calculating the output padding to ensure that the output shape
            # is the same as the input shape.
            # The formula is taken from the PyTorch documentation:
            # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

            out_size = (conv_shapes[i][2] - 1) * stride - 2 * padding + kernel_size
            output_padding = conv_shapes[i + 1][2] - out_size

            decoder += [
                torch.nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    stride,
                    padding,
                    output_padding,
                ),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(),
            ]

        out_size = (conv_shapes[-1][2] - 1) * stride - 2 * padding + kernel_size
        output_padding = in_dim[2] - out_size

        decoder.append(
            torch.nn.ConvTranspose2d(
                in_channels[-1],
                in_dim[0],
                kernel_size,
                stride,
                padding,
                output_padding,
            ),
        )
        self.decoder = torch.nn.Sequential(*decoder)

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
        epsilon = torch.normal(0, 1, size=logvar.shape, device=self.device)
        return mu + std * epsilon

    def decode(self, z: torch.Tensor):
        """
        Decodes the latent space vector into the original input space.

        Parameters
        ----------
        z : torch.Tensor
            The latent space vector.

        Returns
        -------
        torch.Tensor
            The decoded tensor.
        """
        z = self.decoder_input(z)
        z = z.reshape(-1, *self.encoder_output_dim)
        z = self.decoder(z)
        return z

    def generate(self, num_of_samples: int) -> torch.Tensor:
        """
        Generates samples from the latent space.

        Parameters
        ----------
        num_of_samples : int
            The number of samples to generate.

        Returns
        -------
        torch.Tensor
            The generated samples.
        """
        z = torch.normal(0, 1, size=(num_of_samples, self.latent_dim))
        samples = self.decode(z)
        return samples

    def forward(
        self, x: torch.Tensor
    ) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            The input tensor, the mean of the latent space distribution
            the logvar of the latent space distribution and the output tensor.
        """
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        y = self.decode(z)
        return [x, mu, logvar, y]

    def calculate_loss(
        self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the loss of the model, which is the sum of the
        reconstruction loss and the KL divergence.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        mu : torch.Tensor
            The mean of the latent space distribution.
        logvar : torch.Tensor
            The logvar of the latent space distribution.
        y : torch.Tensor
            The output tensor.

        Returns
        -------
        torch.Tensor
            The loss of the model.
        """

        # Reconstruction loss
        reconstruction_loss = torch.nn.functional.mse_loss(x, y)

        # KL divergence
        kl_divergence = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )

        return {
            "loss": reconstruction_loss + 0.00025 * kl_divergence,
            "reconstruction_loss": reconstruction_loss,
            "kl_divergence": kl_divergence,
        }

    def training_step(self, batch: list[torch.Tensor], batch_idx: int):
        """
        Training step of the model.

        Parameters
        ----------
        batch : list[torch.Tensor]
            The batch of data.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The loss of the model.
        """
        x, labels = batch
        _, mu, logvar, y = self.forward(x)
        loss = self.calculate_loss(x, mu, logvar, y)

        self.log("train_loss", loss["loss"])
        self.log("train_reconstruction_loss", loss["reconstruction_loss"])
        self.log("train_kl_divergence", loss["kl_divergence"])

        return loss["loss"]

    def configure_optimizers(self):
        """
        Configures the optimizer of the model.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer of the model.
        """
        return torch.optim.Adam(self.parameters(), lr=0.005)
