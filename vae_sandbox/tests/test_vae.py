"""
Module for tests and test fixtures.

To run the tests, run ``kedro test`` from the project root directory.
"""

from torch import Tensor
import pytest

from vae_sandbox.core.vae import VAE


@pytest.mark.parametrize(
    "in_dim,latent_dim,out_channels",
    [((3, 32, 32), 10, None), ((3, 48, 48), 100, None), ((3, 28, 28), 10, [32, 64])],
)
class TestVae:
    def test_encoder(self, in_dim, latent_dim, out_channels):
        """
        Test the encoder of the VAE.
        """
        vae = VAE(in_dim=in_dim, latent_dim=latent_dim, out_channels=out_channels)
        x = Tensor(1, *in_dim)
        vae.eval()
        mu, var = vae.encode(x)
        assert mu.shape == (1, latent_dim)
        assert var.shape == (1, latent_dim)

    def test_reparametrization(self, in_dim, latent_dim, out_channels):
        """
        Test the reparametrization part of the VAE.
        """
        vae = VAE(in_dim=in_dim, latent_dim=latent_dim, out_channels=out_channels)
        mu = Tensor(1, latent_dim)
        logvar = Tensor(1, latent_dim)
        z = vae.reparametrize(mu, logvar)
        assert z.shape == (1, latent_dim)

    def test_decoder(self, in_dim, latent_dim, out_channels):
        """
        Test the decoder of the VAE.
        """
        vae = VAE(in_dim=in_dim, latent_dim=latent_dim, out_channels=out_channels)
        z = Tensor(1, latent_dim)
        y = vae.decode(z)
        assert y.shape == (1, *in_dim)

    def test_loss(self, in_dim, latent_dim, out_channels):
        """
        Test the loss function of the VAE.
        """
        vae = VAE(in_dim=in_dim, latent_dim=latent_dim, out_channels=out_channels)
        x = Tensor(1, *in_dim)
        out = vae.forward(x)
        loss = vae.calculate_loss(*out)
        assert loss.shape == ()

    def test_vae(self, in_dim, latent_dim, out_channels):
        """
        Test the whole model.
        """
        vae = VAE(in_dim=in_dim, latent_dim=latent_dim, out_channels=out_channels)
        x = Tensor(1, *in_dim)
        out = vae(x)
        assert out[0].shape == (1, *in_dim)
        assert out[1].shape == (1, latent_dim)
        assert out[2].shape == (1, latent_dim)
        assert out[3].shape == (1, *in_dim)
