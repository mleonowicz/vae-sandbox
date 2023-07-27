"""
Module for tests and test fixtures.

To run the tests, run ``kedro test`` from the project root directory.
"""

from torch import Tensor
import pytest

from vae_sandbox.core.vae import VAE


@pytest.mark.parametrize("in_dim,latent_dim", [((3, 32, 32), 10), ((3, 48, 48), 100)])
def test_encoder(in_dim, latent_dim):
    """
    Test the encoder of the VAE.
    """
    vae = VAE(in_dim=in_dim, latent_dim=latent_dim)
    x = Tensor(1, *in_dim)
    # Eval mode to avoid batch norm related errors
    vae.eval()
    mu, var = vae.encode(x)
    assert mu.shape == (1, latent_dim)
    assert var.shape == (1, latent_dim)


@pytest.mark.parametrize("in_dim,latent_dim", [((3, 32, 32), 10), ((3, 48, 48), 100)])
def test_reparametrization(in_dim, latent_dim):
    """
    Test the reparametrization part of the VAE.
    """
    vae = VAE(in_dim=in_dim, latent_dim=latent_dim)
    mu = Tensor(1, latent_dim)
    logvar = Tensor(1, latent_dim)
    z = vae.reparametrize(mu, logvar)
    assert z.shape == (1, latent_dim)


@pytest.mark.parametrize("in_dim,latent_dim", [((3, 32, 32), 10), ((3, 48, 48), 100)])
def test_decoder(in_dim, latent_dim):
    """
    Test the reparametrization part of the VAE.
    """
    vae = VAE(in_dim=in_dim, latent_dim=latent_dim)
    z = Tensor(1, latent_dim)
    y = vae.decode(z)
    assert y.shape == (1, *in_dim)


@pytest.mark.parametrize("in_dim,latent_dim", [((3, 32, 32), 10), ((3, 48, 48), 100)])
def test_loss(in_dim, latent_dim):
    """
    Test the reparametrization part of the VAE.
    """
    vae = VAE(in_dim=in_dim, latent_dim=latent_dim)
    x = Tensor(1, *in_dim)
    out = vae.forward(x)
    loss = vae.calculate_loss(*out)
    assert loss.shape == ()
