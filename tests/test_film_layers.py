import torch

from openspliceai.train_base.openspliceai import ExpressionFiLM


def test_expression_film_identity_with_zero_init():
    film = ExpressionFiLM(channels=4, rbp_dim=3, hidden=8, dropout=0.0, noise_std=0.0)
    rbp = torch.randn(2, 3)
    gamma, beta = film(rbp)
    # gamma should be 1, beta 0 due to zero init on final affine
    assert torch.allclose(gamma, torch.ones_like(gamma), atol=1e-6)
    assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-6)


def test_expression_film_modulates_features():
    film = ExpressionFiLM(channels=1, rbp_dim=1, hidden=2, dropout=0.0, noise_std=0.0)
    with torch.no_grad():
        # zero everything then set bias to control gamma/beta
        film.affine[0].weight.zero_()
        film.affine[0].bias.zero_()
        film.affine[1].weight.fill_(1.0)
        film.affine[1].bias.zero_()
        film.affine[-1].weight.zero_()
        film.affine[-1].bias[:] = torch.tensor([1.0, -0.5])
    rbp = torch.ones(1, 1)
    gamma, beta = film(rbp)
    # gamma = 1 + 1 = 2, beta = -0.5
    x = torch.ones(1, 1, 4)
    modulated = gamma * x + beta
    expected = torch.full_like(modulated, 1.5)
    assert torch.allclose(modulated, expected, atol=1e-6)
