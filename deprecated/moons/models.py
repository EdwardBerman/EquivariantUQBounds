import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import flax
import flax.linen as nn
from e3nn_jax import IrrepsArray

class EquivariantMLP(nn.Module):
    input_irreps: e3nn.Irreps
    output_irreps: e3nn.Irreps
    hidden_dim: int

    def setup(self):
        self.hidden_irreps = e3nn.Irreps(f"{self.hidden_dim}x1e")
        self.linear_mu_one = e3nn.flax.Linear(self.hidden_irreps, name='linear_mu_one')
        self.linear_mu_two = e3nn.flax.Linear(self.output_irreps, name='linear_mu_two')
        self.linear_sigma_one = e3nn.flax.Linear(self.hidden_irreps, name='linear_sigma_one')
        self.linear_sigma_two = e3nn.flax.Linear(self.output_irreps, name='linear_sigma_two')


    def __call__(self, x):
        mu = self.linear_mu_one(x)
        mu = self.linear_mu_two(mu)

        sigma_sq = self.linear_sigma_one(x)
        sigma_sq = self.linear_sigma_two(sigma_sq)

        sigma_sq_array = jax.nn.softplus(sigma_sq.array)
        sigma_sq = e3nn.IrrepsArray(sigma_sq.irreps, sigma_sq_array)
        return mu, sigma_sq

class MLP(nn.Module):
    hidden_dim: int

    def setup(self):
        self.linear_mu_one = nn.Dense(self.hidden_dim)
        self.linear_mu_two = nn.Dense(2)
        self.linear_sigma_one = nn.Dense(self.hidden_dim)
        self.linear_sigma_two = nn.Dense(2)

    def __call__(self, x):
        mu = self.linear_mu_one(x)
        mu = self.linear_mu_two(mu)

        sigma_sq = self.linear_sigma_one(x)
        sigma_sq = self.linear_sigma_two(sigma_sq)

        sigma_sq_array = jax.nn.softplus(sigma_sq)
        return mu, sigma_sq_array
