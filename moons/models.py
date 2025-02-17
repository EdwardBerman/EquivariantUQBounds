import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import flax
import flax.linen as nn

class EquivariantMLP(nn.Module):
    input_irreps: e3nn.Irreps
    output_irreps: e3nn.Irreps
    hidden_dim: int

    def setup(self):
        self.hidden_irreps = e3nn.Irreps(f"{self.hidden_dim}x1e")
        self.linear_one = e3nn.flax.Linear(self.hidden_irreps, name='linear_one')
        self.linear_two = e3nn.flax.Linear(self.output_irreps, name='linear_two')

    def __call__(self, x):
        x = self.linear_one(x)
        x = self.linear_two(x)
        return x
