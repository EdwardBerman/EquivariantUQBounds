import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import flax
import flax.linen as nn
from e3nn_jax import IrrepsArray
from typing import Tuple, Optional

class CorrectEquivariantVectorFieldModel(nn.Module):
    """
    Correct equivariant model for vector field regression.
    Now more expressive: uses multiple equivariant layers and nonlinearities
    while still respecting rotational equivariance.
    """
    input_irreps: e3nn.Irreps
    output_irreps: e3nn.Irreps
    hidden_dim: int
    
    def setup(self):
        # define a hidden irreps that includes scalar (0e) and vector (1e) channels
        self.hidden_irreps = e3nn.Irreps(f"{self.hidden_dim}x0e + {self.hidden_dim}x1e")
        
        # two-stage hidden transformations for both mean (mu) and variance (sigma)
        # each path remains equivariant by using e3nn.flax.Linear (which preserves irreps).
        self.lin_in_mu = e3nn.flax.Linear(self.hidden_irreps)
        self.lin_hid_mu = e3nn.flax.Linear(self.hidden_irreps)
        self.lin_out_mu = e3nn.flax.Linear(self.output_irreps)
        
        self.lin_in_sigma = e3nn.flax.Linear(self.hidden_irreps)
        self.lin_hid_sigma = e3nn.flax.Linear(self.hidden_irreps)
        self.lin_out_sigma = e3nn.flax.Linear(self.output_irreps)

    def _nonlinear(self, x: IrrepsArray) -> IrrepsArray:
        """
        Simple equivariant nonlinearity:
        - Apply ReLU to scalar (0e) channels.
        - Keep vector (1e) channels unchanged.
        """
        scalars = x.filter(keep="0e")
        vectors = x.filter(keep="1e")
        
        # ReLU on scalar part
        scalars_activated = IrrepsArray(
            scalars.irreps,
            jax.nn.relu(scalars.array)
        )
        
        # concatenate activated scalars and original vectors
        return e3nn.concatenate([scalars_activated, vectors], axis=-1)

    def __call__(self, x: IrrepsArray) -> Tuple[IrrepsArray, IrrepsArray]:
        # mean path
        h_mu = self.lin_in_mu(x)
        h_mu = self._nonlinear(h_mu)
        h_mu = self.lin_hid_mu(h_mu)
        h_mu = self._nonlinear(h_mu)
        mu = self.lin_out_mu(h_mu)
        
        # variance path
        h_sigma = self.lin_in_sigma(x)
        h_sigma = self._nonlinear(h_sigma)
        h_sigma = self.lin_hid_sigma(h_sigma)
        h_sigma = self._nonlinear(h_sigma)
        sigma_sq = self.lin_out_sigma(h_sigma)
        
        # apply softplus to ensure positive variance
        sigma_sq_array = jax.nn.softplus(sigma_sq.array)
        sigma_sq = IrrepsArray(sigma_sq.irreps, sigma_sq_array)
        
        return mu, sigma_sq

class IncorrectEquivariantVectorFieldModel(nn.Module):
    input_irreps: e3nn.Irreps
    output_irreps: e3nn.Irreps
    hidden_dim: int
    break_type: str = "perturbation"  # Options: "perturbation", "flip", "scale", "rotate", "mix"
    
    def setup(self):
        self.hidden_irreps = e3nn.Irreps(f"{self.hidden_dim}x0e + {self.hidden_dim}x1e")

        self.lin_in_mu = e3nn.flax.Linear(self.hidden_irreps)
        self.lin_hid_mu = e3nn.flax.Linear(self.hidden_irreps)
        self.lin_out_mu = e3nn.flax.Linear(self.output_irreps)
        
        self.lin_in_sigma = e3nn.flax.Linear(self.hidden_irreps)
        self.lin_hid_sigma = e3nn.flax.Linear(self.hidden_irreps)
        self.lin_out_sigma = e3nn.flax.Linear(self.output_irreps)
        
        # Non-equivariant layers
        self.non_equiv_dense1 = nn.Dense(self.hidden_dim)
        self.non_equiv_dense2 = nn.Dense(self.hidden_dim)

    def _nonlinear_equiv(self, x: IrrepsArray) -> IrrepsArray:
        """Standard equivariant nonlinearity"""
        scalars = x.filter(keep="0e")
        vectors = x.filter(keep="1e")
        
        scalars_activated = IrrepsArray(
            scalars.irreps,
            jax.nn.relu(scalars.array)
        )
        return e3nn.concatenate([scalars_activated, vectors], axis=-1)

    def _break_equivariance(self, x: IrrepsArray) -> IrrepsArray:
        """Apply different types of equivariance breaking"""
        x_array = x.array
        
        if self.break_type == "perturbation":
            # Original perturbation approach
            h = self.non_equiv_dense1(x_array)
            h = jax.nn.relu(h)
            h = self.non_equiv_dense2(h)
            h = jax.nn.relu(h)
            x_broken_array = x_array + 0.3 * h[:, : x_array.shape[-1]]
            
        elif self.break_type == "flip":
            # Flip vectors in certain regions of space (position-dependent)
            # This creates a discontinuity in the vector field
            mask = (jnp.sum(x_array**2, axis=-1) > 0.5).astype(float)
            mask = mask.reshape(-1, 1)
            x_broken_array = x_array * (1 - 2 * mask)
            
        elif self.break_type == "scale":
            # Scale vectors differently based on their position
            # This breaks the norm-preservation property of equivariance
            radius = jnp.sqrt(jnp.sum(x_array**2, axis=-1, keepdims=True))
            scale_factor = 1.0 + jnp.sin(radius * 3.0)
            x_broken_array = x_array * scale_factor
            
        elif self.break_type == "rotate":
            # Apply a position-dependent rotation to vectors
            # This breaks equivariance by making rotations inconsistent
            x_norm = jnp.linalg.norm(x_array, axis=-1, keepdims=True) + 1e-8
            angle = jnp.sum(x_array, axis=-1, keepdims=True) * 0.5
            cos_angle = jnp.cos(angle)
            sin_angle = jnp.sin(angle)
            
            # 2D rotation matrix applied to each vector
            x_rotated_0 = cos_angle * x_array[..., 0:1] - sin_angle * x_array[..., 1:2]
            x_rotated_1 = sin_angle * x_array[..., 0:1] + cos_angle * x_array[..., 1:2]
            x_broken_array = jnp.concatenate([x_rotated_0, x_rotated_1], axis=-1)
            
        elif self.break_type == "mix":
            # Mix different components in a non-equivariant way
            # Process through non-equivariant layers
            h = self.non_equiv_dense1(x_array)
            h = jax.nn.relu(h)
            h = self.non_equiv_dense2(h)
            h = jax.nn.tanh(h)  # bounded activation
            
            # Create a mixture of original and transformed
            x_broken_array = x_array * (1.0 - jnp.abs(h[:, :1])) + h[:, :x_array.shape[-1]] * h[:, :1]
        
        else:
            x_broken_array = x_array  # No breaking if unknown type
            
        return IrrepsArray(x.irreps, x_broken_array)

    def __call__(self, x: IrrepsArray) -> Tuple[IrrepsArray, IrrepsArray]:
        # Break equivariance in interesting ways
        x_broken = self._break_equivariance(x)
        
        # Proceed with "equivariant" transformations
        h_mu = self.lin_in_mu(x_broken)
        h_mu = self._nonlinear_equiv(h_mu)
        h_mu = self.lin_hid_mu(h_mu)
        h_mu = self._nonlinear_equiv(h_mu)
        mu = self.lin_out_mu(h_mu)
        
        # Variance path
        h_sigma = self.lin_in_sigma(x_broken)
        h_sigma = self._nonlinear_equiv(h_sigma)
        h_sigma = self.lin_hid_sigma(h_sigma)
        h_sigma = self._nonlinear_equiv(h_sigma)
        sigma_sq = self.lin_out_sigma(h_sigma)
        
        # Apply softplus to ensure non-negative variance
        sigma_sq_array = jax.nn.softplus(sigma_sq.array)
        sigma_sq = e3nn.IrrepsArray(sigma_sq.irreps, sigma_sq_array)
        
        return mu, sigma_sq


class MLPVectorFieldModel(nn.Module):
    hidden_dim: int
    output_dim: int = 2  # default for 2D vector field (e3nn 3d lol)
    
    def setup(self):
        # mean prediction layers
        self.linear_mu_one = nn.Dense(self.hidden_dim)
        self.linear_mu_two = nn.Dense(self.hidden_dim)
        self.linear_mu_three = nn.Dense(self.output_dim)
        
        # uncertainty prediction layers
        self.linear_sigma_one = nn.Dense(self.hidden_dim)
        self.linear_sigma_two = nn.Dense(self.hidden_dim)
        self.linear_sigma_three = nn.Dense(self.output_dim)

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # process mean path
        mu = self.linear_mu_one(x)
        mu = jax.nn.relu(mu)
        mu = self.linear_mu_two(mu)
        mu = jax.nn.relu(mu)
        mu = self.linear_mu_three(mu)
        
        # process uncertainty path
        sigma_sq = self.linear_sigma_one(x)
        sigma_sq = jax.nn.relu(sigma_sq)
        sigma_sq = self.linear_sigma_two(sigma_sq)
        sigma_sq = jax.nn.relu(sigma_sq)
        sigma_sq = self.linear_sigma_three(sigma_sq)
        
        # apply softplus to ensure positive uncertainty
        sigma_sq = jax.nn.softplus(sigma_sq)
        
        return mu, sigma_sq 
