import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, Sequence

class SyntheticModel(nn.Module):
    """Abstract Synthetic Model representing a function f(x, y) = u."""
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Subclasses should implement __call__")
    
class ResNetBlock(nn.Module):
    features: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = nn.Dense(self.features)(x)
        x = self.activation(x)
        x = nn.Dense(self.features)(x)
        return self.activation(x + residual)

class FeedForwardNet(SyntheticModel):
    """A simple feedforward neural network for synthetic modeling."""
    hidden_dims: Sequence[int] = (64, 64, 64)
    activation: Callable = nn.relu
    output_dim: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        # Concatenate x and y into a single feature vector.
        inputs = jnp.concatenate([x[..., None], y[..., None]], axis=-1)
        x = inputs
        for feat in self.hidden_dims:
            x = nn.Dense(features=feat)(x)
            x = self.activation(x)
        x = nn.Dense(features=self.output_dim)(x)
        # If output_dim is 1, squeeze the trailing dimension.
        if self.output_dim == 1:
            x = jnp.squeeze(x, axis=-1)
        return x



class ResNetSynthetic(SyntheticModel):
    """A ResNet-style synthetic model with skip connections."""
    num_blocks: int = 3
    features: int = 64
    activation: Callable = nn.relu
    output_dim: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([x[..., None], y[..., None]], axis=-1)
        x = nn.Dense(self.features)(inputs)
        x = self.activation(x)
        for _ in range(self.num_blocks):
            x = ResNetBlock(features=self.features, activation=self.activation)(x)
        x = nn.Dense(self.output_dim)(x)
        if self.output_dim == 1:
            x = jnp.squeeze(x, axis=-1)
        return x


