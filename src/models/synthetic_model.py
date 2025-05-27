from flax import nnx
import jax.numpy as jnp
from typing import Callable, Sequence

class SyntheticModel(nnx.Module):
    """Abstract Synthetic Model representing a function f(x, y) = u."""
    def __call__(self, x: float, y: float) -> jnp.ndarray:
        raise NotImplementedError("Subclasses should implement __call__")

class FeedForwardNet(SyntheticModel):
    """A simple feedforward neural network for synthetic modeling."""
    def __init__(self, hidden_dims: Sequence[int] = (64, 64, 64),
                 activation: Callable = nnx.relu,
                 output_dim: int = 1, rngs: nnx.Rngs = None):
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.output_dim = output_dim
        self.input = nnx.Linear(2, hidden_dims[0], rngs=rngs)
        self.hidden = [nnx.Linear(hidden_dims[i], hidden_dims[i + 1], rngs=rngs)
                       for i in range(len(hidden_dims) - 1)]
        self.output = nnx.Linear(hidden_dims[-1], output_dim, rngs=rngs)

    @nnx.jit
    def __call__(self, x, y):
        # native batch support with jnp.stack: x, y can now be scalars or arrays
        inputs = jnp.stack([x, y], axis=-1)
        x_out = self.input(inputs)
        x_out = self.activation(x_out)
        for layer in self.hidden:
            x_out = layer(x_out)
            x_out = self.activation(x_out)
        x_out = self.output(x_out)
        return x_out

class ResNetBlock(nnx.Module):
    def __init__(self, features: int, activation: Callable, rngs: nnx.Rngs = None):
        self.activation = activation
        self.linear = nnx.Linear(features, features, rngs=rngs)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.linear(x)
        x = self.activation(x)
        x = self.linear(x)
        return self.activation(x + residual)

class ResNet(SyntheticModel):
    """A ResNet-style synthetic model with skip connections."""
    def __init__(self, hidden_dims: Sequence[int] = (64, 64, 64),
                 activation: Callable = nnx.relu, output_dim: int = 1, rngs: nnx.Rngs = None):
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.output_dim = output_dim
        self.input = nnx.Linear(2, hidden_dims[0], rngs=rngs)
        self.hidden = [ResNetBlock(features=hidden_dims[i],
                                   activation=activation,
                                   rngs=rngs)
                       for i in range(len(hidden_dims) - 1)]
        self.output = nnx.Linear(hidden_dims[-1], output_dim, rngs=rngs)

    @nnx.jit
    def __call__(self, x: float, y: float) -> float:
        # native batch support with jnp.stack: x, y can now be scalars or arrays
        inputs = jnp.stack([x, y], axis=-1)
        x_out = self.input(inputs)
        x_out = self.activation(x_out)
        for layer in self.hidden:
            x_out = layer(x_out)
            x_out = self.activation(x_out)
        x_out = self.output(x_out)
        return x_out