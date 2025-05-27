import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from models.synthetic_model import FeedForwardNet, ResNetSynthetic

@pytest.mark.parametrize("ModelClass,hidden_dims,output_dim", [
    (FeedForwardNet, (32, 32), 1),
    (ResNetSynthetic, (32, 32, 32), 1),
])
def test_output_shape_scalar(ModelClass, hidden_dims, output_dim):
    # Test output shape for scalar input
    model = ModelClass(hidden_dims=hidden_dims, output_dim=output_dim, rngs=nnx.Rngs(42))
    x, y = 0.5, 0.7
    output = model(x, y)
    assert output.shape == (output_dim,) or output.ndim == 1

@pytest.mark.parametrize("ModelClass,hidden_dims,output_dim", [
    (FeedForwardNet, (32, 32), 1),
    (ResNetSynthetic, (32, 32, 32), 1),
])
def test_output_shape_batched(ModelClass, hidden_dims, output_dim):
    # Test output shape for batched input
    batch_size = 10
    model = ModelClass(hidden_dims=hidden_dims, output_dim=output_dim, rngs=nnx.Rngs(42))
    x = jnp.linspace(0, 1, batch_size)
    y = jnp.linspace(1, 2, batch_size)
    outputs = model(x, y)
    assert outputs.shape == (batch_size, output_dim)

@pytest.mark.parametrize("ModelClass,hidden_dims,output_dim", [
    (FeedForwardNet, (8, 8), 2),
    (ResNetSynthetic, (16, 16, 16), 3),
])
def test_model_initialization(ModelClass, hidden_dims, output_dim):
    # Test model initialization with different hidden_dims and output_dim
    model = ModelClass(hidden_dims=hidden_dims, output_dim=output_dim, rngs=nnx.Rngs(0))
    x, y = 0.1, 0.2
    output = model(x, y)
    assert output.shape[-1] == output_dim

@pytest.mark.parametrize("ModelClass,hidden_dims", [
    (FeedForwardNet, (32, 32)),
    (ResNetSynthetic, (32, 32, 32)),
])
def test_error_on_wrong_input_shape(ModelClass, hidden_dims):
    # Test that model raises an error for mismatched input shapes
    model = ModelClass(hidden_dims=hidden_dims, output_dim=1, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 2))
    y = jnp.ones((3, 3))  # Mismatched shape
    with pytest.raises(Exception):
        model(x, y)

@pytest.mark.parametrize("ModelClass,hidden_dims,output_dim", [
    (FeedForwardNet, (32, 32), 1),
    (ResNetSynthetic, (32, 32, 32), 1),
])
def test_nnx_jit_compatibility(ModelClass, hidden_dims, output_dim):
    model = ModelClass(hidden_dims=hidden_dims, output_dim=output_dim, rngs=nnx.Rngs(42))

    @nnx.jit
    def call_model(m, x, y):
        return m(x, y)

    x, y = 0.5, 0.7
    output = call_model(model, x, y)
    assert output.shape == (output_dim,) or output.ndim == 1

