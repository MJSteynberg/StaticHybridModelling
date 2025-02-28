import os
import sys
from timeit import timeit

import jax
import jax.numpy as jnp
from flax import nnx

# Ensure that the project root is on sys.path.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.synthetic_model import FeedForwardNet, ResNetSynthetic, SyntheticModel


def test_feedforward_net_output_shape():
    # Create dummy inputs
    batch_size = 10
    x = jnp.linspace(0, 1, batch_size)
    y = jnp.linspace(1, 2, batch_size)
    
    # Initialize the model with NNX
    model = FeedForwardNet(hidden_dims=(32, 32), output_dim=1, rngs=nnx.Rngs(42))
    
    # Apply the model with vmap to handle batch dimension
    outputs = jax.vmap(model)(x, y)
    
    # Assert that outputs shape equals batch_size
    assert outputs.shape == (batch_size,1), f"Expected ({batch_size},1) but got {outputs.shape}"

def test_resnet_synthetic_output_shape():
    # Create dummy inputs
    batch_size = 10
    x = jnp.linspace(0, 1, batch_size)
    y = jnp.linspace(1, 2, batch_size)
    
    # Initialize the ResNet synthetic model with NNX
    model = ResNetSynthetic(hidden_dims=(32, 32, 32), output_dim=1, rngs=nnx.Rngs(0))
    
    # Apply the model with vmap to handle batch dimension
    outputs = jax.vmap(model)(x, y)
    
    # Assert that outputs shape equals batch_size
    assert outputs.shape == (batch_size,1), f"Expected ({batch_size},1) but got {outputs.shape}"

def test_model_scalar_inputs():
    """Test that the models can handle scalar inputs (non-batched)."""
    # Initialize models
    ffn = FeedForwardNet(hidden_dims=(32, 32), output_dim=1, rngs=nnx.Rngs(42))
    resnet = ResNetSynthetic(hidden_dims=(32, 32, 32), output_dim=1, rngs=nnx.Rngs(0))
    
    # Test with scalar inputs
    x, y = 0.5, 0.7
    
    # Apply models to single point
    output_ffn = ffn(x, y)
    output_resnet = resnet(x, y)
    
    # Check scalar output
    assert output_ffn.ndim == 1, f"Expected scalar output, got shape {output_ffn.shape}"
    assert output_resnet.ndim == 1, f"Expected scalar output, got shape {output_resnet.shape}"

def test_model_jit_compatibility():
    """Test that the models work with JAX JIT compilation."""
    # Initialize models
    ffn = FeedForwardNet(hidden_dims=(32, 32), output_dim=1, rngs=nnx.Rngs(42))
    resnet = ResNetSynthetic(hidden_dims=(32, 32, 32), output_dim=1, rngs=nnx.Rngs(0))
    
    # Create JIT-compiled versions
    jitted_ffn = jax.jit(lambda x, y: ffn(x, y))
    jitted_resnet = jax.jit(lambda x, y: resnet(x, y))
    
    # Test with scalar inputs
    x, y = 0.5, 0.7
    
    # Apply JIT-compiled functions
    output_ffn = jitted_ffn(x, y)
    output_resnet = jitted_resnet(x, y)
    
    # Check outputs
    assert output_ffn.ndim == 1, f"Expected scalar output, got shape {output_ffn.shape}"
    assert output_resnet.ndim == 1, f"Expected scalar output, got shape {output_resnet.shape}"


if __name__ == "__main__":
    test_feedforward_net_output_shape()
    test_resnet_synthetic_output_shape()
    test_model_scalar_inputs()
    test_model_jit_compatibility()
    print("All tests passed!")