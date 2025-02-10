import os
import sys
from timeit import timeit

import jax
import jax.numpy as jnp

# Ensure that the project root is on sys.path.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


import jax
import jax.numpy as jnp
from models.synthetic_model import FeedForwardNet, ResNetSynthetic, SyntheticModel


def test_feedforward_net_output_shape():
    # Create dummy batched inputs.
    batch_size = 10
    x = jnp.linspace(0, 1, batch_size)
    y = jnp.linspace(1, 2, batch_size)
    
    # Initialize the model.
    model = FeedForwardNet(hidden_dims=(32, 32), output_dim=1)
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, x, y)
    
    # Apply the model.
    outputs = model.apply(params, x, y)
    
    # Assert that outputs shape equals batch_size.
    # When output_dim=1, the final output is squeezed to shape (batch_size,).
    assert outputs.shape == (batch_size,), f"Expected ({batch_size},) but got {outputs.shape}"

def test_resnet_synthetic_output_shape():
    # Create dummy batched inputs.
    batch_size = 10
    x = jnp.linspace(0, 1, batch_size)
    y = jnp.linspace(1, 2, batch_size)
    
    # Initialize the ResNet synthetic model.
    model = ResNetSynthetic(num_blocks=3, features=32, output_dim=1)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, x, y)
    
    # Apply the model.
    outputs = model.apply(params, x, y)
    
    # Assert that outputs shape equals batch_size.
    assert outputs.shape == (batch_size,), f"Expected ({batch_size},) but got {outputs.shape}"

if __name__ == "__main__":
    test_feedforward_net_output_shape()
    test_resnet_synthetic_output_shape()
    print("All tests passed!")