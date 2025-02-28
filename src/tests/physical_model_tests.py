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

from models.physical_model import PhysicalModel

pi = 3.141592653589793

# Forcing function for default test: f(x,y) = 2π² sin(pi*x) sin(pi*y)
def forcing_func(x, y):
    return 2 * pi**2 * jnp.sin(pi * x) * jnp.sin(pi * y)

# Analytical solution for default test: u_exact(x,y) = sin(pi*x)*sin(pi*y)
def u_exact(x, y):
    return jnp.sin(pi * x) * jnp.sin(pi * y)

# Constant coefficient functions for the known-coeff test.
def constant_kappa(parameters, x, y):
    # Ignores parameters.
    return 2.0

def constant_eta(parameters, x, y):
    # Provided for completeness.
    return 3.0

# Forcing function for known-coeff test:
# With u_exact = sin(pi*x)*sin(pi*y), Δ u_exact = -2π² sin(pi*x)*sin(pi*y)
# and so -div(kappa grad u_exact) = -2 Δ u_exact = 4π² sin(pi*x)*sin(pi*y)
def forcing_known(x, y):
    return 4 * pi**2 * jnp.sin(pi * x) * jnp.sin(pi * y)

# A helper to generate evaluation points on a grid.
def eval_points(n=200, domain=(0.0, 1.0)):
    x = jnp.linspace(domain[0], domain[1], n)
    y = jnp.linspace(domain[0], domain[1], n)
    x_grid, y_grid = jnp.meshgrid(x, y)
    points = jnp.stack([x_grid.flatten(), y_grid.flatten()], axis=-1)
    return points

# Default coefficient functions that match the original behavior
def default_kappa(parameters, x, y):
    return 1.0

def default_eta(parameters, x, y):
    return 0.0

def test_poisson_model_defaults():
    # Test with default kappa and eta (kappa=1.0, eta=0.0)
    domain = (0.0, 1.0)
    N = 30  # Mesh resolution
    parameters = jnp.array([1.0, 1.0])
    
    # Instantiate the model with PhysicalModel
    model = PhysicalModel(
        domain=domain, 
        N=N, 
        parameters=parameters, 
        training=False, 
        forcing_func=forcing_func,
        kappa_func=default_kappa,
        eta_func=default_eta,
        rngs=nnx.Rngs(0)
    )
    
    # Create evaluation grid of 200x200 points.
    points = eval_points(n=200, domain=domain)
    
    # Call the model to compute the solution at each point using vmap
    u_computed = jax.vmap(model)(points[:, 0], points[:, 1])
    print("Default Coefficients:")
    print(f"Computed solution shape: {u_computed.shape}")
    
    # Evaluate exact solution on the grid.
    u_expected = jnp.array([u_exact(x, y) for x, y in points])
    error = jnp.abs(u_computed.flatten() - u_expected)
    mse_error = jnp.mean(error ** 2)
    print(f"MSE error: {mse_error:.4f}")
    assert mse_error < 1e-2, f"MSE error too large: {mse_error}"

def test_poisson_model_known_coeff():
    # Test with constant, non-unity coefficients and a forcing function
    # chosen so that u_exact = sin(pi*x)*sin(pi*y) is the exact solution.
    domain = (0.0, 1.0)
    N = 30  # Mesh resolution
    # Parameters are not used by our constant coefficients, but provided for compatibility.
    parameters = jnp.array([0.0])
    
    # Instantiate the model with PhysicalModel
    model = PhysicalModel(
        domain=domain,
        N=N,
        parameters=parameters,
        training=False,
        forcing_func=forcing_known,
        kappa_func=constant_kappa,
        eta_func=constant_eta,
        rngs=nnx.Rngs(0)
    )
    
    # Create evaluation grid of 200x200 points.
    points = eval_points(n=200, domain=domain)
    
    # Call the model to compute the solution at each point using vmap
    u_computed = jax.vmap(model)(points[:, 0], points[:, 1])
    print("\nKnown Coefficients Test:")
    print(f"Computed solution shape: {u_computed.shape}")
    
    # Evaluate exact solution on the grid.
    u_expected = jnp.array([u_exact(x, y) for x, y in points])
    error = jnp.abs(u_computed.flatten() - u_expected)
    mse_error = jnp.mean(error**2)
    print(f"MSE error: {mse_error:.4f}")
    assert mse_error < 1e-2, f"MSE error too large: {mse_error}"

def nonconstant_kappa(parameters, x, y):
    # Nonconstant kappa: 1 + x*y
    return 1.0 + x * y

def nonconstant_eta(parameters, x, y):
    # Nonconstant eta: 1 + x - y (for demonstration)
    return 1.0 + x - y

def forcing_nonc(x, y):
    # For u_exact = sin(pi*x)*sin(pi*y), the forcing function becomes:
    # f(x,y) = -[ y*pi*cos(pi*x)*sin(pi*y) + x*pi*sin(pi*x)*cos(pi*y) ]
    #          + 2*(1 + x*y)*pi**2*sin(pi*x)*sin(pi*y)
    return - ( y * pi * jnp.cos(pi*x) * jnp.sin(pi*y)
               + x * pi * jnp.sin(pi*x) * jnp.cos(pi*y) ) \
           + 2 * (1 + x * y) * pi**2 * jnp.sin(pi*x) * jnp.sin(pi*y)

def test_poisson_model_nonconstant():
    # Test with nonconstant kappa and eta so that
    # u_exact = sin(pi*x)*sin(pi*y) is the exact solution.
    domain = (0.0, 1.0)
    N = 30  # Mesh resolution
    # parameters are not used here but provided for compatibility.
    parameters = jnp.array([0.0])
    
    # Instantiate the model with PhysicalModel
    model = PhysicalModel(
        domain=domain,
        N=N,
        parameters=parameters,
        training=False,
        forcing_func=forcing_nonc,
        kappa_func=nonconstant_kappa,
        eta_func=nonconstant_eta,
        rngs=nnx.Rngs(0)
    )
    
    # Create evaluation grid of 200x200 points.
    points = eval_points(n=200, domain=domain)
    
    # Call the model to compute the solution at each point using vmap
    u_computed = jax.vmap(model)(points[:, 0], points[:, 1])
    print("\nNonconstant Coefficients Test:")
    print(f"Computed solution shape: {u_computed.shape}")
    
    # Evaluate exact solution on the grid.
    u_expected = jnp.array([u_exact(x, y) for x, y in points])
    error = jnp.abs(u_computed.flatten() - u_expected)
    mse_error = jnp.mean(error**2)
    print(f"MSE error: {mse_error:.4f}")
    assert mse_error < 1e-2, f"MSE error too large: {mse_error}"

if __name__ == "__main__":
    test_poisson_model_defaults()
    test_poisson_model_known_coeff()
    test_poisson_model_nonconstant()
    print("All tests passed!")