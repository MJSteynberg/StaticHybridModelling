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

from models.physical_model import PoissonModel

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

def test_poisson_model_defaults():
    # Test with default kappa and eta (kappa=1.0, eta=0.0)
    domain = (0.0, 1.0)
    N = 30  # Mesh resolution
    parameters = jnp.array([1.0, 1.0])
    
    # Instantiate the model (using default coefficients).
    model = PoissonModel(parameters=parameters, domain=domain, N=N, forcing_func=forcing_func)
    
    # Create evaluation grid of 200x200 points.
    points = eval_points(n=200, domain=domain)
    
    # Bind and apply the module.
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, points[:, 0], points[:, 1], mutable=["cache", "state"])
    u_computed, variables = model.apply(variables, points[:, 0], points[:, 1], mutable=["cache", "state"])
    print("Default Coefficients:")
    print(f"Computed solution shape: {u_computed.shape}")
    
    # Evaluate exact solution on the grid.
    u_expected = jnp.array([u_exact(x, y) for x, y in points])
    error = jnp.abs(u_computed.flatten() - u_expected)
    mse_error = jnp.mean(error ** 2)
    print(f"Max error: {mse_error:.4f}")
    assert mse_error < 1e-2, f"Maximum error too large: {mse_error}"

def gaussian_kappa(parameters, x, y):
    # parameters: [amplitude, center_x, center_y, sigma]
    amplitude, cx, cy, sigma = parameters
    return amplitude * jnp.exp(-(((x - cx) ** 2) + ((y - cy) ** 2)) / (2 * sigma ** 2))

def gaussian_eta(parameters, x, y):
    # parameters: [amplitude, center_x, center_y, sigma]
    amplitude, cx, cy, sigma = parameters
    return amplitude * jnp.exp(-(((x - cx) ** 2) + ((y - cy) ** 2)) / (2 * sigma ** 2))

def test_poisson_model_known_coeff():
    # Test with constant, non-unity coefficients and a forcing function
    # chosen so that u_exact = sin(pi*x)*sin(pi*y) is the exact solution.
    domain = (0.0, 1.0)
    N = 30  # Mesh resolution
    # Parameters are not used by our constant coefficients, but provided for compatibility.
    parameters = jnp.array([0.0])
    
    # Forcing function ensuring u_exact is the solution.
    # With u_exact = sin(pi*x)*sin(pi*y), Δ u_exact = -2π² sin(pi*x) sin(pi*y)
    # so -div(kappa grad u) = -2 Δ u_exact = 4π² sin(pi*x)*sin(pi*y)
    model = PoissonModel(
        parameters=parameters,
        domain=domain,
        N=N,
        forcing_func=forcing_known,
        kappa_func=constant_kappa,
        eta_func=constant_eta,  # Provided but not used in assembly.
    )
    
    # Create evaluation grid of 200x200 points.
    points = eval_points(n=200, domain=domain)
    
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, points[:, 0], points[:, 1], mutable=["cache", "state"])
    u_computed, variables = model.apply(variables, points[:, 0], points[:, 1], mutable=["cache", "state"])
    print("\nKnown Coefficients Test:")
    print(f"Computed solution shape: {u_computed.shape}")
    
    # Evaluate exact solution on the grid.
    u_expected = jnp.array([u_exact(x, y) for x, y in points])
    error = jnp.abs(u_computed.flatten() - u_expected)
    mse_error = jnp.mean(error**2)
    print(f"Max error: {mse_error:.4f}")
    assert mse_error < 1e-2, f"Maximum error too large: {mse_error}"

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
    
    # Instantiate the model with nonconstant coefficient functions.
    model = PoissonModel(
        parameters=parameters,
        domain=domain,
        N=N,
        forcing_func=forcing_nonc,
        kappa_func=nonconstant_kappa,
        eta_func=nonconstant_eta,  # Provided but not used in assembly.
    )
    
    # Create evaluation grid of 200x200 points.
    points = eval_points(n=200, domain=domain)
    
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, points[:, 0], points[:, 1], mutable=["cache", "state"])
    u_computed, variables = model.apply(variables, points[:, 0], points[:, 1], mutable=["cache", "state"])
    print("\nNonconstant Coefficients Test:")
    print(f"Computed solution shape: {u_computed.shape}")
    
    # Evaluate exact solution on the grid.
    u_expected = jnp.array([u_exact(x, y) for x, y in points])
    error = jnp.abs(u_computed.flatten() - u_expected)
    mse_error = jnp.mean(error**2)
    print(f"Max error: {mse_error:.4f}")
    assert mse_error < 1e-2, f"Maximum error too large: {mse_error}"


def performance_testing():
    domain = (0.0, 1.0)
    N = 100  # Mesh resolution for FEM assembly
    parameters = jnp.array([1.0])
    
    # Alternative forcing function for performance testing.
    def forcing_perf(x, y):
        return jnp.sin(pi * x) * jnp.cos(pi * y)
    
    # Instantiate the model.
    model = PoissonModel(parameters=parameters, domain=domain, N=N, forcing_func=forcing_perf)
    
    # Create a large evaluation grid (1000 x 1000 points)
    x_eval = jnp.linspace(domain[0], domain[1], 1000)
    y_eval = jnp.linspace(domain[0], domain[1], 1000)
    x_grid, y_grid = jnp.meshgrid(x_eval, y_eval)
    x_eval_flat = x_grid.flatten()
    y_eval_flat = y_grid.flatten()
    
    # Initialize the model.
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, x_eval_flat, y_eval_flat, mutable=["cache", "state"])
    
    num_executions = 100
    
    # Extract the FEM cache for direct PDE solving.
    fem = variables["cache"]["fem"]
    
    # Time the PDE solve.
    solve_time = timeit(lambda: model._solve_pde(fem), number=num_executions)
    
    # Time the complete evaluation (__call__) of the solution.
    eval_time = timeit(lambda: model.apply(variables, x_eval_flat, y_eval_flat, mutable=["cache", "state"]), number=num_executions)
    
    print("Performance Testing:")
    print(f"Average PDE solve time (with JIT): {solve_time / num_executions:.4f} seconds")
    print(f"Average evaluation (__call__) time (with caching): {eval_time / num_executions:.4f} seconds")

if __name__ == "__main__":
    test_poisson_model_defaults()
    test_poisson_model_known_coeff()
    test_poisson_model_nonconstant()
    performance_testing()
    print("All tests passed!")