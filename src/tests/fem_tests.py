import os
import sys
import time
import contextlib

def silent_run(func, *args, **kwargs):
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        return func(*args, **kwargs)

# Determine the directory of this file (tests folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assume the project root is the parent of the current directory, i.e., "src"
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import jax.numpy as jnp
from jax.scipy.linalg import solve
import matplotlib.pyplot as plt
import jax

from tools.finite_element_method import (
    generate_mesh, assemble_stiffness_matrix_2d, assemble_load_vector_2d,
    assemble_mass_matrix_2d, get_interior_indices
)

# Define the forcing function f(x,y) = 2*pi^2*sin(pi*x)*sin(pi*y)
def f_rhs(x, y):
    pi = 3.141592653589793
    return 2 * pi**2 * jnp.sin(pi * x) * jnp.sin(pi * y)

# Known analytical solution u_exact(x,y) = sin(pi*x)*sin(pi*y)
def u_exact(x, y):
    pi = 3.141592653589793
    return jnp.sin(pi * x) * jnp.sin(pi * y)

# Default coefficient functions
def default_kappa(x, y):
    return 1.0

def default_eta(x, y):
    return 0.0

def main(N):
    # Define domain as [0,1] (square domain [0,1]x[0,1])
    domain = (0, 1)

    start = time.perf_counter()
    nodes, elements, _, _ = generate_mesh(domain, N)
    jax.block_until_ready(nodes)
    mesh_time = time.perf_counter() - start
    print("Mesh generation time: {:.6f} sec".format(mesh_time))

    start = time.perf_counter()
    K = assemble_stiffness_matrix_2d(domain, N, default_kappa)
    jax.block_until_ready(K)
    stiffness_time = time.perf_counter() - start
    print("Stiffness matrix assembly time: {:.6f} sec".format(stiffness_time))

    start = time.perf_counter()
    F = assemble_load_vector_2d(domain, N, f_rhs)
    jax.block_until_ready(F)
    load_time = time.perf_counter() - start
    print("Load vector assembly time: {:.6f} sec".format(load_time))
    
    # Get interior indices for Dirichlet boundary conditions
    interior_indices = get_interior_indices(nodes, domain)
    
    # Extract the submatrix and subvector for interior nodes
    K_reduced = K[jnp.ix_(interior_indices, interior_indices)]
    F_reduced = F[interior_indices]
    jax.block_until_ready(K_reduced)
    jax.block_until_ready(F_reduced)

    start = time.perf_counter()
    u_interior = solve(K_reduced, F_reduced)
    jax.block_until_ready(u_interior)
    solve_time = time.perf_counter() - start
    print("Linear system solve time: {:.6f} sec".format(solve_time))

    # Reconstruct the full solution: boundary nodes remain zero
    u = jnp.zeros(nodes.shape[0], dtype=jnp.float32)
    u = u.at[interior_indices].set(u_interior)
    
    u_ex = jnp.array([u_exact(x, y) for (x, y) in nodes])
    error = jnp.abs(u - u_ex)
    max_error = jnp.max(error)
    print("Poisson test maximum error: {:.6f}".format(max_error))

def helmholtz_test(N):
    print("\nRunning Helmholtz test...")
    domain = (0, 1)
    
    start = time.perf_counter()
    nodes, elements, _, _ = generate_mesh(domain, N)
    jax.block_until_ready(nodes)
    mesh_time = time.perf_counter() - start
    print("Mesh generation time: {:.6f} sec".format(mesh_time))

    start = time.perf_counter()
    K = assemble_stiffness_matrix_2d(domain, N, default_kappa)
    jax.block_until_ready(K)
    stiffness_time = time.perf_counter() - start
    print("Stiffness matrix assembly time: {:.6f} sec".format(stiffness_time))

    start = time.perf_counter()
    # Use constant eta=1.0 for Helmholtz equation
    def unit_eta(x, y):
        return 1.0
    
    M = assemble_mass_matrix_2d(domain, N, unit_eta)
    jax.block_until_ready(M)
    mass_time = time.perf_counter() - start
    print("Mass matrix assembly time: {:.6f} sec".format(mass_time))
    
    L = K + M

    # Forcing function for Helmholtz: f = 2π² sin(πx) sin(πy) + sin(πx) sin(πy)
    def f_helmholtz(x, y):
        pi = 3.141592653589793
        return 2*pi**2 * jnp.sin(pi*x)*jnp.sin(pi*y) + jnp.sin(pi*x)*jnp.sin(pi*y)

    start = time.perf_counter()
    F = assemble_load_vector_2d(domain, N, f_helmholtz)
    jax.block_until_ready(F)
    load_time = time.perf_counter() - start
    print("Load vector assembly time: {:.6f} sec".format(load_time))
    
    start = time.perf_counter()
    # Use the helper for interior node indexing
    interior_indices = get_interior_indices(nodes, domain=domain)
    jax.block_until_ready(interior_indices)
    indexing_time = time.perf_counter() - start
    print("Interior nodes indexing time: {:.6f} sec".format(indexing_time))
    
    start = time.perf_counter()
    L_in = L[jnp.ix_(interior_indices, interior_indices)]
    F_in = F[interior_indices]
    jax.block_until_ready(F_in)
    submatrix_time = time.perf_counter() - start
    print("Submatrix extraction time: {:.6f} sec".format(submatrix_time))
    
    start = time.perf_counter()
    u_in = solve(L_in, F_in)
    jax.block_until_ready(u_in)
    linear_solve_time = time.perf_counter() - start
    print("Linear system solve time: {:.6f} sec".format(linear_solve_time))
    
    u = jnp.zeros(nodes.shape[0], dtype=jnp.float32)
    u = u.at[interior_indices].set(u_in)
    
    # u_exact for Helmholtz remains the same: sin(pi*x)*sin(pi*y)
    u_ex = jnp.array([jnp.sin(3.141592653589793*x)*jnp.sin(3.141592653589793*y)
                      for (x, y) in nodes])
    error = jnp.abs(u - u_ex)
    max_error = jnp.max(error)
    print("Helmholtz test maximum error: {:.6f}".format(max_error))

def inhomogeneous_helmholtz_test(N):
    print("\nRunning Inhomogeneous Helmholtz test...")
    domain = (0, 1)

    # Define kappa and eta coefficient functions
    def kappa_func(x, y):
        return 1.0 + x + y

    def eta_func(x, y):
        return 1.0 + 0.2*x + 0.2*y

    pi = 3.141592653589793
    def u_exact(x, y):
        return jnp.sin(pi*x)*jnp.sin(pi*y)

    # Compute f(x,y) = -∇·(kappa∇u) + eta*u by differentiating u_exact analytically
    def f(x, y):
        # For demonstration, we approximate the PDE manually:
        # Let k = kappa_func(x,y), e = eta_func(x,y), then PDE = -div(k grad(u)) + e*u
        k = kappa_func(x, y)
        e = eta_func(x, y)
        # analytic derivatives for -∇·(k grad u) + eta u:
        dudx = pi * jnp.cos(pi*x)*jnp.sin(pi*y)
        dudy = pi * jnp.sin(pi*x)*jnp.cos(pi*y)
        d2udx2 = -pi**2 * jnp.sin(pi*x)*jnp.sin(pi*y)
        d2udy2 = -pi**2 * jnp.sin(pi*x)*jnp.sin(pi*y)
        # approximate chain rule for k * grad(u):
        dkdx = 1.0
        dkdy = 1.0
        term_x = dkdx * dudx + k * d2udx2
        term_y = dkdy * dudy + k * d2udy2
        diffusion_term = -(term_x + term_y)
        return diffusion_term + e * u_exact(x, y)
    
    start = time.perf_counter()
    nodes, elements, _, _ = generate_mesh(domain, N)
    jax.block_until_ready(nodes)
    mesh_time = time.perf_counter() - start
    print("Mesh generation time: {:.6f} sec".format(mesh_time))

    start = time.perf_counter()
    K = assemble_stiffness_matrix_2d(domain, N, kappa_func)
    jax.block_until_ready(K)
    stiffness_time = time.perf_counter() - start
    print("Stiffness matrix assembly time: {:.6f} sec".format(stiffness_time))

    start = time.perf_counter()
    M = assemble_mass_matrix_2d(domain, N, eta_func)
    jax.block_until_ready(M)
    mass_time = time.perf_counter() - start
    print("Mass matrix assembly time: {:.6f} sec".format(mass_time))
    
    L = K + M

    start = time.perf_counter()
    F = assemble_load_vector_2d(domain, N, f)
    jax.block_until_ready(F)
    load_time = time.perf_counter() - start
    print("Load vector assembly time: {:.6f} sec".format(load_time))
    
    start = time.perf_counter()
    interior_indices = get_interior_indices(nodes, domain=domain)
    jax.block_until_ready(interior_indices)
    indexing_time = time.perf_counter() - start
    print("Interior nodes indexing time: {:.6f} sec".format(indexing_time))
    
    start = time.perf_counter()
    L_in = L[jnp.ix_(interior_indices, interior_indices)]
    F_in = F[interior_indices]
    jax.block_until_ready(F_in)
    submatrix_time = time.perf_counter() - start
    print("Submatrix extraction time: {:.6f} sec".format(submatrix_time))
    
    start = time.perf_counter()
    u_in = solve(L_in, F_in)
    jax.block_until_ready(u_in)
    linear_solve_time = time.perf_counter() - start
    print("Linear system solve time: {:.6f} sec".format(linear_solve_time))
    
    u = jnp.zeros(nodes.shape[0], dtype=jnp.float32)
    u = u.at[interior_indices].set(u_in)
    
    # Compare with exact solution
    u_ex = jnp.array([u_exact(x, y) for (x, y) in nodes])
    error = jnp.abs(u - u_ex)
    max_error = jnp.max(error)
    print("Inhomogeneous Helmholtz test maximum error: {:.6f}".format(max_error))

    # Plot the solution
    plt.figure(figsize=(8, 6))
    plt.tripcolor(nodes[:, 0], nodes[:, 1], elements, u, shading='gouraud')
    plt.title("Inhomogeneous Helmholtz solution")
    plt.colorbar()
    plt.savefig('inhomogeneous_helmholtz.png')
    plt.close()
    print("Solution plot saved as 'inhomogeneous_helmholtz.png'")

def benchmark_fem(N_values=None):
    """Run benchmarks with different mesh sizes"""
    if N_values is None:
        N_values = [20, 40, 80]
    
    print("\nRunning FEM benchmarks...")
    results = []
    
    for N in N_values:
        print(f"\nBenchmark with N={N}:")
        domain = (0, 1)
        
        # Mesh generation
        start = time.perf_counter()
        nodes, elements, _, _ = generate_mesh(domain, N)
        jax.block_until_ready(nodes)
        mesh_time = time.perf_counter() - start
        
        # Stiffness assembly
        start = time.perf_counter()
        K = assemble_stiffness_matrix_2d(domain, N, default_kappa)
        jax.block_until_ready(K)
        stiffness_time = time.perf_counter() - start
        
        # Load assembly
        start = time.perf_counter()
        F = assemble_load_vector_2d(domain, N, f_rhs)
        jax.block_until_ready(F)
        load_time = time.perf_counter() - start
        
        # Interior indices
        start = time.perf_counter()
        interior_indices = get_interior_indices(nodes, domain)
        jax.block_until_ready(interior_indices)
        indexing_time = time.perf_counter() - start
        
        # System reduction
        start = time.perf_counter()
        K_in = K[jnp.ix_(interior_indices, interior_indices)]
        F_in = F[interior_indices]
        jax.block_until_ready(K_in)
        jax.block_until_ready(F_in)
        reduction_time = time.perf_counter() - start
        
        # Solve
        start = time.perf_counter()
        u_in = solve(K_in, F_in)
        jax.block_until_ready(u_in)
        solve_time = time.perf_counter() - start
        
        # Error calculation
        u = jnp.zeros(nodes.shape[0], dtype=jnp.float32)
        u = u.at[interior_indices].set(u_in)
        u_ex = jnp.array([u_exact(x, y) for (x, y) in nodes])
        max_error = jnp.max(jnp.abs(u - u_ex))
        
        total_time = mesh_time + stiffness_time + load_time + indexing_time + reduction_time + solve_time
        
        results.append({
            'N': N,
            'nodes': nodes.shape[0],
            'elements': elements.shape[0],
            'mesh_time': mesh_time,
            'stiffness_time': stiffness_time,
            'load_time': load_time,
            'solve_time': solve_time,
            'total_time': total_time,
            'max_error': max_error
        })
        
        print(f"  Nodes: {nodes.shape[0]}, Elements: {elements.shape[0]}")
        print(f"  Mesh time: {mesh_time:.6f} sec")
        print(f"  Stiffness assembly: {stiffness_time:.6f} sec")
        print(f"  Load assembly: {load_time:.6f} sec")
        print(f"  Solve time: {solve_time:.6f} sec")
        print(f"  Total time: {total_time:.6f} sec")
        print(f"  Max error: {max_error:.6f}")
    
    return results

if __name__ == "__main__":
    N = 50
    if len(sys.argv) > 1:
        if sys.argv[1] == "benchmark":
            benchmark_fem()
        elif sys.argv[1].isdigit():
            N = int(sys.argv[1])
    
    silent_run(main, N)  # Warm up JIT
    main(N)
    
    if input("\nRun additional tests? (y/n): ").lower() == 'y':
        helmholtz_test(N)
        inhomogeneous_helmholtz_test(N)