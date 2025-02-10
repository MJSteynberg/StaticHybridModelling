import os
import sys
import time

import os
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

from tools.finite_element_method import (generate_mesh, assemble_stiffness_matrix_2d, 
                                          assemble_load_vector_2d, assemble_mass_matrix_2d, 
                                          get_interior_indices, reduce_system)

# Define the forcing function f(x,y) = 2*pi^2*sin(pi*x)*sin(pi*y)
def f_rhs(x, y):
    pi = 3.141592653589793
    return 2 * pi**2 * jnp.sin(pi * x) * jnp.sin(pi * y)

# Known analytical solution u_exact(x,y) = sin(pi*x)*sin(pi*y)
def u_exact(x, y):
    pi = 3.141592653589793
    return jnp.sin(pi * x) * jnp.sin(pi * y)

def main(N):
    # Define domain as [0,1] (square domain [0,1]x[0,1])
    domain = (0,1)

    start = time.perf_counter()
    nodes, elements, _, _ = generate_mesh(domain, N)
    jax.block_until_ready(nodes)
    mesh_time = time.perf_counter() - start
    print("Mesh generation time: {:.6f} sec".format(mesh_time))

    start = time.perf_counter()
    K = assemble_stiffness_matrix_2d(domain, N)
    jax.block_until_ready(K)
    stiffness_time = time.perf_counter() - start
    print("Stiffness matrix assembly time: {:.6f} sec".format(stiffness_time))

    start = time.perf_counter()
    F = assemble_load_vector_2d(domain, N, f_rhs)
    jax.block_until_ready(F)
    load_time = time.perf_counter() - start
    print("Load vector assembly time: {:.6f} sec".format(load_time))
    
    # Apply reduction to enforce Dirichlet boundary conditions.
    K_reduced, F_reduced, interior_indices = reduce_system(K, F, nodes, domain)
    jax.block_until_ready(K_reduced)
    jax.block_until_ready(F_reduced)

    u_interior = solve(K_reduced, F_reduced)
    jax.block_until_ready(u_interior)

    # Reconstruct the full solution: boundary nodes remain zero.
    u  = jnp.zeros(nodes.shape[0], dtype=jnp.float32)
    u = u.at[interior_indices].set(u_interior)


    
    u_ex = jnp.array([u_exact(x, y) for (x, y) in nodes])
    error = jnp.abs(u - u_ex)
    max_error = jnp.max(error)
    print("Poisson test maximum error: {:.6f}".format(max_error))

def helmholtz_test(N):
    print("\nRunning Helmholtz test...")
    domain = (0,1)
    
    start = time.perf_counter()
    nodes, elements, _, _ = generate_mesh(domain, N)
    jax.block_until_ready(nodes)
    mesh_time = time.perf_counter() - start
    print("Mesh generation time: {:.6f} sec".format(mesh_time))

    start = time.perf_counter()
    K = assemble_stiffness_matrix_2d(domain, N)
    jax.block_until_ready(K)
    stiffness_time = time.perf_counter() - start
    print("Stiffness matrix assembly time: {:.6f} sec".format(stiffness_time))

    start = time.perf_counter()
    M = assemble_mass_matrix_2d(domain, N)
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
    # Use the new helper for interior node indexing.
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
    domain = (0,1)

    # Define kappa, eta, and exact solution:
    def kappa_func(x, y):
        return 1.0 + x + y

    def eta_func(x, y):
        return 1.0 + 0.2*x + 0.2*y

    pi = 3.141592653589793
    def u_exact(x, y):
        return jnp.sin(pi*x)*jnp.sin(pi*y)

    # Compute f(x,y) = -∇·(kappa∇u) + eta*u by differentiating u_exact analytically:
    # Here is a small helper for partial derivatives:
    def f(x, y):
        # For demonstration, we approximate the PDE manually:
        # Let k = kappa_func(x,y), e = eta_func(x,y), then PDE = -div(k grad(u)) + e*u
        # grad(u) = [pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y)]
        # dot(k grad(u)) etc. (Implementation depends on your symbolic approach.)
        # For brevity, just keep it simple or do a direct symbolic calculation externally.
        k = kappa_func(x, y)
        e = eta_func(x, y)
        # analytic second derivatives for -∇·(k grad u) + eta u:
        dudx = pi * jnp.cos(pi*x)*jnp.sin(pi*y)
        dudy = pi * jnp.sin(pi*x)*jnp.cos(pi*y)
        d2udx2 = -pi**2 * jnp.sin(pi*x)*jnp.sin(pi*y)
        d2udy2 = -pi**2 * jnp.sin(pi*x)*jnp.sin(pi*y)
        # approximate chain rule for k * grad(u):
        # div(k*grad(u)) ~ (dk/dx * dudx + k*d2udx2) + (dk/dy * dudy + k*d2udy2)
        dkdx = 1.0
        dkdy = 1.0
        term_x = dkdx * dudx + k * d2udx2
        term_y = dkdy * dudy + k * d2udy2
        diffusion_term = term_x + term_y
        return -diffusion_term + e * u_exact(x, y)
    
    start = time.perf_counter()
    nodes, elements, _, _ = generate_mesh(domain, N)
    jax.block_until_ready(nodes)
    mesh_time = time.perf_counter() - start
    print("Mesh generation time: {:.6f} sec".format(mesh_time))

    start = time.perf_counter()
    K = assemble_stiffness_matrix_2d(domain, N)
    jax.block_until_ready(K)
    stiffness_time = time.perf_counter() - start
    print("Stiffness matrix assembly time: {:.6f} sec".format(stiffness_time))

    start = time.perf_counter()
    M = assemble_mass_matrix_2d(domain, N)
    jax.block_until_ready(M)
    mass_time = time.perf_counter() - start
    print("Mass matrix assembly time: {:.6f} sec".format(mass_time))
    
    L = K + M

    # Forcing function for Helmholtz: f = 2π² sin(πx) sin(πy) + sin(πx) sin(πy)
    def f_helmholtz(x, y):
        pi = 3.141592653589793
        # gaussian
        return f(x,y)

    start = time.perf_counter()
    F = assemble_load_vector_2d(domain, N, f_helmholtz)
    jax.block_until_ready(F)
    load_time = time.perf_counter() - start
    print("Load vector assembly time: {:.6f} sec".format(load_time))
    
    start = time.perf_counter()
    # Use the new helper for interior node indexing.
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

    # plot the solution
    plt.figure(figsize=(8, 6))
    plt.tripcolor(nodes[:, 0], nodes[:, 1], elements, u, shading='gouraud')
    plt.title("Inhomogeneous Helmholtz solution")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    N = 50
    silent_run(main, N)
    main(N)
    # helmholtz_test(N)
    # inhomogeneous_helmholtz_test(N)