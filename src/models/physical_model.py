import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy as jsp

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.finite_element_method import *


class PhysicalModel(nn.Module):
    domain: tuple
    N: int
    parameters: jnp.ndarray
    training: bool

    def setup(self):
        # This abstract base class does not implement setup.
        pass

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Subclasses should implement __call__")


class PoissonModel(PhysicalModel):
    forcing_func: callable
    # Allow user to pass functions that compute the coefficients from parameters, x, and y.
    kappa_func: callable = None
    eta_func: callable = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        # Retrieve (or create) the mesh cache.
        mesh_cache = self.variable("cache", "mesh", lambda: self._create_mesh())
        # Retrieve (or create) the FEM cache.
        fem_cache = self.variable("cache", "fem", lambda: self._assemble_fem(mesh_cache.value))
        # Retrieve (or create) the cached solution.
        sol_cache = self.variable("cache", "solution", lambda: self._solve_pde(mesh_cache.value, fem_cache.value))
        
        # Training flag is used to determine whether to update the cache.
        # Old parameters are stored in the cache.
        old_parameters = self.variable("cache", "parameters", lambda: self.parameters)
        #force the fem_cache to be updated
        do_update = self.training | jnp.all(old_parameters.value != self.parameters)

        def recompute_fn(_):
            new_fem = self._assemble_fem(mesh_cache.value)
            new_sol = self._solve_pde(mesh_cache.value, new_fem)
            return new_fem, new_sol

        # Use a dummy operand (e.g. 0) instead of None.
        fem_cache.value, sol_cache.value = jax.lax.cond(
            do_update,
            recompute_fn,
            lambda _: (fem_cache.value, sol_cache.value),
            operand=None,
        )

        # Evaluate the cached solution at the given points.
        return self._evaluate_solution(mesh_cache.value, sol_cache.value, x, y)
    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False
    
    def _create_mesh(self) -> dict:
        """
        Create and return a dictionary with the mesh nodes and grid.
        """
        nodes, _, grid_x, grid_y = generate_mesh(self.domain, self.N)
        interior_indices = get_interior_indices(nodes, domain=self.domain)
        return {
            "nodes": nodes,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "interior_indices": interior_indices,
        }


    def _assemble_fem(self, mesh) -> dict:
        """
        Assemble and return a dictionary with the mesh and FEM matrices.
        The stiffness matrix is built with the (possibly dynamic) coefficient kappa.
        """
        interior_indices = mesh["interior_indices"]
        # Pass the coefficient function for kappa into the stiffness matrix assembly.
        K = assemble_stiffness_matrix_2d(self.domain, self.N, kappa=self.kappa)
        F = assemble_load_vector_2d(self.domain, self.N, self.forcing_func)
        K_in = K[jnp.ix_(interior_indices, interior_indices)]
        F_in = F[interior_indices]
        return {
            "K_in": K_in,
            "F_in": F_in,
        }

    def _solve_pde(self, mesh: dict, fem: dict) -> jnp.ndarray:
        """
        Solve the PDE using the cached FEM matrices.
        """
        nodes = mesh["nodes"]
        interior_indices = mesh["interior_indices"]
        K_in = fem["K_in"]
        F_in = fem["F_in"]

        @jax.jit
        def solve_inner(K_in, F_in, nodes, interior_indices):
            u_in = jsp.sparse.linalg.cg(K_in, F_in)[0]
            u = jnp.zeros(nodes.shape[0], dtype=jnp.float32)
            u = u.at[interior_indices].set(u_in)
            return u

        return solve_inner(K_in, F_in, nodes, interior_indices)

    def _evaluate_solution(self, mesh: dict, u: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the solution at the provided evaluation points.
        """
        grid_x = mesh["grid_x"]
        grid_y = mesh["grid_y"]
        grid_values = u.reshape((self.N + 1, self.N + 1))
        interpolator = jsp.interpolate.RegularGridInterpolator(
            (grid_x, grid_y),
            grid_values,
            bounds_error=False,
            fill_value=None,
        )
        pts = jnp.stack([x, y], axis=-1)
        return interpolator(pts)

    def kappa(self, x, y):
        """
        Compute the coefficient kappa at position (x,y) based on model parameters.
        If no function is provided, default to 1.0.
        """
        if self.kappa_func is not None:
            return self.kappa_func(self.parameters, x, y)
        return 1.0

    def eta(self, x, y):
        """
        Compute the coefficient eta at position (x,y) based on model parameters.
        If no function is provided, default to 0.0.
        For now, eta is not used in the model since it is the Poisson equation.
        """
        if self.eta_func is not None:
            return self.eta_func(self.parameters, x, y)
        return 0.0
    
class HelmholtzModel(PhysicalModel):
    forcing_func: callable
    # Allow user to pass functions that compute the coefficients from parameters, x, and y.
    kappa_func: callable = None
    eta_func: callable = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        # Retrieve (or create) the mesh cache.
        mesh_cache = self.variable("cache", "mesh", lambda: self._create_mesh())
        # Retrieve (or create) the FEM cache.
        fem_cache = self.variable("cache", "fem", lambda: self._assemble_fem(mesh_cache.value))
        # Retrieve (or create) the cached solution.
        sol_cache = self.variable("cache", "solution", lambda: self._solve_pde(mesh_cache.value, fem_cache.value))
        
        # Training flag is used to determine whether to update the cache.
        # Old parameters are stored in the cache.
        old_parameters = self.variable("cache", "parameters", lambda: self.parameters)
        #force the fem_cache to be updated
        do_update = self.training | jnp.all(old_parameters.value != self.parameters)

        def recompute_fn(_):
            new_fem = self._assemble_fem(mesh_cache.value)
            new_sol = self._solve_pde(mesh_cache.value, new_fem)
            return new_fem, new_sol

        # Use a dummy operand (e.g. 0) instead of None.
        fem_cache.value, sol_cache.value = jax.lax.cond(
            do_update,
            recompute_fn,
            lambda _: (fem_cache.value, sol_cache.value),
            operand=None,
        )

        # Evaluate the cached solution at the given points.
        return self._evaluate_solution(mesh_cache.value, sol_cache.value, x, y)
    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False
    
    def _create_mesh(self) -> dict:
        """
        Create and return a dictionary with the mesh nodes and grid.
        """
        nodes, _, grid_x, grid_y = generate_mesh(self.domain, self.N)
        interior_indices = get_interior_indices(nodes, domain=self.domain)
        return {
            "nodes": nodes,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "interior_indices": interior_indices,
        }
    
    def _assemble_fem(self, mesh) -> dict:
        """
        Assemble and return a dictionary with the mesh and FEM matrices.
        The stiffness matrix is built with the (possibly dynamic) coefficient kappa.
        """
        interior_indices = mesh["interior_indices"]
        # Pass the coefficient function for kappa into the stiffness matrix assembly.
        K = assemble_stiffness_matrix_2d(self.domain, self.N, kappa=self.kappa)
        M = assemble_mass_matrix_2d(self.domain, self.N, eta=self.eta)
        F = assemble_load_vector_2d(self.domain, self.N, self.forcing_func)
        K_in = K[jnp.ix_(interior_indices, interior_indices)]
        M_in = M[jnp.ix_(interior_indices, interior_indices)]
        F_in = F[interior_indices]
        return {
            "K_in": K_in,
            "M_in": M_in,
            "F_in": F_in,
        }
    
    def _solve_pde(self, mesh: dict, fem: dict) -> jnp.ndarray:
        """
        Solve the PDE using the cached FEM matrices.
        """
        nodes = mesh["nodes"]
        interior_indices = mesh["interior_indices"]
        K_in = fem["K_in"]
        M_in = fem["M_in"]
        F_in = fem["F_in"]

        @jax.jit
        def solve_inner(K_in, M_in, F_in, nodes, interior_indices):
            u_in = jsp.sparse.linalg.cg(K_in + M_in, F_in)[0]
            u = jnp.zeros(nodes.shape[0], dtype=jnp.float32)
            u = u.at[interior_indices].set(u_in)
            return u

        return solve_inner(K_in, M_in, F_in, nodes, interior_indices)
    
    def _evaluate_solution(self, mesh: dict, u: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the solution at the provided evaluation points.
        """
        grid_x = mesh["grid_x"]
        grid_y = mesh["grid_y"]
        grid_values = u.reshape((self.N + 1, self.N + 1))
        interpolator = jsp.interpolate.RegularGridInterpolator(
            (grid_x, grid_y),
            grid_values,
            bounds_error=False,
            fill_value=None,
        )
        pts = jnp.stack([x, y], axis=-1)
        return interpolator(pts)
    
    def kappa(self, x, y):
        """
        Compute the coefficient kappa at position (x,y) based on model parameters.
        If no function is provided, default to 1.0.
        """
        if self.kappa_func is not None:
            return self.kappa_func(self.parameters, x, y)
        return 1.0

    def eta(self, x, y):
        """
        Compute the coefficient eta at position (x,y) based on model parameters.
        If no function is provided, default to 0.0.
        For now, eta is not used in the model since it is the Poisson equation.
        """
        if self.eta_func is not None:
            return self.eta_func(self.parameters, x, y)
        return 0.0
        
