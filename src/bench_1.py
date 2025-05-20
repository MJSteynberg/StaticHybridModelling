import os
import sys
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from functools import partial
from flax import nnx
import optimistix
import numpy as np
import orbax.checkpoint as ocp
import jax.scipy as jsp

# Set working directory to the src folder.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
os.chdir(src_dir)

from models.physical_model import PhysicalModel  
from models.synthetic_model import FeedForwardNet
from models.other_models import PINN
from tools.plotting import *  

pi = jnp.pi

# Define coefficient functions as Gaussians.
nx, ny = 21, 21
x_nodes = np.linspace(0,1,nx)
y_nodes = np.linspace(0,1,ny)

def kappa(parameters, x, y):
    """
    parameters: array of length nx*ny, row‐major over x_nodes×y_nodes
    x,y: scalar or array
    returns κ(x,y) by linear interp over the regular grid
    """
    # reshape to (nx,ny)
    grid = parameters.reshape((nx,ny))
    # build interpolator
    interp = jsp.interpolate.RegularGridInterpolator(
        (x_nodes, y_nodes),
        grid,
        method='linear',
        bounds_error=False,
        fill_value=None
    )
    # prepare query points
    pts = jnp.stack([jnp.ravel(x), jnp.ravel(y)], axis=-1)
    vals = interp(pts)                # shape (n_pts,)
    # Set the boundary vallues to 1 / (1 + x**2 + y**2 + (x - 1)**2 + (y - 1)**2)
    mask = jnp.isclose(x, 0) | jnp.isclose(x, 1) | jnp.isclose(y, 0) | jnp.isclose(y, 1)
    vals = jnp.where(mask, 1 / (1 + x**2 + y**2 + (x - 1)**2 + (y - 1)**2), vals)
    return vals.reshape(jnp.shape(x))

def kappa_true(parameters, x, y): 
    return 1 / (1 + x**2 + y**2 + (x - 1)**2 + (y - 1)**2)

eta = None

def f(x, y):
    numerator_1 = 2 * pi**2 * jnp.sin(pi*x) * jnp.sin(pi*y)
    numerator_2 = 2 * pi * ((2 * x - 1) * jnp.cos(pi * x) * jnp.sin(pi * y) + (2 * y - 1) * jnp.sin(pi * x) * jnp.cos(pi * y))
    denominator = 1 + x**2 + y**2 + (x - 1)**2 + (y - 1)**2
    return numerator_1 / denominator + numerator_2 / denominator**2

def u_true(x, y):
    return jnp.sin(pi * x)*jnp.sin(pi * y)
# -----------------------------------------------------------------------------  
# Generate the “true solution” using a high resolution physical model.
x_grid, y_grid = jnp.meshgrid(x_nodes, y_nodes)
true_params = kappa_true(jnp.ones((nx*ny,)), x_grid, y_grid).flatten()

L = 1
domain = (0, 1)



subdomain=[(0,1),(0,1)]
n_train=50
xx_train = jnp.linspace(subdomain[0][0], subdomain[0][1], n_train)
yy_train = jnp.linspace(subdomain[1][0], subdomain[1][1], n_train)
xx_train, yy_train = jnp.meshgrid(xx_train, yy_train)
xx_train = xx_train.flatten()
yy_train = yy_train.flatten()
pts_train = jnp.stack([xx_train, yy_train], axis=-1)

xx_eval = jnp.linspace(subdomain[0][0], subdomain[0][1], 50)
yy_eval = jnp.linspace(subdomain[1][0], subdomain[1][1], 50)
xx_eval, yy_eval = jnp.meshgrid(xx_eval, yy_eval)
xx_eval = xx_eval.flatten()
yy_eval = yy_eval.flatten()
pts_eval = jnp.stack([xx_eval, yy_eval], axis=-1)
# Use vmap over the new scalar __call__ for prediction.
u_train = u_true(xx_train, yy_train).reshape(-1, 1)
print(f"Training data generated with shape: {u_train.shape}")

def train_hybrid(epochs):
    # -------------------------------------------------------------------------
    # Initialize the synthetic model.
    synthetic_model = FeedForwardNet(
        hidden_dims=(128,128),
        activation=nnx.relu,
        output_dim=1,
        rngs=nnx.Rngs(0)
    )

    # -------------------------------------------------------------------------
    # Setup physical model.
    low_res_N = 20
    

    init_params = nnx.Param(jax.random.uniform(jax.random.PRNGKey(0), shape=(nx**2,)) + 1)
    print(f"Initial parameters: {init_params}")
    physical_model = PhysicalModel(
        domain=domain,
        N=low_res_N,
        parameters=init_params,
        training=True,
        forcing_func=f,
        kappa_func=kappa,
        eta_func=eta,
        rngs=nnx.Rngs(0)
    )
    nnx.display(physical_model)
    # -------------------------------------------------------------------------
    # Initialize the optimizers.
    syn_opt = nnx.Optimizer(synthetic_model, optax.adam(5e-4))
    phys_opt = nnx.Optimizer(physical_model, optax.adam(5e-4))

    # Helper to vmap a scalar-call model.
    @nnx.jit
    def vmapped_model(m, xs, ys):
        return jax.vmap(lambda xx, yy: m(xx, yy))(xs, ys)

    @nnx.jit
    def train_step_hyb(model, model_other, optimizer, x, y, u, x_collocation, y_collocation):
        def loss_data(m):
            u_pred = vmapped_model(m, x, y)
            return jnp.mean(optax.squared_error(u_pred, u))
        
        def loss_hyb(m):
            u_pred = vmapped_model(m, x_collocation, y_collocation)
            u_pred_other = vmapped_model(model_other, x_collocation, y_collocation)
            return jnp.mean(optax.squared_error(u_pred, u_pred_other))
        
        def loss_fn(m):
            return  loss_data(m) + loss_hyb(m)

        dloss = loss_data(model)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)

        return loss, dloss
    
    @nnx.jit
    def train_step(model, optimizer, x, y, u):
        def loss_data(m):
            u_pred = vmapped_model(m, x, y)
            return jnp.mean(optax.squared_error(u_pred, u))
        
        def loss_fn(m):
            return loss_data(m)
        
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    # -------------------------------------------------------------------------
    # Train the models.
    u_real = u_true(xx_eval, yy_eval).reshape(-1, 1)
    loss_history_phys = np.zeros(epochs)
    loss_history_syn = np.zeros(epochs)
    param_history = np.zeros((epochs, nx**2))
    rng = jax.random.PRNGKey(6)
    n_collocation = 200
    loss_syn_data = 1
    for epoch in range(epochs):
        if loss_syn_data > 1e-1:
            loss_syn_data = train_step(synthetic_model, syn_opt, xx_train, yy_train, u_train)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss (synthetic): {loss_syn_data}")

        else:
            rng, rng1, rng2 = jax.random.split(rng, 3)
            x_collocation = jax.random.uniform(rng1, shape=(n_collocation,),
                                            minval=domain[0], maxval=domain[1])
            y_collocation = jax.random.uniform(rng2, shape=(n_collocation,),
                                            minval=domain[0], maxval=domain[1])

            
            loss_syn, loss_syn_data = train_step_hyb(synthetic_model, physical_model, syn_opt,
                                    xx_train, yy_train, u_train,
                                    x_collocation, y_collocation)
            
            loss_phy, loss_phy_data = train_step_hyb(physical_model, synthetic_model, phys_opt,
                                xx_train, yy_train, u_train,
                                x_collocation, y_collocation)
            
            # Evaluation
            
            u_pred_syn = vmapped_model(synthetic_model, xx_eval, yy_eval).reshape(-1, 1)
            u_pred_phys = vmapped_model(physical_model, xx_eval, yy_eval).reshape(-1, 1)
            mse_syn = jnp.mean(optax.squared_error(u_pred_syn, u_real))
            mse_phys = jnp.mean(optax.squared_error(u_pred_phys, u_real))
            loss_history_syn[epoch] = mse_syn
            loss_history_phys[epoch] = mse_phys
            param_history[epoch] = physical_model.parameters.value
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss (physical): {loss_phy_data}, Loss (synthetic): {loss_syn}, Parameters: {physical_model.parameters.value}")

    return loss_history_phys, loss_history_syn, synthetic_model, param_history, u_pred_phys, u_pred_syn


if __name__ == "__main__":
    epochs = 3000
    checkpointer = ocp.StandardCheckpointer()
    if input("Train the hybrid model? (y/n): ") == "y":
        loss_history_phys, loss_history_syn, syn_model, param_history, u_pred_phys, u_pred_syn = train_hybrid(epochs)

        # convert them to njp arrays


        # save the model
        _, state = nnx.split(syn_model)
        
        checkpoint_path = os.path.abspath("src/files/bench_1/checkpoint")
        ckpt_dir = ocp.test_utils.erase_and_create_empty(checkpoint_path)
        checkpointer.save(ckpt_dir / 'state', state)
    else:
        model = FeedForwardNet(
            hidden_dims=(128, 128),
            activation=nnx.relu,
            output_dim=1,
            rngs=nnx.Rngs(0)
        )
        
        graphdef, state = nnx.split(model)
        checkpoint_path = os.path.abspath("src/files/bench_1/checkpoint")
        state_restored = checkpointer.restore(checkpoint_path + '/state', state)
        syn_model = nnx.merge(graphdef, state_restored)

    def vmapped_model(m, xs, ys):
        return jax.vmap(lambda xx, yy: m(xx, yy))(xs, ys)
    
    phys_model = PhysicalModel(
        domain=domain,
        N=100,
        parameters=nnx.Param(true_params),
        training=False,
        forcing_func=f,
        kappa_func=kappa,
        eta_func=eta,
        rngs=nnx.Rngs(0)
    )
    # Plot the prediction of the synthetic model, physical model and true solution
    # Plot the prediction of the synthetic model, physical model and true solution
    n_plot = 300
    x_plot = jnp.linspace(-L/2, L/2, n_plot)
    y_plot = jnp.linspace(-L/2, L/2, n_plot)
    xx_plot, yy_plot = jnp.meshgrid(x_plot, y_plot)
    xx_plot = xx_plot.flatten()
    yy_plot = yy_plot.flatten()
    u_true_plot = u_true(xx_plot, yy_plot)
    u_syn_plot = vmapped_model(syn_model, xx_plot, yy_plot)
    u_phy_plot = vmapped_model(phys_model, xx_plot, yy_plot)
    u_true_plot = u_true_plot.reshape(n_plot, n_plot)
    u_syn_plot = u_syn_plot.reshape(n_plot, n_plot)
    u_phy_plot = u_phy_plot.reshape(n_plot, n_plot)
    
    fig = plt.figure(figsize=(13, 2.5))
    gs = fig.add_gridspec(
        1, 8, 
        width_ratios=[1, 1, 1, 0.2, 0.2, 1, 1, 0.2],
        left=0.05, right=0.93, top=0.86, bottom=0.11,
        wspace=0.1, hspace=0.1
    )

    # Top row: Solution plots.
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax_top_cb = fig.add_subplot(gs[0, 3])

    # Bottom row: Error plots.
    ax3 = fig.add_subplot(gs[0, 5])
    ax4 = fig.add_subplot(gs[0, 6])

    ax_bot_cb = fig.add_subplot(gs[0, 7])

    # Remove ticks and enforce square aspect ratio on all axes.
    for axi in [ax0, ax1, ax2, ax3, ax4]:
        axi.set_xticks([])
        axi.set_yticks([])
        axi.set_aspect("equal", adjustable="box")

    # Top row: Plot the solutions with consistent color scale
    vmin_sol = min(u_true_plot.min(), u_syn_plot.min(), u_phy_plot.min())
    vmax_sol = max(u_true_plot.max(), u_syn_plot.max(), u_phy_plot.max())
    
    cs0 = ax0.contourf(x_plot, y_plot, u_true_plot, cmap="viridis", levels=100, vmin=vmin_sol, vmax=vmax_sol)
    ax0.set_title("True")
    
    cs1 = ax1.contourf(x_plot, y_plot, u_phy_plot, cmap="viridis", levels=100, vmin=vmin_sol, vmax=vmax_sol)
    ax1.set_title("Physical")

    cs2 = ax2.contourf(x_plot, y_plot, u_syn_plot, cmap="viridis", levels=100, vmin=vmin_sol, vmax=vmax_sol)
    ax2.set_title("Synthetic")

    # Add a colorbar for the top row.
    cbar_top = fig.colorbar(cs0, cax=ax_top_cb, orientation="vertical", pad=0.02, ticks = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75])

    # Bottom row: Plot the error contours with consistent color scale
    vmax_err = max(jnp.abs(u_syn_plot - u_true_plot).max(), jnp.abs(u_phy_plot - u_true_plot).max())
    
    err1 = ax3.contourf(x_plot, y_plot, jnp.abs(u_phy_plot - u_true_plot), cmap="viridis", levels=100, vmin=0, vmax=vmax_err)
    ax3.set_title("Physical Error")
    err2 = ax4.contourf(x_plot, y_plot, jnp.abs(u_syn_plot - u_true_plot), cmap="viridis", levels=100, vmin=0, vmax=vmax_err)
    ax4.set_title("Synthetic Error")
    # Add a colorbar for the bottom row.
    cbar_bot = fig.colorbar(err2, cax=ax_bot_cb, orientation="vertical", pad=0.02, ticks = [0, 0.02, 0.04, 0.06, 0.08, 0.10])

    # Add overall row labels on the left.

    plt.savefig("src/results/bench_1/prediction_comparison.png")

    
    


    