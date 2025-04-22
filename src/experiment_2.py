import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from functools import partial
from flax import nnx
import orbax.checkpoint as ocp
import optimistix

# Set working directory to the src folder.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
os.chdir(src_dir)

from models.physical_model import PhysicalModel  
from models.synthetic_model import FeedForwardNet, ResNetSynthetic
from models.other_models import PINN
from tools.plotting import *  

pi = jnp.pi

# Define coefficient functions as Gaussians.
def kappa(parameters, x, y): 
    amplitude, cx, cy = parameters[0:3]
    return amplitude * jnp.exp(-(((x - cx)**2 + (y - cy)**2))) + 1

def eta(parameters, x, y):
    amplitude, cx, cy = parameters[3:6]
    return (amplitude * jnp.exp(-(((x - cx)**2 + (y - cy)**2))) + 1)**2

def f_full(parameters, x, y, L):
    A, ax, ay, B, bx, by = parameters
    return -2*pi*A*(2*ax - 2*x)*jnp.exp(-(-ax + x)**2 - (-ay + y)**2)*jnp.sin(2*pi*y/L)*jnp.cos(2*pi*x/L)/L - 2*pi*A*(2*ay - 2*y)*jnp.exp(-(-ax + x)**2 - (-ay + y)**2)*jnp.sin(2*pi*x/L)*jnp.cos(2*pi*y/L)/L + (B*jnp.exp(-(-bx + x)**2 - (-by + y)**2) + 1)**2*jnp.sin(2*pi*x/L)*jnp.sin(2*pi*y/L) + 8*pi**2*(A*jnp.exp(-(-ax + x)**2 - (-ay + y)**2) + 1)*jnp.sin(2*pi*x/L)*jnp.sin(2*pi*y/L)/L**2
def u_true(x, y, L):
    return jnp.sin(2*pi*x/L)*jnp.sin(2*pi*y/L)
# -----------------------------------------------------------------------------  
# Generate the “true solution” using a high resolution physical model.
true_params = jnp.array([
    3.5, -1.0, -1.0,   # kappa parameters
    1.5, 2.0, 1.0       # eta parameters
])
L = 6
domain = (-L//2, L//2)
def f(x, y):
    return f_full(true_params, x, y, L)



subdomain=[(-3, 3), (-3, 3)]
n_train=50
rng_x, rng_y = jax.random.split(jax.random.PRNGKey(0))
xx_train = jax.random.uniform(rng_x, shape=(n_train,), minval=subdomain[0][0], maxval=subdomain[0][1])
yy_train = jax.random.uniform(rng_y, shape=(n_train,), minval=subdomain[1][0], maxval=subdomain[1][1])
pts_train = jnp.stack([xx_train, yy_train], axis=-1)
# Use vmap over the new scalar __call__ for prediction.
u_train = u_true(xx_train, yy_train, L).reshape(-1, 1)
print(f"Training data generated with shape: {u_train.shape}")

def train_hybrid(epochs):
    # -------------------------------------------------------------------------
    # Initialize the synthetic model.
    synthetic_model = ResNetSynthetic(
        hidden_dims=(512, 512, 512, 512), 
        activation=nnx.relu, 
        output_dim=1, 
        rngs=nnx.Rngs(0)
    )

    # -------------------------------------------------------------------------
    # Setup physical model.
    low_res_N = 20
    rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(0), 3)
    amplitudes = jax.random.uniform(rng1, shape=(2,), minval=1, maxval=3)
    centers_x = jax.random.uniform(rng2, shape=(2,), minval=-1, maxval=1)
    centers_y = jax.random.uniform(rng3, shape=(2,), minval=-1, maxval=1)
    init_params = nnx.Param(jnp.array([
        amplitudes[0], centers_x[0], centers_y[0],
        amplitudes[1], centers_x[1], centers_y[1]
    ]))
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
    syn_opt = nnx.Optimizer(synthetic_model, optax.adam(1e-3))
    phys_opt = nnx.Optimizer(physical_model, optax.adam(5e-3))

    # Helper to vmap a scalar-call model.
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
    loss_history = []
    param_history = []
    rng = jax.random.PRNGKey(42)
    n_collocation = 50
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
            loss_history.append(loss_phy_data)
            param_history.append(physical_model.parameters.value)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss (physical): {loss_phy_data}, Loss (synthetic): {loss_syn}, Parameters: {physical_model.parameters.value}")

    
    return loss_history, param_history, synthetic_model



if __name__ == "__main__":
    epochs = 3000
    checkpointer = ocp.StandardCheckpointer()
    if input("Train the hybrid model? (y/n): ") == "y":
        loss_history_hyb, param_history_hyb, syn_model = train_hybrid(epochs)

        # convert them to njp arrays
        loss_history_hyb = jnp.array(loss_history_hyb)
        param_history_hyb = jnp.array(param_history_hyb)

        jnp.save("src/files/experiment_2/loss_history_hyb.npy", loss_history_hyb)
        jnp.save("src/files/experiment_2/param_history_hyb.npy", param_history_hyb)
        # save the model
        _, state = nnx.split(syn_model)
        
        checkpoint_path = os.path.abspath("src/files/experiment_2/checkpoint")
        ckpt_dir = ocp.test_utils.erase_and_create_empty(checkpoint_path)
        checkpointer.save(ckpt_dir / 'state', state)
    else:
        loss_history_hyb = jnp.load("src/files/experiment_2/loss_history_hyb.npy")
        param_history_hyb = jnp.load("src/files/experiment_2/param_history_hyb.npy")
        model = ResNetSynthetic(
            hidden_dims=(512, 512, 512, 512),
            activation=nnx.relu,
            output_dim=1,
            rngs=nnx.Rngs(0)
        )
        
        graphdef, state = nnx.split(model)
        checkpoint_path = os.path.abspath("src/files/experiment_2/checkpoint")
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
    x_plot = jnp.linspace(-L//2, L//2, n_plot)
    y_plot = jnp.linspace(-L//2, L//2, n_plot)
    xx_plot, yy_plot = jnp.meshgrid(x_plot, y_plot)
    xx_plot = xx_plot.flatten()
    yy_plot = yy_plot.flatten()
    u_true_plot = u_true(xx_plot, yy_plot, L)
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
    
    cs1 = ax2.contourf(x_plot, y_plot, u_phy_plot, cmap="viridis", levels=100, vmin=vmin_sol, vmax=vmax_sol)
    ax1.set_title("Physical")

    cs2 = ax1.contourf(x_plot, y_plot, u_syn_plot, cmap="viridis", levels=100, vmin=vmin_sol, vmax=vmax_sol)
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
    cbar_bot = fig.colorbar(err2, cax=ax_bot_cb, orientation="vertical", pad=0.02, ticks = [0, 0.01, 0.02, 0.03, 0.04, 0.05])

    # Add overall row labels on the left.

    plt.savefig("src/results/experiment_2/prediction_comparison.png")

    
    


    