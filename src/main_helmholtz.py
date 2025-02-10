import os
import sys

# Set working directory to the src folder.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
os.chdir(src_dir)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import optax
from flax.training import train_state

from models.physical_model import HelmholtzModel
from models.synthetic_model import ResNetSynthetic

from tools.training import *

pi = jnp.pi

# ----------------------------------------------------------------------------- 
# Define coefficient functions as Gaussians.
def gaussian_kappa(parameters, x, y):
    amplitude, cx, cy = parameters[0:3]
    return amplitude * jnp.exp(-(((x - cx)**2 + (y - cy)**2) / (1))) + 1

def gaussian_eta(parameters, x, y):
    amplitude, cx, cy = parameters[3:6]
    return (amplitude * jnp.exp(-(((x - cx)**2 + (y - cy)**2) / (1))) + 1)**2

# ----------------------------------------------------------------------------- 
def forcing_func(x, y):
    return 4 * jnp.exp(-((x - 2.0)**2 + (y - 2.0)**2) / 0.1) - jnp.exp(-((x + 1.0)**2 + (y + 1.0)**2))


# ----------------------------------------------------------------------------- 
# Generate the "true solution" using a high resolution model.
true_params = jnp.array([
    4.0, 1.0, 1.0,   # kappa parameters
    3.0, -1.0, -1.0    # eta parameters
])
domain = (-3.0, 3.0)
true_N = 100
true_model = HelmholtzModel(
    domain=domain,
    N=true_N,
    parameters=true_params,
    training=False,
    forcing_func=forcing_func,
    kappa_func=gaussian_kappa,
    eta_func=gaussian_eta,
)
print(f"True model initialized with parameters: {true_params}")

x_full = jnp.linspace(0.0, 1.0, true_N + 1)
y_full = jnp.linspace(0.0, 1.0, true_N + 1)
xx_full, yy_full = jnp.meshgrid(x_full, y_full)
pts_full = jnp.stack([xx_full.flatten(), yy_full.flatten()], axis=-1)


# Sample the true solution on a subdomain.
n_train = 5
x_train = jnp.linspace(1.0, 2.5, n_train)
y_train = jnp.linspace(1.0, 2.5, n_train)
xx_train, yy_train = jnp.meshgrid(x_train, y_train)
pts_train = jnp.stack([xx_train.flatten(), yy_train.flatten()], axis=-1)
rng = jax.random.PRNGKey(0)
var_true = true_model.init(rng, pts_train[:, 0], pts_train[:, 1], mutable=["cache", "state"])
u_train = true_model.apply(var_true, pts_train[:, 0], pts_train[:, 1], mutable=["cache", "state"])[0]
print(f"Training subdomain shape: {u_train.shape}")

# ----------------------------------------------------------------------------- 
# Create a lower resolution model with randomized parameters.
init_params = jnp.array([
    6.0, -1.0, 1.0,   # kappa parameters (randomized guess)
    4.0, 1.0, -1.0,    # eta parameters     
])
low_res_N = 21
model_phy = HelmholtzModel(
    domain=domain,
    N=low_res_N,
    parameters=init_params,
    training = True,
    forcing_func=forcing_func,
    kappa_func=gaussian_kappa,
    eta_func=gaussian_eta,
)
var_model_phy = model_phy.init(rng, pts_train[:, 0], pts_train[:, 1], mutable=["cache", "state"])

# -----------------------------------------------------------------------------
# Create a synthetic model
model_syn = ResNetSynthetic(
    num_blocks=3,
    features=32,
    activation=jax.nn.relu,
    output_dim=1
)

# Initialize the synthetic model
rng = jax.random.PRNGKey(42)
var_model_syn = model_syn.init(rng, pts_train[:, 0], pts_train[:, 1])




# JIT compile prediction for speed.
@jax.jit
def predict(state, x, y):
    model_trained = HelmholtzModel(
        domain=domain,
        N=low_res_N,
        parameters=state.params,
        training=False,
        forcing_func=forcing_func,
        kappa_func=gaussian_kappa,
        eta_func=gaussian_eta,
    )
    u_pred, _ = model_trained.apply(state.extra_state, x, y, mutable=["cache", "state"])
    return u_pred

def hybrid_training():

    # ----------------------------------------------------------------------------- 
    # Define the loss function.

    state_phys = TrainStatePhy.create(
    apply_fn=model_phy.apply, params=init_params, tx=optax.adam(1e-1), extra_state=var_model_phy
    )

    state_syn = TrainStateSyn.create(
        apply_fn=model_syn.apply, params=var_model_syn, tx=optax.adam(1e-3)
    )

    def loss(params_phys, params_syn, extra_state, x, y, u_target, flag):
        model_updated = HelmholtzModel(
            domain=domain,
            N=low_res_N,
            parameters=params_phys,
            training=True,
            forcing_func=forcing_func,
            kappa_func=gaussian_kappa,
            eta_func=gaussian_eta,
        )
        rng = jax.random.PRNGKey(0)
        loss_phys, loss_syn, loss_hyb, new_state = loss_fn(model_updated, model_syn, params_phys, params_syn, extra_state, x, y, u_target, rng)
        if flag == "phys":
            return 1e3*loss_phys + loss_hyb, new_state
        else:
            return 1e3*loss_syn + loss_hyb
        

    # JIT compiled training step.
    @jax.jit
    def train_step_phys(state_phys, state_syn, x, y, u_target):
        (loss_val, new_state), grads = jax.value_and_grad(loss, has_aux=True)(
            state_phys.params, state_syn.params, state_phys.extra_state, x, y, u_target, "phys"
        )
        state_phys = state_phys.apply_gradients(grads=grads, extra_state=new_state)
        return state_phys, loss_val

    @jax.jit
    def train_step_syn(state_phys, state_syn, x, y, u_target):
        loss_val, grads = jax.value_and_grad(loss, argnums = 1, has_aux=False)(
            state_phys.params, state_syn.params, state_phys.extra_state, x, y, u_target, "syn"
        )
        state_syn = state_syn.apply_gradients(grads=grads)
        return state_syn, loss_val

    # ----------------------------------------------------------------------------- 
    # Training loop.
    # Set model to training
    model_phy.train()
    num_epochs = 3000
    loss_history_phys = []
    loss_history_syn = []
    for epoch in range(num_epochs):
        state_phys, loss_val_phys = train_step_phys(state_phys, state_syn,
                                                    pts_train[:, 0], pts_train[:, 1], u_train)
        loss_history_phys.append(loss_val_phys)
        # Run several synthetic updates for each physics update.
        for _ in range(10):
            state_syn, loss_val_syn = train_step_syn(state_phys, state_syn,
                                                     pts_train[:, 0], pts_train[:, 1], u_train)
        loss_history_syn.append(loss_val_syn)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, MSE Loss Physics: {loss_val_phys:.6f}, MSE Loss Synthetic: {loss_val_syn:.6f}, \n Parameters: {state_phys.params}")

    return state_phys, loss_history_phys, loss_history_syn


def phys_training():
    # ----------------------------------------------------------------------------- 
    # Define the loss function.
    state_phys = TrainStatePhy.create(
    apply_fn=model_phy.apply, params=init_params, tx=optax.adam(1e-1), extra_state=var_model_phy
    )

    state_syn = TrainStateSyn.create(
        apply_fn=model_syn.apply, params=var_model_syn, tx=optax.adam(1e-3)
    )


    def loss(params_phys, params_syn, extra_state, x, y, u_target, flag):
        model_updated = HelmholtzModel(
            domain=domain,
            N=low_res_N,
            parameters=params_phys,
            training=True,
            forcing_func=forcing_func,
            kappa_func=gaussian_kappa,
            eta_func=gaussian_eta,
        )
        rng = jax.random.PRNGKey(0)
        loss_phys, loss_syn, loss_hyb, new_state = loss_fn(model_updated, model_syn, params_phys, params_syn, extra_state, x, y, u_target, rng)
        return 1e3*loss_phys, new_state
        

    # JIT compiled training step.
    @jax.jit
    def train_step_phys(state_phys, state_syn, x, y, u_target):
        (loss_val, new_state), grads = jax.value_and_grad(loss, has_aux=True)(
            state_phys.params, state_syn.params, state_phys.extra_state, x, y, u_target, "phys"
        )
        state_phys = state_phys.apply_gradients(grads=grads, extra_state=new_state)
        return state_phys, loss_val


    # ----------------------------------------------------------------------------- 
    # Training loop.
    # Set model to training
    model_phy.train()
    num_epochs = 3000
    loss_history_phys = []
    for epoch in range(num_epochs):
        state_phys, loss_val_phys = train_step_phys(state_phys, state_syn,
                                                    pts_train[:, 0], pts_train[:, 1], u_train)
        loss_history_phys.append(loss_val_phys)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, MSE Loss Physics: {loss_val_phys:.6f}, Params: {state_phys.params}")

    return state_phys, loss_history_phys

if __name__ == "__main__":
    state_phys, phys_loss_hist = phys_training()
    state_hyb, hyb_phys_loss_hist, hyb_syn_loss_hist = hybrid_training()

    # ----------------------------------------------------------------------------- 
    # Plot the functions eta and kappa.
    x = jnp.linspace(-3.0, 3.0, 100)
    y = jnp.linspace(-3.0, 3.0, 100)
    xx, yy = jnp.meshgrid(x, y)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    kappa_phys = gaussian_kappa(state_phys.params, xx_flat, yy_flat)
    eta_phys = gaussian_eta(state_phys.params, xx_flat, yy_flat)
    kappa_hyb = gaussian_kappa(state_hyb.params, xx_flat, yy_flat)
    eta_hyb = gaussian_eta(state_hyb.params, xx_flat, yy_flat)

    fig, axs = plt.subplots(3, 2, figsize=(14, 18))
    
    # Contour plots for kappa and eta.
    ax = axs[0, 0]
    ax.set_title("Kappa Physics")
    cf = ax.contourf(xx, yy, kappa_phys.reshape(xx.shape))
    plt.colorbar(cf, ax=ax)
    
    ax = axs[0, 1]
    ax.set_title("Kappa Hybrid")
    cf = ax.contourf(xx, yy, kappa_hyb.reshape(xx.shape))
    plt.colorbar(cf, ax=ax)
    
    ax = axs[1, 0]
    ax.set_title("Eta Physics")
    cf = ax.contourf(xx, yy, eta_phys.reshape(xx.shape))
    plt.colorbar(cf, ax=ax)
    
    ax = axs[1, 1]
    ax.set_title("Eta Hybrid")
    cf = ax.contourf(xx, yy, eta_hyb.reshape(xx.shape))
    plt.colorbar(cf, ax=ax)
    
    # Plot the training losses.
    ax_loss = axs[2, 0]
    ax_loss.plot(phys_loss_hist, label="Physics Training Loss")
    ax_loss.plot(hyb_phys_loss_hist, label="Hybrid Physics Loss", linestyle="--")
    ax_loss.set_title("Physics Loss History")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    
    ax_loss_syn = axs[2, 1]
    ax_loss_syn.plot(hyb_syn_loss_hist, label="Hybrid Synthetic Loss", color="orange")
    ax_loss_syn.set_title("Synthetic Loss History")
    ax_loss_syn.set_xlabel("Epoch")
    ax_loss_syn.set_ylabel("Loss")
    ax_loss_syn.legend()

    plt.tight_layout()
    plt.savefig("results/hybrid_training.png")
