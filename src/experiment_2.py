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

from tools.training import *  # Assumes loss_fn is defined in tools.training
from tools.plotting import *   # Assumes plot is defined in tools.plotting

import time
import jax.profiler

pi = jnp.pi

# -----------------------------------------------------------------------------
# Define coefficient functions as Gaussians.
def gaussian_kappa(parameters, x, y):
    amplitude, cx, cy = parameters[0:3]
    return amplitude * jnp.exp(-(((x - cx)**2 + (y - cy)**2) / 1)) + 1

def gaussian_eta(parameters, x, y):
    amplitude, cx, cy = parameters[3:6]
    return (amplitude * jnp.exp(-(((x - cx)**2 + (y - cy)**2) / 1)) + 1)**2

# -----------------------------------------------------------------------------
def forcing_func(x, y):
    return 4 * (jnp.exp(-((x - 2.0)**2 + (y-2)**2)) - jnp.exp(-((x + 1.0)**2 + (y + 1.0)**2)))

# -----------------------------------------------------------------------------
# Generate the "true solution" using a high resolution model.
true_params = jnp.array([
    3.5, -1.0, -1.0,   # kappa parameters
    1.5, 2.0, 1.0   # eta parameters
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

# Sample the true solution on a subdomain.
n_train = 10
# Sample x_train and y_train randomly from the domain.
xx_train = jax.random.uniform(jax.random.PRNGKey(3), shape=(n_train**2,), minval=-2.5, maxval=2.5)
yy_train = jax.random.uniform(jax.random.PRNGKey(4), shape=(n_train**2,), minval=-2.5, maxval=2.5)

pts_train = jnp.stack([xx_train.flatten(), yy_train.flatten()], axis=-1)
rng = jax.random.PRNGKey(42)
var_true = true_model.init(rng, pts_train[:, 0], pts_train[:, 1], mutable=["cache", "state"])
u_train = true_model.apply(var_true, pts_train[:, 0], pts_train[:, 1], mutable=["cache", "state"])[0]
# dd 30% noise to the true solution.
u_train += 0.3 * jax.random.normal(rng, u_train.shape) * jnp.max(jnp.abs(u_train))
print(f"Training subdomain shape: {u_train.shape}")

# -----------------------------------------------------------------------------
# Create a lower resolution physics model with randomized parameters.
# create random amplitude, center x, and center y for kappa and eta
rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(54), 3)
amplitudes = jax.random.uniform(rng1, shape=(2,), minval=2, maxval=3)
centers_x = jax.random.uniform(rng2, shape=(2,), minval=-1.5, maxval=1.5)
centers_y = jax.random.uniform(rng3, shape=(2,), minval=-1.5, maxval=1.5)

init_params = jnp.array([amplitudes[0], centers_x[0], centers_y[0], amplitudes[1], centers_x[1], centers_y[1]])
print(f"Initial parameters: {init_params}")
low_res_N = 20
model_phy = HelmholtzModel(
    domain=domain,
    N=low_res_N,
    parameters=init_params,
    training=True,
    forcing_func=forcing_func,
    kappa_func=gaussian_kappa,
    eta_func=gaussian_eta,
)
var_model_phy = model_phy.init(rng, pts_train[:, 0], pts_train[:, 1], mutable=["cache", "state"])

# -----------------------------------------------------------------------------
# Create a synthetic model.
model_syn = ResNetSynthetic(
    num_blocks=3,
    features=100,
    activation=jax.nn.relu,
    output_dim=1
)
rng = jax.random.PRNGKey(42)
var_model_syn = model_syn.init(rng, pts_train[:, 0], pts_train[:, 1])

# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Hybrid Training Function
def hybrid_training():
    # Create training states for physics and synthetic models.
    state_phys = TrainStatePhy.create(
        apply_fn=model_phy.apply,
        params=init_params,
        tx=optax.adamw(5e-2),
        extra_state=var_model_phy
    )
    state_syn = TrainStateSyn.create(
        apply_fn=model_syn.apply,
        params=var_model_syn,
        tx=optax.adam(1e-3)
    )

    # Loss function now returns scalar loss and auxiliary info as tuple.
    def loss_fn_and_metrics(params_phys, params_syn, extra_state, x, y, u_target, flag):
        model_updated = HelmholtzModel(
            domain=domain,
            N=low_res_N,
            parameters=params_phys,
            training=True,
            forcing_func=forcing_func,
            kappa_func=gaussian_kappa,
            eta_func=gaussian_eta,
        )
        rng_local = jax.random.PRNGKey(0)
        loss_phys, loss_syn, loss_hyb, new_state = loss_fn(
            model_updated, model_syn,
            params_phys, params_syn, extra_state,
            x, y, u_target, rng_local
        )
        if flag == "phys":
            # Return pure physics loss as scalar.
            loss_val = 1e2 * loss_phys + loss_hyb
            metrics = {"phys_loss": loss_phys, "hyb_loss": loss_hyb}
        else:
            loss_val = 1e2 * loss_syn + loss_hyb
            metrics = {"syn_loss": loss_syn, "hyb_loss": loss_hyb}
        # Return scalar loss and auxiliary tuple.
        return loss_val, (new_state, metrics)

    # JIT compiled physics training step.
    @jax.jit
    def train_step_phys(state_phys, state_syn, x, y, u_target):
        (loss_val, (new_state, metrics)), grads = jax.value_and_grad(
            lambda p, es: loss_fn_and_metrics(p, state_syn.params, es, x, y, u_target, "phys"),
            has_aux=True
        )(state_phys.params, state_phys.extra_state)
        state_phys = state_phys.apply_gradients(grads=grads, extra_state=new_state)
        return state_phys, loss_val, metrics

    # JIT compiled synthetic training step.
    
    @jax.jit
    def train_step_syn(state_phys, state_syn, x, y, u_target):
        (loss_val, (new_state, metrics)), grads = jax.value_and_grad(
            lambda s: loss_fn_and_metrics(state_phys.params, s, state_phys.extra_state, x, y, u_target, "syn"),
            has_aux=True
        )(state_syn.params)
        state_syn = state_syn.apply_gradients(grads=grads)
        return state_syn, loss_val

    # Training loop.
    model_phy.train()  # Set to training mode.
    num_epochs = 3000

    loss_history_phys = []
    metrics_history = []
    state_history = []

    for epoch in range(num_epochs):
        state_phys, loss_val_phys, metrics = train_step_phys(
            state_phys, state_syn, pts_train[:, 0], pts_train[:, 1], u_train
        )
        loss_history_phys.append(jax.device_get(loss_val_phys))
        metrics_history.append(jax.device_get(metrics))
        state_history.append(state_phys.params)
        if epoch % 100 == 0:
            print(f"[Hybrid] Epoch {epoch}, Loss: {loss_val_phys:.6f}, Params: {state_phys.params}")

        # Synthetic updates for each physics update.
        for _ in range(3):
            state_syn, loss_val_syn = train_step_syn(
                state_phys, state_syn, pts_train[:, 0], pts_train[:, 1], u_train
            )
            # Optionally log the synthetic branch loss.

    return state_phys, loss_history_phys, metrics_history, state_history

# -----------------------------------------------------------------------------
# Physics-only Training Function
def phys_training():
    # Create training states for physics and synthetic models.
    state_phys = TrainStatePhy.create(
        apply_fn=model_phy.apply,
        params=init_params,
        tx=optax.adamw(5e-2),
        extra_state=var_model_phy
    )
    state_syn = TrainStateSyn.create(
        apply_fn=model_syn.apply,
        params=var_model_syn,
        tx=optax.adam(1e-3)
    )

    # Loss function returning only pure physics loss.
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
        rng_local = jax.random.PRNGKey(0)
        loss_phys, loss_syn, loss_hyb, new_state = loss_fn(
            model_updated, model_syn,
            params_phys, params_syn, extra_state,
            x, y, u_target, rng_local
        )
        return 1e2 * loss_phys, new_state

    # JIT compiled physics training step.
    @jax.jit
    def train_step_phys_only(state_phys, state_syn, x, y, u_target):
        (loss_val, new_state), grads = jax.value_and_grad(loss, has_aux=True)(
            state_phys.params, state_syn.params, state_phys.extra_state, x, y, u_target, "phys"
        )
        state_phys = state_phys.apply_gradients(grads=grads, extra_state=new_state)
        return state_phys, loss_val

    model_phy.train()  # Set to training mode.
    num_epochs = 3000
    loss_history_phys = []
    state_history = []

    for epoch in range(num_epochs):
        state_phys, loss_val_phys = train_step_phys_only(
            state_phys, state_syn, pts_train[:, 0], pts_train[:, 1], u_train
        )
        loss_history_phys.append(jax.device_get(loss_val_phys))
        state_history.append(state_phys.params)
        if epoch % 100 == 0:
            print(f"[Physics] Epoch {epoch}, Loss: {loss_val_phys:.6f}, Params: {state_phys.params}")

    return state_phys, loss_history_phys, state_history

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if input("Reproduce data? (y/n): ") == "y":
        state_hyb, hyb_phys_loss_hist, hyb_metrics_hist, hyb_state_hist = hybrid_training()
        state_phys, phys_loss_hist, phy_state_hist = phys_training()
        # Save all in files
        jnp.save("src/files/experiment_2/helmholtz_hybrid_loss.npy", hyb_phys_loss_hist)
        jnp.save("src/files/experiment_2/helmholtz_hybrid_metrics.npy", hyb_metrics_hist)
        jnp.save("src/files/experiment_2/helmholtz_hybrid_state_hist.npy", hyb_state_hist)

        jnp.save("src/files/experiment_2/helmholtz_phys_loss.npy", phys_loss_hist)
        jnp.save("src/files/experiment_2/helmholtz_phys_state_hist.npy", phy_state_hist)

    else:
        hyb_phys_loss_hist = jnp.load("src/files/experiment_2/helmholtz_hybrid_loss.npy", allow_pickle=True)
        hyb_metrics_hist = jnp.load("src/files/experiment_2/helmholtz_hybrid_metrics.npy", allow_pickle=True)
        hyb_state_hist = jnp.load("src/files/experiment_2/helmholtz_hybrid_state_hist.npy", allow_pickle=True)

        phys_loss_hist = jnp.load("src/files/experiment_2/helmholtz_phys_loss.npy", allow_pickle=True)
        phy_state_hist = jnp.load("src/files/experiment_2/helmholtz_phys_state_hist.npy", allow_pickle=True)

    plot(
        phy_state_hist,
        hyb_state_hist,
        true_params,
        phys_loss_hist,
        hyb_phys_loss_hist,
        gaussian_kappa,
        gaussian_eta,
        pts_train,
        domain=(-3.0, 3.0),
        N=100,
        filename="experiment_2/helmholtz.png"
    )
    animate(
        phy_state_hist,
        hyb_state_hist,
        true_params,
        phys_loss_hist,
        hyb_phys_loss_hist,
        gaussian_kappa,
        gaussian_eta,
        pts_train,
        domain=(-3.0, 3.0),
        N=100,
        filename="experiment_2/helmholtz.gif"
    )
    