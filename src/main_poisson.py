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

from models.physical_model import PoissonModel
from models.synthetic_model import FeedForwardNet

from tools.training import *

pi = jnp.pi

# ----------------------------------------------------------------------------- 
# Define coefficient functions as Gaussians.
def gaussian_kappa(parameters, x, y):
    amplitude, cx, cy, sigma = parameters[0:4]
    return amplitude * jnp.exp(-(((x - cx)**2 + (y - cy)**2) / (1))) + 1

# ----------------------------------------------------------------------------- 
def forcing_func(x, y):
    return 4 * pi**2 * jnp.sin(pi * x) * jnp.sin(pi * y)


# ----------------------------------------------------------------------------- 
# Generate the "true solution" using a high resolution model.
true_params = jnp.array([
    1.5, 0.5, 0.5, 1.0,   # kappa parameters
])
true_N = 80
true_model = PoissonModel(
    domain=(0.0, 1.0),
    N=true_N,
    parameters=true_params,
    training=False,
    forcing_func=forcing_func,
    kappa_func=gaussian_kappa
)
print(f"True model initialized with parameters: {true_params}")

x_full = jnp.linspace(0.0, 1.0, true_N + 1)
y_full = jnp.linspace(0.0, 1.0, true_N + 1)
xx_full, yy_full = jnp.meshgrid(x_full, y_full)
pts_full = jnp.stack([xx_full.flatten(), yy_full.flatten()], axis=-1)


# Sample the true solution on a subdomain.
n_train = 5
x_train = jnp.linspace(0.7, 1.0, n_train)
y_train = jnp.linspace(0.7, 1.0, n_train)
xx_train, yy_train = jnp.meshgrid(x_train, y_train)
pts_train = jnp.stack([xx_train.flatten(), yy_train.flatten()], axis=-1)
rng = jax.random.PRNGKey(0)
var_true = true_model.init(rng, pts_train[:, 0], pts_train[:, 1], mutable=["cache", "state"])
u_train = true_model.apply(var_true, pts_train[:, 0], pts_train[:, 1], mutable=["cache", "state"])[0]
print(f"Training subdomain shape: {u_train.shape}")

# ----------------------------------------------------------------------------- 
# Create a lower resolution model with randomized parameters.
init_params = jnp.array([
    1.0, 0.4, 0.4, 1.0  # kappa parameters (randomized guess)    
])
low_res_N = 21
model_phy = PoissonModel(
    domain=(0.0, 1.0),
    N=low_res_N,
    parameters=init_params,
    training = True,
    forcing_func=forcing_func,
    kappa_func=gaussian_kappa
)
var_model_phy = model_phy.init(rng, pts_train[:, 0], pts_train[:, 1], mutable=["cache", "state"])

# -----------------------------------------------------------------------------
# Create a synthetic model
model_syn = FeedForwardNet(
    hidden_dims = [32, 32, 32],
    activation=jax.nn.relu,
    output_dim=1)

# Initialize the synthetic model
rng = jax.random.PRNGKey(42)
var_model_syn = model_syn.init(rng, pts_train[:, 0], pts_train[:, 1])


state_phys = TrainStatePhy.create(
    apply_fn=model_phy.apply, params=init_params, tx=optax.adam(1e-2), extra_state=var_model_phy
)

state_syn = TrainStateSyn.create(
    apply_fn=model_syn.apply, params=var_model_syn, tx=optax.adam(1e-2)
)

# ----------------------------------------------------------------------------- 
# Define the loss function.



def loss(params_phys, params_syn, extra_state, x, y, u_target, flag):
    model_updated = PoissonModel(
        domain=(0.0, 1.0),
        N=low_res_N,
        parameters=params_phys,
        training=True,
        forcing_func=forcing_func,
        kappa_func=gaussian_kappa
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
for epoch in range(num_epochs):
    state_phys, loss_val_phys = train_step_phys(state_phys, state_syn, pts_train[:, 0], pts_train[:, 1], u_train)
    state_syn, loss_val_syn = train_step_syn(state_phys, state_syn, pts_train[:, 0], pts_train[:, 1], u_train)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, MSE Loss Physics: {loss_val_phys:.6f}, MSE Loss Synthetic: {loss_val_syn:.6f}, \n Parameters: {state_phys.params}")

# JIT compile prediction for speed.
@jax.jit
def predict(state, x, y):
    model_trained = PoissonModel(
        domain=(0.0, 1.0),
        N=low_res_N,
        parameters=state.params,
        training=False,
        forcing_func=forcing_func,
        kappa_func=gaussian_kappa
    )
    u_pred, _ = model_trained.apply(state.extra_state, x, y, mutable=["cache", "state"])
    return u_pred

# Set model to eval mode
model_phy.eval()
u_pred = predict(state_phys, pts_full[:, 0], pts_full[:, 1])
u_pred_grid = u_pred.reshape(true_N + 1, true_N + 1)

plt.figure(figsize=(6, 5))
plt.contourf(xx_full, yy_full, u_pred_grid, levels=50, cmap="viridis")
plt.colorbar()
plt.title("Trained Model Prediction on Full Domain")
plt.xlabel("x")
plt.ylabel("y")
plt.show()