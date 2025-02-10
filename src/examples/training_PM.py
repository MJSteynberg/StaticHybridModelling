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

pi = jnp.pi

# ----------------------------------------------------------------------------- 
# Define coefficient functions as Gaussians.
def gaussian_kappa(parameters, x, y):
    amplitude, cx, cy, sigma = parameters[0:4]
    return amplitude * jnp.exp(-(((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))) + 1

def gaussian_eta(parameters, x, y):
    amplitude, cx, cy, sigma = parameters[4:8]
    return amplitude * jnp.exp(-(((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))) + 1

# ----------------------------------------------------------------------------- 
# Define the forcing function so that u_exact(x,y)=sin(pi*x)*sin(pi*y)
def forcing_func(x, y):
    return 4 * pi**2 * jnp.sin(pi * x) * jnp.sin(pi * y)

def u_exact(x, y):
    return jnp.sin(pi * x) * jnp.sin(pi * y)

# ----------------------------------------------------------------------------- 
# Generate the "true solution" using a high resolution model.
true_params = jnp.array([
    1.5, 0.5, 0.5, 1.0,   # kappa parameters
    2.0, 0.7, 0.7, 0.15    # eta parameters
])
true_N = 80
true_model = PoissonModel(
    domain=(0.0, 1.0),
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

rng = jax.random.PRNGKey(0)
var_true = true_model.init(rng, pts_full[:, 0], pts_full[:, 1], mutable=["cache", "state"])
u_true, _ = true_model.apply(var_true, pts_full[:, 0], pts_full[:, 1], mutable=["cache", "state"])
u_true_grid = u_true.reshape(true_N + 1, true_N + 1)
print(f"True solution shape: {u_true_grid.shape}")

# Sample the true solution on a subdomain.
n_train = 50
x_train = jnp.linspace(0, 1.0, n_train)
y_train = jnp.linspace(0, 1.0, n_train)
xx_train, yy_train = jnp.meshgrid(x_train, y_train)
pts_train = jnp.stack([xx_train.flatten(), yy_train.flatten()], axis=-1)
u_train = true_model.apply(var_true, pts_train[:, 0], pts_train[:, 1], mutable=["cache", "state"])[0]
print(f"Training subdomain shape: {u_train.shape}")

# ----------------------------------------------------------------------------- 
# Create a lower resolution model with randomized parameters.
init_params = jnp.array([
    1.0, 0.4, 0.4, 0.2,   # kappa parameters (randomized guess)
    1.0, 0.6, 0.6, 0.2    # eta parameters     
])
low_res_N = 21
model = PoissonModel(
    domain=(0.0, 1.0),
    N=low_res_N,
    parameters=init_params,
    training = True,
    forcing_func=forcing_func,
    kappa_func=gaussian_kappa,
    eta_func=gaussian_eta,
)
var_model = model.init(rng, pts_train[:, 0], pts_train[:, 1], mutable=["cache", "state"])

# Define a custom TrainState to include mutable state.
class TrainState(train_state.TrainState):
    extra_state: dict

state = TrainState.create(
    apply_fn=model.apply, params=init_params, tx=optax.adam(1e-2), extra_state=var_model
)

# ----------------------------------------------------------------------------- 
# Define the loss function.
def loss_fn(params, extra_state, x, y, u_target):
    model_updated = PoissonModel(
        domain=(0.0, 1.0),
        N=low_res_N,
        parameters=params,
        training=True,
        forcing_func=forcing_func,
        kappa_func=gaussian_kappa,
        eta_func=gaussian_eta,
    )
    u_pred, new_state = model_updated.apply(extra_state, x, y, mutable=["cache", "state"])
    loss = jnp.mean((u_pred - u_target) ** 2)
    return loss, new_state

# JIT compiled training step.
@jax.jit
def train_step(state, x, y, u_target):
    (loss_val, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, state.extra_state, x, y, u_target
    )
    state = state.apply_gradients(grads=grads, extra_state=new_state)
    return state, loss_val

# ----------------------------------------------------------------------------- 
# Training loop.
# Set model to training
model.train()
num_epochs = 3000
for epoch in range(num_epochs):
    state, loss_val = train_step(state, pts_train[:, 0], pts_train[:, 1], u_train)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, MSE Loss: {loss_val:.6f}, Parameters: {state.params}")

# JIT compile prediction for speed.
@jax.jit
def predict(state, x, y):
    model_trained = PoissonModel(
        domain=(0.0, 1.0),
        N=low_res_N,
        parameters=state.params,
        training=False,
        forcing_func=forcing_func,
        kappa_func=gaussian_kappa,
        eta_func=gaussian_eta,
    )
    u_pred, _ = model_trained.apply(state.extra_state, x, y, mutable=["cache", "state"])
    return u_pred

# Set model to eval mode
model.eval()
u_pred = predict(state, pts_full[:, 0], pts_full[:, 1])
u_pred_grid = u_pred.reshape(true_N + 1, true_N + 1)

plt.figure(figsize=(6, 5))
plt.contourf(xx_full, yy_full, u_pred_grid, levels=50, cmap="viridis")
plt.colorbar()
plt.title("Trained Model Prediction on Full Domain")
plt.xlabel("x")
plt.ylabel("y")
plt.show()