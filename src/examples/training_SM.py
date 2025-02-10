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

from models.physical_model import PoissonModel  # to generate the true solution
from models.synthetic_model import FeedForwardNet  # the synthetic model

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
# Generate the "true solution" using a high resolution physical model.
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

# ----------------------------------------------------------------------------- 
# Collect training data on a subdomain.
n_train = 50
x_train = jnp.linspace(0, 1.0, n_train)
y_train = jnp.linspace(0, 1.0, n_train)
xx_train, yy_train = jnp.meshgrid(x_train, y_train)
pts_train = jnp.stack([xx_train.flatten(), yy_train.flatten()], axis=-1)
u_train = true_model.apply(var_true, pts_train[:, 0], pts_train[:, 1], mutable=["cache", "state"])[0]
print(f"Training subdomain shape: {u_train.shape}")

# ----------------------------------------------------------------------------- 
# Define the synthetic model.
# We use a feedforward neural network with random initial parameters.
syn_model = FeedForwardNet(hidden_dims=(64, 64, 64), output_dim=1)
syn_params = syn_model.init(rng, pts_train[:, 0], pts_train[:, 1])
# For synthetic models no extra mutable state is required.
class TrainState(train_state.TrainState):
    pass

state = TrainState.create(
    apply_fn=syn_model.apply,
    params=syn_params,
    tx=optax.adam(1e-3)
)

# ----------------------------------------------------------------------------- 
# Define the loss function.
def loss_fn(params, x, y, u_target):
    u_pred = syn_model.apply(params, x, y)
    loss = jnp.mean((u_pred - u_target) ** 2)
    return loss

# JIT-compiled training step.
@jax.jit
def train_step(state, x, y, u_target):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, x, y, u_target)
    state = state.apply_gradients(grads=grads)
    return state, loss

# ----------------------------------------------------------------------------- 
# Training loop.
num_epochs = 3000
for epoch in range(num_epochs):
    state, loss_val = train_step(state, pts_train[:, 0], pts_train[:, 1], u_train)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, MSE Loss: {loss_val:.6f}")

# JIT-compile prediction for speed.
@jax.jit
def predict(params, x, y):
    return syn_model.apply(params, x, y)

# Evaluate the trained synthetic model on the full domain.
u_pred = predict(state.params, pts_full[:, 0], pts_full[:, 1])
u_pred_grid = u_pred.reshape(true_N + 1, true_N + 1)

plt.figure(figsize=(6, 5))
plt.contourf(xx_full, yy_full, u_pred_grid, levels=50, cmap="viridis")
plt.colorbar()
plt.title("Recovered Synthetic Model Prediction on Full Domain")
plt.xlabel("x")
plt.ylabel("y")
plt.show()