import os
import sys
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from functools import partial
from flax import nnx
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



subdomain=[(-3.0, 3.0), (-3.0, 3.0)]
n_train=100
rng_x, rng_y = jax.random.split(jax.random.PRNGKey(0))
xx_train = jax.random.uniform(rng_x, shape=(n_train,), minval=subdomain[0][0], maxval=subdomain[0][1])
yy_train = jax.random.uniform(rng_y, shape=(n_train,), minval=subdomain[1][0], maxval=subdomain[1][1])
pts_train = jnp.stack([xx_train, yy_train], axis=-1)
# Use vmap over the new scalar __call__ for prediction.
u_train = u_true(xx_train, yy_train, L).reshape(-1, 1)
u_train += jax.random.normal(jax.random.PRNGKey(0), shape=u_train.shape) * 0.1 * jnp.std(u_train)
print(f"Training data generated with shape: {u_train.shape}")

def train_hybrid(epochs):
    # -------------------------------------------------------------------------
    # Initialize the synthetic model.
    synthetic_model = ResNetSynthetic(
        hidden_dims=(256,256), 
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
    n_collocation = 20
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

    return loss_history, param_history

def train_fem(epochs):
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
    # Initialize the optimizer.
    phys_opt = nnx.Optimizer(physical_model, optax.adam(5e-3))

    def vmapped_model(m, xs, ys):
        return jax.vmap(lambda xx, yy: m(xx, yy))(xs, ys)

    @nnx.jit
    def train_step(model, optimizer, x, y, u):
        def loss_data(m):
            u_pred = vmapped_model(m, x, y)
            return jnp.mean(optax.squared_error(u_pred, u))
        
        def loss_fn(m):
            return  loss_data(m)
        
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)  # In-place update.
        return loss   

    # -------------------------------------------------------------------------
    # Train the physical model.
    loss_history = []
    param_history = []
    for epoch in range(epochs):
        loss_phy = train_step(physical_model, phys_opt, xx_train, yy_train, u_train)
        loss_history.append(loss_phy)
        param_history.append(physical_model.parameters.value)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss (physical): {loss_phy}, Parameters: {physical_model.parameters.value}")

    return loss_history, param_history


def trian_pinn(epochs):
    # Create the PINN model instance.
    rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(0), 3)
    amplitudes = jax.random.uniform(rng1, shape=(2,), minval=1, maxval=3)
    centers_x = jax.random.uniform(rng2, shape=(2,), minval=-1, maxval=1)
    centers_y = jax.random.uniform(rng3, shape=(2,), minval=-1, maxval=1)
    init_params = nnx.Param(jnp.array([
        amplitudes[0], centers_x[0], centers_y[0],
        amplitudes[1], centers_x[1], centers_y[1]
    ]))
    model = ResNetSynthetic(
        hidden_dims=(256,256), 
        activation=nnx.tanh, 
        output_dim=1, 
        rngs=nnx.Rngs(0)
    )
    pinn = PINN(domain=domain, model=model, parameters=init_params, forcing_func=f,
                kappa_func=kappa, eta_func=eta, rngs=nnx.Rngs(0))
    
    tx = optax.multi_transform(
        {
        "model": optax.adam(1e-3),
        "parameters": optax.adam(5e-3),
        },
        nnx.State({
            "model": "model",
            "parameters": "parameters"
        })
    )
    opt = nnx.Optimizer(pinn, tx)
    
    

    # Define vmapped helper functions for the scalar calls.
    def vmapped_model(m, xs, ys):
        return jax.vmap(lambda xx, yy: m(xx, yy))(xs, ys)
    
    def vmapped_residual(m, xs, ys):
        return jax.vmap(lambda xx, yy: m.residual(xx, yy))(xs, ys)
    
    @nnx.jit
    def train_step_pinn(model, optimizer,x_i, y_i, x_b, y_b): 
        def loss_res(m):
            u_res = vmapped_residual(m, x_i, y_i)
            u_b = vmapped_model(m, x_b, y_b)
            return optax.squared_error(u_res).mean() + 1e1*optax.squared_error(u_b).mean()
        
        def loss_fn(m):
            return  loss_res(m)
        
        # Compute the residual loss separately.
        
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        
        # If the residual loss is too high, freeze (zero) the gradients.
        grads['parameters'] = jax.tree.map(lambda g: 0.0, grads['parameters'])
        optimizer.update(grads)  # In-place update.
        return loss

    
    def train_step(model, optimizer, x, y, u, x_i, y_i, x_b, y_b): 
        def loss_data(m):
            u_pred = vmapped_model(m, x, y)
            return optax.squared_error(u_pred, u).mean()
        
        def loss_res(m):
            u_res = vmapped_residual(m, x_i, y_i)
            u_b = vmapped_model(m, x_b, y_b)
            return optax.squared_error(u_res).mean() + 1e1*optax.squared_error(u_b).mean()
        
        def loss_fn(m):
            return  loss_data(m) + loss_res(m)

        dloss = loss_data(model)
        loss, grads = nnx.value_and_grad(loss_fn)(model)

        optimizer.update(grads)  # In-place update.
        return loss, dloss

    # Training loop.
    loss_history = []
    param_history = []

    # Create interior (collocation) points.
    n_interior = 400
    n_boundary = 400
    loss_pinn = 1
    rng = jax.random.PRNGKey(42)
    for epoch in range(epochs):
        if loss_pinn > 1e-1:
            rng, rng1= jax.random.split(rng, 2)
            x_in, y_in, x_b, y_b = pinn.create_collocation_points(n_interior, n_boundary, rng1)
            loss_pinn = train_step_pinn(pinn, opt, x_in, y_in, x_b, y_b)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, PINN Loss: {loss_pinn}")
            
        else: 
            rng, rng1= jax.random.split(rng, 2)
            x_in, y_in, x_b, y_b = pinn.create_collocation_points(n_interior, n_boundary, rng1)
            loss_val, loss_data = train_step(pinn, opt, xx_train, yy_train, u_train,
                                x_in, y_in, x_b, y_b)
            loss_pinn = loss_val - loss_data    
            loss_history.append(loss_data)
            param_history.append(pinn.parameters.value)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, PINN Loss: {loss_val}, PINN Parameters: {pinn.parameters.value}")

    print(f"Final PINN model parameters: {pinn.parameters}")
    return loss_history, param_history


if __name__ == "__main__":
    epochs = 3000
    if input("Run the experiment? (y/n): ") == "y":
        
        loss_history_pinn, param_history_pinn = trian_pinn(epochs)
        loss_history_hyb, param_history_hyb = train_hybrid(epochs)
        loss_history_fem, param_history_fem = train_fem(epochs)

        # convert them to njp arrays
        loss_history_hyb = jnp.array(loss_history_hyb)
        loss_history_fem = jnp.array(loss_history_fem)
        loss_history_pinn = jnp.array(loss_history_pinn)
        param_history_hyb = jnp.array(param_history_hyb)
        param_history_fem = jnp.array(param_history_fem)
        param_history_pinn = jnp.array(param_history_pinn)
        

        # Save the results.
        jnp.save("src/files/experiment_2/hybrid_loss.npy", loss_history_hyb)
        jnp.save("src/files/experiment_2/hybrid_params.npy", param_history_hyb)
        jnp.save("src/files/experiment_2/fem_loss.npy", loss_history_fem)
        jnp.save("src/files/experiment_2/fem_params.npy", param_history_fem)
        jnp.save("src/files/experiment_2/pinn_loss.npy", loss_history_pinn)
        jnp.save("src/files/experiment_2/pinn_params.npy", param_history_pinn)
    else:
        loss_history_hyb = jnp.load("src/files/experiment_2/hybrid_loss.npy")
        loss_history_fem = jnp.load("src/files/experiment_2/fem_loss.npy")
        loss_history_pinn = jnp.load("src/files/experiment_2/pinn_loss.npy")
        param_history_hyb = jnp.load("src/files/experiment_2/hybrid_params.npy")
        param_history_fem = jnp.load("src/files/experiment_2/fem_params.npy")
        param_history_pinn = jnp.load("src/files/experiment_2/pinn_params.npy")



    # If the history is not equal to epochs, pad it at the start with the initial value
    if len(loss_history_hyb) < epochs:
        loss_history_hyb = jnp.pad(loss_history_hyb, (epochs - len(loss_history_hyb), 0), mode="constant", constant_values=loss_history_hyb[0])
    if len(loss_history_fem) < epochs:
        loss_history_fem = jnp.pad(loss_history_fem, (epochs - len(loss_history_fem), 0), mode="constant", constant_values=loss_history_fem[0])
    if len(loss_history_pinn) < epochs:
        loss_history_pinn = jnp.pad(loss_history_pinn, (epochs - len(loss_history_pinn), 0), mode="constant", constant_values=loss_history_pinn[0])
    if len(param_history_hyb) < epochs:
        param_history_hyb = jnp.pad(param_history_hyb, ((epochs - len(param_history_hyb), 0), (0, 0)), mode="edge")
    if len(param_history_fem) < epochs:
        param_history_fem = jnp.pad(param_history_fem, ((epochs - len(param_history_fem), 0), (0, 0)), mode="edge")
    if len(param_history_pinn) < epochs:
        param_history_pinn = jnp.pad(param_history_pinn, ((epochs - len(param_history_pinn), 0), (0, 0)), mode="edge")
    
    # Plot the results.
    plot(
    param_history_fem,
    param_history_hyb,
    param_history_pinn,      # New argument for PINN training results.
    true_params,
    loss_history_fem,
    loss_history_hyb,
    loss_history_pinn,          # New loss history for PINN.
    kappa,
    eta,
    pts_train,
    domain=(-3.0, 3.0),
    N=100,
    filename="experiment_2/experiment_2"
    )

    animate(
    param_history_fem,
    param_history_hyb,
    param_history_pinn,      # New argument for PINN training results.
    true_params,
    loss_history_fem,
    loss_history_hyb,
    loss_history_pinn,          # New loss history for PINN.
    kappa,
    eta,
    pts_train,
    domain=(-3.0, 3.0),
    N=100,
    filename="experiment_2/experiment_2"
    )


    
    


    