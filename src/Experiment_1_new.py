import os
import sys
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from functools import partial
from flax import nnx
import numpy as np

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
    return -A*(2*ax - 2*x)*jnp.exp(-(-ax + x)**2 - (-ay + y)**2)*jnp.sin(y)*jnp.cos(x) - A*(2*ay - 2*y)*jnp.exp(-(-ax + x)**2 - (-ay + y)**2)*jnp.sin(x)*jnp.cos(y) + 2*(A*jnp.exp(-(-ax + x)**2 - (-ay + y)**2) + 1)*jnp.sin(x)*jnp.sin(y) + (B*jnp.exp(-(-bx + x)**2 - (-by + y)**2) + 1)**2*jnp.sin(x)*jnp.sin(y)

def u_true(x, y, L):
    return jnp.sin(x)*jnp.sin(y)
# -----------------------------------------------------------------------------  
# Generate the “true solution” using a high resolution physical model.
true_params = jnp.array([
    4, -1.0, -1.0,   # kappa parameters
    1, 2.0, 1.0       # eta parameters
])
L = 2*pi
domain = (-L/2, L/2)
def f(x, y):
    return f_full(true_params, x, y, L)





# from mpl_toolkits.axes_grid1 import make_axes_locatable

# # plot u, kappa, eta, and f with improved formatting using contourf with 100 levels
# plt.figure(figsize=(16, 4))
# plot_data = [
#     (u_true(xx_eval, yy_eval, L).reshape(100, 100), r'$u$'),
#     (kappa(true_params, xx_eval, yy_eval).reshape(100, 100), r'$\kappa$'),
#     (np.sqrt(eta(true_params, xx_eval, yy_eval)).reshape(100, 100), r'$\eta$'),
#     (f_full(true_params, xx_eval, yy_eval, L).reshape(100, 100), r'$f$'),
# ]

# for i, (data, label) in enumerate(plot_data, start=1):
#     ax = plt.subplot(1, 4, i)
#     im = ax.contourf(data, levels=100, cmap='viridis')
#     ax.set_title(label, fontsize=18)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_aspect('equal')
#     # Create an axes on the right side of ax, make the colorbar broader.
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="16%", pad=0.1)
#     # Set the colorbar to display 5 ticks.
#     plt.colorbar(im, cax=cax, ticks=np.linspace(int(im.get_clim()[0]), int(im.get_clim()[1]), 5))

# plt.tight_layout()
# plt.savefig("src/ground.png")
# plt.show()


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
    low_res_N = 18
    rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(6), 3)
    amplitudes = jax.random.uniform(rng1, shape=(2,), minval=1, maxval=3)
    centers_x = jax.random.uniform(rng2, shape=(2,), minval=0, maxval=1)
    centers_y = jax.random.uniform(rng3, shape=(2,), minval=0, maxval=1)
    init_params = nnx.Param(jnp.array([
        amplitudes[0], centers_x[0], centers_y[0],
        amplitudes[1], centers_x[1], centers_y[1]
    ]))
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
    syn_opt = nnx.Optimizer(synthetic_model, optax.adam(1e-3))
    phys_opt = nnx.Optimizer(physical_model, optax.adam(5e-3))

    # Helper to vmap a scalar-call model.
    @nnx.jit
    def vmapped_model(m, xs, ys):
        return jax.vmap(lambda xx, yy: m(xx, yy))(xs, ys)

    @nnx.jit
    def train_step_hyb(model, model_other, optimizer, x, y, u, x_collocation, y_collocation, ld, lm):
        def loss_data(m):
            u_pred = vmapped_model(m, x, y)
            return jnp.mean(optax.squared_error(u_pred, u))
        
        def loss_hyb(m):
            u_pred = vmapped_model(m, x_collocation, y_collocation)
            u_pred_other = vmapped_model(model_other, x_collocation, y_collocation)
            return jnp.mean(optax.squared_error(u_pred, u_pred_other))
        
        def loss_fn(m):
            return  ld * loss_data(m) + lm * loss_hyb(m)

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
    u_real = u_true(xx_eval, yy_eval, L).reshape(-1, 1)
    loss_history_phys = np.zeros(epochs)
    loss_history_syn = np.zeros(epochs)
    param_history = np.zeros((epochs, 6))
    rng = jax.random.PRNGKey(6)
    n_collocation = 20
    loss_syn_data = 1
    loss_phy = 1
    ld_syn = 1
    lm_syn = 1
    ld_phy = 1
    lm_phy = 1
    for epoch in range(epochs):
        if epoch == 1500:
            n_collocation = 800
            ld_phy = 1e2
            lm_phy = 1e-2
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
                                    x_collocation, y_collocation, ld_syn, lm_syn)
            
            loss_phy, loss_phy_data = train_step_hyb(physical_model, synthetic_model, phys_opt,
                                xx_train, yy_train, u_train,
                                x_collocation, y_collocation, ld_phy, lm_phy)

            # Evaluation
            
            u_pred_syn = vmapped_model(synthetic_model, xx_eval, yy_eval).reshape(-1, 1)
            u_pred_phys = vmapped_model(physical_model, xx_eval, yy_eval).reshape(-1, 1)

            l2_syn = jnp.linalg.norm(u_pred_syn - u_real)/jnp.linalg.norm(u_real)
            l2_phys = jnp.linalg.norm(u_pred_phys - u_real)/jnp.linalg.norm(u_real)
            loss_history_syn[epoch] = l2_syn
            loss_history_phys[epoch] = l2_phys
            param_history[epoch] = physical_model.parameters.value
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss (physical): {loss_phy_data}, Loss (synthetic): {loss_syn}, Parameters: {physical_model.parameters.value}")

    return loss_history_phys, loss_history_syn, param_history, u_pred_phys, u_pred_syn

def train_fem(epochs):
    # -------------------------------------------------------------------------
    # Setup physical model.
    low_res_N = 18
    rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(6), 3)
    amplitudes = jax.random.uniform(rng1, shape=(2,), minval=1, maxval=3)
    centers_x = jax.random.uniform(rng2, shape=(2,), minval=0, maxval=1)
    centers_y = jax.random.uniform(rng3, shape=(2,), minval=0, maxval=1)
    init_params = nnx.Param(jnp.array([
        amplitudes[0], centers_x[0], centers_y[0],
        amplitudes[1], centers_x[1], centers_y[1]
    ]))
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
    # Initialize the optimizer.
    phys_opt = nnx.Optimizer(physical_model, optax.adam(5e-3))

    @nnx.jit
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
    u_real = u_true(xx_eval, yy_eval, L).reshape(-1, 1)
    loss_history = np.zeros(epochs)
    param_history = np.zeros((epochs, 6))
    for epoch in range(epochs):
        loss_phy = train_step(physical_model, phys_opt, xx_train, yy_train, u_train)
        # Evaluation
        u_pred = vmapped_model(physical_model, xx_eval, yy_eval).reshape(-1, 1)
        l2 = jnp.linalg.norm(u_pred - u_real)/jnp.linalg.norm(u_real)
        loss_history[epoch] = l2
        param_history[epoch] = physical_model.parameters.value
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss (physical): {loss_phy}, Parameters: {physical_model.parameters.value}")

    return loss_history, param_history, u_pred


def train_pinn(epochs):
    # Create the PINN model instance.
    rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(6), 3)
    amplitudes = jax.random.uniform(rng1, shape=(2,), minval=1, maxval=3)
    centers_x = jax.random.uniform(rng2, shape=(2,), minval=0, maxval=1)
    centers_y = jax.random.uniform(rng3, shape=(2,), minval=0, maxval=1)
    init_params = nnx.Param(jnp.array([
        amplitudes[0], centers_x[0], centers_y[0],
        amplitudes[1], centers_x[1], centers_y[1]
    ]))
    print(f"Initial parameters: {init_params}")
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
    @nnx.jit
    def vmapped_model(m, xs, ys):
        return jax.vmap(lambda xx, yy: m(xx, yy))(xs, ys)
    
    @nnx.jit
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
    loss_history = np.zeros(epochs)
    param_history = np.zeros((epochs, 6))

    # Create interior (collocation) points.
    u_real = u_true(xx_eval, yy_eval, L).reshape(-1, 1)
    n_interior = 400
    n_boundary = 400
    loss_pinn = 1
    rng = jax.random.PRNGKey(6)
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
            u_pred = vmapped_model(pinn, xx_eval, yy_eval).reshape(-1, 1)
            l2 = jnp.linalg.norm(u_pred - u_real)/jnp.linalg.norm(u_real)
            loss_history[epoch] = l2
            param_history[epoch] = pinn.parameters.value
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, PINN Loss: {loss_val}, PINN Parameters: {pinn.parameters.value}")

    print(f"Final PINN model parameters: {pinn.parameters}")

    return loss_history, param_history, u_pred


if __name__ == "__main__":
    for i in [50, 75, 100]:
        epochs = 3000

        if i == 50:
             subdomain=[(0, pi), (0, pi)]
        elif i == 75:
            subdomain=[(-pi/2, pi), (-pi/2, pi)]
        elif i == 100:
            subdomain=[(-pi, pi), (-pi, pi)]

        n_train=25
        rng_x, rng_y = jax.random.split(jax.random.PRNGKey(6))
        xx_train = jax.random.uniform(rng_x, shape=(n_train,), minval=subdomain[0][0], maxval=subdomain[0][1])
        yy_train = jax.random.uniform(rng_y, shape=(n_train,), minval=subdomain[1][0], maxval=subdomain[1][1])
        pts_train = jnp.stack([xx_train, yy_train], axis=-1)

        xx_eval = jnp.linspace(-pi, pi, 50)
        yy_eval = jnp.linspace(-pi, pi, 50)
        xx_eval, yy_eval = jnp.meshgrid(xx_eval, yy_eval)
        xx_eval = xx_eval.flatten()
        yy_eval = yy_eval.flatten()
        pts_eval = jnp.stack([xx_eval, yy_eval], axis=-1)
        # Use vmap over the new scalar __call__ for prediction.
        u_train = u_true(xx_train, yy_train, L).reshape(-1, 1)
        print(f"Training data generated with shape: {u_train.shape}")
            

        loss_history_hyb_phys, loss_history_hyb_syn, param_history_hyb, u_hyb_phys, u_hyb_syn = train_hybrid(epochs)
        loss_history_fem, param_history_fem, u_fem = train_fem(epochs)
        loss_history_pinn, param_history_pinn, u_pinn = train_pinn(epochs)

        


        # convert them to njp arrays
        loss_history_hyb_phys = jnp.array(loss_history_hyb_phys)
        loss_history_hyb_syn = jnp.array(loss_history_hyb_syn)
        loss_history_fem = jnp.array(loss_history_fem)
        loss_history_pinn = jnp.array(loss_history_pinn)
        param_history_hyb = jnp.array(param_history_hyb)
        param_history_fem = jnp.array(param_history_fem)
        param_history_pinn = jnp.array(param_history_pinn)
        u_hyb_phys = jnp.array(u_hyb_phys)
        u_hyb_syn = jnp.array(u_hyb_syn)
        u_fem = jnp.array(u_fem)
        u_pinn = jnp.array(u_pinn)

        with open(f"src/files/experiment_1/results_summary_{i}.txt", "w") as file:
            file.write("FEM Loss min in last 100: " + str(jnp.min(loss_history_fem[-100:])) + "\n")
            file.write("Hybrid Phys Loss min in last 100: " + str(jnp.min(loss_history_hyb_phys[-100:])) + "\n")
            file.write("Hybrid Syn Loss min in last 100: " + str(jnp.min(loss_history_hyb_syn[-100:])) + "\n")
            file.write("PINN Loss min in last 100: " + str(jnp.min(loss_history_pinn[-100:])) + "\n")


        # Save the results.
        jnp.save(f"src/files/experiment_1/hybrid_loss_phys_new_{i}.npy", loss_history_hyb_phys)
        jnp.save(f"src/files/experiment_1/hybrid_loss_syn_new_{i}.npy", loss_history_hyb_syn)
        jnp.save(f"src/files/experiment_1/hybrid_params_new_{i}.npy", param_history_hyb)
        jnp.save(f"src/files/experiment_1/fem_loss_new_{i}.npy", loss_history_fem)
        jnp.save(f"src/files/experiment_1/fem_params_new_{i}.npy", param_history_fem)
        jnp.save(f"src/files/experiment_1/pinn_loss_new_{i}.npy", loss_history_pinn)
        jnp.save(f"src/files/experiment_1/pinn_params_new_{i}.npy", param_history_pinn)
        jnp.save(f"src/files/experiment_1/u_hyb_phys_new_{i}.npy", u_hyb_phys)
        jnp.save(f"src/files/experiment_1/u_hyb_syn_new_{i}.npy", u_hyb_syn)
        jnp.save(f"src/files/experiment_1/u_fem_new_{i}.npy", u_fem)
        jnp.save(f"src/files/experiment_1/u_pinn_new_{i}.npy", u_pinn)

# Plot the results. 