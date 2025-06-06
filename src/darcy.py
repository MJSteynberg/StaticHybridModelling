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

def kappa(parameters, x, y): 
    # Assume parameters has shape n^2 + 1 
    mu0 = parameters[0]
    coeffs = parameters[1:].reshape(3, 3)
    sin_sum = 0.0
    for m in range(3):
        for p in range(3):
            sin_sum += coeffs[m, p] * jnp.sin(jnp.pi * (m + 1) * x/4) * jnp.sin(jnp.pi * (p + 1) * y/4)
    return mu0**2 + jax.nn.softplus(sin_sum)

def eta(parameters, x, y):
    return jnp.zeros_like(x)

def f_full(parameters, x, y, L):
    return 1

def f(x, y):
    # This is a placeholder for the forcing function.
    # It should be defined based on the problem requirements.
    return f_full(true_params, x, y, L)

# -----------------------------------------------------------------------------  
# Generate the “true solution” using a high resolution physical model.
L = 2 * pi  # Domain size.
rng_key = jax.random.PRNGKey(5)
true_params = jax.random.uniform(rng_key, shape=(10,), minval=-1, maxval=1)
print(f"True parameters: {true_params}")
true_model = PhysicalModel(
    domain=(-L/2, L/2),
    N=100,  # High resolution for true solution.
    parameters=true_params,
    training=False,
    forcing_func=f,
    kappa_func=kappa,
    eta_func=eta,
    rngs=nnx.Rngs(0)
)

# Generate a grid for evaluation.
xx_eval = jnp.linspace(-L/2, L/2, 50)
yy_eval = jnp.linspace(-L/2, L/2, 50)
xx_eval, yy_eval = jnp.meshgrid(xx_eval, yy_eval)
xx_eval = xx_eval.flatten()
yy_eval = yy_eval.flatten()

#plot kappa 
kappa_values = jax.vmap(lambda x, y: kappa(true_params, x, y))(xx_eval, yy_eval).reshape(-1, 1)

# Evaluate the true solution on the grid.
u_true = jax.vmap(lambda x, y: true_model(x, y))(xx_eval, yy_eval).reshape(-1, 1)
# Plot the true solution.

domain = (-L/2, L/2)
def f(x, y):
    return f_full(true_params, x, y, L)

def train_hybrid(epochs, weight):
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
    init_params = nnx.Param(jax.random.uniform(rng3, shape=(10,), minval=-1, maxval=1))
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
    u_real = u_true
    loss_history_phys = np.zeros(epochs)
    loss_history_syn = np.zeros(epochs)
    param_history = np.zeros((epochs, 10))
    rng = jax.random.PRNGKey(6)
    n_collocation = 20
    loss_syn_data = 1
    loss_phy = 1
    ld_syn = weight[0]
    lm_syn = 1
    ld_phy, lm_phy = weight
    for epoch in range(epochs):
        if epoch == 1500:
            n_collocation = 800
            ld_syn = 1e-1
            lm_syn = 1e2
        if loss_syn_data > 5e-1 * max(ld_syn, lm_syn):
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
    init_params = nnx.Param(jax.random.uniform(rng3, shape=(10,), minval=-1, maxval=1))
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
    u_real = u_true
    loss_history = np.zeros(epochs)
    param_history = np.zeros((epochs, 10))
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
    init_params = nnx.Param(jax.random.uniform(rng3, shape=(10,), minval=-1, maxval=1))
    print(f"Initial parameters: {init_params}")
    model = ResNetSynthetic(
        hidden_dims=(256, 256), 
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
    param_history = np.zeros((epochs, 10))

    # Create interior (collocation) points.
    u_real = u_true
    n_interior = 400
    n_boundary = 400
    loss_pinn = 1
    rng = jax.random.PRNGKey(6)
    for epoch in range(epochs):
        if loss_pinn > 1:
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
        errors = [0.25]
        weights = [(0.8, 1.2)]
        for error, weight in zip(errors, weights):
            epochs = 3000
            n_train=50
            subdomain = ((-pi, pi), (-pi, pi))  # Define the subdomain for training.
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
            
            u_train = jax.vmap(lambda x, y: true_model(x, y))(xx_train, yy_train).reshape(-1, 1)
            u_train += jax.random.normal(jax.random.PRNGKey(42), shape=u_train.shape) * error * jnp.max(u_train.flatten())  # Add noise to the training data.
            print(f"Training data generated with shape: {u_train.shape}")
                

            # loss_history_hyb_phys, loss_history_hyb_syn, param_history_hyb, u_hyb_phys, u_hyb_syn = train_hybrid(epochs, weight=weight)
            # loss_history_fem, param_history_fem, u_fem = train_fem(epochs)
            # loss_history_pinn, param_history_pinn, u_pinn = train_pinn(epochs)


            # # convert them to njp arrays
            # loss_history_hyb_phys = jnp.array(loss_history_hyb_phys)
            # loss_history_hyb_syn = jnp.array(loss_history_hyb_syn)
            # loss_history_fem = jnp.array(loss_history_fem)
            # loss_history_pinn = jnp.array(loss_history_pinn)
            # param_history_hyb = jnp.array(param_history_hyb)
            # param_history_fem = jnp.array(param_history_fem)
            # param_history_pinn = jnp.array(param_history_pinn)
            # u_hyb_phys = jnp.array(u_hyb_phys)
            # u_hyb_syn = jnp.array(u_hyb_syn)
            # u_fem = jnp.array(u_fem)
            # u_pinn = jnp.array(u_pinn)

            # with open(f"src/files/darcy/results_{error}.txt", "w") as file:
            #     file.write("FEM Loss min in last 100: " + str(jnp.min(loss_history_fem[-100:])) + "\n")
            #     file.write("Hybrid Phys Loss min in last 100: " + str(jnp.min(loss_history_hyb_phys[-100:])) + "\n")
            #     file.write("Hybrid Syn Loss min in last 100: " + str(jnp.min(loss_history_hyb_syn[-100:])) + "\n")
            #     file.write("PINN Loss min in last 100: " + str(jnp.min(loss_history_pinn[-100:])) + "\n")
            #     file.write("FEM Loss final: " + str(loss_history_fem[-1]) + "\n")
            #     file.write("Hybrid Phys Loss final: " + str(loss_history_hyb_phys[-1]) + "\n")
            #     file.write("Hybrid Syn Loss final: " + str(loss_history_hyb_syn[-1]) + "\n")
            #     file.write("PINN Loss final: " + str(loss_history_pinn[-1]) + "\n")
 

            # # Save the results.
            # jnp.save(f"src/files/darcy/hybrid_loss_phys_{error}.npy", loss_history_hyb_phys)
            # jnp.save(f"src/files/darcy/hybrid_loss_syn_{error}.npy", loss_history_hyb_syn)
            # jnp.save(f"src/files/darcy/hybrid_params_{error}.npy", param_history_hyb)
            # jnp.save(f"src/files/darcy/fem_loss_{error}.npy", loss_history_fem)
            # jnp.save(f"src/files/darcy/fem_params_{error}.npy", param_history_fem)
            # jnp.save(f"src/files/darcy/pinn_loss_{error}.npy", loss_history_pinn)
            # jnp.save(f"src/files/darcy/pinn_params_{error}.npy", param_history_pinn)
            # jnp.save(f"src/files/darcy/u_hyb_phys_{error}.npy", u_hyb_phys)
            # jnp.save(f"src/files/darcy/u_hyb_syn_{error}.npy", u_hyb_syn)
            # jnp.save(f"src/files/darcy/u_fem_{error}.npy", u_fem)
            # jnp.save(f"src/files/darcy/u_pinn_{error}.npy", u_pinn)

            loss_history_hyb_syn = np.load(f"src/files/darcy/hybrid_loss_syn_{error}.npy")
            loss_history_hyb_phys = np.load(f"src/files/darcy/hybrid_loss_phys_{error}.npy")
            loss_history_fem = np.load(f"src/files/darcy/fem_loss_{error}.npy")
            loss_history_pinn = np.load(f"src/files/darcy/pinn_loss_{error}.npy")
            param_history_hyb = np.load(f"src/files/darcy/hybrid_params_{error}.npy")
            param_history_fem = np.load(f"src/files/darcy/fem_params_{error}.npy")
            param_history_pinn = np.load(f"src/files/darcy/pinn_params_{error}.npy")
            u_hyb_phys = np.load(f"src/files/darcy/u_hyb_phys_{error}.npy")
            u_hyb_syn = np.load(f"src/files/darcy/u_hyb_syn_{error}.npy")
            u_fem = np.load(f"src/files/darcy/u_fem_{error}.npy")
            u_pinn = np.load(f"src/files/darcy/u_pinn_{error}.npy")



            def replace_zeros_linear(arr):
                """
                Replace zeros in a 1D numpy array with linear interpolation based on nonzero values.
                """
                indices = np.arange(len(arr))
                mask = arr != 0
                # If no nonzero values exist, return array as is.
                if mask.sum() == 0:
                    return arr
                # np.interp will extrapolate constant for points outside interpolation range.
                return np.interp(indices, indices[mask], arr[mask])
            def replace_zeros_nearest(arr):
                """
                Replace zeros in a nD numpy array with the nearest nonzero value in axis 0.
                """
                # Get the indices of nonzero elements.
                nonzero_indices = np.nonzero(arr)[0]
                # If no nonzero values exist, return array as is.
                if len(nonzero_indices) == 0:
                    return arr
                # Get the indices of zero elements.
                zero_indices = np.where(arr == 0)[0]
                # Create a copy of the array to modify.
                arr_copy = arr.copy()
                # Replace zeros with the nearest nonzero value.
                for zero_index in zero_indices:
                    # Find the nearest nonzero index.
                    nearest_index = nonzero_indices[np.abs(nonzero_indices - zero_index).argmin()]
                    arr_copy[zero_index] = arr[nearest_index]
                return arr_copy
            
            loss_history_hyb_phys = replace_zeros_linear(loss_history_hyb_phys)
            loss_history_hyb_syn = replace_zeros_linear(loss_history_hyb_syn)
            loss_history_fem = replace_zeros_linear(loss_history_fem)
            loss_history_pinn = replace_zeros_linear(loss_history_pinn)
            param_history_hyb = replace_zeros_nearest(param_history_hyb)
            param_history_fem = replace_zeros_nearest(param_history_fem)
            param_history_pinn = replace_zeros_nearest(param_history_pinn)
            # Plot the results.
            plot(
            param_history_fem,
            param_history_hyb,
            param_history_pinn,      # New argument for PINN training results.
            true_params,
            loss_history_fem,
            loss_history_hyb_phys,
            loss_history_pinn,          # New loss history for PINN.
            kappa,
            None,
            pts_train,
            domain=(-pi, pi),
            N=100,
            hyb_synth_loss_hist=loss_history_hyb_syn,
            u_hyb_phys=u_hyb_phys,
            u_hyb_syn=u_hyb_syn,
            u_fem=u_fem,
            u_pinn=u_pinn,
            u_true=u_true,
            filename=f"darcy/darcy_{error}"
            )