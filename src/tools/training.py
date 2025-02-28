
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

import optax
from flax.training import train_state

class TrainStatePhy(train_state.TrainState):
    extra_state: dict

class TrainStateSyn(train_state.TrainState):
    pass

def hybrid_loss(domain, n_hyb, model_phy, model_syn, params_phy, params_syn, extra_state, rng):
    # Total number of random points
    n_pts = n_hyb * n_hyb
    # Sample random points from a uniform distribution over the domain.
    pts_hyb = jax.random.uniform(
        rng, 
        shape=(n_pts, 2), 
        minval=domain[0], 
        maxval=domain[1]
    )
    
    u_pred_phys, new_state_phys = model_phy.apply(
        extra_state, 
        pts_hyb[:, 0], 
        pts_hyb[:, 1], 
        mutable=["cache", "state"]
    )
    
    u_pred_syn = model_syn.apply(
        params_syn, 
        pts_hyb[:, 0], 
        pts_hyb[:, 1]
    )
    
    loss = jnp.mean((u_pred_phys - u_pred_syn) ** 2)
    
    return loss, new_state_phys

def loss_fn(model_phys, model_syn, params_phys, params_syn, extra_state, x, y, u_target, rng):
    u_pred_phys, new_state = model_phys.apply(extra_state, x, y, mutable=["cache", "state"])
    loss_phys = jnp.mean((u_pred_phys - u_target) ** 2)
    
    u_pred_syn = model_syn.apply(params_syn, x, y)
    loss_syn = jnp.mean((u_pred_syn - u_target) ** 2)

    loss_hyb, new_state = hybrid_loss((0.0, 1.0), 50, model_phys, model_syn, params_phys, params_syn, extra_state, rng)
    
    return loss_phys, loss_syn, loss_hyb, new_state

def generate_data(model, params, subdomain, randomized, n_train):
    if randomized:
        rng_x, rng_y = jax.random.split(jax.random.PRNGKey(0))
        x_train = jax.random.uniform(rng_x, shape=(n_train,), minval=subdomain[0][0], maxval=subdomain[0][1])
        y_train = jax.random.uniform(rng_y, shape=(n_train,), minval=subdomain[1][0], maxval=subdomain[1][1])
        pts_train = jnp.stack([x_train, y_train], axis=-1)
        var_true = model.init(jax.random.PRNGKey(0), pts_train[:, 0], pts_train[:, 1], mutable=["cache", "state"])
        u_train = model.apply(var_true, pts_train[:, 0], pts_train[:, 1], mutable=["cache", "state"])[0]
        return pts_train, u_train
        
    else:
        n_train = int(jnp.sqrt(n_train))
        x_train = jnp.linspace(subdomain[0][0], subdomain[0][1], n_train)
        y_train = jnp.linspace(subdomain[1][0], subdomain[1][1], n_train)
        xx_train, yy_train = jnp.meshgrid(x_train, y_train)
        pts_train = jnp.stack([xx_train.flatten(), yy_train.flatten()], axis=-1)
        var_true = model.init(jax.random.PRNGKey(0), pts_train[:, 0], pts_train[:, 1], mutable=["cache", "state"])
        u_train = model.apply(var_true, pts_train[:, 0], pts_train[:, 1], mutable=["cache", "state"])[0]
        return pts_train, u_train

@partial(jax.jit, static_argnames=("loss",))
def train_step_phys(state_phys, state_syn, x, y, u_target, loss):
    (loss_val, (new_state, metrics)), grads = jax.value_and_grad(
        lambda p, es: loss(p, state_syn.params, es, x, y, u_target, "phys"),
        has_aux=True
    )(state_phys.params, state_phys.extra_state)
    state_phys = state_phys.apply_gradients(grads=grads, extra_state=new_state)
    return state_phys, loss_val, metrics
    
@partial(jax.jit, static_argnames=("loss",))
def train_step_phys_only(state_phys, x, y, u_target, loss):
    (loss_val, (new_state, metrics)), grads = jax.value_and_grad(
        lambda p, es: loss(p, es, x, y, u_target, "phys"),
        has_aux=True
    )(state_phys.params, state_phys.extra_state)
    state_phys = state_phys.apply_gradients(grads=grads, extra_state=new_state)
    return state_phys, loss_val, metrics

# JIT compiled synthetic training step.

@partial(jax.jit, static_argnames=("loss",))
def train_step_syn(state_phys, state_syn, x, y, u_target, loss):
    (loss_val, (new_state, metrics)), grads = jax.value_and_grad(
        lambda s: loss(state_phys.params, s, state_phys.extra_state, x, y, u_target, "syn"),
        has_aux=True
    )(state_syn.params)
    state_syn = state_syn.apply_gradients(grads=grads)
    return state_syn, loss_val

def pde_loss(domain, randomized, n_coll, residual, boundary, state_phys, state_syn, model_syn):
    if randomized:
        rng_x, rng_y = jax.random.split(jax.random.PRNGKey(0))
        x_coll = jax.random.uniform(rng_x, shape=(n_coll,), minval=domain[0], maxval=domain[1])
        y_coll = jax.random.uniform(rng_y, shape=(n_coll,), minval=domain[0], maxval=domain[1])
        pts_coll = jnp.stack([xx_coll.flatten(), yy_coll.flatten()], axis=-1)
    else:
        n_coll = int(jnp.sqrt(n_coll))
        x_coll = jnp.linspace(domain[0], domain[1], n_coll)
        y_coll = jnp.linspace(domain[0], domain[1], n_coll)
        xx_coll, yy_coll = jnp.meshgrid(x_coll, y_coll)
        pts_coll = jnp.stack([xx_coll.flatten(), yy_coll.flatten()], axis=-1)
    
    @partial(jax.jit, static_argnames=("model_syn", "residual", "boundary"))
    def compute_loss_for_point(pt, state_phys, state_syn, model_syn, residual, boundary):
        return jax.lax.cond(
            boundary(pt),
            lambda: (model_syn.apply(state_syn.params, pt[0], pt[1])**2),
            lambda: (residual(pt, state_phys, state_syn, model_syn)**2)
        )

    return jnp.mean(jax.vmap(lambda pt: compute_loss_for_point(pt, state_phys, state_syn, model_syn, residual, boundary))(pts_coll))