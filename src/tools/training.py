
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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