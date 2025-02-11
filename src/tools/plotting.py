# filepath: /c:/Users/thyss/Documents/Work/Projects/HybridModelling/UnificationTests/StaticHybridModelling/src/tools/plotting.py
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def plot(state_phys, state_hyb, phys_loss_hist, hyb_phys_loss_hist, true_params, pts_train, kappa, eta, filename):
    x = jnp.linspace(-3.0, 3.0, 100)
    y = jnp.linspace(-3.0, 3.0, 100)
    xx, yy = jnp.meshgrid(x, y)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    # Compute predictions for physics and hybrid, and compute true functions.
    kappa_phys = kappa(state_phys.params, xx_flat, yy_flat)
    eta_phys = eta(state_phys.params, xx_flat, yy_flat)
    kappa_hyb = kappa(state_hyb.params, xx_flat, yy_flat)
    eta_hyb = eta(state_hyb.params, xx_flat, yy_flat)
    real_kappa = kappa(true_params, xx_flat, yy_flat)
    real_eta = eta(true_params, xx_flat, yy_flat)

    # Compute the global min/max true
    vmin = jnp.floor(jnp.min(jnp.array([real_kappa,real_eta])))
    vmax = jnp.ceil(jnp.max(jnp.array([real_kappa,real_eta])))

    fig = plt.figure(figsize=(10, 10))
    gs_top = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.2],
                              left=0.1, right=0.9, top=0.93, bottom=0.43,
                              wspace=0.1, hspace=0.1)

    # Row 1: Kappa plots.
    ax0 = fig.add_subplot(gs_top[0, 0])
    cf0 = ax0.contourf(xx, yy, kappa_phys.reshape(xx.shape), levels=100, vmin=vmin, vmax=vmax)
    ax0.set_title("Kappa Physics")
    ax0.set_aspect('equal')
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs_top[0, 1])
    cf1 = ax1.contourf(xx, yy, kappa_hyb.reshape(xx.shape), levels=100, vmin=vmin, vmax=vmax)
    ax1.set_title("Kappa Hybrid")
    ax1.set_aspect('equal')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs_top[0, 2])
    cf2 = ax2.contourf(xx, yy, real_kappa.reshape(xx.shape), levels=100, vmin=vmin, vmax=vmax)
    ax2.set_title("Kappa True")
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Row 2: Eta plots.
    ax3 = fig.add_subplot(gs_top[1, 0])
    cf3 = ax3.contourf(xx, yy, eta_phys.reshape(xx.shape), levels=100, vmin=vmin, vmax=vmax)
    ax3.set_title("Eta Physics")
    ax3.set_aspect('equal')
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4 = fig.add_subplot(gs_top[1, 1])
    cf4 = ax4.contourf(xx, yy, eta_hyb.reshape(xx.shape), levels=100, vmin=vmin, vmax=vmax)
    ax4.set_title("Eta Hybrid")
    ax4.set_aspect('equal')
    ax4.set_xticks([])
    ax4.set_yticks([])

    ax5 = fig.add_subplot(gs_top[1, 2])
    cf5 = ax5.contourf(xx, yy, real_eta.reshape(xx.shape), levels=100, vmin=vmin, vmax=vmax)
    ax5.set_title("Eta True")
    ax5.set_aspect('equal')
    ax5.set_xticks([])
    ax5.set_yticks([])

    # Scatter training data on subplots.
    for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
        ax.scatter(pts_train[:, 0], pts_train[:, 1], marker='o',
                   c='black', s=20, edgecolors='white')

    # Unified colorbar.
    ax_cb = fig.add_subplot(gs_top[:, 3])
    norm = Normalize(vmin=vmin, vmax=vmax)
    mappable = ScalarMappable(norm=norm, cmap=cf2.cmap)
    mappable.set_array([])
    cb = fig.colorbar(mappable, cax=ax_cb)
    cb.set_label("Parameter Value")

    # Plot the loss below.
    ax_loss = fig.add_axes([0.1, 0.07, 0.8, 0.3])
    ax_loss.plot(phys_loss_hist, label="Physics Loss")
    ax_loss.plot(hyb_phys_loss_hist, label="Hybrid Physics Loss", linestyle="--")
    ax_loss.set_yscale('log')
    ax_loss.set_title("Loss History")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    plt.savefig(f"src/results/{filename}")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def animate_training_evolution(
    state_history,
    true_params,
    kappa_func,
    eta_func,
    pts_train,
    domain=(-3.0, 3.0),
    N=100,
    filename="phy_gaussian_evolution.gif"
):
    """
    Creates an animation showing how the kappa and eta fields 
    evolve over the training epochs in state_history.
    """
    x = jnp.linspace(domain[0], domain[1], N)
    y = jnp.linspace(domain[0], domain[1], N)
    xx, yy = jnp.meshgrid(x, y)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    # Compute the true kappa/eta for reference.
    real_kappa = kappa_func(true_params, xx_flat, yy_flat)
    real_eta = eta_func(true_params, xx_flat, yy_flat)

    # Collect min/max across all states.
    all_kappa_vals = []
    all_eta_vals = []

    for st in state_history:
        kappa_vals = kappa_func(st, xx_flat, yy_flat)
        eta_vals = eta_func(st, xx_flat, yy_flat)
        all_kappa_vals.append(kappa_vals)
        all_eta_vals.append(eta_vals)

    all_kappa_vals.append(real_kappa)
    all_eta_vals.append(real_eta)

    all_kappa_vals = jnp.stack(all_kappa_vals)
    all_eta_vals = jnp.stack(all_eta_vals)

    vmin = float(jnp.minimum(all_kappa_vals.min(), all_eta_vals.min()))
    vmax = float(jnp.maximum(all_kappa_vals.max(), all_eta_vals.max()))

    fig, (ax_kappa, ax_eta) = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.3)

    # Create initial contour plots with placeholder data.
    norm = Normalize(vmin=vmin, vmax=vmax)
    kappa_mappable = ScalarMappable(norm=norm, cmap="viridis")
    eta_mappable = ScalarMappable(norm=norm, cmap="viridis")

    ax_kappa.contourf(xx, yy, jnp.zeros_like(xx), levels=100, cmap="viridis", vmin=vmin, vmax=vmax)
    ax_eta.contourf(xx, yy, jnp.zeros_like(xx), levels=100, cmap="viridis", vmin=vmin, vmax=vmax)

    ax_kappa.set_title("Kappa Evolution")
    ax_eta.set_title("Eta Evolution")
    ax_kappa.set_aspect("equal")
    ax_eta.set_aspect("equal")

    ax_kappa.scatter(pts_train[:, 0], pts_train[:, 1], marker='o', c='white', s=25, edgecolors='black')
    ax_eta.scatter(pts_train[:, 0], pts_train[:, 1], marker='o', c='white', s=25, edgecolors='black')

    # Colorbars.
    cb_kappa = fig.colorbar(kappa_mappable, ax=ax_kappa, fraction=0.05)
    cb_eta = fig.colorbar(eta_mappable, ax=ax_eta, fraction=0.05)
    cb_kappa.set_label("Kappa Value")
    cb_eta.set_label("Eta Value")

    def init():
        # No special initialization needed; return empty (blit=False).
        return []

    def update(frame):
        # Clear plots for the new frame.
        ax_kappa.clear()
        ax_eta.clear()

        # Pull the parameters for current frame.
        st = state_history[frame]
        kappa_vals = kappa_func(st, xx_flat, yy_flat).reshape(xx.shape)
        eta_vals = eta_func(st, xx_flat, yy_flat).reshape(xx.shape)

        # Redraw the contourf plots.
        ax_kappa.contourf(xx, yy, kappa_vals, levels=100, cmap="viridis", vmin=vmin, vmax=vmax)
        ax_eta.contourf(xx, yy, eta_vals, levels=100, cmap="viridis", vmin=vmin, vmax=vmax)

        ax_kappa.scatter(pts_train[:, 0], pts_train[:, 1], marker='o', c='white', s=25, edgecolors='black')
        ax_eta.scatter(pts_train[:, 0], pts_train[:, 1], marker='o', c='white', s=25, edgecolors='black')

        ax_kappa.set_title(f"Kappa at Epoch {frame}")
        ax_eta.set_title(f"Eta at Epoch {frame}")
        ax_kappa.set_aspect("equal")
        ax_eta.set_aspect("equal")

        # Returning an empty list is fine with blit=False.
        return []

    # Build the animation. The interval is ms between frames.
    anim = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=len(state_history),
        init_func=init,
        blit=False,
        interval=200
    )

    # If ffmpeg is unavailable, Pillow (the default) will be used.  
    # Remove the 'writer="ffmpeg"' argument to avoid the "MovieWriter ffmpeg unavailable" message.
    anim.save(filename, dpi=120)
    plt.close(fig)

def animate_both_evolution(
    phy_state_history,
    hyb_state_history,
    true_params,
    kappa_func,
    eta_func,
    pts_train,
    domain=(-3.0, 3.0),
    N=100,
    filename="both_gaussian_evolution.gif"
):
    """
    Animates how Kappa and Eta evolve side-by-side for both physics-only and
    hybrid models throughout training. Each frame shows:
      - Kappa(physics)   | Kappa(hybrid)
      - Eta(physics)     | Eta(hybrid)
    """
    # Build a shared grid of points.
    x = jnp.linspace(domain[0], domain[1], N)
    y = jnp.linspace(domain[0], domain[1], N)
    xx, yy = jnp.meshgrid(x, y)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    # Collect all Kappa/Eta values for min/max across all frames.
    all_kappa_phy = []
    all_kappa_hyb = []
    all_eta_phy = []
    all_eta_hyb = []

    for sp, sh in zip(phy_state_history, hyb_state_history):
        kp_phy = kappa_func(sp, xx_flat, yy_flat)
        kp_hyb = kappa_func(sh, xx_flat, yy_flat)
        et_phy = eta_func(sp, xx_flat, yy_flat)
        et_hyb = eta_func(sh, xx_flat, yy_flat)
        all_kappa_phy.append(kp_phy)
        all_kappa_hyb.append(kp_hyb)
        all_eta_phy.append(et_phy)
        all_eta_hyb.append(et_hyb)

    # Convert to JAX arrays.
    all_kappa_phy = jnp.stack(all_kappa_phy)
    all_kappa_hyb = jnp.stack(all_kappa_hyb)
    all_eta_phy   = jnp.stack(all_eta_phy)
    all_eta_hyb   = jnp.stack(all_eta_hyb)

    # Compute global min/max for Kappa and Eta separately.
    kappa_min = float(jnp.minimum(all_kappa_phy.min(), all_kappa_hyb.min()))
    kappa_max = float(jnp.maximum(all_kappa_phy.max(), all_kappa_hyb.max()))
    eta_min   = float(jnp.minimum(all_eta_phy.min(), all_eta_hyb.min()))
    eta_max   = float(jnp.maximum(all_eta_phy.max(), all_eta_hyb.max()))

    # Prepare figure with 2x2 subplots.
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_kPhy, ax_kHyb = axes[0]
    ax_ePhy, ax_eHyb = axes[1]

    # Shared color norm/mappable for Kappa and Eta.
    kappa_norm = Normalize(vmin=kappa_min, vmax=kappa_max)
    eta_norm   = Normalize(vmin=eta_min, vmax=eta_max)
    kappa_cm   = ScalarMappable(norm=kappa_norm, cmap="viridis")
    eta_cm     = ScalarMappable(norm=eta_norm, cmap="viridis")

    # Initial placeholders for each subplot (just blank contourfs).
    ax_kPhy.contourf(xx, yy, jnp.zeros_like(xx), levels=100, cmap="viridis",
                     vmin=kappa_min, vmax=kappa_max)
    ax_kHyb.contourf(xx, yy, jnp.zeros_like(xx), levels=100, cmap="viridis",
                     vmin=kappa_min, vmax=kappa_max)
    ax_ePhy.contourf(xx, yy, jnp.zeros_like(xx), levels=100, cmap="viridis",
                     vmin=eta_min, vmax=eta_max)
    ax_eHyb.contourf(xx, yy, jnp.zeros_like(xx), levels=100, cmap="viridis",
                     vmin=eta_min, vmax=eta_max)

    # Titles / formatting.
    ax_kPhy.set_title("Kappa (Physics)")
    ax_kHyb.set_title("Kappa (Hybrid)")
    ax_ePhy.set_title("Eta (Physics)")
    ax_eHyb.set_title("Eta (Hybrid)")

    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(pts_train[:,0], pts_train[:,1], marker='o', c='white', s=25, edgecolors='black')

    # Colorbars for each row (Kappa row uses same cbar, Eta row uses same cbar).
    cbar_kappa = fig.colorbar(kappa_cm, ax=[ax_kPhy, ax_kHyb], fraction=0.045)
    cbar_eta   = fig.colorbar(eta_cm, ax=[ax_ePhy, ax_eHyb], fraction=0.045)
    cbar_kappa.set_label("Kappa Value")
    cbar_eta.set_label("Eta Value")

    # Animation update function.
    def update(frame):
        # Clear each subplot.
        for ax in axes.flat:
            ax.clear()
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

        # Plotting Kappa for Physics and Hybrid.
        kp_phy = all_kappa_phy[frame].reshape(xx.shape)
        kp_hyb = all_kappa_hyb[frame].reshape(xx.shape)
        ax_kPhy.contourf(xx, yy, kp_phy, levels=100, cmap="viridis",
                         vmin=kappa_min, vmax=kappa_max)
        ax_kHyb.contourf(xx, yy, kp_hyb, levels=100, cmap="viridis",
                         vmin=kappa_min, vmax=kappa_max)
        ax_kPhy.scatter(pts_train[:,0], pts_train[:,1], marker='o', c='white', s=25, edgecolors='black')
        ax_kHyb.scatter(pts_train[:,0], pts_train[:,1], marker='o', c='white', s=25, edgecolors='black')
        ax_kPhy.set_title(f"Kappa (Physics) Epoch {frame}")
        ax_kHyb.set_title(f"Kappa (Hybrid)  Epoch {frame}")

        # Plotting Eta for Physics and Hybrid.
        et_phy = all_eta_phy[frame].reshape(xx.shape)
        et_hyb = all_eta_hyb[frame].reshape(xx.shape)
        ax_ePhy.contourf(xx, yy, et_phy, levels=100, cmap="viridis",
                         vmin=eta_min, vmax=eta_max)
        ax_eHyb.contourf(xx, yy, et_hyb, levels=100, cmap="viridis",
                         vmin=eta_min, vmax=eta_max)
        ax_ePhy.scatter(pts_train[:,0], pts_train[:,1], marker='o', c='white', s=25, edgecolors='black')
        ax_eHyb.scatter(pts_train[:,0], pts_train[:,1], marker='o', c='white', s=25, edgecolors='black')
        ax_ePhy.set_title(f"Eta (Physics) Epoch {frame}")
        ax_eHyb.set_title(f"Eta (Hybrid)  Epoch {frame}")

        return []

    anim = animation.FuncAnimation(
        fig, update, frames=min(len(phy_state_history), len(hyb_state_history)),
        interval=300, blit=False
    )

    # If ffmpeg is unavailable, Pillow is used. You can remove writer="ffmpeg" if needed.
    anim.save(filename, dpi=120)
    plt.close(fig)