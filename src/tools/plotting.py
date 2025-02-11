# filepath: /c:/Users/thyss/Documents/Work/Projects/HybridModelling/UnificationTests/StaticHybridModelling/src/tools/plotting.py
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.animation as animation
def plot(
    phy_state_history,
    hyb_state_history,
    true_params,
    phys_loss_hist,
    hyb_phys_loss_hist,
    kappa_func,
    eta_func,
    pts_train,
    domain=(-3.0, 3.0),
    N=100,
    filename="final_evolution.png"
):
    """
    Creates a static plot of the final predictions and loss history using the same
    inputs as animate_evolution. It uses the last epoch from the state histories.
    """

    # Extract final states.
    final_phy = phy_state_history[-1]
    final_hyb = hyb_state_history[-1]

    # Build a shared grid.
    x = jnp.linspace(domain[0], domain[1], N)
    y = jnp.linspace(domain[0], domain[1], N)
    xx, yy = jnp.meshgrid(x, y)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    # Compute predictions.
    kappa_phy = kappa_func(final_phy, xx_flat, yy_flat)
    eta_phy   = eta_func(final_phy, xx_flat, yy_flat)
    kappa_hyb = kappa_func(final_hyb, xx_flat, yy_flat)
    eta_hyb   = eta_func(final_hyb, xx_flat, yy_flat)
    real_kappa = kappa_func(true_params, xx_flat, yy_flat)
    real_eta   = eta_func(true_params, xx_flat, yy_flat)

    # Compute the global min/max from the true predictions.
    vmin = float(jnp.floor(jnp.min(jnp.array([real_kappa, real_eta]))))
    vmax = float(jnp.ceil(jnp.max(jnp.array([real_kappa, real_eta]))))

    # Create figure layout.
    fig = plt.figure(figsize=(10, 10))
    gs_top = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.2],
                              left=0.1, right=0.9, top=0.93, bottom=0.43,
                              wspace=0.1, hspace=0.1)

    # Row 1: Kappa plots.
    ax0 = fig.add_subplot(gs_top[0, 0])
    cf0 = ax0.contourf(xx, yy, kappa_phy.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax0.set_title("Kappa Physics (Final)")
    ax0.set_aspect("equal")
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs_top[0, 1])
    cf1 = ax1.contourf(xx, yy, kappa_hyb.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax1.set_title("Kappa Hybrid (Final)")
    ax1.set_aspect("equal")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs_top[0, 2])
    cf2 = ax2.contourf(xx, yy, real_kappa.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax2.set_title("Kappa True")
    ax2.set_aspect("equal")
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Row 2: Eta plots.
    ax3 = fig.add_subplot(gs_top[1, 0])
    cf3 = ax3.contourf(xx, yy, eta_phy.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax3.set_title("Eta Physics (Final)")
    ax3.set_aspect("equal")
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4 = fig.add_subplot(gs_top[1, 1])
    cf4 = ax4.contourf(xx, yy, eta_hyb.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax4.set_title("Eta Hybrid (Final)")
    ax4.set_aspect("equal")
    ax4.set_xticks([])
    ax4.set_yticks([])

    ax5 = fig.add_subplot(gs_top[1, 2])
    cf5 = ax5.contourf(xx, yy, real_eta.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax5.set_title("Eta True")
    ax5.set_aspect("equal")
    ax5.set_xticks([])
    ax5.set_yticks([])

    # Scatter training data on all subplots.
    for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
        ax.scatter(pts_train[:, 0], pts_train[:, 1], marker="o",
                   c="black", s=20, edgecolors="white")

    # Unified colorbar.
    ax_cb = fig.add_subplot(gs_top[:, 3])
    norm = Normalize(vmin=vmin, vmax=vmax)
    mappable = ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array([])
    cb = fig.colorbar(mappable, cax=ax_cb)
    cb.set_label("Parameter Value")

    # Plot the loss history.
    ax_loss = fig.add_axes([0.1, 0.07, 0.8, 0.3])
    ax_loss.plot(phys_loss_hist, label="Physics Loss")
    ax_loss.plot(hyb_phys_loss_hist, label="Hybrid Physics Loss", linestyle="--")
    ax_loss.set_yscale("log")
    ax_loss.set_title("Loss History")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    plt.savefig(f"src/results/{filename}")
    plt.close(fig)


def animate(
    phy_state_history,
    hyb_state_history,
    true_params,
    phys_loss_hist,
    hyb_phys_loss_hist,
    kappa_func,
    eta_func,
    pts_train,
    domain=(-3.0, 3.0),
    N=100,
    filename="evolution.gif"
):
    """
    Animates how Kappa and Eta evolve epoch-by-epoch for physics-only and hybrid models.
    Each frame shows:
      Top row: Kappa: Physics | Hybrid | True
      Bot row: Eta:   Physics | Hybrid | True
      And below a loss history update.
    The layout now matches that of the static plot.
    """

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import matplotlib.animation as animation

    # Build a shared grid.
    x = jnp.linspace(domain[0], domain[1], N)
    y = jnp.linspace(domain[0], domain[1], N)
    xx, yy = jnp.meshgrid(x, y)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    filename = f"src/results/{filename}"

    # Compute predictions history.
    all_kappa_phy = [kappa_func(sp, xx_flat, yy_flat) for sp in phy_state_history]
    all_kappa_hyb = [kappa_func(sh, xx_flat, yy_flat) for sh in hyb_state_history]
    all_eta_phy   = [eta_func(sp, xx_flat, yy_flat) for sp in phy_state_history]
    all_eta_hyb   = [eta_func(sh, xx_flat, yy_flat) for sh in hyb_state_history]

    # Compute true (static) predictions.
    real_kappa = kappa_func(true_params, xx_flat, yy_flat)
    real_eta   = eta_func(true_params, xx_flat, yy_flat)

    # Convert lists to JAX arrays.
    all_kappa_phy_arr = jnp.stack(all_kappa_phy)
    all_kappa_hyb_arr = jnp.stack(all_kappa_hyb)
    all_eta_phy_arr   = jnp.stack(all_eta_phy)
    all_eta_hyb_arr   = jnp.stack(all_eta_hyb)

    # Global vmin/vmax across predictions (including true values).
    vmin = float(jnp.floor(jnp.min(jnp.array([real_kappa, real_eta]))))
    vmax = float(jnp.ceil(jnp.max(jnp.array([real_kappa, real_eta]))))

    # Setup the figure layout to match the static plot.
    fig = plt.figure(figsize=(10, 10))
    gs_top = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.2],
                              left=0.1, right=0.9, top=0.93, bottom=0.43,
                              wspace=0.1, hspace=0.1)
    ax0 = fig.add_subplot(gs_top[0, 0])  # Kappa Physics
    ax1 = fig.add_subplot(gs_top[0, 1])  # Kappa Hybrid
    ax2 = fig.add_subplot(gs_top[0, 2])  # Kappa True
    # Colorbar for top row will be added later.
    
    ax3 = fig.add_subplot(gs_top[1, 0])  # Eta Physics
    ax4 = fig.add_subplot(gs_top[1, 1])  # Eta Hybrid
    ax5 = fig.add_subplot(gs_top[1, 2])  # Eta True

    # Loss history axis.
    ax_loss = fig.add_axes([0.1, 0.07, 0.8, 0.3])
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss History")
    
    # Prepare shared norm and mappables.
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = "viridis"
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    # Draw static true contours on ax2 (Kappa True) and ax5 (Eta True).
    def draw_true(ax, data, title):
        cf = ax.contourf(xx, yy, data.reshape(xx.shape), levels=100,
                         cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(pts_train[:, 0], pts_train[:, 1], marker="o",
                   c="black", s=20, edgecolors="white")
        return cf

    cf_kTrue = draw_true(ax2, real_kappa, "Kappa True")
    cf_eTrue = draw_true(ax5, real_eta, "Eta True")
    
    # Unified colorbar.
    ax_cb = fig.add_subplot(gs_top[:, 3])
    norm = Normalize(vmin=vmin, vmax=vmax)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cb = fig.colorbar(mappable, cax=ax_cb)
    cb.set_label("Parameter Value")
    
    # Initialize loss line objects.
    loss_line_phys, = ax_loss.plot([], [], label="Physics Loss", color="blue")
    loss_line_hyb, = ax_loss.plot([], [], label="Hybrid Physics Loss", color="orange")
    ax_loss.legend()

    epochs = min(len(phy_state_history), len(hyb_state_history))

    def update(frame):
        # Clear dynamic axes (but not the static true axes ax2 and ax5 or the colorbar).
        for ax in [ax0, ax1, ax3, ax4]:
            ax.clear()
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

        # Fetch predictions for this epoch.
        kp_phy = all_kappa_phy_arr[frame].reshape(xx.shape)
        kp_hyb = all_kappa_hyb_arr[frame].reshape(xx.shape)
        et_phy = all_eta_phy_arr[frame].reshape(xx.shape)
        et_hyb = all_eta_hyb_arr[frame].reshape(xx.shape)

        # Plot Kappa predictions.
        cf0 = ax0.contourf(xx, yy, kp_phy, levels=100,
                             cmap=cmap, vmin=vmin, vmax=vmax)
        ax0.set_title(f"Kappa Physics - Epoch {frame}")
        ax0.scatter(pts_train[:, 0], pts_train[:, 1], marker="o",
                    c="black", s=20, edgecolors="white")

        cf1 = ax1.contourf(xx, yy, kp_hyb, levels=100,
                             cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.set_title(f"Kappa Hybrid - Epoch {frame}")
        ax1.scatter(pts_train[:, 0], pts_train[:, 1], marker="o",
                    c="black", s=20, edgecolors="white")

        # Plot Eta predictions.
        cf3 = ax3.contourf(xx, yy, et_phy, levels=100,
                             cmap=cmap, vmin=vmin, vmax=vmax)
        ax3.set_title(f"Eta Physics - Epoch {frame}")
        ax3.scatter(pts_train[:, 0], pts_train[:, 1], marker="o",
                    c="black", s=20, edgecolors="white")

        cf4 = ax4.contourf(xx, yy, et_hyb, levels=100,
                             cmap=cmap, vmin=vmin, vmax=vmax)
        ax4.set_title(f"Eta Hybrid - Epoch {frame}")
        ax4.scatter(pts_train[:, 0], pts_train[:, 1], marker="o",
                    c="black", s=20, edgecolors="white")

        # Update loss plot.
        epoch_range = list(range(frame + 1))
        loss_line_phys.set_data(epoch_range, phys_loss_hist[:frame + 1])
        loss_line_hyb.set_data(epoch_range, hyb_phys_loss_hist[:frame + 1])
        ax_loss.relim()
        ax_loss.autoscale_view()

        # Return updated artists.
        return [cf0, cf1, cf3, cf4, loss_line_phys, loss_line_hyb]

    # Create one frame every 100 epochs.
    anim = animation.FuncAnimation(
        fig, update, frames=range(0, epochs, 5),
        interval=100, blit=False
    )
    anim.save(filename, dpi=120)
    plt.close(fig)