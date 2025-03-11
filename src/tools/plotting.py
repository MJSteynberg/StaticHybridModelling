# filepath: /c:/Users/thyss/Documents/Work/Projects/HybridModelling/UnificationTests/StaticHybridModelling/src/tools/plotting.py
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.animation as animation
def plot(
    phy_state_history,
    hyb_state_history,
    pinn_state_history,      # Still required.
    true_params,
    phys_loss_hist,
    hyb_phys_loss_hist,
    pinn_loss_hist,          # Still required.
    kappa_func,
    eta_func,
    pts_train,
    domain=(-3.0, 3.0),
    N=100,
    filename="final_evolution"
):
    """
    Creates a static plot showing the final predictions for Physics, PINN, Hybrid, and True.
    Also plots their loss history along with column and row labels.
    """
    # Extract final states.
    final_phy  = phy_state_history[-1]
    final_pinn = pinn_state_history[-1]  # PINN now second column.
    final_hyb  = hyb_state_history[-1]   # Hybrid now third column.
    final_true = true_params

    # Build a shared grid.
    x = jnp.linspace(domain[0], domain[1], N)
    y = jnp.linspace(domain[0], domain[1], N)
    xx, yy = jnp.meshgrid(x, y)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    filename = f"src/results/{filename}.png"

    # Compute predictions.
    kappa_phy  = kappa_func(final_phy,  xx_flat, yy_flat)
    kappa_pinn = kappa_func(final_pinn, xx_flat, yy_flat)
    kappa_hyb  = kappa_func(final_hyb,  xx_flat, yy_flat)
    kappa_true = kappa_func(final_true, xx_flat, yy_flat)
    eta_phy    = eta_func(final_phy,  xx_flat, yy_flat)
    eta_pinn   = eta_func(final_pinn, xx_flat, yy_flat)
    eta_hyb    = eta_func(final_hyb,  xx_flat, yy_flat)
    eta_true   = eta_func(final_true, xx_flat, yy_flat)

    # Compute the global min/max from the true predictions.
    vmin = float(jnp.floor(jnp.min(jnp.array([kappa_true, eta_true]))))
    vmax = float(jnp.ceil(jnp.max(jnp.array([kappa_true, eta_true]))))

    # Create figure layout: 4 columns for predictions + a colorbar.
    fig = plt.figure(figsize=(12, 10))
    gs_top = fig.add_gridspec(
        2, 5, width_ratios=[1, 1, 1, 1, 0.2],
        left=0.07, right=0.93, top=0.93, bottom=0.45,
        wspace=0.1, hspace=0.1
    )

    # Row 1: Kappa plots.
    ax0 = fig.add_subplot(gs_top[0, 0])
    cf0 = ax0.contourf(xx, yy, kappa_phy.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs_top[0, 1])
    cf1 = ax1.contourf(xx, yy, kappa_pinn.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs_top[0, 2])
    cf2 = ax2.contourf(xx, yy, kappa_hyb.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(gs_top[0, 3])
    cf3 = ax3.contourf(xx, yy, kappa_true.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Row 2: Eta plots.
    ax4 = fig.add_subplot(gs_top[1, 0])
    cf4 = ax4.contourf(xx, yy, eta_phy.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax4.set_xticks([])
    ax4.set_yticks([])

    ax5 = fig.add_subplot(gs_top[1, 1])
    cf5 = ax5.contourf(xx, yy, eta_pinn.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax5.set_xticks([])
    ax5.set_yticks([])

    ax6 = fig.add_subplot(gs_top[1, 2])
    cf6 = ax6.contourf(xx, yy, eta_hyb.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax6.set_xticks([])
    ax6.set_yticks([])

    ax7 = fig.add_subplot(gs_top[1, 3])
    cf7 = ax7.contourf(xx, yy, eta_true.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax7.set_xticks([])
    ax7.set_yticks([])

    # Scatter training data on all subplots.
    for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.scatter(pts_train[:, 0], pts_train[:, 1], marker="o",
                   c="red", s=15)


    # Add a unified colorbar.
    ax_cb = fig.add_subplot(gs_top[:, 4])
    norm = Normalize(vmin=vmin, vmax=vmax)
    mappable = ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array([])
    cb = fig.colorbar(mappable, cax=ax_cb)
    cb.set_label("Parameter Value", fontsize=14)

    # Plot the loss history.
    ax_loss = fig.add_axes([0.07, 0.07, 0.86, 0.3])
    ax_loss.plot(phys_loss_hist, label="Physics Loss")
    ax_loss.plot(pinn_loss_hist, label="PINN Loss", linestyle=":")
    ax_loss.plot(hyb_phys_loss_hist, label="Hybrid Loss", linestyle="--")
    ax_loss.set_yscale("log")
    ax_loss.set_title("Loss History", fontsize=14)
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_loss.legend(fontsize=12)

    # Add column labels (above the grid).
    # The grid spans from left=0.07 to right=0.93 (width = 0.86). We split this into 4 columns.
    col_centers = [0.07 + 0.81 * (i + 0.5) / 4 for i in range(4)]
    for label, x_pos in zip(["Physics", "PINN", "Hybrid", "True"], col_centers):
        fig.text(x_pos, 0.96, label, ha="center", va="center", fontsize=18)

    # Add row labels on the left (at fixed positions).
    fig.text(0.04, 0.81, r"$\kappa$", ha="center", va="center", fontsize=22)
    fig.text(0.04, 0.57, r"$\eta^2$", ha="center", va="center", fontsize=22)

    plt.savefig(filename)
    plt.close(fig)

def plot_without_loss(
    phy_state_history,
    hyb_state_history,
    pinn_state_history,      # Still required.
    true_params,
    phys_loss_hist,
    hyb_phys_loss_hist,
    pinn_loss_hist,          # Still required.
    kappa_func,
    eta_func,
    pts_train,
    domain=(-3.0, 3.0),
    N=100,
    filename="final_evolution"
):
    """
    Creates a static plot showing the final predictions for Physics, PINN, Hybrid, and True.
    Also plots their loss history along with column and row labels.
    """
    # Extract final states.
    final_phy  = phy_state_history[-1]
    final_pinn = pinn_state_history[-1]  # PINN now second column.
    final_hyb  = hyb_state_history[-1]   # Hybrid now third column.
    final_true = true_params

    # Build a shared grid.
    x = jnp.linspace(domain[0], domain[1], N)
    y = jnp.linspace(domain[0], domain[1], N)
    xx, yy = jnp.meshgrid(x, y)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    filename = f"src/results/{filename}.png"

    # Compute predictions.
    kappa_phy  = kappa_func(final_phy,  xx_flat, yy_flat)
    kappa_pinn = kappa_func(final_pinn, xx_flat, yy_flat)
    kappa_hyb  = kappa_func(final_hyb,  xx_flat, yy_flat)
    kappa_true = kappa_func(final_true, xx_flat, yy_flat)
    eta_phy    = eta_func(final_phy,  xx_flat, yy_flat)
    eta_pinn   = eta_func(final_pinn, xx_flat, yy_flat)
    eta_hyb    = eta_func(final_hyb,  xx_flat, yy_flat)
    eta_true   = eta_func(final_true, xx_flat, yy_flat)

    # Compute the global min/max from the true predictions.
    vmin = float(jnp.floor(jnp.min(jnp.array([kappa_true, eta_true]))))
    vmax = float(jnp.ceil(jnp.max(jnp.array([kappa_true, eta_true]))))

    # Create figure layout: 4 columns for predictions + a colorbar.
    fig = plt.figure(figsize=(12, 5))
    gs_top = fig.add_gridspec(
        2, 5, width_ratios=[1, 1, 1, 1, 0.2],
        left=0.07, right=0.93, top=0.93, bottom=0.07,
        wspace=0.1, hspace=0.1
    )

    # Row 1: Kappa plots.
    ax0 = fig.add_subplot(gs_top[0, 0])
    cf0 = ax0.contourf(xx, yy, kappa_phy.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs_top[0, 1])
    cf1 = ax1.contourf(xx, yy, kappa_pinn.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs_top[0, 2])
    cf2 = ax2.contourf(xx, yy, kappa_hyb.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(gs_top[0, 3])
    cf3 = ax3.contourf(xx, yy, kappa_true.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Row 2: Eta plots.
    ax4 = fig.add_subplot(gs_top[1, 0])
    cf4 = ax4.contourf(xx, yy, eta_phy.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax4.set_xticks([])
    ax4.set_yticks([])

    ax5 = fig.add_subplot(gs_top[1, 1])
    cf5 = ax5.contourf(xx, yy, eta_pinn.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax5.set_xticks([])
    ax5.set_yticks([])

    ax6 = fig.add_subplot(gs_top[1, 2])
    cf6 = ax6.contourf(xx, yy, eta_hyb.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax6.set_xticks([])
    ax6.set_yticks([])

    ax7 = fig.add_subplot(gs_top[1, 3])
    cf7 = ax7.contourf(xx, yy, eta_true.reshape(xx.shape), levels=100,
                         vmin=vmin, vmax=vmax, cmap="viridis")
    ax7.set_xticks([])
    ax7.set_yticks([])

    # Scatter training data on all subplots.
    for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.scatter(pts_train[:, 0], pts_train[:, 1], marker="o",
                   c="red", s=15)

    # Add a unified colorbar.
    ax_cb = fig.add_subplot(gs_top[:, 4])
    norm = Normalize(vmin=vmin, vmax=vmax)
    mappable = ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array([])
    cb = fig.colorbar(mappable, cax=ax_cb)
    cb.set_label("Parameter Value", fontsize=14)

    # Add column labels (above the grid).
    # The grid spans from left=0.07 to right=0.93 (width = 0.86). We split this into 4 columns.
    col_centers = [0.07 + 0.81 * (i + 0.5) / 4 for i in range(4)]
    for label, x_pos in zip(["FEM", "PINN", "HYCO", "True"], col_centers):
        fig.text(x_pos, 0.96, label, ha="center", va="center", fontsize=18)

    # Add row labels on the left (at fixed positions).
    fig.text(0.04, 0.74, r"$\kappa$", ha="center", va="center", fontsize=22)
    fig.text(0.04, 0.27, r"$\eta^2$", ha="center", va="center", fontsize=22)

    plt.savefig(filename)
    plt.close(fig)

def animate(
    phy_state_history,
    hyb_state_history,
    pinn_state_history,       # Still required.
    true_params,
    phys_loss_hist,
    hyb_phys_loss_hist,
    pinn_loss_hist,           # Still required.
    kappa_func,
    eta_func,
    pts_train,
    domain=(-3.0, 3.0),
    N=100,
    filename="evolution.gif"
):
    """
    Animates how κ and η evolve epoch-by-epoch for Physics, PINN, Hybrid, and True models.
    Column labels (above) and row labels (to the left) are added for clarity.
    """
    # Build a shared grid.
    x = jnp.linspace(domain[0], domain[1], N)
    y = jnp.linspace(domain[0], domain[1], N)
    xx, yy = jnp.meshgrid(x, y)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    filename = f"src/results/{filename}.gif"

    # Compute predictions history for each model.
    all_kappa_phy  = [kappa_func(sp, xx_flat, yy_flat) for sp in phy_state_history]
    all_kappa_pinn = [kappa_func(sp, xx_flat, yy_flat) for sp in pinn_state_history]
    all_kappa_hyb  = [kappa_func(sp, xx_flat, yy_flat) for sp in hyb_state_history]
    real_kappa     = kappa_func(true_params, xx_flat, yy_flat)

    all_eta_phy   = [eta_func(sp, xx_flat, yy_flat) for sp in phy_state_history]
    all_eta_pinn  = [eta_func(sp, xx_flat, yy_flat) for sp in pinn_state_history]
    all_eta_hyb   = [eta_func(sp, xx_flat, yy_flat) for sp in hyb_state_history]
    real_eta      = eta_func(true_params, xx_flat, yy_flat)

    # Convert lists to JAX arrays.
    all_kappa_phy_arr  = jnp.stack(all_kappa_phy)
    all_kappa_pinn_arr = jnp.stack(all_kappa_pinn)
    all_kappa_hyb_arr  = jnp.stack(all_kappa_hyb)
    all_eta_phy_arr    = jnp.stack(all_eta_phy)
    all_eta_pinn_arr   = jnp.stack(all_eta_pinn)
    all_eta_hyb_arr    = jnp.stack(all_eta_hyb)

    vmin = float(jnp.floor(jnp.min(jnp.array([real_kappa, real_eta]))))
    vmax = float(jnp.ceil(jnp.max(jnp.array([real_kappa, real_eta]))))

    # Setup the figure layout: 4 columns for predictions.
    fig = plt.figure(figsize=(12, 10))
    gs_top = fig.add_gridspec(
        2, 5, width_ratios=[1, 1, 1, 1, 0.2],
        left=0.07, right=0.93, top=0.93, bottom=0.45,
        wspace=0.1, hspace=0.1
    )
    ax0 = fig.add_subplot(gs_top[0, 0])  # Kappa Physics
    ax1 = fig.add_subplot(gs_top[0, 1])  # Kappa PINN
    ax2 = fig.add_subplot(gs_top[0, 2])  # Kappa Hybrid
    ax3 = fig.add_subplot(gs_top[0, 3])  # Kappa True

    ax4 = fig.add_subplot(gs_top[1, 0])  # Eta Physics
    ax5 = fig.add_subplot(gs_top[1, 1])  # Eta PINN
    ax6 = fig.add_subplot(gs_top[1, 2])  # Eta Hybrid
    ax7 = fig.add_subplot(gs_top[1, 3])  # Eta True

    ax_loss = fig.add_axes([0.07, 0.07, 0.86, 0.3])
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_loss.set_title("Loss History", fontsize=14)

    # Prepare colorbar.
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = "viridis"
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    ax_cb = fig.add_subplot(gs_top[:, 4])
    cb = fig.colorbar(mappable, cax=ax_cb)
    cb.set_label("Parameter Value", fontsize=14)

    loss_line_phy, = ax_loss.plot([], [], label="Physics Loss", color="blue")
    loss_line_pinn, = ax_loss.plot([], [], label="PINN Loss", color="green")
    loss_line_hyb, = ax_loss.plot([], [], label="Hybrid Loss", color="orange")
    ax_loss.legend(fontsize=12)

    col_centers = [0.07 + 0.81 * (i + 0.5) / 4 for i in range(4)]
    for label, x_pos in zip(["Physics", "PINN", "Hybrid", "True"], col_centers):
        fig.text(x_pos, 0.96, label, ha="center", va="center", fontsize=18)

    # Add row labels on the left (at fixed positions).
    fig.text(0.04, 0.81, r"$\kappa$", ha="center", va="center", fontsize=22)
    fig.text(0.04, 0.57, r"$\eta^2$", ha="center", va="center", fontsize=22)

    epochs = min(len(phy_state_history), len(hyb_state_history), len(pinn_state_history))

    def update(frame):
        # Clear dynamic axes.
        for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
            ax.clear()
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

        # Fetch predictions for this epoch.
        kp_phy  = all_kappa_phy_arr[frame].reshape(xx.shape)
        kp_pinn = all_kappa_pinn_arr[frame].reshape(xx.shape)
        kp_hyb  = all_kappa_hyb_arr[frame].reshape(xx.shape)
        et_phy  = all_eta_phy_arr[frame].reshape(xx.shape)
        et_pinn = all_eta_pinn_arr[frame].reshape(xx.shape)
        et_hyb  = all_eta_hyb_arr[frame].reshape(xx.shape)

        # Plot Kappa predictions.
        cf0 = ax0.contourf(xx, yy, kp_phy, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
        cf1 = ax1.contourf(xx, yy, kp_pinn, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
        cf2 = ax2.contourf(xx, yy, kp_hyb, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
        cf3 = ax3.contourf(xx, yy, real_kappa.reshape(xx.shape), levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
        # Plot Eta predictions.
        cf4 = ax4.contourf(xx, yy, et_phy, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
        cf5 = ax5.contourf(xx, yy, et_pinn, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
        cf6 = ax6.contourf(xx, yy, et_hyb, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
        cf7 = ax7.contourf(xx, yy, real_eta.reshape(xx.shape), levels=100, cmap=cmap, vmin=vmin, vmax=vmax)

        # Scatter training data.
        for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
            ax.scatter(pts_train[:, 0], pts_train[:, 1], marker="o", c="black", s=20, edgecolors="white")

        # Update loss plot.
        epoch_range = list(range(frame + 1))
        loss_line_phy.set_data(epoch_range, phys_loss_hist[:frame + 1])
        loss_line_pinn.set_data(epoch_range, pinn_loss_hist[:frame + 1])
        loss_line_hyb.set_data(epoch_range, hyb_phys_loss_hist[:frame + 1])
        ax_loss.relim()
        ax_loss.autoscale_view()

        return [cf0, cf1, cf2, cf3, cf4, cf5, cf6, cf7,
                loss_line_phy, loss_line_pinn, loss_line_hyb]

    anim = animation.FuncAnimation(
        fig, update, frames=range(0, epochs, 12), interval=100, blit=False
    )
    anim.save(filename, dpi=120)
    plt.close(fig)