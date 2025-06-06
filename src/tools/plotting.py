# filepath: /c:/Users/thyss/Documents/Work/Projects/HybridModelling/UnificationTests/StaticHybridModelling/src/tools/plotting.py
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.animation as animation
import numpy as np
def moving_avg_std(loss, window=10):
    """
    Returns the moving average, standard deviation, and x-axis (epoch indices)
    for a given loss history array.
    """
    if len(loss) < window:
        # Not enough points for moving stats, so return original arrays.
        x = np.arange(len(loss))
        return x, loss, np.zeros_like(loss)
    avg = np.convolve(loss, np.ones(window)/window, mode='valid')
    std = np.array([np.std(loss[i:i+window]) for i in range(len(loss)-window+1)])
    x = np.arange(window-1, len(loss))
    return x, avg, std

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
    hyb_synth_loss_hist=None,
    u_hyb_phys=None,
    u_hyb_syn=None,
    u_fem=None,
    u_pinn=None,
    u_true=None,
    filename="final_evolution"
):
    if eta_func is not None:
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
        fig = plt.figure(figsize=(12, 12))
        gs_top = fig.add_gridspec(
            4, 5, width_ratios=[1, 1, 1, 1, 0.2],
            left=0.10, right=0.93, top=0.93, bottom=0.27,
            wspace=0.07, hspace=0.07
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
                    c=u'#d62728', s=15)
            
        N = int(np.sqrt(u_hyb_phys.shape[0]))
        sol = [N, N]
        vmin_sol = min((u_hyb_phys).min(), (u_fem).min(), (u_pinn).min())
        vmax_sol = max((u_hyb_phys).max(), (u_fem).max(), (u_pinn).max())
        vmin_err = min((u_hyb_phys - u_true).min(), (u_fem - u_true).min(), (u_pinn - u_true).min())
        vmax_err = max((u_hyb_phys - u_true).max(), (u_fem - u_true).max(), (u_pinn - u_true).max())
        xx = jnp.linspace(domain[0], domain[1], N)
        yy = jnp.linspace(domain[0], domain[1], N)
        # Plot the solutions 
        ax8 = fig.add_subplot(gs_top[2, 0])
        cf8 = ax8.contourf(xx, yy, u_fem.reshape(sol), levels=100,
                                vmin=vmin_sol, vmax=vmax_sol, cmap="viridis")
        ax8.set_xticks([])
        ax8.set_yticks([])

        ax9 = fig.add_subplot(gs_top[2, 1])
        cf9 = ax9.contourf(xx, yy, u_pinn.reshape(sol), levels=100,
                                vmin=vmin_sol, vmax=vmax_sol, cmap="viridis")
        ax9.set_xticks([])
        ax9.set_yticks([])

        ax10 = fig.add_subplot(gs_top[2, 2])
        cf10 = ax10.contourf(xx, yy, u_hyb_phys.reshape(sol), levels=100,
                                vmin=vmin_sol, vmax=vmax_sol, cmap="viridis")
        ax10.set_xticks([])
        ax10.set_yticks([])

        ax11 = fig.add_subplot(gs_top[2, 3])
        cf11 = ax11.contourf(xx, yy, (u_true.reshape(sol)), levels=100,
                                vmin=vmin_sol, vmax=vmax_sol, cmap="viridis")
        ax11.set_xticks([])
        ax11.set_yticks([])

        ax12 = fig.add_subplot(gs_top[3, 0])
        cf12 = ax12.contourf(xx, yy, u_fem.reshape(sol) - u_true.reshape(sol), levels=100,
                                vmin=vmin_err, vmax=vmax_err, cmap="viridis")
        ax12.set_xticks([])
        ax12.set_yticks([])

        ax13 = fig.add_subplot(gs_top[3, 1])
        cf13 = ax13.contourf(xx, yy, u_pinn.reshape(sol) - u_true.reshape(sol), levels=100,
                                vmin=vmin_err, vmax=vmax_err, cmap="viridis")
        ax13.set_xticks([])
        ax13.set_yticks([])

        ax14 = fig.add_subplot(gs_top[3, 2])
        cf14 = ax14.contourf(xx, yy, u_hyb_phys.reshape(sol) - u_true.reshape(sol), levels=100,
                                vmin=vmin_err, vmax=vmax_err, cmap="viridis")
        ax14.set_xticks([])
        ax14.set_yticks([])

        ax15 = fig.add_subplot(gs_top[3, 3])
        cf15 = ax15.contourf(xx, yy, (u_true.reshape(sol) - u_true.reshape(sol)), levels=100,
                                vmin=vmin_err, vmax=vmax_err, cmap="viridis")
        ax15.set_xticks([])
        ax15.set_yticks([])

        # Add a unified colorbar for gaussians
        ax_cb = fig.add_subplot(gs_top[:2, 4])
        norm = Normalize(vmin=vmin, vmax=vmax)
        mappable = ScalarMappable(norm=norm, cmap="viridis")
        mappable.set_array([])
        cb = fig.colorbar(mappable, cax=ax_cb)
        cb.set_ticks([1.0, 2.0, 3.0, 4.0, 5.0])

        # Add a unified colorbar for solutions
        
        ax_cb = fig.add_subplot(gs_top[2:3, 4])
        norm = Normalize(vmin=vmin_sol, vmax=vmax_sol)
        mappable = ScalarMappable(norm=norm, cmap="viridis")
        mappable.set_array([])
        cb = fig.colorbar(mappable, cax=ax_cb)
        cb.set_ticks([-1.0, 0.0, 1.0, 2.0])

        ax_cb = fig.add_subplot(gs_top[3:4, 4])
        norm = Normalize(vmin=vmin_err, vmax=vmax_err)
        mappable = ScalarMappable(norm=norm, cmap="viridis")
        mappable.set_array([])
        cb = fig.colorbar(mappable, cax=ax_cb)
        cb.set_ticks([0, 1])

        #  u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728'

        if hyb_synth_loss_hist is None:
            # Add a unified colorbar.
            ax_cb = fig.add_subplot(gs_top[:, 4])
            norm = Normalize(vmin=vmin, vmax=vmax)
            mappable = ScalarMappable(norm=norm, cmap="viridis")
            mappable.set_array([])
            cb = fig.colorbar(mappable, cax=ax_cb)

            # Plot the loss history.
            ax_loss = fig.add_axes([0.07, 0.07, 0.86, 0.3])
            ax_loss.plot(phys_loss_hist, label="FEM")
            ax_loss.plot(pinn_loss_hist, label="PINN", linestyle=":")
            ax_loss.plot(hyb_phys_loss_hist, label="HYCO", linestyle="--")
            ax_loss.set_yscale("log")
            ax_loss.set_title("Mean Squared Error History", fontsize=14)
            ax_loss.set_xlabel("Epoch", fontsize=12)
            ax_loss.set_ylabel("Loss", fontsize=12)
            ax_loss.legend(fontsize=12)

            # Add column labels (above the grid).
            # The grid spans from left=0.07 to right=0.93 (width = 0.86). We split this into 4 columns.
            col_centers = [0.07 + 0.81 * (i + 0.5) / 4 for i in range(4)]
            for label, x_pos in zip(["FEM", "PINN", "HYCO", "True"], col_centers):
                fig.text(x_pos, 0.96, label, ha="center", va="center", fontsize=18)

            # Add row labels on the left (at fixed positions).
            fig.text(0.04, 0.81, r"$\kappa$", ha="center", va="center", fontsize=22)
            fig.text(0.04, 0.57, r"$\eta$", ha="center", va="center", fontsize=22)
        else:
            
            window = 100
            # Plot the loss history.
            ax_loss = fig.add_axes([0.1, 0.07, 0.83, 0.191])
            gradient = np.linspace(1, 0.6, 256)  # from white to light blue
            gradient = np.outer(gradient, gradient)  # diagonal fade
            
            x_hyb, avg_hyb, std_hyb = moving_avg_std(hyb_phys_loss_hist, window=window)
            ax_loss.plot(x_hyb, avg_hyb, label="HYCO Physical", color = 'red', lw = 2)

            x_synth, avg_synth, std_synth = moving_avg_std(hyb_synth_loss_hist, window=window)
            ax_loss.plot(x_synth, avg_synth, label="HYCO Synthetic", color = 'green', lw = 2)

            x_phys, avg_phys, std_phys = moving_avg_std(phys_loss_hist, window=window)
            ax_loss.plot(x_phys, avg_phys, label="FEM", color = 'black', lw = 2)
            # ax_loss.fill_between(x_phys, avg_phys - std_phys, avg_phys + std_phys, alpha=0.3)

            x_pinn, avg_pinn, std_pinn = moving_avg_std(pinn_loss_hist, window=window)
            ax_loss.plot(x_pinn, avg_pinn, label="PINN", color = 'blue', lw = 2)
            # ax_loss.fill_between(x_synth, avg_synth - std_synth, avg_synth + std_synth, alpha=0.3)
            ax_loss.imshow(gradient, extent = [100, 3010, 0.05, 1.5], aspect='auto', cmap='Blues', origin='lower', zorder=0, alpha = 0.1)
            ax_loss.imshow(gradient, extent = [100, 3010, 0.05, 1.5], aspect='auto', cmap='Grays', origin='lower', zorder=0, alpha = 0.05)
            # set yrange to not be too far from the average
            avg_min = min(avg_phys.min(), avg_pinn.min(), avg_hyb.min(), avg_synth.min())
            avg_max = max(avg_phys.max(), avg_pinn.max(), avg_hyb.max(), avg_synth.max())
            # ax_loss.set_ylim(avg_min * 0.8, avg_max * 1.4)
        

            ax_loss.set_yscale("log")
            ax_loss.set_xlabel("Epochs", fontsize=18)
            ax_loss.set_ylabel(r"$e_m$", fontsize=22, rotation=0)
            ax_loss.legend(fontsize=12, loc='upper right')

            # Add column labels (above the grid).
            # The grid spans from left=0.07 to right=0.93 (width = 0.86). We split this into 4 columns.
            col_centers = [0.1 + 0.785 * (i + 0.5) / 4 for i in range(4)]
            for label, x_pos in zip(["FEM", "PINN", "HYCO", "True"], col_centers):
                fig.text(x_pos, 0.96, label, ha="center", va="center", fontsize=18)

            # Add row labels on the left (at fixed positions).
            fig.text(0.047, 0.845, r"$\kappa$", ha="center", va="center", fontsize=22)
            fig.text(0.047, 0.685, r"$\eta$", ha="center", va="center", fontsize=22)
            fig.text(0.047, 0.515, r"$u_m$", ha="center", va="center", fontsize=22)
            fig.text(0.047, 0.355, r"$u - u_m$", ha="center", va="center", fontsize=22)
    else:
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

        # Compute the global min/max from the true predictions.
        vmin = float(jnp.floor(jnp.min(jnp.array([kappa_true]))))
        vmax = float(jnp.ceil(jnp.max(jnp.array([kappa_true]))))

        # Create figure layout: 4 columns for predictions + a colorbar.
        fig = plt.figure(figsize=(12, 10))
        gs_top = fig.add_gridspec(
            3, 5, width_ratios=[1, 1, 1, 1, 0.2],
            left=0.10, right=0.93, top=0.93, bottom=0.323,
            wspace=0.07, hspace=0.07
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

    

        # Scatter training data on all subplots.
        for ax in [ax0, ax1, ax2, ax3]:
            ax.scatter(pts_train[:, 0], pts_train[:, 1], marker="o",
                    c=u'#d62728', s=15)
            
        N = int(np.sqrt(u_hyb_phys.shape[0]))
        sol = [N, N]
        vmin_sol = min((u_hyb_phys).min(), (u_pinn).min())
        vmax_sol = max((u_hyb_phys).max(), (u_pinn).max()) + 2
        vmin_err = min((u_hyb_phys - u_true).min(), (u_pinn - u_true).min())
        vmax_err = max((u_hyb_phys - u_true).max(), (u_pinn - u_true).max()) + 4
        xx = jnp.linspace(domain[0], domain[1], N)
        yy = jnp.linspace(domain[0], domain[1], N)
        # Plot the solutions 
        ax8 = fig.add_subplot(gs_top[1, 0])
        cf8 = ax8.contourf(xx, yy, u_fem.reshape(sol), levels=200,
                                vmin=vmin_sol, vmax=vmax_sol, cmap="viridis")
        ax8.set_xticks([])
        ax8.set_yticks([])

        ax9 = fig.add_subplot(gs_top[1, 1])
        cf9 = ax9.contourf(xx, yy, u_pinn.reshape(sol), levels=200,
                                vmin=vmin_sol, vmax=vmax_sol, cmap="viridis")
        ax9.set_xticks([])
        ax9.set_yticks([])

        ax10 = fig.add_subplot(gs_top[1, 2])
        cf10 = ax10.contourf(xx, yy, u_hyb_phys.reshape(sol), levels=200,
                                vmin=vmin_sol, vmax=vmax_sol, cmap="viridis")
        ax10.set_xticks([])
        ax10.set_yticks([])

        ax11 = fig.add_subplot(gs_top[1, 3])
        cf11 = ax11.contourf(xx, yy, (u_true.reshape(sol)), levels=200,
                                vmin=vmin_sol, vmax=vmax_sol, cmap="viridis")
        ax11.set_xticks([])
        ax11.set_yticks([])

        ax12 = fig.add_subplot(gs_top[2, 0])
        cf12 = ax12.contourf(xx, yy, u_fem.reshape(sol) - u_true.reshape(sol), levels=200,
                                vmin=vmin_err, vmax=vmax_err, cmap="viridis")
        ax12.set_xticks([])
        ax12.set_yticks([])

        ax13 = fig.add_subplot(gs_top[2, 1])
        cf13 = ax13.contourf(xx, yy, u_pinn.reshape(sol) - u_true.reshape(sol), levels=200,
                                vmin=vmin_err, vmax=vmax_err, cmap="viridis")
        ax13.set_xticks([])
        ax13.set_yticks([])

        ax14 = fig.add_subplot(gs_top[2, 2])
        cf14 = ax14.contourf(xx, yy, u_hyb_phys.reshape(sol) - u_true.reshape(sol), levels=100,
                                vmin=vmin_err, vmax=vmax_err, cmap="viridis")
        ax14.set_xticks([])
        ax14.set_yticks([])

        ax15 = fig.add_subplot(gs_top[2, 3])
        cf15 = ax15.contourf(xx, yy, (u_true.reshape(sol) - u_true.reshape(sol)), levels=100,
                                vmin=vmin_err, vmax=vmax_err, cmap="viridis")
        ax15.set_xticks([])
        ax15.set_yticks([])

        # Add a unified colorbar for gaussians
        ax_cb = fig.add_subplot(gs_top[:1, 4])
        norm = Normalize(vmin=vmin, vmax=vmax)
        mappable = ScalarMappable(norm=norm, cmap="viridis")
        mappable.set_array([])
        cb = fig.colorbar(mappable, cax=ax_cb)
        cb.set_ticks([1.0, 2.0, 3.0, 4.0])

        # Add a unified colorbar for solutions
        
        ax_cb = fig.add_subplot(gs_top[1:2, 4])
        norm = Normalize(vmin=vmin_sol, vmax=vmax_sol)
        mappable = ScalarMappable(norm=norm, cmap="viridis")
        mappable.set_array([])
        cb = fig.colorbar(mappable, cax=ax_cb)
        cb.set_ticks([0.0, 1.0, 2.0, 3.0, 4.0, 5, 6])

        ax_cb = fig.add_subplot(gs_top[2:3, 4])
        norm = Normalize(vmin=vmin_err, vmax=vmax_err)
        mappable = ScalarMappable(norm=norm, cmap="viridis")
        mappable.set_array([])
        cb = fig.colorbar(mappable, cax=ax_cb)
        cb.set_ticks([0, 1, 2, 3, 4, 5])

        #  u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728'

        if hyb_synth_loss_hist is None:
            # Add a unified colorbar.
            ax_cb = fig.add_subplot(gs_top[:, 4])
            norm = Normalize(vmin=vmin, vmax=vmax)
            mappable = ScalarMappable(norm=norm, cmap="viridis")
            mappable.set_array([])
            cb = fig.colorbar(mappable, cax=ax_cb)

            # Plot the loss history.
            ax_loss = fig.add_axes([0.07, 0.07, 0.86, 0.29])
            ax_loss.plot(phys_loss_hist, label="FEM")
            ax_loss.plot(pinn_loss_hist, label="PINN", linestyle=":")
            ax_loss.plot(hyb_phys_loss_hist, label="HYCO", linestyle="--")
            ax_loss.set_yscale("log")
            ax_loss.set_title("Mean Squared Error History", fontsize=14)
            ax_loss.set_xlabel("Epoch", fontsize=12)
            ax_loss.set_ylabel("Loss", fontsize=12)
            ax_loss.legend(fontsize=12)

            # Add column labels (above the grid).
            # The grid spans from left=0.07 to right=0.93 (width = 0.86). We split this into 4 columns.
            col_centers = [0.07 + 0.81 * (i + 0.5) / 4 for i in range(4)]
            for label, x_pos in zip(["FEM", "PINN", "HYCO", "True"], col_centers):
                fig.text(x_pos, 0.96, label, ha="center", va="center", fontsize=18)

            # Add row labels on the left (at fixed positions).
            fig.text(0.04, 0.81, r"$\kappa$", ha="center", va="center", fontsize=22)
            fig.text(0.04, 0.57, r"$\eta$", ha="center", va="center", fontsize=22)
        else:
            
            window = 100
            # Plot the loss history.
            ax_loss = fig.add_axes([0.1, 0.07, 0.83, 0.24])
            gradient = np.linspace(1, 0.6, 256)  # from white to light blue
            gradient = np.outer(gradient, gradient)  # diagonal fade
            
            x_hyb, avg_hyb, std_hyb = moving_avg_std(hyb_phys_loss_hist, window=window)
            ax_loss.plot(x_hyb, avg_hyb, label="HYCO Physical", color = 'red', lw = 2)

            x_synth, avg_synth, std_synth = moving_avg_std(hyb_synth_loss_hist, window=window)
            ax_loss.plot(x_synth, avg_synth, label="HYCO Synthetic", color = 'green', lw = 2)

            x_phys, avg_phys, std_phys = moving_avg_std(phys_loss_hist, window=window)
            ax_loss.plot(x_phys, avg_phys, label="FEM", color = 'black', lw = 2)
            # ax_loss.fill_between(x_phys, avg_phys - std_phys, avg_phys + std_phys, alpha=0.3)

            x_pinn, avg_pinn, std_pinn = moving_avg_std(pinn_loss_hist, window=window)
            ax_loss.plot(x_pinn, avg_pinn, label="PINN", color = 'blue', lw = 2)
            # ax_loss.fill_between(x_synth, avg_synth - std_synth, avg_synth + std_synth, alpha=0.3)
            ax_loss.imshow(gradient, extent = [100, 3010, 0.05, 1.5], aspect='auto', cmap='Blues', origin='lower', zorder=0, alpha = 0.1)
            ax_loss.imshow(gradient, extent = [100, 3010, 0.05, 1.5], aspect='auto', cmap='Grays', origin='lower', zorder=0, alpha = 0.05)
            # set yrange to not be too far from the average
            avg_min = min(avg_phys.min(), avg_pinn.min(), avg_hyb.min(), avg_synth.min())
            avg_max = max(avg_phys.max(), avg_pinn.max(), avg_hyb.max(), avg_synth.max())
            # ax_loss.set_ylim(avg_min * 0.8, avg_max * 1.4)
        

            ax_loss.set_yscale("log")
            ax_loss.set_xlabel("Epochs", fontsize=18)
            ax_loss.set_ylabel(r"$e_m$", fontsize=22, rotation=0)
            ax_loss.legend(fontsize=12, loc='upper right')

            # Add column labels (above the grid).
            # The grid spans from left=0.07 to right=0.93 (width = 0.86). We split this into 4 columns.
            col_centers = [0.1 + 0.785 * (i + 0.5) / 4 for i in range(4)]
            for label, x_pos in zip(["FEM", "PINN", "HYCO", "True"], col_centers):
                fig.text(x_pos, 0.96, label, ha="center", va="center", fontsize=18)

            # Add row labels on the left (at fixed positions).
            fig.text(0.047, 0.845, r"$\kappa$", ha="center", va="center", fontsize=22)
            fig.text(0.047, 0.63, r"$u_m$", ha="center", va="center", fontsize=22)
            fig.text(0.047, 0.42, r"$u - u_m$", ha="center", va="center", fontsize=22)


        


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
                   c=u'#d62728', s=15)

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
    hyb_synth_loss_hist=None,
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

    all_eta_phy   = [eta_func(sp, xx_flat, yy_flat)**(1./2.) for sp in phy_state_history]
    all_eta_pinn  = [eta_func(sp, xx_flat, yy_flat)**(1./2.) for sp in pinn_state_history]
    all_eta_hyb   = [eta_func(sp, xx_flat, yy_flat)**(1./2.) for sp in hyb_state_history]
    real_eta      = eta_func(true_params, xx_flat, yy_flat)**(1./2.)

    # Convert lists to JAX arrays.
    all_kappa_phy_arr  = jnp.stack(all_kappa_phy)
    all_kappa_pinn_arr = jnp.stack(all_kappa_pinn)
    all_kappa_hyb_arr  = jnp.stack(all_kappa_hyb)
    all_eta_phy_arr    = jnp.stack(all_eta_phy)
    all_eta_pinn_arr   = jnp.stack(all_eta_pinn)
    all_eta_hyb_arr    = jnp.stack(all_eta_hyb)

    # Send the loss hists through the 100 epoch moving average.
    window = 100
    x_phys, avg_phys, std_phys = moving_avg_std(phys_loss_hist, window=window)
    x_pinn, avg_pinn, std_pinn = moving_avg_std(pinn_loss_hist, window=window)
    x_hyb, avg_hyb, std_hyb = moving_avg_std(hyb_phys_loss_hist, window=window)
    if hyb_synth_loss_hist is not None:
        x_synth, avg_synth, std_synth = moving_avg_std(hyb_synth_loss_hist, window=window)
    else:   
        x_synth, avg_synth, std_synth = None, None, None
    # Compute the global min/max from the true predictions.

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

    col_centers = [0.07 + 0.81 * (i + 0.5) / 4 for i in range(4)]
    for label, x_pos in zip(["FEM", "PINN", "HYCO", "True"], col_centers):
        fig.text(x_pos, 0.96, label, ha="center", va="center", fontsize=18)

    # Add row labels on the left (at fixed positions).
    fig.text(0.04, 0.81, r"$\kappa$", ha="center", va="center", fontsize=22)
    fig.text(0.04, 0.57, r"$\eta$", ha="center", va="center", fontsize=22)

    epochs = min(len(phy_state_history), len(hyb_state_history), len(pinn_state_history))
    
    # Compute fixed y-limits using the entire loss histories (i.e. as in the final frame)
    if hyb_synth_loss_hist is not None:
        max_ = max(avg_phys.max(), avg_pinn.max(), avg_hyb.max(), avg_synth.max())
        min_ = min(avg_phys.min(), avg_pinn.min(), avg_hyb.min(), avg_synth.min())
    fixed_ylim = (min_ * 0.8, max_ * 1.4)

    def update(frame):
        # Clear dynamic axes.
        for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
            ax.clear()
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        ax_loss.clear()
        ax_loss.set_yscale("log")

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
            ax.scatter(pts_train[:, 0], pts_train[:, 1], marker="o", c=u'#d62728', s=15)

        # Update loss plot.
        epoch_range = list(range(frame + 1))
        l1 = ax_loss.plot(x_phys[:frame + 1], avg_phys[:frame + 1], label="FEM Error")
        f1 = ax_loss.fill_between(x_phys[:frame + 1], avg_phys[:frame + 1] - std_phys[:frame + 1],
                            avg_phys[:frame + 1] + std_phys[:frame + 1], alpha=0.3, color = u'#1f77b4')
        l2 =ax_loss.plot(x_pinn[:frame + 1], avg_pinn[:frame + 1], label="PINN Error", linestyle=":")
        f2 = ax_loss.fill_between(x_pinn[:frame + 1], avg_pinn[:frame + 1] - std_pinn[:frame + 1],
                            avg_pinn[:frame + 1] + std_pinn[:frame + 1], alpha=0.3, color = u'#ff7f0e')
        l3 =ax_loss.plot(x_hyb[:frame + 1], avg_hyb[:frame + 1], label="HYCO Physical Error", linestyle="--")
        f3 = ax_loss.fill_between(x_hyb[:frame + 1], avg_hyb[:frame + 1] - std_hyb[:frame + 1],
                            avg_hyb[:frame + 1] + std_hyb[:frame + 1], alpha=0.3, color = u'#2ca02c')

        if hyb_synth_loss_hist is not None:
            l4 =ax_loss.plot(x_synth[:frame + 1], avg_synth[:frame + 1], label="HYCO Synthetic Error", linestyle="-.")
            f4 = ax_loss.fill_between(x_synth[:frame + 1], avg_synth[:frame + 1] - std_synth[:frame + 1],
                                avg_synth[:frame + 1] + std_synth[:frame + 1], alpha=0.3, color = u'#d62728')

        ax_loss.legend(fontsize=12)
        # Use the fixed y-limits computed from the full loss histories.
        ax_loss.set_ylim(fixed_ylim)
        ax_loss.set_xlim(0, epochs)

        return [cf0, cf1, cf2, cf3, cf4, cf5, cf6, cf7, l1, l2, l3, l4, f1, f2, f3, f4]

    anim = animation.FuncAnimation(
        fig, update, frames=range(0, epochs, 6), interval=100, blit=False
    )
    anim.save(filename, dpi=120)
    plt.close(fig)