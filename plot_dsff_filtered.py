import numpy as np
import matplotlib.pyplot as plt
import os
from dicke_Liou_lib import *
from parameters import *

α0_arr = [0.0]
# Define gamma values
γ_arr = [2.2]
gc_arr = np.array([np.round(np.sqrt(ω/ω0*(γ**2/4+ω**2))/2, 2) for γ in γ_arr])
θ = 3 * np.pi / 4
M_arr = [40]
α = 2/3

# Define g ranges for each gamma
g_arr = {
    2.2: [0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0],
    4.4: [0.2, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6],
    6.6: [0.2, 0.6, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
}

# Time list
pts = 1000
t_vals_1 = np.linspace(0.0001, 0.001, pts, endpoint=False)
t_vals0 = np.linspace(0.001, 0.01, pts, endpoint=False)
t_vals1 = np.linspace(0.01, 0.1, pts, endpoint=False)
t_vals2 = np.linspace(0.1, 1, pts, endpoint=False)
t_vals3 = np.linspace(1, 10, pts, endpoint=False)
t_vals4 = np.linspace(10, 100, pts, endpoint=False)
t_vals5 = np.linspace(100, 1000, pts)
t_vals6 = np.linspace(1000, 10000, pts)
tlist = np.concatenate([t_vals0, t_vals1, t_vals2, t_vals3, t_vals4, t_vals5, t_vals6])

for α0 in α0_arr:
    for γ_ind, γ in enumerate(γ_arr):
        gc = gc_arr[γ_ind]
        g_list = g_arr[γ]
        num_g = len(g_list)
        num_rows = (num_g + 1) // 2
        fig, axes = plt.subplots(2, 4, figsize=(3.3*2, 2.7), sharex=True, sharey=True)
        axes = axes.flatten()
        
        for g_ind, g in enumerate(g_list):
            print(f"Computing for g = {g}...")
            ax = axes[g_ind]
            ax.tick_params(direction='in', which='both')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
            dsff, dsff_raw, N = compute_dsff(ω, ω0, j, M_arr, γ, g, tlist, β, win=100, θ=θ, α0=α0, n_theta=10, α=α)
            print(f"np.shape(dsff): {np.shape(dsff)}, np.shape(dsff_raw)= {np.shape(dsff_raw)}, eigvals={N}")

            dsff_ginue = theoretical_dsff_ginue(tlist, N)
            dsff_poissonian = K_Poisson(tlist, N) / N**2

            ax.set_xscale('log')
            ax.set_yscale('log')

            # Plot curves
            ax.plot(tlist, dsff_ginue, '--k', label='GinUE', linewidth = 0.8, alpha = 0.8)
            ax.plot(tlist, dsff_poissonian, ':r', label='2d Poisson', linewidth = 0.8)
            ax.plot(tlist, dsff, label='Open Dicke', linewidth = 0.8)

            ax.set_xlim(1e0,1e4)
            ax.set_ylim(1e-7,1.1e0)

            # Label inside plot with frame
            ax.text(
                0.45, 0.90, r"$g/g_{cγ}$" f"={np.round(g/gc,2)}",
                transform=ax.transAxes,
                ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.3'), fontsize = 8
            )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=8, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
        fig.text(0.5, 0.02, r"Time $\tau$", ha='center', va='center', fontsize=8)
        fig.text(0.01, 0.5, r"DSFF ($\tau,\phi$)", ha='center', va='center', rotation='vertical', fontsize=8)

        # Hide unused subplots if any
        for i in range(num_g, len(axes)):
            fig.delaxes(axes[i])

        # fig.suptitle(rf"Dicke model DSFF, $\theta = {θ/np.pi:.2f} \pi$", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # Save
        os.makedirs('plots_dsff_filtered', exist_ok=True)
        plt.savefig(f'plots_dsff_filtered/Dicke_dsff_filtered_j={j}_M={M_arr[0]}_β={β}_γ={γ}_gc={gc}_θ={np.round(θ,2)}_α0={α0}_α={np.round(α,2)}.pdf', dpi=300)
        plt.show()