from parameters import *
from dicke_Liou_lib import *
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gaussian_kde

g_arr = {
    2.2: [0.2],
}

# Use the first gamma value from the array
γ = γ_arr[0]
g_arr_for_gamma = g_arr[γ]
num_g = len(g_arr_for_gamma)

plt.figure(figsize=(6.8, 3.4))  # APS-style figure size

for g_ind, g in enumerate(g_arr_for_gamma):
    if len(M_arr) == 1:
        α = 2/3  # Filtering parameter
        eigvals1 = Dicke_Lop_even_evals_fun(ω, ω0, j, M_arr[0], g, γ)
        eigvals_list = [eigvals1]
        eigvals = filter_eigenvals(j, M_arr[0], γ, eigvals1, α=α)
    elif len(M_arr) >= 3:
        eigvals_list = [Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ) for M in M_arr]
        eigvals = find_converged_eigvals(eigvals_list, j, M_arr, γ, g, rel_tol=0.01, abs_tol=1e-6)

    # Compute density for normal eigenvalues
    x = np.real(eigvals)
    y = np.imag(eigvals)
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), 100),
        np.linspace(y.min(), y.max(), 100)
    )
    positions = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    values = np.vstack([x, y]).T
    kde = gaussian_kde(values.T)
    density_normal = kde(positions.T).reshape(grid_x.shape)

    # --- Normal eigenvalues scatter ---
    plt.subplot(2, 4, 4 * g_ind + 1)
    # plt.scatter(eigvals1.real, eigvals1.imag, s=0.3)
    plt.scatter(x, y, s=0.3, marker=".")
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.tick_params(direction='in', which='both')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # --- Density heatmap normal ---
    plt.subplot(2, 4, 4 * g_ind + 2)
    plt.imshow(
        density_normal, origin='lower', aspect='auto',
        extent=(x.min(), x.max(), y.min(), y.max()), cmap='viridis'
    )
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    # Unfolded eigenvalues
    eigvals_unfolded = transform_spectrum(eigvals, beta = 1/3)
    x_unf = np.real(eigvals_unfolded)
    y_unf = np.imag(eigvals_unfolded)

    # --- Unfolded eigenvalues scatter ---
    plt.subplot(2, 4, 4 * g_ind + 3)
    plt.scatter(x_unf, y_unf, s=0.3, marker=".")
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.tick_params(direction='in', which='both')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # --- Density heatmap unfolded ---
    grid_x_unf, grid_y_unf = np.meshgrid(
        np.linspace(x_unf.min(), x_unf.max(), 100),
        np.linspace(y_unf.min(), y_unf.max(), 100)
    )
    positions_unf = np.vstack([grid_x_unf.ravel(), grid_y_unf.ravel()]).T
    values_unf = np.vstack([x_unf, y_unf]).T
    kde_unf = gaussian_kde(values_unf.T)
    density_unf = kde_unf(positions_unf.T).reshape(grid_x_unf.shape)

    plt.subplot(2, 4, 4 * g_ind + 4)
    plt.imshow(
        density_unf, origin='lower', aspect='auto',
        extent=(x_unf.min(), x_unf.max(), y_unf.min(), y_unf.max()), cmap='viridis'
    )
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

# --- Shared labels ---
# y-axis label on left plots
for i in [1, 5]:  # Subplot numbers (counted 1-indexed)
    plt.subplot(2, 4, i)
    plt.ylabel(r'Im $z$', fontsize=7)
# x-axis label on bottom plots
for i in [5, 6, 7, 8]:
    plt.subplot(2, 4, i)
    plt.xlabel(r'Re $z$', fontsize=7)

# --- Add "Normal" and "Unfolded" text at top ---
# plt.figtext(0.22, 0.95, "Normal", ha='center', va='center', fontsize=8)
# plt.figtext(0.72, 0.95, "Unfolded", ha='center', va='center', fontsize=8)

# --- Add g/gc text on right side for each row ---
for g_ind, g in enumerate(g_arr_for_gamma):
    plt.figtext(0.98, 0.75 - 0.5 * g_ind, r"$g/g_{cγ}$ = %.2f" % (g/gc_arr[0]), 
                ha='right', va='center', fontsize=8, rotation=90)

# --- Save ---
if not os.path.exists("plots"):
    os.makedirs("plots")
plt.tight_layout(rect=[0, 0, 0.96, 0.92])
plt.savefig(f'plots/Dicke_evals_comparison_all_g_j={j}_gc={gc_arr[0]}_γ={γ}_α={np.round(α,2)}.pdf', dpi=600)
plt.show()
