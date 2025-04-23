from parameters import *
from dicke_Liou_lib import *
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gaussian_kde

g_arr = {
	2.2: [0.2, 1.0],
	# 4.4: np.round(np.arange(0.1, 2.05, 0.1), 2),
	# 6.6: np.round(np.arange(0.1, 2.05, 0.1), 2)
}

# Use the first gamma value from the array
γ = γ_arr[0]
g_arr_for_gamma = g_arr[γ]
num_g = len(g_arr_for_gamma)
plt.figure(figsize=(18, 6 * num_g))  # Wider for 4 subplots per row
plt.suptitle(f"Eigenvalues and Density Heatmaps for γ={γ} j={j}, M={M_arr[0]}", fontsize=16)

for g_ind, g in enumerate(g_arr_for_gamma):
    if len(M_arr) == 1:
        eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M_arr[0], g, γ)
        eigvals = filter_eigenvals(j, M_arr[0], γ, eigvals)
        eigvals_list = [eigvals]
    elif len(M_arr) >= 3:
        eigvals_list = [Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ) for M in M_arr]
        eigvals = find_converged_eigvals(eigvals_list, j, M_arr, γ, g, rel_tol=0.1, abs_tol=1e-6)
        print(f"Converged eigenvalues for g={g}: {len(eigvals)} out of {len(eigvals_list[-1])}")

    # Unfold eigenvalues
    eigvals_unfolded = unfold_spectrum(eigvals)

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

    # Plot normal eigenvalues
    plt.subplot(num_g, 4, 4 * g_ind + 1)
    plt.title(f"Normal Spectrum g={g}")
    plt.xlabel("Re E")
    plt.ylabel("Im E")
    plt.scatter(eigvals_list[-1].real, eigvals_list[-1].imag, s=1, marker=".")
    plt.scatter(x, y, s=1, marker="^")

    # Plot density heatmap for normal eigenvalues
    plt.subplot(num_g, 4, 4 * g_ind + 2)
    plt.title(f"Density Heatmap (Normal) g={g}")
    plt.xlabel("Re E")
    plt.ylabel("Im E")
    plt.imshow(
        density_normal, origin='lower', aspect='auto',
        extent=(x.min(), x.max(), y.min(), y.max()), cmap='viridis'
    )
    plt.colorbar(label="Density")

    # Unfolded eigenvalues for plotting
    x_unf = np.real(eigvals_unfolded)
    y_unf = np.imag(eigvals_unfolded)

    # Plot unfolded eigenvalues
    plt.subplot(num_g, 4, 4 * g_ind + 3)
    plt.title(f"Unfolded Spectrum g={g}")
    plt.xlabel("Re E (unfolded)")
    plt.ylabel("Im E (unfolded)")
    plt.scatter(x_unf, y_unf, s=1, marker=".")

    # Compute density for unfolded eigenvalues
    grid_x_unf, grid_y_unf = np.meshgrid(
        np.linspace(x_unf.min(), x_unf.max(), 100),
        np.linspace(y_unf.min(), y_unf.max(), 100)
    )
    positions_unf = np.vstack([grid_x_unf.ravel(), grid_y_unf.ravel()]).T
    values_unf = np.vstack([x_unf, y_unf]).T
    kde_unf = gaussian_kde(values_unf.T)
    density_unf = kde_unf(positions_unf.T).reshape(grid_x_unf.shape)

    # Plot density heatmap for unfolded eigenvalues
    plt.subplot(num_g, 4, 4 * g_ind + 4)
    plt.title(f"Density Heatmap (Unfolded) g={g}")
    plt.xlabel("Re E (unfolded)")
    plt.ylabel("Im E (unfolded)")
    plt.imshow(
        density_unf, origin='lower', aspect='auto',
        extent=(x_unf.min(), x_unf.max(), y_unf.min(), y_unf.max()), cmap='viridis'
    )
    plt.colorbar(label="Density")

if not os.path.exists("plots"):
    os.makedirs("plots")
plt.tight_layout()
plt.savefig(f'plots/Dicke_evals_comparison_all_g_j={j}_gc={gc_arr[0]}_γ={γ}.png')
plt.show()