from parameters import *
from dicke_Liou_lib import *
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gaussian_kde
import scipy.linalg as sl

# Set size and range for both ensembles
N = 1871

# Generate GinUE eigenvalues
eigvals_ginue = ginue_evals_fun(N, traj_ind=0)
# eigvals_ginue = transform_spectrum(eigvals_ginue, beta=0.5)

# Generate Poissonian eigenvalues
eigvals_poisson = poissonian_evals_fun(N, traj_ind=0)
# eigvals_poisson = transform_spectrum(eigvals_poisson, beta=0.5)

# Create visualization
fig = plt.figure(figsize=(10, 8))

def plot_spectrum_and_density(eigvals, row, title_prefix):
    x, y = np.real(eigvals), np.imag(eigvals)
    
    # Spectrum plot
    plt.subplot(2, 2, 2*row + 1)
    plt.title(f"{title_prefix} Spectrum")
    plt.scatter(x, y, s=20, marker=".", alpha=0.5)
    plt.xlabel("Re E"); plt.ylabel("Im E")
    
    # Density plot
    plt.subplot(2, 2, 2*row + 2)
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), 100),
        np.linspace(y.min(), y.max(), 100)
    )
    positions = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    values = np.vstack([x, y]).T
    kde = gaussian_kde(values.T)
    density = kde(positions.T).reshape(grid_x.shape)
    plt.title(f"{title_prefix} Density")
    plt.imshow(density, origin='lower', aspect='auto',
              extent=(x.min(), x.max(), y.min(), y.max()), cmap='viridis')
    plt.colorbar(label="Density")
    plt.xlabel("Re E"); plt.ylabel("Im E")

# Plot both cases
plot_spectrum_and_density(eigvals_ginue, 0, "GinUE")
plot_spectrum_and_density(eigvals_poisson, 1, "Poisson")

plt.tight_layout()
if not os.path.exists("plots"):
    os.makedirs("plots")
plt.savefig('plots/ginue_poisson_comparison.png')
plt.show()

# Create second figure for unfolded spectra
fig2 = plt.figure(figsize=(10, 8))

# Unfold both spectra
eigvals_ginue_unfolded = unfold_spectrum(eigvals_ginue)
eigvals_poisson_unfolded = unfold_spectrum(eigvals_poisson)

# Plot unfolded spectra
plot_spectrum_and_density(eigvals_ginue_unfolded, 0, "Unfolded GinUE")
plot_spectrum_and_density(eigvals_poisson_unfolded, 1, "Unfolded Poisson")

plt.tight_layout()
plt.savefig('plots/ginue_poisson_comparison_unfolded.png')
plt.show()