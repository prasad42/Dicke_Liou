from parameters import *
from dicke_Liou_lib import *
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gaussian_kde
import scipy.linalg as sl

def main():
    N = 1871  # Match the size from previous script
    ntraj = 20  # Number of trajectories to average over
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Arrays to store spacings from all trajectories
    all_spacings_ginue = []
    all_spacings_poisson = []

    for traj_ind in range(ntraj):
        # GinUE analysis
        eigvals_ginue = ginue_evals_fun(N, traj_ind=traj_ind)
        eigvals_ginue = unfold_spectrum(eigvals_ginue)
        spacings_ginue = compute_nearest_neighbor_spacings(eigvals_ginue)
        mean_spacing_ginue = np.mean(spacings_ginue)
        spacings_ginue = spacings_ginue / mean_spacing_ginue
        all_spacings_ginue.append(spacings_ginue)
        
        # Poisson analysis
        eigvals_poisson = poissonian_evals_fun(N)
        eigvals_poisson = unfold_spectrum(eigvals_poisson)
        spacings_poisson = compute_nearest_neighbor_spacings(eigvals_poisson)
        mean_spacing_poisson = np.mean(spacings_poisson)
        spacings_poisson = spacings_poisson / mean_spacing_poisson
        all_spacings_poisson.append(spacings_poisson)

    all_spacings_ginue = np.concatenate(all_spacings_ginue)
    all_spacings_poisson = np.concatenate(all_spacings_poisson)

    # Plot GinUE spacing distribution
    ax1.hist(all_spacings_ginue, bins=100, density=True, alpha=0.7, label=f'Data (ntraj={ntraj})')
    s_range = np.linspace(0, 3, 100)
    ax1.plot(s_range, p_ginue(s_range), 'r--', label='GinUE Theory')
    ax1.plot(s_range, p_2d_poissonian(s_range), 'g--', label='2D Poisson')
    ax1.set_title('GinUE Spacing Distribution')
    ax1.set_xlabel('s')
    ax1.set_ylabel('P(s)')
    ax1.set_xlim(0, 3)
    ax1.legend()

    # Plot Poisson spacing distribution
    ax2.hist(all_spacings_poisson, bins=100, density=True, alpha=0.7, label=f'Data (ntraj={ntraj})')
    ax2.plot(s_range, p_2d_poissonian(s_range), 'r--', label='2D Poisson')
    ax2.plot(s_range, p_ginue(s_range), 'g--', label='GinUE Theory')
    ax2.set_title('Poisson Spacing Distribution')
    ax2.set_xlabel('s')
    ax2.set_ylabel('P(s)')
    ax2.set_xlim(0, 3)
    ax2.legend()

    plt.tight_layout()
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig(f'plots/nnsd_ginue_poisson_comparison_ntraj={ntraj}.png')
    plt.show()

if __name__ == "__main__":
    main()