import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from parameters import *
from dicke_Liou_lib import *

g_arr = {
	2.2: [0.2, 1.0],
	# 4.4: np.round(np.arange(0.1, 2.05, 0.1), 2),
	# 6.6: np.round(np.arange(0.1, 2.05, 0.1), 2)
}

def main():
    for γ in γ_arr:
        g_arr_for_gamma = g_arr[γ]
        num_g = len(g_arr_for_gamma)
        num_rows = (num_g+1) // 2
        plt.figure(figsize=(10,4*num_rows))
        for g_ind, g in enumerate(g_arr_for_gamma):
            plt.subplot(num_rows, 2, g_ind + 1)
            if len(M_arr) == 1:
                eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M_arr[0], g, γ)
                eigvals = filter_eigenvals(j, M_arr[0], γ, eigvals)
                print(f"Eigenvalues for g={g}: {len(eigvals)}")
            elif len(M_arr) >= 3:
                eigvals_list = [Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ) for M in M_arr]
                eigvals = find_converged_eigvals(eigvals_list, j, M_arr, γ, g, rel_tol=0.1, abs_tol=1e-6)
                print(f"Converged eigenvalues for g={g}: {len(eigvals)} out of {len(eigvals_list[-1])}")
             # Filter eigenvalues near the center
            eigvals = filter_circular_patch(eigvals)
            # Unfold eigenvalues
            unfolded_spectrum = unfold_spectrum(eigvals)
            # Filter eigenvalues near the center
            # unfolded_spectrum = filter_rectangular_patch(unfolded_spectrum)
            # compute the spacing distribution
            unfolded_spacings = compute_nearest_neighbor_spacings(unfolded_spectrum)#, j, M_arr[0], γ, g)
            mean_spacing = np.mean(unfolded_spacings)
            unfolded_spacings = unfolded_spacings / mean_spacing

            # plot the spacing distribution
            plot_ps_distribution(unfolded_spacings, bins=100)
            plt.title(f"P(s) Distribution for g = {g}")
        plt.tight_layout()
        plt.savefig(f"plots/Dicke_ps_distribution_gamma_j={j}_M={M_arr[0]}_{γ}.png")
        plt.show()

if __name__=="__main__":
    main()