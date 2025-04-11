import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from parameters import *
from dicke_Liou_lib import *

def main():
    for M in M_arr:
        for g_ind, g in enumerate(g_arr):
            eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ)
            plot_spectrum(eigvals)
            filtered_eigvals = filter_eigenvals(j, M, γ, α, eigvals)
            plot_spectrum(filtered_eigvals)
            # locally unfold the complex eigenvalues
            # spacings = compute_nearest_neighbor_spacings(eigvals)
            unfolded_spacing = unfold_spectrum(filtered_eigvals)
            # plot_unfolding(spacings, unfolded_spacing, g)
            plot_ps_distribution(unfolded_spacing, g)

if __name__=="__main__":
    main()