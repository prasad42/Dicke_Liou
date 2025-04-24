import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from parameters import *
from dicke_Liou_lib import *

def main():
    plt.figure(figsize=(8, 6))
    
    markers = ['o', 's', '^']  # Different markers for each gamma
    colors = ['b', 'r', 'g']   # Different colors for each gamma
    
    for γ_idx, γ in enumerate(γ_arr):
        eta_arr = []
        gc = gc_arr[γ_idx]
        g_gc_arr = g_arr[γ] / gc 
        
        for g in g_arr[γ]:
            if len(M_arr) == 1:
                eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M_arr[0], g, γ)
                eigvals = filter_eigenvals(j, M_arr[0], γ, eigvals)
            elif len(M_arr) >= 3:
                eigvals_list = [Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ) for M in M_arr]
                eigvals = find_converged_eigvals(eigvals_list, j, M_arr, γ, g)
                
            # unfolded_spacings = unfold_spacings(eigvals, j, M_arr[0], γ, g)

            unfolded_spectrum = unfold_spectrum(eigvals)
            unfolded_spacings = compute_nearest_neighbor_spacings(unfolded_spectrum)#, j, M_arr[0], γ, g)
            mean_spacing = np.mean(unfolded_spacings)
            unfolded_spacings = unfolded_spacings / mean_spacing

            eta = compute_eta(unfolded_spacings)
            print(f"γ={γ}, g/gc={g/gc:.2f}, eta={eta}")
            eta_arr.append(eta)
            
        plt.plot(g_gc_arr, eta_arr, f'{markers[γ_idx]}-', color=colors[γ_idx], 
                label=f'γ={γ}', markersize=6)

    plt.axhline(1, linestyle='--', color="k", label="GinUE")
    plt.axhline(0, linestyle='--', color="gray", label="Poissonian")
    plt.axvline(1, linestyle='--', color="black", alpha=0.3, label="g=gc")
    plt.legend()
    plt.xlabel("g/gc")
    plt.ylabel(r"$\eta$")
    plt.title(f"η vs g/gc, j={j}, M={M_arr[0]}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/Dicke_eta_multiple_gamma_j={j}_M={M_arr[0]}.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__=="__main__":
    main()