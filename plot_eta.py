import numpy as np
import matplotlib.pyplot as plt
import os
from parameters import *
from dicke_Liou_lib import *

# Parameters
γ_arr = [2.2, 4.4, 6.6]
# γ_arr = [2.2]
M_arr = [40]
# M_arr = [40, 30, 20]
α = 2/3

tol = 0.01

# Calculate gc for each gamma
gc_arr = np.array([np.round(np.sqrt(ω/ω0*((γ/2)**2+ω**2))/2, 2) for γ in γ_arr])
print(gc_arr)

# Define g ranges for each gamma
g_arr = {
    2.2: np.round(np.arange(0.1, 1.05, 0.1), 2),
    4.4: np.round(np.arange(0.1, 1.45, 0.1), 2),
    6.6: np.round(np.arange(0.1, 2.0, 0.1), 2)
}

def main():
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    
    markers = ['o', 's', '^']
    colors = plt.cm.viridis(np.linspace(0, 1, len(γ_arr)))

    for γ_idx, γ in enumerate(γ_arr):
        eta_arr = []
        gc = gc_arr[γ_idx]
        g_gc_arr = g_arr[γ] / gc 
        
        for g in g_arr[γ]:
            if len(M_arr) == 1:
                eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M_arr[0], g, γ)
                unfolded_spacings = unfold_spacings_filtered(eigvals, j, M_arr[0], γ, g, α=α)
            elif len(M_arr) >= 3:
                eigvals_list = [Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ) for M in M_arr]
                eigvals = find_converged_eigvals(eigvals_list, j, M_arr, γ, g, rel_tol = tol)
                unfolded_spacings = unfold_spacings_converged(eigvals, j, M_arr[0], γ, g, rel_tol = tol)
                print(f"Selected {len(eigvals)} converged eigenvalues out of {len(eigvals_list[-1])}")
            eta = compute_eta(unfolded_spacings)
            print(rf"γ/ω={γ}, g/gc={g/gc:.2f}, eta={eta}")
            eta_arr.append(eta)

        ax.plot(g_gc_arr, eta_arr, marker=markers[γ_idx], linestyle='-', color=colors[γ_idx],
                label=fr'$\gamma/\omega={γ/2}$', markersize=5, linewidth=1)

    # Horizontal and vertical reference lines
    ax.axhline(1, linestyle='--', color='black', linewidth=0.8, label="GinUE")
    ax.axhline(0, linestyle=':', color='r', linewidth=0.8, label="2D Poisson")
    ax.axvline(1, linestyle='--', color='black', alpha=0.3, linewidth=0.8)

    # Axis settings
    ax.set_xlabel(r"$g/g_{cγ}$", fontsize=8)
    ax.set_ylabel(r"$\eta$", fontsize=8)
    ax.tick_params(direction='in', length=4, width=1, labelsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1),
            ncol=3, frameon=False, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve 5% space at top

    # Save
    os.makedirs("plots", exist_ok=True)
    if len(M_arr) == 1:
        plt.savefig(f'plots/Dicke_eta_multiple_gamma_j={j}_M={M_arr[0]}_α={np.round(α,2)}.pdf', dpi=300, bbox_inches='tight')
    elif len(M_arr) >= 3:
        plt.savefig(f'plots/Dicke_eta_multiple_gamma_j={j}_M={M_arr[0]}_tol={tol}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
