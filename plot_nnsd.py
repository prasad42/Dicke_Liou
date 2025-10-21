import numpy as np
import matplotlib.pyplot as plt
import os
from parameters import *
from dicke_Liou_lib import *
from matplotlib.lines import Line2D

# Parameters
γ_arr = [2.2]
gc_arr = np.array([np.round(np.sqrt(ω/ω0* ((γ/2)**2+ω**2) )/2, 2) for γ in γ_arr])
g_arr = {
    2.2: [0.2, 1.0],
}
α = 2/3
rel_tol = 0.01

M_arr = [40, 30, 20]
M_arr = [40]

def main():
    os.makedirs("plots", exist_ok=True)
    
    for γ_ind, γ in enumerate(γ_arr):
        g_arr_for_gamma = g_arr[γ]
        gc = gc_arr[γ_ind]
        num_g = len(g_arr_for_gamma)
        num_rows = (num_g + 1) // 2

        fig, axes = plt.subplots(num_rows, 2, figsize=(3.3, 2.2), sharex=True, sharey=True)
        axes = axes.flatten()

        for g_ind, g in enumerate(g_arr_for_gamma):
            ax = axes[g_ind]

            if len(M_arr) == 1:
                eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M_arr[0], g, γ)
                unfolded_spacings = unfold_spacings_filtered(eigvals, j, M_arr[0], γ, g, α=α)
                print(f"Eigenvalues for g={g}: {len(eigvals)}")
            elif len(M_arr) >= 3:
                eigvals_list = [Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ) for M in M_arr]
                eigvals = find_converged_eigvals(eigvals_list, j, M_arr, γ, g, rel_tol=rel_tol)
                unfolded_spacings = unfold_spacings_converged(eigvals, j, M_arr[0], γ, g)
                print(f"Converged eigenvalues for g={g}: {len(eigvals)} out of {len(eigvals_list[-1])}")

            # eigvals_list = [Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ) for M in M_arr]
            # eigvals = find_converged_eigvals(eigvals_list, j, M_arr, γ, g)
            # eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M_arr[0], g, γ)
            # eigvals = filter_eigenvals(j, M_arr[0], γ, eigvals, α=α)
            # unfolded_spectrum = transform_spectrum(eigvals, beta = 1/3)
            # spacings = compute_nearest_neighbor_spacings(unfolded_spectrum)[:-1]
            # local_density = compute_local_density(eigvals[:-1], spacings)  # Compute density for spacing indices
            # unfolded_spacings = spacings * np.sqrt(local_density)  # Apply unfolding
            # s_bar = np.mean(unfolded_spacings)  # Compute global mean level spacing
            # # print(f"Unfolded spacing: {s_bar}")
            # unfolded_spacings = unfolded_spacings/s_bar

            # Comparison with 2D Poissonian and GinUE distributions
            s_vals = np.linspace(0, 3, 1000)
            ax.plot(s_vals, p_ginue(s_vals), label='GinUE', linestyle='--', color='k', linewidth=0.5)
            ax.plot(s_vals, p_2d_poissonian(s_vals), label='2D Poisson', linestyle='--', color='red', linewidth=0.5)

            # Plot using ax
            ax.hist(unfolded_spacings, bins=100, density=True, histtype='step', linewidth=1, label='Open Dicke')

            custom_lines = [
                Line2D([0], [0], linestyle='--', color='k', linewidth=1, label='GOE'),
                Line2D([0], [0], linestyle=':', color='r', linewidth=1, label='Poisson'),
                Line2D([0], [0], linestyle='-', linewidth=1, label='Dicke Model')
                ]
            fig.legend(custom_lines, ['GinUE', '2D Poisson', 'Dicke Model'], loc='upper center', ncol=3,
                    frameon=False, bbox_to_anchor=(0.5, 1.05), fontsize=7)

            # handles, labels = ax.get_legend_handles_labels()
            # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
            #         ncol=3, frameon=False, fontsize=8)
            
            # Place g label inside the plot
            ax.text(0.6, 0.9, r"$g/g_{cγ}=$" + f"{g/gc:.2f}", transform=ax.transAxes,
                    fontsize=8, ha='center', va='top', bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.7))

            ax.tick_params(direction='in', which='both', length=4, width=1, labelsize=9)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_xlim(0, 4)

        # Global labels
        fig.text(0.5, 0.02, 's', ha='center', va='center', fontsize=8)
        fig.text(0.02, 0.5, 'P(s)', ha='center', va='center', rotation='vertical', fontsize=8)

        # Turn off unused subplots
        # for i in range(num_g, len(axes)):
        #     fig.delaxes(axes[i])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save
        if len(M_arr) == 1:
            plt.savefig(f"plots/Dicke_ps_distribution_gamma_j={j}_M={M_arr[0]}_γ={γ}_α={np.round(α,2)}.pdf", dpi=300, bbox_inches='tight')
        elif len(M_arr) >= 3:
            plt.savefig(f"plots/Dicke_ps_distribution_gamma_j={j}_M={M_arr[0]}_γ={γ}_tol={rel_tol}.pdf", dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()
