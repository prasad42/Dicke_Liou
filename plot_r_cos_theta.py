import numpy as np
import matplotlib.pyplot as plt
import os
from parameters import *
from dicke_Liou_lib import *

# Parameters
j = 5
γ_arr = [2.2, 4.4, 6.6]
M_arr = [40]
α = 2/3
rel_tol = 0.01

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
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(3.4, 4.5), sharey=False)

    markers = ['o', 's', '^']
    colors = plt.cm.viridis(np.linspace(0, 1, len(γ_arr)))

    for γ_idx, γ in enumerate(γ_arr):
        r_avg_arr = []
        cos_avg_arr = []
        gc = gc_arr[γ_idx]
        g_gc_arr = g_arr[γ] / gc

        for g in g_arr[γ]:
            r_avg, cos_avg = z_avg_fun(ω, ω0, j, M_arr, g, γ, α=α, rel_tol=rel_tol)
            print(f"γ={γ}, g/gc={g/gc:.2f}, r_avg={r_avg}, -cos_avg={-cos_avg}")
            r_avg_arr.append(r_avg)
            cos_avg_arr.append(-cos_avg)

        # Left subplot: ⟨r⟩
        ax1.plot(g_gc_arr, r_avg_arr, marker=markers[γ_idx], linestyle='-', color=colors[γ_idx],
                 label=fr'$\gamma/\omega={γ/2}$', markersize=5, linewidth=1)
        ax1.set_ylabel(r"$\left\langle r\right\rangle$", fontsize=11)

        # Right subplot: ⟨cos(θ)⟩
        ax2.plot(g_gc_arr, cos_avg_arr, marker=markers[γ_idx], linestyle='-', color=colors[γ_idx],
                 markersize=5, linewidth=1)
        ax2.set_ylabel(r"$-\left\langle \cos(\theta)\right\rangle$", fontsize=11)

    # Annotations for ⟨r⟩
    ax1.axhline(0.74, linestyle='--', color='black', linewidth=0.8, label="GinUE")
    ax1.axhline(0.67, linestyle=':', color='r', linewidth=0.8, label="2d Poisson")
    ax1.axvline(1, linestyle='--', color='black', alpha=0.3, linewidth=0.8)

    # Annotations for -⟨cos(θ)⟩
    ax2.axhline(0.24, linestyle='--', color='black', linewidth=0.8)
    ax2.axhline(0, linestyle=':', color='r', linewidth=0.8)
    ax2.axvline(1, linestyle='--', color='black', alpha=0.3, linewidth=0.8)

    for ax in [ax1, ax2]:
        ax.tick_params(direction='in', length=4, width=1, labelsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # Shared legend at the top center
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.6, 1.05), ncol=3, fontsize=8, frameon=False, handletextpad=0.5, columnspacing=1.0)

    fig.supxlabel(r"$g/g_c$", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for legend

    # Save
    os.makedirs("plots", exist_ok=True)
    if len(M_arr) == 1:
        plt.savefig(f'plots/Dicke_r_cos_avg_j={j}_M={M_arr[0]}_α={np.round(α,2)}.pdf', dpi=300, bbox_inches='tight')
    elif len(M_arr) >= 3:
        plt.savefig(f'plots/Dicke_r_cos_avg_j={j}_M={M_arr[0]}_tol={rel_tol}.pdf', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
