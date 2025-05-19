import numpy as np
import matplotlib.pyplot as plt
import os
from parameters import *
from dicke_Liou_lib import *

γ = 2.2  # fixed gamma
M_arr = [20, 30, 40]
α = 1/3
rel_tol = 0.001

gc = np.round(np.sqrt(ω/ω0*(γ**2/4 + ω**2))/2, 2)
g_arr = np.round(np.arange(0.1, 1.05, 0.1), 2)
g_gc_arr = g_arr / gc

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6.6, 2.4), sharey=False)

markers = ['o', 's', '^']
colors = plt.cm.viridis(np.linspace(0, 1, len(M_arr)))

for idx, M in enumerate(M_arr):
    r_avg_arr = []
    cos_avg_arr = []

    for g in g_arr:
        r_avg, cos_avg = z_avg_fun(ω, ω0, j, [M], g, γ, α=α, rel_tol=rel_tol)
        print(f"M={M}, g/gc={g/gc:.2f}, r_avg={r_avg}, -cos_avg={-cos_avg}")
        r_avg_arr.append(r_avg)
        cos_avg_arr.append(-cos_avg)

    ax1.plot(g_gc_arr, r_avg_arr, marker=markers[idx], linestyle='-', color=colors[idx],
             label=fr'$M={M}$', markersize=5, linewidth=1)
    ax2.plot(g_gc_arr, cos_avg_arr, marker=markers[idx], linestyle='-', color=colors[idx],
             markersize=5, linewidth=1)

# Plot settings
ax1.set_ylabel(r"$\left\langle r\right\rangle$", fontsize=11)
ax2.set_ylabel(r"$-\left\langle \cos(\theta)\right\rangle$", fontsize=11)

for ax in [ax1, ax2]:
    ax.axvline(1, linestyle='--', color='black', alpha=0.3, linewidth=0.8)
    ax.tick_params(direction='in', length=4, width=1, labelsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

ax1.axhline(0.74, linestyle='--', color='black', linewidth=0.8, label="GinUE")
ax1.axhline(0.67, linestyle=':', color='gray', linewidth=0.8, label="2d Poisson")
ax2.axhline(0.24, linestyle='--', color='black', linewidth=0.8)
ax2.axhline(0, linestyle=':', color='gray', linewidth=0.8)

# Legend inside left subplot
ax1.legend(loc='lower right', fontsize=7, frameon=True)

fig.supxlabel(r"$g/g_c$", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 1.05])
os.makedirs("plots", exist_ok=True)
plt.savefig(f'plots/Dicke_r_cos_avg_convergence_γ={γ}_α={np.round(α,2)}.pdf', dpi=300, bbox_inches='tight')
plt.show()
