from parameters import *
from dicke_Liou_lib import *
import matplotlib.pyplot as plt

num_g = len(g_arr)
num_col = (num_g + 1) // 2
plt.figure(figsize=(5*num_col, 10))
for M in M_arr:
    for g_ind, g in tqdm(enumerate(g_arr)):
        eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ)
        plt.subplot(2, num_col, g_ind+1)
        plt.title(f"g={g}")
        plt.xlabel("Re E")
        plt.ylabel("Im E")
        # plt.xlim(-20,0); plt.ylim(-40,40)
        if M==40:
            plt.scatter(eigvals.real, eigvals.imag, s=.01, marker="^",  label=f"M={M}")
        if M==30:
            plt.scatter(eigvals.real, eigvals.imag, s=.01, marker="v",  label=f"M={M}")
        if M==20:
            plt.scatter(eigvals.real, eigvals.imag, s=.01, marker=".", label=f"M={M}")
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.tight_layout()
    plt.legend()
plt.savefig(f'plots/Dicke_evals_j={j}_gc={gc}_γ={γ}.png')
plt.show()