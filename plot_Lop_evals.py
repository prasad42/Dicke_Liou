from parameters import *
from dicke_Liou_lib import *
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 10))
for M in M_arr:
    plt.suptitle(f"j={j}, γ={γ}")
    for g_ind, g in tqdm(enumerate(g_arr)):
        print(f"[ω, ω0, j, M, g, γ]={[ω, ω0, j, M, g, γ]}")
        eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ)
        plt.subplot(100, 1, g_ind+1)
        plt.title(f"g={g}")
        plt.xlim(-20,0)#; plt.ylim(-40,40)
        if M==20:
            plt.scatter(eigvals.real, eigvals.imag, s=.01, marker=".", label=f"ncutoff={M}")
        if M==30:
            plt.scatter(eigvals.real, eigvals.imag, s=.01, marker="v",  label=f"ncutoff={M}")
        if M==40:
            plt.scatter(eigvals.real, eigvals.imag, s=.01, marker="^",  label=f"ncutoff={M}")
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig(f'plots/Dicke_evals_j={j}_M={M_arr}_β={β}_gc={gc}_γ={γ}.png')
plt.legend()
plt.show()