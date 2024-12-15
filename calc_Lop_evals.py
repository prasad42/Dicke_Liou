from parameters import *
from dicke_Liou_lib import *
import matplotlib.pyplot as plt

g = 0.1
eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ)
# plt.figure(figsize=(10, 5))
# g_arr=np.array([0.2, 1.0])
# for g_ind, g in tqdm(enumerate(g_arr)):
#     eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ)
#     plt.subplot(1, 2, g_ind+1)
#     plt.title(f"g={g}")
#     plt.xlim(-25,0); plt.ylim(-40,40)
#     plt.scatter(eigvals.real, eigvals.imag, s=1, marker=".")
# if not os.path.exists("plots"):
#     os.mkdir("plots")
# plt.savefig(f'plots/Dicke_evals_j={j}_M={M}_β={β}_gc={gc}.png')
# plt.show()