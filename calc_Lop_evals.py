from parameters import *
from dicke_Liou_lib import *
import matplotlib.pyplot as plt

for g in tqdm(g_arr):
    eigvals = Dicke_Lop_even_evals_fun(ω, ω0, g, M, j, γ)

plt.plot(eigvals.imag)
plt.show()