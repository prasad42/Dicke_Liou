from parameters import *
from dicke_Liou_lib import *
import matplotlib.pyplot as plt

<<<<<<< HEAD
for g in tqdm(g_arr):
    Dicke_Lop_even_evals_fun(ω, ω0, g, M, j, γ)
=======
for g in g_arr:
    eigvals = Dicke_Lop_even_evals_fun(ω, ω0, g, M, j, γ)

plt.plot(eigvals.imag)
plt.show()
>>>>>>> 881d721cb431bd987926e2d531294f18e9739d2a
