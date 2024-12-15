from parameters import *
from dicke_Liou_lib import *
import matplotlib.pyplot as plt

for g in tqdm(g_arr):
    eigvals = ginue_evals_fun(j, M, β)

plt.plot(eigvals.imag)
plt.show()