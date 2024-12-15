from parameters import *
from dicke_Liou_lib import *
import matplotlib.pyplot as plt
import time

for traj_ind in range(10):
    eigvals = ginue_evals_fun(j, M, β, traj_ind)

# plt.figure(figsize=(5,5))
# plt.scatter(eigvals.real, eigvals.imag)
# plt.show()
