from parameters import *
from dicke_Liou_lib import *
import matplotlib.pyplot as plt
import time


start_time = time.time()

plt.figure(figsize=(5, 5))
N1 = 25397
N2 = 4916
N_target = 4916
eigvals1 = ginue_evals_fun(N1, traj_ind=0)
eigvals2 = ginue_evals_fun(N2, traj_ind=0)
eigvals_target =  extract_inner_circle(eigvals1, N_target)
plt.xlabel("Re E")
plt.ylabel("Im E")
# plt.scatter(eigvals1.real, eigvals1.imag, s=0.1, marker=".", label=f"N={N1}")
plt.scatter(eigvals2.real, eigvals2.imag, s=0.1, marker="^", label=f"N={N2}")
plt.scatter(eigvals_target.real, eigvals_target.imag, s=0.1, marker="^", label=f"N={N_target}")

if not os.path.exists("plots"):
    os.mkdir("plots")
plt.tight_layout()
plt.legend()

end_time = time.time()
print(f"Time elapsed(sec): {end_time-start_time}")

plt.savefig(f'plots/GinUE_evals.png')
plt.show()