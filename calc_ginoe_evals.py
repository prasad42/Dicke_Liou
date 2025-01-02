from parameters import *
from dicke_Liou_lib import *
import matplotlib.pyplot as plt
import multiprocessing as mp

def main():
    for traj_ind in range(5,11):
        eigvals = ginoe_evals_fun(j, M, β, traj_ind)
        plt.figure(figsize=(5,5))
        plt.scatter(eigvals.real, eigvals.imag)
        plt.show()

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     with mp.Pool(nproc) as pool:
#         args_list = []
#         for traj_ind in range(2,4):
#             args_list.append([j, M, β, traj_ind])
#         for result in pool.starmap(calc_ginue_evals, args_list, chunksize = 1):
#             print(f'Result: {result}')