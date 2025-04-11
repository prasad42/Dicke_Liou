from parameters import *
from dicke_Liou_lib import *
import matplotlib.pyplot as plt
import multiprocessing as mp

def main():
    for traj_ind in range(1,20):
        ginue_evals_fun(25397, traj_ind)
        # plt.figure(figsize=(5,5))
        # plt.scatter(eigvals.real, eigvals.imag, marker='.')
        # plt.show()

if __name__ == "__main__":
    main()