from parameters import *
from dicke_Liou_lib import *
import matplotlib.pyplot as plt
import multiprocessing as mp

def main():
    # for traj_ind in range(5,11):
    eigvals = ginoe_evals_fun(j, M, β, traj_ind=0)
    plt.figure(figsize=(5,5))
    plt.scatter(eigvals.real, eigvals.imag, marker='.')
    plt.show()

if __name__ == "__main__":
    main()