import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from parameters import *
from dicke_Liou_lib import *

def main():
    plt.figure(figsize=(5,5))
    for g_ind, g in enumerate(g_arr):
        eigval_sp = eigval_sp_fun(ω, ω0, j, M, g, γ, v)
        # Reference plots
        x_val = np.arange(0,4,0.1)
        y_gauss = np.pi/2 * x_val * np.exp(-np.pi*x_val**2/4)
        y_poi = np.exp(-x_val)
        # Create a histogram
        plt.subplot(int(len(g_arr)/2)+1,2,g_ind+1)
        plt.title(f'g={g}')
        plt.plot(x_val, y_gauss, linestyle = '--', label = 'Gaussian', linewidth = 1)
        plt.plot(x_val, y_poi, label = 'Poisson', linewidth = 1)
        hist_values,bin_edges, _ = plt.hist(eigval_sp, bins=30, histtype= 'step', density=True)
        plt.xlabel('s')
        plt.ylabel('P(s)')
        # plt.xlim([0,4])
        #plt.legend()
        plt.grid()
        plt.tight_layout()
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig(f'plots/Dicke_NNSD_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={gc}.jpg')
    plt.show()

if __name__=="__main__":
    main()