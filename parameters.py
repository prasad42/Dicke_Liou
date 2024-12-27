import numpy as np
'''
-------------------------------------------------------------------------------------
				        # parameters #
-------------------------------------------------------------------------------------
'''

# SET UP THE CALCULATION

ω  = 1.0; ω0 = 1.0; j = 5; M = 30; v = 30; γ=2.2; β=0; gc={np.round(np.sqrt(ω/ω0*(γ**2+ω**2))/2,2)}; M_arr = [30]
g_arr = np.round(np.arange(1.0,1.55,0.1),2)
# Number of random matrices to average over in GOE and GUE
num_realizations = 1000

# Number of Processes
nproc = 10

# Time list
t_vals_0_to_01 = np.linspace(0, 0.1, 1000, endpoint=False)
t_vals_01_to_1 = np.linspace(0.1, 1, 1000, endpoint=False)
t_vals_1_to_10 = np.linspace(1, 10, 1000, endpoint=False)
t_vals_10_to_100 = np.linspace(10, 100, 1000, endpoint=False)
t_vals_100_to_1000 = np.linspace(100, 1000, 1000)
tlist = np.concatenate([t_vals_0_to_01, t_vals_01_to_1, t_vals_1_to_10, t_vals_10_to_100, t_vals_100_to_1000])

# Time list for open model with MCWF method
StartTime = 0
LateTime = 100
tlist_open = np.arange(StartTime, LateTime, 0.01)