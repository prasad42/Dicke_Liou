import numpy as np
'''
-------------------------------------------------------------------------------------
				        # parameters #
-------------------------------------------------------------------------------------
'''

# SET UP THE CALCULATION

ω  = 1.0; ω0 = 1.0; j = 5; v = 30; γ=1.0; β=0; M_arr = [30] 
gc={np.round(np.sqrt(ω/ω0*(γ**2/4+ω**2))/2,2)}
win=500; σ = 1
axis = 'imag'
# axis = 'real'
kernel = 'rect' 
# kernel = 'gau'
g_arr = np.round(np.arange(0.2,0.55,0.1),2)
# g_arr = np.round(np.arange(0.6,0.95,0.1),2)
# Number of random matrices to average over in GOE and GUE
num_realizations = 1000

# Number of Processes
nproc = 10

# Time list
StartTime = 0
LateTime = 1000
dt = 0.01
tlist = np.arange(StartTime, LateTime, dt)

# Time list
pts = 10000
t_vals0 = np.linspace(0.001, 0.01, pts, endpoint=False)
t_vals1 = np.linspace(0.01, 0.1, pts, endpoint=False)
t_vals2 = np.linspace(0.1, 1, pts, endpoint=False)
t_vals3 = np.linspace(1, 10, pts, endpoint=False)
t_vals4 = np.linspace(10, 100, pts, endpoint=False)
t_vals5 = np.linspace(100, 1000, pts)
tlist_ginue = np.concatenate([t_vals0, t_vals1, t_vals2, t_vals3, t_vals4, t_vals5])