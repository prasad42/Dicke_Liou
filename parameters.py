import numpy as np
'''
-------------------------------------------------------------------------------------
				        # parameters #
-------------------------------------------------------------------------------------
'''

# SET UP THE CALCULATION

# Anisotropic Dicke model
model = "anis"
model = None

ω  = 1.0; ω0 = 1.0; j = 5; v = 30; γ=2.2; β=0; M_arr = [8, 6, 4]; 
# M_arr = [4]

win=500; σ = 1
α = 0.8 # Filtering criterion

kernel = 'rect'
# kernel = 'gau'

g_arr = np.round(np.arange(0.1,1.05,0.05),2)
# g_arr = [0.2, 0.4, 0.6, 0.7]
# g_arr = [0.7, 0.8, 0.9, 1.0]
# g_arr = [0.2, 0.5, 0.7, 1.0]
g_arr = [0.2, 1.0]
# g_arr = [0.1]

# Unfolding for DSFF
unfolding = 'yes'
# unfolding = 'no'

gc=np.round(np.sqrt(ω/ω0*(γ**2/4+ω**2))/2,2)

ntraj = 20

# Number of random matrices to 6average over in GOE and GUE
num_realizations = 1000

# Number of Processes
nproc = 2

# Time list
pts = 10000
t_vals0 = np.linspace(0.001, 0.01, pts, endpoint=False)
t_vals1 = np.linspace(0.01, 0.1, pts, endpoint=False)
t_vals2 = np.linspace(0.1, 1, pts, endpoint=False)
t_vals3 = np.linspace(1, 10, pts, endpoint=False)
t_vals4 = np.linspace(10, 100, pts, endpoint=False)
t_vals5 = np.linspace(100, 1000, pts)
tlist = np.concatenate([t_vals0, t_vals1, t_vals2, t_vals3, t_vals4, t_vals5])