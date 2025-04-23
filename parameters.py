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

ω  = 1.0; ω0 = 1.0; j = 5; v = 30; β=0; M_arr = [40, 30, 20]; 
M_arr = [8, 6, 4]
# M_arr = [20]

win=500

# Define gamma values
γ_arr = [2.2, 4.4, 6.6]
γ_arr = [2.2]

# Define g ranges for each gamma
g_arr = {
	2.2: np.round(np.arange(0.1, 1.05, 0.1), 2),
	4.4: np.round(np.arange(0.1, 2.05, 0.1), 2),
	6.6: np.round(np.arange(0.1, 2.05, 0.1), 2)
}

# Calculate gc for each gamma
gc_arr = np.array([np.round(np.sqrt(ω/ω0*(γ**2/4+ω**2))/2, 2) for γ in γ_arr])

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