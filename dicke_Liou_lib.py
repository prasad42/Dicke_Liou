import numpy as np
import qutip as qt
import os
import scipy.linalg as sl
import scipy.sparse.linalg as ssl
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

def Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ):
    '''
    This function returns the Dicke Hamiltonian for the following parameters.
    Args:
    - w : frequency of the bosonic field
    - w0 : Energy difference in spin states
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    - γ : Decay rate
    '''
    
    if not os.path.exists("evals_par_Lop"):
        os.mkdir("evals_par_Lop")
    file_path = f"evals_par_Lop/evals_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω/ω0*(γ**2+ω**2))/2,2)}_γ={γ}_g={g}.npy"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        a  = qt.tensor(qt.destroy(M), qt.qeye(int(2*j+1)))
        Jp = qt.tensor(qt.qeye(M), qt.jmat(j, '+'))
        Jm = qt.tensor(qt.qeye(M), qt.jmat(j, '-'))
        Jz = qt.tensor(qt.qeye(M), qt.jmat(j, 'z'))
        H0 = ω * a.dag() * a + ω0 * Jz
        H1 = 1.0 / np.sqrt(2*j) * (a + a.dag()) * (Jp + Jm)
        H = H0 + g * H1
        Lop = qt.liouvillian(H,np.sqrt(γ)*a)
        Lop = Lop.data
        Lop = Lop.to_array()
        Lop_even = Lop[::2,::2]

        print(f"g: {g}, Lop: {np.shape(Lop_even)}")

        # time_start = time.perf_counter()
        # eigvals = np.linalg.eigvals(Lop_even)
        # time_end = time.perf_counter()
        # print(f"Numpy: {time_end-time_start}, eigvals: {np.shape(eigvals)}")

        # time_start = time.perf_counter()
        # eigvals = sl.eigvals(Lop_even)
        # time_end = time.perf_counter()
        # print(f"Scipy: {time_end-time_start}, eigvals: {np.shape(eigvals)}")

        time_start = time.perf_counter()
        eigvals = ssl.eigs(Lop_even, k=int((2*j+1)*M)**2/2, return_eigenvectors=False)
        time_end = time.perf_counter()
        print(f"Scipy Sparse: {time_end-time_start}, eigvals: {np.shape(eigvals)}")

        np.save(file_path,eigvals)
    else:
        print(f"{file_path} already exists.")
    Lop_even_eigvals = np.load(file_path)

    return Lop_even_eigvals

def loc_avg_den(σ, eigvals, E):
    '''
    This function gives local density of states.
    ref: https://journals.aps.org/pra/pdf/10.1103/PhysRevA.105.L050201
    Args:   
    - w : frequency of the bosonic field
    - w0 : Energy difference in spin states
    - g : Coupling strength
    - M : Upper limit of bosonic fock states
    - j : Pseudospin
    - γ : Decay rate 
    - σ : Local unfolding parameter
    - i : index of the energy level
    - eigvals : Array of energy eigenvalues
    '''
    N = len(eigvals)
    π = np.pi
    rho_avg = 1/(2*π*σ**2*N)*np.sum(np.exp(-np.abs(E-eigvals))/(2*σ**2))
    
    return rho_avg

def dsff_list_fun(ω, ω0, j, M, g, β, γ, tlist):
    '''
    Calculates the sff with energies of the Dicke Hamiltonian at each time step for a single trajectory.
    Args:
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    - β : Inverse Temperature
    - tlist : pass time list as an array
    '''
    eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ)
    if not os.path.exists("dsff"):
        os.mkdir("dsff")
    file_path = f"dsff/dsff_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω*ω0)/2,2)}_β={β}_g={g}.npy"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        sff_list = []
        for t in tqdm(tlist):
            sff = 0
            norm = 0
            for eigval in eigvals:
                sff += np.exp(-(β-1j*t)*(np.real(eigval)))
                norm += np.exp(-β*eigval)
            sff = np.conjugate(sff)*sff/(norm**2)
            sff_list.append(sff)
            np.save(file_path,np.array(sff_list))
    else:
        print(f"{file_path} already exists.")
    sff_list = np.load(file_path)

    return sff_list

def sff_rl_fun(ω, ω0, j, M, g, β, γ, tlist, win = 50):
    '''
    This function returns the rolling average of the sff over time.
    Args:
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    - β : Inverse Temperature
    - tlist : pass time list as an array
    - win: Window size for rolling average
    '''
    sff_list = dsff_list_fun(ω, ω0, j, M, g, β, γ, tlist)
    sff_rl = []
    for t_ind in range(0,len(tlist),1):
        win_start = int(t_ind)
        win_end = int(t_ind+win)
        sff_rl_val = np.average(sff_list[win_start:win_end], axis=0)
        sff_rl.append(sff_rl_val)

    return sff_rl

def generate_ginue_matrix(N):
    """
    Generate an NxN Gaussian Orthogonal Ensemble (GOE) matrix.
    """
    A = np.random.normal(0, 1, size=(N, N))
    B = np.random.normal(0, 1, size=(N, N))
    A = (A + 1j*B) / np.sqrt(2)
    return A

def ginue_evals_fun(j, M, β, traj_ind):
    # N: Size of the GinUE matrix.
    N = (2*j+1)*M
    N = int(N**2/2)+1
    if not os.path.exists("evals_GinUE"):
        os.mkdir("evals_GinUE")
    file_path = f"evals_GinUE/evals_j={j}_M={M}_N={N}_β={β}_traj_ind={traj_ind}.npy"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        H = generate_ginue_matrix(N)

        # time_start = time.perf_counter()
        # eigvals = np.linalg.eigvals(H)
        # time_end = time.perf_counter()
        # print(f"Numpy: {time_end-time_start}, eigvals: {np.shape(eigvals)}")

        # time_start = time.perf_counter()
        # eigvals = sl.eigvals(H)
        # time_end = time.perf_counter()
        # print(f"Scipy: {time_end-time_start}, eigvals: {np.shape(eigvals)}")

        time_start = time.perf_counter()
        eigvals = ssl.eigs(H, k=N, return_eigenvectors=False)
        time_end = time.perf_counter()
        print(f"Scipy Sparse: {time_end-time_start}, eigvals: {np.shape(eigvals)}")

        np.save(file_path,eigvals)
    else:
        print(f"{file_path} already exists.")
    eigvals = np.load(file_path)

    return eigvals

def sff_ginue_list_fun(j, M, β, tlist, ntraj=10):
    """
    Compute the Spectral Form Factor (sff) for GOE matrices of size N,
    averaged over `num_realizations` random GOE matrices.
    
    Args:
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - β : Inverse Temperature
    - tlist: Array of time values (T) for which to compute sff.
    - ntraj: Number of GOE realizations to average over.
    
    Returns:
    - sff_list: Array of sff values for each T.
    """
    # N: Size of the GinUE matrix.
    N  = int((2*j+1)*M/2)
    N = N*N
    if not os.path.exists("dsff"):
        os.mkdir("dsff")
    file_path = f"dsff/dsff_goe_j={j}_M={M}_N={N}_β={β}.npy"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        sff_list = np.zeros_like(tlist, dtype=np.float64)
        for _ in tqdm(range(ntraj)):
            eigvals = ginue_evals_fun(j, M, β)
            for i, t in enumerate(tlist):
                exp_sum = np.sum(np.exp(-(β + 1j*t) * eigvals))
                sff_list[i] += np.abs(exp_sum)**2
        sff_list /= ntraj * N**2
        np.save(file_path,sff_list)
    else:
        print(f"{file_path} already exists.")

    sff_list = np.load(file_path)

    return sff_list

"""
def eigval_sp_fun(ω, ω0, g, M, j, γ, σ, eigvals):
    '''
    Unfolds the even spectrum locally and returns the unfolded eigenvalue spacings
    Args:
    - σ : Local unfolding parameter
    - eigvals: list of eigenvalues
    '''
    for E_ind, E in enumerate(eigvals):
        eigvals = np.delete(eigvals,E_ind)
        dist = abs(E-eigvals[0])
        for idx, _ in enumerate(eigvals[1:]):
            dist = abs(E-eigvals[idx])
            if min_dist<dist:
                min_dist=dist
                nn = E
                nn_idx = idx

    eigvals_sp = np.diff(Dicke_Lop_even_evals_fun(ω, ω0, g, M, j, γ))

    for E in eigvals:
        eigvals_sp = eigvals_sp * np.sqrt(loc_avg_den(σ, eigvals, E))

    return eigvals_sp

def r_avg_fun(ω, ω0, j, M, g):

    '''
    Calculates the average eigenvalue spacing ratio of the spectrum
    Args:
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    '''
    eigval_sp_arr = []
    r = []
    eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M, g)
    for i in range(len(eigvals)-1):
        eigval_sp_arr.append(eigvals[i+1]-eigvals[i])
    for i in range(len(eigvals)-2):
        r.append(eigval_sp_arr[i+1]/eigval_sp_arr[i])
    for i in range(len(eigvals)-2):
        if r[i] > 1:
            r[i] = 1/ r[i]
        else:
            r[i] = r[i]

    return np.average(r)
"""