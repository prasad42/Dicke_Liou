import numpy as np
import qutip as qt
import os
import scipy.linalg as sl
import scipy.sparse.linalg as ssl
import scipy.sparse as ss
from tqdm import tqdm
import time
import warnings
import subprocess
warnings.filterwarnings('ignore')
from sys import getsizeof

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
    file_path = f"evals_par_Lop/evals_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω/ω0*(γ**2/4+ω**2))/2,2)}_γ={γ}_g={g}.npy"
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
        Lop_even = Lop[::2,::2]
        Lop = Lop.data
        Lop = Lop.to_array()
        Lop = ss.csr_matrix(Lop)
        print(f"Lop is sparse: {ss.issparse(Lop)}, memory size: {getsizeof(Lop)}")
        print(f"g: {g}, Lop: {np.shape(Lop_even)}")
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

def unf_eigval_fun(v, eigvals):
    """
    Unfolds the even spectrum locally and returns the unfolded spectrum
    Args:
    - v : spread of eigenvalues taken into consideration while local unfolding
    - eigvals: list of eigenvalues
    """
    # Unfolded levels
    lvl_unf = []
    unf_val = 0
    for i in range(len(eigvals)):
        # Unfolded value of energy
        unf_val = 0
        for m in range(len(eigvals[:i])):
            # Local density of states
            rho_L = loc_avg_den(v, m, eigvals)
            unf_val += rho_L * (eigvals[m]-eigvals[m-1])
        lvl_unf.append(unf_val)
    lvl_unf = np.sort(lvl_unf)
    
    return lvl_unf

def eigval_sp_fun(ω, ω0, j, M, g, γ, v):
    '''
    The function returns the spacings between the unfolded eigenvalues
    Args:
    - ω : frequency of the bosonic field
    - ω0 : Energy difference in spin states
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    - v : Local unfolding parameter
    '''
    eigvals = (Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ)).imag
    eigvals = np.sort(eigvals)
    eigvals = np.extract(eigvals>0, eigvals)
    # eigvals = unf_eigval_fun(v, eigvals)
    eigvals_sp = []
    for i in range(v, len(eigvals)-v):
        lvl_sp = (2*v/(eigvals[i+v]-eigvals[i-v]))*(eigvals[i+1]-eigvals[i])
        eigvals_sp.append(lvl_sp)
    eigvals_sp = np.sort(eigvals_sp)

    return eigvals_sp

def dsff_list_fun(ω, ω0, j, M, g, β, γ, tlist, axis):
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
    file_path = f"dsff/dsff_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω/ω0*(γ**2/4+ω**2))/2,2)}_β={β}_g={g}_axis={axis}.npy"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        sff_list = []
        for t in tqdm(tlist):
            sff = 0
            norm = 0
            if axis == "imag":
                sff = np.sum(np.exp(-(β-1j*t)*(np.imag(eigvals))))
                norm += np.sum(np.exp(-β*np.imag(eigvals)))
            elif axis == "real":
                sff = np.sum(np.exp(-(β-1j*t)*(np.real(eigvals))))
                norm += np.sum(np.exp(-β*np.real(eigvals)))
            sff = np.conjugate(sff)*sff/(norm**2)
            sff_list.append(sff)
        np.save(file_path,np.array(sff_list))
    else:
        print(f"{file_path} already exists.")
    sff_list = np.load(file_path)

    return sff_list

def dsff_rl_fun(sff_list, tlist, win = 50):
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
    sff_rl = []
    for t_ind in range(0,len(tlist),1):
        win_start = int(t_ind)
        win_end = int(t_ind+win)
        sff_rl_val = np.average(sff_list[win_start:win_end], axis=0)
        sff_rl.append(sff_rl_val)

    return sff_rl

def generate_ginue_matrix(N):
    """
    Generate an NxN Ginibre Unitary Ensemble (GinUE) matrix.
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
    eigvals_list = []
    file_path = f"evals_GinUE/evals_j={j}_M={M}_N={N}_β={β}_traj_ind={traj_ind}.npy"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        H = generate_ginue_matrix(N)
        time_start = time.perf_counter()
        eigvals = sl.eigvals(H)
        time_end = time.perf_counter()
        print(f"Scipy: {time_end-time_start}, eigvals: {np.shape(eigvals)}")
        np.save(file_path,eigvals)
    else:
        print(f"{file_path} already exists.")
    eigvals = np.load(file_path)

    return eigvals

def dsff_ginue_list_fun(j, M, β, tlist, axis, ntraj=1, win=50):
    """
    Compute the Spectral Form Factor (sff) for GOE matrices of size N,
    averaged over `ntraj` random GOE matrices.
    
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
    N  = int((2*j+1)*M)
    N = N**2/2
    if not os.path.exists("dsff"):
        os.mkdir("dsff")
    file_path = f"dsff/dsff_goe_j={j}_M={M}_N={N}_β={β}_axis={axis}.npy"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        dsff_list = np.zeros_like(tlist, dtype=np.float64)
        eigvals = ginue_evals_fun(j, M, β, traj_ind=0)
        for i, t in tqdm(enumerate(tlist)):
            if axis == 'imag':
                exp_sum = np.sum(np.exp(-(β-1j*t)*(np.imag(eigvals))))
            elif axis == 'real':
                exp_sum = np.sum(np.exp(-(β-1j*t)*(np.real(eigvals))))
            dsff_list[i] += np.abs(exp_sum)**2
        dsff_list /= ntraj * N**2
        if ntraj == 1:
            dsff_rl = []
            for t_ind in range(0,len(tlist),1):
                win_start = int(t_ind)
                win_end = int(t_ind+win)
                dsff_rl_val = np.average(dsff_list[win_start:win_end], axis=0)
                dsff_rl.append(dsff_rl_val)
            np.save(file_path,dsff_rl)
        else:
            np.save(file_path,dsff_list)
    else:
        print(f"{file_path} already exists.")
    dsff_list = np.load(file_path)

    return dsff_list

def generate_ginoe_matrix(N):
    """
    Generate an NxN Ginibre Orthogonal Ensemble (GinOE) matrix.
    """
    A = np.random.normal(0, 1, size=(N, N))
    return A

def ginoe_evals_fun(j, M, β, traj_ind):
    # N: Size of the GinUE matrix.
    N = (2*j+1)*M
    N = int(N**2/2)+1
    if not os.path.exists("evals_GinUE"):
        os.mkdir("evals_GinUE")
    eigvals_list = []
    file_path = f"evals_GinUE/evals_j={j}_M={M}_N={N}_β={β}_traj_ind={traj_ind}.npy"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        H = generate_ginoe_matrix(N)
        time_start = time.perf_counter()
        eigvals = sl.eigvals(H)
        time_end = time.perf_counter()
        print(f"Scipy: {time_end-time_start}, eigvals: {np.shape(eigvals)}")
        np.save(file_path,eigvals)
    else:
        print(f"{file_path} already exists.")
    eigvals = np.load(file_path)

    return eigvals

def dsff_ginoe_list_fun(j, M, β, tlist, axis, ntraj=1, win=50):
    """
    Compute the Spectral Form Factor (sff) for GOE matrices of size N,
    averaged over `ntraj` random GOE matrices.
    
    Args:
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - β : Inverse Temperature
    - tlist: Array of time values (T) for which to compute sff.
    - ntraj: Number of GOE realizations to average over.
    
    Returns:
    - sff_list: Array of sff values for each T.
    """
    # N: Size of the GinOE matrix.
    N  = int((2*j+1)*M)
    N = N**2/2
    if not os.path.exists("dsff"):
        os.mkdir("dsff")
    file_path = f"dsff/dsff_goe_j={j}_M={M}_N={N}_β={β}_axis={axis}.npy"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        dsff_list = np.zeros_like(tlist, dtype=np.float64)
        eigvals = ginoe_evals_fun(j, M, β, traj_ind=0)
        for i, t in tqdm(enumerate(tlist)):
            if axis == 'imag':
                exp_sum = np.sum(np.exp(-(β-1j*t)*(np.imag(eigvals))))
            elif axis == 'real':
                exp_sum = np.sum(np.exp(-(β-1j*t)*(np.real(eigvals))))
            dsff_list[i] += np.abs(exp_sum)**2
        dsff_list /= ntraj * N**2
        if ntraj == 1:
            dsff_rl = []
            for t_ind in range(0,len(tlist),1):
                win_start = int(t_ind)
                win_end = int(t_ind+win)
                dsff_rl_val = np.average(dsff_list[win_start:win_end], axis=0)
                dsff_rl.append(dsff_rl_val)
            np.save(file_path,dsff_rl)
        else:
            np.save(file_path,dsff_list)
    else:
        print(f"{file_path} already exists.")
    dsff_list = np.load(file_path)

    return dsff_list