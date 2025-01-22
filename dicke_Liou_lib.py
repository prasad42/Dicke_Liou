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
from multiprocessing import Pool
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

def dsff_fun(ω, ω0, j, M, g, β, γ, tlist_ginue, tlist, axis, win, σ, kernel):
    """
    Calculates the SFF with energies of the Dicke Hamiltonian at each time step for a single trajectory.
    Args:
    - j : Pseudospin
    - M : Upper limit of bosonic Fock states
    - g : Coupling strength
    - β : Inverse Temperature
    - tlist : Pass time list as an array
    """
    # Determine file paths based on kernel type
    suffix = f"j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω/ω0*(γ**2/4+ω**2))/2, 2)}_β={β}_g={g}_axis={axis}"
    if kernel == 'rect':
        tlist = tlist_ginue
        file_path = f"dsff/dsff_{suffix}_kernel={kernel}_win={win}.npy"
        file_path_raw = f"dsff/dsff_raw_{suffix}_kernel={kernel}_win={win}.npy"
    elif kernel == 'gau':
        file_path = f"dsff/dsff_{suffix}_kernel={kernel}_σ={σ}.npy"
        file_path_raw = f"dsff/dsff_raw_{suffix}_kernel={kernel}_σ={σ}.npy"
    
    # Ensure output directory exists
    os.makedirs("dsff", exist_ok=True)

    # Compute eigenvalues once
    eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ)

    def compute_dsff():
        """
        Compute the DSFF for the given eigenvalues and time list.
        """
        dsff = []
        for t in tqdm(tlist, total=len(tlist)):
            if axis == "imag":
                sff = np.sum(np.exp(-(β - 1j * t) * np.imag(eigvals)))
                norm = np.sum(np.exp(-β * np.imag(eigvals)))
            elif axis == "real":
                sff = np.sum(np.exp(-(β - 1j * t) * np.real(eigvals)))
                norm = np.sum(np.exp(-β * np.real(eigvals)))
            sff = np.conjugate(sff) * sff / (norm**2)
            dsff.append(sff)
        return np.array(dsff)

    # Compute or load smoothed DSFF
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        dsff_raw = compute_dsff()
        if kernel == 'rect':
            tlist_dsff = dsff_rl_rect_fun(tlist, dsff_raw, win)
        elif kernel == 'gau':
            tlist_dsff = dsff_rl_gau_fun(tlist, dsff_raw, σ)
        np.save(file_path, tlist_dsff)
    else:
        print(f"{file_path} already exists.")
        tlist_dsff = np.load(file_path)

    # Compute or load raw DSFF
    if not os.path.exists(file_path_raw):
        print(f"{file_path_raw} does not exist, generating data.")
        dsff_raw = compute_dsff()
        np.save(file_path_raw, dsff_raw)
    else:
        print(f"{file_path_raw} already exists.")
        dsff_raw = np.load(file_path_raw)

    # Extract results
    tlist = tlist_dsff[0]
    dsff = tlist_dsff[1]

    return tlist, dsff, dsff_raw

def calc_SFF(params):
    """
    Wrapper function to call `dsff_fun` with given parameters.
    
    Args:
    - params (tuple): Parameters to be passed to `dsff_fun`.
    
    Returns:
    - Tuple of tlist and dsff.
    """
    return dsff_fun(*params)

def parallel_SFF(ω, ω0, j, M_arr, g_arr, β, γ, tlist_ginue, tlist, axis, win, σ, kernel):
    """
    Parallel computation of SFF for multiple g and M values using all available CPU cores.

    Args:
    - ω, ω0, j, β, γ, tlist_ginue, tlist, axis, win, σ, kernel: Arguments for `dsff_fun`.
    - M_vals: List of M values.
    - g_vals: List of g values.

    Returns:
    - results: List of tuples containing (tlist, dsff) for each (M, g) combination.
    """
    # Determine the number of available CPU cores
    n_processes = os.cpu_count()

    # Create a list of parameter tuples
    param_list = [
        (ω, ω0, j, M, g, β, γ, tlist_ginue, tlist, axis, win, σ, kernel)
        for M in M_arr for g in g_arr
    ]

    # Use multiprocessing Pool to compute in parallel
    with Pool(n_processes) as pool:
        results = pool.map(calc_SFF, param_list)

    return results

def dsff_rl_rect_fun(tlist, dsff, win):
    """
    Smooth SFF using a Rectangular kernel.
    
    Args:
    - tlist: Array of time values.
    - sff: Array of SFF values.
    - win: Window size (an odd integer will be incremented to make it even).
    
    Returns:
    - tlist: Same input time values (unchanged).
    - sff_rl: Smoothed SFF values (same length as input).
    """

    half_win = win // 2
    dsff_rl = np.zeros_like(dsff, dtype=np.float64)

    # Perform rolling average with a rectangular kernel
    for i in range(len(dsff)):
        win_start = max(0, i - half_win)
        win_end = min(len(dsff), i + half_win)
        dsff_rl[i] = np.mean(dsff[win_start:win_end])
    
    # Return the original tlist and the smoothed SFF
    return tlist, dsff_rl

def dsff_rl_gau_fun(tlist, dsff, σ):
    """
    Smooth SFF using a Gaussian kernel.
    
    Args:
    - tlist: Array of time values.
    - sff: Array of SFF values.
    - σ: Standard deviation of the Gaussian kernel.
    
    Returns:
    - sff_rl: Smoothed SFF values.
    """
    dt = tlist[1] - tlist[0]
    normalization = 1 / (np.sqrt(2 * np.pi) * σ)
    dsff_rl = np.zeros(len(tlist), dtype=np.float64)

    for idx, t in tqdm(enumerate(tlist), total=len(tlist)):
        # Gaussian weights
        wt = normalization * np.exp(-0.5 * ((tlist - t) / σ) ** 2)
        # Weighted sum
        dsff_rl[idx] = np.sum(dsff * wt) * dt

    return tlist, dsff_rl

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

def dsff_ginue_fun(j, M, β, tlist_ginue, tlist, axis, win, σ, kernel, ntraj=1):
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
    - dsff: Array of sff values for each T.
    """
    # N: Size of the GinUE matrix.
    N  = int((2*j+1)*M)
    N = N**2/2
    if not os.path.exists("dsff"):
        os.mkdir("dsff")
    if kernel == 'rect':
        tlist = tlist_ginue
        file_path = f"dsff/dsff_ginue_j={j}_M={M}_N={N}_β={β}_axis={axis}_win={win}_kernel={kernel}.npy"
    elif kernel == 'gau':
        file_path = f"dsff/dsff_ginue_j={j}_M={M}_N={N}_β={β}_axis={axis}_σ={σ}_kernel={kernel}.npy"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        dsff = np.zeros_like(tlist, dtype=np.float64)
        eigvals = ginue_evals_fun(j, M, β, traj_ind=0)
        for i, t in tqdm(enumerate(tlist), total=len(tlist)):
            if axis == 'imag':
                exp_sum = np.sum(np.exp(-(β-1j*t)*(np.imag(eigvals))))
            elif axis == 'real':
                exp_sum = np.sum(np.exp(-(β-1j*t)*(np.real(eigvals))))
            dsff[i] += np.abs(exp_sum)**2
        dsff /= ntraj * N**2
        if kernel == 'rect':
            tlist_dsff_rl = dsff_rl_rect_fun(tlist, dsff, win)
        elif kernel == 'gau':
            tlist_dsff_rl = dsff_rl_gau_fun(tlist, dsff, σ)
        np.save(file_path,tlist_dsff_rl)
    else:
        print(f"{file_path} already exists.")

    tlist_dsff_rl = np.load(file_path)
    tlist = tlist_dsff_rl[0]
    dsff = tlist_dsff_rl[1]

    return tlist, dsff