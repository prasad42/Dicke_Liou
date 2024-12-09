import numpy as np
import qutip as qt
import os
import scipy.linalg as sl
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def Dicke_Lop_even_evals_fun(ω, ω0, g, M, j, γ):
    '''
    This function returns the Dicke Hamiltonian for the following parameters.
    Args:
    - w : frequency of the bosonic field
    - w0 : Energy difference in spin states
    - g : Coupling strength
    - M : Upper limit of bosonic fock states
    - j : Pseudospin
    - γ : Decay rate
    '''
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
    if not os.path.exists("evals_par_Lop"):
        os.mkdir("evals_par_Lop")
    file_path = f"evals_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω/ω0*(γ**2+ω**2))/2,2)}_γ={γ}_g={g}"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        eigvals = sl.eigvals(Lop_even)
        np.save(file_path,eigvals)
    else:
        print(f"{file_path} already exists.")
    Lop_even_evals = np.load(file_path)

    return Lop_even_evals

def loc_den(v, i, eigvals):
    '''
    This function gives local density of states.
    Args:    
    - v : Local unfolding parameter
    - i : index of the energy level
    - eigvals : Array of energy eigenvalues
    '''
    N = len(eigvals)
    if (v < 1 or v > int(N-1)):
        raise Exception(f"Enter number v between 0 and {N}")
    
    if (i < v):
        rho_L = 2 * v /(eigvals[v+v]-eigvals[0])
    elif (i > N-1-v):
        rho_L = 2 * v /(eigvals[N-1]-eigvals[N-1-v-v])
    else:
        rho_L = 2 * v /(eigvals[i+v]-eigvals[i-v])
    
    return rho_L

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
            rho_L = loc_den(v, m, eigvals)
            unf_val += rho_L * (eigvals[m]-eigvals[m-1])
        lvl_unf.append(unf_val)
    lvl_unf = np.sort(lvl_unf)
    
    return lvl_unf

def eigval_sp_fun(ω, ω0, j, M, g, v):

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
    eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M, g)  
    eigvals = unf_eigval_fun(v, eigvals)
    eigvals_sp = []
    for i in range(len(eigvals)-1):
        lvl_sp = eigvals[i+1]-eigvals[i]
        eigvals_sp.append(lvl_sp)
    eigvals_sp = np.sort(eigvals_sp)

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

def sff_list_fun(ω, ω0, j, M, g, β, tlist):
    '''
    Calculates the sff with energies of the Dicke Hamiltonian at each time step for a single trajectory.
    Args:
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    - β : Inverse Temperature
    - tlist : pass time list as an array
    '''
    eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M, g)
    if not os.path.exists("sff"):
        os.mkdir("sff")
    file_path = f"sff/sff_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω*ω0)/2,2)}_β={β}_g={g}.npy"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        sff_list = []
        for t in tlist:
            sff = 0
            norm = 0
            for eigval in eigvals:
                sff += np.exp(-(β+1j*t)*(eigval))
                norm += np.exp(-β*eigval)
            sff = np.conjugate(sff)*sff/(norm**2)
            sff_list.append(sff)
            np.save(file_path,np.array(sff_list))
    else:
        print(f"{file_path} already exists.")
    sff_list = np.load(file_path)

    return sff_list

def sff_rl_fun(ω, ω0, j, M, g, β, tlist, win = 50):
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
    sff_list = sff_list_fun(ω, ω0, j, M, g, β, tlist)
    sff_rl = []
    for t_ind in range(0,len(tlist),1):
        win_start = int(t_ind)
        win_end = int(t_ind+win)
        sff_rl_val = np.average(sff_list[win_start:win_end], axis=0)
        sff_rl.append(sff_rl_val)

    return sff_rl

def generate_goe_matrix(N):
    """
    Generate an NxN Gaussian Orthogonal Ensemble (GOE) matrix.
    """
    A = np.random.normal(0, 1, size=(N, N))
    A = (A + A.T) / 2
    return A

def sff_goe_list_fun(j, M, β, tlist, ntraj):
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
    # N: Size of the GOE matrix.
    N  = int((2*j+1)*M/2)
    if not os.path.exists("sff"):
        os.mkdir("sff")
    file_path = f"sff/sff_goe_j={j}_M={M}_N={N}_β={β}.npy"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        sff_list = np.zeros_like(tlist, dtype=np.float64)
        for _ in tqdm(range(ntraj)):
            H = generate_goe_matrix(N)
            eigvals = np.linalg.eigvalsh(H)
            for i, t in enumerate(tlist):
                exp_sum = np.sum(np.exp(-(β + 1j*t) * eigvals))
                sff_list[i] += np.abs(exp_sum)**2
        sff_list /= ntraj * N**2 
        np.save(file_path,sff_list)
    else:
        print(f"{file_path} already exists.")

    sff_list = np.load(file_path)

    return sff_list