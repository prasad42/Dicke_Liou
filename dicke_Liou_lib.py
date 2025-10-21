import numpy as np
import qutip as qt
import os
import scipy.linalg as sl
import scipy.sparse.linalg as ssl
import scipy.sparse as ss
from tqdm import tqdm
import time
import warnings
import multiprocessing as mp
warnings.filterwarnings('ignore')
from sys import getsizeof
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from scipy.special import gamma, gammainc
from math import factorial
import scipy.special as spl
from scipy.integrate import quad
from scipy.sparse import coo_matrix
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from numpy.polynomial import Polynomial
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.special import j1  # Bessel function of first kind, order 1

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
    
    os.makedirs("evals_par_Lop",exist_ok=True)
    file_path = f"evals_par_Lop/evals_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω/ω0*(γ**2/4+ω**2))/2,2)}_γ={γ}_g={g}.npy"
    print(f"j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω0/ω*((γ/2)**2+ω**2))/2,2)}_γ={γ}_g={g}")
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

        # Non sparse
        # Lop_even = create_parity_block(Lop, M, j)

        # Sparse
        Lop = Lop.data
        Lop = Lop.to_array()
        Lop = ss.csr_matrix(Lop)
        print(f"Lop is sparse: {ss.issparse(Lop)}, memory size: {getsizeof(Lop)}")
        Lop_even = create_parity_block_sparse(Lop, M, j)
        print(f"g: {g}, Lop: {np.shape(Lop_even)}")
        eigvals = ssl.eigs(Lop_even, k=int((2*j+1)*M)**2/2, return_eigenvectors=False)

        # print(f"Lop is dense, memory size: {getsizeof(Lop)} bytes")
        # print(f"g: {g}, Lop_even shape: {Lop_even.shape}")
        # # Compute eigenvalues using dense matrix solver
        # eigvals = sl.eigvals(Lop_even)

        np.save(file_path,eigvals)

    # else:
        # print(f"{file_path} already exists.")
    Lop_even_eigvals = np.load(file_path)

    return Lop_even_eigvals

def create_parity_block(arr, M, j):
    """
    Selects elements from arr based on block-wise parity:
    p(i,j) = (-1)^(i + j) or with shifts depending on the quadrant of the block.
    
    Then reshapes the selected elements into an (N/2 x N/2) matrix.

    Args:
    - arr: 2D square numpy array (N x N), where N must be even
    - n: integer, controls block size
    - j: integer, pseudospin parameter

    Returns:
    - selected_matrix: 2D numpy array of shape (N/2, N/2)
    """
    
    N = arr.shape[0]
    assert N % 2 == 0, "Array size must be even to reshape into (N/2, N/2)."

    B = 2 * M * (2 * j + 1)  # Full block size
    h = M * (2 * j + 1)      # Half block size

    rows, cols = np.indices(arr.shape)
    
    i_mod = rows % B
    j_mod = cols % B

    shift = np.zeros_like(rows)
    shift[(i_mod < h) & (j_mod >= h)] = 1
    shift[(i_mod >= h) & (j_mod < h)] = 1
    shift[(i_mod >= h) & (j_mod >= h)] = 2

    parity = (-1) ** (rows + cols + shift)
    mask = parity == 1

    # Apply row parity depending on whether i_mod < h or >= h
    row_parity = np.ones_like(rows)
    row_parity[i_mod < h] = (-1) ** rows[i_mod < h]
    row_parity[i_mod >= h] = (-1) ** (rows[i_mod >= h] + 1)

    # Step 4: Only keep elements where row parity == 1
    mask = mask & (row_parity == 1)

    selected_elements = arr[mask]
    new_size = N // 2
    selected_matrix = selected_elements.reshape(new_size, new_size)

    return selected_matrix

def create_parity_block_sparse(arr, M, j, cache_dir="mask_cache"):
    """
    Selects elements from a sparse matrix based on block-wise parity conditions,
    and returns a dense matrix of shape (N/2, N/2).

    Args:
    - arr: 2D square scipy sparse matrix (N x N), where N must be even
    - M: integer, controls block size
    - j: pseudospin parameter
    - cache_dir: directory to store/retrieve precomputed masks

    Returns:
    - selected_matrix: 2D numpy array of shape (N/2, N/2)
    """
    if not ss.issparse(arr):
        raise ValueError("Expected sparse matrix input")

    N = arr.shape[0]
    assert N % 2 == 0, "Matrix size must be even to reshape into (N/2, N/2)."

    # Cache path
    os.makedirs(cache_dir, exist_ok=True)
    mask_file = os.path.join(cache_dir, f"mask_M={M}_j={j}.npz")

    # Convert arr to COO format
    arr = arr.tocoo()
    row, col, data = arr.row, arr.col, arr.data

    if os.path.exists(mask_file):
        print(f"Loading mask from {mask_file}")
        idx_data = np.load(mask_file)
        sel_idx = idx_data["sel_idx"]
        i_new = idx_data["i_new"]
        j_new = idx_data["j_new"]
    else:
        t1 = time.time()
        print(f"Generating and saving mask for M={M}, j={j}")
        B = 2 * M * (2 * j + 1)
        h = M * (2 * j + 1)

        i_mod = row % B
        j_mod = col % B

        shift = np.zeros_like(row)
        shift[(i_mod < h) & (j_mod >= h)] = 1
        shift[(i_mod >= h) & (j_mod < h)] = 1
        shift[(i_mod >= h) & (j_mod >= h)] = 2

        parity = (-1) ** (row + col + shift)

        row_parity = np.ones_like(row)
        row_parity[i_mod < h] = (-1) ** row[i_mod < h]
        row_parity[i_mod >= h] = (-1) ** (row[i_mod >= h] + 1)

        keep = (parity == 1) & (row_parity == 1)

        sel_row = row[keep]
        sel_col = col[keep]

        unique_inds = np.unique(np.concatenate([sel_row, sel_col]))
        idx_map = -np.ones(N, dtype=int)
        idx_map[unique_inds] = np.arange(N // 2)

        i_new = idx_map[sel_row]
        j_new = idx_map[sel_col]
        sel_idx = np.where(keep)[0]

        np.savez(mask_file, sel_idx=sel_idx, i_new=i_new, j_new=j_new)
        t2 = time.time()
        print(f"Time taken to create mask: {t2-t1} s")

    # Apply to data
    selected_matrix = np.zeros((N // 2, N // 2), dtype=arr.dtype)
    selected_matrix[i_new, j_new] = data[sel_idx]

    return selected_matrix

def filter_eigenvals(j, M, γ, eigvals, α = 0.6):
    """
    Filters eigenvalues based on the described condition:
    - Find the eigenvalue with the highest imaginary part.
    - Use its real part as a threshold.
    - Keep eigenvalues whose absolute real part is <= the absolute real part of the identified eigenvalue.
    
    Args:
    - eigvals (array-like): Array of complex eigenvalues.
    
    Returns:
    - filtered_eigvals (array): Array of filtered eigenvalues.
    """

    threshold_real_part = α * M * γ/2
    # Filter eigenvalues based on the condition
    filtered_eigvals = eigvals[np.abs(np.real(eigvals)) <= threshold_real_part]
    eigvals_num = len(filtered_eigvals)
    max_abs_real = np.max(np.abs(np.real(filtered_eigvals)))
    print(f"j={j}, M={M}, γ={γ}, Maximum real part of an eigenvalue after filtering: {max_abs_real} and number of eigvals: {eigvals_num}")
    filtered_eigvals = np.sort_complex(filtered_eigvals)

    return np.array(filtered_eigvals)

def find_converged_eigvals(eigvals_list, j=None, M_arr=None, γ=None, g=None, rel_tol=0.1, abs_tol=1e-6, save_dir="converged_eigvals"):
    """
    Identify eigenvalues stable across cutoffs using:
    - Relative tolerance for magnitude
    - Absolute tolerance for near-zero eigenvalues
    Optionally saves/loads converged eigenvalues to/from disk using parameters j, M_arr, γ, g.
    """
    if all(param is not None for param in [j, M_arr, γ, g]):
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(
            save_dir,
            f"converged_j={j}_M={M_arr[0]}_γ={γ}_g={g}_reltol={rel_tol}_abstol={abs_tol}.npy"
        )
        if os.path.exists(file_path):
            print(f"Loading converged eigenvalues from {file_path}")
            return np.load(file_path)
    else:
        file_path = None

    # M40_evals = eigvals_list[0]
    # M30_evals = eigvals_list[1]
    # M20_evals = eigvals_list[2]

    M40_evals = np.sort_complex(eigvals_list[0])
    M30_evals = np.sort_complex(eigvals_list[1])
    M20_evals = np.sort_complex(eigvals_list[2])

    min_len = min(len(M40_evals), len(M30_evals), len(M20_evals))
    truncated_eigvals_list = np.array([arr[-min_len:] for arr in [M40_evals, M30_evals, M20_evals]])

    M40_evals = truncated_eigvals_list[0]
    M30_evals = truncated_eigvals_list[1]
    M20_evals = truncated_eigvals_list[2]
    print(f"Truncated eigenvalues to {min_len} elements.")

    converged = []
    
    print(f"{file_path} does not exist, generating data.")
    print(f"Finding converged eigenvalues...")
    for eigval40 in tqdm(M40_evals):
        # Find nearest neighbors in lower cutoffs
        eigval30 = find_nearest(eigval40, M30_evals)
        eigval20 = find_nearest(eigval40, M20_evals)
        
        # Calculate differences
        d30 = abs(eigval40 - eigval30)
        d20 = abs(eigval40 - eigval20)
        
        # Convergence criteria
        if abs(eigval40) < 1e-6:  # Handle near-zero values
            if d30 < abs_tol and d20 < abs_tol:
                converged.append(eigval40)
        else:
            rel_diff30 = d30/abs(eigval40)
            rel_diff20 = d20/abs(eigval40)
            if rel_diff30 < rel_tol and rel_diff20 < rel_tol:
                converged.append(eigval40)
                
    converged = np.array(converged)
    if file_path is not None:
        np.save(file_path, converged)

    return converged

def Dicke_eigvals_fun(ω, ω0, j, M_arr, g, γ, α = 0.6, rel_tol = 0.1):
    """
    Computes the eigenvalues of the Dicke Liouvillian for given parameters.

    If M_arr has 3 elements, it computes eigenvalues for each and finds converged ones.
    """

    if len(M_arr) == 3:
        eigvals_list = []
        for M in M_arr:
            eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ)
            eigvals_list.append(eigvals)
        eigvals = find_converged_eigvals(eigvals_list, j, M_arr, γ, g, rel_tol = rel_tol)
        print(f"Converged eigenvalues: {len(eigvals)} out of {len(eigvals_list[-1])}")
    else:
        eigvals1 = Dicke_Lop_even_evals_fun(ω, ω0, j, M_arr[0], g, γ)
        eigvals = filter_eigenvals(j, M_arr[0], γ, eigvals1, α = α)
        print(f"Filtered eigenvalues: {len(eigvals)} out of {len(eigvals1)}")

    return eigvals
    

def find_nearest(target, array):
    """Helper function to find closest eigenvalue"""
    return array[np.argmin(np.abs(array - target))]

def compute_nearest_neighbor_spacings(eigvals):
    """
    Computes the nearest-neighbor spacings for a given set of complex eigenvalues.
    Optionally saves/loads spacings to/from disk using parameters j, M, γ, g.
    """

    N = len(eigvals)
    spacings = np.zeros(N)
    print("Computing the Nearest Neighour Distance")
    for i in tqdm(range(N)):
        distances = np.abs(eigvals - eigvals[i])  # Compute all distances
        distances[i] = np.inf  # Ignore self-distance
        spacings[i] = np.min(distances)  # Find the nearest neighbor

    return spacings

def compute_local_density(eigvals, spacings, sigma=None, j=None, M=None, γ=None, g=None, α =0.6, rel_tol=None, save_dir="local_density", eigval_type = 'filtered'):
    """
    Estimates the local density ρ_av(E).
    Optionally saves/loads density to/from disk using parameters j, M, γ, g.
    """
    if all(param is not None for param in [j, M, γ, g, rel_tol]):
        os.makedirs(save_dir, exist_ok=True)
        if eigval_type == 'filtered':
            file_path = os.path.join(
                save_dir,
                f"local_density_j={j}_M={M}_γ={γ}_g={g}_α={α}.npy"
            )
        elif eigval_type == 'converged':
            file_path = os.path.join(
                save_dir,
                f"local_density_j={j}_M={M}_γ={γ}_g={g}_rel_tol={rel_tol}.npy"
            )
        if os.path.exists(file_path):
            print(f"Loading local density from {file_path}")
            return np.load(file_path)
    else:
        file_path = None

    N = len(eigvals)
    s_tilde = np.mean(spacings) # Global Mean spacing

    if sigma is None:
        sigma = 4.5 * s_tilde # Set σ as 4.5 × mean spacing

    density = np.zeros(N)
    prefactor = 1 / (2 * np.pi * sigma**2 * N)

    print("Computing the local density")
    for i, E in tqdm(enumerate(eigvals), total=len(eigvals)):
        density[i] = prefactor * np.sum(np.exp(-np.abs(E - eigvals)**2 / (2 * sigma**2)))

    if file_path is not None:
        np.save(file_path, density)

    return density

def unfold_spacings_filtered(eigvals, j=None, M=None, γ=None, g=None, α = 0.6):
    """Performs the unfolding procedure following equation (B1) in PhysRevA.105.L050201."""
    eigvals = filter_eigenvals(j, M, γ, eigvals, α = α)  # Filter eigenvalues
    eigvals = np.sort(eigvals)
    spacings = compute_nearest_neighbor_spacings(eigvals)[:-1]

    local_density = compute_local_density(eigvals[:-1], spacings, j = j, M = M, γ = γ, g = g, α = α, eigval_type='filtered')  # Compute density for spacing indices
    unfolded_spacings = spacings * np.sqrt(local_density)  # Apply unfolding
    s_bar = np.mean(unfolded_spacings)  # Compute global mean level spacing
    # print(f"Unfolded spacing: {s_bar}")
    unfolded_spacings = unfolded_spacings/s_bar # Normalise
    # print(f"Unfolded spacing After Normalisation {np.mean(unfolded_spacings)}")

    return unfolded_spacings

def unfold_spacings_converged(eigvals, j=None, M=None, γ=None, g=None, rel_tol = None):
    """Performs the unfolding procedure following equation (B1) in PhysRevA.105.L050201."""
    eigvals = np.sort(eigvals)
    spacings = compute_nearest_neighbor_spacings(eigvals)[:-1]

    local_density = compute_local_density(eigvals[:-1], spacings, j = j, M = M, γ = γ, g = g, rel_tol = rel_tol, eigval_type = 'converged')  # Compute density for spacing indices
    unfolded_spacings = spacings * np.sqrt(local_density)  # Apply unfolding
    s_bar = np.mean(unfolded_spacings)  # Compute global mean level spacing
    # print(f"Unfolded spacing: {s_bar}")
    unfolded_spacings = unfolded_spacings/s_bar # Normalise
    # print(f"Unfolded spacing After Normalisation {np.mean(unfolded_spacings)}")

    return unfolded_spacings

def unfold_spectrum(eigvals, num_y_bins=100, poly_order=6):
    """
    Unfold eigenvalues using a polynomial fit to the cumulative counting function
    (number variance), applied slice-wise along the imaginary axis.

    Args:
    - eigvals: array of complex eigenvalues.
    - num_y_bins: number of bins along the imaginary axis.
    - poly_order: degree of the polynomial to fit to the counting function.

    Returns:
    - unfolded_eigvals: array of complex unfolded eigenvalues.
    """
    eigvals = np.array(eigvals)
    x = np.real(eigvals)
    y = np.imag(eigvals)
    unfolded_x = np.zeros_like(x)

    y_bins = np.linspace(y.min(), y.max(), num_y_bins + 1)
    bin_indices = np.digitize(y, y_bins) - 1

    for i in range(num_y_bins):
        mask = bin_indices == i
        if np.sum(mask) < poly_order + 2:
            continue

        x_bin = x[mask]
        y_bin = y[mask]

        # Sort x within this y-bin
        sort_idx = np.argsort(x_bin)
        x_sorted = x_bin[sort_idx]
        count_indices = np.arange(1, len(x_sorted) + 1)

        # Fit polynomial to the counting function
        coeffs = np.polyfit(x_sorted, count_indices, deg=poly_order)
        cumulative_poly = np.poly1d(coeffs)

        # Evaluate unfolded x values using the fit
        unfolded_x[mask] = cumulative_poly(x_bin)

    unfolded_eigvals = unfolded_x + 1j * y

    return unfolded_eigvals

def p_2d_poissonian(s):
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)

def p_bar_ginue(s_val, j_max=20, k_max=20):
    """Computes the auxiliary function \bar{p}_{GinUE}(s) from the given formula."""

    # print(s_val)

    # Compute the product term separately
    product_term = 1
    for k in range(1, k_max + 1):
        upper_gamma = spl.gamma(1 + k) - spl.gammainc(1 + k, s_val**2) * spl.gamma(1 + k)
        product_term *= upper_gamma / factorial(k)
    # Compute the summation term
    sum_term = 0
    for j in range(1, j_max + 1):
        upper_gamma = spl.gamma(1 + j) - spl.gammainc(1 + j, s_val**2) * spl.gamma(1 + j)
        sum_term += (2 * s_val**(2 * j + 1) * np.exp(-s_val**2)) / upper_gamma

    # Multiply sum_term and product_term
    p_bar = sum_term * product_term

    return p_bar

def integrand_s_bar(s):
    """Defines the integrand function for computing \bar{s}."""

    return s * p_bar_ginue(s)

def compute_s_bar(s):
    """Numerically integrates \bar{p}_{GinUE}(s) to obtain \bar{s}."""
    s_bar = 0

    for i in range(len(s)-1):
        s_bar += (s[i+1] - s[i]) * s[i] * p_bar_ginue(s[i])

    return s_bar

def p_ginue(s):
    """Computes the GinUE nearest-neighbor spacing distribution p_{GinUE}(s)."""
    
    s_bar = compute_s_bar(s)
    return s_bar * p_bar_ginue(s_bar * s)
    
def compute_eta(unfolded_spacings, bins=100):
    """
    Compute the spectral measure η using your histogram data
    and analytic expressions for p_{2D-P}(s) and p_{GinUE}(s).

    Parameters
    ----------
    unfolded_spacing : array-like
        The unfolded spacing data from your system (not histogrammed yet).
    
    bins : int
        Number of bins to use for histogram of p(s).

    Returns
    -------
    eta : float
        Spectral measure η.
    """

    # At module level or during setup, compute and store once:
    fixed_s_vals = np.linspace(0, 3, 1000)
    p_poisson_fixed = p_2d_poissonian(fixed_s_vals)
    p_ginue_fixed = p_ginue(fixed_s_vals)
    eta_denominator = simpson((p_ginue_fixed - p_poisson_fixed)**2, x=fixed_s_vals)

    # Histogram your observed P(s)
    hist, bin_edges = np.histogram(unfolded_spacings, bins=bins, density=True)
    s_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Interpolation range (use only the meaningful support)
    s_vals = np.linspace(0, np.max(s_centres), 1000)

    # Interpolate p(s)
    f_ps = interp1d(s_centres, hist, kind='linear', bounds_error=False, fill_value=0.0)

    # Evaluate all distributions on common support
    ps_interp = f_ps(s_vals)
    p_poisson_vals = p_2d_poissonian(s_vals)
    p_ginue_vals = p_ginue(s_vals)

    # Compute numerator and denominator using Simpson's rule
    numerator = simpson((ps_interp - p_poisson_vals)**2, x = s_vals)
    eta = numerator / eta_denominator
    
    return eta

def z_avg_fun(ω, ω0, j, M_arr, g, γ, α = 0.6, rel_tol = 0.1):
    '''
    Calculates the average modulus <r> and average cosine <cos(theta)> 
    of the complex level-spacing ratio for complex spectra.

    NN = nearest neighbour (smallest distance)
    NNN = next nearest neighbour (second smallest distance)
    
    Args:
    - ω, ω0 : Parameters of the Dicke model
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    - α : Fraction of eigenvalues to keep
    - tol : Numerical tolerance
    '''

    os.makedirs("z_avg", exist_ok=True)
    os.makedirs("z", exist_ok=True)

    eigvals = Dicke_eigvals_fun(ω, ω0, j, M_arr, g, γ, α = α, rel_tol = rel_tol)
    eig_d = len(eigvals)
    z = np.zeros(eig_d, dtype=np.complex128)
    
    for i in range(eig_d):
        diffs = eigvals - eigvals[i]   # differences from Ei
        dists = np.abs(diffs)          # distances
        dists[i] = np.inf              # ignore self

        nn_ind = np.argmin(dists)       # nearest neighbour
        dists[nn_ind] = np.inf          # ignore nearest neighbour
        nnn_ind = np.argmin(dists)      # next nearest neighbour

        num = eigvals[nn_ind] - eigvals[i]
        den = eigvals[nnn_ind] - eigvals[i]
        z[i] = num / den
    
    r = np.abs(z)
    theta = np.angle(z)

    r_avg = np.average(r)
    cos_avg = np.average(np.cos(theta))

    # file_path_r = f"z_avg/r_avg_ω={ω}_ω0={ω0}_j={j}_M={M_arr[0]}_g={g}_γ={γ}_α={α}.npy"
    # file_path_cos = f"z_avg/cos_avg_ω={ω}_ω0={ω0}_j={j}_M={M_arr[0]}_g={g}_γ={γ}_α={α}.npy"
    # file_path_z = f"z/z_ω={ω}_ω0={ω0}_j={j}_M={M_arr[0]}_g={g}_γ={γ}_α={α}.npy"
    
    # if not (os.path.exists(file_path_r) or os.path.exists(file_path_cos) or os.path.exists(file_path_z)):
    #     print(f"{file_path_r} does not exist. Generating data.")
    #     z = np.zeros(eig_d, dtype=np.complex128)

    #     for i in range(eig_d):
    #         diffs = eigvals - eigvals[i]   # differences from Ei
    #         dists = np.abs(diffs)          # distances
    #         dists[i] = np.inf              # ignore self

    #         nn_ind = np.argmin(dists)       # nearest neighbour
    #         dists[nn_ind] = np.inf          # ignore nearest neighbour
    #         nnn_ind = np.argmin(dists)      # next nearest neighbour

    #         num = eigvals[nn_ind] - eigvals[i]
    #         den = eigvals[nnn_ind] - eigvals[i]
    #         z[i] = num / den
        
    #     r = np.abs(z)
    #     theta = np.angle(z)

    #     r_avg = np.average(r)
    #     cos_avg = np.average(np.cos(theta))

    #     np.save(file_path_r, r_avg)
    #     np.save(file_path_cos, cos_avg)
    #     np.save(file_path_z, z)

    # else:
    #     print(f"{file_path_r} already exists.")
    #     r_avg = np.load(file_path_r)
    #     cos_avg = np.load(file_path_cos)
    #     z = np.load(file_path_z)

    return r_avg, cos_avg

def transform_spectrum(eigvals, A=-1j, beta=0.5, sigma=None):
    """
    transform the complex spectrum using the transformation:
        z -> A * (z - z0)^beta

    Args:
    - eigvals (array-like): Array of complex eigenvalues.
    - A (complex): Scaling factor (default: -i).
    - beta (float): Exponent for the transformation (default: 0.5).
    - sigma (float): Standard deviation for local density estimation (optional).

    Returns:
    - unfolded_eigvals (array-like): Unfolded eigenvalues.
    """
    eigvals = np.array(eigvals)

    # Compute spacings
    spacings = compute_nearest_neighbor_spacings(eigvals)

    # Compute local density
    density = compute_local_density(eigvals, spacings, sigma=sigma)

    # Determine z0 as the eigenvalue corresponding to the maximum density
    z0 = eigvals[np.argmax(density)]
    # Shifted eigenvalues
    z_shifted = eigvals - z0

    # Polar form
    r = np.abs(z_shifted)
    theta = np.angle(z_shifted)
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)  # Optional: pick the (0, 2π) branch

    # Single-valued power
    z_beta = r ** beta * np.exp(1j * beta * theta)

    # Apply scaling
    unfolded_eigvals = A * z_beta

    return unfolded_eigvals

def determine_filter_params(eigvals, spacings, j, M, γ, g, eigval_type, α0):
    """
    Automatically determine Gaussian filter parameters (μ_x, μ_y, alpha_x, alpha_y)
    based on the local density of eigenvalues.

    Args:
    - eigvals (array-like): Array of complex eigenvalues (after transformation).
    - spacings (array-like): Array of spacings between eigenvalues.
    - sigma_density (float): Standard deviation for local density estimation (optional).
    - factor (float): Multiplier for sigma if not provided (default 4.5).

    Returns:
    - μ_x, μ_y, α_x, α_x: Filter parameters.
    """

    # Step 1: Compute local density
    density = compute_local_density(eigvals, spacings, eigval_type = eigval_type)

    # Step 2: Find the eigenvalue at maximum density
    peak_index = np.argmax(density)
    peak_eigval = eigvals[peak_index]
    μ_x, μ_y = np.real(peak_eigval), np.imag(peak_eigval)

    # Step 3: Find half-max width along real and imaginary directions separately
    rho_peak = density[peak_index]

    x = np.real(eigvals)
    y = np.imag(eigvals)

    # Along real axis
    mask_real = np.abs(y - μ_y) < 0.1  # small imaginary variation
    x_near = x[mask_real]
    density_near_x = density[mask_real]

    if len(x_near) > 1:
        x_sorted = x_near[np.argsort(x_near)]
        density_sorted = density_near_x[np.argsort(x_near)]
        x_left = x_sorted[density_sorted < rho_peak/2]
        ΔD_x = (np.max(x_left) - np.min(x_left)) / 2 if len(x_left) > 0 else 1.0
    else:
        ΔD_x = 1.0

    # Along imaginary axis
    mask_imag = np.abs(x - μ_x) < 0.1
    y_near = y[mask_imag]
    density_near_y = density[mask_imag]

    if len(y_near) > 1:
        y_sorted = y_near[np.argsort(y_near)]
        density_sorted = density_near_y[np.argsort(y_near)]
        y_left = y_sorted[density_sorted < rho_peak/2]
        ΔD_y = (np.max(y_left) - np.min(y_left)) / 2 if len(y_left) > 0 else 1.0
    else:
        ΔD_y = 1.0

    # Step 4: Compute α_x, α_y
    α_x = α0 / (ΔD_x ** 2)
    α_y = α0 / (ΔD_y ** 2)

    return μ_x, μ_y, α_x, α_y

def compute_dsff(ω, ω0, j, M_arr, γ, g, tlist, β, θ=0, win=100, α0=0.0, n_theta=1, α=0.6):
    """
    Computes the DSFF averaged over a small angular window around θ.
    
    Args:
        delta_theta (float): Half-width of the angular window around θ (in radians).
        n_theta (int): Number of angles in the window to average over.
    """
    δθ = 1 * np.pi / 180  # 1 degrees in radians
    
    os.makedirs("dsff_filtered", exist_ok=True)
    file_path = f"dsff_filtered/dsff_avg_ω={ω}_ω0={ω0}_j={j}_M={M_arr[0]}_g={g}_γ={γ}_θ={np.round(θ,2)}_nθ={n_theta}_win={win}_α0={α0}_α={np.round(α,2)}.npy"
    file_path_raw = f"dsff_filtered/dsff_avg_raw_ω={ω}_ω0={ω0}_j={j}_M={M_arr[0]}_g={g}_γ={γ}_θ={np.round(θ,2)}_nθ={n_theta}_win={win}_α0={α0}_α={np.round(α,2)}.npy"
    file_path_evals_len = f"dsff_filtered/evals_len_ω={ω}_ω0={ω0}_j={j}_M={M_arr[0]}_g={g}_γ={γ}_win={win}_α={np.round(α,2)}.npy"
    
    if not os.path.exists(file_path) or not os.path.exists(file_path_raw) or not os.path.exists(file_path_evals_len):
        print(f"{file_path} does not exist. Generating data.")
        eigvals = Dicke_eigvals_fun(ω, ω0, j, M_arr, g, γ, α=α)
        eigvals = transform_spectrum(eigvals, beta=1/4)
        x = eigvals.real
        y = eigvals.imag
        spacings = compute_nearest_neighbor_spacings(eigvals)
        eigval_type = 'filtered' if len(M_arr) == 1 else 'converged'
        μ_x, μ_y, α_x, α_y = determine_filter_params(eigvals, spacings, j=j, M=M_arr[0], γ=γ, g=g, eigval_type=eigval_type, α0=α0)
        print(f"Filter parameters: μ_x = {np.round(μ_x,2)}, μ_y = {np.round(μ_y,2)}, α_x = {np.round(α_x,2)}, α_y = {np.round(α_y,2)}")
        filter_fun = np.exp(-α_x * (x - μ_x)**2 - α_y * (y - μ_y)**2)

        # Discretise theta window
        if n_theta == 1:
            theta_list = [θ]
        else:
            theta_list = np.linspace(θ - δθ, θ + δθ, n_theta)

        dsff_all = np.zeros((n_theta, len(tlist)), dtype=np.float64)

        for idx, theta_val in tqdm(enumerate(theta_list)):
            proj_eigs = x * np.cos(theta_val) + y * np.sin(theta_val)

            for i, t in tqdm(enumerate(tlist), total=len(tlist), desc=f"Theta {np.round(theta_val, 2)}"):
                exp_term = np.exp(-(β - 1j * t) * proj_eigs * filter_fun)
                sff = np.abs(np.sum(exp_term))**2
                dsff_all[idx, i] = sff

        norm = np.sum(np.exp(-β * filter_fun))
        dsff_raw = np.mean(dsff_all, axis=0) / (norm**2)  # Average over θ
        dsff = dsff_rl_rect_fun(dsff_raw, win)

        np.save(file_path, dsff)
        np.save(file_path_raw, dsff_raw)
        np.save(file_path_evals_len, len(eigvals))
        print(f"Saved averaged DSFF to {file_path} and raw DSFF to {file_path_raw}.")
    else:
        print(f"{file_path} already exists. Loading data.")

    dsff = np.load(file_path)
    dsff_raw = np.load(file_path_raw)
    N = np.load(file_path_evals_len)

    return dsff, dsff_raw, N

def dsff_rl_rect_fun(dsff, win):
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
    return dsff_rl

def K_Poisson(t, N):
    """Poisson spectral form factor"""
    t = np.array(t)
    return N + (N-1)*N *np.exp(-t**2)

def theoretical_dsff_ginue(tlist, N):
    """
    Compute the theoretical DSFF for the GinUE ensemble.    
    ref1: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.127.170602
    ref2: https://arxiv.org/pdf/2306.16262
    
    Args:
    - tlist: Array of time values (τ).
    - N: Matrix dimension (number of eigenvalues).

    Returns:
    - dsff_theory: Array of theoretical DSFF values.
    """
    tau_squared = tlist**2  # since τ is real here (|τ|^2 = τ^2)
    
    term1 = 1 / N
    term2 = 4 * j1(tau_squared) / tau_squared
    term3 = (1 / N) * np.exp(-tau_squared / (4 * N))

    # Handle division by zero at τ = 0
    term2 = np.nan_to_num(term2, nan=2.0, posinf=2.0, neginf=2.0)

    dsff_theory = term1 + term2 - term3

    return dsff_theory

def generate_ginue_matrix(N):
    """
    Generate an NxN Ginibre Unitary Ensemble (GinUE) matrix.
    """
    A = np.random.normal(0, 1, size=(N, N))
    B = np.random.normal(0, 1, size=(N, N))
    A = (A + 1j*B) / np.sqrt(2)
    return A

def extract_inner_circle(eigvals, N_target):
    """
    Extract eigenvalues from a larger set, corresponding to radius ~ sqrt(N_target)
    """
    radius_target = np.sqrt(N_target)
    mask = np.abs(eigvals) <= radius_target

    return eigvals[mask]

def ginue_evals_fun(N, traj_ind):
    # N: Size of the GinUE matrix.
    os.makedirs("evals_GinUE",exist_ok=True)
    file_path = f"evals_GinUE/evals_N={N}_traj_ind={traj_ind}.npy"

    if N <= 25397: # If you generate N = 25397 already, generate N of smaller sizes
        if not os.path.exists(file_path):
            print(f"{file_path} does not exist, generating data.")
            eigvals = np.load(f"evals_GinUE/evals_N={25397}_traj_ind={traj_ind}.npy")
            eigvals = extract_inner_circle(eigvals, N)
            np.save(file_path,eigvals)
        # else:
            # print(f"{file_path} already exists.")
        eigvals = np.load(file_path)
    else:
        if not os.path.exists(file_path):
            print(f"{file_path} does not exist, generating data.")
            H = generate_ginue_matrix(N)
            time_start = time.perf_counter()
            eigvals = sl.eigvals(H)
            time_end = time.perf_counter()
            print(f"Scipy: {time_end-time_start}, eigvals: {np.shape(eigvals)}")
            np.save(file_path,eigvals)
        # else:
            # print(f"{file_path} already exists.")
        
        eigvals = np.load(file_path)

    return eigvals

def compute_single_traj(traj_ind, N, β, tlist, unfolding, thetas) :
    """Compute DSFF for a single trajectory."""
    
    print(f"traj_ind: {traj_ind}, N: {N}, β: {β}")

    eigvals = ginue_evals_fun(N, traj_ind)

    return compute_dsff(eigvals, β, tlist, thetas, unfolding)

def compute_single_traj_wrapper(args):
    """Wrapper function to unpack arguments for multiprocessing."""
    traj_ind, N, β, tlist, unfolding, thetas = args
    return compute_single_traj(traj_ind, N, β, tlist, unfolding, thetas)

def dsff_ginue_fun(N, β, tlist, ntraj, unfolding = 'no', θ1 = 0, θ2 = np.pi, n_theta=10):
    """
    Compute the Spectral Form Factor (SFF) for GOE matrices of size N,
    averaged over `ntraj` random GOE matrices using multiprocessing.
    """
    os.makedirs("dsff", exist_ok=True)

    print(f"unfolding: {unfolding}, N: {N}, ntraj: {ntraj}")

    thetas = np.linspace(θ1, θ2, n_theta)

    file_path = f"dsff/dsff_ginue_N={N}_ntraj={ntraj}"
    if unfolding == 'yes':
        file_path += f"_unfolded.npy"
    elif unfolding == 'projected unfolding':
        file_path += f"_projected_unfolding.npy"
    elif unfolding == 'no':
        file_path += f"_folded.npy"

    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        if ntraj==1:
            traj_ind = 0
            dsff = compute_single_traj(traj_ind, N, β, tlist, unfolding, thetas)
            tlist_rl, dsff_rl = dsff_rl_rect_fun(tlist, dsff, win=50)
            np.save(file_path, np.array([tlist_rl, dsff_rl]))
        else:
            # Set number of processes
            n_jobs = min(mp.cpu_count()-2, ntraj)
            args_list = [(traj_ind, N, β, tlist, unfolding, thetas) for traj_ind in (range(ntraj))]
            # Use multiprocessing Pool
            with mp.Pool(processes=n_jobs) as pool:
                dsff_results = list(pool.imap(compute_single_traj_wrapper, args_list))
            # Compute the average over all trajectories
            dsff = np.sum(dsff_results, axis=0) / ntraj
            np.save(file_path, dsff)
    else:
        print(f"{file_path} already exists.")
        if ntraj==1:
            tlist_rl, dsff_rl = np.load(file_path)
        else:
            dsff = np.load(file_path)

    if ntraj==1:
        return tlist_rl, dsff_rl
    else: 
        return tlist, dsff

def poissonian_evals_fun(N, traj_ind, radius=1.0):
    """
    Generate N uncorrelated complex eigenvalues uniformly distributed inside a disk in the complex plane.
    
    Parameters:
    - N: number of eigenvalues
    - radius: radius of the disk (default: 1.0)
    
    Returns:
    - eigenvalues: complex numpy array of length N
    """
    
    # N: Size of the GinUE matrix.
    os.makedirs("evals_Poi",exist_ok=True)
    file_path = f"evals_Poi/evals_N={N}_traj_ind={traj_ind}.npy"

    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        r = radius * np.sqrt(np.random.uniform(0, 1, N))  # square-root for uniform disk
        theta = np.random.uniform(0, 2*np.pi, N)
        eigvals = r * np.exp(1j * theta)
        np.save(file_path,eigvals)
    else:
        print(f"{file_path} already exists.")
    eigvals = np.load(file_path)
    
    return eigvals

def compute_single_traj_poissonian(traj_ind, N, β, tlist, unfolding, thetas) :
    """Compute DSFF for a single Poissonian trajectory."""

    print(f"traj_ind: {traj_ind}, N: {N}, β: {β}")
    eigvals = poissonian_evals_fun(N, traj_ind)
    return compute_dsff(eigvals, β, tlist, thetas, unfolding)

def compute_single_traj_poissonian_wrapper(args):
    """Wrapper function to unpack arguments for multiprocessing."""
    traj_ind, N, β, tlist, unfolding, thetas = args
    return compute_single_traj_poissonian(traj_ind, N, β, tlist, unfolding, thetas)

def dsff_poissonian_fun(N, β, tlist, ntraj, unfolding = 'no', θ1 = 0, θ2 = np.pi, n_theta=10):
    """
    Compute the Spectral Form Factor (SFF) for Poissonian matrices of size N,
    averaged over `ntraj` random matrices using multiprocessing.
    """
    os.makedirs("dsff", exist_ok=True)

    thetas = np.linspace(θ1, θ2, n_theta)

    print(f"unfolding: {unfolding}, N: {N}, ntraj: {ntraj}")

    file_path = f"dsff/dsff_poissonian_N={N}_ntraj={ntraj}"
    if unfolding == 'yes':
        file_path += f"_unfolded.npy"
    elif unfolding == 'no':
        file_path += f"_folded.npy"

    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        if ntraj==1:
            traj_ind = 0
            dsff = compute_single_traj_poissonian(traj_ind, N, β, tlist, unfolding, thetas)
            tlist_rl, dsff_rl = dsff_rl_rect_fun(tlist, dsff, win=50)
            np.save(file_path, np.array([tlist_rl, dsff_rl]))
        else:
            n_jobs = min(mp.cpu_count()-2, ntraj)
            args_list = [(traj_ind, N, β, tlist, unfolding, thetas) for traj_ind in (range(ntraj))]
            with mp.Pool(processes=n_jobs) as pool:
                dsff_results = list(pool.imap(compute_single_traj_poissonian_wrapper, args_list))
            dsff = np.sum(dsff_results, axis=0) / ntraj
            np.save(file_path, dsff)
    else:
        print(f"{file_path} already exists.")
        if ntraj==1:
            tlist_rl, dsff_rl = np.load(file_path)
        else:
            dsff = np.load(file_path)

    if ntraj==1:
        return tlist_rl, dsff_rl
    else: 
        return tlist, dsff