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
    print(f"j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω/ω0*(γ**2/4+ω**2))/2,2)}_γ={γ}_g={g}")
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

def filter_eigenvals(j, M, γ, eigvals, α = 0.8):
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

    return np.array(converged)

def find_nearest(target, array):
    """Helper function to find closest eigenvalue"""
    return array[np.argmin(np.abs(array - target))]

def compute_nearest_neighbor_spacings(eigvals, j=None, M=None, γ=None, g=None, save_dir="spacings"):
    """
    Computes the nearest-neighbor spacings for a given set of complex eigenvalues.
    Optionally saves/loads spacings to/from disk using parameters j, M, γ, g.
    """
    if all(param is not None for param in [j, M, γ, g]):
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(
            save_dir,
            f"spacings_j={j}_M={M}_γ={γ}_g={g}.npy"
        )
        if os.path.exists(file_path):
            print(f"Loading spacings from {file_path}")
            return np.load(file_path)
    else:
        file_path = None

    N = len(eigvals)
    spacings = np.zeros(N)
    print("Computing the Nearest Neighour Distance")
    for i in tqdm(range(N)):
        distances = np.abs(eigvals - eigvals[i])  # Compute all distances
        distances[i] = np.inf  # Ignore self-distance
        spacings[i] = np.min(distances)  # Find the nearest neighbor

    if file_path is not None:
        np.save(file_path, spacings)

    return spacings

def compute_local_density(eigvals, spacings, sigma=None, j=None, M=None, γ=None, g=None, save_dir="local_density"):
    """
    Estimates the local density ρ_av(E).
    Optionally saves/loads density to/from disk using parameters j, M, γ, g.
    """
    if all(param is not None for param in [j, M, gamma, g]):
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(
            save_dir,
            f"local_density_j={j}_M={M}_γ={γ}_g={g}.npy"
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

def filter_circular_patch(eigvals, min_count=10000, initial_radius=1, radius_step=1, max_radius=20, centre=None):
    """
    Filters a circular patch of eigenvalues around the mean or given centre.

    Parameters:
        eigvals (np.ndarray): Complex eigenvalues.
        min_count (int): Minimum number to retain.
        initial_radius (float): Starting radius.
        radius_step (float): Incremental expansion.
        max_radius (float): Maximum search radius.
        centre (complex or None): Optional centre point. Defaults to np.mean(eigvals).

    Returns:
        np.ndarray: Filtered eigenvalues inside the circular patch.
    """
    if eigvals.size == 0:
        return eigvals

    if centre is None:
        centre = np.mean(eigvals)

    radius = initial_radius
    while radius <= max_radius:
        distances = np.abs(eigvals - centre)
        mask = distances <= radius
        selected = eigvals[mask]
        if selected.size >= min_count:
            print(f"✓ Selected {selected.size} eigenvalues within radius {radius:.3f} around centre {centre:.3f}")
            return selected
        radius += radius_step

    print(f"⚠️ Only {selected.size} eigenvalues found within max radius {max_radius}")
    return selected


def unfold_spacings(eigvals, j=None, M=None, γ=None, g=None):
    """Performs the unfolding procedure following equation (B1) in PhysRevA.105.L050201."""
    eigvals = np.sort(eigvals)
    spacings = compute_nearest_neighbor_spacings(eigvals, j = j, M = M, γ = γ, g = g)[:-1]

    local_density = compute_local_density(eigvals[:-1], spacings, j = j, M = M, γ = γ, g = g)  # Compute density only for spacing indices
    unfolded_spacings = spacings * np.sqrt(local_density)  # Apply unfolding
    s_bar = np.mean(unfolded_spacings)  # Compute global mean level spacing
    # print(f"Unfolded spacing: {s_bar}")
    unfolded_spacings = unfolded_spacings/s_bar # Normalise
    # print(f"Unfolded spacing After Normalisation {np.mean(unfolded_spacings)}")

    return unfolded_spacings

def unfold_spectrum(eigvals, num_y_bins=50, num_x_bins=100, poly_order=4):
    """
    Unfold complex eigenvalues using horizontal slices (fixed y), where ρ_av(x, y)
    is estimated via polynomial fit to histogram and then integrated.

    Args:
        eigvals (np.ndarray): Complex eigenvalues.
        num_y_bins (int): Number of horizontal bins along Im axis.
        num_x_bins (int): Number of bins along Re axis for histogramming.
        poly_order (int): Degree of polynomial for density fitting.

    Returns:
        np.ndarray: Unfolded complex eigenvalues.
    """
    eigvals = np.array(eigvals)
    x = np.real(eigvals)
    y = np.imag(eigvals)
    unfolded_x = np.zeros_like(x)

    y_bins = np.linspace(y.min(), y.max(), num_y_bins + 1)
    bin_indices = np.digitize(y, y_bins) - 1

    for i in range(num_y_bins):
        mask = bin_indices == i
        if np.sum(mask) < poly_order + 3:
            continue

        x_bin = x[mask]
        y_bin = y[mask]

        # Histogram to estimate raw density
        hist_vals, bin_edges = np.histogram(x_bin, bins=num_x_bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Fit low-order polynomial to estimated density
        coeffs = np.polyfit(bin_centers, hist_vals, deg=poly_order)
        rho_poly = np.poly1d(coeffs)

        # Define fine x-grid and integrate fitted density
        x_grid = np.linspace(x_bin.min() - 1e-3, x_bin.max() + 1e-3, 1000)
        rho_vals = rho_poly(x_grid)
        rho_vals[rho_vals < 0] = 0  # clip small negative values

        cumulative = cumulative_trapezoid(rho_vals, x_grid, initial=0)

        # Interpolate unfolded x values
        unfolded_x_vals = np.interp(x_bin, x_grid, cumulative)
        unfolded_x[mask] = unfolded_x_vals

    unfolded_eigvals = unfolded_x + 1j * y
    return unfolded_eigvals

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


def unfold_poly(eigvals, deg=6):
    """
    Unfolds the eigenvalue spectrum polynomially and returns the unfolded spectrum.

    Args:
    - eigvals (array-like): List of eigenvalues.
    - deg (int): Degree of polynomial for fitting.

    Returns:
    - unfolded (np.ndarray): Unfolded spectrum.
    """
    eigvals = np.sort(eigvals)  # Ensure they are sorted
    indices = np.arange(1, len(eigvals) + 1)

    coeffs = np.polyfit(eigvals, indices, deg)
    poly = np.poly1d(coeffs)

    return poly(eigvals)

def plot_spectrum(eigvals):
    plt.xlabel("Re E")
    plt.ylabel("Im E")
    plt.scatter(eigvals.real, eigvals.imag, marker=".")

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

def plot_ps_distribution(spacings, bins=60):
    """Plot nearest-neighbor spacing distribution P(s) for complex eigenvalues."""

    # Histogram of nearest-neighbor distances
    hist, bin_edges = np.histogram(spacings, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot P(s) distribution
    plt.hist(bin_centers, bins=bins, weights=hist, histtype='step', label='P(s) distribution')

    plt.xlim(0, 3)

    # Comparison with 2D Poissonian and GinUE distributions
    s_vals = np.linspace(0, 3, 1000)
    plt.plot(s_vals, p_2d_poissonian(s_vals), label='2D Poisson', linestyle='dashed')
    plt.plot(s_vals, p_ginue(s_vals), label='GinUE', linestyle='dotted')

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

def compute_dsff_theta_avg(eigvals, β, tlist, thetas, unfolding):
    """
    Compute the DSFF averaged over multiple projection angles θ.
    
    Args:
    - eigvals: Complex eigenvalues.
    - β: Inverse temperature.
    - tlist: Time array.
    - thetas: Array of projection angles in radians.
    
    Returns:
    - Averaged DSFF over all thetas.
    """
    # eigvals = transform_spectrum(eigvals)

    if unfolding == "yes":
        eigvals = unfold_spectrum(eigvals)
    elif unfolding == "no":
        eigvals = eigvals

    x = np.real(eigvals)
    y = np.imag(eigvals)
    
    tlist = np.array(tlist)
    dsff_avg = np.zeros(len(tlist), dtype=np.float64)
    
    for theta_ind, theta in enumerate(thetas):
        print(f"theta_ind = {theta_ind} of {len(thetas)}")
        proj_eigs = x * np.cos(theta) + y * np.sin(theta)
        norm = np.sum(np.exp(-β * proj_eigs))  # scalar
        for i, t in enumerate(tlist):
            exp_term = np.exp(-(β - 1j * t) * proj_eigs)
            sff = np.abs(np.sum(exp_term))**2
            dsff_avg[i] += sff / (norm**2)  # add contribution

    dsff_avg /= len(thetas)

    return dsff_avg

def dsff_fun_theta_avg(ω, ω0, j, M_arr, g, β, γ, tlist, win = 100, σ = 1, kernel = "rect", α = 0.8, unfolding="no", n_theta=10, θ1=np.pi/2, θ2=np.pi):
    """
    Computes angle-averaged DSFF for multiple M values or a single one.
    """

    os.makedirs("dsff", exist_ok=True)
    thetas = np.linspace(θ1, θ2, n_theta)
    prefix = f"j={j}_M={M_arr[0]}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω/ω0*(γ**2/4+ω**2))/2, 2)}_β={β}_g={g}_γ={γ}_θ={np.round(θ2,2)}_ntheta={n_theta}"

    # Selecting only converged eigenvalues
    if len(M_arr) == 3:
        eigvals_list = []
        for M in M_arr:
            eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ)
            eigvals_list.append(eigvals)
        eigvals = find_converged_eigvals(eigvals_list, j=j, M_arr=M_arr, γ=γ, g=g)
    else:
        eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M_arr[0], g, γ)
        eigvals = filter_eigenvals(j, M_arr[0], γ, eigvals)
        prefix += f"_α={np.round(α,2)}"

    # Unfolding the spectrum
    if unfolding == 'yes':
        suffix = "unfolded"
    elif unfolding == 'projected unfolding':
        suffix = "projected_unfolding"
    elif unfolding == 'no':
        suffix = "folded"

    N = len(eigvals)
    
    # Averaging the DSFF
    if kernel == 'rect':
        file_path = f"dsff/dsff_{prefix}_kernel={kernel}_win={win}_{suffix}.npy"
        file_path_raw = f"dsff/dsff_raw_{prefix}_kernel={kernel}_win={win}_{suffix}.npy"
    elif kernel == 'gau':
        file_path = f"dsff/dsff_{prefix}_kernel={kernel}_σ={σ}_{suffix}.npy"
        file_path_raw = f"dsff/dsff_raw_{prefix}_kernel={kernel}_σ={σ}_{suffix}.npy"

    # Computing or loading the DSFF
    if not os.path.exists(file_path) or not os.path.exists(file_path_raw):
        print(f"{file_path} does not exist, generating data.")
        print(f"Computing DSFF averaged over θ ∈ [{θ1}, {θ2}], n_theta = {n_theta}")
        dsff_raw = compute_dsff_theta_avg(eigvals, β, tlist, thetas, unfolding)
        np.save(file_path_raw, dsff_raw)

        if kernel == 'rect':
            tlist_dsff = dsff_rl_rect_fun(tlist, dsff_raw, win)
        elif kernel == 'gau':
            tlist_dsff = dsff_rl_gau_fun(tlist, dsff_raw, σ)
        
        np.save(file_path, tlist_dsff)
    else:
        tlist_dsff = np.load(file_path)
        dsff_raw = np.load(file_path_raw)

    tlist = tlist_dsff[0]
    dsff = tlist_dsff[1]

    return tlist, dsff, dsff_raw, N

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

    return compute_dsff_theta_avg(eigvals, β, tlist, thetas, unfolding)

def compute_single_traj_wrapper(args):
    """Wrapper function to unpack arguments for multiprocessing."""
    traj_ind, N, β, tlist, unfolding, thetas = args
    return compute_single_traj(traj_ind, N, β, tlist, unfolding, thetas)

def dsff_ginue_fun(N, β, tlist, ntraj, unfolding = 'no'):
    """
    Compute the Spectral Form Factor (SFF) for GOE matrices of size N,
    averaged over `ntraj` random GOE matrices using multiprocessing.
    """
    os.makedirs("dsff", exist_ok=True)

    print(f"unfolding: {unfolding}, N: {N}, ntraj: {ntraj}")

    θ2 = np.pi / 2
    θ1 = 2 * θ2 / 2.1
    n_theta = 10
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
    return compute_dsff_theta_avg(eigvals, β, tlist, thetas, unfolding)

def compute_single_traj_poissonian_wrapper(args):
    """Wrapper function to unpack arguments for multiprocessing."""
    traj_ind, N, β, tlist, unfolding, thetas = args
    return compute_single_traj_poissonian(traj_ind, N, β, tlist, unfolding, thetas)

def dsff_poissonian_fun(N, β, tlist, ntraj, unfolding = 'no'):
    """
    Compute the Spectral Form Factor (SFF) for Poissonian matrices of size N,
    averaged over `ntraj` random matrices using multiprocessing.
    """
    os.makedirs("dsff", exist_ok=True)

    θ2 = np.pi / 2
    θ1 = 2 * θ2 / 2.1
    n_theta = 10
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