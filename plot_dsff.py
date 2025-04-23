import matplotlib.pyplot as plt
from dicke_Liou_lib import *
from parameters import *

def select_ramp_fun(tlist, dsff):

    def on_click(event):
        """
        Capture the x-coordinate of a click event.
        """
        global x_click
        x_click = event.xdata
        plt.close()

    global x_click
    x_click = None

    # Plot for selecting the start of the ramp
    plt.figure(figsize=(8, 5))
    plt.title("Click on the start of the ramp")
    plt.plot(tlist, dsff)
    plt.yscale('log')
    plt.xscale('log')
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    if x_click is None:
        raise ValueError("No point was clicked for the start of the ramp.")
    ramp_start = x_click
    ramp_start_idx = np.argwhere(np.isclose(tlist, ramp_start, atol=1e-1)).flatten()
    if len(ramp_start_idx) == 0:
        raise ValueError(f"No value in tlist close to ramp_start={ramp_start}.")
    ramp_start_idx = int(ramp_start_idx[0])
    print(f"Ramp start index: {ramp_start_idx}")

    # Plot for selecting the end of the ramp
    plt.figure(figsize=(8, 5))
    plt.title("Click on the end of the ramp")
    plt.plot(tlist, dsff)
    plt.yscale('log')
    plt.xscale('log')
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    if x_click is None:
        raise ValueError("No point was clicked for the end of the ramp.")
    ramp_end = x_click
    ramp_end_idx = np.argwhere(np.isclose(tlist, ramp_end, atol=1e-1)).flatten()
    if len(ramp_end_idx) == 0:
        raise ValueError(f"No value in tlist close to ramp_end={ramp_end}.")
    ramp_end_idx = int(ramp_end_idx[0])
    print(f"Ramp end index: {ramp_end_idx}")

    # Compute the slope of the ramp
    m, b = np.polyfit(
        np.log10(tlist[ramp_start_idx:ramp_end_idx]),
        np.log10(dsff[ramp_start_idx:ramp_end_idx]),
        1
    )

    print(f"Slope of the ramp: {m}")

    # Highlight the selected start and end points
    plt.figure(figsize=(8, 5))
    plt.plot(tlist, dsff, label="SFF")
    plt.plot(tlist[ramp_start_idx], dsff[ramp_start_idx], marker='o', color='red', markersize=10, label='Start Point')
    plt.plot(tlist[ramp_end_idx], dsff[ramp_end_idx], marker='o', color='blue', markersize=10, label='End Point')
    plt.plot(tlist, 10 ** (m * np.log10(tlist)+b), linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Selected Ramp Points")
    plt.legend()
    plt.grid(True)
    plt.show()

    return m, ramp_start_idx, ramp_end_idx

def plot_dicke_dsff():
    for γ_ind, γ in enumerate(γ_arr):
        gc = gc_arr[γ_ind]
        num_g = len(g_arr[γ])
        num_rows = (num_g + 1) // 2
        plt.figure(figsize=(10,5*num_rows))
        for g_ind, g in enumerate(g_arr[γ]):
            tlist2, dsff, dsff_raw, N = dsff_fun_theta_avg(ω, ω0, j, M_arr, g, β, γ, tlist, win, unfolding = "yes")
            tlist1, dsff_ginue = dsff_ginue_fun(N, β, tlist, ntraj, unfolding = "yes")
            tlist3, dsff_poissonian = dsff_poissonian_fun(N, β, tlist, ntraj, unfolding = "yes")
            plt.subplot(num_rows,2,g_ind+1)
            plt.title(f"g={g}")
            plt.xscale('log'); plt.yscale('log')
            plt.xlabel("Time"); plt.ylabel("sff")
            # plt.xlim(1e-3,1e3); plt.ylim(1e-11,2e0)
            # Plot raw data
            plt.plot(tlist2,dsff_raw,color='0.8')
            # Plot GinUE
            plt.plot(tlist1,dsff_ginue,'--k',label=f"GinUE")
            # Plot Poissonian
            plt.plot(tlist3,dsff_poissonian,'--r',label=f"Poissonian")
            # Plot moving average
            plt.plot(tlist2,dsff,label=f"Dicke Model")
            plt.tight_layout()
            plt.grid(True)
        if not os.path.exists("plots"):
            os.mkdir("plots")
        plt.legend()
        plt.savefig(f'plots/Dicke_dsff_j={j}_M={M_arr[0]}_β={β}_γ={γ}_gc={gc}.png')
        plt.show()

    return 0

# def plot_dicke_spectrum():
#     plt.figure(figsize=(10, 8))
#     for g_ind, g in enumerate(g_arr):
#         if len(M_arr) == 1:
#             eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M_arr[0], g, γ)
#             eigvals = filter_eigenvals(j, M_arr[0], γ, eigvals)
#         elif len(M_arr) >= 3:
#             eigvals_list = [Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ) for M in M_arr]
#             eigvals = find_converged_eigvals(eigvals_list, rel_tol=0.1, abs_tol=1e-6)
#         eigvals_unfolded, density = unfold_spectrum(eigvals)
        
#         # Raw spectrum
#         plt.subplot(2, len(g_arr), g_ind + 1)
#         plt.title(f"Raw Spectrum, g={g}")
#         plt.scatter(eigvals.real, eigvals.imag, s=1, alpha=0.5)
#         plt.xlabel("Re E"); plt.ylabel("Im E")
        
#         # Unfolded spectrum
#         plt.subplot(2, len(g_arr), len(g_arr) + g_ind + 1)
#         plt.title(f"Unfolded Spectrum, g={g}")
#         plt.scatter(eigvals_unfolded.real, eigvals_unfolded.imag, s=1, alpha=0.5)
#         plt.xlabel("Re E"); plt.ylabel("Im E")
    
#     plt.tight_layout()
#     plt.savefig(f'plots/Dicke_spectrum_j={j}_M={M_arr[0]}_γ={γ}_gc={gc}.png')
#     plt.show()

def plot_ramp_slope(m_arr, m_ginue):
    plt.figure(figsize=(8,5))
    # plt.title(f"Dicke Model with Cavity Decay "+r"($\gamma=1$)")
    plt.plot(g_arr,m_arr,'.-')
    plt.xlabel('g')
    plt.ylabel('slope of the ramp')
    plt.axhline(m_ginue, linestyle='--', color="k", label="GinUE")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/Dicke_dsff_j={j}_slope_ramp_vs_g")
    plt.show()

def main():
    
    # parallel_SFF(ω, ω0, j, M_arr, g_arr, β, γ, tlist, axis, win, σ, kernel, α, unfolding)
    plot_dicke_dsff()
    # plot_dicke_spectrum()

    return 0

if __name__ == '__main__':
    main()