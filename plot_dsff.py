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

def plot_dicke_dsff(tlist, dsff_ginue):
    
    for M in M_arr:
        num_g = len(g_arr)
        num_rows = (num_g + 1) // 2
        plt.figure(figsize=(10,5*num_rows))
        for g_ind, g in enumerate(g_arr):
            tlist, dsff, dsff_raw = dsff_fun(ω, ω0, j, M, g, β, γ, tlist_ginue, tlist, axis, win, σ, kernel)
            plt.subplot(num_rows,2,g_ind+1)
            plt.title(f"g={g}")
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Time"); plt.ylabel("sff")
            plt.xlim(1e-3,1e3)#; plt.ylim(1e-12,2e0)
            # Plot raw data
            plt.plot(tlist,dsff_raw,color='0.8')
            # Plot moving average
            plt.plot(tlist,dsff,label=f"Dicke Model")
            # Plot GinUE
            plt.plot(tlist,dsff_ginue,'--k',label=f"GinUE")
            plt.tight_layout()
            plt.grid(True)
        if not os.path.exists("plots"):
            os.mkdir("plots")
        plt.legend()
        plt.savefig(f'plots/Dicke_dsff_j={j}_M={M}_β={β}_gc={gc}_axis={axis}.png')
        plt.show()

    return 0

def plot_ramp_slope(m_arr, m_ginue):
    plt.figure(figsize=(8,5))
    # plt.title(f"Dicke Model with Cavity Decay "+r"($\gamma=1$)")
    plt.plot(g_arr,m_arr,'.-')
    plt.xlabel('g')
    plt.ylabel('slope of the ramp')
    plt.axhline(m_ginue, linestyle='--', color="k", label="GinUE")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/Dicke_dsff_j={j}_slope_ramp_vs_g_axis={axis}")
    plt.show()

def main():
    parallel_SFF(ω, ω0, j, M_arr, g_arr, β, γ, tlist_ginue, tlist, axis, win, σ, kernel)
    for M in M_arr:
        tlist_ginue1, dsff_ginue = dsff_ginue_fun(j, M, β, tlist_ginue, tlist, axis, win, σ, kernel)
        plot_dicke_dsff(tlist_ginue1, dsff_ginue)

    return 0

if __name__ == '__main__':
    main()