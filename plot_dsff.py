import matplotlib.pyplot as plt
from dicke_Liou_lib import *
from parameters import *

def main():
    sff_ginue_list = dsff_ginue_list_fun(j, M, β, tlist)
    plt.figure(figsize=(8,5))
    # Isolate the ramp
    indl = int(2e3); indr = int(22e2)
    m,b = np.polyfit(np.log10(tlist[indl:indr]),np.log10(sff_ginue_list[indl:indr]),1)
    m_ginue = m
    # plt.plot(tlist[indl:indr], sff_ginue_list[indl:indr],label=f"Ramp")
    plt.plot(tlist, sff_ginue_list,label=f"GinUE")
    plt.plot(tlist[indl:indr], sff_ginue_list[indl:indr],label=f"Ramp")
    plt.plot(tlist, 10**b*tlist**m,'--')
    plt.savefig(f'plots/ginue_dsff_j={j}_M={M}_β={β}_gc={gc}.png')
    plt.xscale('log'); plt.yscale('log')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,5))
    m_arr = []
    for g_ind, g in enumerate(g_arr):
        dsff_list = dsff_list_fun(ω, ω0, j, M, g, β, γ, tlist)
        dsff_rl = dsff_rl_fun(ω, ω0, j, M, g, β, γ, tlist, win=100)
        plt.subplot(20,2,g_ind+1)
        plt.title(f"g={g}")
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel("Time"); plt.ylabel("sff")
        plt.xlim(1e-3,1e3); plt.ylim(1e-12,2e0)
        # Plot raw data
        plt.plot(tlist,dsff_list,color='0.8')
        # Plot GinUE
        plt.plot(tlist, sff_ginue_list,'--k',label=f"GinUE")
        # Plot moving average
        plt.plot(tlist,dsff_rl,label=f"Dicke Model")
        m,b = np.polyfit(np.log10(tlist[indl:indr]),np.log10(dsff_rl[indl:indr]),1)
        plt.plot(tlist[indl:indr], 10**b*tlist[indl:indr]**m,'--')
        m_arr.append(m)
        plt.tight_layout()
        plt.grid(True)
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.legend()
    plt.savefig(f'plots/Dicke_dsff_j={j}_M={M}_β={β}_gc={gc}.png')
    # plt.show()

    plt.figure(figsize=(8,5))
    # plt.title(f"Dicke Model with Cavity Decay "+r"($\gamma=1$)")
    plt.plot(g_arr,m_arr,'.-')
    plt.xlabel('g')
    plt.ylabel('slope of the ramp')
    plt.axhline(m_ginue, linestyle='--', color="k", label="GinUE")
    plt.legend()
    plt.grid()
    # plt.savefig(f"plots/Dicke_dsff_j={j}_slope_ramp_vs_g")
    plt.show()

if __name__ == '__main__':
    main()