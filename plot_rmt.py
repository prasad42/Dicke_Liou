from dicke_Liou_lib import *
from parameters import *
import matplotlib.pyplot as plt

j = 5
M = 30

sff_ginue_list = dsff_ginue_list_fun(j, M, β, tlist)
N = int(((2*j+1)*M)**2/2)
plt.plot(tlist,sff_ginue_list,label=f"GinUE N={N}")
sff_goe_list = sff_goe_list_fun(N, β, tlist, ntraj=10)
plt.plot(tlist,sff_goe_list,label=f"GOE N={N}")
# plt.plot(tlist,dsff_ginue_list,label="GinUE")
plt.xscale('log');plt.yscale('log')
plt.legend()
plt.show()