from dicke_Liou_lib import *
from parameters import *
import matplotlib.pyplot as plt

j = 5
M = 10

# dsff_ginue_list = dsff_ginue_list_fun(j, M, β, tlist)
N = 100
sff_goe_list = sff_goe_list_fun(N, β, tlist)
plt.plot(tlist,sff_goe_list,label=f"GOE N={N}")
N = 1000
sff_goe_list = sff_goe_list_fun(N, β, tlist)
plt.plot(tlist,sff_goe_list,label=f"GOE N={N}")
# plt.plot(tlist,dsff_ginue_list,label="GinUE")
plt.xscale('log');plt.yscale('log')
plt.legend()
plt.show()