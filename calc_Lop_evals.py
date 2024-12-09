from parameters import *
import numpy as np
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as sla
import os
from tqdm import tqdm
import qutip as qt
import scipy.linalg as sl
from dicke_Liou_lib import *


for g in tqdm(g_arr):
    Dicke_Lop_even_evals_fun(ω, ω0, g, M, j, γ)