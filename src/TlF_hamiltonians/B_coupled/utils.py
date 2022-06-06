from typing import Callable, List

import numpy as np
from centrex_TlF import State

from ..utils import calculate_matrix_reps
from .ld import H_c1p, H_q
from .mhf import H_mhf_F, H_mhf_Tl
from .nsr import Hc1
from .rotational import Hrot
from .stark import HSx, HSy, HSz
from .zeeman import HZx, HZy, HZz


def make_H0(QN: List[State]) -> np.ndarray:
    """
    Returns the field free Hamiltonian
    """
    H_list = [H_c1p, H_mhf_F, H_mhf_Tl, H_q, Hc1, Hrot]
    matrix_reps = calculate_matrix_reps(H_list, QN)
    return sum(matrix_reps)

def make_H_EB(QN: List[State]) -> Callable:
    """
    Returns a function that gives the Hamiltonian as function of electric and
    magnetic field.
    """
    H0 = make_H0(QN)
    MSx, MSy, MSz = calculate_matrix_reps([HSx, HSy, HSz], QN)
    MZx, MZy, MZz = calculate_matrix_reps([HZx, HZy, HZz], QN)
    H_EB = lambda E, B: (H0 + E[0]*MSx + E[1]*MSy + E[2]*MSz
                            + B[0]*MZx + B[1]*MZy + B[2]*MZz)

    return H_EB

