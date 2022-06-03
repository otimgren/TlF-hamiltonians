from typing import List

import numpy as np
from centrex_TlF import CoupledBasisState


def generate_QN(Jmin:int = 1, Jmax:int = 10) -> List[CoupledBasisState]:
    """
    Generates a basis of all states with Jmin <= J <= Jmax
    """
    Omegas = [-1,1]
    I_Tl = 0.5
    I_F = 0.5
    
    QN = [
        1*CoupledBasisState(F,mF,F1,J,I_F,I_Tl, Omega=Omega)
        for J  in np.arange(Jmin, Jmax+1).tolist()
        for F1 in np.arange(np.abs(J-I_F),J+I_F+1).tolist()
        for F in np.arange(np.abs(F1-I_Tl),F1+I_Tl+1).tolist()
        for mF in np.arange(-F, F+1).tolist()
        for Omega in Omegas
    ]

    return QN

def generate_QN_mF(Jmin:int = 1, Jmax:int = 10, mF = 0) -> List[CoupledBasisState]:
    """
    Generates a basis of all states with mF = mJ+m1+m2 and Jmin <= J <= Jmax
    """
    Omegas = [-1,1]
    I_Tl = 0.5
    I_F = 0.5
    
    QN = [
        1*CoupledBasisState(F,mF,F1,J,I_F,I_Tl, Omega=Omega)
        for J  in np.arange(Jmin, Jmax+1).tolist()
        for F1 in np.arange(np.abs(J-I_F),J+I_F+1).tolist()
        for F in np.arange(np.abs(F1-I_Tl),F1+I_Tl+1).tolist()
        for Omega in Omegas
    ]

    return QN
    
