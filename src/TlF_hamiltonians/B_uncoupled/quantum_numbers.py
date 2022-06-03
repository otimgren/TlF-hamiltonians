from typing import List

import numpy as np
from centrex_TlF import UncoupledBasisState


def generate_QN(Jmin:int = 1, Jmax:int = 10) -> List[UncoupledBasisState]:
    """
    Generates a basis of all states with mF = mJ+m1+m2 and Jmin <= J <= Jmax
    """
    Omegas = [-1.,1.]
    I_Tl = 0.5
    I_F = 0.5
    
    QN = [1*UncoupledBasisState(J,mJ,I_Tl,m1,I_F,m2, Omega)
        for J  in np.arange(Jmin, Jmax+1).tolist()
        for mJ in np.arange(-J,J+1).tolist()
        for m1 in np.arange(-I_Tl,I_Tl+1).tolist()
        for m2 in np.arange(-I_F,I_F+1).tolist()
        for Omega in Omegas
        ]
    return QN

def generate_QN_mF(Jmin:int = 1, Jmax:int = 10, mF = 0) -> List[UncoupledBasisState]:
    """
    Generates a basis of all states with mF = mJ+m1+m2 and Jmin <= J <= Jmax
    """
    Omegas = [-1,1]
    I_Tl = 0.5
    I_F = 0.5
    
    QN = []
    for Omega in Omegas:
        for J in np.arange(Jmin, Jmax+1).tolist():
            for m1 in np.arange(-I_Tl,I_Tl+1).tolist():
                for m2 in np.arange(-I_F,I_F+1).tolist():
                    mJ = mF-m1-m2
                    if np.abs(mJ) <= J:
                        QN.append(1*UncoupledBasisState(J,mJ,I_Tl,m1,I_F,m2, Omega))

    return QN
    
