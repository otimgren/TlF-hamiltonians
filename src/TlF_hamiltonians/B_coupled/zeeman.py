"""
Writing down operators functions for the Zeeman Hamiltonian for the B state
"""
import numpy as np
from centrex_TlF import CoupledBasisState, State
from centrex_TlF.constants import constants_B as cst_B
from centrex_TlF.hamiltonian.utils import sixj_f, threej_f

from ..utils import state_operator


def mu_p(p:int, psi: CoupledBasisState, mu_B: float = 1.4e6) -> State:
    """
    Operates on psi using the pth spherical tensor component of the magnetic
    dipole operator.

    mu_B = Bohr magneton in Hz/Gauss
    """
    # Some constants
    gL = 1

    # Find the quantum numbers of the input state
    Jp = psi.J
    I1p = psi.I1
    I2p = psi.I2
    F1p = psi.F1
    Fp = psi.F
    mFp = psi.mF
    Omegap = psi.Omega
    
    #I1, I2 are the same for both states
    I1 = I1p
    I2 = I2p

    # Value of mF changes by p
    mF = mFp + p

    # Omega doesn't change
    Omega = Omegap
    
    #Initialize container for storing states and matrix elements
    data = []
    
    #Loop over possible values of Jprime
    for J in np.arange(np.abs(Jp-1),Jp+2).tolist():
        # Loop over possible values of F1
        for F1 in np.arange(np.abs(J-I1), J+I1+1).tolist():
            # Loop over possible values of F
            for F in np.arange(np.abs(F1-I2), F1+I2+1).tolist():
                amp = (
                    gL* Omega * mu_B * (-1)**(F+Fp+F1+F1p+I1+I2-Omega-mF)
                    *np.sqrt((2*F+1)*(2*Fp+1)*(2*F1+1)*(2*F1p+1)*(2*J+1)*(2*Jp+1))
                    *threej_f(F,1,Fp,-mF,p,mFp)*threej_f(J,1,Jp,-Omega,0,Omegap)
                    *sixj_f(F1p, Fp, I2, F, F1, 1)*sixj_f(Jp, F1p, I1, F1,J,1)
                )

                basis_state = CoupledBasisState(F, mF, F1, J, I1, I2, Omega = Omega)
                if amp != 0:
                    data.append((amp, basis_state))

    return State(data)  

@state_operator
def HZx(psi:CoupledBasisState) -> State:
    """
    Zeeman Hamiltonian operator for x-component of magnetic field
    """
    return -( mu_p(-1,psi) - mu_p(+1,psi) ) / np.sqrt(2)

@state_operator
def HZy(psi:CoupledBasisState) -> State:
    """
    Zeeman Hamiltonian operator for y-component of magnetic field
    """
    return - 1j * ( mu_p(-1,psi) + mu_p(+1,psi) ) / np.sqrt(2)

@state_operator
def HZz(psi:CoupledBasisState) -> State:
    """
    Zeeman Hamiltonian for z-component of magnetic field
    """
    return - mu_p(0,psi)
