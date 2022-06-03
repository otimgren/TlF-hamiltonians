"""
Writing down operators functions for the Zeeman Hamiltonian for the B state
"""
import numpy as np
from centrex_TlF import State, UncoupledBasisState
from centrex_TlF.constants import constants_B as cst_B
from centrex_TlF.hamiltonian.utils import threej_f

from ..utils import state_operator


def mu_p(p:int, psi: UncoupledBasisState, mu_B: float = 1.4e6) -> State:
    """
    Operates on psi using the pth spherical tensor component of the magnetic
    dipole operator.

    mu_B = Bohr magneton in Hz/Gauss
    """
    #Find the quantum numbers of the input state
    Jp = psi.J
    mJp = psi.mJ
    I1p = psi.I1
    m1p = psi.m1
    I2p = psi.I2
    m2p = psi.m2
    Omegap = psi.Omega
    
    # Some constants
    S = 1
    gL = 1 
    
    #The other state must have the same value for I1,m1,I2,m2,mJ and Omega
    I1 = I1p
    m1 = m1p
    I2 = I2p
    m2 = m2p
    Omega = Omegap
    mJ = mJp+p

    #Initialize container for storing states and matrix elements
    data = []

    #Loop over possible values of J
    for J in np.arange(np.abs(Jp-1),Jp+2).tolist():
        #Electron orbital angular momentum term
        L_term = (gL * mu_B * (-1)**(mJ-Omega) * Omega *np.sqrt((2*J+1)*(2*Jp+1))
                    * threej_f(J,1,Jp,-mJ,p,mJp) * threej_f(J,1,Jp,-Omega,0,Omegap))

        amp = L_term
        basis_state = UncoupledBasisState(J, mJ, I1, m1, I2, m2, Omega)
        
        
        if amp != 0:
            data.append((amp, basis_state))
                  
                  
    return State(data)    

@state_operator
def HZx(psi:UncoupledBasisState) -> State:
    """
    Zeeman Hamiltonian operator for x-component of magnetic field
    """
    return -( mu_p(-1,psi) - mu_p(+1,psi) ) / np.sqrt(2)

@state_operator
def HZy(psi:UncoupledBasisState) -> State:
    """
    Zeeman Hamiltonian operator for y-component of magnetic field
    """
    return - 1j * ( mu_p(-1,psi) + mu_p(+1,psi) ) / np.sqrt(2)

@state_operator
def HZz(psi:UncoupledBasisState) -> State:
    """
    Zeeman Hamiltonian for z-component of magnetic field
    """
    return - mu_p(0,psi)
