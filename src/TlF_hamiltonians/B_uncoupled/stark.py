"""
Writing down operators for the Stark Hamiltonian
"""
import numpy as np
from centrex_TlF import State, UncoupledBasisState
from centrex_TlF.constants import constants_B as cst_B
from centrex_TlF.hamiltonian.utils import threej_f

from ..utils import state_operator


def d_p(p:int, psi: UncoupledBasisState, mu_e: float = cst_B.mu_e) -> State:
    """
    Operates on psi using the pth spherical tensor component of the
    dipole operator.
    """
    #Find the quantum numbers of the input state
    Jp = psi.J
    mJp = psi.mJ
    I1p = psi.I1
    m1p = psi.m1
    I2p = psi.I2
    m2p = psi.m2
    Omegap = psi.Omega
    
    #The other state must have the same value for I1,m1,I2,m2,mJ and Omega
    I1 = I1p
    m1 = m1p
    I2 = I2p
    m2 = m2p
    Omega = Omegap
    mJ = mJp+p
    q = 0
    
    #Initialize container for storing states and matrix elements
    data = []
    
    #Loop over possible values of Jprime
    for J in np.arange(np.abs(Jp-1),Jp+2).tolist():
        amp = mu_e*((-1)**(2*J + mJ - Omega) * np.sqrt((2*J+1)*(2*Jp+1))
               * threej_f(J,1,Jp,-mJ,p, mJp) * threej_f(J,1,Jp,-Omega,q, Omegap))

        basis_state = UncoupledBasisState(J, mJ, I1, m1, I2, m2, Omega)

        if amp != 0:
            data.append((amp, basis_state))

    return State(data)  

@state_operator
def HSx(psi:UncoupledBasisState) -> State:
    """
    Stark Hamiltonian operator for x-component of electric field
    """
    return -( d_p(-1,psi) - d_p(+1,psi) ) / np.sqrt(2)

@state_operator
def HSy(psi:UncoupledBasisState) -> State:
    """
    Stark Hamiltonian operator for y-component of electric field
    """
    return - 1j * ( d_p(-1,psi) + d_p(+1,psi) ) / np.sqrt(2)

@state_operator
def HSz(psi:UncoupledBasisState) -> State:
    """
    Stark Hamiltonian for z-component of electric field
    """
    return - d_p(0,psi)
