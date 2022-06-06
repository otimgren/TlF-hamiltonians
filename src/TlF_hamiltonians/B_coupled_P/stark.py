"""
Writing down operators for the Stark Hamiltonian
"""
import centrex_TlF.constants.constants_B as cst_B
import numpy as np
from centrex_TlF import CoupledBasisState, State

from ..B_coupled.stark import d_p
from ..utils import parity_eigenstate_operator, state_operator


@parity_eigenstate_operator
@state_operator
def HSx(psi:State) -> State:
    """
    Stark Hamiltonian operator for x-component of electric field
    """
    return -( d_p(psi,-1) - d_p(psi,+1) ) / np.sqrt(2)

@parity_eigenstate_operator
@state_operator
def HSy(psi:State) -> State:
    """
    Stark Hamiltonian operator for y-component of electric field
    """
    return - 1j * ( d_p(psi,-1) + d_p(psi,+1) ) / np.sqrt(2)

@parity_eigenstate_operator
@state_operator
def HSz(psi:State) -> State:
    """
    Stark Hamiltonian for z-component of electric field
    """
    return - d_p(psi,0)
