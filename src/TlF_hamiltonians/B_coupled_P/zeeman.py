"""
Writing down operators functions for the Zeeman Hamiltonian for the B state
"""
import centrex_TlF.constants.constants_B as cst_B
import numpy as np
from centrex_TlF import CoupledBasisState, State

from ..B_coupled.zeeman import mu_p
from ..utils import parity_eigenstate_operator, state_operator


@state_operator
@parity_eigenstate_operator
def HZx(psi:CoupledBasisState) -> State:
    """
    Zeeman Hamiltonian operator for x-component of magnetic field
    """
    return -( mu_p(-1,psi) - mu_p(+1,psi) ) / np.sqrt(2)

@state_operator
@parity_eigenstate_operator
def HZy(psi:CoupledBasisState) -> State:
    """
    Zeeman Hamiltonian operator for y-component of magnetic field
    """
    return - 1j * ( mu_p(-1,psi) + mu_p(+1,psi) ) / np.sqrt(2)

@state_operator
@parity_eigenstate_operator
def HZz(psi:CoupledBasisState) -> State:
    """
    Zeeman Hamiltonian for z-component of magnetic field
    """
    return - mu_p(0,psi)
