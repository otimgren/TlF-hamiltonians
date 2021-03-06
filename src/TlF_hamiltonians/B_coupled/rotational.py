"""
Operators for making the rotational part of the Hamiltonian are defined here
"""
from centrex_TlF import CoupledBasisState, State
from centrex_TlF.constants import constants_B as cst_B

from ..utils import state_operator


def J2(psi: CoupledBasisState) -> State:
    return State([(psi.J*(psi.J+1),psi)])

def J4(psi: CoupledBasisState) -> State:
    return State([( (psi.J*(psi.J+1))**2, psi)])

def J6(psi: CoupledBasisState) -> State:
    return State([( (psi.J*(psi.J+1))**3, psi)])

@state_operator
def Hrot(psi: CoupledBasisState, B: float = cst_B.B_rot_B, 
    D: float = cst_B.D_rot_B, H: float = cst_B.H_const_B)-> State:
    """
    Rotational Hamiltonian for the B-state.
    """
    return B*J2(psi) + D*J4(psi) + H*J6(psi)
