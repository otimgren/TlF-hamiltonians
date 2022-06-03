"""
Writing down parts of the Hamiltonian that break the degeneracy betweem
Lambda-doubled states
"""
import centrex_TlF.constants.constants_B as cst_B
from centrex_TlF import CoupledBasisState, State

from ..B_coupled import H_c1p as H_c1p_O
from ..B_coupled import H_q as H_q_O
from ..utils import parity_eigenstate_operator, state_operator

@state_operator
@parity_eigenstate_operator
def H_q(psi: CoupledBasisState, q:float = cst_B.q) -> State:
    """
    Calculates the "q-term" that couples states with opposite Omega  
    shifting e-parity up and f-parity down in energy 
    """
    return H_q_O(psi, q)

@state_operator
@parity_eigenstate_operator
def H_c1p(psi: CoupledBasisState, c1p: float = cst_B.c1p_Tl) -> State:
    """
    Calculates the lambda-doubling nuclear spin - rotation term
    """
    return H_c1p_O(psi, c1p)

