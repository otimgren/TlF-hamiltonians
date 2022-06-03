"""
Writing down the nuclear spin - rotation operators 
"""
import centrex_TlF.constants.constants_B as cst_B
from centrex_TlF import CoupledBasisState, State

from ..B_coupled import Hc1 as Hc1_O
from ..utils import parity_eigenstate_operator, state_operator


@state_operator
@parity_eigenstate_operator
def Hc1(psi: CoupledBasisState, c1: float = cst_B.c_Tl) -> State:
    """
    Calculates the effect of the c1 term on the input basis state
    """
    return Hc1_O(psi, c1)
