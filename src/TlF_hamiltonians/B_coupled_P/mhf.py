"""
Writing down operators for magnetic hyperfine interaction
"""
import centrex_TlF.constants.constants_B as cst_B
from centrex_TlF import CoupledBasisState, State

from ..B_coupled import H_mhf_F as H_mhf_F_O
from ..B_coupled import H_mhf_Tl as H_mhf_Tl_O
from ..utils import parity_eigenstate_operator, state_operator


@state_operator
@parity_eigenstate_operator
def H_mhf_Tl(psi: CoupledBasisState, h1_Tl: float = cst_B.h1_Tl) -> State:
    """
    Operator for magnetic hyperfine term for Tl nucleus.
    """
    return H_mhf_Tl_O(psi, h1_Tl = h1_Tl)

@state_operator
@parity_eigenstate_operator
def H_mhf_F(psi: CoupledBasisState, h1_F: float = cst_B.h1_F) -> State:
    """
    Operator for magnetic hyperfine term for F nucleus.
    """
    return H_mhf_F_O(psi, h1_F = h1_F)
