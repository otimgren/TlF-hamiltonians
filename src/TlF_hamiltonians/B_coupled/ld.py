"""
Writing down parts of the Hamiltonian that break the degeneracy betweem
Lambda-doubled states
"""
import centrex_TlF.constants.constants_B as cst_B
import numpy as np
from centrex_TlF import CoupledBasisState, State
from centrex_TlF.hamiltonian.utils import sixj_f, threej_f

from ..utils import state_operator


@state_operator
def H_q(psi: CoupledBasisState, q:float = cst_B.q) -> State:
    """
    Calculates the "q-term" that couples states with opposite Omega  
    shifting e-parity up and f-parity down in energy 
    """
    # All quantum numbers the same, except Omega inverts sign
    J = psi.J
    I1 = psi.I1
    I2 = psi.I2
    F1 = psi.F1
    F = psi.F
    mF = psi.mF
    Omega = -psi.Omega

    amp = q*J*(J+1)/2
    ket = CoupledBasisState(F, mF, F1, J, I1, I2, Omega=Omega)

    return State([(amp,ket)])

@state_operator
def H_c1p(psi: CoupledBasisState, c1p: float = cst_B.c1p_Tl) -> State:
    """
    Calculates the lambda-doubling nuclear spin - rotation term
    """
    # Find the quantum numbers of the input state
    Jp = psi.J
    I1p = psi.I1
    I2p = psi.I2
    F1p = psi.F1
    Fp = psi.F
    mFp = psi.mF
    Omegap = psi.Omega

    #I1, I2, F and mF are the same for both states
    I1 = I1p
    I2 = I2p
    F = Fp
    F1 = F1p
    mF = mFp

    # Omegas are opposite
    Omega = -Omegap

    # Calculate the value of q
    q = Omega

    data = []

    # Loop over possible values of J
    # Need J = |Jp-1| ... |Jp+1|
    for J in np.arange(np.abs(Jp-1), Jp+2).tolist():
        # Calculate matrix element
        amp = (
            -c1p/2 * (-1)**(J+Jp+F1+I1-Omegap) * sixj_f(I1,Jp,F1,J,I1,1)
            *np.sqrt(I1*(I1+1)*(2*I1+1)*(2*J+1)*(2*Jp+1))
            *(
                (-1)**(J)*np.sqrt(J*(J+1)*(2*J+1))
                * threej_f(J,1, Jp, 0, q, Omegap) * threej_f(J,1,J,-Omega,q,0)
                + (-1)**(Jp)*np.sqrt(Jp*(Jp+1)*(2*Jp+1))
                * threej_f(Jp,1, Jp, 0, q, Omegap) * threej_f(J,1,Jp,-Omega,q,0)
            )
        )

        basis_state = CoupledBasisState(F, mF, F1, J, I1, I2, Omega=Omega)

        data.append((amp, basis_state))
    
    return State(data)

