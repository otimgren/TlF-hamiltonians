"""
Writing down parts of the Hamiltonian that break the degeneracy betweem
Lambda-doubled states
"""
import centrex_TlF.constants.constants_B as cst_B
import numpy as np
from centrex_TlF import State, UncoupledBasisState
from centrex_TlF.hamiltonian.utils import threej_f

from ..utils import state_operator


@state_operator
def H_q(psi: UncoupledBasisState, q:float = cst_B.q) -> State:
    """
    Calculates the "q-term" that couples states with opposite Omega  
    shifting e-parity up and f-parity down in energy 
    """
    # All quantum numbers the same, except Omega inverts sign
    J = psi.J
    mJ = psi.mJ
    I1 = psi.I1
    m1 = psi.m1
    I2 = psi.I2
    m2 = psi.m2
    Omega = -psi.Omega

    amp = q*J*(J+1)/2
    ket = UncoupledBasisState(J, mJ, I1, m1, I2, m2, Omega)

    return State([(amp,ket)])

@state_operator
def H_c1p(psi: UncoupledBasisState, c1p: float = cst_B.c1p_Tl) -> State:
    """
    Calculates the lambda-doubling nuclear spin - rotation term
    """
    
    # Find the quantum numbers of the input state
    Jp = psi.J
    mJp = psi.mJ
    I1p = psi.I1
    m1p = psi.m1
    I2p = psi.I2
    m2p = psi.m2
    Omegap = psi.Omega
    
    # I1, I2 and m2 must be the same for non-zero matrix element
    I1 = I1p
    m2 = m2p
    I2 = I2p
    
    # To have non-zero matrix element need OmegaPrime = -Omega
    Omega = -Omegap
    
    # q is chosen such that q == Omega
    q = Omega
    
    # Initialize container for storing states and matrix elements
    data = []
    
    # Loop over the possible values of quantum numbers for which the matrix 
    # element can be non-zero
    # Need J = Jp+1 ... |Jp-1|
    for J in np.arange(np.abs(Jp-1), Jp+2).tolist():
        # Evaluate the part of the matrix element that is common for all p
        common_coefficient = (
            -c1p/2 * np.sqrt(I1*(I1+1)*(2*I1+1)*(2*Jp+1)*(2*J+1))
            * ( (-1)**(J) * np.sqrt(J*(J+1)*(2*J+1))
                * threej_f(J,1,Jp,0,q,Omegap) * threej_f(J,1,J,-Omega, q, 0)
              + (-1)**(Jp) * np.sqrt(Jp*(Jp+1)*(2*Jp+1))
                * threej_f(Jp,1,Jp,0,q,Omegap) * threej_f(J,1,Jp,-Omega, q, 0) )
        )

        # Loop over values of p
        for p in np.arange(-1,2).tolist():
            #To have non-zero matrix element need mJ = mJprime - p
            mJ = mJp - p

            #Also need m1 = m1prime + p
            m1 = m1p + p

            # Check the value of m1 is physical
            if np.abs(m1) <= I1:
                #Calculate rest of matrix element
                p_factor = ((-1)**(p+mJ+I1-m1-Omega)*threej_f(J, 1, Jp, -mJ, -p, mJp)
                               *threej_f(I1, 1, I1p, -m1, p, m1p))

                amp = common_coefficient*p_factor

                basis_state = UncoupledBasisState(J, mJ, I1, m1, I2, m2, Omega)

                if amp != 0:
                        data.append((amp, basis_state))

    return State(data)

