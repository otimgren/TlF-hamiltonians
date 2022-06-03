"""
Writing down the nuclear spin - rotation operators 
"""
import centrex_TlF.constants.constants_B as cst_B
import numpy as np
from centrex_TlF import State, UncoupledBasisState
from centrex_TlF.hamiltonian.utils import threej_f

from ..utils import state_operator


@state_operator
def Hc1(psi: UncoupledBasisState, c1: float = cst_B.c_Tl) -> State:
    """
    Calculates the effect of the c1 term on the input basis state
    """
    #Find the quantum numbers of the input state (the primed state)
    Jp = psi.J
    mJp = psi.mJ
    I1p = psi.I1
    m1p = psi.m1
    I2p = psi.I2
    m2p = psi.m2
    Omegap = psi.Omega

    #J, I1, I2 and m2 must be the same for non-zero matrix element
    J = Jp
    I2 = I2p
    m2 = m2p
    I1 = I1p

    # Omega also doesn't change
    Omega = Omegap

    #Initialize container for storing states and matrix elements
    data = []

    #Evaluate the part of the matrix element that is common for all p
    common_coefficient = c1*np.sqrt(J*(J+1)*(2*J+1)*I1*(I1+1)*(2*I1+1))

    #Loop over the spherical tensor components
    for p in np.arange(-1,2).tolist():
        #To have non-zero matrix element need mJ = mJp + p
        mJ = mJp + p

        #Also need m1 = m1p - p
        m1 = m1p - p

        #Check that mJprime and m2prime are physical
        if np.abs(mJ) <= J and np.abs(m1) <= I1:
            #Calculate rest of matrix element
            p_factor = ((-1)**(p+J-mJ + I1 - m1) * threej_f(J,1,Jp,-mJ,p,mJp)
                        * threej_f(I1, 1, I1p, -m1, -p, m1p))

            amp = common_coefficient*p_factor
            basis_state = UncoupledBasisState(J, mJ, I1, m1, I2, m2, Omega = Omega)
            if amp != 0:
                data.append((amp, basis_state))

    return State(data)
