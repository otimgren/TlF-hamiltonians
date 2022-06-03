"""
Writing down operators for magnetic hyperfine interaction
"""
import centrex_TlF.constants.constants_B as cst_B
import numpy as np
from centrex_TlF import State, UncoupledBasisState
from centrex_TlF.hamiltonian.utils import threej_f

from ..utils import state_operator


@state_operator
def H_mhf_Tl(psi: UncoupledBasisState, h1_Tl: float = cst_B.h1_Tl) -> State:
    """
    Operator for magnetic hyperfine term for Tl nucleus.
    """
    #Find the quantum numbers of the input state (the primed state)
    Jp = psi.J
    mJp = psi.mJ
    I1p = psi.I1
    m1p = psi.m1
    I2p = psi.I2
    m2p = psi.m2
    Omegap = psi.Omega
    
    #I1, I2 and m2 must be the same for non-zero matrix element
    I2 = I2p
    m2 = m2p
    I1 = I1p

    # Omega also doesn't change
    Omega = Omegap
    
    #Initialize container for storing states and matrix elements
    data = []
    
    # Loop over the possible values of quantum numbers for which the matrix element can be non-zero
    # Need J = Jp+1 ... |Jp-1|
    for J in np.arange(np.abs(Jp-1), Jp+2).tolist():
        # Evaluate the part of the matrix element that is common for all p
        common_coefficient = (
            h1_Tl*threej_f(J, 1, Jp, -Omega, 0, Omegap)
            *np.sqrt((2*J+1)*(2*Jp+1)*I1*(I1+1)*(2*I1+1))
        )
        #Loop over the spherical tensor components of I1:
        for p in np.arange(-1,2).tolist():
            #To have non-zero matrix element need mJ = mJprime + p
            mJ = mJp - p
            
            #Also need m1 = m1prime - p
            m1 = m1p + p
            
            #Check that mJprime and m2prime are physical
            if np.abs(mJ) <= J and np.abs(m1) <= I1:
                #Calculate rest of matrix element
                p_factor = ((-1)**(p-mJ+I1-m1-Omega)*threej_f(J, 1, Jp, -mJ, -p, mJp)
                               *threej_f(I1, 1, I1p, -m1, p, m1p))
                               
                amp = common_coefficient*p_factor *Omega # TODO: Why is there a *Omega here?
                basis_state = UncoupledBasisState(J, mJ, I1, m1, I2, m2, Omega = Omega)
                if amp != 0:
                    data.append((amp, basis_state))
    
    return State(data)

@state_operator
def H_mhf_F(psi: UncoupledBasisState, h1_F: float = cst_B.h1_F) -> State:
    """
    Operator for magnetic hyperfine term for F nucleus.
    """
    #Find the quantum numbers of the input state (the primed state in the write up)
    Jp = psi.J
    mJp = psi.mJ
    I1p = psi.I1
    m1p = psi.m1
    I2p = psi.I2
    m2p = psi.m2
    Omegap = psi.Omega
    
    #I1, I2 and m1 must be the same for non-zero matrix element
    I1 = I1p
    m1 = m1p
    I2 = I2p

    # Omega also doesn't change
    Omega = Omegap
    
    #Initialize container for storing states and matrix elements
    data = []
    
    #Loop over the possible values of quantum numbers for which the matrix element can be non-zero
    #Need J = Jp+1 ... |pJ-1|
    for J in np.arange(np.abs(Jp-1), Jp+2).tolist():
        #Evaluate the part of the matrix element that is common for all p
        common_coefficient = (
            h1_F*threej_f(J, 1, Jp, -Omega, 0, Omegap)
            *np.sqrt((2*J+1)*(2*Jp+1)*I2*(I2+1)*(2*I2+1))
        )
        
        #Loop over the spherical tensor components of I2:
        for p in np.arange(-1,2).tolist():
            #To have non-zero matrix element need mJ = mJp - p
            mJ = mJp - p
            
            #Also need m2  = m2p + p
            m2 = m2p + p
            
            #Check that mJprime and m2prime are physical
            if np.abs(mJ) <= J and np.abs(m2) <= I2:
                #Calculate rest of matrix element
                p_factor = ((-1)**(p-mJ+I2-m2-Omega)*threej_f(J, 1, Jp, -mJ, -p, mJp)
                               *threej_f(I2, 1, I2p, -m2, p, m2p))
                               
                amp = common_coefficient*p_factor *Omega # TODO:why factor of Omega here?
                basis_state = UncoupledBasisState(J, mJ, I1, m1, I2, m2, Omega = Omega)
                if amp != 0:
                    data.append((amp, basis_state))
        
        
    return State(data)
