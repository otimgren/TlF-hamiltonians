from typing import Callable, Dict, List, Union

import numpy as np
from centrex_TlF import CoupledBasisState, State, UncoupledBasisState
from joblib import Parallel, delayed
from tqdm import tqdm


def calculate_matrix_rep(H:Callable,
                        QN: List[State],
                        disable = False, args: Dict = {})-> np.ndarray:
    """
    Calculates a matrix representation for H in the basis defined by QN
    """
    result = np.zeros([len(QN),len(QN)], dtype='complex')
    for i,a in tqdm(enumerate(QN), disable = disable):
        for j,b in enumerate(QN):
            result[i,j] = (1*a)@H(b, **args)
            
    return result

def calculate_matrix_reps(H_list: List[Callable], 
            QN: List[Union[UncoupledBasisState, CoupledBasisState]],
            args_list: List[Dict] = None)-> List[np.ndarray]:
    """
    Calculates matrix represenations for operators in H_list in basis defined by
    QN.
    """
    if args_list is None:
        args_list = [{}]*len(H_list)

    result = (Parallel(n_jobs = 6)(delayed(calculate_matrix_rep)(H, QN, disable = True)
                                    for H, args in zip(H_list, args_list)))

    return result

def state_operator(H: Callable) -> Callable:
    """
    Wrapper that returns an operator that can operate on states (i.e.
    superpositions of basis states). The State should be expressed in the
    basis that operator H works on.
    """
    def H_state(psi: State, *args, **kwargs) -> State:
        state_out = State()
        for (amp, ket) in psi.data:
            state_out += amp*H(ket, *args, **kwargs)

        return state_out

    return H_state

def parity_eigenstate_operator(H_state: Callable) -> Callable:
    """
    Wrapper that converts a state_operator that operates on Omega eigenstates (i.e.
    signed Omega) into an operator that operates on parity eigenstates (i.e.
    |Omega| and P are good quantum numbers). 
    
    This is done by changing the basis of the input state (assumed to be a
    parity eigenstate) into the Omega basis and then operating using 
    the Omega basis operator H_state
    """
    def H_P(psi: State, *args, **kwargs) -> State:
        psi = psi.transform_to_omega_basis()
        return H_state(psi, *args, **kwargs).transform_to_parity_basis()

    return H_P

def find_QN_indices(QN:List[State],
                    **kwargs) -> List[int]:
    """
    Finds the indices of states in QN whose quantum numbers match those provided
    in kwargs. E.g. if provide J = 1, will return indices of all states with
    J = 1.
    """
    indices = []
    for i, basis_state in enumerate(QN):
        save = True
        for qn, value in kwargs.items():
            # Compare value of quantumn number to desired value
            largest_component = basis_state.find_largest_component()
            if  getattr(largest_component, qn) != value:
                # If not the same, move to next state
                save = False
                break
            else:
                pass
        # If all quantum numbers match, add indices to list
        if save:
            indices.append(i)

    return indices
