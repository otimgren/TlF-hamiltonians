import numpy as np
import pytest
from centrex_TlF import State
from TlF_hamiltonians.B_coupled import Hc1
from TlF_hamiltonians.B_coupled.quantum_numbers import generate_QN
from TlF_hamiltonians.utils import calculate_matrix_rep


@pytest.fixture
def QN():
    """
    Define a basis for testing Hamiltonians 
    """
    return generate_QN(Jmin = 1, Jmax = 4)

class TestHc1:
    def test_returns_state(self, QN):
        """
        Check that the Hamiltonian function returns a state
        """
        actual = Hc1(QN[0])
        message = f"Hc1 operator should return State object. Returned"\
                  f"{type(actual)} instead"
        assert isinstance(actual, State), message

    def test_hermitian(self, QN):
        """
        Checks that the matrix representation of the Hamiltonian is Hermitian
        """
        M = calculate_matrix_rep(Hc1, QN)
        message = "Hc1 in B_coupled is not Hermitian!"
        assert np.allclose(M, M.conj().T), message
