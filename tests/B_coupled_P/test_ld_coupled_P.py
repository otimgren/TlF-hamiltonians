import numpy as np
import pytest
from centrex_TlF import State
from TlF_hamiltonians.B_coupled_P import H_c1p, H_q
from TlF_hamiltonians.B_coupled_P.quantum_numbers import generate_QN
from TlF_hamiltonians.utils import calculate_matrix_rep


@pytest.fixture
def QN():
    """
    Define a basis for testing Hamiltonians 
    """
    return generate_QN(Jmin = 1, Jmax = 4)

class TestHq:
    def test_returns_state(self, QN):
        """
        Check that the Hamiltonian function returns a state
        """
        actual = H_q(QN[0])
        message = f"H_q operator should return State object. Returned"\
                  f"{type(actual)} instead"
        assert isinstance(actual, State), message

    def test_hermitian(self, QN):
        """
        Checks that the matrix representation of the Hamiltonian is Hermitian
        """
        M = calculate_matrix_rep(H_q, QN)
        message = "H_q in B_coupled_P is not Hermitian!"
        assert np.allclose(M, M.conj().T), message

class TestH_c1p:
    def test_returns_state(self, QN):
        """
        Check that the Hamiltonian function returns a state
        """
        actual = H_c1p(QN[0])
        message = f"H_c1p operator should return State object. Returned"\
                  f"{type(actual)} instead"
        assert isinstance(actual, State), message

    def test_hermitian(self, QN):
        """
        Checks that the matrix representation of the Hamiltonian is Hermitian
        """
        M = calculate_matrix_rep(H_c1p, QN)
        message = "H_c1p in B_coupled_P is not Hermitian!"
        assert np.allclose(M, M.conj().T), message
