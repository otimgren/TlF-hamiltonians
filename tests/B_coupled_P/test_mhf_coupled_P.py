import numpy as np
import pytest
from centrex_TlF import State
from TlF_hamiltonians.B_coupled_P import H_mhf_F, H_mhf_Tl
from TlF_hamiltonians.B_coupled_P.quantum_numbers import generate_QN
from TlF_hamiltonians.utils import calculate_matrix_rep


@pytest.fixture
def QN():
    """
    Define a basis for testing Hamiltonians 
    """
    return generate_QN(Jmin = 1, Jmax = 4)

class TestHmhfF:
    def test_returns_state(self, QN):
        """
        Check that the Hamiltonian function returns a state
        """
        actual = H_mhf_F(QN[0])
        message = f"H_mhf_F operator should return State object. Returned"\
                  f"{type(actual)} instead"
        assert isinstance(actual, State), message

    def test_hermitian(self, QN):
        """
        Checks that the matrix representation of the Hamiltonian is Hermitian
        """
        M = calculate_matrix_rep(H_mhf_F, QN)
        message = "H_mhf_F in B_coupled_P is not Hermitian!"
        assert np.allclose(M, M.conj().T), message

class TestHmhfTl:
    def test_returns_state(self, QN):
        """
        Check that the Hamiltonian function returns a state
        """
        actual = H_mhf_Tl(QN[0])
        message = f"H_mhf_Tl operator should return State object. Returned"\
                  f"{type(actual)} instead"
        assert isinstance(actual, State), message

    def test_hermitian(self, QN):
        """
        Checks that the matrix representation of the Hamiltonian is Hermitian
        """
        M = calculate_matrix_rep(H_mhf_Tl, QN)
        message = "H_mhf_Tl in B_coupled_P is not Hermitian!"
        assert np.allclose(M, M.conj().T), message


