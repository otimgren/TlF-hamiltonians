import numpy as np
import pytest
from centrex_TlF import State
from TlF_hamiltonians.B_coupled import (H_c1p, H_mhf_F, H_mhf_Tl, H_q, Hc1,
                                        Hrot, HSx, HSy, HSz)
from TlF_hamiltonians.B_coupled.quantum_numbers import generate_QN
from TlF_hamiltonians.utils import calculate_matrix_rep, calculate_matrix_reps


@pytest.fixture
def QN():
    """
    Define a basis for testing Hamiltonians 
    """
    return generate_QN(Jmin = 1, Jmax = 4)

@pytest.fixture
def H0(QN):
    H_list = [H_c1p, H_mhf_F, H_mhf_Tl, H_q, Hc1, Hrot]
    matrix_reps = calculate_matrix_reps(H_list, QN)
    return sum(matrix_reps)

class TestHSx:
    def test_returns_state(self, QN):
        """
        Check that the Hamiltonian function returns a state
        """
        actual = HSx(QN[0])
        message = f"HSx operator should return State object. Returned"\
                  f"{type(actual)} instead"
        assert isinstance(actual, State), message

    def test_hermitian(self, QN):
        """
        Checks that the matrix representation of the Hamiltonian is Hermitian
        """
        M = calculate_matrix_rep(HSx, QN)
        message = "HSx in B_coupled is not Hermitian!"
        assert np.allclose(M, M.conj().T), message

class TestHSy:
    def test_returns_state(self, QN):
        """
        Check that the Hamiltonian function returns a state
        """
        actual = HSy(QN[0])
        message = f"HSy operator should return State object. Returned"\
                  f"{type(actual)} instead"
        assert isinstance(actual, State), message

    def test_hermitian(self, QN):
        """
        Checks that the matrix representation of the Hamiltonian is Hermitian
        """
        M = calculate_matrix_rep(HSy, QN)
        message = "HSy in B_coupled is not Hermitian!"
        assert np.allclose(M, M.conj().T), message

class TestHSz:
    def test_returns_state(self, QN):
        """
        Check that the Hamiltonian function returns a state
        """
        actual = HSz(QN[0])
        message = f"HSz operator should return State object. Returned"\
                  f"{type(actual)} instead"
        assert isinstance(actual, State), message

    def test_hermitian(self, QN):
        """
        Checks that the matrix representation of the Hamiltonian is Hermitian
        """
        M = calculate_matrix_rep(HSz, QN)
        message = "HSz in B_coupled is not Hermitian!"
        assert np.allclose(M, M.conj().T), message


class TestIsotropic:
    def test_spectrums_same(self, QN, H0):
        """
        Check that the spectrum of the Hamiltonian is the same for electric field
        in any direction.
        """
        # Calculate matrix representations for each of the Stark hamiltonians
        Mx = calculate_matrix_rep(HSx, QN)
        My = calculate_matrix_rep(HSy, QN)
        Mz = calculate_matrix_rep(HSz, QN)

        # Define Hamiltonian as function of electric field
        H_E = lambda E: H0 + E[0]*Mx + E[1]*My + E[2]*Mz

        # Loop over electric field values and find energies for each direction
        Es = np.linspace(-1000,1000,101)
        energies_x = np.empty((len(QN), len(Es)))
        energies_y = np.empty((len(QN), len(Es)))
        energies_z = np.empty((len(QN), len(Es)))
        for i, E in enumerate(Es):
            energies_x[:,i], _ = np.linalg.eigh(H_E(np.array([E,0,0])))
            energies_y[:,i], _ = np.linalg.eigh(H_E(np.array([0,E,0])))
            energies_z[:,i], _ = np.linalg.eigh(H_E(np.array([0,0,E])))

        message_xy = "Stark effect not isotropic for B_coupled: x!=y"
        assert np.allclose(energies_x, energies_y), message_xy

        message_xz = "Stark effect not isotropic for B_coupled: x!=z"
        assert np.allclose(energies_x, energies_z), message_xz
