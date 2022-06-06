import numpy as np
import pytest
from centrex_TlF import State
from TlF_hamiltonians.B_coupled_P import (H_c1p, H_mhf_F, H_mhf_Tl, H_q, Hc1,
                                          Hrot, HZx, HZy, HZz)
from TlF_hamiltonians.B_coupled_P.quantum_numbers import generate_QN
from TlF_hamiltonians.utils import calculate_matrix_rep, calculate_matrix_reps


@pytest.fixture
def QN():
    """
    Define a basis for testing Hamiltonians 
    """
    return generate_QN(Jmin = 1, Jmax = 2)

@pytest.fixture
def H0(QN):
    H_list = [H_c1p, H_mhf_F, H_mhf_Tl, H_q, Hc1, Hrot]
    matrix_reps = calculate_matrix_reps(H_list, QN)
    return sum(matrix_reps)

class TestHZx:
    def test_returns_state(self, QN):
        """
        Check that the Hamiltonian function returns a state
        """
        actual = HZx(QN[0])
        message = f"HZx operator should return State object. Returned"\
                  f"{type(actual)} instead"
        assert isinstance(actual, State), message

    def test_hermitian(self, QN):
        """
        Checks that the matrix representation of the Hamiltonian is Hermitian
        """
        M = calculate_matrix_rep(HZx, QN)
        message = "HZx in B_coupled_P is not Hermitian!"
        assert np.allclose(M, M.conj().T), message

class TestHZy:
    def test_returns_state(self, QN):
        """
        Check that the Hamiltonian function returns a state
        """
        actual = HZy(QN[0])
        message = f"HZy operator should return State object. Returned"\
                  f"{type(actual)} instead"
        assert isinstance(actual, State), message

    def test_hermitian(self, QN):
        """
        Checks that the matrix representation of the Hamiltonian is Hermitian
        """
        M = calculate_matrix_rep(HZy, QN)
        message = "HZy in B_coupled_P is not Hermitian!"
        assert np.allclose(M, M.conj().T), message

class TestHZz:
    def test_returns_state(self, QN):
        """
        Check that the Hamiltonian function returns a state
        """
        actual = HZz(QN[0])
        message = f"HZz operator should return State object. Returned"\
                  f"{type(actual)} instead"
        assert isinstance(actual, State), message

    def test_hermitian(self, QN):
        """
        Checks that the matrix representation of the Hamiltonian is Hermitian
        """
        M = calculate_matrix_rep(HZz, QN)
        message = "HZz in B_coupled_P is not Hermitian!"
        assert np.allclose(M, M.conj().T), message


class TestIsotropic:
    def test_spectrums_same(self, QN, H0):
        """
        Check that the spectrum of the Hamiltonian is the same for magnetic field
        in any direction.
        """
        # Calculate matrix representations for each of the Stark hamiltonians
        Mx = calculate_matrix_rep(HZx, QN)
        My = calculate_matrix_rep(HZy, QN)
        Mz = calculate_matrix_rep(HZz, QN)

        # Define Hamiltonian as function of magnetic field
        H_B = lambda B: H0 + B[0]*Mx + B[1]*My + B[2]*Mz

        # Loop over electric field values and find energies for each direction
        Bs = np.linspace(-1000,1000,101)
        energies_x = np.empty((len(QN), len(Bs)))
        energies_y = np.empty((len(QN), len(Bs)))
        energies_z = np.empty((len(QN), len(Bs)))
        for i, B in enumerate(Bs):
            energies_x[:,i], _ = np.linalg.eigh(H_B(np.array([B,0,0])))
            energies_y[:,i], _ = np.linalg.eigh(H_B(np.array([0,B,0])))
            energies_z[:,i], _ = np.linalg.eigh(H_B(np.array([0,0,B])))

        message_xy = "Zeeman effect not isotropic for B_coupled_P: x!=y"
        assert np.allclose(energies_x, energies_y), message_xy

        message_xz = "Zeeman effect not isotropic for B_coupled_P: x!=z"
        assert np.allclose(energies_x, energies_z), message_xz
