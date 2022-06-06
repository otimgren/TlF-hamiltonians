import numpy as np
import pytest
from TlF_hamiltonians.B_coupled.quantum_numbers import \
    generate_QN as gen_QN_coupled
from TlF_hamiltonians.B_coupled.utils import make_H0 as make_H0_coupled
from TlF_hamiltonians.B_coupled_P.quantum_numbers import \
    generate_QN as gen_QN_coupled_P
from TlF_hamiltonians.B_coupled_P.utils import make_H0 as make_H0_coupled_P
from TlF_hamiltonians.B_uncoupled.quantum_numbers import \
    generate_QN as gen_QN_uncoupled
from TlF_hamiltonians.B_uncoupled.utils import make_H0 as make_H0_uncoupled


@pytest.fixture
def QN_coupled():
    """
    Define a basis for testing Hamiltonians 
    """
    return gen_QN_coupled(Jmin = 1, Jmax = 4)

@pytest.fixture
def H0_coupled(QN_coupled):
    return make_H0_coupled(QN_coupled)

@pytest.fixture
def QN_coupled_P():
    """
    Define a basis for testing Hamiltonians 
    """
    return gen_QN_coupled_P(Jmin = 1, Jmax = 4)

@pytest.fixture
def H0_coupled_P(QN_coupled_P):
    return make_H0_coupled_P(QN_coupled_P)

@pytest.fixture
def QN_uncoupled():
    """
    Define a basis for testing Hamiltonians 
    """
    return gen_QN_uncoupled(Jmin = 1, Jmax = 4)

@pytest.fixture
def H0_uncoupled(QN_uncoupled):
    return make_H0_uncoupled(QN_uncoupled)

def test_spectra(H0_coupled, H0_uncoupled, H0_coupled_P):
    """
    Test that the eigenenergies of the field free Hamiltonians are all the same.
    """
    energies_coupled, _ = np.linalg.eigh(H0_coupled)
    energies_coupled_P, _ = np.linalg.eigh(H0_coupled_P)
    energies_uncoupled, _ = np.linalg.eigh(H0_uncoupled)

    message1 = "Coupled and uncoupled field free energies not the same"
    assert np.allclose(energies_coupled, energies_uncoupled), message1

    message2 = "Coupled parity vs Omega field free energies not the same"
    assert np.allclose(energies_coupled, energies_coupled_P), message2

