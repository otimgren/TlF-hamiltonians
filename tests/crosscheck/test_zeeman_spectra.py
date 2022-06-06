import numpy as np
import pytest
from TlF_hamiltonians.B_coupled.quantum_numbers import \
    generate_QN as gen_QN_coupled
from TlF_hamiltonians.B_coupled.utils import make_H_EB as make_H_EB_coupled
from TlF_hamiltonians.B_coupled_P.quantum_numbers import \
    generate_QN as gen_QN_coupled_P
from TlF_hamiltonians.B_coupled_P.utils import make_H_EB as make_H_EB_coupled_P
from TlF_hamiltonians.B_uncoupled.quantum_numbers import \
    generate_QN as gen_QN_uncoupled
from TlF_hamiltonians.B_uncoupled.utils import make_H_EB as make_H_EB_uncoupled

Jmax = 3

@pytest.fixture
def QN_coupled():
    """
    Define a basis for testing Hamiltonians 
    """
    return gen_QN_coupled(Jmin = 1, Jmax = Jmax)

@pytest.fixture
def H_EB_coupled(QN_coupled):
    return make_H_EB_coupled(QN_coupled)

@pytest.fixture
def QN_coupled_P():
    """
    Define a basis for testing Hamiltonians 
    """
    return gen_QN_coupled_P(Jmin = 1, Jmax = Jmax)

@pytest.fixture
def H_EB_coupled_P(QN_coupled_P):
    return make_H_EB_coupled_P(QN_coupled_P)

@pytest.fixture
def QN_uncoupled():
    """
    Define a basis for testing Hamiltonians 
    """
    return gen_QN_uncoupled(Jmin = 1, Jmax = Jmax)

@pytest.fixture
def H_EB_uncoupled(QN_uncoupled):
    return make_H_EB_uncoupled(QN_uncoupled)

def test_zeeman_spectra(H_EB_coupled, H_EB_uncoupled, H_EB_coupled_P, QN_coupled):
    """
    Check that the spectra are the same in each basis when electric field is
    varied.
    """

    Bs = np.linspace(-1000,1000,101)
    energies_coupled = np.empty((len(QN_coupled), len(Bs)))
    energies_uncoupled = np.empty((len(QN_coupled), len(Bs)))
    energies_coupled_P = np.empty((len(QN_coupled), len(Bs)))
    for i, B in enumerate(Bs):
        energies_coupled[:,i], _ = np.linalg.eigh(H_EB_coupled(np.array([0,0,0]), np.array([0,0,B])))
        energies_uncoupled[:,i], _ = np.linalg.eigh(H_EB_uncoupled(np.array([0,0,0]), np.array([0,0,B])))
        energies_coupled_P[:,i], _ = np.linalg.eigh(H_EB_coupled_P(np.array([0,0,0]), np.array([0,0,B])))

    message1 = "Zeeman spectrum not the same for coupled and uncoupled"
    assert np.allclose(energies_coupled, energies_uncoupled), message1

    message2 = "Zeeman spectrum not the same for coupled and coupled_P"
    assert np.allclose(energies_coupled, energies_coupled_P), message2
