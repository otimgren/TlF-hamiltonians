# TlF-hamiltonians
Testing grounds for different ways of writing down Hamiltonians for TlF, so they can be compared to ensure the matrix elements have been written down correctly.

## B-state
The B$^3\Pi_1$ state Hamiltonian is written in three bases
- Uncoupled basis with states presented as $|J, \Omega, m_J, I_1, m_1, I_2, m_2 \rangle $ and $\Omega$ is a signed number
- Coupled basis with $|J, \Omega, F_1 = J+I_1, F = F_1+I_2, m_F, \rangle $ and $\Omega$ is a signed number
- Coupled parity eigenstate basis with $|J, |\Omega|, F_1 = J+I_1, F = F_1+I_2, m_F, P \rangle $ and $\Omega$ is an unsigned number

## X-state
Coming soon

## Tests
Tests can be run using [`pytest`](https://docs.pytest.org/). They test that each part of the Hamiltonian is hermitian, and that the spectra of the Hamiltonians are the same in each basis. For field-dependent parts of the Hamiltonian (Stark and Zeeman effect) I'm also checking that the Hamiltonians are isotropic, i.e. yield the same spectrum irrespective of the direction of the field. 
