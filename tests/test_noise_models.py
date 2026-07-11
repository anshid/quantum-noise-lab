import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Kraus, SuperOp
from qiskit_aer.noise import (
    amplitude_damping_error,
    depolarizing_error,
    pauli_error,
    phase_damping_error,
)

from src.noise_models import (
    CHANNEL_KRAUS_BUILDERS,
    amplitude_damping_kraus,
    bit_flip_kraus,
    depolarizing_kraus,
    phase_damping_kraus,
    phase_flip_kraus,
    single_qubit_noise_model,
)
from src.simulation import sample_counts

_PARAMS = [0.0, 0.3, 1.0]


@pytest.mark.parametrize("channel_name", list(CHANNEL_KRAUS_BUILDERS))
@pytest.mark.parametrize("p", _PARAMS)
def test_completeness_relation(channel_name, p):
    # sum_i K_i^dagger K_i = I is required for a channel to be trace-preserving.
    kraus_ops = CHANNEL_KRAUS_BUILDERS[channel_name](p)
    total = sum(k.conj().T @ k for k in kraus_ops)
    assert np.allclose(total, np.eye(2), atol=1e-8)


@pytest.mark.parametrize("p", [0.2, 0.6])
def test_bit_flip_matches_qiskit_builtin(p):
    ours = SuperOp(Kraus(bit_flip_kraus(p)))
    theirs = SuperOp(Kraus(pauli_error([("X", p), ("I", 1 - p)])))
    assert np.allclose(ours.data, theirs.data, atol=1e-8)


@pytest.mark.parametrize("p", [0.2, 0.6])
def test_phase_flip_matches_qiskit_builtin(p):
    ours = SuperOp(Kraus(phase_flip_kraus(p)))
    theirs = SuperOp(Kraus(pauli_error([("Z", p), ("I", 1 - p)])))
    assert np.allclose(ours.data, theirs.data, atol=1e-8)


@pytest.mark.parametrize("p", [0.2, 0.6])
def test_depolarizing_matches_qiskit_builtin(p):
    # Non-unique Kraus decompositions can still represent the same channel, so equality
    # must be checked via SuperOp (or Choi), not by comparing Kraus operators directly.
    ours = SuperOp(Kraus(depolarizing_kraus(p)))
    theirs = SuperOp(Kraus(depolarizing_error(p, 1)))
    assert np.allclose(ours.data, theirs.data, atol=1e-8)


@pytest.mark.parametrize("gamma", [0.2, 0.6])
def test_amplitude_damping_matches_qiskit_builtin(gamma):
    ours = SuperOp(Kraus(amplitude_damping_kraus(gamma)))
    theirs = SuperOp(Kraus(amplitude_damping_error(gamma)))
    assert np.allclose(ours.data, theirs.data, atol=1e-8)


@pytest.mark.parametrize("lam", [0.2, 0.6])
def test_phase_damping_matches_qiskit_builtin(lam):
    ours = SuperOp(Kraus(phase_damping_kraus(lam)))
    theirs = SuperOp(Kraus(phase_damping_error(lam)))
    assert np.allclose(ours.data, theirs.data, atol=1e-8)


@pytest.mark.parametrize("channel_name", list(CHANNEL_KRAUS_BUILDERS))
def test_zero_parameter_is_identity_channel(channel_name):
    rho = DensityMatrix.from_label("+")
    evolved = rho.evolve(Kraus(CHANNEL_KRAUS_BUILDERS[channel_name](0.0)))
    assert np.allclose(evolved.data, rho.data, atol=1e-8)


def test_bit_flip_p1_deterministically_flips():
    rho0 = DensityMatrix.from_label("0")
    evolved = rho0.evolve(Kraus(bit_flip_kraus(1.0)))
    assert np.allclose(evolved.data, DensityMatrix.from_label("1").data, atol=1e-8)


def test_depolarizing_p1_gives_maximally_mixed():
    rho = DensityMatrix.from_label("+")
    evolved = rho.evolve(Kraus(depolarizing_kraus(1.0)))
    assert np.allclose(evolved.data, np.eye(2) / 2, atol=1e-8)


def test_amplitude_damping_gamma1_maps_to_ground_state():
    rho = DensityMatrix.from_label("+")
    evolved = rho.evolve(Kraus(amplitude_damping_kraus(1.0)))
    assert np.allclose(evolved.data, DensityMatrix.from_label("0").data, atol=1e-8)


@pytest.mark.parametrize("lam", [0.0, 0.3, 0.7, 1.0])
def test_phase_damping_preserves_populations_shrinks_coherence(lam):
    rho = DensityMatrix.from_label("+")  # populations 0.5/0.5, off-diagonal 0.5
    evolved = rho.evolve(Kraus(phase_damping_kraus(lam)))
    assert np.isclose(evolved.data[0, 0].real, 0.5, atol=1e-8)
    assert np.isclose(evolved.data[1, 1].real, 0.5, atol=1e-8)
    assert np.isclose(evolved.data[0, 1], 0.5 * np.sqrt(1 - lam), atol=1e-8)


def test_noisy_sample_counts_reproduces_expected_flip_rate():
    p = 0.3
    qc = QuantumCircuit(1)
    qc.id(0)
    qc.add_register(ClassicalRegister(1, name="meas"))
    qc.measure(0, 0)

    noise_model = single_qubit_noise_model(bit_flip_kraus(p), qubits=[0], gate="id")
    counts = sample_counts(qc, shots=4096, seed=42, noise_model=noise_model)
    total = sum(counts.values())
    assert abs(counts.get("1", 0) / total - p) < 0.05
