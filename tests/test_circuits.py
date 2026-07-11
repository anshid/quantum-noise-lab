import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from src.circuits import add_measurements, bell_state, ghz_state, uniform_superposition
from src.simulation import sample_counts


@pytest.mark.parametrize(
    "bell_type, target_amplitudes",
    [
        ("phi_plus", [1, 0, 0, 1]),
        ("phi_minus", [1, 0, 0, -1]),
        ("psi_plus", [0, 1, 1, 0]),
        ("psi_minus", [0, 1, -1, 0]),
    ],
)
def test_bell_state_amplitudes(bell_type, target_amplitudes):
    sv = Statevector.from_instruction(bell_state(bell_type))
    target = Statevector(np.array(target_amplitudes) / np.sqrt(2))
    # Compare up to global phase: global phase is not physically observable,
    # so an exact-amplitude check would fail on a mathematically correct circuit.
    assert sv.equiv(target)


def test_bell_state_unknown_type_raises():
    with pytest.raises(ValueError):
        bell_state("not_a_real_bell_state")


@pytest.mark.parametrize("n", [2, 3, 5])
def test_ghz_state_amplitudes(n):
    sv = Statevector.from_instruction(ghz_state(n))
    expected = np.zeros(2**n)
    expected[0] = expected[-1] = 1 / np.sqrt(2)
    assert np.allclose(sv.data, expected, atol=1e-8)


def test_ghz_state_requires_at_least_two_qubits():
    with pytest.raises(ValueError):
        ghz_state(1)


def test_uniform_superposition_amplitudes():
    sv = Statevector.from_instruction(uniform_superposition(3))
    assert np.allclose(sv.data, np.full(8, 1 / np.sqrt(8)), atol=1e-8)


def test_bell_state_measurement_only_correlated_outcomes():
    counts = sample_counts(bell_state("phi_plus", measure=True), shots=4096, seed=42)
    # Load-bearing, seed-independent: no bitstring outside the entangled subspace can appear.
    assert set(counts) <= {"00", "11"}
    total = sum(counts.values())
    # Secondary, loose statistical check to avoid flakiness from primitives-path RNG plumbing.
    assert abs(counts.get("00", 0) / total - 0.5) < 0.1


def test_add_measurements_classical_bit_count():
    measured = add_measurements(ghz_state(4))
    assert measured.num_clbits == 4
    assert measured.num_qubits == 4
