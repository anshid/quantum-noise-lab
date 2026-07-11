"""Single-qubit quantum noise channels as explicit Kraus operators.

Each channel is a completely positive, trace-preserving (CPTP) map
rho -> sum_i K_i rho K_i^dagger. The five channels here are the standard
building blocks for near-term hardware noise: bit flip, phase flip,
depolarizing, amplitude damping (T1 relaxation), and phase damping (T2
dephasing). Depolarizing follows Qiskit's own convention (the identity term
is included in the mixture), verified in tests/test_noise_models.py against
qiskit_aer.noise.depolarizing_error via SuperOp equality.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from qiskit_aer.noise import NoiseModel, QuantumError, kraus_error

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I = np.eye(2, dtype=complex)


def bit_flip_kraus(p: float) -> list[np.ndarray]:
    """Bit flip channel: with probability p, apply X. K0=sqrt(1-p)I, K1=sqrt(p)X."""
    return [np.sqrt(1 - p) * _I, np.sqrt(p) * _X]


def phase_flip_kraus(p: float) -> list[np.ndarray]:
    """Phase flip channel: with probability p, apply Z. K0=sqrt(1-p)I, K1=sqrt(p)Z."""
    return [np.sqrt(1 - p) * _I, np.sqrt(p) * _Z]


def depolarizing_kraus(p: float) -> list[np.ndarray]:
    """Depolarizing channel: rho -> (1-3p/4)rho + (p/4)(XrhoX+YrhoY+ZrhoZ).

    Matches Qiskit's own convention (identity included in the 4-way Pauli
    mixture), not the "p is the probability of a genuine error" convention
    sometimes seen elsewhere — see docs/interview_questions.md.
    """
    return [
        np.sqrt(1 - 3 * p / 4) * _I,
        np.sqrt(p / 4) * _X,
        np.sqrt(p / 4) * _Y,
        np.sqrt(p / 4) * _Z,
    ]


def amplitude_damping_kraus(gamma: float) -> list[np.ndarray]:
    """Amplitude damping channel (T1 relaxation toward |0>). Non-unital: fixed point is |0><0|."""
    k0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    k1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [k0, k1]


def phase_damping_kraus(lam: float) -> list[np.ndarray]:
    """Phase damping channel (T2 dephasing). Unital; preserves populations exactly."""
    k0 = np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex)
    k1 = np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)
    return [k0, k1]


CHANNEL_KRAUS_BUILDERS: dict[str, Callable[[float], list[np.ndarray]]] = {
    "bit_flip": bit_flip_kraus,
    "phase_flip": phase_flip_kraus,
    "depolarizing": depolarizing_kraus,
    "amplitude_damping": amplitude_damping_kraus,
    "phase_damping": phase_damping_kraus,
}


def single_qubit_noise_model(
    kraus_ops: list[np.ndarray], qubits: Sequence[int], gate: str = "id"
) -> NoiseModel:
    """Attach a single-qubit Kraus channel to `gate` on the given qubits.

    Used to inject noise into circuit-level simulation via
    simulation.sample_counts(..., noise_model=...); qubits must be given
    explicitly since a NoiseModel has no notion of "all qubits" without a
    backend to query.
    """
    error: QuantumError = kraus_error(kraus_ops)
    model = NoiseModel()
    model.add_quantum_error(error, [gate], list(qubits))
    return model
