"""Reusable helpers for noise-comparison experiments (Part 3).

Generalizes Part 2's single-parameter, single-layer noise demos to a
qubit-count x depth x channel x parameter sweep: apply_noise_layers extends
the Bell-state-fidelity demo from notebook 02 (independent per-qubit Kraus
evolution) to an arbitrary number of qubits and repeated noise "layers"
(a proxy for circuit depth / idle time), and success_probability/error_rate
turn raw measurement counts into the two metrics named in CLAUDE.md
alongside fidelity.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from qiskit.quantum_info import DensityMatrix, Kraus


def apply_noise_layers(
    rho: DensityMatrix,
    kraus_builder: Callable[[float], list[np.ndarray]],
    param: float,
    num_layers: int,
    qubits: Sequence[int],
) -> DensityMatrix:
    """Apply a single-qubit channel independently to each qubit, num_layers times.

    Each layer applies kraus_builder(param) once per qubit in `qubits`, in place
    (returned as a new DensityMatrix). num_layers=1 reproduces the per-qubit
    evolve(...) calls used directly in notebook 02's Bell-state fidelity demo.
    """
    kraus_ops = Kraus(kraus_builder(param))
    for _ in range(num_layers):
        for qubit in qubits:
            rho = rho.evolve(kraus_ops, qargs=[qubit])
    return rho


def success_probability(counts: dict[str, int], valid_bitstrings: set[str]) -> float:
    """Fraction of shots landing in `valid_bitstrings` (e.g. the ideal noiseless support)."""
    total = sum(counts.values())
    hits = sum(count for bitstring, count in counts.items() if bitstring in valid_bitstrings)
    return hits / total


def error_rate(counts: dict[str, int], valid_bitstrings: set[str]) -> float:
    """Complement of success_probability: fraction of shots landing outside valid_bitstrings."""
    return 1.0 - success_probability(counts, valid_bitstrings)
