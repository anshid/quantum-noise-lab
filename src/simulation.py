"""Circuit execution helpers: exact statevectors and shot-based sampling.

Sampling targets Qiskit's modern primitives API (qiskit_aer.primitives.SamplerV2)
rather than the deprecated execute()/backend.run() pattern, since this is the
call shape that will later carry a noise_model= into AerSimulator (Part 2).
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer.primitives import SamplerV2


def get_statevector(qc: QuantumCircuit) -> Statevector:
    """Return the exact statevector of an unmeasured circuit.

    Raises if `qc` contains measurement or reset instructions, since those are
    non-unitary and have no well-defined statevector.
    """
    return Statevector.from_instruction(qc)


def sample_counts(qc: QuantumCircuit, shots: int = 4096, seed: int | None = 42) -> dict[str, int]:
    """Run a measured circuit through AerSimulator (via SamplerV2) and return raw bitstring counts.

    `qc` must already contain measurements (see circuits.add_measurements). The classical
    register created by add_measurements is named "meas".
    """
    sampler = SamplerV2(seed=seed)
    result = sampler.run([qc], shots=shots).result()
    data_bin = result[0].data
    creg_name = qc.cregs[0].name
    return getattr(data_bin, creg_name).get_counts()


def probabilities_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """Normalize raw shot counts into empirical probabilities."""
    total = sum(counts.values())
    return {bitstring: count / total for bitstring, count in counts.items()}


def theoretical_probabilities(qc: QuantumCircuit) -> dict[str, float]:
    """Return exact Born-rule probabilities from the statevector of an unmeasured circuit."""
    statevector = get_statevector(qc)
    return statevector.probabilities_dict()
