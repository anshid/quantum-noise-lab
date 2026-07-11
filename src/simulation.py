"""Circuit execution helpers: exact statevectors and shot-based sampling.

Sampling targets Qiskit's modern primitives API (qiskit_aer.primitives.SamplerV2)
rather than the deprecated execute()/backend.run() pattern. As foreshadowed in
Part 1, the same call shape now carries an optional noise_model= into
AerSimulator (Part 2) via SamplerV2's backend_options.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Kraus, Statevector
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import SamplerV2


def get_statevector(qc: QuantumCircuit) -> Statevector:
    """Return the exact statevector of an unmeasured circuit.

    Raises if `qc` contains measurement or reset instructions, since those are
    non-unitary and have no well-defined statevector.
    """
    return Statevector.from_instruction(qc)


def sample_counts(
    qc: QuantumCircuit,
    shots: int = 4096,
    seed: int | None = 42,
    noise_model: NoiseModel | None = None,
) -> dict[str, int]:
    """Run a measured circuit through AerSimulator (via SamplerV2) and return raw bitstring counts.

    `qc` must already contain measurements (see circuits.add_measurements). The classical
    register created by add_measurements is named "meas". Pass `noise_model` (see
    noise_models.single_qubit_noise_model) to inject noise into the simulation; omitting it
    reproduces the exact noiseless Part-1 behavior.
    """
    options = {"backend_options": {"noise_model": noise_model}} if noise_model is not None else None
    sampler = SamplerV2(seed=seed, options=options)
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


def bloch_vector(rho: DensityMatrix) -> tuple[float, float, float]:
    """Return (Tr(rho X), Tr(rho Y), Tr(rho Z)), the Bloch-vector coordinates of a single qubit."""
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    return (
        float(np.real(np.trace(rho.data @ x))),
        float(np.real(np.trace(rho.data @ y))),
        float(np.real(np.trace(rho.data @ z))),
    )


def evolve_density_matrix(rho: DensityMatrix, kraus_ops: list[np.ndarray]) -> DensityMatrix:
    """Apply a Kraus-operator channel to a density matrix: rho -> sum_i K_i rho K_i^dagger."""
    return rho.evolve(Kraus(kraus_ops))
