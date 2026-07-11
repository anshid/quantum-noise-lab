"""Reusable, pure circuit-construction helpers.

No execution or measurement-sampling logic lives here (see simulation.py) —
these functions only build QuantumCircuit objects.
"""

from __future__ import annotations

from typing import Sequence

from qiskit import ClassicalRegister, QuantumCircuit

_BELL_GATES = {
    "phi_plus": (False, False),   # (apply_z_before_cx, apply_x_after_cx)
    "phi_minus": (True, False),
    "psi_plus": (False, True),
    "psi_minus": (True, True),
}


def bell_state(bell_type: str = "phi_plus", *, measure: bool = False) -> QuantumCircuit:
    """Build one of the four maximally-entangled 2-qubit Bell states.

    bell_type is one of "phi_plus" (|00>+|11>)/sqrt2, "phi_minus" (|00>-|11>)/sqrt2,
    "psi_plus" (|01>+|10>)/sqrt2, "psi_minus" (|01>-|10>)/sqrt2.
    """
    if bell_type not in _BELL_GATES:
        raise ValueError(f"Unknown bell_type {bell_type!r}, expected one of {sorted(_BELL_GATES)}")
    apply_z, apply_x = _BELL_GATES[bell_type]

    qc = QuantumCircuit(2, name=f"bell_{bell_type}")
    qc.h(0)
    if apply_z:
        qc.z(0)
    qc.cx(0, 1)
    if apply_x:
        qc.x(1)

    return add_measurements(qc) if measure else qc


def ghz_state(num_qubits: int, *, measure: bool = False) -> QuantumCircuit:
    """Build an n-qubit GHZ state (|0...0> + |1...1>) / sqrt(2)."""
    if num_qubits < 2:
        raise ValueError("GHZ state requires at least 2 qubits")

    qc = QuantumCircuit(num_qubits, name=f"ghz_{num_qubits}")
    qc.h(0)
    for target in range(1, num_qubits):
        qc.cx(0, target)

    return add_measurements(qc) if measure else qc


def uniform_superposition(num_qubits: int, *, measure: bool = False) -> QuantumCircuit:
    """Build H^{otimes n} |0...0>, an equal superposition over all 2^n basis states."""
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")

    qc = QuantumCircuit(num_qubits, name=f"uniform_superposition_{num_qubits}")
    qc.h(range(num_qubits))

    return add_measurements(qc) if measure else qc


def single_qubit_superposition(theta: float, phi: float = 0.0, *, measure: bool = False) -> QuantumCircuit:
    """Build cos(theta/2)|0> + e^{i*phi} sin(theta/2)|1> via the U gate, for Bloch-sphere demos."""
    qc = QuantumCircuit(1, name="single_qubit_superposition")
    qc.u(theta, phi, 0.0, 0)

    return add_measurements(qc) if measure else qc


def add_measurements(qc: QuantumCircuit, qubits: Sequence[int] | None = None) -> QuantumCircuit:
    """Return a copy of qc with a classical register and measurements appended.

    Measures all qubits if `qubits` is None, else only the given qubit indices.
    """
    target_qubits = list(range(qc.num_qubits)) if qubits is None else list(qubits)

    measured = qc.copy()
    creg = ClassicalRegister(len(target_qubits), name="meas")
    measured.add_register(creg)
    measured.barrier()
    measured.measure(target_qubits, creg)

    return measured


def add_idle_layers(qc: QuantumCircuit, num_layers: int) -> QuantumCircuit:
    """Return a copy of qc with num_layers barrier+id(all qubits) blocks appended.

    A depth proxy: each layer is a natural anchor point for one independent
    application of a noise channel per qubit (see noise_models.single_qubit_noise_model
    and experiment_utils.apply_noise_layers), standing in for one unit of idle time
    or circuit depth.
    """
    layered = qc.copy()
    for _ in range(num_layers):
        layered.barrier()
        layered.id(range(layered.num_qubits))

    return layered
