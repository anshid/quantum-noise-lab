"""3-qubit bit-flip repetition code + majority-vote decoding (Part 4).

This protects a classical bit value (or, since the encoding is a linear CNOT
map, more generally the Z-basis populations of an arbitrary single-qubit
input) against independent bit-flip noise on each of the 3 physical qubits.
It is deliberately not full quantum error correction: majority-vote decoding
requires a direct computational-basis measurement of all 3 physical qubits,
which collapses any encoded superposition — see docs/interview_questions.md
Part 4 for the contrast with stabilizer/syndrome-based QEC.
"""

from __future__ import annotations

from qiskit import QuantumCircuit


def repetition_encode(qc: QuantumCircuit) -> QuantumCircuit:
    """Encode a 1-qubit circuit into the 3-qubit bit-flip repetition code.

    qc must be an unmeasured, 1-qubit circuit (the "message"). Returns a
    fresh 3-qubit circuit: qc's qubit 0 is copied onto qubits 1 and 2 via
    2 CNOTs, so all three physical qubits carry the same Z-basis value.
    """
    if qc.num_qubits != 1:
        raise ValueError(f"repetition_encode expects a 1-qubit circuit, got {qc.num_qubits}")

    encoded = QuantumCircuit(3, name="repetition_encoded")
    encoded.compose(qc, qubits=[0], inplace=True)
    encoded.cx(0, 1)
    encoded.cx(0, 2)

    return encoded


def majority_vote_decode(counts: dict[str, int]) -> dict[str, int]:
    """Collapse raw 3-qubit measurement counts to a decoded logical bit via majority vote.

    Each key in `counts` is a 3-character bitstring covering the 3 physical
    code qubits (in any consistent order — majority-of-3 is order-independent
    since it only counts how many characters are "1"). A 3-bit majority never
    ties. Returns a 2-key dict {"0": n0, "1": n1} of decoded logical-bit counts.
    """
    decoded = {"0": 0, "1": 0}
    for bitstring, count in counts.items():
        ones = bitstring.count("1")
        logical_bit = "1" if ones >= 2 else "0"
        decoded[logical_bit] += count

    return decoded


def logical_error_rate_theoretical(p: float) -> float:
    """Probability that majority vote decodes incorrectly, given per-qubit bit-flip rate p.

    Equals P[Binomial(3, p) >= 2] = C(3,2)p^2(1-p) + C(3,3)p^3 = 3p^2 - 2p^3,
    the probability that 2 or 3 of the 3 physical qubits flip.
    """
    return 3 * p**2 - 2 * p**3
