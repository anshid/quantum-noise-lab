import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity

from src.circuits import add_idle_layers, add_measurements
from src.error_mitigation import (
    logical_error_rate_theoretical,
    majority_vote_decode,
    repetition_encode,
)
from src.experiment_utils import apply_noise_layers
from src.noise_models import bit_flip_kraus, phase_flip_kraus, single_qubit_noise_model
from src.simulation import sample_counts

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repetition_encode_all_zero_input_gives_all_000():
    encoded = repetition_encode(QuantumCircuit(1))
    measured_qc = add_measurements(encoded)
    counts = sample_counts(measured_qc, shots=512, seed=42)
    assert counts == {"000": 512}


def test_repetition_encode_x_gate_input_gives_all_111():
    message = QuantumCircuit(1)
    message.x(0)
    encoded = repetition_encode(message)
    measured_qc = add_measurements(encoded)
    counts = sample_counts(measured_qc, shots=512, seed=42)
    assert counts == {"111": 512}


def test_repetition_encode_rejects_multi_qubit_input():
    with pytest.raises(ValueError):
        repetition_encode(QuantumCircuit(2))


def test_majority_vote_decode_hand_checked_counts():
    counts = {"000": 10, "001": 5, "111": 3}
    decoded = majority_vote_decode(counts)
    # "000" -> 0 (10), "001" has one "1" -> majority 0 (5), "111" -> 1 (3)
    assert decoded == {"0": 15, "1": 3}


@pytest.mark.parametrize(
    "bitstring,expected",
    [
        ("000", "0"),
        ("001", "0"),
        ("010", "0"),
        ("100", "0"),
        ("011", "1"),
        ("101", "1"),
        ("110", "1"),
        ("111", "1"),
    ],
)
def test_majority_vote_decode_never_ties(bitstring, expected):
    decoded = majority_vote_decode({bitstring: 1})
    assert decoded[expected] == 1
    assert sum(decoded.values()) == 1


@pytest.mark.parametrize("p", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
def test_logical_error_rate_theoretical_matches_binomial_tail(p):
    binomial_tail = sum(math.comb(3, k) * p**k * (1 - p) ** (3 - k) for k in (2, 3))
    assert np.isclose(logical_error_rate_theoretical(p), binomial_tail)


def test_logical_error_rate_crossover_at_p_half():
    assert logical_error_rate_theoretical(0.5) == 0.5


@pytest.mark.parametrize("p", [0.05, 0.2, 0.4])
def test_monte_carlo_logical_error_rate_matches_theoretical(p):
    encoded = repetition_encode(QuantumCircuit(1))
    measured_qc = add_measurements(add_idle_layers(encoded, 1))
    noise_model = single_qubit_noise_model(bit_flip_kraus(p), qubits=[0, 1, 2], gate="id")
    counts = sample_counts(measured_qc, shots=8192, seed=42, noise_model=noise_model)
    decoded = majority_vote_decode(counts)
    empirical = decoded["1"] / sum(decoded.values())
    assert abs(empirical - logical_error_rate_theoretical(p)) < 0.05


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 1.0])
def test_phase_flip_noise_is_invisible_to_majority_vote_on_computational_input(p):
    # Z|0> = |0> (up to an unobservable global phase), so phase-flip noise is an
    # exact no-op on a computational-basis codeword: majority vote reports zero
    # logical error for every p, not because the code corrects phase flips, but
    # because a Z-basis measurement of a Z-basis eigenstate can never see them.
    encoded = repetition_encode(QuantumCircuit(1))
    measured_qc = add_measurements(add_idle_layers(encoded, 1))
    noise_model = single_qubit_noise_model(phase_flip_kraus(p), qubits=[0, 1, 2], gate="id")
    counts = sample_counts(measured_qc, shots=8192, seed=42, noise_model=noise_model)
    decoded = majority_vote_decode(counts)
    assert decoded == {"0": 8192, "1": 0}


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5])
def test_phase_flip_noise_still_degrades_true_state_fidelity_of_a_superposition_input(p):
    # The classical decoded-bit metric above is blind, but the quantum state is
    # not actually protected: encoding a superposition (|+>) turns phase-flip
    # noise on the physical qubits into a real, measurable loss of fidelity,
    # invisible to the majority-vote/Z-basis view used elsewhere in this module.
    message = QuantumCircuit(1)
    message.h(0)
    ideal_rho = DensityMatrix(Statevector.from_instruction(repetition_encode(message)))
    noisy_rho = apply_noise_layers(ideal_rho, phase_flip_kraus, p, num_layers=1, qubits=[0, 1, 2])
    assert state_fidelity(ideal_rho, noisy_rho) < 1.0 - 1e-9


@pytest.mark.slow
def test_mitigation_sweep_script_runs_end_to_end_on_a_tiny_grid(tmp_path):
    output_csv = tmp_path / "tiny_mitigation.csv"
    config_path = tmp_path / "tiny_config.toml"
    config_path.write_text(
        f"""
[general]
shots = 256
seed = 42
output_csv = "{output_csv.as_posix()}"

[sweep]
params = [0.0, 0.5, 1.0]
"""
    )

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "experiments" / "run_mitigation_sweep.py"), str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    rows = output_csv.read_text().strip().splitlines()
    assert len(rows) == 1 + 3
