import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import DensityMatrix, Kraus, Statevector

from src.circuits import bell_state
from src.experiment_utils import apply_noise_layers, error_rate, success_probability
from src.noise_models import bit_flip_kraus, depolarizing_kraus

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_apply_noise_layers_single_layer_matches_direct_evolve():
    rho = DensityMatrix(Statevector.from_instruction(bell_state("phi_plus")))
    p = 0.3

    expected = rho.evolve(Kraus(depolarizing_kraus(p)), qargs=[0])
    expected = expected.evolve(Kraus(depolarizing_kraus(p)), qargs=[1])

    actual = apply_noise_layers(rho, depolarizing_kraus, p, num_layers=1, qubits=[0, 1])
    assert np.allclose(actual.data, expected.data, atol=1e-8)


def test_apply_noise_layers_composes_across_layers():
    rho = DensityMatrix.from_label("+")
    p = 0.2

    once = apply_noise_layers(rho, bit_flip_kraus, p, num_layers=1, qubits=[0])
    twice_manual = once.evolve(Kraus(bit_flip_kraus(p)), qargs=[0])
    twice = apply_noise_layers(rho, bit_flip_kraus, p, num_layers=2, qubits=[0])

    assert np.allclose(twice.data, twice_manual.data, atol=1e-8)


def test_apply_noise_layers_zero_layers_is_noop():
    rho = DensityMatrix(Statevector.from_instruction(bell_state("phi_plus")))
    unchanged = apply_noise_layers(rho, depolarizing_kraus, 0.5, num_layers=0, qubits=[0, 1])
    assert np.allclose(unchanged.data, rho.data, atol=1e-8)


def test_success_probability_and_error_rate_are_complementary():
    counts = {"00": 3000, "11": 900, "01": 100, "10": 96}
    valid = {"00", "11"}
    succ = success_probability(counts, valid)
    err = error_rate(counts, valid)
    assert np.isclose(succ + err, 1.0)
    assert np.isclose(succ, 3900 / 4096)


def test_success_probability_all_valid():
    counts = {"00": 2048, "11": 2048}
    assert success_probability(counts, {"00", "11"}) == 1.0
    assert error_rate(counts, {"00", "11"}) == 0.0


def test_success_probability_none_valid():
    counts = {"01": 2048, "10": 2048}
    assert success_probability(counts, {"00", "11"}) == 0.0
    assert error_rate(counts, {"00", "11"}) == 1.0


@pytest.mark.slow
def test_sweep_script_runs_end_to_end_on_a_tiny_grid(tmp_path):
    # Smoke test: a deliberately tiny config exercises the same code path as
    # experiments/run_noise_sweep.py without the cost of the full committed grid.
    output_csv = tmp_path / "tiny_sweep.csv"
    config_path = tmp_path / "tiny_config.toml"
    config_path.write_text(
        f"""
[general]
shots = 256
seed = 42
output_csv = "{output_csv.as_posix()}"

[bell]
enabled = true
bell_type = "phi_plus"
channels = ["bit_flip"]
params = [0.0, 0.5]
depths = [1]

[ghz]
enabled = true
channels = ["depolarizing"]
params = [0.0, 0.3]
qubit_counts = [3]
depths = [1]
"""
    )

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "experiments" / "run_noise_sweep.py"), str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    with open(output_csv, "rb") as f:
        pass  # file exists check via open
    rows = output_csv.read_text().strip().splitlines()
    # header + (1 channel x 2 params x 1 depth) bell rows + (1 channel x 2 params x 1 qubit_count x 1 depth) ghz rows
    assert len(rows) == 1 + 2 + 2
