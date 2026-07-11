import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity

from src.circuits import ghz_state
from src.experiment_utils import apply_noise_layers
from src.noise_models import CHANNEL_KRAUS_BUILDERS
from src.research_analysis import (
    fit_geometric_decay_rate,
    predicted_dephasing_ghz_fidelity,
    predicted_phase_damping_decay_factor,
    predicted_phase_flip_decay_factor,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fit_geometric_decay_rate_recovers_known_rate():
    qubit_counts = list(range(2, 7))
    fidelities = [0.9**n for n in qubit_counts]

    r, r_squared = fit_geometric_decay_rate(qubit_counts, fidelities)

    assert np.isclose(r, 0.9, atol=1e-6)
    assert np.isclose(r_squared, 1.0, atol=1e-6)


def test_fit_geometric_decay_rate_on_noisy_data_still_recovers_approximate_rate():
    qubit_counts = list(range(2, 7))
    rng = np.random.default_rng(42)
    fidelities = [0.85**n * (1 + rng.normal(0, 0.01)) for n in qubit_counts]

    r, r_squared = fit_geometric_decay_rate(qubit_counts, fidelities)

    assert abs(r - 0.85) < 0.05
    assert r_squared > 0.9


@pytest.mark.parametrize("p", [0.0, 0.1, 0.25, 0.5])
def test_predicted_phase_flip_decay_factor_matches_hand_calc(p):
    assert predicted_phase_flip_decay_factor(p) == 1 - 2 * p


@pytest.mark.parametrize("lam", [0.0, 0.1, 0.25, 0.5])
def test_predicted_phase_damping_decay_factor_matches_hand_calc(lam):
    assert np.isclose(predicted_phase_damping_decay_factor(lam), np.sqrt(1 - lam))


@pytest.mark.parametrize(
    "channel, param, predictor",
    [
        ("phase_flip", 0.1, lambda p: predicted_phase_flip_decay_factor(p)),
        ("phase_damping", 0.1, lambda p: predicted_phase_damping_decay_factor(p)),
    ],
)
@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_dephasing_ghz_fidelity_closed_form_matches_simulation(channel, param, predictor, n):
    ideal_circuit = ghz_state(n)
    ideal_rho = DensityMatrix(Statevector.from_instruction(ideal_circuit))
    noisy_rho = apply_noise_layers(ideal_rho, CHANNEL_KRAUS_BUILDERS[channel], param, 1, list(range(n)))
    simulated_fidelity = state_fidelity(ideal_rho, noisy_rho)

    predicted_fidelity = predicted_dephasing_ghz_fidelity(predictor(param), n)
    assert np.isclose(simulated_fidelity, predicted_fidelity, atol=1e-9)


def test_dephasing_channels_never_drop_ghz_fidelity_below_half():
    for decay_factor in [0.0, 0.3, 0.7, 1.0]:
        for n in [2, 10, 50]:
            assert predicted_dephasing_ghz_fidelity(decay_factor, n) >= 0.5


def test_bit_flip_ranked_fastest_decaying_channel_on_committed_csv():
    csv_path = REPO_ROOT / "experiments" / "results" / "noise_sweep.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    ghz_rows = [
        row
        for row in rows
        if row["circuit"] == "ghz" and row["depth"] == "1" and np.isclose(float(row["param"]), 0.1)
    ]
    channels = sorted({row["channel"] for row in ghz_rows})

    rates = {}
    for channel in channels:
        channel_rows = sorted(
            (row for row in ghz_rows if row["channel"] == channel),
            key=lambda row: int(row["qubit_count"]),
        )
        qubit_counts = [int(row["qubit_count"]) for row in channel_rows]
        fidelities = [float(row["fidelity"]) for row in channel_rows]
        r, _ = fit_geometric_decay_rate(qubit_counts, fidelities)
        rates[channel] = r

    assert min(rates, key=rates.get) == "bit_flip"


@pytest.mark.slow
def test_ghz_decay_extension_script_runs_end_to_end_on_a_tiny_grid(tmp_path):
    output_csv = tmp_path / "tiny_extension.csv"
    config_path = tmp_path / "tiny_config.toml"
    config_path.write_text(
        f"""
[general]
output_csv = "{output_csv.as_posix()}"

[sweep]
channels = ["bit_flip", "phase_damping"]
qubit_counts = [3, 4]
params = [0.1]
depth = 1
"""
    )

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "experiments" / "run_ghz_decay_extension.py"), str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    rows = output_csv.read_text().strip().splitlines()
    # header + 2 qubit_counts x 2 channels x 1 param
    assert len(rows) == 1 + 4
