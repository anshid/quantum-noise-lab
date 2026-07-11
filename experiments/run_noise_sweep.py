"""Config-driven noise-comparison sweep (Part 3).

Varies noise channel, noise parameter, qubit count, and circuit depth (modeled as
repeated independent noise layers, see src.circuits.add_idle_layers), computing
fidelity (exact, via density-matrix evolution) alongside empirical success
probability and error rate (via shot-based sampling through a matching NoiseModel).
Writes a tidy CSV for notebooks/03_noise_experiments.ipynb to load and discuss.

Usage: python experiments/run_noise_sweep.py [path/to/config.toml]
"""

from __future__ import annotations

import csv
import logging
import sys
import tomllib
from pathlib import Path

from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity

from src.circuits import add_idle_layers, add_measurements, bell_state, ghz_state
from src.experiment_utils import apply_noise_layers, error_rate, success_probability
from src.noise_models import CHANNEL_KRAUS_BUILDERS, single_qubit_noise_model
from src.simulation import sample_counts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

FIELDNAMES = [
    "circuit",
    "qubit_count",
    "channel",
    "param",
    "depth",
    "fidelity",
    "success_probability",
    "error_rate",
]


def _noisy_success_and_error(
    measured_qc, kraus_builder, param, qubits, shots, seed, valid_bitstrings
) -> tuple[float, float]:
    noise_model = single_qubit_noise_model(kraus_builder(param), qubits=qubits, gate="id")
    counts = sample_counts(measured_qc, shots=shots, seed=seed, noise_model=noise_model)
    return success_probability(counts, valid_bitstrings), error_rate(counts, valid_bitstrings)


def run_bell_sweep(cfg: dict, general: dict, writer: csv.DictWriter) -> None:
    bell_type = cfg["bell_type"]
    channels = cfg["channels"]
    params = cfg["params"]
    depths = cfg["depths"]
    qubits = [0, 1]

    ideal_circuit = bell_state(bell_type)
    ideal_rho = DensityMatrix(Statevector.from_instruction(ideal_circuit))
    valid_bitstrings = {"00", "11"} if bell_type in ("phi_plus", "phi_minus") else {"01", "10"}

    for depth in depths:
        measured_qc = add_measurements(add_idle_layers(ideal_circuit, depth))
        for channel in channels:
            builder = CHANNEL_KRAUS_BUILDERS[channel]
            for param in params:
                noisy_rho = apply_noise_layers(ideal_rho, builder, param, depth, qubits)
                fidelity = state_fidelity(ideal_rho, noisy_rho)
                succ, err = _noisy_success_and_error(
                    measured_qc, builder, param, qubits, general["shots"], general["seed"], valid_bitstrings
                )
                writer.writerow(
                    {
                        "circuit": f"bell_{bell_type}",
                        "qubit_count": 2,
                        "channel": channel,
                        "param": param,
                        "depth": depth,
                        "fidelity": fidelity,
                        "success_probability": succ,
                        "error_rate": err,
                    }
                )
    logger.info(
        "Bell sweep complete: %d channels x %d params x %d depths",
        len(channels), len(params), len(depths),
    )


def run_ghz_sweep(cfg: dict, general: dict, writer: csv.DictWriter) -> None:
    channels = cfg["channels"]
    params = cfg["params"]
    depths = cfg["depths"]
    qubit_counts = cfg["qubit_counts"]

    for n in qubit_counts:
        ideal_circuit = ghz_state(n)
        ideal_rho = DensityMatrix(Statevector.from_instruction(ideal_circuit))
        valid_bitstrings = {"0" * n, "1" * n}
        qubits = list(range(n))

        for depth in depths:
            measured_qc = add_measurements(add_idle_layers(ideal_circuit, depth))
            for channel in channels:
                builder = CHANNEL_KRAUS_BUILDERS[channel]
                for param in params:
                    noisy_rho = apply_noise_layers(ideal_rho, builder, param, depth, qubits)
                    fidelity = state_fidelity(ideal_rho, noisy_rho)
                    succ, err = _noisy_success_and_error(
                        measured_qc, builder, param, qubits, general["shots"], general["seed"], valid_bitstrings
                    )
                    writer.writerow(
                        {
                            "circuit": "ghz",
                            "qubit_count": n,
                            "channel": channel,
                            "param": param,
                            "depth": depth,
                            "fidelity": fidelity,
                            "success_probability": succ,
                            "error_rate": err,
                        }
                    )
    logger.info(
        "GHZ sweep complete: %d qubit counts x %d channels x %d params x %d depths",
        len(qubit_counts), len(channels), len(params), len(depths),
    )


def main(config_path: Path) -> None:
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    general = config["general"]
    output_path = REPO_ROOT / general["output_csv"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        if config.get("bell", {}).get("enabled", False):
            run_bell_sweep(config["bell"], general, writer)
        if config.get("ghz", {}).get("enabled", False):
            run_ghz_sweep(config["ghz"], general, writer)

    logger.info("Wrote results to %s", output_path)


if __name__ == "__main__":
    config_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "experiments/configs/noise_sweep.toml"
    main(config_arg)
