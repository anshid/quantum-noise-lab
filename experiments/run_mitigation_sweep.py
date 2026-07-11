"""Config-driven repetition-code + majority-vote sweep (Part 4).

For each physical bit-flip probability p, encodes |0> into the 3-qubit
repetition code (src.error_mitigation.repetition_encode), applies
independent bit-flip noise to each physical qubit, measures, and decodes
via majority vote. Records the theoretical closed-form logical error rate
alongside the Monte Carlo estimate and the unmitigated single-qubit baseline
(= p), for notebooks/04_error_mitigation.ipynb to load and discuss.

Usage: python experiments/run_mitigation_sweep.py [path/to/config.toml]
"""

from __future__ import annotations

import csv
import logging
import sys
import tomllib
from pathlib import Path

from qiskit import QuantumCircuit

from src.circuits import add_idle_layers, add_measurements
from src.error_mitigation import (
    logical_error_rate_theoretical,
    majority_vote_decode,
    repetition_encode,
)
from src.noise_models import bit_flip_kraus, single_qubit_noise_model
from src.simulation import sample_counts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

FIELDNAMES = [
    "p",
    "physical_error_rate",
    "logical_error_rate_theoretical",
    "logical_error_rate_empirical",
]


def run_sweep(cfg: dict, general: dict, writer: csv.DictWriter) -> None:
    params = cfg["params"]

    encoded = repetition_encode(QuantumCircuit(1))
    # single idle layer gives single_qubit_noise_model's "id"-gate hook something
    # to attach to on each physical qubit (same convention as Parts 2-3).
    measured_qc = add_measurements(add_idle_layers(encoded, 1))

    for p in params:
        noise_model = single_qubit_noise_model(bit_flip_kraus(p), qubits=[0, 1, 2], gate="id")
        counts = sample_counts(measured_qc, shots=general["shots"], seed=general["seed"], noise_model=noise_model)
        decoded = majority_vote_decode(counts)
        empirical_logical_error_rate = decoded["1"] / sum(decoded.values())

        writer.writerow(
            {
                "p": p,
                "physical_error_rate": p,
                "logical_error_rate_theoretical": logical_error_rate_theoretical(p),
                "logical_error_rate_empirical": empirical_logical_error_rate,
            }
        )
    logger.info("Mitigation sweep complete: %d values of p", len(params))


def main(config_path: Path) -> None:
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    general = config["general"]
    output_path = REPO_ROOT / general["output_csv"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        run_sweep(config["sweep"], general, writer)

    logger.info("Wrote results to %s", output_path)


if __name__ == "__main__":
    config_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "experiments/configs/mitigation.toml"
    main(config_arg)
