"""Targeted GHZ fidelity-decay extension sweep (Part 5).

Part 3's noise_sweep.csv caps GHZ qubit count at 6 (O(4^n) density-matrix
cost). This script extends just the two extreme channels identified in
Part 5's analysis -- bit_flip (fastest-decaying) and phase_damping (slowest
plateau) -- to qubit counts 7-8, to confirm the geometric-decay trend fitted
from n=2-6 actually extrapolates past Part 3's cap.

Usage: python experiments/run_ghz_decay_extension.py [path/to/config.toml]
"""

from __future__ import annotations

import csv
import logging
import sys
import tomllib
from pathlib import Path

from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity

from src.circuits import ghz_state
from src.experiment_utils import apply_noise_layers
from src.noise_models import CHANNEL_KRAUS_BUILDERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

FIELDNAMES = ["channel", "qubit_count", "param", "fidelity"]


def run_sweep(cfg: dict, writer: csv.DictWriter) -> None:
    channels = cfg["channels"]
    qubit_counts = cfg["qubit_counts"]
    params = cfg["params"]
    depth = cfg["depth"]

    for n in qubit_counts:
        ideal_circuit = ghz_state(n)
        ideal_rho = DensityMatrix(Statevector.from_instruction(ideal_circuit))
        qubits = list(range(n))

        for channel in channels:
            builder = CHANNEL_KRAUS_BUILDERS[channel]
            for param in params:
                noisy_rho = apply_noise_layers(ideal_rho, builder, param, depth, qubits)
                fidelity = state_fidelity(ideal_rho, noisy_rho)
                writer.writerow(
                    {
                        "channel": channel,
                        "qubit_count": n,
                        "param": param,
                        "fidelity": fidelity,
                    }
                )
    logger.info(
        "GHZ decay extension sweep complete: %d qubit counts x %d channels x %d params",
        len(qubit_counts), len(channels), len(params),
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
        run_sweep(config["sweep"], writer)

    logger.info("Wrote results to %s", output_path)


if __name__ == "__main__":
    config_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "experiments/configs/ghz_decay_extension.toml"
    main(config_arg)
