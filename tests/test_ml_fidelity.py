import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml_fidelity import (
    PHYSICS_BASELINE_CHANNELS,
    build_feature_target,
    evaluate,
    extrapolation_split,
    interpolation_split,
    physics_baseline_fidelity,
    train_model,
)
from src.research_analysis import predicted_dephasing_ghz_fidelity, predicted_phase_flip_decay_factor

REPO_ROOT = Path(__file__).resolve().parents[1]


def _toy_sweep_df() -> pd.DataFrame:
    rows = []
    for circuit, n in [("bell_phi_plus", 2), ("ghz", 4)]:
        for channel in ["bit_flip", "phase_flip", "depolarizing", "amplitude_damping", "phase_damping"]:
            for param in [0.0, 0.1, 0.2, 0.3]:
                for depth in [1, 2, 4]:
                    rows.append(
                        {
                            "circuit": circuit,
                            "qubit_count": n,
                            "channel": channel,
                            "param": param,
                            "depth": depth,
                            "fidelity": max(0.5, 1.0 - param * depth),
                            "success_probability": 1.0 - param,
                            "error_rate": param,
                        }
                    )
    return pd.DataFrame(rows)


def test_interpolation_split_is_disjoint_and_stratified_by_channel():
    df = _toy_sweep_df()
    train_df, test_df = interpolation_split(df, test_size=0.25, seed=42)

    assert set(train_df.index).isdisjoint(set(test_df.index))
    assert len(train_df) + len(test_df) == len(df)
    for channel in df["channel"].unique():
        train_frac = (train_df["channel"] == channel).mean()
        test_frac = (test_df["channel"] == channel).mean()
        assert abs(train_frac - test_frac) < 0.05


def test_interpolation_split_is_deterministic_under_fixed_seed():
    df = _toy_sweep_df()
    train_a, test_a = interpolation_split(df, seed=7)
    train_b, test_b = interpolation_split(df, seed=7)
    assert list(train_a.index) == list(train_b.index)
    assert list(test_a.index) == list(test_b.index)


def test_extrapolation_split_holds_out_only_requested_values():
    df = _toy_sweep_df()
    train_df, test_df = extrapolation_split(df, feature="depth", holdout_values=[4])

    assert set(train_df.index).isdisjoint(set(test_df.index))
    assert (train_df["depth"] != 4).all()
    assert (test_df["depth"] == 4).all()
    assert len(train_df) + len(test_df) == len(df)


def test_build_feature_target_selects_expected_columns():
    df = _toy_sweep_df()
    X, y = build_feature_target(df)

    assert list(X.columns) == ["circuit", "channel", "qubit_count", "param", "depth"]
    assert y.name == "fidelity"
    assert len(X) == len(y) == len(df)


@pytest.mark.parametrize("model_type", ["linear", "random_forest"])
def test_train_and_evaluate_recovers_a_known_linear_relationship(model_type):
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "circuit": rng.choice(["bell_phi_plus", "ghz"], size=n),
            "channel": rng.choice(["bit_flip", "phase_flip"], size=n),
            "qubit_count": rng.integers(2, 7, size=n),
            "param": rng.uniform(0, 0.5, size=n),
            "depth": rng.integers(1, 4, size=n),
        }
    )
    df["fidelity"] = 1.0 - 0.5 * df["param"]

    X, y = build_feature_target(df)
    model = train_model(X, y, model_type, seed=42, **({"n_estimators": 50} if model_type == "random_forest" else {}))
    scores = evaluate(model, X, y)

    assert scores["r2"] > 0.9
    assert scores["mae"] < 0.05


def test_physics_baseline_returns_none_for_non_dephasing_channels():
    row = pd.Series({"circuit": "ghz", "channel": "bit_flip", "qubit_count": 4, "param": 0.2, "depth": 1})
    assert physics_baseline_fidelity(row) is None


@pytest.mark.parametrize("channel", sorted(PHYSICS_BASELINE_CHANNELS))
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_physics_baseline_matches_research_analysis_closed_form_scaled_by_depth(channel, depth):
    n, param = 4, 0.2
    row = pd.Series({"circuit": "ghz", "channel": channel, "qubit_count": n, "param": param, "depth": depth})

    if channel == "phase_flip":
        decay_factor = predicted_phase_flip_decay_factor(param)
    else:
        from src.research_analysis import predicted_phase_damping_decay_factor

        decay_factor = predicted_phase_damping_decay_factor(param)
    expected = predicted_dephasing_ghz_fidelity(decay_factor, n * depth)

    assert np.isclose(physics_baseline_fidelity(row), expected)


@pytest.mark.parametrize("channel", sorted(PHYSICS_BASELINE_CHANNELS))
def test_physics_baseline_matches_committed_sweep_for_bell_and_ghz(channel):
    csv_path = REPO_ROOT / "experiments" / "results" / "noise_sweep.csv"
    df = pd.read_csv(csv_path)
    subset = df[(df["channel"] == channel) & np.isclose(df["param"], 0.2)]
    assert len(subset) > 0

    for _, row in subset.iterrows():
        predicted = physics_baseline_fidelity(row)
        assert np.isclose(predicted, row["fidelity"], atol=1e-9)


@pytest.mark.slow
def test_run_ml_fidelity_script_runs_end_to_end_on_a_tiny_grid(tmp_path):
    df = _toy_sweep_df()
    input_csv = tmp_path / "tiny_sweep.csv"
    df.to_csv(input_csv, index=False)

    metrics_csv = tmp_path / "metrics.csv"
    importance_csv = tmp_path / "importance.csv"
    config_path = tmp_path / "tiny_config.toml"
    config_path.write_text(
        f"""
[general]
input_csv = "{input_csv.as_posix()}"
metrics_csv = "{metrics_csv.as_posix()}"
feature_importance_csv = "{importance_csv.as_posix()}"
seed = 42

[interpolation_split]
test_size = 0.25

[extrapolation_split_qubit_count]
feature = "qubit_count"
holdout_values = [4]

[extrapolation_split_depth]
feature = "depth"
holdout_values = [4]

[random_forest]
n_estimators = 20
max_depth = 4
"""
    )

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "experiments" / "run_ml_fidelity.py"), str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    metrics = pd.read_csv(metrics_csv)
    assert {"linear", "random_forest", "physics_baseline"}.issubset(set(metrics["model"]))
    assert {"interpolation", "extrapolation_qubit_count", "extrapolation_depth"}.issubset(set(metrics["split"]))

    importance = pd.read_csv(importance_csv)
    assert {"gini_encoded", "permutation_original"}.issubset(set(importance["importance_type"]))
