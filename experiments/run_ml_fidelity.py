"""ML fidelity regression vs. physics closed form, config-driven (Part 6).

Trains LinearRegression and RandomForestRegressor on Part 3's committed noise
sweep to predict fidelity from (circuit, qubit_count, channel, param, depth),
evaluated under three splits: one interpolation split (random, stratified by
channel) and two extrapolation splits (holding out the top of the qubit_count
and depth ranges respectively). For the two dephasing channels with a Part 5
closed form (phase_flip, phase_damping), the physics-baseline prediction is
scored against the same test rows as a non-ML reference point. Also reports
Random Forest feature importances (Gini, over the one-hot-encoded columns, and
permutation, over the original feature names) on the interpolation split.

Usage: python experiments/run_ml_fidelity.py [path/to/config.toml]
"""

from __future__ import annotations

import csv
import logging
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.ml_fidelity import (
    build_feature_target,
    extrapolation_split,
    interpolation_split,
    physics_baseline_fidelity,
    train_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

METRICS_FIELDNAMES = ["split", "model", "dataset", "n_rows", "r2", "mae", "rmse"]
IMPORTANCE_FIELDNAMES = ["feature", "importance_type", "importance"]


def _score(y_true: pd.Series, y_pred) -> dict[str, float]:
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def _write_metric_row(writer: csv.DictWriter, split: str, model: str, dataset: str, n_rows: int, scores: dict) -> None:
    writer.writerow({"split": split, "model": model, "dataset": dataset, "n_rows": n_rows, **scores})


def _physics_baseline_scores(df: pd.DataFrame) -> tuple[dict, int] | tuple[None, int]:
    """Physics-baseline predictions restricted to rows with a closed form (see
    src/ml_fidelity.py::physics_baseline_fidelity); returns (scores, n_rows), or
    (None, 0) if no eligible rows are present in this split.
    """
    predictions = df.apply(physics_baseline_fidelity, axis=1)
    eligible = predictions.notna()
    if not eligible.any():
        return None, 0
    return _score(df.loc[eligible, "fidelity"], predictions[eligible]), int(eligible.sum())


def run_split(split_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, rf_kwargs: dict, seed: int, metrics_writer: csv.DictWriter) -> dict:
    """Train both models on train_df, score both on train_df/test_df, and score the
    physics baseline on test_df's eligible rows. Returns the fitted random_forest
    pipeline (used by the caller to compute feature importances on the
    interpolation split only).
    """
    X_train, y_train = build_feature_target(train_df)
    X_test, y_test = build_feature_target(test_df)

    fitted = {}
    for model_type, kwargs in [("linear", {}), ("random_forest", rf_kwargs)]:
        model = train_model(X_train, y_train, model_type, seed=seed, **kwargs)
        fitted[model_type] = model
        _write_metric_row(metrics_writer, split_name, model_type, "train", len(train_df), _score(y_train, model.predict(X_train)))
        _write_metric_row(metrics_writer, split_name, model_type, "test", len(test_df), _score(y_test, model.predict(X_test)))

    physics_scores, n_physics_rows = _physics_baseline_scores(test_df)
    if physics_scores is not None:
        _write_metric_row(metrics_writer, split_name, "physics_baseline", "test", n_physics_rows, physics_scores)
    else:
        logger.info("No physics-baseline-eligible rows in the %s test split; skipping.", split_name)

    logger.info("Split '%s' complete: %d train rows, %d test rows.", split_name, len(train_df), len(test_df))
    return fitted


def write_feature_importance(model, X_test: pd.DataFrame, y_test: pd.Series, seed: int, writer: csv.DictWriter) -> None:
    preprocessor = model.named_steps["preprocess"]
    forest = model.named_steps["model"]

    encoded_names = preprocessor.get_feature_names_out()
    for name, importance in zip(encoded_names, forest.feature_importances_):
        writer.writerow({"feature": name, "importance_type": "gini_encoded", "importance": importance})

    perm = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=seed)
    for name, importance in zip(X_test.columns, perm.importances_mean):
        writer.writerow({"feature": name, "importance_type": "permutation_original", "importance": importance})


def main(config_path: Path) -> None:
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    general = config["general"]
    seed = general["seed"]
    df = pd.read_csv(REPO_ROOT / general["input_csv"])

    metrics_path = REPO_ROOT / general["metrics_csv"]
    importance_path = REPO_ROOT / general["feature_importance_csv"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    rf_kwargs = dict(config["random_forest"])

    with open(metrics_path, "w", newline="") as f:
        metrics_writer = csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES)
        metrics_writer.writeheader()

        interp_cfg = config["interpolation_split"]
        train_df, test_df = interpolation_split(df, test_size=interp_cfg["test_size"], seed=seed)
        interp_models = run_split("interpolation", train_df, test_df, rf_kwargs, seed, metrics_writer)

        qc_cfg = config["extrapolation_split_qubit_count"]
        train_df, test_df = extrapolation_split(df, qc_cfg["feature"], qc_cfg["holdout_values"])
        run_split("extrapolation_qubit_count", train_df, test_df, rf_kwargs, seed, metrics_writer)

        depth_cfg = config["extrapolation_split_depth"]
        train_df, test_df = extrapolation_split(df, depth_cfg["feature"], depth_cfg["holdout_values"])
        run_split("extrapolation_depth", train_df, test_df, rf_kwargs, seed, metrics_writer)

    logger.info("Wrote metrics to %s", metrics_path)

    # Feature importance is reported only for the interpolation split's random
    # forest -- extrapolation splits deliberately unbalance the feature
    # distribution (e.g. no high qubit_count in train), which would confound
    # importance with the split itself.
    _, interp_test_df = interpolation_split(df, test_size=config["interpolation_split"]["test_size"], seed=seed)
    X_test, y_test = build_feature_target(interp_test_df)
    with open(importance_path, "w", newline="") as f:
        importance_writer = csv.DictWriter(f, fieldnames=IMPORTANCE_FIELDNAMES)
        importance_writer.writeheader()
        write_feature_importance(interp_models["random_forest"], X_test, y_test, seed, importance_writer)

    logger.info("Wrote feature importances to %s", importance_path)


if __name__ == "__main__":
    config_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "experiments/configs/ml_fidelity.toml"
    main(config_arg)
