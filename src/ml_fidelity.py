"""ML fidelity regression vs. the Part 5 physics closed form (Part 6).

Frames "predict fidelity from (circuit, qubit_count, channel, param, depth)" as
ordinary supervised regression on Part 3's committed sweep, but the point of this
module isn't the R^2 of a single fit -- a 670-row grid is trivial to interpolate,
even for a lookup table. The interesting question is generalization: does a model
recover the *trend* past the edge of the training grid, or only memorize it?

Two split strategies capture this distinction (see interpolation_split and
extrapolation_split below), and two model families bracket the bias-variance
tradeoff at stake: LinearRegression (high bias -- cannot represent the
non-monotonic/saturating decay shapes already derived by hand in Part 5) vs.
RandomForestRegressor (low bias, but piecewise-constant on axis-aligned splits, so
it cannot extrapolate a trend past the max feature value seen in training -- it can
only predict the nearest training leaf's average). physics_baseline_fidelity gives
a third, non-ML reference point for the two channels (phase_flip, phase_damping)
where Part 5 already derived an exact closed form, so the ML model's error can be
compared against a mechanistic prediction rather than only against held-out labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.research_analysis import (
    predicted_dephasing_ghz_fidelity,
    predicted_phase_damping_decay_factor,
    predicted_phase_flip_decay_factor,
)

CATEGORICAL_FEATURES = ["circuit", "channel"]
NUMERIC_FEATURES = ["qubit_count", "param", "depth"]
TARGET = "fidelity"

ModelType = Literal["linear", "random_forest"]

# phase_flip and phase_damping are the only channels with diagonal Kraus operators
# (see src/research_analysis.py), so they're the only ones with a closed-form GHZ
# fidelity prediction to compare the ML model against.
PHYSICS_BASELINE_CHANNELS = {"phase_flip", "phase_damping"}


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load Part 3's noise sweep CSV as-is; this module never regenerates it."""
    return pd.read_csv(csv_path)


def build_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a sweep dataframe into (X, y). Categorical encoding happens in build_pipeline,
    not here, so every split shares one fitted encoder instead of each being encoded alone.
    """
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def build_pipeline(model_type: ModelType, seed: int = 42, **model_kwargs) -> Pipeline:
    """One-hot encode circuit/channel, passthrough the numeric features, then regress."""
    preprocessor = ColumnTransformer(
        transformers=[("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(random_state=seed, **model_kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def interpolation_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random train/test split stratified by channel.

    Every test row's features sit inside the training grid's range on every axis --
    the easiest generalization question a model can be asked.
    """
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df["channel"])


def extrapolation_split(
    df: pd.DataFrame, feature: str, holdout_values: Sequence[float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on df[feature] outside holdout_values, test on df[feature] inside them.

    Test rows lie past the edge of the training range on `feature` -- this is the
    harder question of whether the model learned the underlying trend or only the
    training grid (a tree ensemble in particular cannot extrapolate past the max
    feature value it was trained on; it predicts the nearest leaf's training mean).
    """
    test_mask = df[feature].isin(holdout_values)
    return df[~test_mask], df[test_mask]


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_type: ModelType, seed: int = 42, **model_kwargs) -> Pipeline:
    """Fit a fresh pipeline (preprocessing + model) on the given training rows."""
    pipeline = build_pipeline(model_type, seed=seed, **model_kwargs)
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """R^2, MAE, and RMSE of model's predictions against true fidelities y."""
    y_pred = model.predict(X)
    return {
        "r2": r2_score(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "rmse": mean_squared_error(y, y_pred) ** 0.5,
    }


def physics_baseline_fidelity(row: pd.Series) -> float | None:
    """Closed-form Part 5 fidelity prediction for a dephasing-channel row, else None.

    Generalizes predicted_dephasing_ghz_fidelity to depth > 1: each of `depth`
    sequential noise layers multiplies a qubit's coherence contribution by the same
    per-layer decay_factor, so the total exponent over the whole circuit is
    qubit_count * depth (confirmed against the committed sweep: e.g. GHZ n=4,
    phase_flip, p=0.2 gives fidelity 0.5648/0.5084/0.5001 at depth 1/2/4, matching
    (1 + (1-2p)**(depth*n))/2 exactly). The 2-qubit Bell pair is included too, since
    it is itself a GHZ state with n=2 and the same diagonal-Kraus argument applies
    verbatim -- confirmed against the committed sweep the same way.

    Returns None for the three non-diagonal channels (bit_flip, depolarizing,
    amplitude_damping), which have no simple closed form (see Part 5's writeup),
    rather than approximating them.
    """
    if row["channel"] not in PHYSICS_BASELINE_CHANNELS:
        return None
    n = int(row["qubit_count"])
    depth = int(row["depth"])
    param = float(row["param"])
    if row["channel"] == "phase_flip":
        decay_factor = predicted_phase_flip_decay_factor(param)
    else:
        decay_factor = predicted_phase_damping_decay_factor(param)
    return predicted_dephasing_ghz_fidelity(decay_factor, n * depth)
