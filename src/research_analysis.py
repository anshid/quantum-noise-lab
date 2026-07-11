"""Analysis helpers for the Part 5 mini research question.

GHZ-state fidelity under independent per-qubit noise decays roughly
geometrically with qubit count: fidelity(n) ~ A * r**n. fit_geometric_decay_rate
extracts the per-qubit survival ratio r via a log-linear (least-squares) fit,
turning the qualitative "channel X decays faster" observation from Part 3's
sweep into a single comparable number per channel.

Two of the five channels have a clean closed form, because their Kraus operators
are diagonal in the computational basis: phase-flip (K0=sqrt(1-p) I, K1=sqrt(p) Z)
and phase damping (K0=diag(1, sqrt(1-lam)), K1=diag(0, sqrt(lam))). Both leave
GHZ populations exactly invariant (a diagonal Kraus operator fixes |0><0| and
|1><1|) while shrinking the coherence term |0...0><1...1| by a per-qubit factor
-- (1-2p) for phase-flip, sqrt(1-lam) for phase damping -- giving
fidelity -> (1 + factor**n) / 2 for both. Since 0 <= factor <= 1, GHZ fidelity
under either channel can never drop below 1/2, no matter how large n gets:
these two channels only ever damp coherence, never populations. This is why
Part 3's sweep (n up to 6) showed phase_damping merely as "the slowest-decaying
channel" -- sqrt(1-lam) is close to 1 for small lam, so its approach to the 1/2
floor is barely visible in that window, easy to mistake for ordinary
decay-to-zero. The other three channels (bit-flip, depolarizing, amplitude
damping) all have off-diagonal Kraus contributions that move population out of
the GHZ's two computational-basis sectors entirely, and genuinely drive
fidelity to 0 -- bit-flip does so fastest (see docs/mini_research_question.md).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def fit_geometric_decay_rate(qubit_counts: Sequence[int], fidelities: Sequence[float]) -> tuple[float, float]:
    """Fit fidelity(n) ~ A * r**n via a log-linear least-squares fit.

    Returns (r, r_squared): r is the per-qubit survival ratio (exp of the fitted
    slope), r_squared is the coefficient of determination of the fit in log-space.
    """
    n = np.asarray(qubit_counts, dtype=float)
    log_fidelity = np.log(np.asarray(fidelities, dtype=float))

    slope, intercept = np.polyfit(n, log_fidelity, deg=1)
    r = float(np.exp(slope))

    predicted = slope * n + intercept
    residuals = log_fidelity - predicted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_fidelity - np.mean(log_fidelity)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return r, float(r_squared)


def predicted_phase_flip_decay_factor(p: float) -> float:
    """Closed-form per-qubit coherence-decay factor for phase-flip noise: 1 - 2p."""
    return 1.0 - 2.0 * p


def predicted_phase_damping_decay_factor(lam: float) -> float:
    """Closed-form per-qubit coherence-decay factor for phase damping: sqrt(1 - lam)."""
    return float(np.sqrt(1.0 - lam))


def predicted_dephasing_ghz_fidelity(decay_factor: float, n: int) -> float:
    """GHZ fidelity under n independent applications of a diagonal (dephasing) channel.

    Applies to any channel whose Kraus operators are diagonal in the computational
    basis (phase-flip, phase damping): fidelity = (1 + decay_factor**n) / 2, which
    asymptotes to exactly 1/2 as n grows, since such channels never touch GHZ
    populations -- only the coherence term, which shrinks by decay_factor per qubit.
    """
    return (1.0 + decay_factor**n) / 2.0
