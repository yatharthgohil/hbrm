"""Calibration metrics: Brier, log loss, ECE, reliability diagrams."""

from __future__ import annotations

# Use config.N_BINS_ECE, config.FIGURES_DIR for plots and tables.


def brier_score(y_true, y_prob):
    raise NotImplementedError


def expected_calibration_error(y_true, y_prob, n_bins: int | None = None):
    raise NotImplementedError
