"""Calibration metrics: Brier, log loss, ECE, reliability diagrams."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "brier_score",
    "log_loss_score",
    "expected_calibration_error",
    "compute_all_metrics",
    "per_category_metrics",
]


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Mean squared error between outcomes and probabilities. Lower is better; perfect = 0."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    return float(np.mean((y - p) ** 2))


def log_loss_score(y_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-7) -> float:
    """Binary log loss after clipping probabilities. Lower is better."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def expected_calibration_error(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Equal-width bins on [0, 1]. ECE = sum_b w_b * |acc_b - conf_b|.

    Returns
    -------
    ece : float
    bin_centers, bin_accs, bin_confs, bin_weights : ndarray
        Per-bin accuracy, mean predicted prob (confidence), and sample fraction.
        Empty bins have weight 0 and acc/conf set to NaN.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(p_pred, dtype=float).ravel()
    n = len(y)
    if n == 0:
        raise ValueError("y_true and p_pred must be non-empty.")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_accs = np.full(n_bins, np.nan)
    bin_confs = np.full(n_bins, np.nan)
    bin_weights = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n_b = int(mask.sum())
        bin_weights[i] = n_b / n if n else 0.0
        if n_b > 0:
            bin_accs[i] = float(y[mask].mean())
            bin_confs[i] = float(p[mask].mean())

    # ECE: only bins with data
    valid = bin_weights > 0
    ece = float(
        np.sum(bin_weights[valid] * np.abs(bin_accs[valid] - bin_confs[valid]))
    )
    return ece, bin_centers, bin_accs, bin_confs, bin_weights


_MODEL_ORDER = ("Raw Market", "Logistic", "Isotonic", "HBRM (Bayes)")


def compute_all_metrics(
    y_true: np.ndarray,
    predictions_dict: dict,
) -> pd.DataFrame:
    """Metrics for the four main models; sorted by ECE ascending."""
    y = np.asarray(y_true, dtype=float).ravel()
    clim_p = float(y.mean())
    clim_brier = brier_score(y, np.full_like(y, clim_p, dtype=float))

    rows = []
    for name in _MODEL_ORDER:
        if name not in predictions_dict:
            raise KeyError(f"Missing predictions for '{name}'")
        p = np.asarray(predictions_dict[name], dtype=float).ravel()
        if len(p) != len(y):
            raise ValueError(f"Length mismatch for {name}: {len(p)} vs {len(y)}")
        b = brier_score(y, p)
        ll = log_loss_score(y, p)
        ece, _, _, _, _ = expected_calibration_error(y, p, n_bins=10)
        if clim_brier <= 0 or not np.isfinite(clim_brier):
            skill = np.nan
        else:
            skill = 1.0 - (b / clim_brier)
        rows.append(
            {
                "Model": name,
                "Brier Score": b,
                "Log Loss": ll,
                "ECE": ece,
                "Brier Skill Score": skill,
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("ECE", ascending=True).reset_index(drop=True)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df


def per_category_metrics(
    y_true: np.ndarray,
    predictions_dict: dict,
    cat_labels: np.ndarray,
    min_count: int = 20,
) -> pd.DataFrame:
    """ECE and Brier for Raw vs HBRM per category; improvement_pct from ECE."""
    y = np.asarray(y_true, dtype=float).ravel()
    cat = np.asarray(cat_labels).astype(str).ravel()
    if len(cat) != len(y):
        raise ValueError("cat_labels must match length of y_true.")

    if "Raw Market" not in predictions_dict or "HBRM (Bayes)" not in predictions_dict:
        raise KeyError("predictions_dict must contain 'Raw Market' and 'HBRM (Bayes)'")

    p_raw = np.asarray(predictions_dict["Raw Market"], dtype=float).ravel()
    p_hbrm = np.asarray(predictions_dict["HBRM (Bayes)"], dtype=float).ravel()

    out_rows = []
    for c in np.unique(cat):
        m = cat == c
        n = int(m.sum())
        if n < min_count:
            continue
        y_c = y[m]
        ece_raw, _, _, _, _ = expected_calibration_error(y_c, p_raw[m], n_bins=10)
        ece_h, _, _, _, _ = expected_calibration_error(y_c, p_hbrm[m], n_bins=10)
        br_raw = brier_score(y_c, p_raw[m])
        br_h = brier_score(y_c, p_hbrm[m])
        if ece_raw == 0 or not np.isfinite(ece_raw):
            imp = np.nan
        else:
            imp = (ece_raw - ece_h) / ece_raw * 100.0
        out_rows.append(
            {
                "category": c,
                "n": n,
                "raw_ece": ece_raw,
                "hbrm_ece": ece_h,
                "raw_brier": br_raw,
                "hbrm_brier": br_h,
                "improvement_pct": imp,
            }
        )
    return pd.DataFrame(out_rows)
