"""Baseline and hierarchical Bayesian recalibration models (PyMC)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _interpret_logistic(intercept: float, coef: float, tol_int: float = 0.1, tol_coef: float = 0.1) -> None:
    """Print coefficient diagnostics for single-feature log-odds recalibration."""
    print(f"\n  Intercept: {intercept:.4f}")
    print(f"  Coefficient (log_odds): {coef:.4f}")
    print("\n  Interpretation:")
    if abs(coef - 1.0) < tol_coef and abs(intercept) < tol_int:
        print("    • Market is approximately well-calibrated globally")
    if coef > 1.0 + tol_coef:
        print("    • Market is underconfident (too moderate)")
    elif coef < 1.0 - tol_coef:
        print("    • Market is overconfident (too extreme)")
    if intercept > tol_int:
        print("    • Market systematically underestimates YES probability")
    elif intercept < -tol_int:
        print("    • Market systematically overestimates YES probability")


def fit_logistic_recalibration(df_train: pd.DataFrame) -> LogisticRegression:
    """
    Platt-style logistic recalibration on ``log_odds`` (single feature) → ``outcome``.
    """
    print("Fitting logistic recalibration (LogisticRegression on log_odds)...")
    X = df_train[["log_odds"]].values
    y = df_train["outcome"].astype(int).values
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    intercept = float(model.intercept_[0])
    coef = float(model.coef_[0, 0])
    _interpret_logistic(intercept, coef)
    return model


def fit_isotonic_recalibration(df_train: pd.DataFrame) -> IsotonicRegression:
    """Non-parametric isotonic calibration: ``market_prob`` → ``outcome``."""
    print("Fitting isotonic recalibration...")
    mp = df_train["market_prob"].astype(float).values
    y = df_train["outcome"].astype(int).values
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(mp, y)
    print("  Done (out_of_bounds='clip').")
    return iso


def predict_all_baselines(
    df_test: pd.DataFrame,
    lr_model: LogisticRegression,
    iso_model: IsotonicRegression,
) -> dict[str, np.ndarray]:
    """Raw ``market_prob``, logistic-on-log-odds, and isotonic calibrated probabilities."""
    log_odds = df_test[["log_odds"]].values
    mp = df_test["market_prob"].astype(float).values
    return {
        "Raw Market": np.asarray(mp, dtype=float),
        "Logistic": lr_model.predict_proba(log_odds)[:, 1].astype(float),
        "Isotonic": iso_model.predict(mp).astype(float),
    }


def build_and_sample_hbrm(
    df_train: pd.DataFrame,
    n_cats: int,
    config: dict,
):
    """
    Hierarchical Bayesian logistic calibration: per-category (α_j, β_j) with
    non-centered parameterization and global covariates (volume, liquidity, spread).

    ``config`` keys: mcmc_draws, mcmc_tune, mcmc_chains,
    mcmc_target_accept, random_seed. Optional: save_trace (default True)
    — set False for quick compile checks so ``hbrm_trace.nc`` is not overwritten.
    """
    from pathlib import Path

    import arviz as az
    import pymc as pm

    from . import config as project_config

    n_cats = int(n_cats)

    log_odds_train = df_train["log_odds"].values.astype(float)
    outcome_train = df_train["outcome"].values.astype(int)
    cat_idx_train = df_train["cat_idx"].values.astype(int)
    volume_z_train = df_train["volume_z"].values.astype(float)
    liquidity_z_train = df_train["liquidity_z"].values.astype(float)
    spread_z_train = df_train["spread_z"].values.astype(float)

    if cat_idx_train.min() < 0 or cat_idx_train.max() >= n_cats:
        raise ValueError(
            f"cat_idx must be in [0, {n_cats - 1}], got [{cat_idx_train.min()}, {cat_idx_train.max()}]"
        )

    draws = int(config["mcmc_draws"])
    tune = int(config["mcmc_tune"])
    chains = int(config["mcmc_chains"])
    target_accept = float(config["mcmc_target_accept"])
    seed = int(config["random_seed"])

    with pm.Model() as model:
        mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=1.0)
        mu_beta = pm.Normal("mu_beta", mu=1.0, sigma=0.5)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.5)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.5)

        alpha_offset = pm.Normal("alpha_offset", mu=0.0, sigma=1.0, shape=n_cats)
        beta_offset = pm.Normal("beta_offset", mu=0.0, sigma=1.0, shape=n_cats)

        alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_offset)
        beta = pm.Deterministic("beta", mu_beta + sigma_beta * beta_offset)

        gamma_vol = pm.Normal("gamma_vol", mu=0.0, sigma=0.5)
        gamma_liq = pm.Normal("gamma_liq", mu=0.0, sigma=0.5)
        gamma_spr = pm.Normal("gamma_spr", mu=0.0, sigma=0.5)

        logit_p = (
            alpha[cat_idx_train]
            + beta[cat_idx_train] * log_odds_train
            + gamma_vol * volume_z_train
            + gamma_liq * liquidity_z_train
            + gamma_spr * spread_z_train
        )

        pm.Bernoulli("y_obs", logit_p=logit_p, observed=outcome_train)

        print("Sampling HBRM with NUTS...")
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
        )

    n_div = int(idata.sample_stats["diverging"].values.sum())
    print(f"\nNumber of divergences: {n_div}")
    if n_div > 0:
        print(
            "  WARNING: Divergences detected — consider increasing mcmc_target_accept (e.g. 0.95)."
        )

    summ = az.summary(idata, round_to=4)
    col_ess = "ess_bulk" if "ess_bulk" in summ.columns else "ess_mean"
    rhat = summ["r_hat"].dropna()
    ess = summ[col_ess].dropna()
    print(f"R-hat range (all sampled vars): [{float(rhat.min()):.4f}, {float(rhat.max()):.4f}]")
    print(f"ESS range ({col_ess}): [{float(ess.min()):.1f}, {float(ess.max()):.1f}]")

    if config.get("save_trace", True):
        Path(project_config.POSTERIORS_DIR).mkdir(parents=True, exist_ok=True)
        out_path = Path(project_config.POSTERIORS_DIR) / "hbrm_trace.nc"
        az.to_netcdf(idata, str(out_path))
        print(f"Saved trace: {out_path}")
    else:
        print("save_trace=False — skipping write to hbrm_trace.nc")

    return idata


def build_hbrm_model(*args, **kwargs):
    """Alias for :func:`build_and_sample_hbrm`."""
    return build_and_sample_hbrm(*args, **kwargs)
