"""Load filtered Polymarket data and build modeling features (log-odds, z-scores, splits)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

_ERR_FILTERED = (
    "Run notebook 00_data_quality.ipynb first to generate the filtered dataset."
)


def _resolve_path(filepath: str | Path) -> Path:
    p = Path(filepath)
    if not p.is_absolute():
        p = config.PROJECT_ROOT / filepath
    return p.resolve()


def load_filtered(filepath: str | Path) -> pd.DataFrame:
    """Load ``markets_filtered.csv`` and print basic diagnostics."""
    p = _resolve_path(filepath)
    if not p.is_file():
        raise FileNotFoundError(_ERR_FILTERED)

    df = pd.read_csv(p, low_memory=False)
    print(f"Loaded filtered data from: {p}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("\nColumn names:")
    for i, c in enumerate(df.columns, 1):
        print(f"  {i:3d}. {c!r}")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nFirst 3 rows:")
    with pd.option_context("display.max_columns", 12, "display.width", 120):
        print(df.head(3))
    return df


def prepare_features(
    df_filtered: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, int]]:
    """
    Clip probabilities, log-odds, z-score covariates, encode category, stratified split,
    save train/test CSVs under ``data/processed/``.
    """
    epsilon = float(cfg["epsilon"])
    min_cat = int(cfg["min_category_size"])
    seed = int(cfg["random_seed"])
    test_size = float(cfg["test_size"])

    df = df_filtered.copy()

    print("\n--- Step 1: Clip market_prob ---")
    if "market_prob" not in df.columns:
        raise KeyError("Filtered data must contain a 'market_prob' column.")
    p = df["market_prob"].astype(float).clip(epsilon, 1.0 - epsilon)
    df["market_prob"] = p
    print(f"  Clipped to [{epsilon:g}, {1 - epsilon:g}].")

    print("\n--- Step 2: log_odds ---")
    df["log_odds"] = np.log(p / (1.0 - p))
    print("  Added column log_odds = log(p / (1-p)).")

    print("\n--- Step 3: volume / liquidity / spread (median impute + z-score) ---")
    for name, col in (
        ("volume_z", "volume"),
        ("liquidity_z", "liquidity"),
        ("spread_z", "spread"),
    ):
        if col not in df.columns:
            warnings.warn(
                f"Column {col!r} not found; {name} set to 0.0.",
                stacklevel=2,
            )
            df[name] = 0.0
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        med = x.median()
        if pd.isna(med):
            med = 0.0
        x = x.fillna(med)
        mu = float(x.mean())
        sd = float(x.std(ddof=0))
        if sd == 0.0 or np.isnan(sd):
            df[name] = 0.0
            warnings.warn(
                f"Column {col!r} has zero or NaN std after imputation; {name} set to 0.0.",
                stacklevel=2,
            )
        else:
            df[name] = (x - mu) / sd
        print(f"  {name}: from {col!r} (median impute, z-score).")

    print("\n--- Step 4: category_str ---")
    if "category_clean" in df.columns:
        base = df["category_clean"]
    elif "category" in df.columns:
        base = df["category"]
    else:
        warnings.warn("No category column; using 'unknown'.", stacklevel=2)
        base = "unknown"
    df["category_str"] = (
        base.astype(str).str.strip().str.lower().replace({"nan": "unknown", "": "unknown"})
    )

    print("\n--- Step 5: enforce min_category_size, then cat_idx (alphabetical) ---")
    counts = df["category_str"].value_counts()
    keep = counts[counts >= min_cat].index
    dropped = counts[counts < min_cat]
    n_before = len(df)
    df = df.loc[df["category_str"].isin(keep)].copy()
    print(f"  min_category_size={min_cat}: kept {len(df):,} / {n_before:,} rows.")
    if len(dropped) > 0:
        print(f"  Dropped small categories: {dropped.to_dict()}")

    categories = sorted(df["category_str"].unique())
    cat_to_idx: dict[str, int] = {c: i for i, c in enumerate(categories)}
    df["cat_idx"] = df["category_str"].map(cat_to_idx)
    print(f"  {len(categories)} categories in modeling set; cat_idx 0..{len(categories) - 1}.")

    if "outcome" not in df.columns:
        if "outcome_binary" in df.columns:
            df["outcome"] = df["outcome_binary"].astype(int)
        else:
            raise KeyError("Need 'outcome' or 'outcome_binary' for modeling.")

    print("\n--- Step 6: train/test split (stratified by cat_idx) ---")
    y = df["outcome"].astype(int).values
    X_idx = np.arange(len(df))
    try:
        tr_idx, te_idx = train_test_split(
            X_idx,
            test_size=test_size,
            random_state=seed,
            stratify=df["cat_idx"].values,
        )
    except ValueError as e:
        raise ValueError(
            "Stratified split failed (some category too small?). "
            "Try increasing min_category_size or adjusting test_size."
        ) from e

    df_train = df.iloc[tr_idx].reset_index(drop=True)
    df_test = df.iloc[te_idx].reset_index(drop=True)

    print("\n--- Step 7: Summary ---")
    n_total = len(df_train) + len(df_test)
    print(f"  Total clean markets (train+test): {n_total:,}")
    tbl = (
        df.groupby("category_str", observed=True)
        .size()
        .sort_values(ascending=False)
        .reset_index(name="n_markets")
    )
    print("\n  Categories and counts:")
    print(tbl.to_string(index=False))
    yes_rate = 100.0 * float(df["outcome"].mean())
    print(f"\n  Outcome rate (% YES): {yes_rate:.2f}%")
    print(f"  Train size: {len(df_train):,}  |  Test size: {len(df_test):,}")

    print("\n--- Step 8: Save processed CSVs ---")
    out_dir = Path(config.DATA_PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "markets_train.csv"
    test_path = out_dir / "markets_test.csv"
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    print(f"  Wrote {train_path}")
    print(f"  Wrote {test_path}")

    print("\n--- Done ---")
    return df_train, df_test, categories, cat_to_idx
