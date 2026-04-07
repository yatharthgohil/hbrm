"""Data quality filtering: remove unsuitable markets before any analysis."""

from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config

# --- column name aliases ----------------------------------------------------

_MARKET_PROB_COLS = ("market_prob", "lastTradePrice")
_CREATED_COLS = ("created_at", "createdAt", "startDate", "startDateIso")
_END_COLS = ("endDate", "endDateIso", "resolutionDate", "umaEndDate", "umaEndDateIso")
_VOLUME_COLS = ("volume", "volumeNum")
_LIQUIDITY_COLS = ("liquidityNum", "liquidity")
_OUTCOME_FALLBACK_COLS = ("outcome", "winnerOutcome", "result", "resolution", "winner")


def _find_first_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_raw(data_raw_dir: str | Path) -> pd.DataFrame:
    """Load the single CSV (or CSV.GZ) in ``data_raw_dir`` and print diagnostics."""
    root = Path(data_raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    candidates = sorted(root.glob("*.csv")) + sorted(root.glob("*.csv.gz"))
    if not candidates:
        raise FileNotFoundError(f"No .csv or .csv.gz files found in {root}")

    path = candidates[0]
    if len(candidates) > 1:
        warnings.warn(
            f"Multiple CSV files in {root}; using first by name: {path.name}",
            stacklevel=2,
        )

    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded: {path}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("\nColumn names and dtypes:")
    for col in df.columns:
        print(f"  {col!r}: {df[col].dtype}")
    print("\nFirst 3 rows:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(3))
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"\nApprox. memory usage (deep): {mem_mb:.1f} MB")
    return df


def parse_market_prob(p: Any) -> float:
    """
    Parse implied YES probability into [0, 1] or NaN.
    Handles float/int, strings, list/tuple / stringified lists (first element = YES).
    Values > 1 are treated as percentages and divided by 100.
    """
    if p is None:
        return np.nan
    if isinstance(p, (float, np.floating)) and np.isnan(p):
        return np.nan
    if isinstance(p, (int, float, np.integer, np.floating)):
        v = float(p)
        if np.isnan(v):
            return np.nan
        if v > 1.0:
            v /= 100.0
        if v < 0.0 or v > 1.0:
            return np.nan
        return float(v)

    s = str(p).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return np.nan

    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError, TypeError):
            return np.nan
        if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
            return parse_market_prob(parsed[0])
        return np.nan

    try:
        v = float(s)
    except ValueError:
        return np.nan
    return parse_market_prob(v)


def _outcome_from_explicit(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        if int(value) == 1:
            return 1.0
        if int(value) == 0:
            return 0.0
        return np.nan
    v = str(value).strip().lower()
    if v in ("yes", "true", "1"):
        return 1.0
    if v in ("no", "false", "0"):
        return 0.0
    return np.nan


def parse_outcome(row: pd.Series) -> float:
    """
    Map outcome to 1 (YES) or 0 (NO), or NaN if ambiguous / non-binary.
    Tries explicit columns then Polymarket ``outcomes`` + ``outcomePrices`` (Yes/No only).
    """
    for col in _OUTCOME_FALLBACK_COLS:
        if col in row.index and pd.notna(row[col]):
            y = _outcome_from_explicit(row[col])
            if not np.isnan(y):
                return y

    if "outcomes" not in row.index or "outcomePrices" not in row.index:
        return np.nan
    if pd.isna(row["outcomes"]) or pd.isna(row["outcomePrices"]):
        return np.nan

    try:
        oc = ast.literal_eval(str(row["outcomes"]))
        op = ast.literal_eval(str(row["outcomePrices"]))
    except (ValueError, SyntaxError, TypeError):
        return np.nan

    if not isinstance(oc, (list, tuple)) or not isinstance(op, (list, tuple)):
        return np.nan
    if len(oc) != 2 or len(op) != 2:
        return np.nan

    labels = [str(x).strip().lower() for x in oc]
    if set(labels) != {"yes", "no"}:
        return np.nan

    try:
        prices = [float(x) for x in op]
    except (ValueError, TypeError):
        return np.nan

    mx = max(prices)
    if mx < 0.99:
        return np.nan
    win_idx = max(range(len(prices)), key=lambda i: prices[i])
    winner = labels[win_idx]
    if winner == "yes":
        return 1.0
    if winner == "no":
        return 0.0
    return np.nan


def resolved_market_mask(df: pd.DataFrame) -> pd.Series:
    """
    True if market is treated as resolved (same rule as Filter 1).
    Uses ``resolved`` if present; otherwise ``closed``; otherwise warns and returns all True.
    """
    if "resolved" in df.columns:
        r = df["resolved"]
        if r.dtype == object:
            rs = r.astype(str).str.lower()
            return r.eq(True) | r.eq(1) | rs.isin(("true", "1"))
        return r.eq(True) | r.eq(1)
    if "closed" in df.columns:
        c = df["closed"]
        if c.dtype == object:
            cs = c.astype(str).str.lower()
            return c.eq(True) | c.eq(1) | cs.isin(("true", "1"))
        return c.eq(True) | c.eq(1)
    warnings.warn(
        "No 'resolved' or 'closed' column; cannot identify resolved markets. "
        "Filter 1 will be skipped.",
        stacklevel=2,
    )
    return pd.Series(True, index=df.index)


def _market_prob_series(df: pd.DataFrame) -> pd.Series:
    """
    Implied YES probability for filtering / modeling.

    Prefer bid–ask mid when both sides exist (``lastTradePrice`` is often 0/1 on
    resolved markets). Then ``lastTradePrice``, then first element of ``outcomePrices``.
    """
    if "market_prob" in df.columns:
        return df["market_prob"].map(parse_market_prob)

    out = pd.Series(np.nan, index=df.index, dtype=float)
    if "bestBid" in df.columns and "bestAsk" in df.columns:
        bb = pd.to_numeric(df["bestBid"], errors="coerce")
        ba = pd.to_numeric(df["bestAsk"], errors="coerce")
        mid = (bb + ba) / 2.0
        ok = bb.notna() & ba.notna()
        out.loc[ok] = mid.loc[ok].map(parse_market_prob)

    if "lastTradePrice" in df.columns:
        lt = pd.to_numeric(df["lastTradePrice"], errors="coerce")
        fill = out.isna()
        out.loc[fill] = lt.loc[fill].map(parse_market_prob)

    if "outcomePrices" in df.columns:
        fill = out.isna()
        if bool(fill.any()):
            out.loc[fill] = df.loc[fill, "outcomePrices"].map(parse_market_prob)

    if out.isna().all():
        warnings.warn(
            "Could not derive market_prob (no market_prob, bid/ask, lastTradePrice, "
            "or outcomePrices); column will be all NaN.",
            stacklevel=2,
        )
    return out


def _parse_datetimes(df: pd.DataFrame, col: str | None) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    return pd.to_datetime(df[col], errors="coerce", utc=True)


def parse_and_enrich(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same parsing as ``apply_quality_filters`` (no row removal).
    Adds ``market_prob``, ``outcome_binary``, ``category_clean``, numeric volume/liquidity,
    ``created_at``, ``end_date`` for inspection / EDA.
    """
    df = df_raw.copy()
    df["market_prob"] = _market_prob_series(df_raw)
    df["outcome_binary"] = df.apply(parse_outcome, axis=1)
    if "category" in df.columns:
        cat = df["category"].fillna("unknown")
        df["category_clean"] = (
            cat.astype(str).str.strip().str.lower().replace({"nan": "unknown", "": "unknown"})
        )
    else:
        df["category_clean"] = "unknown"
        warnings.warn("No 'category' column; using 'unknown' for all rows.", stacklevel=2)

    vol_col = _find_first_column(df, _VOLUME_COLS)
    if vol_col:
        df["volume"] = pd.to_numeric(df[vol_col], errors="coerce")
    else:
        df["volume"] = np.nan

    liq_col = _find_first_column(df, _LIQUIDITY_COLS)
    if liq_col:
        df["liquidity"] = pd.to_numeric(df[liq_col], errors="coerce")
    else:
        df["liquidity"] = np.nan

    c_col = _find_first_column(df, _CREATED_COLS)
    e_col = _find_first_column(df, _END_COLS)
    df["created_at"] = _parse_datetimes(df, c_col)
    df["end_date"] = _parse_datetimes(df, e_col)
    return df


def apply_quality_filters(
    df_raw: pd.DataFrame,
    config_dict: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse columns, apply filters in fixed order, print report, return clean df and log."""
    df = parse_and_enrich(df_raw)
    filter_log: dict[str, Any] = {}

    n_raw = len(df)
    filter_log["n_raw"] = n_raw

    # Filter 1 — resolved
    if "resolved" in df_raw.columns or "closed" in df_raw.columns:
        m_res = resolved_market_mask(df_raw)
        df = df.loc[m_res].copy()
        filter_log["n_after_resolved"] = len(df)
        filter_log["removed_after_resolved"] = n_raw - len(df)
    else:
        warnings.warn(
            "Filter 1 (resolved): no 'resolved' or 'closed' column; skipping.",
            stacklevel=2,
        )
        filter_log["filter1_note"] = "skipped - column missing"
        filter_log["n_after_resolved"] = len(df)
        filter_log["removed_after_resolved"] = 0

    n_before = len(df)
    # Filter 2 — binary outcome
    m_bin = df["outcome_binary"].isin([0.0, 1.0])
    df = df.loc[m_bin].copy()
    filter_log["n_after_binary"] = len(df)
    filter_log["removed_after_binary"] = n_before - len(df)

    n_before = len(df)
    # Filter 3 — volume
    vol_note = ""
    if df["volume"].notna().any() and not df["volume"].isna().all():
        thr = float(config_dict["MIN_VOLUME_USD"])
        df = df.loc[df["volume"] >= thr].copy()
        filter_log["n_after_volume"] = len(df)
        filter_log["removed_after_volume"] = n_before - len(df)
    else:
        vol_note = "skipped - column missing"
        warnings.warn(
            "Filter 3 (volume): no usable numeric volume; skipping.",
            stacklevel=2,
        )
        filter_log["n_after_volume"] = len(df)
        filter_log["removed_after_volume"] = 0
        filter_log["filter3_note"] = vol_note

    n_before = len(df)
    # Filter 4 — stuck prices
    mp = df["market_prob"]
    df = df.loc[(mp > config_dict["STUCK_PROB_LOW"]) & (mp < config_dict["STUCK_PROB_HIGH"])].copy()
    filter_log["n_after_stuck"] = len(df)
    filter_log["removed_after_stuck"] = n_before - len(df)

    n_before = len(df)
    # Filter 5 — lifetime
    life_note = ""
    if df["created_at"].notna().any() and df["end_date"].notna().any():
        lifetime_days = (df["end_date"] - df["created_at"]).dt.days
        df = df.loc[lifetime_days >= int(config_dict["MIN_LIFETIME_DAYS"])].copy()
        filter_log["n_after_lifetime"] = len(df)
        filter_log["removed_after_lifetime"] = n_before - len(df)
    else:
        life_note = "skipped - timestamp columns missing"
        warnings.warn(
            "Filter 5 (lifetime): missing parsed created_at/end_date; skipping.",
            stacklevel=2,
        )
        filter_log["n_after_lifetime"] = len(df)
        filter_log["removed_after_lifetime"] = 0
        filter_log["filter5_note"] = life_note

    n_before = len(df)
    # Filter 6 — liquidity
    min_liq = float(config_dict["MIN_LIQUIDITY_USD"])
    if min_liq > 0 and df["liquidity"].notna().any():
        df = df.loc[df["liquidity"] >= min_liq].copy()
        filter_log["n_after_liquidity"] = len(df)
        filter_log["removed_after_liquidity"] = n_before - len(df)
    else:
        filter_log["n_after_liquidity"] = len(df)
        filter_log["removed_after_liquidity"] = 0
        if min_liq > 0:
            filter_log["filter6_note"] = "skipped - liquidity column missing or all NaN"

    n_before = len(df)
    # Filter 7 — category size
    cat_counts = df["category_clean"].value_counts()
    keep_cats = cat_counts[cat_counts >= int(config_dict["MIN_CATEGORY_SIZE"])].index
    removed_cats = cat_counts[cat_counts < int(config_dict["MIN_CATEGORY_SIZE"])].sort_values(
        ascending=False
    )
    df = df.loc[df["category_clean"].isin(keep_cats)].copy()
    filter_log["n_after_category"] = len(df)
    filter_log["removed_after_category"] = n_before - len(df)
    filter_log["categories_removed"] = removed_cats.to_dict()
    filter_log["categories_retained"] = df["category_clean"].value_counts().to_dict()

    df["category"] = df["category_clean"]
    df["outcome"] = df["outcome_binary"].astype(int)

    _print_filter_report(filter_log)
    return df, filter_log


def print_filter_table(filter_log: dict[str, Any]) -> None:
    """Print the filtering summary table (same output as after ``apply_quality_filters``)."""
    _print_filter_report(filter_log)


def _print_filter_report(filter_log: dict[str, Any]) -> None:
    n_r = filter_log["n_raw"]
    n1 = filter_log["n_after_resolved"]
    n2 = filter_log["n_after_binary"]
    n3 = filter_log["n_after_volume"]
    n4 = filter_log["n_after_stuck"]
    n5 = filter_log["n_after_lifetime"]
    n6 = filter_log["n_after_liquidity"]
    n7 = filter_log["n_after_category"]

    def fmt(n: int) -> str:
        return f"{n:,}"

    def rem(a: int, b: int) -> str:
        d = a - b
        if d == 0:
            return "-"
        return f"{d:,}"

    rows = [
        ("Raw dataset", n_r, "-", ""),
        ("Resolved markets only", n1, rem(n_r, n1), filter_log.get("filter1_note", "")),
        ("Binary outcome only", n2, rem(n1, n2), ""),
        ("Volume >= $1,000 USD", n3, rem(n2, n3), filter_log.get("filter3_note", "")),
        ("Stuck prices removed (< 0.02 or > 0.98)", n4, rem(n3, n4), ""),
        ("Market lifetime >= 7 days", n5, rem(n4, n5), filter_log.get("filter5_note", "")),
        ("Liquidity filter", n6, rem(n5, n6), filter_log.get("filter6_note", "")),
        ("Category size >= 20", n7, rem(n6, n7), ""),
        ("FINAL CLEAN DATASET", n7, rem(n_r, n7), ""),
    ]

    w1, w2, w3 = 43, 10, 10
    bar = "+" + "-" * w1 + "+" + "-" * w2 + "+" + "-" * w3 + "+"
    sep = "|" + "-" * w1 + "+" + "-" * w2 + "+" + "-" * w3 + "|"
    print()
    print(bar)
    print(f"| {'Filter':<{w1-2}} | {'Count':^{w2-2}} | {'Removed':^{w3-2}} |")
    print(sep)
    for label, cnt, removed, note in rows:
        note_s = f" ({note})" if note else ""
        lbl = (label + note_s)[: w1 - 2]
        print(f"| {lbl:<{w1-2}} | {fmt(cnt):^{w2-2}} | {removed:^{w3-2}} |")
    print(bar)

    pct = 100.0 * n7 / n_r if n_r else 0.0
    print(f"\nRetention rate: {pct:.1f}% of raw markets kept")

    retained = filter_log.get("categories_retained") or {}
    print("\nCategories retained (count):")
    for k, v in sorted(retained.items(), key=lambda x: -x[1]):
        print(f"  {k!r}: {v:,}")

    removed = filter_log.get("categories_removed") or {}
    print("\nCategories REMOVED (too few markets):")
    if not removed:
        print("  (none)")
    else:
        for k, v in sorted(removed.items(), key=lambda x: -x[1]):
            print(f"  {k!r}: {v:,}")


def save_quality_report(
    filter_log: dict[str, Any],
    df_filtered: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Write filter log, category counts, and filtered CSV; print confirmations."""
    out = Path(output_dir)
    filt_dir = Path(config.DATA_FILTERED_DIR)
    out.mkdir(parents=True, exist_ok=True)
    filt_dir.mkdir(parents=True, exist_ok=True)

    log_rows = []
    for k, v in filter_log.items():
        if k in ("categories_removed", "categories_retained"):
            log_rows.append({"metric": k, "value": json.dumps(v)})
        else:
            log_rows.append({"metric": k, "value": v})
    pd.DataFrame(log_rows).to_csv(out / "filter_log.csv", index=False)
    print(f"Wrote: {out / 'filter_log.csv'}")

    counts = df_filtered["category_clean"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    counts.to_csv(out / "category_counts.csv", index=False)
    print(f"Wrote: {out / 'category_counts.csv'}")

    filtered_path = filt_dir / "markets_filtered.csv"
    df_filtered.to_csv(filtered_path, index=False)
    print(f"Wrote: {filtered_path}")


def filter_raw_to_filtered(
    raw_path: Path | str,
    output_path: Path | str | None = None,
) -> None:
    """Backward-compatible stub: use ``load_raw`` + ``apply_quality_filters`` in notebooks."""
    raise NotImplementedError("Use load_raw, apply_quality_filters, and save_quality_report.")
