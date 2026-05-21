"""
d11_bootstrap_ci.py — Block-bootstrap 95% CI for R^2 (trip-count scale)
on every (city, model) pair for which we have stored test-set predictions.

Reads:
  outputs/d3_<city>_predictions.parquet   (Tier 1 cities, written by d3)
  outputs/d10_<city>_predictions.parquet  (Tier 2 cities, written by d10)

Method:
  Weekly block-bootstrap on the test set. We partition the test set into
  168h calendar blocks (Mon 00h to Sun 23h floor), sample blocks with
  replacement to recover a test-set-sized sample, and recompute R^2 on
  the trip-count scale (after exp transform and clipping to >=0). 500
  bootstrap replicates per (city, model).

Output:
  outputs/d11_bootstrap_ci.csv
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

OUT = Path(__file__).resolve().parents[0] / "outputs"
B = 500
SEED = 42

def r2_trips(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    yt = np.expm1(y_true_log)
    yp = np.expm1(np.clip(y_pred_log, 0, None))
    return float(r2_score(yt, yp))

def weekly_blocks(times: pd.Series) -> np.ndarray:
    """Block id = ISO year*100 + ISO week, so blocks respect calendar weeks."""
    iso = times.dt.isocalendar()
    return (iso["year"].astype(int) * 100 + iso["week"].astype(int)).values

def daily_blocks(times: pd.Series) -> np.ndarray:
    """Block id = ordinal date — fallback when the test set spans < 2 weeks."""
    return times.dt.date.astype(str).values

def bootstrap_one(df: pd.DataFrame, col_pred: str, rng: np.random.Generator) -> tuple[float, float, float]:
    """Return (point estimate, ci_lo, ci_hi) for R^2_trips."""
    blocks = weekly_blocks(pd.to_datetime(df["datetime_hour"]))
    unique_blocks, block_starts = np.unique(blocks, return_index=True)
    # Pre-index rows per block
    by_block = {b: np.where(blocks == b)[0] for b in unique_blocks}

    point = r2_trips(df["y_true_log"].values, df[col_pred].values)
    n_blocks = len(unique_blocks)
    samples = np.empty(B, dtype=float)
    for b in range(B):
        sel = rng.choice(unique_blocks, size=n_blocks, replace=True)
        idx = np.concatenate([by_block[s] for s in sel])
        samples[b] = r2_trips(df["y_true_log"].values[idx],
                              df[col_pred].values[idx])
    return point, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))

def main():
    rng = np.random.default_rng(SEED)
    pred_files = sorted(list(OUT.glob("d1_*_predictions.parquet")) +
                        list(OUT.glob("d3_*_predictions.parquet")) +
                        list(OUT.glob("d10_*_predictions.parquet")) +
                        list(OUT.glob("d14_*_predictions.parquet")))
    rows = []
    for pf in pred_files:
        if pf.name.startswith("d1_"):
            tier, city = 1, pf.stem.replace("d1_", "").replace("_predictions", "")
        elif pf.name.startswith("d3_"):
            tier, city = 1, pf.stem.replace("d3_", "").replace("_predictions", "")
        elif pf.name.startswith("d14_"):
            tier, city = 1, pf.stem.replace("d14_", "").replace("_predictions", "")
        else:
            tier, city = 2, pf.stem.replace("d10_", "").replace("_predictions", "")
        df = pd.read_parquet(pf)
        df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
        n_weeks = pd.Series(weekly_blocks(df["datetime_hour"])).nunique()
        if n_weeks >= 2:
            block_kind = "week"
            blocks = weekly_blocks(df["datetime_hour"])
        else:
            n_days = pd.Series(daily_blocks(df["datetime_hour"])).nunique()
            if n_days < 2:
                print(f"  ✗ {city} (T{tier}): test set spans only {n_days} day"
                      f" — bootstrap not informative, skipping")
                continue
            block_kind = "day"
            blocks = daily_blocks(df["datetime_hour"])
        n_blocks = pd.Series(blocks).nunique()
        print(f"  {city} (T{tier}, {n_blocks} {block_kind}-blocks, n={len(df):,}) ...")
        unique_blocks = np.unique(blocks)
        by_block = {b: np.where(blocks == b)[0] for b in unique_blocks}
        # Point estimates
        p_no = r2_trips(df["y_true_log"].values, df["y_pred_no_imd"].values)
        p_im = r2_trips(df["y_true_log"].values, df["y_pred_imd"].values)
        # Paired block-bootstrap of CI (same resampling for both models)
        no_s = np.empty(B); im_s = np.empty(B); de_s = np.empty(B)
        for b in range(B):
            sel = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
            idx = np.concatenate([by_block[s] for s in sel])
            r_no = r2_trips(df["y_true_log"].values[idx], df["y_pred_no_imd"].values[idx])
            r_im = r2_trips(df["y_true_log"].values[idx], df["y_pred_imd"].values[idx])
            no_s[b], im_s[b], de_s[b] = r_no, r_im, r_im - r_no
        lo_no, hi_no = np.quantile(no_s, [0.025, 0.975])
        lo_im, hi_im = np.quantile(im_s, [0.025, 0.975])
        delta_lo, delta_hi = np.quantile(de_s, [0.025, 0.975])
        rows.append({
            "city": city, "tier": tier, "n_test": len(df),
            "block_kind": block_kind, "n_blocks": int(n_blocks),
            "r2_no_imd": p_no, "r2_no_imd_lo": float(lo_no), "r2_no_imd_hi": float(hi_no),
            "r2_imd":   p_im, "r2_imd_lo":   float(lo_im), "r2_imd_hi":   float(hi_im),
            "delta_r2": p_im - p_no, "delta_r2_lo": float(delta_lo), "delta_r2_hi": float(delta_hi),
        })
    if not rows:
        print("No prediction files found."); return
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "d11_bootstrap_ci.csv", index=False)
    print("\n=== BLOCK-BOOTSTRAP 95% CI (R^2 on trip-count scale, weekly blocks, B=500) ===\n")
    fmt = "{:<28s}  T{:d}  n={:>8d}  R2(G-)={:+.3f} [{:+.3f},{:+.3f}]  R2(G)={:+.3f} [{:+.3f},{:+.3f}]  dR2={:+.3f} [{:+.3f},{:+.3f}]"
    for _, r in out.iterrows():
        print(fmt.format(r.city, r.tier, r.n_test,
                         r.r2_no_imd, r.r2_no_imd_lo, r.r2_no_imd_hi,
                         r.r2_imd,   r.r2_imd_lo,   r.r2_imd_hi,
                         r.delta_r2, r.delta_r2_lo, r.delta_r2_hi))
    print(f"\n✓ Wrote {OUT/'d11_bootstrap_ci.csv'}")

if __name__ == "__main__":
    main()
