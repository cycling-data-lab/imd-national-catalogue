"""
d18_lso_tier1.py — Leave-Station-Out validation on Tier 1 trip-log networks.

Same logic as d17 (LSO on Velomagg) but reads the panel from the Tier 1
trip-log zips and uses the d3/d10 international IMD feature set (7 axes).
Defaults to Boston Bluebikes (497 stations, ~3M bins) which is the
smallest Tier 1 network beyond Velomagg and the cheapest test of the
spatial-generalization hypothesis.

Pass --city to evaluate other Tier 1 networks; --folds and --features-imd
are tunable.

Output (per city):
  outputs/d18_lso_<city>.json
  outputs/d18_lso_<city>_per_station.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, kendalltau

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TIER1_DIR = ROOT / "data_collection" / "tier1_trip_logs"
IMD_INTL_DIR = ROOT / "data_collection" / "imd_international"
OUT = ROOT / "experiments" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Re-use d3's panel loader (schema-aware Lyft vs Hubway) to avoid duplication
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from d3_multicity_benchmark import build_panel, FEATS_T, FEATS_IMD  # noqa: E402

LGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    min_child_samples=30, reg_lambda=0.5, random_state=42,
    n_jobs=-1, verbose=-1,
)


def precision_at_k(true_sorted_ids, pred_sorted_ids, k: int) -> float:
    return len(set(true_sorted_ids[:k]) & set(pred_sorted_ids[:k])) / k


def run(city: str, n_folds: int, seed: int = 42):
    t_start = time.time()
    rng = np.random.default_rng(seed)

    # IMD features
    imd_path = IMD_INTL_DIR / f"{city}.parquet"
    if not imd_path.exists():
        print(f"✗ {imd_path} missing"); return
    imd = pd.read_parquet(imd_path)
    imd["station_id"] = imd["station_id"].astype(str)
    print(f"IMD : {len(imd)} stations  ({imd_path.name})")

    # Trip panel
    print(f"Building trip panel from {TIER1_DIR / city}...")
    panel = build_panel(city)
    if panel.empty:
        print("✗ Empty panel"); return
    print(f"Panel : {len(panel):,} (station,hour) bins  "
          f"{panel['demande'].sum():,} trips")

    df = panel.merge(imd[["station_id"] + FEATS_IMD], on="station_id", how="left")
    df = df.dropna(subset=FEATS_IMD).copy()
    df["hour"] = df["datetime_hour"].dt.hour
    df["day_of_week"] = df["datetime_hour"].dt.dayofweek
    df["month"] = df["datetime_hour"].dt.month
    df["log_demand"] = np.log1p(df["demande"])

    stations = sorted(df["station_id"].unique())
    n = len(stations)
    print(f"After IMD merge : {len(df):,} bins  {n} stations")

    # Filter stations active >=6 months (avoid turnover noise)
    span = df.groupby("station_id")["datetime_hour"].agg(["min", "max"])
    span["months"] = (span["max"] - span["min"]).dt.days / 30
    keep = span[span["months"] >= 6].index.tolist()
    print(f"Stations with >=6 months activity: {len(keep)}/{n}")
    df = df[df["station_id"].isin(keep)].copy()
    stations = sorted(keep)
    n = len(stations)

    perm = list(stations)
    rng.shuffle(perm)
    folds = np.array_split(perm, n_folds)
    print(f"Folds: {[len(f) for f in folds]}")

    per_station = []
    fold_summary = []
    for fi, fold in enumerate(folds):
        fold = list(fold)
        train = df[~df["station_id"].isin(fold)]
        test = df[df["station_id"].isin(fold)].copy()
        print(f"\n=== Fold {fi+1}/{n_folds} : "
              f"{train['station_id'].nunique()} train, {len(fold)} holdout, "
              f"n_test={len(test):,} ===")

        def fit(features, name):
            m = lgb.LGBMRegressor(**LGB_PARAMS)
            t0 = time.time()
            m.fit(train[features].astype("float64").values,
                  train["log_demand"].values)
            ys = m.predict(test[features].astype("float64").values)
            yt = test["log_demand"].values
            yt_trips = np.expm1(yt); yp_trips = np.expm1(np.clip(ys, 0, None))
            return ys, yp_trips, time.time() - t0

        ys_no,  yp_no,  ts_no  = fit(FEATS_T,                "Gminus")
        ys_imd, yp_imd, ts_imd = fit(FEATS_T + FEATS_IMD,    "G")

        test["pred_no_imd"] = yp_no
        test["pred_imd"] = yp_imd

        agg = (test.groupby("station_id")
                  .agg(n_bins=("demande", "size"),
                       true_mean=("demande", "mean"),
                       pred_no_imd_mean=("pred_no_imd", "mean"),
                       pred_imd_mean=("pred_imd", "mean"),
                       true_total=("demande", "sum"),
                       pred_no_imd_total=("pred_no_imd", "sum"),
                       pred_imd_total=("pred_imd", "sum"))
                  .reset_index())
        agg["fold"] = fi + 1
        per_station.append(agg)

        # Hourly R^2 on held-out out-of-station observations
        yt = test["log_demand"].values
        r2_no  = r2_score(np.expm1(yt), yp_no)
        r2_imd = r2_score(np.expm1(yt), yp_imd)
        print(f"  Hourly R²(G-) = {r2_no:+.4f}    R²(G) = {r2_imd:+.4f}    "
              f"ΔR² = {r2_imd-r2_no:+.4f}    (fit {ts_no:.1f}s + {ts_imd:.1f}s)")
        fold_summary.append(dict(fold=fi+1, n_holdout=len(fold),
                                  n_test=int(len(test)),
                                  r2_hourly_no_imd=r2_no,
                                  r2_hourly_imd=r2_imd))

    ps = pd.concat(per_station, ignore_index=True)
    ps.to_csv(OUT / f"d18_lso_{city}_per_station.csv", index=False)

    # Mean-rate ranking metrics across held-out stations
    ordered_obs   = ps.sort_values("true_mean",        ascending=False)["station_id"].tolist()
    ordered_pred  = ps.sort_values("pred_imd_mean",    ascending=False)["station_id"].tolist()
    ordered_pred0 = ps.sort_values("pred_no_imd_mean", ascending=False)["station_id"].tolist()
    rho_imd, _   = spearmanr(ps["true_mean"], ps["pred_imd_mean"])
    rho_no,  _   = spearmanr(ps["true_mean"], ps["pred_no_imd_mean"])
    tau_imd, _   = kendalltau(ps["true_mean"], ps["pred_imd_mean"])
    tau_no,  _   = kendalltau(ps["true_mean"], ps["pred_no_imd_mean"])
    metrics = {
        "city": city,
        "n_stations_evaluated": int(len(ps)),
        "n_folds": n_folds,
        "wall_time_s": round(time.time() - t_start, 1),
        "global_spearman_rho_imd":     float(rho_imd),
        "global_spearman_rho_no_imd":  float(rho_no),
        "global_kendall_tau_imd":      float(tau_imd),
        "global_kendall_tau_no_imd":   float(tau_no),
        "precision_at_k": {},
        "per_fold": fold_summary,
    }
    for K in (5, 10, 20, 50):
        if K > len(ps): continue
        p_imd = precision_at_k(ordered_obs, ordered_pred,  K)
        p_no  = precision_at_k(ordered_obs, ordered_pred0, K)
        p_null = K / len(ps)
        metrics["precision_at_k"][f"K{K}"] = dict(
            precision_imd=float(p_imd),
            precision_no_imd=float(p_no),
            precision_random_null=float(p_null),
        )
        print(f"\nPrecision@{K:>3d}:  IMD = {p_imd:.3f}  |  no-IMD = {p_no:.3f}"
              f"  |  random = {p_null:.3f}")
    print(f"\nSpearman ρ : IMD = {rho_imd:+.3f}  |  no-IMD = {rho_no:+.3f}")
    print(f"Kendall  τ : IMD = {tau_imd:+.3f}  |  no-IMD = {tau_no:+.3f}")

    with open(OUT / f"d18_lso_{city}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Saved {OUT / f'd18_lso_{city}.json'}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", default="boston_bluebikes")
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()
    run(args.city, args.folds)


if __name__ == "__main__":
    main()
