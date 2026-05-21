"""
d17_lso_velomagg.py — Leave-Station-Out cross-validation of the IMD-augmented
demand model on Vélomagg Montpellier.

Setting and motivation
----------------------
The temporal hold-out used in d1 / d3 / d10 / d14 / d16 keeps the same set
of stations in both train and test, so it measures temporal generalisation
only.  The headline operational use of the IMD-4 — ranking candidate
locations for a new station — requires a different kind of generalisation:
predicting demand for stations the model has never seen.

LSO procedure (5 random folds)
------------------------------
For each fold f of the n stations:
  - Train G (IMD-augmented) and G^- (temporal+weather only, no network-state
    because it would leak network-wide demand including the held-out
    stations) on the (n - |f|) stations not in f, over the full panel
  - Predict the full hourly demand series for the stations in f
  - Aggregate to annual total demand per held-out station

Out-of-station metrics
----------------------
Across the 5 fold runs, every station appears exactly once as held-out.
We report:
  - Spearman rho and Kendall tau between predicted-annual and observed-annual
  - Precision@K for K in {5, 10, 20}: of the top-K stations selected by the
    model, what fraction belongs to the realised top-K?
  - R^2 on the held-out hourly observations (for completeness)
G^- serves as the null baseline: with no station-specific spatial information
its predicted ranking should be near-random.

Output:
  outputs/d17_lso_velomagg.json
  outputs/d17_lso_velomagg_per_station.csv
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, kendalltau

warnings.filterwarnings("ignore")

DATA_ROOT = Path(
    os.environ.get(
        "BIKESHARE_DATA_ROOT",
        str(Path(__file__).resolve().parents[3] / "bikeshare-data-explorer" / "data"),
    )
)
OUT_DIR = Path(__file__).resolve().parents[0] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_TEMPORAL = [
    "hour", "day_of_week", "month", "season",
    "temperature", "humidity", "precipitation", "wind_speed",
    "cloud_cover", "is_raining", "is_heavy_rain", "feels_like",
    "bad_weather_score",
]
# Note: network-state features are EXCLUDED for LSO — including network_volume
# computed across all stations would leak demand from the held-out fold into
# the training features.  This matches the "clean" ablation 2 of d12.
FEATURES_IMD = [
    "gtfs_heavy_stops_300m", "gtfs_stops_within_300m_pct",
    "infra_cyclable_km", "infra_cyclable_pct",
    "elevation_m", "topography_roughness_index",
    "n_stations_within_500m", "n_stations_within_1km",
    "catchment_density_per_km2",
    "revenu_median_uc", "revenu_d1",
    "part_menages_voit0", "part_velo_travail",
]

N_FOLDS = 5
SEED = 42
LGB_PARAMS = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=63,
    min_child_samples=30, reg_lambda=0.5, random_state=42,
    n_jobs=-1, verbose=-1,
)


def load_data() -> pd.DataFrame:
    trips = pd.read_csv(DATA_ROOT / "processed" / "dataset_prediction_complet.csv")
    trips["datetime"] = pd.to_datetime(trips["datetime"])
    weather = pd.read_csv(DATA_ROOT / "processed" / "weather_data_enriched.csv")
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    gs = pd.read_parquet(DATA_ROOT / "stations_gold_standard_final.parquet")
    mtp = gs[gs["city"].str.contains("Montpellier", case=False, na=False)].copy()
    imd_cols = ["station_name"] + FEATURES_IMD
    mtp = mtp[imd_cols].rename(columns={"station_name": "station"})
    df = trips.merge(weather, on="datetime", how="inner", suffixes=("", "_wx"))
    df = df.merge(mtp, on="station", how="left")
    df = df.dropna(subset=["gtfs_heavy_stops_300m"]).copy()
    df["log_demand"] = np.log1p(df["demande"])
    return df


def precision_at_k(true_sorted_ids, pred_sorted_ids, k: int) -> float:
    return len(set(true_sorted_ids[:k]) & set(pred_sorted_ids[:k])) / k


def main():
    rng = np.random.default_rng(SEED)
    print("Loading Velomagg panel...")
    df = load_data()
    stations = sorted(df["station"].unique())
    n = len(stations)
    print(f"  n={len(df):,}  stations={n}")
    print(f"  Excluding stations with <6 months of activity:")
    counts = df.groupby("station")["datetime"].agg(["min", "max", "count"])
    counts["months_active"] = (counts["max"] - counts["min"]).dt.days / 30
    keep = counts[counts["months_active"] >= 6].index.tolist()
    n_keep = len(keep)
    print(f"  Retained {n_keep}/{n} stations (>=6 months active)")
    df = df[df["station"].isin(keep)].copy()
    stations = sorted(keep)

    # K-fold split of stations
    perm = list(stations)
    rng.shuffle(perm)
    folds = np.array_split(perm, N_FOLDS)
    print(f"  Folds: {[len(f) for f in folds]}")

    per_station = []
    fold_summary = []
    for fi, fold in enumerate(folds):
        fold = list(fold)
        train = df[~df["station"].isin(fold)].copy()
        test = df[df["station"].isin(fold)].copy()
        print(f"\n=== Fold {fi+1}/{N_FOLDS} : "
              f"{train['station'].nunique()} train, {len(fold)} holdout ===")

        def fit(features, name):
            m = lgb.LGBMRegressor(**LGB_PARAMS)
            t0 = time.time()
            m.fit(train[features].astype("float64").values,
                  train["log_demand"].values)
            ys = m.predict(test[features].astype("float64").values)
            yt = test["log_demand"].values
            yt_trips = np.expm1(yt); yp_trips = np.expm1(np.clip(ys, 0, None))
            return ys, yp_trips, time.time() - t0

        ys_no,  yp_no_trips,  ts_no  = fit(FEATURES_TEMPORAL,                "Gminus")
        ys_imd, yp_imd_trips, ts_imd = fit(FEATURES_TEMPORAL + FEATURES_IMD, "G")

        test = test.copy()
        test["pred_no_imd"] = yp_no_trips
        test["pred_imd"] = yp_imd_trips

        # Aggregate per held-out station: MEAN demand per active hour
        # (using totals would be confounded by the number of active hours
        # per station, which differs by station and dominates the ranking
        # of G^- even though G^- has no spatial signal at all).
        agg = (test.groupby("station")
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

        # Hourly R^2 on held-out (out-of-station)
        yt = test["log_demand"].values
        r2_no  = r2_score(np.expm1(yt), yp_no_trips)
        r2_imd = r2_score(np.expm1(yt), yp_imd_trips)
        print(f"  Hourly R²(G-) = {r2_no:+.4f}    R²(G) = {r2_imd:+.4f}    "
              f"ΔR² = {r2_imd-r2_no:+.4f}")
        fold_summary.append(dict(fold=fi+1, n_holdout=len(fold),
                                  r2_hourly_no_imd=r2_no, r2_hourly_imd=r2_imd))

    per_station_df = pd.concat(per_station, ignore_index=True)
    per_station_df.to_csv(OUT_DIR / "d17_lso_velomagg_per_station.csv", index=False)

    # Global ranking metrics across all held-out stations.
    # We rank by MEAN hourly demand (rate), not by total: the total is
    # confounded by per-station active-hour count, which dominates the
    # ranking for any non-spatial model.
    ordered_obs   = per_station_df.sort_values("true_mean",         ascending=False)["station"].tolist()
    ordered_pred  = per_station_df.sort_values("pred_imd_mean",     ascending=False)["station"].tolist()
    ordered_pred0 = per_station_df.sort_values("pred_no_imd_mean",  ascending=False)["station"].tolist()

    rho_imd, _   = spearmanr(per_station_df["true_mean"], per_station_df["pred_imd_mean"])
    rho_no,  _   = spearmanr(per_station_df["true_mean"], per_station_df["pred_no_imd_mean"])
    tau_imd, _   = kendalltau(per_station_df["true_mean"], per_station_df["pred_imd_mean"])
    tau_no,  _   = kendalltau(per_station_df["true_mean"], per_station_df["pred_no_imd_mean"])

    metrics = {
        "n_stations_evaluated": int(len(per_station_df)),
        "n_folds": N_FOLDS,
        "global_spearman_rho_imd": float(rho_imd),
        "global_spearman_rho_no_imd": float(rho_no),
        "global_kendall_tau_imd": float(tau_imd),
        "global_kendall_tau_no_imd": float(tau_no),
        "precision_at_k": {},
        "per_fold": fold_summary,
    }
    for K in (5, 10, 20):
        p_imd = precision_at_k(ordered_obs, ordered_pred,  K)
        p_no  = precision_at_k(ordered_obs, ordered_pred0, K)
        # Null baseline: expected precision if predictions were random
        # P(uniformly picked) = K / n_evaluated
        p_null = K / len(per_station_df)
        metrics["precision_at_k"][f"K{K}"] = dict(
            precision_imd=float(p_imd),
            precision_no_imd=float(p_no),
            precision_random_null=float(p_null),
        )
        print(f"\nPrecision@{K}:  IMD = {p_imd:.3f}  |  no-IMD = {p_no:.3f}"
              f"  |  random = {p_null:.3f}")

    print(f"\nSpearman ρ : IMD = {rho_imd:+.3f}  |  no-IMD = {rho_no:+.3f}")
    print(f"Kendall  τ : IMD = {tau_imd:+.3f}  |  no-IMD = {tau_no:+.3f}")

    with open(OUT_DIR / "d17_lso_velomagg.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Wrote {OUT_DIR/'d17_lso_velomagg.json'}")


if __name__ == "__main__":
    main()
