"""
d16_london_tfl.py — Tier 1 IMD-augmented demand benchmark on TfL
Santander Cycles (London).

TfL trip-log schema (cycling.data.tfl.gov.uk weekly CSV):
    "Number","Start date","Start station number","Start station",
    "End date","End station number","End station","Bike number",
    "Bike model","Total duration","Total duration (ms)"

We aggregate trips to (Start station number, hour) hourly bins, then
merge with the IMD parquet computed by d15_tfl_setup.py
(world_gb_tfl_santander_cycles.parquet — station_id is the TerminalName,
matching the trip log's "Start station number" with leading zeros).
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TIER1 = ROOT / "data_collection" / "tier1_trip_logs" / "london_tfl"
IMD_INTL_DIR = ROOT / "data_collection" / "imd_international"
OUT = ROOT / "experiments" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

FEATS_T = ["hour", "day_of_week", "month"]
FEATS_IMD = [
    "gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
    "elevation_m", "topography_roughness_index",
    "n_stations_within_500m", "n_stations_within_1km",
    "catchment_density_per_km2",
]


def load_tfl_panel() -> pd.DataFrame:
    """Stream all TfL weekly CSVs in TIER1, aggregate to (station, hour) bins."""
    csvs = sorted(TIER1.glob("*.csv"))
    print(f"  {len(csvs)} TfL weekly CSVs found in {TIER1}")
    if not csvs:
        return pd.DataFrame()
    counts: dict[tuple[str, pd.Timestamp], int] = {}
    n_rows = 0
    for csv in csvs:
        t0 = time.time()
        for chunk in pd.read_csv(
            csv, chunksize=300_000,
            usecols=["Start date", "Start station number"],
            low_memory=False,
        ):
            chunk = chunk.dropna(subset=["Start date", "Start station number"])
            chunk["sid"] = chunk["Start station number"].astype(str).str.zfill(6)
            ts = pd.to_datetime(chunk["Start date"], errors="coerce")
            chunk["hour"] = ts.dt.floor("h")
            chunk = chunk.dropna(subset=["hour"])
            grouped = chunk.groupby(["sid", "hour"]).size().reset_index(name="d")
            for s, h, c in grouped.itertuples(index=False):
                counts[(s, h)] = counts.get((s, h), 0) + int(c)
        n_rows += sum(1 for _ in open(csv, "rb"))
        print(f"    {csv.name} done ({time.time()-t0:.1f}s, {len(counts):,} bins so far)")
    df = pd.DataFrame([(s, h, c) for (s, h), c in counts.items()],
                      columns=["station_id", "datetime_hour", "demande"])
    return df


def main():
    t_start = time.time()
    imd_path = IMD_INTL_DIR / "london_tfl.parquet"
    if not imd_path.exists():
        print(f"✗ IMD parquet missing: {imd_path}. Run d15_tfl_setup.py first."); return
    imd = pd.read_parquet(imd_path)
    imd["station_id"] = imd["station_id"].astype(str).str.zfill(6)
    print(f"IMD : {len(imd)} stations (TfL London)")

    panel = load_tfl_panel()
    if panel.empty:
        print("✗ Empty trip panel"); return
    print(f"Panel : {len(panel):,} (station,hour) bins, {int(panel['demande'].sum()):,} trips")

    df = panel.merge(imd[["station_id"] + FEATS_IMD], on="station_id", how="left")
    n_pre = len(df)
    df = df.dropna(subset=FEATS_IMD).copy()
    print(f"After IMD merge : {len(df):,} bins ({n_pre - len(df):,} dropped, "
          f"{df['station_id'].nunique()} stations)")

    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
    df["hour"] = df["datetime_hour"].dt.hour
    df["day_of_week"] = df["datetime_hour"].dt.dayofweek
    df["month"] = df["datetime_hour"].dt.month
    df["log_demand"] = np.log1p(df["demande"])

    df = df.sort_values("datetime_hour").reset_index(drop=True)
    cut = int(0.8 * len(df))
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    print(f"Train : {len(train):,}  ({train['datetime_hour'].min()} -> {train['datetime_hour'].max()})")
    print(f"Test  : {len(test):,}  ({test['datetime_hour'].min()} -> {test['datetime_hour'].max()})")

    def fit(features, name):
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                              min_child_samples=30, reg_lambda=0.5, random_state=42,
                              n_jobs=-1, verbose=-1)
        t0 = time.time()
        m.fit(train[features].values, train["log_demand"].values)
        ys = m.predict(test[features].values)
        yt = test["log_demand"].values
        yt_trips = np.expm1(yt); yp_trips = np.expm1(np.clip(ys, 0, None))
        print(f"  {name:25s}  R² = {r2_score(yt_trips, yp_trips):+.4f}  "
              f"({time.time()-t0:.1f}s)")
        return {
            "model": name, "n_test": len(test), "n_features": len(features),
            "r2_log": float(r2_score(yt, ys)),
            "r2_trips": float(r2_score(yt_trips, yp_trips)),
            "mae_trips": float(mean_absolute_error(yt_trips, yp_trips)),
            "rmse_trips": float(np.sqrt(mean_squared_error(yt_trips, yp_trips))),
        }, ys

    print("\nFitting G^- (temporal only)...")
    m_no, pred_no = fit(FEATS_T, "london_tfl_G_minus_temporal")
    print("Fitting G  (IMD-augmented)...")
    m_imd, pred_imd = fit(FEATS_T + FEATS_IMD, "london_tfl_G_plus_imd")

    # Save predictions for d11 bootstrap
    pred_df = pd.DataFrame({
        "datetime_hour": test["datetime_hour"].values,
        "station_id":    test["station_id"].values,
        "y_true_log":    test["log_demand"].values,
        "y_pred_no_imd": pred_no,
        "y_pred_imd":    pred_imd,
    })
    pred_df.to_parquet(OUT / "d16_london_tfl_predictions.parquet", index=False)

    summary = {
        "city": "london_tfl", "tier": 1, "target": "trips",
        "n_total": int(len(df)), "n_stations": int(df["station_id"].nunique()),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "no_imd": m_no, "imd_augmented": m_imd,
        "delta_r2_trips": m_imd["r2_trips"] - m_no["r2_trips"],
        "delta_mae_trips": m_no["mae_trips"] - m_imd["mae_trips"],
        "wall_time_s": round(time.time() - t_start, 1),
    }
    with open(OUT / "d16_london_tfl_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved {OUT/'d16_london_tfl_results.json'}")
    print(f"  ΔR² = {summary['delta_r2_trips']:+.4f}")


if __name__ == "__main__":
    main()
