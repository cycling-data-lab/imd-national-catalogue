"""
d14_montreal_bixi.py — Tier 1 IMD-augmented demand benchmark on BIXI Montréal.

BIXI uses a different schema from the Lyft-operated North American networks:
    STARTSTATIONNAME, STARTSTATIONLATITUDE, STARTSTATIONLONGITUDE,
    ENDSTATIONNAME, ENDSTATIONLATITUDE, ENDSTATIONLONGITUDE,
    STARTTIMEMS, ENDTIMEMS
No station_id column. We match against the IMD parquet
(world_ca_bixi_montr_al.parquet) on (lat, lng) with a 50 m haversine
tolerance, then run the standard G vs G^- comparison and save predictions
for downstream d11 bootstrap.

Output:
  outputs/d14_montreal_bixi_results.json
  outputs/d14_montreal_bixi_predictions.parquet
"""
from __future__ import annotations

import json
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import BallTree

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TIER1 = ROOT / "data_collection" / "tier1_trip_logs" / "montreal_bixi"
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


def load_bixi_panel(zip_path: Path, station_lookup: pd.DataFrame,
                    max_dist_m: float = 50.0) -> pd.DataFrame:
    """Stream BIXI 2024 CSV, aggregate to (station_id, hour) using a
    haversine BallTree to map each trip's start coords to the nearest IMD
    station within max_dist_m (otherwise dropped)."""
    print(f"  Reading {zip_path.name} (2.5 GB uncompressed)...")

    # Build BallTree on IMD lat/lng (in radians for haversine)
    pts = np.deg2rad(station_lookup[["lat", "lng"]].values)
    tree = BallTree(pts, metric="haversine")
    EARTH_R = 6_371_000.0  # m

    counts: dict[tuple[str, pd.Timestamp], int] = {}
    n_rows, n_dropped = 0, 0
    with zipfile.ZipFile(zip_path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            for chunk in pd.read_csv(
                f, chunksize=400_000,
                usecols=["STARTSTATIONLATITUDE", "STARTSTATIONLONGITUDE", "STARTTIMEMS"],
                low_memory=False,
            ):
                chunk = chunk.dropna()
                t0 = time.time()
                # Haversine nearest-neighbour lookup
                coords = np.deg2rad(chunk[["STARTSTATIONLATITUDE", "STARTSTATIONLONGITUDE"]].values)
                dist_rad, idx = tree.query(coords, k=1)
                dist_m = (dist_rad[:, 0] * EARTH_R)
                ok = dist_m <= max_dist_m
                n_dropped += int((~ok).sum())
                # Hour bin
                ts = pd.to_datetime(chunk["STARTTIMEMS"].values[ok], unit="ms")
                hours = ts.floor("h")
                sids = station_lookup["station_id"].values[idx[:, 0][ok]]
                # Tally
                bin_keys = list(zip(sids.astype(str), hours))
                for k in bin_keys:
                    counts[k] = counts.get(k, 0) + 1
                n_rows += len(chunk)
                if n_rows % 4_000_000 == 0:
                    print(f"    {n_rows:,} trips read, "
                          f"{n_dropped:,} dropped (>{max_dist_m:.0f} m), "
                          f"{len(counts):,} bins so far, "
                          f"chunk {time.time()-t0:.1f}s")
    print(f"  Done. {n_rows:,} trips total, {n_dropped:,} dropped, {len(counts):,} bins.")
    df = pd.DataFrame([(s, h, c) for (s, h), c in counts.items()],
                      columns=["station_id", "datetime_hour", "demande"])
    return df


def main():
    t_start = time.time()
    imd = pd.read_parquet(IMD_INTL_DIR / "world_ca_bixi_montr_al.parquet")
    imd["station_id"] = imd["station_id"].astype(str)
    print(f"IMD : {len(imd)} stations (BIXI Montréal)")

    zip_path = TIER1 / "DonneesOuvertes2024.zip"
    if not zip_path.exists():
        print(f"✗ {zip_path} not found — see download URL in d13/d14 notes")
        return
    panel = load_bixi_panel(zip_path, imd[["station_id", "lat", "lng"]])
    print(f"Panel : {len(panel):,} (station,hour) bins  "
          f"{panel['demande'].sum():,} trips")

    df = panel.merge(imd[["station_id"] + FEATS_IMD], on="station_id", how="left")
    df = df.dropna(subset=FEATS_IMD).copy()
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
    df["hour"] = df["datetime_hour"].dt.hour
    df["day_of_week"] = df["datetime_hour"].dt.dayofweek
    df["month"] = df["datetime_hour"].dt.month
    df["log_demand"] = np.log1p(df["demande"])

    # BIXI Montréal closes for winter (typically Nov 15 -> Apr 15) and the
    # operator-published trip log contains sporadic off-season records that
    # cannot be predicted from temporal+weather covariates and that dominate
    # the bootstrap variance via the seasonal regime shift.  We restrict the
    # benchmark to the active season (Apr-Nov) to match the year-round
    # coverage of the other Tier 1 networks (NYC/DC/Chicago/Boston/SF).
    n_pre = len(df)
    df = df[df["month"].between(4, 11)].copy()
    print(f"Seasonal filter (Apr-Nov, BIXI active season) : "
          f"{n_pre:,} -> {len(df):,} bins (-{100*(1-len(df)/n_pre):.1f} %)")

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
        print(f"  {name:20s}  R² = {r2_score(yt_trips, yp_trips):+.4f}  "
              f"({time.time()-t0:.1f}s)")
        return {
            "model": name, "n_test": len(test), "n_features": len(features),
            "r2_log": float(r2_score(yt, ys)),
            "r2_trips": float(r2_score(yt_trips, yp_trips)),
            "mae_trips": float(mean_absolute_error(yt_trips, yp_trips)),
            "rmse_trips": float(np.sqrt(mean_squared_error(yt_trips, yp_trips))),
        }, ys

    print("\nFitting G^- (temporal only)...")
    m_no, pred_no = fit(FEATS_T, "montreal_bixi_G_minus_temporal")
    print("Fitting G  (IMD-augmented)...")
    m_imd, pred_imd = fit(FEATS_T + FEATS_IMD, "montreal_bixi_G_plus_imd")

    # Save predictions for d11 bootstrap
    pred_df = pd.DataFrame({
        "datetime_hour": test["datetime_hour"].values,
        "station_id": test["station_id"].values,
        "y_true_log": test["log_demand"].values,
        "y_pred_no_imd": pred_no,
        "y_pred_imd": pred_imd,
    })
    pred_df.to_parquet(OUT / "d14_montreal_bixi_predictions.parquet", index=False)

    summary = {
        "city": "montreal_bixi",
        "tier": 1,
        "target": "trips",
        "n_total": int(len(df)),
        "n_stations": int(df["station_id"].nunique()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "no_imd": m_no,
        "imd_augmented": m_imd,
        "delta_r2_trips": m_imd["r2_trips"] - m_no["r2_trips"],
        "delta_mae_trips": m_no["mae_trips"] - m_imd["mae_trips"],
        "wall_time_s": round(time.time() - t_start, 1),
    }
    with open(OUT / "d14_montreal_bixi_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved {OUT/'d14_montreal_bixi_results.json'}")
    print(f"  ΔR² = {summary['delta_r2_trips']:+.4f}")


if __name__ == "__main__":
    main()
