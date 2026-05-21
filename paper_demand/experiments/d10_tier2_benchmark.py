"""
d10_tier2_benchmark.py — Tier 2 IMD-augmented demand benchmark on GBFS
pseudo-flow targets.

For each requested French city, load all GBFS station-status polling
parquets, derive pseudo-checkouts as max(0, -Delta num_bikes_available)
between consecutive polls within a <=60-minute gap, aggregate to hourly
station bins, merge with IMD features and fit LightGBM G (full) vs
G- (temporal-only) under a strict temporal hold-out.

Outputs:
  outputs/d10_<city>_results.json  per-city full record (mirrors d3 schema)
  outputs/d10_tier2_summary.csv     consolidated multi-city table

Default city: Paris (1,514 stations, ~404k pseudo-trips).

Designed to be invoked the same way as d3_multicity_benchmark.py:
    python d10_tier2_benchmark.py
    python d10_tier2_benchmark.py --cities Paris lyon toulouse
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

ROOT = Path("/home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand")
SNAPSHOTS = Path("/home/rohanfosse/Bureau/Recherche/bikeshare-data-explorer/data/status_snapshots")
IMD_INTL_DIR = ROOT / "data_collection" / "imd_international"
OUT_DIR = ROOT / "experiments" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Map polling-dir slug -> IMD parquet stem (world_fr_*.parquet without the .parquet)
CITY_TO_IMD = {
    "Paris":                  "world_fr_v_lib_metropole",
    "lyon":                   "world_fr_v_lo_v",
    "toulouse":               "world_fr_v_l_toulouse",
    "v_lille":                "world_fr_v_lib_o",
    "velo-tbm-bordeaux":      "world_fr_le_v_lo_par_tbm",
    "levelo_inurba_marseille":"world_fr_lev_lo_marseille",
}

FEATS_T = ["hour", "day_of_week", "month"]
FEATS_IMD = [
    "gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
    "elevation_m", "topography_roughness_index",
    "n_stations_within_500m", "n_stations_within_1km",
    "catchment_density_per_km2",
]


def load_pseudo_flow(city_slug: str) -> pd.DataFrame:
    """Return per-(station,hour) pseudo-checkout count."""
    poll_dir = SNAPSHOTS / city_slug
    parquets = sorted([p for p in poll_dir.glob("*.parquet")
                       if p.name != "station_info.parquet"])
    if not parquets:
        return pd.DataFrame()
    print(f"      {len(parquets)} polling parquets")
    frames = [pd.read_parquet(p) for p in parquets]
    df = pd.concat(frames, ignore_index=True)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    df = df.sort_values(["station_id", "fetched_at"]).reset_index(drop=True)

    df["bikes_prev"] = df.groupby("station_id")["num_bikes_available"].shift(1)
    df["delta"] = df["num_bikes_available"] - df["bikes_prev"]
    df["time_prev"] = df.groupby("station_id")["fetched_at"].shift(1)
    df["gap_min"] = (df["fetched_at"] - df["time_prev"]).dt.total_seconds() / 60
    valid = df["gap_min"].between(0, 60, inclusive="right")
    df["pseudo_checkout"] = np.where(valid & (df["delta"] < 0),
                                     -df["delta"].astype("Int64"), 0)
    df["datetime_hour"] = df["fetched_at"].dt.floor("h").dt.tz_convert(None)
    panel = (df.groupby(["station_id", "datetime_hour"], as_index=False)
               ["pseudo_checkout"].sum()
               .rename(columns={"pseudo_checkout": "demande"}))
    panel["station_id"] = panel["station_id"].astype(str)
    panel["demande"] = panel["demande"].astype(int)
    return panel


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["datetime_hour"].dt.hour
    df["day_of_week"] = df["datetime_hour"].dt.dayofweek
    df["month"] = df["datetime_hour"].dt.month
    df["log_demand"] = np.log1p(df["demande"])
    return df


def evaluate(name, y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(np.clip(y_pred_log, 0, None))
    return {
        "model": name,
        "n_test": int(len(y_true)),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "r2_trips": float(r2_score(y_true, y_pred)),
        "mae_trips": float(mean_absolute_error(y_true, y_pred)),
        "rmse_trips": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def fit_lgb(train, test, features, name):
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        min_child_samples=30, reg_lambda=0.5, random_state=42,
        n_jobs=-1, verbose=-1,
    )
    t0 = time.time()
    model.fit(train[features].values, train["log_demand"].values)
    fit_s = time.time() - t0
    y_pred = model.predict(test[features].values)
    m = evaluate(name, test["log_demand"].values, y_pred)
    m["fit_seconds"] = round(fit_s, 1)
    # Save predictions for downstream bootstrap CI
    return m, y_pred


def run_city(city_slug: str, train_frac: float = 0.8):
    print(f"\n{'='*70}\n[Tier 2 / {city_slug}]\n{'='*70}")
    t0 = time.time()

    imd_stem = CITY_TO_IMD.get(city_slug)
    if not imd_stem:
        print(f"  ✗ No IMD mapping for {city_slug}"); return None
    imd_path = IMD_INTL_DIR / f"{imd_stem}.parquet"
    if not imd_path.exists():
        print(f"  ✗ IMD parquet missing: {imd_path}"); return None
    imd = pd.read_parquet(imd_path)
    imd["station_id"] = imd["station_id"].astype(str)
    print(f"  IMD : {len(imd)} stations  ({imd_path.name})")

    panel = load_pseudo_flow(city_slug)
    if panel.empty:
        print("  ✗ Empty pseudo-flow panel"); return None
    print(f"  Pseudo-flow panel : {len(panel):,} (station,hour) bins, "
          f"{int(panel['demande'].sum()):,} total pseudo-trips")

    df = panel.merge(imd[["station_id"] + FEATS_IMD], on="station_id", how="left")
    df = df.dropna(subset=FEATS_IMD).copy()
    print(f"  After IMD merge   : {len(df):,} bins  "
          f"({df['station_id'].nunique()} stations)")

    df = add_temporal_features(df)
    df = df.sort_values("datetime_hour").reset_index(drop=True)
    cut = int(train_frac * len(df))
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    print(f"  Train : {len(train):,}  ({train['datetime_hour'].min()} -> "
          f"{train['datetime_hour'].max()})")
    print(f"  Test  : {len(test):,}  ({test['datetime_hour'].min()} -> "
          f"{test['datetime_hour'].max()})")

    print("  Training G- (temporal only) ...")
    m_no, pred_no = fit_lgb(train, test, FEATS_T, f"{city_slug}_G_minus_temporal")
    print(f"    R²(G-) = {m_no['r2_trips']:+.4f}")
    print("  Training G  (IMD-augmented) ...")
    m_imd, pred_imd = fit_lgb(train, test, FEATS_T + FEATS_IMD,
                              f"{city_slug}_G_plus_imd")
    print(f"    R²(G)  = {m_imd['r2_trips']:+.4f}    "
          f"ΔR² = {m_imd['r2_trips']-m_no['r2_trips']:+.4f}")

    # Persist predictions for downstream bootstrap (d11)
    pred_df = pd.DataFrame({
        "datetime_hour": test["datetime_hour"].values,
        "station_id":    test["station_id"].values,
        "y_true_log":    test["log_demand"].values,
        "y_pred_no_imd": pred_no,
        "y_pred_imd":    pred_imd,
    })
    pred_path = OUT_DIR / f"d10_{city_slug}_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)

    summary = {
        "city": city_slug,
        "tier": 2,
        "target": "pseudo_flow",
        "n_total": int(len(df)),
        "n_stations": int(df["station_id"].nunique()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "polling_span_h": float((test["datetime_hour"].max()
                                  - train["datetime_hour"].min()).total_seconds()/3600),
        "no_imd": m_no,
        "imd_augmented": m_imd,
        "delta_r2_trips": m_imd["r2_trips"] - m_no["r2_trips"],
        "delta_mae_trips": m_no["mae_trips"] - m_imd["mae_trips"],
        "wall_time_s": round(time.time() - t0, 1),
    }
    out_path = OUT_DIR / f"d10_{city_slug}_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved {out_path}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cities", nargs="*", default=["Paris"],
                   help="Polling-dir slugs (see CITY_TO_IMD)")
    args = p.parse_args()

    results = []
    for c in args.cities:
        try:
            r = run_city(c)
            if r:
                results.append(r)
        except Exception as e:
            print(f"\n[ERROR] {c}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    if results:
        rows = [{
            "city": r["city"],
            "n_stations": r["n_stations"],
            "n_total": r["n_total"],
            "polling_span_h": round(r["polling_span_h"], 1),
            "r2_no_imd": r["no_imd"]["r2_trips"],
            "r2_imd": r["imd_augmented"]["r2_trips"],
            "delta_r2": r["delta_r2_trips"],
            "mae_no_imd": r["no_imd"]["mae_trips"],
            "mae_imd": r["imd_augmented"]["mae_trips"],
        } for r in results]
        df = pd.DataFrame(rows)
        df.to_csv(OUT_DIR / "d10_tier2_summary.csv", index=False)
        print("\n=== TIER 2 MULTI-CITY SUMMARY ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
