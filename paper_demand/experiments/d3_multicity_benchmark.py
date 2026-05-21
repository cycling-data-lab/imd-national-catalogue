"""
d3_multicity_benchmark.py — multi-city version of the IMD-augmented
demand benchmark.

Loops over Tier 1 cities (Lyft trip-log schema), aggregates trip data
to (station, hour) bins, joins with the per-station IMD features
computed by `imd_international.py`, and runs LightGBM with vs without
IMD on a temporal hold-out split.

Per-city outputs to
  paper_demand/experiments/outputs/d3_<city>_results.json
and a combined summary CSV.

Cross-city transfer (train on city A, test on city B) is run as a
companion experiment at the end.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pandas.api.types import is_numeric_dtype

warnings.filterwarnings("ignore")

ROOT = Path("/home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand")
TIER1_DIR = ROOT / "data_collection" / "tier1_trip_logs"
IMD_INTL_DIR = ROOT / "data_collection" / "imd_international"
OUT_DIR = ROOT / "experiments" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LYFT_CITIES = [
    "dc_capitalbikeshare",
    "chicago_divvy",
    "boston_bluebikes",
    "sf_baywheels",
    "nyc_citibike",
]

FEATS_T = ["hour", "day_of_week", "month"]
FEATS_NET = []  # filled per-city from data
FEATS_IMD = [
    "gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
    "elevation_m", "topography_roughness_index",
    "n_stations_within_500m", "n_stations_within_1km",
    "catchment_density_per_km2",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def list_trip_files(city: str) -> list[Path]:
    city_dir = TIER1_DIR / city
    if not city_dir.exists():
        return []
    files = sorted([p for p in city_dir.glob("*.zip") if "tripdata" in p.name])
    return files


def _sniff_schema(zfile: zipfile.ZipFile, csv_name: str) -> tuple[str, str]:
    """Return (started_col, station_col) for the trip-log CSV; supports
    both the modern Lyft schema and the older Hubway-style schema."""
    with zfile.open(csv_name) as f:
        head = pd.read_csv(f, nrows=0)
    cols = set(head.columns)
    if "started_at" in cols and "start_station_id" in cols:
        return "started_at", "start_station_id"
    if "starttime" in cols and "start station id" in cols:
        return "starttime", "start station id"
    raise ValueError(f"Unknown schema in {csv_name}: cols={sorted(cols)}")


def load_trip_zip(zip_path: Path) -> pd.DataFrame:
    """Return a DataFrame with (datetime_hour, station_id, demande) bins
    from a single trip-log zip. Auto-detects Lyft vs Hubway schema per CSV."""
    frames = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.startswith("__MACOSX") or not name.endswith(".csv"):
                continue
            try:
                started_col, station_col = _sniff_schema(z, name)
            except ValueError as e:
                print(f"        ! skip {name}: {e}"); continue
            with z.open(name) as f:
                for chunk in pd.read_csv(
                    f, chunksize=300_000,
                    usecols=[started_col, station_col],
                    low_memory=False,
                ):
                    chunk = chunk.dropna(subset=[station_col])
                    chunk["station_id"] = chunk[station_col].astype(str)
                    chunk["datetime_hour"] = pd.to_datetime(
                        chunk[started_col], errors="coerce").dt.floor("h")
                    chunk = chunk.dropna(subset=["datetime_hour"])
                    grouped = (chunk.groupby(["station_id", "datetime_hour"])
                               .size().reset_index(name="demande"))
                    frames.append(grouped)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = (df.groupby(["station_id", "datetime_hour"], as_index=False)
          ["demande"].sum())
    return df


def build_panel(city: str, max_files: int | None = None) -> pd.DataFrame:
    """Concatenate all monthly trip zips into a per-(station,hour) demand panel."""
    files = list_trip_files(city)
    if max_files:
        files = files[:max_files]
    if not files:
        return pd.DataFrame()
    print(f"      {len(files)} monthly zips found")
    frames = []
    for f in files:
        t0 = time.time()
        df = load_trip_zip(f)
        if not df.empty:
            frames.append(df)
            print(f"        {f.name}: {len(df):,} (station,hour) bins  "
                  f"({time.time()-t0:.1f}s)")
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel = (panel.groupby(["station_id", "datetime_hour"], as_index=False)
             ["demande"].sum())
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


def _encode(X_train, X_test, features):
    Xt, Xv = X_train[features].copy(), X_test[features].copy()
    for c in features:
        if not is_numeric_dtype(Xt[c]):
            combined = pd.concat([Xt[c].astype(str), Xv[c].astype(str)])
            cats = combined.astype("category").cat.categories
            Xt[c] = pd.Categorical(Xt[c].astype(str), categories=cats).codes
            Xv[c] = pd.Categorical(Xv[c].astype(str), categories=cats).codes
    return Xt.astype("float64"), Xv.astype("float64")


def model_lgb(train, test, features, name):
    Xt, Xv = _encode(train, test, features)
    cat_cols = [c for c in features if not is_numeric_dtype(train[c])]
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        min_child_samples=30, reg_lambda=0.5, random_state=42,
        n_jobs=-1, verbose=-1,
    )
    t0 = time.time()
    model.fit(Xt, train["log_demand"].values,
              categorical_feature=cat_cols if cat_cols else "auto")
    fit_s = time.time() - t0
    y_pred = model.predict(Xv)
    m = evaluate(name, test["log_demand"].values, y_pred)
    m["fit_seconds"] = round(fit_s, 1)
    return m, model


# ─── Per-city benchmark ──────────────────────────────────────────────────────
def run_city_benchmark(city: str, train_cutoff_frac: float = 0.8):
    print(f"\n{'='*70}\n[{city}]\n{'='*70}")
    t0 = time.time()

    # IMD features
    imd_path = IMD_INTL_DIR / f"{city}.parquet"
    if not imd_path.exists():
        print(f"  ✗ No IMD features at {imd_path}")
        return None
    imd = pd.read_parquet(imd_path)
    imd["station_id"] = imd["station_id"].astype(str)
    print(f"  IMD features : {len(imd)} stations × {len(imd.columns)} cols")

    # Build trip panel
    print(f"  Loading trip panel from {TIER1_DIR / city}...")
    panel = build_panel(city)
    if panel.empty:
        print(f"  ✗ Empty trip panel")
        return None
    print(f"  Trip panel : {len(panel):,} (station,hour) bins")

    # Merge IMD
    df = panel.merge(imd[["station_id"] + FEATS_IMD], on="station_id", how="left")
    df = df.dropna(subset=FEATS_IMD).copy()
    print(f"  After IMD merge : {len(df):,} bins  ({df['station_id'].nunique()} stations)")

    df = add_temporal_features(df)

    # Temporal split
    df = df.sort_values("datetime_hour").reset_index(drop=True)
    cut = int(train_cutoff_frac * len(df))
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    print(f"  Train : {len(train):,}  Test : {len(test):,}")
    print(f"  Train span : {train['datetime_hour'].min()} -> {train['datetime_hour'].max()}")
    print(f"  Test  span : {test['datetime_hour'].min()} -> {test['datetime_hour'].max()}")

    # No-IMD model (temporal only)
    print(f"  Training G- (no-IMD) ...")
    m_no, mdl_no = model_lgb(train, test, FEATS_T, f"{city}_G_minus_temporal")
    pred_no = mdl_no.predict(_encode(train, test, FEATS_T)[1])

    # IMD-augmented model
    print(f"  Training G+ (IMD-augmented) ...")
    m_imd, mdl_imd = model_lgb(train, test, FEATS_T + FEATS_IMD,
                                f"{city}_G_plus_imd")
    pred_imd = mdl_imd.predict(_encode(train, test, FEATS_T + FEATS_IMD)[1])

    # Save test-set predictions for downstream bootstrap CI (d11)
    pred_df = pd.DataFrame({
        "datetime_hour": test["datetime_hour"].values,
        "station_id":    test["station_id"].values,
        "y_true_log":    test["log_demand"].values,
        "y_pred_no_imd": pred_no,
        "y_pred_imd":    pred_imd,
    })
    pred_df.to_parquet(OUT_DIR / f"d3_{city}_predictions.parquet", index=False)

    summary = {
        "city": city,
        "n_total": int(len(df)),
        "n_stations": int(df["station_id"].nunique()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "no_imd": m_no,
        "imd_augmented": m_imd,
        "delta_r2_trips": m_imd["r2_trips"] - m_no["r2_trips"],
        "delta_mae_trips": m_no["mae_trips"] - m_imd["mae_trips"],
        "wall_time_s": round(time.time() - t0, 1),
    }
    out_path = OUT_DIR / f"d3_{city}_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved {out_path}")
    print(f"  R²(G-) = {m_no['r2_trips']:.4f}    R²(G+) = {m_imd['r2_trips']:.4f}    "
          f"ΔR² = {summary['delta_r2_trips']:+.4f}")

    return summary, mdl_imd, df


def run_combined_summary(per_city: dict):
    rows = []
    for city, (sm, _, _) in per_city.items():
        rows.append({
            "city": city,
            "n_stations": sm["n_stations"],
            "n_total": sm["n_total"],
            "r2_no_imd": sm["no_imd"]["r2_trips"],
            "r2_imd": sm["imd_augmented"]["r2_trips"],
            "delta_r2": sm["delta_r2_trips"],
            "mae_no_imd": sm["no_imd"]["mae_trips"],
            "mae_imd": sm["imd_augmented"]["mae_trips"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "d3_multicity_summary.csv", index=False)
    print("\n=== MULTI-CITY SUMMARY ===")
    print(df.to_string(index=False))
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cities", nargs="*", default=LYFT_CITIES,
                   help="Cities to benchmark (slugs)")
    args = p.parse_args()

    per_city = {}
    for c in args.cities:
        try:
            result = run_city_benchmark(c)
            if result:
                per_city[c] = result
        except Exception as e:
            print(f"\n[ERROR] {c}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    if per_city:
        run_combined_summary(per_city)


if __name__ == "__main__":
    main()
