"""
d1_demand_model.py — Proof of concept: IMD-augmented spatio-temporal
demand model on the Vélomagg Montpellier panel.

Compares three model families on hourly station-level trip demand :
 (A) Temporal-only  : hour, day-of-week, month, season, weather
 (B) Station-fixed-effect : (A) + one-hot station id
 (C) IMD-augmented : (A) + station-level IMD components (M, I, T, D
                     proxies, plus the socio-economic enrichments)

Train: 2021-09 → 2024-08 (3 years)
Test : 2024-09 → 2025-08 (1 year held out)
Target : log1p(demande_hourly)
Metric : R^2, MAE, RMSE on test set (in trips units after expm1)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_ROOT = Path("/home/rohanfosse/Bureau/Recherche/bikeshare-data-explorer/data")
OUT_DIR = Path("/home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/experiments/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    trips = pd.read_csv(DATA_ROOT / "processed" / "dataset_prediction_complet.csv")
    trips["datetime"] = pd.to_datetime(trips["datetime"])
    weather = pd.read_csv(DATA_ROOT / "processed" / "weather_data_enriched.csv")
    weather["datetime"] = pd.to_datetime(weather["datetime"])

    gs = pd.read_parquet(DATA_ROOT / "stations_gold_standard_final.parquet")
    mtp = gs[gs["city"].str.contains("Montpellier", case=False, na=False)].copy()

    # IMD-side features at the station level
    imd_cols = [
        "station_name",
        # M-axis
        "gtfs_heavy_stops_300m",
        "gtfs_stops_within_300m_pct",
        # I-axis
        "infra_cyclable_km",
        "infra_cyclable_pct",
        # T-axis
        "elevation_m",
        "topography_roughness_index",
        # D-axis proxies (per-station, intra-system)
        "n_stations_within_500m",
        "n_stations_within_1km",
        "catchment_density_per_km2",
        # Socio-econ (commune-level, joined to the station)
        "revenu_median_uc",
        "revenu_d1",
        "part_menages_voit0",
        "part_velo_travail",
    ]
    mtp = mtp[imd_cols].rename(columns={"station_name": "station"})

    # Merge trips × weather × IMD enrichment
    df = trips.merge(weather, on="datetime", how="inner", suffixes=("", "_wx"))
    df = df.merge(mtp, on="station", how="left")  # IMD enrichment

    # Keep only stations with IMD enrichment (drop the 14 trip-only stations)
    df = df.dropna(subset=["gtfs_heavy_stops_300m"]).copy()

    # Target on log scale (trip counts are heavy-tailed)
    df["log_demand"] = np.log1p(df["demande"])

    return df


def make_splits(df: pd.DataFrame, test_cutoff: str = "2024-09-01"):
    train = df[df["datetime"] < test_cutoff].copy()
    test = df[df["datetime"] >= test_cutoff].copy()
    return train, test


def evaluate(model_name: str, y_true_log, y_pred_log) -> dict:
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(np.clip(y_pred_log, 0, None))  # avoid negative trips
    return {
        "model": model_name,
        "n_test": int(len(y_true)),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "r2_trips": float(r2_score(y_true, y_pred)),
        "mae_trips": float(mean_absolute_error(y_true, y_pred)),
        "rmse_trips": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mean_true": float(y_true.mean()),
        "mean_pred": float(y_pred.mean()),
    }


FEATURES_TEMPORAL = [
    "hour", "day_of_week", "month", "season",
    "temperature", "humidity", "precipitation", "wind_speed",
    "cloud_cover", "is_raining", "is_heavy_rain", "feels_like",
    "bad_weather_score",
]

FEATURES_NETWORK_BASELINE = [
    "network_volume", "network_entropy", "network_gini",
]

FEATURES_IMD = [
    "gtfs_heavy_stops_300m", "gtfs_stops_within_300m_pct",
    "infra_cyclable_km", "infra_cyclable_pct",
    "elevation_m", "topography_roughness_index",
    "n_stations_within_500m", "n_stations_within_1km",
    "catchment_density_per_km2",
    "revenu_median_uc", "revenu_d1",
    "part_menages_voit0", "part_velo_travail",
]


def train_lgb(train, test, features, target="log_demand", model_name="model"):
    X_train = train[features].copy()
    y_train = train[target].values
    X_test = test[features].copy()
    y_test = test[target].values

    # Encode any non-numeric column as integer codes (manual category)
    from pandas.api.types import is_numeric_dtype
    cat_cols = [c for c in features if not is_numeric_dtype(X_train[c])]
    for c in cat_cols:
        # build a shared category map across train and test
        combined = pd.concat([X_train[c].astype(str), X_test[c].astype(str)])
        all_cats = combined.astype("category").cat.categories
        X_train[c] = pd.Categorical(X_train[c].astype(str), categories=all_cats).codes
        X_test[c] = pd.Categorical(X_test[c].astype(str), categories=all_cats).codes

    # Ensure all numeric float64
    X_train = X_train.astype("float64")
    X_test = X_test.astype("float64")

    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        min_child_samples=30,
        reg_alpha=0.0,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    t0 = time.time()
    model.fit(X_train, y_train, categorical_feature=cat_cols if cat_cols else "auto")
    fit_s = time.time() - t0
    y_pred = model.predict(X_test)

    metrics = evaluate(model_name, y_test, y_pred)
    metrics["fit_seconds"] = round(fit_s, 1)
    metrics["n_features"] = len(features)

    # Feature importance
    fi = pd.DataFrame({"feature": features, "importance": model.booster_.feature_importance(importance_type="gain")})
    fi = fi.sort_values("importance", ascending=False)

    return metrics, fi, y_pred


def main():
    print("[1/5] Loading data...")
    t0 = time.time()
    df = load_data()
    print(f"      n={len(df):,}  stations={df['station'].nunique()}  span={df['datetime'].min()} to {df['datetime'].max()}")

    print("[2/5] Train/test split...")
    train, test = make_splits(df)
    print(f"      train={len(train):,}  test={len(test):,}")

    print("[3/5] Training Model A (Temporal+weather+network only, NO spatial)...")
    feats_A = FEATURES_TEMPORAL + FEATURES_NETWORK_BASELINE
    metrics_A, fi_A, pred_A = train_lgb(train, test, feats_A, model_name="A_temporal_only")
    print(f"      R²_trips = {metrics_A['r2_trips']:.4f}  MAE = {metrics_A['mae_trips']:.3f}")

    print("[4/5] Training Model B (Temporal + Station fixed effect)...")
    feats_B = FEATURES_TEMPORAL + FEATURES_NETWORK_BASELINE + ["station"]
    metrics_B, fi_B, pred_B = train_lgb(train, test, feats_B, model_name="B_temporal_plus_station_FE")
    print(f"      R²_trips = {metrics_B['r2_trips']:.4f}  MAE = {metrics_B['mae_trips']:.3f}")

    print("[5/5] Training Model C (Temporal + IMD components, NO station FE)...")
    feats_C = FEATURES_TEMPORAL + FEATURES_NETWORK_BASELINE + FEATURES_IMD
    metrics_C, fi_C, pred_C = train_lgb(train, test, feats_C, model_name="C_imd_augmented")
    print(f"      R²_trips = {metrics_C['r2_trips']:.4f}  MAE = {metrics_C['mae_trips']:.3f}")

    # Save test-set predictions for bootstrap CI (d11)
    pred_df = pd.DataFrame({
        "datetime_hour": test["datetime"].values,
        "station_id":    test["station"].astype(str).values,
        "y_true_log":    test["log_demand"].values,
        "y_pred_no_imd": pred_A,   # A = temporal-only baseline = G^-
        "y_pred_imd":    pred_C,   # C = IMD-augmented            = G
    })
    pred_df.to_parquet(OUT_DIR / "d1_velomagg_predictions.parquet", index=False)

    # Save
    summary = {
        "experiment": "d1_demand_model",
        "data": {
            "n_total": int(len(df)),
            "n_stations": int(df["station"].nunique()),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_span": f"{train['datetime'].min()} to {train['datetime'].max()}",
            "test_span": f"{test['datetime'].min()} to {test['datetime'].max()}",
        },
        "models": [metrics_A, metrics_B, metrics_C],
        "delta_C_vs_A_r2_trips": metrics_C["r2_trips"] - metrics_A["r2_trips"],
        "delta_C_vs_B_r2_trips": metrics_C["r2_trips"] - metrics_B["r2_trips"],
    }
    with open(OUT_DIR / "d1_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    fi_A.to_csv(OUT_DIR / "d1_feature_importance_A.csv", index=False)
    fi_B.to_csv(OUT_DIR / "d1_feature_importance_B.csv", index=False)
    fi_C.to_csv(OUT_DIR / "d1_feature_importance_C.csv", index=False)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (test set, trips units)")
    print("=" * 60)
    for m in summary["models"]:
        print(f"  {m['model']:30s}  R²={m['r2_trips']:.4f}  MAE={m['mae_trips']:.3f}  RMSE={m['rmse_trips']:.3f}")
    print(f"\n  ΔR² (C vs A, IMD gain over temporal-only) = {summary['delta_C_vs_A_r2_trips']:+.4f}")
    print(f"  ΔR² (C vs B, IMD vs station FE)            = {summary['delta_C_vs_B_r2_trips']:+.4f}")

    print("\nTop-10 features model C (IMD-augmented):")
    print(fi_C.head(10).to_string(index=False))

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
