"""
d12_ablations.py — Two ablation studies on the Velomagg panel.

Both studies use the same LightGBM hyperparameters as d1 (n_estimators=400,
learning_rate=0.05, num_leaves=63, min_child_samples=30, reg_lambda=0.5,
random_state=42) and the same temporal hold-out (train 2021-09-01 to
2024-08-31, test 2024-09-01 to 2025-08-31).

Study 1 — T-axis sensitivity.
    Refits the IMD-augmented model with the two T-axis features
    (elevation_m, topography_roughness_index) zeroed.  This quantifies the
    R^2 cost of the national rollout's T=0 substitution at the predictive
    benchmark level (the paper's §6.4 limit currently only bounds the
    information loss using a univariate FUB correlation argument, ~8%).

Study 2 — Network-state ablation.
    Refits G^- and G with the three network-state features removed
    (network_volume, network_entropy, network_gini).  This isolates the
    IMD gain from any autoregressive leakage carried by network_volume
    (= total trips on the network at t-1, the #1 feature by gain in the
    base specification).

Writes outputs/d12_ablations.json and a paired bootstrap CI to
outputs/d12_ablations_ci.csv.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_ROOT = Path(
    os.environ.get(
        "BIKESHARE_DATA_ROOT",
        str(Path(__file__).resolve().parents[3] / "bikeshare-data-explorer" / "data"),
    )
)
OUT_DIR = Path(__file__).resolve().parents[0] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same feature groups as d1
FEATURES_TEMPORAL = [
    "hour", "day_of_week", "month", "season",
    "temperature", "humidity", "precipitation", "wind_speed",
    "cloud_cover", "is_raining", "is_heavy_rain", "feels_like",
    "bad_weather_score",
]
FEATURES_NETWORK = ["network_volume", "network_entropy", "network_gini"]
FEATURES_T = ["elevation_m", "topography_roughness_index"]
FEATURES_IMD_NON_T = [
    "gtfs_heavy_stops_300m", "gtfs_stops_within_300m_pct",
    "infra_cyclable_km", "infra_cyclable_pct",
    "n_stations_within_500m", "n_stations_within_1km",
    "catchment_density_per_km2",
    "revenu_median_uc", "revenu_d1",
    "part_menages_voit0", "part_velo_travail",
]
FEATURES_IMD = FEATURES_T + FEATURES_IMD_NON_T

B = 500
SEED = 42


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


def split(df: pd.DataFrame, cutoff: str = "2024-09-01"):
    return df[df["datetime"] < cutoff].copy(), df[df["datetime"] >= cutoff].copy()


def fit(train, test, features, name):
    Xt = train[features].astype("float64").values
    Xv = test[features].astype("float64").values
    y_train = train["log_demand"].values
    y_test = test["log_demand"].values
    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        min_child_samples=30, reg_lambda=0.5, random_state=42,
        n_jobs=-1, verbose=-1,
    )
    t0 = time.time()
    model.fit(Xt, y_train)
    fit_s = time.time() - t0
    y_pred = model.predict(Xv)
    yt_trips = np.expm1(y_test)
    yp_trips = np.expm1(np.clip(y_pred, 0, None))
    return {
        "model": name,
        "n_features": len(features),
        "r2_log": float(r2_score(y_test, y_pred)),
        "r2_trips": float(r2_score(yt_trips, yp_trips)),
        "mae_trips": float(mean_absolute_error(yt_trips, yp_trips)),
        "rmse_trips": float(np.sqrt(mean_squared_error(yt_trips, yp_trips))),
        "fit_seconds": round(fit_s, 1),
    }, y_pred


def block_ci(y_true_log: np.ndarray, y_pred_log: np.ndarray,
             times: pd.Series, rng: np.random.Generator) -> tuple[float, float, float]:
    """Paired weekly block bootstrap CI on R^2 (trip-count scale)."""
    iso = times.dt.isocalendar()
    blocks = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).values
    unique_blocks = np.unique(blocks)
    by_block = {b: np.where(blocks == b)[0] for b in unique_blocks}

    def r2(idx):
        yt = np.expm1(y_true_log[idx])
        yp = np.expm1(np.clip(y_pred_log[idx], 0, None))
        return r2_score(yt, yp)

    point = r2(np.arange(len(y_true_log)))
    samples = np.empty(B, dtype=float)
    for b in range(B):
        sel = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([by_block[s] for s in sel])
        samples[b] = r2(idx)
    return point, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main():
    rng = np.random.default_rng(SEED)
    print("Loading Velomagg panel...")
    df = load_data()
    train, test = split(df)
    print(f"  n_train={len(train):,}  n_test={len(test):,}")

    results = {}

    # === Reference ===
    print("\n=== Reference (paper §5) ===")
    print("Fitting A (temporal+weather+network) ...")
    ref_A, pred_A = fit(train, test, FEATURES_TEMPORAL + FEATURES_NETWORK, "A_baseline")
    print(f"  R²_trips = {ref_A['r2_trips']:+.4f}")
    print("Fitting C (temporal+weather+network+IMD) ...")
    ref_C, pred_C = fit(train, test, FEATURES_TEMPORAL + FEATURES_NETWORK + FEATURES_IMD,
                        "C_imd_augmented")
    print(f"  R²_trips = {ref_C['r2_trips']:+.4f}    ΔR² = {ref_C['r2_trips']-ref_A['r2_trips']:+.4f}")
    results["reference"] = {"A": ref_A, "C": ref_C,
                            "delta_r2_trips": ref_C["r2_trips"] - ref_A["r2_trips"]}

    # === Study 1: T-axis ablation ===
    # Refit C with T-axis features zeroed (kept as columns, valued 0 for all rows).
    # This forces the model to predict using only M, I, D axes + socio-econ.
    print("\n=== Study 1 — T-axis sensitivity ===")
    train_T0 = train.copy(); test_T0 = test.copy()
    for c in FEATURES_T:
        train_T0[c] = 0.0; test_T0[c] = 0.0
    s1_C, pred_C_T0 = fit(train_T0, test_T0,
                          FEATURES_TEMPORAL + FEATURES_NETWORK + FEATURES_IMD,
                          "C_T_zeroed")
    print(f"  R²_trips (T=0) = {s1_C['r2_trips']:+.4f}    "
          f"loss vs full C = {ref_C['r2_trips']-s1_C['r2_trips']:+.4f} "
          f"({100*(ref_C['r2_trips']-s1_C['r2_trips'])/ref_C['r2_trips']:+.1f} %)")
    results["study_1_T_zero"] = {
        "C_T_zeroed": s1_C,
        "loss_r2_trips": ref_C["r2_trips"] - s1_C["r2_trips"],
        "loss_pct_relative": 100 * (ref_C["r2_trips"] - s1_C["r2_trips"]) / ref_C["r2_trips"],
    }

    # === Study 2: network-state ablation ===
    print("\n=== Study 2 — Network-state ablation (drops 3 autoregressive features) ===")
    print("Fitting A' (temporal+weather only, NO network state) ...")
    s2_A, _ = fit(train, test, FEATURES_TEMPORAL, "A_minus_network")
    print(f"  R²_trips = {s2_A['r2_trips']:+.4f}")
    print("Fitting C' (temporal+weather+IMD, NO network state) ...")
    s2_C, pred_C2 = fit(train, test, FEATURES_TEMPORAL + FEATURES_IMD,
                        "C_minus_network")
    print(f"  R²_trips = {s2_C['r2_trips']:+.4f}    "
          f"ΔR² (clean) = {s2_C['r2_trips']-s2_A['r2_trips']:+.4f}")
    results["study_2_no_network"] = {
        "A_minus_network": s2_A, "C_minus_network": s2_C,
        "delta_r2_trips_clean": s2_C["r2_trips"] - s2_A["r2_trips"],
    }

    # === Bootstrap CIs (paired weekly blocks on the test set) ===
    print("\n=== Block bootstrap (paired weekly blocks) ===")
    y_test_log = test["log_demand"].values
    times = test["datetime"]
    ci_rows = []
    for name, pred in [("ref_A", pred_A), ("ref_C", pred_C),
                       ("C_T_zeroed", pred_C_T0),
                       ("C_minus_network", pred_C2)]:
        pt, lo, hi = block_ci(y_test_log, pred, times, rng)
        print(f"  {name:20s}  R²={pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
        ci_rows.append({"model": name, "r2": pt, "r2_lo": lo, "r2_hi": hi})
    pd.DataFrame(ci_rows).to_csv(OUT_DIR / "d12_ablations_ci.csv", index=False)
    print(f"\n✓ Wrote {OUT_DIR/'d12_ablations_ci.csv'}")

    with open(OUT_DIR / "d12_ablations.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Wrote {OUT_DIR/'d12_ablations.json'}")


if __name__ == "__main__":
    main()
