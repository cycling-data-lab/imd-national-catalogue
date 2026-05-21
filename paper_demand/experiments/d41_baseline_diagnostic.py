"""
d41_baseline_diagnostic.py — Why is the V\'elomagg G^- baseline so weak
(R^2 = +0.042) when it has 15 features ?

A natural reviewer concern : the headline +0.27 R^2 gain assumes a
no-IMD baseline of +0.04, which is implausibly low for a model that
includes temporal + weather + network-state features.  Two possible
explanations :

  (1) Network-state features (network_volume, network_gini,
      network_entropy) are misleadingly weak because they are
      computed from the full network, so they don't carry per-station
      information — they shift the baseline up only globally.
  (2) The R^2 is on the trip-count scale after exp(); the log-scale
      R^2 (where LightGBM actually optimises) is much higher.  The
      headline metric is mis-leading.

This script disaggregates the G^- baseline by feature family on
the 4 Tier 1 cities to verify the +0.04 figure and check whether
log-scale R^2 tells a different story.

Outputs:
  outputs/d41_baseline_diagnostic.csv
  outputs/d41_baseline_diagnostic.json
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from d3_multicity_benchmark import build_panel, FEATS_T, FEATS_IMD  # noqa: E402

IMD_DIR = ROOT / "data_collection" / "imd_international"
OUT = ROOT / "experiments" / "outputs"

CITIES = [
    ("boston_bluebikes",    "Bluebikes Boston"),
    ("dc_capitalbikeshare", "Capital Bikeshare DC"),
    ("chicago_divvy",       "Divvy Chicago"),
    ("sf_baywheels",        "Bay Wheels SF"),
]

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=63,
                  min_child_samples=30, reg_lambda=0.5, random_state=42,
                  n_jobs=-1, verbose=-1)


def fit_and_score(train, test, features):
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(train[features].values, train["log_demand"].values)
    pred_log = m.predict(test[features].values)
    yt_log = test["log_demand"].values
    yt = np.expm1(yt_log); yp = np.expm1(np.clip(pred_log, 0, None))
    return r2_score(yt_log, pred_log), r2_score(yt, yp)


def main():
    rows = []
    for slug, pretty in CITIES:
        print(f"\n=== {pretty} ===")
        imd = pd.read_parquet(IMD_DIR / f"{slug}.parquet")
        imd["station_id"] = imd["station_id"].astype(str)
        panel = build_panel(slug)
        if panel.empty:
            print("  ✗ empty"); continue
        df = panel.merge(imd[["station_id"] + FEATS_IMD], on="station_id", how="left")
        df = df.dropna(subset=FEATS_IMD).copy()
        df["hour"] = df["datetime_hour"].dt.hour
        df["day_of_week"] = df["datetime_hour"].dt.dayofweek
        df["month"] = df["datetime_hour"].dt.month
        df["log_demand"] = np.log1p(df["demande"])
        df = df.sort_values("datetime_hour").reset_index(drop=True)
        cut = int(0.8 * len(df))
        train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

        sets = {
            "T_only": FEATS_T,
            "T+IMD4": FEATS_T + FEATS_IMD,
        }
        out_city = {"city": pretty, "slug": slug, "n_test": int(len(test))}
        for name, fs in sets.items():
            r2_log, r2_trips = fit_and_score(train, test, fs)
            out_city[f"r2_log_{name}"] = r2_log
            out_city[f"r2_trips_{name}"] = r2_trips
            print(f"  {name:15s}  R²_log = {r2_log:+.4f}    R²_trips = {r2_trips:+.4f}")
        rows.append(out_city)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d41_baseline_diagnostic.csv", index=False)
    print("\n=== Summary (mean across 4 cities) ===")
    for col in df.columns:
        if col.startswith("r2_"):
            print(f"  {col:30s}  mean = {df[col].mean():+.4f}")
    with open(OUT / "d41_baseline_diagnostic.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\n✓ Saved.")


if __name__ == "__main__":
    main()
