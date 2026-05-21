"""
d39_imd1_vs_imd4_ablation.py — Does IMD-1 (M axis alone) match IMD-4?

The Bayesian posterior median puts $w_M = 0.78$ on the
multimodality axis, so a sceptical reviewer will ask : if M
dominates the simplex by an order of magnitude, do the three
other axes (I, T, D) carry any additional predictive value, or
is "IMD-4" a rebranded "transit multimodality" score ?

This script answers that question directly by training the
LightGBM regressor on three nested feature sets, on the same 4-
Tier-1 panel as the K=10 LSO of d35 :

  G^-    : temporal only                  (baseline)
  G_M    : temporal + M-axis only         (IMD-1 simplex collapse)
  G_full : temporal + M, I, T, D          (IMD-4)

On the same temporal hold-out as d3_multicity_benchmark, we report
the R^2 on trip-count scale for the three models per city, the
gap G_full - G_M (i.e.\\ the additional value of I/T/D over M
alone), and the share of the IMD-4 contribution explained by M
alone.

Outputs:
  outputs/d39_imd1_vs_imd4.csv
  outputs/d39_imd1_vs_imd4.json
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

# IMD-1 = M axis only (the posterior-dominant axis at w_M = 0.78)
FEATS_M = ["gtfs_heavy_stops_300m"]

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=63,
                  min_child_samples=30, reg_lambda=0.5, random_state=42,
                  n_jobs=-1, verbose=-1)


def fit_predict(train, test, features):
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(train[features].values, train["log_demand"].values)
    p = m.predict(test[features].values)
    yt = np.expm1(test["log_demand"].values)
    yp = np.expm1(np.clip(p, 0, None))
    return r2_score(yt, yp)


def main():
    rows = []
    t0 = time.time()
    for slug, pretty in CITIES:
        print(f"\n=== {pretty} ({slug}) ===")
        imd = pd.read_parquet(IMD_DIR / f"{slug}.parquet")
        imd["station_id"] = imd["station_id"].astype(str)
        panel = build_panel(slug)
        if panel.empty:
            print("  ✗ empty panel"); continue
        df = panel.merge(imd[["station_id"] + FEATS_IMD], on="station_id", how="left")
        df = df.dropna(subset=FEATS_IMD).copy()
        df["hour"] = df["datetime_hour"].dt.hour
        df["day_of_week"] = df["datetime_hour"].dt.dayofweek
        df["month"] = df["datetime_hour"].dt.month
        df["log_demand"] = np.log1p(df["demande"])
        df = df.sort_values("datetime_hour").reset_index(drop=True)
        cut = int(0.8 * len(df))
        train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
        r2_no = fit_predict(train, test, FEATS_T)
        r2_m  = fit_predict(train, test, FEATS_T + FEATS_M)
        r2_imd = fit_predict(train, test, FEATS_T + FEATS_IMD)
        dr_m_vs_no   = r2_m - r2_no
        dr_imd_vs_no = r2_imd - r2_no
        dr_imd_vs_m  = r2_imd - r2_m
        share_m = (dr_m_vs_no / dr_imd_vs_no) if dr_imd_vs_no > 1e-6 else float("nan")
        rows.append({"city": pretty, "slug": slug,
                     "r2_no_imd": r2_no, "r2_imd1_M": r2_m, "r2_imd4": r2_imd,
                     "delta_M_vs_no": dr_m_vs_no,
                     "delta_imd4_vs_no": dr_imd_vs_no,
                     "delta_imd4_vs_M": dr_imd_vs_m,
                     "share_of_gain_from_M_alone": share_m})
        print(f"  G-       R² = {r2_no:+.4f}")
        print(f"  G_M      R² = {r2_m:+.4f}    ΔR² vs G- = {dr_m_vs_no:+.4f}")
        print(f"  G_full   R² = {r2_imd:+.4f}    ΔR² vs G- = {dr_imd_vs_no:+.4f}")
        print(f"  ΔR²(G_full − G_M) = {dr_imd_vs_m:+.4f}   "
              f"→ M alone explains {share_m*100:.1f}% of the IMD-4 gain")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d39_imd1_vs_imd4.csv", index=False)
    print("\n=== Summary ===")
    print(df.to_string(index=False))
    mean_share = float(df["share_of_gain_from_M_alone"].mean())
    print(f"\nMean share of IMD-4 gain explained by M alone : {mean_share*100:.1f} %")
    print(f"Mean residual ΔR²(IMD-4 − IMD-1) : {df['delta_imd4_vs_M'].mean():+.4f}")
    print(f"Verdict : {'IMD-4 collapses to M' if mean_share > 0.9 else 'I/T/D carry essential residual signal'}")
    with open(OUT / "d39_imd1_vs_imd4.json", "w") as f:
        json.dump({"rows": rows, "mean_share_M_alone": mean_share,
                   "wall_time_s": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\n✓ Saved.")


if __name__ == "__main__":
    main()
