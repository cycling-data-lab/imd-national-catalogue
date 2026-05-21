"""
d36_nonlinear_R2_spectral.py — Non-linear upper bound on what the IMD-4
axes can explain about demand.

The linear R²_spectral of Table tab:gsp uses the orthogonal projection of
y onto the 4-D IMD subspace.  This is the upper bound for any LINEAR
predictor of station-mean demand using only the four IMD axes.

A reviewer would observe : LightGBM is non-linear ; the linear upper
bound is overly pessimistic.  We complement Table tab:gsp by computing
the in-sample R² achievable by a LightGBM regressor on the four IMD
axes alone (no temporal features, no eigenvectors), which gives a
non-linear ceiling on the same quantity.

For each city :
  - Train LightGBM on the 4 IMD axes mapped to mean station demand
  - Report in-sample R² on the trip-count scale
  - Compare to linear R²_spectral (Table tab:gsp)
  - Δ = improvement of non-linear over linear

Output:
  outputs/d36_nonlinear_R2.csv
  outputs/d36_nonlinear_R2.json
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
IMD = ROOT / "data_collection" / "imd_international"
OUT = ROOT / "experiments" / "outputs"

CITIES = [
    ("boston_bluebikes",    "boston_bluebikes",         "Bluebikes Boston"),
    ("dc_capitalbikeshare", "dc_capitalbikeshare",      "Capital Bikeshare DC"),
    ("chicago_divvy",       "chicago_divvy",            "Divvy Chicago"),
    ("sf_baywheels",        "sf_baywheels",             "Bay Wheels SF"),
    ("montreal_bixi",       "world_ca_bixi_montr_al",   "BIXI Montréal"),
    ("london_tfl",          "london_tfl",               "Santander Cycles London"),
    ("paris",               "world_fr_v_lib_metropole", "Vélib Paris"),
    ("lyon",                "world_fr_v_lo_v",          "Vélo'v Lyon"),
    ("toulouse",            "world_fr_v_l_toulouse",    "VéLÔ Toulouse"),
]

FEATS_4AXES = ["gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
               "elevation_m", "topography_roughness_index"]


def load_demand(slug):
    candidates = [
        OUT / f"d3_{slug}_predictions.parquet",
        OUT / f"d14_{slug}_predictions.parquet",
        OUT / f"d16_{slug}_predictions.parquet",
        OUT / f"d10_{slug}_predictions.parquet",
    ]
    paths = [p for p in candidates if p.exists()]
    # Also try the Tier 2 Paris/lyon/toulouse format
    if not paths:
        capitalised = {"paris": "Paris", "lyon": "lyon", "toulouse": "toulouse"}
        if slug in capitalised:
            p = OUT / f"d10_{capitalised[slug]}_predictions.parquet"
            if p.exists(): paths = [p]
    if not paths: return {}
    df = pd.read_parquet(paths[0])
    df["station_id"] = df["station_id"].astype(str)
    df["y_true"] = np.expm1(df["y_true_log"])
    return df.groupby("station_id")["y_true"].mean().to_dict()


def main():
    rows = []
    for slug, stem, pretty in CITIES:
        imd_path = IMD / f"{stem}.parquet"
        if not imd_path.exists():
            continue
        imd = pd.read_parquet(imd_path)
        imd["station_id"] = imd["station_id"].astype(str)
        if slug == "london_tfl":
            imd["station_id"] = imd["station_id"].str.zfill(6)
        y_map = load_demand(slug)
        if not y_map: continue
        imd["y"] = imd["station_id"].map(y_map)
        avail = [f for f in FEATS_4AXES if f in imd.columns]
        if len(avail) < 3: continue
        sub = imd.dropna(subset=["y"] + avail).reset_index(drop=True)
        N = len(sub)
        if N < 50: continue
        X = sub[avail].astype(float).values
        y = sub["y"].values
        # Standardise (matches the §8 GSP setup)
        y_z = (y - y.mean()) / (y.std() + 1e-12)
        # Linear projection on IMD subspace
        Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        try:
            Q, _ = np.linalg.qr(Xs)
            proj_y = Q @ (Q.T @ y_z)
            R2_linear = float((proj_y ** 2).sum() / (y_z @ y_z))
        except Exception:
            R2_linear = float("nan")
        # Non-linear via LightGBM (in-sample R²)
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                              num_leaves=31, min_child_samples=5,
                              random_state=42, verbose=-1)
        m.fit(X, y_z)
        pred = m.predict(X)
        R2_nonlinear_in = float(r2_score(y_z, pred))
        # K-fold (5) out-of-sample non-linear
        rng = np.random.default_rng(42)
        perm = rng.permutation(N); folds = np.array_split(perm, 5)
        preds = np.zeros(N)
        for fold in folds:
            tr = np.setdiff1d(np.arange(N), fold)
            mf = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                   num_leaves=31, min_child_samples=5,
                                   random_state=42, verbose=-1)
            mf.fit(X[tr], y_z[tr])
            preds[fold] = mf.predict(X[fold])
        R2_nonlinear_cv = float(r2_score(y_z, preds))
        rows.append({"city": pretty, "slug": slug, "N": N, "n_axes": len(avail),
                     "R2_linear_subspace": R2_linear,
                     "R2_nonlinear_insample": R2_nonlinear_in,
                     "R2_nonlinear_5fold_cv": R2_nonlinear_cv})
        print(f"  {pretty:25s}  N={N:>4d}  "
              f"R²_lin = {R2_linear:.3f}  "
              f"R²_NL(in) = {R2_nonlinear_in:.3f}  "
              f"R²_NL(5CV) = {R2_nonlinear_cv:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d36_nonlinear_R2.csv", index=False)
    print("\n=== Summary ===")
    print(df.to_string(index=False))
    print(f"\nMean R²_linear        : {df['R2_linear_subspace'].mean():.3f}")
    print(f"Mean R²_nonlinear (in): {df['R2_nonlinear_insample'].mean():.3f}")
    print(f"Mean R²_nonlinear (CV): {df['R2_nonlinear_5fold_cv'].mean():.3f}")
    print(f"Gap NL(in) − linear   : {(df['R2_nonlinear_insample'] - df['R2_linear_subspace']).mean():+.3f}")
    print(f"Gap NL(CV) − linear   : {(df['R2_nonlinear_5fold_cv'] - df['R2_linear_subspace']).mean():+.3f}")

    with open(OUT / "d36_nonlinear_R2.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\n✓ Saved.")


if __name__ == "__main__":
    main()
