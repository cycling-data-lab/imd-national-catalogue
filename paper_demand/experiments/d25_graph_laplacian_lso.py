"""
d25_graph_laplacian_lso.py — Graph Laplacian regularised semi-supervised
prediction of station-mean demand on held-out stations.

Tests whether the spatial generalisation of Section LSO is carried by
(i) the four-axis IMD-4 features specifically, (ii) the smoothness of
the demand signal on the station-proximity graph alone, or (iii) the
synergy of both.

The Laplacian-regularised predictor solves, for each LSO fold:

    f_hat = argmin_f  Σ_{i ∈ train} (f_i - y_i)^2  +  λ f^T L f
          = (S_train + λ L)^{-1} (S_train · y_full)

where S_train is the diagonal indicator of training stations, L is the
unnormalised graph Laplacian, and y_full has the training values on
training stations and zero elsewhere (will be overwritten by the
solution).  This is the closed-form GMRF-prior Bayesian smoother
(Zhu et al. 2003, Belkin & Niyogi 2004).

Three predictors per fold:
  G_Lap     : Laplacian-regularised on graph, no IMD features
  G_LightGBM: classical IMD-augmented LightGBM (re-uses d24 setup)
  G_Lap+IMD : Laplacian smoother applied to LightGBM residual (combines)

Output:
  outputs/d25_graph_laplacian_lso.json
  outputs/d25_graph_laplacian_lso_per_station.csv
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from scipy.sparse import csr_matrix, eye as speye
from scipy.sparse.linalg import spsolve
import lightgbm as lgb
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
IMD_INTL_DIR = ROOT / "data_collection" / "imd_international"
OUT = ROOT / "experiments" / "outputs"

# Cities to test — five LSO networks with good R2_spectral signal
CITIES = [
    ("boston_bluebikes",    "boston_bluebikes",         "Bluebikes Boston"),
    ("dc_capitalbikeshare", "dc_capitalbikeshare",      "Capital Bikeshare DC"),
    ("chicago_divvy",       "chicago_divvy",            "Divvy Chicago"),
    ("sf_baywheels",        "sf_baywheels",             "Bay Wheels SF"),
    ("montreal_bixi",       "world_ca_bixi_montr_al",   "BIXI Montréal"),
]

K_NN = 6
SIGMA_M = 300.0
EARTH_R = 6_371_000.0
N_FOLDS = 5
SEED = 42
LAMBDAS = [0.1, 0.5, 1.0, 2.0, 5.0]   # candidate regularisation strengths
FEATS_IMD = [
    "gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
    "elevation_m", "topography_roughness_index",
    "n_stations_within_500m", "n_stations_within_1km",
    "catchment_density_per_km2",
]


def haversine_matrix(lat, lng):
    lat_r = np.deg2rad(lat); lng_r = np.deg2rad(lng)
    dphi = lat_r[:, None] - lat_r[None, :]
    dlam = lng_r[:, None] - lng_r[None, :]
    a = np.sin(dphi / 2)**2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlam / 2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_graph(lat, lng, k=K_NN, sigma=SIGMA_M):
    N = len(lat); D_mat = haversine_matrix(lat, lng)
    np.fill_diagonal(D_mat, np.inf)
    knn = np.argpartition(D_mat, k, axis=1)[:, :k]
    W = np.zeros((N, N))
    for i in range(N):
        for j in knn[i]:
            w = np.exp(-D_mat[i, j]**2 / (2 * sigma**2))
            W[i, j] = max(W[i, j], w); W[j, i] = W[i, j]
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    return W, deg, L


def lap_reg_predict(L: np.ndarray, y_full: np.ndarray, train_mask: np.ndarray,
                    lam: float) -> np.ndarray:
    """Closed-form Laplacian-regularised smoother. Returns f_hat for all
    nodes; f_hat[i] = y[i] for i in train (approximately) and an
    interpolated value for i in test."""
    N = len(y_full)
    S = np.diag(train_mask.astype(float))
    A = S + lam * L  # (N, N)
    b = S @ y_full
    f_hat = np.linalg.solve(A, b)
    return f_hat


def load_demand(slug: str) -> dict:
    """Return {station_id: mean_demand} for one city."""
    if slug in ("boston_bluebikes", "dc_capitalbikeshare",
                 "chicago_divvy", "sf_baywheels"):
        pred_path = OUT / f"d3_{slug}_predictions.parquet"
    elif slug == "montreal_bixi":
        pred_path = OUT / f"d14_{slug}_predictions.parquet"
    else:
        return {}
    if not pred_path.exists():
        return {}
    df = pd.read_parquet(pred_path)
    df["station_id"] = df["station_id"].astype(str)
    df["y_true"] = np.expm1(df["y_true_log"])
    return df.groupby("station_id")["y_true"].mean().to_dict()


def run_city(slug, imd_stem, pretty, rng):
    print(f"\n=== {pretty} ({slug}) ===")
    imd = pd.read_parquet(IMD_INTL_DIR / f"{imd_stem}.parquet")
    imd["station_id"] = imd["station_id"].astype(str)
    y_map = load_demand(slug)
    if not y_map:
        print("  ✗ no demand data"); return None
    imd["y"] = imd["station_id"].map(y_map)
    sub = imd.dropna(subset=["y", "lat", "lng"] + FEATS_IMD).reset_index(drop=True)
    if len(sub) < 50:
        print(f"  ✗ only {len(sub)} stations matched"); return None
    N = len(sub)
    print(f"  N = {N} stations with demand + IMD")

    print(f"  Building k-NN graph (k={K_NN}, σ={SIGMA_M}m)...")
    W, deg, L = build_graph(sub["lat"].values, sub["lng"].values)
    print(f"    mean degree = {deg.mean():.1f}")

    y = sub["y"].values.astype(float)
    y_z = (y - y.mean()) / (y.std() + 1e-12)

    # 5-fold station LSO
    perm = rng.permutation(N)
    folds = np.array_split(perm, N_FOLDS)

    rows = []
    for fi, fold in enumerate(folds):
        train_mask = np.ones(N, dtype=bool); train_mask[fold] = False
        # Tune λ on a small inner split of the training set
        inner = np.where(train_mask)[0]
        rng.shuffle(inner)
        inner_val = inner[:len(inner)//5]
        inner_train_mask = train_mask.copy()
        inner_train_mask[inner_val] = False
        best_lam, best_score = LAMBDAS[0], -np.inf
        for lam in LAMBDAS:
            f_hat = lap_reg_predict(L, np.where(inner_train_mask, y_z, 0),
                                    inner_train_mask, lam)
            score = -np.mean((f_hat[inner_val] - y_z[inner_val])**2)
            if score > best_score:
                best_score, best_lam = score, lam

        # Final fit with tuned λ
        f_hat = lap_reg_predict(L, np.where(train_mask, y_z, 0), train_mask, best_lam)
        # G_LightGBM: fit on IMD with this fold's split
        X = sub[FEATS_IMD].astype("float64").values
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                              num_leaves=15, min_child_samples=3,
                              n_jobs=1, verbose=-1, random_state=42)
        m.fit(X[train_mask], y_z[train_mask])
        f_lgbm = m.predict(X)
        # G_Lap+IMD: Laplacian-smooth the LightGBM prediction residual on
        # train; for prediction on held-out the LightGBM IMD value is the
        # primary signal augmented by Laplacian smoothing
        resid_train = np.where(train_mask, y_z - f_lgbm, 0)
        f_res = lap_reg_predict(L, resid_train, train_mask, best_lam)
        f_combined = f_lgbm + f_res

        for idx in fold:
            rows.append({"city": pretty, "station_idx": int(idx),
                          "fold": fi+1, "lambda": best_lam,
                          "y_true": float(y_z[idx]),
                          "f_lap": float(f_hat[idx]),
                          "f_lgbm": float(f_lgbm[idx]),
                          "f_combined": float(f_combined[idx])})

    df = pd.DataFrame(rows)
    rho_lap,      _ = spearmanr(df["y_true"], df["f_lap"])
    rho_lgbm,     _ = spearmanr(df["y_true"], df["f_lgbm"])
    rho_combined, _ = spearmanr(df["y_true"], df["f_combined"])
    print(f"  ρ_G_Lap (graph only) = {rho_lap:+.3f}")
    print(f"  ρ_G_LightGBM (IMD only) = {rho_lgbm:+.3f}")
    print(f"  ρ_G_Lap+IMD (combined) = {rho_combined:+.3f}")
    return dict(city=pretty, slug=slug, N=N,
                rho_lap=float(rho_lap),
                rho_lgbm=float(rho_lgbm),
                rho_combined=float(rho_combined),
                per_station=df.to_dict("records"))


def main():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    results = []
    for slug, stem, pretty in CITIES:
        r = run_city(slug, stem, pretty, rng)
        if r: results.append(r)

    summary = pd.DataFrame([{
        "city": r["city"], "N": r["N"],
        "rho_lap": r["rho_lap"],
        "rho_lgbm": r["rho_lgbm"],
        "rho_combined": r["rho_combined"],
    } for r in results])
    summary.to_csv(OUT / "d25_graph_laplacian_lso.csv", index=False)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    with open(OUT / "d25_graph_laplacian_lso.json", "w") as f:
        json.dump({"cities": [{"city": r["city"], "slug": r["slug"],
                                "N": r["N"],
                                "rho_lap": r["rho_lap"],
                                "rho_lgbm": r["rho_lgbm"],
                                "rho_combined": r["rho_combined"]}
                              for r in results],
                   "wall_time_s": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\n✓ Saved.  Total wall time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
