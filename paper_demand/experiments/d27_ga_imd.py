"""
d27_ga_imd.py — Graph-Augmented IMD-4 (GA-IMD) : a hybrid predictor that
concatenates the 4-axis IMD-4 vector with the top-K low-frequency
Laplacian eigenvectors of the station-proximity graph, and feeds the
augmented feature vector to LightGBM in leave-station-out evaluation.

Hypothesis tested : the combined representation captures both
(a) the cold-start interpretability of the IMD-4 (physical features
    computable on any commune before deployment), and
(b) the spatial smoothness extracted by graph-Laplacian regularisation
    (Section sec:gsp-laplacian).

If GA-IMD strictly dominates both IMD-only LightGBM and the
Laplacian smoother on the LSO Spearman ρ, then the combined estimator
is a new operational tool that supersedes both individually.

Output:
  outputs/d27_ga_imd.json
  outputs/d27_ga_imd.csv
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
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
]

K_NN = 6
SIGMA_M = 300.0
EARTH_R = 6_371_000.0
N_FOLDS = 5
SEED = 42
K_EIGS = [10, 20, 50]      # candidate widths for the eigenvector basis
LAMBDAS = [0.1, 0.5, 1.0, 2.0, 5.0]
FEATS_IMD = ["gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
             "elevation_m", "topography_roughness_index",
             "n_stations_within_500m", "n_stations_within_1km",
             "catchment_density_per_km2"]


def haversine_matrix(lat, lng):
    lat_r = np.deg2rad(lat); lng_r = np.deg2rad(lng)
    dphi = lat_r[:, None] - lat_r[None, :]
    dlam = lng_r[:, None] - lng_r[None, :]
    a = np.sin(dphi/2)**2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlam/2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_graph(lat, lng):
    N = len(lat); D = haversine_matrix(lat, lng); np.fill_diagonal(D, np.inf)
    knn = np.argpartition(D, K_NN, axis=1)[:, :K_NN]
    W = np.zeros((N, N))
    for i in range(N):
        for j in knn[i]:
            w = np.exp(-D[i, j]**2 / (2*SIGMA_M**2))
            W[i, j] = max(W[i, j], w); W[j, i] = W[i, j]
    deg = W.sum(axis=1); deg_safe = np.maximum(deg, 1e-12)
    Dinv2 = 1.0/np.sqrt(deg_safe)
    Lsym = np.eye(N) - (W*Dinv2[:, None])*Dinv2[None, :]
    L = np.diag(deg) - W
    eigvals, eigvecs = np.linalg.eigh(Lsym)
    return W, deg, L, Lsym, eigvals, eigvecs


def lap_predict(L, y_full, train_mask, lam):
    """Closed-form Laplacian-regularised smoother."""
    N = len(y_full); S = np.diag(train_mask.astype(float))
    A = S + lam * L
    return np.linalg.solve(A, S @ y_full)


def load_demand(slug):
    if slug in ("boston_bluebikes", "dc_capitalbikeshare",
                "chicago_divvy", "sf_baywheels"):
        path = OUT / f"d3_{slug}_predictions.parquet"
    elif slug == "montreal_bixi":
        path = OUT / f"d14_{slug}_predictions.parquet"
    else:
        return {}
    if not path.exists(): return {}
    df = pd.read_parquet(path)
    df["station_id"] = df["station_id"].astype(str)
    df["y_true"] = np.expm1(df["y_true_log"])
    return df.groupby("station_id")["y_true"].mean().to_dict()


def fit_lgb(X_train, y_train, X_test):
    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                          num_leaves=15, min_child_samples=3,
                          n_jobs=1, verbose=-1, random_state=42)
    m.fit(X_train, y_train)
    return m.predict(X_test)


def run_city(slug, imd_stem, pretty, rng):
    print(f"\n=== {pretty} ({slug}) ===")
    imd = pd.read_parquet(IMD / f"{imd_stem}.parquet")
    imd["station_id"] = imd["station_id"].astype(str)
    y_map = load_demand(slug)
    if not y_map: return None
    imd["y"] = imd["station_id"].map(y_map)
    avail = [f for f in FEATS_IMD if f in imd.columns]
    sub = imd.dropna(subset=["y", "lat", "lng"] + avail).reset_index(drop=True)
    N = len(sub)
    if N < 50: return None
    print(f"  N = {N} stations, {len(avail)} IMD features available")

    W, deg, L, Lsym, eigvals, eigvecs = build_graph(
        sub["lat"].values, sub["lng"].values)
    y = sub["y"].values.astype(float)
    y_z = (y - y.mean()) / (y.std() + 1e-12)
    X_imd = sub[avail].astype("float64").values

    perm = rng.permutation(N)
    folds = np.array_split(perm, N_FOLDS)

    # Storage for predictions per strategy
    preds = {
        "IMD_only":   np.zeros(N),
        "Eig50_only": np.zeros(N),
        "GA_IMD_K10": np.zeros(N),
        "GA_IMD_K20": np.zeros(N),
        "GA_IMD_K50": np.zeros(N),
        "Lap":        np.zeros(N),
    }
    best_lams = []
    for fi, fold in enumerate(folds):
        train_mask = np.ones(N, dtype=bool); train_mask[fold] = False
        idx_tr = np.where(train_mask)[0]; idx_te = np.where(~train_mask)[0]

        # IMD-only
        preds["IMD_only"][idx_te] = fit_lgb(X_imd[idx_tr], y_z[idx_tr], X_imd[idx_te])

        # Eigenvectors-only K=50
        U_K = eigvecs[:, :min(50, N-1)]
        preds["Eig50_only"][idx_te] = fit_lgb(U_K[idx_tr], y_z[idx_tr], U_K[idx_te])

        # GA-IMD with K ∈ {10, 20, 50}
        for K in K_EIGS:
            U = eigvecs[:, :min(K, N-1)]
            X_ga = np.hstack([X_imd, U])
            preds[f"GA_IMD_K{K}"][idx_te] = fit_lgb(X_ga[idx_tr], y_z[idx_tr], X_ga[idx_te])

        # Laplacian smoother (tune λ via internal validation)
        inner = np.random.default_rng(SEED + fi).permutation(idx_tr)
        inner_val = inner[:len(inner)//5]
        inner_train_mask = train_mask.copy(); inner_train_mask[inner_val] = False
        best_lam, best_score = LAMBDAS[0], -np.inf
        for lam in LAMBDAS:
            f_hat = lap_predict(L, np.where(inner_train_mask, y_z, 0),
                                inner_train_mask, lam)
            score = -np.mean((f_hat[inner_val] - y_z[inner_val])**2)
            if score > best_score: best_score, best_lam = score, lam
        best_lams.append(best_lam)
        preds["Lap"][idx_te] = lap_predict(L, np.where(train_mask, y_z, 0),
                                           train_mask, best_lam)[idx_te]

    # Aggregate Spearman across the full set
    results = {}
    for name, pred in preds.items():
        rho, _ = spearmanr(y_z, pred)
        results[name] = float(rho)
        print(f"  {name:14s}  ρ = {rho:+.4f}")
    return dict(city=pretty, slug=slug, N=N, results=results,
                best_lams=best_lams)


def main():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    all_res = []
    for slug, stem, pretty in CITIES:
        r = run_city(slug, stem, pretty, rng)
        if r: all_res.append(r)

    rows = []
    for r in all_res:
        row = {"city": r["city"], "N": r["N"]}
        row.update(r["results"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d27_ga_imd.csv", index=False)
    print("\n=== Summary (Spearman ρ on held-out LSO stations) ===")
    print(df.to_string(index=False))

    # Statistical question: is GA-IMD > IMD-only and > Lap across cities?
    for k in K_EIGS:
        col = f"GA_IMD_K{k}"
        diff_vs_imd = df[col] - df["IMD_only"]
        diff_vs_lap = df[col] - df["Lap"]
        diff_vs_eig = df[col] - df["Eig50_only"]
        print(f"\n  GA-IMD K={k} vs IMD_only :  mean Δρ = {diff_vs_imd.mean():+.3f}  (range [{diff_vs_imd.min():+.3f}, {diff_vs_imd.max():+.3f}])")
        print(f"  GA-IMD K={k} vs Lap      :  mean Δρ = {diff_vs_lap.mean():+.3f}  (range [{diff_vs_lap.min():+.3f}, {diff_vs_lap.max():+.3f}])")
        print(f"  GA-IMD K={k} vs Eig50    :  mean Δρ = {diff_vs_eig.mean():+.3f}  (range [{diff_vs_eig.min():+.3f}, {diff_vs_eig.max():+.3f}])")

    with open(OUT / "d27_ga_imd.json", "w") as f:
        json.dump({"cities": all_res, "wall_time_s": round(time.time()-t0, 1)},
                  f, indent=2)
    print(f"\n✓ Saved.  Total wall time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
