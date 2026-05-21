"""
d31_heat_kernel_signatures.py — Heat Kernel Signature (HKS) features
(Sun, Ovsjanikov, Guibas, EG 2009) as a cross-city-transferable spectral
fingerprint of each station.

HKS at node v for diffusion time t :

    HKS(v, t) = Σ_k exp(-λ_k t) · u_k(v)²

Intrinsic on the graph : invariant under graph isomorphism.  At each
node v we evaluate HKS at a logarithmic time grid t ∈ {t₁, …, t_T} and
obtain a T-dimensional intrinsic descriptor.  This descriptor is
\emph{comparable across cities} (each city's HKS lives in the same
T-dimensional space, with the same meaning per coordinate), making it
a natural complement to the city-specific IMD-4 axes for cross-network
transfer learning.

We test three feature sets on the LSO benchmark of Section sec:lso :
  - IMD-only  (baseline of Table tab:lso)
  - HKS-only  (no physical features, just intrinsic spectral)
  - IMD + HKS (concatenation)

Output :
  outputs/d31_hks.csv
  outputs/d31_hks.json
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

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
HKS_TIMES = np.logspace(-2, 1.5, 10)   # 10 time scales : 0.01 to ~30
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
    eigvals, eigvecs = np.linalg.eigh(Lsym)
    return eigvals, eigvecs


def compute_hks(eigvals, eigvecs, times):
    """HKS(v, t) = Σ_k exp(-λ_k t) u_k(v)².
    Returns (N, T) matrix."""
    N, _ = eigvecs.shape
    T = len(times)
    hks = np.zeros((N, T))
    for ti, t in enumerate(times):
        weights = np.exp(-eigvals * t)
        hks[:, ti] = (eigvecs ** 2 * weights[None, :]).sum(axis=1)
    return hks


def fit_lgb(X_train, y_train, X_test):
    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                          num_leaves=15, min_child_samples=3,
                          n_jobs=1, verbose=-1, random_state=42)
    m.fit(X_train, y_train)
    return m.predict(X_test)


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


def main():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    rows = []
    for slug, stem, pretty in CITIES:
        print(f"\n=== {pretty} ({slug}) ===")
        imd = pd.read_parquet(IMD / f"{stem}.parquet")
        imd["station_id"] = imd["station_id"].astype(str)
        y_map = load_demand(slug)
        if not y_map: continue
        imd["y"] = imd["station_id"].map(y_map)
        avail = [f for f in FEATS_IMD if f in imd.columns]
        sub = imd.dropna(subset=["y", "lat", "lng"] + avail).reset_index(drop=True)
        N = len(sub)
        print(f"  N = {N}")

        evals, evecs = build_graph(sub["lat"].values, sub["lng"].values)
        hks = compute_hks(evals, evecs, HKS_TIMES)
        # Standardise HKS (log-scale, since values vary by orders of magnitude)
        hks_log = np.log(np.maximum(hks, 1e-12))
        hks_std = (hks_log - hks_log.mean(axis=0)) / (hks_log.std(axis=0) + 1e-12)
        print(f"  HKS shape : {hks_std.shape}")

        y = sub["y"].values.astype(float)
        y_z = (y - y.mean()) / (y.std() + 1e-12)
        X_imd = sub[avail].astype("float64").values

        perm = rng.permutation(N)
        folds = np.array_split(perm, N_FOLDS)
        preds = {"imd_only": np.zeros(N), "hks_only": np.zeros(N),
                 "imd_hks": np.zeros(N)}
        for fi, fold in enumerate(folds):
            tr = np.setdiff1d(np.arange(N), fold)
            te = fold
            preds["imd_only"][te] = fit_lgb(X_imd[tr], y_z[tr], X_imd[te])
            preds["hks_only"][te] = fit_lgb(hks_std[tr], y_z[tr], hks_std[te])
            X_combined = np.hstack([X_imd, hks_std])
            preds["imd_hks"][te] = fit_lgb(X_combined[tr], y_z[tr], X_combined[te])

        results = {}
        for name, p in preds.items():
            rho, _ = spearmanr(y_z, p)
            results[f"rho_{name}"] = float(rho)
            print(f"  {name:10s}  ρ = {rho:+.4f}")
        results.update({"city": pretty, "slug": slug, "N": N})
        rows.append(results)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d31_hks.csv", index=False)
    print("\n=== Summary ===")
    print(df.to_string(index=False))
    delta_imd = df["rho_imd_hks"] - df["rho_imd_only"]
    delta_hks = df["rho_imd_hks"] - df["rho_hks_only"]
    print(f"\nMean Δρ (IMD+HKS vs IMD-only) : {delta_imd.mean():+.3f}")
    print(f"Mean Δρ (IMD+HKS vs HKS-only) : {delta_hks.mean():+.3f}")

    with open(OUT / "d31_hks.json", "w") as f:
        json.dump({"cities": rows,
                   "hks_times": list(HKS_TIMES),
                   "wall_time_s": round(time.time()-t0, 1)},
                  f, indent=2)
    print(f"\n✓ Saved.  Total wall time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
