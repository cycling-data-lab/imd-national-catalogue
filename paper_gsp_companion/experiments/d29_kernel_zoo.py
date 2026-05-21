"""
d29_kernel_zoo.py — Comparison of four graph-kernel families for the
infill (LSO) prediction task : regularised Laplacian, heat (diffusion)
kernel, random walk, and Matérn-on-graph.

For every kernel K(λ) = Σ_k g(λ_k) u_k u_k^T defined via the Laplacian
eigenbasis L = U Λ U^T :

  reg_Lap     g(λ) = 1 / (λ + ε)
  heat        g(λ) = exp(-t λ)
  random_walk g(λ) = 1 / (1 - β λ)      (β < 1/λ_max for PD)
  matern      g(λ) = 1 / (ν + λ)^α

we form the kernel ridge regression predictor

  ŷ_{s*} = k_*^T (K_TT + σ² I)^{-1} y_T

at each held-out station s*, with kernel hyperparameters and noise σ²
tuned on an internal validation split.

Output:
  outputs/d29_kernel_zoo.csv
  outputs/d29_kernel_zoo.json
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
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
    return Lsym, eigvals, eigvecs


def kernel_from_spectrum(eigvals, eigvecs, g):
    """Build K = U diag(g(λ_k)) U^T."""
    return (eigvecs * g(eigvals)[None, :]) @ eigvecs.T


def krr_predict(K, train_mask, y_z, sigma2):
    """Kernel ridge regression : f_test = K_{test, train} (K_train + σ² I)^{-1} y_train"""
    N = len(y_z)
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(~train_mask)[0]
    K_TT = K[np.ix_(train_idx, train_idx)] + sigma2 * np.eye(len(train_idx))
    alpha = np.linalg.solve(K_TT, y_z[train_idx])
    K_test = K[np.ix_(test_idx, train_idx)]
    f_test = K_test @ alpha
    f_full = np.zeros(N)
    f_full[test_idx] = f_test
    f_full[train_idx] = y_z[train_idx]  # observed
    return f_full


KERNELS = {
    "reg_Lap":     {"params": [("eps", [0.05, 0.1, 0.5, 1.0])],
                     "g": lambda eigvals, p: 1.0 / (eigvals + p["eps"])},
    "heat":        {"params": [("t", [0.5, 1.0, 2.0, 5.0])],
                     "g": lambda eigvals, p: np.exp(-p["t"] * eigvals)},
    # Random-walk diffusion kernel : g(λ) = 1/(1+βλ).  The textbook form
    # g(λ) = 1/(1-βλ) amplifies high frequencies and is not a smoother ;
    # this corrected sign gives a proper low-pass operator on G.
    "random_walk": {"params": [("beta", [0.2, 0.5, 1.0, 2.0, 5.0])],
                     "g": lambda eigvals, p: 1.0 / (1.0 + p["beta"] * eigvals)},
    "matern":      {"params": [("nu", [0.3, 0.5, 1.0]), ("alpha", [1.0, 2.0])],
                     "g": lambda eigvals, p: 1.0 / (p["nu"] + eigvals) ** p["alpha"]},
}
SIGMA2_GRID = [0.05, 0.1, 0.5, 1.0]


def grid_search(K_builder, params_grid, sigma2_grid, eigvals, eigvecs,
                train_mask, inner_val_idx, y_z):
    """Find (params, σ²) minimising MSE on inner_val_idx, training on the
    remaining training stations."""
    best, best_score = None, np.inf
    inner_train_mask = train_mask.copy()
    inner_train_mask[inner_val_idx] = False
    for params in params_grid:
        try:
            K = kernel_from_spectrum(eigvals, eigvecs,
                                     lambda l, p=params: K_builder(l, p))
            for s2 in sigma2_grid:
                f_hat = krr_predict(K, inner_train_mask, y_z, s2)
                err = np.mean((f_hat[inner_val_idx] - y_z[inner_val_idx]) ** 2)
                if err < best_score:
                    best_score, best = err, (params, s2)
        except Exception:
            continue
    return best


def expand_grid(param_specs):
    """[("a", [1,2]), ("b", [3,4])] -> [{a:1,b:3}, {a:1,b:4}, ...]"""
    if not param_specs: return [dict()]
    name, values = param_specs[0]
    rest = expand_grid(param_specs[1:])
    return [{**{name: v}, **r} for v in values for r in rest]


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
        sub = imd.dropna(subset=["y", "lat", "lng"]).reset_index(drop=True)
        N = len(sub)
        print(f"  N = {N}  building eigendecomposition...")
        _, evals, evecs = build_graph(sub["lat"].values, sub["lng"].values)
        y = sub["y"].values.astype(float)
        y_z = (y - y.mean()) / (y.std() + 1e-12)

        perm = rng.permutation(N)
        folds = np.array_split(perm, N_FOLDS)
        kernel_preds = {name: np.zeros(N) for name in KERNELS}
        kernel_best = {name: [] for name in KERNELS}

        for fi, fold in enumerate(folds):
            train_mask = np.ones(N, dtype=bool); train_mask[fold] = False
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(~train_mask)[0]
            inner_perm = rng.permutation(train_idx)
            inner_val = inner_perm[:len(inner_perm)//5]

            for name, spec in KERNELS.items():
                grid = expand_grid(spec["params"])
                best = grid_search(spec["g"], grid, SIGMA2_GRID,
                                   evals, evecs, train_mask, inner_val, y_z)
                if best is None: continue
                params, sigma2 = best
                kernel_best[name].append({"fold": fi+1, "params": params, "sigma2": sigma2})
                K = kernel_from_spectrum(evals, evecs,
                                         lambda l, p=params: spec["g"](l, p))
                f_hat = krr_predict(K, train_mask, y_z, sigma2)
                kernel_preds[name][test_idx] = f_hat[test_idx]

        city_row = {"city": pretty, "slug": slug, "N": N}
        for name in KERNELS:
            rho, _ = spearmanr(y_z, kernel_preds[name])
            city_row[f"rho_{name}"] = float(rho)
            print(f"  {name:12s}  ρ = {rho:+.4f}")
        rows.append(city_row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d29_kernel_zoo.csv", index=False)
    print("\n=== Summary ===")
    print(df.to_string(index=False))

    # Find the best kernel per city
    for _, r in df.iterrows():
        best_kernel = max(KERNELS, key=lambda k: r[f"rho_{k}"])
        print(f"  {r['city']:25s}  best = {best_kernel}  ρ = {r[f'rho_{best_kernel}']:+.3f}")

    with open(OUT / "d29_kernel_zoo.json", "w") as f:
        json.dump({"cities": rows, "wall_time_s": round(time.time()-t0, 1)},
                  f, indent=2)
    print(f"\n✓ Saved.  Total wall time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
