"""
d26_optimal_siting.py — D-optimal sampling-theoretic siting algorithm on
the station-proximity graph.

Following Anis, Gadde and Ortega (IEEE TSP 2016), and Chen, Varma,
Sandryhaila and Kovacevic (2015), we recover a K-bandlimited signal on
a graph by sampling at a subset S of nodes such that the
K-eigenvectors-restricted-to-S matrix U_K[S, :] has maximum determinant.
This is the D-optimal experimental design problem on a graph; we solve
it greedily.

Algorithm (D-optimal greedy):
    Input  : graph eigenvectors U_K = [u_1, ..., u_K] (low frequencies),
             budget K_target
    Output : S ⊂ V with |S| = K_target maximising
             det(U_K[S, :]^T U_K[S, :])

    S ← {}
    while |S| < K_target :
        if S empty:
            s* ← argmax_i  ||u_i||^2
        else:
            G ← U_K[S, :]^T U_K[S, :]
            G^+ ← pseudo-inverse
            scores_i ← u_i^T G^+ u_i  (informativeness given S)
            s* ← argmax_{i ∉ S} scores_i
        S ← S ∪ {s*}

We apply this to Boston Bluebikes (N = 497 stations) with K_target = 50,
and compare against three baselines:
  - random : uniform random sample of 50 stations
  - top_IMD : 50 stations with the highest IMD-4 score (French weights)
  - oracle : the 50 stations with the highest observed mean demand

Quality is measured by the reconstruction Spearman ρ between
- the band-limited reconstruction of demand from the 50 chosen stations
- the observed mean demand on the 447 held-out stations.

Output:
  outputs/d26_optimal_siting.json
  outputs/d26_optimal_siting_per_city.csv
  figures/fig_optimal_siting.{pdf,png}
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
IMD_INTL_DIR = ROOT / "data_collection" / "imd_international"
OUT = ROOT / "experiments" / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CITIES = [
    ("boston_bluebikes",    "boston_bluebikes",         "Bluebikes Boston"),
    ("dc_capitalbikeshare", "dc_capitalbikeshare",      "Capital Bikeshare DC"),
    ("chicago_divvy",       "chicago_divvy",            "Divvy Chicago"),
    ("sf_baywheels",        "sf_baywheels",             "Bay Wheels SF"),
]

K_NN = 6
SIGMA_M = 300.0
EARTH_R = 6_371_000.0
SEED = 42
N_EIGVECS = 50          # bandwidth K for the band-limited approximation
N_DEPLOYED = 50         # |S| budget for siting


def haversine_matrix(lat, lng):
    lat_r = np.deg2rad(lat); lng_r = np.deg2rad(lng)
    dphi = lat_r[:, None] - lat_r[None, :]
    dlam = lng_r[:, None] - lng_r[None, :]
    a = np.sin(dphi / 2)**2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlam / 2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_graph(lat, lng):
    N = len(lat)
    D = haversine_matrix(lat, lng); np.fill_diagonal(D, np.inf)
    knn = np.argpartition(D, K_NN, axis=1)[:, :K_NN]
    W = np.zeros((N, N))
    for i in range(N):
        for j in knn[i]:
            w = np.exp(-D[i, j]**2 / (2 * SIGMA_M**2))
            W[i, j] = max(W[i, j], w); W[j, i] = W[i, j]
    deg = W.sum(axis=1)
    deg_safe = np.maximum(deg, 1e-12)
    Dinv2 = 1.0 / np.sqrt(deg_safe)
    Lsym = np.eye(N) - (W * Dinv2[:, None]) * Dinv2[None, :]
    eigvals, eigvecs = np.linalg.eigh(Lsym)
    return W, deg, Lsym, eigvals, eigvecs


def d_optimal_greedy(U_K: np.ndarray, K_target: int) -> list[int]:
    """Greedy D-optimal subset selection on the rows of U_K."""
    N, K = U_K.shape
    S = []
    available = set(range(N))
    for _ in range(min(K_target, N)):
        if not S:
            scores = (U_K ** 2).sum(axis=1)
            for s in S: scores[s] = -np.inf
            best = int(np.argmax(scores))
        else:
            U_S = U_K[S, :]
            G = U_S.T @ U_S + 1e-8 * np.eye(K)
            try:
                G_inv = np.linalg.inv(G)
            except np.linalg.LinAlgError:
                G_inv = np.linalg.pinv(G)
            scores = np.einsum("ik,kl,il->i", U_K, G_inv, U_K)
            for s in S: scores[s] = -np.inf
            best = int(np.argmax(scores))
        S.append(best); available.discard(best)
    return S


def band_limited_reconstruct(y_obs: np.ndarray, S: list[int],
                             U_K: np.ndarray) -> np.ndarray:
    """Reconstruct y on all nodes from observations y_obs at sampling
    set S, projecting onto the bandwidth-K subspace.
    f = U_K · (U_K[S, :])^+ · y_obs
    """
    U_S = U_K[S, :]
    c = np.linalg.pinv(U_S) @ y_obs
    return U_K @ c


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


def evaluate_strategy(S, y_z, y_raw, U_K, label):
    """Evaluate a siting strategy : sample standardised y_z at S,
    reconstruct on held-out V\\S, return Spearman ρ. Fraction of total
    demand is computed on RAW y (not on standardised z-scored y, where
    the global sum is ≈ 0)."""
    y_obs = y_z[S]
    f_hat = band_limited_reconstruct(y_obs, S, U_K)
    held = [i for i in range(len(y_z)) if i not in set(S)]
    rho, _ = spearmanr(y_z[held], f_hat[held])
    frac_demand_at_S = float(y_raw[S].sum() / y_raw.sum())
    return dict(strategy=label,
                rho_heldout=float(rho),
                fraction_demand_at_S=frac_demand_at_S)


def run_city(slug, imd_stem, pretty, rng):
    print(f"\n=== {pretty} ({slug}) ===")
    imd = pd.read_parquet(IMD_INTL_DIR / f"{imd_stem}.parquet")
    imd["station_id"] = imd["station_id"].astype(str)
    y_map = load_demand(slug)
    if not y_map: return None
    imd["y"] = imd["station_id"].map(y_map)
    sub = imd.dropna(subset=["y", "lat", "lng"]).reset_index(drop=True)
    N = len(sub)
    if N < 100: return None
    print(f"  N = {N} stations")

    print(f"  Building k-NN graph + eigendecomposition...")
    W, deg, Lsym, eigvals, eigvecs = build_graph(sub["lat"].values, sub["lng"].values)
    K_eig = min(N_EIGVECS, N - 1)
    U_K = eigvecs[:, :K_eig]
    print(f"    using {K_eig} low-frequency eigenvectors")

    K_dep = min(N_DEPLOYED, N // 2)
    y = sub["y"].values
    y_z = (y - y.mean()) / (y.std() + 1e-12)

    # Strategy 1 : D-optimal greedy
    print(f"  D-optimal greedy with K_target = {K_dep}...")
    t0 = time.time()
    S_opt = d_optimal_greedy(U_K, K_dep)
    t_opt = time.time() - t0

    # Strategy 2 : random
    n_random = 50
    rho_random_list = []
    frac_random_list = []
    for b in range(n_random):
        S_rand = rng.choice(N, size=K_dep, replace=False).tolist()
        r = evaluate_strategy(S_rand, y_z, y, U_K, "random")
        rho_random_list.append(r["rho_heldout"])
        frac_random_list.append(r["fraction_demand_at_S"])
    rho_random_mean = float(np.mean(rho_random_list))
    rho_random_sd = float(np.std(rho_random_list))
    frac_random_mean = float(np.mean(frac_random_list))

    # Strategy 3 : top-IMD (highest mean of 4 standardised axes)
    feats = ["gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
             "elevation_m", "topography_roughness_index"]
    avail = [c for c in feats if c in sub.columns]
    if avail:
        feat_mat = sub[avail].astype(float).values
        feat_mat = (feat_mat - feat_mat.mean(axis=0)) / (feat_mat.std(axis=0) + 1e-12)
        imd_score = feat_mat.sum(axis=1)
        S_imd = list(np.argsort(-imd_score)[:K_dep])
        r_imd = evaluate_strategy(S_imd, y_z, y, U_K, "top_IMD")
    else:
        r_imd = None

    # Strategy 4 : oracle (true top-K by demand)
    S_oracle = list(np.argsort(-y)[:K_dep])
    r_oracle = evaluate_strategy(S_oracle, y_z, y, U_K, "oracle")

    r_opt = evaluate_strategy(S_opt, y_z, y, U_K, "D_optimal")

    print(f"  D-optimal           : ρ = {r_opt['rho_heldout']:+.3f}   "
          f"fraction of demand at S = {r_opt['fraction_demand_at_S']:.3f}   ({t_opt:.1f}s)")
    print(f"  Random (B={n_random})        : ρ = {rho_random_mean:+.3f} ± {rho_random_sd:.3f}   "
          f"fraction = {frac_random_mean:.3f}")
    if r_imd:
        print(f"  Top-IMD             : ρ = {r_imd['rho_heldout']:+.3f}   "
              f"fraction = {r_imd['fraction_demand_at_S']:.3f}")
    print(f"  Oracle (top-K dem)  : ρ = {r_oracle['rho_heldout']:+.3f}   "
          f"fraction = {r_oracle['fraction_demand_at_S']:.3f}")

    return dict(
        city=pretty, slug=slug, N=N, K_eig=K_eig, K_deployed=K_dep,
        d_optimal=r_opt,
        random=dict(rho_mean=rho_random_mean, rho_sd=rho_random_sd,
                    fraction_mean=frac_random_mean, n_samples=n_random),
        top_imd=r_imd,
        oracle=r_oracle,
        wall_time_s=round(t_opt, 1),
        S_opt=[int(s) for s in S_opt],
        S_oracle=[int(s) for s in S_oracle],
        coords=sub[["lat", "lng"]].values.tolist(),
    )


def main():
    rng = np.random.default_rng(SEED)
    results = []
    for slug, stem, pretty in CITIES:
        r = run_city(slug, stem, pretty, rng)
        if r: results.append(r)

    rows = []
    for r in results:
        rows.append({
            "city": r["city"], "N": r["N"], "K_deployed": r["K_deployed"],
            "rho_d_optimal": r["d_optimal"]["rho_heldout"],
            "rho_random_mean": r["random"]["rho_mean"],
            "rho_random_sd": r["random"]["rho_sd"],
            "rho_top_imd": r["top_imd"]["rho_heldout"] if r["top_imd"] else None,
            "rho_oracle": r["oracle"]["rho_heldout"],
            "frac_d_optimal": r["d_optimal"]["fraction_demand_at_S"],
            "frac_random_mean": r["random"]["fraction_mean"],
            "frac_top_imd": r["top_imd"]["fraction_demand_at_S"] if r["top_imd"] else None,
            "frac_oracle": r["oracle"]["fraction_demand_at_S"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "d26_optimal_siting_per_city.csv", index=False)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    # ── Figure : ρ comparison + Boston map ──────────────────────────────────
    boston = next((r for r in results if r["slug"] == "boston_bluebikes"), None)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    # Panel a : bar chart of ρ_heldout per strategy across cities
    ax = axes[0]
    strats = ["d_optimal", "random", "top_imd", "oracle"]
    colors = {"d_optimal": "C0", "random": "C7", "top_imd": "C2", "oracle": "C3"}
    width = 0.2
    x = np.arange(len(summary))
    for si, s in enumerate(strats):
        col = "rho_" + s if s != "random" else "rho_random_mean"
        vals = summary[col].values
        sds = summary["rho_random_sd"].values if s == "random" else None
        ax.bar(x + (si - 1.5) * width, vals, width=width, label=s,
               color=colors[s], yerr=sds, capsize=2 if sds is not None else None)
    ax.set_xticks(x); ax.set_xticklabels(summary["city"], rotation=20, fontsize=8, ha="right")
    ax.set_ylabel(r"Held-out Spearman $\rho$")
    ax.set_title("Siting-strategy comparison on the band-limited reconstruction task")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    # Panel b : Boston map
    if boston:
        ax = axes[1]
        coords = np.array(boston["coords"])
        ax.scatter(coords[:, 1], coords[:, 0], s=8, c="lightgray",
                   label="All stations", alpha=0.6)
        S = boston["S_opt"]
        ax.scatter(coords[S, 1], coords[S, 0], s=24, c="C0",
                   marker="o", edgecolor="black", linewidth=0.5,
                   label=f"D-optimal ($K={len(S)}$)", zorder=3)
        S = boston["S_oracle"]
        ax.scatter(coords[S, 1], coords[S, 0], s=14, c="C3",
                   marker="^", edgecolor="black", linewidth=0.5,
                   label=f"Oracle top-{len(S)}", zorder=4)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title("D-optimal vs oracle siting on Bluebikes Boston (N = 497)")
        ax.legend(fontsize=8, loc="best")
        ax.set_aspect("equal")
    fig.savefig(FIG / "fig_optimal_siting.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_optimal_siting.png", bbox_inches="tight", dpi=200)
    print(f"\n✓ Wrote {FIG/'fig_optimal_siting.pdf'}")

    with open(OUT / "d26_optimal_siting.json", "w") as f:
        json.dump({"cities": [{k: v for k, v in r.items() if k not in ("coords",)}
                              for r in results]}, f, indent=2)
    print(f"✓ Saved {OUT/'d26_optimal_siting.json'}")


if __name__ == "__main__":
    main()
