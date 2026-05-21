"""
d33_mclp_comparison.py — Operations-research baselines for the siting
question: MCLP (Maximal Covering Location Problem, Church & ReVelle 1974)
and k-median (Hakimi 1964).  Compared on Boston (N=493 stations, K=50
to deploy) against the GSP-based D-optimal greedy of Section sec:gsp-siting.

MCLP formulation :
    maximise  Σ_i w_i z_i
    subject to  z_i ≤ Σ_{j ∈ N_r(i)} x_j   for all i
                Σ_j x_j ≤ K
                x_j, z_i ∈ {0, 1}

    x_j : binary indicator of deployment at candidate j
    z_i : binary indicator of demand point i covered (within radius r)
    w_i : weight of demand point i (= demand value y_i)
    N_r(i) : candidates within radius r of i

k-median (continuous-relaxation) :
    minimise  Σ_i  w_i · min_{j ∈ S} d(i, j)
    over     S ⊂ V, |S| = K

We solve both with PuLP + CBC and compare to the D-optimal greedy on
two criteria : (i) held-out band-limited reconstruction Spearman ρ
(our metric), (ii) total demand captured at S (the OR-classical objective).

Output:
  outputs/d33_mclp_comparison.json
  outputs/d33_mclp_comparison.csv
  figures/fig_mclp_comparison.{pdf,png}
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pulp

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
IMD = ROOT / "data_collection" / "imd_international"
OUT = ROOT / "experiments" / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CITIES = [
    ("boston_bluebikes",    "boston_bluebikes",         "Bluebikes Boston"),
    ("dc_capitalbikeshare", "dc_capitalbikeshare",      "Capital Bikeshare DC"),
]
K_DEPLOYED = 50
COVER_RADIUS_M = 500.0     # MCLP coverage radius (typical of bike-share planning)
K_NN = 6
SIGMA_M = 300.0
EARTH_R = 6_371_000.0
N_EIGS = 50
SOLVER_TIMELIMIT = 120     # seconds per MIP


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
    return D, deg, Lsym, eigvals, eigvecs


def solve_mclp(D_full: np.ndarray, weights: np.ndarray, K: int, radius: float,
               timelimit: int = SOLVER_TIMELIMIT) -> list[int]:
    """Maximal Covering Location Problem (MCLP). Returns indices of K
    facilities maximising weighted demand coverage."""
    N = len(weights)
    # cover_matrix[i, j] = 1 if candidate j is within `radius` of demand point i
    cover = (D_full <= radius).astype(np.int8)
    np.fill_diagonal(cover, 1)
    prob = pulp.LpProblem("MCLP", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(N)]
    z = [pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(N)]
    prob += pulp.lpSum(weights[i] * z[i] for i in range(N))
    prob += pulp.lpSum(x[j] for j in range(N)) <= K
    for i in range(N):
        cov_set = np.where(cover[i] == 1)[0]
        if len(cov_set) > 0:
            prob += z[i] <= pulp.lpSum(x[j] for j in cov_set)
        else:
            prob += z[i] == 0
    solver = pulp.PULP_CBC_CMD(timeLimit=timelimit, msg=False)
    prob.solve(solver)
    S = [j for j in range(N) if pulp.value(x[j]) > 0.5]
    return S


def solve_kmedian(D_full: np.ndarray, weights: np.ndarray, K: int,
                  timelimit: int = SOLVER_TIMELIMIT) -> list[int]:
    """k-median : minimise weighted assignment cost to nearest facility."""
    N = len(weights)
    prob = pulp.LpProblem("kmedian", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(N)]
    a = [[pulp.LpVariable(f"a_{i}_{j}", lowBound=0, upBound=1) for j in range(N)] for i in range(N)]
    prob += pulp.lpSum(weights[i] * D_full[i, j] * a[i][j] for i in range(N) for j in range(N))
    prob += pulp.lpSum(x[j] for j in range(N)) <= K
    for i in range(N):
        prob += pulp.lpSum(a[i][j] for j in range(N)) == 1
        for j in range(N):
            prob += a[i][j] <= x[j]
    solver = pulp.PULP_CBC_CMD(timeLimit=timelimit, msg=False)
    prob.solve(solver)
    S = [j for j in range(N) if pulp.value(x[j]) > 0.5]
    return S


def d_optimal_greedy(U_K, K_target):
    N, K = U_K.shape
    S = []
    for _ in range(min(K_target, N)):
        if not S:
            scores = (U_K ** 2).sum(axis=1)
        else:
            U_S = U_K[S, :]; G = U_S.T @ U_S + 1e-8 * np.eye(K)
            G_inv = np.linalg.pinv(G)
            scores = np.einsum("ik,kl,il->i", U_K, G_inv, U_K)
        for s in S: scores[s] = -np.inf
        S.append(int(np.argmax(scores)))
    return S


def band_limited_reconstruct(y_obs, S, U_K):
    c = np.linalg.pinv(U_K[S, :]) @ y_obs
    return U_K @ c


def load_demand(slug):
    path = OUT / f"d3_{slug}_predictions.parquet"
    df = pd.read_parquet(path)
    df["station_id"] = df["station_id"].astype(str)
    df["y_true"] = np.expm1(df["y_true_log"])
    return df.groupby("station_id")["y_true"].mean().to_dict()


def evaluate(S, y_z, y_raw, U_K, name):
    y_obs = y_z[S]
    f_hat = band_limited_reconstruct(y_obs, S, U_K)
    held = [i for i in range(len(y_z)) if i not in set(S)]
    rho, _ = spearmanr(y_z[held], f_hat[held]) if held else (float("nan"), 1)
    frac_demand = float(y_raw[S].sum() / y_raw.sum())
    return dict(strategy=name, K=len(S), rho_heldout=float(rho),
                fraction_demand_at_S=frac_demand)


def run_city(slug, imd_stem, pretty):
    print(f"\n=== {pretty} ({slug}) ===")
    imd = pd.read_parquet(IMD / f"{imd_stem}.parquet")
    imd["station_id"] = imd["station_id"].astype(str)
    y_map = load_demand(slug)
    imd["y"] = imd["station_id"].map(y_map)
    sub = imd.dropna(subset=["y", "lat", "lng"]).reset_index(drop=True)
    N = len(sub)
    print(f"  N = {N}, K = {K_DEPLOYED}, coverage radius = {COVER_RADIUS_M:.0f}m")

    D, deg, Lsym, evals, evecs = build_graph(sub["lat"].values, sub["lng"].values)
    U_K = evecs[:, :min(N_EIGS, N - 1)]
    y = sub["y"].values
    y_z = (y - y.mean()) / (y.std() + 1e-12)

    results = []
    # D-optimal greedy
    t0 = time.time()
    S_dopt = d_optimal_greedy(U_K, K_DEPLOYED)
    results.append({**evaluate(S_dopt, y_z, y, U_K, "D-optimal greedy"),
                    "solve_time_s": round(time.time() - t0, 2)})
    print(f"  D-optimal greedy : ρ = {results[-1]['rho_heldout']:+.3f}  "
          f"frac = {results[-1]['fraction_demand_at_S']:.3f}  "
          f"({results[-1]['solve_time_s']}s)")

    # MCLP
    t0 = time.time()
    S_mclp = solve_mclp(D, y, K_DEPLOYED, COVER_RADIUS_M)
    results.append({**evaluate(S_mclp, y_z, y, U_K, "MCLP"),
                    "solve_time_s": round(time.time() - t0, 2)})
    print(f"  MCLP            : ρ = {results[-1]['rho_heldout']:+.3f}  "
          f"frac = {results[-1]['fraction_demand_at_S']:.3f}  "
          f"({results[-1]['solve_time_s']}s, |S|={len(S_mclp)})")

    # Skip k-median for N>200 (too slow with PuLP+CBC)
    if N <= 250:
        t0 = time.time()
        try:
            S_kmed = solve_kmedian(D, y, K_DEPLOYED)
            results.append({**evaluate(S_kmed, y_z, y, U_K, "k-median"),
                            "solve_time_s": round(time.time() - t0, 2)})
            print(f"  k-median        : ρ = {results[-1]['rho_heldout']:+.3f}  "
                  f"frac = {results[-1]['fraction_demand_at_S']:.3f}  "
                  f"({results[-1]['solve_time_s']}s, |S|={len(S_kmed)})")
        except Exception as e:
            print(f"  k-median failed: {e}")

    return dict(city=pretty, slug=slug, N=N, K=K_DEPLOYED,
                cover_radius_m=COVER_RADIUS_M, results=results,
                coords=sub[["lat", "lng"]].values.tolist(),
                S_dopt=S_dopt, S_mclp=S_mclp)


def main():
    all_res = []
    for slug, stem, pretty in CITIES:
        r = run_city(slug, stem, pretty)
        all_res.append(r)

    rows = []
    for r in all_res:
        for res in r["results"]:
            rows.append({"city": r["city"], "N": r["N"],
                         **res})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d33_mclp_comparison.csv", index=False)
    print("\n=== Summary ===")
    print(df.to_string(index=False))

    # Plot Boston: D-optimal vs MCLP map
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, r in zip(axes, all_res):
        coords = np.array(r["coords"])
        ax.scatter(coords[:, 1], coords[:, 0], s=6, c="lightgray", alpha=0.5)
        ax.scatter(coords[r["S_dopt"], 1], coords[r["S_dopt"], 0],
                   s=25, c="C0", marker="o", edgecolor="black", linewidth=0.4,
                   label=f"D-optimal greedy (K={K_DEPLOYED})", zorder=3)
        ax.scatter(coords[r["S_mclp"], 1], coords[r["S_mclp"], 0],
                   s=20, c="C3", marker="s", edgecolor="black", linewidth=0.4,
                   label=f"MCLP (K={K_DEPLOYED})", zorder=4)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(f"{r['city']} : D-optimal greedy vs MCLP siting")
        ax.legend(fontsize=8); ax.set_aspect("equal")
    fig.savefig(FIG / "fig_mclp_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_mclp_comparison.png", bbox_inches="tight", dpi=200)
    print(f"\n✓ Wrote {FIG/'fig_mclp_comparison.pdf'}")

    with open(OUT / "d33_mclp_comparison.json", "w") as f:
        json.dump({"cities": [{k: v for k, v in r.items() if k != "coords"}
                              for r in all_res]}, f, indent=2)
    print(f"✓ Saved")


if __name__ == "__main__":
    main()
