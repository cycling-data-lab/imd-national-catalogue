"""
d28_spectral_bottleneck.py — Spectral Information Bottleneck (B1):
the optimal-rank-K subspace of graph eigenvectors that captures
demand variance, versus the rank-4 subspace spanned by the IMD-4
axes.  Answers : how many axes does the demand actually need ?

For each city :
  1. Compute station-proximity graph and Laplacian eigendecomposition.
  2. Project mean station demand on the eigenbasis : ŷ_k = u_k^T y.
  3. Sort modes by |ŷ_k|² to find the optimal rank-K subspace
     (Eckart–Young : the rank-K subspace of R^N that maximises
     ||P_K y||² is spanned by the K eigenmodes carrying the most
     demand variance).
  4. R²_optimal(K) = (sum of top-K |ŷ_k|²) / Σ_k |ŷ_k|².
  5. Compute R²_IMD = ||P_IMD y||² / ||y||² (rank-4 subspace of IMD-4
     axes projected onto graph eigenbasis).
  6. Find K* = smallest K such that R²_optimal(K) ≥ R²_IMD ; this is
     "how many graph modes are needed to match what IMD-4 achieves".

Output :
  outputs/d28_spectral_bottleneck.csv
  outputs/d28_spectral_bottleneck.json
  figures/fig_spectral_bottleneck.{pdf,png}
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
IMD = ROOT / "data_collection" / "imd_international"
OUT = ROOT / "experiments" / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

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
FEATS_IMD = ["gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
             "elevation_m", "topography_roughness_index"]
K_MAX = 30


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
    t0 = time.time()
    all_curves = {}; summary = []
    for slug, stem, pretty in CITIES:
        imd = pd.read_parquet(IMD / f"{stem}.parquet")
        imd["station_id"] = imd["station_id"].astype(str)
        y_map = load_demand(slug)
        if not y_map: continue
        imd["y"] = imd["station_id"].map(y_map)
        avail = [f for f in FEATS_IMD if f in imd.columns]
        sub = imd.dropna(subset=["y", "lat", "lng"] + avail).reset_index(drop=True)
        N = len(sub)
        print(f"\n=== {pretty}: N = {N} stations, {len(avail)} IMD axes ===")
        _, evals, evecs = build_graph(sub["lat"].values, sub["lng"].values)
        y = sub["y"].values.astype(float)
        y_z = (y - y.mean()) / (y.std() + 1e-12)
        # Spectral coefficients
        y_hat = evecs.T @ y_z
        power = y_hat ** 2
        total = power.sum()
        # Sort by |ŷ_k|² descending → optimal-rank-K subspace
        order = np.argsort(-power)
        cum_opt = np.cumsum(power[order]) / total
        # Also cum by frequency (eigenvalue order, low to high) for comparison
        cum_freq = np.cumsum(power) / total
        # R²_IMD : project y onto IMD subspace
        X = sub[avail].astype(float).values
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        try:
            Q, _ = np.linalg.qr(X)
            R2_imd = float(((Q.T @ y_z) ** 2).sum() / total * total / (y_z @ y_z))
            R2_imd = float(((Q @ (Q.T @ y_z)) ** 2).sum() / (y_z @ y_z))
        except Exception:
            R2_imd = float("nan")
        # K* : smallest K with cum_opt[K-1] >= R2_imd
        if not np.isnan(R2_imd):
            K_star = int(np.searchsorted(cum_opt, R2_imd) + 1)
            K_star_freq = int(np.searchsorted(cum_freq, R2_imd) + 1)
        else:
            K_star = K_star_freq = -1

        K_axes = X.shape[1]
        # Where on cum_opt does K_axes lie?
        R2_opt_at_axes = float(cum_opt[K_axes - 1])
        # Elbow detection : K where derivative drops below 0.01 (1%)
        deriv = np.diff(cum_opt[:K_MAX])
        elbow = int(np.argmax(deriv < 0.01) + 1) if (deriv < 0.01).any() else K_MAX

        all_curves[slug] = dict(cum_opt=cum_opt[:K_MAX].tolist(),
                                cum_freq=cum_freq[:K_MAX].tolist(),
                                pretty=pretty)
        summary.append(dict(
            city=pretty, slug=slug, N=N, K_imd_axes=K_axes,
            R2_imd_subspace=R2_imd,
            R2_optimal_at_K_axes=R2_opt_at_axes,
            K_star_match_imd_optimal=K_star,
            K_star_match_imd_lowfreq=K_star_freq,
            elbow_K=elbow,
        ))
        print(f"  R² IMD-4 subspace projection : {R2_imd:.4f}")
        print(f"  R² optimal rank-{K_axes} subspace : {R2_opt_at_axes:.4f}")
        print(f"  K* (optimal modes to match IMD-4): {K_star}")
        print(f"  K* (low-frequency modes to match IMD-4): {K_star_freq}")
        print(f"  Elbow K (first K with marginal gain < 1%): {elbow}")

    df = pd.DataFrame(summary)
    df.to_csv(OUT / "d28_spectral_bottleneck.csv", index=False)
    print("\n=== Summary ===")
    print(df.to_string(index=False))

    # Plot curves
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_curves)))
    Ks = np.arange(1, K_MAX + 1)
    for (slug, d), c in zip(all_curves.items(), colors):
        ax.plot(Ks, d["cum_opt"], "-", color=c, linewidth=1.4, label=f"{d['pretty']} (optimal)")
        ax.plot(Ks, d["cum_freq"], "--", color=c, linewidth=0.7, alpha=0.5)
    # Mark IMD-4 R² on each city
    for row, c in zip(summary, colors):
        ax.axhline(row["R2_imd_subspace"], color=c, linewidth=0.4, alpha=0.4, linestyle=":")
    ax.axvline(4, color="red", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.text(4.2, 0.05, "IMD-4 rank", color="red", fontsize=8)
    ax.set_xlabel("Rank K (number of graph eigenmodes used)")
    ax.set_ylabel(r"Cumulative explained variance of demand $\|P_K \bar y\|^2 / \|\bar y\|^2$")
    ax.set_title("Spectral information bottleneck on the station-proximity graph\n"
                 "Solid: optimal rank-K subspace.  Dashed: low-frequency K modes (frequency-ordered).")
    ax.set_xlim(1, K_MAX); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG / "fig_spectral_bottleneck.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_spectral_bottleneck.png", bbox_inches="tight", dpi=200)
    print(f"\n✓ Wrote {FIG/'fig_spectral_bottleneck.pdf'}")

    with open(OUT / "d28_spectral_bottleneck.json", "w") as f:
        json.dump({"summary": summary, "curves": all_curves,
                   "wall_time_s": round(time.time()-t0, 1)},
                  f, indent=2)
    print(f"✓ Saved.  Total wall time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
