"""
d24_gsp_real_cities.py — Graph Signal Processing formalisation of IMD-4
on the nine real cities of Table tab:lso.

For each city we build the k-nearest-neighbour station-proximity graph
on the IMD parquet's (lat, lng) coordinates, with Gaussian-RBF edge
weights.  We compute the symmetric-normalised graph Laplacian
L_sym = I - D^{-1/2} W D^{-1/2}, its full eigendecomposition
L_sym = U Λ Uᵀ, and project the empirical demand and the four IMD
axes (M, I, T, D) onto the Laplacian eigenbasis.

We then quantify three theoretical quantities that the paper §6.0
will reference:

(T1) Dirichlet energy / smoothness of each signal on the graph:
        E_G(f) = fᵀ L f
     Lower E_G means smoother signal.  We compare E_G(y), E_G(IMD),
     and the energy of the residual y - ŷ_IMD.

(T2) Spectral concentration of demand: fraction of demand variance
     captured by the bottom-k eigenvectors (low-frequency modes).
     We report the smallest k such that ‖proj_k(y)‖² / ‖y‖² ≥ 0.9.
     A small k confirms that demand is a smooth signal on the
     station graph — the precondition for any local-feature
     indicator like IMD-4 to be predictive.

(T3) Spectral alignment between IMD subspace and demand: the
     fraction of demand variance that lies in the 4-D subspace
     spanned by (M, I, T, D) on the eigenbasis.  This is a
     theoretical upper bound on R²_IMD attainable by any linear
     IMD-augmented predictor.

Output:
  outputs/d24_gsp_per_city.csv
  figures/fig_gsp_spectral_concentration.{pdf,png}
  figures/fig_gsp_alignment_bar.{pdf,png}
"""
from __future__ import annotations

import json
import os
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
IMD_INTL_DIR = ROOT / "data_collection" / "imd_international"
OUT = ROOT / "experiments" / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Same 9 cities as tab:lso
CITIES = [
    # (slug, imd parquet stem, pretty name, panel source)
    ("boston_bluebikes",       "boston_bluebikes",       "Bluebikes Boston",       "tier1"),
    ("dc_capitalbikeshare",    "dc_capitalbikeshare",    "Capital Bikeshare DC",   "tier1"),
    ("chicago_divvy",          "chicago_divvy",          "Divvy Chicago",          "tier1"),
    ("sf_baywheels",           "sf_baywheels",           "Bay Wheels SF",          "tier1"),
    ("london_tfl",             "london_tfl",             "Santander Cycles London","tier1"),
    ("montreal_bixi",          "world_ca_bixi_montr_al", "BIXI Montréal",          "tier1"),
    ("tier2_paris",            "world_fr_v_lib_metropole","Vélib Paris",           "tier2"),
    ("tier2_lyon",             "world_fr_v_lo_v",        "Vélo'v Lyon",            "tier2"),
    ("tier2_toulouse",         "world_fr_v_l_toulouse",  "VéLÔ Toulouse",          "tier2"),
]

K_NN = 6
SIGMA_M = 300.0      # Gaussian RBF length scale on edges (~ 300 m)
EARTH_R = 6_371_000.0


def haversine_matrix(lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    """Haversine distance matrix in metres."""
    lat_r = np.deg2rad(lat); lng_r = np.deg2rad(lng)
    dphi = lat_r[:, None] - lat_r[None, :]
    dlam = lng_r[:, None] - lng_r[None, :]
    a = np.sin(dphi / 2)**2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlam / 2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_knn_graph(lat: np.ndarray, lng: np.ndarray, k: int, sigma: float):
    """k-NN station graph with Gaussian-RBF edge weights based on
    haversine distance.  Returns adjacency W (symmetrised),
    symmetric Laplacian L_sym, and its eigendecomposition."""
    N = len(lat)
    D_mat = haversine_matrix(lat, lng)
    np.fill_diagonal(D_mat, np.inf)
    # k nearest neighbours for each node
    knn_idx = np.argpartition(D_mat, k, axis=1)[:, :k]
    W = np.zeros((N, N))
    for i in range(N):
        for j in knn_idx[i]:
            w = np.exp(-D_mat[i, j]**2 / (2 * sigma**2))
            W[i, j] = max(W[i, j], w); W[j, i] = W[i, j]
    deg = W.sum(axis=1)
    deg_safe = np.maximum(deg, 1e-12)
    Dinv2 = 1.0 / np.sqrt(deg_safe)
    Lsym = np.eye(N) - (W * Dinv2[:, None]) * Dinv2[None, :]
    eigvals, eigvecs = np.linalg.eigh(Lsym)
    return W, deg, Lsym, eigvals, eigvecs


def dirichlet_energy(f: np.ndarray, L: np.ndarray) -> float:
    """E_G(f) = f^T L f"""
    return float(f @ L @ f)


def load_demand_for_city(slug: str, imd_df: pd.DataFrame, source: str):
    """Load mean hourly demand per station for one city, aligned to
    imd_df.station_id ordering.  Returns a vector y of length N (NaN
    for stations without data) and a bool mask of present stations."""
    if source == "tier1" and slug in ("boston_bluebikes", "dc_capitalbikeshare",
                                       "chicago_divvy", "sf_baywheels"):
        path = OUT / f"d3_{slug}_predictions.parquet"
        if not path.exists(): return None
        df = pd.read_parquet(path)
        df["station_id"] = df["station_id"].astype(str)
        y_true = np.expm1(df["y_true_log"].values)
        agg = pd.DataFrame({"station_id": df["station_id"].values, "y": y_true})
        m = agg.groupby("station_id")["y"].mean()
        return m
    if slug == "london_tfl":
        path = OUT / "d16_london_tfl_predictions.parquet"
        if not path.exists(): return None
        df = pd.read_parquet(path)
        df["station_id"] = df["station_id"].astype(str).str.zfill(6)
        m = pd.DataFrame({"sid": df["station_id"], "y": np.expm1(df["y_true_log"])})
        return m.groupby("sid")["y"].mean()
    if slug == "montreal_bixi":
        path = OUT / "d14_montreal_bixi_predictions.parquet"
        if not path.exists(): return None
        df = pd.read_parquet(path)
        df["station_id"] = df["station_id"].astype(str)
        m = pd.DataFrame({"sid": df["station_id"], "y": np.expm1(df["y_true_log"])})
        return m.groupby("sid")["y"].mean()
    if slug.startswith("tier2_"):
        city_map = {"tier2_paris": "Paris", "tier2_lyon": "lyon",
                    "tier2_toulouse": "toulouse"}
        c = city_map[slug]
        path = OUT / f"d10_{c}_predictions.parquet"
        if not path.exists(): return None
        df = pd.read_parquet(path)
        df["station_id"] = df["station_id"].astype(str)
        m = pd.DataFrame({"sid": df["station_id"], "y": np.expm1(df["y_true_log"])})
        return m.groupby("sid")["y"].mean()
    return None


def main():
    t_start = time.time()
    rows = []
    spectral_curves = {}

    for slug, imd_stem, pretty, source in CITIES:
        print(f"\n=== {pretty} ({slug}) ===")
        imd_path = IMD_INTL_DIR / f"{imd_stem}.parquet"
        if not imd_path.exists():
            print(f"  ✗ IMD missing: {imd_path}"); continue
        imd_df = pd.read_parquet(imd_path)
        imd_df["station_id"] = imd_df["station_id"].astype(str)
        if slug == "london_tfl":
            imd_df["station_id"] = imd_df["station_id"].str.zfill(6)
        imd_df = imd_df.dropna(subset=["lat", "lng"]).reset_index(drop=True)
        N = len(imd_df)
        if N < 50:
            print(f"  ✗ too few stations ({N})"); continue
        print(f"  {N} stations  building k-NN graph (k={K_NN}, σ={SIGMA_M}m)...")
        W, deg, Lsym, eigvals, eigvecs = build_knn_graph(
            imd_df["lat"].values, imd_df["lng"].values,
            k=min(K_NN, N - 1), sigma=SIGMA_M)
        print(f"    graph: {N} nodes, mean degree {deg.mean():.1f}")
        print(f"    spectrum: λ ∈ [{eigvals[0]:.4f}, {eigvals[-1]:.4f}]")

        # IMD signals on graph
        feats = ["gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
                 "elevation_m", "topography_roughness_index"]
        signals = {}
        for f in feats:
            if f in imd_df.columns:
                v = imd_df[f].astype(float).values
                v = (v - v.mean()) / (v.std() + 1e-12)
                signals[f] = v
        # Demand
        y_series = load_demand_for_city(slug, imd_df, source)
        if y_series is None:
            print(f"  ✗ demand data missing"); continue
        imd_df["y"] = imd_df["station_id"].map(y_series)
        mask = imd_df["y"].notna().values
        if mask.sum() < 20:
            print(f"  ✗ only {mask.sum()} stations matched"); continue
        y_full = imd_df["y"].fillna(0).values
        y_z = (y_full - y_full[mask].mean()) / (y_full[mask].std() + 1e-12)
        # Restrict to the subset where both demand and IMD are available
        # Use eigenbasis on the FULL graph (it's defined regardless), but
        # compute energies / projections only on the masked subset.
        # For simplicity, compute spectral quantities on the full N
        # using y_z as-is (zeroing missing); this slightly understates
        # E_G(y) but is conservative.

        # T1: Dirichlet energies
        E_y    = float(y_z @ Lsym @ y_z)
        E_imd  = {f: float(s @ Lsym @ s) for f, s in signals.items()}
        # Random reference: a permutation of y on the graph; expected E_perm
        perms = []
        rng = np.random.default_rng(42)
        for _ in range(50):
            yp = y_z[rng.permutation(N)]
            perms.append(float(yp @ Lsym @ yp))
        E_perm_mean = float(np.mean(perms))
        # Smoothness = how far below random permutation
        smoothness_y = 1.0 - E_y / E_perm_mean

        # T2: Spectral concentration of demand
        coefs_y = eigvecs.T @ y_z          # shape (N,)
        total_E = float((coefs_y ** 2).sum())
        cum = np.cumsum(coefs_y ** 2) / total_E
        k90 = int(np.searchsorted(cum, 0.90) + 1)
        k50 = int(np.searchsorted(cum, 0.50) + 1)

        # T3: Spectral alignment IMD subspace ↔ demand
        if signals:
            S = np.column_stack(list(signals.values()))     # (N, k_feat)
            # OLS projection onto IMD subspace (via QR)
            Q, _ = np.linalg.qr(S)
            proj_y = Q @ (Q.T @ y_z)
            R2_spectral_imd = float((proj_y ** 2).sum() / total_E)
        else:
            R2_spectral_imd = float("nan")

        # Also compute mean Dirichlet energy of IMD features (smoother → lower)
        E_imd_mean = float(np.mean(list(E_imd.values()))) if E_imd else float("nan")

        rows.append({
            "slug": slug, "city": pretty, "source": source,
            "N": N,
            "mean_degree": float(deg.mean()),
            "lambda_max": float(eigvals[-1]),
            "E_y_dirichlet": E_y,
            "E_y_permuted_ref": E_perm_mean,
            "smoothness_y": smoothness_y,
            "E_imd_mean_dirichlet": E_imd_mean,
            "k_concentration_50": k50,
            "k_concentration_90": k90,
            "R2_spectral_imd_subspace_upper_bound": R2_spectral_imd,
        })
        print(f"  E_G(y) = {E_y:.2f}  vs E_G(permuted) = {E_perm_mean:.2f}  →  smoothness = {smoothness_y:.3f}")
        print(f"  k_50% concentration = {k50}  k_90% concentration = {k90}  ({100*k90/N:.1f}% of modes)")
        print(f"  R²_spectral (IMD subspace) = {R2_spectral_imd:.3f}")
        spectral_curves[slug] = (cum.copy(), pretty)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d24_gsp_per_city.csv", index=False)
    print(f"\n=== Summary ===")
    print(df[["city", "N", "smoothness_y", "k_concentration_90",
              "R2_spectral_imd_subspace_upper_bound"]].to_string(index=False))

    # ── Figure 1: spectral concentration curves ─────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(spectral_curves)))
    for (slug, (curve, pretty)), c in zip(spectral_curves.items(), colors):
        ax.plot(np.arange(1, len(curve) + 1) / len(curve), curve,
                label=pretty, color=c, linewidth=1.2)
    ax.axhline(0.9, color="red", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Fraction of graph eigenvectors (sorted by frequency)")
    ax.set_ylabel("Cumulative fraction of demand variance captured")
    ax.set_title("Spectral concentration of demand on the station-proximity graph\n"
                 "(low-frequency mass → smoothness)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG / "fig_gsp_spectral_concentration.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_gsp_spectral_concentration.png", bbox_inches="tight", dpi=200)
    print(f"\n✓ Wrote {FIG/'fig_gsp_spectral_concentration.pdf'}")

    # ── Figure 2: spectral R² upper bound vs realised LSO R² ────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    df_plot = df.sort_values("R2_spectral_imd_subspace_upper_bound")
    yidx = np.arange(len(df_plot))
    ax.barh(yidx - 0.2, df_plot["R2_spectral_imd_subspace_upper_bound"],
            height=0.4, label="Spectral upper bound (R²)", color="C0", alpha=0.8)
    ax.barh(yidx + 0.2, df_plot["smoothness_y"],
            height=0.4, label="Smoothness of demand", color="C1", alpha=0.8)
    ax.set_yticks(yidx)
    ax.set_yticklabels(df_plot["city"], fontsize=8)
    ax.set_xlabel("Fraction (0 – 1)")
    ax.set_xlim(0, 1)
    ax.set_title("Spectral characterisation per city: smoothness and IMD-explainable variance")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(FIG / "fig_gsp_alignment_bar.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_gsp_alignment_bar.png", bbox_inches="tight", dpi=200)
    print(f"✓ Wrote {FIG/'fig_gsp_alignment_bar.pdf'}")

    with open(OUT / "d24_gsp_summary.json", "w") as f:
        json.dump({
            "cities": rows,
            "config": dict(K_NN=K_NN, SIGMA_M=SIGMA_M, N_PERMS=50),
            "wall_time_s": round(time.time() - t_start, 1),
        }, f, indent=2)
    print(f"✓ Saved {OUT / 'd24_gsp_summary.json'}")
    print(f"  Total wall time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
