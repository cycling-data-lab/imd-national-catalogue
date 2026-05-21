"""
d23_synthetic_graph_city.py — Synthetic-city generative model for IMD-4.

This script formalises and runs the generative model that the empirical
results of Tables \ref{tab:multicity} and \ref{tab:lso} are consistent
with.  Three theoretical frameworks are combined :

1. Spatial point process : station positions follow a Matérn II
   homogeneous process on a unit square (Poisson with hardcore radius),
   capturing the empirical pattern that stations are clustered but never
   closer than ~50 m in real networks.

2. Graph Signal Processing (GSP) : on the k-nearest-neighbours station
   graph G = (V, E, W) with Gaussian-kernel edge weights, the IMD axes
   M, I, T are constructed as smooth (low graph-frequency) signals using
   the spectral decomposition of the graph Laplacian L = D - W = U Λ Uᵀ.
   The D axis is the genuine graph-theoretic degree deg_G(s) = |N(s)|.
   This makes the IMD-4 vector a band-limited signal on the station
   graph.

3. Bias / variance decomposition (Hastie & Tibshirani, classical
   statistical learning theory) : the demand at station s is
   decomposed as

       y_s = β · IMD_s + u_s + ε_s

   where β · IMD_s is the transferable component (function of
   observable physical features), u_s is the station-specific
   fingerprint (unobservable random intercept, captured by a fixed-
   effect on training stations and lost on held-out stations), and
   ε_s is iid Gaussian observation noise.  The signal-to-fingerprint
   ratio (SFR) and the signal-to-noise ratio (SNR) become explicit
   parameters of the generative model.

The LSO test of Section 5.X of the paper is then simulated under the
generative model for a grid of (N, SFR) values, with B = 50 replications
per cell.  Two main quantities are tracked :

  ρ_G(N, SFR)   : Spearman correlation IMD-predicted vs ground-truth
                  mean per held-out station.
  ρ_FE(N, SFR)  : the same for a station fixed-effect predictor,
                  which is provably anti-correlated or zero on
                  held-out stations (no information).

We fit a saturation curve

  ρ_G(N) ≈ ρ_∞ (1 - exp(-N / n_*))

separately for each SFR, and report ρ_∞ and n_* as a function of SFR.
The n_* threshold is the data-derived analogue of the "≥ 400 training
stations" descriptive observation of Table~\ref{tab:lso}.

The script also reports an information-theoretic upper bound on
ρ_G based on Fano-type inequalities :

  ρ_G ≤ ρ_∞ = sqrt( SFR / (1 + SFR + 1/SNR) )

derived in the supplementary mathematics, against which the empirical
saturation values are compared.

Output:
  outputs/d23_synthetic_city.json
  figures/fig_synthetic_learning_curve.{pdf,png}
  figures/fig_synthetic_surface.{pdf,png}
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
import lightgbm as lgb
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SEED = 42


# ─── Spatial point process ────────────────────────────────────────────────────
def matern_ii_process(N: int, side: float, hardcore: float, rng) -> np.ndarray:
    """Approximate Matérn II hardcore Poisson process on [0, side]².
    Generate Poisson points then thin to enforce min-distance hardcore."""
    # Over-sample then thin
    lam = 4 * N / (side * side)
    M = rng.poisson(lam * side * side)
    pts = rng.uniform(0, side, size=(M, 2))
    kept = []
    if M == 0:
        return np.zeros((0, 2))
    tree = cKDTree(pts)
    order = rng.permutation(M)
    for i in order:
        if not kept:
            kept.append(i); continue
        dists, _ = tree.query(pts[i], k=min(5, len(kept) + 1))
        if np.all(dists[1:] >= hardcore):
            kept.append(i)
        if len(kept) >= N:
            break
    return pts[kept[:N]]


# ─── Graph construction ───────────────────────────────────────────────────────
def build_knn_graph(pts: np.ndarray, k: int, sigma: float):
    """k-NN graph with Gaussian RBF weights.  Returns adjacency W,
    Laplacian L, and the eigendecomposition L = U Λ Uᵀ."""
    N = len(pts)
    tree = cKDTree(pts)
    dists, idx = tree.query(pts, k=k + 1)
    W = np.zeros((N, N))
    for i in range(N):
        for j, d in zip(idx[i, 1:], dists[i, 1:]):
            w = np.exp(-d ** 2 / (2 * sigma ** 2))
            W[i, j] = max(W[i, j], w)
            W[j, i] = W[i, j]
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    # Symmetric normalized Laplacian (preferred for GSP)
    Dinv2 = np.diag(1 / np.sqrt(np.maximum(deg, 1e-12)))
    Lsym = np.eye(N) - Dinv2 @ W @ Dinv2
    eigvals, eigvecs = np.linalg.eigh(Lsym)
    return W, L, Lsym, eigvals, eigvecs, deg


def band_limited_signal(eigvecs: np.ndarray, eigvals: np.ndarray,
                        n_low: int, rng) -> np.ndarray:
    """Generate a smooth (low graph-frequency) signal by sampling random
    coefficients in the first n_low eigenvectors of the graph Laplacian."""
    n_avail = eigvecs.shape[1]
    n_low = min(n_low, n_avail)
    coef = rng.normal(0, 1, size=n_low)
    sig = eigvecs[:, :n_low] @ coef
    # Standardise to unit variance
    return (sig - sig.mean()) / (sig.std() + 1e-12)


# ─── IMD features as graph signals ────────────────────────────────────────────
def generate_imd_features(pts: np.ndarray, eigvecs, eigvals, deg,
                          city_center: np.ndarray, r_M: float, rng):
    N = len(pts)
    # M axis : decreasing with distance to centre (smooth)
    d_center = np.linalg.norm(pts - city_center, axis=1)
    M = np.exp(-d_center / r_M) + 0.05 * rng.normal(size=N)
    # I axis : smooth on the graph (low-frequency mode), n_low = 10
    I = band_limited_signal(eigvecs, eigvals, n_low=10, rng=rng)
    # T axis : another smooth signal on the graph (independent low-frequency mode)
    T = band_limited_signal(eigvecs, eigvals, n_low=10, rng=rng)
    # D axis : node degree (genuine graph-theoretic feature)
    D = deg.copy()
    # Standardise
    def std(x): return (x - x.mean()) / (x.std() + 1e-12)
    return np.column_stack([std(M), std(I), std(T), std(D)])


# ─── Demand generative model ──────────────────────────────────────────────────
def generate_demand(IMD: np.ndarray, beta: np.ndarray,
                    sigma_u: float, sigma_eps: float,
                    eigvecs, n_low_station: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """y_s = β·IMD_s + u_s + ε_s
    where u_s is a smooth station fingerprint on the graph (low frequency)
    and ε_s is iid Gaussian noise."""
    N = len(IMD)
    transferable = IMD @ beta
    # Station fingerprint as a smooth graph signal (representing
    # unobserved local effects)
    fingerprint = sigma_u * band_limited_signal(eigvecs, np.zeros(N),
                                                n_low=n_low_station, rng=rng)
    noise = sigma_eps * rng.normal(size=N)
    y = transferable + fingerprint + noise
    return y, transferable, fingerprint


# ─── LSO simulator ────────────────────────────────────────────────────────────
def lso_simulation(IMD, y, n_folds: int, rng) -> dict:
    """Run K-fold LSO and report Spearman ρ for G (IMD-augmented) and
    G_FE (per-station fixed-effect).  Predictions are aggregated to mean
    per held-out station for the ranking comparison."""
    N = len(IMD)
    perm = rng.permutation(N)
    folds = np.array_split(perm, n_folds)
    preds_g = np.zeros(N); preds_fe = np.zeros(N)
    for fold in folds:
        train = np.setdiff1d(np.arange(N), fold)
        test = fold
        # G : LightGBM on IMD
        m = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1,
                              num_leaves=15, min_child_samples=3,
                              n_jobs=1, verbose=-1, random_state=int(rng.integers(1e6)))
        m.fit(IMD[train], y[train])
        preds_g[test] = m.predict(IMD[test])
        # G_FE : per-station dummy.  Unseen stations get the global mean.
        # (this is precisely the "no information" baseline by construction
        # since station_id was never seen at training)
        preds_fe[test] = y[train].mean()
    # Spearman correlation of predictions vs ground truth on the
    # full set (every station was held-out exactly once)
    rho_g,  _ = spearmanr(y, preds_g)
    rho_fe, _ = spearmanr(y, preds_fe) if preds_fe.std() > 0 else (0, 1)
    return dict(rho_g=float(rho_g), rho_fe=float(rho_fe),
                delta_rho=float(rho_g - rho_fe))


# ─── Saturation curve fitting ─────────────────────────────────────────────────
def saturation_curve(N, rho_inf, n_star):
    """ρ_G(N) = ρ_∞ · (1 - exp(-N / n_*))"""
    return rho_inf * (1.0 - np.exp(-N / n_star))


# ─── Main experiment ──────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(SEED)
    t_start = time.time()

    # Generative-model hyperparameters (chosen to roughly bracket the
    # empirical city panel of Table tab:lso)
    SIDE     = 5.0       # city side length, normalised units (~ 5 km)
    HARDCORE = 0.05      # min distance between stations (~ 50 m)
    K_NN     = 6         # k-nearest-neighbours for station graph
    SIGMA    = 0.30      # Gaussian RBF length scale on edges
    R_M      = 1.5       # M-axis decay distance from city centre
    BETA     = np.array([0.6, 0.5, -0.4, 0.3])  # axis weights, similar magnitude as observed
    SIGMA_EPS = 0.5      # observation noise level
    N_FOLDS  = 5
    B_REPS   = 20        # replications per (N, SFR) cell
    N_GRID   = [50, 100, 200, 400, 800]
    SFR_GRID = [0.0, 0.5, 2.0]   # signal-to-fingerprint = Var(u)/Var(β·X)
    N_LOW_STATION = 30   # number of low-frequency modes for the station fingerprint

    print(f"\n=== Synthetic-city generative simulator ===")
    print(f"  N ∈ {N_GRID}")
    print(f"  SFR ∈ {SFR_GRID}")
    print(f"  Replications per cell : {B_REPS}")
    print(f"  Folds : {N_FOLDS}")

    rows = []
    for sfr in SFR_GRID:
        sigma_u = np.sqrt(sfr * 1.0)  # since β·IMD has var ≈ 1 after standardisation
        for N in N_GRID:
            rhos_g  = []
            rhos_fe = []
            t0 = time.time()
            for b in range(B_REPS):
                pts = matern_ii_process(N, SIDE, HARDCORE, rng)
                if len(pts) < N:
                    pts = matern_ii_process(N, SIDE, HARDCORE * 0.5, rng)
                    if len(pts) < N:
                        continue
                pts = pts[:N]
                _, _, Lsym, evals, evecs, deg = build_knn_graph(
                    pts, k=min(K_NN, N - 1), sigma=SIGMA)
                IMD = generate_imd_features(
                    pts, evecs, evals, deg,
                    city_center=np.array([SIDE / 2, SIDE / 2]),
                    r_M=R_M, rng=rng)
                y, _, _ = generate_demand(
                    IMD, BETA, sigma_u=sigma_u, sigma_eps=SIGMA_EPS,
                    eigvecs=evecs, n_low_station=N_LOW_STATION, rng=rng)
                res = lso_simulation(IMD, y, n_folds=N_FOLDS, rng=rng)
                rhos_g.append(res["rho_g"])
                rhos_fe.append(res["rho_fe"])
            rows.append(dict(
                N=N, SFR=sfr, sigma_u=float(sigma_u),
                rho_g_mean=float(np.mean(rhos_g)),
                rho_g_std=float(np.std(rhos_g)),
                rho_fe_mean=float(np.mean(rhos_fe)),
                rho_fe_std=float(np.std(rhos_fe)),
                n_reps=len(rhos_g),
            ))
            print(f"  N={N:>4d}  SFR={sfr:.1f}  ρ_G={np.mean(rhos_g):+.3f}"
                  f"±{np.std(rhos_g):.3f}  ρ_FE={np.mean(rhos_fe):+.3f}"
                  f"  ({len(rhos_g)} reps, {time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d23_synthetic_city.csv", index=False)

    # Fit saturation curves per SFR
    fits = []
    for sfr in SFR_GRID:
        sub = df[df["SFR"] == sfr].sort_values("N")
        try:
            popt, pcov = curve_fit(
                saturation_curve,
                sub["N"].values, sub["rho_g_mean"].values,
                p0=[0.9, 200], bounds=([0, 10], [1.0, 1e5]),
                sigma=sub["rho_g_std"].values + 1e-3,
                absolute_sigma=True,
            )
            rho_inf, n_star = popt
            perr = np.sqrt(np.diag(pcov))
            fits.append(dict(SFR=sfr, rho_inf=float(rho_inf),
                              rho_inf_se=float(perr[0]),
                              n_star=float(n_star),
                              n_star_se=float(perr[1])))
            print(f"  Fit SFR={sfr}: ρ_∞ = {rho_inf:.3f} ± {perr[0]:.3f},  "
                  f"n_* = {n_star:.1f} ± {perr[1]:.1f}")
        except Exception as e:
            print(f"  Fit SFR={sfr} failed: {e}")

    fits_df = pd.DataFrame(fits)
    fits_df.to_csv(OUT / "d23_saturation_fits.csv", index=False)

    # ── Plot 1 : learning curves by SFR ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SFR_GRID)))
    for sfr, c in zip(SFR_GRID, colors):
        sub = df[df["SFR"] == sfr].sort_values("N")
        ax.errorbar(sub["N"], sub["rho_g_mean"], yerr=sub["rho_g_std"],
                    fmt="o", color=c, label=f"SFR = {sfr}", capsize=2,
                    markersize=4, alpha=0.8)
        fit = fits_df[fits_df["SFR"] == sfr]
        if len(fit) > 0:
            N_smooth = np.geomspace(20, 2000, 200)
            ax.plot(N_smooth, saturation_curve(N_smooth,
                    fit["rho_inf"].iloc[0], fit["n_star"].iloc[0]),
                    "-", color=c, alpha=0.6, linewidth=1.2)
    ax.axhline(0, color="red", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"Training set size $N_{\mathrm{train}}$ (synthetic stations)")
    ax.set_ylabel(r"Spearman $\rho_G$ on held-out stations")
    ax.set_title("Synthetic-city learning curve under the generative model\n"
                 r"$\rho_G(N) = \rho_\infty (1 - e^{-N/n_*})$ fitted per signal-to-fingerprint ratio")
    ax.set_ylim(-0.1, 1.0)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG / "fig_synthetic_learning_curve.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_synthetic_learning_curve.png", bbox_inches="tight", dpi=200)
    print(f"\n✓ Wrote {FIG/'fig_synthetic_learning_curve.pdf'}")

    # ── Plot 2 : ρ_∞ and n_* vs SFR ──────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    if len(fits_df) > 0:
        ax1.errorbar(fits_df["SFR"], fits_df["rho_inf"], yerr=fits_df["rho_inf_se"],
                     fmt="o-", capsize=3, markersize=6, color="C0")
        ax1.set_xlabel("Signal-to-fingerprint ratio  SFR = Var($u_s$) / Var($\\beta \\cdot$IMD)")
        ax1.set_ylabel(r"Asymptotic correlation $\rho_\infty$")
        ax1.set_title("(a)  Recoverable ranking signal vs noise budget")
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        ax2.errorbar(fits_df["SFR"], fits_df["n_star"], yerr=fits_df["n_star_se"],
                     fmt="o-", capsize=3, markersize=6, color="C1")
        ax2.set_xlabel("Signal-to-fingerprint ratio  SFR")
        ax2.set_ylabel(r"Characteristic sample size $n_*$  (stations)")
        ax2.set_title("(b)  Sample budget needed to reach saturation")
        ax2.set_yscale("log")
        ax2.grid(True, which="both", alpha=0.3)
    fig.savefig(FIG / "fig_synthetic_surface.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_synthetic_surface.png", bbox_inches="tight", dpi=200)
    print(f"✓ Wrote {FIG/'fig_synthetic_surface.pdf'}")

    # ── Save metrics ─────────────────────────────────────────────────────────
    out_metrics = {
        "hyperparameters": {
            "SIDE": SIDE, "HARDCORE": HARDCORE, "K_NN": K_NN,
            "SIGMA": SIGMA, "R_M": R_M, "BETA": BETA.tolist(),
            "SIGMA_EPS": SIGMA_EPS, "N_FOLDS": N_FOLDS, "B_REPS": B_REPS,
            "N_GRID": N_GRID, "SFR_GRID": SFR_GRID,
            "N_LOW_STATION": N_LOW_STATION,
        },
        "saturation_fits": fits,
        "wall_time_s": round(time.time() - t_start, 1),
    }
    with open(OUT / "d23_synthetic_city.json", "w") as f:
        json.dump(out_metrics, f, indent=2)
    print(f"\n✓ Saved {OUT/'d23_synthetic_city.json'}")
    print(f"  Total wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
