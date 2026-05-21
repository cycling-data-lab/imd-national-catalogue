"""B1 -- Bayesian IMD-3: full posterior inference and station-level distributions.

Defines the IMD as a three-component Bayesian composite indicator
from scratch:

    IMD_s = w_M * C_M^{(s)} + w_I * C_I^{(s)} + w_T * C_T^{(s)}

where the index s runs over the dock-based stations of the French
Gold Standard panel. C_M, C_I, C_T are the normalised
multimodality, cycling-infrastructure and topography components
defined on station-level Min-Max scaling.

The Bayesian model:

    Priors:
      z ~ Normal(0, sigma_w^2 * I_3),     w = softmax_with_floor(z)
      alpha_y ~ Normal(0, 10^2)            (intercept per reference)
      beta_y ~ Normal(0, 10^2)             (slope per reference)
      sigma_y ~ Half-Cauchy(1)             (obs noise per reference)

    Likelihood (for each reference y in {FUB, EMP}):
      For each city i in the y-panel:
        z_y_i ~ Normal(alpha_y + beta_y * standardised(IMD_city)_i,
                       sigma_y^2)
      where IMD_city is the city-mean of the station-level IMDs
      and z_y_i is the standardised reference value.

We sample the joint posterior on (z, alpha_FUB, beta_FUB,
sigma_FUB, alpha_EMP, beta_EMP, sigma_EMP) via Metropolis-Hastings
with adaptive proposal scale. The resulting posterior on w lives
on the (3-1)-simplex; we summarise it by the marginal mean, 95%
credible intervals, and the joint posterior probability of each
component dominating.

Per-station IMD posterior is obtained by propagating the posterior
samples of w through the fixed component matrix. The city-level
IMD is the weighted mean of station IMDs (cyclically equal
weights here). Each city has a full IMD posterior; we report the
median + 95% CI.

Outputs:
    outputs/b1_results.json
    outputs/b1_weights_posterior.pdf
    outputs/b1_top_ranking_ci.pdf
    outputs/b1_per_station_distribution_examples.pdf
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = next(
    p for p in [HERE, *HERE.parents]
    if (p / ".git").exists() or (p / "CITATION.cff").exists()
)
sys.path.insert(0, str(ROOT))

# Reuse the panel loader from the previous paper's experiments dir
sys.path.insert(0, str(ROOT / "papers" / "02_imd" / "experiments"))
from _common import load_panel  # noqa: E402
from utils.data_loader import load_stations  # noqa: E402

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RNG = np.random.default_rng(2026)

# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------

K = 3                          # number of weight components: M, I, T
W_MIN = 0.05                   # simplex floor for each weight
SIGMA_W = 1.5                  # weight pre-image prior SD
SIGMA_AB = 10.0                # nuisance prior SD
HALFCAUCHY_SCALE = 1.0

# MCMC
N_BURN = 4000
N_KEEP = 12000
THIN = 1
PROPOSAL_SCALE_INIT = {
    "z": 0.35,
    "alpha": 0.30,
    "beta": 0.25,
    "log_sigma": 0.20,
}


def softmax_with_floor(z: np.ndarray, w_min: float = W_MIN) -> np.ndarray:
    z_shift = z - np.max(z)
    soft = np.exp(z_shift)
    soft = soft / soft.sum()
    return w_min + (1.0 - K * w_min) * soft


def normalise_components(stations: pd.DataFrame) -> pd.DataFrame:
    dock = stations[stations["station_type"] == "docked_bike"].copy()

    def _norm(col: str, invert: bool = False) -> np.ndarray:
        s = dock[col].astype(float).fillna(dock[col].median())
        lo, hi = s.min(), s.max()
        if hi == lo:
            out = np.full(len(s), 0.5)
        else:
            out = (s - lo) / (hi - lo)
        return 1.0 - out if invert else out

    dock["M_norm"] = _norm("gtfs_heavy_stops_300m")
    dock["I_norm"] = _norm("infra_cyclable_pct")
    dock["T_norm"] = _norm("topography_roughness_index", invert=True)
    return dock


def standardise(y: np.ndarray) -> np.ndarray:
    return (y - np.nanmean(y)) / np.nanstd(y)


# ---------------------------------------------------------------------------
# Log-posterior
# ---------------------------------------------------------------------------

def log_posterior(
    state: dict,
    component_city_means: np.ndarray,         # (n_cities, K)
    fub: np.ndarray,                          # (n_cities,) standardised, with nan
    emp: np.ndarray,                          # (n_cities,) standardised, with nan
) -> float:
    z = state["z"]
    alpha = state["alpha"]
    beta = state["beta"]
    log_sigma = state["log_sigma"]

    # Prior on z (Normal(0, sigma_w^2 I))
    lp = -0.5 * np.sum(z ** 2) / SIGMA_W ** 2
    # Prior on alpha, beta (Normal(0, SIGMA_AB^2))
    lp += -0.5 * np.sum(alpha ** 2) / SIGMA_AB ** 2
    lp += -0.5 * np.sum(beta ** 2) / SIGMA_AB ** 2
    # Prior on sigma (Half-Cauchy(scale)); sigma = exp(log_sigma)
    # log p(sigma) = -log(pi * scale) - log(1 + (sigma/scale)^2)
    # Jacobian for log_sigma parameterisation adds + log_sigma
    sigmas = np.exp(log_sigma)
    lp += np.sum(-np.log(np.pi * HALFCAUCHY_SCALE)
                 - np.log(1.0 + (sigmas / HALFCAUCHY_SCALE) ** 2)
                 + log_sigma)

    # Likelihood
    w = softmax_with_floor(z)
    imd_city = component_city_means @ w
    imd_city_std = (imd_city - imd_city.mean()) / imd_city.std()

    for i, y in enumerate([fub, emp]):
        mask = np.isfinite(y)
        if mask.sum() < 5:
            continue
        mu = alpha[i] + beta[i] * imd_city_std[mask]
        residual = y[mask] - mu
        sigma = sigmas[i]
        lp += -0.5 * np.sum(residual ** 2) / sigma ** 2 - mask.sum() * np.log(sigma)

    return lp


# ---------------------------------------------------------------------------
# Metropolis-Hastings sampler
# ---------------------------------------------------------------------------

def mh_sample(
    component_city_means: np.ndarray,
    fub: np.ndarray,
    emp: np.ndarray,
    n_burn: int = N_BURN,
    n_keep: int = N_KEEP,
    thin: int = THIN,
) -> dict:
    state = {
        "z": np.zeros(K),
        "alpha": np.zeros(2),
        "beta": np.array([0.5, 0.5]),
        "log_sigma": np.array([0.0, 0.0]),
    }
    scales = {k: float(v) for k, v in PROPOSAL_SCALE_INIT.items()}
    log_p_current = log_posterior(state, component_city_means, fub, emp)

    samples = {k: [] for k in state}
    accepts = {k: 0 for k in state}
    proposals = {k: 0 for k in state}

    total_iter = n_burn + n_keep * thin
    adapt_every = 200
    target_accept = 0.34

    for it in range(total_iter):
        for key in ("z", "alpha", "beta", "log_sigma"):
            prop_state = {k: v.copy() if hasattr(v, "copy") else v
                          for k, v in state.items()}
            prop_state[key] = state[key] + RNG.normal(0, scales[key],
                                                       size=state[key].shape)
            proposals[key] += 1
            try:
                log_p_prop = log_posterior(prop_state, component_city_means, fub, emp)
            except Exception:
                continue
            if np.log(RNG.uniform()) < log_p_prop - log_p_current:
                state = prop_state
                log_p_current = log_p_prop
                accepts[key] += 1

        if it < n_burn and (it + 1) % adapt_every == 0:
            for k in scales:
                rate = accepts[k] / max(proposals[k], 1)
                # gentle adaptation
                if rate < target_accept * 0.5:
                    scales[k] *= 0.7
                elif rate < target_accept:
                    scales[k] *= 0.9
                elif rate > target_accept * 1.8:
                    scales[k] *= 1.3
                elif rate > target_accept * 1.2:
                    scales[k] *= 1.1
                accepts[k] = 0
                proposals[k] = 0

        if it >= n_burn and (it - n_burn) % thin == 0:
            for k in state:
                samples[k].append(state[k].copy())

    return {
        "z": np.array(samples["z"]),
        "alpha": np.array(samples["alpha"]),
        "beta": np.array(samples["beta"]),
        "log_sigma": np.array(samples["log_sigma"]),
        "final_scales": scales,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Loading panel and stations...")
    panel = load_panel()
    stations = load_stations()
    dock = normalise_components(stations)
    log.info("  %d dock-based stations across %d cities",
             len(dock), dock["city"].nunique())

    # Build the panel-aligned city-mean component matrix used as a calibration
    # design, and the per-station component matrix used downstream.
    city_means = dock.groupby("city")[["M_norm", "I_norm", "T_norm"]].mean()
    cmm = city_means.reindex(panel.cities).fillna(city_means.median())
    component_city_means = cmm.to_numpy()  # (n_cities, 3)

    fub = standardise(panel.fub)
    emp = standardise(np.log1p(panel.emp))
    log.info("  calibration matched: FUB n=%d, EMP n=%d",
             int(np.isfinite(fub).sum()), int(np.isfinite(emp).sum()))

    log.info("Running MH sampler: %d burn + %d keep ...", N_BURN, N_KEEP)
    chain = mh_sample(component_city_means, fub, emp)
    log.info("  final proposal scales: %s",
             {k: round(v, 3) for k, v in chain["final_scales"].items()})

    # Posterior on weights
    z_samples = chain["z"]                            # (n_keep, K)
    w_samples = np.array([softmax_with_floor(z) for z in z_samples])
    w_mean = w_samples.mean(axis=0)
    w_q025 = np.percentile(w_samples, 2.5, axis=0)
    w_q975 = np.percentile(w_samples, 97.5, axis=0)
    log.info("\nPosterior on weights:")
    for k, name in enumerate(["M", "I", "T"]):
        log.info("  w_%s  mean = %.3f  95%% CrI = [%.3f, %.3f]",
                 name, w_mean[k], w_q025[k], w_q975[k])
    # Probability that each component is the dominant one
    arg_dom = np.argmax(w_samples, axis=1)
    p_dominant = [float((arg_dom == k).mean()) for k in range(K)]
    log.info("Posterior P(component dominates): M=%.2f, I=%.2f, T=%.2f",
             p_dominant[0], p_dominant[1], p_dominant[2])

    # Posterior on alpha, beta, sigma
    alpha_samples = chain["alpha"]
    beta_samples = chain["beta"]
    sigma_samples = np.exp(chain["log_sigma"])
    for j, label in enumerate(["FUB", "EMP"]):
        log.info(
            "  beta_%s  mean = %.3f [%.3f, %.3f]   sigma_%s mean = %.3f",
            label, beta_samples[:, j].mean(),
            np.percentile(beta_samples[:, j], 2.5),
            np.percentile(beta_samples[:, j], 97.5),
            label, sigma_samples[:, j].mean(),
        )

    # Per-station IMD posterior: sample 500 weight draws, compute IMD for all stations
    log.info("\nPropagating weight posterior to per-station IMD...")
    sub_idx = RNG.choice(len(w_samples), size=500, replace=False)
    w_sub = w_samples[sub_idx]
    components_station = dock[["M_norm", "I_norm", "T_norm"]].to_numpy()
    # station_imd_samples shape: (500, n_stations)
    station_imd_samples = w_sub @ components_station.T * 100.0
    # Median per station + 95% CI
    station_median = np.median(station_imd_samples, axis=0)
    station_q025 = np.percentile(station_imd_samples, 2.5, axis=0)
    station_q975 = np.percentile(station_imd_samples, 97.5, axis=0)
    dock = dock.assign(imd_post_median=station_median,
                       imd_post_q025=station_q025,
                       imd_post_q975=station_q975)

    # City-level posterior: mean of station IMDs per draw, then summary.
    # station_imd_samples is (500, n_stations); we want the per-draw mean
    # over stations grouped by city.
    city_codes, city_index = pd.factorize(dock["city"].values)
    # Build the (n_cities, n_draws) matrix of city-level mean station IMD
    n_cities = len(city_index)
    n_draws = station_imd_samples.shape[0]
    city_draws = np.zeros((n_cities, n_draws))
    for ci in range(n_cities):
        mask = city_codes == ci
        city_draws[ci] = station_imd_samples[:, mask].mean(axis=1)
    n_stations_per_city = pd.Series(dock.groupby("city").size())
    city_summary = pd.DataFrame({
        "city": city_index.tolist(),
        "imd_post_median": np.median(city_draws, axis=1),
        "imd_post_mean":   city_draws.mean(axis=1),
        "imd_post_q025":   np.percentile(city_draws, 2.5, axis=1),
        "imd_post_q975":   np.percentile(city_draws, 97.5, axis=1),
        "imd_post_sd":     city_draws.std(axis=1),
    })
    city_summary["n_stations"] = (
        city_summary["city"].map(n_stations_per_city).astype(int)
    )
    city_summary = city_summary.sort_values("imd_post_median", ascending=False)
    log.info("Top-10 IMD-3 posterior:")
    for _, r in city_summary.head(10).iterrows():
        log.info(
            "  %-22s  IMD = %.1f  [%.1f, %.1f]  sd = %.1f  n = %d",
            r["city"], r["imd_post_median"],
            r["imd_post_q025"], r["imd_post_q975"],
            r["imd_post_sd"], int(r["n_stations"]),
        )

    # Save results
    results = {
        "model": {
            "K": K, "w_min": W_MIN, "sigma_w": SIGMA_W,
            "n_burn": N_BURN, "n_keep": N_KEEP,
        },
        "weights_posterior": {
            "mean": {k: float(v) for k, v in zip(["M", "I", "T"], w_mean)},
            "q025": {k: float(v) for k, v in zip(["M", "I", "T"], w_q025)},
            "q975": {k: float(v) for k, v in zip(["M", "I", "T"], w_q975)},
            "p_dominant": {k: float(v) for k, v in zip(["M", "I", "T"], p_dominant)},
        },
        "calibration_posterior": {
            "beta_FUB": {"mean": float(beta_samples[:, 0].mean()),
                          "q025": float(np.percentile(beta_samples[:, 0], 2.5)),
                          "q975": float(np.percentile(beta_samples[:, 0], 97.5))},
            "beta_EMP": {"mean": float(beta_samples[:, 1].mean()),
                          "q025": float(np.percentile(beta_samples[:, 1], 2.5)),
                          "q975": float(np.percentile(beta_samples[:, 1], 97.5))},
            "sigma_FUB_mean": float(sigma_samples[:, 0].mean()),
            "sigma_EMP_mean": float(sigma_samples[:, 1].mean()),
        },
        "city_summary": city_summary.to_dict("records"),
    }
    out_json = OUT_DIR / "b1_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # ---- Figure 1: weights posterior (3-panel marginal + corner-style) ----
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    labels = ["$w_M$ multimodality", "$w_I$ infrastructure", "$w_T$ topography"]
    colors = ["#1F3A6B", "#7095C8", "#5B7E4F"]
    for k in range(K):
        ax = axes[k]
        ax.hist(w_samples[:, k], bins=40, color=colors[k],
                edgecolor="white", linewidth=0.4, alpha=0.85)
        ax.axvline(w_mean[k], color="#A8201A", linewidth=1.0)
        ax.axvline(w_q025[k], color="#A8201A", linewidth=0.6, linestyle=":")
        ax.axvline(w_q975[k], color="#A8201A", linewidth=0.6, linestyle=":")
        ax.set_xlabel(labels[k])
        ax.set_ylabel("Posterior density")
        ax.set_title(
            f"mean = {w_mean[k]:.2f}, 95% CrI [{w_q025[k]:.2f}, {w_q975[k]:.2f}]",
            fontsize=9,
        )
        ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    fig.suptitle("B1: posterior on IMD-3 weights (MH, "
                 f"$N_{{\\mathrm{{keep}}}}={N_KEEP}$)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "b1_weights_posterior.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote b1_weights_posterior.pdf")

    # ---- Figure 2: Top-20 cities with 95% CI ----
    top20 = city_summary.head(20)
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    y_pos = np.arange(len(top20))
    err = np.vstack([
        top20["imd_post_median"] - top20["imd_post_q025"],
        top20["imd_post_q975"] - top20["imd_post_median"],
    ])
    ax.errorbar(top20["imd_post_median"], y_pos, xerr=err,
                fmt="o", color="#1F3A6B", ecolor="#404040",
                capsize=3, markersize=5, elinewidth=0.9, capthick=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{c} (n={int(n)})" for c, n in zip(top20["city"], top20["n_stations"])],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_xlabel(r"Bayesian IMD-3 (95\% credible interval, "
                  r"$N_{\mathrm{keep}}=$" + f"{N_KEEP})")
    ax.set_title("Top-20 cities by posterior IMD-3", fontsize=10)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "b1_top_ranking_ci.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote b1_top_ranking_ci.pdf")

    # ---- Figure 3: per-station IMD distributions in 4 example cities ----
    examples = ["Strasbourg", "Paris", "Lyon", "Rennes"]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.4))
    axes = axes.flatten()
    for i, city in enumerate(examples):
        ax = axes[i]
        sub = dock[dock["city"] == city]
        if len(sub) == 0:
            ax.set_title(f"{city} (no stations)", fontsize=9)
            continue
        ax.hist(sub["imd_post_median"], bins=20, color="#1F3A6B",
                edgecolor="white", linewidth=0.4, alpha=0.85)
        ax.axvline(sub["imd_post_median"].median(), color="#A8201A",
                    linewidth=1.0)
        ax.set_title(f"{city}  (n = {len(sub)} stations)", fontsize=10)
        ax.set_xlabel("Station-level IMD-3 (posterior median)")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    fig.suptitle("B1: within-city distribution of station IMD-3", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "b1_per_station_distribution_examples.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote b1_per_station_distribution_examples.pdf")


if __name__ == "__main__":
    main()
