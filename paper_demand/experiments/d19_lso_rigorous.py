"""
d19_lso_rigorous.py — Rigorous Leave-Station-Out validation with three
spatial models, paired bootstrap CIs and a spatial-block robustness check.

Improvements over d18:
  (1) Three spatial models compared (instead of two):
        G^-   : temporal-only (no spatial signal)         null A
        G_FE  : temporal + station_id one-hot (fixed-effect)  null B
        G     : temporal + IMD-4                              treatment
      G_FE is the critical comparison: a station fixed-effect carries
      full station-specific information on training stations and zero
      on held-out stations.  If G beats G_FE out-of-station, the IMD
      features carry transferable spatial information that no per-station
      dummy can recover.
  (2) Paired station-bootstrap 95% CIs on Spearman rho, Kendall tau and
      Precision@K, with the bootstrap resampling stations (the unit at
      which the LSO test is conducted), not (station,hour) bins.
  (3) Statistical tests on the paired difference of Spearman rho between
      the three models, with B=1000 paired bootstrap replicates.
  (4) Hypergeometric null for Precision@K (instead of K/n) :
        E[P@K | random] = K/N exactly,
        Var[P@K | random] = (K(N-K))/(N^2 (N-1))
  (5) Spatial-block fold construction (alternative to the random folds of
      d18) — folds defined by 2D k-means clustering of station coordinates
      to control for spatial autocorrelation between neighbouring train
      and held-out stations.

Outputs:
  outputs/d19_lso_<city>.json                  (consolidated metrics)
  outputs/d19_lso_<city>_per_station.csv       (per-station test predictions)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, kendalltau, hypergeom

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TIER1_DIR = ROOT / "data_collection" / "tier1_trip_logs"
IMD_INTL_DIR = ROOT / "data_collection" / "imd_international"
OUT = ROOT / "experiments" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from d3_multicity_benchmark import build_panel, FEATS_T, FEATS_IMD  # noqa: E402

LGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    min_child_samples=30, reg_lambda=0.5, random_state=42,
    n_jobs=-1, verbose=-1,
)

N_BOOT = 1000
SEED = 42


# ── Metric definitions ────────────────────────────────────────────────────────
def precision_at_k(true_sorted, pred_sorted, k: int) -> float:
    return len(set(true_sorted[:k]) & set(pred_sorted[:k])) / k


def hypergeom_null_pk(N: int, K: int) -> tuple[float, float, float]:
    """Hypergeometric null for Precision@K when predictions are uniformly
    random : draw K predicted top-K out of N, count overlap with K true
    top-K.  Returns (mean, std, 95th percentile)."""
    rv = hypergeom(N, K, K)
    mean = rv.mean() / K
    std = rv.std() / K
    p95 = rv.ppf(0.95) / K
    return float(mean), float(std), float(p95)


def station_bootstrap_ci(per_station: pd.DataFrame, rng: np.random.Generator,
                         B: int = N_BOOT) -> dict:
    """Paired station-bootstrap on ranking metrics.  For each of B replicates
    we sample stations with replacement and recompute (rho, tau, P@K) for
    each spatial model on the resample.  Returns point estimate + 95% CI
    for each model and for the paired difference (G - G_FE, G - G^-)."""
    n = len(per_station)
    rho_g, rho_fe, rho_no = [], [], []
    tau_g, tau_fe, tau_no = [], [], []
    dr_g_vs_fe, dr_g_vs_no = [], []
    pk_g, pk_fe, pk_no = {5: [], 10: [], 20: [], 50: []}, \
                         {5: [], 10: [], 20: [], 50: []}, \
                         {5: [], 10: [], 20: [], 50: []}

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        sub = per_station.iloc[idx]
        # Spearman / Kendall on the resample (paired across models)
        r_g,  _ = spearmanr(sub["true_mean"], sub["pred_imd_mean"])
        r_fe, _ = spearmanr(sub["true_mean"], sub["pred_fe_mean"])
        r_no, _ = spearmanr(sub["true_mean"], sub["pred_no_imd_mean"])
        t_g,  _ = kendalltau(sub["true_mean"], sub["pred_imd_mean"])
        t_fe, _ = kendalltau(sub["true_mean"], sub["pred_fe_mean"])
        t_no, _ = kendalltau(sub["true_mean"], sub["pred_no_imd_mean"])
        rho_g.append(r_g); rho_fe.append(r_fe); rho_no.append(r_no)
        tau_g.append(t_g); tau_fe.append(t_fe); tau_no.append(t_no)
        dr_g_vs_fe.append(r_g - r_fe)
        dr_g_vs_no.append(r_g - r_no)
        for K in (5, 10, 20, 50):
            if K > n: continue
            ord_obs = sub.sort_values("true_mean",         ascending=False)["station_id"].tolist()
            ord_g   = sub.sort_values("pred_imd_mean",     ascending=False)["station_id"].tolist()
            ord_fe  = sub.sort_values("pred_fe_mean",      ascending=False)["station_id"].tolist()
            ord_no  = sub.sort_values("pred_no_imd_mean",  ascending=False)["station_id"].tolist()
            pk_g[K].append(precision_at_k(ord_obs, ord_g,  K))
            pk_fe[K].append(precision_at_k(ord_obs, ord_fe, K))
            pk_no[K].append(precision_at_k(ord_obs, ord_no, K))

    def ci(samples):
        return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))

    out = {
        "rho_imd":   dict(mean=float(np.mean(rho_g)),  ci=ci(rho_g)),
        "rho_fe":    dict(mean=float(np.mean(rho_fe)), ci=ci(rho_fe)),
        "rho_no":    dict(mean=float(np.mean(rho_no)), ci=ci(rho_no)),
        "tau_imd":   dict(mean=float(np.mean(tau_g)),  ci=ci(tau_g)),
        "tau_fe":    dict(mean=float(np.mean(tau_fe)), ci=ci(tau_fe)),
        "tau_no":    dict(mean=float(np.mean(tau_no)), ci=ci(tau_no)),
        "delta_rho_imd_vs_fe":  dict(mean=float(np.mean(dr_g_vs_fe)), ci=ci(dr_g_vs_fe)),
        "delta_rho_imd_vs_no":  dict(mean=float(np.mean(dr_g_vs_no)), ci=ci(dr_g_vs_no)),
        "precision_at_k": {},
    }
    for K in (5, 10, 20, 50):
        if K > n: continue
        out["precision_at_k"][f"K{K}"] = dict(
            imd=dict(mean=float(np.mean(pk_g[K])), ci=ci(pk_g[K])),
            fe=dict(mean=float(np.mean(pk_fe[K])), ci=ci(pk_fe[K])),
            no_imd=dict(mean=float(np.mean(pk_no[K])), ci=ci(pk_no[K])),
        )
    return out


# ── Fold construction ─────────────────────────────────────────────────────────
def random_folds(stations: list[str], n_folds: int, rng) -> list[list[str]]:
    perm = list(stations); rng.shuffle(perm)
    return [list(a) for a in np.array_split(perm, n_folds)]


def spatial_folds(stations_df: pd.DataFrame, n_folds: int, seed: int) -> list[list[str]]:
    """K-means on (lat, lng) to define spatial folds — each fold is a
    contiguous geographical cluster, controlling for spatial autocorrelation
    between train and held-out stations."""
    pts = stations_df[["lat", "lng"]].values
    km = KMeans(n_clusters=n_folds, random_state=seed, n_init=10)
    labels = km.fit_predict(pts)
    folds = []
    for f in range(n_folds):
        folds.append(stations_df.loc[labels == f, "station_id"].astype(str).tolist())
    return folds


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run(city: str, n_folds: int = 5, mode: str = "random",
        seed: int = SEED):
    t_start = time.time()
    rng = np.random.default_rng(seed)

    imd_path = IMD_INTL_DIR / f"{city}.parquet"
    if not imd_path.exists():
        print(f"✗ {imd_path} missing"); return
    imd = pd.read_parquet(imd_path)
    imd["station_id"] = imd["station_id"].astype(str)
    print(f"IMD : {len(imd)} stations  ({imd_path.name})")

    print(f"Building trip panel from {TIER1_DIR / city}...")
    panel = build_panel(city)
    if panel.empty:
        print("✗ Empty panel"); return
    print(f"Panel : {len(panel):,} bins  {panel['demande'].sum():,} trips")

    df = panel.merge(imd[["station_id", "lat", "lng"] + FEATS_IMD],
                     on="station_id", how="left")
    df = df.dropna(subset=FEATS_IMD).copy()
    df["hour"] = df["datetime_hour"].dt.hour
    df["day_of_week"] = df["datetime_hour"].dt.dayofweek
    df["month"] = df["datetime_hour"].dt.month
    df["log_demand"] = np.log1p(df["demande"])

    span = df.groupby("station_id")["datetime_hour"].agg(["min", "max"])
    span["months"] = (span["max"] - span["min"]).dt.days / 30
    keep = span[span["months"] >= 6].index.tolist()
    df = df[df["station_id"].isin(keep)].copy()
    print(f"After IMD merge / >=6 months filter: {len(df):,} bins  "
          f"{df['station_id'].nunique()} stations")

    stations_df = (df.groupby("station_id")[["lat", "lng"]].first()
                     .reset_index())

    if mode == "spatial":
        folds = spatial_folds(stations_df, n_folds, seed)
    else:
        folds = random_folds(sorted(stations_df["station_id"]), n_folds, rng)
    print(f"Mode={mode}  folds: {[len(f) for f in folds]}")

    # Encode station_id as integer code (for fixed-effect model)
    station_id_to_code = {s: i for i, s in enumerate(stations_df["station_id"])}
    df["station_code"] = df["station_id"].map(station_id_to_code)

    per_station_blocks = []
    fold_summary = []
    for fi, fold in enumerate(folds):
        fold_set = set(fold)
        train = df[~df["station_id"].isin(fold_set)]
        test = df[df["station_id"].isin(fold_set)].copy()
        if len(test) == 0 or len(train) == 0: continue
        print(f"\n=== Fold {fi+1}/{n_folds} ({mode}): "
              f"{train['station_id'].nunique()} train, "
              f"{len(fold_set)} holdout, n_test={len(test):,} ===")

        def fit(features, name, cat=None):
            m = lgb.LGBMRegressor(**LGB_PARAMS)
            t0 = time.time()
            kwargs = {}
            # Pass DataFrame so LightGBM resolves categorical column NAMES.
            X_train = train[features].copy()
            X_test = test[features].copy()
            if cat:
                for c in cat:
                    X_train[c] = X_train[c].astype("category")
                    X_test[c] = X_test[c].astype("category")
                kwargs["categorical_feature"] = cat
            else:
                X_train = X_train.astype("float64")
                X_test = X_test.astype("float64")
            m.fit(X_train, train["log_demand"].values, **kwargs)
            ys = m.predict(X_test)
            yt = test["log_demand"].values
            yt_trips = np.expm1(yt); yp_trips = np.expm1(np.clip(ys, 0, None))
            return ys, yp_trips, time.time() - t0

        # G^- : temporal only
        ys_no,  yp_no,  ts_no  = fit(FEATS_T,                "Gminus")
        # G_FE : temporal + station_code as categorical
        # Held-out stations have a station_code never seen at training, so
        # LightGBM falls back to the global mean for them (the desired
        # "fixed-effect = useless on new stations" behaviour).
        ys_fe,  yp_fe,  ts_fe  = fit(FEATS_T + ["station_code"], "GFE",
                                       cat=["station_code"])
        # G  : temporal + IMD-4
        ys_imd, yp_imd, ts_imd = fit(FEATS_T + FEATS_IMD,    "G")

        test["pred_no_imd"] = yp_no
        test["pred_fe"] = yp_fe
        test["pred_imd"] = yp_imd

        agg = (test.groupby("station_id")
                  .agg(n_bins=("demande", "size"),
                       true_mean=("demande", "mean"),
                       pred_no_imd_mean=("pred_no_imd", "mean"),
                       pred_fe_mean=("pred_fe", "mean"),
                       pred_imd_mean=("pred_imd", "mean"))
                  .reset_index())
        agg["fold"] = fi + 1
        per_station_blocks.append(agg)

        # Hourly R^2 (out-of-station)
        yt = test["log_demand"].values
        yt_trips = np.expm1(yt)
        r2_no  = r2_score(yt_trips, yp_no)
        r2_fe  = r2_score(yt_trips, yp_fe)
        r2_imd = r2_score(yt_trips, yp_imd)
        print(f"  R²: G^-={r2_no:+.4f}  G_FE={r2_fe:+.4f}  G={r2_imd:+.4f}    "
              f"ΔR²(G vs G^-)={r2_imd-r2_no:+.4f}  "
              f"ΔR²(G vs G_FE)={r2_imd-r2_fe:+.4f}")
        fold_summary.append(dict(fold=fi+1, n_holdout=len(fold_set),
                                  n_test=int(len(test)),
                                  r2_hourly_no_imd=r2_no,
                                  r2_hourly_fe=r2_fe,
                                  r2_hourly_imd=r2_imd))

    ps = pd.concat(per_station_blocks, ignore_index=True)
    ps.to_csv(OUT / f"d19_lso_{city}_{mode}_per_station.csv", index=False)

    # Aggregate hourly R^2 across folds (mean and CI from per-fold values)
    r2_vals_no  = [f["r2_hourly_no_imd"] for f in fold_summary]
    r2_vals_fe  = [f["r2_hourly_fe"]     for f in fold_summary]
    r2_vals_imd = [f["r2_hourly_imd"]    for f in fold_summary]

    # Point estimates of ranking metrics
    ord_obs = ps.sort_values("true_mean",        ascending=False)["station_id"].tolist()
    ord_g   = ps.sort_values("pred_imd_mean",    ascending=False)["station_id"].tolist()
    ord_fe  = ps.sort_values("pred_fe_mean",     ascending=False)["station_id"].tolist()
    ord_no  = ps.sort_values("pred_no_imd_mean", ascending=False)["station_id"].tolist()

    rho_g_pt,  _ = spearmanr(ps["true_mean"], ps["pred_imd_mean"])
    rho_fe_pt, _ = spearmanr(ps["true_mean"], ps["pred_fe_mean"])
    rho_no_pt, _ = spearmanr(ps["true_mean"], ps["pred_no_imd_mean"])
    tau_g_pt,  _ = kendalltau(ps["true_mean"], ps["pred_imd_mean"])

    print(f"\n=== Point estimates ({mode} folds) ===")
    print(f"Spearman ρ:  G={rho_g_pt:+.3f}  G_FE={rho_fe_pt:+.3f}  G^-={rho_no_pt:+.3f}")
    print(f"Kendall  τ:  G={tau_g_pt:+.3f}")

    # Hypergeometric null for Precision@K
    N = len(ps)
    pk_null = {K: hypergeom_null_pk(N, K) for K in (5, 10, 20, 50) if K <= N}

    # Bootstrap CIs
    print(f"\nRunning station-bootstrap (B={N_BOOT})...")
    boot = station_bootstrap_ci(ps, rng, B=N_BOOT)
    print(f"Spearman ρ (IMD):    point={rho_g_pt:+.3f}, "
          f"95% CI = [{boot['rho_imd']['ci'][0]:+.3f}, {boot['rho_imd']['ci'][1]:+.3f}]")
    print(f"Spearman ρ (G_FE):   point={rho_fe_pt:+.3f}, "
          f"95% CI = [{boot['rho_fe']['ci'][0]:+.3f}, {boot['rho_fe']['ci'][1]:+.3f}]")
    print(f"Δρ (IMD - G_FE):     mean={boot['delta_rho_imd_vs_fe']['mean']:+.3f}, "
          f"95% CI = [{boot['delta_rho_imd_vs_fe']['ci'][0]:+.3f}, "
          f"{boot['delta_rho_imd_vs_fe']['ci'][1]:+.3f}]")
    print(f"Δρ (IMD - G^-):      mean={boot['delta_rho_imd_vs_no']['mean']:+.3f}, "
          f"95% CI = [{boot['delta_rho_imd_vs_no']['ci'][0]:+.3f}, "
          f"{boot['delta_rho_imd_vs_no']['ci'][1]:+.3f}]")
    print()
    for K, (mu, sd, p95) in pk_null.items():
        print(f"P@{K:>2d}: G={boot['precision_at_k'][f'K{K}']['imd']['mean']:.3f} "
              f"[{boot['precision_at_k'][f'K{K}']['imd']['ci'][0]:.3f},"
              f"{boot['precision_at_k'][f'K{K}']['imd']['ci'][1]:.3f}]  "
              f"G_FE={boot['precision_at_k'][f'K{K}']['fe']['mean']:.3f}  "
              f"G^-={boot['precision_at_k'][f'K{K}']['no_imd']['mean']:.3f}  "
              f"null E[P@{K}]={mu:.3f} ±{sd:.3f}")

    metrics = {
        "city": city, "mode": mode, "n_folds": n_folds,
        "n_stations_evaluated": int(N),
        "wall_time_s": round(time.time() - t_start, 1),
        "r2_hourly": {
            "no_imd_per_fold": r2_vals_no,
            "fe_per_fold":     r2_vals_fe,
            "imd_per_fold":    r2_vals_imd,
            "no_imd_mean": float(np.mean(r2_vals_no)),
            "fe_mean":     float(np.mean(r2_vals_fe)),
            "imd_mean":    float(np.mean(r2_vals_imd)),
            "imd_vs_no_imd_delta_mean": float(np.mean(r2_vals_imd) - np.mean(r2_vals_no)),
            "imd_vs_fe_delta_mean":     float(np.mean(r2_vals_imd) - np.mean(r2_vals_fe)),
        },
        "rho_point": {"imd": float(rho_g_pt),  "fe": float(rho_fe_pt), "no_imd": float(rho_no_pt)},
        "tau_point": {"imd": float(tau_g_pt)},
        "bootstrap": boot,
        "hypergeometric_null_P_at_K": {f"K{K}": dict(mean=mu, std=sd, p95=p95)
                                       for K, (mu, sd, p95) in pk_null.items()},
        "per_fold": fold_summary,
    }
    with open(OUT / f"d19_lso_{city}_{mode}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Saved {OUT / f'd19_lso_{city}_{mode}.json'}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", default="boston_bluebikes")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--mode", choices=("random", "spatial"), default="random")
    args = p.parse_args()
    run(args.city, args.folds, args.mode)


if __name__ == "__main__":
    main()
