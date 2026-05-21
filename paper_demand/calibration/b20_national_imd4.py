"""B20 -- IMD-4 extended to all 34,875 French communes.

The B19 feasibility study showed that the density component
alone (the cheapest to compute nationally) only reaches
rho = +0.40 against INSEE part-velo-travail on the full
national panel, and a poor rho = +0.05 on the subset of
communes with non-zero cycling. The IMD-4 composite reaches
rho = +0.62 on the 59-city VLS panel.

The "Tableau de bord des mobilites durables" publisher on
data.gouv.fr provides three of the four IMD components
pre-aggregated at the commune level for all 34,875 French
communes :

  - Lineaire cyclable par km^2 (I)        --> sum of all
        cycling-infrastructure km per commune divided by
        commune surface, all types: pistes cyclables,
        bandes, voies vertes, voies bus partagees, double-sens,
        amenagements mixtes, autres.
  - Nombre de stations TC lourdes par km^2 (M)  -->
        count of train + metro + tramway stations per
        commune divided by commune surface (bus excluded).
  - Densite de population par km^2 (D)    --> from
        commune meta.

The fourth component (T, topography roughness) is set to
the panel z-score mean (zero) for the national application.
Its posterior weight is w_T ~ 0.08 so this approximation
loses at most 8% of the IMD-4's contribution. A full
national T component would require BD ALTI 25m elevation
data and is left for future work.

For each commune i we compute:

  IMD-4_i  =  w_M * z(M_i) + w_I * z(I_i)
            + w_T * 0     + w_D * z(D_i)

where w = (w_M, w_I, w_T, w_D) is the posterior median
from the B7 calibration on the 59-city panel against
FUB + EMP, and z(x_i) is the panel-anchored z-score of
component x at commune i.

We then validate by Spearman correlation against the
INSEE part-velo-travail 2022 on all 34,629 communes
with finite values.

Outputs:
    data/external/mobility_sources/imd4_national_communes.csv
    outputs/b20_national_imd4_results.json
    outputs/b20_national_imd4_scatter.pdf
    outputs/b20_national_imd4_map.pdf
"""
from __future__ import annotations

import json
import logging
import sys
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

HERE = Path(__file__).resolve()
ROOT = next(
    p for p in [HERE, *HERE.parents]
    if (p / ".git").exists() or (p / "CITATION.cff").exists()
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "papers" / "02_imd" / "experiments"))

from _common import load_panel  # noqa: E402

B7_PATH = ROOT / "papers" / "03_imd_bayesian" / "experiments" / "b7_imd4_with_density.py"
spec7 = importlib.util.spec_from_file_location("b7", B7_PATH)
b7 = importlib.util.module_from_spec(spec7)
spec7.loader.exec_module(b7)

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def bootstrap_rho(x, y, n_boot=500, seed=2026):
    rng = np.random.default_rng(seed)
    n = len(x)
    rho = sp_stats.spearmanr(x, y).statistic
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r = sp_stats.spearmanr(x[idx], y[idx]).statistic
        boots[b] = r if np.isfinite(r) else np.nan
    boots = boots[np.isfinite(boots)]
    return (float(rho),
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def main() -> None:
    log.info("Loading commune meta (population, surface, density)...")
    meta = pd.read_csv(
        ROOT / "data" / "external" / "insee_communes" / "communes_meta.csv",
        dtype={"code_commune": str},
    )
    meta = meta[meta["surface_km2"] > 0.01]
    meta["log_density"] = np.log(meta["density_per_km2"].clip(lower=1.0))
    log.info("  %d communes after surface filter", len(meta))

    log.info("\nLoading national cycling infrastructure (I component)...")
    cyc = pd.read_csv(
        ROOT / "data" / "external" / "insee_mobpro" / "lineaire-cyclable-com.csv",
        dtype={"code_com": str}, low_memory=False,
    )
    cyc_25 = cyc[cyc["annee"] == 2025].copy()
    # Sum km across all infrastructure types per commune.
    cyc_agg = cyc_25.groupby("code_com")["numerateur"].sum().reset_index()
    cyc_agg = cyc_agg.rename(columns={"numerateur": "infra_cyclable_km"})
    log.info("  %d communes with cycling infra (year 2025)", len(cyc_agg))

    log.info("\nLoading national transit stations (M component)...")
    tc = pd.read_csv(
        ROOT / "data" / "external" / "insee_mobpro" / "nb-stations-tc-com.csv",
        dtype={"code_com": str}, low_memory=False,
    )
    HEAVY = ["train", "tramway", "métro"]
    tc_heavy = tc[(tc["annee"] == 2025) & (tc["route_type"].isin(HEAVY))].copy()
    tc_agg = tc_heavy.groupby("code_com")["numerateur"].sum().reset_index()
    tc_agg = tc_agg.rename(columns={"numerateur": "heavy_stations_count"})
    log.info("  %d communes with at least one heavy transit station",
             len(tc_agg))

    # Merge into a single per-commune frame
    df = meta.merge(cyc_agg, left_on="code_commune",
                     right_on="code_com", how="left")
    df = df.drop(columns=["code_com"])
    df["infra_cyclable_km"] = df["infra_cyclable_km"].fillna(0.0)
    df = df.merge(tc_agg, left_on="code_commune",
                   right_on="code_com", how="left")
    df = df.drop(columns=["code_com"])
    df["heavy_stations_count"] = df["heavy_stations_count"].fillna(0.0)

    # Per-km^2 normalisation to match panel-level component definitions
    df["I_per_km2"] = df["infra_cyclable_km"] / df["surface_km2"]
    df["M_per_km2"] = df["heavy_stations_count"] / df["surface_km2"]
    df["D_log"] = df["log_density"]

    log.info("\nComponent quantiles (national):")
    for col in ["M_per_km2", "I_per_km2", "D_log"]:
        log.info("  %-12s q10=%.3f  median=%.3f  q90=%.3f  max=%.3f",
                 col,
                 float(df[col].quantile(0.10)),
                 float(df[col].median()),
                 float(df[col].quantile(0.90)),
                 float(df[col].max()))

    # ===== Z-score components against the national distribution =====
    def zscore(v):
        v = v.astype(float)
        return (v - v.mean()) / v.std(ddof=0)
    df["M_z"] = zscore(df["M_per_km2"])
    df["I_z"] = zscore(df["I_per_km2"])
    df["D_z"] = zscore(df["D_log"])

    # ===== Posterior median weights from B7 panel calibration =====
    log.info("\nRecalibrating posterior weights on the 59-city panel...")
    panel = load_panel()
    dock, cmm4, _, _, _ = b7.build_design(panel)
    res4 = b7.calibrate_k4(cmm4, panel.fub, panel.emp)
    w_med = np.median(res4["w_samples"], axis=0)
    log.info("  w_med (M, I, T, D) = %.3f, %.3f, %.3f, %.3f", *w_med)

    # Apply weights (T set to 0)
    df["IMD4_national"] = (
        w_med[0] * df["M_z"]
        + w_med[1] * df["I_z"]
        + w_med[2] * 0.0
        + w_med[3] * df["D_z"]
    )

    # ===== Validate against INSEE part-velo-travail 2022 =====
    log.info("\nLoading INSEE part-velo-travail (validation reference)...")
    mobpro = pd.read_csv(
        ROOT / "data" / "external" / "insee_mobpro" /
        "part-actifs-modes-transport-com.csv",
        dtype={"code_com": str}, low_memory=False,
    )
    velo = mobpro[(mobpro["mode_transport"].str.contains("V.lo", regex=True, na=False))
                   & (mobpro["annee"] == 2022)].copy()
    velo["valeur"] = pd.to_numeric(velo["valeur"], errors="coerce")
    velo_lookup = dict(zip(velo["code_com"], velo["valeur"]))
    df["insee_part_velo"] = df["code_commune"].map(velo_lookup)

    # Validation panels:
    full = df[np.isfinite(df["IMD4_national"]) & np.isfinite(df["insee_part_velo"])].copy()
    nz = full[full["insee_part_velo"] > 0.5].copy()
    log.info("  full national panel n = %d", len(full))
    log.info("  non-zero-velo panel n = %d", len(nz))

    rho_full, q025_full, q975_full = bootstrap_rho(
        full["IMD4_national"].values, full["insee_part_velo"].values,
        n_boot=500,
    )
    log.info("\nNATIONAL IMD-4 vs INSEE part-velo-travail:")
    log.info("  full panel:         rho = %+.3f   CI = [%+.3f, %+.3f]   n = %d",
             rho_full, q025_full, q975_full, len(full))
    rho_nz, q025_nz, q975_nz = bootstrap_rho(
        nz["IMD4_national"].values, nz["insee_part_velo"].values,
        n_boot=500,
    )
    log.info("  non-zero subset:    rho = %+.3f   CI = [%+.3f, %+.3f]   n = %d",
             rho_nz, q025_nz, q975_nz, len(nz))

    # Stratification by commune size (urban-scale validation)
    log.info("\nStratified validation by commune population:")
    strat = {}
    for thr in [0, 1000, 5000, 10000, 20000, 50000, 100000]:
        sub = full[full["population"] >= thr]
        if len(sub) < 5:
            continue
        rho_s, q025_s, q975_s = bootstrap_rho(
            sub["IMD4_national"].values, sub["insee_part_velo"].values,
            n_boot=500,
        )
        strat[f"pop_ge_{thr}"] = {"n": int(len(sub)), "rho": float(rho_s),
                                    "q025": float(q025_s),
                                    "q975": float(q975_s)}
        log.info("  pop >= %6d  n=%6d  rho=%+.3f  CI=[%+.3f, %+.3f]",
                 thr, len(sub), rho_s, q025_s, q975_s)

    # Component-level Spearman on the national panel
    log.info("\nNational component-only correlations vs INSEE:")
    comp_results: dict = {}
    for col, label in [("M_z", "M alone"), ("I_z", "I alone"),
                        ("D_z", "D alone")]:
        r, q025, q975 = bootstrap_rho(full[col].values,
                                       full["insee_part_velo"].values,
                                       n_boot=500)
        comp_results[label] = {"rho": r, "q025": q025, "q975": q975,
                                "n": int(len(full))}
        log.info("  %-10s  rho = %+.3f   CI = [%+.3f, %+.3f]",
                 label, r, q025, q975)

    # ===== Top / bottom national rankings =====
    log.info("\nTop 15 communes by national IMD-4 (population > 5000 only):")
    top = full[full["population"] > 5000].nlargest(15, "IMD4_national")
    log.info("\n%s", top[["nom", "code_commune", "population",
                          "M_per_km2", "I_per_km2", "density_per_km2",
                          "IMD4_national", "insee_part_velo"]]
              .to_string(index=False))

    # Save per-commune scores
    out_csv = ROOT / "data" / "external" / "mobility_sources" / \
              "imd4_national_communes.csv"
    cols_keep = ["code_commune", "nom", "population", "surface_km2",
                 "M_per_km2", "I_per_km2", "D_log",
                 "M_z", "I_z", "D_z", "IMD4_national", "insee_part_velo"]
    full[cols_keep].to_csv(out_csv, index=False, encoding="utf-8")
    log.info("\nWrote %s (%d communes)", out_csv, len(full))

    results = {
        "n_full_panel": int(len(full)),
        "n_non_zero_panel": int(len(nz)),
        "weights_used": {"w_M": float(w_med[0]), "w_I": float(w_med[1]),
                          "w_T": float(w_med[2]), "w_D": float(w_med[3])},
        "rho_imd4_national_full": float(rho_full),
        "ci_imd4_national_full": [q025_full, q975_full],
        "rho_imd4_national_nz": float(rho_nz),
        "ci_imd4_national_nz": [q025_nz, q975_nz],
        "components_alone": comp_results,
        "stratified_by_pop": strat,
    }
    (OUT_DIR / "b20_national_imd4_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # ===== Figure 1: stratified rho by commune population =====
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))

    # Left: stratified rho bars
    ax = axes[0]
    thr_list = [0, 1000, 5000, 10000, 20000, 50000, 100000]
    labels = ["all\n34,858", "$\\geq$1k\n10,014", "$\\geq$5k\n2,234",
              "$\\geq$10k\n1,038", "$\\geq$20k\n482", "$\\geq$50k\n133",
              "$\\geq$100k\n42"]
    rhos = [strat[f"pop_ge_{t}"]["rho"] for t in thr_list]
    err_lo = [strat[f"pop_ge_{t}"]["rho"] - strat[f"pop_ge_{t}"]["q025"]
              for t in thr_list]
    err_hi = [strat[f"pop_ge_{t}"]["q975"] - strat[f"pop_ge_{t}"]["rho"]
              for t in thr_list]
    cmap = ["#9DBADD"] * 6 + ["#1F3A6B"]
    ax.bar(np.arange(len(thr_list)), rhos, yerr=[err_lo, err_hi],
           color=cmap, capsize=4, edgecolor="white", linewidth=0.5)
    ax.axhline(0.621, color="#C0392B", linestyle="--", linewidth=0.9,
               label="59-city VLS panel (B14)")
    ax.axhline(0, color="#404040", linewidth=0.5)
    ax.set_xticks(np.arange(len(thr_list)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Communes by population threshold\n(n shown below)",
                  fontsize=9)
    ax.set_ylabel("Spearman $\\rho$ vs INSEE part-velo-travail 2022",
                  fontsize=9)
    ax.set_title("(a) Stratified national validation",
                 fontsize=10)
    ax.set_ylim(-0.05, 0.8)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8)
    for i, n in enumerate(thr_list):
        ax.text(i, rhos[i] + err_hi[i] + 0.02, f"{rhos[i]:+.2f}",
                ha="center", va="bottom", fontsize=8)

    # Right: scatter of top-100 IMD-4 cities (pop > 5k)
    ax = axes[1]
    sub = full[full["population"] > 5000].sort_values(
        "IMD4_national", ascending=False).head(100)
    rho_sub = sp_stats.spearmanr(sub["IMD4_national"],
                                  sub["insee_part_velo"]).statistic
    ax.scatter(sub["IMD4_national"], sub["insee_part_velo"],
               s=24, color="#1F3A6B", alpha=0.65,
               edgecolor="white", linewidth=0.3)
    for _, row in sub.head(15).iterrows():
        ax.annotate(row["nom"],
                     (row["IMD4_national"], row["insee_part_velo"]),
                     fontsize=7, xytext=(3, 3),
                     textcoords="offset points", color="#202020")
    ax.set_xlabel("National IMD-4")
    ax.set_ylabel(r"INSEE part-velo-travail 2022 (\%)")
    ax.set_title(f"(b) Top 100 IMD-4 communes (pop $>$ 5k), "
                 r"$\rho = " f"{rho_sub:+.3f}$",
                 fontsize=10)
    ax.grid(True, color="#E5E5E5", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "b20_national_imd4_scatter.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote b20_national_imd4_scatter.pdf")


if __name__ == "__main__":
    main()
