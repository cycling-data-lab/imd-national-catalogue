"""
d13_forest_plot.py — Forest plot of the 24-city per-network IMD-4 gain.

Reads the consolidated bootstrap CI table d11_bootstrap_ci.csv and
produces a forest plot (one row per city, ordered Tier 1 then Tier 2 by
decreasing dR^2) with horizontal CI bars.

Output:
  paper_demand/figures/fig_multicity_forest.pdf
  paper_demand/figures/fig_multicity_forest.png  (raster fallback for slides)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[0] / "outputs"
FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PRETTY = {
    "velomagg": "Vélomagg Montpellier",
    "dc_capitalbikeshare": "Capital Bikeshare DC",
    "chicago_divvy": "Divvy Chicago",
    "boston_bluebikes": "Bluebikes Boston",
    "sf_baywheels": "Bay Wheels SF",
    "Paris": "Vélib Paris",
    "lyon": "Vélo'v Lyon",
    "toulouse": "VélÔToulouse",
    "levelo_inurba_marseille": "LeVélo Marseille",
    "velo-tbm-bordeaux": "TBM Vélo Bordeaux",
    "velivert_saint_etienne": "Vélivert Saint-Étienne",
    "velonecy60minutes_annecy": "Vélonecy Annecy",
    "nantes": "Naolib Nantes",
    "vilvolt_epinal": "Vilvolt Épinal",
    "velozef": "Vélozef Brest",
    "velopop": "Vélopop Avignon",
    "inurba-rouen": "Lovélo Rouen",
    "twisto_velolib_caen": "Twisto Caen",
    "tanlib": "Tanlib Le Mans",
    "amiens": "Vélam Amiens",
    "nancy": "Vélostan'lib Nancy",
    "zebullo": "Zebullo Belfort",
    "capcotentin": "Cap Cotentin",
    "le_velo_star": "Le Vélo Star Rennes",
    "montreal_bixi": "BIXI Montréal",
}


def main():
    df = pd.read_csv(OUT / "d11_bootstrap_ci.csv")
    df["pretty"] = df["city"].map(PRETTY).fillna(df["city"])
    # Sort: Tier 1 first (descending dR²), then Tier 2 (descending dR²)
    df["sort_key"] = -df["tier"] * 100 + df["delta_r2"]
    df = df.sort_values("sort_key", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.5, 8.5), constrained_layout=True)
    y = range(len(df))
    colors = ["#1f77b4" if t == 1 else "#d95f02" for t in df["tier"]]
    ax.errorbar(
        df["delta_r2"], y,
        xerr=[df["delta_r2"] - df["delta_r2_lo"], df["delta_r2_hi"] - df["delta_r2"]],
        fmt="o", color="black", ecolor="grey", elinewidth=1.0, capsize=2,
        markersize=4, markerfacecolor="white", markeredgecolor="black",
    )
    for i, (xi, yi, c) in enumerate(zip(df["delta_r2"], y, colors)):
        ax.plot(xi, yi, "o", color=c, markersize=4, zorder=3)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["pretty"], fontsize=8)
    ax.set_xlabel(r"$\Delta R^2$ (trip-count scale, IMD vs no-IMD)", fontsize=10)
    ax.set_xlim(-0.05, 0.65)
    ax.set_title(
        r"IMD-4 contribution across 24 networks (paired block-bootstrap 95% CI, $B=500$)",
        fontsize=10,
    )
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    # Legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markersize=6, label="Tier 1 (trip log)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d95f02",
               markersize=6, label="Tier 2 (GBFS pseudo-flow)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.95)

    fig.savefig(FIG / "fig_multicity_forest.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_multicity_forest.png", bbox_inches="tight", dpi=200)
    print(f"✓ Wrote {FIG/'fig_multicity_forest.pdf'} and .png")


if __name__ == "__main__":
    main()
