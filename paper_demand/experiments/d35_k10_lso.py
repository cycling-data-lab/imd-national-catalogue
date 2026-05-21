"""
d35_k10_lso.py — K=10 LSO replication on the 4 Tier 1 cities.

Re-runs the rigorous Leave-Station-Out pipeline of d19_lso_rigorous with
n_folds=10 (instead of 5) on Boston, DC, Chicago and SF.  The motivation
is a referee-anticipated criticism : K=5 leaves 20% of stations out per
fold, which still permits some leakage through neighbouring training
stations.  K=10 (10% holdout) is the more conservative standard used in
the spatial-ML literature (Roberts et al. 2017, Pohjankukka 2017).

For each city we record:
  - hourly R² (per-fold and mean) for G^-, G_FE and G
  - paired Δρ (G - G^-, G - G_FE) with 95% station-bootstrap CIs
  - Spearman ρ and Precision@K with hypergeometric null

Outputs:
  outputs/d35_k10_<city>.json   (per-city, same schema as d19)
  outputs/d35_k10_summary.csv
  outputs/d35_k10_summary.json
"""
from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd

from d19_lso_rigorous import run as run_lso

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "outputs"

CITIES = [
    ("boston_bluebikes",    "Bluebikes Boston"),
    ("dc_capitalbikeshare", "Capital Bikeshare DC"),
    ("chicago_divvy",       "Divvy Chicago"),
    ("sf_baywheels",        "Bay Wheels SF"),
]

N_FOLDS = 10


def main():
    t0 = time.time()
    rows = []
    for slug, pretty in CITIES:
        print(f"\n{'='*70}\n=== {pretty} ({slug})  K={N_FOLDS}\n{'='*70}")
        metrics = run_lso(slug, n_folds=N_FOLDS, mode="random")
        if metrics is None:
            print(f"  ✗ {pretty} skipped"); continue
        # Rename output file to d35_*
        src = OUT / f"d19_lso_{slug}_random.json"
        dst = OUT / f"d35_k10_{slug}.json"
        if src.exists():
            src.rename(dst)
        ps_src = OUT / f"d19_lso_{slug}_random_per_station.csv"
        ps_dst = OUT / f"d35_k10_{slug}_per_station.csv"
        if ps_src.exists():
            ps_src.rename(ps_dst)

        rho = metrics["rho_point"]
        r2 = metrics["r2_hourly"]
        boot = metrics["bootstrap"]
        rows.append({
            "city": pretty,
            "slug": slug,
            "K": N_FOLDS,
            "N_stations": metrics["n_stations_evaluated"],
            "rho_no_imd": rho["no_imd"],
            "rho_fe":     rho["fe"],
            "rho_imd":    rho["imd"],
            "delta_rho_imd_vs_fe":      boot["delta_rho_imd_vs_fe"]["mean"],
            "delta_rho_imd_vs_fe_lo":   boot["delta_rho_imd_vs_fe"]["ci"][0],
            "delta_rho_imd_vs_fe_hi":   boot["delta_rho_imd_vs_fe"]["ci"][1],
            "delta_rho_imd_vs_no":      boot["delta_rho_imd_vs_no"]["mean"],
            "delta_rho_imd_vs_no_lo":   boot["delta_rho_imd_vs_no"]["ci"][0],
            "delta_rho_imd_vs_no_hi":   boot["delta_rho_imd_vs_no"]["ci"][1],
            "r2_hourly_no_imd": r2["no_imd_mean"],
            "r2_hourly_fe":     r2["fe_mean"],
            "r2_hourly_imd":    r2["imd_mean"],
            "delta_r2_imd_vs_no": r2["imd_vs_no_imd_delta_mean"],
            "delta_r2_imd_vs_fe": r2["imd_vs_fe_delta_mean"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d35_k10_summary.csv", index=False)
    with open(OUT / "d35_k10_summary.json", "w") as f:
        json.dump({"rows": rows, "n_folds": N_FOLDS,
                   "wall_time_s": round(time.time() - t0, 1)}, f, indent=2)
    print("\n=== K=10 summary ===")
    print(df.to_string(index=False))
    print(f"\n✓ Saved.  Wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
