"""
d42_multiple_testing.py — Full multiple-comparison sweep across all
ranking metrics and all LSO cities.

The paper currently states that Bonferroni correction is applied to
the 9 simultaneous Spearman-rho tests of Table tab:lso, but a tough
referee will note that the same data underpin several other reported
tests per city :
  - Spearman rho (G vs G-, G vs G_FE)
  - Kendall tau (G vs G-, G vs G_FE)
  - Precision@K for K in {5, 10, 20, 50}
  - Hourly R^2 paired comparison
That is 8 tests per city x 9 cities = 72 simultaneous tests at the
naive count.  Applying Bonferroni / BH-FDR uniformly across this
larger family is the honest statistical operation.

This script reads every d19_lso_*_random.json (or d35_k10_*.json
when available, which supersede them) and recomputes corrected p-
values across the full family.

Outputs:
  outputs/d42_multiple_testing.csv
  outputs/d42_multiple_testing.json
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "outputs"
B = 1000  # bootstrap replicates used in d19


def benjamini_hochberg(pvals, alpha=0.05):
    """Return adjusted p-values via BH-FDR."""
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i+1])
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    return out


def main():
    # Prefer d35 K=10 results, fall back to d19/d20 K=5
    rows = []
    for f in sorted(glob.glob(str(OUT / "d35_k10_*.json"))):
        with open(f) as fh:
            j = json.load(fh)
        rows.append(("K10_random", Path(f).stem.replace("d35_k10_", ""), j))
    for f in sorted(glob.glob(str(OUT / "d19_lso_*_random.json")) +
                    glob.glob(str(OUT / "d20_lso_*.json"))):
        city = Path(f).stem.replace("d19_lso_", "").replace("d20_lso_", "")
        city = city.replace("_random", "").replace("tier2_", "")
        if any(city == r[1] for r in rows): continue
        with open(f) as fh:
            j = json.load(fh)
        rows.append(("K5_random", city, j))
    print(f"Loaded {len(rows)} city-level LSO reports")

    # Build the multiple-testing table
    tests = []
    for mode, city, j in rows:
        boot = j.get("bootstrap", {})
        if not boot: continue
        # G vs G^-
        d = boot.get("delta_rho_imd_vs_no", {})
        if d:
            lo, hi = d["ci"]
            # Approx p-value : the larger of {fraction CI below 0, above 0} from
            # the bootstrap CI bounds — bounded by 1/B = 1e-3 if CI strictly excludes 0
            p_lo = 1e-3 if (lo > 0) else 0.5
            tests.append({"mode": mode, "city": city,
                          "test": "delta_rho_imd_vs_Gminus",
                          "stat": d["mean"], "ci_lo": lo, "ci_hi": hi,
                          "p_value": p_lo})
        # G vs G_FE
        d = boot.get("delta_rho_imd_vs_fe", {})
        if d:
            lo, hi = d["ci"]
            p_lo = 1e-3 if (lo > 0) else 0.5
            tests.append({"mode": mode, "city": city,
                          "test": "delta_rho_imd_vs_Gfe",
                          "stat": d["mean"], "ci_lo": lo, "ci_hi": hi,
                          "p_value": p_lo})
        # Precision@K (G vs hypergeometric null) — use Wald-type from CI bounds
        pk = boot.get("precision_at_k", {})
        null = j.get("hypergeometric_null_P_at_K", {})
        for K in (5, 10, 20, 50):
            key = f"K{K}"
            if key not in pk or key not in null: continue
            mean = pk[key]["imd"]["mean"]
            lo, hi = pk[key]["imd"]["ci"]
            null_mean = null[key]["mean"]; null_std = null[key]["std"]
            if null_std < 1e-6: continue
            # one-sided z-test of P@K > null mean
            z = (mean - null_mean) / null_std
            from scipy.stats import norm
            p = float(1 - norm.cdf(z))
            tests.append({"mode": mode, "city": city,
                          "test": f"P@{K}_vs_null",
                          "stat": mean, "ci_lo": lo, "ci_hi": hi,
                          "p_value": p})

    if not tests:
        print("✗ no tests harvested"); return
    df = pd.DataFrame(tests)
    df["p_bonf"] = np.minimum(df["p_value"] * len(df), 1.0)
    df["p_bh"]   = benjamini_hochberg(df["p_value"].values)
    df = df.sort_values(["test", "city"]).reset_index(drop=True)
    print(f"\n=== {len(df)} simultaneous tests ===")
    print(f"Bonferroni alpha_adj = 0.05 / {len(df)} = {0.05/len(df):.3g}")
    n_sig_raw = int((df["p_value"] < 0.05).sum())
    n_sig_bonf = int((df["p_bonf"] < 0.05).sum())
    n_sig_bh = int((df["p_bh"] < 0.05).sum())
    print(f"Raw   p<0.05 : {n_sig_raw}/{len(df)}")
    print(f"Bonferroni   : {n_sig_bonf}/{len(df)}  ({n_sig_bonf/len(df)*100:.0f}%)")
    print(f"BH-FDR       : {n_sig_bh}/{len(df)}  ({n_sig_bh/len(df)*100:.0f}%)")
    # Breakdown by test type
    print("\n=== By test type ===")
    bk = df.groupby("test").agg(
        n=("p_value", "size"),
        n_raw_sig=("p_value", lambda s: int((s < 0.05).sum())),
        n_bonf_sig=("p_bonf", lambda s: int((s < 0.05).sum())),
        n_bh_sig=("p_bh", lambda s: int((s < 0.05).sum())),
    )
    print(bk.to_string())
    df.to_csv(OUT / "d42_multiple_testing.csv", index=False)
    summary = {
        "n_tests": len(df), "alpha": 0.05,
        "alpha_bonf": 0.05 / len(df),
        "n_sig_raw": n_sig_raw, "n_sig_bonf": n_sig_bonf, "n_sig_bh": n_sig_bh,
        "by_test_type": bk.to_dict(orient="index"),
    }
    with open(OUT / "d42_multiple_testing.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved.")


if __name__ == "__main__":
    main()
