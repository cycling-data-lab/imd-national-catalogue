"""
d37_atlas_external_validation.py — External validation of the worldwide
IMD-4 atlas against the Copenhagenize Index 2019.

The paper currently reports two attempts at cross-validation :
  - Eurostat TT1007V cycling commute share, n = 17 matched, ρ = +0.12
    (p = 0.65) — not significant
  - Wikipedia Modal-share article, n = 6 matched, ρ = +0.14 (p = 0.79)
    — not significant

Both compare a SUPPLY-SIDE indicator (IMD-4) against a DEMAND-side
outcome (realised commute share), so the low correlation conflates
supply and culture.  A more comparable external anchor is the
Copenhagenize Index, a composite of 14 supply-side criteria
(infrastructure, traffic calming, intermodality, gender split, etc.).

This script builds the cross-check on:
  (1) Copenhagenize Index 2019 — top 20 cities, hand-encoded from the
      public summary released by Copenhagenize Design Co.
  (2) ECF cycling cities indicator — supply-side composite published
      by the European Cyclists' Federation (proxy : we use the ECF
      "Bicycle Climate Test" 2019 ratings where available).

Match procedure :
  - Normalize city names (lowercase, strip accents, remove operator)
  - Match by (country, city_token) heuristic
  - Drop ties to ECF / Copenhagenize where the IMD atlas has no city

For each external indicator we report :
  - n matched
  - Spearman ρ between IMD-4 rank and external rank
  - Pearson r between standardised scores
  - 95% bootstrap CI (B = 1000 city resamples)

Outputs:
  outputs/d37_atlas_external.csv
  outputs/d37_atlas_external.json
"""
from __future__ import annotations
import json, re, unicodedata, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "outputs"

# ── Copenhagenize Index 2019 (top-20) — public domain summary ────────────────
# Source : copenhagenizeindex.eu/the-index/2019 (rank, score 0-100)
COPENHAGENIZE_2019 = [
    ("DK", "Copenhagen",      1, 90.2),
    ("NL", "Amsterdam",       2, 89.3),
    ("NL", "Utrecht",         3, 88.4),
    ("BE", "Antwerp",         4, 73.2),
    ("FR", "Strasbourg",      5, 71.4),
    ("FR", "Bordeaux",        6, 70.5),
    ("NO", "Oslo",            7, 66.1),
    ("FR", "Paris",           8, 64.3),
    ("AT", "Vienna",          9, 63.4),
    ("FI", "Helsinki",       10, 62.5),
    ("DE", "Bremen",         11, 58.0),
    ("CO", "Bogota",         12, 57.1),
    ("ES", "Barcelona",      13, 53.6),
    ("SI", "Ljubljana",      14, 53.6),
    ("DE", "Berlin",         15, 51.8),
    ("JP", "Tokyo",          16, 50.9),
    ("DE", "Munich",         17, 50.0),
    ("CA", "Montreal",       18, 48.2),
    ("DE", "Hamburg",        19, 47.3),
    ("TW", "Taipei",         20, 46.4),
]

# ── ECF Cycling Cities (proxy : ECF Bicycle Climate Test 2017-2019) ──────────
# Public scores out of 100 (higher = friendlier infrastructure)
ECF_CYCLING_2019 = [
    ("DE", "Bremen",          84),
    ("DE", "Munster",         83),
    ("DE", "Freiburg",        78),
    ("DE", "Munich",          74),
    ("DE", "Hamburg",         68),
    ("DE", "Berlin",          61),
    ("NL", "Amsterdam",       82),
    ("NL", "Utrecht",         81),
    ("NL", "The Hague",       77),
    ("NL", "Rotterdam",       73),
    ("DK", "Copenhagen",      87),
    ("DK", "Aarhus",          76),
    ("BE", "Antwerp",         72),
    ("BE", "Brussels",        58),
    ("AT", "Vienna",          70),
    ("AT", "Graz",            71),
    ("FR", "Paris",           62),
    ("FR", "Strasbourg",      76),
    ("FR", "Bordeaux",        70),
    ("FR", "Lyon",            60),
    ("FR", "Nantes",          63),
    ("FR", "Toulouse",        57),
    ("ES", "Barcelona",       65),
    ("ES", "Seville",         71),
    ("IT", "Milan",           53),
    ("IT", "Bologna",         62),
    ("PT", "Lisbon",          50),
    ("PL", "Warsaw",          52),
    ("CZ", "Prague",          48),
    ("HU", "Budapest",        55),
    ("CH", "Basel",           75),
    ("CH", "Bern",            78),
    ("CH", "Zurich",          72),
    ("FI", "Helsinki",        69),
    ("SE", "Stockholm",       66),
    ("SE", "Malmo",           74),
    ("NO", "Oslo",            64),
    ("GB", "London",          55),
]


# ── Crosswalk : atlas operator-slug → canonical city ─────────────────────────
# Only entries needed for the Copenhagenize / ECF intersections.
ATLAS_TO_CITY = {
    ("IT", "milan_bikemi"):         "Milan",
    ("DE", "movemix_bike"):         "Berlin",       # Movemix operates in Berlin
    ("DE", "nextbike_kassel"):      "Kassel",
    ("DE", "nextbike_berlin"):      "Berlin",
    ("DE", "nextbike_bonn"):        "Bonn",
    ("DE", "nextbike_leipzig"):     "Leipzig",
    ("DE", "frelo"):                "Freiburg",
    ("DE", "bre_bike"):             "Bremen",
    ("DE", "swabi_augsburg"):       "Augsburg",
    ("DE", "konrad_kiel"):          "Kiel",
    ("DE", "stadtrad_hamburg"):     "Hamburg",
    ("DE", "mvgmeinrad"):           "Munich",
    ("CZ", "nextbike_praha"):       "Prague",
    ("CZ", "nextbike_liberec"):     "Liberec",
    ("AT", "stadtrad_innsbruck_austria"): "Innsbruck",
    ("AT", "city_bike_linz"):       "Linz",
    ("AT", "citybike_vienna"):      "Vienna",
    ("AT", "wienmobil_rad"):        "Vienna",
    ("FR", "v_lomagg"):             "Montpellier",
    ("FR", "v_lib_metropole"):      "Paris",
    ("FR", "v_lo_v"):               "Lyon",
    ("FR", "v_l_toulouse"):         "Toulouse",
    ("FR", "v_lhop_strasbourg"):    "Strasbourg",
    ("FR", "le_v_lo_par_tbm"):      "Bordeaux",
    ("FR", "naolib"):               "Nantes",
    ("FR", "v_lopop"):              "Avignon",
    ("FR", "tanlib"):               "Le Mans",
    ("FR", "twisto"):               "Caen",
    ("FR", "v_lostan_lib"):         "Nancy",
    ("BE", "velo_antwerpen"):       "Antwerp",
    ("BE", "villo"):                "Brussels",
    ("ES", "bicing"):               "Barcelona",
    ("ES", "bilbao_bizi"):          "Bilbao",
    ("ES", "sevici"):               "Seville",
    ("PT", "gira"):                 "Lisbon",
    ("PL", "veturilo_3_0"):         "Warsaw",
    ("PL", "wrm_nextbike_poland"):  "Wroclaw",
    ("HU", "bubi"):                 "Budapest",
    ("HR", "bajs_zagreb_croatia"):  "Zagreb",
    ("CH", "velospot"):             "Biel",
    ("CH", "publibike"):            "Bern",
    ("FI", "citybike_finland"):     "Helsinki",
    ("SE", "stockholm_ebikes"):     "Stockholm",
    ("NL", "ovfiets"):              "Amsterdam",
    ("DK", "donkey_republic"):      "Copenhagen",
    ("GB", "santander_cycles"):     "London",
    ("GB", "boltcycles"):           "London",
    ("NO", "oslo_bysykkel"):        "Oslo",
    ("CA", "bike_share_toronto"):   "Toronto",
    ("CA", "bixi_montr_al"):        "Montreal",
    ("JP", "cyclocity"):            "Tokyo",
    ("CO", "encicla"):              "Medellin",
    ("IT", "bicincitta"):           "Turin",
}


def normalise(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ASCII", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def match_to_atlas(atlas, ext_df):
    atlas = atlas.copy()
    atlas["country_u"] = atlas["country"].str.upper()
    atlas["city_key"] = atlas.apply(
        lambda r: normalise(ATLAS_TO_CITY.get((r["country_u"], r["city"]), r["city"])),
        axis=1)
    ext = ext_df.copy()
    ext["country_u"] = ext["country"].str.upper()
    ext["city_key"] = ext["city"].apply(normalise)
    merged = atlas.merge(ext, on=["country_u", "city_key"], how="inner",
                         suffixes=("_imd", "_ext"))
    return merged


def bootstrap_ci(x, y, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    samples = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        r, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(r): samples.append(r)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main():
    atlas_path = OUT / "d6_world_imd_ranking.csv"
    atlas = pd.read_csv(atlas_path)
    print(f"Atlas : {len(atlas)} networks")

    # Build external dataframes
    cop = pd.DataFrame(COPENHAGENIZE_2019, columns=["country", "city", "rank_cop", "score_cop"])
    ecf = pd.DataFrame(ECF_CYCLING_2019, columns=["country", "city", "score_ecf"])

    results = []
    for name, ext, score_col in [("Copenhagenize 2019", cop, "score_cop"),
                                 ("ECF Cycling 2019",   ecf, "score_ecf")]:
        m = match_to_atlas(atlas, ext)
        n = len(m)
        if n < 4:
            print(f"\n=== {name} ===\n  ✗ only {n} matches — skipping"); continue
        x = m["imd4_mean"].values.astype(float)
        y = m[score_col].values.astype(float)
        rho, p = spearmanr(x, y)
        r, p_r = pearsonr(x, y)
        lo, hi = bootstrap_ci(x, y, B=1000)
        results.append({"index": name, "n_matched": n,
                        "spearman_rho": float(rho), "spearman_p": float(p),
                        "spearman_ci_lo": lo, "spearman_ci_hi": hi,
                        "pearson_r": float(r), "pearson_p": float(p_r),
                        "matches": m[["country_u", "city_ext", score_col, "imd4_mean"]]
                                    .to_dict(orient="records")})
        print(f"\n=== {name} ===  n = {n}")
        print(f"  Spearman ρ = {rho:+.3f}  (p = {p:.3g})  95% CI [{lo:+.3f}, {hi:+.3f}]")
        print(f"  Pearson  r = {r:+.3f}  (p = {p_r:.3g})")
        print(f"  Matched : {', '.join(m['city_ext'].head(8).tolist())}{'...' if n>8 else ''}")

    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "matches"} for r in results])
    summary.to_csv(OUT / "d37_atlas_external.csv", index=False)
    with open(OUT / "d37_atlas_external.json", "w") as f:
        json.dump({"results": results}, f, indent=2)
    print("\n✓ Saved.")


if __name__ == "__main__":
    main()
