"""
d38_velomagg_diagnostic.py — Why does V\'elomagg fail the LSO test?

The paper currently attributes the V\'elomagg LSO failure
(\\rho_G = -0.08 on $N = 53$ stations) to small-sample
underpowerment.  The learning-curve experiment of d32 refutes that
hypothesis : Boston N=25 achieves \\rho = +0.61, so sample size
alone cannot explain a negative correlation.

This script tests three alternative hypotheses :
  H1  Distributional outlier — V\'elomagg IMD axes are degenerate
      (low variance, off-center mean) relative to other Tier 1 / 2
      networks, so the LightGBM model has no exploitable signal.
  H2  Demand-IMD correlation reversal at the station level —
      higher infrastructure (I axis) actually predicts LOWER demand
      in Montpellier due to the geography of the historic centre
      vs the residential perimeter.
  H3  Topography saturation — the T axis is near-constant within
      the 53 stations (all on the plain, ignoring the hilly part
      of the metropole), so the dominant feature has no
      discriminating power.

For each hypothesis we compute a numeric diagnostic and compare
V\'elomagg to the other 8 LSO networks.

Outputs:
  outputs/d38_velomagg_diagnostic.csv
  outputs/d38_velomagg_diagnostic.json
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
IMD = ROOT / "data_collection" / "imd_international"
OUT = ROOT / "experiments" / "outputs"

CITIES = [
    ("velomagg",            "velomagg",                  "Vélomagg Montpellier"),
    ("boston_bluebikes",    "boston_bluebikes",          "Bluebikes Boston"),
    ("dc_capitalbikeshare", "dc_capitalbikeshare",       "Capital Bikeshare DC"),
    ("chicago_divvy",       "chicago_divvy",             "Divvy Chicago"),
    ("sf_baywheels",        "sf_baywheels",              "Bay Wheels SF"),
    ("montreal_bixi",       "world_ca_bixi_montr_al",    "BIXI Montréal"),
    ("london_tfl",          "london_tfl",                "Santander Cycles London"),
    ("paris",               "world_fr_v_lib_metropole",  "Vélib Paris"),
    ("lyon",                "world_fr_v_lo_v",           "Vélo'v Lyon"),
    ("toulouse",            "world_fr_v_l_toulouse",     "VéLÔ Toulouse"),
]
FEATS = ["gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
         "elevation_m", "topography_roughness_index"]


def load_demand(slug):
    if slug == "velomagg":
        p = OUT / "d1_velomagg_predictions.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            id_col = "station" if "station" in df.columns else "station_id"
            df[id_col] = df[id_col].astype(str)
            y = np.expm1(df["y_true_log"]) if "y_true_log" in df else df.get("y_true", df.get("demande"))
            df["__y__"] = y
            return df.groupby(id_col)["__y__"].mean().rename_axis("station_id")
    candidates = [
        OUT / f"d3_{slug}_predictions.parquet",
        OUT / f"d14_{slug}_predictions.parquet",
        OUT / f"d16_{slug}_predictions.parquet",
        OUT / f"d10_{slug}_predictions.parquet",
    ]
    if slug in ("paris", "lyon", "toulouse"):
        cap = {"paris": "Paris", "lyon": "lyon", "toulouse": "toulouse"}[slug]
        candidates.insert(0, OUT / f"d10_{cap}_predictions.parquet")
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            df["station_id"] = df["station_id"].astype(str)
            df["y_true"] = np.expm1(df["y_true_log"])
            return df.groupby("station_id")["y_true"].mean()
    return None


VELOMAGG_GS = Path("/home/rohanfosse/Bureau/Recherche/bikeshare-data-explorer/data/stations_gold_standard_final.parquet")


def load_velomagg_imd():
    gs = pd.read_parquet(VELOMAGG_GS)
    m = gs[gs["city"].str.contains("Montpellier", case=False, na=False)].copy()
    # The GS parquet already has a station_id column ; we use station_name as
    # the canonical identifier (matches the d1 predictions parquet schema).
    if "station_id" in m.columns:
        m = m.drop(columns=["station_id"])
    m = m.rename(columns={"station_name": "station_id",
                          "infra_cyclable_km": "infra_cyclable_features_300m"})
    m["station_id"] = m["station_id"].astype(str)
    return m


def main():
    rows = []
    for slug, stem, pretty in CITIES:
        if slug == "velomagg":
            if not VELOMAGG_GS.exists():
                print(f"  ✗ {pretty}: gold-standard parquet missing"); continue
            imd = load_velomagg_imd()
        else:
            imd_path = IMD / f"{stem}.parquet"
            if not imd_path.exists():
                print(f"  ✗ {pretty}: no IMD parquet at {imd_path}"); continue
            imd = pd.read_parquet(imd_path)
            imd["station_id"] = imd["station_id"].astype(str)
        if slug == "london_tfl":
            imd["station_id"] = imd["station_id"].str.zfill(6)
        avail = [f for f in FEATS if f in imd.columns]
        if len(avail) < 3:
            print(f"  ✗ {pretty}: only {len(avail)} axes"); continue
        y = load_demand(slug)
        if y is None:
            print(f"  ✗ {pretty}: no demand parquet"); continue
        imd["y"] = imd["station_id"].map(y.to_dict())
        sub = imd.dropna(subset=["y"] + avail).copy()
        N = len(sub)
        if N < 20:
            print(f"  ✗ {pretty}: only {N} matched stations"); continue
        # H1 — distributional check : z-scored coefficient of variation on each axis
        cv = {f: float(sub[f].std() / (abs(sub[f].mean()) + 1e-9)) for f in avail}
        # Range as fraction of pooled global range (loaded later)
        rng = {f: float(sub[f].max() - sub[f].min()) for f in avail}
        # H2 — per-axis Spearman with y on this city
        rho = {}
        for f in avail:
            r, _ = spearmanr(sub[f], sub["y"])
            rho[f] = float(r)
        # H3 — topography saturation : variance ratio of T-axis on Velomagg vs others
        t_var = float(sub["elevation_m"].var()) if "elevation_m" in sub else None
        t_iqr = float(np.subtract(*np.percentile(sub["elevation_m"], [75, 25]))) \
            if "elevation_m" in sub else None
        rows.append({
            "city": pretty, "slug": slug, "N": N,
            "cv_M":  cv.get("gtfs_heavy_stops_300m"),
            "cv_I":  cv.get("infra_cyclable_features_300m"),
            "cv_T":  cv.get("elevation_m"),
            "cv_R":  cv.get("topography_roughness_index"),
            "rng_T": rng.get("elevation_m"),
            "rng_R": rng.get("topography_roughness_index"),
            "T_var": t_var, "T_iqr": t_iqr,
            "rho_M_y": rho.get("gtfs_heavy_stops_300m"),
            "rho_I_y": rho.get("infra_cyclable_features_300m"),
            "rho_T_y": rho.get("elevation_m"),
            "rho_R_y": rho.get("topography_roughness_index"),
        })
        print(f"  {pretty:25s} N={N:>4d}  "
              f"σ_M={cv.get('gtfs_heavy_stops_300m', 0):+.2f}  "
              f"σ_I={cv.get('infra_cyclable_features_300m', 0):+.2f}  "
              f"σ_T={cv.get('elevation_m', 0):+.2f}  "
              f"σ_R={cv.get('topography_roughness_index', 0):+.2f}")
        print(f"     ρ(IMD_axis, y_station)  M={rho.get('gtfs_heavy_stops_300m', 0):+.2f}  "
              f"I={rho.get('infra_cyclable_features_300m', 0):+.2f}  "
              f"T={rho.get('elevation_m', 0):+.2f}  "
              f"R={rho.get('topography_roughness_index', 0):+.2f}")

    if not rows:
        print("✗ no cities loaded"); return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d38_velomagg_diagnostic.csv", index=False)

    # Comparative diagnostic : Velomagg vs others
    if "Vélomagg Montpellier" in df["city"].values:
        v = df[df["city"] == "Vélomagg Montpellier"].iloc[0]
        o = df[df["city"] != "Vélomagg Montpellier"]
        print("\n=== Vélomagg vs others (mean) ===")
        for col in ["cv_M", "cv_I", "cv_T", "cv_R",
                    "rho_M_y", "rho_I_y", "rho_T_y", "rho_R_y",
                    "T_iqr"]:
            if col in v and not pd.isna(v[col]) and o[col].notna().any():
                vv, om = v[col], o[col].mean()
                ratio = vv / om if abs(om) > 1e-9 else float("inf")
                flag = " <— OUTLIER" if abs(ratio) > 2 or abs(ratio) < 0.5 else ""
                print(f"  {col:10s}  Vélomagg={vv:+.3f}  others_mean={om:+.3f}  "
                      f"ratio={ratio:+.2f}{flag}")
        # Diagnosis call
        print("\n=== Diagnosis ===")
        diag = []
        # H1 — distributional collapse on multiple axes
        narrow = [a for a in ("cv_M", "cv_I", "cv_T", "cv_R")
                  if not pd.isna(v[a]) and o[a].notna().any()
                  and abs(v[a]) < 0.6 * abs(o[a].mean())]
        if len(narrow) >= 3:
            diag.append(f"H1 STRONGLY supported: Vélomagg is below 60% of peer dispersion on {len(narrow)}/4 axes ({','.join(narrow)}) — LightGBM has narrow feature ranges to split on")
        elif len(narrow) == 2:
            diag.append(f"H1 partially supported: 2/4 axes ({','.join(narrow)}) have collapsed dispersion")
        else:
            diag.append(f"H1 weak: only {len(narrow)}/4 axes show dispersion collapse")
        # H3 — topography-specific saturation (subset of H1)
        if not pd.isna(v["cv_T"]) and abs(v["cv_T"]) < 0.5 * abs(o["cv_T"].mean()):
            diag.append("H3 supported: T-axis dispersion on Vélomagg is <50% of peer mean — topography saturated within the 48 stations")
        else:
            diag.append("H3 not supported as primary cause: T-axis dispersion is in peer range")
        # H2 — sign reversal on demand correlation
        sign_rev = sum(1 for f in ("rho_M_y", "rho_I_y", "rho_T_y", "rho_R_y")
                       if not pd.isna(v[f]) and not pd.isna(o[f].mean())
                       and np.sign(v[f]) != np.sign(o[f].mean()))
        if sign_rev >= 2:
            diag.append(f"H2 supported: {sign_rev}/4 axes have OPPOSITE sign of demand correlation vs peer mean")
        else:
            diag.append(f"H2 weak: only {sign_rev}/4 axes show sign reversal")
        # H4 — topography amplification (Vélomagg-specific finding)
        if not pd.isna(v["rho_T_y"]) and abs(v["rho_T_y"]) > 3 * abs(o["rho_T_y"].mean() or 0.01):
            diag.append(f"H4 emergent: ρ(T, y) on Vélomagg = {v['rho_T_y']:+.2f} is {abs(v['rho_T_y']/o['rho_T_y'].mean()):.1f}× the peer mean — the elevation→demand relationship is anomalously strong in Montpellier (valley centre, residential heights)")
        for d in diag:
            print(f"  • {d}")
        with open(OUT / "d38_velomagg_diagnostic.json", "w") as f:
            json.dump({"rows": rows, "diagnosis": diag}, f, indent=2)
    print(f"\n✓ Saved.")


if __name__ == "__main__":
    main()
