# bikeshare-demand-forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Status: submission draft](https://img.shields.io/badge/Status-submission%20draft-orange.svg)](#)

**Paper:** *From cycling-environment supply to bike-share demand:
an IMD-augmented spatio-temporal forecasting benchmark for French
networks.* Rohan Fossé and Gaël Pallares, CESI LINEACT, 2026.

This repository packages the predictive paper and all the code,
data and intermediate outputs needed to reproduce the results.
It is one of five repositories in the
[cycling-data-lab](https://github.com/cycling-data-lab) GitHub
organisation.

## TL;DR

Operational bike-share demand modelling has two distinct use-cases
that the literature usually conflates:

1. **Forecasting demand at existing stations** (a time-series problem).
2. **Ranking candidate sites for new stations** (a spatial problem
   with no historical demand to lean on).

Existing models do well on (1) and largely speculate on (2). We
close that gap with the **IMD-4** (*Indice de Mobilité Douce à
quatre composantes*), a Bayesian composite indicator of
cycling-environment quality computed from public open data on
every French commune ($n = 34{,}858$), and validate it empirically
on two complementary tests:

- **Temporal forecasting on 27 networks.** IMD-augmented LightGBM
  beats its non-IMD ablation by $\Delta R^2 \in [+0.06, +0.55]$
  with no bootstrap CI overlapping zero. Spans two continents,
  two trip-data paradigms (true logs + GBFS pseudo-flow), three
  operator families and network sizes from 29 to 2,230 stations.
- **Spatial generalisation by leave-station-out.** On four Tier 1
  networks $\geq 400$ training stations, a paired station-bootstrap
  LSO ($B = 1000$) recovers Spearman $\rho \in [+0.49, +0.79]$ for
  the IMD-augmented model vs $\rho \in [-0.68, -0.32]$ for the
  station fixed-effect. The latter is a direct test: a per-station
  dummy has zero information on held-out stations.

Cross-referencing the two evaluations, the +0.47 R² gap recorded
on Boston decomposes into ~+0.15 transferable-spatial + ~+0.32
station-fingerprint components.

## Build the paper

```bash
pdflatex imd_demand.tex
bibtex   imd_demand
pdflatex imd_demand.tex
pdflatex imd_demand.tex
```

Compiles cleanly on TeX Live 2026. Current PDF: 25 pages, A4.

## Reproduce the experiments

Setup once:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then run any of the `experiments/d*.py` scripts. The naming
convention: `d<NN>_<short_description>.py`. Each script pins a
random seed and writes its outputs to `experiments/outputs/`.

Highlights:

| Script | Topic |
|---|---|
| `d1_demand_model.py` | Vélomagg Montpellier single-city headline |
| `d3_multicity_benchmark.py` | 27-network temporal forecasting |
| `d19_lso_rigorous.py` | Leave-station-out spatial generalisation |
| `d34_nyc_bootstrap_subsample.py` | NYC Citi Bike bootstrap CI (closed in v2.3) |
| `d35_k10_lso.py` | K=10 LSO replication on 4 Tier 1 cities |
| `d36_nonlinear_R2_spectral.py` | Non-linear R²_spectral ceiling |
| `d37_atlas_external_validation.py` | Atlas vs Copenhagenize/ECF cross-check |
| `d38_velomagg_diagnostic.py` | Why Vélomagg LSO fails (multi-axis dispersion collapse) |
| `d42_multiple_testing.py` | Bonferroni / BH-FDR sweep across the LSO family |

The full list (43 scripts) is in `experiments/`.

## Data

Computed inputs (commune-level IMD-4 layers) come from the
parent indicator paper:

> Fossé R, Pallares G. *A National Cycling-Environment Composite
> Indicator for French Communes.* CESI LINEACT, 2026.
> [cycling-data-lab/imd-national-catalogue](https://github.com/cycling-data-lab/imd-national-catalogue).

Trip-level inputs are downloaded by `data_collection/tier1_downloader.py`
from Lyft, Bluebikes, Capital Bikeshare, Divvy, BIXI, TfL, Citi Bike
and Vélomagg public archives.

GBFS pseudo-flow inputs are collected by the polling daemon documented
in [cycling-data-lab/gbfs-audit-catalogue](https://github.com/cycling-data-lab/gbfs-audit-catalogue).

## How to cite

```bibtex
@unpublished{FossePallares2026bikeshareDemand,
  author = {Foss\'e, Rohan and Pallares, Ga\"el},
  title  = {From Cycling-Environment Supply to Bike-Share Demand:
            An {IMD}-Augmented Spatio-Temporal Forecasting Benchmark
            for French Networks},
  note   = {CESI LINEACT, 2026.
            \url{https://github.com/cycling-data-lab/bikeshare-demand-forecasting}},
  year   = {2026}
}
```

A machine-readable [CITATION.cff](./CITATION.cff) is provided in
the root.

## Companion papers

- [imd-national-catalogue](https://github.com/cycling-data-lab/imd-national-catalogue) — the IMD-4 indicator on 34,858 communes.
- [bikeshare-gsp-tools](https://github.com/cycling-data-lab/bikeshare-gsp-tools) — graph-signal-processing companion paper (spectral bounds, D-optimal siting, learning curves).
- [penality-analysis](https://github.com/cycling-data-lab/penality-analysis) — triple-penalty mobility-justice diagnostic.
- [gbfs-audit-catalogue](https://github.com/cycling-data-lab/gbfs-audit-catalogue) — the underlying station inventory.

## License

[MIT](./LICENSE). Data products inherit upstream licenses (OdbL,
INSEE open data, GBFS provider terms).

## Contact

Rohan Fossé — [rfosse@cesi.fr](mailto:rfosse@cesi.fr) — [ORCID](https://orcid.org/0009-0002-2195-0198)
Gaël Pallares — [ORCID](https://orcid.org/0009-0002-8680-604X)
