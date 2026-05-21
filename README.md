# IMD National Catalogue

> A commune-level reference dataset for the **Indice de Mobilité Douce
> (IMD)** and the **Indice d'Équité Sociale (IES)** across **34,858
> French communes**, with the open-source pipeline that produced it.

[![Code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Data: ODbL](https://img.shields.io/badge/data-ODbL%20v1.0-blue)](LICENSE-DATA)
[![Status](https://img.shields.io/badge/status-v0.2%20beta-yellow)](#roadmap)

> **Status — v0.2 beta.** The Parquet catalogue, the scoring pipeline
> and a 13-test suite are in. The Hugging Face dataset, the Streamlit
> dashboard and the Zenodo DOI are staged for the upcoming `v1.0.0`
> tag — see the [roadmap](#roadmap) for what remains.

## Quick start (local, no Hugging Face yet)

```bash
git clone https://github.com/cycling-data-lab/imd-national-catalogue.git
cd imd-national-catalogue
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"

# Rebuild the Parquet from the bundled CSV sources
python scripts/regenerate.py

# Run the schema + value tests
pytest -v

# Use the catalogue
python -c "
import pandas as pd
df = pd.read_parquet('catalogue/communes_imd_national.parquet')
print(df.shape, '— 17 columns:', list(df.columns))
print('deserts:', int(df['double_penalty'].sum()))
"
```

## What this is

Volumetric metrics (stations per capita, kilometres of cycle lane)
predict cycling *supply*, not cycling *quality*. The IMD is a four-
component composite — **M**ultimodality, **I**nfrastructure,
**S**afety, **T**opography — whose weights are calibrated against the
Baromètre FUB and the EMP 2019 via a Bayesian simplex. Once calibrated
at urban-area level (Paper 03), the same weights are pushed to **all
34,858 French communes** (Paper 04) and paired with the **Indice
d'Équité Sociale (IES)**, which flags communes that under-perform
relative to their socio-economic profile — the *cycling-poverty
deserts*.

Headline results (Paper 04):

* IMD-4 wins **4 / 5** of the tournament references against the
  Cerema infrastructure baseline.
* **+18 pts R²** over the standard Cerema indicator
  (Cerema-residual regression).
* National panel ρ = **+0.41** overall, ρ = **+0.55** on the 42 cities
  with population ≥ 100 k.
* **362 cycling-poverty deserts** identified, 6 of the top-15 in
  La Réunion.
* ρ(IMD, median income) = +0.001 in the urban subset — the IMD is
  **not** a wealth proxy.

## Schema

`catalogue/communes_imd_national.parquet` (34,858 rows × 17 cols, zstd-compressed, ~2.5 MB):

| Column | Type | Meaning |
|---|---|---|
| `code_commune` | string | INSEE COG 2024 commune code (PK, leading zeros preserved) |
| `nom` | string | Commune name |
| `population` | Int64 | Population (INSEE) |
| `surface_km2` | float | Commune surface |
| `M_per_km2` | float | Multimodality density (heavy-transit stations / km²) |
| `I_per_km2` | float | Infrastructure density (cyclable-km / km²) |
| `D_log` | float | log Population density |
| `M_z`, `I_z`, `D_z` | float | National-panel z-scores of the three components |
| `IMD4_national` | float | Composite IMD-4 score (Bayesian weights from B7) |
| `insee_part_velo` | float | Reference: INSEE part-vélo-travail 2022 (%) |
| `income_median` | float / NaN | Niveau de vie médian, EUR/UC (INSEE Filosofi) |
| `poverty_rate` | float / NaN | Taux de pauvreté, % (INSEE Filosofi) |
| `IES` | float / NaN | Indice d'Équité Sociale (residual after log income) |
| `ies_available` | bool | True for the 31,266 communes with Filosofi income |
| `double_penalty` | bool | True for the 362 cycling-poverty deserts |

## Once v1.0.0 is published on Hugging Face

```python
from datasets import load_dataset

imd = load_dataset("rohanfosse/imd-national-catalogue", split="train").to_pandas()

# The 362 priority deserts: high social fragility, low IMD
deserts = imd[imd["double_penalty"]].sort_values("IMD4_national")
```

## Companion artefacts

| Resource | Where |
| --- | --- |
| Sister catalogue (stations, Paper 01) | [github.com/cycling-data-lab/gbfs-audit-catalogue](https://github.com/cycling-data-lab/gbfs-audit-catalogue) |
| Manuscript (LaTeX) | [`paper/`](paper/) — Paper 04 of the BikeShare-ICT series |
| Bayesian methodology (Paper 03) | upstream calibration; weights ingested by `imd_pipeline/` |
| Dataset card + full schema | _coming with v1.0.0 on Hugging Face_ |
| Live dashboard | _coming with v1.0.0 on Streamlit Cloud_ |
| Zenodo DOI | _minted at v1.0.0 release_ |

## Roadmap

* **Phase 1 — Scaffold** *(done)*. Licensing, README, package layout,
  citation metadata.
* **Phase 2 — Pipeline & catalogue** *(done, v0.2)*. `imd_pipeline.core`
  ports the b20/b21 join, `scripts/regenerate.py` materialises
  `catalogue/communes_imd_national.parquet` (34,858 × 17), and 13
  tests pin the schema and headline counts (362 deserts, 31,266 with
  IES).
* **Phase 3 — Hugging Face** *(next)*. Dataset card with explicit
  recipes (top-15 deserts, INSEE join, Cerema-vs-IMD comparison) and
  Croissant metadata; upload to `rohanfosse/imd-national-catalogue`.
* **Phase 4 — Streamlit.** Commune-level map with strata filters,
  IES double-penalty layer, M / I / D decomposition tab; deploy at
  `imd-national.streamlit.app`.
* **Phase 5 — Zenodo.** `v1.0.0` tag, archived DOI, cite-as updated
  across the paper, the dataset card and `CITATION.cff`.

## Relation to the cycling-data-lab series

This repository is one of five sibling repositories under the
[cycling-data-lab](https://github.com/cycling-data-lab) GitHub
organisation, each packaging one publication-grade contribution
to cycling-data research (CESI LINEACT, 2025–2026). The five-repo
series:

1. **[gbfs-audit-catalogue](https://github.com/cycling-data-lab/gbfs-audit-catalogue)**
   — 1,509-feed GBFS infrastructure audit (the certified station
   inventory).
2. **[imd-national-catalogue](https://github.com/cycling-data-lab/imd-national-catalogue)** (this repo)
   — IMD-4 + IES on 34,858 French communes (the supply-side
   indicator substrate).
3. **[bikeshare-demand-forecasting](https://github.com/cycling-data-lab/bikeshare-demand-forecasting)**
   — IMD-augmented spatio-temporal forecasting on 27 networks +
   leave-station-out spatial generalisation on 9 networks.
4. **[bikeshare-gsp-tools](https://github.com/cycling-data-lab/bikeshare-gsp-tools)**
   — graph-signal-processing companion (spectral bounds, D-optimal
   siting, learning curves). *Working draft.*
5. **[penality-analysis](https://github.com/cycling-data-lab/penality-analysis)**
   — triple-penalty mobility-justice diagnostic (cycling-poverty
   deserts at the intersection of cycling-environment deprivation,
   monetary poverty and structural geographic vulnerability).

## Citation

A [`CITATION.cff`](CITATION.cff) is provided. Once the manuscript is
deposited, the BibTeX block below will be filled in.

```bibtex
@article{Fosse2026imdnational,
  author  = {Foss\'e, Rohan and Pallares, Ga\"el},
  title   = {A National Cycling-Environment Composite Indicator for
             French Communes: The IMD and the Cycling-Poverty Deserts},
  year    = {2026},
  note    = {Manuscript in preparation; submission target:
             Transportation Research Part D / Transport Policy /
             Journal of Transport Geography}
}
```

## Licences

Code under [MIT](LICENSE); data (once released) under
[ODbL v1.0](LICENSE-DATA). Upstream attributions are listed in
`LICENSE-DATA`.

## Contact

**Rohan Fossé** ([rfosse@cesi.fr](mailto:rfosse@cesi.fr)) — CESI École d'Ingénieurs, Montpellier.
**Gaël Pallares** — CESI LINEACT (EA 7527), Montpellier.

Issues and contributions are welcome on the
[issue tracker](https://github.com/cycling-data-lab/imd-national-catalogue/issues).
