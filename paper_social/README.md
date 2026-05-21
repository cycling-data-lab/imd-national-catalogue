# penality-analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Status: submission draft](https://img.shields.io/badge/Status-submission%20draft-orange.svg)](#)

**Paper:** *Cycling poverty in France: a triple-penalty mobility-justice
diagnostic on 34,858 communes.* Rohan Fossé and Gaël Pallares,
CESI LINEACT, 2026.

This repository packages a mobility-justice diagnostic that turns
the IMD-4 cycling-environment indicator into an actionable
priority list for the second tranche of the French Plan Vélo
(2023–2027). It is one of five repositories in the
[cycling-data-lab](https://github.com/cycling-data-lab) GitHub
organisation.

> **Naming note.** The repository is named `penality-analysis`
> (with the typo) to keep stable URLs once cited; the paper itself
> uses the correct *triple-penalty* terminology throughout.

## TL;DR

The standard French cycling-policy diagnostic answers *where*
the cycling environment is weak. It does not answer the policy
question that follows — *who pays the cost of that weakness*.
We close this gap by stacking three vulnerability layers on top
of the commune-level IMD-4:

1. **Cycling-environment deprivation** (bottom-33% IMD-4)
2. **Monetary-poverty exposure** (top-33% INSEE poverty rate, > 15%)
3. **Structural geographic isolation** (lowest income tertile
   ∪ overseas *départements*)

The **triple-penalty intersection** (the product of the three
indicator flags) identifies:

- **322 of the 362 cycling-poverty deserts** as also monetary-
  poverty-vulnerable, covering 1.89 M inhabitants (91.7% of
  desert population).
- **42 of the 362 deserts** in the overseas *départements* —
  11.6% of the count but **36.3% of the population** — making
  Outre-mer a structurally over-represented mobility-justice
  priority.
- **90 metropolitan deserts** in the national first income decile
  but outside Outre-mer, aggregating to ~450,000 inhabitants and
  invisible under any single-axis equity filter. This is the
  policy-blind subset of the current Plan Vélo distribution.

## Build the paper

```bash
pdflatex imd_social.tex
bibtex   imd_social
pdflatex imd_social.tex
pdflatex imd_social.tex
```

Compiles cleanly on TeX Live 2026.

## Reproduce the analysis

The triple-penalty diagnostic is a transparent overlay of three
public data layers:

- **IMD-4** : from the parent repository
  [imd-national-catalogue](https://github.com/cycling-data-lab/imd-national-catalogue).
- **Income median, poverty rate** : INSEE Filosofi commune-level
  open data, most recent vintage.
- **Outre-mer status** : binary flag from INSEE commune code
  (department code starts with `97`).

No model fitting is required — the three flags are deterministic
thresholds (Section "Method" of the paper). The computation is
sub-second on a modern laptop and is documented inline.

## Companion papers

- [imd-national-catalogue](https://github.com/cycling-data-lab/imd-national-catalogue) — the IMD-4 indicator (substrate of this analysis).
- [bikeshare-demand-forecasting](https://github.com/cycling-data-lab/bikeshare-demand-forecasting) — predictive paper on the IMD-4.
- [bikeshare-gsp-tools](https://github.com/cycling-data-lab/bikeshare-gsp-tools) — graph-signal-processing companion paper.
- [gbfs-audit-catalogue](https://github.com/cycling-data-lab/gbfs-audit-catalogue) — the underlying station inventory.

## How to cite

```bibtex
@unpublished{FossePallares2026cyclingPoverty,
  author = {Foss\'e, Rohan and Pallares, Ga\"el},
  title  = {Cycling Poverty in {F}rance: A Triple-Penalty
            Mobility-Justice Diagnostic on 34{,}858 Communes},
  note   = {CESI LINEACT, 2026.
            \url{https://github.com/cycling-data-lab/penality-analysis}},
  year   = {2026}
}
```

A machine-readable [CITATION.cff](./CITATION.cff) is provided.

## License

[MIT](./LICENSE). The underlying INSEE Filosofi data is published
under the Licence Ouverte 2.0 (Etalab).

## Contact

Rohan Fossé — [rfosse@cesi.fr](mailto:rfosse@cesi.fr) — [ORCID](https://orcid.org/0009-0002-2195-0198)
Gaël Pallares — [ORCID](https://orcid.org/0009-0002-8680-604X)
