# Paper B — GSP Companion (working draft, v0.1)

**Status:** extracted 2026-05-21 from the predictive paper (v2.3)
at `paper_demand/imd_demand.tex`. Not shippable as-is — needs
~6-9 months of additional work before submission to a venue like
*Transportation Research Part B* or *IEEE TITS*.

## What's here

```
paper_gsp_companion/
├── paper_gsp_companion.tex       # Main TeX file (intro + extracted §8 + discussion + limits placeholders)
├── _section8_extract.tex         # Verbatim §8 from the predictive paper v2.3
├── experiments/                  # 14 GSP-related Python scripts (d22-d33, d36, d40)
├── figures/                      # 5 GSP-related figures (PDF + PNG)
├── outputs/                      # JSON/CSV outputs from the experiments
├── references/
│   └── references_gsp.bib        # Copy of the predictive paper bib (will need pruning)
└── README.md                     # This file
```

## What needs work before submission

### Theoretical novelty (the binding constraint)
Theorems 1-5 are applications of standard results to bike-share,
not new mathematics:

| Theorem | Status | Action needed |
|---------|--------|---------------|
| T1 Dirichlet energy permuted | Trivial / well-known | Likely cut |
| T2 Spectral upper bound on R² | Orthogonal projection R² | Apply to a new setting or strengthen |
| T3 FE structural collapse | Mostly definitional | Cut or move to background |
| T4 Submodularity (1-1/e) | Nemhauser-Wolfe-Fisher 1978 | Add a new bound for the bike-share-specific objective |
| T5 Anis-Gadde-Ortega bound | Direct citation | Specialise to the station-proximity Laplacian |
| T6 Learning curve | Empirical fit on 1 city | Replicate on 5+ cities |
| T7 MCLP comparison | Empirical, not really a theorem | Re-frame as a proposition with a clear analytical statement |

**Minimum bar for a TransRes B / TITS submission:**
- 1-2 genuinely new theoretical results (proofs needed)
- Replication of learning curve on 5+ cities
- MCLP comparison on at least 4 cities at 3 coverage radii
- GCN baseline with proper hyperparameter sweep

### Empirical depth
- D-opt / MCLP only on Boston + DC at present → extend to 9 LSO cities
- HKS features defined but cross-city transfer test not run (cf. companion paper backlog)
- GCN single configuration; needs sweep on depth, dropout, edge weighting

### Scope tightening
The current draft tries to cover three angles at once:
1. Spectral diagnostics (Th. 1-3)
2. Optimal sampling theory (Th. 4-5)
3. Learning curves (Th. 6)
4. (+ regularised-Laplacian smoother, GCN, HKS, GA-IMD)

Pick **one** as the main contribution and demote the others to
supplementary or future-work. The most novel angle is probably
(2) sampling theory with a new bound tailored to the bike-share
demand prediction objective.

## Relation to the predictive paper

Paper A (predictive paper) is the empirical/applied contribution.
Paper B (this one) is the theoretical/methodological contribution.
The two papers share:
- The 9-LSO data infrastructure (Tier 1 + Tier 2 panels)
- The IMD-4 feature definition (Paper A introduces, Paper B uses as fixed input)
- The same authors and codebase

A reader of Paper A can ignore Paper B and get a complete story
("composite indicator predicts demand"). A reader of Paper B
needs Paper A as a citation for the data and the IMD-4
definition.

## Reproducibility

The Python scripts in `experiments/` are *archive copies* of the
canonical versions in `../paper_demand/experiments/`.  They
hard-code `Path(__file__).resolve().parents[1]` to locate the
shared data folders (`data_collection/`, `experiments/outputs/`)
which live under `paper_demand/`, not under
`paper_gsp_companion/`.  To re-run an experiment, execute the
**original** script from the predictive-paper tree :

```bash
cd ../paper_demand/experiments
python d24_gsp_real_cities.py    # spectral measurements
python d26_optimal_siting.py     # D-opt baseline
python d33_mclp_comparison.py    # MCLP / k-median
# … etc
```

The pre-computed JSON/CSV outputs and PDF figures used in this
companion paper are already in `outputs/` and `figures/` here ;
re-running the scripts is only needed if the inputs change.

All scripts pin random seeds (typically 42). Wall-time is
typically <5 minutes per script on a 16 GB laptop.

## Building this paper

```bash
cd paper_gsp_companion
pdflatex paper_gsp_companion.tex
bibtex   paper_gsp_companion
pdflatex paper_gsp_companion.tex
pdflatex paper_gsp_companion.tex
```

Compiles cleanly (0 warnings, 0 undefined refs/citations) on
TeX Live 2026.  Current PDF : 15 pages, A4.

## License
MIT, same as the parent repository.
