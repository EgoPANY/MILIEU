# Figure bar significance (paired multiseed tests)

CSV exports live under [`data/manuscript/`](../data/manuscript/README.md):

- `figure_significance_ours_vs_comparator_panel_summary.csv` — panel-level means, paired two-sided *t*-tests, star labels.
- `figure_significance_paired_seed_multiseed_tests.csv` — one row per seed per comparison.

## Scope

| Figure | Comparison |
|--------|------------|
| Fig. 2b | MMD bars · Essential, RPE1, Norman · Ours vs GEARS, BIOLORD, STATE, Ridge, Mean |
| Fig. 3a | MMD bars · hippocampus Stereo-seq and PerturbMap Visium · Ours vs Celcomen, CONCERT, Ridge, Mean |
| Fig. 3c | VCC metrics · Stk39 and Fasn · Ours vs Celcomen, CONCERT, Mean-control |
| Fig. 4c | Tgfbr2 module Spearman ρ · Ours vs Mean, CONCERT, Celcomen |
| Supp. Fig. S2 | Ablation subpanels (context, ESM-2, niche encoder) |
| Supp. Fig. S3 | Context-type holdout subpanels |

## Method

- **Unit of pairing:** matched random seeds (or seed index intersection) between Ours and each comparator on the same plot panel and metric.
- **Test:** paired two-sided Student *t*-test on per-seed scalar metrics (SciPy `ttest_rel`).
- **Direction:** `better_direction` is `higher` or `lower` per metric (e.g. MMD uses `lower`).
- **Stars:** conventional thresholds on *p*-value (\* *p* < 0.05, \*\* < 0.01, \*\*\* < 0.001, \*\*\*\* < 1e-4).

Recompute from authority multiseed plot CSVs with the project’s significance export script (not shipped in this repository).
