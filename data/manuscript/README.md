# Manuscript result tables

CSV files supporting benchmark numbers cited in the MILIEU paper. File names refer to biological screens or atlases (for example Replogle K562 Essential, hippocampus Perturb-map).

## Included in this repository

| File | Role |
|------|------|
| `sc_gene_ood_adamson_jurkat_hepg2_multiseed_metrics.csv` | Extended dissociated screens (Supplementary). |
| `spatial_benchmark_radar_metrics_hippocampus_perturbmap_16seed_multiseed.csv` | Main-text spatial radar metrics. |
| `spatial_benchmark_mmd_hippocampus_perturbmap_16seed_multiseed.csv` | Main-text spatial MMD bars. |
| `spatial_hippocampus_vcc_stk39_fasn_clu_ridge_mean_train_multiseed.csv` | VCC-style probe scores (hippocampus). |
| `figure_significance_ours_vs_comparator_panel_summary.csv` | Panel-level significance summaries. |
| `figure_significance_paired_seed_multiseed_tests.csv` | Paired seed-level tests. |

**Figure replay CSV** (main Figs 2–4, Supp Figs S1–S3): [`../figure_tables/README.md`](../figure_tables/README.md).

**Inference demo** (Essential seed 0 slice + checkpoint): [`../demo/sc/README.json`](../demo/sc/README.json).

Per-gene holdout split ledgers, full per-spot spatial readout tables, and model checkpoints used for publication figures are **not** included here.

## Integrity

[`SHA256SUMS`](SHA256SUMS) · [`MANIFEST.json`](MANIFEST.json)

```bash
python scripts/verify_manuscript_data_integrity.py
```
