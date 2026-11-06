# MILIEU

**MILIEU: Towards virtual tissue through context-aware modelling of perturbation responses.**

*Milieu* (from Claude Bernard’s *milieu intérieur*) denotes the local cellular and spatial context in which a perturbation acts. MILIEU models perturbation responses as **context-conditioned operators** on control-referenced expression change, with a shared formulation for dissociated Perturb-seq and spatial transcriptomics.

This repository accompanies the manuscript under review. It provides **benchmark result tables**, **CSV files to replay main and supplementary figures**, and **short demos** (matplotlib bar replay + infer-only on a small Essential slice).

Repository: https://github.com/EgoPANY/MILIEU

## Quick start

```bash
git clone https://github.com/EgoPANY/MILIEU.git
cd MILIEU
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-demo.txt
bash examples/demo/run_public_demo.sh
```

Outputs:

- `data/demo/sc/fig2b_essential_skd_seed0_demo.png` — example bar chart from [`data/figure_tables/`](data/figure_tables/README.md)
- `data/demo/sc/demo_pred.npy`, `demo_score.json` — inference on the bundled Essential test pack and demo checkpoint

Details: [docs/DEMO.md](docs/DEMO.md).

## Data in this repository

| Path | Contents |
|------|----------|
| [`data/manuscript/`](data/manuscript/README.md) | Multiseed benchmark and significance **summary tables** |
| [`data/figure_tables/`](data/figure_tables/README.md) | Plot and display CSV for main Figs 2–4 and Supp Figs S1–S3 |
| [`data/demo/sc/`](data/demo/sc/README.json) | Essential gene-OOD **seed 0** infer slice (`te_cond.npz`) + matched **demo checkpoint** (not full benchmark weights) |

## Layout

```text
examples/demo/       Figure replay and inference scripts (see docs/DEMO.md)
data/manuscript/     Result tables
data/figure_tables/  Figure replay CSV
data/demo/sc/        Demo pack and checkpoint
docs/DEMO.md
requirements-demo.txt
```

## Contact

Manuscript under review. For questions about the code:

- Open an issue on GitHub
- Email co-first authors **Yao Pan** (3200101597@zju.edu.cn) and **Shuhui Wang** (shuhui.wang@genesight.cn), or corresponding author **Yixuan Wang** (yixuan.wang@genesight.cn)

## License

MIT — [LICENSE](LICENSE). Citation: [CITATION.cff](CITATION.cff).
