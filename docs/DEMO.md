# Demos

Two scripts illustrate how to use the bundled data. Neither requires training code in this repository.

## 1 · Replay a figure panel (CPU only)

Reads a plot CSV and writes a PNG (default: Essential screen, SKD metric, seed 0):

```bash
python examples/demo/plot_fig2b_demo.py
```

Input default: `data/figure_tables/main_fig2/Fig2B_ed5_essential_multiseed_plot_data.csv`.

## 2 · Run inference on the Essential demo slice

Loads `data/demo/sc/te_cond.npz` and `demo_cloud_ft.pt`, runs one forward pass, and writes `demo_pred.npy` and `demo_score.json`:

```bash
python examples/demo/run_infer_demo.py
```

The network definition used for loading the checkpoint lives in `examples/demo/sc_infer/` (inference only).

## Both demos

```bash
bash examples/demo/run_public_demo.sh
```

## Scope

The inference bundle is a **worked example** from the same Replogle K562 Essential gene-OOD setup used in the paper: split **seed 0**, held-out primaries **SON**, **MED19**, and **DDX10**, with real condition-level Δ and ESM inputs from that screen (HVG subset plus a short cloud fine-tune that produces the bundled checkpoint). It shows how to load the pack and run one forward pass together with the table and figure CSVs here.

**Context pools (16-D).** Full training uses a higher-dimensional PCA-N pool per condition. Here we keep the same pooling recipe but ship only the **first 16** components in `te_cond.npz`, so `demo_cloud_ft.pt` fits GitHub’s **100 MiB** per-file limit. The bundled checkpoint matches that 16-D representation.

## Reference environment (inference)

Example outputs on the machine used to smoke-test the demo:

| | |
|--|--|
| CPU | Intel Xeon, x86_64 (90 logical CPUs) |
| GPU | NVIDIA H20, driver 535.183.06 |
| PyTorch / CUDA | 2.10.0+cu128 / 12.8 (`requirements-demo.txt`) |

The script uses CUDA when available, otherwise CPU. Minor differences in `demo_score.json` on other hardware (device, sampling in the cloud head) are normal.
