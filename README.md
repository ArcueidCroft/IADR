# IADR — Code & Data Supplement

Anonymous NeurIPS 2026 supplementary material for the paper
**"IADR: Interface-Augmented Neural Operator for Phase-Field Mean-Curvature Flow"**.

This package provides a **formally runnable, structurally complete** codebase
covering every component of the paper's experimental pipeline: the IADR model
implementation, all eight baseline / ablation classes, configuration files,
training and evaluation scripts, dataset-generation utilities, and a small
subset of the training data so the pipeline can be loaded and stepped through
end-to-end. **It is not intended to bit-reproduce the paper's reported
numbers**: the supplement is constrained by the conference upload size limit
(~100 MB), so the shipped training tensors are a sliced 80 MB subset of the
~1.8 GB standard training set, and trained checkpoints are not included.
Reproducing the paper's metrics requires regenerating the full datasets with
the scripts in Section 3.2 and retraining each model under the protocol of
the paper. The full training set, all trained checkpoints, and the public
source repository will be released on GitHub upon acceptance, under the
**Apache-2.0** licence (placeholder until release).

---

## 1. Repository layout

```
material/
├── README.md                       (this file)
├── code/
│   ├── config/                     # 9 YAML configs, one per baseline (+ template + Blind DR)
│   │   ├── IADR.yml                  # IADR
│   │   ├── IADR_DiffOnly.yml         # Hyper-D ablation
│   │   ├── IADR_ReactOnly.yml        # Hyper-R ablation
│   │   ├── IADR_UNET.yml             # U-Net variant
│   │   ├── DR1_blind_5eps.yml             # Blind DR ablation
│   │   ├── DR1_per_eps_template.yml       # template for the 8 per-eps specialists
│   │   ├── FNO_concat.yml                 # FNO baseline
│   │   ├── DR1_temp.yml                   # bare DR1 schedule template
│   │   └── FullDisco_A.yml                # DISCO baseline
│   │
│   ├── models_IADR.py              # IADR (IADR class) and the higher-order IADR2
│   ├── models_per_epsilon.py       # per-eps DR / Blind DR (DR1, DR2 standalone classes)
│   ├── models_disco_unet.py        # U-Net variant target (UNet_DISCO)
│   ├── models_FNO.py               # FNO baseline (FNO_Epsilon, FNO_Epsilon_FiLM)
│   ├── models_full_disco_A.py      # DISCO baseline core
│   ├── models_full_disco_A_runnable.py  # PL wrapper of DISCO with our dataloader
│   ├── models_ours_ablation.py     # Hyper-D, Hyper-R ablation classes
│   │
│   ├── get_datasets_circle_random.{py,sh}  # circle dataset generation
│   ├── get_datasets_fs.{py,sh}             # five-pointed-star dataset generation
│   ├── get_datasets_sd.{py,sh}             # spiral-disk dataset generation
│   │
│   ├── test_dr1_bootstrap.py       # rollout evaluation + bootstrap per-metric statistics
│   ├── log_to_xlsx_bootstrap.py    # parse rollout logs → consolidated xlsx
│   ├── run_dr1_bootstrap_serial.sh # batch evaluation driver across (model, shape, eps)
│   └── train0.sh                   # per-eps training sweep template
│
└── datasets_light/
    └── train_std_small/            # 80 MB sliced subset of the standard training set
        ├── train_dataset_5.npy           # (5, 2, 8, 1, 256, 256), float64
        ├── train_dataset_eps_5.npy       # (5,), the five training eps values
        ├── valid_dataset_5.npy           # (5, 6, 8, 1, 256, 256), float64
        └── valid_dataset_eps_5.npy       # (5,)
```

The `datasets_light/` subset is sliced from the full ~1.8 GB standard training
set so that the package fits a typical supplementary-material size budget.
It is large enough for every model class to *load and train for a few epochs*
on representative data; reproducing the paper's reported metrics requires
regenerating the full dataset (see Section 4).

---

## 2. Installation

The minimum environment for IADR, the per-eps DR specialists, the U-Net
variant, the FNO baseline, and the Hyper-D / Hyper-R / Blind DR ablations is

```bash
pip install nnpf
```

`nnpf` brings in the rest of the standard scientific stack (PyTorch, NumPy,
PyTorch Lightning, PyYAML, matplotlib, Pillow, tqdm). For the evaluation
post-processing pipeline (`log_to_xlsx_bootstrap.py`) you additionally need

```bash
pip install openpyxl
```

The DISCO baseline (`models_full_disco_A*.py` + `config/FullDisco_A.yml`) needs
the upstream DISCO repository checked out separately, plus two extra
dependencies:

```bash
pip install timm einops
git clone <DISCO repository URL>     # path elided for anonymous review
export DISCO_ROOT=/absolute/path/to/disco_repo
export PYTHONPATH="${DISCO_ROOT}:${PYTHONPATH}"
```

Without `DISCO_ROOT` set, the DISCO baseline will refuse to start; all other models run under the minimum environment alone.

A Python version of 3.9 or newer is recommended; we tested under CUDA on a single NVIDIA V100 GPU.

---

## 3. Data

### 3.1 What is in `datasets_light/`

`datasets_light/train_std_small/` contains the sliced *standard-variant*
training data used by every model in the paper except the DISCO baseline:

| File                              | Shape                          | Dtype    |
|-----------------------------------|--------------------------------|----------|
| `train_dataset_5.npy`             | `(5, 2, 8, 1, 256, 256)`       | float64  |
| `train_dataset_eps_5.npy`         | `(5,)`                         | float64  |
| `valid_dataset_5.npy`             | `(5, 6, 8, 1, 256, 256)`       | float64  |
| `valid_dataset_eps_5.npy`         | `(5,)`                         | float64  |

Axes are `(N_eps, T, N_per_eps, channel=1, H, W)`. The eight retained samples
per eps are simply the first eight along the original `N_per_eps` axis of the
full set.

The DISCO variant (16-frame context) requires a different layout
(`(N_eps, T+1 or T+6, N_per_eps, 1, H, W)` with `T = n_past = 16`). It is *not*
included in this lightweight subset; regenerate it with
`get_datasets_circle_random.py` (see Section 3.2). Likewise, the per-eps
single-eps datasets used by the per-eps DR specialists in
`config/DR1_per_eps_template.yml` are not in the subset.

### 3.2 Regenerating the full data

Each shape has a paired `.py` generator and a `.sh` launcher:

```bash
# circles (training + validation + per-eps test set)
bash get_datasets_circle_random.sh

# five-pointed star (test only)
bash get_datasets_fs.sh

# spiral disk (test only)
bash get_datasets_sd.sh
```

Each script writes `.npy` files into `train_datasets/` (creating the directory
if missing). Adjust the `N_PER_EPS`, `T`, and resolution constants near the top
of each script if you need a different sample budget.

---

## 4. Quickstart

The smallest end-to-end check is to train IADR for a handful of epochs on the
sliced data and evaluate the resulting checkpoint. From `material/`:

```bash
# 1. make the sliced data visible at the path the YAML configs expect
mkdir -p train_datasets
ln -s ../datasets_light/train_std_small train_datasets/train_std

# 2. train IADR for a few epochs (override max_epochs from the CLI)
cd code
nnpf train config/IADR.yml --max_epochs 50

# 3. evaluate the resulting checkpoint on the in-distribution circle test
python test_dr1_bootstrap.py \
    --checkpoint logs/IADR/last.ckpt \
    --test ../datasets_light/train_std_small/valid_dataset_5.npy \
    --eps_values 0.0078 \
    --batch_size 4 --device cuda
```

Step 2 produces a checkpoint under `code/logs/IADR/`. Step 3 prints a
single rollout's `L2_Tinf`, `V_Tinf`, and `AIW_t` values together with their
bootstrap standard deviations and 95% confidence intervals. Both steps complete
in a few minutes on a single GPU; they verify the pipeline rather than
reproducing the paper's reported numbers, which require the full dataset and
20,000 training epochs.

---

## 5. Models and configs

The eight rows of the paper's main results table map onto the following files:

| Paper baseline      | Model class                                                   | Config                                |
|---------------------|---------------------------------------------------------------|---------------------------------------|
| **IADR**            | `models_IADR.py:IADR`                                    | `config/IADR.yml`                |
| Per-eps DR          | `models_per_epsilon.py:DR1`                                   | `config/DR1_per_eps_template.yml`     |
| U-Net variant       | `models_disco_unet.py:UNet_DISCO`                             | `config/IADR_UNET.yml`           |
| FNO                 | `models_FNO.py:FNO_Epsilon`                                   | `config/FNO_concat.yml`               |
| DISCO               | `models_full_disco_A_runnable.py:FullDiscoA_Run`              | `config/FullDisco_A.yml`              |
| Blind DR            | `models_per_epsilon.py:DR1` (no eps input, pooled five-eps)   | `config/DR1_blind_5eps.yml`           |
| Hyper-D ablation    | `models_ours_ablation.py:IADR_DiffOnly`                  | `config/IADR_DiffOnly.yml`       |
| Hyper-R ablation    | `models_ours_ablation.py:IADR_ReactOnly`                 | `config/IADR_ReactOnly.yml`      |

`DR1` and `DR2` are defined in two places: `models_IADR.py` keeps a self-
contained copy used by the IADR / IADR pipeline, while
`models_per_epsilon.py` keeps a standalone copy that the per-eps and Blind DR
yaml configs point at. The two copies share an architecture and a state-dict
layout but live in separate files to keep the two training pipelines
independent.

---

## 6. Training

All training entry points use the `nnpf` CLI, which reads the YAML and
dispatches to the model class named in its `model:` field:

```bash
cd code
nnpf train config/<some_config>.yml
```

Every config writes into `code/logs/<version>/`, where `<version>` is the YAML
`version:` field. Override individual hyperparameters from the CLI with
`--max_epochs N`, `--seed S`, etc.

### 6.1 IADR and the partial-conditioning ablations

```bash
nnpf train config/IADR.yml             # IADR
nnpf train config/IADR_DiffOnly.yml    # Hyper-D ablation
nnpf train config/IADR_ReactOnly.yml   # Hyper-R ablation
nnpf train config/IADR_UNET.yml        # U-Net variant
```

### 6.2 Per-eps DR specialists

The paper trains eight specialists at the eps values
`{0.0065, 0.0069, 0.0072, 0.0078, 0.0081, 0.008425, 0.0090, 0.0093}`. The
template yaml in `config/DR1_per_eps_template.yml` is a starting point. For
each target eps, copy the template and replace its `version:`, `train_u`,
`train_eps`, `valid_u`, and `valid_eps` paths with the corresponding single-
eps dataset. `train0.sh` is a shell-driven sweep template that automates this
substitution and launches the eight runs one at a time.

### 6.3 Blind DR

```bash
nnpf train config/DR1_blind_5eps.yml
```

This trains a `DR1` instance on the pooled five-eps training tensor without
any eps input, providing the eps-agnostic baseline reported in Section 4.4.

### 6.4 FNO

```bash
nnpf train config/FNO_concat.yml
```

### 6.5 DISCO baseline

The DISCO config requires the supplementary environment of Section 2:

```bash
export DISCO_ROOT=/absolute/path/to/disco_repo
export PYTHONPATH="${DISCO_ROOT}:${PYTHONPATH}"
nnpf train config/FullDisco_A.yml
```

DISCO uses a 16-frame context per sample, so its training tensors are not
shared with the standard models; regenerate the appropriate `(N_eps, 17,
N_per_eps, 1, 256, 256)` train and `(N_eps, 22, N_per_eps, 1, 256, 256)`
validation tensors with `get_datasets_circle_random.py` before training.

---

## 7. Evaluation

A single rollout evaluation reads a checkpoint, replays the autoregressive
trajectory, and produces per-trajectory metrics with bootstrap standard
deviations and 95% CIs:

```bash
python test_dr1_bootstrap.py \
    --checkpoint logs/<version>/last.ckpt \
    --test path/to/test_dataset.npy \
    --eps_values 0.0078 \
    --batch_size 4 --device cuda
```

Common flags:

| flag                  | purpose                                              |
|-----------------------|------------------------------------------------------|
| `--checkpoint`        | path to a Lightning checkpoint or a directory       |
| `--test`              | path to the test `.npy` (shape `(N, T, H, W)` or grouped) |
| `--eps_values`        | comma-separated eps values, e.g. `0.0078,0.0093`     |
| `--batch_size`        | rollout batch size                                   |
| `--device`            | `cpu` or `cuda`                                      |
| `--save_rollout_preds`| optional path to dump prediction tensors             |

`run_dr1_bootstrap_serial.sh` is the batch driver that loops the call above
across the (model, shape, eps) matrix used in the paper, redirecting each
run's stdout/stderr to a per-cell log under `results_logs/`. After all cells
have run, consolidate the logs into a single workbook:

```bash
python log_to_xlsx_bootstrap.py \
    --log_dir results_logs/dr1_bootstrap_total_best/ \
    --out results_dr1_bootstrap_total_best.xlsx
```

The resulting xlsx contains one row per `(model_id, shape, eps_id)` triple
with the bootstrap mean, std, and CI for every metric, plus the per-step
inference timings.

---

## 8. Reproducing the paper's tables and figures

The table below names the script that *produces* each paper artefact;
**bit-reproducing the reported numbers is not possible from this supplement
alone**, because (a) the shipped training data is the 80 MB sliced subset and
(b) trained checkpoints are not bundled. To match the paper's numbers,
regenerate the full datasets per Section 3.2, retrain every model under the
protocol of Section 6, then run the evaluation pipeline of Section 7.

| Paper artefact                     | How it is produced                                                                                                              |
|------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| Table 1 (`tab:main`)               | Eight training runs (one per baseline) → `run_dr1_bootstrap_serial.sh` → `log_to_xlsx_bootstrap.py`. Bootstrap mean of every cell goes to Table 1. |
| Table 2 (`tab:step-time`)          | The same xlsx; column `step_time_per_sample_mean` is averaged across eps within each shape.                                     |
| Table E.1 (`tab:main-full`, App. E)| Same xlsx as Table 1; column shows `mean ± boot_std`.                                                                            |
| Table F.1 (`tab:ablation`, App. F) | Three additional training runs (Blind DR + Hyper-D + Hyper-R) → same evaluation pipeline → same xlsx.                            |
| Figure 1 (architecture)            | Static schematic, included as `fig/IADR_architecture.png` in the paper repository.                                              |
| Figure 2 (case study)              | Per-cell rollout trajectories saved by `test_dr1_bootstrap.py --save_rollout_preds`, then composited with a small plotting script (released with the GitHub version). |
| Figure E.1 (case study, App. E)    | Same procedure on the spiral-disk geometry-OOD cell.                                                                            |

The xlsx column names map to paper notation as follows:

| xlsx model_id           | paper baseline       |
|-------------------------|----------------------|
| `IADR_s1`          | IADR                 |
| `DR1_0_<eps>_s<seed>`   | per-eps DR specialist (trained at the encoded eps) |
| `DISCO_UNet`            | U-Net variant        |
| `FNO_concat`            | FNO                  |
| `FullDisco_A_s0`        | DISCO                |
| `BlindDR_5eps_s0`       | Blind DR             |
| `IADR_DiffOnly`    | Hyper-D              |
| `IADR_ReactOnly`   | Hyper-R              |

---

## 9. Notes and caveats

* **Full dataset.** The full standard training set (`train_dataset_5.npy` =
  `(5, 2, 100, 1, 256, 256)` and `valid_dataset_5.npy` =
  `(5, 6, 200, 1, 256, 256)`, both float64) is roughly 1.8 GB, which is why we
  ship a sliced 80 MB subset. The DISCO-context variant
  (`(5, 17, 100, ...)`, `(5, 22, 100, ...)`) is larger again and is generated
  on demand by `get_datasets_circle_random.py`.
* **Trained checkpoints are not released.** Reviewers can train from scratch
  with `nnpf train` against the configs above; under the protocol of the paper
  this takes between 3 h and 72 h per model on a single V100.
* **Two `DR1` definitions.** `models_IADR.py` and `models_per_epsilon.py` each
  define a `DR1` (and a `DR2`) class. The yaml configs in `config/` always
  name the source file explicitly through the `model:` field, so there is no
  ambiguity at runtime. If you import `DR1` from Python directly, pick the
  file matching your use case.
* **Post-acceptance release.** The full source repository, the trained
  checkpoints, the bootstrap evaluation logs, and the dataset-generation
  pipeline will be released on GitHub upon acceptance, under the Apache-2.0
  licence.
# IADR
