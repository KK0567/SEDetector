# SEDetector Experiments

This directory contains all experiment scripts for the SEDetector paper revision.
Each subdirectory corresponds to a specific experiment or analysis.

## Directory Structure

| Directory | Description |
|-----------|-------------|
| `audit/` | Leakage audit of training data and features |
| `baselines/` | Baseline models (HGAT, AllSet) for comparison |
| `irreversibility/` | Irreversibility / singling-out & linkability analysis |
| `interpretability/` | Perturbation-based faithfulness analysis |
| `cross_domain/` | Cross-domain transfer experiments |
| `dapt2020/` | DAPT2020 chronological split experiments |
| `scalability/` | Runtime measurement and scalability experiments |
| `privacy/` | Privacy attack framework (attribute inference) |
| `k_anonymity/` | K-anonymity analysis |
| `seed_experiments/` | Multi-seed (5 seeds) main experiment orchestration |
| `ablation/` | Ablation study scripts (model-level + data-level) |
| `ablation_analysis/` | RawHG v2, dummy baselines, balanced accuracy analysis |
| `stat_test/` | Statistical significance testing (paired t-test) |

## Project Layout

These scripts expect the following directory structure at the **project root** (parent of `experiments/`):

```
SEDetector/
  DAPT/                    # DAPT dataset
    data/                  # train/val/test JSON files
    dataset/               # preprocessing scripts
    src/                   # model source code
    Result/                # trained checkpoints + predictions
  OpTC/                    # OpTC dataset (same structure)
  TCE5/                    # TCE5 dataset (same structure)
  experiments/             # this directory
  progress_OPTC/           # training outputs (generated)
  progress_TCE5/           # training outputs (generated)
  progress_DAPT/           # training outputs (generated)
  data_OPTC/               # preprocessed hypergraph data
  data_TCE5/               # preprocessed hypergraph data
  data_DAPT/               # preprocessed hypergraph data
```

The `progress_*` and `data_*` directories are generated during training and
contain checkpoints, predictions, and preprocessed hypergraph files.

## Environment

- Python 3.10+ with PyTorch, NumPy, scikit-learn, SciPy, NLTK
- Activate the conda environment before running any script:
  ```bash
  conda activate torch
  ```

## Usage

Most scripts can be run from the project root:

```bash
# Statistical significance test
python experiments/stat_test/run_significance_test.py

# Run ablation studies
python experiments/ablation/run_all.py --variants Abl_NoKD --datasets OPTC

# Scalability measurement
python experiments/scalability/run_all.py
```

Each script contains usage instructions in its docstring.

## Ablation Variants

| Variant | Description |
|---------|-------------|
| `Abl_NoKD` | Without knowledge distillation |
| `Abl_NoSEU` | Without Semantic Evidence Unit |
| `Abl_RawHG` | Raw hypergraph features (no SEU abstraction) |
| `Abl_SEU_MLP` | SEU with MLP encoder (no hypergraph message passing) |
| `Abl_SEU_GCN` | SEU with GCN encoder |
| `NoOpCat` | Without operator-category aggregation (data-level) |
| `NoTemplAbs` | Without template abstraction (data-level) |
| `NoRole` | Without role tokens (data-level) |
