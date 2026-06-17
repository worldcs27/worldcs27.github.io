# AdaPCLA Anonymous Code Release

This archive accompanies the anonymous submission **"AdaPCLA: Curriculum Prior Internalization for Long-Tailed Longitudinal EHR Generation"**.

It contains the code used to train AdaPCLA, run the main ablations, evaluate downstream utility and tail plausibility, analyze prior internalization, run zero-shot distribution control, and reproduce the paper figures. Preprocessed MIMIC data and pretrained checkpoints are not included because of data-use restrictions and file size.

## Repository Layout

```text
AdaPCLA_anonymous_code/
|-- README.md
|-- requirements.txt
|-- paths_config.py
|-- fame/myfame/
|   |-- baseline/model7/      # MIMIC-III HALO/PCLA-style backbone code
|   |-- baseline/model8/      # MIMIC-IV HALO/PCLA-style backbone code
|   `-- evaluate/             # Downstream and tail-plausibility evaluation
|-- experiments/
|   |-- model1/               # PCLA baseline reproduction
|   |-- model2/               # Learnable-bias ablation
|   |-- model3/               # AdaPCLA training on MIMIC-III
|   |-- model4/               # Prior-only baseline
|   |-- model5/               # AdaPCLA training on MIMIC-IV
|   |-- model6/               # Prior-internalization and empirical NTK analysis
|   `-- zero_shot/            # Zero-shot distribution control
|-- analysis/                 # Figure and case-study scripts
`-- docs/                     # Experiment notes
```

Core AdaPCLA entry points:

- `experiments/model3/run_pcla_fixed_bias_anneal.py` for MIMIC-III.
- `experiments/model5/run_pcla_fixed_bias_anneal_mimiciv.py` for MIMIC-IV.

## Environment Setup

```bash
conda create -n adapcla python=3.10 -y
conda activate adapcla
pip install -r requirements.txt
export ADAPCLA_ROOT=/path/to/AdaPCLA_anonymous_code
```

For multi-GPU baseline reproduction, make sure `torchrun` is available in the active environment.

## Required External Assets

The following assets are required for full reproduction but are not distributed in this archive.

| Asset | Environment variable | Expected content |
| --- | --- | --- |
| MIMIC-III task data | `DATA_MIMICIII` | `trainDataset.pkl`, `valDataset.pkl`, `testDataset.pkl`, vocabulary files |
| MIMIC-IV task data | `DATA_MIMICIV` | Same task-file schema as MIMIC-III |
| MIMIC-III HALO checkpoint | `HALO_MIMICIII_CKPT` | Warm-start checkpoint for the generator |
| MIMIC-IV HALO checkpoint | `HALO_MIMICIV_CKPT` | Warm-start checkpoint for the generator |

All paths can be overridden through environment variables. Default locations are defined in `paths_config.py`.

## Quick Start

### MIMIC-III

```bash
cd $ADAPCLA_ROOT/experiments/model3
python run_pcla_fixed_bias_anneal.py \
  --data_dir $DATA_MIMICIII \
  --init_ckpt $HALO_MIMICIII_CKPT \
  --save_dir ./save_anneal/seed1 \
  --epochs 10 \
  --eval
```

### MIMIC-IV

```bash
cd $ADAPCLA_ROOT/experiments/model5
python run_pcla_fixed_bias_anneal_mimiciv.py \
  --data_dir $DATA_MIMICIV \
  --init_ckpt $HALO_MIMICIV_CKPT \
  --save_dir ./save_anneal_mimiciv/seed1 \
  --epochs 10 \
  --eval
```

Both scripts apply a data-derived prior bias during training, anneal the bias to zero, and sample without the training-time prior at generation time.

## Other Experiments

| Experiment | Directory | Entry script |
| --- | --- | --- |
| PCLA baseline reproduction | `experiments/model1` | `run_pcla_best_3seeds.py` |
| Learnable-bias ablation | `experiments/model2` | `run_pcla_learnable_bias.py` |
| Prior-only baseline | `experiments/model4` | `run_prior_only_bias.py` |
| Prior-internalization analysis | `experiments/model5`, `experiments/model6` | `compute_internalization_error.py`, `compute_entk_trajectory.py` |
| Zero-shot distribution control | `experiments/zero_shot` | `run_all_zeroshot.sh` |
| Paper figures | `analysis/star`, `analysis/extend`, `analysis/output` | See local README files |

## Reproducibility Notes

- The released scripts expect preprocessed task files rather than raw MIMIC tables.
- MIMIC data access must be obtained through the official credentialed process.
- Full training requires the external checkpoints listed above.
- The code has been anonymized for double-blind review; author names, machine-specific paths, and generated data artifacts are not included.

## Citation

Citation is withheld for double-blind review.
