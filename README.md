# AdaPCLA Anonymous Release

This repository accompanies the anonymous submission of **AdaPCLA: Curriculum Prior Internalization for Long-Tailed Longitudinal EHR Generation**.

## Contents

- `AdaPCLA_anonymous_code_upload_ready.zip`: anonymous code release
- `index.html`: lightweight GitHub Pages landing page

## What is included

The code release contains:

- training scripts for AdaPCLA
- ablation and analysis scripts
- zero-shot distribution control scripts
- figure-generation scripts
- configuration helpers and documentation

## What is not included

- raw MIMIC data
- preprocessed task files
- pretrained checkpoints

These assets must be obtained separately under the relevant data-use agreements.

## Quick start

```bash
conda create -n adapcla python=3.10 -y
conda activate adapcla
pip install -r requirements.txt
export ADAPCLA_ROOT=/path/to/AdaPCLA_anonymous_code
```

See `AdaPCLA_anonymous_code_upload_ready.zip` for the full anonymous release and usage details.

