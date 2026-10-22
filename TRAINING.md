# Training and Transfer-Learning Scope

## What this release contains

This repository release is designed primarily for checkpoint-verified inference,
evaluation, and circuit-level integration. It contains:

- the TCN-BiLSTM architecture definitions and material registry for A-E;
- preprocessing and inference semantics for the supported prediction modes;
- checkpoint identity metadata in `weights/manifest.json`;
- tests that construct the models and validate inference-related contracts; and
- retained per-material `train_report.json` files under
  `results/paper_evaluation/training_reports/`.

Those reports record completed runs and help trace the architecture, seed,
selected epoch, and recorded internal metrics. They do **not** constitute a
complete, independently executable training pipeline.

## What is not currently released

The current public source tree does not include the complete code and immutable
data manifests needed to reproduce the paper's full training and few-shot
cross-material transfer-learning experiments from raw measurements. In
particular, it does not provide a supported command that regenerates every A-E
checkpoint, the paper's source-material selection study, or the final transfer
results from scratch.

The source datasets and trained binaries also have separate provenance and usage
conditions. The code license alone does not grant permission to redistribute the
MagNetX data, independently measured waveforms, or model checkpoints.

Accordingly, the present release should not be cited as proof that the paper's
training pipeline is fully reproducible. It supports reproducible execution of
the released inference code when authorized checkpoints are supplied and verified.

## Documented experimental contracts

The deployed models use 1,000 samples per window at the documented 16 MHz
sampling rate. Materials A, B, D, and E support history/prediction partitions of
100/900, 500/500, and 900/100 samples. Material C uses a one-sample zero
placeholder and predicts the remaining 999 samples.

For leakage-safe development, independent acquisition sequences must be assigned
to training, validation, and test partitions **before** overlapping window
extraction. Windows derived from one acquisition must not cross those partitions.
Preprocessing statistics and checkpoint selection must be fitted using the
training/validation scope only, with the test set retained for final evaluation.

These contracts describe the intended methodology and evaluation boundary. They
are not a substitute for missing raw-data manifests or a complete trainer.

## Requirements for a future full training release

A release claiming end-to-end training reproducibility should add, at minimum:

1. an executable trainer and few-shot fine-tuning entry point;
2. exact dependency and accelerator environment records;
3. source-data identifiers, licenses, checksums, and acquisition-level manifests;
4. deterministic split files created before window extraction;
5. all preprocessing and target-renormalization code;
6. architecture and optimizer configuration for every material;
7. fixed random seeds and nondeterminism disclosures;
8. checkpoint-selection rules that do not use the held-out test set;
9. scripts implementing the reported sequence and energy-error definitions; and
10. commands that regenerate result tables and figures from saved predictions.

Few-shot transfer experiments should additionally record the source model,
target-data budget, source-selection rule, freeze/unfreeze schedule,
layer-specific learning rates, and every target validation decision. For a
truly new material, source selection should use predefined physical/model
criteria and a validation subset rather than the held-out final test set.

## Checkpoint handling

Authorized checkpoints are placed in `weights/` using the filenames documented
in `weights/README.md`. Verify them before inference:

```bash
python scripts/verify_weight_manifest.py --manifest weights/manifest.json
```

If redistribution rights are confirmed, publish large binaries through GitHub
Release assets or Git LFS and publish their hashes. Do not silently replace a
checkpoint while retaining the old manifest or version number.
