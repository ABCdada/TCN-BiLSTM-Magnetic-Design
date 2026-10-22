# Results and evidence

This directory contains stable, reviewable artifacts intended for a source-code
release.  It is deliberately separate from `outputs/`, which is a disposable
runtime directory.

## Contents

- `verification/` records software checks performed on the local CPU and on an
  AutoDL CUDA 12.8 GPU.  These files prove checkpoint integrity, strict model
  loading, output shape, finite inference, API behavior, and test status.  They
  do **not** establish prediction accuracy.
- `reproducible_examples/` contains 16 deterministic request/response pairs,
  pointwise CSV exports, editable SVG figures, summary tables, and a provenance
  manifest generated from synthetic waveforms. They are smoke-test fixtures,
  not measurements and not benchmark evidence.
- `cold_start_smoke/` contains real-checkpoint outputs for all four browser
  grades at 1,000 points plus a 2,200-point recursive-stitching case. The input
  is project-generated and the outputs verify deployment behavior, not accuracy.
- `cold_start_training/` records the held-out acquisition-row metrics,
  parameter counts, training caps, and checkpoint identities for A, B, D, and E.
- `paper_evaluation/` contains reported manuscript metrics and compact copies of
  the corresponding training reports/error-distribution figures.  Reported
  test metrics are transcribed from the manuscript and are not recomputed from
  the synthetic examples.
- `pfc_validation_metrics.csv` records the independent measured PFC validation
  described in the revised manuscript; [`PFC_VALIDATION.md`](PFC_VALIDATION.md)
  records the setup, split context, table, and figure links.

Every result class carries its own provenance note. The reproducible-example
manifest also records SHA-256 hashes for all request, response, and pointwise
CSV files. See [`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for the distinction
between software reproducibility, checkpoint reproducibility, and scientific
result reproducibility.

Never compare a synthetic example output against its synthetic fixture and cite
the result as experimental accuracy.  Reproducing the paper's P95 values
requires the original independent test records and the official evaluation
protocol.
