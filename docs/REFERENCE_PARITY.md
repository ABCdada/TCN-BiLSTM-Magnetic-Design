# Reference-project parity and scope

## Purpose

The public
[LSTM-DAO-Inference](https://github.com/ouihon/LSTM-DAO-Inference) repository
demonstrates a web-accessible magnetic-component workflow for a different
LSTM-DAO model. This release adapts its MIT-licensed web layout while using a
different TCN-BiLSTM backend and a stricter separation among
inference, deterministic circuit examples, software verification, and
experimental accuracy.

This document is a feature-category map, not a claim of numerical equivalence
or superiority. The adapted templates and selected media are identified in
`THIRD_PARTY_NOTICES.md`; upstream model code and weights are not included.

## Category map

| Reference category | This repository | Parity note |
|---|---|---|
| Web design interface | Adapted upstream Design Center, Model Libraries, Test Center, and Documents layout | Preserves the page layout with selectable 3C92, 3C95, FEC014, and T37 cold-start models while applying a distinct Aurora visual theme; pages show grade names without internal letter identifiers, and scientific APIs remain separate |
| Pretrained material models | Five-model scientific registry for 3C92, 3C95, FEC007, FEC014, and T37; separate four-grade browser cold-start registry for 3C92, 3C95, FEC014, and T37 | Different network definitions, checkpoints, and normalization from the reference project; C/FEC007 remains preserved in the scientific registry but is not registered in the browser cold-start release |
| Magnetic prediction API | `/api/predict-window`, `/api/predict-sequence`, and CLI commands | Adds explicit mask modes, input contracts, and structured errors |
| Energy display | Trapezoidal `H dB` line integral divided by interval duration | Open paths are labelled interval magnetic-energy exchange rate; only a complete closed cycle is interpreted as core-loss density |
| Grid-connected converter examples | Single-phase full bridge, three-phase full bridge, and three-phase T-type examples | Deterministic SPWM/R-L simulation plus AI window post-processing |
| Circuit-level integration | `/api/circuit-demo` voltage-to-`B`, model, and current workflow | Window/block interface; not timestep neural feedback |
| Downloadable example data | Tracked 1,000-point upload CSV, a format-only synthetic fine-tuning ZIP, and complete JSON/CSV request-response bundles | Every synthetic artifact is labelled as non-measured and is not accuracy evidence |
| Fine-tuning interface | Original Model Libraries layout retained with four grade-only baseline choices and editable, browser-persisted hyperparameters; upload, server training, and export remain disabled | Editing the configuration does not train a model; training and few-shot transfer require a separate controlled workflow |
| Model/download export | JSON responses, CLI output, verification JSON, and deployable Python package | Does not generate a trained-model/PDF-report bundle from the web UI |
| Integrity controls | Versioned SHA-256 manifest and strict model loading | Release-specific addition |
| Automated tests | Automated release regression suite; released checkpoints are not required for the unit-test suite | Tests software contracts, including four-grade browser mapping and recursive-window boundaries, not paper accuracy |
| Documentation | English/Chinese README, model card, data card, reproducibility guide, API docs | Explicitly documents inference and energy limitations |

## Numerical-result files

At the time this parity document was prepared, the visible reference repository
tree and README described interactive predictions, downloadable/generated
reports, and website-hosted toy data, but did **not** commit a machine-readable
benchmark-results table or a repository results directory containing the
paper's numerical evaluation records. Therefore, there is no committed
reference numerical-result artifact to mirror byte-for-byte in this project.

This repository adds a clearly separated `results/` hierarchy: a
machine-readable transcription of manuscript metrics and independent PFC
validation, checkpoint/software verification reports, and deterministic
synthetic request/response examples. The aggregate manuscript numbers are not
presented as freshly recomputed because the raw independent targets are not
redistributed. The generated examples remain synthetic smoke tests rather than
experimental evidence.

To make a future numerical comparison reproducible, publish independent target
records, immutable split identifiers, metric definitions and denominators,
per-record predictions, an aggregation script, checkpoint hashes, and full
environment metadata as described in `REPRODUCIBILITY.md`.

## Original assets

The diagrams under `docs/images/` were created for this repository from its own
software and physical-interface definitions. Selected upstream interface media
under `static/images/` are redistributed under the upstream MIT license.
