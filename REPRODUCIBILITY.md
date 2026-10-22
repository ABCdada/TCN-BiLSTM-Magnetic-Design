# Reproducibility guide

This guide separates three levels of reproducibility:

1. **Software reproducibility**: install, test, and exercise the interfaces.
2. **Artifact reproducibility**: verify and strictly load the exact A-E model
   checkpoints.
3. **Scientific-result reproducibility**: rerun a defined evaluation on the
   original independent records and recompute the stated metrics.

The repository supports levels 1 and 2 when authorized weights are available.
Level 3 additionally requires the original data, split definitions,
normalization state, evaluation code/version, and target records.

## 1. Record the release

Before running an experiment, record:

```bash
git rev-parse HEAD
python --version
python -m pip freeze > environment.lock.txt
```

Also record the operating system, CPU/GPU model, PyTorch version, CUDA runtime,
material ID, prediction mode, sampling rate, temperature, and all environment
variables that override checkpoint paths.

For a packaged release, verify stable evidence files against
`SHA256SUMS.txt`. Maintainers can rebuild that list after an intentional release
change with `python scripts/build_release_checksums.py`.

## 2. Install

Create a fresh Python 3.10-or-newer environment and install dependencies:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a GPU run, install the official PyTorch wheel compatible with the host
driver before installing the remaining dependencies. Do not infer CUDA
compatibility from the repository alone.

## 3. Verify checkpoints

Place the five authorized checkpoint files beside the authoritative manifest:

```text
weights/A_best_model.pth
weights/B_best_model.pth
weights/C_best_model.pth
weights/D_best_model.pth
weights/E_best_model.pth
weights/manifest.json
```

Then run:

```bash
python scripts/verify_weight_manifest.py --manifest weights/manifest.json
python scripts/verify_all_models.py \
  --manifest weights/manifest.json \
  --device cpu \
  --output outputs/verification_cpu.json
```

Use `--device cuda` for a separate GPU report. The verifier checks file size,
SHA-256, material mapping, strict state-dictionary loading, parameter count,
output shape, finite values, and a deterministic output summary. Do not rebuild
the manifest before verification: that would replace trusted hashes with hashes
of the files being checked.

Models A-D must have five inputs. Model E must have six inputs, adding `ddB`.
The expected parameter counts are documented in `MODEL_CARD.md` and the
manifest.

## 4. Run automated tests

```bash
pytest -q
```

The current release contains 40 tests; 39 run without the released checkpoint
files. They cover software contracts and numerical primitives but do not
establish paper accuracy. If the count changes in a later release, preserve the
full pytest output instead of assuming a fixed number.

## 5. Generate smoke-test requests

```bash
python scripts/generate_sample_inputs.py --output-dir examples/generated
flask --app app run --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/models
curl -X POST http://127.0.0.1:8000/api/predict-window \
  -H "Content-Type: application/json" \
  --data-binary @examples/generated/predict_window_A.json
```

Generated inputs are synthetic interface fixtures. A finite response confirms
only that the software path runs.

## 6. Reproduce a scientific evaluation

For each reported evaluation:

1. Freeze independent acquisition-level train, validation, and test lists.
2. Split those acquisitions before generating overlapping windows.
3. Use the checkpoint's original normalization state.
4. Fix the material mapping, 16 MHz sample rate, 1,000-point window, prediction
   mode, window stride, and aggregation rule.
5. Preserve both predictions and measured targets with stable record IDs.
6. Compute sequence and energy metrics using the exact manuscript/challenge
   definitions and denominators.
7. Report each prediction mode separately before any across-mode or
   across-material average.
8. Save the command, code revision, environment lock, checkpoint manifest, and
   machine-readable results.

The repository includes a machine-readable transcription of the manuscript's
aggregate accuracy table under `results/paper_evaluation/`, but it does not
contain the original independent target records. Consequently, the table can
be inspected and plotted but cannot be recomputed from raw targets using this
repository alone. Verification JSON must not be presented as an accuracy
benchmark.

## 7. Energy accounting

The numerical path integral is evaluated along the supplied `B` trajectory.
Label the output according to the selected interval:

- open interval: net transient magnetic-energy exchange;
- complete closed path: cumulative closed-path energy;
- exactly one closed magnetic period: loss energy density per cycle;
- one-period loss times its cycle rate: loss power density.

Do not use switching frequency as a generic multiplier unless the selected path
really represents the corresponding closed periods.

## 8. Circuit examples

The deterministic converter examples cover a single-phase full bridge, a
three-phase full bridge, and a three-phase T-type three-level topology. They
solve a simplified SPWM plus series `R-L` state equation and then apply neural
window post-processing. They are reproducible integration demonstrations, not
device-level validation, closed-loop neural co-simulation, or measured accuracy
evidence.

See [the circuit workflow](docs/images/circuit_workflow.svg) and the three
conceptual topology diagrams in `docs/images/`.

## 9. Minimum result bundle

A result intended for comparison should contain:

- a code commit or source archive hash;
- the authoritative checkpoint manifest;
- environment and hardware metadata;
- immutable split/record identifiers;
- all input-contract fields and preprocessing settings;
- per-record predictions and metrics;
- an aggregation script and its output;
- a statement identifying synthetic, simulated, and independently measured
  records;
- energy-interval classification and units.

This bundle allows another researcher to distinguish implementation checks from
scientific validation and to trace every reported number to its inputs.
