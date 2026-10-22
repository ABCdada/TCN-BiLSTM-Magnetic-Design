# TCN-BiLSTM Transient Magnetic Inference

Reproducible, window-level inference for transient magnetic-field reconstruction
with the TCN-BiLSTM models described in the accompanying manuscript. The
service accepts magnetic flux density and available operating information,
reconstructs the masked part of the magnetic-field-strength trajectory, and
reports trajectory and energy-related quantities through a web interface and a
JSON API.

The repository is designed for **inference and reproducible evaluation**. It
does not contain private training data, and trained weights are not committed.
See [Model checkpoints](#model-checkpoints-and-integrity) before starting the
server.

## Scope and important limitations

- Inference uses fixed **1,000-sample windows**. The reference sampling rate is
  **16 MHz**, so one window spans 62.5 microseconds.
- Models A-D use five input channels: the complete `B` sequence, `dB`,
  temperature `T`, known values of `H`, and an `H` mask. The final-paper
  Material E checkpoint adds `ddB`, the second discrete flux increment. These
  derived channels and the mask are constructed by the backend.
- The standard history/prediction scenarios are `900-100`, `500-500`, and
  `100-900`. In each name, the first value is the number of known `H` samples
  and the second is the number reconstructed by the model.
- The network predicts the unknown interval in one forward pass. The
  `/api/predict-sequence` endpoint applies that window model repeatedly to a
  longer sequence.
- In the Design and Test Center cold-start workflow, each of the four browser
  grades (3C92, 3C95, FEC014, and T37) starts the first window from `H=0`.
  Test Center long traces use a
  one-point overlap (999-sample hop). For numerical parity with the reference
  website, Design Center instead uses the browser's raw display samples in
  non-overlapping 1,000-point windows and carries the preceding final `H` into
  the next boundary. Prediction is non-autoregressive inside each window but
  recursive between windows, so cross-window errors can still propagate.
- The public Design Center exposes only the single-phase full-bridge workflow.
  Its toroidal outer radius `R`, inner radius `r`, and thickness `d` are
  independently editable, subject only to positive dimensions and `r < R`.
  Three-phase examples remain available through the separate converter API.
- The circuit example is **window-level inference plus post-processing**. It is
  not a causal, one-timestep closed-loop co-simulation and should not be
  represented as one. A circuit solver must supply a complete `B` window (or
  sufficient voltage data to reconstruct it) before the model evaluates that
  window.
- The simple `B`/`H` geometry conversion assumes a near-uniform toroidal or
  equivalent magnetic path. Gapped, hybrid-material, strongly nonuniform, or
  leakage-dominated structures need a segmented magnetic circuit or field
  solver that supplies the applicable local quantities.
- The path integral of `H dB` over a general open transient interval represents
  net magnetic-energy exchange over that interval. Only a closed B-H cycle,
  with consistent units and orientation, yields irreversible hysteresis-loss
  energy density. Multiplication by cycles per second converts closed-cycle
  energy density (J/m3/cycle) to loss density (W/m3).

## Included material models

| ID | Material grade | Trainable parameters | Reference architecture |
|---|---|---:|---|
| A | 3C92 | 497,793 | 5-layer, 64-channel TCN + 2-layer BiLSTM (96 units per direction) |
| B | 3C95 | 2,092,545 | 7-layer, 128-channel TCN + 2-layer BiLSTM (192 units per direction) |
| C | FEC007 | 2,092,545 | 7-layer, 128-channel TCN + 2-layer BiLSTM (192 units per direction) |
| D | FEC014 | 2,512,593 | 7-layer, 144-channel TCN + 2-layer BiLSTM (208 units per direction) |
| E | T37 | 2,092,865 | 7-layer, 128-channel TCN + 2-layer BiLSTM (192 units per direction), with `ddB` input |

These identifiers define the five-model **scientific** registry and remain
fixed throughout its API, checksum manifest, and tests. The browser cold-start
release is intentionally narrower: Design Center, Test Center, and Model
Libraries expose only 3C92, 3C95, FEC014, and T37, displaying the grade names
without the internal A/B/D/E identifiers. The C/FEC007 scientific checkpoint is
retained for reproducible scientific requests but is not registered in the
browser cold-start family. Do not silently substitute similarly named ferrites.

## Repository layout

```text
.
|-- app.py                    # Flask application and JSON endpoints
|-- magnetic_model/           # model, registry, loading, inference, physics
|-- templates/                # browser interface
|-- static/                   # browser assets
|-- weights/                  # local checkpoints; not committed
|-- examples/                 # API clients and manifest example
|-- scripts/                  # deterministic sample and manifest utilities
|-- tests/                    # private-data-free verification tests
|-- results/                  # paper, PFC, verification, and synthetic artifacts
|-- docs/                     # original architecture/workflow/topology figures
|-- MODEL_CARD.md             # model scope, inputs, limitations, validation
|-- DATA_CARD.md              # data contract, split integrity, provenance
|-- REPRODUCIBILITY.md        # three-level reproduction guide
|-- Dockerfile
|-- requirements.txt
`-- README.md
```

Key release documents and evidence:

- [Model card](MODEL_CARD.md), [data card](DATA_CARD.md), and
  [reproducibility guide](REPRODUCIBILITY.md);
- [results index](results/README.md), including the manuscript P95 table,
  independent PFC validation, strict CPU/GPU checks, and 16 complete synthetic
  request/response cases;
- [reference-project parity map](docs/REFERENCE_PARITY.md) and original
  [model](docs/images/model_architecture.svg) and
  [circuit-workflow](docs/images/circuit_workflow.svg) diagrams;
- [citation metadata](CITATION.cff), [training-scope note](TRAINING.md), and
  release/community files for GitHub.
- [deployment guide](DEPLOYMENT.md), [API reference](API.md),
  [testing guide](TESTING.md), [release notes](RELEASE_NOTES.md), and
  [GitHub publication checklist](GITHUB_UPLOAD_CHECKLIST.md).

## Release packages

The release is distributed in two forms. The **GitHub source archive** contains
the application, documentation, tests, examples, results, and checksum
manifests, but intentionally omits `.pth` files. The **full deployment archive**
contains the same tree plus the five scientific and four browser cold-start
checkpoints verified by their manifests. Optional scientific-weight and
cold-start-weight ZIP files are also suitable as GitHub Release assets after a
redistribution-rights review. See [MODEL_WEIGHTS_NOTICE.md](MODEL_WEIGHTS_NOTICE.md).

The browser pages currently obtain Tailwind CSS and Chart.js from public CDNs.
Backend inference works locally, but the complete browser presentation requires
network access unless those third-party assets are vendored.

## Installation

The project has been verified on Windows with Python 3.13/PyTorch 2.7 CPU and
on AutoDL with Python 3.12/PyTorch 2.8/CUDA 12.8. `requirements.txt` declares
compatible version ranges rather than a frozen environment; save `pip freeze`
when publishing a result. Use a fresh virtual environment.

### CPU (Windows PowerShell)

```powershell
cd C:\path\to\tcn_bilstm_magnetic_circuit
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### CPU (Linux/macOS)

```bash
cd /path/to/tcn_bilstm_magnetic_circuit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### NVIDIA GPU

Install a PyTorch wheel that matches the available driver using the official
PyTorch package selector. For a driver that supports CUDA 12.8:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Changing only CPU versus CUDA kernels can produce very small floating-point
differences. Record the environment and hardware when publishing metrics.

### AutoDL

The following commands keep the project and model weights on the data disk.
Replace the repository URL and checkpoint source with locations you control;
never commit an SSH password or access token.

```bash
ssh -p <PORT> root@<AUTODL_HOST>
cd /path/to/persistent-data
git clone https://github.com/ABCdada/TCN-BiLSTM-Magnetic-Design.git tcn_bilstm_magnetic_circuit
cd tcn_bilstm_magnetic_circuit
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p weights
# Copy A_best_model.pth ... E_best_model.pth together with the release's
# authoritative manifest.json. Do not rebuild it during ordinary deployment.
python scripts/verify_weight_manifest.py --manifest weights/manifest.json
python scripts/verify_all_models.py --device cpu --output outputs/model_verification.json
pytest -q
waitress-serve --listen=0.0.0.0:8000 app:app
```

Expose TCP port 8000 using the AutoDL custom-service/port-mapping interface.
Use the assigned HTTPS URL rather than opening an unrestricted public port.

## Model checkpoints and integrity

Place the five original checkpoints without renaming their internal keys:

```text
weights/
|-- A_best_model.pth          # A = 3C92
|-- B_best_model.pth          # B = 3C95
|-- C_best_model.pth          # C = FEC007
|-- D_best_model.pth          # D = FEC014
|-- E_best_model.pth          # E = T37
`-- manifest.json
```

Each checkpoint is expected to be a dictionary containing at least a `model`
state dictionary and the normalization data used during training. The server
loads with `map_location`, validates architecture compatibility, and verifies
file size plus SHA-256 against the release manifest before use; it never uses
a filename alone as evidence that a checkpoint is correct.

Set `MAGNET_MODEL_DIR` to use a different checkpoint directory. Individual
files can be overridden with `MAGNET_CHECKPOINT_A` through
`MAGNET_CHECKPOINT_E`; this is useful for controlled comparisons and must be
recorded with the reported results.

Verify the authoritative SHA-256 manifest supplied with the release:

```bash
python scripts/verify_weight_manifest.py --manifest weights/manifest.json
python scripts/verify_all_models.py --device cpu --output outputs/model_verification.json
```

`scripts/build_weight_manifest.py` is a maintainer-only release utility. Do not
run it over downloaded files before verification, because doing so would
replace trusted hashes with hashes of whatever files are present.

The manifest format is versioned. A minimal record is:

```json
{
  "format_version": 1,
  "algorithm": "sha256",
  "models": {
    "A": {
      "material": "3C92",
      "parameter_count": 497793,
      "path": "A_best_model.pth",
      "bytes": 2006453,
      "sha256": "64 lowercase hexadecimal characters"
    }
  }
}
```

An example containing all five records is provided at
`examples/model_manifest.example.json`. Commit the completed manifest only if
publishing the corresponding checkpoints is permitted.

## Run the service

For local development:

```bash
flask --app app run --host 127.0.0.1 --port 8000
```

For a reproducible local/remote service:

```bash
waitress-serve --listen=0.0.0.0:8000 app:app
```

Open <http://127.0.0.1:8000>. Health and model discovery do not require a
prediction request:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/models
```

Design Center, Test Center, and Model Libraries show the four browser grades by
name only: 3C92, 3C95, FEC014, and T37. In Model Libraries, the baseline and
hyperparameter controls are editable and the selected configuration is stored
locally in the browser. File upload, server-side training, and model/report
export remain deliberately unavailable in this inference-only release; editing
the fields does not launch or imply a training run.

## API

All errors use a structured envelope:

```json
{"error": {"code": "invalid_request", "message": "...", "details": {}}}
```

### `POST /api/predict-window`

Reconstruct one 1,000-sample window. Generate a complete request rather than
hand-writing 1,000 numbers:

```bash
python scripts/generate_sample_inputs.py --output-dir examples/generated
curl -X POST http://127.0.0.1:8000/api/predict-window \
  -H "Content-Type: application/json" \
  --data-binary @examples/generated/predict_window_A.json
```

Core request fields:

```json
{
  "material": "A",
  "B": ["1000 finite values in tesla"],
  "temperature": 25.0,
  "known_H": ["known prefix in A/m"],
  "mode": "100-900",
  "sampling_rate_hz": 16000000
}
```

`temperature` may be a scalar or a 1,000-value sequence. For A, B, D, and E,
`known_H` is required and may be the measured prefix or a 1,000-value sequence
whose unknown tail is omitted or null. Material C was trained without an
effective historical trajectory and therefore uses exactly one raw zero
placeholder (`known_H: [0.0]`). Material C also reuses that independent zero
placeholder in every stitched long-sequence window, matching its training
distribution. The response contains the reconstructed `H`
trajectory, target boundary, energy-related post-processing, and model/sampling
metadata.

### `POST /api/predict-sequence`

Apply the same 1,000-point model to a longer sequence. The request supplies
`B` with at least 1,000 values, `temperature`, `known_H`, `history_length`, and
optionally `hop_length`. Window stitching is explicit in the response metadata.
The public service rejects requests requiring more than 256 model windows.
This endpoint does not convert the network into a causal recurrent solver.
This scientific endpoint is intentionally distinct from the Design Center's
carry-last-H compatibility workflow.

### `POST /api/predict` (adapted website compatibility)

The upstream-layout Design and Test pages use a compact browser contract:

```json
{"materialId": 1, "sample": {"B": ["finite values"], "T": 25, "real_time": 0.0000625}}
```

For this scientific compatibility endpoint, material IDs 1-5 remain mappings
to A-E (3C92, 3C95, FEC007, FEC014, and T37). Design Center and Test Center use
the separate browser-facing `/api/predict-cold-start` endpoint, whose compact
IDs are instead 1=3C92, 2=3C95, 3=FEC014, and 4=T37. The pages display only
those grade names; A, B, D, and E are internal registry identifiers. Each
browser material follows the native
1-known/999-predicted contract, whose sole initial H value is the raw zero
placeholder used in training. The response preserves the upstream flat keys `H_pred`,
`mu`, and `core_loss`, with `H_pred` and `mu` equal in length to the submitted
B sequence. Test Center supplies its uploaded sequence on the 16 MHz model
grid. Design Center explicitly requests `original_design_raw` compatibility:
the browser's 10 points per switching period are passed directly to the model
without 16 MHz interpolation, matching the reference site's numerical path.

For a native cold-start selection longer than one model window, the compatibility
endpoint preserves the complete selected interval. Test Center uses the native
one-sample-overlap recursion with a 999-sample hop. Design Center uses the
reference-compatible non-overlapping 1,000-sample hop, carries the previous
final H, and applies the reference tail-padding rule. The response `stitch`
object reports the selected strategy, window count, hop length, model sample
count, and returned sample count.

For contract compatibility the field is still named `core_loss`, but its
value is the magnitude of the trapezoidal `H dB` line integral divided by the
submitted interval duration, in W/m3. For an open transient path it is an
interval magnetic-energy exchange rate, not irreversible core loss. It may be
interpreted as core-loss density only when the submitted path is a complete
closed magnetization cycle.

After an interval is selected, the Design Center also multiplies this rate
density by the effective core volume `Ae * Le` and displays the resulting total
power in watts.  The total is labelled as estimated core loss only when the
predicted B-H path closes; for an open path it remains interval magnetic-energy
exchange power.

The compact upstream request does not include measured historical H. The Design
and Test Centers therefore use the separate four-grade cold-start family, in
which each material starts from one raw-zero H placeholder. Backward-compatible
direct `/api/predict` requests for A, B, D, or E still use the documented
zero-history compatibility convention and are **not** equivalent to the
measured-history evaluation protocol used for the manuscript. Use
`/api/predict-window` or `/api/predict-sequence` with explicit `known_H` for
scientific evaluation of those materials.

### `POST /api/predict-cold-start` (isolated four-grade cold-start family)

This endpoint uses the same compact request/response shape as `/api/predict`,
but resolves models through a separate registry and
`weights/cold_start_manifest.json`. Every model has the native `1-999` contract:
the first H value is a raw zero placeholder and the remaining 999 values are
predicted. Long sequences use one-sample-overlap recursive windows and carry the
previous final H at each boundary. The deployed release contains dedicated
cold-start checkpoints for the four exposed grades (internally A, B, D, and E).
The C/FEC007 scientific checkpoint, and any separately archived C cold-start
artifact, remain preserved but are deliberately absent from
`cold_start_manifest.json`, this endpoint, and the browser selectors. A source
checkout without the optional four cold-start weight files reports
`model_unavailable` until the files are installed and registered. Installing
them never replaces any scientific A--E weight, including C/FEC007.

### `POST /api/circuit-demo`

Demonstrate electrical-to-magnetic preprocessing and window inference:

```json
{
  "material": "A",
  "induced_voltage": ["finite magnetic-core induced-voltage samples"],
  "current": ["100 known winding-current samples"],
  "temperature": 25.0,
  "turns": 20,
  "area_m2": 0.0001,
  "path_length_m": 0.08,
  "sampling_rate_hz": 16000000,
  "initial_B": 0.0,
  "history_length": 100
}
```

The full induced-voltage window is integrated to obtain `B`. It must represent
the magnetic-core induced voltage after subtracting winding-resistance,
leakage-inductance, and relevant parasitic voltage drops; raw terminal voltage
is not interchangeable. Supply either `current`
with exactly the known-prefix length or `known_H` with that length; the unknown
target current is not required. For an independent validation, optionally add
`reference_current` with the full waveform. A legacy full-length `current`
array is also accepted and is treated as both the prefix source and validation
reference. Relative validation metrics are `null` with an explanatory reason
when no full reference is supplied.

If the paper/challenge energy-error normalization is required, supply the
measured complete-cycle denominator as positive `reference_loss_density` in
J/m3. Without it, the reported relative energy error uses the magnitude of the
measured target-interval H-dB integral, which is intentionally labeled as a
different denominator.

The response returns reconstructed `B`, full reconstructed `H`, target-only
`predicted_H`, and the magnetic-branch
current obtained from `H`, effective path length, and turns. Because the
BiLSTM needs the complete B window, this is a block/window interface rather
than a causal one-timestep constitutive element. A tightly coupled circuit
simulation must provide an external block iteration and convergence policy.
Real deployment must additionally address sensor offsets, voltage-integration
drift, synchronization, anti-aliasing, core geometry uncertainty, and the
solver's chosen window-update policy.

The `H = N i / le` and `B = integral(v dt)/(N Ae)` conversions are direct only
for a near-uniform toroidal or equivalent magnetic path. For gapped,
hybrid-material, strongly nonuniform, or leakage-dominated structures, use a
segmented magnetic network or field solver and pass the appropriate local
quantities to the model interface.

Python clients are in `examples/python_client.py` and
`examples/circuit_demo_client.py`.

### `POST /api/converter-demo`

Run a deterministic 1,000-sample grid-side converter/filter window before the
magnetic post-processing step. Supported topologies are
`single_phase_full_bridge`, `three_phase_full_bridge`, and
`three_phase_t_type`. The numerical model uses SPWM and the explicit Euler
state equation `L di/dt = v_inverter - v_grid - R i` at 16 MHz. For example:

```json
{
  "material": "A",
  "topology": "single_phase_full_bridge",
  "mode": "100-900",
  "temperature": 25.0,
  "dc_voltage": 50.0,
  "grid_voltage_rms": 20.0,
  "grid_frequency_hz": 50.0,
  "switching_frequency_hz": 100000.0,
  "filter_inductance_h": 0.000145,
  "series_resistance_ohm": 0.05,
  "initial_current_A": 0.0,
  "grid_phase_deg": 0.0,
  "turns": 20,
  "area_m2": 0.000025,
  "path_length_m": 0.06,
  "initial_B": 0.0,
  "sampling_rate_hz": 16000000
}
```

The response includes grid, inverter, induced-voltage, and nominal fixed-L
current waveforms, followed by the reconstructed `B`, `H`, and magnetic-branch
current. The nominal current supplies only the known model history; it is not
treated as measured validation data. This mirrors the scope of a deterministic
converter-state example plus AI window post-processing. It is not a switching-
device model, a controller validation, or a neural constitutive law fed back at
every circuit time step. Use `examples/converter_demo_client.py` or the
`converter-demo` CLI command for scripted runs.

## Energy reporting

For discrete samples, the implementation evaluates the line integral using
trapezoidal `H` values along increments of `B`. Keep the following quantities
distinct:

1. **Open-interval integral**: net transient magnetic-energy exchange per unit
   volume over the selected interval. It may include changes in stored energy.
2. **Closed-path integral**: absolute integral over the complete supplied
   closed path. It is loss energy density per cycle only if that path contains
   exactly one magnetic period; a multi-period record is cumulative.
3. **Core-loss density**: one-period loss energy density multiplied by the cycle
   rate. A switching frequency is not a valid multiplier unless the selected
   samples actually form the corresponding closed cycles.

The API labels these separately and does not call every open-window integral a
"core loss."

## Reproduce and test

Generate deterministic, non-proprietary smoke-test input:

```bash
python scripts/generate_sample_inputs.py --output-dir examples/generated
```

Generate the complete tracked request/response/CSV/SVG example bundle:

```bash
python scripts/generate_reproducible_bundle.py \
  --project-root . \
  --output results/reproducible_examples \
  --device cpu
```

The generated bundle is explicitly labelled as synthetic software
verification. It does not contain a measured `H` continuation and therefore
does not report prediction-accuracy metrics.

Run the private-data-free test suite:

```bash
pytest -q
```

After installing real checkpoints, verify every hash, load all five state
dictionaries strictly, confirm exact parameter counts, and run one deterministic
1,000-point forward pass per model:

```bash
python scripts/verify_all_models.py \
  --manifest weights/manifest.json \
  --device cpu \
  --output outputs/model_verification.json
```

Use `--device cuda` to produce a separate GPU report. The report records the
environment, checkpoint hashes, output summaries, and an output-byte hash.
CPU and GPU output hashes may differ because floating-point kernels differ;
finite output, shape, checkpoint integrity, and parameter counts are the
portable checks.

The tests cover the hand-calculated B-H path integral, the material/parameter
registry, JSON schema rejection, and a dummy checkpoint/model forward pass.
They verify software wiring, not the accuracy claims of the manuscript. Exact
accuracy reproduction requires the original checkpoints, normalization state,
evaluation records, and split definitions.

Before reporting a result, preserve:

- Git commit ID and `weights/manifest.json`;
- `python --version`, `pip freeze`, PyTorch/CUDA versions, and GPU model;
- material ID, sampling rate, history length, hop length, and temperature;
- whether a metric covers an open interval or complete closed cycles;
- the original acquisition-level train/validation/test split performed before
  overlapping windows are generated.

## Docker

The default image installs CPU-only PyTorch:

```bash
docker build -t tcn-bilstm-magnetic:cpu .
docker run --rm -p 8000:8000 \
  -v "$PWD/weights:/app/weights:ro" \
  tcn-bilstm-magnetic:cpu
```

On PowerShell, use `${PWD}/weights:/app/weights:ro` for the volume argument.
The image does not embed model weights. For GPU serving, start from an NVIDIA
CUDA runtime compatible with the chosen PyTorch wheel, install the CUDA build,
and run with the NVIDIA Container Toolkit; record the exact base-image digest.

## Relationship to prior open-source work

The public [LSTM-DAO-Inference](https://github.com/ouihon/LSTM-DAO-Inference)
project provides the MIT-licensed web layout adapted by this release. The
prediction backend, five-model scientific registry, four-grade browser
cold-start registry, checkpoints, normalization, validation code, and
scientific result files use the different TCN-BiLSTM implementation supplied
by this project. A file/function-level mapping and the different safety
boundaries are documented in
[`docs/REFERENCE_PARITY.md`](docs/REFERENCE_PARITY.md). See also
`THIRD_PARTY_NOTICES.md`.

## License

Project-original source code and documentation are released under the MIT
License. Adapted interface portions retain the upstream MIT notice at
`licenses/LSTM-DAO-Inference-LICENSE.txt`. Checkpoints and datasets may have
separate terms; see `LICENSE` and `THIRD_PARTY_NOTICES.md`.
