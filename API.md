# HTTP API Reference

This document describes the routes implemented by release 1.3.2. Examples use
`http://127.0.0.1:8000`; replace that origin with the deployment's HTTPS origin
when calling a remote service.

## Conventions and limits

- JSON request endpoints require `Content-Type: application/json`.
- All numbers and array entries must be finite; `NaN` and infinity are rejected.
- Released model inference uses exactly 16 MHz. `sampling_rate_hz` may be
  omitted, but if present it must be 16000000 (within the implementation's
  one-hertz tolerance).
- A model window contains exactly 1,000 samples.
- The ordinary scientific sequence API accepts 1,000--50,000 samples and at
  most 256 model windows.
- The compact compatibility API accepts 2--200,000 browser samples and, after
  optional 16 MHz resampling, at most 2,000,000 model-grid samples and 2,048
  recursive windows.
- Flask rejects request bodies larger than 16 MiB with HTTP 413.
- Temperature may be a scalar or an array matching the associated waveform.

The two model families are isolated:

| Family | Endpoint(s) | Internal IDs | Grades | History contract |
|---|---|---|---|---|
| Scientific | `/api/predict-window`, `/api/predict-sequence`, `/api/circuit-demo`, `/api/converter-demo`, legacy `/api/predict` | A, B, C, D, E | 3C92, 3C95, FEC007, FEC014, T37 | A/B/D/E: 100-900, 500-500, 900-100; C: 1-999 |
| Browser cold start | `/api/predict-cold-start` | A, B, D, E | 3C92, 3C95, FEC014, T37 | one raw-zero H placeholder, then 999 predictions |

`C/FEC007` remains available in the scientific registry but is intentionally
absent from the Design Center, Test Center, Model Libraries, and cold-start API.

## Discovery and health

### `GET /health`

Returns process, device, and model-family availability:

```bash
curl --fail http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "version": "1.3.2",
  "device": "cpu",
  "loaded_models": [],
  "available_models": ["A", "B", "C", "D", "E"],
  "cold_start": {
    "device": "cpu",
    "loaded_models": [],
    "available_models": ["A", "B", "D", "E"]
  }
}
```

Availability means that the configured file matches its trusted manifest.
Models are loaded lazily, so an empty `loaded_models` list is normal before the
first prediction.

### `GET /api/models`

Returns the five scientific records under `models`. Each record includes its
ID, grade, architecture configuration, supported modes, input features,
parameter count, availability, and integrity state.

### `GET /api/cold-start-models`

Returns only A/3C92, B/3C95, D/FEC014, and E/T37. Their public labels are the
grade names, every record supports only `1-999`, and
`known_h_placeholder_zero` is true.

## Scientific window inference

### `POST /api/predict-window`

Reconstructs one 1,000-sample H trajectory using the scientific family.

Request fields:

| Field | Required | Description |
|---|---:|---|
| `material` | yes | `A`, `B`, `C`, `D`, or `E` |
| `B` | yes | Exactly 1,000 flux-density samples in tesla |
| `temperature` | yes | Scalar degrees Celsius or exactly 1,000 values |
| `known_H` | conditional | Known H prefix in A/m; C may omit it and defaults to `[0.0]` |
| `mode` | optional | A/B/D/E: `100-900`, `500-500`, or `900-100`; C: `1-999` |
| `history_length` | optional | 100, 500, 900, or 1, consistent with `mode` and the material |
| `sampling_rate_hz` | optional | Defaults to 16000000; no other model rate is accepted |

For A, B, D, and E, `known_H` must contain the selected prefix length. A
1,000-entry masked array is also accepted; only its known prefix is read. C
requires the literal raw-zero placeholder `[0.0]`.

Create an executable request rather than manually entering 1,000 values:

```bash
python scripts/generate_sample_inputs.py --output-dir examples/generated
curl --fail -X POST http://127.0.0.1:8000/api/predict-window \
  -H "Content-Type: application/json" \
  --data-binary @examples/generated/predict_window_A.json
```

The response includes:

```json
{
  "material": "A",
  "grade": "3C92",
  "mode": "100-900",
  "history_length": 100,
  "prediction_length": 900,
  "sampling_rate_hz": 16000000.0,
  "H": ["1,000 reconstructed values"],
  "known_H": ["100 input values"],
  "predicted_H": ["900 predicted values"],
  "energy": {
    "h_db_integral_j_per_m3": 0.0,
    "closed_path_energy_density_j_per_m3": null,
    "note": "..."
  }
}
```

The strings in the abbreviated arrays above describe lengths; they are not
literal model output.

## Scientific long-sequence inference

### `POST /api/predict-sequence`

Applies the 1,000-point scientific model to a longer B sequence. It does not
turn the bidirectional model into a causal timestep solver.

The request uses the same fields as `/api/predict-window`, except that `B` and
an array-valued `temperature` may contain 1,000--50,000 samples. Optional
`hop_length` controls window overlap and must be between 1 and
`1000 - history_length`. The default is the predicted-tail length. A request
requiring more than 256 windows is rejected.

```json
{
  "material": "B",
  "B": ["at least 1,000 finite values"],
  "temperature": 25.0,
  "known_H": ["500 measured prefix values"],
  "mode": "500-500",
  "hop_length": 500,
  "sampling_rate_hz": 16000000
}
```

The response contains the full `H`, target-only `predicted_H`, energy fields,
`window_count`, `window_length`, `hop_length`, and a `windows` list of the
stitched intervals. Overlapping estimates are averaged. For scientific
C/FEC007, every evaluation window uses the independent zero placeholder that
matches its training contract; this endpoint does not use Design Center's
carry-last-H compatibility recursion.

## Compact compatibility endpoints

### `POST /api/predict`

Legacy scientific compatibility route used by the adapted upstream contract.
Select a model with either `materialId` (1=A, 2=B, 3=C, 4=D, 5=E) or `modelId`
(`A`--`E`). Do not provide both as competing selections.

### `POST /api/predict-cold-start`

Browser-facing cold-start route. Its numeric mapping is 1=3C92, 2=3C95,
3=FEC014, and 4=T37; the equivalent internal `modelId` values are A, B, D, and
E. Every model starts from one raw-zero H placeholder and predicts the
remaining 999 samples in each model window.

Both routes accept the same compact body:

```json
{
  "materialId": 1,
  "sample": {
    "B": [0.0, 0.001],
    "T": 25.0,
    "sampling_mode": "model_16mhz"
  }
}
```

Compact `sample` fields:

| Field | Required | Description |
|---|---:|---|
| `B` | yes | 2--200,000 finite browser samples |
| `T` | yes | Scalar or array matching `B` |
| `real_time` | optional | Positive duration in seconds; omission means `len(B) / 16e6` |
| `sampling_mode` | optional | `model_16mhz` (default) or `original_design_raw` |
| `duration_semantics` | optional | `sample_count` (default) or `endpoint_span` |

With `model_16mhz`, a supplied `real_time` determines the number of samples on
the 16 MHz model grid and the server linearly resamples B and T. With
`endpoint_span`, the inclusive grid contains one additional endpoint sample.
With `original_design_raw`, the submitted points are sent directly to the model
without 16 MHz interpolation; this path exists to reproduce the reference
Design Center's numerical workflow.

Long compatibility reconstruction is available only to a model with a native
cold-start contract. Scientific A/B/D/E calls through `/api/predict` remain
limited to 50,000 model-grid samples; use the explicit scientific APIs and a
controlled offline batch workflow when that bound is unsuitable.

For a long cold-start trace, the ordinary `model_16mhz` path uses one-sample
overlap, a 999-sample hop, and carries the prior final H into the next window.
`original_design_raw` instead uses non-overlapping 1,000-point windows, carries
the preceding final H to the new boundary, and applies the reference tail
padding rule. The `stitch` response object states the exact strategy, hop,
window count, model sample count, and returned sample count.

Compact response fields are:

```json
{
  "H_pred": ["same length as submitted B"],
  "mu": ["same length as submitted B"],
  "core_loss": 0.0,
  "energy_quantity": "interval_magnetic_energy_exchange_rate",
  "energy_unit": "W/m^3",
  "material": "A",
  "grade": "3C92",
  "mode": "1-999",
  "history_length": 1,
  "stitch": {},
  "sampling_mode": "model_16mhz",
  "initial_state": "material_A_native_cold_start_H0_zero"
}
```

The legacy key `core_loss` is the magnitude of the trapezoidal H-dB line
integral divided by the submitted interval duration. For an open transient it
is an interval magnetic-energy exchange rate, not irreversible core-loss
density. It is core-loss density only when the supplied path is one complete,
closed magnetic cycle.

Direct `/api/predict` calls for scientific A/B/D/E do not contain measured H
history and therefore use a compatibility zero-history convention. They are
not equivalent to the manuscript evaluation protocol. Use
`/api/predict-window` or `/api/predict-sequence` with measured `known_H` for
scientific evaluation.

## Circuit integration

### `POST /api/circuit-demo`

Integrates a magnetic-core induced-voltage trace to obtain B, runs scientific
window or sequence inference, and converts predicted H to magnetizing current.

```json
{
  "material": "A",
  "mode": "100-900",
  "induced_voltage": ["1,000 to 50,000 finite voltage samples"],
  "current": ["100 known winding-current samples"],
  "temperature": 25.0,
  "turns": 20,
  "area_m2": 0.0001,
  "path_length_m": 0.08,
  "sampling_rate_hz": 16000000,
  "initial_B": 0.0
}
```

Required physical fields are positive `turns`, `area_m2`, and
`path_length_m`; `initial_B` defaults to zero. Use `induced_voltage`. The
deprecated `voltage` alias is accepted only when `induced_voltage` is absent.
The input must be the voltage induced by the core after removing winding
resistance, leakage inductance, and relevant parasitic drops; raw terminal
voltage is not interchangeable.

Supply the known state in one of these forms:

- `known_H` with the selected history length; or
- `current` with the selected history length, which is converted using
  `H = N i / le`; or
- legacy full-length `current`, which provides both the prefix and an optional
  validation reference.

Scientific C/FEC007 always uses its trained one-value zero placeholder. A
supplied current prefix does not replace that placeholder; a full
`reference_current` may still be used for independent validation.

For validation, prefer a separate full-length `reference_current`. Optional
positive `reference_loss_density` supplies the complete-cycle energy-error
denominator in J/m3. Without a reference current, relative metrics are `null`
and include a reason. For sequences, optional `hop_length` follows the
scientific sequence rules.

The response contains reconstructed `B`, full reconstructed `H`, target-only
`predicted_H`, predicted magnetizing current, optional reference quantities,
prediction metadata, geometry, and validation/energy metrics. Because the
network is bidirectional, the complete B window must be available before
inference; this is a block interface, not a causal constitutive element.

### `POST /api/converter-demo`

Generates a deterministic 1,000-sample SPWM converter/filter window and then
uses the same scientific circuit pipeline. Supported `topology` values are:

- `single_phase_full_bridge`
- `three_phase_full_bridge`
- `three_phase_t_type`

```json
{
  "material": "A",
  "mode": "100-900",
  "topology": "single_phase_full_bridge",
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

DC voltage, RMS grid voltage, grid and switching frequencies, filter
inductance, turns, area, and path length must be positive. Series resistance
must be nonnegative. Initial current, phase, and B may be zero or signed.

The response extends `/api/circuit-demo` with a `simulation` object containing
time, grid voltage, inverter voltage, induced voltage, and nominal fixed-L
current. That nominal current supplies only the known model prefix; it is not
measured data and does not produce validation metrics. This endpoint is an
integration example, not a switching-device or controller validation.

## Disabled training routes

### `POST /api/upload`

### `POST /api/finetune`

Both return HTTP 501 in this inference release:

```json
{
  "error": "Browser-side training is not enabled in this inference release. Use the reproducible offline training workflow and the downloadable data-format example from the Resources page."
}
```

The Model Libraries page permits local editing of a configuration preview, but
it does not upload data, start server-side training, create weights, or export
a trained model.

## Error responses

Most API failures use this envelope:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "human-readable explanation",
    "details": {}
  }
}
```

| HTTP status | Code | Meaning |
|---:|---|---|
| 400 | `invalid_request` | JSON schema, length, mode, sampling, or physical-value validation failed |
| 400 | `invalid_value` | A downstream numerical or checkpoint-value check failed |
| 400 | `bad_request` | Malformed JSON was rejected by Flask/Werkzeug |
| 404 | `not_found` | Unknown API route or unavailable registered download |
| 413 | `request_too_large` | Body exceeds 16 MiB |
| 500 | `internal_error` | Unexpected server failure; implementation detail is not exposed |
| 503 | `model_unavailable` | Required checkpoint is absent or deliberately pending |

The two disabled training routes are the documented exception: they return
HTTP 501 with a flat string-valued `error` field.

Treat non-2xx responses as failures and read the JSON body before retrying. A
400 response caused by an integrity mismatch is not retryable until the
deployment's weights and manifests are repaired.

## Browser and artifact routes

These routes return HTML or files rather than JSON:

| Method and route | Purpose |
|---|---|
| `GET /` | Landing page |
| `GET /predict` | Scientific window-inference page |
| `GET /test` | Reference-layout Test Center |
| `GET /circuit` | Circuit/converter demonstration page |
| `GET /design` | Reference-layout Design Center |
| `GET /docs` | Technical documentation page |
| `GET /document` | Reference-layout documentation page |
| `GET /fine_tune` | Model Libraries configuration page |
| `GET /resources` | Downloadable artifact index |
| `GET /downloads/<artifact_id>` | Allow-listed release artifact download |
| `GET /resources/<resource_name>` | Static resource download |
| `GET /licenses/<license_name>` | Bundled license text |

Registered artifact IDs for `/downloads/<artifact_id>` are
`paper-p95-csv`, `paper-metrics-json`, `pfc-validation-csv`,
`pareto-sequence-svg`, `pareto-energy-svg`, `synthetic-upload-csv`,
`synthetic-finetune-format`, `synthetic-manifest-json`, and
`gpu-verification-json`. The server returns 404 when an ID is not registered or
its file is absent.
