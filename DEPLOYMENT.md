# Deployment Guide

This guide deploys release 1.3.1 as an inference service. The application is
not a browser-based training service: `/api/upload` and `/api/finetune` are
deliberately disabled, and production deployments should treat model files and
their manifests as immutable release artifacts.

## 1. Requirements

- Python 3.10 or newer.
- A CPU installation of PyTorch, or a CUDA build compatible with the host GPU
  driver.
- Enough memory for the selected checkpoints. Models are loaded lazily on the
  first prediction and then cached in the process.
- Network access from the browser if the reference-layout pages are used. Some
  pages load Tailwind CSS, Chart.js 4.4.0, and chartjs-plugin-zoom 2.0.1 from
  public CDNs. The JSON API and `/health` do not depend on those CDNs.

Use a fresh virtual environment. `requirements.txt` declares supported version
ranges; retain the environment snapshot supplied with a release, or record
`python -m pip freeze`, when a result must be reproduced exactly.

## 2. Checkpoint layout and integrity

The complete deployment bundle uses one directory for two isolated model
families:

```text
weights/
|-- manifest.json
|-- A_best_model.pth
|-- B_best_model.pth
|-- C_best_model.pth
|-- D_best_model.pth
|-- E_best_model.pth
|-- cold_start_manifest.json
|-- A_cold_start_best_model.pth
|-- B_cold_start_best_model.pth
|-- D_cold_start_best_model.pth
`-- E_cold_start_best_model.pth
```

`manifest.json` is the scientific A--E family. Its public IDs and grades are
A/3C92, B/3C95, C/FEC007, D/FEC014, and E/T37. The separate
`cold_start_manifest.json` has profile `cold_start_1_999` and contains only the
four browser-facing cold-start models A/3C92, B/3C95, D/FEC014, and E/T37.
Do not rename, substitute, or copy a scientific checkpoint into a cold-start
slot.

Verify both manifests before starting the service. The first command validates
the five scientific files. The second uses the same loader used by the running
cold-start service and validates the cold-start profile, model set, byte counts,
and SHA-256 hashes:

```bash
python scripts/verify_weight_manifest.py --manifest weights/manifest.json
python scripts/verify_cold_start_manifest.py --manifest weights/cold_start_manifest.json
```

The application checks manifest integrity again before a checkpoint is loaded.
Do not run `scripts/build_weight_manifest.py` over downloaded files as a
substitute for verification; that would create new trust values from the files
being checked.

## 3. Windows deployment

From the release root in PowerShell:

```powershell
py -3 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\verify_weight_manifest.py --manifest weights\manifest.json
python scripts\verify_cold_start_manifest.py --manifest weights\cold_start_manifest.json
python -m pytest -q
waitress-serve --listen=127.0.0.1:8000 app:app
```

Keep the listener on `127.0.0.1` for a local workstation. For a managed server,
place a TLS reverse proxy in front of Waitress before binding to an externally
reachable interface.

## 4. Linux deployment

From the release root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_weight_manifest.py --manifest weights/manifest.json
python scripts/verify_cold_start_manifest.py --manifest weights/cold_start_manifest.json
python -m pytest -q
waitress-serve --listen=127.0.0.1:8000 app:app
```

A minimal `systemd` unit may look like this; replace the generic paths and user
with values owned by the deployment account:

```ini
[Unit]
Description=TCN-BiLSTM magnetic inference
After=network.target

[Service]
Type=simple
User=magnetic-service
Group=magnetic-service
WorkingDirectory=/opt/tcn-bilstm-magnetic
Environment=MAGNET_DEVICE=cpu
Environment=MAGNET_MODEL_DIR=/opt/tcn-bilstm-magnetic/weights
Environment=MAGNET_COLD_START_MODEL_DIR=/opt/tcn-bilstm-magnetic/weights
ExecStart=/opt/tcn-bilstm-magnetic/.venv/bin/waitress-serve --listen=127.0.0.1:8000 app:app
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Run it behind a reverse proxy that terminates TLS and applies the site's access
policy, request-rate limits, and timeouts.

## 5. Waitress process configuration

The Flask development server is acceptable only for local inspection. The
recommended application target is `app:app`:

```bash
waitress-serve --listen=127.0.0.1:8000 app:app
```

Do not enable Flask debug mode in production. If multiple Waitress processes
are launched, each process maintains its own lazy model cache and therefore
duplicates model memory. Start with one process, measure inference latency and
memory, and scale only after accounting for that duplication.

## 6. Docker

The supplied Dockerfile installs a CPU PyTorch build by default, creates an
unprivileged user, exposes port 8000, runs Waitress, and includes a `/health`
health check.

```bash
docker build \
  --build-arg TORCH_VERSION=2.8.0 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  -t tcn-bilstm-magnetic:1.3.1 .

docker run --rm \
  --name tcn-bilstm-magnetic \
  -p 127.0.0.1:8000:8000 \
  --mount type=bind,src="$(pwd)/weights",dst=/app/weights,readonly \
  -e MAGNET_DEVICE=cpu \
  -e MAGNET_MODEL_DIR=/app/weights \
  -e MAGNET_COLD_START_MODEL_DIR=/app/weights \
  tcn-bilstm-magnetic:1.3.1
```

On PowerShell, resolve the mount source before invoking Docker:

```powershell
$weights = (Resolve-Path .\weights).Path
docker run --rm --name tcn-bilstm-magnetic `
  -p 127.0.0.1:8000:8000 `
  --mount "type=bind,src=$weights,dst=/app/weights,readonly" `
  -e MAGNET_DEVICE=cpu `
  -e MAGNET_MODEL_DIR=/app/weights `
  -e MAGNET_COLD_START_MODEL_DIR=/app/weights `
  tcn-bilstm-magnetic:1.3.1
```

For CUDA, use a CUDA-compatible base/runtime and PyTorch wheel, pass the GPU to
the container, and set `MAGNET_DEVICE=cuda`. Record the driver, CUDA, PyTorch,
and hardware versions with published results.

## 7. Configuration

| Variable | Purpose | Default |
|---|---|---|
| `MAGNET_DEVICE` | `auto`, `cpu`, `cuda`, or a concrete CUDA device | `auto` |
| `MAGNET_MODEL_DIR` | Directory containing scientific weights and `manifest.json` | project `weights/` |
| `MAGNET_COLD_START_MODEL_DIR` | Directory containing cold-start weights and `cold_start_manifest.json` | scientific model directory |
| `MAGNET_CHECKPOINT_A` ... `MAGNET_CHECKPOINT_E` | Controlled per-file scientific override | unset |
| `MAGNET_COLD_START_CHECKPOINT_A`, `B`, `D`, `E` | Controlled per-file cold-start override | unset |
| `HOST`, `PORT` | Used only by `python app.py` | `127.0.0.1`, `8000` |

Checkpoint overrides still use the trusted manifest metadata. Record every
override with any reported result. Prefer the directory variables for normal
deployments.

## 8. Health and smoke checks

After startup:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/api/models
curl --fail http://127.0.0.1:8000/api/cold-start-models
```

A complete bundle should report `status: "ok"`, scientific
`available_models` equal to `A,B,C,D,E`, and cold-start `available_models`
equal to `A,B,D,E`. An empty `loaded_models` list is normal before the first
prediction: availability verifies the manifest and file hash, whereas network
construction and checkpoint loading are lazy.

Generate and submit a private-data-free 1,000-point request:

```bash
python scripts/generate_sample_inputs.py --output-dir examples/generated
curl --fail -X POST http://127.0.0.1:8000/api/predict-window \
  -H "Content-Type: application/json" \
  --data-binary @examples/generated/predict_window_A.json
```

See [API.md](API.md) for all service endpoints and response semantics.

## 9. Browser assets and offline operation

The API, tests, model loading, and health check operate without a browser CDN.
The upstream-layout `/design`, `/test`, `/fine_tune`, and `/document` pages load
some presentation libraries from `cdn.tailwindcss.com` and `cdn.jsdelivr.net`.
If those hosts are unavailable, JSON inference remains functional but styling,
charts, zoom, or pan controls on those pages may be incomplete.

For an air-gapped deployment, download reviewed copies of the exact referenced
assets, serve them from `static/vendor/`, update the templates, preserve their
licenses, and re-run the UI and regression tests. Do not claim that the shipped
reference-layout UI is fully offline until that vendoring step is complete.

## 10. Production security checklist

- Run Waitress as a dedicated unprivileged account; never use Flask debug mode.
- Bind the application to loopback or a private interface and terminate HTTPS
  at a maintained reverse proxy.
- Add authentication and authorization if predictions or downloaded artifacts
  are not intended to be public.
- Preserve the 16 MiB request-body limit and add proxy-level rate, connection,
  and request-time limits. Inference is compute-intensive even when a request is
  syntactically valid.
- Mount weights read-only. Protect both manifests and checkpoint files from
  runtime modification.
- Never place credentials, SSH keys, tokens, private datasets, or environment
  files in the repository or image.
- Restrict outbound network access if CDN-backed browser pages are not needed.
- Retain application and reverse-proxy logs without logging full B/H payloads
  when they may contain confidential measurements.
- Monitor `/health`, disk space, process memory, and model-load failures. A
  successful health response does not validate scientific accuracy for a new
  operating domain.
- Back up the exact release archive and its external SHA-256 checksum before
  upgrading. Roll back code, manifests, and weights as one versioned unit.

## 11. Common deployment failures

| Symptom | Likely cause | Action |
|---|---|---|
| Application fails during import | One of the two manifest files is missing or malformed | Restore the release manifests beside their model family |
| Model appears unavailable | Missing file, wrong byte count, wrong SHA-256, pending status, or wrong material set | Re-run both verification commands; do not rebuild trust values |
| `CUDA was requested but is not available` | CPU PyTorch build or unavailable GPU | Set `MAGNET_DEVICE=cpu` or install a compatible CUDA build |
| First prediction is slow | Lazy checkpoint load and model construction | Warm only the required materials after startup |
| API works but charts do not render | Browser cannot reach the public CDNs | Allow the pinned CDN assets or vendor them locally |
| HTTP 413 | JSON body exceeds 16 MiB | Reduce the interval or use a controlled offline batch workflow |
