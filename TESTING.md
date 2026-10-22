# Verification and testing

Run these checks from the project root in a clean virtual environment. They do
not read private training data.

## 1. Install dependencies

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements-validated.txt` records the direct versions used by the validated
AutoDL deployment. Select the correct PyTorch CPU or CUDA wheel for the target
machine before treating it as a lock file.

## 2. Verify model artifacts

The full deployment archive must pass both checks:

```bash
python scripts/verify_weight_manifest.py --manifest weights/manifest.json
python scripts/verify_cold_start_manifest.py --manifest weights/cold_start_manifest.json
python scripts/verify_all_models.py --manifest weights/manifest.json --device cpu
```

The GitHub source archive intentionally has no `.pth` files; model verification
is expected to fail until authorized checkpoints are restored.

## 3. Run the automated suite

```bash
python -m pytest -q
```

The tests cover registry isolation, manifest integrity, 1,000-point window
semantics, long-sequence stitching, browser compatibility, physical
post-processing, circuit/converter interfaces, resources, and API validation.

## 4. Generate real-checkpoint smoke evidence

```bash
python scripts/generate_cold_start_smoke_results.py
```

This sends deterministic synthetic B sequences through the same Flask API used
by the website. It writes four 1,000-point outputs, one 2,200-point recursive
output, and a machine-readable summary to `results/cold_start_smoke/`.

## 5. HTTP smoke test

Start the service and check the public registries:

```bash
python -m waitress --listen=127.0.0.1:8000 app:app
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/models
curl http://127.0.0.1:8000/api/cold-start-models
```

Expected scientific IDs are A, B, C, D, and E. Expected browser cold-start IDs
are A, B, D, and E, displayed as 3C92, 3C95, FEC014, and T37.

## Interpretation boundary

Passing these checks demonstrates software, schema, checkpoint-integrity, and
deterministic inference behavior. It is not an independent accuracy benchmark
and does not replace the held-out and measured-PFC evidence under `results/`.
