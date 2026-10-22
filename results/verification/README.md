# Verification reports

The JSON reports record the Python, PyTorch, CUDA, platform, checkpoint hash,
parameter count, output shape, output summary, and deterministic output-byte
hash for each A-E model.  CPU and CUDA hashes may differ slightly because their
floating-point kernels differ.

The retained AutoDL text log records the original 37-test GPU validation run,
manifest verification, and live API smoke tests. The current source tree has
56 tests; run `pytest -q` to obtain a report for the exact revision being used.
`pytest_local.txt` records the current 56-test local run. All model files are
checked against `weights/manifest.json` before loading.

These are software-verification artifacts, not accuracy measurements.

