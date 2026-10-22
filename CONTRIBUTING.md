# Contributing

Thank you for helping improve the TCN-BiLSTM magnetic-circuit inference project.
The public repository is centered on model definitions, inference, evaluation,
and circuit-level examples. It does not currently contain the complete paper
training and few-shot transfer-learning pipeline.

## Development setup

Use Python 3.10 or newer in a dedicated environment:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The five released checkpoint filenames are listed in `weights/README.md` and
authenticated by `weights/manifest.json`. Checkpoints are not required for most
unit tests. Do not add or redistribute a checkpoint unless its license and data
provenance permit it.

## Before opening a pull request

Run syntax checks and all tests that do not require the released checkpoint
files:

```bash
python -m compileall -q app.py cli.py magnetic_model scripts
python -m pytest -q --ignore=tests/test_checkpoint_integrity.py
python -m build --outdir ci_dist
```

When authorized checkpoints are available locally, verify them and run the full
suite as an additional check:

```bash
python scripts/verify_weight_manifest.py --manifest weights/manifest.json
python -m pytest -q
```

`test_checkpoint_integrity.py` intentionally depends on a real checkpoint. The
remaining tests use pure numerical logic, fake predictors, or a freshly created
dummy checkpoint and therefore run in the no-weight CI environment.

## Change expectations

- Keep material identifiers, grades, input features, modes, parameter counts,
  and checkpoint hashes synchronized where applicable.
- Preserve the 1,000-sample input contract unless a versioned interface change
  is proposed and documented.
- Label result inputs as measured, simulated, or synthetic. Synthetic software
  demonstrations must not be presented as experimental accuracy evidence.
- Split independent acquisition sequences into training, validation, and test
  sets before extracting overlapping windows.
- Add focused tests for behavior changes and document externally visible API or
  CLI changes.
- Do not commit credentials, private server addresses, private paths, participant
  data, or source datasets without explicit redistribution permission.

## Training-related contributions

Read `TRAINING.md` before proposing training or transfer-learning changes. A
training contribution should include executable code, environment information,
fixed seeds, data-manifest and split-generation procedures, metric definitions,
and a clear record of checkpoint selection. A report copied from a completed run
is useful provenance, but it is not by itself an independently reproducible
training pipeline.

## Pull requests and review

Keep changes focused and explain how they were validated. Complete the pull
request checklist, including data provenance and licensing items. Large model
files should normally be distributed as approved GitHub Release assets or through
Git LFS rather than committed directly to ordinary Git history.

Security issues should follow `SECURITY.md` instead of a public issue.

