## Summary

Describe the change and the problem it solves.

## Scope

- [ ] Inference/model definition
- [ ] API, CLI, or Web interface
- [ ] Circuit/converter integration
- [ ] Tests or packaging
- [ ] Documentation/results metadata

## Validation

List the exact commands executed and their results. At minimum, run the relevant
non-checkpoint tests. If checkpoint-dependent behavior changed, also report the
manifest verification and checkpoint-dependent test results from an authorized
local environment.

```text
python -m compileall -q app.py cli.py magnetic_model scripts
python -m pytest -q --ignore=tests/test_checkpoint_integrity.py
```

## Reproducibility and data handling

- [ ] I did not commit credentials, private host details, or proprietary data.
- [ ] New data and assets include their provenance and redistribution terms.
- [ ] New numerical results identify whether inputs are measured, simulated, or synthetic.
- [ ] Training, validation, and test acquisition sequences remain independent before window extraction.
- [ ] Any checkpoint change is reflected in `weights/manifest.json` and has redistribution approval.
- [ ] Documentation and tests were updated where behavior changed.

## Reviewer notes

Call out compatibility changes, assumptions, limitations, or items that cannot be
verified in the public no-checkpoint CI environment.

