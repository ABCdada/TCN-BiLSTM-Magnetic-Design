# Checkpoints

The runtime expects these files in this directory:

- `A_best_model.pth` - 3C92
- `B_best_model.pth` - 3C95
- `C_best_model.pth` - FEC007
- `D_best_model.pth` - FEC014
- `E_best_model.pth` - T37 final-paper six-channel model (`ddB` included)

Checkpoints are excluded from Git. `manifest.json` records their expected byte
counts, model parameter counts, and SHA-256 hashes. Verify after every transfer:

```bash
python scripts/verify_weight_manifest.py --manifest weights/manifest.json
```

The independent `cold_start_manifest.json` describes the one-known-H / 999-
predicted model family used by `/api/predict-cold-start`. The browser release
registers exactly four grades: 3C92, 3C95, FEC014, and T37 (internal IDs A, B,
D, and E). Their files are named `<ID>_cold_start_best_model.pth`. Browser
`materialId` values 1-4 follow that grade order; the pages display only the
grade names.

C/FEC007 is intentionally different: `C_best_model.pth` and its scientific
manifest record remain preserved for scientific A-E inference, but C is not a
member of `cold_start_manifest.json` and is not exposed by Design Center, Test
Center, Model Libraries, or `/api/predict-cold-start`. If an older deployment
also retains `C_cold_start_best_model.pth` for provenance, leaving the file on
disk does not register it and it must not be substituted for any of the four
browser models.

A source checkout that does not redistribute these optional four cold-start
weights keeps their records as `pending`; the deployed release marks them
`ready` only after installing and validating the real files. Register real
files (and compute, never invent, their byte counts and SHA-256 hashes) with:

```bash
python scripts/update_cold_start_manifest.py --weights-dir weights
```

The scientific files and `manifest.json` must not be overwritten by cold-start
weights.

The code license does not automatically grant permission to redistribute model
weights or their training data.
