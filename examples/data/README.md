# Browser-upload example data

`synthetic_window_A_1000.csv` contains exactly 1,000 rows with `B` and `H`
columns and can be uploaded directly on the Prediction page. It is generated
by `scripts/generate_upload_example.py`.

Both columns are deterministic synthetic fixtures. The `H` column is only a
convenient known-history input and optional visual reference; it is not an
experimental target and must not be used to claim model accuracy.

`synthetic_finetune_format.zip` is the format-only counterpart to a small
fine-tuning dataset. It contains `B.csv`, `H.csv`, `T.csv`, and an explicit
`acquisition_id.csv` whose identifiers must be split before overlapping-window
generation. This inference-focused release does not expose a Web fine-tuning
endpoint, so the ZIP documents the data contract rather than claiming that the
paper's transfer experiment can be reproduced from synthetic data.
