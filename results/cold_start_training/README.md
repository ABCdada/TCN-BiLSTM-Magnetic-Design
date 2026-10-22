# Native cold-start training results for A, B, D, and E

Materials A, B, D, and E were trained as independent single-sequence
checkpoints with one raw-zero H anchor followed by 999 predicted samples.
Acquisition rows were split into disjoint training, validation, and held-out
test sets before overlapping windows were generated. The browser release
registers exactly these four cold-start checkpoints; Material C/FEC007 remains
only in the separate scientific registry.

The held-out metrics below come from the acquisition-row test split. The energy
column is a trajectory-proxy metric used by the trainer; it is **not** the
MagNet Challenge measured-Loss metric and must not be compared directly with
the manuscript's official energy-error results.

| Material | Grade | Parameters | Completed / configured epochs | Best epoch | Sequence mean | Sequence P95 | Energy-proxy mean | Energy-proxy P95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | 3C92 | 497,793 | 70 / 150 (early stop) | 33 | 18.7730% | 62.8348% | 19.1745% | 60.9080% |
| B | 3C95 | 2,092,545 | 60 / 150 (requested cap) | 60 | 2.7722% | 5.9773% | 3.9369% | 12.2167% |
| D | FEC014 | 2,512,593 | 60 / 150 (requested cap) | 59 | 5.8994% | 14.7910% | 8.9913% | 27.3134% |
| E | T37 | 2,092,865 | 60 / 180 (requested cap) | 59 | 6.5155% | 15.5314% | 6.8457% | 25.0981% |

The exact machine-readable metrics, byte counts, and SHA-256 values are stored
in `metrics.json`. Material A has only 14 held-out acquisition rows and a weak
tail result; its deployed checkpoint is functional but should not be described
as high-accuracy without further data or retraining. B, D, and E stopped at the
user-requested epoch cap while preserving their original optimizer and cosine
schedule horizons.
