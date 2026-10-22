# Data card

## Scope

This inference repository does not redistribute the training, validation, or
test measurements used to train the A-E checkpoints. It also does not package
the independent experimental waveforms described in the accompanying
manuscript. Dataset access, attribution, licensing, and any competition rules
must be checked at their original source.

The accompanying study identifies its main measured transient magnetic data as
coming from the MagNetX/MagNet Challenge data source. This repository supplies
the software contract needed to use compatible measurements, but it does not
independently certify or re-host that dataset.

## Supported material identifiers

| Repository ID | Material grade | Historical-`H` convention |
|---|---|---|
| A | 3C92 | measured known prefix |
| B | 3C95 | measured known prefix |
| C | FEC007 | one raw zero placeholder; no effective history |
| D | FEC014 | measured known prefix |
| E | T37 | measured known prefix |

The IDs are part of the checkpoint and normalization contract and must not be
reassigned to other grades.

## Record structure

A direct inference record contains:

- exactly 1,000 values of magnetic flux density `B` in tesla;
- temperature `T` as one value or a 1,000-value sequence in degrees Celsius;
- an available `H` prefix in ampere per metre, except for Material C's defined
  zero-placeholder convention;
- one of the `900-100`, `500-500`, or `100-900` history/prediction modes;
- a sampling-rate declaration of 16,000,000 samples per second.

The backend calculates `dB`, and for Material E also `ddB`, using discrete
increments on that grid. It constructs the mask from the requested prediction
boundary. Users should not pre-append these channels.

The circuit-window interface can instead receive a magnetic-core induced
voltage, winding turns, effective area, effective path length, and the known
current or `H` prefix. Raw terminal voltage must first be corrected for winding
resistance, leakage inductance, and relevant parasitic voltage drops.

## Sampling and preprocessing contract

The reference rate is fixed at 16 MHz, so a 1,000-point window spans 62.5
microseconds. Discrete increments are meaningful only with that known interval.
The current checkpoints should not be assumed invariant to arbitrary
resampling. A different acquisition interval requires an explicitly defined
rate feature and suitable retraining or fine-tuning.

Normalization parameters are stored with the authorized checkpoints. Exact
reproduction requires the original normalization state; estimating replacement
statistics from evaluation data would change the evaluation protocol.

## Split integrity

Acquisition-level training, validation, and test sequences must be separated
**before** overlapping windows are extracted. Windows derived from the same
acquisition must not cross those partitions. Record acquisition identifiers,
random seed, split membership, window length, and stride so leakage can be
audited.

For independent evaluation, keep complete acquisition sequences and measured
targets unavailable to model selection and fine-tuning. A reported metric
should identify the material, acquisition split, prediction mode, and whether
aggregation is per sequence, per material, or across materials.

## Synthetic examples

Files created by `scripts/generate_sample_inputs.py` are deterministic,
non-proprietary API examples. They are intended only to verify schemas,
software wiring, and finite model execution. They are not sampled from the
experimental dataset and cannot support a claim about predictive accuracy,
generalization, energy loss, or circuit performance.

## Energy labels

Store enough metadata to determine whether a record is an open transient
interval, one complete closed magnetic cycle, or multiple cycles. The integral
of `H dB` over an open interval is net transient magnetic-energy exchange, not
necessarily irreversible loss. A one-cycle loss label requires a closed path;
conversion to power density additionally requires its valid cycle rate.

## Sensitive information and governance

Magnetic waveforms normally contain no personal data. They can nevertheless be
proprietary engineering data and may reveal material or converter operating
details. Do not commit restricted raw data, access credentials, or checkpoints
without redistribution permission. Publish hashes and provenance metadata when
the underlying artifact cannot be redistributed.

## Known gaps

This repository alone does not provide the complete acquisition protocol,
original split lists, or ground-truth evaluation records. It includes an
aggregate benchmark-table transcription, but those missing records are still
needed, together with authorized weights, to independently recompute the
manuscript's numerical accuracy results.
