# Independent PFC validation

This validation is separate from the MagNetX evaluation records. A 3C90
toroidal inductor with a nominal inductance of 145 microhenries was measured in
a PFC circuit operating at a 100 kHz switching frequency, 20 V RMS input, and
50 V RMS output. Voltage and current were sampled at 16 MHz. The measured
signals were converted to synchronized `B` and `H` trajectories using the core
geometry and winding relationships documented in the manuscript.

The PFC excitation and the MagNetX T-type-inverter excitation have a material
domain difference. A 20.58 MB subset of independently measured PFC records was
therefore used to adapt the pretrained 3C90 model; the metrics below were then
evaluated on the held-out validation records.

| Known / predicted samples | Sequence relative error | Energy relative error |
|---:|---:|---:|
| 100 / 900 | 6.20% | 6.46% |
| 500 / 500 | 4.91% | 1.92% |
| 900 / 100 | 4.32% | 1.51% |

The same values, together with the acquisition and component metadata, are in
[`pfc_validation_metrics.csv`](pfc_validation_metrics.csv). The manuscript
figures are retained as
[`pfc_window_extraction.png`](../docs/screenshots/pfc_window_extraction.png)
and
[`pfc_prediction_examples.png`](../docs/screenshots/pfc_prediction_examples.png).

These reported values are experimental evidence, unlike the deterministic
fixtures under `results/reproducible_examples/`. Raw oscilloscope records are
not redistributed here and may be subject to separate data terms.

