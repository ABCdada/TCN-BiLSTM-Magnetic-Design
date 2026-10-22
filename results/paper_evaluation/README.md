# Paper-evaluation artifacts

`p95_metrics.csv` is a machine-readable transcription of the formal test results
reported in the revised manuscript.  For A, B, D, and E, each material-level
entry averages the P95 result across the 900-100, 500-500, and 100-900
history/prediction scenarios.  Material C is evaluated under its dedicated
cold-start setting because its historical `H` channel contains only the zero
placeholder used by the released model.

`training_reports/` contains the compact JSON reports produced during training.
Their `internal_seq` and `internal_ene` fields are internal validation
statistics and must not be substituted for the formal independent-test P95
values in `p95_metrics.csv`.

`error_distributions/` contains the retained evaluation figures associated with
the released checkpoints.  E uses the final six-input-channel `model2` report
and figures.

`pareto_comparison/` contains the machine-readable comparison data, MATLAB
source, PNG previews, and editable SVG exports for the parameter-count versus
average P95 sequence/energy comparisons. Missing values remain explicitly
marked as unavailable rather than being estimated.

The underlying full prediction tables and raw evaluation records are not
included here because they are large and may be governed by separate dataset
terms.  Therefore these reported metrics are evidence from the manuscript, not
a claim that a fresh clone can independently recompute every paper table.
