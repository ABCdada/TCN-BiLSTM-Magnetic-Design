# Parameter/error Pareto comparison

The two plots compare the arithmetic mean of trainable-parameter counts with
the five-material average P95 sequence error and P95 energy error reported for
the representative methods discussed in the manuscript.

- `pareto_data.csv`: compact machine-readable values used by both plots.
- `plot_pareto_comparison.m`: MATLAB source and documented aggregation inputs.
- `pareto_sequence_p95.svg` and `pareto_energy_p95.svg`: editable vector files.
- the corresponding PNG files: fixed previews for browsers and documents.

Lower values are preferable on both axes. MagLearn 2 has no energy-P95 point
because that metric was not reported in the source used for this comparison;
the missing value is not imputed. These files reproduce the manuscript's
reported comparison and do not imply that all methods were retrained under
identical hardware or implementation conditions.

