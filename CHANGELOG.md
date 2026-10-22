# Changelog

All notable public-release changes are documented in this file. The format is
based on Keep a Changelog, and versions follow Semantic Versioning where possible.

## [Unreleased]

## [1.3.1] - 2026-10-22

### Added

- Total magnetic power in watts after Design Center interval selection, while
  retaining the duration-normalized rate density in W/m³ and distinguishing
  closed-path core loss from open-interval reversible energy exchange.
- An isolated four-material cold-start model registry, integrity manifest, and
  `/api/predict-cold-start` endpoint. Dedicated checkpoints for 3C92, 3C95,
  FEC014, and T37 use the native 1--999 protocol and are registered only after
  hash verification; scientific checkpoints and APIs remain unchanged.
- An atomic cold-start manifest update utility that validates checkpoint shape
  and normalization metadata before recording actual file size and SHA-256.
- Separate GitHub-source and full-deployment release layouts, production start
  scripts, API/deployment/testing guides, and release-asset checksums.
- GitHub CI for syntax checks, no-checkpoint tests, distribution builds, and
  installed-wheel import/API validation.
- Issue and pull-request templates, contribution guidance, security policy, and
  an explicit description of the current training-release boundary.

### Changed

- The Design Center, Test Center, and Model Libraries expose only the four
  public grade names 3C92, 3C95, FEC014, and T37. Each uses its own
  one-zero-anchor/999-predicted checkpoint;
  scientific APIs continue to expose the original A--E models with explicit
  historical-H inputs.
- Design selections now identify their elapsed time as an inclusive endpoint
  span, preventing the 16 MHz compatibility resampler from dropping one point.
- The Design Center no longer clips selections to 50,000 model samples. Long
  browser intervals use non-overlapping 1,000-sample windows, carry the
  preceding final H state, and right-pad only the final partial window. The
  scientific sequence endpoint retains its original bounded,
  training-consistent semantics.
- Zooming or panning the input charts now preserves the selected interval and
  completed long-sequence result, preventing an ordinary page scroll from
  clearing an expensive recursive prediction.
- The public Design Center now exposes only the single-phase full-bridge
  workflow. Toroidal dimensions `R`, `r`, and `d` are independently editable
  while retaining the physical `r < R` constraint; converter APIs continue to
  support their documented three-phase demonstrations.
- The magnetic-core diagram no longer implies fixed `r = 0.4R` or `d = 0.3R`
  ratios, and the inherited team contact footer has been removed.

## [1.3.0] - 2026-10-20

### Added

- MIT-licensed LSTM-DAO-Inference website layout adapted for the TCN-BiLSTM
  material models, with preserved Design Center, Model Libraries, Test Center,
  and Documents navigation.
- `/api/predict` compatibility endpoint for the upstream browser contract,
  including A-E material mapping, 16 MHz resampling, short-window padding, and
  long-sequence reconstruction.
- Upstream MIT license, attribution, project-generated browser examples, and
  compatibility regression tests.

### Changed

- Browser material labels now map to 3C92, 3C95, FEC007, FEC014, and T37.
- The architecture image and documentation now describe the released
  TCN-BiLSTM models rather than the upstream LSTM-DAO backend.
- The adapted Design and Test pages now treat model output directly as H in
  A/m and label open-path H-dB integrals as interval magnetic-energy exchange
  rates instead of irreversible core loss.
- The preserved Model Libraries layout is now an explicitly disabled,
  read-only reference for the separate offline training workflow.

## [1.1.0] - 2026-10-20

### Added

- PyTorch definitions and inference registry for Materials A-E: 3C92, 3C95,
  FEC007, FEC014, and T37.
- Validated 1,000-sample window prediction modes, including the dedicated
  one-known/999-predicted cold-start contract for Material C.
- Python service, REST API, command-line interface, and Web interface.
- Circuit-level and converter-topology demonstrations with explicit separation
  between simulated demonstration signals and measured validation data.
- Checkpoint manifest containing expected filenames, parameter counts, byte
  counts, and SHA-256 identities.
- Unit and integration tests for model construction, sequence semantics,
  physics calculations, API validation, and circuit/converter interfaces.
- Model, data, and reproducibility documentation plus retained evaluation and
  verification artifacts.

### Limitations

- Ordinary Git history excludes trained checkpoint binaries and source datasets.
- The complete paper training and few-shot transfer-learning implementation is
  not part of this release; retained training reports are provenance artifacts,
  not an executable end-to-end training workflow.

Release comparison links can be added after the final GitHub repository URL is
assigned.
