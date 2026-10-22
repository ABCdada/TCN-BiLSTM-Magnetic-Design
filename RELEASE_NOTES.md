# Release v1.3.1

This release packages the TCN-BiLSTM scientific inference API together with the
separate four-grade browser cold-start family used by Design Center and Test
Center.

Compared with v1.3.0, this release completes the browser cold-start rollout,
matches the reference Design Center's raw display-sample and H-dB integration
workflow, and narrows the public design page to the single-phase full bridge.
Toroidal `R`, `r`, and `d` dimensions are now independently editable, and the
diagram and documentation no longer imply fixed geometry ratios.

## Included

- Flask/Waitress service, CLI, Design Center, Test Center, Model Libraries,
  Documents, Resources, circuit demonstration, and converter demonstration;
- scientific model registry A--E (3C92, 3C95, FEC007, FEC014, and T37);
- browser cold-start registry for 3C92, 3C95, FEC014, and T37;
- checksum manifests, model/data cards, API and deployment documentation;
- paper evaluation tables, independent PFC validation evidence, training
  provenance summaries, synthetic examples, and real-checkpoint smoke outputs;
- automated tests and GitHub Actions workflow.
- configurable toroidal geometry in the single-phase Design Center, with the
  independent converter API retaining its documented three-phase examples.

## Deliberate boundaries

- browser-side training endpoints return HTTP 501; Model Libraries documents
  and configures the separate offline training workflow;
- the page UI loads Tailwind CSS and Chart.js from public CDNs, so the browser
  presentation needs network access unless those assets are vendored locally;
- open-path H-dB integration is magnetic-energy exchange, not automatically
  irreversible core loss;
- the source archive omits checkpoints, while the full deployment archive
  contains owner-supplied verified checkpoints.

See `DEPLOYMENT.md`, `API.md`, `TESTING.md`,
`GITHUB_UPLOAD_CHECKLIST.md`, and `MODEL_WEIGHTS_NOTICE.md` before publishing or
deploying.
