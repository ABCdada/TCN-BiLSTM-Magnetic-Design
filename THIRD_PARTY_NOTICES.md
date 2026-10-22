# Third-party notices

The TCN-BiLSTM inference backend, model definitions, verification code, and
project-specific documentation are original portions of this repository.

## Adapted web user interface

Portions of this web user interface, including its page structure, styling,
client-side behavior, and selected media, are adapted from
**LSTM-DAO-Inference**, <https://github.com/ouihon/LSTM-DAO-Inference>, source
revision `59fdc474c3e9a8b2b8e2be88bf936169330a6f13`.

Copyright (c) 2024 Magnetic Design Team. Licensed under the MIT License; the
full license text is provided at
`licenses/LSTM-DAO-Inference-LICENSE.txt`. The TCN-BiLSTM prediction backend,
model weights, material registry, and result files are supplied by this
project and are not part of, or endorsed by, the upstream project.

Adapted upstream files comprise `templates/index.html`, `design.html`,
`test.html`, `document.html`, and `fine_tune.html`, together with selected
assets under `static/images/`. The files under `static/resources/` are
project-generated counterparts based on this project's synthetic examples.
The upstream LSTM-DAO model code, weights, and sample datasets are not included.

## Runtime libraries

The project uses third-party Python packages, including PyTorch, NumPy, Flask,
Waitress, Requests, and pytest. Each package remains subject to its own license.
Installed package versions can be recorded with:

```bash
python -m pip freeze > environment.lock.txt
```

## Browser libraries

The adapted pages load Tailwind CSS, Chart.js 4.4.0, and
chartjs-plugin-zoom 2.0.1 from their public CDNs. The legacy project pages also
retain a vendored copy of Chart.js 4.4.7 under `static/vendor/`; these projects
are distributed under their respective licenses. The retained Chart.js
license is at `static/vendor/Chart.js.LICENSE.md`.

## Model weights and datasets

The MIT license in this repository applies to the original source code and
documentation only. Trained checkpoints and datasets are not automatically
covered by that license. Users must verify that they have the right to use and
redistribute each checkpoint and dataset. Large checkpoints are intentionally
excluded from Git; only a checksum manifest should be committed.
