# Third-party notices

The TCN-BiLSTM inference backend, model definitions, and verification code are
original portions of this repository.

## Adapted web user interface

Portions of this web user interface, including its page structure, styling,
client-side behavior, and selected media, are adapted from
**LSTM-DAO-Inference**, <https://github.com/ouihon/LSTM-DAO-Inference>, source
revision `59fdc474c3e9a8b2b8e2be88bf936169330a6f13`.

Copyright (c) 2024 Magnetic Design Team. Licensed under the MIT License; the
full license text is provided at
`licenses/LSTM-DAO-Inference-LICENSE.txt`. The TCN-BiLSTM prediction backend,
model weights and material registry are supplied by this
project and are not part of, or endorsed by, the upstream project.

Adapted upstream files comprise `templates/index.html`, `design.html`,
`test.html`, `document.html`, and `fine_tune.html`, together with selected
assets under `static/images/`. The upstream LSTM-DAO model code, weights, and
sample datasets are not included.

## Runtime libraries

The project uses PyTorch, NumPy, Flask, and Waitress. Each package remains
subject to its own license.

## Browser libraries

The adapted pages load Tailwind CSS, Chart.js 4.4.0, and
chartjs-plugin-zoom 2.0.1 from their public CDNs. These projects remain subject
to their respective licenses.

## Model weights and datasets

The MIT license in this repository applies to the original source code and
documentation only. Trained checkpoints and datasets are not automatically
covered by that license. Users must verify that they have the right to use and
redistribute each checkpoint and dataset. Large checkpoints are intentionally
excluded from Git; only a checksum manifest should be committed.
