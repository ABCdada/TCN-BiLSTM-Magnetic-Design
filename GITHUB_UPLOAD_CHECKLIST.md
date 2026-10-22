# GitHub publication checklist

Two archives are produced for different purposes:

1. `*-github-source.zip` is the recommended repository import. It contains
   code, templates, documentation, tests, examples, results, and manifests, but
   no `.pth` checkpoint binaries.
2. `*-full-deployment.zip` is an owner-authorized, ready-to-run deployment
   bundle containing the verified checkpoints.

Optional scientific-weights and cold-start-weights ZIP files are prepared as
separate release assets. Publish them only after reviewing
`MODEL_WEIGHTS_NOTICE.md` and confirming redistribution rights.

Before the first push:

- create a new empty GitHub repository without an auto-generated README;
- extract the GitHub source archive and review `README.md`, `LICENSE`,
  `THIRD_PARTY_NOTICES.md`, and `CITATION.cff`;
- replace placeholder maintainer/repository fields if applicable;
- run a secret scan and confirm that no `.env`, SSH key, password, private
  waveform, or local absolute path is present;
- run `python -m pytest -q` (tests that require absent checkpoints should be
  skipped or report model unavailability as documented);
- verify the outer archive hashes in `RELEASE_SHA256SUMS.txt`;
- push the source tree, create tag `v1.3.2`, and attach authorized binary assets
  to the GitHub Release rather than ordinary Git history.

If checkpoints must live in the repository, use Git LFS and explicitly track
`*.pth` before adding them. Git LFS changes storage mechanics, not licensing.

Suggested first publication commands:

```bash
git init
git add .
git commit -m "Release v1.3.2"
git branch -M main
git remote add origin https://github.com/ABCdada/TCN-BiLSTM-Magnetic-Design.git
git push -u origin main
git tag -a v1.3.2 -m "v1.3.2"
git push origin v1.3.2
```

Do not copy the reference project's checkpoints, author identity, fixed hosting
address, or private data. Retained MIT-derived code/assets are documented in
`THIRD_PARTY_NOTICES.md` and `licenses/LSTM-DAO-Inference-LICENSE.txt`.
