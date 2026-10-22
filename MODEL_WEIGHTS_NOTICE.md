# Model checkpoint notice

The MIT license in `LICENSE` applies to the original software in this
repository unless a file says otherwise. It does **not** automatically grant a
license to training data or trained checkpoint files.

The full deployment archive may contain nine owner-supplied checkpoints:

- five scientific checkpoints (`A_best_model.pth` through
  `E_best_model.pth`); and
- four browser cold-start checkpoints for 3C92, 3C95, FEC014, and T37.

Their exact filenames, byte counts, parameter counts, and SHA-256 identities
are recorded in `weights/manifest.json` and
`weights/cold_start_manifest.json`. The GitHub source archive intentionally
omits the checkpoint binaries. Put authorized copies in `weights/` before
starting the service and verify both manifests.

No public redistribution license for the checkpoints or their source datasets
is asserted by this notice. Before publishing checkpoint binaries in a GitHub
repository or Release, the publisher must confirm that they own the artifacts
or have permission to distribute them. Removing a checkpoint from Git history
does not revoke copies already downloaded; use GitHub Release assets or Git LFS
only after that rights review.
