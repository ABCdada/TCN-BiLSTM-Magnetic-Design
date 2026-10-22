# Security Policy

## Supported versions

Security fixes are intended for the current release line.

| Version | Supported |
| --- | --- |
| 1.3.x | Yes |
| Earlier versions | No |

## Reporting a vulnerability

Do not disclose an exploitable vulnerability in a public issue. When this project
is hosted on GitHub, use the repository's **Security** tab and **Report a
vulnerability** option if private vulnerability reporting is enabled. Otherwise,
contact the repository maintainers through a private channel listed by the
repository owner before sharing technical details.

Include the affected version or commit, impact, minimal reproduction, and any
suggested mitigation. Remove credentials, private waveforms, restricted datasets,
and checkpoint files from the report. No fixed response-time service level is
promised, but reports will be reviewed as maintainers are available.

## Security scope

Reports may cover the Web/API service, unsafe parsing or file handling, dependency
risks, checkpoint integrity verification, denial-of-service paths, and unintended
information disclosure. Scientific disagreement about a metric or model result is
not a security issue and should use a normal issue instead.

Checkpoint files are deserialized only after their size and SHA-256 identity are
checked by the runtime manifest workflow. Obtain checkpoints only from a trusted,
authorized source and verify `weights/manifest.json` after transfer.
