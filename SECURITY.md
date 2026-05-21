# Security Policy

`gpu-holder` is designed for shared GPU machines. Its safety boundary is intentionally narrow:

- It may start and stop only the guard and worker processes it created.
- It must treat external CUDA processes as read-only scheduling signals.
- It must not kill, suspend, renice, ptrace, or otherwise control external jobs.
- It should prefer releasing its own holder over risking interference with a training job.

## Reporting

Please report safety or security issues privately through GitHub's private vulnerability reporting for this repository:

https://github.com/BITnene465/gpu-holder/security/advisories/new

If private vulnerability reporting is unavailable, open a minimal issue without sensitive logs and ask for a private contact path.

## Useful Context

When reporting behavior issues, include:

- `gpu-holder doctor --json`
- `gpu-holder status --json`
- the exact command used to start the guard
- relevant `~/.gpu-holder/gpu-holder.log` lines
- GPU model, driver version, and whether the run happened inside a container

Do not include secrets, access tokens, private dataset paths, or proprietary training logs.
