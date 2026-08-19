# Laptop Bootstrap and Private Repository Design

## Goal

Create a private `Brasaking1/TuinaDex` repository that preserves the current
TuinaDex source and submodule layout, then provide reproducible environment
files and Ubuntu 22.04 setup instructions on a separate branch.

## Repository layout

- Push the current upstream `main` unchanged to `Brasaking1/TuinaDex`.
- Keep `Arm-robot_VLA` and `Leap_Hand` as Git submodules pointing at their
  existing public MacroBright repositories and pinned commits.
- Put migration assets on `codex/laptop-bootstrap`; do not merge this branch
  into `main` automatically.

## Tracked content

The bootstrap branch will contain:

- a minimal explicit Conda specification for `arm_vla`;
- a minimal explicit Conda specification for `huawei_contest`;
- Ubuntu 22.04 clone, submodule, environment creation, and verification steps;
- notes separating server-only, Ubuntu hardware, and Mac client tasks.

Generated environment files must not contain absolute server paths. Package
versions will reflect the environments that were verified on the server.

## Excluded content

Do not commit Conda environment directories, credentials, SSH keys, datasets,
training checkpoints, model weights, caches, or machine-specific device data.
Large artifacts remain on server storage. A later artifact policy may put
selected release weights in Git LFS or a private model registry.

## Safety and verification

Before pushing, scan tracked additions for credentials and absolute server
paths. Validate that both Conda files parse, verify submodule URLs and pinned
commits, and confirm the root repository is clean after committing. Create the
GitHub repository as private, push `main` and `codex/laptop-bootstrap`, and do
not change the upstream MacroBright repositories.

## Success criteria

The private repository is visible under `Brasaking1`, both branches are
available, no secret or large artifact is uploaded, and an Ubuntu 22.04 laptop
can follow the documented commands to reconstruct the two project
environments.
