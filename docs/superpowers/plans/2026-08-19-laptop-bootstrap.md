# Laptop Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish TuinaDex to a private personal repository and provide reproducible Ubuntu 22.04 environment setup files without uploading large artifacts or secrets.

**Architecture:** Keep the upstream `main` commit unchanged and publish migration assets only on `codex/laptop-bootstrap`. Use small Conda history exports as the human-maintained dependency inputs, plus a setup guide that recreates environments and validates imports. Continue referencing the two public MacroBright repositories as pinned Git submodules.

**Tech Stack:** Git, GitHub CLI, Git submodules, Conda, pip, Ubuntu 22.04, Markdown, YAML

---

### Task 1: Capture sanitized environment inputs

**Files:**
- Create: `environments/arm_vla.yml`
- Create: `environments/huawei_contest.yml`

- [ ] **Step 1: Export the two verified server environments**

Run on the server:

```bash
conda env export -n arm_vla --from-history
conda env export -n huawei_contest --from-history
```

Expected: two YAML documents containing environment names, channels, and direct dependencies.

- [ ] **Step 2: Sanitize the exports**

Create the two files locally. Remove `prefix:` lines, editable local paths, and server-specific absolute paths. Preserve Python and direct package version constraints reported by Conda.

- [ ] **Step 3: Validate YAML syntax and prohibited content**

Run:

```bash
python3 -c "import pathlib, yaml; [yaml.safe_load(p.read_text()) for p in pathlib.Path('environments').glob('*.yml')]"
grep -RInE '/home/|BEGIN .*PRIVATE KEY|111111|218\\.194\\.|192\\.168\\.' environments && exit 1 || true
```

Expected: YAML command exits zero and secret/path scan prints nothing.

- [ ] **Step 4: Commit environment inputs**

```bash
git add environments/arm_vla.yml environments/huawei_contest.yml
git commit -m "build: add reproducible conda environment inputs"
```

### Task 2: Document Ubuntu laptop reconstruction

**Files:**
- Create: `docs/environment/ubuntu22-laptop-setup.md`

- [ ] **Step 1: Write the setup guide**

Document these exact stages: install Git/Conda, clone the private repository with `--recurse-submodules`, create both environments from `environments/*.yml`, install the arm and top-level projects editable, run offline imports/tests, and reserve USB/Dynamixel/RealSense checks for the Ubuntu laptop. State that training and GPU-heavy work remains on the server.

- [ ] **Step 2: Check every referenced repository path**

Run:

```bash
grep -nE 'Arm-robot_VLA|Leap_Hand|arm_vla|huawei_contest' docs/environment/ubuntu22-laptop-setup.md
git submodule status
```

Expected: both submodules and both environments are documented; submodule commits match the root repository pins.

- [ ] **Step 3: Scan the guide for credentials and server-only paths**

```bash
grep -nE '111111|218\\.194\\.|192\\.168\\.|/home/maziqi' docs/environment/ubuntu22-laptop-setup.md && exit 1 || true
```

Expected: no output.

- [ ] **Step 4: Commit the guide**

```bash
git add docs/environment/ubuntu22-laptop-setup.md
git commit -m "docs: add Ubuntu laptop bootstrap guide"
```

### Task 3: Verify the bootstrap branch

**Files:**
- Verify: `.gitmodules`
- Verify: `environments/arm_vla.yml`
- Verify: `environments/huawei_contest.yml`
- Verify: `docs/environment/ubuntu22-laptop-setup.md`

- [ ] **Step 1: Check formatting and repository state**

```bash
git diff main...HEAD --check
git status --short
git submodule status
```

Expected: no whitespace errors, clean working tree, and two pinned submodules.

- [ ] **Step 2: Confirm no large added files**

```bash
find environments docs/environment -type f -size +1M -print
```

Expected: no output.

- [ ] **Step 3: Confirm no credential-like additions**

```bash
git diff main...HEAD -- . ':!docs/superpowers' | grep -Ei 'password|private key|token=|111111' && exit 1 || true
```

Expected: no output.

### Task 4: Create and publish the private repository

**Files:**
- No new local files.

- [ ] **Step 1: Create the empty private GitHub repository**

```bash
gh repo create Brasaking1/TuinaDex --private --description "TuinaDex massage robot integration workspace"
```

Expected: GitHub returns the new repository URL and reports private visibility.

- [ ] **Step 2: Preserve upstream and add personal remote**

```bash
git remote rename origin upstream
git remote add origin git@github.com:Brasaking1/TuinaDex.git
git remote -v
```

Expected: `origin` points to Brasaking1 and `upstream` points to MacroBright.

- [ ] **Step 3: Push unchanged main and the bootstrap branch**

```bash
git push origin main:main
git push -u origin codex/laptop-bootstrap
```

Expected: both remote branches are created without changing MacroBright.

- [ ] **Step 4: Verify GitHub state**

```bash
gh repo view Brasaking1/TuinaDex --json nameWithOwner,visibility,defaultBranchRef,url
git ls-remote --heads origin main codex/laptop-bootstrap
```

Expected: repository is `PRIVATE`, default branch is `main`, and both branch heads are present.
