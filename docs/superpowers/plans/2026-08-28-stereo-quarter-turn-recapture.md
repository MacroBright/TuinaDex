# Stereo Quarter-Turn Recapture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Digitally normalize the fixed camera pair to upright 960×1280 frames, reject consecutive duplicate poses, and explicitly verify a horizontal stereo baseline before formal recapture.

**Architecture:** Keep `CaptureConfig.image_size` as the raw V4L2 request size and add `AppConfig.logical_image_size` for all post-rotation consumers. Normalize frames at the camera boundary, compare current corners with the last active pair in the service layer, and add translation-axis metrics to the existing immutable calibration artifacts.

**Tech Stack:** Python 3.10, OpenCV 5, NumPy 2.2, pytest, stdlib HTTP server, V4L2.

---

### Task 1: Quarter-turn configuration and frame normalization

**Files:**
- Modify: `tools/stereo_calibration/config.py`
- Modify: `tools/stereo_calibration/detection.py`
- Modify: `tools/stereo_calibration/cameras.py`
- Modify: `tools/stereo_calibration/service.py`
- Modify: `tools/stereo_calibration/calibration.py`
- Test: `tools/stereo_calibration/tests/test_config.py`
- Test: `tools/stereo_calibration/tests/test_detection.py`
- Test: `tools/stereo_calibration/tests/test_cameras.py`

- [ ] **Step 1: Write failing tests**

Add tests asserting rotations are limited to `(0, 90, 180, 270)`, mixed quarter-turn parity is rejected, left 270/right 90 produces `logical_image_size == (960, 1280)`, and 90°/270° pixel output equals OpenCV's quarter-turn constants.

```python
assert AppConfig.from_mapping(mapping).logical_image_size == (960, 1280)
np.testing.assert_array_equal(normalize_frame(frame, 90), cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))
np.testing.assert_array_equal(normalize_frame(frame, 270), cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE))
```

- [ ] **Step 2: Verify RED on Ubuntu**

```bash
python -m pytest tools/stereo_calibration/tests/test_config.py tools/stereo_calibration/tests/test_detection.py tools/stereo_calibration/tests/test_cameras.py -q
```

Expected: new tests fail for unsupported quarter turns and missing logical size.

- [ ] **Step 3: Implement the minimal size/rotation behavior**

```python
@property
def logical_image_size(self) -> tuple[int, int]:
    if self.left.rotation_degrees in (90, 270):
        return self.capture.height, self.capture.width
    return self.capture.image_size
```

Support the two OpenCV quarter-turn constants. Validate normalized frames against the logical size while keeping device open/readback checks at `capture.image_size`. Replace post-rotation bounds, metadata and artifact consumers with `config.logical_image_size`.

- [ ] **Step 4: Verify GREEN and commit**

```bash
python -m pytest tools/stereo_calibration/tests/test_config.py tools/stereo_calibration/tests/test_detection.py tools/stereo_calibration/tests/test_cameras.py -q
git add tools/stereo_calibration/config.py tools/stereo_calibration/detection.py tools/stereo_calibration/cameras.py tools/stereo_calibration/service.py tools/stereo_calibration/calibration.py tools/stereo_calibration/tests
git commit -m "feat: support quarter-turn stereo normalization"
```

### Task 2: Consecutive duplicate-pose guard

**Files:**
- Modify: `tools/stereo_calibration/service.py`
- Modify: `tools/stereo_calibration/static/index.html`
- Test: `tools/stereo_calibration/tests/test_web.py`

- [ ] **Step 1: Write failing service tests**

Cover first pair allowed; either camera below 15 px blocks; both cameras at or above 15 px allow; rejecting the latest pair changes the reference. Require `capture_blocker` in status.

```python
assert status["can_capture"] is False
assert status["capture_blocker"] == "与上一组过于相似，请明显移动、倾斜或改变距离"
```

- [ ] **Step 2: Verify RED**

```bash
python -m pytest tools/stereo_calibration/tests/test_web.py -q
```

- [ ] **Step 3: Implement and render the guard**

Load only the last active pair's corner metadata. For each side compute `median(norm(current - previous))` and require the smaller side value to be at least 15.0. Expose the safe blocker text, render it in the page, and remove the misleading unconditional movement warning.

- [ ] **Step 4: Verify GREEN and commit**

```bash
python -m pytest tools/stereo_calibration/tests/test_web.py -q
git add tools/stereo_calibration/service.py tools/stereo_calibration/static/index.html tools/stereo_calibration/tests/test_web.py
git commit -m "feat: reject consecutive duplicate calibration poses"
```

### Task 3: Horizontal-baseline reporting

**Files:**
- Modify: `tools/stereo_calibration/calibration.py`
- Test: `tools/stereo_calibration/tests/test_calibration.py`

- [ ] **Step 1: Write failing report tests**

Require translation X/Y/Z and `horizontal_baseline` in JSON and Markdown. Y- or Z-dominant translation must downgrade an otherwise passing report to `warning`.

```python
assert report["metrics"]["horizontal_baseline"] is False
assert report["status"] == "warning"
```

- [ ] **Step 2: Verify RED**

```bash
python -m pytest tools/stereo_calibration/tests/test_calibration.py -q
```

- [ ] **Step 3: Implement, verify and commit**

Flatten the validated `(3, 1)` translation; define `abs(tx) > max(abs(ty), abs(tz))`; add fields to JSON and Chinese Markdown; include the condition in status calculation.

```bash
python -m pytest tools/stereo_calibration/tests/test_calibration.py -q
git add tools/stereo_calibration/calibration.py tools/stereo_calibration/tests/test_calibration.py
git commit -m "feat: report horizontal stereo baseline direction"
```

### Task 4: Operator configuration and hardware proof

**Files:**
- Modify: `tools/stereo_calibration/example_config.json`
- Modify: `tools/stereo_calibration/README.md`
- Modify: `tools/stereo_calibration/tests/test_main.py`

- [ ] **Step 1: Update example and guide**

Set left rotation 270 and right rotation 90. Document raw 1280×960 versus logical 960×1280, unique poses, new sessions, and horizontal baseline acceptance.

- [ ] **Step 2: Run full verification**

```bash
python -m pytest tools/stereo_calibration/tests -q
python -m compileall -q tools/stereo_calibration
git diff --check
```

- [ ] **Step 3: Commit, push only the feature branch and deploy**

```bash
git add tools/stereo_calibration/example_config.json tools/stereo_calibration/README.md tools/stereo_calibration/tests/test_main.py
git commit -m "docs: prepare upright stereo recapture"
git push upstream codex/stereo-calibration
```

Fast-forward `/home/mzq/projects/TuinaDex-stereo-calibration` at `mzq@113.54.195.195`. Do not touch `/home/mzq/projects/TuinaDex` and do not merge `main`.

- [ ] **Step 4: Prove the real-camera path**

Create a new private dry-run config/session, run `--check-cameras`, confirm each logical preview is 960×1280 and upright, prove an immediate duplicate is blocked, capture three genuinely different poses, then stop and resume the same session.

- [ ] **Step 5: Start formal capture only after measuring the baseline**

Put the lens-center distance to the nearest millimetre in a separate formal config and collect 24–30 distinct poses. Preserve `stereo-dry-run-20260827` unchanged.
