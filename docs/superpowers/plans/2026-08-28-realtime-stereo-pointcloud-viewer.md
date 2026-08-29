# Realtime Stereo Point Cloud Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one Ubuntu-local PyQtGraph window that displays the live left image, depth colormap, and interactive coloured point cloud at a balanced target of roughly 5 FPS.

**Architecture:** A reusable processor validates the saved calibration, rectifies one stereo pair, computes half-resolution SGBM disparity, rescales it to the calibrated resolution, and publishes an immutable display snapshot. A single worker thread exclusively owns `OpenCVCameraPair`; the Qt GUI main thread polls only the latest snapshot and never queues stale frames.

**Tech Stack:** Python 3.10, NumPy 2.2, OpenCV 5, PyQt5 5.15, PyQtGraph 0.13, PyOpenGL 3.1, V4L2, pytest.

---

## File map

- Create `tools/stereo_calibration/realtime.py`: calibration loading, balanced-mode depth/point-cloud processing, latest-frame worker state.
- Create `tools/stereo_calibration/realtime_viewer.py`: CLI and Ubuntu-local PyQtGraph GUI.
- Create `tools/stereo_calibration/tests/test_realtime.py`: focused processor and worker lifecycle tests.
- Modify `tools/stereo_calibration/README.md`: installation and exact launch command.

### Task 1: Balanced realtime processing core

**Files:**
- Create: `tools/stereo_calibration/realtime.py`
- Create: `tools/stereo_calibration/tests/test_realtime.py`

- [ ] **Step 1: Write failing calibration and scale tests**

Add tests that create a temporary NPZ with `map_left_x`, `map_left_y`, `map_right_x`, `map_right_y`, and `disparity_to_depth`; assert malformed shapes are rejected. Inject a fake matcher returning half-resolution disparity `-80 px`, call the processor, and assert the full-resolution disparity is approximately `-160 px`, the output images match the logical image size, and finite point colours correspond to the rectified left image.

```python
def test_balanced_processor_restores_full_resolution_disparity(tmp_path):
    calibration = write_identity_calibration(tmp_path, width=8, height=6)
    matcher = FakeMatcher(np.full((3, 4), -80 * 16, dtype=np.int16))
    processor = RealtimeProcessor(calibration, matcher_factory=lambda **_: matcher)
    snapshot = processor.process(frame_pair(8, 6))
    assert snapshot.disparity_px.shape == (6, 8)
    assert np.allclose(snapshot.disparity_px, -160.0)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m pytest -q tools/stereo_calibration/tests/test_realtime.py
```

Expected: collection fails because `tools.stereo_calibration.realtime` does not exist.

- [ ] **Step 3: Implement the minimal processor**

Define immutable `CalibrationMaps` and `RealtimeSnapshot` dataclasses. `load_calibration()` must copy arrays out of the NPZ and validate full-resolution float maps plus a finite `4x4` Q matrix. `RealtimeProcessor.process()` must rectify, resize to 50%, compute `minDisparity=-128` and `numDisparities=128`, resize disparity back with nearest-neighbour interpolation, multiply disparity by two, reproject with the full-resolution Q matrix, filter `200 < z < 5000 mm`, subsample every fourth row/column, and build RGB colours plus a Turbo depth image.

Core scale conversion:

```python
small_disparity = matcher.compute(left_small, right_small).astype(np.float32) / 16.0
full_disparity = cv2.resize(
    small_disparity, (width, height), interpolation=cv2.INTER_NEAREST
) / scale
points = cv2.reprojectImageTo3D(full_disparity, calibration.q)
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the same focused pytest command. Expected: all tests in `test_realtime.py` pass.

- [ ] **Step 5: Commit the processing core**

```bash
git add tools/stereo_calibration/realtime.py tools/stereo_calibration/tests/test_realtime.py
git commit -m "feat: add realtime stereo point cloud processor"
```

### Task 2: Latest-frame worker and single PyQtGraph window

**Files:**
- Modify: `tools/stereo_calibration/realtime.py`
- Create: `tools/stereo_calibration/realtime_viewer.py`
- Modify: `tools/stereo_calibration/tests/test_realtime.py`

- [ ] **Step 1: Write failing worker lifecycle tests**

Use a fake pair source and fake processor. Assert that repeated frames replace the stored snapshot rather than accumulating a queue, `stop()` joins the worker and closes the source exactly once, and a processing exception publishes an error with no stale snapshot.

```python
def test_worker_replaces_old_snapshot_and_releases_camera():
    source = FakeSource([pair(1), pair(2)])
    worker = RealtimeWorker(lambda: source, FakeProcessor())
    worker.start()
    wait_until(lambda: worker.state().sequence >= 2)
    worker.stop()
    assert worker.state().snapshot.marker == 2
    assert source.close_calls == 1
```

- [ ] **Step 2: Run the focused test and verify RED**

Run the focused pytest command. Expected: fails because `RealtimeWorker` and its state API do not exist.

- [ ] **Step 3: Implement worker ownership and latest-state publication**

Add a condition-protected `RealtimeState(sequence, snapshot, fps, error)` and `RealtimeWorker`. The worker opens the pair source inside its thread, processes continuously, replaces one stored snapshot, calculates a rolling FPS, and always closes the source in `finally`. `stop()` sets an event and joins with a bounded timeout; timeout is reported as an explicit runtime error.

- [ ] **Step 4: Implement the PyQtGraph GUI CLI**

`realtime_viewer.py` must:

- validate `DISPLAY` or `WAYLAND_DISPLAY` before opening cameras;
- import PyQt5/PyQtGraph lazily and explain how to install the pinned GUI dependencies if missing;
- parse `--config`, `--calibration`, `--min-depth-mm`, `--max-depth-mm`, and `--stride`;
- create one Qt GUI window with two image labels on the left, one `GLViewWidget` on the right, status text, pause/resume, reset-view, and exit buttons;
- update widgets from the main thread using the latest immutable snapshot;
- replace the point-cloud geometry under a fixed name instead of adding new geometry each frame;
- call `worker.stop()` once when the window closes.

The GUI must use a Qt main-thread timer to poll immutable worker state rather than modifying widgets from the camera worker.

- [ ] **Step 5: Run focused tests and compile checks**

```bash
python -m pytest -q tools/stereo_calibration/tests/test_realtime.py
python -m compileall -q tools/stereo_calibration/realtime.py \
  tools/stereo_calibration/realtime_viewer.py
git diff --check
```

Expected: focused tests pass; compile and diff checks have exit code 0.

- [ ] **Step 6: Commit the viewer**

```bash
git add tools/stereo_calibration/realtime.py \
  tools/stereo_calibration/realtime_viewer.py \
  tools/stereo_calibration/tests/test_realtime.py
git commit -m "feat: add Ubuntu realtime point cloud viewer"
```

### Task 3: Ubuntu deployment and one real-camera smoke check

**Files:**
- Modify: `tools/stereo_calibration/README.md`

- [ ] **Step 1: Document exact local launch instructions**

Add a section that installs pinned PyQt5/PyQtGraph/PyOpenGL packages in `tuinadex_hw` and runs:

```bash
cd ~/projects/TuinaDex-stereo-calibration
python -m tools.stereo_calibration.realtime_viewer \
  --config ~/.config/tuinadex/stereo-upright-dry-run.json \
  --calibration ~/projects/TuinaDex/data/stereo_calibration/stereo-recalibration-v2-20260828/results/20260828-215540665234/stereo_calibration.npz
```

State explicitly that the command belongs in an Ubuntu desktop terminal, not an SSH shell.

- [ ] **Step 2: Commit and push only the feature branch**

```bash
git add tools/stereo_calibration/README.md
git commit -m "docs: explain local realtime point cloud viewer"
git push upstream codex/stereo-calibration
```

- [ ] **Step 3: Install and deploy on Ubuntu**

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tuinadex_hw
python -m pip install "PyQt5==5.15.11" "pyqtgraph==0.13.7" "PyOpenGL==3.1.10"
cd ~/projects/TuinaDex-stereo-calibration
git pull --ff-only origin codex/stereo-calibration
```

- [ ] **Step 4: Perform one local-display smoke check**

From the Ubuntu desktop terminal, launch the documented command. Verify that one window shows live colour, depth, and interactive point cloud; the status reports nonzero FPS and point count; pause/resume and reset-view respond; closing the window exits the process and `fuser` reports neither camera device remains open.

- [ ] **Step 5: Record fresh verification evidence**

Run only:

```bash
python -m pytest -q tools/stereo_calibration/tests/test_realtime.py
python -m compileall -q tools/stereo_calibration/realtime.py \
  tools/stereo_calibration/realtime_viewer.py
git status --short
```

Expected: focused tests pass, compile exits 0, and the feature worktree is clean.
