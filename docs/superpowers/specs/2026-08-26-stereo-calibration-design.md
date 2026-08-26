# Stereo Camera Calibration Design

**Date:** 2026-08-26
**Status:** User-approved design
**Scope:** Guided capture, stereo calibration, rectification, and calibration validation. Dense point-cloud generation and LeRobot integration are explicitly deferred to the next phase.

## 1. Goal

Build a beginner-friendly calibration workflow for the two fixed USB cameras on the Ubuntu laptop. The user operates the workflow from a Mac browser over an SSH tunnel. The workflow must produce trustworthy stereo calibration parameters for later depth and point-cloud generation, not merely report that OpenCV completed without an exception.

Success means:

- the user can see both camera feeds and capture valid checkerboard pairs from a browser;
- invalid pairs are rejected before saving when possible;
- the workflow produces camera intrinsics, distortion coefficients, stereo extrinsics, rectification parameters, and the disparity-to-depth matrix;
- the workflow generates numerical quality metrics and human-readable rectified previews;
- a completed calibration can be reproduced from the saved raw images and configuration snapshot.

## 2. Physical Setup and Fixed Assumptions

The approved physical setup is:

- two identical LRCP U3-JX02 USB cameras;
- approximately parallel mounting;
- approximately 200 mm optical-center baseline;
- expected working distance of 700–800 mm;
- 1280×960 MJPEG capture at 30 frames per second;
- a printed 9×6 inner-corner checkerboard with measured 35 mm squares;
- a rigid camera bar that remains unchanged throughout capture and later stereo use.

The two cameras share the same serial number, so `/dev/v4l/by-id` is ambiguous. Configuration must use the stable physical USB port paths:

- logical camera 1: `/dev/v4l/by-path/pci-0000:04:00.4-usb-0:1:1.0-video-index0`;
- logical camera 2: `/dev/v4l/by-path/pci-0000:04:00.4-usb-0:2:1.0-video-index0`.

Logical camera 2 is physically rolled by approximately 180 degrees. Every frame from that camera must therefore be rotated with `cv2.ROTATE_180` before checkerboard detection, calibration, rectification, preview, or later depth processing. The configuration stores this rotation explicitly rather than relying on transient `/dev/videoN` numbers.

The two cameras must use the same resolution, frame rate, exposure mode, exposure value, gain, and white-balance mode during one calibration session. Exact exposure and gain values will be selected during the three-pair dry run, then frozen in the saved session configuration before the formal capture begins. Before the formal run, the user also measures lens-center-to-lens-center baseline to the nearest millimetre and records it as a validation reference; OpenCV still estimates `T` independently from the checkerboard observations.

Stereo calibration estimates the complete rigid transform between the cameras, including translation, yaw, pitch, and roll. Preserving only the 200 mm baseline is insufficient: moving or rotating either camera invalidates the stereo extrinsics and requires recalibration. Moving the entire rigid camera assembly does not invalidate camera-to-camera calibration, although it will invalidate any future camera-to-robot or camera-to-world extrinsic calibration.

## 3. System Architecture

The Ubuntu laptop owns all camera and calibration work. The Mac is a remote control surface only.

### Ubuntu responsibilities

- open both cameras using their configured `/dev/v4l/by-path` endpoints;
- apply the configured 180-degree rotation to logical camera 2;
- acquire paired frames and record timestamps;
- detect and refine checkerboard corners;
- evaluate whether a pair is suitable for capture;
- store raw image pairs and metadata;
- run monocular and stereo calibration;
- generate rectification maps, metrics, reports, and previews;
- serve the browser interface on `127.0.0.1:8765`.

### Mac responsibilities

- establish an SSH tunnel, for example:

  ```bash
  ssh -N -L 8765:127.0.0.1:8765 mzq@<ubuntu-ip>
  ```

- open `http://localhost:8765`;
- observe capture quality and move the checkerboard;
- save or delete image pairs and start calibration.

The Ubuntu service binds only to loopback. It is not exposed to the campus or public network.

## 4. Component Boundaries

Implementation will live under `tools/stereo_calibration/` and will not modify dexterous-hand control code or LeRobot behavior.

The tool is separated into focused components:

1. **Configuration**
   - Loads camera paths, capture mode, camera rotations, checkerboard dimensions, square size, web port, and data root.
   - Validates values before either camera is opened.
   - Writes an immutable configuration snapshot into each session.

2. **Camera pair acquisition**
   - Opens both configured devices and verifies the requested mode.
   - Calls `grab()` for both cameras before `retrieve()` to minimize software-side skew.
   - Applies the camera-2 rotation immediately after retrieval.
   - Publishes only complete pairs and preserves both acquisition timestamps.

3. **Checkerboard detection and capture-quality checks**
   - Uses OpenCV's robust checkerboard detector for 9×6 inner corners.
   - Requires 54 refined corners in both images.
   - Requires the outer detected corners to stay at least 12 pixels from every image edge.
   - Uses configurable gross-quality defaults: Laplacian variance of at least 60 in the checkerboard bounding region and fewer than 45% of full-frame pixels at grayscale values of 250 or above. The dry run may raise these thresholds, after which the values are frozen in the session snapshot.
   - Returns explicit reasons when a pair is not saveable.

4. **Session storage**
   - Creates a new dated session directory.
   - Assigns monotonically increasing pair numbers.
   - Saves the left image, rotated right image, timestamps, corner coordinates, and quality metadata together.
   - Supports resuming an interrupted session without overwriting existing pairs.

5. **Calibration engine**
   - Calibrates each camera independently.
   - Runs stereo calibration with the accepted intrinsics fixed.
   - Computes stereo rectification, projection matrices, rectification maps, and the `Q` disparity-to-depth matrix.
   - Computes per-view reprojection error and post-rectification vertical correspondence error.

6. **Local web server and static browser page**
   - Uses Python's `ThreadingHTTPServer` plus static HTML, CSS, and JavaScript rather than adding a web framework or requiring a full ROS stack.
   - Provides paired preview images, quality status, capture progress, save/delete actions, and calibration status.
   - Keeps camera access in one owner thread so concurrent browser requests cannot race the devices.

Each component has a narrow interface and can be tested without running the complete workflow.

## 5. Capture Workflow

### Dry run

The user first captures three trial pairs:

1. checkerboard centered near the 700–800 mm working distance;
2. checkerboard in an upper/left area with a modest tilt;
3. checkerboard in a lower/right area with the opposite tilt.

The dry run validates camera identity, digital rotation, focus, exposure, full 54-corner detection, pair storage, and browser controls. Exposure and gain are then frozen for the formal session.

### Formal capture

The user records 30 pairs with the cameras fixed and the checkerboard stationary at the moment of capture. The goal is to retain approximately 25 high-quality pairs after reviewing any outliers.

The desired distribution is:

- 12 pairs spanning the center, edges, and corners of the shared field of view;
- 6 pairs with different left/right tilts;
- 6 pairs with different up/down tilts;
- 3 pairs at approximately 550–650 mm;
- 3 pairs at approximately 900–1000 mm.

Most samples remain near the real 650–850 mm work range. Tilts should generally remain within approximately 15–30 degrees. The complete checkerboard must be visible and sharp in both images.

The browser enables **Save pair** only when both images contain all 54 corners and pass the gross quality checks. It displays the reason otherwise. The user can delete the most recent pair explicitly, but the tool never silently replaces or deletes images.

The cameras do not provide hardware synchronization. Software acquisition cannot make them truly simultaneous. This is acceptable for a stationary checkerboard: the user waits for it to stop moving before saving. The limitation must be documented for the later moving-hand and dense-point-cloud phase.

## 6. Calibration and Data Flow

For each saved pair, checkerboard object points are generated in millimetres from the verified 35 mm square size. The same orientation convention is used for both logical images after camera 2 has been rotated.

The computation sequence is:

1. load and validate all saved pairs;
2. calibrate camera 1 and camera 2 independently;
3. calculate per-camera reprojection errors;
4. stereo-calibrate with the intrinsics fixed to estimate relative rotation `R` and translation `T`;
5. stereo-rectify to produce `R1`, `R2`, `P1`, `P2`, `Q`, and valid image regions;
6. construct and save OpenCV remapping arrays;
7. rectify representative pairs and draw shared horizontal epipolar lines;
8. compute vertical residuals for corresponding rectified checkerboard corners;
9. write machine-readable parameters and a human-readable report.

Suspected outlier pairs are reported with their numbers and errors. They are not automatically deleted or silently excluded. If quality is unacceptable, the user removes or replaces the identified pairs and deliberately reruns calibration.

## 7. Data and Output Layout

Captured pair images and generated calibration results are local runtime data and must not be committed to Git. Only source code, tests, documentation, and an example configuration belong in the repository. The saved images are canonical logical-camera images: the camera-2 files have already received the configured 180-degree rotation, and that fact is recorded in the session configuration snapshot.

A session has a layout equivalent to:

```text
data/stereo_calibration/<session-id>/
├── config_snapshot.json
├── manifest.jsonl
├── pairs/
│   ├── pair_0001_left.png
│   ├── pair_0001_right.png
│   └── ...
└── results/<calibration-run-id>/
    ├── stereo_calibration.npz
    ├── stereo_calibration.yaml
    ├── report.json
    ├── report.md
    └── rectified_previews/
```

`stereo_calibration.npz` contains numeric arrays intended for the later rectification and point-cloud code. `stereo_calibration.yaml` contains the same essential parameters in an inspectable, interoperable form. The reports include the session configuration, retained image count, errors, estimated baseline, warnings, and suspected outlier pair numbers.

Every calibration run gets a new result directory. A failed run never overwrites a previous valid result.

## 8. Quality Gates

The report applies these interpretation thresholds:

- at least 18 valid pairs are required to run; approximately 25 well-distributed pairs are the target;
- left and right monocular reprojection RMS and stereo RMS should ideally be below 1.0 pixel;
- 1.0–1.5 pixels is a warning range that requires visual review;
- above 1.5 pixels is treated as a failed result and triggers pair review or recapture;
- rectified vertical correspondence error should have a median below 1.0 pixel and a 95th percentile below 2.0 pixels;
- estimated baseline should be physically plausible and close to the measured approximately 200 mm value;
- rectified preview images must show corresponding checkerboard features on the same horizontal guide lines.

These are validation gates, not proof that a later stereo matcher will work on textureless skin. Dense-depth quality will be evaluated separately in the point-cloud phase.

## 9. Error Handling and Recovery

- **Missing or ambiguous camera:** refuse to start and identify the failed configured by-path.
- **Unexpected capture mode:** report the actual resolution/format and refuse formal capture until corrected.
- **Camera disconnection or read failure:** mark the browser status red, disable saving, close/reopen devices only through an explicit retry action, and preserve saved data.
- **Checkerboard rejection:** display separate left/right corner counts and the applicable blur, border, or exposure reason.
- **Interrupted process:** recover the existing manifest and continue at the next unused pair number.
- **Malformed or mismatched pair:** exclude it from computation with an explicit report entry; never guess its partner.
- **Insufficient views or calibration failure:** preserve raw data, write diagnostics to a new failed run directory, and leave previous valid output untouched.
- **Browser disconnect:** continue camera operation safely on Ubuntu and allow the browser to reconnect without starting another camera owner.

No part of this workflow sends commands to the dexterous hand or robot arm.

## 10. Testing and Verification

Testing proceeds in layers:

1. **Automated unit tests**
   - configuration validation;
   - stable device-path resolution and camera-2 rotation;
   - pair numbering and resume behavior;
   - checkerboard object-point dimensions and millimetre scale;
   - metric calculations and NPZ/YAML round trips;
   - refusal to overwrite existing results.

2. **Offline integration tests**
   - run detection against known captured checkerboard images;
   - verify 54 corners after the configured right-camera rotation;
   - verify browser actions against a fake camera-pair source;
   - verify calibration output structure using deterministic sample observations.

3. **Three-pair hardware dry run**
   - validate both physical paths, matching capture modes, frame orientation, exposure, web preview, storage, and resume behavior.

4. **Thirty-pair formal session**
   - complete calibration;
   - inspect all numerical quality gates;
   - inspect rectified previews with horizontal lines;
   - confirm the estimated baseline against a direct physical measurement.

The phase is complete only when the calibration artifacts exist, the automated tests pass, the quality report meets the accepted thresholds, and the rectified preview passes visual inspection.

## 11. Deferred Work

The following items are intentionally outside this design:

- stereo matching and disparity tuning;
- dense depth maps and point clouds;
- filtering skin/back point clouds;
- locating acupoints;
- tracking a moving dexterous hand;
- LeRobot dataset recording or policy inference;
- camera-to-robot/world extrinsic calibration;
- hardware synchronization or replacing the USB cameras.

The next phase will consume `stereo_calibration.npz` and the rectification outputs to design and validate dense point-cloud generation.
