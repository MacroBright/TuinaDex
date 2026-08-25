# Stereo Camera Bar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a one-piece, 300 mm reinforced stereo-camera mounting bar for two M6-mounted cameras and a central 1/4\"-20 tripod connection.

**Architecture:** Keep the editable design in one parameterized OpenSCAD source file. Use a standard-library Python geometry test to export a fresh binary STL through OpenSCAD, validate the declared dimensions, parse the mesh, check its bounds and manifold edge counts, and prevent the checked-in STL from drifting away from the source.

**Tech Stack:** OpenSCAD CLI, Python 3 standard library, STL, Git.

---

## File structure

- Create `hardware/stereo_camera_bar/stereo_camera_bar.scad`: canonical parameterized solid model.
- Create `hardware/stereo_camera_bar/tests/test_geometry.py`: source-parameter, STL-bound, triangle, and watertightness checks.
- Create `hardware/stereo_camera_bar/README.md`: hardware list, printing orientation, assembly, and safety notes.
- Create `hardware/stereo_camera_bar/output/stereo_camera_bar.stl`: ready-to-slice mesh exported from the source.
- Create `hardware/stereo_camera_bar/output/stereo_camera_bar_preview.png`: visual review image.

### Task 1: Prepare an isolated modeling worktree and OpenSCAD CLI

**Files:**
- No project files changed.

- [ ] **Step 1: Create an isolated worktree**

Use the `superpowers:using-git-worktrees` skill to create a worktree for branch `codex/stereo-camera-bar`. Do not carry the modified `Leap_Hand` submodule or unrelated `.superpowers/` working-tree files into the new branch.

- [ ] **Step 2: Check for OpenSCAD**

Run:

```bash
if command -v openscad >/dev/null 2>&1; then
  openscad --version
elif [[ -x /Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD ]]; then
  /Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD --version
else
  exit 1
fi
```

Expected: an OpenSCAD version line. The current machine is expected to exit with status 1 because OpenSCAD was not found during planning.

- [ ] **Step 3: Install OpenSCAD when missing**

Run:

```bash
brew install --cask openscad
/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD --version
```

Expected: Homebrew completes successfully and the second command prints an OpenSCAD version.

### Task 2: Define the geometry contract as a failing test

**Files:**
- Create: `hardware/stereo_camera_bar/tests/test_geometry.py`
- Test: `hardware/stereo_camera_bar/tests/test_geometry.py`

- [ ] **Step 1: Write the geometry test before the model exists**

Create `hardware/stereo_camera_bar/tests/test_geometry.py` with:

```python
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCAD = ROOT / "stereo_camera_bar.scad"

EXPECTED_PARAMETERS = {
    "bar_length": 300.0,
    "bar_width": 42.0,
    "plate_thickness": 8.0,
    "rib_height": 12.0,
    "mount_hole_diameter": 6.6,
    "mount_hole_pitch": 10.0,
    "mount_hole_min_x": 40.0,
    "mount_hole_max_x": 120.0,
    "tripod_bore_diameter": 7.0,
    "tripod_nut_across_flats": 11.6,
}


def openscad_executable() -> str:
    candidates = [
        os.environ.get("OPENSCAD"),
        "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD",
        "openscad",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if candidate == "openscad":
            result = subprocess.run(
                ["/usr/bin/env", "which", candidate],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        elif Path(candidate).is_file():
            return candidate
    raise RuntimeError("OpenSCAD CLI was not found")


def declared_parameter(source: str, name: str) -> float:
    match = re.search(
        rf"^\s*{re.escape(name)}\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*;",
        source,
        flags=re.MULTILINE,
    )
    if not match:
        raise AssertionError(f"parameter not declared: {name}")
    return float(match.group(1))


def load_binary_stl(path: Path) -> list[tuple[tuple[float, float, float], ...]]:
    data = path.read_bytes()
    if len(data) < 84:
        raise AssertionError("STL is shorter than its binary header")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if len(data) != expected_size:
        raise AssertionError(
            f"unexpected binary STL size: {len(data)} != {expected_size}"
        )
    triangles = []
    offset = 84
    for _ in range(triangle_count):
        values = struct.unpack_from("<12fH", data, offset)
        triangles.append((values[3:6], values[6:9], values[9:12]))
        offset += 50
    return triangles


def cross_length(a, b, c) -> float:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return math.sqrt(sum(component * component for component in cross))


def quantized(vertex) -> tuple[int, int, int]:
    return tuple(round(value * 100_000) for value in vertex)


class StereoCameraBarGeometryTest(unittest.TestCase):
    def test_declared_parameters_match_approved_specification(self):
        source = SCAD.read_text(encoding="utf-8")
        for name, expected in EXPECTED_PARAMETERS.items():
            self.assertEqual(declared_parameter(source, name), expected, name)
        self.assertIn(
            "[mount_hole_min_x : mount_hole_pitch : mount_hole_max_x]",
            source,
        )
        self.assertIn("for (side = [-1, 1])", source)

    def test_exported_stl_is_watertight_and_has_expected_bounds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stl_path = Path(temp_dir) / "stereo_camera_bar.stl"
            result = subprocess.run(
                [
                    openscad_executable(),
                    "--hardwarnings",
                    "--export-format",
                    "binstl",
                    "-o",
                    str(stl_path),
                    str(SCAD),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            triangles = load_binary_stl(stl_path)

        self.assertGreater(len(triangles), 500)
        vertices = [vertex for triangle in triangles for vertex in triangle]
        for vertex in vertices:
            self.assertTrue(all(math.isfinite(value) for value in vertex))

        minimum = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
        maximum = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
        size = [maximum[axis] - minimum[axis] for axis in range(3)]
        for actual, expected in zip(size, (300.0, 42.0, 20.0)):
            self.assertAlmostEqual(actual, expected, places=3)

        edge_counts = {}
        for triangle in triangles:
            self.assertGreater(cross_length(*triangle), 1e-8)
            points = [quantized(vertex) for vertex in triangle]
            for start, end in ((0, 1), (1, 2), (2, 0)):
                edge = tuple(sorted((points[start], points[end])))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        nonmanifold = [edge for edge, count in edge_counts.items() if count != 2]
        self.assertEqual(nonmanifold, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify the intended failure**

Run:

```bash
python3 -m unittest hardware/stereo_camera_bar/tests/test_geometry.py -v
```

Expected: errors because `hardware/stereo_camera_bar/stereo_camera_bar.scad` does not exist yet. This proves the test is exercising the missing deliverable.

- [ ] **Step 3: Commit the failing contract test**

Run:

```bash
git add hardware/stereo_camera_bar/tests/test_geometry.py
git commit -m "test: define stereo camera bar geometry contract"
```

Expected: one commit containing only the test.

### Task 3: Implement the parameterized OpenSCAD model

**Files:**
- Create: `hardware/stereo_camera_bar/stereo_camera_bar.scad`
- Test: `hardware/stereo_camera_bar/tests/test_geometry.py`

- [ ] **Step 1: Create the minimum model that satisfies the approved geometry**

Create `hardware/stereo_camera_bar/stereo_camera_bar.scad` with:

```scad
$fn = 64;

bar_length = 300.0;
bar_width = 42.0;
plate_thickness = 8.0;
corner_radius = 2.0;

rib_height = 12.0;
rib_thickness = 4.0;
rib_end_margin = 2.0;

mount_hole_diameter = 6.6;
mount_hole_pitch = 10.0;
mount_hole_min_x = 40.0;
mount_hole_max_x = 120.0;

center_boss_length = 48.0;
tripod_bore_diameter = 7.0;
tripod_nut_across_flats = 11.6;
tripod_nut_depth = 6.2;

epsilon = 0.2;

module rounded_prism(length, width, height, radius) {
    linear_extrude(height = height)
        offset(r = radius)
            square([length - 2 * radius, width - 2 * radius], center = true);
}

module hex_pocket(across_flats, height) {
    cylinder(h = height, r = across_flats / sqrt(3), $fn = 6);
}

difference() {
    union() {
        rounded_prism(
            bar_length,
            bar_width,
            plate_thickness,
            corner_radius
        );

        for (side = [-1, 1]) {
            translate([
                -(bar_length - 2 * rib_end_margin) / 2,
                side < 0 ? -bar_width / 2 : bar_width / 2 - rib_thickness,
                plate_thickness
            ])
                cube([
                    bar_length - 2 * rib_end_margin,
                    rib_thickness,
                    rib_height
                ]);
        }

        translate([
            -center_boss_length / 2,
            -bar_width / 2,
            plate_thickness
        ])
            cube([center_boss_length, bar_width, rib_height]);
    }

    for (side = [-1, 1]) {
        for (
            x = [mount_hole_min_x : mount_hole_pitch : mount_hole_max_x]
        ) {
            translate([side * x, 0, -epsilon])
                cylinder(
                    h = plate_thickness + 2 * epsilon,
                    d = mount_hole_diameter
                );
        }
    }

    translate([0, 0, -epsilon])
        cylinder(
            h = plate_thickness + rib_height + 2 * epsilon,
            d = tripod_bore_diameter
        );

    translate([
        0,
        0,
        plate_thickness + rib_height - tripod_nut_depth
    ])
        hex_pocket(tripod_nut_across_flats, tripod_nut_depth + epsilon);
}
```

- [ ] **Step 2: Run the geometry test**

Run:

```bash
python3 -m unittest hardware/stereo_camera_bar/tests/test_geometry.py -v
```

Expected: two tests pass. If OpenSCAD reports a geometry warning, treat it as a failure because `--hardwarnings` is enabled.

- [ ] **Step 3: Inspect the generated geometry in OpenSCAD**

Run:

```bash
open -a OpenSCAD hardware/stereo_camera_bar/stereo_camera_bar.scad
```

Expected: a single 300 mm bar with a flat camera face, two underside ribs, symmetric M6 holes, and one central tripod boss. The nut pocket must open on the rib side of the part.

- [ ] **Step 4: Commit the source model**

Run:

```bash
git add hardware/stereo_camera_bar/stereo_camera_bar.scad
git commit -m "feat: add reinforced stereo camera bar model"
```

Expected: one commit containing only the OpenSCAD model.

### Task 4: Export deliverables and document printing and assembly

**Files:**
- Create: `hardware/stereo_camera_bar/README.md`
- Create: `hardware/stereo_camera_bar/output/stereo_camera_bar.stl`
- Create: `hardware/stereo_camera_bar/output/stereo_camera_bar_preview.png`
- Test: `hardware/stereo_camera_bar/tests/test_geometry.py`

- [ ] **Step 1: Export the binary STL from the approved source**

Run:

```bash
mkdir -p hardware/stereo_camera_bar/output
/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD \
  --hardwarnings \
  --export-format binstl \
  -o hardware/stereo_camera_bar/output/stereo_camera_bar.stl \
  hardware/stereo_camera_bar/stereo_camera_bar.scad
```

Expected: exit status 0 and a non-empty STL.

- [ ] **Step 2: Render a preview image**

Run:

```bash
/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD \
  --autocenter \
  --viewall \
  --projection o \
  --imgsize 1600,900 \
  --render \
  -o hardware/stereo_camera_bar/output/stereo_camera_bar_preview.png \
  hardware/stereo_camera_bar/stereo_camera_bar.scad
```

Expected: exit status 0 and a PNG showing the complete model.

- [ ] **Step 3: Write assembly and slicing guidance**

Create `hardware/stereo_camera_bar/README.md` containing:

```markdown
# Stereo camera mounting bar

This folder contains the editable and ready-to-print versions of the reinforced stereo-camera bar.

## Files

- `stereo_camera_bar.scad`: parameterized source; dimensions are millimetres.
- `output/stereo_camera_bar.stl`: ready to import into Bambu Studio.
- `output/stereo_camera_bar_preview.png`: orientation and geometry preview.

## Required hardware

- Two M6 camera mounting screws. Start with M6×12 mm and verify thread engagement by hand before tightening.
- Two M6 flat washers.
- One standard 1/4\"-20 UNC hex nut with 7/16\" (about 11.11 mm) width across flats.
- One tripod or quick-release plate with a standard 1/4\"-20 camera screw.

## Printing orientation

The STL is already in its support-free printing orientation. The large, completely flat camera-contact face goes on the build plate. The ribs and central tripod boss point upward while printing. Flip the part after printing so the ribs face downward during use.

Material is selected in the slicer and is not encoded in the STL. A 0.20 mm layer height, at least five wall loops, at least five top and bottom layers, and 35% gyroid infill are conservative starting settings.

## Assembly

1. Press the 1/4\"-20 metal nut into the central hexagonal recess on the rib side. Use a small amount of epoxy only if the fit is loose.
2. Attach the tripod plate from below and tighten its 1/4\"-20 screw into the metal nut.
3. Place both camera brackets on the flat face.
4. Insert the M6 screws from the rib side and tighten the cameras at symmetric hole positions.
5. Use `x = -80 mm` and `x = +80 mm` for an initial 160 mm camera baseline.
6. Tighten only enough to prevent movement. Confirm that screws do not bottom out in either camera bracket.
7. Complete stereo calibration only after the bar, tripod, camera spacing, and camera angles are final.

## Changing dimensions

Edit the named values at the top of `stereo_camera_bar.scad`. Re-export the STL and rerun `python3 -m unittest hardware/stereo_camera_bar/tests/test_geometry.py -v` after every geometry change.
```

- [ ] **Step 4: Run final verification**

Run:

```bash
python3 -m unittest hardware/stereo_camera_bar/tests/test_geometry.py -v
file hardware/stereo_camera_bar/output/stereo_camera_bar.stl
shasum -a 256 hardware/stereo_camera_bar/output/stereo_camera_bar.stl
git status --short
```

Expected: two passing tests; `file` identifies STL data; SHA-256 is printed; Git status contains only the README and the two intended output files.

- [ ] **Step 5: Commit the print deliverables**

Run:

```bash
git add \
  hardware/stereo_camera_bar/README.md \
  hardware/stereo_camera_bar/output/stereo_camera_bar.stl \
  hardware/stereo_camera_bar/output/stereo_camera_bar_preview.png
git commit -m "docs: add stereo camera bar print package"
```

Expected: one commit containing the README, STL, and preview PNG.

### Task 5: Confirm the branch is ready for handoff

**Files:**
- Verify all files created in Tasks 2–4.

- [ ] **Step 1: Run the complete verification again from a clean state**

Run:

```bash
python3 -m unittest hardware/stereo_camera_bar/tests/test_geometry.py -v
git status --short
git log --oneline -4
```

Expected: two passing tests, a clean worktree, and separate commits for the test, source model, and print package.

- [ ] **Step 2: Inspect the final preview and STL metadata**

Open `hardware/stereo_camera_bar/output/stereo_camera_bar_preview.png` and import `hardware/stereo_camera_bar/output/stereo_camera_bar.stl` into Bambu Studio. Confirm the displayed size is 300 mm × 42 mm × 20 mm and the flat face is on the build plate.

