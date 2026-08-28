# Third-Party Notices

This file identifies the principal third-party software and model components
used by the project. The repository's AGPL-3.0-only license does not replace
the licenses that apply to third-party materials.

## Ultralytics YOLO

- Project: Ultralytics
- Source: https://github.com/ultralytics/ultralytics
- Declared dependency: `ultralytics = "^8.3.38"`
- License: GNU Affero General Public License version 3 (AGPL-3.0)
- License text: https://github.com/ultralytics/ultralytics/blob/main/LICENSE

The project imports Ultralytics' Python package for YOLO training and
inference. Users must comply with the Ultralytics licensing terms applicable to
the exact software version and model weights they obtain. Ultralytics also
offers a separate Enterprise License; this notice does not grant one.

No Ultralytics model weights are included in this repository snapshot.

## Segment Anything (SAM)

- Project: Segment Anything
- Copyright: Meta Platforms, Inc. and affiliates
- Source: https://github.com/facebookresearch/segment-anything
- Declared dependency: `segment-anything = "^1.0"`
- License: Apache License 2.0
- License text: https://github.com/facebookresearch/segment-anything/blob/main/LICENSE

The project imports Segment Anything for segmentation. Preserve the upstream
copyright, attribution, modification, and license notices when redistributing
SAM code or checkpoints.

No SAM checkpoint is included in this repository snapshot.

## AprilTag

- Project: AprilTag 3
- Source: https://github.com/AprilRobotics/apriltag
- License: BSD 2-Clause License at the time this notice was prepared
- License text: https://github.com/AprilRobotics/apriltag/blob/master/LICENSE.md

The project imports the separately built AprilTag Python module. Record the
exact commit used for each reproducible release and verify its license at that
commit.

## Other Python dependencies

Other dependencies are listed in `pyproject.toml` and
`ear_scanner_control/requirements.txt`. Each remains subject to its own license.
Before a release, generate and review a dependency license report for the exact
locked environment, and preserve any notices required by those licenses.
