"""Frame-level wrapper around a pose detector and pose estimator.

The wrapper normalises the outputs of a detector backend (person boxes)
and an estimator backend (COCO-17 joints per box) into one validated,
JSON-compatible result for a single image. It performs no inference of
its own and depends only on NumPy.
"""

from __future__ import annotations

from copy import deepcopy
import math
import time

import numpy as np


class PoseExtractionError(Exception):
    """Raised when an image, a backend result, or an extraction step is invalid."""


CANONICAL_JOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

_CANONICAL_SET = frozenset(CANONICAL_JOINT_NAMES)


def _is_number(value):
    """True for a finite int/float/NumPy number; bools and strings are not numbers."""
    if isinstance(value, bool) or isinstance(value, np.bool_):
        return False
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _as_python_number(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _is_sequence(value):
    # A zero-dimensional NumPy array is not a sequence: iterating over it
    # raises a raw TypeError, so it must fail the sequence check and be
    # reported as a PoseExtractionError naming the field, not crash.
    if isinstance(value, np.ndarray):
        return value.ndim >= 1
    return isinstance(value, (list, tuple))


def _validate_box(box, field, width, height):
    """Validate one box_xyxy and return it as a new list of plain numbers."""
    if not _is_sequence(box) or isinstance(box, np.ndarray) and box.ndim != 1:
        raise PoseExtractionError(f"{field}: box_xyxy must be a sequence of four numbers")
    values = list(box)
    if len(values) != 4:
        raise PoseExtractionError(f"{field}: box_xyxy must have exactly four numbers")
    for value in values:
        if not _is_number(value):
            raise PoseExtractionError(f"{field}: box_xyxy entries must be finite numbers")
    x0, y0, x1, y1 = (_as_python_number(v) for v in values)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise PoseExtractionError(
            f"{field}: box_xyxy [{x0}, {y0}, {x1}, {y1}] is not a valid box "
            f"inside a {width}x{height} image"
        )
    return [x0, y0, x1, y1]


def _validate_joint_names(names):
    if not _is_sequence(names):
        raise PoseExtractionError("joint_names: must be a list of the 17 joint names")
    names = list(names)
    if len(names) != len(CANONICAL_JOINT_NAMES):
        raise PoseExtractionError(
            f"joint_names: expected {len(CANONICAL_JOINT_NAMES)} names, got {len(names)}"
        )
    for name in names:
        if not isinstance(name, str):
            raise PoseExtractionError("joint_names: every joint name must be a string")
    if len(set(names)) != len(names):
        raise PoseExtractionError("joint_names: duplicate joint names are not allowed")
    unknown = set(names) - _CANONICAL_SET
    if unknown:
        raise PoseExtractionError(f"joint_names: unknown joint names {sorted(unknown)}")
    missing = _CANONICAL_SET - set(names)
    if missing:
        raise PoseExtractionError(f"joint_names: missing joint names {sorted(missing)}")
    return names


def _validate_xy(xy):
    if not _is_sequence(xy):
        raise PoseExtractionError("xy: must be a list of 17 joint positions or nulls")
    entries = list(xy)
    if len(entries) != len(CANONICAL_JOINT_NAMES):
        raise PoseExtractionError(
            f"xy: expected {len(CANONICAL_JOINT_NAMES)} entries, got {len(entries)}"
        )
    validated = []
    for entry in entries:
        if entry is None:
            validated.append(None)
            continue
        if not _is_sequence(entry):
            raise PoseExtractionError("xy: each entry must be null or an [x, y] pair")
        pair = list(entry)
        if len(pair) != 2:
            raise PoseExtractionError("xy: each entry must be null or an [x, y] pair")
        if not all(_is_number(v) for v in pair):
            raise PoseExtractionError("xy: coordinates must be finite numbers")
        validated.append([_as_python_number(pair[0]), _as_python_number(pair[1])])
    return validated


def _validate_scores(scores):
    if not _is_sequence(scores):
        raise PoseExtractionError("scores_raw: must be a list of 17 scores or nulls")
    entries = list(scores)
    if len(entries) != len(CANONICAL_JOINT_NAMES):
        raise PoseExtractionError(
            f"scores_raw: expected {len(CANONICAL_JOINT_NAMES)} entries, "
            f"got {len(entries)}"
        )
    validated = []
    for entry in entries:
        if entry is None:
            validated.append(None)
        elif _is_number(entry):
            validated.append(_as_python_number(entry))
        else:
            raise PoseExtractionError("scores_raw: scores must be finite numbers or null")
    return validated


def _validate_transform(to_original):
    if to_original is None:
        return None
    if not _is_sequence(to_original):
        raise PoseExtractionError("to_original: must be null or a 2x3 matrix")
    rows = list(to_original)
    if len(rows) != 2:
        raise PoseExtractionError("to_original: must be null or a 2x3 matrix")
    matrix = []
    for row in rows:
        if not _is_sequence(row):
            raise PoseExtractionError("to_original: must be null or a 2x3 matrix")
        values = list(row)
        if len(values) != 3:
            raise PoseExtractionError("to_original: must be null or a 2x3 matrix")
        if not all(_is_number(v) for v in values):
            raise PoseExtractionError("to_original: entries must be finite numbers")
        matrix.append([_as_python_number(v) for v in values])
    return matrix


class DWPoseFrameExtractor:
    """Wrap a detector and an estimator into one validated per-frame result."""

    def __init__(self, detector, estimator, model_info):
        self.detector = detector
        self.estimator = estimator
        self.model_info = deepcopy(model_info)

    def extract_frame(self, image, reviewed_boxes=None):
        total_start = time.perf_counter()

        height, width = self._validate_image(image)

        detector_ms = 0.0
        estimator_ms = 0.0

        if reviewed_boxes is not None:
            boxes, scores = self._reviewed_boxes(reviewed_boxes, width, height)
            box_source = "reviewed"
        else:
            stage_start = time.perf_counter()
            try:
                raw_detections = self.detector.detect(image)
            except Exception as error:
                raise PoseExtractionError(f"detector failed: {error}") from error
            detector_ms = (time.perf_counter() - stage_start) * 1000.0
            boxes, scores = self._detector_boxes(raw_detections, width, height)
            box_source = "detector"

        candidates = []
        if boxes:
            stage_start = time.perf_counter()
            try:
                raw_results = self.estimator.estimate(
                    image, [list(box) for box in boxes]
                )
            except Exception as error:
                raise PoseExtractionError(f"estimator failed: {error}") from error
            estimator_ms = (time.perf_counter() - stage_start) * 1000.0

            poses = self._validate_estimator_results(raw_results, len(boxes))
            built = [
                self._build_candidate(box, score, box_source, pose, width, height)
                for box, score, pose in zip(boxes, scores, poses)
            ]
            candidates = self._sort_candidates(built, box_source)

        total_ms = (time.perf_counter() - total_start) * 1000.0

        return {
            "width": width,
            "height": height,
            "status": "completed" if candidates else "no_detections",
            "candidates": candidates,
            "model_info": deepcopy(self.model_info),
            "timing_ms": {
                "detector": float(detector_ms),
                "estimator": float(estimator_ms),
                "total": float(total_ms),
            },
        }

    # -- inputs ---------------------------------------------------------

    @staticmethod
    def _validate_image(image):
        if not isinstance(image, np.ndarray):
            raise PoseExtractionError("image: must be a NumPy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise PoseExtractionError(
                "image: must have shape (height, width, 3)"
            )
        if image.dtype != np.uint8:
            raise PoseExtractionError("image: dtype must be uint8")
        height, width = int(image.shape[0]), int(image.shape[1])
        if height < 2 or width < 2:
            raise PoseExtractionError("image: height and width must each be at least 2")
        return height, width

    @staticmethod
    def _detector_boxes(raw_detections, width, height):
        if not isinstance(raw_detections, (list, tuple)):
            raise PoseExtractionError("detector output: must be a list of detections")
        boxes, scores = [], []
        for index, detection in enumerate(raw_detections):
            field = f"detector output {index}"
            if not isinstance(detection, dict):
                raise PoseExtractionError(f"{field}: each detection must be a mapping")
            if "box_xyxy" not in detection:
                raise PoseExtractionError(f"{field}: missing box_xyxy")
            if "score_raw" not in detection:
                raise PoseExtractionError(f"{field}: missing score_raw")
            boxes.append(_validate_box(detection["box_xyxy"], field, width, height))
            score = detection["score_raw"]
            if not _is_number(score):
                raise PoseExtractionError(f"{field}: score_raw must be a finite number")
            scores.append(_as_python_number(score))
        return boxes, scores

    @staticmethod
    def _reviewed_boxes(reviewed_boxes, width, height):
        if not isinstance(reviewed_boxes, (list, tuple)):
            raise PoseExtractionError("reviewed_boxes: must be a list of boxes")
        boxes = []
        for index, item in enumerate(reviewed_boxes):
            field = f"reviewed_boxes {index}"
            raw_box = item.get("box_xyxy") if isinstance(item, dict) else item
            if isinstance(item, dict) and "box_xyxy" not in item:
                raise PoseExtractionError(f"{field}: missing box_xyxy")
            boxes.append(_validate_box(raw_box, field, width, height))
        return boxes, [None] * len(boxes)

    # -- estimator output -------------------------------------------------

    @staticmethod
    def _validate_estimator_results(raw_results, expected_count):
        if not isinstance(raw_results, (list, tuple)):
            raise PoseExtractionError("estimator output: must be a list of results")
        if len(raw_results) != expected_count:
            raise PoseExtractionError(
                f"estimator output: expected one result per box "
                f"({expected_count}), got {len(raw_results)}"
            )
        validated = []
        for index, result in enumerate(raw_results):
            if not isinstance(result, dict):
                raise PoseExtractionError(
                    f"estimator output {index}: each result must be a mapping"
                )
            for key in ("joint_names", "xy", "scores_raw", "coordinate_space", "to_original"):
                if key not in result:
                    raise PoseExtractionError(f"estimator output {index}: missing {key}")
            names = _validate_joint_names(result["joint_names"])
            xy = _validate_xy(result["xy"])
            scores = _validate_scores(result["scores_raw"])
            coordinate_space = result["coordinate_space"]
            if coordinate_space not in ("original_pixels", "model_pixels"):
                raise PoseExtractionError(
                    "coordinate_space: must be 'original_pixels' or 'model_pixels'"
                )
            to_original = _validate_transform(result["to_original"])
            if coordinate_space == "model_pixels" and to_original is None:
                raise PoseExtractionError(
                    "to_original: a transform is required for model_pixels results"
                )
            if coordinate_space == "original_pixels" and to_original is not None:
                raise PoseExtractionError(
                    "to_original: must be null when coordinate_space is "
                    "'original_pixels'"
                )
            validated.append(
                {
                    "joint_names": names,
                    "xy": xy,
                    "scores_raw": scores,
                    "coordinate_space": coordinate_space,
                    "to_original": to_original,
                }
            )
        return validated

    @staticmethod
    def _build_candidate(box, score, box_source, pose, width, height):
        positions = {name: index for index, name in enumerate(pose["joint_names"])}
        xy_pixels = []
        joint_scores = []
        invalid_reasons = []
        for name in CANONICAL_JOINT_NAMES:
            source_index = positions[name]
            joint_scores.append(deepcopy(pose["scores_raw"][source_index]))
            point = pose["xy"][source_index]
            if point is None:
                xy_pixels.append(None)
                invalid_reasons.append("model_missing")
                continue
            x, y = point
            if pose["coordinate_space"] == "model_pixels":
                (a, b, c), (d, e, f) = pose["to_original"]
                new_x = a * x + b * y + c
                new_y = d * x + e * y + f
                try:
                    finite = math.isfinite(float(new_x)) and math.isfinite(float(new_y))
                except (OverflowError, ValueError):
                    finite = False
                if not finite:
                    raise PoseExtractionError(
                        "xy: transformed coordinates must be finite"
                    )
                x, y = new_x, new_y
            if 0 <= x <= width - 1 and 0 <= y <= height - 1:
                xy_pixels.append([x, y])
                invalid_reasons.append(None)
            else:
                xy_pixels.append(None)
                invalid_reasons.append("outside_frame")
        return {
            "box_xyxy": list(box),
            "box_source": box_source,
            "detector_score_raw": deepcopy(score),
            "joint_names": list(CANONICAL_JOINT_NAMES),
            "xy_pixels": xy_pixels,
            "joint_scores_raw": joint_scores,
            "invalid_reasons": invalid_reasons,
            "diagnostics": {
                "source_joint_names": list(pose["joint_names"]),
                "coordinate_space_input": pose["coordinate_space"],
                "to_original": deepcopy(pose["to_original"]),
                "raw_box_xyxy": list(box),
            },
        }

    # -- ordering ---------------------------------------------------------
    # Python's sort is stable, so equal keys keep the original backend
    # order, which is exactly the final tie-break the spec requires.

    @staticmethod
    def _sort_candidates(built, box_source):
        if box_source == "detector":
            ordered = sorted(
                built,
                key=lambda candidate: (
                    -candidate["detector_score_raw"],
                    candidate["box_xyxy"],
                ),
            )
        else:
            ordered = sorted(built, key=lambda candidate: candidate["box_xyxy"])
        return [
            {"candidate_id": f"candidate_{index}", **candidate}
            for index, candidate in enumerate(ordered)
        ]
