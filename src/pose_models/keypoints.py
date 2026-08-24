"""Canonical Body-18 contract used by pose preparation and inference."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


# This is the ControlNet/OpenPose body ordering after the DWPose conversion.
BODY_18_JOINT_NAMES: tuple[str, ...] = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
)

BODY_18_INDEX = {name: index for index, name in enumerate(BODY_18_JOINT_NAMES)}

# Each entry is a zero-based pair in BODY_18_JOINT_NAMES order.
BODY_18_LIMBS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
)

# Official ControlNet OpenPose palette, represented explicitly as RGB.
BODY_18_RGB_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
)

DEFAULT_CONFIDENCE_THRESHOLD = 0.30


def validate_body_18_keypoints(
    keypoints: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Validate normalized Body-18 ``[joint, (x, y, confidence)]`` data."""

    array = np.asarray(keypoints, dtype=np.float32)
    if array.shape != (len(BODY_18_JOINT_NAMES), 3):
        raise ValueError(
            "expected Body-18 keypoints with shape "
            f"({len(BODY_18_JOINT_NAMES)}, 3), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("keypoints contain NaN or Inf")
    if (array[:, :2] < 0.0).any() or (array[:, :2] > 1.0).any():
        raise ValueError("normalized keypoint coordinates must be in [0, 1]")
    if (array[:, 2] < 0.0).any() or (array[:, 2] > 1.0).any():
        raise ValueError("keypoint confidences must be in [0, 1]")
    return array


def body_bounding_box(
    keypoints: np.ndarray | Sequence[Sequence[float]],
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> tuple[float, float, float, float] | None:
    """Return ``(left, top, right, bottom)`` for confident Body-18 joints."""

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    array = validate_body_18_keypoints(keypoints)
    visible = array[:, 2] >= confidence_threshold
    if not visible.any():
        return None
    coordinates = array[visible, :2]
    left, top = coordinates.min(axis=0)
    right, bottom = coordinates.max(axis=0)
    return float(left), float(top), float(right), float(bottom)


def sort_people_deterministically(
    people: Iterable[np.ndarray | Sequence[Sequence[float]]],
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[np.ndarray]:
    """Sort valid people by center, then decreasing box area, deterministically."""

    sortable: list[tuple[tuple[float, float, float, int], np.ndarray]] = []
    for original_index, person in enumerate(people):
        array = validate_body_18_keypoints(person)
        bbox = body_bounding_box(array, confidence_threshold=confidence_threshold)
        if bbox is None:
            center_x = center_y = area = 0.0
        else:
            left, top, right, bottom = bbox
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            area = (right - left) * (bottom - top)
        sortable.append(((center_x, center_y, -area, original_index), array))
    return [person for _, person in sorted(sortable, key=lambda item: item[0])]
