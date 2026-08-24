"""Deterministic, non-antialiased RGB renderer for Body-18 skeleton controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .keypoints import (
    BODY_18_LIMBS,
    BODY_18_RGB_COLORS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    sort_people_deterministically,
)


@dataclass(frozen=True)
class PoseRenderConfig:
    """Frozen renderer settings for one pose-preprocessing version."""

    output_size: tuple[int, int] = (128, 128)
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    joint_radius: int = 3
    limb_thickness: int = 2

    def __post_init__(self) -> None:
        if len(self.output_size) != 2 or any(value <= 0 for value in self.output_size):
            raise ValueError("output_size must contain two positive integers")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.joint_radius <= 0:
            raise ValueError("joint_radius must be positive")
        if self.limb_thickness <= 0:
            raise ValueError("limb_thickness must be positive")


def _pixel_coordinate(value: float, size: int) -> int:
    """Map normalized [0, 1] coordinates to valid output pixel positions."""

    return int(np.rint(value * (size - 1)))


def _draw_disk(
    canvas: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    """Draw an opaque, non-antialiased disk directly into an RGB canvas."""

    center_x, center_y = center
    output_h, output_w = canvas.shape[:2]
    left = max(0, center_x - radius)
    right = min(output_w - 1, center_x + radius)
    top = max(0, center_y - radius)
    bottom = min(output_h - 1, center_y + radius)
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2:
                canvas[y, x] = color


def _draw_line(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    thickness: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a deterministic opaque line using disk stamps along the segment."""

    start_x, start_y = start
    end_x, end_y = end
    steps = max(abs(end_x - start_x), abs(end_y - start_y), 1)
    radius = max(0, (thickness - 1) // 2)
    for step in range(steps + 1):
        fraction = step / steps
        point = (
            int(np.rint(start_x + fraction * (end_x - start_x))),
            int(np.rint(start_y + fraction * (end_y - start_y))),
        )
        _draw_disk(canvas, point, radius, color)


def render_pose(
    people: Iterable[np.ndarray | Sequence[Sequence[float]]],
    config: PoseRenderConfig = PoseRenderConfig(),
) -> np.ndarray:
    """Render every confident person as a canonical RGB ``[H,W,3]`` uint8 map."""

    output_h, output_w = config.output_size
    canvas = np.zeros((output_h, output_w, 3), dtype=np.uint8)
    people_sorted = sort_people_deterministically(
        people,
        confidence_threshold=config.confidence_threshold,
    )

    for person in people_sorted:
        valid = person[:, 2] >= config.confidence_threshold
        pixels = np.empty((len(person), 2), dtype=np.int32)
        for joint_index, (x, y, _) in enumerate(person):
            pixels[joint_index] = (
                _pixel_coordinate(float(x), output_w),
                _pixel_coordinate(float(y), output_h),
            )

        for limb_index, (first, second) in enumerate(BODY_18_LIMBS):
            if not (valid[first] and valid[second]):
                continue
            _draw_line(
                canvas,
                tuple(pixels[first]),
                tuple(pixels[second]),
                config.limb_thickness,
                BODY_18_RGB_COLORS[limb_index],
            )

        for joint_index, pixel in enumerate(pixels):
            if not valid[joint_index]:
                continue
            _draw_disk(
                canvas,
                tuple(pixel),
                config.joint_radius,
                BODY_18_RGB_COLORS[joint_index],
            )

    return canvas
