"""ONNX Runtime backends for the official DWPose detector and pose model.

This module is deliberately separate from the NumPy-only frame wrapper. Importing
it does not load ONNX Runtime, OpenCV, or model files. Constructing a backend is
explicit and verifies both model files against caller-supplied SHA-256 pins.

The preprocessing and SimCC decoding follow the public DWPose ONNX implementation
from IDEA-Research/DWPose. This adapter returns COCO-17 body joints before
whole-body neck insertion, OpenPose reordering, confidence thresholding, or drawing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


class DWPoseONNXError(RuntimeError):
    """Raised when DWPose model setup or inference fails."""


_COCO17_JOINT_NAMES = (
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
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGENET_RGB_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
_IMAGENET_RGB_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


@dataclass(frozen=True)
class DWPoseONNXConfig:
    """Pinned files and explicit runtime choices for one DWPose installation.

    ``providers`` is ordered as ONNX Runtime's preferred provider list. For CUDA,
    pass an explicit ``provider_options["CUDAExecutionProvider"]["device_id"]``;
    this avoids silently choosing a GPU. CPU-only use is supported by selecting
    ``("CPUExecutionProvider",)``.
    """

    detector_path: Path | str
    pose_path: Path | str
    detector_sha256: str
    pose_sha256: str
    expected_onnxruntime_version: str
    expected_opencv_version: str
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    provider_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    detector_input_size: tuple[int, int] = (640, 640)  # height, width
    detection_score_threshold: float = 0.1
    final_person_score_threshold: float = 0.3
    nms_iou_threshold: float = 0.45
    pose_padding_factor: float = 1.25
    simcc_split_ratio: float = 2.0

    def __post_init__(self) -> None:
        for name, digest in (
            ("detector_sha256", self.detector_sha256),
            ("pose_sha256", self.pose_sha256),
        ):
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
        for name in ("expected_onnxruntime_version", "expected_opencv_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be an explicit version string")
        if not self.providers or any(not isinstance(p, str) or not p for p in self.providers):
            raise ValueError("providers must be a non-empty tuple of provider names")
        if len(set(self.providers)) != len(self.providers):
            raise ValueError("providers must not contain duplicates")
        if len(self.detector_input_size) != 2 or any(
            not isinstance(v, int) or isinstance(v, bool) or v <= 0
            for v in self.detector_input_size
        ):
            raise ValueError("detector_input_size must be positive integer (height, width)")
        if any(value % 32 for value in self.detector_input_size):
            raise ValueError("detector_input_size dimensions must be multiples of 32")
        for name in (
            "detection_score_threshold",
            "final_person_score_threshold",
            "nms_iou_threshold",
            "pose_padding_factor",
            "simcc_split_ratio",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        if not 0.0 <= self.detection_score_threshold <= 1.0:
            raise ValueError("detection_score_threshold must be in [0, 1]")
        if not 0.0 <= self.final_person_score_threshold <= 1.0:
            raise ValueError("final_person_score_threshold must be in [0, 1]")
        if not 0.0 <= self.nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be in [0, 1]")
        if self.pose_padding_factor <= 0 or self.simcc_split_ratio <= 0:
            raise ValueError("pose_padding_factor and simcc_split_ratio must be positive")
        if "CUDAExecutionProvider" in self.providers:
            cuda_options = self.provider_options.get("CUDAExecutionProvider", {})
            if "device_id" not in cuda_options:
                raise ValueError(
                    "set provider_options['CUDAExecutionProvider']['device_id'] explicitly"
                )
            device_id = cuda_options["device_id"]
            if not isinstance(device_id, int) or isinstance(device_id, bool) or device_id < 0:
                raise ValueError("CUDA provider device_id must be a non-negative integer")


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 of a model file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_model_file(path: Path | str, expected_sha256: str, label: str) -> str:
    model_path = Path(path).expanduser()
    if not model_path.is_file():
        raise DWPoseONNXError(f"{label} model file not found: {model_path}")
    actual = sha256_file(model_path)
    if actual != expected_sha256:
        raise DWPoseONNXError(
            f"{label} model SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return actual


def _provider_session_options(config: DWPoseONNXConfig) -> list[tuple[str, dict[str, Any]]]:
    options: list[tuple[str, dict[str, Any]]] = []
    for provider in config.providers:
        provider_options = dict(config.provider_options.get(provider, {}))
        options.append((provider, provider_options))
    return options


def _model_manifest(
    config: DWPoseONNXConfig,
    runtime_version: str,
    opencv_version: str,
    pose_input_size_hw: tuple[int, int],
    pose_output_names: Sequence[str],
) -> dict[str, Any]:
    settings = {
        "implementation": "dwpose-onnx-co-co17-v1",
        "upstream": "https://github.com/IDEA-Research/DWPose",
        "detector": {
            "filename": Path(config.detector_path).name,
            "sha256": config.detector_sha256,
            "input_size_hw": list(config.detector_input_size),
            "score_threshold": float(config.detection_score_threshold),
            "person_score_threshold": float(config.final_person_score_threshold),
            "nms_iou_threshold": float(config.nms_iou_threshold),
        },
        "pose": {
            "filename": Path(config.pose_path).name,
            "sha256": config.pose_sha256,
            "input_size_hw": list(pose_input_size_hw),
            "output_names": list(pose_output_names),
            "padding_factor": float(config.pose_padding_factor),
            "simcc_split_ratio": float(config.simcc_split_ratio),
            "input_color_order": "RGB",
            "normalization_mean": _IMAGENET_RGB_MEAN.tolist(),
            "normalization_std": _IMAGENET_RGB_STD.tolist(),
            "body_joint_mapping": list(_COCO17_JOINT_NAMES),
            "missing_point_rule": "non-positive SimCC maximum or negative decoded coordinate",
        },
        "runtime": {
            "onnxruntime_version": runtime_version,
            "opencv_version": opencv_version,
            "requested_providers": list(config.providers),
            "provider_options": {
                name: dict(options) for name, options in config.provider_options.items()
            },
        },
    }
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**settings, "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def _require_cv2(cv2_module: Any | None = None) -> Any:
    if cv2_module is not None:
        return cv2_module
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise DWPoseONNXError(
            "OpenCV is required for DWPose preprocessing; install opencv-python-headless"
        ) from error
    return cv2


def _validate_rgb_image(image: np.ndarray) -> tuple[int, int]:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise DWPoseONNXError("image must be a NumPy RGB array with shape (height, width, 3)")
    if image.dtype != np.uint8:
        raise DWPoseONNXError("image must have dtype uint8")
    height, width = (int(image.shape[0]), int(image.shape[1]))
    if height < 2 or width < 2:
        raise DWPoseONNXError("image height and width must each be at least 2")
    return height, width


def _letterbox_top_left(
    image: np.ndarray, input_size_hw: tuple[int, int], cv2_module: Any
) -> tuple[np.ndarray, float]:
    """Apply the DWPose YOLOX 114-padding and top-left letterbox preprocessing."""

    height, width = image.shape[:2]
    target_h, target_w = input_size_hw
    ratio = min(target_h / height, target_w / width)
    resized_w = max(1, int(width * ratio))
    resized_h = max(1, int(height * ratio))
    resized = cv2_module.resize(
        image, (resized_w, resized_h), interpolation=cv2_module.INTER_LINEAR
    )
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[:resized_h, :resized_w] = resized
    chw = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32)
    return chw[None, ...], ratio


def _yolox_grid(input_size_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    strides = (8, 16, 32)
    height, width = input_size_hw
    grids: list[np.ndarray] = []
    expanded_strides: list[np.ndarray] = []
    for stride in strides:
        hsize, wsize = height // stride, width // stride
        if hsize <= 0 or wsize <= 0:
            raise DWPoseONNXError("YOLOX input size must be at least 32x32")
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((1, grid.shape[1], 1), stride))
    return np.concatenate(grids, axis=1), np.concatenate(expanded_strides, axis=1)


def _decode_yolox_output(
    output: np.ndarray, input_size_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Decode YOLOX center/size outputs into corner boxes and class scores."""

    predictions = np.asarray(output)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.ndim != 2 or predictions.shape[1] < 6:
        raise DWPoseONNXError(
            f"YOLOX output must have shape (candidates, 5+classes), got {predictions.shape}"
        )
    grids, strides = _yolox_grid(input_size_hw)
    if predictions.shape[0] != grids.shape[1]:
        raise DWPoseONNXError(
            f"YOLOX output has {predictions.shape[0]} candidates; expected {grids.shape[1]} "
            f"for input size {input_size_hw}"
        )
    decoded = predictions.astype(np.float32, copy=True)
    decoded[..., :2] = (decoded[..., :2] + grids) * strides
    try:
        decoded[..., 2:4] = np.exp(decoded[..., 2:4]) * strides
    except FloatingPointError as error:
        raise DWPoseONNXError("YOLOX width/height decoding overflowed") from error
    centers = decoded[:, :2]
    sizes = decoded[:, 2:4]
    boxes = np.concatenate((centers - sizes / 2.0, centers + sizes / 2.0), axis=1)
    scores = decoded[:, 4:5] * decoded[:, 5:]
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise DWPoseONNXError("YOLOX output contains non-finite decoded values")
    return boxes, scores


def _nms_indices(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Single-class NMS with the inclusive-pixel area convention used upstream."""

    if boxes.size == 0:
        return []
    x0, y0, x1, y1 = (boxes[:, index] for index in range(4))
    areas = np.maximum(0.0, x1 - x0 + 1.0) * np.maximum(0.0, y1 - y0 + 1.0)
    order = np.argsort(scores, kind="stable")[::-1]
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        remaining = order[1:]
        if remaining.size == 0:
            break
        ix0 = np.maximum(x0[current], x0[remaining])
        iy0 = np.maximum(y0[current], y0[remaining])
        ix1 = np.minimum(x1[current], x1[remaining])
        iy1 = np.minimum(y1[current], y1[remaining])
        intersection = (
            np.maximum(0.0, ix1 - ix0 + 1.0)
            * np.maximum(0.0, iy1 - iy0 + 1.0)
        )
        denominator = areas[current] + areas[remaining] - intersection
        overlap = np.divide(
            intersection,
            denominator,
            out=np.zeros_like(intersection),
            where=denominator > 0,
        )
        order = remaining[overlap <= threshold]
    return kept


class DWPoseONNXDetector:
    """YOLOX person detector implementing ``detect(image_rgb)`` for the wrapper."""

    def __init__(
        self,
        session: Any,
        input_size_hw: tuple[int, int] = (640, 640),
        score_threshold: float = 0.1,
        final_person_score_threshold: float = 0.3,
        nms_iou_threshold: float = 0.45,
        cv2_module: Any | None = None,
    ) -> None:
        self.session = session
        self.input_size_hw = input_size_hw
        self.score_threshold = float(score_threshold)
        self.final_person_score_threshold = float(final_person_score_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.cv2 = _require_cv2(cv2_module)
        inputs = session.get_inputs()
        if not inputs:
            raise DWPoseONNXError("detector model has no input")
        self.input_name = inputs[0].name
        self.output_names = [item.name for item in session.get_outputs()]

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        _validate_rgb_image(image)
        tensor, ratio = _letterbox_top_left(image, self.input_size_hw, self.cv2)
        try:
            outputs = self.session.run(self.output_names or None, {self.input_name: tensor})
        except Exception as error:
            raise DWPoseONNXError(f"YOLOX ONNX inference failed: {error}") from error
        if not outputs:
            raise DWPoseONNXError("YOLOX model returned no output tensors")
        boxes, class_scores = _decode_yolox_output(outputs[0], self.input_size_hw)
        person_scores = class_scores[:, 0]
        selected = person_scores > self.score_threshold
        boxes = boxes[selected]
        person_scores = person_scores[selected]
        kept = _nms_indices(boxes, person_scores, self.nms_iou_threshold)
        results: list[dict[str, Any]] = []
        for index in kept:
            score = float(person_scores[index])
            if score <= self.final_person_score_threshold:
                continue
            box = (boxes[index] / ratio).astype(np.float64)
            results.append(
                {"box_xyxy": [float(value) for value in box], "score_raw": score}
            )
        return results


def _bbox_center_scale(
    box: Sequence[float], input_size_wh: tuple[int, int], padding: float
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = (float(value) for value in box)
    center = np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)
    scale = np.asarray([x1 - x0, y1 - y0], dtype=np.float32) * padding
    input_w, input_h = input_size_wh
    aspect_ratio = input_w / input_h
    if scale[0] > scale[1] * aspect_ratio:
        scale[1] = scale[0] / aspect_ratio
    else:
        scale[0] = scale[1] * aspect_ratio
    return center, scale


def _crop_affine(
    image: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    input_size_wh: tuple[int, int],
    cv2_module: Any,
) -> np.ndarray:
    input_w, input_h = input_size_wh
    transform = np.asarray(
        [
            [input_w / scale[0], 0.0, input_w * 0.5 - center[0] * input_w / scale[0]],
            [0.0, input_h / scale[1], input_h * 0.5 - center[1] * input_h / scale[1]],
        ],
        dtype=np.float32,
    )
    return cv2_module.warpAffine(
        image,
        transform,
        (input_w, input_h),
        flags=cv2_module.INTER_LINEAR,
    )


def _decode_simcc(
    simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one batch of SimCC vectors into crop coordinates and raw scores."""

    x_dist = np.asarray(simcc_x)
    y_dist = np.asarray(simcc_y)
    if x_dist.ndim == 3 and x_dist.shape[0] == 1:
        x_dist = x_dist[0]
    if y_dist.ndim == 3 and y_dist.shape[0] == 1:
        y_dist = y_dist[0]
    if x_dist.ndim != 2 or y_dist.ndim != 2 or x_dist.shape[0] != y_dist.shape[0]:
        raise DWPoseONNXError(
            f"SimCC outputs must be (joints, bins) arrays; got {x_dist.shape} and {y_dist.shape}"
        )
    x_indices = np.argmax(x_dist, axis=1)
    y_indices = np.argmax(y_dist, axis=1)
    x_max = np.max(x_dist, axis=1)
    y_max = np.max(y_dist, axis=1)
    scores = np.minimum(x_max, y_max).astype(np.float64)
    points = np.stack((x_indices, y_indices), axis=1).astype(np.float64) / float(split_ratio)
    missing = scores <= 0.0
    points[missing] = -1.0
    if not np.isfinite(points).all() or not np.isfinite(scores).all():
        raise DWPoseONNXError("SimCC output contains non-finite values")
    return points, scores


class DWPoseONNXEstimator:
    """RTMPose/SimCC estimator implementing ``estimate(image_rgb, boxes)``."""

    def __init__(
        self,
        session: Any,
        padding_factor: float = 1.25,
        split_ratio: float = 2.0,
        cv2_module: Any | None = None,
    ) -> None:
        self.session = session
        self.padding_factor = float(padding_factor)
        self.split_ratio = float(split_ratio)
        self.cv2 = _require_cv2(cv2_module)
        inputs = session.get_inputs()
        if not inputs:
            raise DWPoseONNXError("pose model has no input")
        self.input_name = inputs[0].name
        input_shape = inputs[0].shape
        if len(input_shape) != 4 or not all(
            isinstance(value, int) and value > 0 for value in input_shape[2:]
        ):
            raise DWPoseONNXError(
                f"pose model must declare static NCHW spatial dimensions; got {input_shape}"
            )
        self.input_h, self.input_w = int(input_shape[2]), int(input_shape[3])
        if input_shape[1] != 3:
            raise DWPoseONNXError(f"pose model must have 3 input channels; got {input_shape}")
        outputs = session.get_outputs()
        if len(outputs) < 2:
            raise DWPoseONNXError("pose model must return SimCC x and y outputs")
        output_names = [item.name for item in outputs]
        if "simcc_x" in output_names and "simcc_y" in output_names:
            self.output_names = ["simcc_x", "simcc_y"]
        elif len(output_names) == 2:
            self.output_names = output_names
        else:
            raise DWPoseONNXError(
                "pose model outputs must be named simcc_x/simcc_y or contain exactly two outputs"
            )
        self.all_output_names = output_names

    def estimate(
        self, image: np.ndarray, boxes: Sequence[Sequence[float]]
    ) -> list[dict[str, Any]]:
        _validate_rgb_image(image)
        results: list[dict[str, Any]] = []
        for box in boxes:
            center, scale = _bbox_center_scale(
                box, (self.input_w, self.input_h), self.padding_factor
            )
            crop = _crop_affine(
                image, center, scale, (self.input_w, self.input_h), self.cv2
            ).astype(np.float32)
            normalized = (crop - _IMAGENET_RGB_MEAN) / _IMAGENET_RGB_STD
            tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])
            try:
                outputs = self.session.run(
                    self.output_names, {self.input_name: tensor}
                )
            except Exception as error:
                raise DWPoseONNXError(f"pose ONNX inference failed: {error}") from error
            if len(outputs) < 2:
                raise DWPoseONNXError("pose model returned fewer than two SimCC tensors")
            crop_points, scores = _decode_simcc(
                outputs[0], outputs[1], self.split_ratio
            )
            if crop_points.shape[0] != 133:
                raise DWPoseONNXError(
                    f"pose model returned {crop_points.shape[0]} joints; "
                    "the pinned DWPose whole-body model must return 133"
                )

            # DWPose's ONNX decoder emits COCO-WholeBody order. The first 17
            # points are the COCO body joints in the project's named order.
            xy: list[list[float] | None] = []
            raw_scores: list[float] = []
            for point, score in zip(crop_points[:17], scores[:17]):
                raw_score = float(score)
                raw_scores.append(raw_score)
                if raw_score <= 0.0 or point[0] < 0.0 or point[1] < 0.0:
                    xy.append(None)
                    continue
                original_x = point[0] / self.input_w * scale[0] + center[0] - scale[0] / 2.0
                original_y = point[1] / self.input_h * scale[1] + center[1] - scale[1] / 2.0
                xy.append([float(original_x), float(original_y)])
            results.append(
                {
                    "joint_names": list(_COCO17_JOINT_NAMES),
                    "xy": xy,
                    "scores_raw": raw_scores,
                    "coordinate_space": "original_pixels",
                    "to_original": None,
                }
            )
        return results


def create_dwpose_onnx_extractor(
    config: DWPoseONNXConfig,
    *,
    wrapper_class: type | None = None,
    runtime: Any | None = None,
    cv2_module: Any | None = None,
) -> Any:
    """Verify pins, load ONNX sessions, and build the frame wrapper explicitly.

    ``wrapper_class`` and ``runtime`` are injectable for tests. In normal use,
    the wrapper import occurs only when this function is called, so importing
    the package does not import DWPose, ONNX Runtime, or OpenCV.
    """

    if wrapper_class is None:
        try:
            from .dwpose import DWPoseFrameExtractor
        except ImportError as error:
            raise DWPoseONNXError(
                "the DWPose frame wrapper is not installed yet; merge its PR or pass wrapper_class"
            ) from error
        wrapper_class = DWPoseFrameExtractor
    if runtime is None:
        try:
            import onnxruntime as runtime  # type: ignore[no-redef,import-not-found]
        except ImportError as error:
            raise DWPoseONNXError(
                "ONNX Runtime is required; install the runtime matching the selected provider"
            ) from error
    cv2_instance = _require_cv2(cv2_module)
    runtime_version = str(runtime.__version__)
    if runtime_version != config.expected_onnxruntime_version:
        raise DWPoseONNXError(
            "ONNX Runtime version mismatch: expected "
            f"{config.expected_onnxruntime_version}, got {runtime_version}"
        )
    cv2_version = str(getattr(cv2_instance, "__version__", ""))
    if cv2_version != config.expected_opencv_version:
        raise DWPoseONNXError(
            "OpenCV version mismatch: expected "
            f"{config.expected_opencv_version}, got {cv2_version or 'unknown'}"
        )
    detector_hash = _check_model_file(
        config.detector_path, config.detector_sha256, "detector"
    )
    pose_hash = _check_model_file(config.pose_path, config.pose_sha256, "pose estimator")
    available = set(runtime.get_available_providers())
    unavailable = [provider for provider in config.providers if provider not in available]
    if unavailable:
        raise DWPoseONNXError(
            f"requested ONNX Runtime provider(s) unavailable: {unavailable}; "
            f"available providers: {sorted(available)}"
        )
    providers = _provider_session_options(config)
    try:
        detector_session = runtime.InferenceSession(
            str(config.detector_path), providers=providers
        )
        pose_session = runtime.InferenceSession(str(config.pose_path), providers=providers)
    except Exception as error:
        raise DWPoseONNXError(f"failed to create ONNX Runtime sessions: {error}") from error

    detector = DWPoseONNXDetector(
        detector_session,
        input_size_hw=config.detector_input_size,
        score_threshold=config.detection_score_threshold,
        final_person_score_threshold=config.final_person_score_threshold,
        nms_iou_threshold=config.nms_iou_threshold,
        cv2_module=cv2_instance,
    )
    estimator = DWPoseONNXEstimator(
        pose_session,
        padding_factor=config.pose_padding_factor,
        split_ratio=config.simcc_split_ratio,
        cv2_module=cv2_instance,
    )
    model_info = _model_manifest(
        config,
        runtime_version,
        cv2_version,
        (estimator.input_h, estimator.input_w),
        estimator.all_output_names,
    )
    model_info["detector"]["sha256"] = detector_hash
    model_info["pose"]["sha256"] = pose_hash
    detector_session_providers = detector_session.get_providers()
    pose_session_providers = pose_session.get_providers()
    if config.providers[0] not in detector_session_providers:
        raise DWPoseONNXError(
            f"detector session did not activate the requested provider {config.providers[0]}"
        )
    if config.providers[0] not in pose_session_providers:
        raise DWPoseONNXError(
            f"pose session did not activate the requested provider {config.providers[0]}"
        )
    model_info["runtime"]["detector_session_providers"] = detector_session_providers
    model_info["runtime"]["pose_session_providers"] = pose_session_providers
    return wrapper_class(detector=detector, estimator=estimator, model_info=model_info)
