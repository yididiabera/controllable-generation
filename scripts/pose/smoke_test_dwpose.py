#!/usr/bin/env python3
"""Run DWPose on one image or source-video frame and save inspectable outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_COCO17_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one real DWPose frame through the frame wrapper. "
            "Choose exactly one input image or source video frame."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="path to one decoded frame image")
    source.add_argument("--video", type=Path, help="path to a source video")
    parser.add_argument(
        "--frame-index",
        type=int,
        help="zero-based source-video frame index; required with --video",
    )
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--detector-sha256", required=True)
    parser.add_argument("--pose-sha256", required=True)
    parser.add_argument("--onnxruntime-version", required=True)
    parser.add_argument("--opencv-version", required=True)
    parser.add_argument(
        "--provider",
        action="append",
        required=True,
        help="ONNX Runtime provider in priority order; repeat for a fallback",
    )
    parser.add_argument(
        "--cuda-device-id",
        type=int,
        help="explicit CUDA device id when CUDAExecutionProvider is selected",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dwpose-smoke"))
    parser.add_argument("--detector-size", type=int, default=640)
    return parser.parse_args()


def _read_rgb_frame(args: argparse.Namespace, cv2: Any) -> tuple[Any, dict[str, Any]]:
    if args.video is not None:
        if args.frame_index is None or args.frame_index < 0:
            raise ValueError("--frame-index must be a non-negative integer with --video")
        capture = cv2.VideoCapture(str(args.video))
        try:
            if not capture.isOpened():
                raise ValueError(f"could not open video: {args.video}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
            ok, bgr = capture.read()
        finally:
            capture.release()
        if not ok or bgr is None:
            raise ValueError(
                f"could not decode frame {args.frame_index} from {args.video}"
            )
        source_info = {
            "kind": "video_frame",
            "path": str(args.video.resolve()),
            "source_frame_index": args.frame_index,
        }
    else:
        if args.frame_index is not None:
            raise ValueError("--frame-index can only be used with --video")
        bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not read image: {args.image}")
        source_info = {"kind": "image", "path": str(args.image.resolve())}

    # OpenCV decoders return BGR. Convert once at the source boundary because
    # the wrapper and the DWPose ONNX preprocessing here use RGB arrays.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    source_info["width"] = int(rgb.shape[1])
    source_info["height"] = int(rgb.shape[0])
    return rgb, source_info


def _draw_overlay(rgb: Any, result: dict[str, Any], cv2: Any) -> Any:
    overlay = rgb.copy()
    for candidate in result["candidates"]:
        box = candidate["box_xyxy"]
        box_start = (round(box[0]), round(box[1]))
        box_end = (round(box[2]), round(box[3]))
        color = (255, 190, 0)
        cv2.rectangle(overlay, box_start, box_end, color, 2)
        cv2.putText(
            overlay,
            f"{candidate['candidate_id']} score={candidate['detector_score_raw']}",
            (box_start[0], max(14, box_start[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        points = candidate["xy_pixels"]
        for first, second in _COCO17_EDGES:
            if points[first] is not None and points[second] is not None:
                cv2.line(
                    overlay,
                    tuple(round(value) for value in points[first]),
                    tuple(round(value) for value in points[second]),
                    (0, 220, 80),
                    2,
                    cv2.LINE_AA,
                )
        for point_index, point in enumerate(points):
            if point is not None:
                position = tuple(round(value) for value in point)
                cv2.circle(overlay, position, 3, (255, 40, 40), -1, cv2.LINE_AA)
                cv2.putText(
                    overlay,
                    str(point_index),
                    (position[0] + 3, position[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
    return overlay


def main() -> int:
    args = _parse_args()
    if args.video is not None and args.frame_index is None:
        raise SystemExit("--frame-index is required with --video")
    if args.video is None and args.frame_index is not None:
        raise SystemExit("--frame-index can only be used with --video")

    try:
        import cv2
        import onnxruntime
    except ImportError as error:
        raise SystemExit(
            "This smoke test needs OpenCV and ONNX Runtime in the selected server environment. "
            f"Import failed: {error}"
        ) from error

    from src.pose_models.extraction.dwpose import DWPoseFrameExtractor
    from src.pose_models.extraction.dwpose_onnx import (
        DWPoseONNXConfig,
        create_dwpose_onnx_extractor,
    )

    provider_options = {}
    if args.cuda_device_id is not None:
        provider_options["CUDAExecutionProvider"] = {"device_id": args.cuda_device_id}
    config = DWPoseONNXConfig(
        detector_path=args.detector_model,
        pose_path=args.pose_model,
        detector_sha256=args.detector_sha256,
        pose_sha256=args.pose_sha256,
        expected_onnxruntime_version=args.onnxruntime_version,
        expected_opencv_version=args.opencv_version,
        providers=tuple(args.provider),
        provider_options=provider_options,
        detector_input_size=(args.detector_size, args.detector_size),
    )
    extractor = create_dwpose_onnx_extractor(config, wrapper_class=DWPoseFrameExtractor)
    rgb, source_info = _read_rgb_frame(args, cv2)
    result = extractor.extract_frame(rgb)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_rgb = _draw_overlay(rgb, result, cv2)
    overlay_path = args.output_dir / "pose_overlay.png"
    result_path = args.output_dir / "frame_result.json"
    if not cv2.imwrite(
        str(overlay_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    ):
        raise SystemExit(f"could not write overlay: {overlay_path}")
    payload = {
        "source": source_info,
        "result": result,
        "overlay_path": str(overlay_path.resolve()),
    }
    result_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print(f"status: {result['status']}")
    print(f"candidates: {len(result['candidates'])}")
    for candidate in result["candidates"]:
        visible = sum(point is not None for point in candidate["xy_pixels"])
        print(
            f"{candidate['candidate_id']}: {visible}/17 joints visible; "
            f"raw detector score={candidate['detector_score_raw']}"
        )
    print(f"result: {result_path.resolve()}")
    print(f"overlay: {overlay_path.resolve()}")
    print(f"config SHA-256: {result['model_info']['config_sha256']}")
    print(f"active providers: {result['model_info']['runtime']['detector_session_providers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
