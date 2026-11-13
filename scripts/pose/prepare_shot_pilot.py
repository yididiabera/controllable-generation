#!/usr/bin/env python3
"""Create one annotated-shot pose-map fixture for a full loader diagnostic.

The output contains all detected people and uncalibrated in-frame keypoints.
It is not a validated pose record or approved training control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.data.frame_sampling import resolve_frame_interval, select_frame_indices
from src.pose_models.extraction.dwpose_onnx import DWPoseONNXConfig, create_dwpose_onnx_extractor
from scripts.pose.prepare_video_pilot import build_sequence


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('video', 'annotations', 'split-manifest', 'detector-model', 'pose-model', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('video-id', 'shot-id', 'detector-sha256', 'pose-sha256', 'onnxruntime-version', 'opencv-version'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--num-frames', type=int, default=8)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--sigma', type=float, default=2.0)
    args = parser.parse_args()
    if args.num_frames <= 0 or args.size <= 0 or args.sigma <= 0:
        raise ValueError('frame count, size, and sigma must be positive')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    annotations = json.loads(args.annotations.read_text())
    matches = [item for item in annotations if str(item.get('video_id')) == args.video_id
               and str(item.get('shot_id')) == args.shot_id]
    if len(matches) != 1:
        raise ValueError(f'expected exactly one matching annotation, found {len(matches)}')
    split = json.loads(args.split_manifest.read_text())
    partitions = [name for name in ('train', 'val', 'test')
                  if args.video_id in {str(x) for x in split[name]}]
    if len(partitions) != 1 or partitions[0] not in ('train', 'val'):
        raise ValueError(f'expected one development partition, found {partitions}')
    if args.video.name != args.video_id + '.mp4':
        raise ValueError('video filename must match the annotated video ID')
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f'could not open {args.video}')
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        annotation = matches[0]
        start, end = resolve_frame_interval(int(annotation['segment_start_frame']),
                                             int(annotation['segment_end_frame']), frame_count)
        indices = [int(i) for i in select_frame_indices(start, end, args.num_frames)]
        config = DWPoseONNXConfig(detector_path=args.detector_model, pose_path=args.pose_model,
            detector_sha256=args.detector_sha256, pose_sha256=args.pose_sha256,
            expected_onnxruntime_version=args.onnxruntime_version,
            expected_opencv_version=args.opencv_version, providers=('CPUExecutionProvider',))
        extractor = create_dwpose_onnx_extractor(config)
        observations = {}
        for index in dict.fromkeys(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise RuntimeError(f'could not decode frame {index}')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            observations[index] = extractor.extract_frame(rgb)
            print(f'Frame {index}: {len(observations[index]["candidates"])} candidates', flush=True)
    finally:
        capture.release()
    rows = [{'source_frame_index': i, 'result': result} for i, result in observations.items()]
    maps, selected_indices, selected = build_sequence(rows, start, end, args.num_frames,
                                                       args.size, args.size, args.sigma)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    control = output / 'controls' / args.video_id / f'shot_{args.shot_id}_controls_encoded.npz'
    control.parent.mkdir(parents=True)
    np.savez_compressed(control, pose_encoded=maps)
    raw_path = output / 'raw_predictions.json'
    raw_path.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    manifest = {
        'purpose': 'annotated-shot loader diagnostic; not eligible for training',
        'training_eligible': False, 'video_id': args.video_id, 'shot_id': args.shot_id,
        'partition': partitions[0], 'source_frame_count': frame_count,
        'source_interval_half_open': [start, end], 'source_frame_indices': selected_indices,
        'source_video_sha256': sha256(args.video), 'annotations_sha256': sha256(args.annotations),
        'split_manifest_sha256': sha256(args.split_manifest), 'raw_predictions_sha256': sha256(raw_path),
        'extractor_config_sha256': selected[0]['model_info']['config_sha256'],
        'model_info': selected[0]['model_info'],
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'shape_cthw': list(maps.shape), 'sigma_pixels': args.sigma,
        'channels': ['head', 'torso', 'limbs'],
        'candidate_counts': [len(result['candidates']) for result in selected],
        'policy': 'all detections; all in-frame positions; no calibrated visibility, score filtering or tracking',
        'limitations': 'diagnostic only; potential false positives, hidden-joint guesses and mixed identities',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print('Source frame indices:', selected_indices)
    print('Control:', control.resolve())
    print('Output:', output.resolve())
    print('DIAGNOSTIC ONLY: not eligible for training.')


if __name__ == '__main__':
    main()
