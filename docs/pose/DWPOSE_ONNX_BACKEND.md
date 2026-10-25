# DWPose ONNX backend

This backend connects the official two-stage DWPose ONNX models to the
single-frame wrapper interface. It does not assemble video records or choose a
subject across frames.

## What it does

- Runs the YOLOX person detector at the configured input size. It applies the
  official score thresholds and non-maximum suppression, then returns each
  person box with its raw objectness-times-class score.
- Crops each person for the DWPose RTMPose model, applies the model's RGB
  normalization, decodes its SimCC outputs, and restores joint locations to the
  original frame.
- Returns the first 17 COCO body joints by name. It does not add a neck, reorder
  joints for OpenPose drawing, threshold scores, or turn scores into drawing
  indices.
- Converts the model's non-positive SimCC missing sentinel to `xy: null` while
  preserving the raw score. The frame wrapper then applies its shared rules.
- Verifies both model-file SHA-256 values and exact ONNX Runtime and OpenCV
  versions before constructing sessions. It records model hashes, preprocessing
  settings and requested/active execution providers in `model_info`.

The implementation follows the official [DWPose repository](https://github.com/IDEA-Research/DWPose),
particularly its [YOLOX ONNX preprocessing and decoding](https://github.com/IDEA-Research/DWPose/blob/onnx/ControlNet-v1-1-nightly/annotator/dwpose/onnxdet.py),
[RTMPose ONNX preprocessing and SimCC decoding](https://github.com/IDEA-Research/DWPose/blob/onnx/ControlNet-v1-1-nightly/annotator/dwpose/onnxpose.py),
and [whole-body output mapping](https://github.com/IDEA-Research/DWPose/blob/onnx/ControlNet-v1-1-nightly/annotator/dwpose/wholebody.py).
The project uses the named COCO-17 body output before the whole-body wrapper's
derived-neck and OpenPose drawing transformations.

## Configure on the model server

The model paths, file hashes, ONNX Runtime version, OpenCV version, and CUDA
device are installation-specific. Do not copy placeholder hashes or infer these
values from a filename. On the server, record the actual values first:

```bash
python -c 'import cv2, onnxruntime; print("OpenCV", cv2.__version__); print("ONNX Runtime", onnxruntime.__version__); print("Providers", onnxruntime.get_available_providers())'
sha256sum /path/to/yolox_l.onnx /path/to/dw-ll_ucoco_384.onnx
```

Construct `DWPoseONNXConfig` with those exact values. For CUDA, set the chosen
device explicitly in `provider_options`, for example
`{"CUDAExecutionProvider": {"device_id": 1}}`. Select a device available to
that server process. A missing requested provider, hash mismatch, or version
mismatch stops setup with an error. CPU execution is available by explicitly
selecting `("CPUExecutionProvider",)`.

Call `create_dwpose_onnx_extractor(config)` only when extraction is intended.
It verifies the pins before loading model sessions. Keep the model files in the
shared model store; do not copy them into Git or the intern Drive.

## One-frame server smoke test

After merging the frame-wrapper PR and making this branch available on the
server, run the following from the repository root. Replace the paths,
frame index, hashes, installed versions, and CUDA device with the values checked
on that server. Use a video from the train or validation split.

```bash
python scripts/pose/smoke_test_dwpose.py \
  --video /path/to/train-or-val-video.mp4 \
  --frame-index 120 \
  --detector-model /path/to/yolox_l.onnx \
  --pose-model /path/to/dw-ll_ucoco_384.onnx \
  --detector-sha256 DETECTOR_SHA256 \
  --pose-sha256 POSE_SHA256 \
  --onnxruntime-version ONNXRUNTIME_VERSION \
  --opencv-version OPENCV_VERSION \
  --provider CUDAExecutionProvider \
  --provider CPUExecutionProvider \
  --cuda-device-id 1 \
  --output-dir /tmp/dwpose-smoke
```

The command writes `frame_result.json` and `pose_overlay.png`. Inspect the
overlay against the source frame. Check body-joint order, left/right placement,
crop restoration, missing joints, and boxes near image borders. The preview
draws every returned point and does not apply a confidence threshold; it is a
debug view, not a training control image. If the server selects CPU, replace the
provider arguments with `--provider CPUExecutionProvider` and omit
`--cuda-device-id`.

## Local checks and remaining server check

`tests/test_dwpose_onnx.py` uses fake ONNX sessions and a small OpenCV stand-in.
It checks detector decoding, person scores, SimCC decoding, missing-point
handling, coordinate restoration, hash verification, version checks and
provider selection without requiring model weights or GPU libraries.

The real model smoke test is still required on the server. Use one selected
training or validation frame, save the model metadata, inspect the returned
17-point overlay against the source image, and record empty detections, malformed
boxes, and any missing joints. This verifies integration only; it does not
establish anime pose accuracy or temporal identity consistency.
