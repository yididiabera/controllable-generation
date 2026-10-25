# Pose wrapper examples

These are synthetic model outputs, not anime annotations or evidence that a pose model works. They are separate from the pose-record validator fixtures.

Place all files from this folder under `tests/fixtures/pose_extraction/` in your checkout. JSON keys beginning with `expected` belong to the review examples, not to the wrapper's input or output API. Read the accompanying intern brief and interface reference before implementing the wrapper.

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv-pose-wrapper
source .venv-pose-wrapper/bin/activate
python -m pip install -r tests/fixtures/pose_extraction/requirements.txt
python -m json.tool tests/fixtures/pose_extraction/mapped_partial.json
```

The last command works before the wrapper is implemented. It shows the complete input and expected result. The input image is created locally as a 100 by 160 RGB array filled with zeros; no image download is needed.

After implementing the wrapper, run:

```bash
python tests/fixtures/pose_extraction/try_example.py
python -m unittest discover -s tests -p 'test_dwpose_frame_extraction.py' -v
```

The first command checks all five supplied cases. The second runs the intern's own tests, which the intern must add. This package supplies example backends and an acceptance runner, not the wrapper or a complete test suite. No Torch, ONNX Runtime, WAN, model weights, or GPU is required.

- `mapped_partial.json`: source joints are deliberately reversed. A nose at model position `[20,5]` becomes original-image `[80,30]`. A missing left ankle stays null; an outside right ankle becomes null. Scores 1.2 and -0.2 are preserved.
- `no_detections.json`: the detector returns no boxes; the estimator must never run.
- `reviewed_empty.json`: explicitly passing an empty reviewed-box list skips both backends, even if the fake detector could return a box.
- `duplicate_candidates.json`: retain both identical candidate detections; assign distinct call-local IDs. Do not introduce post-detector suppression in this wrapper.
- `invalid_output.json`: one score is missing from the array; raise `PoseExtractionError`.

`example_backends.py` implements the exact `detect(image)` and `estimate(image, boxes)` interfaces with stored outputs. `try_example.py` uses those helpers to check the intern's implementation. Timing is checked for structure and finite nonnegative values, not exact equality. The helper does not establish real-model compatibility.

Add tests for reviewed nonempty boxes, original-pixel coordinates without double transformation, score/box ordering ties, invalid image inputs, malformed matrices, nonfinite values, missing scores, and backend exceptions. Preserve caller-owned images, boxes, and metadata.
