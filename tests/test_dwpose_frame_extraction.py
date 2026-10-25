"""Tests for the DWPose frame-extraction wrapper.

Covers the five supplied fixture cases in
``tests/fixtures/pose_extraction/`` plus edge cases around reviewed
boxes, coordinate transforms, ordering, validation, and backend
failures.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

import numpy as np

from src.pose_models.extraction.dwpose import (
    CANONICAL_JOINT_NAMES,
    DWPoseFrameExtractor,
    PoseExtractionError,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "pose_extraction"

MODEL_INFO = {"kind": "simulated", "pipeline_id": "pose_wrapper_test_v1"}
BOX = [40, 20, 120, 80]


class FakeDetector:
    def __init__(self, outputs=None, error=None):
        self.outputs = deepcopy(outputs) if outputs is not None else []
        self.error = error
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return deepcopy(self.outputs)


class FakeEstimator:
    def __init__(self, outputs=None, error=None):
        self.outputs = deepcopy(outputs) if outputs is not None else []
        self.error = error
        self.calls = 0
        self.received_boxes = None

    def estimate(self, image, boxes):
        self.calls += 1
        self.received_boxes = deepcopy(boxes)
        if self.error is not None:
            raise self.error
        return deepcopy(self.outputs)


def make_image(height=100, width=160):
    return np.zeros((height, width, 3), dtype=np.uint8)


def detection(box=None, score=0.93):
    return {"box_xyxy": list(box if box is not None else BOX), "score_raw": score}


def estimator_output(xy=None, scores=None, names=None,
                     coordinate_space="original_pixels", to_original=None):
    return {
        "joint_names": list(names if names is not None else CANONICAL_JOINT_NAMES),
        "xy": xy if xy is not None else [[10, 10] for _ in range(17)],
        "scores_raw": scores if scores is not None else [0.5] * 17,
        "coordinate_space": coordinate_space,
        "to_original": to_original,
    }


def make_wrapper(detector_outputs=None, estimator_outputs=None):
    detector = FakeDetector(detector_outputs)
    estimator = FakeEstimator(estimator_outputs)
    wrapper = DWPoseFrameExtractor(
        detector=detector, estimator=estimator, model_info=deepcopy(MODEL_INFO)
    )
    return wrapper, detector, estimator


class SuppliedExamplesTests(unittest.TestCase):
    def test_all_supplied_fixture_cases(self):
        paths = sorted(FIXTURE_DIR.glob("*.json"))
        self.assertEqual(len(paths), 5)
        for path in paths:
            with self.subTest(case=path.name):
                self._run_fixture(path)

    def _run_fixture(self, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        spec = data["image"]
        image = np.full(
            (spec["height"], spec["width"], spec["channels"]),
            spec["fill"], dtype=np.uint8,
        )
        unchanged = image.copy()
        detector = FakeDetector(data["detector_outputs"])
        estimator = FakeEstimator(data["estimator_outputs"])
        wrapper = DWPoseFrameExtractor(
            detector=detector, estimator=estimator, model_info=data["model_info"]
        )
        if data.get("expected_error") == "PoseExtractionError":
            with self.assertRaises(PoseExtractionError):
                wrapper.extract_frame(image, reviewed_boxes=data["reviewed_boxes"])
        else:
            result = dict(wrapper.extract_frame(
                image, reviewed_boxes=data["reviewed_boxes"]
            ))
            timing = result.pop("timing_ms")
            self.assertEqual(set(timing), {"detector", "estimator", "total"})
            for value in timing.values():
                self.assertNotIsInstance(value, bool)
                self.assertIsInstance(value, (int, float))
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, 0)
            if detector.calls == 0:
                self.assertEqual(timing["detector"], 0)
            if estimator.calls == 0:
                self.assertEqual(timing["estimator"], 0)
            self.assertEqual(result, data["expected"])
            json.dumps(result, allow_nan=False)
        self.assertTrue(np.array_equal(image, unchanged))
        self.assertEqual(
            {"detector": detector.calls, "estimator": estimator.calls},
            data["expected_calls"],
        )


class ReviewedBoxesTests(unittest.TestCase):
    def test_reviewed_nonempty_skips_detector(self):
        wrapper, detector, estimator = make_wrapper(
            detector_outputs=[detection(box=[0, 0, 10, 10], score=0.1)],
            estimator_outputs=[estimator_output()],
        )
        result = wrapper.extract_frame(make_image(), reviewed_boxes=[list(BOX)])
        self.assertEqual(detector.calls, 0)
        self.assertEqual(estimator.calls, 1)
        self.assertEqual(estimator.received_boxes, [BOX])
        self.assertEqual(result["timing_ms"]["detector"], 0)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["box_xyxy"], BOX)
        self.assertEqual(candidate["box_source"], "reviewed")
        self.assertIsNone(candidate["detector_score_raw"])
        self.assertEqual(result["status"], "completed")

    def test_reviewed_boxes_sorted_by_coordinates(self):
        wrapper, _, estimator = make_wrapper(
            estimator_outputs=[estimator_output(), estimator_output()],
        )
        result = wrapper.extract_frame(
            make_image(), reviewed_boxes=[[50, 20, 120, 80], list(BOX)]
        )
        self.assertEqual(
            [c["box_xyxy"] for c in result["candidates"]],
            [BOX, [50, 20, 120, 80]],
        )
        self.assertEqual(
            [c["candidate_id"] for c in result["candidates"]],
            ["candidate_0", "candidate_1"],
        )
        # The estimator saw the boxes in the caller's original order.
        self.assertEqual(estimator.received_boxes, [[50, 20, 120, 80], BOX])


class CoordinateTests(unittest.TestCase):
    def test_original_pixels_are_not_transformed(self):
        output = estimator_output(
            xy=[[20, 5]] + [[10, 10]] * 16,
            coordinate_space="original_pixels",
            to_original=None,
        )
        wrapper, _, _ = make_wrapper([detection()], [output])
        result = wrapper.extract_frame(make_image())
        self.assertEqual(result["candidates"][0]["xy_pixels"][0], [20, 5])

    def test_model_pixels_transform_applied_once(self):
        output = estimator_output(
            xy=[[20, 5]] + [[10, 10]] * 16,
            coordinate_space="model_pixels",
            to_original=[[2, 0, 40], [0, 2, 20]],
        )
        wrapper, _, _ = make_wrapper([detection()], [output])
        result = wrapper.extract_frame(make_image())
        self.assertEqual(result["candidates"][0]["xy_pixels"][0], [80, 30])

    def test_missing_and_outside_joints_keep_scores(self):
        xy = [[10, 10]] * 17
        xy[0] = None          # nose: model-missing
        xy[1] = [500, 10]     # left_eye: outside the 160x100 frame
        scores = [0.5] * 17
        scores[0] = 1.2
        scores[1] = -0.2
        output = estimator_output(xy=xy, scores=scores)
        wrapper, _, _ = make_wrapper([detection()], [output])
        result = wrapper.extract_frame(make_image())
        candidate = result["candidates"][0]
        self.assertIsNone(candidate["xy_pixels"][0])
        self.assertIsNone(candidate["xy_pixels"][1])
        self.assertEqual(candidate["invalid_reasons"][0], "model_missing")
        self.assertEqual(candidate["invalid_reasons"][1], "outside_frame")
        self.assertEqual(candidate["joint_scores_raw"][0], 1.2)
        self.assertEqual(candidate["joint_scores_raw"][1], -0.2)
        self.assertEqual(result["status"], "completed")


class OrderingTests(unittest.TestCase):
    def test_candidates_sorted_by_descending_score(self):
        wrapper, _, _ = make_wrapper(
            detector_outputs=[detection(box=[50, 20, 120, 80], score=0.4),
                              detection(box=BOX, score=0.9)],
            estimator_outputs=[estimator_output(), estimator_output()],
        )
        result = wrapper.extract_frame(make_image())
        self.assertEqual(
            [c["detector_score_raw"] for c in result["candidates"]], [0.9, 0.4]
        )
        self.assertEqual(result["candidates"][0]["box_xyxy"], BOX)

    def test_score_tie_broken_by_box_coordinates(self):
        wrapper, _, _ = make_wrapper(
            detector_outputs=[detection(box=[50, 20, 120, 80], score=0.9),
                              detection(box=BOX, score=0.9)],
            estimator_outputs=[estimator_output(), estimator_output()],
        )
        result = wrapper.extract_frame(make_image())
        self.assertEqual(
            [c["box_xyxy"] for c in result["candidates"]],
            [BOX, [50, 20, 120, 80]],
        )
        self.assertEqual(
            [c["candidate_id"] for c in result["candidates"]],
            ["candidate_0", "candidate_1"],
        )

    def test_duplicate_candidates_are_retained(self):
        wrapper, _, _ = make_wrapper(
            detector_outputs=[detection(), detection()],
            estimator_outputs=[estimator_output(), estimator_output()],
        )
        result = wrapper.extract_frame(make_image())
        self.assertEqual(len(result["candidates"]), 2)
        self.assertNotEqual(
            result["candidates"][0]["candidate_id"],
            result["candidates"][1]["candidate_id"],
        )


class InvalidInputTests(unittest.TestCase):
    def test_invalid_images_raise(self):
        wrapper, _, _ = make_wrapper()
        bad_images = [
            np.zeros((100, 160), dtype=np.uint8),          # 2D
            np.zeros((100, 160, 4), dtype=np.uint8),       # 4 channels
            np.zeros((100, 160, 3), dtype=np.float32),     # wrong dtype
            np.zeros((1, 160, 3), dtype=np.uint8),         # height < 2
            np.zeros((100, 1, 3), dtype=np.uint8),         # width < 2
            [[0, 0, 0]],                                   # not an array
        ]
        for bad in bad_images:
            with self.subTest(image=repr(bad)[:40]):
                with self.assertRaises(PoseExtractionError):
                    wrapper.extract_frame(bad)

    def test_invalid_boxes_raise_and_are_not_clipped(self):
        for bad_box in ([-1, 20, 120, 80], [40, 20, 40, 80],
                        [40, 20, 120, 101], [40, 20, 120], "box"):
            wrapper, _, _ = make_wrapper([detection(box=bad_box if isinstance(bad_box, list) else [0, 0, 10, 10])], [estimator_output()])
            if isinstance(bad_box, str):
                wrapper.detector.outputs = [{"box_xyxy": bad_box, "score_raw": 0.5}]
            with self.subTest(box=bad_box):
                with self.assertRaises(PoseExtractionError):
                    wrapper.extract_frame(make_image())

    def test_malformed_matrices_raise(self):
        for bad_matrix in ([[1, 0]], [[1, 0, 0], [0, 1]],
                           [[1, 0, "x"], [0, 1, 0]], [[1, 0, 0]] * 3):
            output = estimator_output(
                coordinate_space="model_pixels", to_original=bad_matrix
            )
            wrapper, _, _ = make_wrapper([detection()], [output])
            with self.subTest(matrix=bad_matrix):
                with self.assertRaises(PoseExtractionError):
                    wrapper.extract_frame(make_image())

    def test_model_pixels_without_transform_raises(self):
        output = estimator_output(coordinate_space="model_pixels", to_original=None)
        wrapper, _, _ = make_wrapper([detection()], [output])
        with self.assertRaises(PoseExtractionError):
            wrapper.extract_frame(make_image())

    def test_original_pixels_with_non_null_transform_raises(self):
        output = estimator_output(
            coordinate_space="original_pixels",
            to_original=[[2, 0, 40], [0, 2, 20]],
        )
        wrapper, _, _ = make_wrapper([detection()], [output])
        with self.assertRaises(PoseExtractionError):
            wrapper.extract_frame(make_image())

    def test_nonfinite_values_raise(self):
        nan_detector = [detection(score=float("nan"))]
        wrapper, _, _ = make_wrapper(nan_detector, [estimator_output()])
        with self.assertRaises(PoseExtractionError):
            wrapper.extract_frame(make_image())

        inf_xy = estimator_output(xy=[[float("inf"), 5]] + [[10, 10]] * 16)
        wrapper, _, _ = make_wrapper([detection()], [inf_xy])
        with self.assertRaises(PoseExtractionError):
            wrapper.extract_frame(make_image())

        wrapper, _, _ = make_wrapper([detection(score=True)], [estimator_output()])
        with self.assertRaises(PoseExtractionError):
            wrapper.extract_frame(make_image())

    def test_missing_scores_raise(self):
        short_scores = estimator_output(scores=[0.5] * 16)
        wrapper, _, _ = make_wrapper([detection()], [short_scores])
        with self.assertRaises(PoseExtractionError):
            wrapper.extract_frame(make_image())

        missing_key = estimator_output()
        del missing_key["scores_raw"]
        wrapper, _, _ = make_wrapper([detection()], [missing_key])
        with self.assertRaises(PoseExtractionError):
            wrapper.extract_frame(make_image())

    def test_unknown_and_duplicate_joint_names_raise(self):
        names = list(CANONICAL_JOINT_NAMES)
        names[0] = "neck"
        wrapper, _, _ = make_wrapper(
            [detection()], [estimator_output(names=names)]
        )
        with self.assertRaises(PoseExtractionError):
            wrapper.extract_frame(make_image())

        names = list(CANONICAL_JOINT_NAMES)
        names[1] = names[0]
        wrapper, _, _ = make_wrapper(
            [detection()], [estimator_output(names=names)]
        )
        with self.assertRaises(PoseExtractionError):
            wrapper.extract_frame(make_image())

    def test_estimator_result_count_mismatch_raises(self):
        wrapper, _, _ = make_wrapper(
            [detection(), detection(box=[50, 20, 120, 80])],
            [estimator_output()],
        )
        with self.assertRaises(PoseExtractionError):
            wrapper.extract_frame(make_image())


class NumpyArrayHandlingTests(unittest.TestCase):
    def test_zero_dimensional_arrays_raise_pose_extraction_error(self):
        # Regression: np.array(1.0) used to leak a raw TypeError from
        # iteration; it must surface as PoseExtractionError naming the field.
        # The 0-d values are assigned directly, because the test helper
        # itself materialises list-valued fields.
        def with_field(field, value):
            output = estimator_output()
            output[field] = value
            return output

        cases = [
            ("joint_names", with_field("joint_names", np.array(1.0))),
            ("xy", with_field("xy", np.array(1.0))),
            ("scores_raw", with_field("scores_raw", np.array(1.0))),
            ("to_original", estimator_output(
                coordinate_space="model_pixels", to_original=np.array(1.0))),
            ("xy entry", estimator_output(
                xy=[np.array(1.0)] + [[10, 10]] * 16)),
            ("to_original row", estimator_output(
                coordinate_space="model_pixels",
                to_original=[np.array(1.0), [0, 1, 0]])),
        ]
        for label, output in cases:
            with self.subTest(field=label):
                wrapper, _, _ = make_wrapper([detection()], [output])
                with self.assertRaises(PoseExtractionError):
                    wrapper.extract_frame(make_image())

    def test_correctly_shaped_numpy_arrays_are_supported(self):
        output = estimator_output(
            names=np.array(CANONICAL_JOINT_NAMES),
            xy=np.array([[10, 10]] * 17, dtype=float),
            scores=np.array([0.5] * 17),
            coordinate_space="model_pixels",
            to_original=np.array([[1, 0, 0], [0, 1, 0]]),
        )
        wrapper, _, _ = make_wrapper([detection()], [output])
        result = wrapper.extract_frame(make_image())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["candidates"][0]["xy_pixels"][0], [10.0, 10.0])


class BackendFailureTests(unittest.TestCase):
    def test_detector_exception_is_wrapped(self):
        detector = FakeDetector(error=RuntimeError("detector boom"))
        wrapper = DWPoseFrameExtractor(
            detector=detector, estimator=FakeEstimator(),
            model_info=deepcopy(MODEL_INFO),
        )
        with self.assertRaises(PoseExtractionError):
            wrapper.extract_frame(make_image())

    def test_estimator_exception_is_wrapped(self):
        estimator = FakeEstimator(error=ValueError("estimator boom"))
        wrapper = DWPoseFrameExtractor(
            detector=FakeDetector([detection()]), estimator=estimator,
            model_info=deepcopy(MODEL_INFO),
        )
        with self.assertRaises(PoseExtractionError):
            wrapper.extract_frame(make_image())


class PreservationTests(unittest.TestCase):
    def test_caller_owned_data_is_not_mutated(self):
        image = make_image()
        image[0, 0] = [9, 9, 9]
        unchanged = image.copy()
        reviewed = [list(BOX)]
        model_info = {"kind": "simulated", "nested": {"run": 1}}
        estimator = FakeEstimator([estimator_output()])
        wrapper = DWPoseFrameExtractor(
            detector=FakeDetector([detection()]), estimator=estimator,
            model_info=model_info,
        )
        result = wrapper.extract_frame(image, reviewed_boxes=reviewed)
        self.assertTrue(np.array_equal(image, unchanged))
        self.assertEqual(reviewed, [BOX])
        result["model_info"]["nested"]["run"] = 999
        result["candidates"][0]["box_xyxy"][0] = -1
        self.assertEqual(model_info, {"kind": "simulated", "nested": {"run": 1}})
        # A second call still returns pristine metadata and boxes.
        second = wrapper.extract_frame(image, reviewed_boxes=reviewed)
        self.assertEqual(second["model_info"]["nested"]["run"], 1)
        self.assertEqual(second["candidates"][0]["box_xyxy"], BOX)


if __name__ == "__main__":
    unittest.main()
