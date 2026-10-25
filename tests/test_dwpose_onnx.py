"""Unit tests for the DWPose ONNX backend without ONNX Runtime or OpenCV."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.pose_models.extraction.dwpose_onnx import (
    DWPoseONNXConfig,
    DWPoseONNXDetector,
    DWPoseONNXError,
    DWPoseONNXEstimator,
    _IMAGENET_RGB_MEAN,
    _IMAGENET_RGB_STD,
    _letterbox_top_left,
    _decode_simcc,
)


class FakeCV2:
    INTER_LINEAR = 1
    __version__ = "test-cv2"

    @staticmethod
    def resize(image, size_wh, interpolation):
        width, height = size_wh
        return np.zeros((height, width, 3), dtype=np.uint8)

    @staticmethod
    def warpAffine(image, transform, size_wh, flags):
        width, height = size_wh
        return np.zeros((height, width, 3), dtype=np.uint8)


class FakeValue:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape


class FakeSession:
    def __init__(self, output, input_shape=None):
        self.output = output
        self.input_shape = input_shape
        self.feed = None

    def get_inputs(self):
        return [FakeValue("input", self.input_shape)]

    def get_outputs(self):
        if isinstance(self.output, list):
            return [FakeValue(f"output_{index}") for index in range(len(self.output))]
        return [FakeValue("output")]

    @staticmethod
    def get_providers():
        return ["CPUExecutionProvider"]

    def run(self, output_names, feed):
        self.feed = feed
        return self.output if isinstance(self.output, list) else [self.output]


class DWPoseDetectorTests(unittest.TestCase):
    def test_letterbox_shape_padding_and_channel_layout(self):
        image = np.full((16, 32, 3), [10, 20, 30], dtype=np.uint8)
        tensor, ratio = _letterbox_top_left(image, (32, 32), FakeCV2)
        self.assertEqual(tensor.shape, (1, 3, 32, 32))
        self.assertEqual(ratio, 1.0)
        # FakeCV2 returns zero for the resized area; unused rows retain 114.
        self.assertTrue(np.all(tensor[0, :, :16, :] == 0))
        self.assertTrue(np.all(tensor[0, :, 16:, :] == 114))

    def test_returns_person_box_and_raw_score(self):
        # A 32x32 input has 4x4 + 2x2 + 1x1 YOLOX candidates.
        output = np.zeros((1, 21, 6), dtype=np.float32)
        output[0, 0] = [2.0, 2.0, np.log(2.0), np.log(2.0), 0.9, 0.9]
        session = FakeSession(output)
        detector = DWPoseONNXDetector(
            session,
            input_size_hw=(32, 32),
            cv2_module=FakeCV2,
        )
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        detections = detector.detect(image)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["box_xyxy"], [8.0, 8.0, 24.0, 24.0])
        self.assertAlmostEqual(detections[0]["score_raw"], 0.81, places=6)
        self.assertEqual(session.feed["input"].shape, (1, 3, 32, 32))

    def test_rejects_invalid_input_image(self):
        detector = DWPoseONNXDetector(
            FakeSession(np.zeros((1, 21, 6), dtype=np.float32)),
            input_size_hw=(32, 32),
            cv2_module=FakeCV2,
        )
        with self.assertRaisesRegex(DWPoseONNXError, "RGB"):
            detector.detect(np.zeros((32, 32), dtype=np.uint8))

    def test_rejects_wrong_yolox_candidate_shape(self):
        detector = DWPoseONNXDetector(
            FakeSession(np.zeros((1, 20, 6), dtype=np.float32)),
            input_size_hw=(32, 32),
            cv2_module=FakeCV2,
        )
        with self.assertRaisesRegex(DWPoseONNXError, "expected 21"):
            detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))

    def test_rejects_nonfinite_yolox_values(self):
        output = np.zeros((1, 21, 6), dtype=np.float32)
        output[0, 0] = [2.0, 2.0, np.nan, 0.0, 0.9, 0.9]
        detector = DWPoseONNXDetector(
            FakeSession(output),
            input_size_hw=(32, 32),
            cv2_module=FakeCV2,
        )
        with self.assertRaisesRegex(DWPoseONNXError, "non-finite"):
            detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))


class DWPoseEstimatorTests(unittest.TestCase):
    @staticmethod
    def make_session():
        x = np.zeros((1, 133, 64), dtype=np.float32)
        y = np.zeros((1, 133, 64), dtype=np.float32)
        x[:, :, 32] = 0.8
        y[:, :, 32] = 0.7
        return FakeSession([x, y], input_shape=[1, 3, 32, 32])

    def test_restores_original_frame_pixels_and_emits_coco17(self):
        session = self.make_session()
        estimator = DWPoseONNXEstimator(session, cv2_module=FakeCV2)
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        result = estimator.estimate(image, [[10, 10, 30, 30]])[0]
        self.assertEqual(result["joint_names"][0], "nose")
        self.assertEqual(len(result["joint_names"]), 17)
        self.assertEqual(len(result["xy"]), 17)
        self.assertEqual(len(result["scores_raw"]), 17)
        self.assertEqual(result["xy"][0], [20.0, 20.0])
        self.assertAlmostEqual(result["scores_raw"][0], 0.7)
        self.assertEqual(result["coordinate_space"], "original_pixels")
        self.assertIsNone(result["to_original"])
        self.assertEqual(session.feed["input"].shape, (1, 3, 32, 32))
        np.testing.assert_allclose(
            session.feed["input"][0, :, 0, 0],
            -_IMAGENET_RGB_MEAN / _IMAGENET_RGB_STD,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_restores_coordinates_for_non_square_model_input(self):
        x = np.zeros((1, 133, 64), dtype=np.float32)
        y = np.zeros((1, 133, 128), dtype=np.float32)
        x[:, :, 32] = 0.8
        y[:, :, 64] = 0.7
        session = FakeSession([x, y], input_shape=[1, 3, 64, 32])
        estimator = DWPoseONNXEstimator(session, cv2_module=FakeCV2)
        result = estimator.estimate(
            np.zeros((80, 80, 3), dtype=np.uint8), [[20, 10, 60, 50]]
        )[0]
        self.assertEqual(result["xy"][0], [40.0, 30.0])

    def test_maps_nonpositive_simcc_peak_to_missing_without_changing_score(self):
        x = np.zeros((1, 133, 64), dtype=np.float32)
        y = np.zeros((1, 133, 64), dtype=np.float32)
        x[:, :, 10] = 0.5
        y[:, :, 10] = 0.4
        x[0, 0, 10] = 0.0
        y[0, 0, 10] = 0.0
        session = FakeSession([x, y], input_shape=[1, 3, 32, 32])
        estimator = DWPoseONNXEstimator(session, cv2_module=FakeCV2)
        result = estimator.estimate(
            np.zeros((40, 40, 3), dtype=np.uint8), [[10, 10, 30, 30]]
        )[0]
        self.assertIsNone(result["xy"][0])
        self.assertEqual(result["scores_raw"][0], 0.0)
        self.assertIsNotNone(result["xy"][1])

    def test_empty_boxes_do_not_call_model(self):
        session = self.make_session()
        estimator = DWPoseONNXEstimator(session, cv2_module=FakeCV2)
        result = estimator.estimate(np.zeros((40, 40, 3), dtype=np.uint8), [])
        self.assertEqual(result, [])
        self.assertIsNone(session.feed)

    def test_simcc_decoder_rejects_bad_shapes(self):
        with self.assertRaisesRegex(DWPoseONNXError, "SimCC"):
            _decode_simcc(np.zeros(17), np.zeros(18), 2.0)

    def test_estimator_rejects_wrong_joint_count(self):
        x = np.zeros((1, 132, 64), dtype=np.float32)
        y = np.zeros((1, 132, 64), dtype=np.float32)
        session = FakeSession([x, y], input_shape=[1, 3, 32, 32])
        estimator = DWPoseONNXEstimator(session, cv2_module=FakeCV2)
        with self.assertRaisesRegex(DWPoseONNXError, "132 joints"):
            estimator.estimate(
                np.zeros((40, 40, 3), dtype=np.uint8), [[10, 10, 30, 30]]
            )


class ConfigAndFactoryTests(unittest.TestCase):
    @staticmethod
    def config(detector_path, pose_path, detector_hash, pose_hash, **kwargs):
        return DWPoseONNXConfig(
            detector_path=detector_path,
            pose_path=pose_path,
            detector_sha256=detector_hash,
            pose_sha256=pose_hash,
            expected_onnxruntime_version="test-runtime",
            expected_opencv_version="test-cv2",
            **kwargs,
        )

    def test_cuda_requires_explicit_device_id(self):
        with self.assertRaisesRegex(ValueError, "device_id"):
            self.config("det.onnx", "pose.onnx", "0" * 64, "1" * 64,
                        providers=("CUDAExecutionProvider",))

    def test_factory_verifies_hashes_and_records_provenance(self):
        class Runtime:
            __version__ = "test-runtime"

            def __init__(self):
                self.calls = []

            @staticmethod
            def get_available_providers():
                return ["CPUExecutionProvider"]

            def InferenceSession(self, path, providers):
                self.calls.append((path, providers))
                if path.endswith("detector.onnx"):
                    output = np.zeros((1, 21, 6), dtype=np.float32)
                    return FakeSession(output)
                return DWPoseEstimatorTests.make_session()

        class Wrapper:
            def __init__(self, detector, estimator, model_info):
                self.detector = detector
                self.estimator = estimator
                self.model_info = model_info

        with tempfile.TemporaryDirectory() as temp_dir:
            detector_path = Path(temp_dir) / "detector.onnx"
            pose_path = Path(temp_dir) / "pose.onnx"
            detector_path.write_bytes(b"detector model test bytes")
            pose_path.write_bytes(b"pose model test bytes")
            detector_hash = hashlib.sha256(detector_path.read_bytes()).hexdigest()
            pose_hash = hashlib.sha256(pose_path.read_bytes()).hexdigest()
            config = self.config(detector_path, pose_path, detector_hash, pose_hash)
            runtime = Runtime()
            wrapper = __import__(
                "src.pose_models.extraction.dwpose_onnx",
                fromlist=["create_dwpose_onnx_extractor"],
            ).create_dwpose_onnx_extractor(
                config,
                wrapper_class=Wrapper,
                runtime=runtime,
                cv2_module=FakeCV2,
            )
            self.assertEqual(len(runtime.calls), 2)
            self.assertEqual(wrapper.model_info["detector"]["sha256"], detector_hash)
            self.assertEqual(wrapper.model_info["pose"]["sha256"], pose_hash)
            self.assertRegex(wrapper.model_info["config_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(wrapper.model_info["runtime"]["requested_providers"],
                             ["CPUExecutionProvider"])

    def test_factory_rejects_hash_mismatch_before_loading(self):
        class Runtime:
            __version__ = "test-runtime"
            session_count = 0

            @staticmethod
            def get_available_providers():
                return ["CPUExecutionProvider"]

            def InferenceSession(self, *args, **kwargs):
                self.session_count += 1
                raise AssertionError("sessions must not load before pin verification")

        with tempfile.TemporaryDirectory() as temp_dir:
            detector_path = Path(temp_dir) / "detector.onnx"
            pose_path = Path(temp_dir) / "pose.onnx"
            detector_path.write_bytes(b"wrong detector bytes")
            pose_path.write_bytes(b"pose model test bytes")
            config = self.config(detector_path, pose_path, "0" * 64,
                                 hashlib.sha256(pose_path.read_bytes()).hexdigest())
            runtime = Runtime()
            from src.pose_models.extraction.dwpose_onnx import create_dwpose_onnx_extractor

            with self.assertRaisesRegex(DWPoseONNXError, "SHA-256 mismatch"):
                create_dwpose_onnx_extractor(
                    config,
                    wrapper_class=object,
                    runtime=runtime,
                    cv2_module=FakeCV2,
                )
            self.assertEqual(runtime.session_count, 0)

    def test_factory_rejects_missing_requested_provider(self):
        class Runtime:
            __version__ = "test-runtime"

            @staticmethod
            def get_available_providers():
                return ["CPUExecutionProvider"]

        with tempfile.TemporaryDirectory() as temp_dir:
            detector_path = Path(temp_dir) / "detector.onnx"
            pose_path = Path(temp_dir) / "pose.onnx"
            detector_path.write_bytes(b"detector")
            pose_path.write_bytes(b"pose")
            config = self.config(
                detector_path,
                pose_path,
                hashlib.sha256(detector_path.read_bytes()).hexdigest(),
                hashlib.sha256(pose_path.read_bytes()).hexdigest(),
                providers=("CUDAExecutionProvider",),
                provider_options={"CUDAExecutionProvider": {"device_id": 2}},
            )
            from src.pose_models.extraction.dwpose_onnx import create_dwpose_onnx_extractor

            with self.assertRaisesRegex(DWPoseONNXError, "unavailable"):
                create_dwpose_onnx_extractor(
                    config,
                    wrapper_class=object,
                    runtime=Runtime(),
                    cv2_module=FakeCV2,
                )

    def test_factory_rejects_runtime_version_mismatch(self):
        class Runtime:
            __version__ = "different-runtime"

        with tempfile.TemporaryDirectory() as temp_dir:
            detector_path = Path(temp_dir) / "detector.onnx"
            pose_path = Path(temp_dir) / "pose.onnx"
            detector_path.write_bytes(b"detector")
            pose_path.write_bytes(b"pose")
            config = self.config(
                detector_path,
                pose_path,
                hashlib.sha256(detector_path.read_bytes()).hexdigest(),
                hashlib.sha256(pose_path.read_bytes()).hexdigest(),
            )
            from src.pose_models.extraction.dwpose_onnx import create_dwpose_onnx_extractor

            with self.assertRaisesRegex(DWPoseONNXError, "version mismatch"):
                create_dwpose_onnx_extractor(
                    config,
                    wrapper_class=object,
                    runtime=Runtime(),
                    cv2_module=FakeCV2,
                )


if __name__ == "__main__":
    unittest.main()
