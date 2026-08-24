import unittest

import numpy as np

from src.pose_models.keypoints import BODY_18_RGB_COLORS
from src.pose_models.renderer import PoseRenderConfig, render_pose


def body_18_person(*, confidence=1.0):
    person = np.zeros((18, 3), dtype=np.float32)
    person[:, :2] = (0.5, 0.5)
    person[:, 2] = confidence
    person[0, :2] = (0.2, 0.2)
    person[1, :2] = (0.5, 0.25)
    person[2, :2] = (0.7, 0.35)
    person[3, :2] = (0.8, 0.5)
    person[4, :2] = (0.9, 0.6)
    return person


class PoseRendererTests(unittest.TestCase):
    def test_empty_people_render_black_rgb_uint8(self):
        rendered = render_pose([], PoseRenderConfig(output_size=(32, 48)))
        self.assertEqual(rendered.shape, (32, 48, 3))
        self.assertEqual(rendered.dtype, np.uint8)
        self.assertEqual(np.count_nonzero(rendered), 0)

    def test_repeated_rendering_is_byte_identical(self):
        person = body_18_person()
        config = PoseRenderConfig(output_size=(64, 64))
        first = render_pose([person], config)
        second = render_pose([person], config)
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_renderer_returns_rgb_joint_color(self):
        person = body_18_person()
        rendered = render_pose([person], PoseRenderConfig(output_size=(101, 101)))
        # The nose is joint 0 at (0.2, 0.2); circles are drawn after limbs.
        self.assertTupleEqual(tuple(rendered[20, 20]), BODY_18_RGB_COLORS[0])

    def test_low_confidence_joints_are_not_rendered(self):
        person = body_18_person(confidence=0.0)
        rendered = render_pose([person], PoseRenderConfig(output_size=(64, 64)))
        self.assertEqual(np.count_nonzero(rendered), 0)

    def test_invalid_coordinates_fail_loudly(self):
        person = body_18_person()
        person[0, 0] = -0.1
        with self.assertRaisesRegex(ValueError, "coordinates"):
            render_pose([person])


if __name__ == "__main__":
    unittest.main()
