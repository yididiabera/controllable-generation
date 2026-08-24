import unittest

import numpy as np

from src.pose_models.keypoints import (
    BODY_18_INDEX,
    BODY_18_JOINT_NAMES,
    BODY_18_LIMBS,
    body_bounding_box,
    sort_people_deterministically,
    validate_body_18_keypoints,
)


def body_18_person(*, center_x=0.5, center_y=0.5, confidence=1.0):
    person = np.zeros((18, 3), dtype=np.float32)
    person[:, :2] = (center_x, center_y)
    person[:, 2] = confidence
    person[1, :2] = (center_x, center_y - 0.1)
    person[8, :2] = (center_x - 0.08, center_y + 0.1)
    person[11, :2] = (center_x + 0.08, center_y + 0.1)
    return person


class Body18ContractTests(unittest.TestCase):
    def test_joint_names_and_limb_graph_are_frozen(self):
        self.assertEqual(len(BODY_18_JOINT_NAMES), 18)
        self.assertEqual(BODY_18_INDEX["neck"], 1)
        self.assertEqual(BODY_18_INDEX["right_shoulder"], 2)
        self.assertEqual(BODY_18_INDEX["left_shoulder"], 5)
        self.assertEqual(len(BODY_18_LIMBS), 17)
        self.assertEqual(BODY_18_LIMBS[0], (1, 2))
        self.assertEqual(BODY_18_LIMBS[-1], (15, 17))

    def test_validator_rejects_wrong_shape_and_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_body_18_keypoints(np.zeros((17, 3), dtype=np.float32))
        person = body_18_person()
        person[0, 0] = 1.1
        with self.assertRaisesRegex(ValueError, "coordinates"):
            validate_body_18_keypoints(person)

    def test_bbox_requires_confident_joints(self):
        person = body_18_person(confidence=0.0)
        self.assertIsNone(body_bounding_box(person))
        person[1] = (0.2, 0.3, 1.0)
        person[8] = (0.8, 0.9, 1.0)
        np.testing.assert_allclose(
            body_bounding_box(person),
            (0.2, 0.3, 0.8, 0.9),
        )

    def test_sorting_is_deterministic_by_center_then_area(self):
        right = body_18_person(center_x=0.8)
        left = body_18_person(center_x=0.2)
        first = sort_people_deterministically([right, left])
        second = sort_people_deterministically([left, right])
        self.assertTrue(np.array_equal(first[0], left))
        self.assertTrue(np.array_equal(second[0], left))


if __name__ == "__main__":
    unittest.main()
