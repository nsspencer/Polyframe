import unittest
import polyframe
import numpy as np

Transform = polyframe.define_convention(
    polyframe.Direction.FORWARD,
    polyframe.Direction.LEFT,
    polyframe.Direction.UP
)


class TestCreation(unittest.TestCase):
    def test_identity(self):
        t = Transform.identity()
        np.testing.assert_array_equal(t.matrix, np.eye(4))
        np.testing.assert_array_equal(t.translation, [0, 0, 0])
        np.testing.assert_array_equal(t.rotation, np.eye(3))
        np.testing.assert_array_equal(t.scale, [1, 1, 1])
        np.testing.assert_array_equal(t.perspective, [0, 0, 0, 1])

    def test_from_values_uniform_scale(self):
        trans = np.array([1, 2, 3])
        rot = np.eye(3)
        scale = np.array([2, 2, 2])
        persp = np.array([0, 0, 0, 1])
        t = Transform.from_values(trans, rot, scale, perspective=persp)
        np.testing.assert_array_equal(t.translation, trans)
        np.testing.assert_array_equal(t.rotation, rot)
        np.testing.assert_array_equal(t.scale, scale)
        np.testing.assert_array_equal(t.perspective, persp)

    def test_from_values_translation_only(self):
        t = Transform.from_values(translation=[7, 8, 9])
        # matrix should be identity except for the translation column
        expected = np.eye(4)
        expected[:3, 3] = [7, 8, 9]
        np.testing.assert_array_equal(t.matrix, expected)
        np.testing.assert_array_equal(t.translation, [7, 8, 9])
        np.testing.assert_array_equal(t.rotation, np.eye(3))
        np.testing.assert_array_equal(t.scale, [1, 1, 1])
        np.testing.assert_array_equal(t.perspective, [0, 0, 0, 1])

    def test_from_values_rotation_only(self):
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        t = Transform.from_values(rotation=R)
        expected = np.eye(4)
        expected[:3, :3] = R
        np.testing.assert_array_equal(t.matrix, expected)
        np.testing.assert_array_equal(t.rotation, R)
        np.testing.assert_array_equal(t.translation, [0, 0, 0])
        np.testing.assert_array_equal(t.scale, [1, 1, 1])
        np.testing.assert_array_equal(t.perspective, [0, 0, 0, 1])

    def test_from_values_scale_vector(self):
        # non-uniform scale via 3-element vector
        s = np.array([2.0, 3.0, 4.0])
        t = Transform.from_values(scale=s)
        expected = np.diag([2.0, 3.0, 4.0, 1.0])
        np.testing.assert_array_equal(t.matrix, expected)
        np.testing.assert_array_equal(t.scale, s)

    def test_from_values_perspective_only(self):
        p = np.array([9, 8, 7, 6])
        t = Transform.from_values(perspective=p)
        expected = np.eye(4)
        expected[3, :] = p
        np.testing.assert_array_equal(t.matrix, expected)
        np.testing.assert_array_equal(t.perspective, p)
        # everything else stays default
        np.testing.assert_array_equal(t.rotation, np.eye(3))
        np.testing.assert_array_equal(t.translation, [0, 0, 0])
        np.testing.assert_array_equal(t.scale, [1, 1, 1])

    def test_from_values_rotation_and_scale(self):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        s = np.array([2.0, 2.0, 2.0])
        t = Transform.from_values(rotation=R, scale=s)
        # scale should post‐multiply rotation
        expected3 = R @ np.diag(s)
        np.testing.assert_array_equal(t.matrix[:3, :3], expected3)
        # and rotation property (pure) should strip out scale
        np.testing.assert_array_equal(t.rotation, R)

    def test_from_values_invalid_scale_shape_raises(self):
        with self.assertRaises(ValueError):
            # invalid: neither 1-, 3- nor 3×3-shaped
            Transform.from_values(scale=[1.0, 2.0])

    def test_from_quaternion(self):
        # 90° about Z: quat = [0,0,sin45,cos45]
        q = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])
        t = Transform.from_quaternion(q, w_last=True)
        Rz90 = np.array([[0, -1, 0],
                         [1, 0, 0],
                         [0, 0, 1]])
        np.testing.assert_allclose(t.rotation, Rz90, atol=1e-6)
        np.testing.assert_array_equal(t.translation, [0, 0, 0])
        np.testing.assert_array_equal(t.scale, [1, 1, 1])
        np.testing.assert_array_equal(t.perspective, [0, 0, 0, 1])

    def test_from_euler_angles(self):
        # yaw=90 about Z gives same Rz90
        t = Transform.from_euler_angles(roll=0, pitch=0, yaw=90, degrees=True)
        Rz90 = np.array([[0, -1, 0],
                         [1, 0, 0],
                         [0, 0, 1]])
        np.testing.assert_allclose(t.rotation, Rz90, atol=1e-6)
        np.testing.assert_array_equal(t.translation, [0, 0, 0])
        np.testing.assert_array_equal(t.scale, [1, 1, 1])
        np.testing.assert_array_equal(t.perspective, [0, 0, 0, 1])

    def test_from_flat_array(self):
        flat = np.arange(16, dtype=float)
        t = Transform.from_flat_array(flat)
        expected = flat.reshape(4, 4)
        np.testing.assert_array_equal(t.matrix, expected)
        # also round‐trip
        np.testing.assert_array_equal(t.to_flat_array(), flat)

    def test_from_list(self):
        lst = list(range(16))
        t = Transform.from_list(lst)
        expected = np.array(lst, dtype=float).reshape(4, 4)
        np.testing.assert_array_equal(t.matrix, expected)
        # list conversion
        self.assertEqual(t.to_list(), [float(i) for i in lst])


if __name__ == "__main__":
    unittest.main()
