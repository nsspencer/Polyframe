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
        t = Transform.from_values(
            translation=trans,
            rotation=rot,
            scale=scale,
            perspective=persp
        )
        np.testing.assert_array_equal(t.translation, trans)
        np.testing.assert_array_equal(t.rotation, rot)
        np.testing.assert_array_equal(t.scale, scale)
        np.testing.assert_array_equal(t.perspective, persp)

    def test_from_values_translation_only(self):
        t = Transform.from_values(translation=[7, 8, 9])
        expected = np.eye(4)
        expected[:3, 3] = [7, 8, 9]
        np.testing.assert_array_equal(t.matrix, expected)
        np.testing.assert_array_equal(t.translation, [7, 8, 9])
        np.testing.assert_array_equal(t.rotation, np.eye(3))
        np.testing.assert_array_equal(t.scale, [1, 1, 1])
        np.testing.assert_array_equal(t.perspective, [0, 0, 0, 1])

    def test_from_values_rotation_only(self):
        R = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [-1, 0, 0]])
        t = Transform.from_values(rotation=R)
        expected = np.eye(4)
        expected[:3, :3] = R
        np.testing.assert_array_equal(t.matrix, expected)
        np.testing.assert_array_equal(t.rotation, R)
        np.testing.assert_array_equal(t.translation, [0, 0, 0])
        np.testing.assert_array_equal(t.scale, [1, 1, 1])
        np.testing.assert_array_equal(t.perspective, [0, 0, 0, 1])

    def test_from_values_scale_vector(self):
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
        np.testing.assert_array_equal(t.rotation, np.eye(3))
        np.testing.assert_array_equal(t.scale, [1, 1, 1])
        np.testing.assert_array_equal(t.translation, [0, 0, 0])

    def test_from_values_rotation_and_scale(self):
        R = np.array([[0, -1, 0],
                      [1,  0, 0],
                      [0,  0, 1]])
        s = np.array([2.0, 2.0, 2.0])
        t = Transform.from_values(rotation=R, scale=s)
        expected3 = R @ np.diag(s)
        np.testing.assert_array_equal(t.matrix[:3, :3], expected3)
        np.testing.assert_array_equal(t.rotation, R)

    def test_from_values_invalid_scale_shape_raises(self):
        with self.assertRaises(ValueError):
            Transform.from_values(scale=[1.0, 2.0])  # wrong length

    def test_from_values_shear_only(self):
        H = np.array([
            [1.0, 1.5, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 1.0]
        ])
        t = Transform.from_values(shear=H)
        # raw 3×3 block must be exactly H
        np.testing.assert_array_equal(t.matrix[:3, :3], H)
        # translation/perspective defaults
        np.testing.assert_array_equal(t.translation, [0, 0, 0])
        np.testing.assert_array_equal(t.perspective, [0, 0, 0, 1])
        # rotation & scale *extract* a polar factor and unit‐scale; do not check here

    def test_from_values_invalid_shear_raises(self):
        # wrong shape or non-unit diagonal
        with self.assertRaises(ValueError):
            Transform.from_values(shear=[[1, 0], [0, 1]])

    def test_from_quaternion(self):
        q = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])
        t = Transform.from_quaternion(q, w_last=True)
        Rz90 = np.array([[0, -1, 0],
                         [1,  0, 0],
                         [0,  0, 1]])
        np.testing.assert_allclose(t.rotation, Rz90, atol=1e-6)
        np.testing.assert_array_equal(t.translation, [0, 0, 0])
        np.testing.assert_array_equal(t.scale, [1, 1, 1])
        np.testing.assert_array_equal(t.perspective, [0, 0, 0, 1])

    def test_from_euler_angles(self):
        t = Transform.from_euler_angles(0, 0, 90, degrees=True)
        Rz90 = np.array([[0, -1, 0],
                         [1,  0, 0],
                         [0,  0, 1]])
        np.testing.assert_allclose(t.rotation, Rz90, atol=1e-6)
        np.testing.assert_array_equal(t.translation, [0, 0, 0])
        np.testing.assert_array_equal(t.scale, [1, 1, 1])
        np.testing.assert_array_equal(t.perspective, [0, 0, 0, 1])

    def test_from_flat_array(self):
        flat = np.arange(16, dtype=float)
        t = Transform.from_flat_array(flat)
        expected = flat.reshape(4, 4)
        np.testing.assert_array_equal(t.matrix, expected)
        np.testing.assert_array_equal(t.to_flat_array(), flat)

    def test_from_list(self):
        lst = list(range(16))
        t = Transform.from_list(lst)
        expected = np.array(lst, dtype=float).reshape(4, 4)
        np.testing.assert_array_equal(t.matrix, expected)
        self.assertEqual(t.to_list(), [float(i) for i in lst])

    def test_from_values_all_components(self):
        # translation, rotation, scale, shear, perspective together
        trans = np.array([5.0, -3.0, 2.0])
        R = np.array([[0, -1,  0],
                      [1,  0,  0],
                      [0,  0,  1]])
        scale = np.array([2.0, 3.0, 4.0])
        H = np.array([[1.0, 0.5, 0.0],
                      [0.0, 1.0, 1.5],
                      [0.0, 0.0, 1.0]])
        persp = np.array([9.0, 8.0, 7.0, 6.0])

        t = Transform.from_values(
            translation=trans,
            rotation=R,
            scale=scale,
            shear=H,
            perspective=persp
        )

        # raw matrix check
        expected3 = R @ (np.diag(scale) @ H)
        expected = np.eye(4)
        expected[:3, :3] = expected3
        expected[:3, 3] = trans
        expected[3, :] = persp

        np.testing.assert_array_equal(t.matrix, expected)

        # still verify translation & perspective
        np.testing.assert_array_equal(t.translation, trans)
        np.testing.assert_array_equal(t.perspective, persp)


if __name__ == "__main__":
    unittest.main()
