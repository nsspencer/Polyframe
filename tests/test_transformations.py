import unittest
import numpy as np
from polyframe import Direction, define_convention

Transform = define_convention(
    Direction.FORWARD, Direction.LEFT, Direction.UP
)


class TestTransformMethods(unittest.TestCase):
    def setUp(self):
        # identity transform for reuse
        self.I = Transform.identity()

    def test_transform_point_identity(self):
        # both ndarray and list inputs should pass through unchanged
        points = [
            np.array([1.0, 2.0, 3.0]),
            [4, 5, 6],
            np.array([0.0, 0.0, 0.0]),
        ]
        for p in points:
            out = self.I.transform_point(p)
            expected = np.asarray(p, dtype=float)
            np.testing.assert_allclose(out, expected, atol=1e-9)

    def test_transform_vector_identity(self):
        vecs = [
            np.array([1.0, -1.0, 0.5]),
            [0, 10, -10],
        ]
        for v in vecs:
            out = self.I.transform_vector(v)
            expected = np.asarray(v, dtype=float)
            np.testing.assert_allclose(out, expected, atol=1e-9)

    def test_transform_point_translation(self):
        t = Transform.from_values(translation=[1, -2, 3])
        p = np.array([5.0, 5.0, 5.0])
        out = t.transform_point(p)
        # only translation should apply
        np.testing.assert_allclose(out, p + np.array([1, -2, 3]), atol=1e-9)

    def test_transform_vector_translation_ignored(self):
        t = Transform.from_values(translation=[1, 2, 3])
        v = np.array([7.0, 8.0, 9.0])
        # transform_vector ignores translation
        out = t.transform_vector(v)
        np.testing.assert_allclose(out, v, atol=1e-9)

    def test_transform_point_scale(self):
        s = Transform.from_values(scale=[2, 3, 4])
        p = np.array([1.0, 1.0, 1.0])
        out = s.transform_point(p)
        # only scale should apply
        np.testing.assert_allclose(out, np.array([2, 3, 4]), atol=1e-9)

    def test_transform_vector_scale(self):
        s = Transform.from_values(scale=[2, 3, 4])
        v = np.array([1.0, 1.0, 1.0])
        out = s.transform_vector(v)
        np.testing.assert_allclose(out, np.array([2, 3, 4]), atol=1e-9)

    def test_transform_point_rotation(self):
        # 90° about Z takes X→Y
        r = Transform.from_euler_angles(0, 0, 90)
        p = np.array([1.0, 0.0, 0.0])
        out = r.transform_point(p)
        np.testing.assert_allclose(out, np.array([0.0, 1.0, 0.0]), atol=1e-6)

    def test_transform_vector_rotation(self):
        # same rotation on a direction vector
        r = Transform.from_euler_angles(0, 0, 90)
        v = np.array([1.0, 0.0, 0.0])
        out = r.transform_vector(v)
        np.testing.assert_allclose(out, np.array([0.0, 1.0, 0.0]), atol=1e-6)

    def test_transform_point_combined(self):
        # rotate 90° about Z then translate by (1,2,3)
        t = Transform.from_euler_angles(0, 0, 90, translation=[1, 2, 3])
        p = np.array([1.0, 0.0, 0.0])
        # first rotates to (0,1,0), then adds translation
        np.testing.assert_allclose(
            t.transform_point(p),
            np.array([1.0, 3.0, 3.0]),
            atol=1e-6
        )

    def test_transform_point_perspective(self):
        # set bottom row so w = z
        # identity bottom row is [0,0,0,1]; adding [0,0,1,0] makes it [0,0,1,1]
        # so ph[3] = z + 1*1 = z + 1; to isolate z use a slightly different hack:
        p = Transform.identity()
        # overwrite bottom row directly to [0,0,1,0]
        p.matrix[3, :] = np.array([0.0, 0.0, 1.0, 0.0])
        pt = np.array([2.0, 4.0, 2.0])
        # ph = [2,4,2,2] so output = [2/2,4/2,2/2] = [1,2,1]
        out = p.transform_point(pt)
        np.testing.assert_allclose(out, np.array([1.0, 2.0, 1.0]), atol=1e-9)

        # but transform_vector does not use bottom row at all:
        v = np.array([2.0, 4.0, 2.0])
        np.testing.assert_allclose(p.transform_vector(v), v, atol=1e-9)

    def test_transform_point_zero_w_raises(self):
        # force bottom row to all zeros so w = 0 for any input
        p = Transform.identity()
        p.matrix[3, :] = np.zeros(4)
        with self.assertRaises(ZeroDivisionError):
            p.transform_point([1.0, 2.0, 3.0])

    def test_input_types(self):
        # lists and tuples should both be accepted
        t = Transform.from_values(translation=[10, 20, 30], scale=[2, 2, 2])
        for pt in ([1, 1, 1], (1, 1, 1)):
            out_p = t.transform_point(pt)
            expected = (np.array(pt) * 2.0) + np.array([10, 20, 30])
            np.testing.assert_allclose(out_p, expected, atol=1e-9)

            out_v = t.transform_vector(pt)
            np.testing.assert_allclose(out_v, np.array(pt) * 2.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
