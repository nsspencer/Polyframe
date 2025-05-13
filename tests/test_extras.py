import unittest
import copy
import numpy as np
import polyframe
from polyframe import Direction


class TestTransformUtilities(unittest.TestCase):
    def setUp(self):
        # default convention: x→forward, y→left, z→up
        self.F = polyframe.define_convention(
            Direction.FORWARD, Direction.LEFT, Direction.UP
        )
        self.I = self.F.identity()

    def test_is_rigid(self):
        # identity is rigid
        self.assertTrue(self.I.is_rigid())

        # pure rotation + translation is still rigid
        t = self.F.from_euler_angles(45, 0, 0).apply_translation([5, -3, 2])
        self.assertTrue(t.is_rigid())

        # scale breaks rigidity
        s = self.F.from_values(scale=[2, 2, 2])
        self.assertFalse(s.is_rigid())

        # shear breaks rigidity
        shear = np.eye(3)
        shear[0, 1] = 0.5
        sh = self.F.from_values(shear=shear)
        self.assertFalse(sh.is_rigid())

    def test_orthonormalize(self):
        # create a "bad" transform: scale X axis and add translation
        bad = self.F.identity()
        bad.matrix[:3, 3] = [1, 2, 3]
        bad.matrix[0, 0] = 2.0

        self.assertFalse(bad.is_rigid())

        # non-inplace orthonormalize
        good = bad.orthonormalize(inplace=False)
        # original still bad
        self.assertFalse(bad.is_rigid())
        # result must be rigid
        self.assertTrue(good.is_rigid())
        # rotation part should have been corrected back to identity
        np.testing.assert_allclose(good.rotation, np.eye(3), atol=1e-6)
        # translation preserved
        np.testing.assert_allclose(good.translation, [1, 2, 3], atol=1e-6)

        # inplace orthonormalize
        bad2 = bad.orthonormalize(inplace=True)
        self.assertIs(bad2, bad)
        self.assertTrue(bad2.is_rigid())
        np.testing.assert_allclose(bad2.rotation, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(bad2.translation, [1, 2, 3], atol=1e-6)

    def test_inverse(self):
        t = self.F.from_euler_angles(10, 20, 30).apply_translation([1, 2, 3])

        # non-inplace inverse
        inv = t.inverse(inplace=False)
        self.assertIsInstance(inv, self.F)

        # inv @ t should be identity
        comb = inv @ t
        np.testing.assert_allclose(comb.matrix, np.eye(4), atol=1e-6)

        # matches numpy.linalg.inv
        np.testing.assert_allclose(
            inv.matrix, np.linalg.inv(t.matrix), atol=1e-6)

        # inplace inverse
        t2 = copy.copy(t)
        inv2 = t2.inverse(inplace=True)
        self.assertIs(inv2, t2)
        np.testing.assert_allclose(
            t2.matrix, np.linalg.inv(t.matrix), atol=1e-6)

    def test_transpose(self):
        # build an arbitrary 4×4
        M = np.arange(16, dtype=float).reshape(4, 4)
        t = self.F(M)

        # non-inplace transpose
        tr = t.transpose(inplace=False)
        self.assertIsInstance(tr, self.F)
        np.testing.assert_array_equal(tr.matrix, M.T)
        # original untouched
        np.testing.assert_array_equal(t.matrix, M)

        # inplace transpose
        t2 = self.F(M.copy())
        ret = t2.transpose(inplace=True)
        self.assertIs(ret, t2)
        np.testing.assert_array_equal(t2.matrix, M.T)


class TestLookAt(unittest.TestCase):
    def setUp(self):
        # default convention: x→forward, y→left, z→up
        self.F = polyframe.define_convention(
            Direction.FORWARD, Direction.LEFT, Direction.UP
        )
        # a fresh identity transform
        self.I = self.F.identity()

    def assert_forward_aligned(self, T, target_vec, tol=1e-6):
        """Helper: after look_at, T.forward should align with direction_to(target_vec)."""
        desired = target_vec / np.linalg.norm(target_vec)
        actual = T.forward / np.linalg.norm(T.forward)
        np.testing.assert_allclose(actual, desired, atol=tol)

    def test_look_at_point_ahead_is_noop(self):
        # target directly along +X
        P = np.array([10.0, 0.0, 0.0])
        L = self.I.look_at(P, inplace=False)
        # no change in rotation
        np.testing.assert_allclose(L.rotation, np.eye(3), atol=1e-6)
        # identity forward stays [1,0,0]
        np.testing.assert_array_equal(L.forward, [1, 0, 0])

    def test_look_at_point_above(self):
        P = np.array([0.0, 0.0, 5.0])
        L = self.I.look_at(P)
        # now forward axis should point upward
        self.assert_forward_aligned(L, P)
        # right and up axes remain orthonormal
        self.assertAlmostEqual(np.dot(L.forward, L.up), 0.0, places=6)
        self.assertAlmostEqual(np.dot(L.forward, L.right), 0.0, places=6)
        self.assertAlmostEqual(np.dot(L.up, L.right), 0.0, places=6)
        self.assertAlmostEqual(np.linalg.norm(L.up), 1.0, places=6)
        self.assertAlmostEqual(np.linalg.norm(L.right), 1.0, places=6)

    def test_look_at_point_behind(self):
        P = np.array([-1.0, 0.0, 0.0])
        L = self.I.look_at(P)
        # forward should flip to [-1,0,0]
        np.testing.assert_allclose(L.forward, [-1, 0, 0], atol=1e-6)
        # up should remain [0,0,1]
        np.testing.assert_allclose(L.up, [0, 0, 1], atol=1e-6)

    def test_look_at_zero_vector_is_noop(self):
        # target at same origin: zero vector → no change
        L = self.I.look_at([0.0, 0.0, 0.0])
        np.testing.assert_allclose(L.rotation, np.eye(3), atol=1e-6)

    def test_preserves_translation_scale_and_shear(self):
        # build a transform with translation, scale, shear
        base = self.F.identity() \
            .apply_translation([1, 2, 3]) \
            .apply_scale([2, 0.5, 1.5], order="before") \
            .apply_shear(np.array([[1, 0.2, 0], [0, 1, 0.3], [0, 0, 1]]), order="after")
        # capture non‐rotation parts
        orig_trans = base.translation.copy()
        orig_scale = base.scale.copy()
        orig_shear = base.shear.copy()

        # look_at some arbitrary point
        L = base.look_at([0, 0, 1], inplace=False)
        # translation unchanged
        np.testing.assert_allclose(L.translation, orig_trans, atol=1e-6)
        # scale and shear unchanged
        np.testing.assert_allclose(L.scale, orig_scale, atol=1e-6)
        np.testing.assert_allclose(L.shear, orig_shear, atol=1e-6)

    def test_inplace_and_return_identity(self):
        # non‐inplace returns new object
        t0 = self.F.identity()
        t1 = t0.look_at([0, 1, 0], inplace=False)
        self.assertIsNot(t0, t1)
        # inplace returns same
        t2 = t0.look_at([0, 1, 0], inplace=True)
        self.assertIs(t0, t2)

    def test_accepts_transform_target(self):
        # target given as a Transform instance
        target = self.F.identity().apply_translation([0, 0, 1])
        L = self.I.look_at(target)
        # forward points to target.translation
        self.assert_forward_aligned(L, target.translation)

    def test_non_default_convention(self):
        # pick a frame where the z‐axis is labeled FORWARD
        C = polyframe.define_convention(
            Direction.UP, Direction.RIGHT, Direction.FORWARD
        )
        I2 = C.identity()

        # in this convention, identity().forward should equal basis_forward()
        np.testing.assert_array_equal(I2.forward, I2.basis_forward())

        # look_at a world‐point along +X → forward should align with that direction
        P = np.array([1.0, 0.0, 0.0])
        L = I2.look_at(P)
        # now its world‐forward should align with (P - origin)
        desired = P / np.linalg.norm(P)
        actual = L.forward / np.linalg.norm(L.forward)
        np.testing.assert_allclose(actual, desired, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
