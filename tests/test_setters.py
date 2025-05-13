import unittest
import numpy as np
from numpy.testing import assert_allclose as _assert_allclose
import polyframe
from polyframe._polyframe import (
    decompose_scale_shear,
    pure_rotation_if_possible,
    quaternion_to_rotation,
    euler_to_rotation,
)
from polyframe._polyframe import Direction


def assert_allclose(a, b, **kwargs):
    return _assert_allclose(a, b, rtol=1e-2, atol=1e-2, **kwargs)


# pick the default forward‐left‐up convention
T = polyframe.define_convention(
    Direction.FORWARD, Direction.LEFT, Direction.UP)


class TestSetterMethodsExhaustive(unittest.TestCase):
    def setUp(self):
        self.I = T.identity()

    # ------------------------
    # translation.setter + set_translation
    # ------------------------
    def test_translation_property_and_setter(self):
        t = self.I.copy()
        t.translation = [1, -2, 3]
        assert_allclose(t.translation, [1, -2, 3])

        # copy vs inplace
        t0 = self.I.copy()
        t1 = t0.set_translation([5, 6, 7])
        # original unchanged
        assert_allclose(t0.translation, [0, 0, 0])
        assert_allclose(t1.translation, [5, 6, 7])
        self.assertIsNot(t0, t1)

        t2 = t0.set_translation([8, 9, 10], inplace=True)
        self.assertIs(t2, t0)
        assert_allclose(t0.translation, [8, 9, 10])

    # ------------------------
    # rotation.setter + set_rotation + set_rotation_from_quaternion + set_rotation_from_euler
    # ------------------------
    def test_rotation_property_preserves_scale_and_shear(self):
        # start with nontrivial scale+shear
        base = T.from_values(scale=[2, 3, 4], shear=[
                             [1, 0.5, 0], [0, 1, 0.2], [0, 0, 1]])
        orig_S, orig_H = decompose_scale_shear(base.matrix[:3, :3])
        # new rotation
        R_new = euler_to_rotation(10, 20, 30)
        base.rotation = R_new
        # the pure-rotation part should match
        R_after = pure_rotation_if_possible(base.matrix[:3, :3])
        assert_allclose(R_after, R_new)
        # scale+shear preserved
        S2, H2 = decompose_scale_shear(base.matrix[:3, :3])
        assert_allclose(S2, orig_S)
        assert_allclose(H2, orig_H)

    def test_set_rotation_copy_and_inplace(self):
        R = euler_to_rotation(45, 0, 0)
        t0 = self.I.copy()
        t1 = t0.set_rotation(R)
        # original unchanged
        assert_allclose(self.I.rotation, np.eye(3))
        assert_allclose(t1.rotation, R)
        self.assertIsNot(t1, t0)

        t2 = t0.set_rotation(R, inplace=True)
        self.assertIs(t2, t0)
        assert_allclose(t0.rotation, R)

    def test_set_rotation_from_quaternion_and_euler(self):
        # 180deg about X
        q = np.array([1, 0, 0, 0], float)
        t_q = self.I.set_rotation_from_quaternion(q)
        assert_allclose(t_q.rotation, quaternion_to_rotation(q))

        # from euler
        angs = (30, 60, 90)
        t_e = self.I.set_rotation_from_euler(*angs)
        assert_allclose(t_e.rotation, euler_to_rotation(*angs))

    # ------------------------
    # shear.setter + set_shear
    # ------------------------
    def test_shear_property_preserves_scale_and_rotation(self):
        base = T.from_values(
            scale=[2, 3, 4],
            rotation=euler_to_rotation(10, 20, 30),
        )
        # capture original pure‐rotation R0 and scale S0/H0 decomposition
        R0 = pure_rotation_if_possible(base.matrix[:3, :3])
        S0, H0 = decompose_scale_shear(base.matrix[:3, :3])

        new_H = np.array([[1, 1, 0],
                          [0, 1, 1],
                          [0, 0, 1]])

        # apply the property setter
        base.shear = new_H

        # the block should now equal R0 @ diag(S0) @ new_H
        expected = R0 @ np.diag(S0) @ new_H
        assert_allclose(base.matrix[:3, :3], expected)

    def test_shear_property_invalid(self):
        t = self.I.copy()
        with self.assertRaises(ValueError):
            t.shear = np.zeros((2, 2))
        bad = np.eye(3)
        bad[1, 1] = 2
        with self.assertRaises(ValueError):
            t.shear = bad

    def test_set_shear_before_after_and_inplace(self):
        # base has only scale, no shear or rotation
        base = T.from_values(scale=[2, 2, 2])
        S = base.scale  # [2,2,2]

        H = np.array([[1, 2, 0],
                      [0, 1, 3],
                      [0, 0, 1]])

        # should produce diag(S) @ H (since original H was identity)
        t_a = base.set_shear(H)
        expected_a = np.diag(S) @ H
        assert_allclose(t_a.matrix[:3, :3], expected_a)

        # inplace also writes into matrix exactly
        base2 = T.from_values(scale=[1, 2, 3])
        S2 = base2.scale
        ret = base2.set_shear(np.eye(3), inplace=True)
        self.assertIs(ret, base2)
        expected_eye = np.diag(S2) @ np.eye(3)
        assert_allclose(base2.matrix[:3, :3], expected_eye)

    # ------------------------
    # scale.setter + set_scale
    # ------------------------
    def test_scale_property_preserves_rotation_and_shear(self):
        base = T.from_values(rotation=euler_to_rotation(5, 10, 15),
                             shear=[[1, 0.2, 0], [0, 1, 0.3], [0, 0, 1]])
        orig_R = pure_rotation_if_possible(base.matrix[:3, :3])
        S_new = [5, 6, 7]
        base.scale = S_new
        S2, H2 = decompose_scale_shear(base.matrix[:3, :3])
        assert_allclose(S2, S_new)
        # shear preserved
        assert_allclose(H2, decompose_scale_shear(
            T.from_values(shear=[[1, 0.2, 0], [0, 1, 0.3],
                          [0, 0, 1]]).matrix[:3, :3]
        )[1])
        # rotation preserved
        R2 = pure_rotation_if_possible(base.matrix[:3, :3])
        assert_allclose(R2, orig_R)

    def test_scale_property_invalid(self):
        t = self.I.copy()
        with self.assertRaises(ValueError):
            t.scale = [1, 2]  # wrong shape

    def test_set_scale_before_after_and_inplace(self):
        # start from pure shear so we can decompose its original R/S/H
        H0 = np.array([[1, 2, 0],
                       [0, 1, 3],
                       [0, 0, 1]])
        base = T.from_values(shear=H0)

        # capture original rotation
        R0 = pure_rotation_if_possible(base.matrix[:3, :3])

        S_new = np.array([2, 3, 4])

        t_a = base.set_scale(S_new)
        Pa = R0.T @ t_a.matrix[:3, :3]
        assert_allclose(np.diag(Pa), S_new)
        _, Ha = decompose_scale_shear(Pa)
        assert_allclose(Ha, H0)

        # ---------- inplace path ----------
        base2 = self.I.copy()
        ret = base2.set_scale([7, 8, 9], inplace=True)
        self.assertIs(ret, base2)
        Pin = pure_rotation_if_possible(
            base2.matrix[:3, :3]).T @ base2.matrix[:3, :3]
        assert_allclose(np.diag(Pin), [7, 8, 9])

    # ------------------------
    # perspective.setter + set_perspective
    # ------------------------
    def test_perspective_property_and_setter(self):
        t = self.I.copy()
        p = [9, 8, 7, 6]
        t.perspective = p
        assert_allclose(t.perspective, p)

        t0 = self.I.copy()
        t1 = t0.set_perspective(p)
        # copy unchanged
        assert_allclose(t0.perspective, [0, 0, 0, 1])
        assert_allclose(t1.perspective, p)
        self.assertIsNot(t0, t1)

        t2 = t0.set_perspective(p, inplace=True)
        self.assertIs(t2, t0)
        assert_allclose(t0.perspective, p)


if __name__ == "__main__":
    unittest.main()
