# tests/test_dunders.py

import unittest
import copy
import pickle
import numpy as np
import polyframe
from polyframe import Direction


class TestTransformDunders(unittest.TestCase):
    def setUp(self):
        # default frame: X→forward, Y→left, Z→up
        self.F = polyframe.define_convention(
            Direction.FORWARD, Direction.LEFT, Direction.UP
        )
        # build a non‐trivial transform: 10° roll, 20° pitch, 30° yaw, then translate
        self.t1 = (
            self.F
            .from_euler_angles(10, 20, 30)
            .apply_translation([1, 2, 3])
        )

    def test_matmul_with_vector(self):
        v = np.array([1.0, 1.0, 1.0])
        # __matmul__ on a 3‐vector should call transform_point
        out1 = self.t1 @ v
        out2 = self.t1.transform_point(v)
        np.testing.assert_allclose(out1, out2, atol=1e-6)

    def test_matmul_with_matrix(self):
        M = np.arange(16).reshape(4, 4).astype(float)
        # __matmul__ on a 4×4 just falls back to raw matmul
        self.assertTrue(isinstance(self.t1 @ M, np.ndarray))
        np.testing.assert_array_equal(self.t1 @ M, self.t1.matrix @ M)

    def test_matmul_with_transform(self):
        # composing two transforms is just 4×4 matrix multiplication
        t2 = self.F.from_euler_angles(0, 0, 90).apply_translation([4, 5, 6])
        combo = t2 @ self.t1
        expected = t2.matrix @ self.t1.matrix
        np.testing.assert_allclose(combo.matrix, expected, atol=1e-6)

    def test_mul_alias(self):
        # __mul__ is an alias for __matmul__
        t2 = self.F.from_euler_angles(0, 90, 0)
        matmul = (t2 @ self.t1).matrix
        mul_op = (t2 * self.t1).matrix
        np.testing.assert_allclose(mul_op, matmul, atol=1e-6)

    def test_eq_and_not_eq(self):
        # a shallow copy should be equal
        t_copy = copy.copy(self.t1)
        self.assertTrue(self.t1 == t_copy)

        # mutating the copy makes it unequal
        t_copy.apply_translation([1, 0, 0], inplace=True)
        self.assertFalse(self.t1 == t_copy)

        # different convention / class → not equal
        G = polyframe.define_convention(
            Direction.UP, Direction.RIGHT, Direction.FORWARD
        )
        other = G.identity()
        self.assertFalse(bool(self.t1 == other))

        # comparing to a non‐Transform always returns False
        self.assertFalse(bool(self.t1 == 123))

    def test_repr_and_str(self):
        r = repr(self.t1)
        # repr must mention class name and the word 'matrix='
        self.assertIn(self.F.__name__, r)
        self.assertIn('matrix=', r)
        # str delegates to repr
        self.assertEqual(str(self.t1), r)

    def test_copy_and_deepcopy(self):
        c1 = copy.copy(self.t1)
        dc1 = copy.deepcopy(self.t1)

        # must be distinct objects but equal in value
        self.assertIsNot(c1, self.t1)
        self.assertIsNot(dc1, self.t1)
        self.assertTrue(bool(c1 == self.t1))
        self.assertTrue(bool(dc1 == self.t1))

        # mutating the shallow copy must not affect the original
        c1.apply_translation([9, 0, 0], inplace=True)
        self.assertFalse(bool(c1 == self.t1))

    def test_reduce_and_pickle(self):
        # __reduce__ should support pickling round-trip
        data = pickle.dumps(self.t1)
        loaded = pickle.loads(data)
        self.assertIsInstance(loaded, self.F)
        np.testing.assert_allclose(loaded.matrix, self.t1.matrix, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
