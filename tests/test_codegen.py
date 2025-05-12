import unittest
import polyframe
import numpy as np


class TestWorldDirections(unittest.TestCase):
    def test_forward_left_up(self):
        # Test the identity transformation
        Transform = polyframe.define_convention(
            polyframe.Direction.FORWARD, polyframe.Direction.LEFT, polyframe.Direction.UP)
        identity = Transform.identity()
        np.testing.assert_array_equal(np.eye(4), identity.matrix)
        np.testing.assert_array_equal(identity.forward, np.array([1, 0, 0]))
        np.testing.assert_array_equal(identity.backward, np.array([-1, 0, 0]))
        np.testing.assert_array_equal(identity.left, np.array([0, 1, 0]))
        np.testing.assert_array_equal(identity.right, np.array([0, -1, 0]))
        np.testing.assert_array_equal(identity.up, np.array([0, 0, 1]))
        np.testing.assert_array_equal(identity.down, np.array([0, 0, -1]))


if __name__ == "__main__":
    unittest.main()
