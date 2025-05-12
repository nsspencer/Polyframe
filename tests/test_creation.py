import unittest
import polyframe
import numpy as np

Transform = polyframe.define_convention(
    polyframe.Direction.FORWARD, polyframe.Direction.LEFT, polyframe.Direction.UP)


class TestCreation(unittest.TestCase):
    def test_identity(self):
        # Test the identity transformation
        transform = Transform.identity()
        np.testing.assert_array_equal(np.eye(4), transform.matrix)
        np.testing.assert_array_equal(
            transform.translation, np.array([0, 0, 0]))
        np.testing.assert_array_equal(transform.rotation, np.eye(3))
        np.testing.assert_array_equal(transform.scale, np.array([1, 1, 1]))
        np.testing.assert_array_equal(
            transform.perspective, np.array([0, 0, 0, 1]))

    def test_from_values(self):
        # Test creating a transformation from translation, rotation, scale, and perspective
        translation = np.array([1, 2, 3])
        rotation = np.eye(3)
        scale = np.array([2, 2, 2])
        perspective = np.array([0, 0, 0, 1])
        transform = Transform.from_values(
            translation, rotation, scale, perspective)
        np.testing.assert_array_equal(transform.translation, translation)
        np.testing.assert_array_equal(transform.rotation, rotation)
        np.testing.assert_array_equal(transform.scale, scale)
        np.testing.assert_array_equal(transform.perspective, perspective)

    def test_from_translation(self):
        # Test creating a transformation from a translation vector
        translation = np.array([1, 2, 3])
        transform = Transform.from_translation(translation)
        np.testing.assert_array_equal(transform.translation, translation)
        np.testing.assert_array_equal(transform.rotation, np.eye(3))
        np.testing.assert_array_equal(transform.scale, np.array([1, 1, 1]))
        np.testing.assert_array_equal(
            transform.perspective, np.array([0, 0, 0, 1]))

    def test_from_rotation(self):
        # Test creating a transformation from a rotation matrix
        rotation_matrix = np.array([[0, -1, 0],
                                    [1, 0, 0],
                                    [0, 0, 1]])
        transform = Transform.from_rotation(rotation_matrix)
        np.testing.assert_array_equal(
            transform.translation, np.array([0, 0, 0]))
        np.testing.assert_array_equal(transform.rotation, rotation_matrix)
        np.testing.assert_array_equal(transform.scale, np.array([1, 1, 1]))
        np.testing.assert_array_equal(
            transform.perspective, np.array([0, 0, 0, 1]))


if __name__ == "__main__":
    unittest.main()
