import unittest
import numpy as np
import polyframe
from polyframe import Direction
import polyframe._polyframe


class TestWorldDirections(unittest.TestCase):
    def test_world_directions(self):
        # Test the identity transformation in the default convention
        Transform = polyframe.define_convention(Direction.FORWARD,
                                                Direction.LEFT,
                                                Direction.UP)
        I = Transform.identity()
        np.testing.assert_array_equal(I.matrix, np.eye(4))
        np.testing.assert_array_equal(I.forward,  np.array([1, 0, 0]))
        np.testing.assert_array_equal(I.backward, np.array([-1, 0, 0]))
        np.testing.assert_array_equal(I.left,     np.array([0, 1, 0]))
        np.testing.assert_array_equal(I.right,    np.array([0, -1, 0]))
        np.testing.assert_array_equal(I.up,       np.array([0, 0, 1]))
        np.testing.assert_array_equal(I.down,     np.array([0, 0, -1]))

    def test_identity_directions_for_every_convention(self):
        for x in Direction:
            for y in Direction:
                for z in Direction:
                    try:
                        T = polyframe.define_convention(x, y, z)
                    except ValueError:
                        continue
                    I = T.identity()
                    expected = {
                        'forward':  T.basis_forward(),
                        'backward': T.basis_backward(),
                        'left':     T.basis_left(),
                        'right':    T.basis_right(),
                        'up':       T.basis_up(),
                        'down':     T.basis_down(),
                    }
                    for prop, exp in expected.items():
                        got = getattr(I, prop)
                        np.testing.assert_array_almost_equal(
                            got, exp,
                            err_msg=f"{T.__name__}.{prop} should be {exp}, got {got}"
                        )


class TestRotationEffects(unittest.TestCase):
    def setUp(self):
        # default convention: x→forward, y→left, z→up
        self.T0 = polyframe.define_convention(
            Direction.FORWARD, Direction.LEFT, Direction.UP
        )

    def test_yaw_90_about_z(self):
        # yaw=+90° rotates +X→+Y
        T = self.T0.from_euler_angles(roll=0, pitch=0, yaw=90)
        np.testing.assert_allclose(T.forward, [0, 1, 0], atol=1e-6)
        np.testing.assert_allclose(T.left,   [-1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(T.up,     [0, 0, 1], atol=1e-6)

    def test_pitch_90_about_y(self):
        # pitch rotates about Y: +X→−Z in this convention
        T = self.T0.from_euler_angles(roll=0, pitch=90, yaw=0)
        np.testing.assert_allclose(T.forward, [0, 0, -1], atol=1e-6)
        np.testing.assert_allclose(T.up,      [1, 0,  0], atol=1e-6)

    def test_roll_90_about_x(self):
        # roll rotates about X: +Y→−Z, +Z→+Y
        T = self.T0.from_euler_angles(roll=90, pitch=0, yaw=0)
        np.testing.assert_allclose(T.up,   [0, -1, 0], atol=1e-6)
        np.testing.assert_allclose(T.left, [0,  0, 1], atol=1e-6)


class TestScaleShearEffects(unittest.TestCase):
    def setUp(self):
        self.T0 = polyframe.define_convention(
            Direction.FORWARD, Direction.LEFT, Direction.UP
        )

    def test_scale_does_not_affect_directions(self):
        S = np.array([2.0, 3.0, 4.0])
        T = self.T0.from_values(scale=S)
        # scale only affects the P block, directions come from pure rotation
        np.testing.assert_array_equal(T.forward, self.T0.identity().forward)
        np.testing.assert_array_equal(T.left,    self.T0.identity().left)
        np.testing.assert_array_equal(T.up,      self.T0.identity().up)

    def test_shear_changes_directions(self):
        # shear generally induces a non‐trivial polar rotation
        shear = np.eye(3)
        shear[0, 1] = 0.5
        shear[1, 2] = -0.3
        T = self.T0.from_values(shear=shear)
        # we no longer assume shear leaves directions unchanged
        # but we can assert the result is exactly the polar‐rotation of the shear bar basis
        R_pure = polyframe._polyframe.pure_rotation_if_possible(
            np.array(shear, float))
        np.testing.assert_allclose(
            T.forward,  R_pure @ np.array([1, 0, 0]), atol=1e-6)
        np.testing.assert_allclose(
            T.left,     R_pure @ np.array([0, 1, 0]), atol=1e-6)
        np.testing.assert_allclose(
            T.up,       R_pure @ np.array([0, 0, 1]), atol=1e-6)


class TestBasisStaticsAndMatrix(unittest.TestCase):
    def setUp(self):
        self.T = polyframe.define_convention(
            Direction.FORWARD, Direction.LEFT, Direction.UP
        )

    def test_basis_vectors_match_direction_enum(self):
        # hard‐code the expected mapping
        expected = {
            'basis_x': np.array([1, 0, 0]),
            'basis_y': np.array([0, 1, 0]),
            'basis_z': np.array([0, 0, 1]),
        }
        for name, vec in expected.items():
            got = getattr(self.T, name)()
            np.testing.assert_array_equal(got, vec)


class TestHandednessAndChangeCoordinateSystem(unittest.TestCase):
    def test_is_right_and_left_handed(self):
        R = polyframe.define_convention(
            Direction.FORWARD, Direction.LEFT, Direction.UP
        )
        self.assertTrue(R.is_right_handed())
        self.assertFalse(R.is_left_handed())

        L = polyframe.define_convention(
            Direction.FORWARD, Direction.UP, Direction.LEFT
        )
        self.assertFalse(L.is_right_handed())
        self.assertTrue(L.is_left_handed())

    def test_change_coordinate_system_round_trip(self):
        # if we move tA into B’s axes and then back into A’s,
        # we should recover the original world‐directions
        A = polyframe.define_convention(
            Direction.FORWARD, Direction.LEFT, Direction.UP
        )
        tA = A.from_euler_angles(0, 0, 90)

        B = polyframe.define_convention(
            Direction.UP, Direction.RIGHT, Direction.FORWARD
        )
        # move into B’s local axes
        tB = tA.change_coordinate_system(B)
        # then move back into A’s axes
        tA2 = tB.change_coordinate_system(A)

        # now in A’s frame again, the world‐directions must match
        np.testing.assert_allclose(tA2.forward, tA.forward, atol=1e-6)
        np.testing.assert_allclose(tA2.left,    tA.left,    atol=1e-6)
        np.testing.assert_allclose(tA2.up,      tA.up,      atol=1e-6)


class TestGeneratedStaticsAndLabels(unittest.TestCase):
    def setUp(self):
        # pick a “mixed” convention
        self.C = polyframe.define_convention(
            Direction.RIGHT, Direction.DOWN, Direction.BACKWARD
        )

    def test_labels(self):
        # label_x/y/z must return exactly the enum you passed
        self.assertIs(self.C.label_x(), Direction.RIGHT)
        self.assertIs(self.C.label_y(), Direction.DOWN)
        self.assertIs(self.C.label_z(), Direction.BACKWARD)

    def test_handedness(self):
        # this one is left‐handed by construction (x × y = +z ?)
        self.assertFalse(self.C.is_right_handed())
        self.assertTrue(self.C.is_left_handed())

    def test_registry_size_and_naming(self):
        # 48 orthonormal triples
        self.assertEqual(len(polyframe._polyframe._FRAME_REGISTRY), 48)
        # check that the class names encode the directions
        name = self.C.__name__
        self.assertIn("RIGHT", name)
        self.assertIn("DOWN", name)
        self.assertIn("BACKWARD", name)


class TestBasisMatrixAndVectors(unittest.TestCase):
    def setUp(self):
        self.D = polyframe.define_convention(
            Direction.LEFT, Direction.BACKWARD, Direction.UP
        )

    def test_basis_matrix_inverse(self):
        B = self.D.basis_matrix()
        Binv = self.D.basis_matrix_inv()
        np.testing.assert_allclose(B @ Binv, np.eye(3), atol=1e-8)

    def test_basis_vectors_consistency(self):
        # rows of basis_matrix should match basis_x, basis_y, basis_z
        B = self.D.basis_matrix()
        np.testing.assert_array_equal(B[0], self.D.basis_x())
        np.testing.assert_array_equal(B[1], self.D.basis_y())
        np.testing.assert_array_equal(B[2], self.D.basis_z())

    def test_forward_backward_symmetry(self):
        f = self.D.basis_forward()
        b = self.D.basis_backward()
        np.testing.assert_array_equal(b, -f)
        l = self.D.basis_left()
        r = self.D.basis_right()
        np.testing.assert_array_equal(r, -l)
        u = self.D.basis_up()
        d = self.D.basis_down()
        np.testing.assert_array_equal(d, -u)


class TestPropertyTranslationIndependence(unittest.TestCase):
    def setUp(self):
        self.T = polyframe.define_convention(
            Direction.FORWARD, Direction.LEFT, Direction.UP
        ).identity()

    def test_translation_does_not_change_directions(self):
        moved = self.T.apply_translation([5, -3, 2])
        for prop in ("forward", "left", "up", "backward", "right", "down"):
            np.testing.assert_array_equal(
                getattr(moved, prop),
                getattr(self.T, prop)
            )


class TestCompositionAndDirections(unittest.TestCase):
    def setUp(self):
        self.T0 = polyframe.define_convention(
            Direction.FORWARD, Direction.LEFT, Direction.UP
        )

    def test_composing_rotations(self):
        # yaw 90 then yaw 90 again should be equivalent to yaw 180
        t1 = self.T0.from_euler_angles(0, 0, 90)
        t2 = self.T0.from_euler_angles(0, 0, 90)
        combined = t2 @ t1
        direct = self.T0.from_euler_angles(0, 0, 180)
        np.testing.assert_allclose(combined.forward, direct.forward, atol=1e-6)
        np.testing.assert_allclose(combined.left,    direct.left,    atol=1e-6)
        np.testing.assert_allclose(combined.up,      direct.up,      atol=1e-6)

    def test_change_coordinate_system_type(self):
        A = self.T0
        B = polyframe.define_convention(
            Direction.UP, Direction.RIGHT, Direction.FORWARD
        )
        tA = A.identity()
        tB = tA.change_coordinate_system(B)
        # the result must be an instance of B
        self.assertIsInstance(tB, B)
        # round‐trip back to A yields A.identity()
        tA2 = tB.change_coordinate_system(A)
        np.testing.assert_allclose(tA2.matrix, A.identity().matrix)


class TestInvalidConventions(unittest.TestCase):
    def test_define_convention_rejects_colinear(self):
        with self.assertRaises(ValueError):
            polyframe.define_convention(
                Direction.FORWARD, Direction.BACKWARD, Direction.UP
            )
        with self.assertRaises(ValueError):
            polyframe.define_convention(
                Direction.UP, Direction.UP, Direction.RIGHT
            )


if __name__ == "__main__":
    unittest.main()
