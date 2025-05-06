# tests.py
import unittest
import numpy as np
import pickle
from polyframe import Transform, FrameRegistry, Direction

X_FORWARD_Z_UP = FrameRegistry.from_directions(
    Direction.FORWARD, Direction.LEFT, Direction.UP
)


class TestFrameRegistryExtras(unittest.TestCase):
    def test_str_and_repr(self):
        cs = FrameRegistry.from_directions(
            Direction.UP, Direction.RIGHT, Direction.BACKWARD)
        # Both __str__ and __repr__ should mention the same triple
        self.assertIn("UP", str(cs))
        self.assertIn("UP", repr(cs))
        self.assertEqual(str(cs), repr(cs))

    def test_pickle_roundtrip(self):
        # Make sure the CCS is pickleable and comes back equal
        cs = FrameRegistry.from_directions(
            Direction.FORWARD, Direction.LEFT, Direction.DOWN)
        s = pickle.dumps(cs)
        cs2 = pickle.loads(s)
        self.assertEqual(cs2, cs)


class TestTransformExtras2(unittest.TestCase):
    def test_pickle_roundtrip(self):
        tr = Transform.from_values(
            translation=np.array([3, 4, 5]),
            rotation=np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        )
        s = pickle.dumps(tr)
        tr2 = pickle.loads(s)
        np.testing.assert_array_equal(tr2.matrix, tr.matrix)
        self.assertEqual(tr2.coordinate_system, tr.coordinate_system)

    def test_change_basis_to_input_transformed(self):
        """
        If you want to preserve the *same world point* you must
        re‐express the input in the new coordinate system before feeding
        it to the converted transform.
        """
        cs1 = X_FORWARD_Z_UP
        # pick a frame that's rotated 90° about Z
        cs2 = FrameRegistry.from_directions(
            Direction.LEFT, Direction.BACKWARD, Direction.UP
        )
        tr = Transform.from_values(translation=np.array(
            [5, 0, 0]), coordinate_system=cs1)
        # world point of (1,2,3) in frame 1:
        world = tr.transform_point(np.array([1, 2, 3]))
        # to get same world via tr2 we must feed in the *coords of [1,2,3]* expressed in cs2:
        tr2 = tr.change_basis_to(cs2)
        # compute how to get coords2 so that tr2.coords2 → world
        # tr2 is R @ p + t  so coords2 = R^T @ (p_world - t)
        R = FrameRegistry.change_of_basis(cs1, cs2)
        coords2 = R.T @ (world - tr2.translation)
        world2 = tr2.transform_point(coords2)
        np.testing.assert_allclose(world2, world, atol=1e-8)

    def test_invalid_matmul_operand(self):
        """If you @ something that isn’t a Transform or array, you get NotImplemented."""
        tr = Transform()

        class Foo:
            pass
        with self.assertRaises(TypeError):
            _ = tr @ Foo()   # should bubble up NotImplemented → TypeError


class TestTransformExtras(unittest.TestCase):

    def test_T_property(self):
        """T should return the matrix transpose."""
        R = np.random.randn(4, 4)
        tr = Transform(R, X_FORWARD_Z_UP)
        np.testing.assert_array_equal(tr.T, R.T)

    def test_matmul_with_array(self):
        """__matmul__ with raw numpy array should apply the transform matrix."""
        tr = Transform.from_values(translation=np.array([1, 2, 3]))
        A = np.eye(4) * 2
        result = tr @ A
        np.testing.assert_array_equal(result, tr.matrix @ A)

    def test_matmul_frame_mismatch(self):
        """__matmul__ between transforms in different frames should reframe second."""
        cs1 = X_FORWARD_Z_UP
        cs2 = FrameRegistry.from_directions(
            Direction.RIGHT, Direction.BACKWARD, Direction.UP
        )
        A = Transform.from_values(translation=np.array(
            [1, 0, 0]), coordinate_system=cs1)
        B = Transform.from_values(translation=np.array(
            [0, 2, 0]), coordinate_system=cs2)

        # the composition under test
        C = A @ B

        # manual reframe of B into A’s frame:
        R3 = FrameRegistry.change_of_basis(cs2, cs1)  # 3×3 ref-frame rotation
        T4 = np.eye(4, dtype=R3.dtype)             # lift to homogeneous
        T4[:3, :3] = R3
        B_reframed = Transform(T4 @ B.matrix, cs1)

        expected = Transform(A.matrix @ B_reframed.matrix, cs1)

        np.testing.assert_array_equal(C.matrix, expected.matrix)
        self.assertEqual(C.coordinate_system, cs1)

    def test_change_basis_to_reexpress_input(self):
        """
        To get the *same world point* under a converted transform,
        you must first re‐express your input in the new frame.
        """
        cs1 = X_FORWARD_Z_UP
        cs2 = FrameRegistry.from_directions(
            Direction.LEFT, Direction.DOWN, Direction.BACKWARD
        )
        tr = Transform.from_values(translation=np.array(
            [5, 0, 0]), coordinate_system=cs1)
        p = np.array([1, 2, 3])
        world = tr.transform_point(p)

        tr2 = tr.change_basis_to(cs2)
        # compute the coords in cs2 that represent the *same world point*:
        # world = tr2.rotation @ q + tr2.translation  ⇒  q = tr2.rotation.T @ (world - tr2.translation)
        q = tr2.rotation.T @ (world - tr2.translation)
        world2 = tr2.transform_point(q)
        np.testing.assert_allclose(world2, world, atol=1e-8)

    def test_roundtrip_inverse(self):
        """Both inverse() and inverse(inplace=True) should round-trip correctly."""

        # a known rigid-body transform
        R = np.array([[0, -1, 0],
                      [1,  0, 0],
                      [0,  0, 1]], float)
        t = np.array([1.0, 2.0, 3.0])

        # ---- non-inplace ----
        tr = Transform.from_values(translation=t, rotation=R)
        orig = tr.matrix.copy()

        inv = tr.inverse()  # non-inplace
        # original tr.matrix must be unchanged
        np.testing.assert_allclose(tr.matrix, orig, atol=1e-8)
        # inv.matrix @ orig == I
        np.testing.assert_allclose(inv.matrix @ orig, np.eye(4), atol=1e-8)

        inv2 = inv.inverse()  # non-inplace again
        # inv stays unchanged
        np.testing.assert_allclose(
            inv.matrix @ inv2.matrix, np.eye(4), atol=1e-8)
        # and inv2 is back to orig
        np.testing.assert_allclose(inv2.matrix, orig, atol=1e-8)

        # ---- inplace ----
        tr_ip = Transform.from_values(translation=t, rotation=R)
        orig_ip = tr_ip.matrix.copy()

        inv_ip = tr_ip.inverse(inplace=True)
        # tr_ip (and inv_ip) now holds the inverse of orig_ip
        np.testing.assert_allclose(
            inv_ip.matrix @ orig_ip, np.eye(4), atol=1e-8)

        inv2_ip = tr_ip.inverse(inplace=True)
        # double-invert should restore to the original
        np.testing.assert_allclose(inv2_ip.matrix, orig_ip, atol=1e-8)

    def test_scale_then_rotate_vs_rotate_then_scale(self):
        """Ensure non‐commutativity of scale and rotate is respected."""
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        S = np.array([2.0, 3.0, 4.0])
        tr_sr = Transform.from_values().apply_scale(S).apply_rotation(R)
        tr_rs = Transform.from_values().apply_rotation(R).apply_scale(S)
        # The resulting matrices should differ
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(tr_sr.matrix, tr_rs.matrix)


class TestFrameRegistry(unittest.TestCase):
    def test_basic_directions(self):
        cs = FrameRegistry.from_directions(
            Direction.FORWARD, Direction.LEFT, Direction.UP
        )
        # enum properties
        self.assertEqual(cs.x, Direction.FORWARD)
        self.assertEqual(cs.y, Direction.LEFT)
        self.assertEqual(cs.z, Direction.UP)
        # vector properties
        self.assertTrue(np.all(cs.forward == np.array((1, 0, 0))))
        self.assertTrue(np.all(cs.left == np.array((0, 1, 0))))
        self.assertTrue(np.all(cs.up == np.array((0, 0, 1))))
        self.assertTrue(cs.right_handed)

    def test_from_directions_all_permutations(self):
        # all valid direction triples produce a byte in VALID_BYTES
        seen = set()
        for x in Direction:
            for y in Direction:
                for z in Direction:
                    try:
                        cs = FrameRegistry.from_directions(x, y, z)
                    except KeyError:
                        continue
                    seen.add(cs)
        self.assertEqual(len(seen), 48)

    def test_rotation_to_identity_and_det(self):
        from itertools import permutations
        # helper to build all valid frames

        def all_frames():
            frames = []
            for x, y, z in permutations(list(Direction), 3):
                try:
                    cs = FrameRegistry.from_directions(x, y, z)
                except KeyError:
                    continue
                # determine handedness tag
                fwd = cs.forward
                lft = cs.left
                upv = cs.up
                det = np.linalg.det(np.stack((fwd, lft, upv), axis=1))
                tag = "RH" if det > 0 else "LH"
                frames.append((tag, cs))
            return frames

        frames = all_frames()
        n = len(frames)
        print(
            f"Testing rotation_to across {n} valid frames → {n} valid frames ({n*n} pairs)")
        tol = 1e-16

        for i, (tag1, cs1) in enumerate(frames):
            for j, (tag2, cs2) in enumerate(frames):
                cs1: FrameRegistry
                R = FrameRegistry.change_of_basis(cs1, cs2)

                # 1) basis mapping
                for vec_name in ("forward", "left", "up"):
                    v1 = getattr(cs1, vec_name)
                    v2 = getattr(cs2, vec_name)
                    mapped = R @ v1
                    self.assertTrue(np.allclose(
                        mapped, v2, atol=tol), f"[{i}->{j}] mapping {vec_name} failed:\n{tag1}→{tag2}, R@{vec_name} = {mapped}, expected {v2}")

                # 2) determinant
                detR = np.linalg.det(R)
                expected = +1.0 if tag1 == tag2 else -1.0
                self.assertTrue(np.isclose(detR, expected, atol=tol),
                                f"[{i}->{j}] det(R)={detR}, expected {expected} (tags {tag1}->{tag2})")


class TestTransform(unittest.TestCase):
    def setUp(self):
        self.cs = X_FORWARD_Z_UP
        self.identity = Transform()

    def test_identity_matrix(self):
        np.testing.assert_array_equal(self.identity.matrix, np.eye(4))

    def test_from_values_translation(self):
        t = np.array([1.0, 2.0, 3.0])
        tr = Transform.from_values(translation=t)
        np.testing.assert_array_equal(tr.translation, t)
        np.testing.assert_array_equal(tr.rotation, np.eye(3))

    def test_from_values_rotation(self):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
        tr = Transform.from_values(rotation=R)
        np.testing.assert_array_equal(tr.rotation, R)
        np.testing.assert_array_equal(tr.translation, np.zeros(3))

    def test_from_values_scale(self):
        s = np.array([2.0, 3.0, 4.0])
        tr = Transform.from_values(scale=s)
        expected = np.diag([2.0, 3.0, 4.0, 1.0])
        np.testing.assert_array_equal(tr.matrix, expected)

    def test_translate_method(self):
        tr = self.identity.apply_translation(np.array([5, 6, 7]))
        np.testing.assert_array_equal(tr.translation, [5, 6, 7])

    def test_rotate_method(self):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
        tr = self.identity.apply_rotation(R)
        np.testing.assert_array_equal(tr.rotation, R)

    def test_scale_method(self):
        s = np.array([2, 3, 4])
        tr = self.identity.apply_scale(s)
        np.testing.assert_array_equal(tr.scaler, s)

    def test_inverse(self):
        # compose translate+rotate then invert
        tr = self.identity.apply_translation(np.array([1, 2, 3])) \
            .apply_rotation(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
        inv = tr.inverse()
        # tr @ inv == identity
        I = tr.matrix @ inv.matrix
        np.testing.assert_allclose(I, np.eye(4), atol=1e-8)

    def test_transform_point_and_vector(self):
        tr = Transform.from_values(translation=np.array([1, 2, 3]))
        p = tr.transform_point(np.array([1, 1, 1]))
        self.assertTrue(np.allclose(p, [2, 3, 4]))
        v = tr.transform_vector(np.array([1, 0, 0]))
        self.assertTrue(np.allclose(v, [1, 0, 0]))  # unaffected by translation

    def test_change_basis_to(self):
        cs1 = FrameRegistry.from_directions(
            Direction.FORWARD, Direction.LEFT, Direction.UP
        )
        cs2 = FrameRegistry.from_directions(
            Direction.RIGHT, Direction.BACKWARD, Direction.UP
        )
        tr = Transform(coordinate_system=cs1)
        tr2 = tr.change_basis_to(cs2)
        # new coordinate system should be cs2
        self.assertEqual(tr2.coordinate_system, cs2)

    def test_matmul_with_transform(self):
        A = Transform.from_values(translation=np.array([1, 0, 0]))
        B = Transform.from_values(translation=np.array([0, 2, 0]))
        C = A @ B
        np.testing.assert_array_equal(C.translation, [1, 2, 0])


class TestOrientTo(unittest.TestCase):
    def setUp(self):
        # identity pose at the origin, default forward = +X
        self.tr = Transform()

    def assertForward(self, tr: Transform, expected: np.ndarray, tol=1e-6):
        """Helper: check that tr.forward ≈ expected."""
        np.testing.assert_allclose(tr.rotation @ tr.coordinate_system.forward,
                                   expected, atol=tol)

    def test_noop_when_already_facing(self):
        # Target out along +X; identity already faces that way
        target = np.array([10.0, 0.0, 0.0])
        tr2 = self.tr.look_at(target)
        # rotation should be identity
        np.testing.assert_allclose(tr2.rotation, np.eye(3), atol=1e-6)
        # forward remains +X
        self.assertForward(tr2, np.array([1.0, 0.0, 0.0]))
        # translation unchanged
        np.testing.assert_allclose(tr2.translation, self.tr.translation)

    def test_90deg_turn_upwards(self):
        # Point up along +Y
        target = np.array([0.0, 5.0, 0.0])
        tr2 = self.tr.look_at(target)
        # new forward should be ≈ +Y
        self.assertForward(tr2, np.array([0.0, 1.0, 0.0]))
        # ensure rotation is orthonormal + det=+1
        R = tr2.rotation
        det = np.linalg.det(R)
        self.assertAlmostEqual(det, +1.0, places=6)

    def test_180deg_flip(self):
        # Point directly behind along -X
        target = np.array([-3.0, 0.0, 0.0])
        tr2 = self.tr.look_at(target)
        # forward ≈ -X
        self.assertForward(tr2, np.array([-1.0, 0.0, 0.0]))
        # rotation should be 180° about some perpendicular axis; det=+1
        det = np.linalg.det(tr2.rotation)
        self.assertAlmostEqual(det, +1.0, places=6)

    def test_rotation_to_transform_target(self):
        # place a small translation on the target
        Tgt = Transform.from_values(translation=np.array([0.0, 0.0, 7.0]))
        tr2 = self.tr.look_at(Tgt)
        # since target is straight “up” in world Z, forward vector points toward (0,0,7)
        # i.e. along Z axis
        self.assertForward(tr2, np.array([0.0, 0.0, 1.0]))
        # translation unchanged
        np.testing.assert_allclose(tr2.translation, self.tr.translation)

    def test_preserves_translation(self):
        # start with a nonzero translation
        tr = Transform.from_values(translation=np.array([4.0, 5.0, 6.0]))
        # target somewhere else
        target = np.array([4.0, 5.0, 10.0])
        tr2 = tr.look_at(target)
        # translation must remain the same
        np.testing.assert_allclose(tr2.translation, [4.0, 5.0, 6.0])
        # forward should point from original translation toward target
        desired = target - tr.translation
        desired /= np.linalg.norm(desired)
        self.assertForward(tr2, desired)

###############################################################################
#  Additional, targeted edge‑case tests
###############################################################################


class TestOrientToInplace(unittest.TestCase):
    """Exercise the `inplace=True` branch of look_at."""

    def test_inplace_updates_and_returns_same_object(self):
        tr = Transform()                       # faces +X
        target = np.array([0.0, 8.0, 0.0])         # lies on +Y
        out_ref = tr.look_at(target, inplace=True)

        # must *be* the same instance
        self.assertIs(out_ref, tr)

        # forward now points at +Y
        np.testing.assert_allclose(tr.forward, [0.0, 1.0, 0.0], atol=1e-6)
        # translation unchanged
        np.testing.assert_array_equal(tr.translation, [0.0, 0.0, 0.0])


class TestAzElRangeVertical(unittest.TestCase):
    """Straight‑up / straight‑down targets should yield az=0, |el|=90°."""

    def setUp(self):
        self.origin = Transform()                   # at (0,0,0), +X fwd, +Z up

    def test_straight_up(self):
        az, el, rng = self.origin.az_el_range_to([0, 0, 5])
        self.assertAlmostEqual(az, 0.0, places=6)
        self.assertAlmostEqual(el, 90.0, places=6)
        self.assertAlmostEqual(rng, 5.0,  places=6)

    def test_straight_down(self):
        az, el, rng = self.origin.az_el_range_to([0, 0, -7])
        self.assertAlmostEqual(az, 0.0,   places=6)
        self.assertAlmostEqual(el, -90.0, places=6)
        self.assertAlmostEqual(rng, 7.0,  places=6)


class TestAzElRangeOptions(unittest.TestCase):
    """Verify azimuth sign conventions and radian mode."""

    def setUp(self):
        self.origin = Transform()                 # +X forward ; +Y left ; –Y right
        self.to_right = np.array([0.0, -10.0, 0.0])  # pure “right” direction

    def test_clockwise_vs_counterclockwise(self):
        # default (clockwise, unsigned) → –90° wrapped to 270°
        az, _, _ = self.origin.az_el_range_to(self.to_right)
        self.assertAlmostEqual(az, 270.0, places=6)

        # CCW convention with signed output → +90°
        az_ccw, _, _ = self.origin.az_el_range_to(
            self.to_right,
            counterclockwise_azimuth=True,
            signed_azimuth=True
        )
        self.assertAlmostEqual(az_ccw, 90.0, places=6)

    def test_radian_mode(self):
        # default (clockwise, unsigned) in radians → –π/2 wrapped to 3π/2
        az_rad, el_rad, _ = self.origin.az_el_range_to(
            self.to_right,
            degrees=False
        )
        self.assertAlmostEqual(az_rad, 3*np.pi/2, places=6)
        self.assertAlmostEqual(el_rad, 0.0,        places=6)


class TestAzElRangeExhaustive(unittest.TestCase):
    def setUp(self):
        cs = FrameRegistry.from_directions(
            Direction.FORWARD, Direction.LEFT, Direction.UP
        )
        self.origin = Transform(coordinate_system=cs)
        # six cardinal directions, unit length
        self.targets = {
            'forward':  np.array(cs.forward),
            'backward': np.array(cs.backward),
            'left':     np.array(cs.left),
            'right':    np.array(cs.right),
            'up':       np.array(cs.up),
            'down':     np.array(cs.down),
        }

    def test_all_combinations(self):
        import itertools
        # iterate all keyword‐arg combos
        for degrees, signed, ccw, flip in itertools.product(
            [True, False],   # degrees vs radians
            [True, False],   # signed_azimuth
            [True, False],   # counterclockwise_azimuth
            [True, False],   # flip_elevation
        ):
            period = 360.0 if degrees else 2*np.pi
            for name, vec in self.targets.items():
                az, el, rng = self.origin.az_el_range_to(
                    vec,
                    degrees=degrees,
                    signed_azimuth=signed,
                    counterclockwise_azimuth=ccw,
                    flip_elevation=flip
                )

                # 1) Range must be exactly 1.0 for unit‐length targets
                self.assertAlmostEqual(rng, 1.0, places=8,
                                       msg=f"range for {name} failed ({degrees=},{signed=},{ccw=},{flip=})")

                # 2) Azimuth is in correct interval
                if signed:
                    low, high = (-180.0, 180.0) if degrees else (-np.pi, np.pi)
                else:
                    low, high = (0.0, period)
                self.assertTrue(low <= az <= high,
                                msg=f"azimuth out of bounds for {name}: {az} not in [{low},{high})")

                # 3) flip_elevation truly negates elevation
                #    compare flip vs non‐flip, other flags held constant
                el_noflip = self.origin.az_el_range_to(
                    vec,
                    degrees=degrees,
                    signed_azimuth=signed,
                    counterclockwise_azimuth=ccw,
                    flip_elevation=False
                )[1]
                el_flip = self.origin.az_el_range_to(
                    vec,
                    degrees=degrees,
                    signed_azimuth=signed,
                    counterclockwise_azimuth=ccw,
                    flip_elevation=True
                )[1]
                self.assertAlmostEqual(el_flip, -el_noflip, places=6,
                                       msg=f"flip_elevation failed for {name}")

                # 4) CW vs CCW consistency:
                #    compute both in unsigned (signed=False) mode,
                #    they should sum to 0 mod period
                az_cw = self.origin.az_el_range_to(vec, degrees=degrees,
                                                   signed_azimuth=False,
                                                   counterclockwise_azimuth=False,
                                                   flip_elevation=False)[0]
                az_ccw = self.origin.az_el_range_to(vec, degrees=degrees,
                                                    signed_azimuth=False,
                                                    counterclockwise_azimuth=True,
                                                    flip_elevation=False)[0]
                self.assertAlmostEqual((az_cw + az_ccw) % period, 0.0, places=6,
                                       msg=f"CW/CCW symmetry failed for {name}")

                # 5) Degrees ↔ radians consistency
                #    if we convert this degrees result to radians we get the radian‐mode result
                if degrees:
                    az_rad, el_rad, _ = self.origin.az_el_range_to(
                        vec,
                        degrees=False,
                        signed_azimuth=signed,
                        counterclockwise_azimuth=ccw,
                        flip_elevation=flip
                    )
                    self.assertAlmostEqual(az_rad, az * np.pi/180.0, places=6,
                                           msg=f"az degree→rad mismatch for {name}")
                    self.assertAlmostEqual(el_rad, el * np.pi/180.0, places=6,
                                           msg=f"el degree→rad mismatch for {name}")


class TestPhiThetaTo(unittest.TestCase):
    def setUp(self):
        # identity at origin, default frame X-forward, Z-up
        self.tr = Transform()

    def test_default_cardinals(self):
        # (vector → expected φ,θ) in degrees, polar convention
        data = {
            (1,  0,  0): (0.0,  90.0),  # forward
            (0,  1,  0): (90.0,  90.0),  # left
            (-1,  0,  0): (180.0,  90.0),  # backward
            (0, -1,  0): (270.0,  90.0),  # right
            (0,  0,  1): (0.0,   0.0),  # up
            (0,  0, -1): (0.0, 180.0),  # down
        }
        for vec, (exp_phi, exp_theta) in data.items():
            phi, th = self.tr.phi_theta_to(np.array(vec))
            self.assertAlmostEqual(phi, exp_phi, places=6,
                                   msg=f"φ for {vec}")
            self.assertAlmostEqual(th,  exp_theta, places=6,
                                   msg=f"θ for {vec}")

    def test_elevation_mode(self):
        # elevation (θ measured from horizontal) instead of polar
        # forward → θ = 0, up → θ = 90
        _, th_fwd = self.tr.phi_theta_to(np.array([1, 0, 0]), polar=False)
        _, th_up = self.tr.phi_theta_to(np.array([0, 0, 1]), polar=False)
        self.assertAlmostEqual(th_fwd, 0.0, places=6)
        self.assertAlmostEqual(th_up,  90.0, places=6)

    def test_signed_and_counterclockwise_phi(self):
        # left vector, signed φ in [-180,180], CCW convention → +90
        phi, _ = self.tr.phi_theta_to(
            np.array([0, 1, 0]),
            signed_phi=True,
            counterclockwise_phi=True
        )
        self.assertAlmostEqual(phi, 90.0, places=6)

    def test_flip_theta(self):
        # forward polar θ=90 → flipped → -90
        _, th = self.tr.phi_theta_to([1, 0, 0], flip_theta=True)
        self.assertAlmostEqual(th, -90.0, places=6)

    def test_internal_njit_matches_public(self):
        # compare non‐degree mode of public vs static _phi_theta_to
        vec = np.random.randn(3)
        phi1, th1 = self.tr.phi_theta_to(vec, degrees=False)
        up = self.tr.up
        lat = self.tr.right   # note lateral = right
        fwd = self.tr.forward
        # args: vec, up, lateral, forward, degrees, signed_phi, counterclockwise_phi, polar, flip_theta
        phi2, th2 = Transform._phi_theta_to(
            vec, up, lat, fwd,
            False, False, True, True, False
        )
        self.assertAlmostEqual(phi1, phi2, places=6)
        self.assertAlmostEqual(th1, th2, places=6)


class TestLatLonTo(unittest.TestCase):
    def setUp(self):
        self.tr = Transform()

    def test_default_cardinals(self):
        # (vector → expected lat,lon) in degrees
        data = {
            (1,  0,  0): (0.0,   0.0),  # forward
            (0,  1,  0): (0.0,  -90.0),  # left
            (-1,  0,  0): (0.0, 180.0),  # backward
            (0, -1,  0): (0.0, 90),  # right
            (0,  0,  1): (90.0,   0.0),  # up
            (0,  0, -1): (-90.0,   0.0),  # down
        }
        for vec, (exp_lat, exp_lon) in data.items():
            lat, lon = self.tr.lat_lon_to(np.array(vec))
            self.assertAlmostEqual(lat, exp_lat, places=6,
                                   msg=f"lat for {vec}")
            self.assertAlmostEqual(lon, exp_lon, places=6,
                                   msg=f"lon for {vec}")

    def test_signed_and_ccw_longitude(self):
        # backward: default lon=180; with signed & CCW, 180→ -180
        lat, lon = self.tr.lat_lon_to(
            np.array([-1, 0, 0]),
            signed_longitude=True,
            counterclockwise_longitude=True
        )
        self.assertAlmostEqual(lat, 0.0,   places=6)
        self.assertAlmostEqual(lon, 180.0, places=6)

    def test_flip_latitude(self):
        # up → lat=90; flip_latitude → -90
        lat, _ = self.tr.lat_lon_to([0, 0, 1], flip_latitude=True)
        self.assertAlmostEqual(lat, -90.0, places=6)

    def test_internal_njit_matches_public(self):
        vec = np.random.randn(3)
        lat1, lon1 = self.tr.lat_lon_to(vec, degrees=False)
        up = self.tr.up
        lat_axis = self.tr.right   # lateral axis is right for longitude
        fwd = self.tr.forward
        # args: vec, up, lateral, forward, degrees, signed_longitude, counterclockwise_longitude, flip_latitude
        lat2, lon2 = Transform._latitude_longitude_to(
            vec, up, lat_axis, fwd,
            False, True, True, False
        )
        self.assertAlmostEqual(lat1, lat2, places=6)
        self.assertAlmostEqual(lon1, lon2, places=6)


if __name__ == "__main__":
    unittest.main()
