import unittest
import numpy as np
import polyframe
from polyframe import Direction
from polyframe._polyframe import quaternion_to_rotation


class TestTargetMethodsExhaustive(unittest.TestCase):
    def setUp(self):
        # default FLU convention: X→FORWARD, Y→LEFT, Z→UP
        F = polyframe.define_convention(
            Direction.FORWARD, Direction.LEFT, Direction.UP
        )
        self.I = F.identity()
        self.T = self.I.apply_translation([1.0, -2.0, 0.5])  # nonzero origin

    # ---- distance_to & vector_to ----

    def test_distance_and_vector_array_vs_transform(self):
        pts = [
            (np.array([0, 0, 0]), np.linalg.norm([0, 0, 0])),
            (np.array([3, 4, 0]), 5.0),
        ]
        for arr, dist in pts:
            # from I
            self.assertAlmostEqual(self.I.distance_to(arr), dist, places=7)
            # as Transform
            Tarr = self.I.apply_translation(arr)
            self.assertAlmostEqual(self.I.distance_to(Tarr), dist, places=7)
            # vector_to
            np.testing.assert_allclose(self.I.vector_to(arr), arr, atol=1e-7)
            np.testing.assert_allclose(self.I.vector_to(Tarr), arr, atol=1e-7)

    def test_zero_distance_and_vector(self):
        # distance zero to itself
        self.assertAlmostEqual(self.I.distance_to(self.I), 0.0, places=7)
        np.testing.assert_array_equal(self.I.vector_to(self.I), np.zeros(3))

    # ---- direction_to ----

    def test_direction_normalized(self):
        v = np.array([0.0, 3.0, 4.0])
        dir_arr = self.I.direction_to(v)
        np.testing.assert_allclose(dir_arr, v/5.0, atol=1e-7)

    def test_direction_transform_input(self):
        Tv = self.I.apply_translation([0, 3, 4])
        dir_tr = self.I.direction_to(Tv)
        np.testing.assert_allclose(dir_tr, np.array([0, 3, 4])/5.0, atol=1e-7)

    def test_direction_zero_fallback(self):
        # exactly zero vector
        np.testing.assert_array_equal(
            self.I.direction_to([0, 0, 0]), self.I.forward)
        # distance < 1e-8 (just inside threshold)
        tiny = np.array([1e-9, 0, 0])
        np.testing.assert_array_equal(
            self.I.direction_to(tiny), self.I.forward)

    # ---- rotation_to / quaternion_to / euler_angles_to ----

    def test_rotation_to_zero(self):
        R = self.I.rotation_to([0, 0, 0])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-7)

    def test_rotation_to_cardinal_axes(self):
        for target in (
            np.array([1, 0, 0]),   # forward
            np.array([0, 1, 0]),   # left
            np.array([0, 0, 1]),   # up
            np.array([-1, 0, 0]),  # backward
            np.array([0, -1, 0]),  # right
            np.array([0, 0, -1]),  # down
        ):
            R = self.I.rotation_to(target)
            # new forward should align
            np.testing.assert_allclose(
                R.dot(self.I.basis_forward()),
                target/np.linalg.norm(target),
                atol=1e-7
            )

    def test_quaternion_to_matches_rotation_to(self):
        v = np.array([2.0, -1.0, 5.0])
        R = self.I.rotation_to(v)
        for w_last in (True, False):
            q = self.I.quaternion_to(v, w_last=w_last)
            Rq = quaternion_to_rotation(q, w_last=w_last)
            np.testing.assert_allclose(Rq, R, atol=1e-7)

    def test_euler_angles_to_degrees_and_radians(self):
        # pitch-down: target +Z gives roll=0, pitch=-90, yaw=0
        roll, pitch, yaw = self.I.euler_angles_to([0, 0, 1], degrees=True)
        self.assertAlmostEqual(roll,    0.0,   places=6)
        self.assertAlmostEqual(pitch, -90.0,   places=6)
        self.assertAlmostEqual(yaw,     0.0,   places=6)

        r2, p2, y2 = self.I.euler_angles_to([0, 0, 1], degrees=False)
        self.assertAlmostEqual(p2, -np.pi/2,   places=6)

    # ---- azimuth_elevation_to ----

    def test_az_el_zero_vector(self):
        az, el = self.I.azimuth_elevation_to([0, 0, 0])
        self.assertEqual((az, el), (0.0, 0.0))

    def test_az_el_cardinals(self):
        # +X
        az, el = self.I.azimuth_elevation_to([5, 0, 0])
        self.assertAlmostEqual(az, 0.0, places=6)
        self.assertAlmostEqual(el, 0.0, places=6)
        # +Y
        az, el = self.I.azimuth_elevation_to([0, 5, 0])
        self.assertAlmostEqual(az,  90.0, places=6)
        self.assertAlmostEqual(el,   0.0, places=6)
        # +Z
        az, el = self.I.azimuth_elevation_to([0, 0, 5])
        self.assertAlmostEqual(az,  0.0, places=6)
        self.assertAlmostEqual(el, 90.0, places=6)
        # –Z
        az, el = self.I.azimuth_elevation_to([0, 0, -5])
        self.assertAlmostEqual(az,  0.0, places=6)
        self.assertAlmostEqual(el, -90.0, places=6)

    def test_az_el_hnorm_small(self):
        # put almost directly up but tiny horizontal part
        v = np.array([1e-9, 0, 5.0])
        az, el = self.I.azimuth_elevation_to(v)
        # h_norm<1e-8 branch → az=0
        self.assertAlmostEqual(az, 0.0, places=6)

    def test_az_el_flag_combinations(self):
        # signed & ccw flips the sign
        az1, _ = self.I.azimuth_elevation_to(
            [0, 5, 0],
            signed_azimuth=True,
            counterclockwise_azimuth=True
        )
        self.assertAlmostEqual(az1, -90.0, places=6)

        # flip_elevation makes up negative
        _, el2 = self.I.azimuth_elevation_to([0, 0, 5], flip_elevation=True)
        self.assertAlmostEqual(el2, -90.0, places=6)

        # radians mode
        azr, _ = self.I.azimuth_elevation_to([0, 5, 0], degrees=False)
        self.assertAlmostEqual(azr, np.pi/2, places=6)

    # ---- phi_theta_to ----

    def test_phi_theta_zero_vector(self):
        phi, theta = self.I.phi_theta_to([0, 0, 0])
        self.assertEqual((phi, theta), (0.0, 0.0))

    def test_phi_theta_cardinals_and_modes(self):
        # polar: +Z → (0°,0°)
        phi1, theta1 = self.I.phi_theta_to([0, 0, 5], polar=True)
        self.assertAlmostEqual(phi1, 0.0, places=6)
        self.assertAlmostEqual(theta1, 0.0, places=6)
        # polar: +Y → phi=90, theta=90
        phi2, theta2 = self.I.phi_theta_to([0, 5, 0], polar=True)
        self.assertAlmostEqual(phi2, 90.0, places=6)
        self.assertAlmostEqual(theta2, 90.0, places=6)
        # elevation mode: +Z → theta=90
        _, theta3 = self.I.phi_theta_to([0, 0, 5], polar=False)
        self.assertAlmostEqual(theta3, 90.0, places=6)

    def test_phi_theta_sign_wrap_and_flip(self):
        # signed_phi & ccw=False → wrap into (–π,π]
        phi4, _ = self.I.phi_theta_to(
            [-5, 0, 0], signed_phi=True, counterclockwise_phi=False)
        # should be about ±180
        self.assertTrue(abs(abs(phi4) - 180.0) < 1e-6)

        # flip_theta in elevation mode → negative
        _, theta5 = self.I.phi_theta_to(
            [0, 0, 5], polar=False, flip_theta=True)
        self.assertAlmostEqual(theta5, -90.0, places=6)

        # radians
        phir, thetar = self.I.phi_theta_to([0, 5, 0], degrees=False)
        self.assertAlmostEqual(phir, np.pi/2, places=6)
        self.assertAlmostEqual(thetar, np.pi/2, places=6)

    # ---- latitude_longitude_to ----

    def test_lat_lon_zero_vector(self):
        lat, lon = self.I.latitude_longitude_to([0, 0, 0])
        self.assertEqual((lat, lon), (0.0, 0.0))

    def test_lat_lon_cardinals(self):
        # north pole
        lat1, lon1 = self.I.latitude_longitude_to([0, 0, 5])
        self.assertAlmostEqual(lat1,  90.0, places=6)
        self.assertAlmostEqual(lon1,   0.0, places=6)
        # on equator, +Y
        lat2, lon2 = self.I.latitude_longitude_to([0, 5, 0])
        self.assertAlmostEqual(lat2,   0.0, places=6)
        self.assertAlmostEqual(lon2, -90.0, places=6)
        # +X backward (-X)
        _, lon3 = self.I.latitude_longitude_to(
            [-5, 0, 0], signed_longitude=False)
        self.assertAlmostEqual(lon3, 180.0, places=6)

    def test_lat_lon_flags_and_radians(self):
        # flip_latitude
        lat4, _ = self.I.latitude_longitude_to([0, 0, 5], flip_latitude=True)
        self.assertAlmostEqual(lat4, -90.0, places=6)
        # radians
        latr, lonr = self.I.latitude_longitude_to([0, 5, 0], degrees=False)
        self.assertAlmostEqual(latr, 0.0, places=6)
        self.assertAlmostEqual(lonr, -np.pi/2, places=6)


if __name__ == "__main__":
    unittest.main()
