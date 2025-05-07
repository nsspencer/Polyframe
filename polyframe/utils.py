# utils.py

import numpy as np
from numba import njit
from typing import Tuple

from numpy.linalg import norm as np_norm
from numpy import cross as np_cross
from numpy import eye as np_eye
from numpy import dot as np_dot
from numpy import array as np_array

import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


@njit
def quaternion_to_rotation(quaternion: np.ndarray, w_last: bool = True) -> np.ndarray:
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Args:
        quaternion: shape-(4,) array, either [x,y,z,w] if w_last=True,
                    or [w,x,y,z] if w_last=False.

    Returns:
        R: shape-(3,3) rotation matrix.
    """
    # unpack
    if w_last:
        x, y, z, w = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    else:
        w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]

    # precompute products
    xx = x*x
    yy = y*y
    zz = z*z
    xy = x*y
    xz = x*z
    yz = y*z
    wx = w*x
    wy = w*y
    wz = w*z

    R = np.empty((3, 3), dtype=np.float64)
    R[0, 0] = 1 - 2*(yy + zz)
    R[0, 1] = 2*(xy - wz)
    R[0, 2] = 2*(xz + wy)

    R[1, 0] = 2*(xy + wz)
    R[1, 1] = 1 - 2*(xx + zz)
    R[1, 2] = 2*(yz - wx)

    R[2, 0] = 2*(xz - wy)
    R[2, 1] = 2*(yz + wx)
    R[2, 2] = 1 - 2*(xx + yy)
    return R


@njit
def rotation_to_quaternion(rotation: np.ndarray, w_last: bool = True) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion.

    Args:
        rotation: shape-(3,3) rotation matrix.

    Returns:
        quaternion: shape-(4,), in [x,y,z,w] if w_last=True else [w,x,y,z].
    """
    tr = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
    qx = 0.0
    qy = 0.0
    qz = 0.0
    qw = 0.0

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (rotation[2, 1] - rotation[1, 2]) / S
        qy = (rotation[0, 2] - rotation[2, 0]) / S
        qz = (rotation[1, 0] - rotation[0, 1]) / S
    else:
        # find which major diagonal element has greatest value
        if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            S = np.sqrt(1.0 + rotation[0, 0] -
                        rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / S
            qx = 0.25 * S
            qy = (rotation[0, 1] + rotation[1, 0]) / S
            qz = (rotation[0, 2] + rotation[2, 0]) / S
        elif rotation[1, 1] > rotation[2, 2]:
            S = np.sqrt(1.0 + rotation[1, 1] -
                        rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / S
            qx = (rotation[0, 1] + rotation[1, 0]) / S
            qy = 0.25 * S
            qz = (rotation[1, 2] + rotation[2, 1]) / S
        else:
            S = np.sqrt(1.0 + rotation[2, 2] -
                        rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / S
            qx = (rotation[0, 2] + rotation[2, 0]) / S
            qy = (rotation[1, 2] + rotation[2, 1]) / S
            qz = 0.25 * S

    out = np.empty(4, dtype=np.float64)
    if w_last:
        out[0], out[1], out[2], out[3] = qx, qy, qz, qw
    else:
        out[0], out[1], out[2], out[3] = qw, qx, qy, qz
    return out


@njit
def rotation_to_euler(rotation: np.ndarray, degrees: bool = True) -> tuple[float, float, float]:
    """
    Convert a 3x3 rotation matrix to (roll-pitch-yaw) Euler angles.

    Returns angles [roll, pitch, yaw].
    """
    # pitch = asin(-R[2,0])
    sp = -rotation[2, 0]
    if sp > 1.0:
        sp = 1.0
    elif sp < -1.0:
        sp = -1.0
    pitch = np.arcsin(sp)

    # roll  = atan2( R[2,1],  R[2,2] )
    # yaw   = atan2( R[1,0],  R[0,0] )
    cp = np.cos(pitch)
    roll = np.arctan2(rotation[2, 1]/cp, rotation[2, 2]/cp)
    yaw = np.arctan2(rotation[1, 0]/cp, rotation[0, 0]/cp)

    if degrees:
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)

    return roll, pitch, yaw


@njit
def quaternion_to_euler(
    quaternion: np.ndarray,
    w_last: bool = True,
    degrees: bool = True
) -> Tuple[float, float, float]:
    """
    Convert a quaternion to Euler angles [roll, pitch, yaw].
    """
    R = quaternion_to_rotation(quaternion, w_last)
    return rotation_to_euler(R, degrees)


@njit
def euler_to_rotation(
        roll: float,
        pitch: float,
        yaw: float,
        degrees: bool = True) -> np.ndarray:
    """
    Convert Euler angles [roll, pitch, yaw] to a 3x3 rotation matrix.
    """
    if degrees:
        roll *= np.pi/180.0
        pitch *= np.pi/180.0
        yaw *= np.pi/180.0

    sr, cr = np.sin(roll),  np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw),   np.cos(yaw)

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = np.empty((3, 3), dtype=np.float64)
    R[0, 0] = cy*cp
    R[0, 1] = cy*sp*sr - sy*cr
    R[0, 2] = cy*sp*cr + sy*sr

    R[1, 0] = sy*cp
    R[1, 1] = sy*sp*sr + cy*cr
    R[1, 2] = sy*sp*cr - cy*sr

    R[2, 0] = -sp
    R[2, 1] = cp*sr
    R[2, 2] = cp*cr
    return R


@njit
def euler_to_quaternion(
    roll: float,
    pitch: float,
    yaw: float,
    degrees: bool = True,
    w_last: bool = True
) -> np.ndarray:
    """
    Convert Euler angles [roll, pitch, yaw] to a quaternion.
    """
    if degrees:
        roll *= np.pi/180.0
        pitch *= np.pi/180.0
        yaw *= np.pi/180.0

    hr, hp, hy = roll*0.5, pitch*0.5, yaw*0.5
    sr, cr = np.sin(hr), np.cos(hr)
    sp, cp = np.sin(hp), np.cos(hp)
    sy, cy = np.sin(hy), np.cos(hy)

    # quaternion for R = Rz * Ry * Rx  is q = qz * qy * qx
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy

    out = np.empty(4, dtype=np.float64)
    if w_last:
        out[0], out[1], out[2], out[3] = qx, qy, qz, qw
    else:
        out[0], out[1], out[2], out[3] = qw, qx, qy, qz
    return out


@njit
def _rotation_to(
    target_vector: np.ndarray,
    current_R: np.ndarray,
    forward: np.ndarray
) -> np.ndarray:
    """
    Compute a new 3x3 rotation matrix that takes the “forward” axis
    of the current rotation and re-aims it at the direction of `target_vector`.
    """
    # length of target_vector
    d = np_norm(target_vector)
    # if almost zero, no change
    if d < 1e-8:
        return current_R.copy()

    # normalize desired direction
    v_des = target_vector / d
    # current forward in world coords
    v_curr = np_dot(current_R, forward)

    # rotation axis = v_curr × v_des
    axis = np_cross(v_curr, v_des)
    s = np_norm(axis)
    c = np_dot(v_curr, v_des)

    # degenerate: either aligned (c≈1) or opposite (c≈-1)
    if s < 1e-8:
        if c > 0.0:
            # already pointing the right way
            R_delta = np_eye(3)
        else:
            # flip 180° about any perpendicular axis
            # pick axis orthogonal to v_curr
            perp = np_cross(v_curr, np_array([1.0, 0.0, 0.0]))
            if np_norm(perp) < 1e-3:
                perp = np_cross(v_curr, np_array([0.0, 1.0, 0.0]))
            perp /= np_norm(perp)
            # Rodrigues 180°: R = I + 2 * (K @ K)
            K = np_array([[0, -perp[2],  perp[1]],
                          [perp[2],      0, -perp[0]],
                          [-perp[1],  perp[0],     0]])
            R_delta = np_eye(3) + 2.0 * (K @ K)
    else:
        # general case:
        axis = axis / s
        K = np_array([[0, -axis[2],  axis[1]],
                      [axis[2],      0, -axis[0]],
                      [-axis[1],  axis[0],      0]])
        R_delta = np_eye(3) + K * s + (K @ K) * (1.0 - c)

    # final new world rotation = R_delta @ current_R
    return np_dot(R_delta, current_R)


@njit
def _az_el_range_to(target_vector: np.ndarray, up: np.ndarray, lateral: np.ndarray, forward: np.ndarray, degrees: bool = True, signed_azimuth: bool = False, counterclockwise_azimuth: bool = False, flip_elevation: bool = False) -> tuple[float, float, float]:
    """
    Calculate azimuth, elevation, and range from origin to target
    in the origin's own coordinate frame.

    Args:
        target_vector: the vector from origin to target.
        up: the up vector of the origin.
        lateral: the lateral vector of the origin.
        forward: the forward vector of the origin.
        degrees: if True, return az/el in degrees, else radians.
        signed_azimuth: if True, az ∈ [-180,180] (or [-π,π]), else [0,360).
        counterclockwise_azimuth: if True, positive az is from forward → left,
                        otherwise forward → right.
        flip_elevation: if True, positive el means downward (down vector),
                        otherwise positive means upward (up vector).

    Returns:
        (azimuth, elevation, range)
    """
    rng = np_norm(target_vector)
    if rng < 1e-12:
        return (0.0, 0.0, 0.0)

    # 3) horizontal projection: subtract off the component along 'up'
    #    (always use up for defining the horizontal plane)
    target_vector_h = target_vector - np_dot(target_vector, up) * up
    h_norm = np_norm(target_vector_h)
    if h_norm < 1e-8:
        # looking straight up/down: azimuth undefined → zero
        az_rad = 0.0
    else:
        # choose which lateral axis to project onto for azimuth
        if not counterclockwise_azimuth:
            lateral = -lateral
        comp = np_dot(target_vector_h, lateral)
        az_rad = np.arctan2(comp, np_dot(target_vector_h, forward))

    # 4) optionally wrap into [0,2π)
    if not signed_azimuth:
        az_rad = az_rad % (2*np.pi)

    # 5) elevation: angle between target_vector and horizontal plane
    #    choose up vs down as positive direction
    e_ref = -up if flip_elevation else up
    el_rad = np.arctan2(np_dot(target_vector, e_ref), h_norm)

    # 6) degrees?
    if degrees:
        az_rad = np.degrees(az_rad)
        el_rad = np.degrees(el_rad)

    return az_rad, el_rad, rng


@njit
def _phi_theta_to(
    target_vector: np.ndarray,
    up: np.ndarray,
    lateral: np.ndarray,
    forward: np.ndarray,
    degrees: bool,
    signed_phi: bool,
    counterclockwise_phi: bool,
    polar: bool,
    flip_theta: bool
) -> tuple[float, float]:
    # normalize
    r = np_norm(target_vector)
    if r < 1e-12:
        return 0.0, 0.0
    unit = target_vector / r

    # φ: positive around up-axis; CCW=forward→left, else forward→right
    axis = -lateral if counterclockwise_phi else lateral
    phi = np.arctan2(
        np_dot(unit, axis),
        np_dot(unit, forward)
    )
    if signed_phi:
        # wrap into (–π, π]
        phi = (phi + np.pi) % (2*np.pi) - np.pi
    else:
        # wrap into [0, 2π)
        phi = phi % (2*np.pi)

    # θ
    if polar:
        # polar angle from up-axis:
        theta = np.arccos(np_dot(unit, up))
    else:
        # elevation from horizontal:
        # elevation = atan2(dot(unit, up), norm of horizontal component)
        horiz = target_vector - np_dot(target_vector, up) * up
        hnorm = np_norm(horiz)
        theta = np.arctan2(np_dot(unit, up), hnorm)

    if flip_theta:
        theta = -theta

    if degrees:
        phi = np.degrees(phi)
        theta = np.degrees(theta)
    return phi, theta


@njit
def _latitude_longitude_to(
    target_vector: np.ndarray,
    up: np.ndarray,
    lateral: np.ndarray,
    forward: np.ndarray,
    degrees: bool,
    signed_longitude: bool,
    counterclockwise_longitude: bool,
    flip_latitude: bool
) -> tuple[float, float]:
    # normalize
    r = np_norm(target_vector)
    if r < 1e-12:
        return 0.0, 0.0
    unit = target_vector / r

    # longitude
    if not counterclockwise_longitude:
        lateral = -lateral
    lon = np.arctan2(
        np_dot(unit, lateral),
        np_dot(unit, forward)
    )
    if not signed_longitude:
        lon = lon % (2*np.pi)

    # latitude = arcsin(z/r) = angle above/below equatorial plane
    lat = np.arcsin(np_dot(unit, up))
    if flip_latitude:
        lat = -lat

    if degrees:
        lat = np.degrees(lat)
        lon = np.degrees(lon)
    return lat, lon
