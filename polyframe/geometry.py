# geometry.py
import math
from numpy import float64 as np_float64
from numpy import ndarray
import numpy as np
from typing import Tuple
from numba import njit

from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


@njit(cache=True)
def quaternion_to_rotation(quaternion: ndarray, w_last: bool = True) -> ndarray:
    """
    Convert a quaternion to a 3x3 rotation matrix.

    This function computes the rotation matrix corresponding to the given quaternion.
    The quaternion can be provided in two formats:
    - If w_last is True (default), the quaternion is expected to be in the form [x, y, z, w].
    - If w_last is False, the quaternion should be in the form [w, x, y, z].

    Parameters:
        quaternion (ndarray): A 4-element array representing the quaternion.
        w_last (bool, optional): Determines the order of the quaternion components.
            - True: quaternion is [x, y, z, w] (default).
            - False: quaternion is [w, x, y, z].

    Returns:
        ndarray: A 3x3 rotation matrix corresponding to the input quaternion.
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

    R = np.empty((3, 3), dtype=np_float64)
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


@njit(cache=True, fastmath=True)
def rotation_to_quaternion(rotation, w_last=True):
    """
    Converts a 3x3 rotation matrix to a normalized quaternion.

    Depending on the value of the trace of the rotation matrix, the algorithm selects an appropriate
    computation method to extract the quaternion components, ensuring numerical stability by normalizing
    the result.

    Parameters:
        rotation (array_like): A 3x3 rotation matrix.
        w_last (bool, optional): Determines the ordering of the quaternion components in the output array.
                                 If True, the quaternion is returned as [x, y, z, w] (with the real part last);
                                 otherwise, it is returned as [w, x, y, z] (with the real part first). Default is True.

    Returns:
        numpy.ndarray: A 1D numpy array of 4 floats representing the normalized quaternion in the specified order.

    Notes:
        - The rotation matrix should be a valid orthogonal matrix.
        - The function normalizes the quaternion to guard against numerical drift.

    Example:
        >>> import numpy as np
        >>> R = np.eye(3)
        >>> q = rotation_to_quaternion(R)
        >>> print(q)  # For w_last=True, output will be [0.0, 0.0, 0.0, 1.0]
    """
    # unpack to locals (avoids repeated indexing)
    a00, a01, a02 = rotation[0, 0], rotation[0, 1], rotation[0, 2]
    a10, a11, a12 = rotation[1, 0], rotation[1, 1], rotation[1, 2]
    a20, a21, a22 = rotation[2, 0], rotation[2, 1], rotation[2, 2]

    tr = a00 + a11 + a22

    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (a21 - a12) / S
        qy = (a02 - a20) / S
        qz = (a10 - a01) / S
    else:
        # pick largest diagonal element
        if a00 > a11 and a00 > a22:
            S = math.sqrt(1.0 + a00 - a11 - a22) * 2.0
            qw = (a21 - a12) / S
            qx = 0.25 * S
            qy = (a01 + a10) / S
            qz = (a02 + a20) / S
        elif a11 > a22:
            S = math.sqrt(1.0 + a11 - a00 - a22) * 2.0
            qw = (a02 - a20) / S
            qx = (a01 + a10) / S
            qy = 0.25 * S
            qz = (a12 + a21) / S
        else:
            S = math.sqrt(1.0 + a22 - a00 - a11) * 2.0
            qw = (a10 - a01) / S
            qx = (a02 + a20) / S
            qy = (a12 + a21) / S
            qz = 0.25 * S

    # normalize (guards against numerical drift)
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    out = np.empty(4, dtype=np_float64)
    if w_last:
        out[0], out[1], out[2], out[3] = qx, qy, qz, qw
    else:
        out[0], out[1], out[2], out[3] = qw, qx, qy, qz

    return out


@njit(cache=True)
def rotation_to_euler(rotation: ndarray, degrees: bool = True) -> tuple[float, float, float]:
    """
    Converts a 3x3 rotation matrix to Euler angles (roll, pitch, yaw).

    The conversion is based on the following assumptions:
    - The pitch angle is computed using the arcsine of the negative value of element [2, 0] of the rotation matrix.
    - The roll and yaw angles are computed using arctan2 functions on appropriate elements of the rotation matrix,
        taking into account a cosine correction for the pitch.
    - Optionally, the resulting angles are converted from radians to degrees if the 'degrees' flag is True.

    Parameters:
            rotation (ndarray): A 3x3 rotation matrix.
            degrees (bool, optional): If True, the Euler angles are returned in degrees. Defaults to True.

    Returns:
            tuple of float: A tuple containing the roll, pitch, and yaw angles in the specified unit (degrees or radians).
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


@njit(cache=True)
def quaternion_to_euler(
    quaternion: ndarray,
    w_last: bool = True,
    degrees: bool = True
) -> Tuple[float, float, float]:
    """
    Converts a quaternion to Euler angles.

    This function first converts the given quaternion into a rotation matrix using the
    quaternion_to_rotation function, and then converts that rotation matrix into Euler
    angles using the rotation_to_euler function.

    Parameters:
        quaternion (ndarray): The input quaternion representing orientation.
        w_last (bool, optional): Indicates the position of the scalar component in the quaternion.
                                 If True, it assumes the scalar is the last element; if False, the first.
                                 Defaults to True.
        degrees (bool, optional): If True, the resulting Euler angles are in degrees; otherwise, in radians.
                                  Defaults to True.

    Returns:
        Tuple[float, float, float]: A tuple containing the three Euler angles corresponding to
                                    the input quaternion.
    """
    R = quaternion_to_rotation(quaternion, w_last)
    return rotation_to_euler(R, degrees)


@njit(cache=True)
def euler_to_rotation(
        roll: float,
        pitch: float,
        yaw: float,
        degrees: bool = True) -> ndarray:
    """
    Compute a rotation matrix from Euler angles.

    This function calculates the rotation matrix corresponding to the provided Euler angles.
    The rotation is constructed as R = Rz(yaw) @ Ry(pitch) @ Rx(roll), where the angles are
    applied in the order: roll (rotation about the x-axis), pitch (rotation about the y-axis),
    and yaw (rotation about the z-axis). When the 'degrees' flag is True, the angles are assumed
    to be in degrees and are converted to radians before the computation.

    Parameters:
        roll (float): Rotation angle about the x-axis. If 'degrees' is True, the value is in degrees.
        pitch (float): Rotation angle about the y-axis. If 'degrees' is True, the value is in degrees.
        yaw (float): Rotation angle about the z-axis. If 'degrees' is True, the value is in degrees.
        degrees (bool, optional): Flag indicating whether the input angles are in degrees. Defaults to True.

    Returns:
        ndarray: A 3x3 NumPy array (of type np_float64) representing the rotation matrix.
    """
    if degrees:
        roll *= np.pi/180.0
        pitch *= np.pi/180.0
        yaw *= np.pi/180.0

    sr, cr = np.sin(roll),  np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw),   np.cos(yaw)

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = np.empty((3, 3), dtype=np_float64)
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


@njit(cache=True)
def euler_to_quaternion(
    roll: float,
    pitch: float,
    yaw: float,
    degrees: bool = True,
    w_last: bool = True
) -> ndarray:
    """
    Converts Euler angles to a quaternion representation.

    This function converts Euler angles (roll, pitch, yaw) into a quaternion that represents
    the same rotation. If the 'degrees' parameter is True, the input angles are assumed to be in degrees
    and are converted to radians before computation. The ordering of the quaternion components in the
    returned array is determined by the 'w_last' parameter: when True, the quaternion is returned in the order
    [qx, qy, qz, qw]; otherwise, it is returned as [qw, qx, qy, qz].

    Parameters:
        roll (float): Rotation angle around the x-axis.
        pitch (float): Rotation angle around the y-axis.
        yaw (float): Rotation angle around the z-axis.
        degrees (bool, optional): Indicates whether the input angles are in degrees. Defaults to True.
        w_last (bool, optional): Determines the output order of the quaternion. If True, the scalar part (w)
                                 is placed last; if False, it is placed first. Defaults to True.

    Returns:
        numpy.ndarray: A 4-element array representing the quaternion. The ordering of the elements is based
                       on the 'w_last' parameter.
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

    out = np.empty(4, dtype=np_float64)
    if w_last:
        out[0], out[1], out[2], out[3] = qx, qy, qz, qw
    else:
        out[0], out[1], out[2], out[3] = qw, qx, qy, qz
    return out


@njit(cache=True, fastmath=True)
def rotation_to(target: np.ndarray, R_in: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """
    Rotate the given rotation matrix so that its transformed forward vector aligns with the target direction.

    Parameters:
        target (np.ndarray): A 3-element vector representing the desired direction to align with.
        R_in (np.ndarray): A 3x3 rotation matrix representing the current orientation.
        forward (np.ndarray): A 3-element vector indicating the local "forward" direction in the original coordinate system.

    Returns:
        np.ndarray: A new 3x3 rotation matrix obtained by applying the necessary rotation to R_in
                    so that the transformed forward vector points in the direction of target.

    Notes:
        - If the target vector has a magnitude close to zero (norm² < 1e-16), no rotation is applied, and R_in is returned.
        - The function computes the necessary rotation using the Rodrigues rotation formula.
        - If the current forward and target directions are aligned, the original rotation is returned.
        - In the case of a 180° rotation (opposite vectors), a fallback mechanism is used to choose a suitable perpendicular axis.
    """
    # 1) unpack and norm² of target
    tx, ty, tz = target[0], target[1], target[2]
    d2 = tx*tx + ty*ty + tz*tz
    if d2 < 1e-16:
        return R_in         # no change

    # 2) normalize
    d = math.sqrt(d2)
    vx, vy, vz = tx/d, ty/d, tz/d

    # 3) current “forward” in world coords
    fx, fy, fz = forward[0], forward[1], forward[2]
    vcx = R_in[0, 0]*fx + R_in[0, 1]*fy + R_in[0, 2]*fz
    vcy = R_in[1, 0]*fx + R_in[1, 1]*fy + R_in[1, 2]*fz
    vcz = R_in[2, 0]*fx + R_in[2, 1]*fy + R_in[2, 2]*fz

    # 4) axis = v_curr × v_des
    ax = vcy*vz - vcz*vy
    ay = vcz*vx - vcx*vz
    az = vcx*vy - vcy*vx

    # 5) norm of axis and dot
    s2 = ax*ax + ay*ay + az*az
    c = vcx*vx + vcy*vy + vcz*vz

    # 6) handle aligned or opposite
    if s2 < 1e-16:
        if c > 0.0:
            return R_in   # already aligned
        # 180° flip: pick a perp to v_curr
        # here we cross with world-X (1,0,0),
        # but if v_curr is nearly parallel, cross with Y instead:
        px = vcy*0.0 - vcz*1.0
        py = vcz*1.0 - vcx*0.0
        pz = vcx*0.0 - vcy*0.0
        if px*px + py*py + pz*pz < 1e-6:
            # fallback cross with Y
            px = vcz*0.0 - vcx*1.0
            py = vcx*0.0 - vcz*0.0
            pz = vcy*1.0 - vcy*0.0
        # normalize perp
        pn = math.sqrt(px*px + py*py + pz*pz)
        ux, uy, uz = px/pn, py/pn, pz/pn
        s = 0.0
        c = -1.0          # cos(π)
    else:
        s = math.sqrt(s2)
        ux, uy, uz = ax/s, ay/s, az/s

    # 7) Rodrigues formula (explicit)
    one_c = 1.0 - c
    r00 = c + ux*ux*one_c
    r01 = ux*uy*one_c - uz*s
    r02 = ux*uz*one_c + uy*s

    r10 = uy*ux*one_c + uz*s
    r11 = c + uy*uy*one_c
    r12 = uy*uz*one_c - ux*s

    r20 = uz*ux*one_c - uy*s
    r21 = uz*uy*one_c + ux*s
    r22 = c + uz*uz*one_c

    # 8) apply R_delta @ R_in
    out = np.empty((3, 3), np.float64)
    # row 0
    out[0, 0] = r00*R_in[0, 0] + r01*R_in[1, 0] + r02*R_in[2, 0]
    out[0, 1] = r00*R_in[0, 1] + r01*R_in[1, 1] + r02*R_in[2, 1]
    out[0, 2] = r00*R_in[0, 2] + r01*R_in[1, 2] + r02*R_in[2, 2]
    # row 1
    out[1, 0] = r10*R_in[0, 0] + r11*R_in[1, 0] + r12*R_in[2, 0]
    out[1, 1] = r10*R_in[0, 1] + r11*R_in[1, 1] + r12*R_in[2, 1]
    out[1, 2] = r10*R_in[0, 2] + r11*R_in[1, 2] + r12*R_in[2, 2]
    # row 2
    out[2, 0] = r20*R_in[0, 0] + r21*R_in[1, 0] + r22*R_in[2, 0]
    out[2, 1] = r20*R_in[0, 1] + r21*R_in[1, 1] + r22*R_in[2, 1]
    out[2, 2] = r20*R_in[0, 2] + r21*R_in[1, 2] + r22*R_in[2, 2]

    return out


@njit(cache=True)
def azimuth_elevation_to(target_vector: ndarray, up: ndarray, lateral: ndarray, forward: ndarray, degrees: bool = True, signed_azimuth: bool = False, counterclockwise_azimuth: bool = False, flip_elevation: bool = False) -> tuple[float, float]:
    """
    Calculate azimuth and elevation from origin to target
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
        (azimuth, elevation)
    """
    rng = np.linalg.norm(target_vector)
    if rng < 1e-12:
        return (0.0, 0.0)

    # 3) horizontal projection: subtract off the component along 'up'
    #    (always use up for defining the horizontal plane)
    target_vector_h = target_vector - np.dot(target_vector, up) * up
    h_norm = np.linalg.norm(target_vector_h)
    if h_norm < 1e-8:
        # looking straight up/down: azimuth undefined → zero
        az_rad = 0.0
    else:
        # choose which lateral axis to project onto for azimuth
        if not counterclockwise_azimuth:
            lateral = -lateral
        comp = np.dot(target_vector_h, lateral)
        az_rad = np.arctan2(comp, np.dot(target_vector_h, forward))

    # 4) optionally wrap into [0,2π)
    if not signed_azimuth:
        az_rad = az_rad % (2*np.pi)

    # 5) elevation: angle between target_vector and horizontal plane
    #    choose up vs down as positive direction
    e_ref = -up if flip_elevation else up
    el_rad = np.arctan2(np.dot(target_vector, e_ref), h_norm)

    # 6) degrees?
    if degrees:
        az_rad = np.degrees(az_rad)
        el_rad = np.degrees(el_rad)

    return az_rad, el_rad


@njit(cache=True)
def phi_theta_to(
    target_vector: ndarray,
    up: ndarray,
    lateral: ndarray,
    forward: ndarray,
    degrees: bool = True,
    signed_phi: bool = False,
    counterclockwise_phi: bool = True,
    polar: bool = True,
    flip_theta: bool = False
) -> tuple[float, float]:
    """
    Calculates the azimuth (φ) and elevation (θ) angles of a target vector with respect to given reference axes.

    This function converts a Cartesian target vector into a pair of angles by projecting the vector onto
    a coordinate system defined by the 'up', 'lateral', and 'forward' axes. The computation of φ (azimuth)
    and θ (elevation or polar angle) is performed as follows:

    1. The target vector is normalized. If its norm is below a very small threshold (1e-12), both angles
        are set to 0.0.
    2. φ is computed as the arctan2 of the dot product of the unit target vector with the relevant lateral axis
        (which is negated if counterclockwise_phi is False) and the dot product with the forward axis.
        - If signed_phi is True, φ is wrapped into the range (–π, π]; otherwise into [0, 2π).
    3. θ is computed differently depending on the 'polar' flag:
        - If polar is True, θ is the angle between the target unit vector and the up axis (i.e., the polar angle).
        - If polar is False, θ represents the elevation angle, calculated as the arctan2 of the dot product of
          the unit vector with the up direction and the norm of the horizontal component (target vector with
          its up component removed).
        - The computed θ may be flipped in sign if flip_theta is True.
    4. If degrees is True, both φ and θ are converted from radians to degrees.

    Parameters:
         target_vector (ndarray): The target vector whose angles are to be calculated.
         up (ndarray): The upward axis vector.
         lateral (ndarray): The lateral axis vector, used for computing the azimuth angle.
         forward (ndarray): The forward axis vector.
         degrees (bool): If True, converts the resulting angles from radians to degrees.
         signed_phi (bool): Determines if φ should be returned as a signed angle (range: (–π, π]) or
                                  as an unsigned angle (range: [0, 2π)).
         counterclockwise_phi (bool): If True, defines the positive φ direction as counterclockwise from the
                                                forward axis toward the lateral axis; otherwise, the direction is reversed.
         polar (bool): If True, computes θ as the polar angle from the up axis; if False, computes θ as the
                            elevation from the horizontal plane.
         flip_theta (bool): If True, reverses the sign of θ.

    Returns:
         tuple[float, float]: A tuple containing the calculated azimuth (φ) and elevation (θ) angles.
    """
    # normalize
    r = np.linalg.norm(target_vector)
    if r < 1e-12:
        return 0.0, 0.0
    unit = target_vector / r

    # φ: positive around up-axis; CCW=forward→left, else forward→right
    axis = -lateral if counterclockwise_phi else lateral
    phi = np.arctan2(
        np.dot(unit, axis),
        np.dot(unit, forward)
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
        theta = np.arccos(np.dot(unit, up))
    else:
        # elevation from horizontal:
        # elevation = atan2(dot(unit, up), norm of horizontal component)
        horiz = target_vector - np.dot(target_vector, up) * up
        hnorm = np.linalg.norm(horiz)
        theta = np.arctan2(np.dot(unit, up), hnorm)

    if flip_theta:
        theta = -theta

    if degrees:
        phi = np.degrees(phi)
        theta = np.degrees(theta)
    return phi, theta


@njit(cache=True)
def latitude_longitude_to(
    target_vector: ndarray,
    up: ndarray,
    lateral: ndarray,
    forward: ndarray,
    degrees: bool = True,
    signed_longitude: bool = True,
    counterclockwise_longitude: bool = True,
    flip_latitude: bool = False
) -> tuple[float, float]:
    """
    Compute latitude and longitude from a 3D target vector relative to specified coordinate axes.

    This function computes the angular coordinates (latitude and longitude) of a given target vector
    by projecting it onto a coordinate system defined by the 'up', 'lateral', and 'forward' vectors.
    The latitude is determined as the arcsine of the dot product between the normalized target vector and the 'up' vector.
    The longitude is computed using an arctangent of the projections onto the 'lateral' and 'forward' axes,
    with optional adjustments for sign and rotation direction.

    Parameters:
        target_vector (ndarray):
            The input 3D vector whose latitude and longitude angles are to be calculated.
        up (ndarray):
            The vector representing the upward direction (normal to the equatorial plane).
        lateral (ndarray):
            The vector representing the lateral (sideways) direction.
        forward (ndarray):
            The vector representing the forward direction.
        degrees (bool, optional):
            If True, converts the resulting angles from radians to degrees (default is True).
        signed_longitude (bool, optional):
            If True, allows the longitude to be negative; if False, maps the longitude to [0, 2π) (default is True).
        counterclockwise_longitude (bool, optional):
            If True, uses the lateral vector as provided (assuming a counterclockwise orientation); if False, negates it (default is True).
        flip_latitude (bool, optional):
            If True, flips (negates) the computed latitude (default is False).

    Returns:
        tuple[float, float]:
            A tuple containing two floats:
                - latitude: The computed latitude angle.
                - longitude: The computed longitude angle.

    Notes:
        If the norm of the target vector is below 1e-12, the function returns (0.0, 0.0) to avoid division by zero.
    """
    # normalize
    r = np.linalg.norm(target_vector)
    if r < 1e-12:
        return 0.0, 0.0
    unit = target_vector / r

    # longitude
    if not counterclockwise_longitude:
        lateral = -lateral
    lon = np.arctan2(
        np.dot(unit, lateral),
        np.dot(unit, forward)
    )
    if not signed_longitude:
        lon = lon % (2*np.pi)

    # latitude = arcsin(z/r) = angle above/below equatorial plane
    lat = np.arcsin(np.dot(unit, up))
    if flip_latitude:
        lat = -lat

    if degrees:
        lat = np.degrees(lat)
        lon = np.degrees(lon)
    return lat, lon


@njit(cache=True, fastmath=True)
def polar_rotation_svd(M: ndarray) -> ndarray:
    """
    Return the orthonormal rotation R from the polar decomposition M = R · P
    using a single SVD call (U Σ Vᵀ).  Handles all edge cases:

    * arbitrary scale, shear, or reflection in M
    * singular / nearly singular M   → best-fit R is still defined
    * det(R) enforced to +1 (proper rotation)
    """
    # 1) SVD – works for any real 3×3, even rank‑deficient
    U, _, Vt = np.linalg.svd(M)          # M = U Σ Vᵀ

    # 2) Draft rotation
    R = U @ Vt                           # U Vᵀ is orthogonal; det may be −1

    # 3) Force det(R)=+1  (avoid improper reflection)
    if np.linalg.det(R) < 0.0:
        # Flip sign of last column of U (equivalent to Σ33 → −Σ33)
        U[:, 2] *= -1.0
        R = U @ Vt                       # recompute with proper handedness

    return R


@njit(cache=True, fastmath=True)
def pure_rotation_if_possible(M: ndarray) -> ndarray:
    """
    Fast path:  ▸ If M is already a proper rotation (orthonormal, det≈+1)
                ▸ return it unchanged.
    Slow path:  ▸ Otherwise perform SVD-based polar decomposition and
                  return the R factor.
    """
    # --- 1.  column norms --------------------------------------------------
    c0 = M[:, 0]
    c1 = M[:, 1]
    c2 = M[:, 2]

    n0 = c0[0]*c0[0] + c0[1]*c0[1] + c0[2]*c0[2]
    n1 = c1[0]*c1[0] + c1[1]*c1[1] + c1[2]*c1[2]
    n2 = c2[0]*c2[0] + c2[1]*c2[1] + c2[2]*c2[2]

    if (abs(n0-1.0) > 1e-8 or
        abs(n1-1.0) > 1e-8 or
            abs(n2-1.0) > 1e-8):
        return polar_rotation_svd(M)          # scale present

    # --- 2.  cross‑column orthogonality -----------------------------------
    d01 = c0[0]*c1[0] + c0[1]*c1[1] + c0[2]*c1[2]
    d02 = c0[0]*c2[0] + c0[1]*c2[1] + c0[2]*c2[2]
    d12 = c1[0]*c2[0] + c1[1]*c2[1] + c1[2]*c2[2]

    if (abs(d01) > 1e-8 or
        abs(d02) > 1e-8 or
            abs(d12) > 1e-8):
        return polar_rotation_svd(M)          # shear / skew

    # --- 3.  determinant +1 ? --------------------------------------------
    detM = (
        M[0, 0]*(M[1, 1]*M[2, 2] - M[1, 2]*M[2, 1])
        - M[0, 1]*(M[1, 0]*M[2, 2] - M[1, 2]*M[2, 0])
        + M[0, 2]*(M[1, 0]*M[2, 1] - M[1, 1]*M[2, 0])
    )
    if abs(detM - 1.0) > 1e-8:
        return polar_rotation_svd(M)          # reflection or numeric drift

    # --- Already a clean rotation ----------------------------------------
    return M
