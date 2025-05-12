# _polyframe.py

# Written by: Nathan Spencer
# Licensed under the Apache License, Version 2.0 (the "License")

from numpy.linalg import qr as np_qr
from numpy.linalg import det as np_det
from numpy.linalg import norm as np_norm
from numpy.linalg import svd as np_svd
from numpy.linalg import inv as np_inv
from numpy import diag as np_diag
from numpy import shape as np_shape
from numpy import asarray as np_asarray
from numpy import allclose as np_allclose
from numpy import append as np_append
from numpy import array as np_array
from numpy import dot as np_dot
from numpy import eye as np_eye
from numpy import cross as np_cross
from numpy import array2string as np_array2string
from numpy import float64 as np_float64
from numpy import ndarray
import numpy as np

from typing import Union, Optional, List, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
from numba import njit

from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# preallocate the identity matrix for performance
_EYE4 = np_eye(4, dtype=np_float64)


class Direction(Enum):
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    UP = 4
    DOWN = 5


# map each Direction to its unit‐vector in the *world* frame
_DIR_TO_VEC = {
    Direction.FORWARD:  np_array([1,  0,  0]),
    Direction.BACKWARD: np_array([-1,  0,  0]),
    Direction.LEFT:    np_array([0,  1,  0]),
    Direction.RIGHT:     np_array([0, -1,  0]),
    Direction.UP:       np_array([0,  0,  1]),
    Direction.DOWN:     np_array([0,  0, -1]),
}


@njit(cache=True)
def decompose_scale_shear(mat3: ndarray) -> tuple[ndarray, ndarray]:
    """
    Split the affine block A = R · P into
        S = (sx,sy,sz)      (pure scale)
        H = S⁻¹ · P             (unit-diagonal shear)
    Returns S, H.
    """
    R = pure_rotation_if_possible(mat3)
    # polar “stretch” block
    P = R.T @ mat3
    # S⁻¹·P   (unit diag by construction)
    S = np_diag(P)
    H = np.linalg.solve(np_diag(S), P)
    return S, H


@njit(cache=True)
def polar_rotation_svd(M: ndarray) -> ndarray:
    """
    Return the orthonormal rotation R from the polar decomposition M = R · P
    using a single SVD call (U Σ Vᵀ).  Handles all edge cases:

    * arbitrary scale, shear, or reflection in M
    * singular / nearly singular M   → best-fit R is still defined
    * det(R) enforced to +1 (proper rotation)
    """
    # 1) SVD – works for any real 3×3, even rank‑deficient
    U, _, Vt = np_svd(M)          # M = U Σ Vᵀ

    # 2) Draft rotation
    R = U @ Vt                           # U Vᵀ is orthogonal; det may be −1

    # 3) Force det(R)=+1  (avoid improper reflection)
    if np_det(R) < 0.0:
        # Flip sign of last column of U (equivalent to Σ33 → −Σ33)
        U[:, 2] *= -1.0
        R = U @ Vt                       # recompute with proper handedness

    return R


_TOL_ORTHO = 1e-6
_TOL_DET = 1e-6


@njit(cache=True)
def pure_rotation_if_possible(M: ndarray) -> ndarray:
    """
    Fast path:  ▸ If M is already a proper rotation (orthonormal, det≈+1)
                ▸ return it unchanged.
    Slow path:  ▸ Otherwise perform SVD‑based polar decomposition and
                  return the R factor.
    """
    # --- 1.  column norms --------------------------------------------------
    c0 = M[:, 0]
    c1 = M[:, 1]
    c2 = M[:, 2]

    n0 = c0[0]*c0[0] + c0[1]*c0[1] + c0[2]*c0[2]
    n1 = c1[0]*c1[0] + c1[1]*c1[1] + c1[2]*c1[2]
    n2 = c2[0]*c2[0] + c2[1]*c2[1] + c2[2]*c2[2]

    if (abs(n0-1.0) > _TOL_ORTHO or
        abs(n1-1.0) > _TOL_ORTHO or
            abs(n2-1.0) > _TOL_ORTHO):
        return polar_rotation_svd(M)          # scale present

    # --- 2.  cross‑column orthogonality -----------------------------------
    d01 = c0[0]*c1[0] + c0[1]*c1[1] + c0[2]*c1[2]
    d02 = c0[0]*c2[0] + c0[1]*c2[1] + c0[2]*c2[2]
    d12 = c1[0]*c2[0] + c1[1]*c2[1] + c1[2]*c2[2]

    if (abs(d01) > _TOL_ORTHO or
        abs(d02) > _TOL_ORTHO or
            abs(d12) > _TOL_ORTHO):
        return polar_rotation_svd(M)          # shear / skew

    # --- 3.  determinant +1 ? --------------------------------------------
    detM = (
        M[0, 0]*(M[1, 1]*M[2, 2] - M[1, 2]*M[2, 1])
        - M[0, 1]*(M[1, 0]*M[2, 2] - M[1, 2]*M[2, 0])
        + M[0, 2]*(M[1, 0]*M[2, 1] - M[1, 1]*M[2, 0])
    )
    if abs(detM - 1.0) > _TOL_DET:
        return polar_rotation_svd(M)          # reflection or numeric drift

    # --- Already a clean rotation ----------------------------------------
    return M


@njit(cache=True)
def quaternion_to_rotation(quaternion: ndarray, w_last: bool = True) -> ndarray:
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


@njit(cache=True)
def rotation_to_quaternion(rotation: ndarray, w_last: bool = True) -> ndarray:
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


@njit(cache=True)
def rotation_to_euler(rotation: ndarray, degrees: bool = True) -> tuple[float, float, float]:
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


@njit(cache=True)
def quaternion_to_euler(
    quaternion: ndarray,
    w_last: bool = True,
    degrees: bool = True
) -> Tuple[float, float, float]:
    """
    Convert a quaternion to Euler angles [roll, pitch, yaw].
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


@njit(cache=True)
def euler_to_quaternion(
    roll: float,
    pitch: float,
    yaw: float,
    degrees: bool = True,
    w_last: bool = True
) -> ndarray:
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


@njit(cache=True)
def rotation_to(
    target_vector: ndarray,
    current_R: ndarray,
    forward: ndarray
) -> ndarray:
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
    rng = np_norm(target_vector)
    if rng < 1e-12:
        return (0.0, 0.0)

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

    return az_rad, el_rad


@njit(cache=True)
def phi_theta_to(
    target_vector: ndarray,
    up: ndarray,
    lateral: ndarray,
    forward: ndarray,
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


@njit(cache=True)
def latitude_longitude_to(
    target_vector: ndarray,
    up: ndarray,
    lateral: ndarray,
    forward: ndarray,
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


@dataclass(slots=True)
class Transform:
    """
    A 4x4 homogeneous transformation in 3D space.

    Attributes:
        matrix (ndarray): 4x4 transformation matrix.
    """

    matrix: ndarray = field(default_factory=lambda: _EYE4.copy())

    @classmethod
    def identity(cls) -> "Transform":
        """
        Create an identity Transform.

        Returns:
            A new Transform whose `matrix` is the identity matrix.
        """
        return cls()

    @classmethod
    def from_values(
        cls,
        translation: Optional[Union[ndarray, List, Tuple]] = None,
        rotation: Optional[Union[ndarray, List, Tuple]] = None,
        scale: Optional[Union[ndarray, List, Tuple]] = None,
        shear: Optional[Union[ndarray, List, Tuple]] = None,
        perspective: Optional[Union[ndarray, List, Tuple]] = None,
    ) -> "Transform":
        """
        Create a Transform by assembling translation, rotation, scale, shear, and perspective into a 4x4 matrix.

        Args:
            translation: length-3 array to place in last column.
            rotation: 3x3 rotation matrix to place in upper-left.
            scale: length-3 scale factors applied along the diagonal.
            shear: 3x3 shear matrix to place in upper-left.
            perspective: length-4 perspective array.

        Returns:
            A new Transform whose `matrix` encodes the provided information.
        """
        instance = cls()
        if translation is not None:
            instance.set_translation(translation, inplace=True)
        if rotation is not None:
            instance.set_rotation(rotation, inplace=True)
        if scale is not None:
            instance.set_scale(scale, inplace=True)
        if shear is not None:
            instance.set_shear(shear, inplace=True)
        if perspective is not None:
            instance.set_perspective(perspective, inplace=True)
        return instance

    @classmethod
    def from_quaternion(
        cls,
        quaternion: ndarray,
        w_last: bool = True,
        translation: Optional[Union[ndarray, List, Tuple]] = None,
        scale: Optional[Union[ndarray, List, Tuple]] = None,
        shear: Optional[Union[ndarray, List, Tuple]] = None,
        perspective: Optional[Union[ndarray, List, Tuple]] = None,
    ) -> "Transform":
        """
        Create a Transform from a quaternion.

        Args:
            quaternion: 4-element array representing the quaternion.
            w_last: if True, the quaternion is in [x, y, z, w] format.
            translation: length-3 array to place in last column.
            scale: length-3 scale factors applied along the diagonal.
            shear: 3x3 shear matrix to place in upper-left.
            perspective: length-4 perspective array.

        Returns:
            A new Transform whose `matrix` encodes R.
        """
        return cls.from_values(translation=translation, rotation=quaternion_to_rotation(quaternion, w_last=w_last), scale=scale, shear=shear, perspective=perspective)

    @classmethod
    def from_euler_angles(
        cls,
        roll: float,
        pitch: float,
        yaw: float,
        degrees: bool = True,
        translation: Optional[Union[ndarray, List, Tuple]] = None,
        scale: Optional[Union[ndarray, List, Tuple]] = None,
        shear: Optional[Union[ndarray, List, Tuple]] = None,
        perspective: Optional[Union[ndarray, List, Tuple]] = None,
    ) -> "Transform":
        """
        Create a Transform from Euler angles.

        Args:
            roll: rotation around x-axis.
            pitch: rotation around y-axis.
            yaw: rotation around z-axis.
            degrees: if True, angles are in degrees, else radians.
            translation: length-3 array to place in last column.
            scale: length-3 scale factors applied along the diagonal.
            shear: 3x3 shear matrix to place in upper-left.
            perspective: length-4 perspective array.

        Returns:
            A new Transform whose `matrix` encodes R.
        """
        return cls.from_values(translation=translation, rotation=euler_to_rotation(
            roll, pitch, yaw, degrees=degrees), scale=scale, shear=shear, perspective=perspective)

    @classmethod
    def from_flat_array(
        cls,
        flat_array: ndarray,
    ) -> "Transform":
        """
        Create a Transform from a flat array.

        Args:
            flat_array: 1D array of 16 floats representing the matrix.

        Returns:
            A new Transform whose `matrix` is constructed from the flat array.
        """
        if flat_array.shape != (16,):
            raise ValueError(f"Invalid flat array shape: {flat_array.shape}")
        flat_array = np_asarray(flat_array, dtype=np_float64)
        mat = flat_array.reshape((4, 4))
        return cls(mat)

    @classmethod
    def from_list(
        cls,
        list_array: List[float],
    ) -> "Transform":
        """
        Create a Transform from a list.

        Args:
            list_array: 1D list of 16 floats representing the matrix.

        Returns:
            A new Transform whose `matrix` is constructed from the list.
        """
        if len(list_array) != 16:
            raise ValueError(f"Invalid list array length: {len(list_array)}")
        mat = np_array(list_array, dtype=np_float64).reshape((4, 4))
        return cls(mat)

    #########
    # Getters for fundamental properties
    #

    @property
    def translation(self) -> ndarray:
        """
        Extract the translation vector.

        Returns:
            A length-3 array from the first three entries of the fourth column.
        """
        return self.matrix[:3, 3]

    @property
    def rotation(self) -> ndarray:
        """
        Extract the 3x3 rotation submatrix.

        Returns:
            The upper-left 3x3 of `matrix`.
        """
        return pure_rotation_if_possible(self.matrix[:3, :3])

    @property
    def scale(self) -> ndarray:
        """
        Compute and return the scale factors from the object's transformation matrix.

        Returns the scale portion of the upper-left 3x3 decomposition.

        Returns:
            numpy.ndarray: A 1D array containing the scale factors along the x, y, and z axes.
        """
        return decompose_scale_shear(self.matrix[:3, :3])[0]

    @property
    def shear(self) -> ndarray:
        """
        Compute and return the shear component of the 3x3 transformation matrix.

        Returns the shear portion of the upper-left 3x3 decomposition.

        Returns:
            numpy.ndarray: The 3x3 shear component extracted from the transformation matrix.
        """
        return decompose_scale_shear(self.matrix[:3, :3])[1]

    @property
    def perspective(self) -> ndarray:
        """
        Extract the perspective component of the matrix.

        Returns:
            The last row of the matrix.
        """
        return self.matrix[3, :]

    ########
    # Apply methods
    #

    def apply_translation(self, translation: ndarray, *, inplace: bool = False) -> "Transform":
        """
        Apply a translation to this Transform.

        Args:
            translation: length-3 vector to add to current translation.
            inplace: if True, modify this Transform in place.

        Returns:
            Transform with updated translation.
        """
        mat = self.matrix if inplace else self.matrix.copy()
        mat[:3, 3] += translation
        return self if inplace else self.__class__(mat)

    def apply_rotation(self, rotation: ndarray, *, inplace: bool = False) -> "Transform":
        """
        Apply a rotation to this Transform.

        Args:
            rotation: 3x3 matrix to left-multiply current rotation.
            inplace: if True, modify this Transform in place.

        Returns:
            Transform with updated rotation.
        """
        R = self.rotation                          # already the polar R
        P = R.T @ self.matrix[:3, :3]              # hence P = Rᵀ (R·P)
        new_block = rotation @ R @ P                # (Q · R) P
        mat = self.matrix if inplace else self.matrix.copy()
        mat[:3, :3] = new_block
        return self if inplace else self.__class__(mat)

    def apply_rotation_from_quaternion(self, quaternion: ndarray, w_last: bool = True, *, inplace: bool = False) -> "Transform":
        """
        Apply a quaternion to this Transform.

        Args:
            quaternion: 4-element array representing the quaternion.
            inplace: if True, modify this Transform in place.

        Returns:
            Transform with updated rotation.
        """
        return self.apply_rotation(quaternion_to_rotation(quaternion, w_last=w_last), inplace=inplace)

    def apply_rotation_from_euler(self, roll: float, pitch: float, yaw: float, degrees: bool = True, *, inplace: bool = False) -> "Transform":
        """
        Apply Euler angles to this Transform.

        Args:
            euler_angles: 3-element array representing the Euler angles (roll, pitch, yaw).
            inplace: if True, modify this Transform in place.

        Returns:
            Transform with updated rotation.
        """
        return self.apply_rotation(euler_to_rotation(roll, pitch, yaw, degrees=degrees), inplace=inplace)

    def apply_scale(self, scale: ndarray, *, inplace: bool = False) -> "Transform":
        """
        Apply a scale to this Transform.

        Args:
            scale: length-3 factors to multiply each axis.
            inplace: if True, modify this Transform in place.

        Returns:
            Transform with updated scale.
        """
        shape = np_shape(scale)
        if shape == (3,):
            raise ValueError(f"Invalid scale shape: {shape}")

        R = self.rotation                          # already the polar R
        P = R.T @ self.matrix[:3, :3]              # hence P = Rᵀ (R·P)
        P_new = P @ np_diag(scale)                 # keep orientation
        mat = self.matrix if inplace else self.matrix.copy()
        mat[:3, :3] = R @ P_new
        return self if inplace else self.__class__(mat)

    def apply_shear(self, shear: ndarray, *, inplace=False) -> "Transform":
        """
        Apply a shear transformation to the current affine block in local space.

        This method post-multiplies the current 3x3 linear (affine) submatrix of the transformation matrix
        with the provided shear matrix. The shear is applied in the object's local space without altering
        the scaling of the axes (i.e., each axis retains its current length).

        Parameters
        ----------
        shear : ndarray
            A 3x3 shear matrix that must have a unit diagonal (i.e., np.diag(shear) == [1, 1, 1]).
        inplace : bool, optional
            If True, modifies the transformation matrix in place. If False, the transformation matrix is copied,
            and a new instance of the transformation is returned. Default is False.

        Returns
        -------
        Transform
            The updated transformation instance. Returns `self` if inplace is True; otherwise, returns a new instance
            with the modified matrix.

        Raises
        ------
        ValueError
            If `shear` does not have a shape of (3, 3) or if its diagonal elements are not all ones.
        """
        if np_shape(shear) != (3, 3) or not np_allclose(np_diag(shear), 1):
            raise ValueError("shear must be 3x3 with unit diagonal")
        mat = self.matrix if inplace else self.matrix.copy()
        mat[:3, :3] = mat[:3, :3] @ shear
        return self if inplace else self.__class__(mat)

    def apply_perspective(self, perspective: ndarray, *, inplace: bool = False) -> "Transform":
        """
        Apply a perspective to this Transform.

        Args:
            perspective: length-4 array to add to current perspective.
            inplace: if True, modify this Transform in place.

        Returns:
            Transform with updated perspective.
        """
        mat = self.matrix if inplace else self.matrix.copy()
        mat[3, :] += perspective
        return self if inplace else self.__class__(mat)

    #########
    # Setter methods
    #

    @rotation.setter
    def rotation(self, value: ndarray) -> None:
        """
        Assign a new pure-rotation R_new but preserve the original P = Rᵀ·(R·P).

        Args:
            value: 3x3 rotation matrix to set as rotation.
        """
        # current R and P
        P = self.rotation.T @ self.matrix[:3, :3]
        # write back R_new·P
        self.matrix[:3, :3] = value @ P

    @translation.setter
    def translation(self, value: ndarray) -> None:
        """
        Set the translation vector.

        Args:
            value: length-3 array to set as translation.
        """
        self.matrix[:3, 3] = value

    @shear.setter
    def shear(self, value: ndarray) -> None:
        """
        Overwrite shear while preserving rotation R and scale S.
        `value` must be a 3x3 with ones on the diagonal.
        """
        if np_shape(value) != (3, 3):
            raise ValueError("shear must be 3x3")
        if not np_allclose(np_diag(value), 1):
            raise ValueError("shear diagonal must all be 1")
        R = self.rotation
        S, _ = decompose_scale_shear(self.matrix[:3, :3])
        self.matrix[:3, :3] = R @ S @ value

    @scale.setter
    def scale(self, value: ndarray) -> None:
        """
        Set the scale factors.

        Args:
            value: length-3 array to set as scale.
        """
        if np_shape(value) != (3,):
            raise ValueError(f"scale must be 3x1.")

        self.matrix[:3, :3] = self.rotation @ np_diag(value)

    @perspective.setter
    def perspective(self, value: ndarray) -> None:
        """
        Set the perspective component of the matrix.

        Args:
            value: length-4 array to set as perspective.
        """
        self.matrix[3, :] = value

    def set_translation(self, translation: ndarray, *, inplace: bool = False) -> "Transform":
        """
        Assign a translation to this Transform.

        Args:
            translation: length-3 vector to set as translation.

        Returns:
            self with updated translation.
        """
        mat = self.matrix if inplace else self.matrix.copy()
        mat[:3, 3] = translation
        return self if inplace else self.__class__(mat)

    def set_rotation(self, rotation: ndarray, *, inplace: bool = False) -> "Transform":
        """
        Assign a rotation to this Transform.

        Args:
            rotation: 3x3 matrix to set as rotation.

        Returns:
            self with updated rotation.
        """
        # hence P = Rᵀ (R·P)
        P = self.rotation.T @ self.matrix[:3, :3]
        mat = self.matrix if inplace else self.matrix.copy()
        mat[:3, :3] = rotation @ P
        return self if inplace else self.__class__(mat)

    def set_rotation_from_quaternion(self, quaternion: ndarray, w_last: bool = True, *, inplace: bool = False) -> "Transform":
        """
        Assign a quaternion to this Transform.

        Args:
            quaternion: 4-element array representing the quaternion.

        Returns:
            self with updated rotation.
        """
        return self.set_rotation(quaternion_to_rotation(quaternion, w_last=w_last), inplace=inplace)

    def set_rotation_from_euler(self, roll: float, pitch: float, yaw: float, degrees: bool = True, *, inplace: bool = False) -> "Transform":
        """
        Assign Euler angles to this Transform.

        Args:
            euler_angles: 3-element array representing the Euler angles (roll, pitch, yaw).
            inplace: if True, modify this Transform in place.

        Returns:
            self with updated rotation.
        """
        return self.set_rotation(euler_to_rotation(roll, pitch, yaw, degrees=degrees), inplace=inplace)

    def set_scale(self, scale: ndarray, *, inplace: bool = False) -> "Transform":
        """
        Assign a scale to this Transform.

        Args:
            scale: length-3 factors to set as scale.

        Returns:
            self with updated scale.
        """
        if np_shape(scale) != (3,):
            raise ValueError(f"scale must be 3x1.")

        mat = self.matrix if inplace else self.matrix.copy()
        mat[:3, :3] = self.rotation @ np_diag(scale)
        return self if inplace else self.__class__(mat)

    def set_shear(self, shear: ndarray, *, inplace=False) -> "Transform":
        """
        Replace the shear component, keep R and S unchanged.
        """
        if np_shape(shear) != (3, 3) or not np_allclose(np_diag(shear), 1):
            raise ValueError("shear must be 3x3 with unit diagonal")
        S, _ = decompose_scale_shear(self.matrix[:3, :3])
        mat = self.matrix if inplace else self.matrix.copy()
        mat[:3, :3] = self.rotation @ S @ shear
        return self if inplace else self.__class__(mat)

    def set_perspective(self, perspective: ndarray, *, inplace: bool = False) -> "Transform":
        """
        Assign a perspective to this Transform.

        Args:
            perspective: length-4 array to set as perspective.

        Returns:
            self with updated perspective.
        """
        mat = self.matrix if inplace else self.matrix.copy()
        mat[3, :] = perspective
        return self if inplace else self.__class__(mat)

    ########
    # Transform methods
    #

    def transform_point(self, point: ndarray) -> ndarray:
        """
        Apply this transform to a 3D point (affine).

        Args:
            point: length-3 array.

        Returns:
            Transformed length-3 point.
        """
        p = np_append(point, 1.0)
        return (self.matrix @ p)[:3]

    def transform_vector(self, vector: ndarray) -> ndarray:
        """
        Apply this transform to a 3D direction (no translation).

        Args:
            vector: length-3 array.

        Returns:
            Transformed length-3 vector.
        """
        v = np_append(vector, 0.0)
        return (self.matrix @ v)[:3]

    ########
    # Target methods
    #

    def distance_to(self, target: Union["Transform", ndarray]) -> float:
        """
        Compute the distance to another Transform or translation vector.

        Args:
            target: the target Transform or translation vector.

        Returns:
            The distance to the target.
        """
        if isinstance(target, Transform):
            tgt = target.matrix[:3, 3]
        else:
            tgt = np_asarray(target, float)

        return np_norm(tgt - self.matrix[:3, 3])

    def vector_to(self, target: Union["Transform", ndarray]) -> ndarray:
        """
        Compute the vector to another Transform or translation vector.

        Args:
            target: the target Transform or translation vector.

        Returns:
            The vector to the target.
        """
        if isinstance(target, Transform):
            tgt = target.matrix[:3, 3]
        else:
            tgt = np_asarray(target, float)

        return tgt - self.matrix[:3, 3]

    def direction_to(self, target: Union["Transform", ndarray]) -> ndarray:
        """
        Compute the direction to another Transform or translation vector.

        Args:
            target: the target Transform or translation vector.

        Returns:
            The direction to the target.
        """
        if isinstance(target, Transform):
            tgt = target.matrix[:3, 3]
        else:
            tgt = np_asarray(target, float)
        target_vector = tgt - self.matrix[:3, 3]
        distance = np_norm(target_vector)
        if distance < 1e-8:
            # avoid division by zero by returning forward vector
            return self.forward

        return target_vector / distance

    def rotation_to(
        self,
        target: Union["Transform", ndarray],
    ) -> ndarray:
        """
        Get the rotation from this Trasnform to the target.

        Args:
            target: the target Transform or translation vector.

        Returns:
            Rotation matrix 3x3.
        """
        # 1) grab the world-space target translation
        if isinstance(target, Transform):
            tgt = target.matrix[:3, 3]
        else:
            tgt = np_asarray(target, float)

        # 2) form the vector from this.origin → target
        target_vector = tgt - self.matrix[:3, 3]

        # 3) call into our compiled routine
        return rotation_to(
            target_vector,
            self.rotation,
            self.basis_forward()
        )

    def quaternion_to(self, target: Union["Transform", ndarray], w_last: bool = True) -> ndarray:
        """
        Get the quaternion from this Transform to the target.

        Args:
            target: the target Transform or translation vector.

        Returns:
            Quaternion 4-element array.
        """
        # 1) grab the world-space target translation
        if isinstance(target, Transform):
            tgt = target.matrix[:3, 3]
        else:
            tgt = np_asarray(target, float)

        # 2) form the vector from this.origin → target
        target_vector = tgt - self.matrix[:3, 3]

        # 3) call into our compiled routine
        return rotation_to_quaternion(
            rotation_to(
                target_vector,
                self.rotation,
                self.basis_forward()
            ),
            w_last=w_last
        )

    def euler_angles_to(self, target: Union["Transform", ndarray], degrees: bool = True) -> tuple[float, float, float]:
        """
        Get the Euler angles from this Transform to the target.

        Args:
            target: the target Transform or translation vector.
            degrees: if True, return angles in degrees, else radians.

        Returns:
            (roll, pitch, yaw)
        """
        # 1) grab the world-space target translation
        if isinstance(target, Transform):
            tgt = target.matrix[:3, 3]
        else:
            tgt = np_asarray(target, float)

        # 2) form the vector from this.origin → target
        target_vector = tgt - self.matrix[:3, 3]

        # 3) call into our compiled routine
        return rotation_to_euler(
            rotation_to(
                target_vector,
                self.rotation,
                self.basis_forward()
            ),
            degrees=degrees
        )

    def azimuth_elevation_to(
        self,
        target: Union["Transform", ndarray],
        *,
        degrees: bool = True,
        signed_azimuth: bool = False,
        counterclockwise_azimuth: bool = False,
        flip_elevation: bool = False
    ) -> tuple[float, float]:
        """
        Calculate azimuth, elevation, and range to the target.

        Args:
            origin: the observer Transform.
            target: the target Transform or translation vector.
            degrees: if True, return az/el in degrees, else radians.
            signed_azimuth: if True, az ∈ [-180,180] (or [-π,π]), else [0,360).
            counterclockwise_azimuth: if True, positive az is from forward → left,
                            otherwise forward → right.
            flip_elevation: if True, positive el means downward (down vector),
                            otherwise positive means upward (up vector).

        Returns:
            (azimuth, elevation)
        """
        if isinstance(target, Transform):
            target_vector = target.matrix[:3, 3] - self.matrix[:3, 3]
        else:
            target_vector = np_asarray(target, float) - self.matrix[:3, 3]
        return azimuth_elevation_to(target_vector, self.up, self.right, self.forward, degrees=degrees, signed_azimuth=signed_azimuth, counterclockwise_azimuth=counterclockwise_azimuth, flip_elevation=flip_elevation)

    def phi_theta_to(
        self,
        target: Union["Transform", ndarray],
        *,
        degrees: bool = True,
        signed_phi: bool = False,
        counterclockwise_phi: bool = True,
        polar: bool = True,
        flip_theta: bool = False
    ) -> tuple[float, float]:
        """
        Calculate (φ, θ) to the target.

        Args:
            target: the target Transform or translation vector.
            degrees: if True, return angles in degrees, else radians.
            signed_phi: if True, φ in [-π,π] (or [-180,180]), else [0,2π) (or [0,360)).
            counterclockwise_phi: if True, φ positive from forward → left, else forward → right.
            polar: if True, θ is the polar angle from up (0…π), else θ is elevation from horizontal (−π/2…π/2).
            flip_theta: if True, flip the sign of θ.

        Returns:
            (φ, θ)
        """
        if isinstance(target, Transform):
            tv = target.matrix[:3, 3] - self.matrix[:3, 3]
        else:
            tv = np_asarray(target, float) - self.matrix[:3, 3]

        return phi_theta_to(
            tv,
            self.up, self.right, self.forward,
            degrees,
            signed_phi,
            counterclockwise_phi,
            polar,
            flip_theta
        )

    def latitude_longitude_to(
        self,
        target: Union["Transform", ndarray],
        *,
        degrees: bool = True,
        signed_longitude: bool = True,
        counterclockwise_longitude: bool = True,
        flip_latitude: bool = False
    ) -> tuple[float, float]:
        """
        Calculate (latitude, longitude) to the target.

        Args:
            target: the target Transform or translation vector.
            degrees: if True, return lat/lon in degrees, else radians.
            signed_longitude: if True, lon in [-π,π] (or [-180,180]), else [0,2π).
            counterclockwise_longitude: if True, lon positive from forward → left, else forward → right.
            flip_latitude: if True, flip the sign of latitude.

        Returns:
            (latitude, longitude)
        """
        if isinstance(target, Transform):
            tv = target.matrix[:3, 3] - self.matrix[:3, 3]
        else:
            tv = np_asarray(target, float) - self.matrix[:3, 3]

        return latitude_longitude_to(
            tv,
            self.up, self.right, self.forward,
            degrees,
            signed_longitude,
            counterclockwise_longitude,
            flip_latitude
        )

    ########
    # Camera methods
    #

    def look_at(
        self,
        target: Union["Transform", ndarray],
        *,
        inplace: bool = False
    ) -> "Transform":
        """
        Rotate this Transform so that its forward axis points at `target`.
        Rotate only the R part so forward→target, leaving P intact.

        Args:
            target: the target Transform or translation vector.
            inplace: if True, modify this Transform in place.

        Returns:
            Transform with updated rotation.
        """
        # 1) compute pure‐rotation that points forward at target
        if isinstance(target, Transform):
            tgt = target.matrix[:3, 3]
        else:
            tgt = np_asarray(target, float)
        v = tgt - self.matrix[:3, 3]
        R_new = rotation_to(v, self.rotation, self.basis_forward())

        # 2) grab old R and P
        R_old = self.rotation
        P = R_old.T @ self.matrix[:3, :3]

        # 3) compose back and write
        block = R_new @ P
        if inplace:
            self.matrix[:3, :3] = block
            return self

        M = self.matrix.copy()
        M[:3, :3] = block
        return self.__class__(M)

    ########
    # Convenience/utility methods
    #

    def is_rigid(self, tol: float = 1e-6) -> bool:
        """
        Check if the transform encodes a pure rotation + translation.

        Parameters:
        -----------
        tol : float
            Tolerance for orthonormality and det≈1 check.

        Returns:
        --------
        bool
        """
        rot = self.matrix[:3, :3]
        return (
            np_allclose(rot @ rot.T, np_eye(3), atol=tol) and
            np_allclose(np_det(rot), 1.0, atol=tol) and
            np_allclose(self.scale, 1, atol=tol)
        )

    def orthonormalize(self, *, inplace: bool = True) -> "Transform":
        """
        Re-orthonormalize the rotation block to remove drift.

        Returns:
        --------
        Transform
        """
        if inplace:
            self.matrix[:3, :3] = np_qr(self.matrix[:3, :3])[0]
            return self

        new = self.matrix.copy()
        new[:3, :3] = np_qr(self.matrix[:3, :3])[0]
        return self.__class__(new)

    def inverse(self, *, inplace: bool = False) -> "Transform":
        """
        Invert this Transform.

        Args:
            inplace: if True, modify this Transform in place.

        Returns:
            Inverted Transform.
        """
        mat = self.matrix if inplace else self.matrix.copy()
        mat = np_inv(mat)
        return self if inplace else self.__class__(mat)

    def transpose(self, *, inplace: bool = False) -> "Transform":
        """
        Transpose of the 4x4 matrix.

        Returns:
            The matrix transposed.
        """
        if inplace:
            self.matrix[:] = self.matrix.T
            return self

        return self.__class__(self.matrix.copy().T)

    ###########
    # World frame properties derived from coordinate system convention
    #

    @property
    def forward(self) -> ndarray:
        """
        Rotate the coordinate system's forward vector into world frame.

        Returns:
            The 3D “forward” direction after applying this transform's rotation.
        """
        ...

    @property
    def backward(self) -> ndarray:
        """
        Rotate the coordinate system's backward vector into world frame.

        Returns:
            The 3D “backward” direction after applying this transform's rotation.
        """
        ...

    @property
    def left(self) -> ndarray:
        """
        Rotate the coordinate system's left vector into world frame.

        Returns:
            The 3D “left” direction after applying this transform's rotation.
        """
        ...

    @property
    def right(self) -> ndarray:
        """
        Rotate the coordinate system's right vector into world frame.

        Returns:
            The 3D “right” direction after applying this transform's rotation.
        """
        ...

    @property
    def up(self) -> ndarray:
        """
        Rotate the coordinate system's up vector into world frame.

        Returns:
            The 3D “up” direction after applying this transform's rotation.
        """
        ...

    @property
    def down(self) -> ndarray:
        """
        Rotate the coordinate system's down vector into world frame.

        Returns:
            The 3D “down” direction after applying this transform's rotation.
        """
        ...

    ############
    # Basis information and conventions
    #

    @staticmethod
    def is_right_handed() -> bool:
        """
        Check if the coordinate system is right-handed.
        Implemented in the generated code.

        Returns:
            True if the coordinate system is right-handed, False otherwise.
        """
        ...

    @staticmethod
    def label_x() -> Direction:
        """
        Returns the label for the x-axis of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def label_y() -> Direction:
        """
        Returns the label for the y-axis of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def label_z() -> Direction:
        """
        Returns the label for the z-axis of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_x() -> ndarray:
        """
        Returns the basis x vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_y() -> ndarray:
        """
        Returns the basis y vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_z() -> ndarray:
        """
        Returns the basis z vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_matrix() -> ndarray:
        """
        Returns the basis matrix of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_matrix_inv() -> ndarray:
        """
        Returns the inverse of the basis matrix of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_forward() -> ndarray:
        """
        Returns the basis forward vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_backward() -> ndarray:
        """
        Returns the basis backward vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_left() -> ndarray:
        """
        Returns the basis left vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_right() -> ndarray:
        """
        Returns the basis right vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_up() -> ndarray:
        """
        Returns the basis up vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_down() -> ndarray:
        """
        Returns the basis down vector of the transform.
        Implemented in the generated code.
        """
        ...

    #########
    # Basis methods
    #

    def change_coordinate_system(
        self,
        other: Type["Transform"],
        *,
        inplace: bool = False
    ) -> "Transform":
        """
        Re-express this transform in the coordinate system defined by `other` by right multiplying.

        Both `self` and `other` map *their* local axes→world.  This method
        computes the 3x3 change-of-basis P that carries `other`'s local
        coords into `self`'s local coords, then right-multiplies your
        4x4 matrix by the homogeneous P to yield a new 4x4 that, when
        called on `other`-local points, gives the same world result.

        Mathematically:
            B_old = self.basis_matrix()      # rows = old-local axes in world
            B_new = other.basis_matrix()     # rows = new-local axes in world
            P     = B_old.T @ B_new          # new-local → old-local
            C     = [[P, 0],
                     [0, 1]]                # 4x4 homogeneous
            M_new = self.matrix @ C

        Args:
            other:     The Transform *class* whose basis you want to switch into.
            inplace:   If True, overwrite self.matrix; otherwise return a new instance.

        Returns:
            A Transform of type `other` whose numeric matrix does the same
            world-space mapping, but expects `other`-local inputs.
        """
        # 1) grab the 3×3 basis matrices (rows = basis vectors in world coords)
        B_old = self.basis_matrix()    # ndarray (3×3)
        B_new = other.basis_matrix()      # ndarray (3×3)

        # 2) compute P = B_old⁻¹ B_new; since B_old is orthonormal, inv = transpose
        P = B_old.T @ B_new                # (3×3)

        # 3) build a 4×4 homogeneous change‐of‐basis
        C = _EYE4.copy()               # ndarray (4×4)
        C[:3, :3] = P

        # 4) apply on the right
        M_new = self.matrix @ C

        if inplace:
            self.matrix[:] = M_new
            return self

        # return an instance of the *other* subclass
        return other(M_new)

    #########
    # To-styled/representation methods
    #

    def to_quaternion(self, w_last: bool = True) -> ndarray:
        """
        Extract the quaternion from the rotation matrix.

        Returns:
            A 4-element array representing the quaternion.
        """
        return rotation_to_quaternion(self.rotation, w_last=w_last)

    def to_euler_angles(self, degrees: bool = True) -> Tuple[float, float, float]:
        """
        Extract the Euler angles from the rotation matrix.

        Returns:
            A 3-element tuple representing the Euler angles.
        """
        return rotation_to_euler(self.rotation, degrees=degrees)

    def to_list(self) -> List[float]:
        """
        Convert the matrix to a list of floats.

        Returns:
            A list of 16 floats representing the matrix.
        """
        return self.matrix.flatten().tolist()

    def to_flat_array(self) -> ndarray:
        """
        Convert the matrix to a flat array.

        Returns:
            A 1D array of 16 floats representing the matrix.
        """
        return self.matrix.flatten()

    #########
    # Dunder methods
    #

    def __matmul__(self, other: Union["Transform", ndarray]) -> Union["Transform", ndarray]:
        """
        Compute the matrix multiplication or composition of transforms.
        This magic method overloads the @ operator. It supports two types of operands:
        1. If 'other' is a numpy ndarray, this method applies the transform's 4x4 matrix
            to the array and returns the resulting array.
        2. If 'other' is another Transform, it composes the transforms in such a way that
            the transformation represented by 'other' is applied first, followed by this transform.
            If the two transforms have different coordinate system representations,
            'other' is converted to the same basis as self before composition.
        Parameters:
             other (Union["Transform", ndarray]): The right-hand operand. It can be either:
                  - A numpy ndarray, in which case the transform's matrix is applied to it.
                  - Another Transform object, in which case the transforms are composed (self ∘ other).
        Returns:
             Union["Transform", ndarray]:
                  - A numpy ndarray if 'other' is an ndarray.
                  - A new Transform instance representing the composition if 'other' is a Transform.
        Raises:
             NotImplemented: If 'other' is neither a numpy ndarray nor a Transform instance.
        """
        if isinstance(other, ndarray):
            return self.matrix @ other

        if not isinstance(other, Transform):
            return NotImplemented

        # if other has a different labeling/basis subclass, convert it
        if type(other) is not type(self):
            other = other.change_coordinate_system(type(self))

        # Compose: first apply `other`, then `self`
        M_combined = self.matrix @ other.matrix
        return self.__class__(M_combined)

    def __mul__(self, other: Union["Transform", ndarray]) -> Union["Transform", ndarray]:
        """
        Alias for the @ operator: allows `self * other` as well as `self @ other`.
        """
        return self.__matmul__(other)

    def __eq__(self, other: object) -> bool:
        """
        True if `other` is the same class and matrices are equal within a small tolerance.
        """
        if not isinstance(other, Transform) or type(self) is not type(other):
            return False
        return np_allclose(self.matrix, other.matrix)

    def __repr__(self) -> str:
        """
        Unambiguous representation including class name and matrix.
        """
        cls = type(self).__name__
        mat = np_array2string(self.matrix, precision=6, separator=', ')
        return f"{cls}(matrix=\n{mat}\n)"

    def __str__(self) -> str:
        """
        Friendly string: delegating to repr for now.
        """
        return self.__repr__()

    def __copy__(self) -> "Transform":
        """
        Shallow copy of this Transform (matrix is copied).
        """
        return type(self)(self.matrix.copy())

    def __deepcopy__(self, memo) -> "Transform":
        """
        Deep copy support for the copy module.
        """
        # matrices are numeric, so shallow vs deep is effectively the same here
        return self.__copy__()

    def __reduce__(self):
        """
        Pickle support: reprunes to (class, (matrix,))
        """
        return (type(self), (self.matrix.copy(),))


def _create_frame_convention(
    x: Direction, y: Direction, z: Direction
) -> Type[Transform]:
    # sanity check
    if len({x, y, z}) != 3:
        raise ValueError("x, y, z must be three distinct Directions")

    # compute handedness
    x_vec = _DIR_TO_VEC[x]
    y_vec = _DIR_TO_VEC[y]
    z_vec = _DIR_TO_VEC[z]
    is_right_handed = bool(np_allclose(np_cross(x_vec, y_vec), z_vec))
    # ensure orthobonality
    if not np_allclose(np_dot(x_vec, y_vec), 0) or not np_allclose(np_dot(x_vec, z_vec), 0) or not np_allclose(np_dot(y_vec, z_vec), 0):
        raise ValueError("x, y, z must be orthogonal Directions")

    # define the basis vectors for the class type
    def x_fn(self): return pure_rotation_if_possible(
        self.matrix[:3, :3])[:3, 0]

    def x_inv_fn(self): return - \
        pure_rotation_if_possible(self.matrix[:3, :3])[:3, 0]

    def y_fn(self): return pure_rotation_if_possible(
        self.matrix[:3, :3])[:3, 1]

    def y_inv_fn(self): return - \
        pure_rotation_if_possible(self.matrix[:3, :3])[:3, 1]

    def z_fn(self): return pure_rotation_if_possible(
        self.matrix[:3, :3])[:3, 2]
    def z_inv_fn(self): return - \
        pure_rotation_if_possible(self.matrix[:3, :3])[:3, 2]

    if x == Direction.FORWARD:
        forward = x_fn
        backward = x_inv_fn
        forward_basis = x_vec
    elif x == Direction.BACKWARD:
        backward = x_fn
        forward = x_inv_fn
        backward_basis = x_vec
    elif x == Direction.LEFT:
        left = x_fn
        right = x_inv_fn
        left_basis = x_vec
    elif x == Direction.RIGHT:
        right = x_fn
        left = x_inv_fn
        right_basis = x_vec
    elif x == Direction.UP:
        up = x_fn
        down = x_inv_fn
        up_basis = x_vec
    elif x == Direction.DOWN:
        down = x_fn
        up = x_inv_fn
        down_basis = x_vec
    else:
        raise ValueError("Invalid direction for x")

    if y == Direction.FORWARD:
        forward = y_fn
        backward = y_inv_fn
        forward_basis = y_vec
    elif y == Direction.BACKWARD:
        backward = y_fn
        forward = y_inv_fn
        backward_basis = y_vec
    elif y == Direction.LEFT:
        left = y_fn
        right = y_inv_fn
        left_basis = y_vec
    elif y == Direction.RIGHT:
        right = y_fn
        left = y_inv_fn
        right_basis = y_vec
    elif y == Direction.UP:
        up = y_fn
        down = y_inv_fn
        up_basis = y_vec
    elif y == Direction.DOWN:
        down = y_fn
        up = y_inv_fn
        down_basis = y_vec
    else:
        raise ValueError("Invalid direction for y")

    if z == Direction.FORWARD:
        forward = z_fn
        backward = z_inv_fn
        forward_basis = z_vec
    elif z == Direction.BACKWARD:
        backward = z_fn
        forward = z_inv_fn
        backward_basis = z_vec
    elif z == Direction.LEFT:
        left = z_fn
        right = z_inv_fn
        left_basis = z_vec
    elif z == Direction.RIGHT:
        right = z_fn
        left = z_inv_fn
        right_basis = z_vec
    elif z == Direction.UP:
        up = z_fn
        down = z_inv_fn
        up_basis = z_vec
    elif z == Direction.DOWN:
        down = z_fn
        up = z_inv_fn
        down_basis = z_vec
    else:
        raise ValueError("Invalid direction for z")

    # pack onto a new subclass
    cls_name = f"Transform<{x.name},{y.name},{z.name}>"
    props = {
        # world directions
        "forward": property(forward),
        "backward": property(backward),
        "left": property(left),
        "right": property(right),
        "up": property(up),
        "down": property(down),

        # basis information
        "is_right_handed": staticmethod(lambda: is_right_handed),
        "label_x": staticmethod(lambda: x),
        "label_y": staticmethod(lambda: y),
        "label_z": staticmethod(lambda: z),
        "basis_x": staticmethod(lambda: x_vec),
        "basis_y": staticmethod(lambda: y_vec),
        "basis_z": staticmethod(lambda: z_vec),
        "basis_matrix": staticmethod(lambda: np_array([x_vec, y_vec, z_vec], dtype=np_float64)),
        "basis_matrix_inv": staticmethod(lambda: np_array([x_vec, y_vec, z_vec], dtype=np_float64).T),
        "basis_forward": staticmethod(lambda: forward_basis),
        "basis_backward": staticmethod(lambda: backward_basis),
        "basis_left": staticmethod(lambda: left_basis),
        "basis_right": staticmethod(lambda: right_basis),
        "basis_up": staticmethod(lambda: up_basis),
        "basis_down": staticmethod(lambda: down_basis),
    }
    NewFrame = type(cls_name, (Transform,), props)
    # make it a slots dataclass
    return dataclass(NewFrame, slots=True)


# needed for pickling
_FRAME_REGISTRY = {}
for x in Direction:
    for y in Direction:
        for z in Direction:
            try:
                frame = _create_frame_convention(x, y, z)
                globals()[frame.__name__] = frame
                _FRAME_REGISTRY[(x, y, z)] = frame
            except ValueError:
                pass


def define_convention(x: Direction = Direction.FORWARD, y: Direction = Direction.LEFT, z: Direction = Direction.UP) -> Type[Transform]:
    """
    Get the transform type for the given frame convention.

    Parameters
    ----------
    x : Direction
        The x direction of the frame convention.
    y : Direction
        The y direction of the frame convention.
    z : Direction
        The z direction of the frame convention.

    Returns
    -------
    Type[Transform]
        The transform type for the given frame convention.
    """
    val = _FRAME_REGISTRY.get((x, y, z), None)
    if val is None:
        raise ValueError(
            f"Frame convention {x}, {y}, {z} not valid. Must be orthogonal.")
    return val
