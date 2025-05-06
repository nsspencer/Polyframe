# utils.py

import numpy as np
from numba import njit
from typing import Tuple


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
