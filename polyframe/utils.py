# utils.py

import numpy as np
from typing import Union
from polyframe.geometry import pure_rotation_if_possible
from numba import njit
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
_EYE4 = np.eye(4)


@njit(cache=True)
def to_matrix(translation: np.ndarray, rotation: np.ndarray, scale: Union[float, np.ndarray]) -> np.ndarray:
    """Convert the transform to a 4x4 transformation matrix. Scale first, then rotate."""
    m = _EYE4.copy()
    # Scale each column of the rotation matrix by the corresponding scale factor.
    m[:3, :3] = rotation * scale
    m[:3, 3] = translation
    return m


@njit(cache=True)
def to_inv_matrix(translation: np.ndarray, rotation: np.ndarray, scale: Union[float, np.ndarray]) -> np.ndarray:
    """Convert the transform to a 4x4 transformation matrix. Scale first, then rotate."""
    m = _EYE4.copy()
    # Scale each column of the rotation matrix by the corresponding scale factor.
    m[:3, :3] = rotation.T / scale
    m[:3, 3] = -rotation.T @ translation
    return m


@njit(cache=True)
def decompose_matrix(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Decompose a 4x4 transformation matrix into translation, rotation, and scale.
    Ignore shear and non-uniform scale.

    Args:
        mat (_type_): 4x4 transformation matrix

    Returns:
        tuple[np.ndarray, np.ndarray, float]: translation, rotation, scale
    """
    # mat: float64[4,4]
    # 1) translation
    t = mat[:3, 3].copy()

    # 2) upper 3×3
    M = mat[:3, :3]

    # 3) SVD
    U, sigma, Vt = np.linalg.svd(M)

    # 4) proper rotation
    R = U @ Vt
    # if reflection, flip last column of Vt
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = U @ Vt

    # 5) uniform scale = max singular value
    scale = sigma.max()

    return t, R, scale


@njit(cache=True)
def is_rigid(mat4: np.ndarray, tol=1e-6) -> bool:
    R = mat4[:3, :3]
    # 1) R must be orthonormal, det≈+1
    if not (np.allclose(R @ R.T, np.eye(3), atol=tol)
            and abs(np.linalg.det(R) - 1.0) <= tol):
        return False

    # 2) bottom row must be [0,0,0,1]
    if not np.allclose(mat4[3, :], [0.0, 0.0, 0.0, 1.0], atol=tol):
        return False

    # (the translation column can be anything)
    return True


@njit(cache=True)
def decompose_scale_shear(mat3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the affine block A into

        S  - pure scale vector (sx, sy, sz)
        H  - unit-diagonal shear matrix      (A = R · diag(S) · H)

    The implementation is deliberately simple and numerically stable:
    obtain R from the polar decomposition and peel it off.
    """
    R = pure_rotation_if_possible(mat3)
    P = R.T @ mat3                     # stretch/shear block   (diag = scale)
    S_vec = np.diag(P)
    H_mat = np.linalg.solve(np.diag(S_vec), P)
    return S_vec, H_mat


@njit(cache=True)
def pure_rotation_if_possible_and_basis_matmul(M: np.ndarray, basis_vector: np.ndarray) -> np.ndarray:
    return pure_rotation_if_possible(M) @ basis_vector
