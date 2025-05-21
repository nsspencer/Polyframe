from numba import njit, float64
import numpy as np
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


@njit(fastmath=True, inline='always', cache=True)
def det3(M):
    """Determinant of a 3 x 3 (faster than np.linalg.det for tiny mats)."""
    return (
        M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
        - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
        + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])
    )


@njit(float64(float64[:, :]), fastmath=True, cache=True, inline="always")
def det4(m):
    """
    Determinant of a 4x4 matrix using the 12-subfactor scheme
    (fewer multiplies than Laplace expansion; zero temporaries).

    Parameters
    ----------
    m : (4,4) float64 array (C-contiguous)

    Returns
    -------
    float64
        det(m)
    """
    # sub-factors from the first two rows (s-vector)
    s0 = m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]
    s1 = m[0, 0] * m[1, 2] - m[1, 0] * m[0, 2]
    s2 = m[0, 0] * m[1, 3] - m[1, 0] * m[0, 3]
    s3 = m[0, 1] * m[1, 2] - m[1, 1] * m[0, 2]
    s4 = m[0, 1] * m[1, 3] - m[1, 1] * m[0, 3]
    s5 = m[0, 2] * m[1, 3] - m[1, 2] * m[0, 3]

    # complementary sub-factors from the last two rows (c-vector)
    c5 = m[2, 2] * m[3, 3] - m[3, 2] * m[2, 3]
    c4 = m[2, 1] * m[3, 3] - m[3, 1] * m[2, 3]
    c3 = m[2, 1] * m[3, 2] - m[3, 1] * m[2, 2]
    c2 = m[2, 0] * m[3, 3] - m[3, 0] * m[2, 3]
    c1 = m[2, 0] * m[3, 2] - m[3, 0] * m[2, 2]
    c0 = m[2, 0] * m[3, 1] - m[3, 0] * m[2, 1]

    # assemble determinant (6 FMAs after LLVM fusion)
    return (
        s0 * c5 - s1 * c4 + s2 * c3
        + s3 * c2 - s4 * c1 + s5 * c0
    )


@njit(fastmath=True, inline='always', cache=True)
def inv3(M):
    """Analytic inverse of a 3 x 3"""
    d = det3(M)
    invd = 1.0 / d
    out = np.empty((3, 3), dtype=M.dtype)
    out[0, 0] = (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]) * invd
    out[0, 1] = -(M[0, 1] * M[2, 2] - M[0, 2] * M[2, 1]) * invd
    out[0, 2] = (M[0, 1] * M[1, 2] - M[0, 2] * M[1, 1]) * invd
    out[1, 0] = -(M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0]) * invd
    out[1, 1] = (M[0, 0] * M[2, 2] - M[0, 2] * M[2, 0]) * invd
    out[1, 2] = -(M[0, 0] * M[1, 2] - M[0, 2] * M[1, 0]) * invd
    out[2, 0] = (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]) * invd
    out[2, 1] = -(M[0, 0] * M[2, 1] - M[0, 1] * M[2, 0]) * invd
    out[2, 2] = (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) * invd
    return out


# -------------------------------------------------------------------------
# analytic inverse for a 4x4 matrix
# -------------------------------------------------------------------------
@njit(float64[:, :](float64[:, :]), fastmath=True, cache=True)
def inv4(m):
    """
    Fast analytic inverse of a 4x4 matrix.
    Raises ZeroDivisionError if the matrix is singular.
    """

    # ---- step 1: re-use the 12 sub-factors (exactly as det4) -------------
    s0 = m[0, 0]*m[1, 1] - m[1, 0]*m[0, 1]
    s1 = m[0, 0]*m[1, 2] - m[1, 0]*m[0, 2]
    s2 = m[0, 0]*m[1, 3] - m[1, 0]*m[0, 3]
    s3 = m[0, 1]*m[1, 2] - m[1, 1]*m[0, 2]
    s4 = m[0, 1]*m[1, 3] - m[1, 1]*m[0, 3]
    s5 = m[0, 2]*m[1, 3] - m[1, 2]*m[0, 3]

    c5 = m[2, 2]*m[3, 3] - m[3, 2]*m[2, 3]
    c4 = m[2, 1]*m[3, 3] - m[3, 1]*m[2, 3]
    c3 = m[2, 1]*m[3, 2] - m[3, 1]*m[2, 2]
    c2 = m[2, 0]*m[3, 3] - m[3, 0]*m[2, 3]
    c1 = m[2, 0]*m[3, 2] - m[3, 0]*m[2, 2]
    c0 = m[2, 0]*m[3, 1] - m[3, 0]*m[2, 1]

    # ---- step 2: determinant & reciprocal --------------------------------
    det = (s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0)
    if det == 0.0:
        raise ZeroDivisionError("Matrix is singular and cannot be inverted")
    inv_det = 1.0 / det

    # ---- step 3: build the adjugate (transposed cofactor matrix) ---------
    out = np.empty((4, 4), dtype=m.dtype)

    out[0, 0] = (m[1, 1]*c5 - m[1, 2]*c4 + m[1, 3]*c3) * inv_det
    out[0, 1] = (-m[0, 1]*c5 + m[0, 2]*c4 - m[0, 3]*c3) * inv_det
    out[0, 2] = (m[3, 1]*s5 - m[3, 2]*s4 + m[3, 3]*s3) * inv_det
    out[0, 3] = (-m[2, 1]*s5 + m[2, 2]*s4 - m[2, 3]*s3) * inv_det

    out[1, 0] = (-m[1, 0]*c5 + m[1, 2]*c2 - m[1, 3]*c1) * inv_det
    out[1, 1] = (m[0, 0]*c5 - m[0, 2]*c2 + m[0, 3]*c1) * inv_det
    out[1, 2] = (-m[3, 0]*s5 + m[3, 2]*s2 - m[3, 3]*s1) * inv_det
    out[1, 3] = (m[2, 0]*s5 - m[2, 2]*s2 + m[2, 3]*s1) * inv_det

    out[2, 0] = (m[1, 0]*c4 - m[1, 1]*c2 + m[1, 3]*c0) * inv_det
    out[2, 1] = (-m[0, 0]*c4 + m[0, 1]*c2 - m[0, 3]*c0) * inv_det
    out[2, 2] = (m[3, 0]*s4 - m[3, 1]*s2 + m[3, 3]*s0) * inv_det
    out[2, 3] = (-m[2, 0]*s4 + m[2, 1]*s2 - m[2, 3]*s0) * inv_det

    out[3, 0] = (-m[1, 0]*c3 + m[1, 1]*c1 - m[1, 2]*c0) * inv_det
    out[3, 1] = (m[0, 0]*c3 - m[0, 1]*c1 + m[0, 2]*c0) * inv_det
    out[3, 2] = (-m[3, 0]*s3 + m[3, 1]*s1 - m[3, 2]*s0) * inv_det
    out[3, 3] = (m[2, 0]*s3 - m[2, 1]*s1 + m[2, 2]*s0) * inv_det

    return out


@njit(fastmath=True, cache=True)
def decompose_affine_matrix(matrix: float64[:, :]):  # type: ignore
    """
    Decompose a 4x4 affine matrix into

        mat = R @ diag(scale) @ shear

    where
        R      : nearest right-handed rotation (3x3, det = +1)
        scale  : length-3 vector of positive scales
        shear  : 3x3 upper-triangular, unit diagonal

    Returns (R, scale, shear).
    """
    M = matrix[:3, :3]

    # (M = U Σ Vᵀ)
    U, scale, Vh = np.linalg.svd(M)

    # Ensure a proper rotation (det = +1)
    if det3(U @ Vh) < 0.0:
        U[:, 2] = -U[:, 2]
        scale[2] = -scale[2]

    # nearest rotation
    R = U @ Vh

    # shear_raw = Vᵀ Σ V   (symmetric PSD)
    # Vᵀ·diag(s)·V
    shear_raw = (Vh.T * scale) @ Vh

    # Convert symmetric shear_raw into upper-triangular with unit diagonal
    # by dividing rows (or columns) by the scales:
    shear = np.empty((3, 3), dtype=mat.dtype)
    for i in range(3):
        for j in range(3):
            shear[i, j] = shear_raw[i, j] / scale[j]

    # Force exact 1’s on the diagonal (numerical tidy-up)
    shear[0, 0] = shear[1, 1] = shear[2, 2] = 1.0

    return R, scale, shear


@njit(fastmath=True, cache=True)
def svd_rotation(matrix: float64[:, :]):  # type: ignore
    """
    Nearest orthonormal rotation matrix to a given 3x3 matrix.
    Uses Singular Value Decomposition (SVD) to find the nearest orthonormal
    rotation matrix.

    Parameters
    ----------
    matrix : (4, 4) float64 array
        The input matrix.

    Returns
    -------
    (3, 3) float64 array
        The nearest orthonormal rotation matrix.
    """
    U, _, Vh = np.linalg.svd(matrix[:3, :3])
    # Ensure a proper rotation (det = +1)
    if det3(U @ Vh) < 0.0:
        U[:, 2] = -U[:, 2]
    return U @ Vh


@njit(fastmath=True, cache=True)
def svd_scale(matrix: float64[:, :]):  # type: ignore
    """
    Decompose a 4x4 matrix into scale components.

    Parameters
    ----------
    matrix : (4, 4) float64 array
        The input matrix.

    Returns
    -------
    (3,) float64 array
        The scale vector.
    """
    M = matrix[:3, :3]

    # (M = U Σ Vᵀ)
    U, scale, Vh = np.linalg.svd(M)

    # Ensure a proper rotation (det = +1)
    if det3(U @ Vh) < 0.0:
        scale[2] = -scale[2]

    return scale


@njit(fastmath=True, cache=True)
def svd_shear(matrix: float64[:, :]):  # type: ignore
    """
    Decompose a 4x4 matrix into normalized shear components.

    Parameters
    ----------
    matrix : (4, 4) float64 array
        The input matrix.

    Returns
    -------
    (3, 3) float64 array
        The shear matrix normalized by scale.
    """
    M = matrix[:3, :3]

    # (M = U Σ Vᵀ)
    U, scale, Vh = np.linalg.svd(M)

    # Ensure a proper rotation (det = +1)
    if det3(U @ Vh) < 0.0:
        scale[2] = -scale[2]

    # shear_raw = Vᵀ Σ V   (symmetric PSD)
    # Vᵀ·diag(s)·V
    shear_raw = (Vh.T * scale) @ Vh

    # Convert symmetric shear_raw into upper-triangular with unit diagonal
    # by dividing rows (or columns) by the scales:
    shear = np.empty((3, 3), dtype=mat.dtype)
    for i in range(3):
        for j in range(3):
            shear[i, j] = shear_raw[i, j] / scale[j]

    # Force exact 1’s on the diagonal (numerical tidy-up)
    shear[0, 0] = shear[1, 1] = shear[2, 2] = 1.0

    return shear


if __name__ == "__main__":
    import timeit
    from scipy.spatial.transform import Rotation as R

    rot = R.from_euler('xyz', [45, 45, 45], degrees=True)
    scale = 2.0
    mat = rot.as_matrix()
    mat4 = np.eye(4)
    mat4[:3, :3] = mat * scale

    det3(mat)
    det4(mat4)
    inv3(mat)
    inv4(mat4)

    N = 1_000_000
    print("det3:", timeit.timeit(lambda: det3(mat), number=N))
    print("det4:", timeit.timeit(lambda: det4(mat4), number=N))

    print("inv3:", timeit.timeit(lambda: inv3(mat), number=N))
    print("inv4:", timeit.timeit(lambda: inv4(mat4), number=N))

    np.testing.assert_allclose(
        inv4(mat4) @ mat4, np.eye(4), atol=1e-6, err_msg="inv4 failed"
    )
    np.testing.assert_allclose(
        inv3(mat) @ mat, np.eye(3), atol=1e-6, err_msg="inv3 failed"
    )
    np.testing.assert_allclose(
        det4(mat4), np.linalg.det(mat4), atol=1e-6, err_msg="det4 failed"
    )
    np.testing.assert_allclose(
        det3(mat), np.linalg.det(mat), atol=1e-6, err_msg="det3 failed"
    )

    out_rot, out_scale, out_shear = decompose_affine_matrix(mat4)
    np.testing.assert_allclose(out_rot, mat)
    np.testing.assert_allclose(out_scale, np.array([scale] * 3))
    np.testing.assert_allclose(out_shear, np.eye(3), atol=1e-10)

    out_rot = svd_rotation(mat4)
    np.testing.assert_allclose(out_rot, mat)
    out_scale = svd_scale(mat4)
    np.testing.assert_allclose(out_scale, np.array([scale] * 3))
    out_shear = svd_shear(mat4)
    np.testing.assert_allclose(out_shear, np.eye(3), atol=1e-10)
    print("svd_rotation:", timeit.timeit(
        lambda: svd_rotation(mat4), number=N))
    print("svd_scale:", timeit.timeit(
        lambda: svd_scale(mat4), number=N))
    print("svd_shear:", timeit.timeit(
        lambda: svd_shear(mat4), number=N))
    print("decompose_affine_matrix svd:", timeit.timeit(
        lambda: decompose_affine_matrix(mat4), number=N))
    pass
