import numpy as np
from numpy import asarray as np_asarray
from typing import Union, Type, TYPE_CHECKING
from polyframe2.direction import Direction
from polyframe2.math import svd_rotation, svd_scale, svd_shear, inv4, inv3

if TYPE_CHECKING:
    # Avoid circular import issues
    from polyframe2.local_transform import LocalTransform

_EYE4 = np.eye(4, dtype=np.float64)


class AffineMatrix:
    __slots__ = ('matrix',)
    _local_transform_type: Type[LocalTransform]

    def __init__(self, matrix: np.ndarray):
        matrix = np_asarray(matrix)
        if matrix.shape != (4, 4):
            raise ValueError("Matrix must be 4x4.")
        self.matrix = matrix

    @classmethod
    def identity(cls) -> 'AffineMatrix':
        """Create an identity affine matrix."""
        return cls(_EYE4.copy())

    @classmethod
    def from_unsafe(cls, matrix: np.ndarray) -> 'AffineMatrix':
        """Create an affine matrix without checking its shape.

        Args:
            matrix (np.ndarray): The matrix to be used.

        Returns:
            AffineMatrix: An instance of AffineMatrix with the provided matrix.
        """
        instance = object.__new__(cls)
        instance.matrix = matrix
        return instance

    @property
    def translation(self) -> np.ndarray:
        """
        Get the translation vector from the affine matrix.

        Returns:
            The translation vector as a 3D numpy array.
        """
        return self.matrix[:3, 3]

    @property
    def rotation(self) -> np.ndarray:
        """
        Get the orthonormal rotation matrix from the affine matrix.

        Returns:
            The orthonormal rotation matrix as a 3x3 numpy array.
        """
        return svd_rotation(self.matrix)

    @property
    def upper_left(self) -> np.ndarray:
        """
        Get the upper-left 3x3 submatrix from the affine matrix.

        Returns:
            The upper-left 3x3 submatrix as a numpy array.
        """
        return self.matrix[:3, :3]

    @property
    def scale(self) -> np.ndarray:
        """
        Get the pure scale vector from the affine matrix.

        Returns:
            The scale vector as a 3D numpy array.
        """
        return svd_scale(self.matrix)

    @property
    def shear(self) -> np.ndarray:
        """
        Get the pure shear matrix from the affine matrix.

        Returns:
            The shear matrix as a 3x3 numpy array.
        """
        return svd_shear(self.matrix)

    @property
    def perspective(self) -> np.ndarray:
        """
        Get the perspective vector from the affine matrix.

        Returns:
            The perspective vector as a 3D numpy array.
        """
        return self.matrix[3, :3]

    @property
    def perspective_multiplier(self) -> float:
        """
        Get the perspective multiplier from the affine matrix.

        Returns:
            The perspective multiplier as a float.
        """
        return self.matrix[3, 3]

    ###########
    # Directional properties derived from coordinate system convention
    #

    @property
    def forward(self) -> np.ndarray:
        """
        Get the forward vector from the current rotation and basis vector.

        Returns:
            The 3D “forward” direction after applying this transform's rotation.
        """
        ...

    @property
    def forward_orthonormal(self) -> np.ndarray:
        """
        Get the forward orthonormal vector from the current rotation and basis vector.

        Returns:
            The 3D “forward” direction after applying this transform's rotation.
        """
        ...

    @property
    def backward(self) -> np.ndarray:
        """
        Get the backward vector from the current rotation and basis vector.

        Returns:
            The 3D “backward” direction after applying this transform's rotation.
        """
        ...

    @property
    def backward_orthonormal(self) -> np.ndarray:
        """
        Get the backward orthonormal vector from the current rotation and basis vector.

        Returns:
            The 3D “backward” direction after applying this transform's rotation.
        """
        ...

    @property
    def left(self) -> np.ndarray:
        """
        Get the left vector from the current rotation and basis vector.

        Returns:
            The 3D “left” direction after applying this transform's rotation.
        """
        ...

    @property
    def left_orthonormal(self) -> np.ndarray:
        """
        Get the left orthonormal vector from the current rotation and basis vector.

        Returns:
            The 3D “left” direction after applying this transform's rotation.
        """
        ...

    @property
    def right(self) -> np.ndarray:
        """
        Get the right vector from the current rotation and basis vector.

        Returns:
            The 3D “right” direction after applying this transform's rotation.
        """
        ...

    @property
    def right_orthonormal(self) -> np.ndarray:
        """
        Get the right orthonormal vector from the current rotation and basis vector.

        Returns:
            The 3D “right” direction after applying this transform's rotation.
        """
        ...

    @property
    def up(self) -> np.ndarray:
        """
        Get the up vector from the current rotation and basis vector.

        Returns:
            The 3D “up” direction after applying this transform's rotation.
        """
        ...

    @property
    def up_orthonormal(self) -> np.ndarray:
        """
        Get the up orthonormal vector from the current rotation and basis vector.

        Returns:
            The 3D “up” direction after applying this transform's rotation.
        """
        ...

    @property
    def down(self) -> np.ndarray:
        """
        Get the down vector from the current rotation and basis vector.

        Returns:
            The 3D “down” direction after applying this transform's rotation.
        """
        ...

    @property
    def down_orthonormal(self) -> np.ndarray:
        """
        Get the down orthonormal vector from the current rotation and basis vector.

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
    def is_left_handed() -> bool:
        """
        Check if the coordinate system is left-handed.
        Implemented in the generated code.

        Returns:
            True if the coordinate system is left-handed, False otherwise.
        """
        ...

    @staticmethod
    def get_x_label() -> Direction:
        """
        Returns the label for the x-axis of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def get_y_label() -> Direction:
        """
        Returns the label for the y-axis of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def get_z_label() -> Direction:
        """
        Returns the label for the z-axis of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def get_x_basis() -> np.ndarray:
        """
        Returns the basis x vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def get_y_basis() -> np.ndarray:
        """
        Returns the basis y vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def get_z_basis() -> np.ndarray:
        """
        Returns the basis z vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def get_basis_matrix() -> np.ndarray:
        """
        Returns the basis matrix of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def get_basis_matrix_inv() -> np.ndarray:
        """
        Returns the inverse of the basis matrix of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def get_basis_forward() -> np.ndarray:
        """
        Returns the basis forward vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def get_basis_backward() -> np.ndarray:
        """
        Returns the basis backward vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def get_basis_left() -> np.ndarray:
        """
        Returns the basis left vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def get_basis_right() -> np.ndarray:
        """
        Returns the basis right vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def get_basis_up() -> np.ndarray:
        """
        Returns the basis up vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def get_basis_down() -> np.ndarray:
        """
        Returns the basis down vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def get_change_of_basis_matrix(origin: Type["AffineMatrix"], destination: Type["AffineMatrix"]) -> np.ndarray:
        """
        Returns the change of basis matrix from the origin to the destination.

        Args:
            origin: The origin coordinate system.
            destination: The destination coordinate system.

        Returns:
            The change of basis matrix as a 3x3 numpy array.
        """
        return destination.get_basis_matrix_inv() @ origin.get_basis_matrix()

    def change_basis_to(self, other: "AffineMatrix") -> "AffineMatrix":
        """
        Change the basis of this transform to another transform.

        Args:
            other: The target AffineMatrix to change the basis to.

        Returns:
            A new AffineMatrix with the same translation, rotation, and scale,
            but in the new coordinate system.
        """
        t_other = type(other)
        R = self.get_change_of_basis_matrix(type(self), t_other)
        return t_other.from_unsafe(R @ self.matrix @ inv3(R))

    def to_local_transform(self) -> "LocalTransform":
        """Convert this affine matrix to a local transform.

        If shear is present, it will be ignored.
        If scale is non-uniform, the max will be used.

        Returns:
            LocalTransform: A new LocalTransform instance with the same translation and rotation, with the same basis matrix.
        """
        return self._local_transform_type.from_unsafe(
            self.translation, self.rotation, max(self.scale))

    def inverse(self) -> 'AffineMatrix':
        """Return the inverse of this affine matrix."""
        return AffineMatrix(inv4(self.matrix))

    def __matmul__(self, other: Union['AffineMatrix', np.ndarray]) -> 'AffineMatrix':
        if isinstance(other, AffineMatrix):
            other_matrix = other.matrix
        elif isinstance(other, np.ndarray):
            if other.shape != (4, 4):
                raise ValueError("Matrix must be 4x4.")
            other_matrix = other
        else:
            raise TypeError("Unsupported type for matrix multiplication.")

        result_matrix = self.matrix @ other_matrix
        return AffineMatrix(result_matrix)

    def __rmatmul__(self, other: Union['AffineMatrix', np.ndarray]) -> 'AffineMatrix':
        if isinstance(other, AffineMatrix):
            other_matrix = other.matrix
        elif isinstance(other, np.ndarray):
            if other.shape != (4, 4):
                raise ValueError("Matrix must be 4x4.")
            other_matrix = other
        else:
            raise TypeError("Unsupported type for matrix multiplication.")

        result_matrix = other_matrix @ self.matrix
        return AffineMatrix(result_matrix)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.matrix})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.matrix})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, np.ndarray):
            return np.array_equal(self.matrix, other)

        elif isinstance(other, AffineMatrix):
            return np.array_equal(self.matrix, other.matrix)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        if isinstance(other, np.ndarray):
            return not np.array_equal(self.matrix, other)

        elif isinstance(other, AffineMatrix):
            return not np.array_equal(self.matrix, other.matrix)
        return NotImplemented
