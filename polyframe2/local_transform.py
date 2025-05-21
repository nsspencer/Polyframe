import numpy as np
from numpy import asarray as np_asarray
from polyframe2.utils import to_matrix, to_inv_matrix, decompose_matrix
from polyframe2.geometry import quaternion_to_rotation, euler_to_rotation, rotation_to_quaternion, rotation_to_euler
from polyframe2.direction import Direction, _DIR_TO_VEC
from typing import Union, Iterable, Type, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular import issues
    from polyframe2.affine_matrix import AffineMatrix


_BASE_TRANSLATION = np.zeros(3)
_BASE_ROTATION = np.eye(3)


class LocalTransform:
    """
    A class representing a local transformation in 3D space.
    """
    __slots__ = ('_translation', '_rotation', '_scale', '_dirty', '_matrix')
    _affine_matrix_type: Type[AffineMatrix]

    def __init__(self, translation: Union[None, Iterable] = None, rotation: Union[None, Iterable] = None, scale: Union[None, float] = None):
        self._dirty: bool = True  # dirty flag on init
        self._matrix: Union[None, AffineMatrix] = None  # init to None

        if translation is not None:
            self._translation = np_asarray(translation)
            if self._translation.shape != (3,):
                raise ValueError(
                    f"Translation must be a 3D vector, got {self._translation.shape}")
        else:
            self._translation = _BASE_TRANSLATION.copy()

        if rotation is not None:
            self._rotation = np_asarray(rotation)
            if self._rotation.shape != (3, 3):
                raise ValueError(
                    f"Rotation must be a 3x3 matrix, got {self._rotation.shape}")
            if not np.allclose(np.dot(self._rotation, self._rotation.T), _BASE_ROTATION):
                raise ValueError("Rotation matrix must be orthogonal.")
        else:
            self._rotation = _BASE_ROTATION.copy()

        if scale is None:
            self._scale = 1.0
        else:
            scale = float(scale)
            if scale == 0.0:
                raise ValueError("Scale cannot be zero.")
            self._scale = scale

    @classmethod
    def identity(cls) -> "LocalTransform":
        """
        Create an identity LocalTransform.

        Returns:
            A LocalTransform object with zero translation, identity rotation, and scale of 1.
        """
        return cls.from_unsafe(_BASE_TRANSLATION.copy(), _BASE_ROTATION.copy(), 1.0)

    @classmethod
    def from_unsafe(cls, translation: np.ndarray, rotation: np.ndarray, scale: float):
        """Create a LocalTransform without checking the validity of the inputs. Useful for performance when you are sure of the input shapes and types."""
        instance = object.__new__(cls)
        instance._dirty = True  # dirty flag on init
        instance._matrix = None
        instance._translation = translation
        instance._rotation = rotation
        instance._scale = scale
        return instance

    @classmethod
    def from_translation(cls, translation: np.ndarray):
        """
        Create a LocalTransform from a translation vector.

        Args:
            translation: A 3D numpy array representing the translation.

        Returns:
            A LocalTransform object with the given translation and default rotation and scale.
        """
        translation = np_asarray(translation)
        if translation.shape != (3,):
            raise ValueError(
                f"Translation must be a 3D vector, got {translation.shape}")
        return cls.from_unsafe(translation, _BASE_ROTATION.copy(), 1.0)

    @classmethod
    def from_translation_rotation(cls, translation: np.ndarray, rotation: np.ndarray):
        """
        Create a LocalTransform from a translation vector and rotation matrix.

        Args:
            translation: A 3D numpy array representing the translation.
            rotation: A 3x3 numpy array representing the rotation matrix.

        Returns:
            A LocalTransform object with the given translation and rotation.
        """
        translation = np_asarray(translation)
        if translation.shape != (3,):
            raise ValueError(
                f"Translation must be a 3D vector, got {translation.shape}")
        rotation = np_asarray(rotation)
        if rotation.shape != (3, 3):
            raise ValueError(
                f"Rotation must be a 3x3 matrix, got {rotation.shape}")
        return cls.from_unsafe(translation, rotation, 1.0)

    @classmethod
    def from_translation_rotation_scale(cls, translation: np.ndarray, rotation: np.ndarray, scale: float):
        """
        Create a LocalTransform from a translation vector, rotation matrix, and scale.

        Args:
            translation: A 3D numpy array representing the translation.
            rotation: A 3x3 numpy array representing the rotation matrix.
            scale: A float representing the scale.

        Returns:
            A LocalTransform object with the given translation, rotation, and scale.
        """
        translation = np_asarray(translation)
        if translation.shape != (3,):
            raise ValueError(
                f"Translation must be a 3D vector, got {translation.shape}")
        rotation = np_asarray(rotation)
        if rotation.shape != (3, 3):
            raise ValueError(
                f"Rotation must be a 3x3 matrix, got {rotation.shape}")
        scale = float(scale)
        if scale == 0.0:
            raise ValueError("Scale cannot be zero.")
        return cls.from_unsafe(translation, rotation, scale)

    @classmethod
    def from_rotation(cls, rotation: np.ndarray):
        """
        Create a LocalTransform from a rotation matrix.

        Args:
            rotation: A 3x3 numpy array representing the rotation matrix.

        Returns:
            A LocalTransform object with the given rotation and default translation and scale.
        """
        rotation = np_asarray(rotation)
        if rotation.shape != (3, 3):
            raise ValueError(
                f"Rotation must be a 3x3 matrix, got {rotation.shape}")
        return cls.from_unsafe(_BASE_TRANSLATION.copy(), rotation, 1.0)

    @classmethod
    def from_scale(cls, scale: float):
        """
        Create a LocalTransform from a scale factor.

        Args:
            scale: A float representing the scale.

        Returns:
            A LocalTransform object with the given scale and default translation and rotation.
        """
        scale = float(scale)
        if scale == 0.0:
            raise ValueError("Scale cannot be zero.")
        return cls.from_unsafe(_BASE_TRANSLATION.copy(), _BASE_ROTATION.copy(), scale)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        """
        Create a LocalTransform from a 4x4 transformation matrix.

        If the input matrix contains non-uniform scale, the largest value will be used. Any shear in the input matrix will be ignored.

        Args:
            matrix: A 4x4 numpy array representing the transformation matrix.

        Returns:
            A LocalTransform object.
        """
        return cls.from_unsafe(*decompose_matrix(matrix))

    @property
    def translation(self) -> np.ndarray:
        """
        Get the translation vector of the transform.

        Returns:
            The translation vector as a 3D numpy array.
        """
        return self._translation

    @translation.setter
    def translation(self, value: Union[Iterable, np.ndarray]):
        """Set the translation vector."""
        self._translation = np_asarray(value)
        self._dirty = True

    @property
    def rotation(self) -> np.ndarray:
        """
        Get the pure rotation matrix of the transform.

        Returns:
            The rotation matrix as a 3x3 numpy array.
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value: Union[Iterable, np.ndarray]):
        """Set the rotation matrix."""
        self._rotation = np_asarray(value)
        self._dirty = True

    @property
    def scale(self) -> float:
        """
        Get the uniform scale factor of the transform.

        Returns:
            The scale factor as a float.
        """
        return self._scale

    @scale.setter
    def scale(self, value: float):
        """Set the scale factor."""
        self._scale = float(value)
        self._dirty = True

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
    def backward(self) -> np.ndarray:
        """
        Get the backward vector from the current rotation and basis vector.

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
    def right(self) -> np.ndarray:
        """
        Get the right vector from the current rotation and basis vector.

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
    def down(self) -> np.ndarray:
        """
        Get the down vector from the current rotation and basis vector.

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
    def get_change_of_basis_matrix(origin: Type["LocalTransform"], destination: Type["LocalTransform"]) -> np.ndarray:
        """
        Returns the change of basis matrix from the origin to the destination.

        Args:
            origin: The origin coordinate system.
            destination: The destination coordinate system.

        Returns:
            The change of basis matrix as a 3x3 numpy array.
        """
        return destination.get_basis_matrix_inv() @ origin.get_basis_matrix()

    def change_basis_to(self, other: "LocalTransform") -> "LocalTransform":
        """
        Change the basis of this transform to another transform.

        Args:
            other: The target LocalTransform to change the basis to.

        Returns:
            A new LocalTransform with the same translation, rotation, and scale,
            but in the new coordinate system.
        """
        t_other = type(other)
        R = self.get_change_of_basis_matrix(type(self), t_other)
        return t_other.from_unsafe(
            R @ self._translation,
            R @ self._rotation,
            self._scale
        )

    def apply_scale(self, scale: float) -> "LocalTransform":
        """
        Apply a scale to this LocalTransform.

        Args:
            scale: A float representing the scale factor.

        Returns:
            A new LocalTransform with the same translation and rotation,
            but scaled by the given factor.
        """
        return self.__class__.from_unsafe(
            self._translation.copy(),
            self._rotation.copy(),
            self._scale * float(scale)
        )

    def translate(self, translation: np.ndarray) -> "LocalTransform":
        """
        Translate this LocalTransform by a given translation vector.

        Args:
            translation: A 3D numpy array representing the translation vector.

        Returns:
            A new LocalTransform with the same rotation and scale,
            but translated by the given vector.
        """
        return self.__class__.from_unsafe(
            self._translation + np_asarray(translation),
            self._rotation.copy(),
            self._scale
        )

    def rotate(self, rotation: np.ndarray) -> "LocalTransform":
        """
        Rotate this LocalTransform by a given rotation matrix.

        Args:
            rotation: A 3x3 numpy array representing the rotation matrix.

        Returns:
            A new LocalTransform with the same translation and scale,
            but rotated by the given rotation.
        """
        return self.__class__.from_unsafe(
            self._translation,
            self._rotation @ rotation,
            self._scale
        )

    def rotate_by_quaternion(self, quaternion: np.ndarray) -> "LocalTransform":
        """
        Rotate this LocalTransform by a given quaternion.

        Args:
            quaternion: A 4D numpy array representing the quaternion.

        Returns:
            A new LocalTransform with the same translation and scale,
            but rotated by the given quaternion.
        """
        return self.rotate(quaternion_to_rotation(quaternion))

    def rotate_by_euler(self, x: float, y: float, z: float, degrees: bool = True) -> "LocalTransform":
        """
        Rotate this LocalTransform by a given Euler angles.

        Args:
            x: The roll angle in degrees or radians.
            y: The pitch angle in degrees or radians.
            z: The yaw angle in degrees or radians.
            degrees: If True, the angles are in degrees. If False, the angles are in radians.

        Returns:
            A new LocalTransform with the same translation and scale,
            but rotated by the given Euler angles.
        """
        return self.rotate(euler_to_rotation(x, y, z, degrees=degrees))

    def rotate_x(self, angle: float, degrees: bool = True) -> "LocalTransform":
        """
        Rotate this LocalTransform around the x-axis by a given angle.

        Args:
            angle: The angle to rotate in degrees or radians.
            degrees: If True, the angle is in degrees. If False, the angle is in radians.

        Returns:
            A new LocalTransform with the same translation and scale,
            but rotated around the x-axis by the given angle.
        """
        return self.rotate(euler_to_rotation(angle, 0, 0, degrees=degrees))

    def rotate_y(self, angle: float, degrees: bool = True) -> "LocalTransform":
        """
        Rotate this LocalTransform around the y-axis by a given angle.

        Args:
            angle: The angle to rotate in degrees or radians.
            degrees: If True, the angle is in degrees. If False, the angle is in radians.

        Returns:
            A new LocalTransform with the same translation and scale,
            but rotated around the y-axis by the given angle.
        """
        return self.rotate(euler_to_rotation(0, angle, 0, degrees=degrees))

    def rotate_z(self, angle: float, degrees: bool = True) -> "LocalTransform":
        """
        Rotate this LocalTransform around the z-axis by a given angle.

        Args:
            angle: The angle to rotate in degrees or radians.
            degrees: If True, the angle is in degrees. If False, the angle is in radians.

        Returns:
            A new LocalTransform with the same translation and scale,
            but rotated around the z-axis by the given angle.
        """
        return self.rotate(euler_to_rotation(0, 0, angle, degrees=degrees))

    def rotation_as_quaternion(self, w_last: bool = True) -> np.ndarray:
        """
        Get the rotation of this LocalTransform as a quaternion.

        Args:
            w_last: If True, the quaternion is in the format (x, y, z, w).

        Returns:
            The rotation as a 4D numpy array representing the quaternion.
        """
        return rotation_to_quaternion(self._rotation, w_last=w_last)

    def rotation_as_euler(self, degrees: bool = True) -> Tuple[float, float, float]:
        """
        Get the rotation of this LocalTransform as Euler angles.
        The angles are in the order of (roll, pitch, yaw) or (x, y, z).

        Args:
            degrees: If True, the angles are in degrees. If False, the angles are in radians.

        Returns:
            The rotation as a 3D numpy array representing the Euler angles.
        """
        return rotation_to_euler(self._rotation, degrees=degrees)

    def inverse(self) -> "LocalTransform":
        """
        Invert this LocalTransform.

        Returns:
            A new LocalTransform that is the inverse of this transform.
        """
        return self.__class__.from_unsafe(
            -self._rotation.T @ self._translation,
            self._rotation.T,
            1.0 / self._scale
        )

    def to_inverse_matrix(self) -> AffineMatrix:
        """
        Get the inverse transformation matrix of this LocalTransform.

        Returns:
            The inverse transformation matrix as a 4x4 numpy array.
        """
        return self._affine_matrix_type.from_unsafe(to_inv_matrix(self._translation, self._rotation, self._scale))

    def to_matrix(self) -> AffineMatrix:
        """
        Get the transformation matrix of this LocalTransform.

        Returns:
            The transformation matrix as a 4x4 numpy array.
        """
        if self._dirty:
            self._matrix = self._affine_matrix_type.from_unsafe(to_matrix(
                self._translation, self._rotation, self._scale))
            self._dirty = False
        return self._matrix  # type: ignore[return-value]

    def with_translation(self, translation: np.ndarray) -> "LocalTransform":
        """
        Create a new LocalTransform with the same rotation and scale, but with a new translation.

        Args:
            translation: A 3D numpy array representing the new translation.

        Returns:
            A new LocalTransform with the same rotation and scale, but with the new translation.
        """
        return self.__class__.from_unsafe(
            np_asarray(translation), self._rotation.copy(), self._scale
        )

    def with_rotation(self, rotation: np.ndarray) -> "LocalTransform":
        """
        Create a new LocalTransform with the same translation and scale, but with a new rotation.

        Args:
            rotation: A 3x3 numpy array representing the new rotation matrix.

        Returns:
            A new LocalTransform with the same translation and scale, but with the new rotation.
        """
        return self.__class__.from_unsafe(
            self._translation.copy(), np_asarray(rotation), self._scale
        )

    def with_scale(self, scale: float) -> "LocalTransform":
        """
        Create a new LocalTransform with the same translation and rotation, but with a new scale.

        Args:
            scale: A float representing the new scale.

        Returns:
            A new LocalTransform with the same translation and rotation, but with the new scale.
        """
        return self.__class__.from_unsafe(
            self._translation.copy(), self._rotation.copy(), float(scale)
        )

    def transform_direction(self, direction: np.ndarray) -> np.ndarray:
        """
        Transform a direction vector using this LocalTransform.

        Args:
            direction: A 3D numpy array representing the direction vector.

        Returns:
            The transformed direction vector as a 3D numpy array.
        """
        return self._rotation @ (self._scale * direction)

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """
        Transform a point using this LocalTransform.

        Args:
            point: A 3D numpy array representing the point.

        Returns:
            The transformed point as a 3D numpy array.
        """
        return self._translation + self._rotation @ (self._scale * point)

    def transform_rotation(self, rotation: np.ndarray) -> np.ndarray:
        """
        Transform a rotation matrix using this LocalTransform.

        Args:
            rotation: A 3x3 numpy array representing the rotation matrix.

        Returns:
            The transformed rotation matrix as a 3x3 numpy array.
        """
        return self._rotation @ rotation

    def transform_scale(self, scale: float) -> float:
        """
        Transform a scale factor using this LocalTransform.

        Args:
            scale: A float representing the scale factor.

        Returns:
            The transformed scale factor as a float.
        """
        return self._scale * scale

    def transform_transform(self, transform: "LocalTransform") -> "LocalTransform":
        """
        Transform another LocalTransform using this LocalTransform.

        Args:
            transform: The LocalTransform to transform.

        Returns:
            A new LocalTransform that is the result of the transformation.
        """
        return self.__class__.from_unsafe(
            self._translation +
            self._rotation @ (self._scale * transform._translation),
            self._rotation @ transform._rotation,
            self._scale * transform._scale
        )

    def copy(self) -> "LocalTransform":
        """
        Create a copy of this LocalTransform.

        Returns:
            A new LocalTransform with independent copies of translation, rotation, and scale.
        """
        return self.__class__.from_unsafe(
            self._translation.copy(), self._rotation.copy(), self._scale
        )

    def __matmul__(self, other: "LocalTransform") -> "LocalTransform":
        """
        Right-multiply this LocalTransform by another LocalTransform.

        Args:
            other: The LocalTransform to multiply with.

        Returns:
            A new LocalTransform that is the result of the multiplication,
            with independent copies of translation, rotation, and scale.
        """
        # parent=self, child=other
        translation = self._translation + \
            self._rotation @ (self._scale * other._translation)
        rotation = self._rotation @ other._rotation
        scale = self._scale * other._scale
        return self.__class__.from_unsafe(translation, rotation, scale)

    def __rmatmul__(self, other: "LocalTransform") -> "LocalTransform":
        """
        Left-multiply this LocalTransform by another LocalTransform.

        Args:
            other: The LocalTransform to multiply with.

        Returns:
            A new LocalTransform that is the result of the multiplication,
            with independent copies of translation, rotation, and scale.
        """
        # parent=other, child=self
        translation = other._translation + \
            other._rotation @ (other._scale * self._translation)
        rotation = other._rotation @ self._rotation
        scale = other._scale * self._scale
        return self.__class__.from_unsafe(translation, rotation, scale)

    def __mult__(self, other: "LocalTransform") -> "LocalTransform":
        """
        Right-multiply this LocalTransform by another LocalTransform.

        Args:
            other: The LocalTransform to multiply with.

        Returns:
            A new LocalTransform that is the result of the multiplication.
        """
        return self.__matmul__(other)

    def __repr__(self):
        return f"{self.__class__.__name__}(translation={self._translation}, rotation={self._rotation}, scale={self._scale})"

    def __str__(self):
        return self.__repr__()

    def __copy__(self) -> "LocalTransform":
        """
        Shallow copy of this LocalTransform (matrix is copied).
        """
        return self.__class__.from_unsafe(self._translation.copy(), self._rotation.copy(), self._scale)

    def __deepcopy__(self, memo) -> "LocalTransform":
        """
        Deep copy support for the copy module.
        """
        # matrices are numeric, so shallow vs deep is effectively the same here
        return self.__copy__()

    def __reduce__(self):
        """
        Pickle support: reprunes to (class, (translation, rotation, scale))
        """
        return (self.__class__, (self._translation.copy(), self._rotation.copy(), self._scale))

    def __eq__(self, other: "LocalTransform") -> bool:
        """
        True if `other` is the same class and matrices are equal within a small tolerance.
        An identity check is performed to quickly handle comparisons to self.
        """
        if self is other:
            return True
        if self.__class__ is not other.__class__:
            return False
        return np.array_equal(self._translation, other._translation) and \
            np.array_equal(self._rotation, other._rotation) and \
            abs(self._scale - other._scale) < 1e-9

    def __ne__(self, other: "LocalTransform") -> bool:
        """
        Check if two transforms are not equal.
        Args:
            other: The Transform to compare with.
        Returns:
            True if the transforms are not equal, False otherwise.
        """
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """
        Hash function for the LocalTransform class.
        Returns:
            A hash value for the transform.
        """
        return hash((self.__class__, tuple(self._translation), tuple(self._rotation.flatten()), self._scale))
