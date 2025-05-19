import numpy as np
from numpy import asarray as np_asarray
from polyframe.utils import to_matrix, to_inv_matrix, decompose_matrix
from polyframe.geometry import quaternion_to_rotation, euler_to_rotation, rotation_to_quaternion, rotation_to_euler
from polyframe.direction import Direction, _DIR_TO_VEC
from typing import Union, Iterable, Type, Tuple


_BASE_TRANSLATION = np.zeros(3)
_BASE_ROTATION = np.eye(3)


class LocalTransform:
    """
    A class representing a local transformation in 3D space.
    """

    __slots__ = ('_translation', '_rotation', '_scale', '_dirty', '_matrix')

    def __init__(self, translation: Union[None, Iterable] = None, rotation: Union[None, Iterable] = None, scale: Union[None, float] = None):
        self._dirty: bool = True  # dirty flag on init
        self._matrix: Union[None, np.ndarray] = None  # init to None

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
        return cls.from_unchecked_values(_BASE_TRANSLATION.copy(), _BASE_ROTATION.copy(), 1.0)

    @classmethod
    def from_unchecked_values(cls, translation: np.ndarray, rotation: np.ndarray, scale: float):
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
        return cls.from_unchecked_values(translation, _BASE_ROTATION.copy(), 1.0)

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
        return cls.from_unchecked_values(translation, rotation, 1.0)

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
        return cls.from_unchecked_values(translation, rotation, scale)

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
        return cls.from_unchecked_values(_BASE_TRANSLATION.copy(), rotation, 1.0)

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
        return cls.from_unchecked_values(_BASE_TRANSLATION.copy(), _BASE_ROTATION.copy(), scale)

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
        return cls.from_unchecked_values(*decompose_matrix(matrix))

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
        return t_other.from_unchecked_values(
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
        return self.__class__.from_unchecked_values(
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
        return self.__class__.from_unchecked_values(
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
        return self.__class__.from_unchecked_values(
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
        return self.__class__.from_unchecked_values(
            -self._rotation.T @ self._translation,
            self._rotation.T,
            1.0 / self._scale
        )

    def to_inverse_matrix(self) -> np.ndarray:
        """
        Get the inverse transformation matrix of this LocalTransform.

        Returns:
            The inverse transformation matrix as a 4x4 numpy array.
        """
        return to_inv_matrix(self._translation, self._rotation, self._scale)

    def to_matrix(self) -> np.ndarray:
        """
        Get the transformation matrix of this LocalTransform.

        Returns:
            The transformation matrix as a 4x4 numpy array.
        """
        if self._dirty:
            self._matrix = to_matrix(
                self._translation, self._rotation, self._scale)
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
        return self.__class__.from_unchecked_values(
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
        return self.__class__.from_unchecked_values(
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
        return self.__class__.from_unchecked_values(
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
        return self.__class__.from_unchecked_values(
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
        return self.__class__.from_unchecked_values(
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
        return self.__class__.from_unchecked_values(translation, rotation, scale)

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
        return self.__class__.from_unchecked_values(translation, rotation, scale)

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
        return self.__class__.from_unchecked_values(self._translation.copy(), self._rotation.copy(), self._scale)

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


def _create_frame_convention(
    x: Direction, y: Direction, z: Direction
) -> Type[LocalTransform]:
    # sanity check
    if len({x, y, z}) != 3:
        raise ValueError("x, y, z must be three distinct Directions")

    # compute handedness attributes
    x_vec = _DIR_TO_VEC[x]
    y_vec = _DIR_TO_VEC[y]
    z_vec = _DIR_TO_VEC[z]
    is_right_handed = bool(np.allclose(np.cross(x_vec, y_vec), z_vec))
    is_left_handed = not is_right_handed

    # ensure orthobonality
    if not np.allclose(np.dot(x_vec, y_vec), 0) or not np.allclose(np.dot(x_vec, z_vec), 0) or not np.allclose(np.dot(y_vec, z_vec), 0):
        raise ValueError("x, y, z must be orthogonal Directions")

    # create basis vectors for the x axes
    if x == Direction.FORWARD:
        forward_basis = x_vec
        backward_basis = -x_vec
    elif x == Direction.BACKWARD:
        backward_basis = x_vec
        forward_basis = -x_vec
    elif x == Direction.LEFT:
        left_basis = x_vec
        right_basis = -x_vec
    elif x == Direction.RIGHT:
        right_basis = x_vec
        left_basis = -x_vec
    elif x == Direction.UP:
        up_basis = x_vec
        down_basis = -x_vec
    elif x == Direction.DOWN:
        down_basis = x_vec
        up_basis = -x_vec
    else:
        raise ValueError("Invalid direction for x")

    # create basis vectors for the y axes
    if y == Direction.FORWARD:
        forward_basis = y_vec
        backward_basis = -y_vec
    elif y == Direction.BACKWARD:
        backward_basis = y_vec
        forward_basis = -y_vec
    elif y == Direction.LEFT:
        left_basis = y_vec
        right_basis = -y_vec
    elif y == Direction.RIGHT:
        right_basis = y_vec
        left_basis = -y_vec
    elif y == Direction.UP:
        up_basis = y_vec
        down_basis = -y_vec
    elif y == Direction.DOWN:
        down_basis = y_vec
        up_basis = -y_vec
    else:
        raise ValueError("Invalid direction for y")

    # create basis vectors for the z axes
    if z == Direction.FORWARD:
        forward_basis = z_vec
        backward_basis = -z_vec
    elif z == Direction.BACKWARD:
        backward_basis = z_vec
        forward_basis = -z_vec
    elif z == Direction.LEFT:
        left_basis = z_vec
        right_basis = -z_vec
    elif z == Direction.RIGHT:
        right_basis = z_vec
        left_basis = -z_vec
    elif z == Direction.UP:
        up_basis = z_vec
        down_basis = -z_vec
    elif z == Direction.DOWN:
        down_basis = z_vec
        up_basis = -z_vec
    else:
        raise ValueError("Invalid direction for z")

    # create the world direction functions
    def _make_world_dir(reverse: bool, column_number: int):
        if reverse:
            def direction(self) -> np.ndarray:
                return -self._rotation[:, column_number]
            return direction
        else:
            def direction(self) -> np.ndarray:
                return self._rotation[:, column_number]
            return direction

    # create the basis matrix
    basis_matrix = np.array([x_vec, y_vec, z_vec], dtype=np.float64)
    basis_matrix_inv = basis_matrix.T

    # define the direction functions from the basis vectors
    forward_idx = int(np.argmax(forward_basis != 0))
    backward_idx = int(np.argmax(backward_basis != 0))
    left_idx = int(np.argmax(left_basis != 0))
    right_idx = int(np.argmax(right_basis != 0))
    up_idx = int(np.argmax(up_basis != 0))
    down_idx = int(np.argmax(down_basis != 0))
    forward_fn = _make_world_dir(
        forward_basis[forward_idx] < 0, forward_idx)  # type: ignore
    backward_fn = _make_world_dir(
        backward_basis[backward_idx] < 0, backward_idx)  # type: ignore
    left_fn = _make_world_dir(
        left_basis[left_idx] < 0, left_idx)  # type: ignore
    right_fn = _make_world_dir(
        right_basis[right_idx] < 0, right_idx)  # type: ignore
    up_fn = _make_world_dir(up_basis[up_idx] < 0, up_idx)  # type: ignore
    down_fn = _make_world_dir(
        down_basis[down_idx] < 0, down_idx)  # type: ignore

    props = {
        # direction properties
        "forward":  property(forward_fn),
        "backward": property(backward_fn),
        "left":     property(left_fn),
        "right":    property(right_fn),
        "up":       property(up_fn),
        "down":     property(down_fn),
        # basis-info
        "is_right_handed":  staticmethod(lambda: is_right_handed),
        "is_left_handed":   staticmethod(lambda: is_left_handed),
        "get_x_label":          staticmethod(lambda: x),
        "get_y_label":          staticmethod(lambda: y),
        "get_z_label":          staticmethod(lambda: z),
        "get_x_basis":          staticmethod(lambda: x_vec),
        "get_y_basis":          staticmethod(lambda: y_vec),
        "get_z_basis":          staticmethod(lambda: z_vec),
        "get_basis_matrix":     staticmethod(lambda: basis_matrix),
        "get_basis_matrix_inv": staticmethod(lambda: basis_matrix_inv),
        "get_basis_forward":    staticmethod(lambda: forward_basis),
        "get_basis_backward":   staticmethod(lambda: backward_basis),
        "get_basis_left":       staticmethod(lambda: left_basis),
        "get_basis_right":      staticmethod(lambda: right_basis),
        "get_basis_up":         staticmethod(lambda: up_basis),
        "get_basis_down":       staticmethod(lambda: down_basis),
    }

    cls_name = f"LocalTransform<{x.name},{y.name},{z.name}>"
    return type(cls_name, (LocalTransform,), props)


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


def define_transform_convention(x: Direction = Direction.FORWARD, y: Direction = Direction.LEFT, z: Direction = Direction.UP) -> Type[LocalTransform]:
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
    Type[LocalTransform]
        The transform type for the given frame convention.
    """
    val = _FRAME_REGISTRY.get((x, y, z), None)
    if val is None:
        raise ValueError(
            f"Frame convention {x}, {y}, {z} not valid. Must be orthogonal.")
    return val
