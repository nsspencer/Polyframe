import numpy as np
from numpy.linalg import norm as np_norm
from numpy import asarray as np_asarray
from polyframe.geometry import (
    rotation_to_quaternion,
    rotation_to_euler,
    rotation_to,
    latitude_longitude_to,
    phi_theta_to,
    azimuth_elevation_to
)
from polyframe.utils import to_matrix
from typing import Tuple, Type, Union, Iterable
from polyframe.direction import Direction, _DIR_TO_VEC

_BASE_TRANSLATION = np.zeros(3)
_BASE_ROTATION = np.eye(3)
_BASE_SCALE = np.ones(3)


class LocalTransform:
    __slots__ = ('_translation', '_rotation', '_scale', '_dirty', '_matrix')

    def __init__(self, translation: Union[None, Iterable] = None, rotation: Union[None, Iterable] = None, scale: Union[None, Iterable] = None):
        self._dirty = True
        self._matrix = None
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

        if scale is not None:
            self._scale = np_asarray(scale)
            if self._scale.shape != (3,):
                raise ValueError(
                    f"Scale must be a 3D vector, got {self._scale.shape}")
        else:
            self._scale = _BASE_SCALE.copy()

    @classmethod
    def from_unchecked_values(cls, translation: np.ndarray, rotation: np.ndarray, scale: np.ndarray):
        """Create a LocalTransform without checking the validity of the inputs. Useful for performance when you are sure of the input shapes and types."""
        instance = object.__new__(cls)
        instance._dirty = True
        instance._matrix = None
        instance._translation = translation
        instance._rotation = rotation
        instance._scale = scale
        return instance

    @property
    def translation(self) -> np.ndarray:
        """Get the translation vector."""
        return self._translation

    @translation.setter
    def translation(self, value: Union[Iterable, np.ndarray]):
        """Set the translation vector."""
        self._translation = np_asarray(value)
        self._dirty = True

    @property
    def rotation(self) -> np.ndarray:
        """Get the rotation matrix."""
        return self._rotation

    @rotation.setter
    def rotation(self, value: Union[Iterable, np.ndarray]):
        """Set the rotation matrix."""
        self._rotation = np_asarray(value)
        self._dirty = True

    @property
    def scale(self) -> np.ndarray:
        """Get the scale vector."""
        return self._scale

    @scale.setter
    def scale(self, value: Union[Iterable, np.ndarray]):
        """Set the scale vector."""
        self._scale = np_asarray(value)
        self._dirty = True

    def bake(self) -> np.ndarray:
        """Convert the transform to a 4x4 transformation matrix."""
        if self._dirty:
            self._matrix = to_matrix(
                self._translation, self._rotation, self._scale)
            self._dirty = False
        return self._matrix

    def rotation_as_quaternion(self, w_last: bool = True) -> np.ndarray:
        """Convert the transform to a quaternion."""
        return rotation_to_quaternion(self._rotation, w_last=w_last)

    def rotation_as_euler(self, degrees: bool = True) -> Tuple[float, float, float]:
        """Convert the transform to Euler angles (roll, pitch, yaw)."""
        return rotation_to_euler(self._rotation, degrees=degrees)

    def copy(self) -> "LocalTransform":
        """
        Return a copy of this LocalTransform.

        Returns:
            A new LocalTransform with the same matrix.
        """
        return self.__class__.from_unchecked_values(self._translation.copy(), self._rotation.copy(), self._scale.copy())

    def distance_to(self, target: Union["LocalTransform", np.ndarray]) -> np.floating:
        """
        Compute the distance to another LocalTransform or translation vector.

        Args:
            target: the target LocalTransform or translation vector.

        Returns:
            The distance to the target.
        """
        if isinstance(target, LocalTransform):
            tgt = target._translation
        else:
            tgt = np_asarray(target, float)

        return np_norm(tgt - self._translation)

    def vector_to(self, target: Union["LocalTransform", np.ndarray]) -> np.ndarray:
        """
        Compute the vector to another LocalTransform or translation vector.

        Args:
            target: the target LocalTransform or translation vector.

        Returns:
            The vector to the target.
        """
        if isinstance(target, LocalTransform):
            tgt = target._translation
        else:
            tgt = np_asarray(target, float)

        return tgt - self._translation

    def direction_to(self, target: Union["LocalTransform", np.ndarray]) -> np.ndarray:
        """
        Compute the direction to another LocalTransform or translation vector.

        Args:
            target: the target LocalTransform or translation vector.

        Returns:
            The direction to the target.
        """
        if isinstance(target, LocalTransform):
            tgt = target._translation
        else:
            tgt = np_asarray(target, float)
        target_vector = tgt - self._translation
        distance = np_norm(target_vector)
        if distance < 1e-8:
            # avoid division by zero by returning forward vector
            return self.forward

        return target_vector / distance

    def rotation_to(
        self,
        target: Union["LocalTransform", np.ndarray],
    ) -> np.ndarray:
        """
        Get the rotation from this Trasnform to the target.

        Args:
            target: the target LocalTransform or translation vector.

        Returns:
            Rotation matrix 3x3.
        """
        # 1) grab the world-space target translation
        if isinstance(target, LocalTransform):
            tgt = target._translation
        else:
            tgt = np_asarray(target, float)

        # 2) form the vector from this.origin → target
        target_vector = tgt - self._translation

        # 3) call into our compiled routine
        return rotation_to(
            target_vector,
            self._rotation,
            self.basis_forward()
        )

    def quaternion_to(self, target: Union["LocalTransform", np.ndarray], w_last: bool = True) -> np.ndarray:
        """
        Get the quaternion from this LocalTransform to the target.

        Args:
            target: the target LocalTransform or translation vector.

        Returns:
            Quaternion 4-element array.
        """
        # 1) grab the world-space target translation
        if isinstance(target, LocalTransform):
            tgt = target._translation
        else:
            tgt = np_asarray(target, float)

        # 2) form the vector from this.origin → target
        target_vector = tgt - self._translation

        # 3) call into our compiled routine
        return rotation_to_quaternion(
            rotation_to(
                target_vector,
                self._rotation,
                self.basis_forward()
            ),
            w_last=w_last
        )

    def euler_angles_to(self, target: Union["LocalTransform", np.ndarray], degrees: bool = True) -> tuple[float, float, float]:
        """
        Get the Euler angles from this LocalTransform to the target.

        Args:
            target: the target LocalTransform or translation vector.
            degrees: if True, return angles in degrees, else radians.

        Returns:
            (roll, pitch, yaw)
        """
        # 1) grab the world-space target translation
        if isinstance(target, LocalTransform):
            tgt = target._translation
        else:
            tgt = np_asarray(target, float)

        # 2) form the vector from this.origin → target
        target_vector = tgt - self._translation

        R = rotation_to(
            target_vector,
            self._rotation,
            self.basis_forward()
        )

        R_standard = self.basis_matrix_inv() @ R @ self.basis_matrix()

        # 3) call into our compiled routine
        return rotation_to_euler(
            R_standard,
            degrees=degrees
        )

    def azimuth_elevation_to(
        self,
        target: Union["LocalTransform", np.ndarray],
        *,
        degrees: bool = True,
        signed_azimuth: bool = False,
        counterclockwise_azimuth: bool = False,
        flip_elevation: bool = False
    ) -> tuple[float, float]:
        """
        Calculate azimuth, elevation, and range to the target.

        Args:
            origin: the observer LocalTransform.
            target: the target LocalTransform or translation vector.
            degrees: if True, return az/el in degrees, else radians.
            signed_azimuth: if True, az ∈ [-180,180] (or [-π,π]), else [0,360).
            counterclockwise_azimuth: if True, positive az is from forward → left,
                            otherwise forward → right.
            flip_elevation: if True, positive el means downward (down vector),
                            otherwise positive means upward (up vector).

        Returns:
            (azimuth, elevation)
        """
        if isinstance(target, LocalTransform):
            target_vector = target._translation - self._translation
        else:
            target_vector = np_asarray(target, float) - self._translation

        return azimuth_elevation_to(
            target_vector,
            self.up,
            self.right,
            self.forward,
            degrees=degrees,
            signed_azimuth=signed_azimuth,
            counterclockwise_azimuth=counterclockwise_azimuth,
            flip_elevation=flip_elevation
        )

    def phi_theta_to(
        self,
        target: Union["LocalTransform", np.ndarray],
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
            target: the target LocalTransform or translation vector.
            degrees: if True, return angles in degrees, else radians.
            signed_phi: if True, φ in [-π,π] (or [-180,180]), else [0,2π) (or [0,360)).
            counterclockwise_phi: if True, φ positive from forward → left, else forward → right.
            polar: if True, θ is the polar angle from up (0…π), else θ is elevation from horizontal (−π/2…π/2).
            flip_theta: if True, flip the sign of θ.

        Returns:
            (φ, θ)
        """
        if isinstance(target, LocalTransform):
            target_vector = target._translation - self._translation
        else:
            target_vector = np_asarray(target, float) - self._translation

        return phi_theta_to(
            target_vector,
            self.up, self.right, self.forward,
            degrees,
            signed_phi,
            counterclockwise_phi,
            polar,
            flip_theta
        )

    def latitude_longitude_to(
        self,
        target: Union["LocalTransform", np.ndarray],
        *,
        degrees: bool = True,
        signed_longitude: bool = True,
        counterclockwise_longitude: bool = True,
        flip_latitude: bool = False
    ) -> tuple[float, float]:
        """
        Calculate (latitude, longitude) to the target.

        Args:
            target: the target LocalTransform or translation vector.
            degrees: if True, return lat/lon in degrees, else radians.
            signed_longitude: if True, lon in [-π,π] (or [-180,180]), else [0,2π).
            counterclockwise_longitude: if True, lon positive from forward → left, else forward → right.
            flip_latitude: if True, flip the sign of latitude.

        Returns:
            (latitude, longitude)
        """
        if isinstance(target, LocalTransform):
            target_vector = target._translation - self._translation
        else:
            target_vector = np_asarray(target, float) - self._translation

        return latitude_longitude_to(
            target_vector,
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
        target: Union["LocalTransform", np.ndarray],
        *,
        inplace: bool = False
    ) -> "LocalTransform":
        """
        Rotate this LocalTransform so that its forward axis points at `target`.
        Rotate only the R part so forward→target, leaving P intact.

        Args:
            target: the target LocalTransform or translation vector.
            inplace: if True, modify this LocalTransform in place.

        Returns:
            LocalTransform with updated rotation.
        """
        # 1) compute pure‐rotation that points forward at target
        if isinstance(target, LocalTransform):
            tgt = target._translation
        else:
            tgt = np_asarray(target, float)
        target_vector = tgt - self._translation
        R_new = rotation_to(target_vector, self._rotation,
                            self.basis_forward())

        # 3) compose back and write
        block = R_new @ self._rotation
        if inplace:
            self._rotation[:] = block
            self._dirty = True
            return self

        return self.__class__.from_unchecked_values(self._translation.copy(), block.copy(), self._scale.copy())

    ###########
    # Directional properties derived from coordinate system convention
    #

    @property
    def forward(self) -> np.ndarray:
        """
        Get the local forward vector from the current rotation and basis vector.

        Returns:
            The 3D “forward” direction after applying this transform's rotation.
        """
        ...

    @property
    def backward(self) -> np.ndarray:
        """
        Get the local backward vector from the current rotation and basis vector.

        Returns:
            The 3D “backward” direction after applying this transform's rotation.
        """
        ...

    @property
    def left(self) -> np.ndarray:
        """
        Get the local left vector from the current rotation and basis vector.

        Returns:
            The 3D “left” direction after applying this transform's rotation.
        """
        ...

    @property
    def right(self) -> np.ndarray:
        """
        Get the local right vector from the current rotation and basis vector.

        Returns:
            The 3D “right” direction after applying this transform's rotation.
        """
        ...

    @property
    def up(self) -> np.ndarray:
        """
        Get the local up vector from the current rotation and basis vector.

        Returns:
            The 3D “up” direction after applying this transform's rotation.
        """
        ...

    @property
    def down(self) -> np.ndarray:
        """
        Get the local down vector from the current rotation and basis vector.

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
    def basis_x() -> np.ndarray:
        """
        Returns the basis x vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_y() -> np.ndarray:
        """
        Returns the basis y vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_z() -> np.ndarray:
        """
        Returns the basis z vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_matrix() -> np.ndarray:
        """
        Returns the basis matrix of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_matrix_inv() -> np.ndarray:
        """
        Returns the inverse of the basis matrix of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_forward() -> np.ndarray:
        """
        Returns the basis forward vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_backward() -> np.ndarray:
        """
        Returns the basis backward vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_left() -> np.ndarray:
        """
        Returns the basis left vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_right() -> np.ndarray:
        """
        Returns the basis right vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_up() -> np.ndarray:
        """
        Returns the basis up vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_down() -> np.ndarray:
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
        other: Type["LocalTransform"],
    ) -> "LocalTransform":
        """
        Re-express this transform in the coordinate system defined by `other` by right multiplying.

        Both `self` and `other` map *their* local axes→world.  This method
        computes the 3x3 change-of-basis P that carries `other`'s local
        coords into `self`'s local coords, then right-multiplies your
        3x3 rotation matrix by it to yield the rotation in the new basis, 
        when called on `other`-local points, gives the same world result.

        Mathematically:
            B_old = self.basis_matrix()      # rows = old-local axes in world
            B_new = other.basis_matrix()     # rows = new-local axes in world
            P     = B_old.T @ B_new          # new-local → old-local
            M_new = self._rotation @ C

        Args:
            other:     The LocalTransform *class* whose basis you want to switch into.

        Returns:
            A LocalTransform of type `other` whose numeric matrix does the same
            world-space mapping, but expects `other`-local inputs.
        """
        # 1) grab the 3×3 basis matrices (rows = basis vectors in world coords)
        B_old = self.basis_matrix()    # np.ndarray (3×3)
        B_new = other.basis_matrix()      # np.ndarray (3×3)

        # 2) compute P = B_old⁻¹ B_new; since B_old is orthonormal, inv = transpose
        P = B_old.T @ B_new                # (3×3)

        # return an instance of the *other* subclass
        return self.__class__.from_unchecked_values(
            self._translation.copy(),  # copy translation
            self._rotation @ P,       # rotate the rotation matrix
            self._scale.copy()        # copy scale
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
        translation = (self._translation + self._rotation @
                       other._translation)
        rotation = (self._rotation @ other._rotation)
        scale = (self._scale * other._scale)
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
        return self.__class__.from_unchecked_values(
            (other._translation + other._rotation @ self._translation),
            (other._rotation @ self._rotation),
            (other._scale * self._scale)
        )

    def __mult__(self, other: "LocalTransform") -> "LocalTransform":
        """
        Right-multiply this LocalTransform by another LocalTransform.

        Args:
            other: The LocalTransform to multiply with.

        Returns:
            A new LocalTransform that is the result of the multiplication.
        """
        return self.__matmul__(other)

    def __eq__(self, other: "LocalTransform") -> bool:
        """
        True if `other` is the same class and matrices are equal within a small tolerance.
        An identity check is performed to quickly handle comparisons to self.
        """
        if self is other:
            return True
        if self.__class__ is not other.__class__:
            return False
        return np.allclose(self._translation, other._translation) and \
            np.allclose(self._rotation, other._rotation) and \
            np.allclose(self._scale, other._scale)

    def __repr__(self):
        return f"{self.__class__.__name__}(translation={self._translation}, rotation={self._rotation}, scale={self._scale})"

    def __str__(self):
        return self.__repr__()

    def __copy__(self) -> "LocalTransform":
        """
        Shallow copy of this LocalTransform (matrix is copied).
        """
        return self.__class__.from_unchecked_values(self._translation.copy(), self._rotation.copy(), self._scale.copy())

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
        return (self.__class__, (self._translation.copy(), self._rotation.copy(), self._scale.copy()))


def _create_frame_convention(
    x: Direction, y: Direction, z: Direction
) -> Type[LocalTransform]:
    # sanity check
    if len({x, y, z}) != 3:
        raise ValueError("x, y, z must be three distinct Directions")

    # compute handedness
    x_vec = _DIR_TO_VEC[x]
    y_vec = _DIR_TO_VEC[y]
    z_vec = _DIR_TO_VEC[z]
    is_right_handed = bool(np.allclose(np.cross(x_vec, y_vec), z_vec))

    # ensure orthobonality
    if not np.allclose(np.dot(x_vec, y_vec), 0) or not np.allclose(np.dot(x_vec, z_vec), 0) or not np.allclose(np.dot(y_vec, z_vec), 0):
        raise ValueError("x, y, z must be orthogonal Directions")

    # placeholders so every basis_* name exists
    forward_basis = backward_basis = None
    left_basis = right_basis = None
    up_basis = down_basis = None

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

    def _make_world_dir(reverse: bool, column_number: int):
        if reverse:
            def direction(self) -> np.ndarray:
                return -self._rotation[:, column_number]
            return direction
        else:
            def direction(self) -> np.ndarray:
                return self._rotation[:, column_number]
            return direction

    basis_matrix = np.array([x_vec, y_vec, z_vec], dtype=np.float64)
    basis_matrix_inv = basis_matrix.T

    forward_idx = int(np.argmax(forward_basis != 0))
    backward_idx = int(np.argmax(backward_basis != 0))
    left_idx = int(np.argmax(left_basis != 0))
    right_idx = int(np.argmax(right_basis != 0))
    up_idx = int(np.argmax(up_basis != 0))
    down_idx = int(np.argmax(down_basis != 0))

    props = {
        # world-directions
        "forward":  property(_make_world_dir(forward_basis[forward_idx] < 0, forward_idx)),
        "backward": property(_make_world_dir(backward_basis[backward_idx] < 0, backward_idx)),
        "left":     property(_make_world_dir(left_basis[left_idx] < 0, left_idx)),
        "right":    property(_make_world_dir(right_basis[right_idx] < 0, right_idx)),
        "up":       property(_make_world_dir(up_basis[up_idx] < 0, up_idx)),
        "down":     property(_make_world_dir(down_basis[down_idx] < 0, down_idx)),

        # basis-info statics (unchanged)
        "is_right_handed":  staticmethod(lambda: is_right_handed),
        "is_left_handed":  staticmethod(lambda: not is_right_handed),
        "label_x":          staticmethod(lambda: x),
        "label_y":          staticmethod(lambda: y),
        "label_z":          staticmethod(lambda: z),
        "basis_x":          staticmethod(lambda: x_vec),
        "basis_y":          staticmethod(lambda: y_vec),
        "basis_z":          staticmethod(lambda: z_vec),
        "basis_matrix":     staticmethod(lambda: basis_matrix),
        "basis_matrix_inv": staticmethod(lambda: basis_matrix_inv),
        "basis_forward":    staticmethod(lambda: forward_basis),
        "basis_backward":   staticmethod(lambda: backward_basis),
        "basis_left":       staticmethod(lambda: left_basis),
        "basis_right":      staticmethod(lambda: right_basis),
        "basis_up":         staticmethod(lambda: up_basis),
        "basis_down":       staticmethod(lambda: down_basis),
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


def define_convention(x: Direction = Direction.FORWARD, y: Direction = Direction.LEFT, z: Direction = Direction.UP) -> Type[LocalTransform]:
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
