# transform.py

from dataclasses import dataclass, field
from typing import Union, Optional, List, Tuple
import numpy as np
from numpy.linalg import norm as np_norm
from numpy import eye as np_eye
from numpy import array as np_array
from numpy import asarray as np_asarray
from numpy import diag as np_diag
from numpy import array_equal as np_array_equal
from numpy import float64 as np_float64

from polyframe.frame_registry import FrameRegistry, CoordinateFrameType
from polyframe.utils import quaternion_to_rotation, euler_to_rotation, _rotation_to, _az_el_range_to, _phi_theta_to, _latitude_longitude_to


# preallocate the identity matrix for performance
EYE4 = np_eye(4, dtype=np_float64)


@dataclass(slots=True)
class Transform:
    """
    A 4x4 homogeneous transformation in 3D space, plus its coordinate system.

    Attributes:
        matrix (np.ndarray): 4x4 transformation matrix.
        coordinate_system (FrameRegistry): Defines forward/up/etc.
    """

    matrix: np.ndarray = field(default_factory=lambda: EYE4.copy())
    coordinate_system: CoordinateFrameType = field(
        default_factory=lambda: FrameRegistry.default)

    @classmethod
    def from_values(
        cls,
        translation: Optional[Union[np.ndarray, List, Tuple]] = None,
        rotation: Optional[Union[np.ndarray, List, Tuple]] = None,
        scale: Optional[Union[np.ndarray, List, Tuple]] = None,
        coordinate_system: CoordinateFrameType = FrameRegistry.default,
        *,
        dtype: np.dtype = np_float64
    ) -> "Transform":
        """
        Create a Transform by assembling translation, rotation, and scale into a 4x4 matrix.

        Args:
            translation: length-3 array to place in last column.
            rotation: 3x3 rotation matrix to place in upper-left.
            scale: length-3 scale factors applied along the diagonal.
            coordinate_system: which frame's forward/up/etc. to use.
            dtype: element type for the matrix (default float64).

        Returns:
            A new Transform whose `matrix` encodes T·R·S.
        """
        mat = np_eye(4, dtype=dtype)
        if translation is not None:
            mat[:3, 3] = translation
        if rotation is not None:
            mat[:3, :3] = rotation
        if scale is not None:
            mat[:3, :3] *= np_diag(scale)
        return cls(mat, coordinate_system)

    @property
    def rotation(self) -> np.ndarray:
        """
        Extract the 3x3 rotation submatrix.

        Returns:
            The upper-left 3x3 of `matrix`.
        """
        return self.matrix[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        """
        Extract the translation vector.

        Returns:
            A length-3 array from the first three entries of the fourth column.
        """
        return self.matrix[:3, 3]

    @property
    def scaler(self) -> np.ndarray:
        """
        Compute per-axis scale from the rotation columns' norms.

        Returns:
            Length-3 array of Euclidean norms of each column of `rotation`.
        """
        return np_norm(self.rotation, axis=0)

    @property
    def forward(self) -> np.ndarray:
        """
        Rotate the coordinate system's forward vector into world frame.

        Returns:
            The 3D “forward” direction after applying this transform's rotation.
        """
        return self.rotation @ self.coordinate_system.forward

    @property
    def backward(self) -> np.ndarray:
        """
        Rotate the coordinate system's backward vector into world frame.

        Returns:
            The 3D “backward” direction after applying this transform's rotation.
        """
        return self.rotation @ self.coordinate_system.backward

    @property
    def left(self) -> np.ndarray:
        """
        Rotate the coordinate system's left vector into world frame.

        Returns:
            The 3D “left” direction after applying this transform's rotation.
        """
        return self.rotation @ self.coordinate_system.left

    @property
    def right(self) -> np.ndarray:
        """
        Rotate the coordinate system's right vector into world frame.

        Returns:
            The 3D “right” direction after applying this transform's rotation.
        """
        return self.rotation @ self.coordinate_system.right

    @property
    def up(self) -> np.ndarray:
        """
        Rotate the coordinate system's up vector into world frame.

        Returns:
            The 3D “up” direction after applying this transform's rotation.
        """
        return self.rotation @ self.coordinate_system.up

    @property
    def down(self) -> np.ndarray:
        """
        Rotate the coordinate system's down vector into world frame.

        Returns:
            The 3D “down” direction after applying this transform's rotation.
        """
        return self.rotation @ self.coordinate_system.down

    @property
    def T(self) -> np.ndarray:
        """
        Transpose of the 4x4 matrix.

        Returns:
            The matrix transposed.
        """
        return self.matrix.T

    def apply_translation(self, translation: np.ndarray, *, inplace: bool = False) -> "Transform":
        """
        Apply a translation to this Transform.

        Args:
            translation: length-3 vector to add to current translation.
            inplace: if True, modify this Transform in place.

        Returns:
            Transform with updated translation.
        """
        if inplace:
            self.matrix[:3, 3] += translation
            return self

        new = self.matrix.copy()
        new[:3, 3] += translation
        t = object.__new__(Transform)
        t.matrix = new
        t.coordinate_system = self.coordinate_system
        return t

    def assign_translation(self, translation: np.ndarray) -> "Transform":
        """
        Assign a translation to this Transform.

        Args:
            translation: length-3 vector to set as translation.

        Returns:
            self with updated translation.
        """
        self.matrix[:3, 3] = translation
        return self

    def apply_rotation(self, rotation: np.ndarray, *, inplace: bool = False) -> "Transform":
        """
        Apply a rotation to this Transform.

        Args:
            rotation: 3x3 matrix to left-multiply current rotation.
            inplace: if True, modify this Transform in place.

        Returns:
            Transform with updated rotation.
        """
        if inplace:
            self.matrix[:3, :3] = rotation @ self.rotation
            return self

        new = self.matrix.copy()
        new[:3, :3] = rotation @ self.rotation
        t = object.__new__(Transform)
        t.matrix = new
        t.coordinate_system = self.coordinate_system
        return t

    def apply_quaternion(self, quaternion: np.ndarray, *, inplace: bool = False) -> "Transform":
        """
        Apply a quaternion to this Transform.

        Args:
            quaternion: 4-element array representing the quaternion.
            inplace: if True, modify this Transform in place.

        Returns:
            Transform with updated rotation.
        """
        R = quaternion_to_rotation(quaternion)
        if inplace:
            self.matrix[:3, :3] = R @ self.rotation
            return self

        new = self.matrix.copy()
        new[:3, :3] = R @ self.rotation
        t = object.__new__(Transform)
        t.matrix = new
        t.coordinate_system = self.coordinate_system
        return t

    def apply_euler_rotation(self, roll: float, pitch: float, yaw: float, degrees: bool = True, *, inplace: bool = False) -> "Transform":
        """
        Apply Euler angles to this Transform.

        Args:
            euler_angles: 3-element array representing the Euler angles (roll, pitch, yaw).
            inplace: if True, modify this Transform in place.

        Returns:
            Transform with updated rotation.
        """
        R = euler_to_rotation(roll, pitch, yaw, degrees=degrees)
        if inplace:
            self.matrix[:3, :3] = R @ self.rotation
            return self

        new = self.matrix.copy()
        new[:3, :3] = R @ self.rotation
        t = object.__new__(Transform)
        t.matrix = new
        t.coordinate_system = self.coordinate_system
        return t

    def assign_rotation(self, rotation: np.ndarray) -> "Transform":
        """
        Assign a rotation to this Transform.

        Args:
            rotation: 3x3 matrix to set as rotation.

        Returns:
            self with updated rotation.
        """
        self.matrix[:3, :3] = rotation
        return self

    def apply_scale(self, scale: np.ndarray, *, inplace: bool = False) -> "Transform":
        """
        Apply a scale to this Transform.

        Args:
            scale: length-3 factors to multiply each axis.
            inplace: if True, modify this Transform in place.

        Returns:
            Transform with updated scale.
        """
        shape = np.shape(scale)
        if shape == (1,):
            s = float(scale[0])
            S = np.diag([s, s, s])
        elif shape == (3,):
            S = np.diag(scale)
        elif shape == (3, 3):
            S = scale
        else:
            raise ValueError(f"Invalid scale shape: {shape}")

        if inplace:
            self.matrix[:3, :3] *= S
            return self

        new = self.matrix.copy()
        new[:3, :3] *= S
        t = object.__new__(Transform)
        t.matrix = new
        t.coordinate_system = self.coordinate_system
        return t

    def assign_scale(self, scale: np.ndarray) -> "Transform":
        """
        Assign a scale to this Transform.

        Args:
            scale: length-3 factors to set as scale.

        Returns:
            self with updated scale.
        """
        shape = np.shape(scale)
        if shape == (1,):
            s = float(scale[0])
            S = np.diag([s, s, s])
        elif shape == (3,):
            S = np.diag(scale)
        elif shape == (3, 3):
            S = scale
        else:
            raise ValueError(f"Invalid scale shape: {shape}")

        self.matrix[:3, :3] = S
        return self

    def inverse(self, *, inplace: bool = False) -> "Transform":
        """
        Invert this Transform analytically:
          T = [R t; 0 1]  ⇒  T⁻¹ = [Rᵀ  -Rᵀ t; 0 1]

        Args:
            inplace: if True, modify this Transform in place.

        Returns:
            Inverted Transform.
        """
        R = self.rotation
        t = self.translation

        R_inv = R.T
        t_inv = -R_inv @ t

        M = np_eye(4, dtype=self.matrix.dtype)
        M[:3, :3] = R_inv
        M[:3,  3] = t_inv

        if inplace:
            self.matrix[:] = M
            return self

        t = object.__new__(Transform)
        t.matrix = M
        t.coordinate_system = self.coordinate_system
        return t

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """
        Apply this transform to a 3D point (affine).

        Args:
            point: length-3 array.

        Returns:
            Transformed length-3 point.
        """
        p = np.append(point, 1.0)
        return (self.matrix @ p)[:3]

    def transform_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Apply this transform to a 3D direction (no translation).

        Args:
            vector: length-3 array.

        Returns:
            Transformed length-3 vector.
        """
        v = np.append(vector, 0.0)
        return (self.matrix @ v)[:3]

    def change_coordinate_system(self, new_coordinate_system: CoordinateFrameType, *, inplace: bool = False) -> "Transform":
        """
        Re-express this Transform in another coordinate system.

        Args:
            new_coordinate_system: target frame.
            inplace: if True, modify this Transform in place.

        Returns:
            Transform in the target coordinate system.
        """
        # 1) get 3×3 rotation to new frame
        R = FrameRegistry.get_system_rotation(
            self.coordinate_system, new_coordinate_system)

        if inplace:
            self.matrix[:3, :3] = R @ self.rotation
            self.matrix[:3, 3] = R @ self.translation
            self.coordinate_system = new_coordinate_system
            return self

        # apply to old rotation and translation
        old_R = self.rotation        # 3×3
        old_t = self.translation     # length-3

        new_R = R @ old_R            # 3×3
        new_t = R @ old_t            # length-3

        # build the new 4×4 homogeneous matrix
        M = np_eye(4, dtype=self.matrix.dtype)
        M[:3, :3] = new_R
        M[:3,  3] = new_t

        t = object.__new__(Transform)
        t.matrix = M
        t.coordinate_system = new_coordinate_system
        return t

    def distance_to(self, target: Union["Transform", np.ndarray]) -> float:
        """
        Compute the distance to another Transform or translation vector.

        Args:
            target: the target Transform or translation vector.

        Returns:
            The distance to the target.
        """
        if isinstance(target, Transform):
            tgt = target.translation
        else:
            tgt = np_asarray(target, float)

        return np_norm(tgt - self.translation)

    def vector_to(self, target: Union["Transform", np.ndarray]) -> np.ndarray:
        """
        Compute the vector to another Transform or translation vector.

        Args:
            target: the target Transform or translation vector.

        Returns:
            The vector to the target.
        """
        if isinstance(target, Transform):
            tgt = target.translation
        else:
            tgt = np_asarray(target, float)

        return tgt - self.translation

    def direction_to(self, target: Union["Transform", np.ndarray]) -> np.ndarray:
        """
        Compute the direction to another Transform or translation vector.

        Args:
            target: the target Transform or translation vector.

        Returns:
            The direction to the target.
        """
        if isinstance(target, Transform):
            tgt = target.translation
        else:
            tgt = np_asarray(target, float)
        target_vector = tgt - self.translation
        distance = np_norm(target_vector)
        if distance < 1e-8:
            # avoid division by zero by returning forward vector
            return self.forward

        return target_vector / distance

    def look_at(
        self,
        target: Union["Transform", np.ndarray],
        *,
        inplace: bool = False
    ) -> "Transform":
        """
        Rotate this Transform so that its forward axis points at `target`.

        Args:
            target: the target Transform or translation vector.
            inplace: if True, modify this Transform in place.

        Returns:
            Transform with updated rotation.
        """
        # 1) grab the world-space target translation
        if isinstance(target, Transform):
            tgt = target.translation
        else:
            tgt = np_asarray(target, float)

        # 2) form the vector from this.origin → target
        target_vector = tgt - self.translation

        # 3) call into our compiled routine
        R_new = _rotation_to(
            target_vector,
            self.rotation,
            np_array(self.coordinate_system.forward, dtype=float)
        )

        # 4) build the new 4×4
        if inplace:
            self.matrix[:3, :3] = R_new
            return self

        M = self.matrix.copy()
        M[:3, :3] = R_new
        t = object.__new__(Transform)
        t.matrix = M
        t.coordinate_system = self.coordinate_system
        return t

    def az_el_range_to(
        self,
        target: Union["Transform", np.ndarray],
        *,
        degrees: bool = True,
        signed_azimuth: bool = False,
        counterclockwise_azimuth: bool = False,
        flip_elevation: bool = False
    ) -> tuple[float, float, float]:
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
            (azimuth, elevation, range)
        """
        if isinstance(target, Transform):
            target_vector = target.translation - self.translation
        else:
            target_vector = np_asarray(target, float) - self.translation
        return _az_el_range_to(target_vector, self.up, self.right, self.forward, degrees=degrees, signed_azimuth=signed_azimuth, counterclockwise_azimuth=counterclockwise_azimuth, flip_elevation=flip_elevation)

    def phi_theta_to(
        self,
        target: Union["Transform", np.ndarray],
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
            tv = target.translation - self.translation
        else:
            tv = np_asarray(target, float) - self.translation

        return _phi_theta_to(
            tv,
            self.up, self.right, self.forward,
            degrees,
            signed_phi,
            counterclockwise_phi,
            polar,
            flip_theta
        )

    def lat_lon_to(
        self,
        target: Union["Transform", np.ndarray],
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
            tv = target.translation - self.translation
        else:
            tv = np_asarray(target, float) - self.translation

        return _latitude_longitude_to(
            tv,
            self.up, self.right, self.forward,
            degrees,
            signed_longitude,
            counterclockwise_longitude,
            flip_latitude
        )

    def __matmul__(self, other: Union["Transform", np.ndarray]) -> Union["Transform", np.ndarray]:
        """
        Compose this Transform with another (or apply to a raw matrix).

        Args:
            other: either another Transform or a 4xN array.

        Returns:
            The composed Transform.
        """
        if isinstance(other, np.ndarray):
            return self.matrix @ other

        if not isinstance(other, Transform):
            return NotImplemented

        if self.coordinate_system == other.coordinate_system:
            M = other.matrix
        else:
            # re‐frame `other` into self’s frame:
            R3 = FrameRegistry.get_system_rotation(
                other.coordinate_system, self.coordinate_system)
            # build 4×4 homogeneous re‐frame:
            T = np_eye(4, dtype=R3.dtype)
            T[:3, :3] = R3
            M = T @ other.matrix

        t = object.__new__(Transform)
        t.matrix = self.matrix @ M
        t.coordinate_system = self.coordinate_system
        return t

    def __eq__(self, other: "Transform") -> bool:
        return np_array_equal(self.matrix, other.matrix) and self.coordinate_system == other.coordinate_system

    def __repr__(self) -> str:
        return f"Transform(matrix={self.matrix}, coordinate_system={self.coordinate_system})"

    def __str__(self) -> str:
        return f"Transform(matrix={self.matrix}, coordinate_system={self.coordinate_system})"

    def __copy__(self) -> "Transform":
        return Transform(self.matrix.copy(), self.coordinate_system)

    def __reduce__(self):
        return (self.__class__, (self.matrix.copy(), self.coordinate_system))
