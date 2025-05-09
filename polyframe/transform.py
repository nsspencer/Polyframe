# transform.py

from dataclasses import dataclass, field
from typing import Union, Optional, List, Tuple, Type
import numpy as np
from numpy.linalg import norm as np_norm
from numpy.linalg import det as np_det
from numpy.linalg import qr as np_qr
from numpy import append as np_append
from numpy import eye as np_eye
from numpy import allclose as np_allclose
from numpy import asarray as np_asarray
from numpy import diag as np_diag
from numpy import array_equal as np_array_equal
from numpy import float64 as np_float64
from polyframe.direction import Direction
from polyframe.utils import quaternion_to_rotation, euler_to_rotation, rotation_to_euler, rotation_to_quaternion, rotation_to, azimuth_elevation_to, phi_theta_to, latitude_longitude_to

# preallocate the identity matrix for performance
EYE4 = np_eye(4, dtype=np_float64)


@dataclass(slots=True)
class Transform:
    """
    A 4x4 homogeneous transformation in 3D space.

    Attributes:
        matrix (np.ndarray): 4x4 transformation matrix.
    """

    matrix: np.ndarray = field(default_factory=lambda: EYE4.copy())

    @classmethod
    def identity(cls) -> "Transform":
        """
        Create an identity Transform.

        Returns:
            A new Transform whose `matrix` is the identity matrix.
        """
        return cls(EYE4.copy())

    @classmethod
    def from_values(
        cls,
        translation: Optional[Union[np.ndarray, List, Tuple]] = None,
        rotation: Optional[Union[np.ndarray, List, Tuple]] = None,
        scale: Optional[Union[np.ndarray, List, Tuple]] = None,
        perspective: Optional[Union[np.ndarray, List, Tuple]] = None,
    ) -> "Transform":
        """
        Create a Transform by assembling translation, rotation, and scale into a 4x4 matrix.

        Args:
            translation: length-3 array to place in last column.
            rotation: 3x3 rotation matrix to place in upper-left.
            scale: length-3 scale factors applied along the diagonal.
            perspective: length-4 perspective array.

        Returns:
            A new Transform whose `matrix` encodes T·R·S.
        """
        mat = EYE4.copy()
        if translation is not None:
            mat[:3, 3] = translation
        if rotation is not None:
            mat[:3, :3] = rotation
        if scale is not None:
            shape = np.shape(scale)
            if shape == (1,):
                s = float(scale[0])
                S = np_diag([s, s, s])
            elif shape == (3,):
                S = np_diag(scale)
            elif shape == (3, 3):
                S = scale
            else:
                raise ValueError(f"Invalid scale shape: {shape}")
            mat[:3, :3] = mat[:3, :3] @ S
        if perspective is not None:
            mat[3, :] = perspective
        return cls(mat)

    @classmethod
    def from_translation(
        cls,
        translation: Union[np.ndarray, List, Tuple],
    ) -> "Transform":
        """
        Create a Transform from a translation vector.

        Args:
            translation: length-3 array to place in last column.

        Returns:
            A new Transform whose `matrix` encodes T.
        """
        mat = EYE4.copy()
        mat[:3, 3] = translation
        return cls(mat)

    @classmethod
    def from_rotation(
        cls,
        rotation: np.ndarray,
    ) -> "Transform":
        """
        Create a Transform from a rotation matrix.

        Args:
            rotation: 3x3 rotation matrix to place in upper-left.

        Returns:
            A new Transform whose `matrix` encodes R.
        """
        mat = EYE4.copy()
        mat[:3, :3] = rotation
        return cls(mat)

    @classmethod
    def from_scale(
        cls,
        scale: Union[np.ndarray, List, Tuple],
    ) -> "Transform":
        """
        Create a Transform from a scale vector.

        Args:
            scale: length-3 array to place in diagonal.

        Returns:
            A new Transform whose `matrix` encodes S.
        """
        shape = np.shape(scale)
        if shape == (1,):
            s = float(scale[0])
            S = np_diag([s, s, s])
        elif shape == (3,):
            S = np_diag(scale)
        elif shape == (3, 3):
            S = scale
        else:
            raise ValueError(f"Invalid scale shape: {shape}")

        mat = EYE4.copy()
        mat[:3, :3] = S
        return cls(mat)

    @classmethod
    def from_quaternion(
        cls,
        quaternion: np.ndarray,
        w_last: bool = True,
    ) -> "Transform":
        """
        Create a Transform from a quaternion.

        Args:
            quaternion: 4-element array representing the quaternion.
            w_last: if True, the quaternion is in [x, y, z, w] format.

        Returns:
            A new Transform whose `matrix` encodes R.
        """
        mat = EYE4.copy()
        mat[:3, :3] = quaternion_to_rotation(quaternion, w_last=w_last)
        return cls(mat)

    @classmethod
    def from_euler_angles(
        cls,
        roll: float,
        pitch: float,
        yaw: float,
        degrees: bool = True,
    ) -> "Transform":
        """
        Create a Transform from Euler angles.

        Args:
            roll: rotation around x-axis.
            pitch: rotation around y-axis.
            yaw: rotation around z-axis.
            degrees: if True, angles are in degrees, else radians.

        Returns:
            A new Transform whose `matrix` encodes R.
        """
        mat = EYE4.copy()
        mat[:3, :3] = euler_to_rotation(roll, pitch, yaw, degrees=degrees)
        return cls(mat)

    @classmethod
    def from_flat_array(
        cls,
        flat_array: np.ndarray,
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
        flat_array = np.asarray(flat_array, dtype=np_float64)
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
        mat = np.array(list_array, dtype=np_float64).reshape((4, 4))
        return cls(mat)

    #########
    # Getters for fundamental properties
    #

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
    def scale(self) -> np.ndarray:
        """
        Compute per-axis scale from the rotation columns' norms.

        Returns:
            Length-3 array of Euclidean norms of each column of `rotation`.
        """
        return np_norm(self.matrix[:3, :3], axis=0)

    @property
    def perspective(self) -> np.ndarray:
        """
        Extract the perspective component of the matrix.

        Returns:
            The last row of the matrix.
        """
        return self.matrix[3, :]

    ########
    # Apply methods
    #

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
        return self.__class__(new)

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
            self.matrix[:3, :3] = rotation @ self.matrix[:3, :3]
            return self

        new = self.matrix.copy()
        new[:3, :3] = rotation @ self.matrix[:3, :3]
        return self.__class__(new)

    def apply_rotation_from_quaternion(self, quaternion: np.ndarray, w_last: bool = True, *, inplace: bool = False) -> "Transform":
        """
        Apply a quaternion to this Transform.

        Args:
            quaternion: 4-element array representing the quaternion.
            inplace: if True, modify this Transform in place.

        Returns:
            Transform with updated rotation.
        """
        R = quaternion_to_rotation(quaternion, w_last=w_last)
        if inplace:
            self.matrix[:3, :3] = R @ self.matrix[:3, :3]
            return self

        new = self.matrix.copy()
        new[:3, :3] = R @ self.matrix[:3, :3]
        return self.__class__(new)

    def apply_rotation_from_euler(self, roll: float, pitch: float, yaw: float, degrees: bool = True, *, inplace: bool = False) -> "Transform":
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
            self.matrix[:3, :3] = R @ self.matrix[:3, :3]
            return self

        new = self.matrix.copy()
        new[:3, :3] = R @ self.matrix[:3, :3]
        return self.__class__(new)

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
            self.matrix[:3, :3] = self.matrix[:3, :3] @ S      # post‑mult
            return self

        new = self.matrix.copy()
        new[:3, :3] = new[:3, :3] @ S      # post‑mult
        return self.__class__(new)

    def apply_perspective(self, perspective: np.ndarray, *, inplace: bool = False) -> "Transform":
        """
        Apply a perspective to this Transform.

        Args:
            perspective: length-4 array to add to current perspective.
            inplace: if True, modify this Transform in place.

        Returns:
            Transform with updated perspective.
        """
        if inplace:
            self.matrix[3, :] += perspective
            return self

        new = self.matrix.copy()
        new[3, :] += perspective
        return self.__class__(new)

    #########
    # Setter methods
    #

    @rotation.setter
    def rotation(self, value: np.ndarray) -> None:
        """
        Set the rotation submatrix.

        Args:
            value: 3x3 matrix to set as rotation.
        """
        self.matrix[:3, :3] = value

    @translation.setter
    def translation(self, value: np.ndarray) -> None:
        """
        Set the translation vector.

        Args:
            value: length-3 array to set as translation.
        """
        self.matrix[:3, 3] = value

    @scale.setter
    def scale(self, value: np.ndarray) -> None:
        """
        Set the scale factors.

        Args:
            value: length-3 array to set as scale.
        """
        shape = np.shape(value)
        if shape == (1,):
            s = float(value[0])
            S = np_diag([s, s, s])
        elif shape == (3,):
            S = np_diag(value)
        elif shape == (3, 3):
            S = value
        else:
            raise ValueError(f"Invalid scale shape: {shape}")

        self.matrix[:3, :3] = self.matrix[:3, :3] @ S

    @perspective.setter
    def perspective(self, value: np.ndarray) -> None:
        """
        Set the perspective component of the matrix.

        Args:
            value: length-4 array to set as perspective.
        """
        self.matrix[3, :] = value

    def set_translation(self, translation: np.ndarray, *, inplace: bool = False) -> "Transform":
        """
        Assign a translation to this Transform.

        Args:
            translation: length-3 vector to set as translation.

        Returns:
            self with updated translation.
        """
        if inplace:
            self.matrix[:3, 3] = translation
            return self

        M = self.matrix.copy()
        M[:3, 3] = translation
        return self.__class__(M)

    def set_rotation(self, rotation: np.ndarray, *, inplace: bool = False) -> "Transform":
        """
        Assign a rotation to this Transform.

        Args:
            rotation: 3x3 matrix to set as rotation.

        Returns:
            self with updated rotation.
        """
        if inplace:
            self.matrix[:3, :3] = rotation
            return self

        M = self.matrix.copy()
        M[:3, :3] = rotation
        return self.__class__(M)

    def set_rotation_from_quaternion(self, quaternion: np.ndarray, w_last: bool = True, *, inplace: bool = False) -> "Transform":
        """
        Assign a quaternion to this Transform.

        Args:
            quaternion: 4-element array representing the quaternion.

        Returns:
            self with updated rotation.
        """
        if inplace:
            self.matrix[:3, :3] = quaternion_to_rotation(
                quaternion, w_last=w_last)
            return self

        M = self.matrix.copy()
        M[:3, :3] = quaternion_to_rotation(quaternion, w_last=w_last)
        return self.__class__(M)

    def set_rotation_from_euler(self, roll: float, pitch: float, yaw: float, degrees: bool = True, *, inplace: bool = False) -> "Transform":
        """
        Assign Euler angles to this Transform.

        Args:
            euler_angles: 3-element array representing the Euler angles (roll, pitch, yaw).
            inplace: if True, modify this Transform in place.

        Returns:
            self with updated rotation.
        """
        if inplace:
            self.matrix[:3, :3] = euler_to_rotation(
                roll, pitch, yaw, degrees=degrees)
            return self

        M = self.matrix.copy()
        M[:3, :3] = euler_to_rotation(roll, pitch, yaw, degrees=degrees)
        return self.__class__(M)

    def set_scale(self, scale: np.ndarray, *, inplace: bool = False) -> "Transform":
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

        if inplace:
            self.matrix[:3, :3] = S
            return self

        M = self.matrix.copy()
        M[:3, :3] = S
        return self.__class__(M)

    def set_perspective(self, perspective: np.ndarray, *, inplace: bool = False) -> "Transform":
        """
        Assign a perspective to this Transform.

        Args:
            perspective: length-4 array to set as perspective.

        Returns:
            self with updated perspective.
        """
        if inplace:
            self.matrix[3, :] = perspective
            return self

        M = self.matrix.copy()
        M[3, :] = perspective
        return self.__class__(M)

    ########
    # Transform methods
    #

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """
        Apply this transform to a 3D point (affine).

        Args:
            point: length-3 array.

        Returns:
            Transformed length-3 point.
        """
        p = np_append(point, 1.0)
        return (self.matrix @ p)[:3]

    def transform_vector(self, vector: np.ndarray) -> np.ndarray:
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

    def distance_to(self, target: Union["Transform", np.ndarray]) -> float:
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

    def vector_to(self, target: Union["Transform", np.ndarray]) -> np.ndarray:
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

    def direction_to(self, target: Union["Transform", np.ndarray]) -> np.ndarray:
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
        target: Union["Transform", np.ndarray],
    ) -> np.ndarray:
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
            self.matrix[:3, :3],
            self.basis_forward()
        )

    def quaternion_to(self, target: Union["Transform", np.ndarray], w_last: bool = True) -> np.ndarray:
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
                self.matrix[:3, :3],
                self.basis_forward()
            ),
            w_last=w_last
        )

    def euler_angles_to(self, target: Union["Transform", np.ndarray], degrees: bool = True) -> tuple[float, float, float]:
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
                self.matrix[:3, :3],
                self.basis_forward()
            ),
            degrees=degrees
        )

    def azimuth_elevation_to(
        self,
        target: Union["Transform", np.ndarray],
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
            tgt = target.matrix[:3, 3]
        else:
            tgt = np_asarray(target, float)

        # 2) form the vector from this.origin → target
        target_vector = tgt - self.matrix[:3, 3]

        # 3) call into our compiled routine
        R_new = rotation_to(
            target_vector,
            self.matrix[:3, :3],
            self.basis_forward()
        )

        # 4) build the new 4×4
        if inplace:
            self.matrix[:3, :3] = R_new
            return self

        M = self.matrix.copy()
        M[:3, :3] = R_new
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
        Invert this Transform analytically:
          T = [R t; 0 1]  ⇒  T⁻¹ = [Rᵀ  -Rᵀ t; 0 1]

        Args:
            inplace: if True, modify this Transform in place.

        Returns:
            Inverted Transform.
        """
        R = self.matrix[:3, :3]  # rotation
        t = self.matrix[:3, 3]  # translation

        R_inv = R.T
        t_inv = -R_inv @ t

        M = self.matrix.copy()
        M[:3, :3] = R_inv
        M[:3,  3] = t_inv

        if inplace:
            self.matrix[:] = M
            return self

        return self.__class__(M)

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
    def forward(self) -> np.ndarray:
        """
        Rotate the coordinate system's forward vector into world frame.

        Returns:
            The 3D “forward” direction after applying this transform's rotation.
        """
        ...

    @property
    def backward(self) -> np.ndarray:
        """
        Rotate the coordinate system's backward vector into world frame.

        Returns:
            The 3D “backward” direction after applying this transform's rotation.
        """
        ...

    @property
    def left(self) -> np.ndarray:
        """
        Rotate the coordinate system's left vector into world frame.

        Returns:
            The 3D “left” direction after applying this transform's rotation.
        """
        ...

    @property
    def right(self) -> np.ndarray:
        """
        Rotate the coordinate system's right vector into world frame.

        Returns:
            The 3D “right” direction after applying this transform's rotation.
        """
        ...

    @property
    def up(self) -> np.ndarray:
        """
        Rotate the coordinate system's up vector into world frame.

        Returns:
            The 3D “up” direction after applying this transform's rotation.
        """
        ...

    @property
    def down(self) -> np.ndarray:
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
        B_old = self.basis_matrix()    # np.ndarray (3×3)
        B_new = other.basis_matrix()      # np.ndarray (3×3)

        # 2) compute P = B_old⁻¹ B_new; since B_old is orthonormal, inv = transpose
        P = B_old.T @ B_new                # (3×3)

        # 3) build a 4×4 homogeneous change‐of‐basis
        C = np.eye(4, dtype=self.matrix.dtype)
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

    def to_quaternion(self, w_last: bool = True) -> np.ndarray:
        """
        Extract the quaternion from the rotation matrix.

        Returns:
            A 4-element array representing the quaternion.
        """
        return rotation_to_quaternion(self.matrix[:3, :3], w_last=w_last)

    def to_euler_angles(self, degrees: bool = True) -> Tuple[float, float, float]:
        """
        Extract the Euler angles from the rotation matrix.

        Returns:
            A 3-element tuple representing the Euler angles.
        """
        return rotation_to_euler(self.matrix[:3, :3], degrees=degrees)

    def to_list(self) -> List[float]:
        """
        Convert the matrix to a list of floats.

        Returns:
            A list of 16 floats representing the matrix.
        """
        return self.matrix.flatten().tolist()

    def to_flat_array(self) -> np.ndarray:
        """
        Convert the matrix to a flat array.

        Returns:
            A 1D array of 16 floats representing the matrix.
        """
        return self.matrix.flatten()

    #########
    # Dunder methods
    #

    def __matmul__(self, other: Union["Transform", np.ndarray]) -> Union["Transform", np.ndarray]:
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
             other (Union["Transform", np.ndarray]): The right-hand operand. It can be either:
                  - A numpy ndarray, in which case the transform's matrix is applied to it.
                  - Another Transform object, in which case the transforms are composed (self ∘ other).
        Returns:
             Union["Transform", np.ndarray]:
                  - A numpy ndarray if 'other' is an ndarray.
                  - A new Transform instance representing the composition if 'other' is a Transform.
        Raises:
             NotImplemented: If 'other' is neither a numpy ndarray nor a Transform instance.
        """
        if isinstance(other, np.ndarray):
            return self.matrix @ other

        if not isinstance(other, Transform):
            return NotImplemented

        # if other has a different labeling/basis subclass, convert it
        if type(other) is not type(self):
            other = other.change_coordinate_system(type(self))

        # Compose: first apply `other`, then `self`
        M_combined = self.matrix @ other.matrix
        return self.__class__(M_combined)

    def __mul__(self, other: Union["Transform", np.ndarray]) -> Union["Transform", np.ndarray]:
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
        mat = np.array2string(self.matrix, precision=6, separator=', ')
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
