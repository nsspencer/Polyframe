from dataclasses import dataclass, field
from typing import Union, Optional, List, Tuple
import numpy as np
from numpy.linalg import norm as np_norm
from numpy import cross as np_cross
from numpy import eye as np_eye
from numpy import dot as np_dot
from numpy import array as np_array
from numpy import asarray as np_asarray
from numpy import diag as np_diag
from numpy import array_equal as np_array_equal
from numpy import float64 as np_float64
from polyframe.frame_registry import FrameRegistry, CoordinateFrameType

from numba import njit
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


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

    def convert_to(self, new_coordinate_system: CoordinateFrameType, *, inplace: bool = False) -> "Transform":
        """
        Re-express this Transform in another coordinate system.

        Args:
            new_coordinate_system: target frame.
            inplace: if True, modify this Transform in place.

        Returns:
            Transform in the target coordinate system.
        """
        # 1) get 3×3 rotation to new frame
        R = FrameRegistry.change_of_basis(
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
        R_new = Transform._rotation_to(
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

    @staticmethod
    @njit
    def _rotation_to(
        target_vector: np.ndarray,
        current_R: np.ndarray,
        forward: np.ndarray
    ) -> np.ndarray:
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
        return Transform._az_el_range_to(target_vector, self.up, self.right, self.forward, degrees=degrees, signed_azimuth=signed_azimuth, counterclockwise_azimuth=counterclockwise_azimuth, flip_elevation=flip_elevation)

    @staticmethod
    @njit
    def _az_el_range_to(target_vector: np.ndarray, up: np.ndarray, lateral: np.ndarray, forward: np.ndarray, degrees: bool = True, signed_azimuth: bool = False, counterclockwise_azimuth: bool = False, flip_elevation: bool = False) -> tuple[float, float, float]:
        """
        Calculate azimuth, elevation, and range from origin to target
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
            (azimuth, elevation, range)
        """
        rng = np_norm(target_vector)
        if rng < 1e-12:
            return (0.0, 0.0, 0.0)

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

        return az_rad, el_rad, rng

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

        return Transform._phi_theta_to(
            tv,
            self.up, self.right, self.forward,
            degrees,
            signed_phi,
            counterclockwise_phi,
            polar,
            flip_theta
        )

    @staticmethod
    @njit
    def _phi_theta_to(
        target_vector: np.ndarray,
        up: np.ndarray,
        lateral: np.ndarray,
        forward: np.ndarray,
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

        return Transform._latitude_longitude_to(
            tv,
            self.up, self.right, self.forward,
            degrees,
            signed_longitude,
            counterclockwise_longitude,
            flip_latitude
        )

    @staticmethod
    @njit
    def _latitude_longitude_to(
        target_vector: np.ndarray,
        up: np.ndarray,
        lateral: np.ndarray,
        forward: np.ndarray,
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
            R3 = FrameRegistry.change_of_basis(
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
