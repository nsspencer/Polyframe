# _polyframe.py

# Written by: Nathan Spencer
# Licensed under the Apache License, Version 2.0 (the "License")

from numpy.linalg import det as np_det
from numpy.linalg import norm as np_norm
from numpy.linalg import svd as np_svd
from numpy.linalg import inv as np_inv
from numpy import diag as np_diag
from numpy import shape as np_shape
from numpy import asarray as np_asarray
from numpy import allclose as np_allclose
from numpy import append as np_append
from numpy import array as np_array
from numpy import dot as np_dot
from numpy import eye as np_eye
from numpy import cross as np_cross
from numpy import array2string as np_array2string
from numpy import float64 as np_float64
from numpy import ndarray
import numpy as np

from typing import Union, Optional, List, Tuple, Type, Literal
from polyframe.utils import *
from polyframe.geometry import *
from polyframe.direction import Direction, _DIR_TO_VEC

# preallocate the identity matrix for performance
_EYE4 = np_eye(4, dtype=np_float64)


class WorldTransform:
    """
    A 4x4 homogeneous transformation in 3D space.

    Attributes:
        matrix (ndarray): 4x4 transformation matrix.
    """
    __slots__ = ("matrix",)

    def __init__(self, matrix: Optional[ndarray] = None):
        if matrix is None:
            self.matrix = _EYE4.copy()
        else:
            if matrix.shape != (4, 4):
                raise ValueError(f"Invalid matrix shape: {matrix.shape}")
            self.matrix = np_asarray(matrix, dtype=np_float64)

    @classmethod
    def identity(cls) -> "WorldTransform":
        """
        Create an identity WorldTransform.

        Returns:
            A new WorldTransform whose `matrix` is the identity matrix.
        """
        return cls(_EYE4.copy())

    @classmethod
    def from_values(
        cls,
        translation: Optional[Union[ndarray, List, Tuple]] = None,
        rotation: Optional[Union[ndarray, List, Tuple]] = None,
        scale: Optional[Union[ndarray, List, Tuple]] = None,
        shear: Optional[Union[ndarray, List, Tuple]] = None,
        perspective: Optional[Union[ndarray, List, Tuple]] = None,
    ) -> "WorldTransform":
        """
        Create a WorldTransform by assembling translation, rotation, scale, shear, and perspective into a 4x4 matrix.
        Order of application is: scale → shear → rotate → translate → perspective.
        Args:
            translation: length-3 array to place in last column.
            rotation: 3x3 rotation matrix to place in upper-left.
            scale: length-3 scale factors applied along the diagonal.
            shear: 3x3 shear matrix to place in upper-left.
            perspective: length-4 perspective array.

        Returns:
            A new WorldTransform whose `matrix` encodes the provided information.
        """
        instance = cls()
        if translation is not None:
            instance.set_translation(translation, inplace=True)
        if rotation is not None:
            instance.set_rotation(rotation, inplace=True)
        if scale is not None:
            instance.set_scale(scale, inplace=True)
        if shear is not None:
            instance.set_shear(shear, inplace=True)
        if perspective is not None:
            instance.set_perspective(perspective, inplace=True)
        return instance

    @classmethod
    def from_quaternion(
        cls,
        quaternion: ndarray,
        w_last: bool = True,
        translation: Optional[Union[ndarray, List, Tuple]] = None,
        scale: Optional[Union[ndarray, List, Tuple]] = None,
        shear: Optional[Union[ndarray, List, Tuple]] = None,
        perspective: Optional[Union[ndarray, List, Tuple]] = None,
    ) -> "WorldTransform":
        """
        Create a WorldTransform from a quaternion.

        Args:
            quaternion: 4-element array representing the quaternion.
            w_last: if True, the quaternion is in [x, y, z, w] format.
            translation: length-3 array to place in last column.
            scale: length-3 scale factors applied along the diagonal.
            shear: 3x3 shear matrix to place in upper-left.
            perspective: length-4 perspective array.

        Returns:
            A new WorldTransform whose `matrix` encodes R.
        """
        return cls.from_values(translation=translation, rotation=quaternion_to_rotation(quaternion, w_last=w_last), scale=scale, shear=shear, perspective=perspective)

    @classmethod
    def from_euler_angles(
        cls,
        roll: float,
        pitch: float,
        yaw: float,
        degrees: bool = True,
        translation: Optional[Union[ndarray, List, Tuple]] = None,
        scale: Optional[Union[ndarray, List, Tuple]] = None,
        shear: Optional[Union[ndarray, List, Tuple]] = None,
        perspective: Optional[Union[ndarray, List, Tuple]] = None,
    ) -> "WorldTransform":
        """
        Create a WorldTransform from Euler angles.

        Args:
            roll: rotation around x-axis.
            pitch: rotation around y-axis.
            yaw: rotation around z-axis.
            degrees: if True, angles are in degrees, else radians.
            translation: length-3 array to place in last column.
            scale: length-3 scale factors applied along the diagonal.
            shear: 3x3 shear matrix to place in upper-left.
            perspective: length-4 perspective array.

        Returns:
            A new WorldTransform whose `matrix` encodes R.
        """
        return cls.from_values(translation=translation, rotation=euler_to_rotation(
            roll, pitch, yaw, degrees=degrees), scale=scale, shear=shear, perspective=perspective)

    @classmethod
    def from_flat_array(
        cls,
        flat_array: ndarray,
    ) -> "WorldTransform":
        """
        Create a WorldTransform from a flat array.

        Args:
            flat_array: 1D array of 16 floats representing the matrix.

        Returns:
            A new WorldTransform whose `matrix` is constructed from the flat array.
        """
        shape = np_shape(flat_array)
        if shape != (16,):
            raise ValueError(
                f"Invalid flat array shape: {shape}")
        flat_array = np_asarray(flat_array, dtype=np_float64)
        mat = flat_array.reshape((4, 4))
        return cls(mat)

    @classmethod
    def from_list(
        cls,
        list_array: List[float],
    ) -> "WorldTransform":
        """
        Create a WorldTransform from a list.

        Args:
            list_array: 1D list of 16 floats representing the matrix.

        Returns:
            A new WorldTransform whose `matrix` is constructed from the list.
        """
        if len(list_array) != 16:
            raise ValueError(f"Invalid list array length: {len(list_array)}")
        mat = np_array(list_array, dtype=np_float64).reshape((4, 4))
        return cls(mat)

    #########
    # Getters for fundamental properties
    #

    @property
    def translation(self) -> ndarray:
        """
        Extract the translation vector.

        Returns:
            A length-3 array from the first three entries of the fourth column.
        """
        return self.matrix[:3, 3]

    @property
    def rotation(self) -> ndarray:
        """
        Extract the 3x3 rotation submatrix.

        Returns:
            The upper-left 3x3 of `matrix`.
        """
        return pure_rotation_if_possible(self.matrix[:3, :3])

    @property
    def scale(self) -> ndarray:
        """
        Compute and return the scale factors from the object's transformation matrix.

        Returns the scale portion of the upper-left 3x3 decomposition.

        Returns:
            numpy.ndarray: A 1D array containing the scale factors along the x, y, and z axes.
        """
        return decompose_scale_shear(self.matrix[:3, :3])[0]

    @property
    def shear(self) -> ndarray:
        """
        Compute and return the shear component of the 3x3 transformation matrix.

        Returns the shear portion of the upper-left 3x3 decomposition.

        Returns:
            numpy.ndarray: The 3x3 shear component extracted from the transformation matrix.
        """
        return decompose_scale_shear(self.matrix[:3, :3])[1]

    @property
    def perspective(self) -> ndarray:
        """
        Extract the perspective component of the matrix.

        Returns:
            The last row of the matrix.
        """
        return self.matrix[3, :]

    ########
    # Apply methods
    #

    def apply_translation(self, translation: ndarray, *, inplace: bool = False) -> "WorldTransform":
        """
        Apply a translation to this WorldTransform.

        Args:
            translation: length-3 vector to add to current translation.
            inplace: if True, modify this WorldTransform in place.

        Returns:
            WorldTransform with updated translation.
        """
        mat = self.matrix if inplace else self.matrix.copy()
        mat[:3, 3] += translation
        return self if inplace else self.__class__(mat)

    def apply_rotation(self, rotation: ndarray, *, inplace: bool = False) -> "WorldTransform":
        """
        Apply a rotation to this WorldTransform.

        Args:
            rotation: 3x3 matrix to left-multiply current rotation.
            inplace: if True, modify this WorldTransform in place.

        Returns:
            WorldTransform with updated rotation.
        """
        R = self.rotation                          # already the polar R
        P = R.T @ self.matrix[:3, :3]              # hence P = Rᵀ (R·P)
        new_block = rotation @ R @ P               # (Q · R) P
        mat = self.matrix if inplace else self.matrix.copy()
        mat[:3, :3] = new_block
        return self if inplace else self.__class__(mat)

    def apply_rotation_from_quaternion(self, quaternion: ndarray, w_last: bool = True, *, inplace: bool = False) -> "WorldTransform":
        """
        Apply a quaternion to this WorldTransform.

        Args:
            quaternion: 4-element array representing the quaternion.
            inplace: if True, modify this WorldTransform in place.

        Returns:
            WorldTransform with updated rotation.
        """
        return self.apply_rotation(quaternion_to_rotation(quaternion, w_last=w_last), inplace=inplace)

    def apply_rotation_from_euler(self, roll: float, pitch: float, yaw: float, degrees: bool = True, *, inplace: bool = False) -> "WorldTransform":
        """
        Apply Euler angles to this WorldTransform.

        Args:
            euler_angles: 3-element array representing the Euler angles (roll, pitch, yaw).
            inplace: if True, modify this WorldTransform in place.

        Returns:
            WorldTransform with updated rotation.
        """
        return self.apply_rotation(euler_to_rotation(roll, pitch, yaw, degrees=degrees), inplace=inplace)

    def apply_scale(
        self,
        scale: ndarray,
        *,
        order: Literal["before", "after"] = "before",
        inplace: bool = False
    ) -> "WorldTransform":
        """
        Apply a scale to this WorldTransform, either before or after the existing shear.

        Args:
            scale: length-3 factors to multiply each axis.
            order:  "before" to scale before existing shear (default),
                    "after"  to scale after existing shear.
            inplace: if True, modify this WorldTransform in place.

        Returns:
            WorldTransform with updated scale/shear in the requested order.
        """
        scale = np_asarray(scale, float)
        if scale.shape != (3,):
            raise ValueError(f"scale must be (3,), got {scale.shape}")

        # pull apart current affine: matrix = R @ (S · H)
        S, H = decompose_scale_shear(self.matrix[:3, :3])
        scale_mat = np_diag(scale)

        if order == "before":
            P_new = scale_mat @ np_diag(S) @ H
        elif order == "after":
            P_new = np_diag(S) @ H @ scale_mat
        else:
            raise ValueError(
                f"order must be 'before' or 'after', got {order!r}")

        block = self.rotation @ P_new

        M = self.matrix if inplace else self.matrix.copy()
        M[:3, :3] = block
        return self if inplace else self.__class__(M)

    def apply_shear(
        self,
        shear: ndarray,
        *,
        order: Literal["before", "after"] = "after",
        inplace: bool = False
    ) -> "WorldTransform":
        """
        Apply a shear to this WorldTransform, either before or after the existing scale.

        Args:
            shear:   3x3 shear matrix (unit diagonal).
            order:   "before" to shear before existing scale,
                     "after"  to shear after existing scale (default).
            inplace: if True, modify this WorldTransform in place.

        Returns:
            WorldTransform with updated scale/shear in the requested order.
        """
        shear = np_asarray(shear, float)
        if shear.shape != (3, 3) or not np_allclose(np_diag(shear), 1):
            raise ValueError("shear must be 3x3 with unit diagonal")

        # pull apart current affine: matrix = R @ (S · H)
        S, H = decompose_scale_shear(self.matrix[:3, :3])

        if order == "before":
            P_new = np_diag(S) @ shear @ H
        elif order == "after":
            P_new = np_diag(S) @ H @ shear
        else:
            raise ValueError(
                f"order must be 'before' or 'after', got {order!r}")

        block = self.rotation @ P_new

        M = self.matrix if inplace else self.matrix.copy()
        M[:3, :3] = block
        return self if inplace else self.__class__(M)

    def apply_perspective(self, perspective: ndarray, *, inplace: bool = False) -> "WorldTransform":
        """
        Apply a perspective to this WorldTransform.

        Args:
            perspective: length-4 array to add to current perspective.
            inplace: if True, modify this WorldTransform in place.

        Returns:
            WorldTransform with updated perspective.
        """
        mat = self.matrix if inplace else self.matrix.copy()
        mat[3, :] += perspective
        return self if inplace else self.__class__(mat)

    #########
    # Setter methods
    #

    @rotation.setter
    def rotation(self, value: ndarray) -> None:
        """
        Assign a new pure-rotation R_new but preserve the original P = Rᵀ·(R·P).

        Args:
            value: 3x3 rotation matrix to set as rotation.
        """
        # current R and P
        P = self.rotation.T @ self.matrix[:3, :3]
        # write back R_new·P
        self.matrix[:3, :3] = value @ P

    @translation.setter
    def translation(self, value: ndarray) -> None:
        """
        Set the translation vector.

        Args:
            value: length-3 array to set as translation.
        """
        self.matrix[:3, 3] = value

    @shear.setter
    def shear(self, value: ndarray) -> None:
        """
        Overwrite shear while preserving rotation R and scale S.
        `value` must be a 3x3 with ones on the diagonal.
        """
        if np_shape(value) != (3, 3):
            raise ValueError("shear must be 3x3")
        if not np_allclose(np_diag(value), 1):
            raise ValueError("shear diagonal must all be 1")
        S, _ = decompose_scale_shear(self.matrix[:3, :3])
        self.matrix[:3, :3] = self.rotation @ np_diag(S) @ value

    @scale.setter
    def scale(self, value: ndarray) -> None:
        """
        Set the scale factors.

        Args:
            value: length-3 array to set as scale.
        """
        if np_shape(value) != (3,):
            raise ValueError(f"scale must be 3x1.")

        _, H = decompose_scale_shear(self.matrix[:3, :3])
        self.matrix[:3, :3] = self.rotation @ np_diag(value) @ H

    @perspective.setter
    def perspective(self, value: ndarray) -> None:
        """
        Set the perspective component of the matrix.

        Args:
            value: length-4 array to set as perspective.
        """
        self.matrix[3, :] = value

    def set_translation(self, translation: ndarray, *, inplace: bool = False) -> "WorldTransform":
        """
        Assign a translation to this WorldTransform.

        Args:
            translation: length-3 vector to set as translation.

        Returns:
            self with updated translation.
        """
        mat = self.matrix if inplace else self.matrix.copy()
        mat[:3, 3] = translation
        return self if inplace else self.__class__(mat)

    def set_rotation(self, rotation: ndarray, *, inplace: bool = False) -> "WorldTransform":
        """
        Assign a rotation to this WorldTransform.

        Args:
            rotation: 3x3 matrix to set as rotation.

        Returns:
            self with updated rotation.
        """
        # hence P = Rᵀ (R·P)
        P = self.rotation.T @ self.matrix[:3, :3]
        mat = self.matrix if inplace else self.matrix.copy()
        mat[:3, :3] = rotation @ P
        return self if inplace else self.__class__(mat)

    def set_rotation_from_quaternion(self, quaternion: ndarray, w_last: bool = True, *, inplace: bool = False) -> "WorldTransform":
        """
        Assign a quaternion to this WorldTransform.

        Args:
            quaternion: 4-element array representing the quaternion.

        Returns:
            self with updated rotation.
        """
        return self.set_rotation(quaternion_to_rotation(quaternion, w_last=w_last), inplace=inplace)

    def set_rotation_from_euler(self, roll: float, pitch: float, yaw: float, degrees: bool = True, *, inplace: bool = False) -> "WorldTransform":
        """
        Assign Euler angles to this WorldTransform.

        Args:
            euler_angles: 3-element array representing the Euler angles (roll, pitch, yaw).
            inplace: if True, modify this WorldTransform in place.

        Returns:
            self with updated rotation.
        """
        return self.set_rotation(euler_to_rotation(roll, pitch, yaw, degrees=degrees), inplace=inplace)

    def set_scale(
        self,
        scale: ndarray,
        *,
        inplace: bool = False
    ) -> "WorldTransform":
        """Replace the scale vector while preserving rotation *and* shear."""
        scale = np_asarray(scale, float)
        if scale.shape != (3,):
            raise ValueError(f"scale must be (3,), got {scale.shape}")

        # keep the current shear only
        _, H = decompose_scale_shear(self.matrix[:3, :3])
        P_new = np_diag(scale) @ H      # shear stays intact

        M = self.matrix if inplace else self.matrix.copy()
        M[:3, :3] = self.rotation @ P_new
        return self if inplace else self.__class__(M)

    def set_shear(
        self,
        shear: ndarray,
        *,
        inplace: bool = False
    ) -> "WorldTransform":
        """
        Overwrite the shear (preserving R and S) in the requested order.
        """
        shear = np_asarray(shear, float)
        if shear.shape != (3, 3) or not np_allclose(np_diag(shear), 1):
            raise ValueError("shear must be 3x3 with unit diagonal")

        # pull out the current scale (ignore any old shear)
        S, _ = decompose_scale_shear(self.matrix[:3, :3])
        # always rebuild as: R @ (diag(S) @ new_shear)
        P_new = np_diag(S) @ shear
        M = self.matrix if inplace else self.matrix.copy()
        M[:3, :3] = self.rotation @ P_new
        return self if inplace else self.__class__(M)

    def set_perspective(self, perspective: ndarray, *, inplace: bool = False) -> "WorldTransform":
        """
        Assign a perspective to this WorldTransform.

        Args:
            perspective: length-4 array to set as perspective.

        Returns:
            self with updated perspective.
        """
        mat = self.matrix if inplace else self.matrix.copy()
        mat[3, :] = perspective
        return self if inplace else self.__class__(mat)

    ########
    # WorldTransform methods
    #

    def transform_point(self, point: ndarray) -> ndarray:
        """
        Apply the full 4x4 projective transform to a 3D point:
          X_h ≔ M @ [x, y, z, 1]^T
          return  (X_h[:3] / X_h[3])
        """
        ph = self.matrix @ np_append(point, 1.0)
        # guard against w≈0
        w = ph[3]
        if abs(w) < 1e-12:
            raise ZeroDivisionError("projective w coordinate is zero")
        return ph[:3] / w

    def transform_vector(self, v: ndarray) -> ndarray:
        # apply only the top-left 3×3
        return self.matrix[:3, :3] @ v

    ########
    # Target methods
    #

    def distance_to(self, target: Union["WorldTransform", ndarray]) -> np.floating:
        """
        Compute the distance to another WorldTransform or translation vector.

        Args:
            target: the target WorldTransform or translation vector.

        Returns:
            The distance to the target.
        """
        if isinstance(target, WorldTransform):
            tgt = target.matrix[:3, 3]
        else:
            tgt = np_asarray(target, float)

        return np_norm(tgt - self.matrix[:3, 3])

    def vector_to(self, target: Union["WorldTransform", ndarray]) -> ndarray:
        """
        Compute the vector to another WorldTransform or translation vector.

        Args:
            target: the target WorldTransform or translation vector.

        Returns:
            The vector to the target.
        """
        if isinstance(target, WorldTransform):
            tgt = target.matrix[:3, 3]
        else:
            tgt = np_asarray(target, float)

        return tgt - self.matrix[:3, 3]

    def direction_to(self, target: Union["WorldTransform", ndarray]) -> ndarray:
        """
        Compute the direction to another WorldTransform or translation vector.

        Args:
            target: the target WorldTransform or translation vector.

        Returns:
            The direction to the target.
        """
        if isinstance(target, WorldTransform):
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
        target: Union["WorldTransform", ndarray],
    ) -> ndarray:
        """
        Get the rotation from this Trasnform to the target.

        Args:
            target: the target WorldTransform or translation vector.

        Returns:
            Rotation matrix 3x3.
        """
        # 1) grab the world-space target translation
        if isinstance(target, WorldTransform):
            tgt = target.matrix[:3, 3]
        else:
            tgt = np_asarray(target, float)

        # 2) form the vector from this.origin → target
        target_vector = tgt - self.matrix[:3, 3]

        # 3) call into our compiled routine
        return rotation_to(
            target_vector,
            self.rotation,
            self.basis_forward()
        )

    def quaternion_to(self, target: Union["WorldTransform", ndarray], w_last: bool = True) -> ndarray:
        """
        Get the quaternion from this WorldTransform to the target.

        Args:
            target: the target WorldTransform or translation vector.

        Returns:
            Quaternion 4-element array.
        """
        # 1) grab the world-space target translation
        if isinstance(target, WorldTransform):
            tgt = target.matrix[:3, 3]
        else:
            tgt = np_asarray(target, float)

        # 2) form the vector from this.origin → target
        target_vector = tgt - self.matrix[:3, 3]

        # 3) call into our compiled routine
        return rotation_to_quaternion(
            rotation_to(
                target_vector,
                self.rotation,
                self.basis_forward()
            ),
            w_last=w_last
        )

    def euler_angles_to(self, target: Union["WorldTransform", ndarray], degrees: bool = True) -> tuple[float, float, float]:
        """
        Get the Euler angles from this WorldTransform to the target.

        Args:
            target: the target WorldTransform or translation vector.
            degrees: if True, return angles in degrees, else radians.

        Returns:
            (roll, pitch, yaw)
        """
        # 1) grab the world-space target translation
        if isinstance(target, WorldTransform):
            tgt = target.matrix[:3, 3]
        else:
            tgt = np_asarray(target, float)

        # 2) form the vector from this.origin → target
        target_vector = tgt - self.matrix[:3, 3]

        # 3) call into our compiled routine
        return rotation_to_euler(
            rotation_to(
                target_vector,
                self.rotation,
                self.basis_forward()
            ),
            degrees=degrees
        )

    def azimuth_elevation_to(
        self,
        target: Union["WorldTransform", ndarray],
        *,
        degrees: bool = True,
        signed_azimuth: bool = False,
        counterclockwise_azimuth: bool = False,
        flip_elevation: bool = False
    ) -> tuple[float, float]:
        """
        Calculate azimuth, elevation, and range to the target.

        Args:
            origin: the observer WorldTransform.
            target: the target WorldTransform or translation vector.
            degrees: if True, return az/el in degrees, else radians.
            signed_azimuth: if True, az ∈ [-180,180] (or [-π,π]), else [0,360).
            counterclockwise_azimuth: if True, positive az is from forward → left,
                            otherwise forward → right.
            flip_elevation: if True, positive el means downward (down vector),
                            otherwise positive means upward (up vector).

        Returns:
            (azimuth, elevation)
        """
        if isinstance(target, WorldTransform):
            target_vector = target.matrix[:3, 3] - self.matrix[:3, 3]
        else:
            target_vector = np_asarray(target, float) - self.matrix[:3, 3]
        return azimuth_elevation_to(target_vector, self.up, self.right, self.forward, degrees=degrees, signed_azimuth=signed_azimuth, counterclockwise_azimuth=counterclockwise_azimuth, flip_elevation=flip_elevation)

    def phi_theta_to(
        self,
        target: Union["WorldTransform", ndarray],
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
            target: the target WorldTransform or translation vector.
            degrees: if True, return angles in degrees, else radians.
            signed_phi: if True, φ in [-π,π] (or [-180,180]), else [0,2π) (or [0,360)).
            counterclockwise_phi: if True, φ positive from forward → left, else forward → right.
            polar: if True, θ is the polar angle from up (0…π), else θ is elevation from horizontal (−π/2…π/2).
            flip_theta: if True, flip the sign of θ.

        Returns:
            (φ, θ)
        """
        if isinstance(target, WorldTransform):
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
        target: Union["WorldTransform", ndarray],
        *,
        degrees: bool = True,
        signed_longitude: bool = True,
        counterclockwise_longitude: bool = True,
        flip_latitude: bool = False
    ) -> tuple[float, float]:
        """
        Calculate (latitude, longitude) to the target.

        Args:
            target: the target WorldTransform or translation vector.
            degrees: if True, return lat/lon in degrees, else radians.
            signed_longitude: if True, lon in [-π,π] (or [-180,180]), else [0,2π).
            counterclockwise_longitude: if True, lon positive from forward → left, else forward → right.
            flip_latitude: if True, flip the sign of latitude.

        Returns:
            (latitude, longitude)
        """
        if isinstance(target, WorldTransform):
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
        target: Union["WorldTransform", ndarray],
        *,
        inplace: bool = False
    ) -> "WorldTransform":
        """
        Rotate this WorldTransform so that its forward axis points at `target`.
        Rotate only the R part so forward→target, leaving P intact.

        Args:
            target: the target WorldTransform or translation vector.
            inplace: if True, modify this WorldTransform in place.

        Returns:
            WorldTransform with updated rotation.
        """
        # 1) compute pure‐rotation that points forward at target
        if isinstance(target, WorldTransform):
            tgt = target.matrix[:3, 3]
        else:
            tgt = np_asarray(target, float)
        v = tgt - self.matrix[:3, 3]
        R_new = rotation_to(v, self.rotation, self.basis_forward())

        # 2) grab old R and P
        R_old = self.rotation
        P = R_old.T @ self.matrix[:3, :3]

        # 3) compose back and write
        block = R_new @ P
        if inplace:
            self.matrix[:3, :3] = block
            return self

        M = self.matrix.copy()
        M[:3, :3] = block
        return self.__class__(M)

    ########
    # Convenience/utility methods
    #

    def is_rigid(self, tol: float = 1e-6) -> bool:
        """
        True if this is exactly a rotation+translation (no scale/shear/perspective).
        """
        return is_rigid(self.matrix, tol=tol)

    def orthonormalize(self, *, inplace: bool = True) -> "WorldTransform":
        """
        Re-orthonormalize the rotation block to remove drift,
        using the polar decomposition so the result is always
        a proper rotation (det=+1) and never flips scale.

        Returns:
        --------
        WorldTransform
        """
        # polar_rotation_svd returns the closest proper rotation R
        mat = self.matrix if inplace else self.matrix.copy()
        mat[:3, :3] = polar_rotation_svd(self.matrix[:3, :3])
        return self if inplace else self.__class__(mat)

    def inverse(self, *, inplace: bool = False) -> "WorldTransform":
        """
        Invert this WorldTransform.

        Args:
            inplace: if True, modify this WorldTransform in place.

        Returns:
            Inverted WorldTransform.
        """
        inv_mat = np_inv(self.matrix)
        if inplace:
            self.matrix[:] = inv_mat
            return self
        else:
            return self.__class__(inv_mat)

    def transpose(self, *, inplace: bool = False) -> "WorldTransform":
        """
        Transpose of the 4x4 matrix.

        Returns:
            The matrix transposed.
        """
        if inplace:
            self.matrix[:] = self.matrix.T
            return self

        return self.__class__(self.matrix.T.copy())

    def copy(self) -> "WorldTransform":
        """
        Return a copy of this WorldTransform.

        Returns:
            A new WorldTransform with the same matrix.
        """
        return self.__class__(self.matrix.copy())

    ###########
    # World frame properties derived from coordinate system convention
    #

    @property
    def forward(self) -> ndarray:
        """
        Rotate the coordinate system's forward vector into world frame.

        Returns:
            The 3D “forward” direction after applying this transform's rotation.
        """
        ...

    @property
    def backward(self) -> ndarray:
        """
        Rotate the coordinate system's backward vector into world frame.

        Returns:
            The 3D “backward” direction after applying this transform's rotation.
        """
        ...

    @property
    def left(self) -> ndarray:
        """
        Rotate the coordinate system's left vector into world frame.

        Returns:
            The 3D “left” direction after applying this transform's rotation.
        """
        ...

    @property
    def right(self) -> ndarray:
        """
        Rotate the coordinate system's right vector into world frame.

        Returns:
            The 3D “right” direction after applying this transform's rotation.
        """
        ...

    @property
    def up(self) -> ndarray:
        """
        Rotate the coordinate system's up vector into world frame.

        Returns:
            The 3D “up” direction after applying this transform's rotation.
        """
        ...

    @property
    def down(self) -> ndarray:
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
    def basis_x() -> ndarray:
        """
        Returns the basis x vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_y() -> ndarray:
        """
        Returns the basis y vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_z() -> ndarray:
        """
        Returns the basis z vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_matrix() -> ndarray:
        """
        Returns the basis matrix of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_matrix_inv() -> ndarray:
        """
        Returns the inverse of the basis matrix of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_forward() -> ndarray:
        """
        Returns the basis forward vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_backward() -> ndarray:
        """
        Returns the basis backward vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_left() -> ndarray:
        """
        Returns the basis left vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_right() -> ndarray:
        """
        Returns the basis right vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_up() -> ndarray:
        """
        Returns the basis up vector of the transform.
        Implemented in the generated code.
        """
        ...

    @staticmethod
    def basis_down() -> ndarray:
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
        other: Type["WorldTransform"],
        *,
        inplace: bool = False
    ) -> "WorldTransform":
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
            other:     The WorldTransform *class* whose basis you want to switch into.
            inplace:   If True, overwrite self.matrix; otherwise return a new instance.

        Returns:
            A WorldTransform of type `other` whose numeric matrix does the same
            world-space mapping, but expects `other`-local inputs.
        """
        # 1) grab the 3×3 basis matrices (rows = basis vectors in world coords)
        B_old = self.basis_matrix()    # ndarray (3×3)
        B_new = other.basis_matrix()      # ndarray (3×3)

        # 2) compute P = B_old⁻¹ B_new; since B_old is orthonormal, inv = transpose
        P = B_old.T @ B_new                # (3×3)

        # 3) build a 4×4 homogeneous change‐of‐basis
        C = _EYE4.copy()               # ndarray (4×4)
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

    def to_quaternion(self, w_last: bool = True) -> ndarray:
        """
        Extract the quaternion from the rotation matrix.

        Returns:
            A 4-element array representing the quaternion.
        """
        return rotation_to_quaternion(self.rotation, w_last=w_last)

    def to_euler_angles(self, degrees: bool = True) -> Tuple[float, float, float]:
        """
        Extract the Euler angles from the rotation matrix.

        Returns:
            A 3-element tuple representing the Euler angles.
        """
        return rotation_to_euler(self.rotation, degrees=degrees)

    def to_list(self) -> List[float]:
        """
        Convert the matrix to a list of floats.

        Returns:
            A list of 16 floats representing the matrix.
        """
        return self.matrix.flatten().tolist()

    def to_flat_array(self) -> ndarray:
        """
        Convert the matrix to a flat array.

        Returns:
            A 1D array of 16 floats representing the matrix.
        """
        return self.matrix.flatten()

    #########
    # Dunder methods
    #

    def __matmul__(self, other: Union["WorldTransform", ndarray]) -> Union["WorldTransform", ndarray]:
        """
        Compute the matrix multiplication or composition of transforms.
        This magic method overloads the @ operator. It supports two types of operands:
        1. If 'other' is a numpy ndarray, this method applies the transform's 4x4 matrix
            to the array and returns the resulting array.
        2. If 'other' is another WorldTransform, it composes the transforms in such a way that
            the transformation represented by 'other' is applied first, followed by this transform.
            If the two transforms have different coordinate system representations,
            'other' is converted to the same basis as self before composition.
        Parameters:
             other (Union["WorldTransform", ndarray]): The right-hand operand. It can be either:
                  - A numpy ndarray, in which case the transform's matrix is applied to it.
                  - Another WorldTransform object, in which case the transforms are composed (self ∘ other).
        Returns:
             Union["WorldTransform", ndarray]:
                  - A numpy ndarray if 'other' is an ndarray.
                  - A new WorldTransform instance representing the composition if 'other' is a WorldTransform.
        Raises:
             NotImplemented: If 'other' is neither a numpy ndarray nor a WorldTransform instance.
        """
        if isinstance(other, ndarray):
            if np_shape(other) == (3,):
                return self.transform_point(other)
            else:
                return self.matrix @ other

        if not isinstance(other, WorldTransform):
            return NotImplemented

        # if other has a different labeling/basis subclass, convert it
        if other.__class__ is not self.__class__:
            other = other.change_coordinate_system(self.__class__)

        # Compose: first apply `other`, then `self`
        M_combined = self.matrix @ other.matrix
        return self.__class__(M_combined)

    def __mul__(self, other: Union["WorldTransform", ndarray]) -> Union["WorldTransform", ndarray]:
        """
        Alias for the @ operator: allows `self * other` as well as `self @ other`.
        """
        return self.__matmul__(other)

    def __eq__(self, other: object) -> bool:
        """
        True if `other` is the same class and matrices are equal within a small tolerance.
        """
        if not isinstance(other, WorldTransform) or self.__class__ is not other.__class__:
            return False
        return np_allclose(self.matrix, other.matrix)

    def __repr__(self) -> str:
        """
        Unambiguous representation including class name and matrix.
        """
        cls = self.__class__.__name__
        mat = np_array2string(self.matrix, precision=6, separator=', ')
        return f"{cls}(matrix=\n{mat}\n)"

    def __str__(self) -> str:
        """
        Friendly string: delegating to repr for now.
        """
        return self.__repr__()

    def __copy__(self) -> "WorldTransform":
        """
        Shallow copy of this WorldTransform (matrix is copied).
        """
        return self.__class__(self.matrix.copy())

    def __deepcopy__(self, memo) -> "WorldTransform":
        """
        Deep copy support for the copy module.
        """
        # matrices are numeric, so shallow vs deep is effectively the same here
        return self.__copy__()

    def __reduce__(self):
        """
        Pickle support: reprunes to (class, (matrix,))
        """
        return (self.__class__, (self.matrix.copy(),))


def _create_frame_convention(
    x: Direction, y: Direction, z: Direction
) -> Type[WorldTransform]:
    # sanity check
    if len({x, y, z}) != 3:
        raise ValueError("x, y, z must be three distinct Directions")

    # compute handedness
    x_vec = _DIR_TO_VEC[x]
    y_vec = _DIR_TO_VEC[y]
    z_vec = _DIR_TO_VEC[z]
    is_right_handed = bool(np_allclose(np_cross(x_vec, y_vec), z_vec))

    # ensure orthobonality
    if not np_allclose(np_dot(x_vec, y_vec), 0) or not np_allclose(np_dot(x_vec, z_vec), 0) or not np_allclose(np_dot(y_vec, z_vec), 0):
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
                return -pure_rotation_if_possible(self.matrix[:3, :3])[:, column_number]
            return direction
        else:
            def direction(self) -> np.ndarray:
                return pure_rotation_if_possible(self.matrix[:3, :3])[:, column_number]
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

    cls_name = f"WorldTransform<{x.name},{y.name},{z.name}>"
    return type(cls_name, (WorldTransform,), props)


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


def define_convention(x: Direction = Direction.FORWARD, y: Direction = Direction.LEFT, z: Direction = Direction.UP) -> Type[WorldTransform]:
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
    Type[WorldTransform]
        The transform type for the given frame convention.
    """
    val = _FRAME_REGISTRY.get((x, y, z), None)
    if val is None:
        raise ValueError(
            f"Frame convention {x}, {y}, {z} not valid. Must be orthogonal.")
    return val
