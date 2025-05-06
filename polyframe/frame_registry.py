# frame_registry.py

import numpy as np
from numpy import cross as np_cross
from numpy import dot as np_dot
from numpy import array_equal as np_array_equal
from polyframe.direction import Direction

GENERATED_COORDINATE_SYSTEMS = {}


class CoordinateFrameType(type):
    def __call__(cls, *args, **kwargs):
        raise TypeError(
            f"{cls.__name__} cannot be instantiated; use the class itself for lookups.")

    def __setattr__(cls, name, value):
        if name in cls.__dict__:
            raise AttributeError(f"Cannot reassign '{name}' on {cls.__name__}")
        super().__setattr__(name, value)


class FrameRegistry:
    """Container for every valid generated coordinate system."""
    __slots__ = ()
    default: CoordinateFrameType = None

    @staticmethod
    def from_directions(x: Direction, y: Direction, z: Direction) -> CoordinateFrameType:
        """
        Returns the coordinate system class for the given x, y, z directions.
        Raises KeyError if the combination is not valid.
        """
        try:
            return GENERATED_COORDINATE_SYSTEMS[(x, y, z)]
        except KeyError:
            raise KeyError(
                f"Invalid coordinate system: ({x}, {y}, {z}). Basis vectors must be orthogonal.")

    @staticmethod
    def from_name(name: str) -> CoordinateFrameType:
        """
        Returns the coordinate system class for the given name.
        """
        return getattr(FrameRegistry, name, None)

    @staticmethod
    def change_of_basis(from_frame: CoordinateFrameType, to_frame: CoordinateFrameType) -> np.ndarray:
        """
        Returns the change of basis matrix from one frame to another.
        """
        return to_frame.matrix @ from_frame.inv_matrix

    @staticmethod
    def valid_frames() -> list[CoordinateFrameType]:
        """
        Returns a list of all valid coordinate frame classes.
        """
        return [getattr(FrameRegistry, name) for name in GENERATED_COORDINATE_SYSTEMS.keys()]

    @staticmethod
    def set_default(new_default: CoordinateFrameType) -> None:
        """
        Change the default coordinate frame to the given one.
        """
        FrameRegistry.default = new_default


###
# construct the valid coordinate frames
__dir_to_vec = {
    Direction.FORWARD:  (1,  0,  0),
    Direction.BACKWARD: (-1,  0,  0),
    Direction.LEFT:     (0,  1,  0),
    Direction.RIGHT:    (0, -1,  0),
    Direction.UP:       (0,  0,  1),
    Direction.DOWN:     (0,  0, -1),
}

for dx in Direction:
    fwd = __dir_to_vec[dx]
    for dy in Direction:
        left = __dir_to_vec[dy]
        # must be orthogonal
        if np_dot(fwd, left) != 0:
            continue

        cross_fl = np_cross(fwd, left)
        # degenerate if zero
        if not np.any(cross_fl):
            continue

        for dz in Direction:
            up = __dir_to_vec[dz]
            # up must be orthogonal to both
            if np_dot(up, fwd) != 0 or np_dot(up, left) != 0:
                continue

            # check handedness
            if np_array_equal(cross_fl, up) or np_array_equal(cross_fl, tuple(-i for i in up)):
                # valid frame
                basis = np.stack((fwd, left, up), axis=1, dtype=np.int8)
                x, y, z = dx, dy, dz
                class_name = f"X_{str(x).split('.')[1]}_Y_{str(y).split('.')[1]}_Z_{str(z).split('.')[1]}"
                is_right_handed = np_array_equal(
                    np_cross(basis[:, 0], basis[:, 1]), basis[:, 2])
                namespace = {
                    '__slots__': (),               # remove instance __dict__
                    '__doc__': f"CoordinateFrameType '{class_name}', use attributes only.",
                    '__module__': __name__,        # for pickling
                    'forward':  basis[:, 0],
                    'backward': -basis[:, 0],
                    'left':     basis[:, 1],
                    'right': -basis[:, 1],
                    'up':       basis[:, 2],
                    'down': -basis[:, 2],
                    'matrix':   basis,
                    'inv_matrix': basis.T,
                    'right_handed': is_right_handed,
                    'determinant': np.linalg.det(basis),
                    'x': x,
                    'y': y,
                    'z': z,
                    'xyz': (x, y, z),
                }
                generated_class = CoordinateFrameType(
                    class_name, (), namespace)
                GENERATED_COORDINATE_SYSTEMS[(x, y, z)] = generated_class

                # add the class to the global namespace
                setattr(FrameRegistry, class_name, generated_class)
                globals()[class_name] = generated_class
del __dir_to_vec  # cleanup temporary variable

# set the default coordinate system to X_FORWARD_Y_LEFT_Z_UP
FrameRegistry.set_default(FrameRegistry.from_directions(
    Direction.FORWARD, Direction.LEFT, Direction.UP))
