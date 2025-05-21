from typing import Type
import numpy as np
from polyframe2.direction import Direction, _DIR_TO_VEC
from polyframe2.local_transform import LocalTransform


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
