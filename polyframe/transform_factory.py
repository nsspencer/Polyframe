
from dataclasses import dataclass
import numpy as np
from typing import Type
from numpy import float64 as np_float64
from polyframe.direction import Direction
from polyframe.transform import Transform

EYE4 = np.eye(4, dtype=np.float64)


def create_frame_convention(
    x: Direction, y: Direction, z: Direction
) -> Type[Transform]:
    # sanity check
    if len({x, y, z}) != 3:
        raise ValueError("x, y, z must be three distinct Directions")

    # map each Direction to its unit‐vector in the *world* frame
    dir_to_vec = {
        Direction.FORWARD:  np.array([1,  0,  0]),
        Direction.BACKWARD: np.array([-1,  0,  0]),
        Direction.LEFT:    np.array([0,  1,  0]),
        Direction.RIGHT:     np.array([0, -1,  0]),
        Direction.UP:       np.array([0,  0,  1]),
        Direction.DOWN:     np.array([0,  0, -1]),
    }

    # compute handedness
    x_vec = dir_to_vec[x]
    y_vec = dir_to_vec[y]
    z_vec = dir_to_vec[z]
    is_right_handed = bool(np.allclose(np.cross(x_vec, y_vec), z_vec))
    # ensure orthobonality
    if not np.allclose(np.dot(x_vec, y_vec), 0) or not np.allclose(np.dot(x_vec, z_vec), 0) or not np.allclose(np.dot(y_vec, z_vec), 0):
        raise ValueError("x, y, z must be orthogonal Directions")

    def x_fn(self): return self.matrix[:3, 0]
    def x_inv_fn(self): return -self.matrix[:3, 0]
    def y_fn(self): return self.matrix[:3, 1]
    def y_inv_fn(self): return -self.matrix[:3, 1]
    def z_fn(self): return self.matrix[:3, 2]
    def z_inv_fn(self): return -self.matrix[:3, 2]

    if x == Direction.FORWARD:
        forward = x_fn
        backward = x_inv_fn
        forward_basis = x_vec
    elif x == Direction.BACKWARD:
        backward = x_fn
        forward = x_inv_fn
        backward_basis = x_vec
    elif x == Direction.LEFT:
        left = x_fn
        right = x_inv_fn
        left_basis = x_vec
    elif x == Direction.RIGHT:
        right = x_fn
        left = x_inv_fn
        right_basis = x_vec
    elif x == Direction.UP:
        up = x_fn
        down = x_inv_fn
        up_basis = x_vec
    elif x == Direction.DOWN:
        down = x_fn
        up = x_inv_fn
        down_basis = x_vec
    else:
        raise ValueError("Invalid direction for x")

    if y == Direction.FORWARD:
        forward = y_fn
        backward = y_inv_fn
        forward_basis = y_vec
    elif y == Direction.BACKWARD:
        backward = y_fn
        forward = y_inv_fn
        backward_basis = y_vec
    elif y == Direction.LEFT:
        left = y_fn
        right = y_inv_fn
        left_basis = y_vec
    elif y == Direction.RIGHT:
        right = y_fn
        left = y_inv_fn
        right_basis = y_vec
    elif y == Direction.UP:
        up = y_fn
        down = y_inv_fn
        up_basis = y_vec
    elif y == Direction.DOWN:
        down = y_fn
        up = y_inv_fn
        down_basis = y_vec
    else:
        raise ValueError("Invalid direction for y")

    if z == Direction.FORWARD:
        forward = z_fn
        backward = z_inv_fn
        forward_basis = z_vec
    elif z == Direction.BACKWARD:
        backward = z_fn
        forward = z_inv_fn
        backward_basis = z_vec
    elif z == Direction.LEFT:
        left = z_fn
        right = z_inv_fn
        left_basis = z_vec
    elif z == Direction.RIGHT:
        right = z_fn
        left = z_inv_fn
        right_basis = z_vec
    elif z == Direction.UP:
        up = z_fn
        down = z_inv_fn
        up_basis = z_vec
    elif z == Direction.DOWN:
        down = z_fn
        up = z_inv_fn
        down_basis = z_vec
    else:
        raise ValueError("Invalid direction for z")

    # pack onto a new subclass
    cls_name = f"Transform<{x.name},{y.name},{z.name}>"
    props = {
        # world directions
        "forward": property(forward),
        "backward": property(backward),
        "left": property(left),
        "right": property(right),
        "up": property(up),
        "down": property(down),

        # basis information
        "is_right_handed": staticmethod(lambda: is_right_handed),
        "label_x": staticmethod(lambda: x),
        "label_y": staticmethod(lambda: y),
        "label_z": staticmethod(lambda: z),
        "basis_x": staticmethod(lambda: x_vec),
        "basis_y": staticmethod(lambda: y_vec),
        "basis_z": staticmethod(lambda: z_vec),
        "basis_matrix": staticmethod(lambda: np.array([x_vec, y_vec, z_vec], dtype=np_float64)),
        "basis_matrix_inv": staticmethod(lambda: np.array([x_vec, y_vec, z_vec], dtype=np_float64).T),
        "basis_forward": staticmethod(lambda: forward_basis),
        "basis_backward": staticmethod(lambda: backward_basis),
        "basis_left": staticmethod(lambda: left_basis),
        "basis_right": staticmethod(lambda: right_basis),
        "basis_up": staticmethod(lambda: up_basis),
        "basis_down": staticmethod(lambda: down_basis),
    }
    NewFrame = type(cls_name, (Transform,), props)
    # make it a slots dataclass
    return dataclass(NewFrame, slots=True)
