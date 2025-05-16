import numpy as np
from enum import Enum


class Direction(Enum):
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    UP = 4
    DOWN = 5


# map each Direction to its unit‐vector in the *world* frame
_DIR_TO_VEC = {
    Direction.FORWARD:  np.array([1,  0,  0], dtype=np.float64),
    Direction.BACKWARD: np.array([-1,  0,  0], dtype=np.float64),
    Direction.LEFT:    np.array([0,  1,  0], dtype=np.float64),
    Direction.RIGHT:     np.array([0, -1,  0], dtype=np.float64),
    Direction.UP:       np.array([0,  0,  1], dtype=np.float64),
    Direction.DOWN:     np.array([0,  0, -1], dtype=np.float64),
}
