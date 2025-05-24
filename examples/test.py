from polyframe import RigidTransform, Direction, FrameConvention
import timeit
import numpy as np
from scipy.spatial.transform import Rotation as R

conv = FrameConvention(x=Direction.Forward,
                       y=Direction.Left, z=Direction.Up)

if __name__ == "__main__":
    t = RigidTransform()

    N = 1_000_000
    print("creation: ", timeit.timeit(lambda: RigidTransform(), number=N))

    translation = np.array([1, 2, 3], dtype=np.float64)
    rotation = R.from_euler('xyz', [0.1, 0.2, 0.3]).as_matrix()
    scale = 1
    print("creation from values: ", timeit.timeit(
        lambda: RigidTransform(translation, rotation, scale), number=N))

    # axis times
    print("forward: ", timeit.timeit(lambda: conv.forward(t), number=N))
    print("backward: ", timeit.timeit(lambda: conv.backward(t), number=N))
    print("left: ", timeit.timeit(lambda: conv.left(t), number=N))
    print("right: ", timeit.timeit(lambda: conv.right(t), number=N))
    print("up: ", timeit.timeit(lambda: conv.up(t), number=N))
    print("down: ", timeit.timeit(lambda: conv.down(t), number=N))

    # getters
    print("translation getter: ", timeit.timeit(
        lambda: t.translation, number=N))
    print("rotation getter: ", timeit.timeit(lambda: t.rotation, number=N))
    print("scale getter: ", timeit.timeit(lambda: t.scale, number=N))

    # applying methods
    translation = np.array([1, 2, 3], dtype=np.float64)
    print("translate float64: ", timeit.timeit(
        lambda: t.translate(translation), number=N))
    translation = np.array([1, 2, 3])
    print("translate int: ", timeit.timeit(
        lambda: t.translate(translation), number=N))
    translation = [1, 2, 3]
    print("translate list: ", timeit.timeit(
        lambda: t.translate(translation), number=N))
    translation = (1, 2, 3)
    print("translate tuple: ", timeit.timeit(
        lambda: t.translate(translation), number=N))
    print("translate xyz: ", timeit.timeit(
        lambda: t.translate_xyz(1, 2, 3), number=N))

    translation = [1.0, 2.0, 3.0]
    print("translate list float: ", timeit.timeit(
        lambda: t.translate(translation), number=N))
    translation = (1.0, 2.0, 3.0)
    print("translate tuple float: ", timeit.timeit(
        lambda: t.translate(translation), number=N))
    print("translate xyz float: ", timeit.timeit(
        lambda: t.translate_xyz(1, 2, 3), number=N))

    rot = R.from_euler('xyz', [0.1, 0.2, 0.3]).as_matrix()
    print("rotate: ", timeit.timeit(lambda: t.rotate(rot), number=N))
    print("rotate_x: ", timeit.timeit(
        lambda: t.rotate_x(0.1, degrees=True), number=N))
    print("rotate_y: ", timeit.timeit(
        lambda: t.rotate_y(0.1, degrees=True), number=N))
    print("rotate_z: ", timeit.timeit(
        lambda: t.rotate_z(0.1, degrees=True), number=N))
    print("apply_scale: ", timeit.timeit(lambda: t.apply_scale(2.0), number=N))

    print("inverse: ", timeit.timeit(lambda: t.inverse(), number=N))
    print("to_matrix: ", timeit.timeit(lambda: t.to_matrix(), number=N))
    print("to_matrix_transpose: ", timeit.timeit(
        lambda: t.to_matrix_transpose(), number=N))

    M = t.to_matrix()
    q_list = [0.0, 0.0, 0.0, 1.0]
    q_arr = np.array(q_list, dtype=np.float64)
    eye3 = np.eye(3)

    # from_matrix
    print("from_matrix (eye4):      ", timeit.timeit(
        lambda: RigidTransform.from_matrix(M), number=N))

    # with_* helpers
    translation_list = [1, 2, 3]
    print("with_translation:        ", timeit.timeit(
        lambda: t.with_translation(translation_list), number=N))
    print("with_rotation:           ", timeit.timeit(
        lambda: t.with_rotation(eye3), number=N))
    print("with_scale:              ", timeit.timeit(
        lambda: t.with_scale(2.0), number=N))

    # composition / scalar mul/div
    print("compose (t * t):         ", timeit.timeit(lambda: t * t, number=N))
    print("mul by scalar:           ", timeit.timeit(lambda: t * 2.0, number=N))
    print("div by scalar:           ", timeit.timeit(lambda: t / 2.0, number=N))

    # rotate_by_quaternion (list vs ndarray)
    print("rotate_by_quaternion(lst):", timeit.timeit(
        lambda: t.rotate_by_quaternion(q_list), number=N))
    print("rotate_by_quaternion(arr):", timeit.timeit(
        lambda: t.rotate_by_quaternion(q_arr), number=N))

    # as_quaternion + as_euler
    print("as_quaternion:           ", timeit.timeit(
        lambda: t.as_quaternion(), number=N))
    print("as_euler('xyz'):         ", timeit.timeit(
        lambda: t.as_euler("xyz"), number=N))
    print("as_euler('zyx', False):  ", timeit.timeit(
        lambda: t.as_euler("zyx", False), number=N))

    # comparisons and repr
    print("eq (t==t):               ", timeit.timeit(lambda: t == t, number=N))
    print("ne (t!=t):               ", timeit.timeit(lambda: t != t, number=N))
    print("repr(t):                 ", timeit.timeit(lambda: repr(t), number=N))
