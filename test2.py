from polyframe.local_transform import define_convention, Direction
import numpy as np
import timeit

LocalTransform = define_convention(
    Direction.FORWARD, Direction.LEFT, Direction.UP)

if __name__ == "__main__":
    # Example usage
    t = LocalTransform()
    p = LocalTransform([1, 2, 3])
    t.euler_angles_to(p)
    translation, rotation, scale = np.array(
        [1, 2, 3]), np.eye(3), np.array([1, 2, 3])
    p = LocalTransform(translation, rotation, scale)
    OtherType = define_convention(
        Direction.FORWARD, Direction.RIGHT, Direction.UP)
    t.rotation_as_quaternion()
    t.rotation_as_euler()
    t.to_matrix()
    t.look_at(p)

    print("init", timeit.timeit(lambda: LocalTransform(), number=1_000_000))
    print("from uncheched", timeit.timeit(lambda: LocalTransform.from_unchecked_values(
        translation, rotation, scale), number=1_000_000))

    print("to matrix", timeit.timeit(lambda: t.to_matrix(), number=1_000_000))
    print("forward", timeit.timeit(lambda: t.forward, number=1_000_000))
    print("backward", timeit.timeit(lambda: t.backward, number=1_000_000))
    print("left", timeit.timeit(lambda: t.left, number=1_000_000))
    print("right", timeit.timeit(lambda: t.right, number=1_000_000))
    print("up", timeit.timeit(lambda: t.up, number=1_000_000))
    print("down", timeit.timeit(lambda: t.down, number=1_000_000))
    print("to quaternion", timeit.timeit(
        lambda: t.rotation_as_quaternion(), number=1_000_000))
    print("to euler", timeit.timeit(
        lambda: t.rotation_as_euler(), number=1_000_000))
    print("Change coordinate system", timeit.timeit(
        lambda: t.change_coordinate_system(OtherType), number=100_000))

    print("Equality", timeit.timeit(
        lambda: t == p, number=100_000))

    print("copy", timeit.timeit(
        lambda: t.copy(), number=1_000_000))
    print("look at inplace + copy", timeit.timeit(
        lambda: t.copy().look_at(p, inplace=True), number=1_000_000))
    print("look at + copy", timeit.timeit(
        lambda: t.copy().look_at(p), number=1_000_000))
    print("look at no copy", timeit.timeit(
        lambda: t.look_at(p), number=1_000_000))

    print("euler angles to", timeit.timeit(
        lambda: t.euler_angles_to(p), number=1_000_000))
    print("quaternion to", timeit.timeit(
        lambda: t.quaternion_to(p), number=1_000_000))
    print("rotation to", timeit.timeit(
        lambda: t.rotation_to(p), number=1_000_000))

    print("is_right_handed", timeit.timeit(
        lambda: t.is_right_handed(), number=1_000_000))
    print("is_left_handed", timeit.timeit(
        lambda: t.is_left_handed(), number=1_000_000))
    print("label_x", timeit.timeit(lambda: t.label_x(), number=1_000_000))
    print("label_y", timeit.timeit(lambda: t.label_y(), number=1_000_000))
    print("label_z", timeit.timeit(lambda: t.label_z(), number=1_000_000))
    print("basis_x", timeit.timeit(lambda: t.basis_x(), number=1_000_000))
    print("basis_y", timeit.timeit(lambda: t.basis_y(), number=1_000_000))
    print("basis_z", timeit.timeit(lambda: t.basis_z(), number=1_000_000))
    print("basis_matrix", timeit.timeit(
        lambda: t.basis_matrix(), number=1_000_000))
    print("basis_matrix_inv", timeit.timeit(
        lambda: t.basis_matrix_inv(), number=1_000_000))
    print("basis_forward", timeit.timeit(
        lambda: t.basis_forward(), number=1_000_000))
    print("basis_backward", timeit.timeit(
        lambda: t.basis_backward(), number=1_000_000))
    print("basis_left", timeit.timeit(lambda: t.basis_left(), number=1_000_000))
    print("basis_right", timeit.timeit(
        lambda: t.basis_right(), number=1_000_000))
    print("basis_up", timeit.timeit(lambda: t.basis_up(), number=1_000_000))
    print("basis_down", timeit.timeit(lambda: t.basis_down(), number=1_000_000))
