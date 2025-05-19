from polyframe.direction import Direction
from polyframe.base_transform import define_transform_convention

LocalTransform = define_transform_convention(
    Direction.FORWARD, Direction.LEFT, Direction.UP)

if __name__ == "__main__":
    t = LocalTransform()
    p = LocalTransform([1, 2, 3])

    r = t.change_basis_to(p)
    pass
