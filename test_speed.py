from polyframe import Direction, Transform, create_frame_convention
import timeit
import numpy as np


MyTransform = create_frame_convention(
    Direction.FORWARD, Direction.LEFT, Direction.UP)


print("Creation")
print(timeit.timeit(lambda: MyTransform(), number=1_000_000))
translation = np.array([1, 0, 0])
print(timeit.timeit(lambda: MyTransform.from_translation(
    translation), number=1_000_000))

valid_frames = []
for x in Direction:
    for y in Direction:
        for z in Direction:
            try:
                convention = create_frame_convention(x, y, z)
                valid_frames.append(convention)
            except ValueError:
                pass

t = MyTransform()
t.apply_translation(translation, inplace=True)
print("forward")
print(timeit.timeit(lambda: t.forward, number=1_000_000))


print("translation")
print(timeit.timeit(lambda: t.translation, number=1_000_000))


print("apply translation")
print(timeit.timeit(lambda: t.apply_translation(
    translation, inplace=False), number=1_000_000))

print("apply translation in place")
print(timeit.timeit(lambda: t.apply_translation(
    translation, inplace=True), number=1_000_000))

print("apply translation setter")


def assign_translation():
    t.translation += translation


print(timeit.timeit(assign_translation, number=1_000_000))

print("rotation")
print(timeit.timeit(lambda: t.rotation, number=1_000_000))
