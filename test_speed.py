from polyframe import Direction, define_convention
import timeit
import numpy as np


MyTransform = define_convention(
    Direction.FORWARD, Direction.LEFT, Direction.UP)


print("Creation")
print(timeit.timeit(lambda: MyTransform(), number=1_000_000))
translation = np.array([1, 0, 0])
print(timeit.timeit(lambda: MyTransform.from_values(
    translation=translation), number=1_000_000))

t = MyTransform()
t.rotation  # warmup

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
