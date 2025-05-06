# Polyframe

Polyframe is a modern Python library for 3D homogeneous transforms and coordinate system utilities. It supports easy re-framing and coordinate system conversions, making it especially useful in robotics, computer graphics, and simulation applications.

Polyframe philosophies:
* Bring your own conventions
    Polyframe lets you define your own cartesian and spherical coordinate frame conventions, and enforces them using zero cost abstractions
* Performance and flexability
    Polyframe uses dynamic code creation ahead of time to construct frame types that are used to modify the behavior of Transforms
    Polyframe uses numba to optimize many linear algebra routines and transformations
    Polyframe maximizes memory efficiency by using static types to represent frame basis information