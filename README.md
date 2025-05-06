# Polyframe

[![PyPI version](https://img.shields.io/pypi/v/polyframe.svg)](https://pypi.org/project/polyframe/)  
[![License](https://img.shields.io/pypi/l/polyframe.svg)](LICENSE)

A **fast**, **flexible**, zero-cost Python library for 3D homogeneous transforms and arbitrary coordinate-system conventions.

---

## 🚀 Features

- **Arbitrary Cartesian frames**  
  Define any right- or left-handed `(x,y,z)` basis via a simple enum.  
  Zero-cost: all “bring your own conventions” work happens at import time.

- **Flexible spherical conventions**  
  Compute azimuth/elevation, φ/θ or latitude/longitude with pluggable:
  - signed vs. unsigned angles  
  - CW vs. CCW positive rotation  
  - polar vs. elevation definitions  
  - custom “up”/“forward”/“lateral” axes

- **Effortless re-framing**  
  Change the basis of a `Transform` in one call:
  ```python
  from polyframe.frame_registry import FrameRegistry, Direction
  from polyframe.transform import Transform

  tr_world = Transform.from_values(translation=[1,2,3])
  cs_robot = FrameRegistry.from_directions(
      Direction.RIGHT, Direction.FORWARD, Direction.UP
  )
  tr_robot = tr_world.change_basis_to(cs_robot)
  ```

- **Quaternion & Euler support**  
  Apply quaternion or Euler rotations directly:
  ```python
  tr = Transform()
  tr_q = tr.apply_quaternion([0,0,0,1])           # [x,y,z,w]
  tr_e = tr.apply_euler_rotation(roll, pitch, yaw)
  ```

- **Distance & direction utilities**  
  - `distance_to()`, `vector_to()`, `direction_to()`

- **Numba-accelerated core routines**  
  Under the hood, heavy linear-algebra paths (`look_at`, angle computations, quaternion ↔ matrix) are JIT-compiled for max throughput.

- **Memory-efficient**  
  Coordinate frames are static types; no per-instance overhead for storing conventions.  

---

## 📦 Installation

```bash
pip install polyframe
```

---

## 🎬 Quickstart

```python
from polyframe.frame_registry import FrameRegistry, Direction
from polyframe.transform import Transform

# 1) identity at origin, facing +X
tr = Transform()

# 2) translate to (1,2,3)
tr = tr.apply_translation([1,2,3])

# 3) rotate via Euler angles
tr = tr.apply_euler_rotation(roll=0, pitch=45, yaw=90)

# 4) apply a quaternion
tr = tr.apply_quaternion([0,0,0,1])  # x,y,z,w

# 5) make it look at a point
tr = tr.look_at([4,5,6])

# 6) compute spherical angles
az, el, rng = tr.az_el_range_to([7,8,9])

# 7) compute φ/θ
phi, theta = tr.phi_theta_to([1,1,0])

# 8) distance/direction helpers
dist = tr.distance_to([2,2,2])
vec  = tr.vector_to([2,2,2])
dirn = tr.direction_to([2,2,2])

# 9) re-frame into a robot basis
robot_cs = FrameRegistry.from_directions(
    Direction.FORWARD, Direction.DOWN, Direction.LEFT
)
tr_robot = tr.change_basis_to(robot_cs)
```

---

## 🔍 Coordinate-System Conventions

### Cartesian frames

Choose from any of 48 valid `(x_dir,y_dir,z_dir)` triples:

```python
from polyframe.frame_registry import Direction, FrameRegistry

# World: X forward, Y left, Z up
world_cs = FrameRegistry.from_directions(
    Direction.FORWARD,
    Direction.LEFT,
    Direction.UP,
)

# Robot: X right, Y forward, Z down
robot_cs = FrameRegistry.from_directions(
    Direction.RIGHT,
    Direction.FORWARD,
    Direction.DOWN,
)
```

Each frame type exposes `.forward` / `.backward` / `.left` / `.right` / `.up` / `.down` unit vectors.

### Spherical coordinates

- **Azimuth/Elevation** (`az_el_range_to`)
- **Longitude/Latitude** (`lat_lon_to`)
- **φ/θ (polar)** (`phi_theta_to`)

All accept flags:

| Flag                         | Meaning                                             |
| ---------------------------- | --------------------------------------------------- |
| `degrees`                    | degrees (default) or radians                        |
| `signed_*`                   | signed (±180°/π) vs. unsigned (0…360°/2π)           |
| `counterclockwise_*`         | CCW-positive vs. CW-positive                        |
| `polar` (φ/θ only)           | θ as polar (0…π) vs. elevation (±90°)               |
| `flip_*`                     | flip sign of elevation/latitude/θ                   |

---

## ⚙️ API Highlights

```python
from polyframe.transform import Transform

# Construction
tr = Transform()                             # identity
tr2 = Transform.from_values(
    translation=[1,2,3],
    rotation=[[...]],    # 3×3
    scale=[2,2,2],
)

# Translation / Rotation / Scale
tr_t = tr.apply_translation([1,0,0])
tr_r = tr.apply_rotation(R)                  # 3×3
tr_q = tr.apply_quaternion(q)                # 4-vector [x,y,z,w]
tr_e = tr.apply_euler_rotation(roll,pitch,yaw)
tr_s = tr.apply_scale([2,3,4])

# Inverse / Combine
inv    = tr.inverse()
inv_ip = tr.inverse(inplace=True)
C      = tr @ tr2                            # compose

# Point / Vector
p2 = tr.transform_point([1,2,3])
v2 = tr.transform_vector([1,0,0])

# Re-frame
tr_new = tr.change_basis_to(other_frame)

# Look-at
tr_look = tr.look_at([x,y,z])

# Distance & Direction
d = tr.distance_to([x,y,z])
v = tr.vector_to([x,y,z])
dir = tr.direction_to([x,y,z])

# Angles
az, el, rng = tr.az_el_range_to([x,y,z])
phi, th     = tr.phi_theta_to([x,y,z])
lat, lon    = tr.lat_lon_to([x,y,z])
```

See the full [docs](https://github.com/your-org/polyframe).

---

## 🤝 Contributing

1. Fork  
2. Add a branch  
3. Write tests under `tests/`  
4. PR — must pass CI (including Numba builds)

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE).
