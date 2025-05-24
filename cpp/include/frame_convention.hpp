#pragma once
#include <array>
#include <Eigen/Dense>
#include "polyframe/rigid_transform.hpp"


/// All six semantic directions in world space.
enum class Direction : int {
    Forward  = 0,
    Backward = 1,
    Left     = 2,
    Right    = 3,
    Up       = 4,
    Down     = 5
};

/// Pre‐computed integer vectors for each Direction.
/// [Forward=(0,0,1), Backward=(0,0,-1), Left=(-1,0,0), …]
static constexpr std::array<std::array<int,3>,6> _dir_vec = {{
    {{  0,  0,  1 }},  // Forward
    {{  0,  0, -1 }},  // Backward
    {{ -1,  0,  0 }},  // Left
    {{  1,  0,  0 }},  // Right
    {{  0,  1,  0 }},  // Up
    {{  0, -1,  0 }}   // Down
}};

/// Cross product on 3‐element integer arrays (constexpr).
static constexpr std::array<int,3> cross3(const std::array<int,3> &a,
                                         const std::array<int,3> &b) {
    return {{
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    }};
}

/// Return –a component-wise (constexpr).
constexpr std::array<int,3> negate(const std::array<int,3>& a) {
    return {{ -a[0], -a[1], -a[2] }};
}

/// constexpr equality for three-component integer arrays.
constexpr bool equal3(const std::array<int,3>& a,
                      const std::array<int,3>& b) {
    return a[0]==b[0] && a[1]==b[1] && a[2]==b[2];
}

/// A compile‐time‐checked orientation convention mapping
/// local X/Y/Z axes → world‐space semantic Directions.
template<Direction XAxis, Direction YAxis, Direction ZAxis>
struct FrameConvention {
    // 1) Enforce at compile time that the three labels are distinct.
    static_assert(int(XAxis) != int(YAxis) &&
                  int(XAxis) != int(ZAxis) &&
                  int(YAxis) != int(ZAxis),
                  "Axes must be distinct Directions");

    // 2) Enforce that they form a valid (right- or left-handed) basis.
    static constexpr auto vX  = _dir_vec[int(XAxis)];
    static constexpr auto vY  = _dir_vec[int(YAxis)];
    static constexpr auto vZ  = _dir_vec[int(ZAxis)];
    static constexpr auto cXY = cross3(vX, vY);
    static_assert(
        equal3(cXY, vZ) || equal3(cXY, negate(vZ)),
        "Axes must form a valid 3-D basis (right- or left-handed)");

    // 3) A constexpr helper for “opposite” of a Direction.
    static constexpr Direction opposite(Direction d) {
        switch (d) {
            case Direction::Forward:  return Direction::Backward;
            case Direction::Backward: return Direction::Forward;
            case Direction::Left:     return Direction::Right;
            case Direction::Right:    return Direction::Left;
            case Direction::Up:       return Direction::Down;
            case Direction::Down:     return Direction::Up;
        }
        return Direction::Forward; // unreachable
    }

    // 4) Map any Direction → which column (0,1,2) to pick from R, at compile time.
    static constexpr int indexOf(Direction d) noexcept {
        return (d == XAxis || d == opposite(XAxis)) ? 0
             : (d == YAxis || d == opposite(YAxis)) ? 1
             :                                        2;
    }

    // 5) Sign: +1 if it's the “positive” mapping, –1 if it’s the opposite.
    static constexpr int signOf(Direction d) noexcept {
        return (d == XAxis || d == YAxis || d == ZAxis) ? +1 : -1;
    }

    // 6) The zero-overhead getter: one multiply + one col fetch.
    static inline Eigen::Vector3d get(const RigidTransform &t,
                                      Direction d) noexcept {
        return signOf(d) * t.rotation().col(indexOf(d));
    }

    // 7) Named accessors
    static inline Eigen::Vector3d forward (const RigidTransform &t) noexcept {
        return get(t, Direction::Forward);
    }
    static inline Eigen::Vector3d backward(const RigidTransform &t) noexcept {
        return get(t, Direction::Backward);
    }
    static inline Eigen::Vector3d left    (const RigidTransform &t) noexcept {
        return get(t, Direction::Left);
    }
    static inline Eigen::Vector3d right   (const RigidTransform &t) noexcept {
        return get(t, Direction::Right);
    }
    static inline Eigen::Vector3d up      (const RigidTransform &t) noexcept {
        return get(t, Direction::Up);
    }
    static inline Eigen::Vector3d down    (const RigidTransform &t) noexcept {
        return get(t, Direction::Down);
    }
};

template<Direction X, Direction Y, Direction Z>
constexpr auto make_frame_convention() {
    return FrameConvention<X,Y,Z>{};   // variable is the instantiation
}

// // can define a convention like this
// using Conv = FrameConvention<
//                     Direction::Forward,
//                     Direction::Left,
//                     Direction::Right>;


// // or use the helper to define the convention
// static constexpr auto X_FORWARD_Z_UP = make_frame_convention<
//     Direction::Forward,
//     Direction::Left,
//     Direction::Up>();

// static constexpr auto Z_FORWARD_X_RIGHT = make_frame_convention<
//     Direction::Right,
//     Direction::Up,
//     Direction::Forward>();