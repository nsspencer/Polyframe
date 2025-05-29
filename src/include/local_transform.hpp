// local_transform.hpp
#pragma once

#include <array>
#include <Eigen/Dense>
#include <cmath>

namespace local
{

/// All six semantic directions in world space.
enum class Direction : int {
    Forward  = 0,
    Backward = 1,
    Left     = 2,
    Right    = 3,
    Up       = 4,
    Down     = 5
};

namespace detail {
    // Precomputed integer vectors for each Direction
    static constexpr std::array<std::array<int,3>,6> dir_vec = {{
        {{  0,  0,  1 }},  // Forward
        {{  0,  0, -1 }},  // Backward
        {{ -1,  0,  0 }},  // Left
        {{  1,  0,  0 }},  // Right
        {{  0,  1,  0 }},  // Up
        {{  0, -1,  0 }}   // Down
    }};

    // Cross product of integer 3-vectors
    static constexpr std::array<int,3> cross3(const std::array<int,3>& a,
                                             const std::array<int,3>& b) noexcept {
        return {{
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        }};
    }

    // Negate vector
    static constexpr std::array<int,3> negate3(const std::array<int,3>& a) noexcept {
        return {{ -a[0], -a[1], -a[2] }};
    }

    // Equality
    static constexpr bool equal3(const std::array<int,3>& a,
                                 const std::array<int,3>& b) noexcept {
        return a[0]==b[0] && a[1]==b[1] && a[2]==b[2];
    }

    // Opposite Direction
    static constexpr Direction opposite(Direction d) noexcept {
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

    // Column index in rotation matrix for a given direction
    static constexpr int indexOf(Direction d,
                                 Direction X, Direction Y, Direction Z) noexcept {
        return (d == X || d == opposite(X)) ? 0
             : (d == Y || d == opposite(Y)) ? 1
             : 2; // must be Z or its opposite
    }

    // Sign of mapping (+1 if matches axis, -1 if opposite)
    static constexpr int signOf(Direction d,
                                Direction X, Direction Y, Direction Z) noexcept {
        return (d == X || d == Y || d == Z) ? +1 : -1;
    }

    // Helper to obtain rotated, scaled axis with compile-time index/sign
    template<int IDX, int SGN>
    static inline Eigen::Vector3d axisDir(const Eigen::Quaterniond& q,
                                          double scale) noexcept {
        if constexpr (IDX == 0)
            return scale * SGN * (q * Eigen::Vector3d::UnitX());
        else if constexpr (IDX == 1)
            return scale * SGN * (q * Eigen::Vector3d::UnitY());
        else
            return scale * SGN * (q * Eigen::Vector3d::UnitZ());
    }


    template<int IDX, int SGN>
    static inline Eigen::Vector3d axisDirPure(const Eigen::Quaterniond& q) noexcept {
        if constexpr (IDX == 0)
            return SGN * (q * Eigen::Vector3d::UnitX());
        else if constexpr (IDX == 1)
            return SGN * (q * Eigen::Vector3d::UnitY());
        else
            return SGN * (q * Eigen::Vector3d::UnitZ());
    }
}

/// A local transform with baked-in frame convention.
/// XAxis/YAxis/ZAxis must be distinct and form a valid (right- or left-handed) basis.
template<
    Direction XAxis = Direction::Forward,
    Direction YAxis = Direction::Left,
    Direction ZAxis = Direction::Up
>
class LocalTransform {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

public:
    // Compile-time validation of distinct axes
    static_assert(int(XAxis) != int(YAxis) &&
                  int(XAxis) != int(ZAxis) &&
                  int(YAxis) != int(ZAxis),
                  "Axes must be distinct Directions");

private:
    // Validate that {X,Y,Z} form a basis
    static constexpr auto vX  = detail::dir_vec[int(XAxis)];
    static constexpr auto vY  = detail::dir_vec[int(YAxis)];
    static constexpr auto vZ  = detail::dir_vec[int(ZAxis)];
    static constexpr auto cXY = detail::cross3(vX, vY);
    static_assert(detail::equal3(cXY, vZ) ||
                  detail::equal3(cXY, detail::negate3(vZ)),
                  "Axes must form a valid 3-D basis (right- or left-handed)");

    // Pre-compute index & sign constants for each named access
    static constexpr int FWD_IDX = detail::indexOf(Direction::Forward , XAxis, YAxis, ZAxis);
    static constexpr int FWD_SGN = detail::signOf (Direction::Forward , XAxis, YAxis, ZAxis);
    static constexpr int BWD_IDX = detail::indexOf(Direction::Backward, XAxis, YAxis, ZAxis);
    static constexpr int BWD_SGN = detail::signOf (Direction::Backward, XAxis, YAxis, ZAxis);
    static constexpr int LFT_IDX = detail::indexOf(Direction::Left    , XAxis, YAxis, ZAxis);
    static constexpr int LFT_SGN = detail::signOf (Direction::Left    , XAxis, YAxis, ZAxis);
    static constexpr int RGT_IDX = detail::indexOf(Direction::Right   , XAxis, YAxis, ZAxis);
    static constexpr int RGT_SGN = detail::signOf (Direction::Right   , XAxis, YAxis, ZAxis);
    static constexpr int UP_IDX  = detail::indexOf(Direction::Up      , XAxis, YAxis, ZAxis);
    static constexpr int UP_SGN  = detail::signOf (Direction::Up      , XAxis, YAxis, ZAxis);
    static constexpr int DWN_IDX = detail::indexOf(Direction::Down    , XAxis, YAxis, ZAxis);
    static constexpr int DWN_SGN = detail::signOf (Direction::Down    , XAxis, YAxis, ZAxis);

    /*---------------- Static basis matrix (once per specialization) ----------------*/
public:
    // Columns are the world‑space unit vectors of this local X/Y/Z.
    inline static const Eigen::Matrix3d BasisMatrix = []{
        Eigen::Matrix3d M;
        auto load = [&](int col,const std::array<int,3>& v){
            M.col(col)=Eigen::Vector3d(double(v[0]),double(v[1]),double(v[2]));};
        load(0,vX); load(1,vY); load(2,vZ);
        return M;
    }();

public:
    // Data members
    Eigen::Vector3d    m_position;
    Eigen::Quaterniond m_orientation;
    double             m_scale;

    // Constructors
    LocalTransform() noexcept
        : m_position(Eigen::Vector3d::Zero()),
          m_orientation(Eigen::Quaterniond::Identity()),
          m_scale(1.0) {}

    explicit LocalTransform(const Eigen::Vector3d& position,
                             const Eigen::Quaterniond& orientation,
                             double scale = 1.0) noexcept
        : m_position(position),
          m_orientation(orientation),
          m_scale(scale) {}
    
    // Copy constructor
    LocalTransform(const LocalTransform& other) noexcept
        : m_position(other.m_position),
          m_orientation(other.m_orientation),
          m_scale(other.m_scale) {}
    
    // Move constructor
    LocalTransform(LocalTransform&& other) noexcept
        : m_position(std::move(other.m_position)),
          m_orientation(std::move(other.m_orientation)),
          m_scale(other.m_scale) {}
    
    // Implicit converting‑ctor between frame specialisations
    template<Direction Ox,Direction Oy,Direction Oz,
        typename = std::enable_if_t<!(Ox==XAxis && Oy==YAxis && Oz==ZAxis)>>
    LocalTransform(const LocalTransform<Ox,Oy,Oz>& o) noexcept
        : m_position(o.m_position), m_scale(o.m_scale)
    {
        auto R = BasisMatrix * LocalTransform<Ox,Oy,Oz>::BasisMatrix.transpose();
        m_orientation = o.m_orientation * Eigen::Quaterniond(R);
    }

    // Explicit conversion operator as alternative to ctor
    template<Direction Dx,Direction Dy,Direction Dz>
    explicit operator LocalTransform<Dx,Dy,Dz>() const noexcept {
        return LocalTransform<Dx,Dy,Dz>(*this);
    }

    // Copy assignment
    LocalTransform& operator=(const LocalTransform& other) noexcept {
        if (this != &other) {
            m_position = other.m_position;
            m_orientation = other.m_orientation;
            m_scale = other.m_scale;
        }
        return *this;
    }

    // Move assignment
    LocalTransform& operator=(LocalTransform&& other) noexcept {
        if (this != &other) {
            m_position = std::move(other.m_position);
            m_orientation = std::move(other.m_orientation);
            m_scale = other.m_scale;
        }
        return *this;
    }

    bool operator==(LocalTransform const& other) const noexcept {
        return m_position.isApprox(other.m_position) &&
                m_orientation.isApprox(other.m_orientation) &&
                m_scale == other.m_scale;
    }

    bool operator!=(LocalTransform const& other) const noexcept { return !(*this == other); }

    // Destructor
    ~LocalTransform() noexcept = default;

    // Basic accessors / mutators
    const Eigen::Vector3d& position()    const noexcept { return m_position; }
    const Eigen::Quaterniond& orientation() const noexcept { return m_orientation; }
    double scale()                       const noexcept { return m_scale; }

    void set_position(double x, double y, double z) noexcept { m_position = Eigen::Vector3d(x, y, z); }
    void set_position(const std::array<double, 3>& p) noexcept {m_position = Eigen::Vector3d(p[0], p[1], p[2]); }
    void set_position(const Eigen::Vector3d& p) noexcept { m_position = p; }
    void set_orientation(const Eigen::Quaterniond& q) noexcept { m_orientation = q; }
    void set_orientation(const Eigen::Matrix3d& R) noexcept { m_orientation = Eigen::Quaterniond(R); }
    void set_orientation(double x, double y, double z, double w) noexcept { m_orientation = Eigen::Quaterniond(w, x, y, z); }
    void set_orientation(const std::array<double, 4>& q) noexcept { m_orientation = Eigen::Quaterniond(q[3], q[0], q[1], q[2]); }
    void set_scale(double s) noexcept { m_scale = s; }

    // *** Named direction accessors – fully resolved at compile time ***
    inline Eigen::Vector3d forward()  const noexcept { return detail::axisDir<FWD_IDX, FWD_SGN>(m_orientation, m_scale); }
    inline Eigen::Vector3d forward_pure() const noexcept { return detail::axisDirPure<FWD_IDX, FWD_SGN>(m_orientation); }
    inline Eigen::Vector3d backward() const noexcept { return detail::axisDir<BWD_IDX, BWD_SGN>(m_orientation, m_scale); }
    inline Eigen::Vector3d backward_pure() const noexcept { return detail::axisDirPure<BWD_IDX, BWD_SGN>(m_orientation); }
    inline Eigen::Vector3d left()     const noexcept { return detail::axisDir<LFT_IDX, LFT_SGN>(m_orientation, m_scale); }
    inline Eigen::Vector3d left_pure() const noexcept { return detail::axisDirPure<LFT_IDX, LFT_SGN>(m_orientation); }
    inline Eigen::Vector3d right()    const noexcept { return detail::axisDir<RGT_IDX, RGT_SGN>(m_orientation, m_scale); }
    inline Eigen::Vector3d right_pure() const noexcept { return detail::axisDirPure<RGT_IDX, RGT_SGN>(m_orientation); }
    inline Eigen::Vector3d up()       const noexcept { return detail::axisDir<UP_IDX , UP_SGN >(m_orientation, m_scale); }
    inline Eigen::Vector3d up_pure()  const noexcept { return detail::axisDirPure<UP_IDX , UP_SGN >(m_orientation); }
    inline Eigen::Vector3d down()     const noexcept { return detail::axisDir<DWN_IDX, DWN_SGN>(m_orientation, m_scale); }
    inline Eigen::Vector3d down_pure() const noexcept { return detail::axisDirPure<DWN_IDX, DWN_SGN>(m_orientation); }

    // Convenience transforms
    inline Eigen::Vector3d transform_point(const Eigen::Vector3d& p) const noexcept { return m_position + (m_orientation * p) * m_scale; }
    inline Eigen::Vector3d transform_vector(const Eigen::Vector3d& v) const noexcept { return (m_orientation * v) * m_scale; }

    LocalTransform operator*(LocalTransform const& r) const noexcept {
        return {
            m_position + (m_orientation * (r.m_position * m_scale)),
            m_orientation * r.m_orientation,
            m_scale * r.m_scale
        };
    }

    LocalTransform inverse() const noexcept {
        auto invQ = m_orientation.conjugate();
        double invS = 1.0 / m_scale;
        return {
            invQ * ((m_position * -1.0) * invS),
            invQ,
            invS
        };
    }
};

} // namespace local