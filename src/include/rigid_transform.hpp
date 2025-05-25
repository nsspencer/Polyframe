#pragma once
#include <array>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector3d;

// Tolerance for numerical stability checks
static constexpr double kTol = 1e-6;
static constexpr double kPI  = 3.14159265358979323846;


class RigidTransform
{
    private:
        Vector3d m_translation;
        Matrix3d m_rotation;
        double m_scale;

        /// Compute the nearest proper (determinant +1) orthogonal matrix via SVD.
        static Eigen::Matrix3d sanitize_rotation(const Eigen::Matrix3d &R_in) {
            // 1) ensure all entries are finite
            if (!R_in.allFinite()) {
                throw std::invalid_argument("Rotation contains NaN or Inf"); 
            }
            // 2) SVD decomposition
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(
                R_in, Eigen::ComputeFullU | Eigen::ComputeFullV);
            const Eigen::Matrix3d &U = svd.matrixU();
            const Eigen::Matrix3d &V = svd.matrixV();

            // 3) polar factor = U * V^T
            Eigen::Matrix3d R = U * V.transpose();

            // 4) ensure a proper rotation (determinant +1)
            if (R.determinant() < 0) {
                // flip the last column of U and recompute
                Eigen::Matrix3d U2 = U;
                U2.col(2) *= -1;
                R = U2 * V.transpose();
            }
            return R;
        }

        /// Validate scale and rotation, correcting small drift in rotation.
        void validate_state() {
            // Scale must be positive and finite
            if (!(m_scale > 0.0) || !std::isfinite(m_scale)) {
                throw std::invalid_argument("Scale must be > 0 and finite");
            }

            // Sanitize small numerical errors in the rotation matrix
            m_rotation = sanitize_rotation(m_rotation);
        }
    
    public:
        RigidTransform()
            : m_translation(Vector3d::Zero()), m_rotation(Matrix3d::Identity()), m_scale(1.0) {}

        RigidTransform(const Vector3d &translation, const Matrix3d &rotation, double scale, bool validate = true)
            : m_translation(translation), m_rotation(rotation), m_scale(scale)
            {
                if (validate) {
                    validate_state();
                }
            }

        RigidTransform(const RigidTransform &other)
            : m_translation(other.m_translation),
              m_rotation(other.m_rotation),
              m_scale(other.m_scale) {}

        RigidTransform(RigidTransform &&other) noexcept
            : m_translation(std::move(other.m_translation)),
              m_rotation(std::move(other.m_rotation)),
              m_scale(std::move(other.m_scale)) {}
        
        /// Construct a RigidTransform from a 4x4 homogeneous matrix.
        /// Throws if the matrix is not a valid (finite) 4x4 homogeneous transform
        /// with positive uniform scale and a proper rotation.
        static RigidTransform from_matrix(const Matrix4d &M) {
            /* 1. Finite + bottom‑row checks (no temporaries) */
            if (!M.allFinite())
                throw std::invalid_argument("from_matrix: NaN/Inf detected");

            if (!( M(3,0)==0 && M(3,1)==0 && M(3,2)==0
                && std::abs(M(3,3)-1.0) < kTol ))
                throw std::invalid_argument("from_matrix: bottom row must be [0 0 0 1]");

            /* 2. Extract views (zero‑copy) */
            const auto  A = M.topLeftCorner<3,3>();      // view
            const auto  t = M.col(3).head<3>();          // view

            /* 3. Uniform scale & quick reject */
            double detA = A.determinant();
            if (detA <= 0.0)
                throw std::invalid_argument("from_matrix: determinant must be > 0");
            double scale = std::cbrt(detA);

            Matrix3d R = A / scale;

            /* 4. Fast orthonormality check; cheap fix‑up if needed */
            if (!((R.transpose()*R).isApprox(Matrix3d::Identity(), 1e-6) &&
                std::abs(R.determinant() - 1.0) < 1e-6))
            {
                // About 4× cheaper than SVD; still robust
                Eigen::Quaterniond q(R);
                q.normalize();
                R = q.toRotationMatrix();
            }

            /* 5. Construct without re‑validating */
            return RigidTransform(t, R, scale, false);
        }
        
        // getter methods
        const Matrix3d& rotation() const { return m_rotation; }
        const Vector3d& translation() const { return m_translation; }
        double scale() const { return m_scale; }

        RigidTransform with_translation(const Vector3d &translation) const
        {
            return RigidTransform(translation, m_rotation, m_scale, false);
        }
        
        RigidTransform with_rotation(const Matrix3d &rotation) const
        {
            return RigidTransform(m_translation, rotation, m_scale, true);
        }

        RigidTransform with_scale(double scale) const
        {
            if (scale <= 0.0 || !std::isfinite(scale)) {
                throw std::invalid_argument("Scale must be > 0 and finite");
            }
            return RigidTransform(m_translation, m_rotation, scale, false);
        }
        
        RigidTransform rotate(const Matrix3d &rotation, bool validate = true) const
        {
            return RigidTransform(m_translation, rotation * m_rotation, m_scale, validate);
        }

        RigidTransform rotate_x(double angle, bool degrees = true) const {
            double rad = degrees ? angle * kPI / 180.0 : angle;
            Matrix3d R = Eigen::AngleAxisd(rad, Vector3d::UnitX()).toRotationMatrix();
            return rotate(R, false);
        }

        RigidTransform rotate_y(double angle, bool degrees = true) const {
            double rad = degrees ? angle * kPI / 180.0 : angle;
            Matrix3d R = Eigen::AngleAxisd(rad, Vector3d::UnitY()).toRotationMatrix();
            return rotate(R, false);
        }

        RigidTransform rotate_z(double angle, bool degrees = true) const {
            double rad = degrees ? angle * kPI / 180.0 : angle;
            Matrix3d R = Eigen::AngleAxisd(rad, Vector3d::UnitZ()).toRotationMatrix();
            return rotate(R, false);
        }

        RigidTransform rotate_xyz(double rx, double ry, double rz,
                                    bool degrees = true) const {
            if (degrees) {
                rx *= kPI/180.0; ry *= kPI/180.0; rz *= kPI/180.0;
            }
            Matrix3d R =
                (Eigen::AngleAxisd(rz, Vector3d::UnitZ()) *
                Eigen::AngleAxisd(ry, Vector3d::UnitY()) *
                Eigen::AngleAxisd(rx, Vector3d::UnitX()))
                .toRotationMatrix();
            return rotate(R, false);
        }

        RigidTransform rotate_by_quaternion(const Eigen::Quaterniond& q) const
        {
            Matrix3d R = q.normalized().toRotationMatrix();
            return rotate(R, false);
        }

        RigidTransform translate(const Vector3d &translation) const
        {
            return RigidTransform(m_translation + translation, m_rotation, m_scale, false);
        }

        RigidTransform translate_x(double x) const {
            return translate(Vector3d{x, 0.0, 0.0});
        }
        RigidTransform translate_y(double y) const {
            return translate(Vector3d{0.0, y, 0.0});
        }
        RigidTransform translate_z(double z) const {
            return translate(Vector3d{0.0, 0.0, z});
        }

        RigidTransform translate_xyz(double x, double y, double z) const {
            return translate(Vector3d{x, y, z});
        }

        RigidTransform apply_scale(double scale) const
        {
            if (scale <= 0.0 || !std::isfinite(scale)) {
                throw std::invalid_argument("Scale must be > 0 and finite");
            }
            return RigidTransform(m_translation, m_rotation, m_scale * scale, false);
        }
        
        RigidTransform inverse() const
        {
            Matrix3d inv_rotation = m_rotation.transpose();
            Vector3d inv_translation = -inv_rotation * m_translation;
            return RigidTransform(inv_translation, inv_rotation, 1.0 / m_scale, false);
        }

        Matrix4d to_matrix() const
        {
            Matrix4d matrix = Matrix4d::Identity();
            matrix.block<3, 3>(0, 0) = m_rotation * m_scale;
            matrix.block<3, 1>(0, 3) = m_translation;
            return matrix;
        }
        Matrix4d to_matrix_transpose() const
        {
            Matrix4d matrix = Matrix4d::Identity();
            auto rotation_transpose = m_rotation.transpose();
            matrix.block<3, 3>(0, 0) = rotation_transpose * (1.0 / m_scale);
            matrix.block<3, 1>(0, 3) = -rotation_transpose * m_translation;
            return matrix;
        }

        Eigen::Quaterniond as_quaternion() const
        {
            return Eigen::Quaterniond(m_rotation);
        }

        Eigen::AngleAxisd as_euler() const
        {
            return Eigen::AngleAxisd(m_rotation);
        }

        RigidTransform compose(const RigidTransform &other) const
        {
            return RigidTransform(
                m_translation + m_rotation * other.m_translation,
                m_rotation * other.m_rotation,
                m_scale * other.m_scale,
                false
            );
        }
        RigidTransform operator*(const RigidTransform &other) const
        {
            return compose(other);
        }
        RigidTransform operator*(double scale) const
        {
            return RigidTransform(
                m_translation,
                m_rotation,
                m_scale * scale,
                false
            );
        }
        RigidTransform operator/(double scale) const
        {
            return RigidTransform(
                m_translation,
                m_rotation,
                m_scale / scale,
                false
            );
        }
        RigidTransform& operator=(const RigidTransform &other)
        {
            if (this != &other) {
                m_translation = other.m_translation;
                m_rotation = other.m_rotation;
                m_scale = other.m_scale;
            }
            return *this;
        }

        bool operator==(const RigidTransform &other) const
        {
            return m_translation.isApprox(other.m_translation) &&
                   m_rotation.isApprox(other.m_rotation) &&
                   m_scale == other.m_scale;
        }
        bool operator!=(const RigidTransform &other) const
        {
            return !(*this == other);
        }
        friend std::ostream& operator<<(std::ostream &os, const RigidTransform &t)
        {
            os << "RigidTransform(translation=" << t.m_translation
               << ", rotation=" << t.m_rotation
               << ", scale=" << t.m_scale << ")";
            return os;
        }
        friend std::ostream& operator<<(std::ostream &os, const RigidTransform *t)
        {
            os << "RigidTransform(translation=" << t->m_translation
               << ", rotation=" << t->m_rotation
               << ", scale=" << t->m_scale << ")";
            return os;
        }

};