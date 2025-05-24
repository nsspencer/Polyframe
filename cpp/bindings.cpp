#define PY_ARRAY_UNIQUE_SYMBOL transform_ARRAY_API
#define NPY_NO_DEPRECATED_API  NPY_1_7_API_VERSION

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <numpy/arrayobject.h>   // NumPy C‐API
#include <unsupported/Eigen/EulerAngles> // for euler angle helpers
#include <cmath>

#include "frame_convention.hpp"
#include "rigid_transform.hpp"  // RigidTransform class

namespace nb = nanobind;
using namespace nb::literals;          // for "_a" named-parameter literals

// Forward declaration of the raw init function:
extern "C" int _import_array();


namespace {                          // ── helpers kept in an anon namespace

// ---------------------------------------------------------------------
// Small helpers for cheap NumPy array creation (all RETURN NEW REF)
// ---------------------------------------------------------------------
PyObject* make_array(int nd, const npy_intp* dims, int flags = 0)
{
    return PyArray_New(
        &PyArray_Type,
        nd, const_cast<npy_intp*>(dims), NPY_DOUBLE,
        /*strides=*/nullptr, /*data=*/nullptr,
        /*itemsize=*/0, flags | NPY_ARRAY_WRITEABLE | NPY_ARRAY_ALIGNED,
        /*obj=*/nullptr);
}

PyObject* make_vec3()                { npy_intp d[1]{3};     return make_array(1,d,NPY_ARRAY_C_CONTIGUOUS); }
PyObject* make_mat3()                { npy_intp d[2]{3,3};   return make_array(2,d,NPY_ARRAY_F_CONTIGUOUS); }
PyObject* make_mat4()                { npy_intp d[2]{4,4};   return make_array(2,d,NPY_ARRAY_F_CONTIGUOUS); }
PyObject* make_vec4()                { npy_intp d[1]{4};     return make_array(1,d,NPY_ARRAY_C_CONTIGUOUS); }

// ---------------------------------------------------------------------
// Converter: Python handle  ->  Eigen type (Vector3d / Matrix3d / Matrix4d)
// ---------------------------------------------------------------------
template<typename EigenType, int Rows, int Cols>
NB_INLINE EigenType as_eigen(nb::handle h, const char* name)
{
    /* ---------- 1. zero‑copy ndarray paths ---------- */
    if (PyArray_Check(h.ptr())) {
        PyArrayObject* ao = reinterpret_cast<PyArrayObject*>(h.ptr());

        if (PyArray_NDIM(ao)==2 && Rows>1 && Cols>1 &&
            PyArray_DIM(ao,0)==Rows && PyArray_DIM(ao,1)==Cols &&
            PyArray_TYPE(ao)==NPY_DOUBLE && PyArray_ISCARRAY(ao))
            return Eigen::Map<EigenType>(static_cast<double*>(PyArray_DATA(ao)));

        if (PyArray_NDIM(ao)==1 && Cols==1 &&
            PyArray_DIM(ao,0)==Rows &&
            PyArray_TYPE(ao)==NPY_DOUBLE && PyArray_ISCARRAY(ao))
            return Eigen::Map<EigenType>(static_cast<double*>(PyArray_DATA(ao)));

        /* optional int‑vector fast‑path */
        if constexpr (Cols == 1) {
            if (PyArray_NDIM(ao)==1 && PyArray_DIM(ao, 0)==Rows &&
                (PyArray_TYPE(ao)==NPY_INT32 || PyArray_TYPE(ao)==NPY_INT64) &&
                PyArray_ISCARRAY(ao))
            {
                using EigenI = Eigen::Matrix<long long, Rows, 1>;
                return Eigen::Map<EigenI>(static_cast<long long*>(PyArray_DATA(ao)))
                         .template cast<double>();
            }
        }
        /* fall through → generic cast */
    }

    /* ---------- 2. tuple / list / arbitrary iterable (vector only) ----- */
    if constexpr (Cols == 1) {
        PyObject* fast = PySequence_Fast(h.ptr(), nullptr);
        if (fast) {
            Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
            if (n != Rows) {
                Py_DECREF(fast);
                throw std::invalid_argument(std::string(name) + " must have length "
                                            + std::to_string(Rows));
            }

            EigenType out;
            PyObject** items = PySequence_Fast_ITEMS(fast);
            if constexpr (Rows == 3) {        // fully unrolled
                for (int i=0;i<3;++i) {
                    PyObject* obj = items[i];
                    if (PyFloat_CheckExact(obj))
                        out[i] = PyFloat_AS_DOUBLE(obj);
                    else if (PyLong_CheckExact(obj))
                        out[i] = static_cast<double>(PyLong_AS_LONG(obj));
                    else
                        out[i] = nb::cast<double>(nb::handle(obj));
                }
            } else {
                for (int i=0;i<Rows;++i) {
                    PyObject* obj = items[i];
                    if (PyFloat_CheckExact(obj))
                        out[i] = PyFloat_AS_DOUBLE(obj);
                    else if (PyLong_CheckExact(obj))
                        out[i] = static_cast<double>(PyLong_AS_LONG(obj));
                    else
                        out[i] = nb::cast<double>(nb::handle(obj));
                }
            }
            Py_DECREF(fast);
            return out;
        }
        PyErr_Clear();              // iterator was not materialisable
    }

    /* ---------- 3. generic (PyArray_FromAny) ------------- */
    const int nd = (Cols == 1 ? 1 : 2);
    PyArray_Descr* descr = PyArray_DescrFromType(NPY_DOUBLE);
    PyObject* tmp = PyArray_FromAny(
        h.ptr(), descr, nd, nd,
        NPY_ARRAY_CARRAY | NPY_ARRAY_ALIGNED |
        NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_FORCECAST, nullptr);
    if (!tmp)
        throw std::invalid_argument(std::string("Cannot convert to ") + name);

    EigenType out = Eigen::Map<EigenType>(
        static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(tmp))));
    Py_DECREF(tmp);
    return out;
}

inline Vector3d  as_vec3 (nb::handle h){ return as_eigen<Vector3d ,3,1>(h,"Vector3"); }
inline Matrix3d  as_mat3 (nb::handle h){ return as_eigen<Matrix3d ,3,3>(h,"Matrix3"); }
inline Matrix4d  as_mat4 (nb::handle h){ return as_eigen<Matrix4d ,4,4>(h,"Matrix4"); }

// Quaternion extractor (expects [w,x,y,z] or (4,) ndarray)
Eigen::Quaterniond as_quat(nb::handle h)
{
    if (PyArray_Check(h.ptr())) {
        PyArrayObject* ao = reinterpret_cast<PyArrayObject*>(h.ptr());
        if (PyArray_NDIM(ao)==1 && PyArray_DIM(ao,0)==4 &&
            PyArray_TYPE(ao)==NPY_DOUBLE && PyArray_ISCARRAY(ao))
        {
            double* d = static_cast<double*>(PyArray_DATA(ao));
            return {d[0],d[1],d[2],d[3]};
        }
    }
    std::vector<double> v = nb::cast<std::vector<double>>(h);
    if (v.size()!=4)
        throw std::invalid_argument("Quaternion must have length 4");
    return {v[0],v[1],v[2],v[3]};
}
// ---------------------------------------------------------------------
} // anon‑ns

// ──────────────────────────────────────────────────────────────────────────────
// MODULE
// ──────────────────────────────────────────────────────────────────────────────
NB_MODULE(transform_module, m)
{
    if (_import_array()<0) { PyErr_Print(); return; }

    m.doc() = "Nanobind wrapper for RigidTransform (translation, rotation, scale)";

    using RT = RigidTransform;
    using ColM4d = Eigen::Matrix<double,4,4,Eigen::ColMajor>;

    nb::class_<RT>(m,"RigidTransform")
    // ----------------------------------------------------------------- Ctors
        .def(nb::init<>())
        .def("__init__",
             [](RT& self, nb::handle t, nb::handle R,
                double s, bool validate)
             {
                 new (&self) RT(as_vec3(t), as_mat3(R), s, validate);
             },
             "translation"_a, "rotation"_a, "scale"_a,
             nb::arg("validate")=true)

        .def_static("from_matrix",
            [](nb::handle M){ return RT::from_matrix(as_mat4(M)); },
            "M"_a, nb::rv_policy::move)

    // ----------------------------------------------------------- with_* helpers
        .def("with_translation",
             [](const RT& self, nb::handle t){ return self.with_translation(as_vec3(t)); },
             "translation"_a, nb::rv_policy::move)
        .def("with_rotation",
             [](const RT& self, nb::handle R){ return self.with_rotation(as_mat3(R)); },
             "rotation"_a, nb::rv_policy::move)
        .def("with_scale", &RT::with_scale, "scale"_a, nb::rv_policy::move)

    // -------------------------------------------------------- read‑only props
        .def_prop_ro("translation",
            [](const RT& self){
                PyObject* py = make_vec3();
                Eigen::Map<Vector3d>(static_cast<double*>(
                    PyArray_DATA(reinterpret_cast<PyArrayObject*>(py)))) = self.translation();
                return nb::steal<nb::object>(py);
            }, nb::rv_policy::move)

        .def_prop_ro("rotation",
            [](const RT& self){
                PyObject* py = make_mat3();
                Eigen::Map<Matrix3d>(static_cast<double*>(
                    PyArray_DATA(reinterpret_cast<PyArrayObject*>(py)))) = self.rotation();
                return nb::steal<nb::object>(py);
            }, nb::rv_policy::move)

        .def_prop_ro("scale", &RT::scale)

    // ------------------------------------------------------------- transforms
        .def("translate",
             [](const RT& self, nb::handle v){ return self.translate(as_vec3(v)); },
             "v"_a, nb::rv_policy::move)
        .def("translate_x",&RT::translate_x,"x"_a)
        .def("translate_y",&RT::translate_y,"y"_a)
        .def("translate_z",&RT::translate_z,"z"_a)
        .def("translate_xyz",&RT::translate_xyz,"x"_a,"y"_a,"z"_a)

        .def("rotate",
             [](const RT& self, nb::handle R){ return self.rotate(as_mat3(R)); },
             "R"_a, nb::rv_policy::move)
        .def("rotate_x",&RT::rotate_x,"angle"_a,nb::arg("degrees")=true)
        .def("rotate_y",&RT::rotate_y,"angle"_a,nb::arg("degrees")=true)
        .def("rotate_z",&RT::rotate_z,"angle"_a,nb::arg("degrees")=true)
        .def("rotate_xyz",&RT::rotate_xyz,"rx"_a,"ry"_a,"rz"_a,nb::arg("degrees")=true)
        .def("rotate_by_quaternion",
             [](const RT& self, nb::handle q){ return self.rotate_by_quaternion(as_quat(q)); },
             "q"_a, nb::rv_policy::move)

        .def("apply_scale",&RT::apply_scale,"s"_a,nb::rv_policy::move)
        .def("inverse",&RT::inverse, nb::rv_policy::move)
        .def("compose",&RT::compose,"other"_a, nb::rv_policy::move)

    // --------------------------------------------------------- matrix access
        .def("to_matrix",
             [](const RT& self){
                 PyObject* py = make_mat4();
                 Eigen::Map<ColM4d>(static_cast<double*>(
                     PyArray_DATA(reinterpret_cast<PyArrayObject*>(py)))) = self.to_matrix();
                 return nb::steal<nb::object>(py);
             }, nb::rv_policy::move)

        .def("to_matrix_transpose",
             [](const RT& self){
                 PyObject* py = make_mat4();
                 Eigen::Map<ColM4d>(static_cast<double*>(
                     PyArray_DATA(reinterpret_cast<PyArrayObject*>(py)))) = self.to_matrix_transpose();
                 return nb::steal<nb::object>(py);
             }, nb::rv_policy::move)

    // ------------------------------------------------------- quaternion/euler
        .def("as_quaternion",
             [](const RT& self){
                 PyObject* py = make_vec4();
                 const auto q = self.as_quaternion();
                 double* d = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(py)));
                 d[0]=q.x(); d[1]=q.y(); d[2]=q.z(); d[3]=q.w();
                 return nb::steal<nb::object>(py);
             }, nb::rv_policy::move)

        .def("as_euler",
            [](const RT& self,
                const std::string& order,
                bool degrees) -> nb::object
            {
                const Matrix3d &R = self.rotation();
                Vector3d ang;                         // will hold α,β,γ  (rad)

                /* -------- use the new EulerAngles<> typedefs ---------- */
                if      (order == "xyz") ang = Eigen::EulerAnglesXYZd(R).angles();
                else if (order == "xzy") ang = Eigen::EulerAnglesXZYd(R).angles();
                else if (order == "yxz") ang = Eigen::EulerAnglesYXZd(R).angles();
                else if (order == "yzx") ang = Eigen::EulerAnglesYZXd(R).angles();
                else if (order == "zxy") ang = Eigen::EulerAnglesZXYd(R).angles();
                else if (order == "zyx") ang = Eigen::EulerAnglesZYXd(R).angles();
                else
                    throw std::invalid_argument("Invalid Euler order: " + order);

                if (degrees)
                    ang *= (180.0 / M_PI);

                /* pack into a (3,) float64 ndarray (same helper you already have) */
                PyObject *py = make_vec3();
                std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(py)),
                            ang.data(), 3*sizeof(double));
                return nb::steal<nb::object>(py);
            },
            "order"_a, nb::arg("degrees") = true, nb::rv_policy::move,
            R"pbdoc(
                as_euler(order: str, degrees: bool = True) -> ndarray
                Return the three Euler angles for *order* as a NumPy (3,) array.
            )pbdoc")

    // ------------------------------------------------------- dunder helpers
        .def("__mul__",   nb::overload_cast<const RT&>(&RT::operator* ,nb::const_), nb::is_operator())
        .def("__mul__",   nb::overload_cast<double    >(&RT::operator* ,nb::const_), nb::is_operator())
        .def("__truediv__", nb::overload_cast<double>(&RT::operator/ ,nb::const_), nb::is_operator())
        .def("__eq__",&RT::operator==)
        .def("__ne__",&RT::operator!=)

        .def("__repr__",[](const RT& r){
            std::ostringstream s;
            auto t=r.translation(); auto R=r.rotation();
            s<<"RigidTransform(translation=[" << t.x() << ", " << t.y() << ", " << t.z() << "], "
             <<"rotation=[["<<R(0,0)<<","<<R(0,1)<<","<<R(0,2)<<"],["
                       <<R(1,0)<<","<<R(1,1)<<","<<R(1,2)<<"],["
                       <<R(2,0)<<","<<R(2,1)<<","<<R(2,2)<<"]], scale="<<r.scale()<<")";
            return s.str();
        })
        .def("__str__",[](const RT& r){
            std::ostringstream s;
            auto t=r.translation(); auto R=r.rotation();
            s<<"RigidTransform(translation=[" << t.x() << ", " << t.y() << ", " << t.z() << "], "
             <<"rotation=[["<<R(0,0)<<","<<R(0,1)<<","<<R(0,2)<<"],["
                       <<R(1,0)<<","<<R(1,1)<<","<<R(1,2)<<"],["
                       <<R(2,0)<<","<<R(2,1)<<","<<R(2,2)<<"]], scale="<<r.scale()<<")";
            return s.str();
        });

    // 1) Re-declare the enum in your binding unit (it should match your C++ enum)
    nb::enum_<Direction>(m, "Direction")
        .value("Forward",  Direction::Forward)
        .value("Backward", Direction::Backward)
        .value("Left",     Direction::Left)
        .value("Right",    Direction::Right)
        .value("Up",       Direction::Up)
        .value("Down",     Direction::Down)
        .export_values();

    struct PyConvention {
        int  col_[6];          // which column of R to pick
        int  sign_[6];         // ±1 for each semantic direction

        /* -------- runtime helper for the opposite direction ---------- */
        static constexpr Direction opposite(Direction d) noexcept {
            switch (d) {
                case Direction::Forward:  return Direction::Backward;
                case Direction::Backward: return Direction::Forward;
                case Direction::Left:     return Direction::Right;
                case Direction::Right:    return Direction::Left;
                case Direction::Up:       return Direction::Down;
                case Direction::Down:     return Direction::Up;
            }
            return Direction::Forward;        // unreachable, silences warnings
        }

        PyConvention(Direction x, Direction y, Direction z) {
            /* ---- 1. validity checks (same as before) ---------------- */
            if (int(x)==int(y) || int(x)==int(z) || int(y)==int(z))
                throw std::invalid_argument("Axes must be distinct Directions");

            auto vX = _dir_vec[int(x)], vY = _dir_vec[int(y)], vZ = _dir_vec[int(z)];
            auto cXY = cross3(vX, vY);
            if (!(equal3(cXY, vZ) || equal3(cXY, negate(vZ))))
                throw std::invalid_argument("Axes must form a valid basis");

            /* ---- 2. pre‑compute lookup tables ----------------------- */
            auto fill = [&](Direction d, int column) {
                col_[int(d)]  = column;
                sign_[int(d)] = +1;
                Direction od  = opposite(d);          // **runtime** opposite
                col_[int(od)]  = column;
                sign_[int(od)] = -1;
            };
            fill(x,0);   // local X
            fill(y,1);   // local Y
            fill(z,2);   // local Z
        }

        /* ---- 3. getters -------------------------------------------- */
        inline Eigen::Vector3d dir(const RigidTransform& t, Direction d) const {
            return sign_[int(d)] * t.rotation().col(col_[int(d)]);
        }
        Eigen::Vector3d forward (const RigidTransform& t) const { return dir(t, Direction::Forward); }
        Eigen::Vector3d backward(const RigidTransform& t) const { return dir(t, Direction::Backward); }
        Eigen::Vector3d left    (const RigidTransform& t) const { return dir(t, Direction::Left); }
        Eigen::Vector3d right   (const RigidTransform& t) const { return dir(t, Direction::Right); }
        Eigen::Vector3d up      (const RigidTransform& t) const { return dir(t, Direction::Up); }
        Eigen::Vector3d down    (const RigidTransform& t) const { return dir(t, Direction::Down); }
    };

    auto vec_as_ndarray = [](const Eigen::Vector3d &v) -> nb::object
    {
        PyObject *py = make_vec3();              // ❶ allocate (3,)
        if (!py) throw std::bad_alloc();

        double *buf = static_cast<double*>(
                        PyArray_DATA(reinterpret_cast<PyArrayObject*>(py)));
        Eigen::Map<Eigen::Vector3d, Eigen::Unaligned> dst(buf);
        dst = v;                                    // ❷ copy into NumPy buffer

        return nb::steal<nb::object>(py);           // ❸ hand ownership to Python
    };

    nb::class_<PyConvention>(m, "FrameConvention")
        .def(nb::init<Direction,Direction,Direction>(),
            "x_axis"_a, "y_axis"_a, "z_axis"_a,
            "Create a frame convention mapping local X/Y/Z → semantic directions")

        /* each accessor: grab Eigen vec → wrap into fresh ndarray */
        .def("forward",
            [&](const PyConvention &self, const RigidTransform &rt)
            { return vec_as_ndarray(self.forward(rt)); },
            "rt"_a, nb::rv_policy::move)

        .def("backward",
            [&](const PyConvention &self, const RigidTransform &rt)
            { return vec_as_ndarray(self.backward(rt)); },
            "rt"_a, nb::rv_policy::move)

        .def("left",
            [&](const PyConvention &self, const RigidTransform &rt)
            { return vec_as_ndarray(self.left(rt)); },
            "rt"_a, nb::rv_policy::move)

        .def("right",
            [&](const PyConvention &self, const RigidTransform &rt)
            { return vec_as_ndarray(self.right(rt)); },
            "rt"_a, nb::rv_policy::move)

        .def("up",
            [&](const PyConvention &self, const RigidTransform &rt)
            { return vec_as_ndarray(self.up(rt)); },
            "rt"_a, nb::rv_policy::move)

        .def("down",
            [&](const PyConvention &self, const RigidTransform &rt)
            { return vec_as_ndarray(self.down(rt)); },
            "rt"_a, nb::rv_policy::move);

    /* Factory function simply forwards to the ctor above */
    m.def("make_frame_convention",
        [](Direction x, Direction y, Direction z) {
            return PyConvention(x,y,z);
        },
        "x_axis"_a, "y_axis"_a, "z_axis"_a,
        "Build a FrameConvention at runtime with full validity checks");

    }