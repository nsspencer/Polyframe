// ──────────────────────────────────────────────────────────────────────────────
// polyframe bindings – full type-hinted & documented nanobind module
// ──────────────────────────────────────────────────────────────────────────────
#define PY_ARRAY_UNIQUE_SYMBOL transform_ARRAY_API
#define NPY_NO_DEPRECATED_API  NPY_1_7_API_VERSION

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <numpy/arrayobject.h>
#include <unsupported/Eigen/EulerAngles>
#include <cmath>
#include "frame_convention.hpp"
#include "rigid_transform.hpp"
#include "local_transform.hpp"

namespace nb = nanobind;
using namespace nb::literals;            // for "_a" literals
using nb::sig;

// or use the helper to define the convention
// static auto X_FORWARD_Z_UP = local::LocalTransform<
//     local::Direction::Forward,
//     local::Direction::Left,
//     local::Direction::Up>();

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
    /* ---------- 1. zero-copy ndarray paths ---------- */
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

        /* optional int-vector fast-path */
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

// Quaternion extractor (expects list, tuple, or ndarray of shape (4,))
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
} // anon-ns

// ──────────────────────────────────────────────────────────────────────────────
// MODULE
// ──────────────────────────────────────────────────────────────────────────────
NB_MODULE(_polyframe, m)
{
    if (_import_array()<0) { PyErr_Print(); return; }

    m.doc() = R"pbdoc(
        polyframe
        =========
        Nanobind bindings for **RigidTransform** (3-D similarity transform) and
        compile-time/ run-time **FrameConvention** helpers.
    )pbdoc";

    using RT = RigidTransform;
    using ColM4d = Eigen::Matrix<double,4,4,Eigen::ColMajor>;

    nb::class_<RT>(m,"RigidTransform",
        R"pbdoc(
            RigidTransform(translation: ndarray, rotation: ndarray, scale: float, validate: bool = True)

            A 3D similarity transform with translation, rotation, and uniform scale.
            The rotation is represented as a 3x3 matrix or a quaternion.

            Parameters
            ----------
            translation : ndarray
                A (3,) array representing the translation vector.
            rotation : ndarray
                A (3,3) array representing the rotation matrix or a (4,) array for quaternion.
            scale : float
                A positive float representing the uniform scale factor.
            validate : bool, optional
                If True, validates the state of the transform (default is True).
        )pbdoc")

    // ----------------------------------------------------------------- Ctors
        .def(nb::init<>(), "Default constructor, initializes to identity transform.")
        .def("__init__",
            [](RigidTransform &self,
            nb::handle t,
            nb::handle R,
            double s,
            bool validate)
            {
                // if t is None → zero vector, else cast
                Eigen::Vector3d tv = t.is_none()
                    ? Eigen::Vector3d::Zero()
                    : as_vec3(t);

                // if R is None → identity, else cast
                Eigen::Matrix3d Rm = R.is_none()
                    ? Eigen::Matrix3d::Identity()
                    : as_mat3(R);

                // placement‐new with chosen defaults
                new (&self) RigidTransform(tv, Rm, s, validate);
            },
            // default‐args so Python sees: translation=None, rotation=None, scale=1.0, validate=True
            nb::arg("translation") = nb::none(),
            nb::arg("rotation")    = nb::none(),
            nb::arg("scale")       = 1.0,
            nb::arg("validate")    = true,

            // signature must start with "def __init__"
            sig("def __init__("
                "self, "
                "translation: Union[None, numpy.ndarray, Sequence[float]] = None, "
                "rotation: Union[None, numpy.ndarray] = None, "
                "scale: float = 1.0, "
                "validate: bool = True"
                ") -> None"),
            R"pbdoc(
                Construct a RigidTransform from translation, rotation, and scale.

                Parameters
                ----------
                translation : Union[None, numpy.ndarray, Sequence[float]]
                    A (3,) array representing the translation vector. If None, defaults to zero vector.
                rotation : Union[None, numpy.ndarray]
                    A (3,3) array for rotation matrix. If None, defaults to identity.
                scale : float
                    A positive float representing the uniform scale factor. Defaults to 1.0.
                validate : bool, optional
                    If True, validates the validity of the rotation (default is True).
                )pbdoc")

        .def_static("from_matrix",
            [](nb::handle M){ return RT::from_matrix(as_mat4(M)); },
            "matrix"_a,
            sig("def from_matrix(matrix: numpy.ndarray) -> RigidTransform"),
            R"pbdoc(
                Build from a 4x4 homogeneous matrix.

                • The bottom row must equal ``[0 0 0 1]``  
                • Extracts positive uniform scale ``s = ³√(det A)``  
                • Corrects small numerical drift in the rotational part.  
                • Raises ``ValueError`` if the matrix is not finite or has
                  non-uniform / negative scale.
            )pbdoc",
            nb::rv_policy::move)

    // -------------------------------------------------------- read-only props
        .def_prop_ro("translation",
            [](const RT& self){
                PyObject* py = make_vec3();
                Eigen::Map<Vector3d>(static_cast<double*>(
                    PyArray_DATA(reinterpret_cast<PyArrayObject*>(py)))) = self.translation();
                return nb::steal<nb::object>(py);
            }, 
            sig("def translation(self) -> numpy.ndarray"),
            R"pbdoc(
                Get the read-only translation vector as a NumPy (3,) array.
            )pbdoc",
            nb::rv_policy::move)

        .def_prop_ro("rotation",
            [](const RT& self){
                PyObject* py = make_mat3();
                Eigen::Map<Matrix3d>(static_cast<double*>(
                    PyArray_DATA(reinterpret_cast<PyArrayObject*>(py)))) = self.rotation();
                return nb::steal<nb::object>(py);
            },
            sig("def rotation(self) -> numpy.ndarray"),
            R"pbdoc(
                Get the read-only rotation matrix as a NumPy (3,3) array.
            )pbdoc",
            nb::rv_policy::move)

        .def_prop_ro("scale", &RT::scale, sig("def scale(self) -> float"),
            R"pbdoc(
                Get the read-only uniform scale factor as a float.
            )pbdoc")

    // ----------------------------------------------------------- with_* helpers
        .def("with_translation",
             [](const RT& self, nb::handle t){ return self.with_translation(as_vec3(t)); },
             "translation"_a, 
             sig("def with_translation(self, translation: Union[numpy.ndarray, Sequence[float]]) -> RigidTransform"),
                R"pbdoc(
                    Return a new RigidTransform with the given translation vector.
    
                    Parameters
                    ----------
                    translation : Union[numpy.ndarray, Sequence[float]]
                        A (3,) array representing the new translation vector.
                )pbdoc",
             nb::rv_policy::move)
        .def("with_rotation",
             [](const RT& self, nb::handle R){ return self.with_rotation(as_mat3(R)); },
             "rotation"_a,
             sig("def with_rotation(self, rotation: numpy.ndarray) -> RigidTransform"),
                R"pbdoc(
                    Return a new RigidTransform with the given rotation matrix.
    
                    Parameters
                    ----------
                    rotation : numpy.ndarray
                        A (3,3) array representing the new rotation matrix.
                )pbdoc",
             nb::rv_policy::move)
        .def("with_scale", &RT::with_scale,
            "scale"_a,
            sig("def with_scale(self, scale: float) -> RigidTransform"),
                R"pbdoc(
                    Return a new RigidTransform with the given uniform scale factor.
    
                    Parameters
                    ----------
                    scale : float
                        A positive float representing the new uniform scale factor.
                )pbdoc",
            nb::rv_policy::move)

    // ------------------------------------------------------------- transforms
        .def("translate",
             [](const RT& self, nb::handle translation){ return self.translate(as_vec3(translation)); },
             "translation"_a,
             sig("def translate(self, translation: Union[numpy.ndarray, Sequence[float]]) -> RigidTransform"),
                R"pbdoc(
                    Translate the transform by a vector.
                    Returns a new RigidTransform with the translation applied.
    
                    Parameters
                    ----------
                    v : Union[numpy.ndarray, Sequence[float]]
                        A (3,) array representing the translation vector.
                )pbdoc",
             nb::rv_policy::move)
        .def("translate_x",&RT::translate_x,"x"_a
            ,sig("def translate_x(self, x: float) -> RigidTransform"),
                R"pbdoc(
                    Translate the transform along the X axis.
                    Returns a new RigidTransform with the translation applied.
    
                    Parameters
                    ----------
                    x : float
                        The translation distance along the X axis.
                )pbdoc")
        .def("translate_y",&RT::translate_y,"y"_a,
            sig("def translate_y(self, y: float) -> RigidTransform"),
                R"pbdoc(
                    Translate the transform along the Y axis.
                    Returns a new RigidTransform with the translation applied.
    
                    Parameters
                    ----------
                    y : float
                        The translation distance along the Y axis.
                )pbdoc")
        .def("translate_z",&RT::translate_z,"z"_a,
            sig("def translate_z(self, z: float) -> RigidTransform"),
                R"pbdoc(
                    Translate the transform along the Z axis.
                    Returns a new RigidTransform with the translation applied.
    
                    Parameters
                    ----------
                    z : float
                        The translation distance along the Z axis.
                )pbdoc")
        .def("translate_xyz",&RT::translate_xyz,"x"_a,"y"_a,"z"_a,
            sig("def translate_xyz(self, x: float, y: float, z: float) -> RigidTransform"),
                R"pbdoc(
                    Translate the transform by a vector (x, y, z).
                    Returns a new RigidTransform with the translation applied.
    
                    Parameters
                    ----------
                    x : float
                        The translation distance along the X axis.
                    y : float
                        The translation distance along the Y axis.
                    z : float
                        The translation distance along the Z axis.
                )pbdoc")

        .def("rotate",
             [](const RT& self, nb::handle R){ return self.rotate(as_mat3(R)); },
             "R"_a, 
             sig("def rotate(self, R: numpy.ndarray) -> RigidTransform"),
                R"pbdoc(
                    Rotate the transform by a rotation matrix.
                    Returns a new RigidTransform with the rotation applied.
    
                    Parameters
                    ----------
                    R : numpy.ndarray
                        A (3,3) array representing the rotation matrix.
                )pbdoc",
             nb::rv_policy::move)
        .def("rotate_x",&RT::rotate_x,"angle"_a,nb::arg("degrees")=true,
            sig("def rotate_x(self, angle: float, degrees: bool = True) -> RigidTransform"),
                R"pbdoc(
                    Rotate the transform around the X axis.
                    Returns a new RigidTransform with the rotation applied.
    
                    Parameters
                    ----------
                    angle : float
                        The rotation angle in radians or degrees (if `degrees` is True).
                    degrees : bool, optional
                        If True, interprets `angle` as degrees (default is True).
                )pbdoc")
        .def("rotate_y",&RT::rotate_y,"angle"_a,nb::arg("degrees")=true,
            sig("def rotate_y(self, angle: float, degrees: bool = True) -> RigidTransform"),
                R"pbdoc(
                    Rotate the transform around the Y axis.
                    Returns a new RigidTransform with the rotation applied.
    
                    Parameters
                    ----------
                    angle : float
                        The rotation angle in radians or degrees (if `degrees` is True).
                    degrees : bool, optional
                        If True, interprets `angle` as degrees (default is True).
                )pbdoc")
        .def("rotate_z",&RT::rotate_z,"angle"_a,nb::arg("degrees")=true,
            sig("def rotate_z(self, angle: float, degrees: bool = True) -> RigidTransform"),
                R"pbdoc(
                    Rotate the transform around the Z axis.
                    Returns a new RigidTransform with the rotation applied.
    
                    Parameters
                    ----------
                    angle : float
                        The rotation angle in radians or degrees (if `degrees` is True).
                    degrees : bool, optional
                        If True, interprets `angle` as degrees (default is True).
                )pbdoc")
        .def("rotate_xyz",&RT::rotate_xyz,"rx"_a,"ry"_a,"rz"_a,nb::arg("degrees")=true,
            sig("def rotate_xyz(self, rx: float, ry: float, rz: float, degrees: bool = True) -> RigidTransform"),
                R"pbdoc(
                    Rotate the transform by Euler angles (rx, ry, rz).
                    Returns a new RigidTransform with the rotation applied.
    
                    Parameters
                    ----------
                    rx : float
                        The rotation angle around the X axis in radians or degrees (if `degrees` is True).
                    ry : float
                        The rotation angle around the Y axis in radians or degrees (if `degrees` is True).
                    rz : float
                        The rotation angle around the Z axis in radians or degrees (if `degrees` is True).
                    degrees : bool, optional
                        If True, interprets angles as degrees (default is True).
                )pbdoc")
        .def("rotate_by_quaternion",
             [](const RT& self, nb::handle q){ return self.rotate_by_quaternion(as_quat(q)); },
             "q"_a, 
             sig("def rotate_by_quaternion(self, q: Union[numpy.ndarray, Sequence[float]]) -> RigidTransform"),
                R"pbdoc(
                    Rotate the transform by a quaternion.
                    Returns a new RigidTransform with the rotation applied.
                    
                    Assumes quaternion in [x, y, z, w] order.
    
                    Parameters
                    ----------
                    q : Union[numpy.ndarray, Sequence[float]]
                        A (4,) array representing the quaternion.
                )pbdoc",
             nb::rv_policy::move)

        .def("apply_scale",&RT::apply_scale,"s"_a, 
            sig("def apply_scale(self, s: float) -> RigidTransform"),
                R"pbdoc(
                    Apply a uniform scale to the transform.
                    Returns a new RigidTransform with the scale applied.
    
                    Parameters
                    ----------
                    s : float
                        A positive float representing the uniform scale factor.
                )pbdoc",
            nb::rv_policy::move)
        .def("inverse",&RT::inverse,
            sig("def inverse(self) -> RigidTransform"),
                R"pbdoc(
                    Return the inverse of this RigidTransform.
                    The inverse is a new RigidTransform that undoes the translation, rotation, and scale.
                )pbdoc",
            nb::rv_policy::move)
        .def("compose",&RT::compose,"other"_a,
            sig("def compose(self, other: RigidTransform) -> RigidTransform"),
                R"pbdoc(
                    Compose this RigidTransform with another one.
                    Returns a new RigidTransform that is the result of applying `other` after this one.
    
                    Parameters
                    ----------
                    other : RigidTransform
                        The other RigidTransform to compose with.
                )pbdoc",
            nb::rv_policy::move)

    // --------------------------------------------------------- matrix access
        .def("to_matrix",
             [](const RT& self){
                 PyObject* py = make_mat4();
                 Eigen::Map<ColM4d>(static_cast<double*>(
                     PyArray_DATA(reinterpret_cast<PyArrayObject*>(py)))) = self.to_matrix();
                 return nb::steal<nb::object>(py);
             },
             sig("def to_matrix(self) -> numpy.ndarray"),
             R"pbdoc(
                to_matrix() -> ndarray
                Return the 4x4 homogeneous matrix representation of this RigidTransform.
                The bottom row is always [0, 0, 0, 1].
                The matrix is in column-major order.
                )pbdoc",
             nb::rv_policy::move)

        .def("to_matrix_transpose",
             [](const RT& self){
                 PyObject* py = make_mat4();
                 Eigen::Map<ColM4d>(static_cast<double*>(
                     PyArray_DATA(reinterpret_cast<PyArrayObject*>(py)))) = self.to_matrix_transpose();
                 return nb::steal<nb::object>(py);
             },
             sig("def to_matrix_transpose(self) -> numpy.ndarray"),
             R"pbdoc(
                to_matrix_transpose() -> ndarray
                Return the transposed 4x4 homogeneous matrix representation of this RigidTransform.
                The bottom row is always [0, 0, 0, 1].
                The matrix is in column-major order.
                )pbdoc",
             nb::rv_policy::move)

    // ------------------------------------------------------- quaternion/euler
        .def("as_quaternion",
             [](const RT& self){
                 PyObject* py = make_vec4();
                 const auto q = self.as_quaternion();
                 double* d = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(py)));
                 d[0]=q.x(); d[1]=q.y(); d[2]=q.z(); d[3]=q.w();
                 return nb::steal<nb::object>(py);
             }, 
             sig("def as_quaternion(self) -> numpy.ndarray"),
            R"pbdoc(
                as_quaternion() -> ndarray
                Return the quaternion representation of this RigidTransform as a NumPy (4,) array.
                The order is [x, y, z, w].
            )pbdoc",
             nb::rv_policy::move)

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
                    ang *= (180.0 / kPI);

                /* pack into a (3,) float64 ndarray (same helper you already have) */
                PyObject *py = make_vec3();
                std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(py)),
                            ang.data(), 3*sizeof(double));
                return nb::steal<nb::object>(py);
            },
            "order"_a, nb::arg("degrees") = true,
            sig("def as_euler(self, order: str, degrees: bool = True) -> numpy.ndarray"),
            R"pbdoc(
                as_euler(order: str, degrees: bool = True) -> numpy.ndarray
                Return the Euler angles representation of this RigidTransform.
                The angles are returned as a NumPy (3,) array in the specified order.

                Parameters
                ----------
                order : str
                    The order of rotations, e.g. "xyz", "xzy", "yxz", etc.
                degrees : bool, optional
                    If True, returns angles in degrees (default is True).
            )pbdoc",
            nb::rv_policy::move)

    // ------------------------------------------------------- dunder helpers
        .def("__matmul__", nb::overload_cast<const RT&>(&RT::operator* ,nb::const_), nb::is_operator()
            ,sig("def __matmul__(self, other: RigidTransform) -> RigidTransform"),
            R"pbdoc(
                Matrix multiplication with another RigidTransform.
                Returns a new RigidTransform that is the result of applying `other` after this one.
            )pbdoc",
            nb::rv_policy::move)
        .def("__mul__",   nb::overload_cast<const RT&>(&RT::operator* ,nb::const_), nb::is_operator()
            ,sig("def __mul__(self, other: RigidTransform) -> RigidTransform"),
            R"pbdoc(
                Multiply this RigidTransform with another one.
                Returns a new RigidTransform that is the result of applying `other` after this one.
            )pbdoc",
            nb::rv_policy::move)
        .def("__mul__",   nb::overload_cast<double    >(&RT::operator* ,nb::const_), nb::is_operator(),
            sig("def __mul__(self, scale: float) -> RigidTransform"),
            R"pbdoc(
                Scale this RigidTransform by a uniform factor.
                Returns a new RigidTransform with the scale applied.
            )pbdoc",
            nb::rv_policy::move)
        .def("__truediv__", nb::overload_cast<double>(&RT::operator/ ,nb::const_), nb::is_operator(),
            sig("def __truediv__(self, scale: float) -> RigidTransform"),
            R"pbdoc(
                Divide this RigidTransform by a uniform scale factor.
                Returns a new RigidTransform with the scale applied.
            )pbdoc",
            nb::rv_policy::move)
        .def("__eq__",&RT::operator==,
            sig("def __eq__(self, other: RigidTransform) -> bool"),
            R"pbdoc(
                Check if this RigidTransform is equal to another one.
                Returns True if the translation, rotation, and scale are all equal.
            )pbdoc")
        .def("__ne__",&RT::operator!=,
            sig("def __ne__(self, other: RigidTransform) -> bool"),
            R"pbdoc(
                Check if this RigidTransform is not equal to another one.
                Returns True if any of the translation, rotation, or scale differ.
            )pbdoc")

        .def("__repr__",[](const RT& r){
            std::ostringstream s;
            auto t=r.translation(); auto R=r.rotation();
            s<<"RigidTransform(translation=[" << t.x() << ", " << t.y() << ", " << t.z() << "], "
             <<"rotation=[["<<R(0,0)<<","<<R(0,1)<<","<<R(0,2)<<"],["
                       <<R(1,0)<<","<<R(1,1)<<","<<R(1,2)<<"],["
                       <<R(2,0)<<","<<R(2,1)<<","<<R(2,2)<<"]], scale="<<r.scale()<<")";
            return s.str();
            },
            sig("def __repr__(self) -> str"),
            R"pbdoc(
                Return a string representation of this RigidTransform.
                The format is:
                RigidTransform(translation=[x, y, z], rotation=[[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]], scale=s)
            )pbdoc")

        .def("__str__",[](const RT& r){
            std::ostringstream s;
            auto t=r.translation(); auto R=r.rotation();
            s<<"RigidTransform(translation=[" << t.x() << ", " << t.y() << ", " << t.z() << "], "
             <<"rotation=[["<<R(0,0)<<","<<R(0,1)<<","<<R(0,2)<<"],["
                       <<R(1,0)<<","<<R(1,1)<<","<<R(1,2)<<"],["
                       <<R(2,0)<<","<<R(2,1)<<","<<R(2,2)<<"]], scale="<<r.scale()<<")";
            return s.str();
            },
            sig("def __str__(self) -> str"),
            R"pbdoc(
                Return a string representation of this RigidTransform.
                The format is:
                RigidTransform(translation=[x, y, z], rotation=[[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]], scale=s)
            )pbdoc");

    // 1) Re-declare the enum in your binding unit (it should match your C++ enum)
    nb::enum_<Direction>(m, "Direction",
        R"pbdoc(
            Semantic world-space directions.
        )pbdoc")
        .value("Forward",  Direction::Forward)
        .value("Backward", Direction::Backward)
        .value("Left",     Direction::Left)
        .value("Right",    Direction::Right)
        .value("Up",       Direction::Up)
        .value("Down",     Direction::Down);

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

            /* ---- 2. pre-compute lookup tables ----------------------- */
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
            "x"_a, "y"_a, "z"_a,
            sig("def __init__(self, x: Direction, y: Direction, z: Direction) -> None"),
            R"pbdoc(
                FrameConvention(x: Direction, y: Direction, z: Direction)

                Create a frame convention mapping local X/Y/Z → semantic directions.

                Parameters
                ----------
                x : Direction
                    The semantic direction for the local X axis.
                y : Direction
                    The semantic direction for the local Y axis.
                z : Direction
                    The semantic direction for the local Z axis.
            )pbdoc"
            )

        /* each accessor: grab Eigen vec → wrap into fresh ndarray */
        .def("forward",
            [&](const PyConvention &self, const RigidTransform &rt)
            { return vec_as_ndarray(self.forward(rt)); },
            "rt"_a, 
            sig("def forward(self, rt: RigidTransform) -> numpy.ndarray"),
            R"pbdoc(
                Get the forward direction vector from the RigidTransform.
                Returns a NumPy (3,) array representing the forward direction.
            )pbdoc",
            nb::rv_policy::move)

        .def("backward",
            [&](const PyConvention &self, const RigidTransform &rt)
            { return vec_as_ndarray(self.backward(rt)); },
            "rt"_a,
            sig("def backward(self, rt: RigidTransform) -> numpy.ndarray"),
            R"pbdoc(
                Get the backward direction vector from the RigidTransform.
                Returns a NumPy (3,) array representing the backward direction.
            )pbdoc",
            nb::rv_policy::move)

        .def("left",
            [&](const PyConvention &self, const RigidTransform &rt)
            { return vec_as_ndarray(self.left(rt)); },
            "rt"_a,
            sig("def left(self, rt: RigidTransform) -> numpy.ndarray"),
            R"pbdoc(
                Get the left direction vector from the RigidTransform.
                Returns a NumPy (3,) array representing the left direction.
            )pbdoc",
            nb::rv_policy::move)

        .def("right",
            [&](const PyConvention &self, const RigidTransform &rt)
            { return vec_as_ndarray(self.right(rt)); },
            "rt"_a, 
            sig("def right(self, rt: RigidTransform) -> numpy.ndarray"),
            R"pbdoc(
                Get the right direction vector from the RigidTransform.
                Returns a NumPy (3,) array representing the right direction.
            )pbdoc",
            nb::rv_policy::move)

        .def("up",
            [&](const PyConvention &self, const RigidTransform &rt)
            { return vec_as_ndarray(self.up(rt)); },
            "rt"_a,
            sig("def up(self, rt: RigidTransform) -> numpy.ndarray"),
            R"pbdoc(
                Get the up direction vector from the RigidTransform.
                Returns a NumPy (3,) array representing the up direction.
            )pbdoc",
            nb::rv_policy::move)

        .def("down",
            [&](const PyConvention &self, const RigidTransform &rt)
            { return vec_as_ndarray(self.down(rt)); },
            "rt"_a, 
            sig("def down(self, rt: RigidTransform) -> numpy.ndarray"),
            R"pbdoc(
                Get the down direction vector from the RigidTransform.
                Returns a NumPy (3,) array representing the down direction.
            )pbdoc",
            nb::rv_policy::move)

        // Helper lambdas to convert stored convention back to the original enum values.
        .def("__repr__",
            [](const PyConvention &self) {
                auto get_direction_for_col = [&self](int col) -> Direction {
                    for (int i = 0; i < 6; ++i) {
                        if (self.col_[i] == col && self.sign_[i] > 0)
                            return static_cast<Direction>(i);
                    }
                    return Direction::Forward; // fallback
                };
                auto direction_to_string = [](Direction d) -> std::string {
                    switch (d) {
                        case Direction::Forward:  return "Forward";
                        case Direction::Backward: return "Backward";
                        case Direction::Left:     return "Left";
                        case Direction::Right:    return "Right";
                        case Direction::Up:       return "Up";
                        case Direction::Down:     return "Down";
                    }
                    return "Unknown";
                };
                Direction xdir = get_direction_for_col(0);
                Direction ydir = get_direction_for_col(1);
                Direction zdir = get_direction_for_col(2);

                std::ostringstream s;
                s << "FrameConvention(x=" << direction_to_string(xdir)
                  << ", y=" << direction_to_string(ydir)
                  << ", z=" << direction_to_string(zdir) << ")";
                return s.str();
            },
            sig("def __repr__(self) -> str"),
            R"pbdoc(
                Return a string representation of this FrameConvention.
                The format is: FrameConvention(x=..., y=..., z=...)
            )pbdoc")
        
        .def("__str__",
            [](const PyConvention &self) {
                auto get_direction_for_col = [&self](int col) -> Direction {
                    for (int i = 0; i < 6; ++i) {
                        if (self.col_[i] == col && self.sign_[i] > 0)
                            return static_cast<Direction>(i);
                    }
                    return Direction::Forward; // fallback
                };
                auto direction_to_string = [](Direction d) -> std::string {
                    switch (d) {
                        case Direction::Forward:  return "Forward";
                        case Direction::Backward: return "Backward";
                        case Direction::Left:     return "Left";
                        case Direction::Right:    return "Right";
                        case Direction::Up:       return "Up";
                        case Direction::Down:     return "Down";
                    }
                    return "Unknown";
                };
                Direction xdir = get_direction_for_col(0);
                Direction ydir = get_direction_for_col(1);
                Direction zdir = get_direction_for_col(2);

                std::ostringstream s;
                s << "FrameConvention(x=" << direction_to_string(xdir)
                  << ", y=" << direction_to_string(ydir)
                  << ", z=" << direction_to_string(zdir) << ")";
                return s.str();
            },
            sig("def __str__(self) -> str"),
            R"pbdoc(
                Return a string representation of this FrameConvention.
                The format is: FrameConvention(x=..., y=..., z=...)
            )pbdoc");

    }