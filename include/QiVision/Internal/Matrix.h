#pragma once

/**
 * @file Matrix.h
 * @brief Small matrix operations library for QiVision
 *
 * This module provides:
 * - Fixed-size vectors: Vec<2>, Vec<3>, Vec<4>
 * - Fixed-size matrices: Mat<2,2>, Mat<3,3>, Mat<4,4>, Mat<2,3>, Mat<3,4>
 * - Dynamic-size vectors: VecX
 * - Dynamic-size matrices: MatX
 * - Matrix decomposition result structures (for Solver.h)
 * - Special matrix factory functions (rotation, translation, etc.)
 *
 * Used by:
 * - Solver.h (linear equation solving)
 * - Eigen.h (eigenvalue decomposition)
 * - Fitting.h (geometric fitting)
 * - Homography.h (perspective transforms)
 * - Calib module (camera calibration)
 *
 * Design principles:
 * - Small matrices (<=4x4) use stack memory for performance
 * - Dynamic matrices use aligned heap memory
 * - Row-major storage
 * - Double precision only
 */

#include <QiVision/Core/Constants.h>
#include <QiVision/Core/Types.h>
#include <QiVision/Platform/Memory.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Qi::Vision {
// Forward declaration
class QMatrix;
}

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Tolerance for matrix singularity detection
constexpr double MATRIX_SINGULAR_THRESHOLD = 1e-10;

/// Tolerance for matrix element comparison
constexpr double MATRIX_EPSILON = 1e-12;

/// Stack threshold for VecX (number of elements)
constexpr int VECX_STACK_SIZE = 16;

// =============================================================================
// Forward Declarations
// =============================================================================

template<int N> class Vec;
template<int M, int N> class Mat;
class VecX;
class MatX;

// =============================================================================
// Fixed-Size Vector: Vec<N>
// =============================================================================

/**
 * @brief Fixed-size vector template
 * @tparam N Vector dimension (2, 3, or 4)
 *
 * Stack-allocated, compile-time optimized.
 */
template<int N>
class Vec {
    static_assert(N >= 1 && N <= 16, "Vec dimension must be between 1 and 16");

public:
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Default constructor (zero vector)
    Vec() {
        std::fill(data_, data_ + N, 0.0);
    }

    /// Construct from initializer list
    Vec(std::initializer_list<double> init) {
        std::fill(data_, data_ + N, 0.0);
        int i = 0;
        for (auto val : init) {
            if (i >= N) break;
            data_[i++] = val;
        }
    }

    /// Construct from array
    explicit Vec(const double* data) {
        std::copy(data, data + N, data_);
    }

    // =========================================================================
    // Element Access
    // =========================================================================

    double& operator[](int i) { return data_[i]; }
    const double& operator[](int i) const { return data_[i]; }

    double& operator()(int i) { return data_[i]; }
    const double& operator()(int i) const { return data_[i]; }

    // =========================================================================
    // Vector Operations
    // =========================================================================

    Vec operator+(const Vec& v) const {
        Vec result;
        for (int i = 0; i < N; ++i) result.data_[i] = data_[i] + v.data_[i];
        return result;
    }

    Vec operator-(const Vec& v) const {
        Vec result;
        for (int i = 0; i < N; ++i) result.data_[i] = data_[i] - v.data_[i];
        return result;
    }

    Vec operator*(double s) const {
        Vec result;
        for (int i = 0; i < N; ++i) result.data_[i] = data_[i] * s;
        return result;
    }

    Vec operator/(double s) const {
        Vec result;
        double inv = 1.0 / s;
        for (int i = 0; i < N; ++i) result.data_[i] = data_[i] * inv;
        return result;
    }

    Vec& operator+=(const Vec& v) {
        for (int i = 0; i < N; ++i) data_[i] += v.data_[i];
        return *this;
    }

    Vec& operator-=(const Vec& v) {
        for (int i = 0; i < N; ++i) data_[i] -= v.data_[i];
        return *this;
    }

    Vec& operator*=(double s) {
        for (int i = 0; i < N; ++i) data_[i] *= s;
        return *this;
    }

    Vec& operator/=(double s) {
        double inv = 1.0 / s;
        for (int i = 0; i < N; ++i) data_[i] *= inv;
        return *this;
    }

    Vec operator-() const {
        Vec result;
        for (int i = 0; i < N; ++i) result.data_[i] = -data_[i];
        return result;
    }

    // =========================================================================
    // Vector Properties
    // =========================================================================

    /// Dot product
    double Dot(const Vec& v) const {
        double sum = 0.0;
        for (int i = 0; i < N; ++i) sum += data_[i] * v.data_[i];
        return sum;
    }

    /// L2 norm (Euclidean length)
    double Norm() const {
        return std::sqrt(NormSquared());
    }

    /// L2 norm squared
    double NormSquared() const {
        double sum = 0.0;
        for (int i = 0; i < N; ++i) sum += data_[i] * data_[i];
        return sum;
    }

    /// L1 norm (Manhattan distance)
    double NormL1() const {
        double sum = 0.0;
        for (int i = 0; i < N; ++i) sum += std::abs(data_[i]);
        return sum;
    }

    /// Infinity norm (max absolute value)
    double NormInf() const {
        double maxVal = 0.0;
        for (int i = 0; i < N; ++i) maxVal = std::max(maxVal, std::abs(data_[i]));
        return maxVal;
    }

    /// Return normalized vector (unit length)
    Vec Normalized() const {
        double n = Norm();
        if (n < MATRIX_EPSILON) return Zero();
        return *this / n;
    }

    /// Normalize in place
    void Normalize() {
        double n = Norm();
        if (n < MATRIX_EPSILON) {
            std::fill(data_, data_ + N, 0.0);
            return;
        }
        *this /= n;
    }

    // =========================================================================
    // Data Access
    // =========================================================================

    double* Data() { return data_; }
    const double* Data() const { return data_; }
    static constexpr int Size() { return N; }

    // =========================================================================
    // Factory Methods
    // =========================================================================

    static Vec Zero() {
        Vec result;
        return result;  // Already zero-initialized
    }

    static Vec Ones() {
        Vec result;
        std::fill(result.data_, result.data_ + N, 1.0);
        return result;
    }

    /// Unit vector along specified axis
    static Vec Unit(int axis) {
        Vec result;
        if (axis >= 0 && axis < N) result.data_[axis] = 1.0;
        return result;
    }

private:
    double data_[N];
};

// =============================================================================
// Vec<2> Specialization
// =============================================================================

template<>
inline Vec<2>::Vec(std::initializer_list<double> init) {
    data_[0] = data_[1] = 0.0;
    auto it = init.begin();
    if (it != init.end()) { data_[0] = *it++; }
    if (it != init.end()) { data_[1] = *it; }
}

// Additional Vec2 constructor
inline Vec<2> MakeVec2(double x, double y) {
    return Vec<2>{x, y};
}

// =============================================================================
// Vec<3> Specialization with Cross Product
// =============================================================================

template<>
inline Vec<3>::Vec(std::initializer_list<double> init) {
    data_[0] = data_[1] = data_[2] = 0.0;
    auto it = init.begin();
    if (it != init.end()) { data_[0] = *it++; }
    if (it != init.end()) { data_[1] = *it++; }
    if (it != init.end()) { data_[2] = *it; }
}

/// Cross product (only for Vec<3>)
inline Vec<3> Cross(const Vec<3>& a, const Vec<3>& b) {
    return Vec<3>{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

// Additional Vec3 constructor
inline Vec<3> MakeVec3(double x, double y, double z) {
    return Vec<3>{x, y, z};
}

// =============================================================================
// Vec<4> Specialization
// =============================================================================

template<>
inline Vec<4>::Vec(std::initializer_list<double> init) {
    data_[0] = data_[1] = data_[2] = data_[3] = 0.0;
    auto it = init.begin();
    if (it != init.end()) { data_[0] = *it++; }
    if (it != init.end()) { data_[1] = *it++; }
    if (it != init.end()) { data_[2] = *it++; }
    if (it != init.end()) { data_[3] = *it; }
}

// Additional Vec4 constructor
inline Vec<4> MakeVec4(double x, double y, double z, double w) {
    return Vec<4>{x, y, z, w};
}

// =============================================================================
// Type Aliases
// =============================================================================

using Vec2 = Vec<2>;
using Vec3 = Vec<3>;
using Vec4 = Vec<4>;

// =============================================================================
// Point2d/Point3d Conversion
// =============================================================================

inline Vec2 ToVec(const Point2d& p) {
    return Vec2{p.x, p.y};
}

inline Vec3 ToVec(const Point3d& p) {
    return Vec3{p.x, p.y, p.z};
}

inline Point2d ToPoint2d(const Vec2& v) {
    return Point2d(v[0], v[1]);
}

inline Point3d ToPoint3d(const Vec3& v) {
    return Point3d(v[0], v[1], v[2]);
}

// Scalar * Vec (commutative)
template<int N>
inline Vec<N> operator*(double s, const Vec<N>& v) {
    return v * s;
}

// =============================================================================
// Fixed-Size Matrix: Mat<M,N>
// =============================================================================

/**
 * @brief Fixed-size matrix template
 * @tparam M Number of rows
 * @tparam N Number of columns
 *
 * Row-major storage, stack-allocated.
 */
template<int M, int N>
class Mat {
    static_assert(M >= 1 && M <= 16, "Mat rows must be between 1 and 16");
    static_assert(N >= 1 && N <= 16, "Mat cols must be between 1 and 16");

public:
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Default constructor (zero matrix)
    Mat() {
        std::fill(data_, data_ + M * N, 0.0);
    }

    /// Construct from initializer list (row-major order)
    Mat(std::initializer_list<double> init) {
        std::fill(data_, data_ + M * N, 0.0);
        int i = 0;
        for (auto val : init) {
            if (i >= M * N) break;
            data_[i++] = val;
        }
    }

    /// Construct from array (row-major)
    explicit Mat(const double* data) {
        std::copy(data, data + M * N, data_);
    }

    // =========================================================================
    // Element Access
    // =========================================================================

    double& operator()(int row, int col) { return data_[row * N + col]; }
    const double& operator()(int row, int col) const { return data_[row * N + col]; }

    // =========================================================================
    // Row/Column Access
    // =========================================================================

    Vec<N> Row(int i) const {
        Vec<N> result;
        for (int j = 0; j < N; ++j) result[j] = data_[i * N + j];
        return result;
    }

    Vec<M> Col(int j) const {
        Vec<M> result;
        for (int i = 0; i < M; ++i) result[i] = data_[i * N + j];
        return result;
    }

    void SetRow(int i, const Vec<N>& row) {
        for (int j = 0; j < N; ++j) data_[i * N + j] = row[j];
    }

    void SetCol(int j, const Vec<M>& col) {
        for (int i = 0; i < M; ++i) data_[i * N + j] = col[i];
    }

    // =========================================================================
    // Matrix Arithmetic
    // =========================================================================

    Mat operator+(const Mat& m) const {
        Mat result;
        for (int i = 0; i < M * N; ++i) result.data_[i] = data_[i] + m.data_[i];
        return result;
    }

    Mat operator-(const Mat& m) const {
        Mat result;
        for (int i = 0; i < M * N; ++i) result.data_[i] = data_[i] - m.data_[i];
        return result;
    }

    Mat operator*(double s) const {
        Mat result;
        for (int i = 0; i < M * N; ++i) result.data_[i] = data_[i] * s;
        return result;
    }

    Mat operator/(double s) const {
        Mat result;
        double inv = 1.0 / s;
        for (int i = 0; i < M * N; ++i) result.data_[i] = data_[i] * inv;
        return result;
    }

    Mat& operator+=(const Mat& m) {
        for (int i = 0; i < M * N; ++i) data_[i] += m.data_[i];
        return *this;
    }

    Mat& operator-=(const Mat& m) {
        for (int i = 0; i < M * N; ++i) data_[i] -= m.data_[i];
        return *this;
    }

    Mat& operator*=(double s) {
        for (int i = 0; i < M * N; ++i) data_[i] *= s;
        return *this;
    }

    Mat operator-() const {
        Mat result;
        for (int i = 0; i < M * N; ++i) result.data_[i] = -data_[i];
        return result;
    }

    // =========================================================================
    // Matrix Multiplication
    // =========================================================================

    /// Matrix-matrix multiplication: this (M x N) * other (N x P) = result (M x P)
    template<int P>
    Mat<M, P> operator*(const Mat<N, P>& other) const {
        Mat<M, P> result;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < P; ++j) {
                double sum = 0.0;
                for (int k = 0; k < N; ++k) {
                    sum += data_[i * N + k] * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    /// Matrix-vector multiplication: this (M x N) * v (N) = result (M)
    Vec<M> operator*(const Vec<N>& v) const {
        Vec<M> result;
        for (int i = 0; i < M; ++i) {
            double sum = 0.0;
            for (int j = 0; j < N; ++j) {
                sum += data_[i * N + j] * v[j];
            }
            result[i] = sum;
        }
        return result;
    }

    // =========================================================================
    // Transpose
    // =========================================================================

    Mat<N, M> Transpose() const {
        Mat<N, M> result;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                result(j, i) = data_[i * N + j];
            }
        }
        return result;
    }

    // =========================================================================
    // Norms
    // =========================================================================

    /// Frobenius norm
    double NormFrobenius() const {
        double sum = 0.0;
        for (int i = 0; i < M * N; ++i) sum += data_[i] * data_[i];
        return std::sqrt(sum);
    }

    /// L1 norm (max column sum of absolute values)
    double NormL1() const {
        double maxSum = 0.0;
        for (int j = 0; j < N; ++j) {
            double colSum = 0.0;
            for (int i = 0; i < M; ++i) {
                colSum += std::abs(data_[i * N + j]);
            }
            maxSum = std::max(maxSum, colSum);
        }
        return maxSum;
    }

    /// Infinity norm (max row sum of absolute values)
    double NormInf() const {
        double maxSum = 0.0;
        for (int i = 0; i < M; ++i) {
            double rowSum = 0.0;
            for (int j = 0; j < N; ++j) {
                rowSum += std::abs(data_[i * N + j]);
            }
            maxSum = std::max(maxSum, rowSum);
        }
        return maxSum;
    }

    // =========================================================================
    // Data Access
    // =========================================================================

    double* Data() { return data_; }
    const double* Data() const { return data_; }
    static constexpr int Rows() { return M; }
    static constexpr int Cols() { return N; }

    // =========================================================================
    // Factory Methods
    // =========================================================================

    static Mat Zero() {
        return Mat();  // Already zero-initialized
    }

    /// Identity matrix (only for square matrices)
    template<int M2 = M, int N2 = N>
    static typename std::enable_if<M2 == N2, Mat>::type Identity() {
        Mat result;
        for (int i = 0; i < M; ++i) result(i, i) = 1.0;
        return result;
    }

    /// Diagonal matrix from vector
    template<int M2 = M, int N2 = N>
    static typename std::enable_if<M2 == N2, Mat>::type Diagonal(const Vec<M>& diag) {
        Mat result;
        for (int i = 0; i < M; ++i) result(i, i) = diag[i];
        return result;
    }

    // =========================================================================
    // Square Matrix Operations (only for M == N)
    // =========================================================================

    /// Trace (sum of diagonal elements)
    template<int M2 = M, int N2 = N>
    typename std::enable_if<M2 == N2, double>::type Trace() const {
        double sum = 0.0;
        for (int i = 0; i < M; ++i) sum += data_[i * N + i];
        return sum;
    }

    /// Determinant - implemented via specialization for 2x2, 3x3, 4x4
    template<int M2 = M, int N2 = N>
    typename std::enable_if<M2 == N2, double>::type Determinant() const;

    /// Inverse - implemented via specialization for 2x2, 3x3, 4x4
    template<int M2 = M, int N2 = N>
    typename std::enable_if<M2 == N2, Mat>::type Inverse() const;

    /// Check if invertible
    template<int M2 = M, int N2 = N>
    typename std::enable_if<M2 == N2, bool>::type IsInvertible(double eps = MATRIX_SINGULAR_THRESHOLD) const {
        return std::abs(Determinant()) > eps;
    }

private:
    double data_[M * N];  // Row-major storage
};

// Scalar * Mat (commutative)
template<int M, int N>
inline Mat<M, N> operator*(double s, const Mat<M, N>& m) {
    return m * s;
}

// =============================================================================
// Mat<2,2> Specializations
// =============================================================================

template<>
template<>
inline double Mat<2, 2>::Determinant<2, 2>() const {
    return data_[0] * data_[3] - data_[1] * data_[2];
}

template<>
template<>
inline Mat<2, 2> Mat<2, 2>::Inverse<2, 2>() const {
    double det = Determinant();
    if (std::abs(det) < MATRIX_SINGULAR_THRESHOLD) {
        return Mat<2, 2>::Zero();
    }
    double invDet = 1.0 / det;
    Mat<2, 2> result;
    result(0, 0) =  data_[3] * invDet;
    result(0, 1) = -data_[1] * invDet;
    result(1, 0) = -data_[2] * invDet;
    result(1, 1) =  data_[0] * invDet;
    return result;
}

// =============================================================================
// Mat<3,3> Specializations
// =============================================================================

template<>
template<>
inline double Mat<3, 3>::Determinant<3, 3>() const {
    // Expansion by first row
    double a = data_[0], b = data_[1], c = data_[2];
    double d = data_[3], e = data_[4], f = data_[5];
    double g = data_[6], h = data_[7], i = data_[8];
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

template<>
template<>
inline Mat<3, 3> Mat<3, 3>::Inverse<3, 3>() const {
    double det = Determinant();
    if (std::abs(det) < MATRIX_SINGULAR_THRESHOLD) {
        return Mat<3, 3>::Zero();
    }
    double invDet = 1.0 / det;

    // Elements of original matrix
    double a = data_[0], b = data_[1], c = data_[2];
    double d = data_[3], e = data_[4], f = data_[5];
    double g = data_[6], h = data_[7], i = data_[8];

    // Cofactor matrix (transposed = adjugate)
    Mat<3, 3> result;
    result(0, 0) = (e * i - f * h) * invDet;
    result(0, 1) = (c * h - b * i) * invDet;
    result(0, 2) = (b * f - c * e) * invDet;
    result(1, 0) = (f * g - d * i) * invDet;
    result(1, 1) = (a * i - c * g) * invDet;
    result(1, 2) = (c * d - a * f) * invDet;
    result(2, 0) = (d * h - e * g) * invDet;
    result(2, 1) = (b * g - a * h) * invDet;
    result(2, 2) = (a * e - b * d) * invDet;

    return result;
}

// =============================================================================
// Mat<4,4> Specializations
// =============================================================================

template<>
template<>
inline double Mat<4, 4>::Determinant<4, 4>() const {
    // Use Laplace expansion along first row
    const double* m = data_;

    double s0 = m[0] * m[5] - m[1] * m[4];
    double s1 = m[0] * m[6] - m[2] * m[4];
    double s2 = m[0] * m[7] - m[3] * m[4];
    double s3 = m[1] * m[6] - m[2] * m[5];
    double s4 = m[1] * m[7] - m[3] * m[5];
    double s5 = m[2] * m[7] - m[3] * m[6];

    double c5 = m[10] * m[15] - m[11] * m[14];
    double c4 = m[9]  * m[15] - m[11] * m[13];
    double c3 = m[9]  * m[14] - m[10] * m[13];
    double c2 = m[8]  * m[15] - m[11] * m[12];
    double c1 = m[8]  * m[14] - m[10] * m[12];
    double c0 = m[8]  * m[13] - m[9]  * m[12];

    return s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
}

template<>
template<>
inline Mat<4, 4> Mat<4, 4>::Inverse<4, 4>() const {
    const double* m = data_;

    double s0 = m[0] * m[5] - m[1] * m[4];
    double s1 = m[0] * m[6] - m[2] * m[4];
    double s2 = m[0] * m[7] - m[3] * m[4];
    double s3 = m[1] * m[6] - m[2] * m[5];
    double s4 = m[1] * m[7] - m[3] * m[5];
    double s5 = m[2] * m[7] - m[3] * m[6];

    double c5 = m[10] * m[15] - m[11] * m[14];
    double c4 = m[9]  * m[15] - m[11] * m[13];
    double c3 = m[9]  * m[14] - m[10] * m[13];
    double c2 = m[8]  * m[15] - m[11] * m[12];
    double c1 = m[8]  * m[14] - m[10] * m[12];
    double c0 = m[8]  * m[13] - m[9]  * m[12];

    double det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;

    if (std::abs(det) < MATRIX_SINGULAR_THRESHOLD) {
        return Mat<4, 4>::Zero();
    }

    double invDet = 1.0 / det;

    Mat<4, 4> result;

    result(0, 0) = ( m[5] * c5 - m[6] * c4 + m[7] * c3) * invDet;
    result(0, 1) = (-m[1] * c5 + m[2] * c4 - m[3] * c3) * invDet;
    result(0, 2) = ( m[13] * s5 - m[14] * s4 + m[15] * s3) * invDet;
    result(0, 3) = (-m[9] * s5 + m[10] * s4 - m[11] * s3) * invDet;

    result(1, 0) = (-m[4] * c5 + m[6] * c2 - m[7] * c1) * invDet;
    result(1, 1) = ( m[0] * c5 - m[2] * c2 + m[3] * c1) * invDet;
    result(1, 2) = (-m[12] * s5 + m[14] * s2 - m[15] * s1) * invDet;
    result(1, 3) = ( m[8] * s5 - m[10] * s2 + m[11] * s1) * invDet;

    result(2, 0) = ( m[4] * c4 - m[5] * c2 + m[7] * c0) * invDet;
    result(2, 1) = (-m[0] * c4 + m[1] * c2 - m[3] * c0) * invDet;
    result(2, 2) = ( m[12] * s4 - m[13] * s2 + m[15] * s0) * invDet;
    result(2, 3) = (-m[8] * s4 + m[9] * s2 - m[11] * s0) * invDet;

    result(3, 0) = (-m[4] * c3 + m[5] * c1 - m[6] * c0) * invDet;
    result(3, 1) = ( m[0] * c3 - m[1] * c1 + m[2] * c0) * invDet;
    result(3, 2) = (-m[12] * s3 + m[13] * s1 - m[14] * s0) * invDet;
    result(3, 3) = ( m[8] * s3 - m[9] * s1 + m[10] * s0) * invDet;

    return result;
}

// =============================================================================
// Matrix Type Aliases
// =============================================================================

using Mat22 = Mat<2, 2>;
using Mat33 = Mat<3, 3>;
using Mat44 = Mat<4, 4>;
using Mat23 = Mat<2, 3>;  // 2D affine (for 2D points)
using Mat34 = Mat<3, 4>;  // 3D projection (K * [R|t])

// =============================================================================
// Dynamic-Size Vector: VecX
// =============================================================================

/**
 * @brief Dynamic-size vector
 *
 * Uses stack memory for small vectors (<=16 elements),
 * heap memory for larger vectors.
 */
class VecX {
public:
    // =========================================================================
    // Constructors
    // =========================================================================

    VecX() : size_(0), heapData_(nullptr) {
        std::fill(stackData_, stackData_ + VECX_STACK_SIZE, 0.0);
    }

    explicit VecX(int size) : size_(size), heapData_(nullptr) {
        if (size_ <= 0) {
            size_ = 0;
            return;
        }
        if (UseHeap()) {
            heapData_ = static_cast<double*>(Platform::AlignedAlloc(size_ * sizeof(double)));
            std::fill(heapData_, heapData_ + size_, 0.0);
        } else {
            std::fill(stackData_, stackData_ + size_, 0.0);
        }
    }

    VecX(int size, double value) : VecX(size) {
        double* ptr = Data();
        std::fill(ptr, ptr + size_, value);
    }

    VecX(std::initializer_list<double> init) : VecX(static_cast<int>(init.size())) {
        double* ptr = Data();
        int i = 0;
        for (auto val : init) {
            ptr[i++] = val;
        }
    }

    // Copy constructor
    VecX(const VecX& other) : size_(other.size_), heapData_(nullptr) {
        std::fill(stackData_, stackData_ + VECX_STACK_SIZE, 0.0);
        if (UseHeap()) {
            heapData_ = static_cast<double*>(Platform::AlignedAlloc(size_ * sizeof(double)));
            std::copy(other.heapData_, other.heapData_ + size_, heapData_);
        } else {
            std::copy(other.stackData_, other.stackData_ + size_, stackData_);
        }
    }

    // Move constructor
    VecX(VecX&& other) noexcept : size_(other.size_), heapData_(other.heapData_) {
        std::fill(stackData_, stackData_ + VECX_STACK_SIZE, 0.0);
        if (!other.UseHeap()) {
            std::copy(other.stackData_, other.stackData_ + size_, stackData_);
        }
        other.size_ = 0;
        other.heapData_ = nullptr;
    }

    // Copy assignment
    VecX& operator=(const VecX& other) {
        if (this != &other) {
            if (heapData_) {
                Platform::AlignedFree(heapData_);
                heapData_ = nullptr;
            }
            size_ = other.size_;
            if (UseHeap()) {
                heapData_ = static_cast<double*>(Platform::AlignedAlloc(size_ * sizeof(double)));
                std::copy(other.heapData_, other.heapData_ + size_, heapData_);
            } else {
                std::copy(other.stackData_, other.stackData_ + size_, stackData_);
            }
        }
        return *this;
    }

    // Move assignment
    VecX& operator=(VecX&& other) noexcept {
        if (this != &other) {
            if (heapData_) {
                Platform::AlignedFree(heapData_);
            }
            size_ = other.size_;
            heapData_ = other.heapData_;
            if (!other.UseHeap()) {
                std::copy(other.stackData_, other.stackData_ + size_, stackData_);
            }
            other.size_ = 0;
            other.heapData_ = nullptr;
        }
        return *this;
    }

    ~VecX() {
        if (heapData_) {
            Platform::AlignedFree(heapData_);
        }
    }

    // =========================================================================
    // Conversion from Fixed-Size
    // =========================================================================

    template<int N>
    explicit VecX(const Vec<N>& v) : VecX(N) {
        double* ptr = Data();
        for (int i = 0; i < N; ++i) ptr[i] = v[i];
    }

    template<int N>
    Vec<N> ToFixed() const {
        if (size_ != N) {
            throw std::invalid_argument("VecX size does not match Vec<N>");
        }
        Vec<N> result;
        const double* ptr = Data();
        for (int i = 0; i < N; ++i) result[i] = ptr[i];
        return result;
    }

    // =========================================================================
    // Element Access
    // =========================================================================

    double& operator[](int i) { return Data()[i]; }
    const double& operator[](int i) const { return Data()[i]; }

    double& operator()(int i) { return Data()[i]; }
    const double& operator()(int i) const { return Data()[i]; }

    // =========================================================================
    // Vector Operations
    // =========================================================================

    VecX operator+(const VecX& v) const {
        if (size_ != v.size_) throw std::invalid_argument("VecX size mismatch");
        VecX result(size_);
        const double* a = Data();
        const double* b = v.Data();
        double* r = result.Data();
        for (int i = 0; i < size_; ++i) r[i] = a[i] + b[i];
        return result;
    }

    VecX operator-(const VecX& v) const {
        if (size_ != v.size_) throw std::invalid_argument("VecX size mismatch");
        VecX result(size_);
        const double* a = Data();
        const double* b = v.Data();
        double* r = result.Data();
        for (int i = 0; i < size_; ++i) r[i] = a[i] - b[i];
        return result;
    }

    VecX operator*(double s) const {
        VecX result(size_);
        const double* a = Data();
        double* r = result.Data();
        for (int i = 0; i < size_; ++i) r[i] = a[i] * s;
        return result;
    }

    VecX operator/(double s) const {
        VecX result(size_);
        const double* a = Data();
        double* r = result.Data();
        double inv = 1.0 / s;
        for (int i = 0; i < size_; ++i) r[i] = a[i] * inv;
        return result;
    }

    VecX& operator+=(const VecX& v) {
        if (size_ != v.size_) throw std::invalid_argument("VecX size mismatch");
        double* a = Data();
        const double* b = v.Data();
        for (int i = 0; i < size_; ++i) a[i] += b[i];
        return *this;
    }

    VecX& operator-=(const VecX& v) {
        if (size_ != v.size_) throw std::invalid_argument("VecX size mismatch");
        double* a = Data();
        const double* b = v.Data();
        for (int i = 0; i < size_; ++i) a[i] -= b[i];
        return *this;
    }

    VecX& operator*=(double s) {
        double* a = Data();
        for (int i = 0; i < size_; ++i) a[i] *= s;
        return *this;
    }

    VecX operator-() const {
        VecX result(size_);
        const double* a = Data();
        double* r = result.Data();
        for (int i = 0; i < size_; ++i) r[i] = -a[i];
        return result;
    }

    // =========================================================================
    // Vector Properties
    // =========================================================================

    double Dot(const VecX& v) const {
        if (size_ != v.size_) throw std::invalid_argument("VecX size mismatch");
        const double* a = Data();
        const double* b = v.Data();
        double sum = 0.0;
        for (int i = 0; i < size_; ++i) sum += a[i] * b[i];
        return sum;
    }

    double Norm() const { return std::sqrt(NormSquared()); }

    double NormSquared() const {
        const double* a = Data();
        double sum = 0.0;
        for (int i = 0; i < size_; ++i) sum += a[i] * a[i];
        return sum;
    }

    VecX Normalized() const {
        double n = Norm();
        if (n < MATRIX_EPSILON) return Zero(size_);
        return *this / n;
    }

    void Normalize() {
        double n = Norm();
        if (n < MATRIX_EPSILON) {
            SetZero();
            return;
        }
        *this *= (1.0 / n);
    }

    // =========================================================================
    // Size Operations
    // =========================================================================

    int Size() const { return size_; }

    void Resize(int newSize) {
        if (newSize == size_) return;

        VecX temp(newSize);
        int copySize = std::min(size_, newSize);
        const double* src = Data();
        double* dst = temp.Data();
        std::copy(src, src + copySize, dst);

        *this = std::move(temp);
    }

    void SetZero() {
        double* ptr = Data();
        std::fill(ptr, ptr + size_, 0.0);
    }

    void SetOnes() {
        double* ptr = Data();
        std::fill(ptr, ptr + size_, 1.0);
    }

    void SetConstant(double value) {
        double* ptr = Data();
        std::fill(ptr, ptr + size_, value);
    }

    // =========================================================================
    // Data Access
    // =========================================================================

    double* Data() { return UseHeap() ? heapData_ : stackData_; }
    const double* Data() const { return UseHeap() ? heapData_ : stackData_; }

    // =========================================================================
    // Sub-Vector Operations
    // =========================================================================

    VecX Segment(int start, int length) const {
        if (start < 0 || length < 0 || start + length > size_) {
            throw std::out_of_range("VecX segment out of range");
        }
        VecX result(length);
        const double* src = Data() + start;
        double* dst = result.Data();
        std::copy(src, src + length, dst);
        return result;
    }

    void SetSegment(int start, const VecX& segment) {
        if (start < 0 || start + segment.size_ > size_) {
            throw std::out_of_range("VecX segment out of range");
        }
        double* dst = Data() + start;
        const double* src = segment.Data();
        std::copy(src, src + segment.size_, dst);
    }

    // =========================================================================
    // Factory Methods
    // =========================================================================

    static VecX Zero(int size) {
        return VecX(size, 0.0);
    }

    static VecX Ones(int size) {
        return VecX(size, 1.0);
    }

    static VecX LinSpace(double start, double end, int count) {
        if (count < 1) return VecX();
        if (count == 1) return VecX{start};

        VecX result(count);
        double* ptr = result.Data();
        double step = (end - start) / (count - 1);
        for (int i = 0; i < count; ++i) {
            ptr[i] = start + i * step;
        }
        return result;
    }

private:
    bool UseHeap() const { return size_ > VECX_STACK_SIZE; }

    int size_;
    double stackData_[VECX_STACK_SIZE];
    double* heapData_;
};

// Scalar * VecX
inline VecX operator*(double s, const VecX& v) {
    return v * s;
}

// =============================================================================
// Dynamic-Size Matrix: MatX
// =============================================================================

/**
 * @brief Dynamic-size matrix
 *
 * Row-major storage, uses aligned heap memory.
 */
class MatX {
public:
    // =========================================================================
    // Constructors
    // =========================================================================

    MatX() : rows_(0), cols_(0), data_(nullptr) {}

    MatX(int rows, int cols) : rows_(rows), cols_(cols), data_(nullptr) {
        if (rows_ <= 0 || cols_ <= 0) {
            rows_ = cols_ = 0;
            return;
        }
        size_t size = static_cast<size_t>(rows_) * cols_ * sizeof(double);
        data_ = static_cast<double*>(Platform::AlignedAlloc(size));
        std::fill(data_, data_ + rows_ * cols_, 0.0);
    }

    MatX(int rows, int cols, double value) : MatX(rows, cols) {
        if (data_) {
            std::fill(data_, data_ + rows_ * cols_, value);
        }
    }

    // Copy constructor
    MatX(const MatX& other) : rows_(other.rows_), cols_(other.cols_), data_(nullptr) {
        if (rows_ > 0 && cols_ > 0) {
            size_t size = static_cast<size_t>(rows_) * cols_ * sizeof(double);
            data_ = static_cast<double*>(Platform::AlignedAlloc(size));
            std::copy(other.data_, other.data_ + rows_ * cols_, data_);
        }
    }

    // Move constructor
    MatX(MatX&& other) noexcept : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {
        other.rows_ = other.cols_ = 0;
        other.data_ = nullptr;
    }

    // Copy assignment
    MatX& operator=(const MatX& other) {
        if (this != &other) {
            if (data_) Platform::AlignedFree(data_);
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = nullptr;
            if (rows_ > 0 && cols_ > 0) {
                size_t size = static_cast<size_t>(rows_) * cols_ * sizeof(double);
                data_ = static_cast<double*>(Platform::AlignedAlloc(size));
                std::copy(other.data_, other.data_ + rows_ * cols_, data_);
            }
        }
        return *this;
    }

    // Move assignment
    MatX& operator=(MatX&& other) noexcept {
        if (this != &other) {
            if (data_) Platform::AlignedFree(data_);
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = other.data_;
            other.rows_ = other.cols_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }

    ~MatX() {
        if (data_) Platform::AlignedFree(data_);
    }

    // =========================================================================
    // Conversion from Fixed-Size
    // =========================================================================

    template<int M, int N>
    explicit MatX(const Mat<M, N>& m) : MatX(M, N) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                data_[i * cols_ + j] = m(i, j);
            }
        }
    }

    template<int M, int N>
    Mat<M, N> ToFixed() const {
        if (rows_ != M || cols_ != N) {
            throw std::invalid_argument("MatX size does not match Mat<M,N>");
        }
        Mat<M, N> result;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                result(i, j) = data_[i * cols_ + j];
            }
        }
        return result;
    }

    // =========================================================================
    // Element Access
    // =========================================================================

    double& operator()(int row, int col) { return data_[row * cols_ + col]; }
    const double& operator()(int row, int col) const { return data_[row * cols_ + col]; }

    // =========================================================================
    // Row/Column Access
    // =========================================================================

    VecX Row(int i) const {
        VecX result(cols_);
        for (int j = 0; j < cols_; ++j) result[j] = data_[i * cols_ + j];
        return result;
    }

    VecX Col(int j) const {
        VecX result(rows_);
        for (int i = 0; i < rows_; ++i) result[i] = data_[i * cols_ + j];
        return result;
    }

    void SetRow(int i, const VecX& row) {
        if (row.Size() != cols_) throw std::invalid_argument("Row size mismatch");
        for (int j = 0; j < cols_; ++j) data_[i * cols_ + j] = row[j];
    }

    void SetCol(int j, const VecX& col) {
        if (col.Size() != rows_) throw std::invalid_argument("Column size mismatch");
        for (int i = 0; i < rows_; ++i) data_[i * cols_ + j] = col[i];
    }

    // =========================================================================
    // Matrix Arithmetic
    // =========================================================================

    MatX operator+(const MatX& m) const {
        if (rows_ != m.rows_ || cols_ != m.cols_) {
            throw std::invalid_argument("MatX size mismatch");
        }
        MatX result(rows_, cols_);
        for (int i = 0; i < rows_ * cols_; ++i) {
            result.data_[i] = data_[i] + m.data_[i];
        }
        return result;
    }

    MatX operator-(const MatX& m) const {
        if (rows_ != m.rows_ || cols_ != m.cols_) {
            throw std::invalid_argument("MatX size mismatch");
        }
        MatX result(rows_, cols_);
        for (int i = 0; i < rows_ * cols_; ++i) {
            result.data_[i] = data_[i] - m.data_[i];
        }
        return result;
    }

    MatX operator*(double s) const {
        MatX result(rows_, cols_);
        for (int i = 0; i < rows_ * cols_; ++i) {
            result.data_[i] = data_[i] * s;
        }
        return result;
    }

    MatX operator/(double s) const {
        MatX result(rows_, cols_);
        double inv = 1.0 / s;
        for (int i = 0; i < rows_ * cols_; ++i) {
            result.data_[i] = data_[i] * inv;
        }
        return result;
    }

    MatX& operator+=(const MatX& m) {
        if (rows_ != m.rows_ || cols_ != m.cols_) {
            throw std::invalid_argument("MatX size mismatch");
        }
        for (int i = 0; i < rows_ * cols_; ++i) data_[i] += m.data_[i];
        return *this;
    }

    MatX& operator-=(const MatX& m) {
        if (rows_ != m.rows_ || cols_ != m.cols_) {
            throw std::invalid_argument("MatX size mismatch");
        }
        for (int i = 0; i < rows_ * cols_; ++i) data_[i] -= m.data_[i];
        return *this;
    }

    MatX& operator*=(double s) {
        for (int i = 0; i < rows_ * cols_; ++i) data_[i] *= s;
        return *this;
    }

    MatX operator-() const {
        MatX result(rows_, cols_);
        for (int i = 0; i < rows_ * cols_; ++i) {
            result.data_[i] = -data_[i];
        }
        return result;
    }

    // =========================================================================
    // Matrix Multiplication
    // =========================================================================

    MatX operator*(const MatX& m) const {
        if (cols_ != m.rows_) {
            throw std::invalid_argument("MatX dimension mismatch for multiplication");
        }
        MatX result(rows_, m.cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < m.cols_; ++j) {
                double sum = 0.0;
                for (int k = 0; k < cols_; ++k) {
                    sum += data_[i * cols_ + k] * m.data_[k * m.cols_ + j];
                }
                result.data_[i * m.cols_ + j] = sum;
            }
        }
        return result;
    }

    VecX operator*(const VecX& v) const {
        if (cols_ != v.Size()) {
            throw std::invalid_argument("MatX-VecX dimension mismatch");
        }
        VecX result(rows_);
        for (int i = 0; i < rows_; ++i) {
            double sum = 0.0;
            for (int j = 0; j < cols_; ++j) {
                sum += data_[i * cols_ + j] * v[j];
            }
            result[i] = sum;
        }
        return result;
    }

    // =========================================================================
    // Transpose
    // =========================================================================

    MatX Transpose() const {
        MatX result(cols_, rows_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result.data_[j * rows_ + i] = data_[i * cols_ + j];
            }
        }
        return result;
    }

    // =========================================================================
    // Square Matrix Operations
    // =========================================================================

    double Trace() const {
        if (rows_ != cols_) {
            throw std::invalid_argument("Trace requires square matrix");
        }
        double sum = 0.0;
        for (int i = 0; i < rows_; ++i) sum += data_[i * cols_ + i];
        return sum;
    }

    /// Determinant (only for small square matrices)
    double Determinant() const;

    /// Inverse (only for small square matrices)
    MatX Inverse() const;

    // =========================================================================
    // Norms
    // =========================================================================

    double NormFrobenius() const {
        double sum = 0.0;
        for (int i = 0; i < rows_ * cols_; ++i) sum += data_[i] * data_[i];
        return std::sqrt(sum);
    }

    // =========================================================================
    // Size Operations
    // =========================================================================

    int Rows() const { return rows_; }
    int Cols() const { return cols_; }
    int Stride() const { return cols_; }  // Row-major, stride = cols

    void Resize(int rows, int cols) {
        if (rows == rows_ && cols == cols_) return;

        MatX temp(rows, cols);
        int copyRows = std::min(rows_, rows);
        int copyCols = std::min(cols_, cols);
        for (int i = 0; i < copyRows; ++i) {
            for (int j = 0; j < copyCols; ++j) {
                temp.data_[i * cols + j] = data_[i * cols_ + j];
            }
        }
        *this = std::move(temp);
    }

    void SetZero() {
        if (data_) std::fill(data_, data_ + rows_ * cols_, 0.0);
    }

    void SetIdentity() {
        if (rows_ != cols_) throw std::invalid_argument("SetIdentity requires square matrix");
        SetZero();
        for (int i = 0; i < rows_; ++i) data_[i * cols_ + i] = 1.0;
    }

    // =========================================================================
    // Data Access
    // =========================================================================

    double* Data() { return data_; }
    const double* Data() const { return data_; }

    // =========================================================================
    // Block Operations
    // =========================================================================

    MatX Block(int startRow, int startCol, int blockRows, int blockCols) const {
        if (startRow < 0 || startCol < 0 ||
            startRow + blockRows > rows_ || startCol + blockCols > cols_) {
            throw std::out_of_range("Block out of range");
        }
        MatX result(blockRows, blockCols);
        for (int i = 0; i < blockRows; ++i) {
            for (int j = 0; j < blockCols; ++j) {
                result.data_[i * blockCols + j] = data_[(startRow + i) * cols_ + (startCol + j)];
            }
        }
        return result;
    }

    void SetBlock(int startRow, int startCol, const MatX& block) {
        if (startRow < 0 || startCol < 0 ||
            startRow + block.rows_ > rows_ || startCol + block.cols_ > cols_) {
            throw std::out_of_range("Block out of range");
        }
        for (int i = 0; i < block.rows_; ++i) {
            for (int j = 0; j < block.cols_; ++j) {
                data_[(startRow + i) * cols_ + (startCol + j)] = block.data_[i * block.cols_ + j];
            }
        }
    }

    // =========================================================================
    // Factory Methods
    // =========================================================================

    static MatX Zero(int rows, int cols) {
        return MatX(rows, cols, 0.0);
    }

    static MatX Identity(int size) {
        MatX result(size, size);
        for (int i = 0; i < size; ++i) result.data_[i * size + i] = 1.0;
        return result;
    }

    static MatX Diagonal(const VecX& diag) {
        int n = diag.Size();
        MatX result(n, n);
        for (int i = 0; i < n; ++i) result.data_[i * n + i] = diag[i];
        return result;
    }

private:
    int rows_;
    int cols_;
    double* data_;
};

// Scalar * MatX
inline MatX operator*(double s, const MatX& m) {
    return m * s;
}

// =============================================================================
// Decomposition Result Structures
// =============================================================================

/**
 * @brief LU decomposition result
 * PA = LU, where P is permutation matrix
 */
struct LUResult {
    MatX L;                    ///< Lower triangular (diagonal = 1)
    MatX U;                    ///< Upper triangular
    std::vector<int> P;        ///< Permutation vector (row swaps)
    int sign;                  ///< Sign of permutation (+1 or -1)
    bool valid;                ///< Whether decomposition succeeded

    /// Compute determinant from U
    double Determinant() const {
        if (!valid) return 0.0;
        double det = static_cast<double>(sign);
        int n = U.Rows();
        for (int i = 0; i < n; ++i) det *= U(i, i);
        return det;
    }
};

/**
 * @brief Cholesky decomposition result
 * A = L * L^T (lower triangular)
 */
struct CholeskyResult {
    MatX L;                    ///< Lower triangular matrix
    bool valid;                ///< Whether matrix is positive definite
};

/**
 * @brief QR decomposition result
 * A = Q * R, where Q is orthogonal, R is upper triangular
 */
struct QRResult {
    MatX Q;                    ///< Orthogonal matrix (m x m or m x n)
    MatX R;                    ///< Upper triangular matrix
    bool valid;                ///< Whether decomposition succeeded
    bool thinQR;               ///< true: Q is m x n, R is n x n
};

/**
 * @brief SVD decomposition result
 * A = U * S * V^T
 */
struct SVDResult {
    MatX U;                    ///< Left singular vectors (m x k)
    VecX S;                    ///< Singular values (descending order)
    MatX V;                    ///< Right singular vectors (n x k)
    bool valid;                ///< Whether decomposition succeeded
    int rank;                  ///< Numerical rank

    /// Condition number
    double Condition() const {
        if (!valid || rank < 1) return std::numeric_limits<double>::infinity();
        return S[0] / S[rank - 1];
    }
};

// =============================================================================
// Special Matrix Factory Functions
// =============================================================================

// -------------------------------------------------------------------------
// 2D Transformations (3x3 homogeneous coordinates)
// -------------------------------------------------------------------------

/**
 * @brief 2D rotation matrix (3x3)
 * @param angle Rotation angle (radians), counter-clockwise positive
 */
Mat33 Rotation2D(double angle);

/**
 * @brief 2D translation matrix (3x3)
 */
Mat33 Translation2D(double tx, double ty);
Mat33 Translation2D(const Vec2& t);

/**
 * @brief 2D scaling matrix (3x3)
 */
Mat33 Scaling2D(double sx, double sy);
Mat33 Scaling2D(double s);  // Uniform scaling

/**
 * @brief 2D affine transformation matrix (3x3)
 * Order: scale -> rotate -> translate
 */
Mat33 Affine2D(double tx, double ty, double angle, double sx, double sy);

// -------------------------------------------------------------------------
// 3D Transformations (4x4 homogeneous coordinates)
// -------------------------------------------------------------------------

/**
 * @brief Rotation around X axis
 */
Mat44 RotationX(double angle);

/**
 * @brief Rotation around Y axis
 */
Mat44 RotationY(double angle);

/**
 * @brief Rotation around Z axis
 */
Mat44 RotationZ(double angle);

/**
 * @brief Euler angle rotation (ZYX order)
 * Equivalent to: Rz(yaw) * Ry(pitch) * Rx(roll)
 */
Mat44 RotationEulerZYX(double roll, double pitch, double yaw);

/**
 * @brief Axis-angle rotation (Rodrigues formula)
 * @param axis Unit rotation axis
 * @param angle Rotation angle (radians)
 */
Mat44 RotationAxisAngle(const Vec3& axis, double angle);

/**
 * @brief 3D translation matrix (4x4)
 */
Mat44 Translation3D(double tx, double ty, double tz);
Mat44 Translation3D(const Vec3& t);

/**
 * @brief 3D scaling matrix (4x4)
 */
Mat44 Scaling3D(double sx, double sy, double sz);
Mat44 Scaling3D(double s);  // Uniform scaling

// -------------------------------------------------------------------------
// 3x3 Rotation Matrices (no homogeneous coordinates)
// -------------------------------------------------------------------------

/**
 * @brief 3x3 rotation matrix from Euler angles (ZYX order)
 */
Mat33 Rotation3x3EulerZYX(double roll, double pitch, double yaw);

/**
 * @brief 3x3 rotation matrix from axis-angle
 */
Mat33 Rotation3x3AxisAngle(const Vec3& axis, double angle);

/**
 * @brief Extract Euler angles from rotation matrix (ZYX order)
 * @return Vec3(roll, pitch, yaw)
 */
Vec3 ExtractEulerZYX(const Mat33& R);

/**
 * @brief Extract axis-angle from rotation matrix
 * @return (axis, angle)
 */
std::pair<Vec3, double> ExtractAxisAngle(const Mat33& R);

// -------------------------------------------------------------------------
// Camera Matrices
// -------------------------------------------------------------------------

/**
 * @brief Camera intrinsic matrix K
 * K = [fx  0  cx]
 *     [0  fy  cy]
 *     [0   0   1]
 */
Mat33 CameraIntrinsic(double fx, double fy, double cx, double cy);

/**
 * @brief Projection matrix P = K * [R|t]
 */
Mat34 ProjectionMatrix(const Mat33& K, const Mat33& R, const Vec3& t);

// -------------------------------------------------------------------------
// QMatrix Conversion
// -------------------------------------------------------------------------

/**
 * @brief Convert from QMatrix (2D affine) to Mat33
 */
Mat33 FromQMatrix(const QMatrix& qmat);

/**
 * @brief Convert Mat33 to QMatrix (2D affine)
 * @note Third row must be [0, 0, 1]
 */
QMatrix ToQMatrix(const Mat33& mat);

} // namespace Qi::Vision::Internal
