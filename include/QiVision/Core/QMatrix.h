#pragma once

/**
 * @file QMatrix.h
 * @brief 2D Affine transformation matrix for QiVision
 *
 * Represents a 2D affine transformation as a 3x3 matrix:
 * | m00  m01  m02 |   | a  b  tx |
 * | m10  m11  m12 | = | c  d  ty |
 * | 0    0    1   |   | 0  0  1  |
 *
 * Transforms point (x, y) to:
 *   x' = m00*x + m01*y + m02
 *   y' = m10*x + m11*y + m12
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/Export.h>
#include <array>
#include <initializer_list>

namespace Qi::Vision {

/**
 * @brief 2D Affine transformation matrix (3x3, homogeneous)
 */
class QIVISION_API QMatrix {
public:
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Default constructor (identity matrix)
    QMatrix();

    /// Construct from 6 elements (row-major: m00, m01, m02, m10, m11, m12)
    QMatrix(double m00, double m01, double m02,
            double m10, double m11, double m12);

    /// Construct from array of 6 elements
    explicit QMatrix(const double (&elements)[6]);

    // =========================================================================
    // Static Factory Methods
    // =========================================================================

    /// Identity matrix
    static QMatrix Identity();

    /// Translation matrix
    static QMatrix Translation(double tx, double ty);
    static QMatrix Translation(const Point2d& t);

    /// Rotation matrix (angle in radians, around origin)
    static QMatrix Rotation(double angle);

    /// Rotation matrix around a center point
    static QMatrix Rotation(double angle, const Point2d& center);
    static QMatrix Rotation(double angle, double cx, double cy);

    /// Uniform scaling matrix (around origin)
    static QMatrix Scaling(double scale);

    /// Non-uniform scaling matrix (around origin)
    static QMatrix Scaling(double sx, double sy);

    /// Scaling around a center point
    static QMatrix Scaling(double sx, double sy, const Point2d& center);

    /// Shearing matrix
    static QMatrix Shearing(double shx, double shy);

    /// Create from 3 point correspondences (source → target)
    static QMatrix FromPoints(const Point2d src[3], const Point2d dst[3]);

    // =========================================================================
    // Matrix Operations
    // =========================================================================

    /// Matrix multiplication: this * other
    QMatrix operator*(const QMatrix& other) const;

    /// Compound assignment multiplication
    QMatrix& operator*=(const QMatrix& other);

    /// Equality comparison
    bool operator==(const QMatrix& other) const;
    bool operator!=(const QMatrix& other) const;

    /// Invert the matrix (returns identity if not invertible)
    QMatrix Inverse() const;

    /// Check if matrix is invertible
    bool IsInvertible() const;

    /// Transpose the linear part (swap m01 and m10)
    QMatrix TransposeLinear() const;

    /// Determinant of the linear part (m00*m11 - m01*m10)
    double Determinant() const;

    // =========================================================================
    // Element Access
    // =========================================================================

    /// Access element at (row, col), 0-indexed
    double At(int row, int col) const;

    /// Set element at (row, col)
    void SetAt(int row, int col, double value);

    /// Get all 6 elements as array (row-major)
    void GetElements(double (&elements)[6]) const;

    /// Get matrix as 3x3 array (for OpenGL etc.)
    void GetMatrix3x3(double (&matrix)[9]) const;

    /// Direct element access
    double M00() const { return m_[0]; }
    double M01() const { return m_[1]; }
    double M02() const { return m_[2]; }
    double M10() const { return m_[3]; }
    double M11() const { return m_[4]; }
    double M12() const { return m_[5]; }

    // =========================================================================
    // Point Transformation
    // =========================================================================

    /// Transform a single point
    Point2d Transform(const Point2d& p) const;
    Point2d Transform(double x, double y) const;

    /// Transform a single point (convenience operator)
    Point2d operator*(const Point2d& p) const { return Transform(p); }

    /// Transform multiple points in place
    void TransformPoints(Point2d* points, size_t count) const;

    /// Transform multiple points (output to separate array)
    void TransformPoints(const Point2d* src, Point2d* dst, size_t count) const;

    /// Transform a vector (ignores translation)
    Point2d TransformVector(const Point2d& v) const;
    Point2d TransformVector(double vx, double vy) const;

    // =========================================================================
    // Decomposition
    // =========================================================================

    /// Extract translation component
    Point2d GetTranslation() const { return {m_[2], m_[5]}; }

    /// Extract rotation angle (radians)
    double GetRotation() const;

    /// Extract scale factors (approximate, assumes no shear)
    void GetScale(double& sx, double& sy) const;

    /// Check if matrix is identity
    bool IsIdentity() const;

    /// Check if matrix has only translation (no rotation/scale/shear)
    bool IsTranslationOnly() const;

    /// Check if matrix preserves orientation (det > 0)
    bool PreservesOrientation() const;

private:
    // Storage: [m00, m01, m02, m10, m11, m12]
    // The third row is always [0, 0, 1]
    std::array<double, 6> m_;
};

// =============================================================================
// Additional Type Aliases
// =============================================================================

/// Alias for 2D homogeneous transformation matrix
using QHomMat2d = QMatrix;

} // namespace Qi::Vision
