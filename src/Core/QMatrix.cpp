#include <QiVision/Core/QMatrix.h>
#include <QiVision/Core/Constants.h>
#include <cmath>
#include <stdexcept>

namespace Qi::Vision {

// =============================================================================
// Constructors
// =============================================================================

QMatrix::QMatrix() : m_{1.0, 0.0, 0.0, 0.0, 1.0, 0.0} {}

QMatrix::QMatrix(double m00, double m01, double m02,
                 double m10, double m11, double m12)
    : m_{m00, m01, m02, m10, m11, m12} {}

QMatrix::QMatrix(const double (&elements)[6])
    : m_{elements[0], elements[1], elements[2],
         elements[3], elements[4], elements[5]} {}

// =============================================================================
// Static Factory Methods
// =============================================================================

QMatrix QMatrix::Identity() {
    return QMatrix();
}

QMatrix QMatrix::Translation(double tx, double ty) {
    return QMatrix(1.0, 0.0, tx, 0.0, 1.0, ty);
}

QMatrix QMatrix::Translation(const Point2d& t) {
    return Translation(t.x, t.y);
}

QMatrix QMatrix::Rotation(double angle) {
    double c = std::cos(angle);
    double s = std::sin(angle);
    return QMatrix(c, -s, 0.0, s, c, 0.0);
}

QMatrix QMatrix::Rotation(double angle, const Point2d& center) {
    return Rotation(angle, center.x, center.y);
}

QMatrix QMatrix::Rotation(double angle, double cx, double cy) {
    // Rotate around (cx, cy):
    // 1. Translate to origin
    // 2. Rotate
    // 3. Translate back
    double c = std::cos(angle);
    double s = std::sin(angle);
    double tx = cx - c * cx + s * cy;
    double ty = cy - s * cx - c * cy;
    return QMatrix(c, -s, tx, s, c, ty);
}

QMatrix QMatrix::Scaling(double scale) {
    return Scaling(scale, scale);
}

QMatrix QMatrix::Scaling(double sx, double sy) {
    return QMatrix(sx, 0.0, 0.0, 0.0, sy, 0.0);
}

QMatrix QMatrix::Scaling(double sx, double sy, const Point2d& center) {
    // Scale around center:
    // 1. Translate to origin
    // 2. Scale
    // 3. Translate back
    double tx = center.x * (1.0 - sx);
    double ty = center.y * (1.0 - sy);
    return QMatrix(sx, 0.0, tx, 0.0, sy, ty);
}

QMatrix QMatrix::Shearing(double shx, double shy) {
    return QMatrix(1.0, shx, 0.0, shy, 1.0, 0.0);
}

QMatrix QMatrix::FromPoints(const Point2d src[3], const Point2d dst[3]) {
    // Solve affine transformation from 3 point correspondences
    // Using direct solution for 2D affine (6 unknowns, 6 equations)

    // Source matrix:
    // | x0 y0 1  0  0  0 |   | m00 |   | x0' |
    // | 0  0  0  x0 y0 1 |   | m01 |   | y0' |
    // | x1 y1 1  0  0  0 | * | m02 | = | x1' |
    // | 0  0  0  x1 y1 1 |   | m10 |   | y1' |
    // | x2 y2 1  0  0  0 |   | m11 |   | x2' |
    // | 0  0  0  x2 y2 1 |   | m12 |   | y2' |

    // Simplified: solve two 3x3 systems
    double x0 = src[0].x, y0 = src[0].y;
    double x1 = src[1].x, y1 = src[1].y;
    double x2 = src[2].x, y2 = src[2].y;

    double dx0 = dst[0].x, dy0 = dst[0].y;
    double dx1 = dst[1].x, dy1 = dst[1].y;
    double dx2 = dst[2].x, dy2 = dst[2].y;

    // Determinant of source matrix
    double det = x0 * (y1 - y2) - y0 * (x1 - x2) + (x1 * y2 - x2 * y1);

    if (std::abs(det) < EPSILON) {
        // Points are collinear, return identity
        return Identity();
    }

    double invDet = 1.0 / det;

    // Solve for m00, m01, m02 (x transformation)
    double m00 = ((y1 - y2) * dx0 + (y2 - y0) * dx1 + (y0 - y1) * dx2) * invDet;
    double m01 = ((x2 - x1) * dx0 + (x0 - x2) * dx1 + (x1 - x0) * dx2) * invDet;
    double m02 = ((x1 * y2 - x2 * y1) * dx0 + (x2 * y0 - x0 * y2) * dx1 +
                  (x0 * y1 - x1 * y0) * dx2) * invDet;

    // Solve for m10, m11, m12 (y transformation)
    double m10 = ((y1 - y2) * dy0 + (y2 - y0) * dy1 + (y0 - y1) * dy2) * invDet;
    double m11 = ((x2 - x1) * dy0 + (x0 - x2) * dy1 + (x1 - x0) * dy2) * invDet;
    double m12 = ((x1 * y2 - x2 * y1) * dy0 + (x2 * y0 - x0 * y2) * dy1 +
                  (x0 * y1 - x1 * y0) * dy2) * invDet;

    return QMatrix(m00, m01, m02, m10, m11, m12);
}

// =============================================================================
// Matrix Operations
// =============================================================================

QMatrix QMatrix::operator*(const QMatrix& other) const {
    // Matrix multiplication:
    // | m00 m01 m02 |   | o00 o01 o02 |
    // | m10 m11 m12 | * | o10 o11 o12 |
    // | 0   0   1   |   | 0   0   1   |

    return QMatrix(
        m_[0] * other.m_[0] + m_[1] * other.m_[3],           // m00
        m_[0] * other.m_[1] + m_[1] * other.m_[4],           // m01
        m_[0] * other.m_[2] + m_[1] * other.m_[5] + m_[2],   // m02
        m_[3] * other.m_[0] + m_[4] * other.m_[3],           // m10
        m_[3] * other.m_[1] + m_[4] * other.m_[4],           // m11
        m_[3] * other.m_[2] + m_[4] * other.m_[5] + m_[5]    // m12
    );
}

QMatrix& QMatrix::operator*=(const QMatrix& other) {
    *this = *this * other;
    return *this;
}

bool QMatrix::operator==(const QMatrix& other) const {
    for (int i = 0; i < 6; ++i) {
        if (std::abs(m_[i] - other.m_[i]) > EPSILON) {
            return false;
        }
    }
    return true;
}

bool QMatrix::operator!=(const QMatrix& other) const {
    return !(*this == other);
}

QMatrix QMatrix::Inverse() const {
    double det = Determinant();
    if (std::abs(det) < EPSILON) {
        // Not invertible, return identity
        return Identity();
    }

    double invDet = 1.0 / det;

    // Inverse of 2D affine:
    // | m11/det  -m01/det  (m01*m12-m02*m11)/det |
    // | -m10/det  m00/det  (m02*m10-m00*m12)/det |
    // | 0         0        1                     |

    double m00 = m_[4] * invDet;
    double m01 = -m_[1] * invDet;
    double m02 = (m_[1] * m_[5] - m_[2] * m_[4]) * invDet;
    double m10 = -m_[3] * invDet;
    double m11 = m_[0] * invDet;
    double m12 = (m_[2] * m_[3] - m_[0] * m_[5]) * invDet;

    return QMatrix(m00, m01, m02, m10, m11, m12);
}

bool QMatrix::IsInvertible() const {
    return std::abs(Determinant()) > EPSILON;
}

QMatrix QMatrix::TransposeLinear() const {
    // Transpose only the linear part (swap m01 and m10)
    return QMatrix(m_[0], m_[3], m_[2], m_[1], m_[4], m_[5]);
}

double QMatrix::Determinant() const {
    return m_[0] * m_[4] - m_[1] * m_[3];
}

// =============================================================================
// Element Access
// =============================================================================

double QMatrix::At(int row, int col) const {
    if (row < 0 || row > 2 || col < 0 || col > 2) {
        throw std::out_of_range("Matrix index out of range");
    }
    if (row == 2) {
        return (col == 2) ? 1.0 : 0.0;
    }
    return m_[row * 3 + col];
}

void QMatrix::SetAt(int row, int col, double value) {
    if (row < 0 || row > 1 || col < 0 || col > 2) {
        throw std::out_of_range("Matrix index out of range (third row is fixed)");
    }
    m_[row * 3 + col] = value;
}

void QMatrix::GetElements(double (&elements)[6]) const {
    for (int i = 0; i < 6; ++i) {
        elements[i] = m_[i];
    }
}

void QMatrix::GetMatrix3x3(double (&matrix)[9]) const {
    // Row-major 3x3 for OpenGL etc.
    matrix[0] = m_[0]; matrix[1] = m_[1]; matrix[2] = m_[2];
    matrix[3] = m_[3]; matrix[4] = m_[4]; matrix[5] = m_[5];
    matrix[6] = 0.0;   matrix[7] = 0.0;   matrix[8] = 1.0;
}

// =============================================================================
// Point Transformation
// =============================================================================

Point2d QMatrix::Transform(const Point2d& p) const {
    return Transform(p.x, p.y);
}

Point2d QMatrix::Transform(double x, double y) const {
    return {
        m_[0] * x + m_[1] * y + m_[2],
        m_[3] * x + m_[4] * y + m_[5]
    };
}

void QMatrix::TransformPoints(Point2d* points, size_t count) const {
    for (size_t i = 0; i < count; ++i) {
        points[i] = Transform(points[i]);
    }
}

void QMatrix::TransformPoints(const Point2d* src, Point2d* dst, size_t count) const {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = Transform(src[i]);
    }
}

Point2d QMatrix::TransformVector(const Point2d& v) const {
    return TransformVector(v.x, v.y);
}

Point2d QMatrix::TransformVector(double vx, double vy) const {
    // Transform vector (ignores translation)
    return {
        m_[0] * vx + m_[1] * vy,
        m_[3] * vx + m_[4] * vy
    };
}

// =============================================================================
// Decomposition
// =============================================================================

double QMatrix::GetRotation() const {
    // Extract rotation angle from the linear part
    // Assuming no shear, rotation is atan2(m10, m00) or atan2(-m01, m11)
    return std::atan2(m_[3], m_[0]);
}

void QMatrix::GetScale(double& sx, double& sy) const {
    // Extract scale factors (assumes no shear)
    // Scale is the magnitude of the column vectors
    sx = std::sqrt(m_[0] * m_[0] + m_[3] * m_[3]);
    sy = std::sqrt(m_[1] * m_[1] + m_[4] * m_[4]);

    // Check for reflection (negative determinant)
    if (Determinant() < 0) {
        sy = -sy;
    }
}

bool QMatrix::IsIdentity() const {
    return ApproxEqual(m_[0], 1.0) && ApproxEqual(m_[1], 0.0) &&
           ApproxEqual(m_[2], 0.0) && ApproxEqual(m_[3], 0.0) &&
           ApproxEqual(m_[4], 1.0) && ApproxEqual(m_[5], 0.0);
}

bool QMatrix::IsTranslationOnly() const {
    return ApproxEqual(m_[0], 1.0) && ApproxEqual(m_[1], 0.0) &&
           ApproxEqual(m_[3], 0.0) && ApproxEqual(m_[4], 1.0);
}

bool QMatrix::PreservesOrientation() const {
    return Determinant() > 0;
}

} // namespace Qi::Vision
