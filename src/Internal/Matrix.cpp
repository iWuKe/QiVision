/**
 * @file Matrix.cpp
 * @brief Implementation of matrix operations
 */

#include <QiVision/Internal/Matrix.h>
#include <QiVision/Core/QMatrix.h>
#include <QiVision/Core/Exception.h>

#include <cmath>
#include <algorithm>

namespace Qi::Vision::Internal {

// =============================================================================
// MatX Determinant and Inverse
// =============================================================================

double MatX::Determinant() const {
    if (rows_ != cols_) {
        throw InvalidArgumentException("MatX::Determinant: requires square matrix");
    }

    // Use specialized implementations for small matrices
    switch (rows_) {
        case 1:
            return data_[0];

        case 2: {
            return data_[0] * data_[3] - data_[1] * data_[2];
        }

        case 3: {
            // Expansion by first row
            double a = data_[0], b = data_[1], c = data_[2];
            double d = data_[3], e = data_[4], f = data_[5];
            double g = data_[6], h = data_[7], i = data_[8];
            return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        }

        case 4: {
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

        default:
            // For larger matrices, use LU decomposition
            // For now, throw an error
            throw UnsupportedException("MatX::Determinant: not implemented for matrices larger than 4x4");
    }
}

MatX MatX::Inverse() const {
    if (rows_ != cols_) {
        throw InvalidArgumentException("MatX::Inverse: requires square matrix");
    }

    switch (rows_) {
        case 1: {
            if (std::abs(data_[0]) < MATRIX_SINGULAR_THRESHOLD) {
                return MatX::Zero(1, 1);
            }
            MatX result(1, 1);
            result.data_[0] = 1.0 / data_[0];
            return result;
        }

        case 2: {
            double det = data_[0] * data_[3] - data_[1] * data_[2];
            if (std::abs(det) < MATRIX_SINGULAR_THRESHOLD) {
                return MatX::Zero(2, 2);
            }
            double invDet = 1.0 / det;
            MatX result(2, 2);
            result.data_[0] =  data_[3] * invDet;
            result.data_[1] = -data_[1] * invDet;
            result.data_[2] = -data_[2] * invDet;
            result.data_[3] =  data_[0] * invDet;
            return result;
        }

        case 3: {
            double a = data_[0], b = data_[1], c = data_[2];
            double d = data_[3], e = data_[4], f = data_[5];
            double g = data_[6], h = data_[7], i = data_[8];

            double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
            if (std::abs(det) < MATRIX_SINGULAR_THRESHOLD) {
                return MatX::Zero(3, 3);
            }
            double invDet = 1.0 / det;

            MatX result(3, 3);
            result.data_[0] = (e * i - f * h) * invDet;
            result.data_[1] = (c * h - b * i) * invDet;
            result.data_[2] = (b * f - c * e) * invDet;
            result.data_[3] = (f * g - d * i) * invDet;
            result.data_[4] = (a * i - c * g) * invDet;
            result.data_[5] = (c * d - a * f) * invDet;
            result.data_[6] = (d * h - e * g) * invDet;
            result.data_[7] = (b * g - a * h) * invDet;
            result.data_[8] = (a * e - b * d) * invDet;
            return result;
        }

        case 4: {
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
                return MatX::Zero(4, 4);
            }
            double invDet = 1.0 / det;

            MatX result(4, 4);
            result.data_[0]  = ( m[5] * c5 - m[6] * c4 + m[7] * c3) * invDet;
            result.data_[1]  = (-m[1] * c5 + m[2] * c4 - m[3] * c3) * invDet;
            result.data_[2]  = ( m[13] * s5 - m[14] * s4 + m[15] * s3) * invDet;
            result.data_[3]  = (-m[9] * s5 + m[10] * s4 - m[11] * s3) * invDet;

            result.data_[4]  = (-m[4] * c5 + m[6] * c2 - m[7] * c1) * invDet;
            result.data_[5]  = ( m[0] * c5 - m[2] * c2 + m[3] * c1) * invDet;
            result.data_[6]  = (-m[12] * s5 + m[14] * s2 - m[15] * s1) * invDet;
            result.data_[7]  = ( m[8] * s5 - m[10] * s2 + m[11] * s1) * invDet;

            result.data_[8]  = ( m[4] * c4 - m[5] * c2 + m[7] * c0) * invDet;
            result.data_[9]  = (-m[0] * c4 + m[1] * c2 - m[3] * c0) * invDet;
            result.data_[10] = ( m[12] * s4 - m[13] * s2 + m[15] * s0) * invDet;
            result.data_[11] = (-m[8] * s4 + m[9] * s2 - m[11] * s0) * invDet;

            result.data_[12] = (-m[4] * c3 + m[5] * c1 - m[6] * c0) * invDet;
            result.data_[13] = ( m[0] * c3 - m[1] * c1 + m[2] * c0) * invDet;
            result.data_[14] = (-m[12] * s3 + m[13] * s1 - m[14] * s0) * invDet;
            result.data_[15] = ( m[8] * s3 - m[9] * s1 + m[10] * s0) * invDet;

            return result;
        }

        default:
            throw UnsupportedException("MatX::Inverse: not implemented for matrices larger than 4x4");
    }
}

// =============================================================================
// 2D Transformation Matrices (3x3)
// =============================================================================

Mat33 Rotation2D(double angle) {
    double c = std::cos(angle);
    double s = std::sin(angle);
    return Mat33{
        c, -s, 0.0,
        s,  c, 0.0,
        0.0, 0.0, 1.0
    };
}

Mat33 Translation2D(double tx, double ty) {
    return Mat33{
        1.0, 0.0, tx,
        0.0, 1.0, ty,
        0.0, 0.0, 1.0
    };
}

Mat33 Translation2D(const Vec2& t) {
    return Translation2D(t[0], t[1]);
}

Mat33 Scaling2D(double sx, double sy) {
    return Mat33{
        sx, 0.0, 0.0,
        0.0, sy, 0.0,
        0.0, 0.0, 1.0
    };
}

Mat33 Scaling2D(double s) {
    return Scaling2D(s, s);
}

Mat33 Affine2D(double tx, double ty, double angle, double sx, double sy) {
    // Order: scale -> rotate -> translate
    double c = std::cos(angle);
    double s = std::sin(angle);
    return Mat33{
        sx * c, -sy * s, tx,
        sx * s,  sy * c, ty,
        0.0,     0.0,    1.0
    };
}

// =============================================================================
// 3D Transformation Matrices (4x4)
// =============================================================================

Mat44 RotationX(double angle) {
    double c = std::cos(angle);
    double s = std::sin(angle);
    return Mat44{
        1.0, 0.0, 0.0, 0.0,
        0.0,  c,  -s, 0.0,
        0.0,  s,   c, 0.0,
        0.0, 0.0, 0.0, 1.0
    };
}

Mat44 RotationY(double angle) {
    double c = std::cos(angle);
    double s = std::sin(angle);
    return Mat44{
         c,  0.0,  s, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -s,  0.0,  c, 0.0,
        0.0, 0.0, 0.0, 1.0
    };
}

Mat44 RotationZ(double angle) {
    double c = std::cos(angle);
    double s = std::sin(angle);
    return Mat44{
         c,  -s, 0.0, 0.0,
         s,   c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    };
}

Mat44 RotationEulerZYX(double roll, double pitch, double yaw) {
    // Rz(yaw) * Ry(pitch) * Rx(roll)
    double cr = std::cos(roll);
    double sr = std::sin(roll);
    double cp = std::cos(pitch);
    double sp = std::sin(pitch);
    double cy = std::cos(yaw);
    double sy = std::sin(yaw);

    return Mat44{
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0,
           -sp,                cp * sr,                cp * cr, 0.0,
           0.0,                    0.0,                    0.0, 1.0
    };
}

Mat44 RotationAxisAngle(const Vec3& axis, double angle) {
    // Rodrigues' rotation formula
    Vec3 n = axis.Normalized();
    double c = std::cos(angle);
    double s = std::sin(angle);
    double t = 1.0 - c;

    double x = n[0], y = n[1], z = n[2];

    return Mat44{
        t*x*x + c,   t*x*y - s*z, t*x*z + s*y, 0.0,
        t*x*y + s*z, t*y*y + c,   t*y*z - s*x, 0.0,
        t*x*z - s*y, t*y*z + s*x, t*z*z + c,   0.0,
        0.0,         0.0,         0.0,         1.0
    };
}

Mat44 Translation3D(double tx, double ty, double tz) {
    return Mat44{
        1.0, 0.0, 0.0, tx,
        0.0, 1.0, 0.0, ty,
        0.0, 0.0, 1.0, tz,
        0.0, 0.0, 0.0, 1.0
    };
}

Mat44 Translation3D(const Vec3& t) {
    return Translation3D(t[0], t[1], t[2]);
}

Mat44 Scaling3D(double sx, double sy, double sz) {
    return Mat44{
        sx,  0.0, 0.0, 0.0,
        0.0, sy,  0.0, 0.0,
        0.0, 0.0, sz,  0.0,
        0.0, 0.0, 0.0, 1.0
    };
}

Mat44 Scaling3D(double s) {
    return Scaling3D(s, s, s);
}

// =============================================================================
// 3x3 Rotation Matrices
// =============================================================================

Mat33 Rotation3x3EulerZYX(double roll, double pitch, double yaw) {
    double cr = std::cos(roll);
    double sr = std::sin(roll);
    double cp = std::cos(pitch);
    double sp = std::sin(pitch);
    double cy = std::cos(yaw);
    double sy = std::sin(yaw);

    return Mat33{
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
           -sp,                cp * sr,                cp * cr
    };
}

Mat33 Rotation3x3AxisAngle(const Vec3& axis, double angle) {
    Vec3 n = axis.Normalized();
    double c = std::cos(angle);
    double s = std::sin(angle);
    double t = 1.0 - c;

    double x = n[0], y = n[1], z = n[2];

    return Mat33{
        t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
        t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
        t*x*z - s*y, t*y*z + s*x, t*z*z + c
    };
}

Vec3 ExtractEulerZYX(const Mat33& R) {
    // Extract Euler angles from rotation matrix (ZYX order)
    // R = Rz(yaw) * Ry(pitch) * Rx(roll)
    double pitch, roll, yaw;

    // Check for gimbal lock
    if (std::abs(R(2, 0)) >= 1.0 - MATRIX_EPSILON) {
        // Gimbal lock: pitch = +/- 90 degrees
        yaw = 0.0;  // Arbitrary choice
        if (R(2, 0) < 0.0) {
            pitch = HALF_PI;
            roll = std::atan2(R(0, 1), R(0, 2));
        } else {
            pitch = -HALF_PI;
            roll = std::atan2(-R(0, 1), -R(0, 2));
        }
    } else {
        pitch = std::asin(-R(2, 0));
        roll = std::atan2(R(2, 1), R(2, 2));
        yaw = std::atan2(R(1, 0), R(0, 0));
    }

    return Vec3{roll, pitch, yaw};
}

std::pair<Vec3, double> ExtractAxisAngle(const Mat33& R) {
    // Trace = 1 + 2*cos(angle)
    double trace = R.Trace();
    double cosAngle = (trace - 1.0) * 0.5;
    cosAngle = Clamp(cosAngle, -1.0, 1.0);
    double angle = std::acos(cosAngle);

    Vec3 axis;

    if (std::abs(angle) < MATRIX_EPSILON) {
        // Zero rotation - any axis works
        axis = Vec3{1.0, 0.0, 0.0};
        angle = 0.0;
    } else if (std::abs(angle - PI) < MATRIX_EPSILON) {
        // 180 degree rotation - need to find axis from R
        // R = 2*n*n^T - I, so diagonal of R+I = 2*n^2
        double x2 = (R(0, 0) + 1.0) * 0.5;
        double y2 = (R(1, 1) + 1.0) * 0.5;
        double z2 = (R(2, 2) + 1.0) * 0.5;

        if (x2 >= y2 && x2 >= z2) {
            double x = std::sqrt(x2);
            double y = R(0, 1) / (2.0 * x);
            double z = R(0, 2) / (2.0 * x);
            axis = Vec3{x, y, z};
        } else if (y2 >= x2 && y2 >= z2) {
            double y = std::sqrt(y2);
            double x = R(0, 1) / (2.0 * y);
            double z = R(1, 2) / (2.0 * y);
            axis = Vec3{x, y, z};
        } else {
            double z = std::sqrt(z2);
            double x = R(0, 2) / (2.0 * z);
            double y = R(1, 2) / (2.0 * z);
            axis = Vec3{x, y, z};
        }
        axis.Normalize();
    } else {
        // General case: axis from skew-symmetric part
        double sinAngle = std::sin(angle);
        axis = Vec3{
            (R(2, 1) - R(1, 2)) / (2.0 * sinAngle),
            (R(0, 2) - R(2, 0)) / (2.0 * sinAngle),
            (R(1, 0) - R(0, 1)) / (2.0 * sinAngle)
        };
        axis.Normalize();
    }

    return {axis, angle};
}

// =============================================================================
// Camera Matrices
// =============================================================================

Mat33 CameraIntrinsic(double fx, double fy, double cx, double cy) {
    return Mat33{
        fx,  0.0, cx,
        0.0, fy,  cy,
        0.0, 0.0, 1.0
    };
}

Mat34 ProjectionMatrix(const Mat33& K, const Mat33& R, const Vec3& t) {
    // P = K * [R | t]
    Mat34 Rt;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Rt(i, j) = R(i, j);
        }
        Rt(i, 3) = t[i];
    }

    Mat34 P;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += K(i, k) * Rt(k, j);
            }
            P(i, j) = sum;
        }
    }
    return P;
}

// =============================================================================
// QMatrix Conversion
// =============================================================================

Mat33 FromQMatrix(const QMatrix& qmat) {
    return Mat33{
        qmat.M00(), qmat.M01(), qmat.M02(),
        qmat.M10(), qmat.M11(), qmat.M12(),
        0.0,        0.0,        1.0
    };
}

QMatrix ToQMatrix(const Mat33& mat) {
    return QMatrix(
        mat(0, 0), mat(0, 1), mat(0, 2),
        mat(1, 0), mat(1, 1), mat(1, 2)
    );
}

} // namespace Qi::Vision::Internal
