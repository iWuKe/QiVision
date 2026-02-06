/**
 * @file HandEyeCalib.cpp
 * @brief Hand-eye calibration implementation (Tsai-Lenz)
 */

#include <QiVision/Calib/HandEyeCalib.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Internal/Solver.h>

#include <algorithm>
#include <cmath>

namespace Qi::Vision::Calib {

namespace {

constexpr double EPS = 1e-10;

Internal::Vec3 MatrixToRodrigues(const Internal::Mat33& R) {
    double trace = R(0, 0) + R(1, 1) + R(2, 2);
    double cosTheta = (trace - 1.0) * 0.5;
    cosTheta = std::clamp(cosTheta, -1.0, 1.0);
    double theta = std::acos(cosTheta);

    if (theta < EPS) {
        return Internal::Vec3{0.0, 0.0, 0.0};
    }

    double sinTheta = std::sin(theta);
    Internal::Vec3 rvec;
    rvec[0] = (R(2, 1) - R(1, 2)) / (2.0 * sinTheta);
    rvec[1] = (R(0, 2) - R(2, 0)) / (2.0 * sinTheta);
    rvec[2] = (R(1, 0) - R(0, 1)) / (2.0 * sinTheta);
    rvec *= theta;
    return rvec;
}

Internal::Mat33 RodriguesToMatrix(const Internal::Vec3& rvec) {
    double theta = rvec.Norm();
    if (theta < EPS) {
        Internal::Mat33 R = Internal::Mat33::Identity();
        R(0, 1) = -rvec[2];
        R(0, 2) = rvec[1];
        R(1, 0) = rvec[2];
        R(1, 2) = -rvec[0];
        R(2, 0) = -rvec[1];
        R(2, 1) = rvec[0];
        return R;
    }

    Internal::Vec3 axis = rvec / theta;
    double kx = axis[0], ky = axis[1], kz = axis[2];

    Internal::Mat33 K;
    K(0, 0) = 0;    K(0, 1) = -kz; K(0, 2) = ky;
    K(1, 0) = kz;   K(1, 1) = 0;   K(1, 2) = -kx;
    K(2, 0) = -ky;  K(2, 1) = kx;  K(2, 2) = 0;

    Internal::Mat33 K2 = K * K;
    double c = std::cos(theta);
    double s = std::sin(theta);

    Internal::Mat33 R = Internal::Mat33::Identity();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R(i, j) = R(i, j) + s * K(i, j) + (1.0 - c) * K2(i, j);
        }
    }
    return R;
}

Internal::Mat33 Skew(const Internal::Vec3& v) {
    Internal::Mat33 S;
    S(0, 0) = 0.0;    S(0, 1) = -v[2]; S(0, 2) = v[1];
    S(1, 0) = v[2];   S(1, 1) = 0.0;   S(1, 2) = -v[0];
    S(2, 0) = -v[1];  S(2, 1) = v[0];  S(2, 2) = 0.0;
    return S;
}

} // namespace

HandEyeCalibrationResult CalibrateHandEye(
    const std::vector<Internal::Mat44>& A,
    const std::vector<Internal::Mat44>& B,
    HandEyeMethod method)
{
    HandEyeCalibrationResult result;
    result.success = false;

    if (A.size() != B.size() || A.size() < 2) {
        throw InvalidArgumentException("CalibrateHandEye: A/B size mismatch or too few pairs");
    }

    if (method != HandEyeMethod::TsaiLenz) {
        throw InvalidArgumentException("CalibrateHandEye: unsupported method");
    }

    const size_t n = A.size();

    // Solve rotation using Tsai-Lenz: skew(ra+rb) * r = rb - ra
    Internal::MatX M(static_cast<int>(3 * n), 3);
    Internal::VecX b(static_cast<int>(3 * n));

    for (size_t i = 0; i < n; ++i) {
        Internal::Mat33 Ra;
        Internal::Mat33 Rb;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                Ra(r, c) = A[i](r, c);
                Rb(r, c) = B[i](r, c);
            }
        }

        Internal::Vec3 ra = MatrixToRodrigues(Ra);
        Internal::Vec3 rb = MatrixToRodrigues(Rb);

        Internal::Mat33 S = Skew(ra + rb);
        Internal::Vec3 rhs = rb - ra;

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                M(static_cast<int>(3 * i + r), c) = S(r, c);
            }
            b[static_cast<int>(3 * i + r)] = rhs[r];
        }
    }

    Internal::VecX rvec = Internal::SolveLeastSquaresNormal(M, b);
    Internal::Mat33 R = RodriguesToMatrix(Internal::Vec3{rvec[0], rvec[1], rvec[2]});

    // Solve translation: (Ra - I) t = R * tb - ta
    Internal::MatX Aeq(static_cast<int>(3 * n), 3);
    Internal::VecX beq(static_cast<int>(3 * n));

    for (size_t i = 0; i < n; ++i) {
        Internal::Mat33 Ra;
        Internal::Vec3 ta;
        Internal::Vec3 tb;

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                Ra(r, c) = A[i](r, c);
            }
            ta[r] = A[i](r, 3);
            tb[r] = B[i](r, 3);
        }

        Internal::Mat33 RaI = Ra - Internal::Mat33::Identity();
        Internal::Vec3 rhs = R * tb - ta;

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                Aeq(static_cast<int>(3 * i + r), c) = RaI(r, c);
            }
            beq[static_cast<int>(3 * i + r)] = rhs[r];
        }
    }

    Internal::VecX tvec = Internal::SolveLeastSquaresNormal(Aeq, beq);
    Internal::Vec3 t{tvec[0], tvec[1], tvec[2]};

    // Build X
    Internal::Mat44 X;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            X(r, c) = (r == c) ? 1.0 : 0.0;
        }
    }
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            X(r, c) = R(r, c);
        }
        X(r, 3) = t[r];
    }

    result.R = R;
    result.t = t;
    result.X = X;
    result.success = true;
    return result;
}

} // namespace Qi::Vision::Calib
