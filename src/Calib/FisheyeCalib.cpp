/**
 * @file FisheyeCalib.cpp
 * @brief Fisheye camera calibration implementation (Kannala-Brandt)
 */

#include <QiVision/Calib/FisheyeCalib.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Internal/Solver.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace Qi::Vision::Calib {

namespace {

constexpr double EPS = 1e-10;

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

void ComputeResiduals(
    const std::vector<std::vector<Point2d>>& imagePoints,
    const std::vector<std::vector<Point3d>>& objectPoints,
    const FisheyeCameraModel& camera,
    const std::vector<FisheyeExtrinsicParams>& extrinsics,
    std::vector<double>& residuals)
{
    residuals.clear();
    for (size_t v = 0; v < imagePoints.size(); ++v) {
        const auto& imgPts = imagePoints[v];
        const auto& objPts = objectPoints[v];
        const auto& ext = extrinsics[v];

        auto projected = FisheyeProjectPoints(objPts, camera, ext.rvec, ext.t);
        for (size_t i = 0; i < imgPts.size(); ++i) {
            residuals.push_back(projected[i].x - imgPts[i].x);
            residuals.push_back(projected[i].y - imgPts[i].y);
        }
    }
}

} // namespace

// =============================================================================
// FisheyeExtrinsicParams
// =============================================================================

Internal::Mat44 FisheyeExtrinsicParams::ToTransformMatrix() const {
    Internal::Mat44 T;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            T(i, j) = R(i, j);
        }
    }
    T(0, 3) = t[0];
    T(1, 3) = t[1];
    T(2, 3) = t[2];
    T(3, 0) = 0.0;
    T(3, 1) = 0.0;
    T(3, 2) = 0.0;
    T(3, 3) = 1.0;
    return T;
}

FisheyeExtrinsicParams FisheyeExtrinsicParams::FromRt(const Internal::Mat33& R_, const Internal::Vec3& t_) {
    FisheyeExtrinsicParams params;
    params.R = R_;
    params.t = t_;
    params.rvec = MatrixToRodrigues(R_);
    return params;
}

FisheyeExtrinsicParams FisheyeExtrinsicParams::FromRvecTvec(const Internal::Vec3& rvec,
                                                            const Internal::Vec3& tvec) {
    FisheyeExtrinsicParams params;
    params.rvec = rvec;
    params.t = tvec;
    params.R = RodriguesToMatrix(rvec);
    return params;
}

size_t FisheyeCalibrationResult::TotalPoints() const {
    size_t total = 0;
    for (const auto& view : perPointErrors) {
        total += view.size();
    }
    return total;
}

// =============================================================================
// Calibration Core
// =============================================================================

FisheyeCalibrationResult CalibrateFisheye(
    const std::vector<std::vector<Point2d>>& imagePoints,
    const std::vector<std::vector<Point3d>>& objectPoints,
    const Size2i& imageSize,
    FisheyeCalibFlags flags,
    const FisheyeCameraModel* initialCamera)
{
    FisheyeCalibrationResult result;
    result.success = false;

    const int numViews = static_cast<int>(imagePoints.size());
    if (numViews < 3) {
        throw InvalidArgumentException("CalibrateFisheye: requires at least 3 views");
    }
    if (imagePoints.size() != objectPoints.size()) {
        throw InvalidArgumentException("CalibrateFisheye: imagePoints/objectPoints size mismatch");
    }
    if (imageSize.width <= 0 || imageSize.height <= 0) {
        throw InvalidArgumentException("CalibrateFisheye: imageSize must be positive");
    }

    for (int v = 0; v < numViews; ++v) {
        if (imagePoints[v].size() != objectPoints[v].size()) {
            throw InvalidArgumentException("CalibrateFisheye: per-view point count mismatch");
        }
        if (imagePoints[v].size() < 4) {
            throw InvalidArgumentException("CalibrateFisheye: each view must have at least 4 points");
        }
        for (size_t i = 0; i < imagePoints[v].size(); ++i) {
            if (!imagePoints[v][i].IsValid() || !objectPoints[v][i].IsValid()) {
                throw InvalidArgumentException("CalibrateFisheye: invalid point data");
            }
        }
    }

    CameraIntrinsics K;
    FisheyeDistortion dist;

    if ((flags & FisheyeCalibFlags::UseIntrinsicGuess) && initialCamera) {
        K = initialCamera->Intrinsics();
        dist = initialCamera->Distortion();
    } else {
        K = EstimateFisheyeInitialIntrinsics(imageSize, 180.0);
    }

    if (flags & FisheyeCalibFlags::FixPrincipalPoint) {
        K.cx = imageSize.width * 0.5;
        K.cy = imageSize.height * 0.5;
    }
    if (flags & FisheyeCalibFlags::FixAspectRatio) {
        K.fy = K.fx;
    }
    if (flags & FisheyeCalibFlags::FixSkew) {
        // Skew not modeled explicitly
    }

    if (flags & FisheyeCalibFlags::FixK1) dist.k1 = 0.0;
    if (flags & FisheyeCalibFlags::FixK2) dist.k2 = 0.0;
    if (flags & FisheyeCalibFlags::FixK3) dist.k3 = 0.0;
    if (flags & FisheyeCalibFlags::FixK4) dist.k4 = 0.0;

    result.camera = FisheyeCameraModel(K, dist, imageSize);

    // Initialize extrinsics via PnP
    result.extrinsics.resize(numViews);
    for (int v = 0; v < numViews; ++v) {
        Internal::Vec3 rvec{0.0, 0.0, 0.0};
        Internal::Vec3 tvec{0.0, 0.0, 1.0};
        bool ok = FisheyeSolvePnP(objectPoints[v], imagePoints[v], result.camera, rvec, tvec, false);
        if (!ok) {
            return result;
        }
        result.extrinsics[v] = FisheyeExtrinsicParams::FromRvecTvec(rvec, tvec);
    }

    // Simple alternating refinement (intrinsics/distortion + extrinsics)
    const int maxIter = 10;
    for (int iter = 0; iter < maxIter; ++iter) {
        // Update extrinsics per view
        for (int v = 0; v < numViews; ++v) {
            Internal::Vec3 rvec = result.extrinsics[v].rvec;
            Internal::Vec3 tvec = result.extrinsics[v].t;
            bool ok = FisheyeSolvePnP(objectPoints[v], imagePoints[v], result.camera, rvec, tvec, true);
            if (ok) {
                result.extrinsics[v] = FisheyeExtrinsicParams::FromRvecTvec(rvec, tvec);
            }
        }

        // Update intrinsics/distortion with Gauss-Newton (numeric Jacobian)
        std::vector<int> paramMap;
        paramMap.reserve(8);
        // 0 fx,1 fy,2 cx,3 cy,4 k1,5 k2,6 k3,7 k4
        auto allowParam = [&](int idx) {
            switch (idx) {
                case 0: return !(flags & FisheyeCalibFlags::FixFocalLength);
                case 1: return !(flags & FisheyeCalibFlags::FixFocalLength);
                case 2: return !(flags & FisheyeCalibFlags::FixPrincipalPoint);
                case 3: return !(flags & FisheyeCalibFlags::FixPrincipalPoint);
                case 4: return !(flags & FisheyeCalibFlags::FixK1);
                case 5: return !(flags & FisheyeCalibFlags::FixK2);
                case 6: return !(flags & FisheyeCalibFlags::FixK3);
                case 7: return !(flags & FisheyeCalibFlags::FixK4);
                default: return false;
            }
        };

        for (int i = 0; i < 8; ++i) {
            if (allowParam(i)) paramMap.push_back(i);
        }
        if (paramMap.empty()) {
            break;
        }

        std::vector<double> baseResiduals;
        ComputeResiduals(imagePoints, objectPoints, result.camera, result.extrinsics, baseResiduals);

        const int m = static_cast<int>(baseResiduals.size());
        const int n = static_cast<int>(paramMap.size());
        Internal::MatX J(m, n);
        Internal::VecX r(m);
        for (int i = 0; i < m; ++i) {
            r[i] = -baseResiduals[static_cast<size_t>(i)];
        }

        for (int pi = 0; pi < n; ++pi) {
            int idx = paramMap[pi];
            FisheyeCameraModel camPert = result.camera;

            double* paramPtr = nullptr;
            if (idx == 0) paramPtr = &camPert.Intrinsics().fx;
            if (idx == 1) paramPtr = &camPert.Intrinsics().fy;
            if (idx == 2) paramPtr = &camPert.Intrinsics().cx;
            if (idx == 3) paramPtr = &camPert.Intrinsics().cy;
            if (idx == 4) paramPtr = &camPert.Distortion().k1;
            if (idx == 5) paramPtr = &camPert.Distortion().k2;
            if (idx == 6) paramPtr = &camPert.Distortion().k3;
            if (idx == 7) paramPtr = &camPert.Distortion().k4;

            double baseVal = *paramPtr;
            double eps = 1e-6 * std::max(1.0, std::abs(baseVal));
            *paramPtr = baseVal + eps;

            std::vector<double> pertResiduals;
            ComputeResiduals(imagePoints, objectPoints, camPert, result.extrinsics, pertResiduals);

            for (int ri = 0; ri < m; ++ri) {
                double d = (pertResiduals[static_cast<size_t>(ri)] - baseResiduals[static_cast<size_t>(ri)]) / eps;
                J(ri, pi) = d;
            }
        }

        Internal::VecX delta = Internal::SolveLeastSquaresNormal(J, r);
        if (delta.Size() != n) {
            break;
        }

        // Damping
        const double lambda = 1e-3;
        for (int i = 0; i < n; ++i) {
            delta[i] *= (1.0 / (1.0 + lambda));
        }

        for (int i = 0; i < n; ++i) {
            int idx = paramMap[i];
            switch (idx) {
                case 0: result.camera.Intrinsics().fx += delta[i]; break;
                case 1: result.camera.Intrinsics().fy += delta[i]; break;
                case 2: result.camera.Intrinsics().cx += delta[i]; break;
                case 3: result.camera.Intrinsics().cy += delta[i]; break;
                case 4: result.camera.Distortion().k1 += delta[i]; break;
                case 5: result.camera.Distortion().k2 += delta[i]; break;
                case 6: result.camera.Distortion().k3 += delta[i]; break;
                case 7: result.camera.Distortion().k4 += delta[i]; break;
                default: break;
            }
        }
    }

    // Final stats
    result.perViewErrors.resize(numViews);
    result.perPointErrors.resize(numViews);

    double totalError = 0.0;
    double maxError = 0.0;
    int totalPoints = 0;

    for (int v = 0; v < numViews; ++v) {
        auto errors = ComputeFisheyeReprojectionErrors(
            objectPoints[v], imagePoints[v], result.camera,
            result.extrinsics[v].rvec, result.extrinsics[v].t);

        result.perPointErrors[v] = errors;
        double viewError = 0.0;
        for (double e : errors) {
            totalError += e * e;
            viewError += e * e;
            maxError = std::max(maxError, e);
            ++totalPoints;
        }
        result.perViewErrors[v] = std::sqrt(viewError / errors.size());
    }

    if (totalPoints > 0) {
        result.rmsError = std::sqrt(totalError / totalPoints);
        result.meanError = totalError / totalPoints;
        result.maxError = maxError;
    }

    result.success = true;
    return result;
}

// =============================================================================
// PnP and Projection
// =============================================================================

bool FisheyeSolvePnP(
    const std::vector<Point3d>& objectPoints,
    const std::vector<Point2d>& imagePoints,
    const FisheyeCameraModel& camera,
    Internal::Vec3& rvec,
    Internal::Vec3& tvec,
    bool useExtrinsicGuess)
{
    if (objectPoints.size() < 4 || objectPoints.size() != imagePoints.size()) {
        throw InvalidArgumentException("FisheyeSolvePnP: requires >= 4 matched points");
    }
    for (size_t i = 0; i < objectPoints.size(); ++i) {
        if (!objectPoints[i].IsValid() || !imagePoints[i].IsValid()) {
            throw InvalidArgumentException("FisheyeSolvePnP: invalid point data");
        }
    }

    if (!useExtrinsicGuess) {
        rvec = Internal::Vec3{0.0, 0.0, 0.0};
        tvec = Internal::Vec3{0.0, 0.0, 1.0};
    }

    const int maxIter = 15;
    for (int iter = 0; iter < maxIter; ++iter) {
        const size_t n = objectPoints.size();
        const int m = static_cast<int>(n * 2);
        Internal::MatX J(m, 6);
        Internal::VecX r(m);

        auto projected = FisheyeProjectPoints(objectPoints, camera, rvec, tvec);
        for (size_t i = 0; i < n; ++i) {
            r[static_cast<int>(2 * i)] = imagePoints[i].x - projected[i].x;
            r[static_cast<int>(2 * i + 1)] = imagePoints[i].y - projected[i].y;
        }

        const double eps = 1e-6;
        for (int p = 0; p < 6; ++p) {
            Internal::Vec3 rvecPert = rvec;
            Internal::Vec3 tvecPert = tvec;
            if (p < 3) {
                rvecPert[p] += eps;
            } else {
                tvecPert[p - 3] += eps;
            }

            auto projPert = FisheyeProjectPoints(objectPoints, camera, rvecPert, tvecPert);
            for (size_t i = 0; i < n; ++i) {
                double dx = (projPert[i].x - projected[i].x) / eps;
                double dy = (projPert[i].y - projected[i].y) / eps;
                J(static_cast<int>(2 * i), p) = -dx;
                J(static_cast<int>(2 * i + 1), p) = -dy;
            }
        }

        Internal::VecX delta = Internal::SolveLeastSquaresNormal(J, r);
        if (delta.Size() != 6) {
            break;
        }

        double stepNorm = 0.0;
        for (int i = 0; i < 3; ++i) {
            rvec[i] += delta[i];
            tvec[i] += delta[i + 3];
            stepNorm += delta[i] * delta[i] + delta[i + 3] * delta[i + 3];
        }

        if (stepNorm < 1e-12) {
            break;
        }
    }

    return true;
}

std::vector<Point2d> FisheyeProjectPoints(
    const std::vector<Point3d>& objectPoints,
    const FisheyeCameraModel& camera,
    const Internal::Vec3& rvec,
    const Internal::Vec3& tvec)
{
    std::vector<Point2d> out;
    out.reserve(objectPoints.size());

    Internal::Mat33 R = RodriguesToMatrix(rvec);
    for (const auto& p : objectPoints) {
        Internal::Vec3 Pw{p.x, p.y, p.z};
        Internal::Vec3 Pc = R * Pw + tvec;
        out.push_back(camera.ProjectPoint(Point3d(Pc[0], Pc[1], Pc[2])));
    }

    return out;
}

std::vector<double> ComputeFisheyeReprojectionErrors(
    const std::vector<Point3d>& objectPoints,
    const std::vector<Point2d>& imagePoints,
    const FisheyeCameraModel& camera,
    const Internal::Vec3& rvec,
    const Internal::Vec3& tvec)
{
    std::vector<double> errors;
    errors.reserve(objectPoints.size());

    auto projected = FisheyeProjectPoints(objectPoints, camera, rvec, tvec);
    for (size_t i = 0; i < projected.size(); ++i) {
        double dx = projected[i].x - imagePoints[i].x;
        double dy = projected[i].y - imagePoints[i].y;
        errors.push_back(std::sqrt(dx * dx + dy * dy));
    }

    return errors;
}

CameraIntrinsics EstimateFisheyeInitialIntrinsics(
    const Size2i& imageSize,
    double fovDegrees)
{
    if (imageSize.width <= 0 || imageSize.height <= 0) {
        throw InvalidArgumentException("EstimateFisheyeInitialIntrinsics: invalid image size");
    }
    if (!std::isfinite(fovDegrees) || fovDegrees <= 0.0) {
        throw InvalidArgumentException("EstimateFisheyeInitialIntrinsics: invalid fovDegrees");
    }

    double fov = fovDegrees * DEG_TO_RAD;
    double diag = std::sqrt(static_cast<double>(imageSize.width * imageSize.width +
                                               imageSize.height * imageSize.height));
    double f = (diag * 0.5) / (fov * 0.5);

    return CameraIntrinsics(f, f, imageSize.width * 0.5, imageSize.height * 0.5);
}

void ComputeFisheyeProjectionJacobian(
    const Point3d& p3d,
    const FisheyeCameraModel& camera,
    double J[2][3])
{
    const double eps = 1e-6;
    Point2d base = camera.ProjectPoint(p3d);
    Point3d px(p3d.x + eps, p3d.y, p3d.z);
    Point3d py(p3d.x, p3d.y + eps, p3d.z);
    Point3d pz(p3d.x, p3d.y, p3d.z + eps);

    Point2d dx = camera.ProjectPoint(px);
    Point2d dy = camera.ProjectPoint(py);
    Point2d dz = camera.ProjectPoint(pz);

    J[0][0] = (dx.x - base.x) / eps;
    J[1][0] = (dx.y - base.y) / eps;
    J[0][1] = (dy.x - base.x) / eps;
    J[1][1] = (dy.y - base.y) / eps;
    J[0][2] = (dz.x - base.x) / eps;
    J[1][2] = (dz.y - base.y) / eps;
}

void ComputeFisheyeParameterJacobian(
    const Point3d& p3d,
    const FisheyeCameraModel& camera,
    double Jintr[2][4],
    double Jdist[2][4])
{
    const double eps = 1e-6;

    FisheyeCameraModel temp = camera;
    Point2d base = camera.ProjectPoint(p3d);

    double* intr[4] = {
        &temp.Intrinsics().fx,
        &temp.Intrinsics().fy,
        &temp.Intrinsics().cx,
        &temp.Intrinsics().cy
    };

    for (int i = 0; i < 4; ++i) {
        double backup = *intr[i];
        *intr[i] = backup + eps;
        Point2d p = temp.ProjectPoint(p3d);
        Jintr[0][i] = (p.x - base.x) / eps;
        Jintr[1][i] = (p.y - base.y) / eps;
        *intr[i] = backup;
    }

    double* dist[4] = {
        &temp.Distortion().k1,
        &temp.Distortion().k2,
        &temp.Distortion().k3,
        &temp.Distortion().k4
    };

    for (int i = 0; i < 4; ++i) {
        double backup = *dist[i];
        *dist[i] = backup + eps;
        Point2d p = temp.ProjectPoint(p3d);
        Jdist[0][i] = (p.x - base.x) / eps;
        Jdist[1][i] = (p.y - base.y) / eps;
        *dist[i] = backup;
    }
}

} // namespace Qi::Vision::Calib
