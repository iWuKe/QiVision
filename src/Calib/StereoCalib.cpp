/**
 * @file StereoCalib.cpp
 * @brief Stereo camera calibration implementation
 */

#include <QiVision/Calib/StereoCalib.h>
#include <QiVision/Core/Exception.h>

#include <algorithm>
#include <cmath>

namespace Qi::Vision::Calib {


StereoCalibrationResult CalibrateStereo(
    const std::vector<std::vector<Point2d>>& leftImagePoints,
    const std::vector<std::vector<Point2d>>& rightImagePoints,
    const std::vector<std::vector<Point3d>>& objectPoints,
    const Size2i& imageSize,
    CalibFlags flags,
    const CameraModel* initialLeft,
    const CameraModel* initialRight)
{
    StereoCalibrationResult result;
    result.success = false;

    if (leftImagePoints.size() != rightImagePoints.size() ||
        leftImagePoints.size() != objectPoints.size()) {
        throw InvalidArgumentException("CalibrateStereo: view count mismatch");
    }

    if (leftImagePoints.size() < 3) {
        throw InvalidArgumentException("CalibrateStereo: requires at least 3 views");
    }

    // Calibrate each camera independently
    CalibrationResult left = CalibrateCamera(leftImagePoints, objectPoints, imageSize, flags, initialLeft);
    CalibrationResult right = CalibrateCamera(rightImagePoints, objectPoints, imageSize, flags, initialRight);

    if (!left.success || !right.success) {
        return result;
    }

    result.leftCamera = left.camera;
    result.rightCamera = right.camera;
    result.leftExtrinsics = left.extrinsics;
    result.rightExtrinsics = right.extrinsics;

    // Estimate relative R,t using per-view extrinsics
    std::vector<Internal::Vec3> rvecs;
    std::vector<Internal::Vec3> tvecs;
    for (size_t i = 0; i < left.extrinsics.size(); ++i) {
        Internal::Mat33 Rl = left.extrinsics[i].R;
        Internal::Mat33 Rr = right.extrinsics[i].R;
        Internal::Vec3 tl = left.extrinsics[i].t;
        Internal::Vec3 tr = right.extrinsics[i].t;

        Internal::Mat33 Rlr = Rr * Rl.Transpose();
        Internal::Vec3 tlr = tr - Rlr * tl;

        rvecs.push_back(MatrixToRodrigues(Rlr));
        tvecs.push_back(tlr);
    }

    // Average rotation in Rodrigues space (simple average)
    Internal::Vec3 rmean{0.0, 0.0, 0.0};
    Internal::Vec3 tmean{0.0, 0.0, 0.0};
    for (size_t i = 0; i < rvecs.size(); ++i) {
        rmean += rvecs[i];
        tmean += tvecs[i];
    }
    rmean /= static_cast<double>(rvecs.size());
    tmean /= static_cast<double>(tvecs.size());

    result.R = RodriguesToMatrix(rmean);
    result.t = tmean;

    // Compute basic reprojection stats (sum of both cameras)
    double totalError = 0.0;
    double maxError = 0.0;
    int totalPoints = 0;

    for (size_t v = 0; v < leftImagePoints.size(); ++v) {
        auto leftErrors = ComputeReprojectionErrors(objectPoints[v], leftImagePoints[v],
                                                    result.leftCamera, left.extrinsics[v].rvec, left.extrinsics[v].t);
        auto rightErrors = ComputeReprojectionErrors(objectPoints[v], rightImagePoints[v],
                                                     result.rightCamera, right.extrinsics[v].rvec, right.extrinsics[v].t);
        for (double e : leftErrors) {
            totalError += e * e;
            maxError = std::max(maxError, e);
            ++totalPoints;
        }
        for (double e : rightErrors) {
            totalError += e * e;
            maxError = std::max(maxError, e);
            ++totalPoints;
        }
    }

    if (totalPoints > 0) {
        result.rmsError = std::sqrt(totalError / totalPoints);
        result.meanError = totalError / totalPoints;
        result.maxError = maxError;
    }

    result.success = true;
    return result;
}

} // namespace Qi::Vision::Calib
