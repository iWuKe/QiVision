#pragma once

/**
 * @file StereoCalib.h
 * @brief Stereo camera calibration
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Calib/CameraCalib.h>
#include <QiVision/Calib/CameraModel.h>
#include <QiVision/Internal/Matrix.h>
#include <QiVision/Core/Export.h>

#include <vector>

namespace Qi::Vision::Calib {

/**
 * @brief Result of stereo calibration
 */
struct QIVISION_API StereoCalibrationResult {
    bool success = false;

    CameraModel leftCamera;
    CameraModel rightCamera;

    Internal::Mat33 R;    ///< Rotation from left to right
    Internal::Vec3 t;     ///< Translation from left to right

    double rmsError = 0.0;
    double meanError = 0.0;
    double maxError = 0.0;

    std::vector<ExtrinsicParams> leftExtrinsics;
    std::vector<ExtrinsicParams> rightExtrinsics;
};

/**
 * @brief Calibrate stereo cameras (minimal implementation)
 *
 * Steps:
 * - Calibrate left and right cameras independently
 * - Estimate relative R,t from per-view extrinsics
 * - Compute reprojection error statistics
 */
QIVISION_API StereoCalibrationResult CalibrateStereo(
    const std::vector<std::vector<Point2d>>& leftImagePoints,
    const std::vector<std::vector<Point2d>>& rightImagePoints,
    const std::vector<std::vector<Point3d>>& objectPoints,
    const Size2i& imageSize,
    CalibFlags flags = CalibFlags::None,
    const CameraModel* initialLeft = nullptr,
    const CameraModel* initialRight = nullptr
);

} // namespace Qi::Vision::Calib
