#pragma once

/**
 * @file HandEyeCalib.h
 * @brief Hand-eye calibration (AX=XB)
 */

#include <QiVision/Internal/Matrix.h>
#include <QiVision/Core/Export.h>

#include <vector>

namespace Qi::Vision::Calib {

/**
 * @brief Hand-eye calibration method
 */
enum class HandEyeMethod {
    TsaiLenz
};

/**
 * @brief Hand-eye calibration result
 */
struct QIVISION_API HandEyeCalibrationResult {
    bool success = false;
    Internal::Mat33 R;
    Internal::Vec3 t;
    Internal::Mat44 X;
};

/**
 * @brief Solve hand-eye calibration (AX=XB)
 *
 * @param A List of robot motions (4x4): A_i
 * @param B List of camera motions (4x4): B_i
 * @param method Calibration method
 */
QIVISION_API HandEyeCalibrationResult CalibrateHandEye(
    const std::vector<Internal::Mat44>& A,
    const std::vector<Internal::Mat44>& B,
    HandEyeMethod method = HandEyeMethod::TsaiLenz
);

} // namespace Qi::Vision::Calib
