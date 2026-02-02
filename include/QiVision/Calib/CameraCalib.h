#pragma once

/**
 * @file CameraCalib.h
 * @brief Camera calibration using Zhang's method
 *
 * Provides:
 * - Camera intrinsic calibration from multiple images
 * - Distortion coefficient estimation
 * - Extrinsic parameter estimation
 *
 * Reference: Z. Zhang, "A flexible new technique for camera calibration",
 * IEEE Transactions on Pattern Analysis and Machine Intelligence, 2000
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Calib/CameraModel.h>
#include <QiVision/Internal/Matrix.h>
#include <QiVision/Core/Export.h>

#include <vector>

namespace Qi::Vision::Calib {

/**
 * @brief Camera calibration flags
 */
enum class CalibFlags : uint32_t {
    None = 0,
    FixPrincipalPoint = 1,      ///< Fix principal point at image center
    FixAspectRatio = 2,         ///< Fix fx = fy
    ZeroTangentDist = 4,        ///< Assume p1 = p2 = 0
    FixK1 = 8,                  ///< Fix k1 = 0
    FixK2 = 16,                 ///< Fix k2 = 0
    FixK3 = 32,                 ///< Fix k3 = 0
    RationalModel = 64,         ///< Use rational distortion model (k4, k5, k6)
    UseIntrinsicGuess = 128     ///< Use provided intrinsics as initial guess
};

inline CalibFlags operator|(CalibFlags a, CalibFlags b) {
    return static_cast<CalibFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline bool operator&(CalibFlags a, CalibFlags b) {
    return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) != 0;
}

/**
 * @brief Extrinsic parameters for a single calibration image
 */
struct QIVISION_API ExtrinsicParams {
    Internal::Mat33 R;      ///< Rotation matrix (3x3)
    Internal::Vec3 t;       ///< Translation vector
    Internal::Vec3 rvec;    ///< Rodrigues rotation vector (for optimization)

    /// Get 4x4 transformation matrix
    Internal::Mat44 ToTransformMatrix() const;

    /// Create from rotation matrix and translation
    static ExtrinsicParams FromRt(const Internal::Mat33& R, const Internal::Vec3& t);
};

/**
 * @brief Result of camera calibration
 */
struct QIVISION_API CalibrationResult {
    bool success = false;           ///< Whether calibration succeeded
    CameraModel camera;             ///< Calibrated camera model
    double rmsError = 0.0;          ///< RMS reprojection error (pixels)
    double meanError = 0.0;         ///< Mean reprojection error (pixels)
    double maxError = 0.0;          ///< Maximum reprojection error (pixels)

    std::vector<ExtrinsicParams> extrinsics;    ///< Per-image extrinsics
    std::vector<double> perViewErrors;          ///< RMS error per image
    std::vector<std::vector<double>> perPointErrors;  ///< Error per point per image

    /// Get number of calibration views
    size_t NumViews() const { return extrinsics.size(); }
};

/**
 * @brief Calibrate camera using Zhang's method
 *
 * Requires at least 3 images of a planar calibration pattern from
 * different viewpoints.
 *
 * @param imagePoints 2D corner coordinates for each image
 * @param objectPoints 3D corner coordinates (same for all images, z=0)
 * @param imageSize Image size in pixels
 * @param flags Calibration flags
 * @param initialCamera Optional initial camera model (for UseIntrinsicGuess)
 * @return Calibration result
 */
QIVISION_API CalibrationResult CalibrateCamera(
    const std::vector<std::vector<Point2d>>& imagePoints,
    const std::vector<std::vector<Point3d>>& objectPoints,
    const Size2i& imageSize,
    CalibFlags flags = CalibFlags::None,
    const CameraModel* initialCamera = nullptr
);

/**
 * @brief Estimate camera pose from known calibration
 *
 * Given a calibrated camera and 2D-3D correspondences, estimate
 * the camera pose (extrinsic parameters).
 *
 * @param objectPoints 3D object points
 * @param imagePoints Corresponding 2D image points
 * @param camera Calibrated camera model
 * @param[out] rvec Output rotation vector (Rodrigues)
 * @param[out] tvec Output translation vector
 * @param useExtrinsicGuess If true, use rvec/tvec as initial guess
 * @return true if successful
 */
QIVISION_API bool SolvePnP(
    const std::vector<Point3d>& objectPoints,
    const std::vector<Point2d>& imagePoints,
    const CameraModel& camera,
    Internal::Vec3& rvec,
    Internal::Vec3& tvec,
    bool useExtrinsicGuess = false
);

/**
 * @brief Compute reprojection error for calibration
 *
 * @param objectPoints 3D object points
 * @param imagePoints Observed 2D image points
 * @param camera Camera model
 * @param rvec Rotation vector
 * @param tvec Translation vector
 * @return Vector of per-point reprojection errors
 */
QIVISION_API std::vector<double> ComputeReprojectionErrors(
    const std::vector<Point3d>& objectPoints,
    const std::vector<Point2d>& imagePoints,
    const CameraModel& camera,
    const Internal::Vec3& rvec,
    const Internal::Vec3& tvec
);

/**
 * @brief Project 3D points to image plane
 *
 * @param objectPoints 3D object points
 * @param camera Camera model
 * @param rvec Rotation vector
 * @param tvec Translation vector
 * @return Projected 2D points
 */
QIVISION_API std::vector<Point2d> ProjectPoints(
    const std::vector<Point3d>& objectPoints,
    const CameraModel& camera,
    const Internal::Vec3& rvec,
    const Internal::Vec3& tvec
);

/**
 * @brief Convert rotation vector to rotation matrix (Rodrigues)
 *
 * @param rvec Rotation vector (axis * angle)
 * @return 3x3 rotation matrix
 */
QIVISION_API Internal::Mat33 RodriguesToMatrix(const Internal::Vec3& rvec);

/**
 * @brief Convert rotation matrix to rotation vector (Rodrigues)
 *
 * @param R 3x3 rotation matrix
 * @return Rotation vector
 */
QIVISION_API Internal::Vec3 MatrixToRodrigues(const Internal::Mat33& R);

} // namespace Qi::Vision::Calib
