#pragma once

/**
 * @file FisheyeCalib.h
 * @brief Fisheye camera calibration functions
 *
 * Provides comprehensive fisheye camera calibration including:
 * - Multi-view intrinsic calibration using Zhang's method adapted for fisheye
 * - Extrinsic parameter estimation (PnP solving)
 * - Point projection with fisheye model
 * - Reprojection error computation
 *
 * The calibration uses the Kannala-Brandt equidistant projection model
 * with 4 radial distortion coefficients (k1, k2, k3, k4).
 *
 * Reference:
 * - J. Kannala and S. Brandt, "A generic camera model and calibration method
 *   for conventional, wide-angle, and fish-eye lenses", IEEE TPAMI, 2006
 * - OpenCV fisheye calibration (cv::fisheye::calibrate)
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Calib/FisheyeModel.h>
#include <QiVision/Internal/Matrix.h>
#include <QiVision/Core/Export.h>

#include <vector>
#include <cstdint>

namespace Qi::Vision::Calib {

/**
 * @brief Fisheye calibration flags
 *
 * These flags control which parameters are estimated during calibration
 * and which are held fixed. Use bitwise OR to combine multiple flags.
 *
 * @code
 * auto flags = FisheyeCalibFlags::FixK3 | FisheyeCalibFlags::FixK4;
 * @endcode
 */
enum class FisheyeCalibFlags : uint32_t {
    None = 0,                       ///< Estimate all parameters

    // Intrinsic parameter flags
    FixPrincipalPoint = 1 << 0,     ///< Fix principal point at image center
    FixFocalLength = 1 << 1,        ///< Fix focal length (requires UseIntrinsicGuess)
    FixAspectRatio = 1 << 2,        ///< Fix aspect ratio (fx/fy)
    FixSkew = 1 << 3,               ///< Fix skew coefficient to zero

    // Distortion parameter flags
    FixK1 = 1 << 4,                 ///< Fix k1 = 0 (or initial value if UseIntrinsicGuess)
    FixK2 = 1 << 5,                 ///< Fix k2 = 0 (or initial value if UseIntrinsicGuess)
    FixK3 = 1 << 6,                 ///< Fix k3 = 0 (or initial value if UseIntrinsicGuess)
    FixK4 = 1 << 7,                 ///< Fix k4 = 0 (or initial value if UseIntrinsicGuess)

    // Optimization flags
    UseIntrinsicGuess = 1 << 8,     ///< Use provided intrinsics as initial guess
    RecomputeExtrinsic = 1 << 9,    ///< Recompute extrinsics after intrinsic optimization
    CheckCond = 1 << 10,            ///< Check condition number of Jacobian

    // Common combinations
    FixHigherOrderDistortion = FixK3 | FixK4,  ///< Fix k3 and k4 to zero
};

/// Bitwise OR operator for FisheyeCalibFlags
inline FisheyeCalibFlags operator|(FisheyeCalibFlags a, FisheyeCalibFlags b) {
    return static_cast<FisheyeCalibFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

/// Bitwise AND test for FisheyeCalibFlags
inline bool operator&(FisheyeCalibFlags a, FisheyeCalibFlags b) {
    return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) != 0;
}

/**
 * @brief Extrinsic parameters for a single calibration view
 *
 * Represents the pose of the calibration target (or equivalently,
 * the inverse of the camera pose) for one calibration image.
 */
struct QIVISION_API FisheyeExtrinsicParams {
    Internal::Mat33 R;      ///< Rotation matrix (3x3), world to camera
    Internal::Vec3 t;       ///< Translation vector, world to camera
    Internal::Vec3 rvec;    ///< Rodrigues rotation vector (axis * angle)

    /**
     * @brief Get 4x4 homogeneous transformation matrix
     *
     * Returns the transformation matrix that transforms points from
     * world coordinates to camera coordinates.
     *
     * @return 4x4 transformation matrix [R|t; 0 0 0 1]
     */
    Internal::Mat44 ToTransformMatrix() const;

    /**
     * @brief Create from rotation matrix and translation
     * @param R 3x3 rotation matrix
     * @param t Translation vector
     * @return ExtrinsicParams object
     */
    static FisheyeExtrinsicParams FromRt(const Internal::Mat33& R, const Internal::Vec3& t);

    /**
     * @brief Create from Rodrigues vector and translation
     * @param rvec Rodrigues rotation vector
     * @param tvec Translation vector
     * @return ExtrinsicParams object
     */
    static FisheyeExtrinsicParams FromRvecTvec(const Internal::Vec3& rvec,
                                                const Internal::Vec3& tvec);
};

/**
 * @brief Result of fisheye camera calibration
 *
 * Contains the calibrated camera model, per-view extrinsic parameters,
 * and comprehensive error statistics.
 */
struct QIVISION_API FisheyeCalibrationResult {
    bool success = false;               ///< Whether calibration converged successfully
    FisheyeCameraModel camera;          ///< Calibrated fisheye camera model

    // Error statistics
    double rmsError = 0.0;              ///< Root mean square reprojection error (pixels)
    double meanError = 0.0;             ///< Mean reprojection error (pixels)
    double maxError = 0.0;              ///< Maximum reprojection error (pixels)

    // Per-view data
    std::vector<FisheyeExtrinsicParams> extrinsics;     ///< Extrinsics for each view
    std::vector<double> perViewErrors;                   ///< RMS error per view
    std::vector<std::vector<double>> perPointErrors;     ///< Error per point per view

    // Optimization info
    int iterations = 0;                 ///< Number of optimization iterations
    double finalCost = 0.0;             ///< Final optimization cost

    /**
     * @brief Get number of calibration views
     * @return Number of images used in calibration
     */
    size_t NumViews() const { return extrinsics.size(); }

    /**
     * @brief Get total number of calibration points
     * @return Total points across all views
     */
    size_t TotalPoints() const;

    /**
     * @brief Check if calibration quality is acceptable
     * @param maxRmsError Maximum acceptable RMS error (default: 1.0 pixel)
     * @return true if success and rmsError <= maxRmsError
     */
    bool IsGoodCalibration(double maxRmsError = 1.0) const {
        return success && rmsError <= maxRmsError;
    }
};

// =============================================================================
// Core Calibration Functions
// =============================================================================

/**
 * @brief Calibrate fisheye camera from multiple views
 *
 * Performs full fisheye camera calibration using multiple images of a
 * planar calibration pattern. Uses the Kannala-Brandt equidistant
 * projection model with 4 radial distortion coefficients.
 *
 * The calibration process:
 * 1. Estimate initial homographies for each view
 * 2. Compute initial intrinsics from homographies
 * 3. Refine all parameters using Levenberg-Marquardt optimization
 *
 * Requirements:
 * - At least 3 images of the calibration pattern
 * - Pattern should be viewed from different angles (varied poses)
 * - Same number of points detected in each image
 * - Points should cover a significant portion of the image
 *
 * @param imagePoints 2D corner coordinates for each image [numImages][numPoints]
 * @param objectPoints 3D corner coordinates (same pattern for all images, z=0)
 *                     [numImages][numPoints]
 * @param imageSize Image size in pixels
 * @param flags Calibration flags controlling which parameters to estimate
 * @param initialCamera Optional initial camera model (when UseIntrinsicGuess set)
 * @return Calibration result with camera model, extrinsics, and error statistics
 *
 * @code
 * // Example: Basic calibration
 * std::vector<std::vector<Point2d>> imagePoints;  // Detected corners per image
 * std::vector<std::vector<Point3d>> objectPoints; // Known 3D positions (z=0)
 *
 * auto result = CalibrateFisheye(imagePoints, objectPoints, Size2i(1920, 1080));
 * if (result.IsGoodCalibration(0.5)) {
 *     // Use result.camera for undistortion
 * }
 *
 * // Example: Calibrate with fixed k3, k4
 * auto flags = FisheyeCalibFlags::FixK3 | FisheyeCalibFlags::FixK4;
 * auto result = CalibrateFisheye(imagePoints, objectPoints, imageSize, flags);
 * @endcode
 *
 * @see FisheyeCalibFlags for available calibration options
 */
QIVISION_API FisheyeCalibrationResult CalibrateFisheye(
    const std::vector<std::vector<Point2d>>& imagePoints,
    const std::vector<std::vector<Point3d>>& objectPoints,
    const Size2i& imageSize,
    FisheyeCalibFlags flags = FisheyeCalibFlags::None,
    const FisheyeCameraModel* initialCamera = nullptr
);

/**
 * @brief Estimate camera pose from known fisheye calibration (PnP)
 *
 * Given a calibrated fisheye camera and 2D-3D point correspondences,
 * estimates the camera pose (rotation and translation) that explains
 * the observed projections.
 *
 * Uses iterative optimization starting from an initial guess obtained
 * via DLT (Direct Linear Transform) or P3P algorithm.
 *
 * @param objectPoints 3D object points in world coordinates (at least 4 points)
 * @param imagePoints Corresponding 2D image points (same count as objectPoints)
 * @param camera Calibrated fisheye camera model
 * @param[out] rvec Output Rodrigues rotation vector (axis * angle)
 * @param[out] tvec Output translation vector (world to camera)
 * @param useExtrinsicGuess If true, use rvec/tvec as initial guess for refinement
 * @return true if pose estimation succeeded
 *
 * @code
 * Internal::Vec3 rvec, tvec;
 * if (FisheyeSolvePnP(objectPts, imagePts, fisheyeCamera, rvec, tvec)) {
 *     // Transform: P_camera = R * P_world + t
 *     Internal::Mat33 R = RodriguesToMatrix(rvec);
 * }
 * @endcode
 */
QIVISION_API bool FisheyeSolvePnP(
    const std::vector<Point3d>& objectPoints,
    const std::vector<Point2d>& imagePoints,
    const FisheyeCameraModel& camera,
    Internal::Vec3& rvec,
    Internal::Vec3& tvec,
    bool useExtrinsicGuess = false
);

/**
 * @brief Project 3D points to image plane using fisheye model
 *
 * Projects an array of 3D world points to 2D image coordinates using
 * the fisheye camera model and specified extrinsic parameters.
 *
 * Projection pipeline:
 * 1. Transform to camera frame: P_cam = R * P_world + t
 * 2. Project using fisheye model with distortion
 *
 * @param objectPoints 3D points in world coordinates
 * @param camera Fisheye camera model (intrinsics + distortion)
 * @param rvec Rodrigues rotation vector (world to camera)
 * @param tvec Translation vector (world to camera)
 * @return Projected 2D points in pixel coordinates
 *
 * @code
 * auto projectedPts = FisheyeProjectPoints(worldPts, camera, rvec, tvec);
 * // projectedPts[i] is the projection of worldPts[i]
 * @endcode
 */
QIVISION_API std::vector<Point2d> FisheyeProjectPoints(
    const std::vector<Point3d>& objectPoints,
    const FisheyeCameraModel& camera,
    const Internal::Vec3& rvec,
    const Internal::Vec3& tvec
);

/**
 * @brief Compute reprojection errors for fisheye model
 *
 * Computes the Euclidean distance between observed 2D points and
 * their reprojected positions for each point.
 *
 * @param objectPoints 3D points in world coordinates
 * @param imagePoints Observed 2D points in image
 * @param camera Fisheye camera model
 * @param rvec Rodrigues rotation vector
 * @param tvec Translation vector
 * @return Per-point reprojection errors (in pixels)
 *
 * @code
 * auto errors = ComputeFisheyeReprojectionErrors(objPts, imgPts, camera, rvec, tvec);
 * double meanError = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
 * @endcode
 */
QIVISION_API std::vector<double> ComputeFisheyeReprojectionErrors(
    const std::vector<Point3d>& objectPoints,
    const std::vector<Point2d>& imagePoints,
    const FisheyeCameraModel& camera,
    const Internal::Vec3& rvec,
    const Internal::Vec3& tvec
);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Estimate initial intrinsics for fisheye calibration
 *
 * Provides a reasonable initial estimate of camera intrinsics based on
 * image size and expected field of view. Useful when no prior calibration
 * exists.
 *
 * Assumes:
 * - Principal point at image center
 * - Square pixels (fx = fy)
 * - Focal length derived from FOV and image diagonal
 *
 * @param imageSize Image dimensions in pixels
 * @param fovDegrees Expected diagonal field of view in degrees (default: 180)
 * @return Initial intrinsics estimate
 *
 * @code
 * auto initIntr = EstimateFisheyeInitialIntrinsics(Size2i(1920, 1080), 190.0);
 * FisheyeCameraModel initCamera(initIntr, FisheyeDistortion());
 * auto result = CalibrateFisheye(imgPts, objPts, imageSize,
 *                                FisheyeCalibFlags::UseIntrinsicGuess, &initCamera);
 * @endcode
 */
QIVISION_API CameraIntrinsics EstimateFisheyeInitialIntrinsics(
    const Size2i& imageSize,
    double fovDegrees = 180.0
);

/**
 * @brief Compute Jacobian for fisheye projection
 *
 * Computes the 2x3 Jacobian matrix of the projection function with respect
 * to the 3D point coordinates. Used internally for optimization but exposed
 * for advanced users implementing custom optimization.
 *
 * J[i][j] = d(projected_coord_i) / d(point_coord_j)
 *
 * @param p3d 3D point in camera frame
 * @param camera Fisheye camera model
 * @param[out] J Output 2x3 Jacobian matrix (row-major: J[row][col])
 */
QIVISION_API void ComputeFisheyeProjectionJacobian(
    const Point3d& p3d,
    const FisheyeCameraModel& camera,
    double J[2][3]
);

/**
 * @brief Compute Jacobian with respect to camera parameters
 *
 * Computes the Jacobian of the projection with respect to intrinsic
 * and distortion parameters. Used for camera calibration optimization.
 *
 * @param p3d 3D point in camera frame
 * @param camera Fisheye camera model
 * @param[out] Jintr Output Jacobian w.r.t. intrinsics (2x4: fx, fy, cx, cy)
 * @param[out] Jdist Output Jacobian w.r.t. distortion (2x4: k1, k2, k3, k4)
 */
QIVISION_API void ComputeFisheyeParameterJacobian(
    const Point3d& p3d,
    const FisheyeCameraModel& camera,
    double Jintr[2][4],
    double Jdist[2][4]
);

} // namespace Qi::Vision::Calib
