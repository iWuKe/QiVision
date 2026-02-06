#pragma once

/**
 * @file FisheyeModel.h
 * @brief Fisheye camera model with Kannala-Brandt distortion
 *
 * Implements the equidistant projection model commonly used for fisheye lenses:
 *   theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
 *
 * Where:
 *   - theta = atan(r), r = sqrt(x^2 + y^2) in normalized coordinates
 *   - theta_d is the distorted angle
 *   - k1, k2, k3, k4 are radial distortion coefficients
 *
 * This model can handle field of view > 180 degrees and is widely used
 * in robotics, automotive, and surveillance applications.
 *
 * Reference:
 * - J. Kannala and S. Brandt, "A generic camera model and calibration method
 *   for conventional, wide-angle, and fish-eye lenses", IEEE TPAMI, 2006
 * - OpenCV fisheye module (cv::fisheye)
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Calib/CameraModel.h>
#include <QiVision/Internal/Matrix.h>
#include <QiVision/Core/Export.h>
#include <array>

namespace Qi::Vision::Calib {

/**
 * @brief Fisheye distortion coefficients (Kannala-Brandt model)
 *
 * Equidistant projection distortion model:
 *   theta = atan(r)  where r = sqrt(x^2 + y^2) in normalized coordinates
 *   theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
 *   x_distorted = (theta_d / r) * x
 *   y_distorted = (theta_d / r) * y
 *
 * This model is suitable for wide-angle and fisheye lenses (FOV > 180 degrees).
 * Unlike standard Brown-Conrady model, this uses the incidence angle theta
 * rather than the radial distance r for distortion modeling.
 */
struct QIVISION_API FisheyeDistortion {
    double k1 = 0.0;    ///< Radial distortion coefficient k1 (quadratic term)
    double k2 = 0.0;    ///< Radial distortion coefficient k2 (quartic term)
    double k3 = 0.0;    ///< Radial distortion coefficient k3 (sextic term)
    double k4 = 0.0;    ///< Radial distortion coefficient k4 (octic term)

    /// Default constructor (zero distortion)
    FisheyeDistortion() = default;

    /**
     * @brief Construct with distortion coefficients
     * @param k1_ Quadratic coefficient
     * @param k2_ Quartic coefficient
     * @param k3_ Sextic coefficient (default: 0)
     * @param k4_ Octic coefficient (default: 0)
     */
    FisheyeDistortion(double k1_, double k2_, double k3_ = 0.0, double k4_ = 0.0);

    /**
     * @brief Check if all coefficients are zero (no distortion)
     * @return true if distortion-free model
     */
    bool IsZero() const;

    /**
     * @brief Convert to array format [k1, k2, k3, k4]
     * @return Array of 4 distortion coefficients
     */
    std::array<double, 4> ToArray() const;

    /**
     * @brief Create from array format [k1, k2, k3, k4]
     * @param arr Array of 4 distortion coefficients
     * @return FisheyeDistortion object
     */
    static FisheyeDistortion FromArray(const std::array<double, 4>& arr);
};

/**
 * @brief Fisheye camera model with equidistant projection
 *
 * This class combines camera intrinsics with fisheye distortion to provide
 * a complete fisheye camera model. It supports:
 * - 3D to 2D projection with distortion
 * - 2D to 3D ray unprojection with undistortion
 * - Field of view calculation
 *
 * Projection pipeline:
 * 1. 3D point (X, Y, Z) -> normalized (x, y) = (X/Z, Y/Z)
 * 2. Compute angle: theta = atan(sqrt(x^2 + y^2))
 * 3. Apply distortion: theta_d = theta * (1 + k1*theta^2 + ...)
 * 4. Project to image: u = fx * theta_d * x/r + cx, v = fy * theta_d * y/r + cy
 *
 * @note Uses CameraIntrinsics from standard camera model for compatibility
 */
class QIVISION_API FisheyeCameraModel {
public:
    /// Default constructor
    FisheyeCameraModel() = default;

    /**
     * @brief Construct with intrinsics, distortion, and optional image size
     * @param intr Camera intrinsic parameters (fx, fy, cx, cy)
     * @param dist Fisheye distortion coefficients
     * @param imgSize Image dimensions (optional, for FOV calculation)
     */
    FisheyeCameraModel(const CameraIntrinsics& intr, const FisheyeDistortion& dist,
                       const Size2i& imgSize = Size2i(0, 0));

    // =========================================================================
    // Accessors
    // =========================================================================

    /**
     * @brief Get intrinsic parameters (const)
     * @return Reference to camera intrinsics
     */
    const CameraIntrinsics& Intrinsics() const;

    /**
     * @brief Get intrinsic parameters (mutable)
     * @return Reference to camera intrinsics
     */
    CameraIntrinsics& Intrinsics();

    /**
     * @brief Get distortion coefficients (const)
     * @return Reference to fisheye distortion
     */
    const FisheyeDistortion& Distortion() const;

    /**
     * @brief Get distortion coefficients (mutable)
     * @return Reference to fisheye distortion
     */
    FisheyeDistortion& Distortion();

    /**
     * @brief Get image size
     * @return Image dimensions
     */
    const Size2i& ImageSize() const;

    /**
     * @brief Set image size
     * @param size Image dimensions
     */
    void SetImageSize(const Size2i& size);

    /**
     * @brief Get intrinsic matrix K as 3x3 matrix
     * @return Camera intrinsic matrix
     *         | fx  0  cx |
     *         | 0  fy  cy |
     *         | 0   0   1 |
     */
    Internal::Mat33 GetCameraMatrix() const;

    // =========================================================================
    // Core Projection Operations
    // =========================================================================

    /**
     * @brief Apply fisheye distortion to normalized point
     *
     * Converts undistorted normalized coordinates to distorted normalized
     * coordinates using the equidistant projection model.
     *
     * @param normalized Undistorted normalized coordinates (x/z, y/z)
     * @return Distorted normalized coordinates
     */
    Point2d Distort(const Point2d& normalized) const;

    /**
     * @brief Remove fisheye distortion from normalized point
     *
     * Uses Newton-Raphson iteration to find the undistorted normalized
     * coordinates from distorted coordinates. The iteration solves:
     *   theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
     *
     * @param distorted Distorted normalized coordinates
     * @param maxIterations Maximum iterations for Newton-Raphson (default: 15)
     * @return Undistorted normalized coordinates
     */
    Point2d Undistort(const Point2d& distorted, int maxIterations = 15) const;

    /**
     * @brief Project 3D point to distorted pixel coordinates
     *
     * Projects a 3D point in camera frame to 2D pixel coordinates,
     * applying the fisheye distortion model.
     *
     * @param p3d 3D point in camera frame (Z should be positive)
     * @return 2D pixel coordinates with distortion
     */
    Point2d ProjectPoint(const Point3d& p3d) const;

    /**
     * @brief Unproject pixel to normalized ray
     *
     * Converts a pixel coordinate to a 3D ray direction in camera frame.
     * Removes fisheye distortion and returns a normalized ray.
     *
     * @param pixel Pixel coordinates
     * @return Normalized ray direction (x, y, 1) - unit z-component
     */
    Point3d UnprojectPixel(const Point2d& pixel) const;

    // =========================================================================
    // Utility Functions
    // =========================================================================

    /**
     * @brief Check if camera model is valid
     * @return true if focal lengths are positive
     */
    bool IsValid() const;

    /**
     * @brief Get horizontal field of view
     *
     * Computes the horizontal FOV based on image width and focal length.
     * Accounts for fisheye distortion at image edges.
     *
     * @return Horizontal FOV in radians
     */
    double HorizontalFOV() const;

    /**
     * @brief Get vertical field of view
     *
     * Computes the vertical FOV based on image height and focal length.
     * Accounts for fisheye distortion at image edges.
     *
     * @return Vertical FOV in radians
     */
    double VerticalFOV() const;

private:
    CameraIntrinsics intrinsics_;   ///< Camera intrinsic parameters
    FisheyeDistortion distortion_;  ///< Fisheye distortion coefficients
    Size2i imageSize_;              ///< Image dimensions

    /**
     * @brief Apply distortion to theta angle
     * @param theta Undistorted incidence angle
     * @return Distorted angle theta_d
     */
    double DistortTheta(double theta) const;

    /**
     * @brief Remove distortion from theta angle (Newton-Raphson)
     * @param thetaD Distorted incidence angle
     * @param maxIterations Maximum iterations
     * @return Undistorted angle theta
     */
    double UndistortTheta(double thetaD, int maxIterations) const;
};

// =============================================================================
// OpenCV Compatibility Functions
// =============================================================================

/**
 * @brief Convert OpenCV fisheye parameters to FisheyeCameraModel
 *
 * Creates a FisheyeCameraModel from OpenCV-style parameters.
 * OpenCV fisheye uses the same equidistant projection model with
 * 4 distortion coefficients (k1, k2, k3, k4).
 *
 * @param K 3x3 camera intrinsic matrix
 * @param D Distortion coefficients [k1, k2, k3, k4]
 * @param imageSize Image size in pixels (optional)
 * @return Fisheye camera model
 *
 * @code
 * Internal::Mat33 K = Internal::CameraIntrinsic(fx, fy, cx, cy);
 * std::array<double, 4> D = {k1, k2, k3, k4};
 * auto model = FisheyeFromOpenCV(K, D, Size2i(1920, 1080));
 * @endcode
 */
QIVISION_API FisheyeCameraModel FisheyeFromOpenCV(
    const Internal::Mat33& K,
    const std::array<double, 4>& D,
    const Size2i& imageSize = Size2i(0, 0)
);

/**
 * @brief Convert FisheyeCameraModel to OpenCV format
 *
 * Exports the camera model to OpenCV-compatible format.
 *
 * @param model Fisheye camera model
 * @param[out] K Output 3x3 camera intrinsic matrix
 * @param[out] D Output distortion coefficients [k1, k2, k3, k4]
 *
 * @code
 * Internal::Mat33 K;
 * std::array<double, 4> D;
 * FisheyeToOpenCV(model, K, D);
 * @endcode
 */
QIVISION_API void FisheyeToOpenCV(
    const FisheyeCameraModel& model,
    Internal::Mat33& K,
    std::array<double, 4>& D
);

} // namespace Qi::Vision::Calib
