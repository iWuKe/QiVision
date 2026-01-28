#pragma once

/**
 * @file CameraModel.h
 * @brief Camera intrinsic model with distortion coefficients
 *
 * Provides:
 * - Camera intrinsics (fx, fy, cx, cy)
 * - Radial and tangential distortion coefficients
 * - Point distortion/undistortion operations
 *
 * Reference: OpenCV camera model
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Internal/Matrix.h>

namespace Qi::Vision::Calib {

/**
 * @brief Camera intrinsic parameters
 *
 * Intrinsic matrix K:
 * | fx  0  cx |
 * | 0  fy  cy |
 * | 0   0   1 |
 */
struct CameraIntrinsics {
    double fx = 1.0;    ///< Focal length X (pixels)
    double fy = 1.0;    ///< Focal length Y (pixels)
    double cx = 0.0;    ///< Principal point X (pixels)
    double cy = 0.0;    ///< Principal point Y (pixels)

    CameraIntrinsics() = default;
    CameraIntrinsics(double fx_, double fy_, double cx_, double cy_)
        : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

    /// Get intrinsic matrix K as Mat33
    Internal::Mat33 ToMatrix() const;

    /// Create from Mat33
    static CameraIntrinsics FromMatrix(const Internal::Mat33& K);
};

/**
 * @brief Lens distortion coefficients
 *
 * Distortion model (Brown-Conrady):
 *   x_distorted = x * (1 + k1*r^2 + k2*r^4 + k3*r^6) + 2*p1*x*y + p2*(r^2 + 2*x^2)
 *   y_distorted = y * (1 + k1*r^2 + k2*r^4 + k3*r^6) + p1*(r^2 + 2*y^2) + 2*p2*x*y
 *
 * where r^2 = x^2 + y^2, (x, y) are normalized coordinates
 */
struct DistortionCoeffs {
    double k1 = 0.0;    ///< Radial distortion k1
    double k2 = 0.0;    ///< Radial distortion k2
    double k3 = 0.0;    ///< Radial distortion k3
    double p1 = 0.0;    ///< Tangential distortion p1
    double p2 = 0.0;    ///< Tangential distortion p2

    DistortionCoeffs() = default;
    DistortionCoeffs(double k1_, double k2_, double p1_, double p2_, double k3_ = 0.0)
        : k1(k1_), k2(k2_), k3(k3_), p1(p1_), p2(p2_) {}

    /// Check if all coefficients are zero (no distortion)
    bool IsZero() const {
        return k1 == 0.0 && k2 == 0.0 && k3 == 0.0 && p1 == 0.0 && p2 == 0.0;
    }
};

/**
 * @brief Complete camera model with intrinsics and distortion
 */
class CameraModel {
public:
    CameraModel() = default;

    CameraModel(const CameraIntrinsics& intr, const DistortionCoeffs& dist,
                const Size2i& imgSize = Size2i(0, 0))
        : intrinsics_(intr), distortion_(dist), imageSize_(imgSize) {}

    // Accessors
    const CameraIntrinsics& Intrinsics() const { return intrinsics_; }
    CameraIntrinsics& Intrinsics() { return intrinsics_; }

    const DistortionCoeffs& Distortion() const { return distortion_; }
    DistortionCoeffs& Distortion() { return distortion_; }

    const Size2i& ImageSize() const { return imageSize_; }
    void SetImageSize(const Size2i& size) { imageSize_ = size; }

    /// Get intrinsic matrix
    Internal::Mat33 GetCameraMatrix() const { return intrinsics_.ToMatrix(); }

    // Point operations

    /**
     * @brief Apply distortion to normalized point
     * @param normalized Normalized image coordinates (x/z, y/z)
     * @return Distorted normalized coordinates
     */
    Point2d Distort(const Point2d& normalized) const;

    /**
     * @brief Remove distortion from normalized point (iterative)
     * @param distorted Distorted normalized coordinates
     * @param maxIterations Maximum iterations for Newton-Raphson
     * @return Undistorted normalized coordinates
     */
    Point2d Undistort(const Point2d& distorted, int maxIterations = 10) const;

    /**
     * @brief Project 3D point to distorted pixel coordinates
     * @param p3d 3D point in camera frame
     * @return 2D pixel coordinates with distortion
     */
    Point2d ProjectPoint(const Point3d& p3d) const;

    /**
     * @brief Unproject pixel to normalized ray (removes distortion)
     * @param pixel Pixel coordinates
     * @return Normalized ray direction (x, y, 1)
     */
    Point3d UnprojectPixel(const Point2d& pixel) const;

    /// Check if camera model is valid
    bool IsValid() const {
        return intrinsics_.fx > 0 && intrinsics_.fy > 0;
    }

private:
    CameraIntrinsics intrinsics_;
    DistortionCoeffs distortion_;
    Size2i imageSize_;
};

} // namespace Qi::Vision::Calib
