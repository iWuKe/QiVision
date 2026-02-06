/**
 * @file FisheyeModel.cpp
 * @brief Fisheye camera model implementation (Kannala-Brandt distortion)
 *
 * Implements the equidistant projection model for fisheye lenses:
 *   theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
 */

#include <QiVision/Calib/FisheyeModel.h>
#include <QiVision/Core/Exception.h>

#include <cmath>
#include <algorithm>
#include <limits>

namespace Qi::Vision::Calib {

// ============================================================================
// FisheyeDistortion Implementation
// ============================================================================

FisheyeDistortion::FisheyeDistortion(double k1_, double k2_, double k3_, double k4_)
    : k1(k1_), k2(k2_), k3(k3_), k4(k4_) {}

bool FisheyeDistortion::IsZero() const {
    return k1 == 0.0 && k2 == 0.0 && k3 == 0.0 && k4 == 0.0;
}

std::array<double, 4> FisheyeDistortion::ToArray() const {
    return {k1, k2, k3, k4};
}

FisheyeDistortion FisheyeDistortion::FromArray(const std::array<double, 4>& arr) {
    return FisheyeDistortion(arr[0], arr[1], arr[2], arr[3]);
}

// ============================================================================
// FisheyeCameraModel Implementation
// ============================================================================

FisheyeCameraModel::FisheyeCameraModel(const CameraIntrinsics& intr,
                                       const FisheyeDistortion& dist,
                                       const Size2i& imgSize)
    : intrinsics_(intr), distortion_(dist), imageSize_(imgSize) {}

// ============================================================================
// Accessors
// ============================================================================

const CameraIntrinsics& FisheyeCameraModel::Intrinsics() const {
    return intrinsics_;
}

CameraIntrinsics& FisheyeCameraModel::Intrinsics() {
    return intrinsics_;
}

const FisheyeDistortion& FisheyeCameraModel::Distortion() const {
    return distortion_;
}

FisheyeDistortion& FisheyeCameraModel::Distortion() {
    return distortion_;
}

const Size2i& FisheyeCameraModel::ImageSize() const {
    return imageSize_;
}

void FisheyeCameraModel::SetImageSize(const Size2i& size) {
    imageSize_ = size;
}

Internal::Mat33 FisheyeCameraModel::GetCameraMatrix() const {
    return intrinsics_.ToMatrix();
}

// ============================================================================
// Private Theta Distortion Functions
// ============================================================================

double FisheyeCameraModel::DistortTheta(double theta) const {
    // theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    double theta2 = theta * theta;
    double theta4 = theta2 * theta2;
    double theta6 = theta4 * theta2;
    double theta8 = theta6 * theta2;

    return theta * (1.0 + distortion_.k1 * theta2
                        + distortion_.k2 * theta4
                        + distortion_.k3 * theta6
                        + distortion_.k4 * theta8);
}

double FisheyeCameraModel::UndistortTheta(double thetaD, int maxIterations) const {
    // Newton-Raphson iteration to solve:
    // f(theta) = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8) - thetaD = 0
    //
    // f'(theta) = 1 + 3*k1*theta^2 + 5*k2*theta^4 + 7*k3*theta^6 + 9*k4*theta^8

    // Handle edge case
    if (std::abs(thetaD) < 1e-12) {
        return thetaD;
    }

    // If no distortion, theta = thetaD
    if (distortion_.IsZero()) {
        return thetaD;
    }

    double k1 = distortion_.k1;
    double k2 = distortion_.k2;
    double k3 = distortion_.k3;
    double k4 = distortion_.k4;

    // Initial guess: theta = thetaD
    double theta = thetaD;

    for (int i = 0; i < maxIterations; ++i) {
        double theta2 = theta * theta;
        double theta4 = theta2 * theta2;
        double theta6 = theta4 * theta2;
        double theta8 = theta6 * theta2;

        // f(theta) = theta * (1 + k1*theta^2 + ...) - thetaD
        double f = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8) - thetaD;

        // f'(theta) = 1 + 3*k1*theta^2 + 5*k2*theta^4 + 7*k3*theta^6 + 9*k4*theta^8
        double df = 1.0 + 3.0 * k1 * theta2
                        + 5.0 * k2 * theta4
                        + 7.0 * k3 * theta6
                        + 9.0 * k4 * theta8;

        // Avoid division by very small derivative
        if (std::abs(df) < 1e-12) {
            break;
        }

        double delta = f / df;
        theta -= delta;

        // Check convergence
        if (std::abs(delta) < 1e-12) {
            break;
        }
    }

    return theta;
}

// ============================================================================
// Core Projection Operations
// ============================================================================

Point2d FisheyeCameraModel::Distort(const Point2d& normalized) const {
    if (!normalized.IsValid()) {
        throw InvalidArgumentException("FisheyeCameraModel::Distort: invalid point");
    }

    double x = normalized.x;
    double y = normalized.y;
    double r = std::sqrt(x * x + y * y);

    // At center, no distortion
    if (r < 1e-10) {
        return normalized;
    }

    // theta = atan(r), where r is the distance in normalized coordinates
    double theta = std::atan(r);

    // Apply theta distortion
    double thetaD = DistortTheta(theta);

    // Scale factor to convert from undistorted to distorted normalized coordinates
    double scale = thetaD / r;

    return Point2d(x * scale, y * scale);
}

Point2d FisheyeCameraModel::Undistort(const Point2d& distorted, int maxIterations) const {
    if (maxIterations <= 0) {
        throw InvalidArgumentException("FisheyeCameraModel::Undistort: maxIterations must be > 0");
    }
    if (!distorted.IsValid()) {
        throw InvalidArgumentException("FisheyeCameraModel::Undistort: invalid point");
    }

    double xD = distorted.x;
    double yD = distorted.y;
    double rD = std::sqrt(xD * xD + yD * yD);

    // At center, no distortion
    if (rD < 1e-10) {
        return distorted;
    }

    // If no distortion, just return
    if (distortion_.IsZero()) {
        return distorted;
    }

    // theta_d = r_d for fisheye in normalized distorted coordinates
    double thetaD = rD;

    // Solve for undistorted theta
    double theta = UndistortTheta(thetaD, maxIterations);

    // Undistorted r in normalized coordinates: r = tan(theta)
    double r = std::tan(theta);

    // Avoid issues with very large theta (near 90 degrees)
    // tan approaches infinity at pi/2
    if (!std::isfinite(r) || std::abs(theta) > 1.5) {
        // Fallback: use a safe approximation for extreme angles
        r = theta / std::cos(theta);
        if (!std::isfinite(r)) {
            r = rD; // Last resort: return distorted
        }
    }

    // Scale factor to convert from distorted to undistorted
    double scale = r / rD;

    return Point2d(xD * scale, yD * scale);
}

Point2d FisheyeCameraModel::ProjectPoint(const Point3d& p3d) const {
    if (!p3d.IsValid()) {
        throw InvalidArgumentException("FisheyeCameraModel::ProjectPoint: invalid point");
    }

    // Points at or behind camera
    if (p3d.z <= 0.0) {
        double nan = std::numeric_limits<double>::quiet_NaN();
        return Point2d(nan, nan);
    }

    // 1. Normalize to camera plane (z = 1)
    double x = p3d.x / p3d.z;
    double y = p3d.y / p3d.z;

    // 2. Apply fisheye distortion
    Point2d distorted = Distort(Point2d(x, y));

    // 3. Convert to pixel coordinates
    double u = intrinsics_.fx * distorted.x + intrinsics_.cx;
    double v = intrinsics_.fy * distorted.y + intrinsics_.cy;

    return Point2d(u, v);
}

Point3d FisheyeCameraModel::UnprojectPixel(const Point2d& pixel) const {
    if (!pixel.IsValid()) {
        throw InvalidArgumentException("FisheyeCameraModel::UnprojectPixel: invalid pixel");
    }
    if (!IsValid()) {
        throw InvalidArgumentException("FisheyeCameraModel::UnprojectPixel: invalid intrinsics");
    }

    // 1. Convert to normalized coordinates (remove intrinsics)
    double x = (pixel.x - intrinsics_.cx) / intrinsics_.fx;
    double y = (pixel.y - intrinsics_.cy) / intrinsics_.fy;

    // 2. Remove distortion
    Point2d undistorted = Undistort(Point2d(x, y));

    // 3. Return as 3D ray (unit z-component)
    return Point3d(undistorted.x, undistorted.y, 1.0);
}

// ============================================================================
// Utility Functions
// ============================================================================

bool FisheyeCameraModel::IsValid() const {
    return intrinsics_.fx > 0 && intrinsics_.fy > 0;
}

double FisheyeCameraModel::HorizontalFOV() const {
    if (imageSize_.width <= 0 || intrinsics_.fx <= 0) {
        return 0.0;
    }

    // Distance from principal point to image edge in normalized coordinates
    double left = intrinsics_.cx;
    double right = static_cast<double>(imageSize_.width - 1) - intrinsics_.cx;
    double maxHalf = std::max(left, right);
    double xEdge = maxHalf / intrinsics_.fx;

    // For fisheye: theta_d = r_d in normalized distorted coordinates
    // We need to find the undistorted angle that corresponds to the edge

    // The distorted normalized coordinate at the edge
    // Since we're at the horizontal edge, y = 0
    double thetaD = std::abs(xEdge);

    // Undistort to get the actual field angle
    double theta = UndistortTheta(thetaD, 15);

    // Full horizontal FOV is 2 * theta
    return 2.0 * theta;
}

double FisheyeCameraModel::VerticalFOV() const {
    if (imageSize_.height <= 0 || intrinsics_.fy <= 0) {
        return 0.0;
    }

    // Distance from principal point to image edge in normalized coordinates
    double top = intrinsics_.cy;
    double bottom = static_cast<double>(imageSize_.height - 1) - intrinsics_.cy;
    double maxHalf = std::max(top, bottom);
    double yEdge = maxHalf / intrinsics_.fy;

    // For fisheye: theta_d = r_d in normalized distorted coordinates
    double thetaD = std::abs(yEdge);

    // Undistort to get the actual field angle
    double theta = UndistortTheta(thetaD, 15);

    // Full vertical FOV is 2 * theta
    return 2.0 * theta;
}

// ============================================================================
// OpenCV Compatibility Functions
// ============================================================================

FisheyeCameraModel FisheyeFromOpenCV(
    const Internal::Mat33& K,
    const std::array<double, 4>& D,
    const Size2i& imageSize)
{
    CameraIntrinsics intr = CameraIntrinsics::FromMatrix(K);
    FisheyeDistortion dist = FisheyeDistortion::FromArray(D);
    return FisheyeCameraModel(intr, dist, imageSize);
}

void FisheyeToOpenCV(
    const FisheyeCameraModel& model,
    Internal::Mat33& K,
    std::array<double, 4>& D)
{
    K = model.GetCameraMatrix();
    D = model.Distortion().ToArray();
}

} // namespace Qi::Vision::Calib
