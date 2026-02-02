/**
 * @file CameraModel.cpp
 * @brief Camera intrinsic model implementation
 */

#include <QiVision/Calib/CameraModel.h>
#include <QiVision/Core/Exception.h>

#include <cmath>
#include <algorithm>

namespace Qi::Vision::Calib {

// ============================================================================
// CameraIntrinsics Implementation
// ============================================================================

Internal::Mat33 CameraIntrinsics::ToMatrix() const {
    Internal::Mat33 K;
    K(0, 0) = fx;
    K(0, 1) = 0.0;
    K(0, 2) = cx;
    K(1, 0) = 0.0;
    K(1, 1) = fy;
    K(1, 2) = cy;
    K(2, 0) = 0.0;
    K(2, 1) = 0.0;
    K(2, 2) = 1.0;
    return K;
}

CameraIntrinsics CameraIntrinsics::FromMatrix(const Internal::Mat33& K) {
    return CameraIntrinsics(K(0, 0), K(1, 1), K(0, 2), K(1, 2));
}

// ============================================================================
// CameraModel Implementation
// ============================================================================

Point2d CameraModel::Distort(const Point2d& normalized) const {
    if (!normalized.IsValid()) {
        throw InvalidArgumentException("CameraModel::Distort: invalid point");
    }
    // Brown-Conrady distortion model
    // x' = x * (1 + k1*r^2 + k2*r^4 + k3*r^6) + 2*p1*x*y + p2*(r^2 + 2*x^2)
    // y' = y * (1 + k1*r^2 + k2*r^4 + k3*r^6) + p1*(r^2 + 2*y^2) + 2*p2*x*y

    double x = normalized.x;
    double y = normalized.y;
    double k1 = distortion_.k1;
    double k2 = distortion_.k2;
    double k3 = distortion_.k3;
    double p1 = distortion_.p1;
    double p2 = distortion_.p2;

    double r2 = x * x + y * y;
    double r4 = r2 * r2;
    double r6 = r4 * r2;

    // Radial distortion factor
    double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

    // Tangential distortion
    double dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    double dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

    return Point2d(x * radial + dx, y * radial + dy);
}

Point2d CameraModel::Undistort(const Point2d& distorted, int maxIterations) const {
    if (maxIterations <= 0) {
        throw InvalidArgumentException("CameraModel::Undistort: maxIterations must be > 0");
    }
    if (!distorted.IsValid()) {
        throw InvalidArgumentException("CameraModel::Undistort: invalid point");
    }

    // Newton-Raphson iterative undistortion
    // Start with the distorted point as initial guess
    // Iterate: x_undist = distorted - (distort(x_undist) - distorted)

    if (distortion_.IsZero()) {
        return distorted;
    }

    double x = distorted.x;
    double y = distorted.y;

    // Iterative refinement using fixed-point iteration
    // This is more stable than Newton-Raphson for distortion
    for (int iter = 0; iter < maxIterations; ++iter) {
        double k1 = distortion_.k1;
        double k2 = distortion_.k2;
        double k3 = distortion_.k3;
        double p1 = distortion_.p1;
        double p2 = distortion_.p2;

        double r2 = x * x + y * y;
        double r4 = r2 * r2;
        double r6 = r4 * r2;

        // Radial distortion factor
        double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

        // Tangential distortion
        double dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        double dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

        // Compute distorted coordinates
        double x_dist = x * radial + dx;
        double y_dist = y * radial + dy;

        // Compute error
        double err_x = x_dist - distorted.x;
        double err_y = y_dist - distorted.y;

        // Check convergence
        if (std::abs(err_x) < 1e-10 && std::abs(err_y) < 1e-10) {
            break;
        }

        // Update estimate (fixed-point iteration)
        x = distorted.x - dx;
        y = distorted.y - dy;

        // Recompute with new r2
        r2 = x * x + y * y;
        r4 = r2 * r2;
        r6 = r4 * r2;
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

        // Avoid division by very small numbers
        if (std::abs(radial) > 1e-10) {
            x = (distorted.x - dx) / radial;
            y = (distorted.y - dy) / radial;
        }
    }

    return Point2d(x, y);
}

Point2d CameraModel::ProjectPoint(const Point3d& p3d) const {
    if (!p3d.IsValid()) {
        throw InvalidArgumentException("CameraModel::ProjectPoint: invalid point");
    }
    // Check for points at or behind camera
    if (p3d.z <= 0.0) {
        return Point2d(0.0, 0.0);
    }

    // Normalize to camera plane
    double x_norm = p3d.x / p3d.z;
    double y_norm = p3d.y / p3d.z;

    // Apply distortion
    Point2d distorted = Distort(Point2d(x_norm, y_norm));

    // Convert to pixel coordinates
    double u = intrinsics_.fx * distorted.x + intrinsics_.cx;
    double v = intrinsics_.fy * distorted.y + intrinsics_.cy;

    return Point2d(u, v);
}

Point3d CameraModel::UnprojectPixel(const Point2d& pixel) const {
    if (!pixel.IsValid()) {
        throw InvalidArgumentException("CameraModel::UnprojectPixel: invalid pixel");
    }
    // Convert to normalized coordinates
    double x_norm = (pixel.x - intrinsics_.cx) / intrinsics_.fx;
    double y_norm = (pixel.y - intrinsics_.cy) / intrinsics_.fy;

    // Remove distortion
    Point2d undistorted = Undistort(Point2d(x_norm, y_norm));

    // Return normalized ray (z = 1)
    return Point3d(undistorted.x, undistorted.y, 1.0);
}

} // namespace Qi::Vision::Calib
