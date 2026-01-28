#pragma once

/**
 * @file PolarTransform.h
 * @brief Polar coordinate transformation
 *
 * Provides:
 * - Cartesian to Polar transformation
 * - Polar to Cartesian transformation (inverse)
 * - Linear and Semi-Log modes
 *
 * Reference: OpenCV warpPolar implementation
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Internal/Interpolate.h>

namespace Qi::Vision::Internal {

/**
 * @brief Polar transformation mode
 */
enum class PolarMode {
    Linear,     ///< Linear radial mapping: rho = r
    SemiLog     ///< Semi-log radial mapping: rho = M * log(r + 1)
};

/**
 * @brief Warp image to/from polar coordinates
 *
 * Maps image from Cartesian to Polar coordinates:
 * - x axis (dst) = angle [0, 2*pi)
 * - y axis (dst) = radius [0, maxRadius]
 *
 * Or inverse: Polar to Cartesian.
 *
 * @param src Source image
 * @param center Center point for polar transformation
 * @param maxRadius Maximum radius to map
 * @param dstWidth Output width (angle resolution). 0 = auto (2*pi * maxRadius)
 * @param dstHeight Output height (radial resolution). 0 = auto (maxRadius)
 * @param mode Polar mapping mode (Linear or SemiLog)
 * @param inverse If true, transform from Polar back to Cartesian
 * @param method Interpolation method
 * @param borderMode Border handling mode
 * @param borderValue Value for constant border
 * @return Transformed image
 */
QImage WarpPolar(
    const QImage& src,
    const Point2d& center,
    double maxRadius,
    int32_t dstWidth = 0,
    int32_t dstHeight = 0,
    PolarMode mode = PolarMode::Linear,
    bool inverse = false,
    InterpolationMethod method = InterpolationMethod::Bilinear,
    BorderMode borderMode = BorderMode::Constant,
    double borderValue = 0.0
);

/**
 * @brief Convert Cartesian point to polar coordinates
 *
 * @param pt Point in Cartesian coordinates
 * @param center Center of polar system
 * @return Point where x = angle (radians), y = radius
 */
Point2d CartesianToPolar(const Point2d& pt, const Point2d& center);

/**
 * @brief Convert polar coordinates to Cartesian point
 *
 * @param r Radius
 * @param theta Angle in radians
 * @param center Center of polar system
 * @return Point in Cartesian coordinates
 */
Point2d PolarToCartesian(double r, double theta, const Point2d& center);

/**
 * @brief Linear to log-polar mapping
 * @param r Linear radius
 * @param M Log scale factor
 * @return Log-polar radius
 */
inline double LinearToLogPolar(double r, double M) {
    return M * std::log(r + 1.0);
}

/**
 * @brief Log-polar to linear mapping
 * @param rho Log-polar radius
 * @param M Log scale factor
 * @return Linear radius
 */
inline double LogPolarToLinear(double rho, double M) {
    return std::exp(rho / M) - 1.0;
}

} // namespace Qi::Vision::Internal
