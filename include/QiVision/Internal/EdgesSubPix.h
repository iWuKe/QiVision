#pragma once

/**
 * @file EdgesSubPix.h
 * @brief Sub-pixel edge detection (internal use only)
 *
 * Implements HALCON-compatible edges_sub_pix_gray:
 *   Gaussian smoothing → Sobel gradient → Gradient-direction NMS →
 *   2D quadratic surface fitting → projection check → convexity check
 *
 * Used by:
 *   - Matching/ShapeModelCreate: extract model edge point set
 *   - Internal/AnglePyramid:     EdgePoint type (re-exported via include)
 *
 * NOT exported as public API.
 */

#include <cstdint>
#include <vector>

namespace Qi::Vision::Internal {

// =============================================================================
// EdgePoint
// =============================================================================

/**
 * @brief Edge point with sub-pixel position and gradient information
 */
struct EdgePoint {
    double x = 0.0;           ///< Sub-pixel X coordinate (column direction)
    double y = 0.0;           ///< Sub-pixel Y coordinate (row direction)
    double angle = 0.0;       ///< Gradient direction in radians, [0, 2π)
    double magnitude = 0.0;   ///< Gradient magnitude
    int32_t angleBin = 0;     ///< Quantized angle bin index [0, numAngleBins)

    EdgePoint() = default;
    EdgePoint(double x_, double y_, double ang, double mag, int32_t bin)
        : x(x_), y(y_), angle(ang), magnitude(mag), angleBin(bin) {}
};

// =============================================================================
// EdgesSubPixGray
// =============================================================================

/**
 * @brief Extract sub-pixel edge points from a float grayscale image
 *
 * Full pipeline:
 *   1. Gaussian pre-smoothing (smoothSigma)
 *   2. Sobel 3×3 gradient → (Gx, Gy), magnitude, direction
 *   3. Gradient-direction NMS with lowThreshold
 *   4. 2D quadratic surface fitting on 3×3 magnitude window:
 *        f(x,y) = a + b·x + f_·y + e·x² + c·y² + d·x·y
 *      Closed-form LS coefficients → extremum (sub_x, sub_y)
 *   5. Projection check: |sub_x·Gx + sub_y·Gy| / mag ≥ 0.5
 *   6. Convexity check:  2e·cos²θ + 2d·cosθ·sinθ + 2c·sin²θ < 0
 *   7. High-threshold filter (magnitude ≥ highThreshold)
 *
 * @param src           Float grayscale input (contiguous row-major, width×height)
 * @param width         Image width  (pixels)
 * @param height        Image height (pixels)
 * @param highThreshold Strong-edge gradient threshold
 * @param lowThreshold  Weak-edge threshold for NMS pre-filtering (0 = highThreshold)
 * @param smoothSigma   Gaussian pre-smoothing sigma (≥ 0; 0 = skip smoothing)
 * @param numAngleBins  Number of angle quantization bins for EdgePoint::angleBin
 * @return Unsorted vector of sub-pixel edge points with magnitude ≥ highThreshold
 */
std::vector<EdgePoint> EdgesSubPixGray(
    const float* src, int32_t width, int32_t height,
    double highThreshold,
    double lowThreshold  = 0.0,
    double smoothSigma   = 0.5,
    int32_t numAngleBins = 16);

} // namespace Qi::Vision::Internal
