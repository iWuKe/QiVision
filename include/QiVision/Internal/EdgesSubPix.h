#pragma once

/**
 * @file EdgesSubPix.h
 * @brief Sub-pixel edge detection (internal use only)
 *
 * Implements HALCON-compatible edges_sub_pix_gray:
 *   A) Sobel gradient field → magnitude terrain (no separate NMS pass)
 *   B) Facet Model + Hessian eigensystem per pixel:
 *      λ_n < 0 (≡ NMS) + |t| ≤ 0.5 (sub-pixel) + dual-threshold flag
 *   C) Greedy chain linking from strong seeds with angle consistency
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
    double angle = 0.0;       ///< Edge normal direction in radians, [0, 2π) (Hessian eigenvector, gradient-hemisphere adjusted)
    double magnitude = 0.0;   ///< Gradient magnitude
    int32_t angleBin = 0;     ///< Quantized tangent direction bin [0, 4), 4 bins over [0, π)

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
 *   2. Sobel 3×3 gradient (scale 0.25) → magnitude terrain
 *   3. Per pixel with magnitude ≥ lowThreshold:
 *      a) 3×3 quadratic surface fit on magnitude
 *      b) Hessian eigensystem → λ_n < 0 (≡ NMS), sub-pixel offset |t|≤0.5
 *      c) Dual-threshold flag: mag ≥ high → strong(2), mag ≥ low → weak(1)
 *   4. Greedy chain linking: strong seeds extend bidirectionally to
 *      angle-consistent neighbors (< 67.5°), accepting weak points
 *
 * EdgePoint::angleBin is always quantized to 4 bins over [0, π) on the
 * tangent direction (fixed, matching the original library convention).
 *
 * @param src           Float grayscale input (contiguous row-major, width×height)
 * @param width         Image width  (pixels)
 * @param height        Image height (pixels)
 * @param highThreshold Strong-edge threshold (seeds hysteresis)
 * @param lowThreshold  Weak-edge threshold for NMS + hysteresis (0 = 0.5×high)
 * @param smoothSigma   Gaussian pre-smoothing sigma (≥ 0; 0 = skip smoothing)
 * @param mask          Optional binary mask (non-zero = include)
 * @param maskStride    Row stride of mask buffer (0 = width)
 * @param angleBins     Angle quantization bins for this level (reserved; internally fixed to 4)
 * @param contrastScale Per-level contrast multiplier (reserved; 1.0 = no-op)
 * @return Vector of sub-pixel edge points, sorted by magnitude descending
 */
std::vector<EdgePoint> EdgesSubPixGray(
    const float* src, int32_t width, int32_t height,
    double highThreshold,
    double lowThreshold  = 0.0,
    double smoothSigma   = 0.5,
    const uint8_t* mask  = nullptr,
    int32_t maskStride   = 0,
    int32_t angleBins    = 0,
    double contrastScale = 1.0);

} // namespace Qi::Vision::Internal
