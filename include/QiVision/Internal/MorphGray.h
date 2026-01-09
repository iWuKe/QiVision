#pragma once

/**
 * @file MorphGray.h
 * @brief Gray-scale morphological operations on images
 *
 * This module provides gray-scale morphological operations that work
 * directly on image intensity values, unlike binary morphology which
 * operates on regions.
 *
 * Operations include:
 * - Basic: dilation, erosion
 * - Compound: opening, closing
 * - Derived: gradient, top-hat, black-hat
 * - Reconstruction-based operations
 *
 * Reference Halcon operators:
 * - gray_dilation, gray_erosion
 * - gray_opening, gray_closing
 * - gray_tophat, gray_bothat (blackhat)
 * - gray_range_rect, gray_range_circle
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Internal/StructElement.h>

namespace Qi::Vision::Internal {

// =============================================================================
// Basic Gray Morphology
// =============================================================================

/**
 * @brief Gray-scale dilation
 *
 * For each pixel, takes the maximum value in the SE neighborhood.
 * Expands bright regions and fills dark holes.
 *
 * @param src Input grayscale image (8-bit)
 * @param se Structuring element
 * @return Dilated image
 */
QImage GrayDilate(const QImage& src, const StructElement& se);

/**
 * @brief Gray-scale erosion
 *
 * For each pixel, takes the minimum value in the SE neighborhood.
 * Expands dark regions and removes bright spots.
 *
 * @param src Input grayscale image (8-bit)
 * @param se Structuring element
 * @return Eroded image
 */
QImage GrayErode(const QImage& src, const StructElement& se);

/**
 * @brief Gray-scale dilation with rectangular SE
 *
 * Optimized implementation using separable filtering.
 *
 * @param src Input grayscale image
 * @param width SE width
 * @param height SE height
 * @return Dilated image
 */
QImage GrayDilateRect(const QImage& src, int32_t width, int32_t height);

/**
 * @brief Gray-scale erosion with rectangular SE
 *
 * Optimized implementation using separable filtering.
 *
 * @param src Input grayscale image
 * @param width SE width
 * @param height SE height
 * @return Eroded image
 */
QImage GrayErodeRect(const QImage& src, int32_t width, int32_t height);

/**
 * @brief Gray-scale dilation with circular SE
 *
 * @param src Input grayscale image
 * @param radius Circle radius
 * @return Dilated image
 */
QImage GrayDilateCircle(const QImage& src, int32_t radius);

/**
 * @brief Gray-scale erosion with circular SE
 *
 * @param src Input grayscale image
 * @param radius Circle radius
 * @return Eroded image
 */
QImage GrayErodeCircle(const QImage& src, int32_t radius);

// =============================================================================
// Compound Operations
// =============================================================================

/**
 * @brief Gray-scale opening (erosion followed by dilation)
 *
 * Removes bright spots smaller than SE, smooths bright peaks.
 *
 * @param src Input grayscale image
 * @param se Structuring element
 * @return Opened image
 */
QImage GrayOpening(const QImage& src, const StructElement& se);

/**
 * @brief Gray-scale closing (dilation followed by erosion)
 *
 * Removes dark spots smaller than SE, fills dark valleys.
 *
 * @param src Input grayscale image
 * @param se Structuring element
 * @return Closed image
 */
QImage GrayClosing(const QImage& src, const StructElement& se);

/**
 * @brief Gray-scale opening with rectangular SE
 *
 * @param src Input grayscale image
 * @param width SE width
 * @param height SE height
 * @return Opened image
 */
QImage GrayOpeningRect(const QImage& src, int32_t width, int32_t height);

/**
 * @brief Gray-scale closing with rectangular SE
 *
 * @param src Input grayscale image
 * @param width SE width
 * @param height SE height
 * @return Closed image
 */
QImage GrayClosingRect(const QImage& src, int32_t width, int32_t height);

/**
 * @brief Gray-scale opening with circular SE
 *
 * @param src Input grayscale image
 * @param radius Circle radius
 * @return Opened image
 */
QImage GrayOpeningCircle(const QImage& src, int32_t radius);

/**
 * @brief Gray-scale closing with circular SE
 *
 * @param src Input grayscale image
 * @param radius Circle radius
 * @return Closed image
 */
QImage GrayClosingCircle(const QImage& src, int32_t radius);

// =============================================================================
// Derived Operations
// =============================================================================

/**
 * @brief Morphological gradient (dilation - erosion)
 *
 * Highlights edges by showing the local contrast.
 *
 * @param src Input grayscale image
 * @param se Structuring element
 * @return Gradient image
 */
QImage GrayMorphGradient(const QImage& src, const StructElement& se);

/**
 * @brief Internal gradient (original - erosion)
 *
 * Shows inner edge of bright regions.
 *
 * @param src Input grayscale image
 * @param se Structuring element
 * @return Internal gradient image
 */
QImage GrayInternalGradient(const QImage& src, const StructElement& se);

/**
 * @brief External gradient (dilation - original)
 *
 * Shows outer edge of bright regions.
 *
 * @param src Input grayscale image
 * @param se Structuring element
 * @return External gradient image
 */
QImage GrayExternalGradient(const QImage& src, const StructElement& se);

/**
 * @brief White top-hat (original - opening)
 *
 * Extracts bright features smaller than SE against a dark background.
 * Useful for background correction and small bright spot detection.
 *
 * @param src Input grayscale image
 * @param se Structuring element
 * @return Top-hat image
 */
QImage GrayTopHat(const QImage& src, const StructElement& se);

/**
 * @brief Black top-hat / bottom-hat (closing - original)
 *
 * Extracts dark features smaller than SE against a bright background.
 * Useful for background correction and small dark spot detection.
 *
 * @param src Input grayscale image
 * @param se Structuring element
 * @return Black-hat image
 */
QImage GrayBlackHat(const QImage& src, const StructElement& se);

// =============================================================================
// Range Operations (Local Contrast)
// =============================================================================

/**
 * @brief Local range (max - min) with rectangular window
 *
 * Computes local contrast/range at each pixel.
 * Result = GrayDilate(src) - GrayErode(src)
 *
 * @param src Input grayscale image
 * @param width Window width
 * @param height Window height
 * @return Range image
 */
QImage GrayRangeRect(const QImage& src, int32_t width, int32_t height);

/**
 * @brief Local range with circular window
 *
 * @param src Input grayscale image
 * @param radius Circle radius
 * @return Range image
 */
QImage GrayRangeCircle(const QImage& src, int32_t radius);

// =============================================================================
// Iterative Operations
// =============================================================================

/**
 * @brief Apply gray dilation N times
 *
 * @param src Input grayscale image
 * @param se Structuring element
 * @param iterations Number of iterations
 * @return Result after N dilations
 */
QImage GrayDilateN(const QImage& src, const StructElement& se, int iterations);

/**
 * @brief Apply gray erosion N times
 *
 * @param src Input grayscale image
 * @param se Structuring element
 * @param iterations Number of iterations
 * @return Result after N erosions
 */
QImage GrayErodeN(const QImage& src, const StructElement& se, int iterations);

/**
 * @brief Apply gray opening N times
 *
 * @param src Input grayscale image
 * @param se Structuring element
 * @param iterations Number of iterations
 * @return Result after N openings
 */
QImage GrayOpeningN(const QImage& src, const StructElement& se, int iterations);

/**
 * @brief Apply gray closing N times
 *
 * @param src Input grayscale image
 * @param se Structuring element
 * @param iterations Number of iterations
 * @return Result after N closings
 */
QImage GrayClosingN(const QImage& src, const StructElement& se, int iterations);

// =============================================================================
// Geodesic Operations
// =============================================================================

/**
 * @brief Geodesic gray dilation
 *
 * Dilate marker constrained by mask (pointwise minimum with mask).
 * Result = min(GrayDilate(marker, se), mask)
 *
 * @param marker Marker image (seed)
 * @param mask Constraint mask (upper bound)
 * @param se Structuring element
 * @return Geodesic dilated image
 */
QImage GrayGeodesicDilate(const QImage& marker,
                          const QImage& mask,
                          const StructElement& se);

/**
 * @brief Geodesic gray erosion
 *
 * Erode marker constrained by mask (pointwise maximum with mask).
 * Result = max(GrayErode(marker, se), mask)
 *
 * @param marker Marker image
 * @param mask Constraint mask (lower bound)
 * @param se Structuring element
 * @return Geodesic eroded image
 */
QImage GrayGeodesicErode(const QImage& marker,
                         const QImage& mask,
                         const StructElement& se);

/**
 * @brief Morphological reconstruction by dilation
 *
 * Iteratively apply geodesic dilation until stable.
 * Recovers connected bright regions from marker within mask bounds.
 *
 * @param marker Marker image (seed, must be <= mask)
 * @param mask Constraint mask (upper bound)
 * @return Reconstructed image
 */
QImage GrayReconstructByDilation(const QImage& marker, const QImage& mask);

/**
 * @brief Morphological reconstruction by erosion
 *
 * Iteratively apply geodesic erosion until stable.
 *
 * @param marker Marker image (seed, must be >= mask)
 * @param mask Constraint mask (lower bound)
 * @return Reconstructed image
 */
QImage GrayReconstructByErosion(const QImage& marker, const QImage& mask);

/**
 * @brief Opening by reconstruction
 *
 * Erosion followed by reconstruction by dilation from eroded result.
 * Preserves shapes of objects that survive erosion.
 *
 * @param src Input grayscale image
 * @param se Structuring element
 * @return Opening by reconstruction result
 */
QImage GrayOpeningByReconstruction(const QImage& src, const StructElement& se);

/**
 * @brief Closing by reconstruction
 *
 * Dilation followed by reconstruction by erosion from dilated result.
 * Preserves shapes of background regions.
 *
 * @param src Input grayscale image
 * @param se Structuring element
 * @return Closing by reconstruction result
 */
QImage GrayClosingByReconstruction(const QImage& src, const StructElement& se);

/**
 * @brief Fill holes in grayscale image
 *
 * Fills dark regions that are completely surrounded by brighter pixels.
 *
 * @param src Input grayscale image
 * @return Image with holes filled
 */
QImage GrayFillHoles(const QImage& src);

// =============================================================================
// Background Correction
// =============================================================================

/**
 * @brief Rolling ball background subtraction
 *
 * Uses morphological opening with large SE to estimate background,
 * then subtracts it from original.
 *
 * @param src Input grayscale image
 * @param radius Ball radius (larger = smoother background)
 * @return Background-corrected image
 */
QImage RollingBallBackground(const QImage& src, int32_t radius);

/**
 * @brief Estimate background using opening
 *
 * @param src Input grayscale image
 * @param se Structuring element
 * @return Estimated background
 */
QImage EstimateBackground(const QImage& src, const StructElement& se);

/**
 * @brief Subtract background from image
 *
 * Result = saturate(src - background + offset)
 *
 * @param src Input image
 * @param background Background image
 * @param offset Offset to add (default 0)
 * @return Background-subtracted image
 */
QImage SubtractBackground(const QImage& src,
                          const QImage& background,
                          int32_t offset = 0);

} // namespace Qi::Vision::Internal
