#pragma once

/**
 * @file MorphBinary.h
 * @brief Binary morphological operations on regions
 *
 * This module provides:
 * - Basic operations: dilation, erosion
 * - Compound operations: opening, closing
 * - Derived operations: gradient, tophat, blackhat
 * - Hit-or-miss transform
 * - Skeleton and thinning
 *
 * Reference Halcon operators:
 * - dilation1, erosion1, dilation_circle, erosion_circle
 * - opening, closing, opening_circle, closing_circle
 * - boundary, skeleton, thinning
 * - hit_or_miss
 */

#include <QiVision/Core/QRegion.h>
#include <QiVision/Internal/StructElement.h>

#include <vector>

namespace Qi::Vision::Internal {

// =============================================================================
// Basic Morphological Operations
// =============================================================================

/**
 * @brief Dilate region with structuring element
 *
 * Dilation expands the region by adding pixels where the SE overlaps.
 * Result = Union of all translations of SE where SE origin touches region.
 *
 * @param region Input region
 * @param se Structuring element
 * @return Dilated region
 */
QRegion Dilate(const QRegion& region, const StructElement& se);

/**
 * @brief Erode region with structuring element
 *
 * Erosion shrinks the region by removing boundary pixels.
 * Result = Set of all positions where SE fits entirely within region.
 *
 * @param region Input region
 * @param se Structuring element
 * @return Eroded region
 */
QRegion Erode(const QRegion& region, const StructElement& se);

/**
 * @brief Dilate region with rectangular SE
 *
 * @param region Input region
 * @param width SE width
 * @param height SE height
 * @return Dilated region
 */
QRegion DilateRect(const QRegion& region, int32_t width, int32_t height);

/**
 * @brief Erode region with rectangular SE
 *
 * @param region Input region
 * @param width SE width
 * @param height SE height
 * @return Eroded region
 */
QRegion ErodeRect(const QRegion& region, int32_t width, int32_t height);

/**
 * @brief Dilate region with circular SE
 *
 * @param region Input region
 * @param radius Circle radius
 * @return Dilated region
 */
QRegion DilateCircle(const QRegion& region, int32_t radius);

/**
 * @brief Erode region with circular SE
 *
 * @param region Input region
 * @param radius Circle radius
 * @return Eroded region
 */
QRegion ErodeCircle(const QRegion& region, int32_t radius);

// =============================================================================
// Compound Operations
// =============================================================================

/**
 * @brief Opening operation (erosion followed by dilation)
 *
 * Opening removes small protrusions and separates weakly connected regions.
 * Opening = Dilate(Erode(region, se), se)
 *
 * @param region Input region
 * @param se Structuring element
 * @return Opened region
 */
QRegion Opening(const QRegion& region, const StructElement& se);

/**
 * @brief Closing operation (dilation followed by erosion)
 *
 * Closing fills small holes and connects nearby regions.
 * Closing = Erode(Dilate(region, se), se)
 *
 * @param region Input region
 * @param se Structuring element
 * @return Closed region
 */
QRegion Closing(const QRegion& region, const StructElement& se);

/**
 * @brief Opening with rectangular SE
 *
 * @param region Input region
 * @param width SE width
 * @param height SE height
 * @return Opened region
 */
QRegion OpeningRect(const QRegion& region, int32_t width, int32_t height);

/**
 * @brief Closing with rectangular SE
 *
 * @param region Input region
 * @param width SE width
 * @param height SE height
 * @return Closed region
 */
QRegion ClosingRect(const QRegion& region, int32_t width, int32_t height);

/**
 * @brief Opening with circular SE
 *
 * @param region Input region
 * @param radius Circle radius
 * @return Opened region
 */
QRegion OpeningCircle(const QRegion& region, int32_t radius);

/**
 * @brief Closing with circular SE
 *
 * @param region Input region
 * @param radius Circle radius
 * @return Closed region
 */
QRegion ClosingCircle(const QRegion& region, int32_t radius);

// =============================================================================
// Derived Operations
// =============================================================================

/**
 * @brief Morphological gradient
 *
 * Highlights region boundaries.
 * Gradient = Dilate(region, se) - Erode(region, se)
 *
 * @param region Input region
 * @param se Structuring element
 * @return Gradient region (boundary)
 */
QRegion MorphGradient(const QRegion& region, const StructElement& se);

/**
 * @brief Internal gradient
 *
 * InternalGradient = region - Erode(region, se)
 *
 * @param region Input region
 * @param se Structuring element
 * @return Internal gradient region
 */
QRegion InternalGradient(const QRegion& region, const StructElement& se);

/**
 * @brief External gradient
 *
 * ExternalGradient = Dilate(region, se) - region
 *
 * @param region Input region
 * @param se Structuring element
 * @return External gradient region
 */
QRegion ExternalGradient(const QRegion& region, const StructElement& se);

/**
 * @brief Top-hat transform
 *
 * Extracts bright regions smaller than SE.
 * TopHat = region - Opening(region, se)
 *
 * @param region Input region
 * @param se Structuring element
 * @return Top-hat result
 */
QRegion TopHat(const QRegion& region, const StructElement& se);

/**
 * @brief Black-hat transform
 *
 * Extracts dark regions smaller than SE.
 * BlackHat = Closing(region, se) - region
 *
 * @param region Input region
 * @param se Structuring element
 * @return Black-hat result
 */
QRegion BlackHat(const QRegion& region, const StructElement& se);

// =============================================================================
// Hit-or-Miss Transform
// =============================================================================

/**
 * @brief Hit-or-miss transform
 *
 * Detects patterns defined by hit and miss structuring elements.
 * Result = Erode(region, hit) ∩ Erode(complement(region), miss)
 *
 * @param region Input region
 * @param hit SE for foreground match
 * @param miss SE for background match
 * @param bounds Bounding rectangle for complement
 * @return Matched positions
 */
QRegion HitOrMiss(const QRegion& region,
                  const StructElement& hit,
                  const StructElement& miss,
                  const Rect2i& bounds);

/**
 * @brief Hit-or-miss with combined SE pair
 *
 * @param region Input region
 * @param sePair Pair of (hit, miss) structuring elements
 * @param bounds Bounding rectangle
 * @return Matched positions
 */
QRegion HitOrMiss(const QRegion& region,
                  const std::pair<StructElement, StructElement>& sePair,
                  const Rect2i& bounds);

// =============================================================================
// Thinning and Skeleton
// =============================================================================

/**
 * @brief Thin region by one iteration
 *
 * Removes pixels that match hit-or-miss pattern.
 * Thin = region - HitOrMiss(region, hit, miss)
 *
 * @param region Input region
 * @param hit SE for foreground
 * @param miss SE for background
 * @param bounds Bounding rectangle
 * @return Thinned region
 */
QRegion ThinOnce(const QRegion& region,
                 const StructElement& hit,
                 const StructElement& miss,
                 const Rect2i& bounds);

/**
 * @brief Morphological thinning
 *
 * Iteratively thin region until stable.
 *
 * @param region Input region
 * @param maxIterations Maximum iterations (0 = until stable)
 * @return Thinned region
 */
QRegion Thin(const QRegion& region, int maxIterations = 0);

/**
 * @brief Thicken region by one iteration
 *
 * Opposite of thinning.
 * Thicken = region ∪ HitOrMiss(complement(region), hit, miss)
 *
 * @param region Input region
 * @param hit SE for background
 * @param miss SE for foreground
 * @param bounds Bounding rectangle
 * @return Thickened region
 */
QRegion ThickenOnce(const QRegion& region,
                    const StructElement& hit,
                    const StructElement& miss,
                    const Rect2i& bounds);

/**
 * @brief Morphological skeleton
 *
 * Extract medial axis of region.
 *
 * @param region Input region
 * @return Skeleton region
 */
QRegion Skeleton(const QRegion& region);

/**
 * @brief Prune skeleton branches
 *
 * Remove short branches from skeleton.
 *
 * @param skeleton Input skeleton
 * @param iterations Number of pruning iterations
 * @return Pruned skeleton
 */
QRegion PruneSkeleton(const QRegion& skeleton, int iterations = 1);

// =============================================================================
// Iterative Operations
// =============================================================================

/**
 * @brief Apply dilation N times
 *
 * @param region Input region
 * @param se Structuring element
 * @param iterations Number of iterations
 * @return Result after N dilations
 */
QRegion DilateN(const QRegion& region, const StructElement& se, int iterations);

/**
 * @brief Apply erosion N times
 *
 * @param region Input region
 * @param se Structuring element
 * @param iterations Number of iterations
 * @return Result after N erosions
 */
QRegion ErodeN(const QRegion& region, const StructElement& se, int iterations);

/**
 * @brief Apply opening N times
 *
 * @param region Input region
 * @param se Structuring element
 * @param iterations Number of iterations
 * @return Result after N openings
 */
QRegion OpeningN(const QRegion& region, const StructElement& se, int iterations);

/**
 * @brief Apply closing N times
 *
 * @param region Input region
 * @param se Structuring element
 * @param iterations Number of iterations
 * @return Result after N closings
 */
QRegion ClosingN(const QRegion& region, const StructElement& se, int iterations);

// =============================================================================
// Geodesic Operations
// =============================================================================

/**
 * @brief Geodesic dilation
 *
 * Dilation constrained within a mask region.
 * Result = Dilate(marker, se) ∩ mask
 *
 * @param marker Marker region (seed)
 * @param mask Constraint mask
 * @param se Structuring element
 * @return Geodesic dilated region
 */
QRegion GeodesicDilate(const QRegion& marker,
                       const QRegion& mask,
                       const StructElement& se);

/**
 * @brief Geodesic erosion
 *
 * Erosion constrained within a mask region.
 * Result = Erode(marker, se) ∪ mask^c
 *
 * @param marker Marker region
 * @param mask Constraint mask
 * @param se Structuring element
 * @return Geodesic eroded region
 */
QRegion GeodesicErode(const QRegion& marker,
                      const QRegion& mask,
                      const StructElement& se);

/**
 * @brief Morphological reconstruction by dilation
 *
 * Iterative geodesic dilation until stable.
 *
 * @param marker Marker region (seed)
 * @param mask Constraint mask
 * @return Reconstructed region
 */
QRegion ReconstructByDilation(const QRegion& marker, const QRegion& mask);

/**
 * @brief Morphological reconstruction by erosion
 *
 * Iterative geodesic erosion until stable.
 *
 * @param marker Marker region
 * @param mask Constraint mask
 * @return Reconstructed region
 */
QRegion ReconstructByErosion(const QRegion& marker, const QRegion& mask);

/**
 * @brief Fill holes using reconstruction
 *
 * @param region Input region
 * @return Region with holes filled
 */
QRegion FillHolesByReconstruction(const QRegion& region);

/**
 * @brief Clear border-touching regions
 *
 * @param region Input region
 * @param bounds Image bounds
 * @return Region without border-touching components
 */
QRegion ClearBorder(const QRegion& region, const Rect2i& bounds);

} // namespace Qi::Vision::Internal
