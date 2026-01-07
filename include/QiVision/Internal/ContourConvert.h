#pragma once

/**
 * @file ContourConvert.h
 * @brief Conversion between contours and regions for QiVision
 *
 * This module provides:
 * - Contour to Region: Fill closed contour interior
 * - Region to Contour: Extract region boundary as contour
 * - Contour to Polygon: Convert to simplified polygon
 *
 * Reference Halcon operators:
 * - gen_region_contour_xld: XLD contour to region (fill)
 * - gen_contour_region_xld: Region boundary to XLD contour
 * - gen_polygons_xld: XLD to polygon approximation
 *
 * Design principles:
 * - All functions are pure (input not modified)
 * - Contour coordinates are double (subpixel)
 * - Region coordinates are int32_t (pixel)
 */

#include <QiVision/Core/QContour.h>
#include <QiVision/Core/QContourArray.h>
#include <QiVision/Core/QRegion.h>

#include <string>
#include <vector>

namespace Qi::Vision::Internal {

// =============================================================================
// Contour Fill Mode
// =============================================================================

/**
 * @brief Fill mode for contour to region conversion
 */
enum class ContourFillMode {
    Filled,         ///< Fill interior (default)
    Margin          ///< Only the contour line (1 pixel wide)
};

// =============================================================================
// Region Boundary Mode
// =============================================================================

/**
 * @brief Boundary extraction mode for region to contour
 */
enum class BoundaryMode {
    Outer,          ///< Outer boundary only (default)
    Inner,          ///< Inner boundaries (holes) only
    Both            ///< Both outer and inner boundaries
};

/**
 * @brief Connectivity for boundary tracing
 */
enum class BoundaryConnectivity {
    FourConnected,  ///< 4-connectivity (axis-aligned)
    EightConnected  ///< 8-connectivity (includes diagonals, default)
};

// =============================================================================
// Contour to Region Conversion
// =============================================================================

/**
 * @brief Convert a closed contour to a filled region
 *
 * Uses scanline fill algorithm to fill the interior of the contour.
 *
 * @param contour Input contour (should be closed)
 * @param mode Fill mode (Filled or Margin)
 * @return Filled region
 *
 * @note Open contours are treated as closed by connecting last to first.
 * @note Self-intersecting contours use even-odd fill rule.
 */
QRegion ContourToRegion(const QContour& contour,
                         ContourFillMode mode = ContourFillMode::Filled);

/**
 * @brief Convert multiple contours to a single region
 *
 * All contours are filled and combined using union operation.
 *
 * @param contours Input contour array
 * @param mode Fill mode
 * @return Combined region
 */
QRegion ContoursToRegion(const QContourArray& contours,
                          ContourFillMode mode = ContourFillMode::Filled);

/**
 * @brief Convert a contour to region with hierarchy support
 *
 * Handles parent-child relationships: children create holes.
 *
 * @param contour Outer contour
 * @param holes Inner contours (holes)
 * @param mode Fill mode
 * @return Region with holes
 */
QRegion ContourWithHolesToRegion(const QContour& contour,
                                  const QContourArray& holes,
                                  ContourFillMode mode = ContourFillMode::Filled);

// =============================================================================
// Region to Contour Conversion
// =============================================================================

/**
 * @brief Extract boundary contours from a region
 *
 * Uses boundary tracing algorithm to extract region boundaries.
 *
 * @param region Input region
 * @param mode Boundary mode (Outer/Inner/Both)
 * @param connectivity Boundary connectivity
 * @return Array of boundary contours
 *
 * @note Returns closed contours.
 * @note Multiple disconnected regions produce multiple contours.
 */
QContourArray RegionToContours(const QRegion& region,
                                BoundaryMode mode = BoundaryMode::Outer,
                                BoundaryConnectivity connectivity = BoundaryConnectivity::EightConnected);

/**
 * @brief Extract single outer boundary from a region
 *
 * Convenience function for simple regions.
 *
 * @param region Input region
 * @param connectivity Boundary connectivity
 * @return Single outer boundary contour
 *
 * @note For multi-component regions, returns the largest component's boundary.
 */
QContour RegionToContour(const QRegion& region,
                          BoundaryConnectivity connectivity = BoundaryConnectivity::EightConnected);

/**
 * @brief Extract boundary with subpixel precision
 *
 * Interpolates boundary positions for smoother contours.
 *
 * @param region Input region
 * @param mode Boundary mode
 * @return Subpixel-precision boundary contours
 */
QContourArray RegionToSubpixelContours(const QRegion& region,
                                        BoundaryMode mode = BoundaryMode::Outer);

// =============================================================================
// Contour Line to Region (Margin)
// =============================================================================

/**
 * @brief Convert contour to region representing the contour line itself
 *
 * Creates a 1-pixel-wide region following the contour path.
 * Uses Bresenham's line algorithm for rasterization.
 *
 * @param contour Input contour
 * @return Region representing the contour line
 */
QRegion ContourLineToRegion(const QContour& contour);

/**
 * @brief Convert contour to thick line region
 *
 * Creates a region with specified thickness along the contour.
 *
 * @param contour Input contour
 * @param thickness Line thickness in pixels
 * @return Thick line region
 */
QRegion ContourToThickLineRegion(const QContour& contour, double thickness);

// =============================================================================
// Polygon Conversion
// =============================================================================

/**
 * @brief Convert contour to simplified polygon
 *
 * Reduces number of points while preserving shape.
 * Uses Douglas-Peucker algorithm.
 *
 * @param contour Input contour
 * @param tolerance Maximum deviation tolerance
 * @return Simplified polygon as contour
 *
 * @note This is similar to ContourProcess::SimplifyContour
 */
QContour ContourToPolygon(const QContour& contour, double tolerance = 1.0);

/**
 * @brief Convert region boundary to simplified polygon
 *
 * Extracts boundary and simplifies in one step.
 *
 * @param region Input region
 * @param tolerance Simplification tolerance
 * @return Simplified polygon contour
 */
QContour RegionToPolygon(const QRegion& region, double tolerance = 1.0);

// =============================================================================
// Point Set Conversion
// =============================================================================

/**
 * @brief Convert contour points to region (individual pixels)
 *
 * Each contour point becomes a single pixel in the region.
 *
 * @param contour Input contour
 * @return Region of individual pixels
 */
QRegion ContourPointsToRegion(const QContour& contour);

/**
 * @brief Convert region pixels to contour points
 *
 * Each pixel in the region becomes a contour point.
 * Points are ordered by row then column.
 *
 * @param region Input region
 * @return Open contour with all region pixels as points
 */
QContour RegionPixelsToContour(const QRegion& region);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Check if a point is inside a closed contour
 *
 * Uses ray casting (even-odd) algorithm.
 *
 * @param contour Closed contour
 * @param point Point to test
 * @return True if point is inside the contour
 */
bool IsPointInsideContour(const QContour& contour, const Point2d& point);

/**
 * @brief Compute winding number of a point relative to contour
 *
 * @param contour Closed contour
 * @param point Point to test
 * @return Winding number (0 = outside, non-zero = inside)
 */
int ContourWindingNumber(const QContour& contour, const Point2d& point);

/**
 * @brief Get the fill direction (CW or CCW) of a contour
 *
 * @param contour Input contour
 * @return True if counter-clockwise, false if clockwise
 */
bool IsContourCCW(const QContour& contour);

// Note: ReverseContour is declared in ContourProcess.h to avoid duplication

} // namespace Qi::Vision::Internal
