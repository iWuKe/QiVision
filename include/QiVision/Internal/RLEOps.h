#pragma once

/**
 * @file RLEOps.h
 * @brief RLE (Run-Length Encoding) operations for region processing
 *
 * This module provides:
 * - Image to region conversion (thresholding)
 * - Region to image conversion (painting)
 * - Advanced RLE set operations
 * - Boundary extraction
 * - Fill and connection operations
 * - Region analysis helpers
 *
 * Reference Halcon operators:
 * - threshold, dyn_threshold, binary_threshold
 * - paint_region, gen_region_runs
 * - union1, intersection, difference, complement
 * - boundary, fill_up, fill_up_shape
 * - connection, select_shape
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>
#include <QiVision/Core/Types.h>

#include <vector>
#include <functional>

namespace Qi::Vision::Internal {

// =============================================================================
// Types
// =============================================================================

// Use Connectivity from Qi::Vision namespace (defined in Types.h)
using Qi::Vision::Connectivity;

/// Threshold mode
enum class ThresholdMode {
    Binary,         ///< Simple binary threshold
    BinaryInv,      ///< Inverted binary threshold
    Range,          ///< Value within range
    RangeInv        ///< Value outside range
};

/// Fill direction for fill operations
enum class FillDirection {
    Horizontal,     ///< Fill horizontal gaps
    Vertical,       ///< Fill vertical gaps
    Both            ///< Fill both directions
};

// =============================================================================
// Image to Region Conversion
// =============================================================================

/**
 * @brief Threshold image to create region
 *
 * Creates a region containing all pixels where:
 * - Binary: value >= threshold
 * - BinaryInv: value < threshold
 * - Range: minVal <= value <= maxVal
 * - RangeInv: value < minVal || value > maxVal
 *
 * @param image Input image (grayscale)
 * @param minVal Minimum threshold value
 * @param maxVal Maximum threshold value
 * @param mode Threshold mode
 * @return Region containing thresholded pixels
 */
QRegion ThresholdToRegion(const QImage& image,
                          double minVal,
                          double maxVal = 255.0,
                          ThresholdMode mode = ThresholdMode::Range);

/**
 * @brief Dynamic threshold using local mean
 *
 * Compares each pixel to local neighborhood mean.
 *
 * @param image Input image
 * @param maskSize Local neighborhood size (must be odd)
 * @param offset Offset from local mean (pixel selected if value > localMean + offset)
 * @return Thresholded region
 */
QRegion DynamicThreshold(const QImage& image,
                         int maskSize,
                         double offset = 0.0);

/**
 * @brief Automatic threshold using histogram analysis
 *
 * Uses Otsu's method or similar to find optimal threshold.
 *
 * @param image Input image
 * @param method "otsu", "mean", "median", "triangle"
 * @return Thresholded region
 */
QRegion AutoThreshold(const QImage& image,
                      const std::string& method = "otsu");

/**
 * @brief Create region from non-zero pixels
 *
 * @param image Input image
 * @return Region containing all non-zero pixels
 */
QRegion NonZeroToRegion(const QImage& image);

// =============================================================================
// Region to Image Conversion
// =============================================================================

/**
 * @brief Paint region onto image with specified value
 *
 * @param region Region to paint
 * @param image Target image (modified in place)
 * @param value Pixel value to paint
 */
void PaintRegion(const QRegion& region,
                 QImage& image,
                 double value);

/**
 * @brief Create binary mask image from region
 *
 * @param region Input region
 * @param width Output image width (0 = use region bounding box)
 * @param height Output image height (0 = use region bounding box)
 * @return Binary mask (255 inside, 0 outside)
 */
QImage RegionToMask(const QRegion& region,
                    int32_t width = 0,
                    int32_t height = 0);

/**
 * @brief Create labeled image from multiple regions
 *
 * @param regions Vector of regions
 * @param width Output image width
 * @param height Output image height
 * @return Labeled image (0=background, 1,2,3...=region indices)
 */
QImage RegionsToLabels(const std::vector<QRegion>& regions,
                       int32_t width,
                       int32_t height);

// =============================================================================
// Set Operations on Run Vectors
// =============================================================================

using Run = QRegion::Run;
using RunVector = std::vector<Run>;

/**
 * @brief Union of two run vectors
 *
 * @param runs1 First run vector
 * @param runs2 Second run vector
 * @return Union of runs
 */
RunVector UnionRuns(const RunVector& runs1, const RunVector& runs2);

/**
 * @brief Intersection of two run vectors
 *
 * @param runs1 First run vector
 * @param runs2 Second run vector
 * @return Intersection of runs
 */
RunVector IntersectRuns(const RunVector& runs1, const RunVector& runs2);

/**
 * @brief Difference of two run vectors (runs1 - runs2)
 *
 * @param runs1 First run vector
 * @param runs2 Second run vector
 * @return Difference of runs
 */
RunVector DifferenceRuns(const RunVector& runs1, const RunVector& runs2);

/**
 * @brief Complement of runs within bounds
 *
 * @param runs Input runs
 * @param bounds Bounding rectangle
 * @return Complemented runs
 */
RunVector ComplementRuns(const RunVector& runs, const Rect2i& bounds);

/**
 * @brief Symmetric difference (XOR) of two run vectors
 *
 * @param runs1 First run vector
 * @param runs2 Second run vector
 * @return Symmetric difference
 */
RunVector SymmetricDifferenceRuns(const RunVector& runs1, const RunVector& runs2);

// =============================================================================
// Boundary Operations
// =============================================================================

/**
 * @brief Extract boundary pixels of region
 *
 * @param region Input region
 * @param connectivity 4 or 8 connected boundary
 * @return Region containing only boundary pixels
 */
QRegion ExtractBoundary(const QRegion& region,
                        Connectivity connectivity = Connectivity::Eight);

/**
 * @brief Extract inner boundary (pixels inside region adjacent to outside)
 *
 * @param region Input region
 * @return Inner boundary region
 */
QRegion InnerBoundary(const QRegion& region);

/**
 * @brief Extract outer boundary (pixels outside region adjacent to inside)
 *
 * @param region Input region
 * @return Outer boundary region
 */
QRegion OuterBoundary(const QRegion& region);

// =============================================================================
// Fill Operations
// =============================================================================

/**
 * @brief Fill horizontal gaps in region
 *
 * Fills gaps smaller than maxGap between runs on same row.
 *
 * @param region Input region
 * @param maxGap Maximum gap size to fill
 * @return Filled region
 */
QRegion FillHorizontalGaps(const QRegion& region, int32_t maxGap);

/**
 * @brief Fill vertical gaps in region
 *
 * Fills gaps smaller than maxGap between runs on adjacent rows.
 *
 * @param region Input region
 * @param maxGap Maximum gap size to fill
 * @return Filled region
 */
QRegion FillVerticalGaps(const QRegion& region, int32_t maxGap);

/**
 * @brief Fill all holes in region
 *
 * Fills enclosed areas that are not connected to image border.
 *
 * @param region Input region
 * @return Region with holes filled
 */
QRegion FillHoles(const QRegion& region);

/**
 * @brief Fill up to convex hull
 *
 * @param region Input region
 * @return Convex hull region
 */
QRegion FillConvex(const QRegion& region);

// =============================================================================
// Connection Operations
// =============================================================================

/**
 * @brief Split region into connected components
 *
 * @param region Input region
 * @param connectivity 4 or 8 connected
 * @return Vector of connected component regions
 */
std::vector<QRegion> SplitConnectedComponents(const QRegion& region,
                                               Connectivity connectivity = Connectivity::Eight);

/**
 * @brief Check if region is connected
 *
 * @param region Input region
 * @param connectivity 4 or 8 connected
 * @return True if region is a single connected component
 */
bool IsConnected(const QRegion& region,
                 Connectivity connectivity = Connectivity::Eight);

/**
 * @brief Get number of connected components
 *
 * @param region Input region
 * @param connectivity 4 or 8 connected
 * @return Number of connected components
 */
size_t CountConnectedComponents(const QRegion& region,
                                Connectivity connectivity = Connectivity::Eight);

// =============================================================================
// Analysis Operations
// =============================================================================

/**
 * @brief Compute region area
 *
 * @param runs Run vector
 * @return Area in pixels
 */
int64_t ComputeArea(const RunVector& runs);

/**
 * @brief Compute bounding box
 *
 * @param runs Run vector
 * @return Bounding box rectangle
 */
Rect2i ComputeBoundingBox(const RunVector& runs);

/**
 * @brief Compute centroid
 *
 * @param runs Run vector
 * @return Centroid point
 */
Point2d ComputeCentroid(const RunVector& runs);

/**
 * @brief Compute perimeter (boundary length)
 *
 * @param region Input region
 * @param connectivity 4 or 8 connected
 * @return Perimeter length
 */
double ComputePerimeter(const QRegion& region,
                        Connectivity connectivity = Connectivity::Eight);

/**
 * @brief Compute circularity (4*pi*area / perimeter^2)
 *
 * @param region Input region
 * @return Circularity value [0, 1], 1 = perfect circle
 */
double ComputeCircularity(const QRegion& region);

/**
 * @brief Compute compactness (perimeter^2 / area)
 *
 * @param region Input region
 * @return Compactness value
 */
double ComputeCompactness(const QRegion& region);

/**
 * @brief Compute rectangularity (area / bounding_box_area)
 *
 * @param region Input region
 * @return Rectangularity value [0, 1]
 */
double ComputeRectangularity(const QRegion& region);

// =============================================================================
// RLE Utilities
// =============================================================================

/**
 * @brief Sort runs by (row, colBegin)
 *
 * @param runs Run vector to sort (modified in place)
 */
void SortRuns(RunVector& runs);

/**
 * @brief Merge overlapping/adjacent runs
 *
 * @param runs Run vector (must be sorted)
 */
void MergeRuns(RunVector& runs);

/**
 * @brief Sort and merge runs
 *
 * @param runs Run vector to normalize
 */
void NormalizeRuns(RunVector& runs);

/**
 * @brief Check if runs are valid (sorted, non-overlapping)
 *
 * @param runs Run vector to check
 * @return True if valid
 */
bool ValidateRuns(const RunVector& runs);

/**
 * @brief Translate runs by offset
 *
 * @param runs Run vector
 * @param dx Column offset
 * @param dy Row offset
 * @return Translated runs
 */
RunVector TranslateRuns(const RunVector& runs, int32_t dx, int32_t dy);

/**
 * @brief Clip runs to rectangle
 *
 * @param runs Run vector
 * @param bounds Clipping rectangle
 * @return Clipped runs
 */
RunVector ClipRuns(const RunVector& runs, const Rect2i& bounds);

/**
 * @brief Get runs for a specific row
 *
 * @param runs Run vector (must be sorted)
 * @param row Row index
 * @return Runs on specified row
 */
RunVector GetRunsForRow(const RunVector& runs, int32_t row);

/**
 * @brief Get row range of runs
 *
 * @param runs Run vector
 * @param minRow Output: minimum row
 * @param maxRow Output: maximum row
 */
void GetRowRange(const RunVector& runs, int32_t& minRow, int32_t& maxRow);

} // namespace Qi::Vision::Internal
