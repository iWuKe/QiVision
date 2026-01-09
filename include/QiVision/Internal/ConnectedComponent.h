#pragma once

/**
 * @file ConnectedComponent.h
 * @brief Connected component labeling and analysis
 *
 * This module provides algorithms for labeling connected components
 * in binary images/regions and extracting individual components.
 *
 * Algorithms:
 * - Two-pass labeling with union-find
 * - Single-pass labeling for RLE regions
 * - Component extraction and filtering
 *
 * Reference Halcon operators:
 * - connection, select_shape, count_obj
 * - area_center, smallest_rectangle1/2
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

/**
 * @brief Statistics for a connected component
 */
struct ComponentStats {
    int32_t label = 0;          ///< Component label (1-based)
    int64_t area = 0;           ///< Area in pixels
    double centroidX = 0.0;     ///< Centroid X coordinate
    double centroidY = 0.0;     ///< Centroid Y coordinate
    Rect2i boundingBox;         ///< Axis-aligned bounding box
    int32_t minRow = 0;         ///< Minimum row
    int32_t maxRow = 0;         ///< Maximum row
    int32_t minCol = 0;         ///< Minimum column
    int32_t maxCol = 0;         ///< Maximum column
};

// =============================================================================
// Connected Component Labeling - Image Based
// =============================================================================

/**
 * @brief Label connected components in a binary image
 *
 * Uses two-pass algorithm with union-find for efficient labeling.
 *
 * @param binary Input binary image (non-zero = foreground)
 * @param connectivity 4 or 8 connectivity
 * @param[out] numLabels Number of components found (excluding background)
 * @return Label image (0 = background, 1..N = components)
 */
QImage LabelConnectedComponents(const QImage& binary,
                                 Connectivity connectivity,
                                 int32_t& numLabels);

/**
 * @brief Label connected components (simplified interface)
 *
 * @param binary Input binary image
 * @param connectivity 4 or 8 connectivity
 * @return Label image
 */
QImage LabelConnectedComponents(const QImage& binary,
                                 Connectivity connectivity = Connectivity::Eight);

/**
 * @brief Get statistics for all labeled components
 *
 * @param labels Label image from LabelConnectedComponents
 * @param numLabels Number of labels
 * @return Vector of component statistics (index 0 = label 1)
 */
std::vector<ComponentStats> GetComponentStats(const QImage& labels, int32_t numLabels);

/**
 * @brief Extract a single component as binary image
 *
 * @param labels Label image
 * @param label Component label to extract (1-based)
 * @return Binary image with only the specified component
 */
QImage ExtractComponent(const QImage& labels, int32_t label);

/**
 * @brief Extract all components as separate binary images
 *
 * @param labels Label image
 * @param numLabels Number of labels
 * @return Vector of binary images, one per component
 */
std::vector<QImage> ExtractAllComponents(const QImage& labels, int32_t numLabels);

// =============================================================================
// Connected Component Labeling - Region Based (RLE)
// =============================================================================

// Note: SplitConnectedComponents and CountConnectedComponents are declared
// in RLEOps.h. Include that header for those functions.

/**
 * @brief Get the largest connected component
 *
 * @param region Input region
 * @param connectivity 4 or 8 connectivity
 * @return Largest component by area
 */
QRegion GetLargestComponent(const QRegion& region,
                            Connectivity connectivity = Connectivity::Eight);

/**
 * @brief Get the N largest connected components
 *
 * @param region Input region
 * @param n Number of components to return
 * @param connectivity 4 or 8 connectivity
 * @return Vector of largest components (sorted by area descending)
 */
std::vector<QRegion> GetLargestComponents(const QRegion& region,
                                           int32_t n,
                                           Connectivity connectivity = Connectivity::Eight);

// =============================================================================
// Component Filtering
// =============================================================================

/**
 * @brief Filter components by area
 *
 * @param components Input components
 * @param minArea Minimum area (inclusive)
 * @param maxArea Maximum area (inclusive, 0 = no limit)
 * @return Filtered components
 */
std::vector<QRegion> FilterByArea(const std::vector<QRegion>& components,
                                   int64_t minArea,
                                   int64_t maxArea = 0);

/**
 * @brief Filter components by bounding box dimensions
 *
 * @param components Input components
 * @param minWidth Minimum width
 * @param maxWidth Maximum width (0 = no limit)
 * @param minHeight Minimum height
 * @param maxHeight Maximum height (0 = no limit)
 * @return Filtered components
 */
std::vector<QRegion> FilterBySize(const std::vector<QRegion>& components,
                                   int32_t minWidth, int32_t maxWidth,
                                   int32_t minHeight, int32_t maxHeight);

/**
 * @brief Filter components by aspect ratio
 *
 * @param components Input components
 * @param minRatio Minimum aspect ratio (width/height)
 * @param maxRatio Maximum aspect ratio
 * @return Filtered components
 */
std::vector<QRegion> FilterByAspectRatio(const std::vector<QRegion>& components,
                                          double minRatio,
                                          double maxRatio);

/**
 * @brief Filter components by custom predicate
 *
 * @param components Input components
 * @param predicate Function returning true for components to keep
 * @return Filtered components
 */
std::vector<QRegion> FilterByPredicate(const std::vector<QRegion>& components,
                                        std::function<bool(const QRegion&)> predicate);

/**
 * @brief Select components touching the image border
 *
 * @param components Input components
 * @param bounds Image bounds
 * @return Components touching any border
 */
std::vector<QRegion> SelectBorderComponents(const std::vector<QRegion>& components,
                                             const Rect2i& bounds);

/**
 * @brief Remove components touching the image border
 *
 * @param components Input components
 * @param bounds Image bounds
 * @return Components not touching any border
 */
std::vector<QRegion> RemoveBorderComponents(const std::vector<QRegion>& components,
                                             const Rect2i& bounds);

// =============================================================================
// Component Merging
// =============================================================================

/**
 * @brief Merge all components into a single region
 *
 * @param components Input components
 * @return Union of all components
 */
QRegion MergeComponents(const std::vector<QRegion>& components);

/**
 * @brief Merge nearby components
 *
 * Components closer than the threshold distance are merged.
 *
 * @param components Input components
 * @param maxDistance Maximum distance to merge
 * @return Merged components
 */
std::vector<QRegion> MergeNearbyComponents(const std::vector<QRegion>& components,
                                            double maxDistance);

// =============================================================================
// Hole Detection
// =============================================================================

/**
 * @brief Find holes in a region
 *
 * Holes are background regions completely surrounded by foreground.
 *
 * @param region Input region
 * @param bounds Bounding rectangle for hole detection
 * @return Vector of hole regions
 */
std::vector<QRegion> FindHoles(const QRegion& region, const Rect2i& bounds);

/**
 * @brief Check if a region has holes
 *
 * @param region Input region
 * @param bounds Bounding rectangle
 * @return True if region contains holes
 */
bool HasHoles(const QRegion& region, const Rect2i& bounds);

/**
 * @brief Count holes in a region
 *
 * @param region Input region
 * @param bounds Bounding rectangle
 * @return Number of holes
 */
int32_t CountHoles(const QRegion& region, const Rect2i& bounds);

} // namespace Qi::Vision::Internal
