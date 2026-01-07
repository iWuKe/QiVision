#pragma once

/**
 * @file ContourSelect.h
 * @brief Contour selection/filtering by properties for QiVision
 *
 * This module provides:
 * - Select contours by geometric properties (length, area, curvature)
 * - Select contours by shape descriptors (circularity, convexity, etc.)
 * - Select contours by index
 * - Combined multi-criteria selection
 *
 * Reference Halcon operators:
 * - select_contours_xld: Select contours by geometric features
 * - select_shape_xld: Select contours by shape features
 * - select_obj: Select objects by index
 *
 * Design principles:
 * - Reuses ContourAnalysis for all property calculations
 * - All functions are pure (input not modified)
 * - Returns new QContourArray with selected contours
 * - Empty result if no contours match criteria
 */

#include <QiVision/Core/QContour.h>
#include <QiVision/Core/QContourArray.h>

#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Default minimum value for range filtering
constexpr double SELECT_MIN_DEFAULT = -std::numeric_limits<double>::max();

/// Default maximum value for range filtering
constexpr double SELECT_MAX_DEFAULT = std::numeric_limits<double>::max();

// =============================================================================
// Feature Types for Selection
// =============================================================================

/**
 * @brief Contour feature types for selection
 */
enum class ContourFeature {
    // Geometric properties
    Length,             ///< Arc length of contour
    Area,               ///< Enclosed area (closed contours)
    Perimeter,          ///< Perimeter (same as length)
    NumPoints,          ///< Number of points

    // Curvature properties
    MeanCurvature,      ///< Mean absolute curvature
    MaxCurvature,       ///< Maximum absolute curvature

    // Shape descriptors
    Circularity,        ///< 4*pi*A/P^2, 1.0 for perfect circle
    Compactness,        ///< P^2/A
    Convexity,          ///< Convex hull perimeter / contour perimeter
    Solidity,           ///< Area / convex hull area
    Eccentricity,       ///< sqrt(1 - (b/a)^2)
    Elongation,         ///< 1 - minor/major axis
    Rectangularity,     ///< Area / min bounding rect area
    Extent,             ///< Area / AABB area
    AspectRatio,        ///< Major axis / minor axis

    // Position properties
    CentroidRow,        ///< Centroid Y coordinate
    CentroidCol,        ///< Centroid X coordinate

    // Bounding box properties
    BoundingBoxWidth,   ///< Width of axis-aligned bounding box
    BoundingBoxHeight,  ///< Height of axis-aligned bounding box

    // Orientation
    Orientation         ///< Principal axis angle (radians)
};

/**
 * @brief Convert ContourFeature to string name
 */
std::string ContourFeatureToString(ContourFeature feature);

/**
 * @brief Parse ContourFeature from string name
 * @param name Feature name (case-insensitive)
 * @return Feature enum, or Length if not recognized
 */
ContourFeature StringToContourFeature(const std::string& name);

// =============================================================================
// Selection Criteria
// =============================================================================

/**
 * @brief Selection criterion for a single feature
 */
struct SelectionCriterion {
    ContourFeature feature = ContourFeature::Length;
    double minValue = SELECT_MIN_DEFAULT;
    double maxValue = SELECT_MAX_DEFAULT;

    SelectionCriterion() = default;
    SelectionCriterion(ContourFeature f, double minVal, double maxVal)
        : feature(f), minValue(minVal), maxValue(maxVal) {}

    /// Check if a value passes this criterion
    bool Passes(double value) const {
        return value >= minValue && value <= maxValue;
    }
};

/**
 * @brief Logic operation for combining multiple criteria
 */
enum class SelectionLogic {
    And,    ///< All criteria must be satisfied
    Or      ///< At least one criterion must be satisfied
};

// =============================================================================
// Feature Computation
// =============================================================================

/**
 * @brief Compute a specific feature value for a contour
 *
 * @param contour Input contour
 * @param feature Feature to compute
 * @return Feature value
 *
 * @note Some features require closed contours (Area, Circularity, etc.)
 *       Returns 0 for such features on open contours.
 */
double ComputeContourFeature(const QContour& contour, ContourFeature feature);

/**
 * @brief Compute multiple feature values for a contour
 *
 * More efficient than calling ComputeContourFeature multiple times
 * as it caches intermediate calculations.
 *
 * @param contour Input contour
 * @param features Features to compute
 * @return Vector of feature values in same order as input features
 */
std::vector<double> ComputeContourFeatures(const QContour& contour,
                                            const std::vector<ContourFeature>& features);

// =============================================================================
// Single-Feature Selection Functions
// =============================================================================

/**
 * @brief Select contours by length
 *
 * @param contours Input contour array
 * @param minLength Minimum length (inclusive)
 * @param maxLength Maximum length (inclusive)
 * @return Contours with length in [minLength, maxLength]
 */
QContourArray SelectContoursByLength(const QContourArray& contours,
                                      double minLength = 0.0,
                                      double maxLength = SELECT_MAX_DEFAULT);

/**
 * @brief Select contours by enclosed area
 *
 * @param contours Input contour array
 * @param minArea Minimum area (inclusive)
 * @param maxArea Maximum area (inclusive)
 * @return Contours with area in [minArea, maxArea]
 *
 * @note Only closed contours have meaningful area.
 */
QContourArray SelectContoursByArea(const QContourArray& contours,
                                    double minArea = 0.0,
                                    double maxArea = SELECT_MAX_DEFAULT);

/**
 * @brief Select contours by number of points
 *
 * @param contours Input contour array
 * @param minPoints Minimum number of points
 * @param maxPoints Maximum number of points
 * @return Contours with point count in [minPoints, maxPoints]
 */
QContourArray SelectContoursByNumPoints(const QContourArray& contours,
                                         size_t minPoints = 0,
                                         size_t maxPoints = std::numeric_limits<size_t>::max());

/**
 * @brief Select contours by circularity
 *
 * Circularity = 4*pi*A/P^2, equals 1.0 for perfect circle.
 *
 * @param contours Input contour array
 * @param minCircularity Minimum circularity (0.0-1.0)
 * @param maxCircularity Maximum circularity (0.0-1.0)
 * @return Contours with circularity in [minCircularity, maxCircularity]
 */
QContourArray SelectContoursByCircularity(const QContourArray& contours,
                                           double minCircularity = 0.0,
                                           double maxCircularity = 1.0);

/**
 * @brief Select contours by convexity
 *
 * Convexity = convex hull perimeter / contour perimeter.
 *
 * @param contours Input contour array
 * @param minConvexity Minimum convexity (0.0-1.0)
 * @param maxConvexity Maximum convexity (0.0-1.0)
 * @return Contours with convexity in [minConvexity, maxConvexity]
 */
QContourArray SelectContoursByConvexity(const QContourArray& contours,
                                         double minConvexity = 0.0,
                                         double maxConvexity = 1.0);

/**
 * @brief Select contours by solidity
 *
 * Solidity = area / convex hull area.
 *
 * @param contours Input contour array
 * @param minSolidity Minimum solidity (0.0-1.0)
 * @param maxSolidity Maximum solidity (0.0-1.0)
 * @return Contours with solidity in [minSolidity, maxSolidity]
 */
QContourArray SelectContoursBySolidity(const QContourArray& contours,
                                        double minSolidity = 0.0,
                                        double maxSolidity = 1.0);

/**
 * @brief Select contours by compactness
 *
 * Compactness = P^2/A.
 *
 * @param contours Input contour array
 * @param minCompactness Minimum compactness
 * @param maxCompactness Maximum compactness
 * @return Contours with compactness in [minCompactness, maxCompactness]
 */
QContourArray SelectContoursByCompactness(const QContourArray& contours,
                                           double minCompactness = 0.0,
                                           double maxCompactness = SELECT_MAX_DEFAULT);

/**
 * @brief Select contours by elongation
 *
 * Elongation = 1 - minor_axis/major_axis.
 *
 * @param contours Input contour array
 * @param minElongation Minimum elongation (0.0 = circle)
 * @param maxElongation Maximum elongation (1.0 = line)
 * @return Contours with elongation in [minElongation, maxElongation]
 */
QContourArray SelectContoursByElongation(const QContourArray& contours,
                                          double minElongation = 0.0,
                                          double maxElongation = 1.0);

/**
 * @brief Select contours by aspect ratio
 *
 * AspectRatio = major_axis / minor_axis.
 *
 * @param contours Input contour array
 * @param minRatio Minimum aspect ratio (>= 1.0)
 * @param maxRatio Maximum aspect ratio
 * @return Contours with aspect ratio in [minRatio, maxRatio]
 */
QContourArray SelectContoursByAspectRatio(const QContourArray& contours,
                                           double minRatio = 1.0,
                                           double maxRatio = SELECT_MAX_DEFAULT);

/**
 * @brief Select contours by mean curvature
 *
 * @param contours Input contour array
 * @param minCurvature Minimum mean curvature
 * @param maxCurvature Maximum mean curvature
 * @return Contours with mean curvature in [minCurvature, maxCurvature]
 */
QContourArray SelectContoursByMeanCurvature(const QContourArray& contours,
                                             double minCurvature = 0.0,
                                             double maxCurvature = SELECT_MAX_DEFAULT);

/**
 * @brief Select contours by maximum curvature
 *
 * @param contours Input contour array
 * @param minCurvature Minimum max curvature
 * @param maxCurvature Maximum max curvature
 * @return Contours with max curvature in [minCurvature, maxCurvature]
 */
QContourArray SelectContoursByMaxCurvature(const QContourArray& contours,
                                            double minCurvature = 0.0,
                                            double maxCurvature = SELECT_MAX_DEFAULT);

// =============================================================================
// Generic Selection Functions
// =============================================================================

/**
 * @brief Select contours by any single feature
 *
 * @param contours Input contour array
 * @param feature Feature to filter by
 * @param minValue Minimum value (inclusive)
 * @param maxValue Maximum value (inclusive)
 * @return Contours with feature value in [minValue, maxValue]
 */
QContourArray SelectContoursByFeature(const QContourArray& contours,
                                       ContourFeature feature,
                                       double minValue = SELECT_MIN_DEFAULT,
                                       double maxValue = SELECT_MAX_DEFAULT);

/**
 * @brief Select contours by multiple criteria
 *
 * @param contours Input contour array
 * @param criteria List of selection criteria
 * @param logic How to combine criteria (And/Or)
 * @return Contours that satisfy the criteria according to logic
 */
QContourArray SelectContoursByCriteria(const QContourArray& contours,
                                        const std::vector<SelectionCriterion>& criteria,
                                        SelectionLogic logic = SelectionLogic::And);

/**
 * @brief Select contours using a custom predicate
 *
 * @param contours Input contour array
 * @param predicate Function that returns true for contours to keep
 * @return Contours for which predicate returns true
 */
QContourArray SelectContoursIf(const QContourArray& contours,
                                const std::function<bool(const QContour&)>& predicate);

// =============================================================================
// Index-Based Selection
// =============================================================================

/**
 * @brief Select contours by indices
 *
 * @param contours Input contour array
 * @param indices Indices of contours to select (0-based)
 * @return Contours at specified indices
 *
 * @note Invalid indices are silently ignored.
 */
QContourArray SelectContoursByIndex(const QContourArray& contours,
                                     const std::vector<size_t>& indices);

/**
 * @brief Select contour range by indices
 *
 * @param contours Input contour array
 * @param startIndex Starting index (inclusive)
 * @param endIndex Ending index (exclusive)
 * @return Contours in range [startIndex, endIndex)
 */
QContourArray SelectContourRange(const QContourArray& contours,
                                  size_t startIndex,
                                  size_t endIndex);

/**
 * @brief Select first N contours
 *
 * @param contours Input contour array
 * @param count Number of contours to select
 * @return First count contours (or all if count > size)
 */
QContourArray SelectFirstContours(const QContourArray& contours, size_t count);

/**
 * @brief Select last N contours
 *
 * @param contours Input contour array
 * @param count Number of contours to select
 * @return Last count contours (or all if count > size)
 */
QContourArray SelectLastContours(const QContourArray& contours, size_t count);

// =============================================================================
// Sorting and Ranking
// =============================================================================

/**
 * @brief Sort contours by a feature
 *
 * @param contours Input contour array
 * @param feature Feature to sort by
 * @param ascending True for ascending order, false for descending
 * @return Sorted contour array
 */
QContourArray SortContoursByFeature(const QContourArray& contours,
                                     ContourFeature feature,
                                     bool ascending = true);

/**
 * @brief Select top N contours by a feature
 *
 * @param contours Input contour array
 * @param feature Feature to rank by
 * @param count Number of top contours to select
 * @param largest True to select largest values, false for smallest
 * @return Top N contours by feature value
 */
QContourArray SelectTopContoursByFeature(const QContourArray& contours,
                                          ContourFeature feature,
                                          size_t count,
                                          bool largest = true);

// =============================================================================
// Spatial Selection
// =============================================================================

/**
 * @brief Select contours whose centroid is within a rectangle
 *
 * @param contours Input contour array
 * @param minRow Minimum row (inclusive)
 * @param maxRow Maximum row (inclusive)
 * @param minCol Minimum column (inclusive)
 * @param maxCol Maximum column (inclusive)
 * @return Contours with centroid in rectangle
 */
QContourArray SelectContoursInRect(const QContourArray& contours,
                                    double minRow, double maxRow,
                                    double minCol, double maxCol);

/**
 * @brief Select contours whose centroid is within a circle
 *
 * @param contours Input contour array
 * @param centerRow Circle center row
 * @param centerCol Circle center column
 * @param radius Circle radius
 * @return Contours with centroid in circle
 */
QContourArray SelectContoursInCircle(const QContourArray& contours,
                                      double centerRow, double centerCol,
                                      double radius);

// =============================================================================
// Closed/Open Selection
// =============================================================================

/**
 * @brief Select only closed contours
 *
 * @param contours Input contour array
 * @return Closed contours only
 */
QContourArray SelectClosedContours(const QContourArray& contours);

/**
 * @brief Select only open contours
 *
 * @param contours Input contour array
 * @return Open contours only
 */
QContourArray SelectOpenContours(const QContourArray& contours);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Get indices of contours that satisfy a criterion
 *
 * @param contours Input contour array
 * @param feature Feature to check
 * @param minValue Minimum value
 * @param maxValue Maximum value
 * @return Vector of indices
 */
std::vector<size_t> GetContourIndicesByFeature(const QContourArray& contours,
                                                ContourFeature feature,
                                                double minValue = SELECT_MIN_DEFAULT,
                                                double maxValue = SELECT_MAX_DEFAULT);

/**
 * @brief Partition contours into two groups by criterion
 *
 * @param contours Input contour array
 * @param feature Feature to partition by
 * @param threshold Threshold value
 * @param[out] below Contours with feature < threshold
 * @param[out] aboveOrEqual Contours with feature >= threshold
 */
void PartitionContoursByFeature(const QContourArray& contours,
                                 ContourFeature feature,
                                 double threshold,
                                 QContourArray& below,
                                 QContourArray& aboveOrEqual);

} // namespace Qi::Vision::Internal
