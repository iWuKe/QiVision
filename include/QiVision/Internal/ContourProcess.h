#pragma once

/**
 * @file ContourProcess.h
 * @brief Contour processing operations for QiVision
 *
 * This module provides:
 * - Smoothing: Gaussian, Moving Average, Bilateral
 * - Simplification: Douglas-Peucker, Visvalingam-Whyatt, Radial Distance
 * - Resampling: By distance, by count, by arc length
 * - Other operations: Reverse, Close, Remove duplicates
 *
 * Used by:
 * - Core/QContour: Member function implementations
 * - Internal/ContourAnalysis: Pre-processing for analysis
 * - Feature/Matching: Model contour preparation
 * - Feature/Edge: Edge detection post-processing
 *
 * Design principles:
 * - All functions are pure (input not modified)
 * - All coordinates use double for subpixel precision
 * - Support both open and closed contours
 * - Optional attribute preservation/interpolation
 */

#include <QiVision/Core/QContour.h>
#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>

#include <vector>
#include <cstdint>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Default sigma for Gaussian smoothing (in pixels)
constexpr double DEFAULT_SMOOTH_SIGMA = 1.0;

/// Default window size for moving average smoothing
constexpr int32_t DEFAULT_SMOOTH_WINDOW = 5;

/// Minimum window size for smoothing
constexpr int32_t MIN_SMOOTH_WINDOW = 3;

/// Maximum window size for smoothing
constexpr int32_t MAX_SMOOTH_WINDOW = 101;

/// Default tolerance for Douglas-Peucker simplification (in pixels)
constexpr double DEFAULT_SIMPLIFY_TOLERANCE = 1.0;

/// Minimum tolerance for simplification
constexpr double MIN_SIMPLIFY_TOLERANCE = 0.01;

/// Minimum number of points for smoothing operations
constexpr size_t MIN_CONTOUR_POINTS_FOR_SMOOTH = 3;

/// Minimum number of points for simplification
constexpr size_t MIN_CONTOUR_POINTS_FOR_SIMPLIFY = 3;

/// Default resampling distance (in pixels)
constexpr double DEFAULT_RESAMPLE_DISTANCE = 1.0;

/// Minimum distance for resampling
constexpr double MIN_RESAMPLE_DISTANCE = 0.01;

/// Tolerance for duplicate point detection
constexpr double DUPLICATE_POINT_TOLERANCE = 1e-9;

/// Tolerance for collinear point detection
constexpr double COLLINEAR_POINT_TOLERANCE = 1e-6;

// =============================================================================
// Enumerations
// =============================================================================

/**
 * @brief Smoothing method enumeration
 */
enum class SmoothMethod {
    Gaussian,       ///< Gaussian smoothing (default)
    MovingAverage,  ///< Moving average (box filter)
    Bilateral       ///< Bilateral filter (edge-preserving)
};

/**
 * @brief Simplification method enumeration
 */
enum class SimplifyMethod {
    DouglasPeucker,   ///< Douglas-Peucker algorithm (default)
    Visvalingam,      ///< Visvalingam-Whyatt algorithm
    RadialDistance,   ///< Radial distance algorithm
    NthPoint          ///< Keep every Nth point
};

/**
 * @brief Resampling method enumeration
 */
enum class ResampleMethod {
    ByDistance,     ///< Fixed distance between points (default)
    ByCount,        ///< Fixed number of points
    ByArcLength     ///< Equal arc length intervals
};

/**
 * @brief Attribute handling mode during processing
 */
enum class AttributeMode {
    None,           ///< Discard attributes
    Interpolate,    ///< Linearly interpolate attributes (default)
    NearestNeighbor ///< Use nearest neighbor attributes
};

// =============================================================================
// Smoothing Parameters
// =============================================================================

/**
 * @brief Parameters for Gaussian smoothing
 */
struct GaussianSmoothParams {
    double sigma = DEFAULT_SMOOTH_SIGMA;        ///< Standard deviation
    int32_t windowSize = 0;                     ///< Window size (0 = auto from sigma)
    AttributeMode attrMode = AttributeMode::Interpolate; ///< How to handle attributes

    static GaussianSmoothParams Default() { return {}; }
};

/**
 * @brief Parameters for moving average smoothing
 */
struct MovingAverageSmoothParams {
    int32_t windowSize = DEFAULT_SMOOTH_WINDOW; ///< Window size (must be odd)
    AttributeMode attrMode = AttributeMode::Interpolate; ///< How to handle attributes

    static MovingAverageSmoothParams Default() { return {}; }
};

/**
 * @brief Parameters for bilateral smoothing
 */
struct BilateralSmoothParams {
    double sigmaSpace = 2.0;    ///< Spatial sigma
    double sigmaRange = 30.0;   ///< Range (curvature) sigma
    int32_t windowSize = 0;     ///< Window size (0 = auto)
    AttributeMode attrMode = AttributeMode::Interpolate;

    static BilateralSmoothParams Default() { return {}; }
};

// =============================================================================
// Simplification Parameters
// =============================================================================

/**
 * @brief Parameters for Douglas-Peucker simplification
 */
struct DouglasPeuckerParams {
    double tolerance = DEFAULT_SIMPLIFY_TOLERANCE;  ///< Maximum perpendicular distance

    static DouglasPeuckerParams Default() { return {}; }
};

/**
 * @brief Parameters for Visvalingam-Whyatt simplification
 */
struct VisvalingamParams {
    double minArea = 1.0;       ///< Minimum triangle area to preserve
    size_t minPoints = 0;       ///< Minimum number of points (0 = use minArea)

    static VisvalingamParams Default() { return {}; }
};

/**
 * @brief Parameters for radial distance simplification
 */
struct RadialDistanceParams {
    double tolerance = DEFAULT_SIMPLIFY_TOLERANCE;  ///< Radial distance tolerance

    static RadialDistanceParams Default() { return {}; }
};

// =============================================================================
// Resampling Parameters
// =============================================================================

/**
 * @brief Parameters for distance-based resampling
 */
struct ResampleByDistanceParams {
    double distance = DEFAULT_RESAMPLE_DISTANCE;    ///< Target distance between points
    bool preserveEndpoints = true;                  ///< Always include first/last points
    AttributeMode attrMode = AttributeMode::Interpolate;

    static ResampleByDistanceParams Default() { return {}; }
};

/**
 * @brief Parameters for count-based resampling
 */
struct ResampleByCountParams {
    size_t count = 100;                             ///< Target number of points
    bool preserveEndpoints = true;                  ///< Always include first/last points
    AttributeMode attrMode = AttributeMode::Interpolate;

    static ResampleByCountParams Default() { return {}; }
};

// =============================================================================
// Smoothing Functions
// =============================================================================

/**
 * @brief Apply Gaussian smoothing to a contour
 *
 * Smooths the contour using a Gaussian kernel. For closed contours,
 * the smoothing wraps around. For open contours, edge handling uses
 * reflection (mirroring).
 *
 * @param contour Input contour
 * @param params Gaussian smoothing parameters
 * @return Smoothed contour
 *
 * @note If contour has fewer than MIN_CONTOUR_POINTS_FOR_SMOOTH points,
 *       returns the original contour unchanged.
 *
 * @par Example:
 * @code
 * QContour smoothed = SmoothContourGaussian(contour, {.sigma = 2.0});
 * @endcode
 */
QContour SmoothContourGaussian(const QContour& contour, const GaussianSmoothParams& params = {});

/**
 * @brief Apply moving average smoothing to a contour
 *
 * Smooths the contour using a simple moving average (box filter).
 * Faster than Gaussian but may produce less smooth results.
 *
 * @param contour Input contour
 * @param params Moving average parameters
 * @return Smoothed contour
 */
QContour SmoothContourMovingAverage(const QContour& contour, const MovingAverageSmoothParams& params = {});

/**
 * @brief Apply bilateral smoothing to a contour
 *
 * Edge-preserving smoothing that considers both spatial distance and
 * local curvature difference. Preserves corners while smoothing noise.
 *
 * @param contour Input contour
 * @param params Bilateral smoothing parameters
 * @return Smoothed contour
 */
QContour SmoothContourBilateral(const QContour& contour, const BilateralSmoothParams& params = {});

/**
 * @brief Unified smoothing interface
 *
 * @param contour Input contour
 * @param method Smoothing method to use
 * @param sigma Smoothing strength (interpreted based on method)
 * @param windowSize Window size (0 = auto)
 * @return Smoothed contour
 */
QContour SmoothContour(const QContour& contour, SmoothMethod method = SmoothMethod::Gaussian,
                       double sigma = DEFAULT_SMOOTH_SIGMA, int32_t windowSize = 0);

// =============================================================================
// Simplification Functions
// =============================================================================

/**
 * @brief Simplify contour using Douglas-Peucker algorithm
 *
 * Iteratively removes points that are within 'tolerance' distance from
 * the line connecting their neighbors. Preserves overall shape well.
 *
 * @param contour Input contour
 * @param params Douglas-Peucker parameters
 * @return Simplified contour
 *
 * @par Algorithm:
 * 1. Start with line from first to last point
 * 2. Find point with maximum perpendicular distance
 * 3. If distance > tolerance, split and recurse
 * 4. Otherwise, remove intermediate points
 *
 * @note Time complexity: O(n log n) average, O(n^2) worst case
 */
QContour SimplifyContourDouglasPeucker(const QContour& contour, const DouglasPeuckerParams& params = {});

/**
 * @brief Simplify contour using Visvalingam-Whyatt algorithm
 *
 * Iteratively removes points that form the smallest triangle area
 * with their neighbors. Good for preserving topological features.
 *
 * @param contour Input contour
 * @param params Visvalingam parameters
 * @return Simplified contour
 *
 * @par Algorithm:
 * 1. Compute effective area for each point (triangle with neighbors)
 * 2. Remove point with smallest area
 * 3. Recompute affected areas
 * 4. Repeat until minArea threshold reached or minPoints count
 *
 * @note Time complexity: O(n log n) using priority queue
 */
QContour SimplifyContourVisvalingam(const QContour& contour, const VisvalingamParams& params = {});

/**
 * @brief Simplify contour using radial distance algorithm
 *
 * Removes consecutive points that are within 'tolerance' radial
 * distance from each other. Simple and fast.
 *
 * @param contour Input contour
 * @param params Radial distance parameters
 * @return Simplified contour
 *
 * @note Time complexity: O(n)
 */
QContour SimplifyContourRadialDistance(const QContour& contour, const RadialDistanceParams& params = {});

/**
 * @brief Simplify contour by keeping every Nth point
 *
 * Simple decimation - keeps every Nth point, always including first and last.
 *
 * @param contour Input contour
 * @param n Keep every Nth point (n >= 2)
 * @return Simplified contour
 */
QContour SimplifyContourNthPoint(const QContour& contour, size_t n);

/**
 * @brief Unified simplification interface
 *
 * @param contour Input contour
 * @param method Simplification method to use
 * @param tolerance Tolerance parameter (interpreted based on method)
 * @return Simplified contour
 */
QContour SimplifyContour(const QContour& contour, SimplifyMethod method = SimplifyMethod::DouglasPeucker,
                         double tolerance = DEFAULT_SIMPLIFY_TOLERANCE);

// =============================================================================
// Resampling Functions
// =============================================================================

/**
 * @brief Resample contour with fixed distance between points
 *
 * Creates a new contour where consecutive points are approximately
 * 'distance' apart (measured along the contour).
 *
 * @param contour Input contour
 * @param params Distance-based resampling parameters
 * @return Resampled contour
 *
 * @note For closed contours, the last point may be slightly closer to
 *       the first point to ensure proper closure.
 */
QContour ResampleContourByDistance(const QContour& contour, const ResampleByDistanceParams& params = {});

/**
 * @brief Resample contour to have a fixed number of points
 *
 * Creates a new contour with exactly 'count' points, uniformly
 * distributed along the contour (by arc length).
 *
 * @param contour Input contour
 * @param params Count-based resampling parameters
 * @return Resampled contour
 *
 * @note For closed contours with preserveEndpoints=true, first point
 *       is included but last point is omitted (as it would duplicate first).
 */
QContour ResampleContourByCount(const QContour& contour, const ResampleByCountParams& params = {});

/**
 * @brief Resample contour with equal arc length intervals
 *
 * Similar to ResampleContourByDistance but guarantees exact equal
 * arc length between all consecutive points.
 *
 * @param contour Input contour
 * @param numSegments Number of segments (points = segments + 1 for open)
 * @param attrMode Attribute handling mode
 * @return Resampled contour
 */
QContour ResampleContourByArcLength(const QContour& contour, size_t numSegments,
                                     AttributeMode attrMode = AttributeMode::Interpolate);

/**
 * @brief Unified resampling interface
 *
 * @param contour Input contour
 * @param method Resampling method to use
 * @param param Method-specific parameter (distance or count)
 * @return Resampled contour
 */
QContour ResampleContour(const QContour& contour, ResampleMethod method = ResampleMethod::ByDistance,
                         double param = DEFAULT_RESAMPLE_DISTANCE);

// =============================================================================
// Other Processing Functions
// =============================================================================

/**
 * @brief Reverse the direction of a contour
 *
 * Reverses the point order and adjusts direction attributes.
 *
 * @param contour Input contour
 * @return Reversed contour
 *
 * @note Direction attributes are rotated by PI.
 */
QContour ReverseContour(const QContour& contour);

/**
 * @brief Close an open contour
 *
 * If the contour is not already closed, marks it as closed.
 * Does not add a duplicate point.
 *
 * @param contour Input contour
 * @return Closed contour
 */
QContour CloseContour(const QContour& contour);

/**
 * @brief Open a closed contour
 *
 * Marks a closed contour as open.
 *
 * @param contour Input contour
 * @return Open contour
 */
QContour OpenContour(const QContour& contour);

/**
 * @brief Remove duplicate consecutive points
 *
 * Removes points that are within 'tolerance' distance of their predecessor.
 *
 * @param contour Input contour
 * @param tolerance Distance tolerance for duplicate detection
 * @return Contour with duplicates removed
 */
QContour RemoveDuplicatePoints(const QContour& contour, double tolerance = DUPLICATE_POINT_TOLERANCE);

/**
 * @brief Remove collinear points
 *
 * Removes points that lie on the line between their neighbors
 * (within tolerance).
 *
 * @param contour Input contour
 * @param tolerance Perpendicular distance tolerance
 * @return Contour with collinear points removed
 *
 * @note This is similar to Douglas-Peucker with tolerance=tolerance,
 *       but operates locally rather than recursively.
 */
QContour RemoveCollinearPoints(const QContour& contour, double tolerance = COLLINEAR_POINT_TOLERANCE);

/**
 * @brief Shift the starting point of a closed contour
 *
 * Rotates the point sequence so that the point nearest to 'newStart'
 * becomes the first point.
 *
 * @param contour Input contour (must be closed)
 * @param newStart Point near desired new start
 * @return Contour with shifted start (or original if open)
 */
QContour ShiftContourStart(const QContour& contour, const Point2d& newStart);

/**
 * @brief Shift the starting point of a closed contour by index
 *
 * @param contour Input contour (must be closed)
 * @param startIndex New starting point index
 * @return Contour with shifted start (or original if open/invalid index)
 */
QContour ShiftContourStartByIndex(const QContour& contour, size_t startIndex);

/**
 * @brief Extract a sub-contour between two parameters
 *
 * Extracts the portion of the contour between parameter t1 and t2,
 * where t in [0, 1] represents position along the contour.
 *
 * @param contour Input contour
 * @param t1 Start parameter [0, 1]
 * @param t2 End parameter [0, 1]
 * @return Extracted sub-contour
 *
 * @note If t2 < t1 and contour is closed, wraps around.
 *       If t2 < t1 and contour is open, swaps t1 and t2.
 */
QContour ExtractSubContour(const QContour& contour, double t1, double t2);

/**
 * @brief Extract a sub-contour between two point indices
 *
 * @param contour Input contour
 * @param startIdx Start point index
 * @param endIdx End point index (exclusive)
 * @return Extracted sub-contour
 */
QContour ExtractSubContourByIndex(const QContour& contour, size_t startIdx, size_t endIdx);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Compute total length of a contour
 *
 * @param contour Input contour
 * @return Total arc length
 */
double ComputeContourLength(const QContour& contour);

/**
 * @brief Compute cumulative arc length at each point
 *
 * @param contour Input contour
 * @return Vector of cumulative lengths (same size as contour)
 *
 * @note First element is 0, last element is total length.
 */
std::vector<double> ComputeCumulativeLength(const QContour& contour);

/**
 * @brief Find point on contour at given arc length
 *
 * @param contour Input contour
 * @param arcLength Target arc length from start
 * @param attrMode Attribute interpolation mode
 * @return Interpolated contour point at the given arc length
 *
 * @note If arcLength exceeds total length, returns last point.
 *       If arcLength < 0, returns first point.
 */
ContourPoint FindPointByArcLength(const QContour& contour, double arcLength,
                                   AttributeMode attrMode = AttributeMode::Interpolate);

/**
 * @brief Find point on contour at given parameter t
 *
 * @param contour Input contour
 * @param t Parameter in [0, 1] (by arc length)
 * @param attrMode Attribute interpolation mode
 * @return Interpolated contour point
 */
ContourPoint FindPointByParameter(const QContour& contour, double t,
                                   AttributeMode attrMode = AttributeMode::Interpolate);

/**
 * @brief Interpolate a contour point between two existing points
 *
 * @param p1 First point
 * @param p2 Second point
 * @param t Interpolation parameter [0, 1] (0=p1, 1=p2)
 * @param attrMode Attribute interpolation mode
 * @return Interpolated contour point
 */
ContourPoint InterpolateContourPoint(const ContourPoint& p1, const ContourPoint& p2,
                                      double t, AttributeMode attrMode = AttributeMode::Interpolate);

/**
 * @brief Find the segment index containing a given arc length
 *
 * @param contour Input contour
 * @param arcLength Target arc length
 * @param localT Output: local parameter within segment [0, 1]
 * @return Segment index (or last segment if arcLength exceeds total)
 */
size_t FindSegmentByArcLength(const QContour& contour, double arcLength, double& localT);

/**
 * @brief Interpolate angle using shortest path on circle
 *
 * @param a1 First angle (radians)
 * @param a2 Second angle (radians)
 * @param t Interpolation parameter [0, 1]
 * @return Interpolated angle in [-PI, PI]
 */
double InterpolateAngle(double a1, double a2, double t);

} // namespace Qi::Vision::Internal
