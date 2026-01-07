#pragma once

/**
 * @file Distance.h
 * @brief Distance calculation between geometric primitives
 *
 * This module provides:
 * - Point to primitive distance (line, segment, circle, ellipse, arc, rotatedRect)
 * - Primitive to primitive distance (line-line, segment-segment, circle-circle)
 * - Point to contour distance with nearest point
 *
 * Used by:
 * - Internal/Fitting: RANSAC residual calculation
 * - Feature/Measure: Caliper measurement
 * - Feature/Matching: Contour matching distance
 * - Feature/Metrology: Geometric measurement
 *
 * Design principles:
 * - All functions are pure (no global state)
 * - All coordinates use double for subpixel precision
 * - Returns both distance and closest point when applicable
 * - Graceful handling of degenerate cases
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Internal/Geometry2d.h>

#include <cmath>
#include <vector>
#include <optional>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Default maximum iterations for ellipse distance calculation
constexpr int ELLIPSE_MAX_ITERATIONS = 10;

/// Default tolerance for ellipse distance convergence
constexpr double ELLIPSE_TOLERANCE = 1e-10;

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief Distance calculation result with closest point
 */
struct DistanceResult {
    double distance = 0.0;           ///< Unsigned distance
    Point2d closestPoint;            ///< Closest point on the target primitive
    double parameter = 0.0;          ///< Parameter on target (t for segment, theta for circle/arc)

    /// Check if result is valid
    bool IsValid() const { return distance >= 0.0; }

    /// Create invalid result
    static DistanceResult Invalid() {
        DistanceResult r;
        r.distance = -1.0;
        return r;
    }
};

/**
 * @brief Signed distance result (positive = left/outside, negative = right/inside)
 */
struct SignedDistanceResult {
    double signedDistance = 0.0;     ///< Signed distance (convention depends on function)
    Point2d closestPoint;            ///< Closest point on the target primitive
    double parameter = 0.0;          ///< Parameter on target

    /// Get unsigned distance
    double Distance() const { return std::abs(signedDistance); }

    /// Check if point is on positive side
    bool IsOnPositiveSide() const { return signedDistance > 0.0; }

    /// Check if point is on negative side
    bool IsOnNegativeSide() const { return signedDistance < 0.0; }
};

/**
 * @brief Result of segment-to-segment distance computation
 */
struct SegmentDistanceResult {
    double distance = 0.0;           ///< Minimum distance between segments
    Point2d closestPoint1;           ///< Closest point on segment 1
    Point2d closestPoint2;           ///< Closest point on segment 2
    double parameter1 = 0.0;         ///< Parameter t1 in [0,1] on segment 1
    double parameter2 = 0.0;         ///< Parameter t2 in [0,1] on segment 2

    /// Check if segments intersect (distance ~ 0)
    bool Intersects(double tolerance = GEOM_TOLERANCE) const {
        return distance <= tolerance;
    }
};

/**
 * @brief Result of circle-to-circle distance computation
 */
struct CircleDistanceResult {
    double distance = 0.0;           ///< Distance between circle boundaries (can be negative if overlapping)
    Point2d closestPoint1;           ///< Closest point on circle 1
    Point2d closestPoint2;           ///< Closest point on circle 2

    /// Check if circles are externally separated
    bool AreSeparated() const { return distance > 0.0; }

    /// Check if circles are externally tangent
    bool AreExternallyTangent(double tolerance = GEOM_TOLERANCE) const {
        return std::abs(distance) <= tolerance;
    }

    /// Check if one circle contains the other (distance < 0 and |d| < |r1-r2|)
    bool OneContainsOther() const { return distance < 0.0; }
};

/**
 * @brief Result of point-to-contour distance computation
 */
struct ContourDistanceResult {
    double distance = 0.0;           ///< Minimum distance to contour
    Point2d closestPoint;            ///< Closest point on contour
    size_t segmentIndex = 0;         ///< Index of segment containing closest point
    double segmentParameter = 0.0;   ///< Parameter on segment [0,1]

    /// Check if result is valid
    bool IsValid() const { return distance >= 0.0; }

    /// Create invalid result
    static ContourDistanceResult Invalid() {
        ContourDistanceResult r;
        r.distance = -1.0;
        return r;
    }
};

// =============================================================================
// Point-to-Point Distance
// =============================================================================

/**
 * @brief Compute distance between two points
 *
 * @param p1 First point
 * @param p2 Second point
 * @return Euclidean distance
 *
 * @note This is equivalent to p1.DistanceTo(p2) in Point2d
 */
inline double DistancePointToPoint(const Point2d& p1, const Point2d& p2) {
    return p1.DistanceTo(p2);
}

/**
 * @brief Compute squared distance between two points (faster, no sqrt)
 *
 * @param p1 First point
 * @param p2 Second point
 * @return Squared Euclidean distance
 */
inline double DistancePointToPointSquared(const Point2d& p1, const Point2d& p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    return dx * dx + dy * dy;
}

// =============================================================================
// Point-to-Line Distance
// =============================================================================

/**
 * @brief Compute unsigned distance from point to infinite line
 *
 * @param point Query point
 * @param line Target line (assumed normalized: a^2 + b^2 = 1)
 * @return Distance result with closest point on line
 *
 * @note If line is not normalized, it will be normalized internally
 */
DistanceResult DistancePointToLine(const Point2d& point, const Line2d& line);

/**
 * @brief Compute signed distance from point to infinite line
 *
 * Sign convention:
 * - Positive: point is on the side of the normal vector (a, b)
 * - Negative: point is on the opposite side
 *
 * @param point Query point
 * @param line Target line
 * @return Signed distance result
 */
SignedDistanceResult SignedDistancePointToLine(const Point2d& point, const Line2d& line);

// =============================================================================
// Point-to-Segment Distance
// =============================================================================

/**
 * @brief Compute distance from point to line segment
 *
 * Returns distance to the closest point on the segment,
 * which may be an endpoint or an interior point.
 *
 * @param point Query point
 * @param segment Target segment
 * @return Distance result with closest point and parameter t in [0,1]
 *
 * @note For degenerate segment (p1 == p2), returns distance to p1
 */
DistanceResult DistancePointToSegment(const Point2d& point, const Segment2d& segment);

/**
 * @brief Compute signed distance from point to line segment
 *
 * Sign is determined by the direction from p1 to p2:
 * - Positive: point is on the left side of the segment direction
 * - Negative: point is on the right side
 *
 * @param point Query point
 * @param segment Target segment
 * @return Signed distance result
 */
SignedDistanceResult SignedDistancePointToSegment(const Point2d& point, const Segment2d& segment);

// =============================================================================
// Point-to-Circle Distance
// =============================================================================

/**
 * @brief Compute distance from point to circle boundary
 *
 * @param point Query point
 * @param circle Target circle
 * @return Distance result with closest point on circle boundary
 *         - If point is at circle center, returns arbitrary point on boundary
 *
 * @note Distance is always to the circumference, not the disk
 */
DistanceResult DistancePointToCircle(const Point2d& point, const Circle2d& circle);

/**
 * @brief Compute signed distance from point to circle boundary
 *
 * Sign convention:
 * - Positive: point is outside the circle
 * - Negative: point is inside the circle
 * - Zero: point is on the circle boundary
 *
 * @param point Query point
 * @param circle Target circle
 * @return Signed distance result
 */
SignedDistanceResult SignedDistancePointToCircle(const Point2d& point, const Circle2d& circle);

// =============================================================================
// Point-to-Ellipse Distance
// =============================================================================

/**
 * @brief Compute distance from point to ellipse boundary
 *
 * Uses iterative Newton's method to find the closest point on ellipse.
 *
 * @param point Query point
 * @param ellipse Target ellipse
 * @param maxIterations Maximum Newton iterations (default: 10)
 * @param tolerance Convergence tolerance (default: 1e-10)
 * @return Distance result with closest point on ellipse boundary
 *
 * @note For ellipse with a == b (circle), use DistancePointToCircle for efficiency
 */
DistanceResult DistancePointToEllipse(const Point2d& point, const Ellipse2d& ellipse,
                                       int maxIterations = ELLIPSE_MAX_ITERATIONS,
                                       double tolerance = ELLIPSE_TOLERANCE);

/**
 * @brief Compute signed distance from point to ellipse boundary
 *
 * Sign convention:
 * - Positive: point is outside the ellipse
 * - Negative: point is inside the ellipse
 *
 * @param point Query point
 * @param ellipse Target ellipse
 * @param maxIterations Maximum Newton iterations
 * @param tolerance Convergence tolerance
 * @return Signed distance result
 */
SignedDistanceResult SignedDistancePointToEllipse(const Point2d& point, const Ellipse2d& ellipse,
                                                   int maxIterations = ELLIPSE_MAX_ITERATIONS,
                                                   double tolerance = ELLIPSE_TOLERANCE);

// =============================================================================
// Point-to-Arc Distance
// =============================================================================

/**
 * @brief Compute distance from point to circular arc
 *
 * The closest point is either on the arc itself or at one of the endpoints.
 *
 * @param point Query point
 * @param arc Target arc
 * @return Distance result with closest point on arc
 *         - parameter: angle of closest point, or -1/-2 for start/end endpoint
 */
DistanceResult DistancePointToArc(const Point2d& point, const Arc2d& arc);

// =============================================================================
// Point-to-RotatedRect Distance
// =============================================================================

/**
 * @brief Compute distance from point to rotated rectangle boundary
 *
 * Returns the minimum distance to any of the four edges.
 *
 * @param point Query point
 * @param rect Target rotated rectangle
 * @return Distance result with closest point on rectangle boundary
 *         - parameter: edge index (0-3: top, right, bottom, left)
 */
DistanceResult DistancePointToRotatedRect(const Point2d& point, const RotatedRect2d& rect);

/**
 * @brief Compute signed distance from point to rotated rectangle
 *
 * Sign convention:
 * - Positive: point is outside the rectangle
 * - Negative: point is inside the rectangle
 *
 * @param point Query point
 * @param rect Target rotated rectangle
 * @return Signed distance result
 */
SignedDistanceResult SignedDistancePointToRotatedRect(const Point2d& point, const RotatedRect2d& rect);

// =============================================================================
// Line-to-Line Distance
// =============================================================================

/**
 * @brief Compute distance between two parallel lines
 *
 * @param line1 First line
 * @param line2 Second line
 * @return Distance between lines if parallel, nullopt if lines intersect
 *
 * @note For intersecting lines, distance is 0 at intersection point
 */
std::optional<double> DistanceLineToLine(const Line2d& line1, const Line2d& line2);

/**
 * @brief Compute signed distance between two parallel lines
 *
 * Returns the signed distance from line1 to line2, measured
 * along line1's normal direction.
 *
 * @param line1 First line (reference)
 * @param line2 Second line
 * @return Signed distance if parallel, nullopt if lines intersect
 */
std::optional<double> SignedDistanceLineToLine(const Line2d& line1, const Line2d& line2);

// =============================================================================
// Segment-to-Segment Distance
// =============================================================================

/**
 * @brief Compute minimum distance between two line segments
 *
 * Finds the closest pair of points, one on each segment.
 *
 * @param seg1 First segment
 * @param seg2 Second segment
 * @return Segment distance result with closest points on both segments
 */
SegmentDistanceResult DistanceSegmentToSegment(const Segment2d& seg1, const Segment2d& seg2);

// =============================================================================
// Circle-to-Circle Distance
// =============================================================================

/**
 * @brief Compute distance between two circle boundaries
 *
 * Distance conventions:
 * - Positive: circles are separated
 * - Zero: circles are tangent (internally or externally)
 * - Negative: circles overlap or one contains the other
 *
 * @param circle1 First circle
 * @param circle2 Second circle
 * @return Circle distance result
 */
CircleDistanceResult DistanceCircleToCircle(const Circle2d& circle1, const Circle2d& circle2);

// =============================================================================
// Point-to-Contour Distance
// =============================================================================

/**
 * @brief Compute distance from point to contour (polyline)
 *
 * Finds the closest point on the contour, which may be on any segment.
 *
 * @param point Query point
 * @param contourPoints Points defining the contour
 * @param closed If true, contour is closed (last point connects to first)
 * @return Contour distance result
 *
 * @note Returns Invalid() if contour has less than 2 points
 */
ContourDistanceResult DistancePointToContour(const Point2d& point,
                                              const std::vector<Point2d>& contourPoints,
                                              bool closed = false);

/**
 * @brief Compute signed distance from point to closed contour
 *
 * Sign convention:
 * - Positive: point is outside the contour
 * - Negative: point is inside the contour
 *
 * Uses ray casting to determine inside/outside.
 *
 * @param point Query point
 * @param contourPoints Points defining the closed contour
 * @return Signed distance (positive = outside, negative = inside)
 *
 * @note Contour is assumed to be closed
 * @note Returns Invalid() if contour has less than 3 points
 */
SignedDistanceResult SignedDistancePointToContour(const Point2d& point,
                                                   const std::vector<Point2d>& contourPoints);

// =============================================================================
// Batch Distance Functions
// =============================================================================

/**
 * @brief Compute distances from multiple points to a line
 *
 * @param points Query points
 * @param line Target line
 * @return Vector of distance values
 */
std::vector<double> DistancePointsToLine(const std::vector<Point2d>& points, const Line2d& line);

/**
 * @brief Compute signed distances from multiple points to a line
 *
 * @param points Query points
 * @param line Target line
 * @return Vector of signed distance values
 */
std::vector<double> SignedDistancePointsToLine(const std::vector<Point2d>& points, const Line2d& line);

/**
 * @brief Compute distances from multiple points to a circle
 *
 * @param points Query points
 * @param circle Target circle
 * @return Vector of distance values
 */
std::vector<double> DistancePointsToCircle(const std::vector<Point2d>& points, const Circle2d& circle);

/**
 * @brief Compute signed distances from multiple points to a circle
 *
 * @param points Query points
 * @param circle Target circle
 * @return Vector of signed distance values (positive = outside)
 */
std::vector<double> SignedDistancePointsToCircle(const std::vector<Point2d>& points, const Circle2d& circle);

/**
 * @brief Compute distances from multiple points to an ellipse
 *
 * @param points Query points
 * @param ellipse Target ellipse
 * @return Vector of distance values
 */
std::vector<double> DistancePointsToEllipse(const std::vector<Point2d>& points, const Ellipse2d& ellipse);

/**
 * @brief Compute distances from multiple points to a contour
 *
 * @param points Query points
 * @param contourPoints Points defining the contour
 * @param closed If true, contour is closed
 * @return Vector of distance values
 */
std::vector<double> DistancePointsToContour(const std::vector<Point2d>& points,
                                             const std::vector<Point2d>& contourPoints,
                                             bool closed = false);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Find the closest point on a contour to a query point
 *
 * @param point Query point
 * @param contourPoints Points defining the contour
 * @param closed If true, contour is closed
 * @return Closest point on contour, or nullopt if contour is empty
 */
std::optional<Point2d> NearestPointOnContour(const Point2d& point,
                                              const std::vector<Point2d>& contourPoints,
                                              bool closed = false);

/**
 * @brief Compute Hausdorff distance between two contours
 *
 * The Hausdorff distance is the maximum of the minimum distances
 * from each point on one contour to the other contour.
 *
 * @param contour1 First contour points
 * @param contour2 Second contour points
 * @param closed1 If true, first contour is closed
 * @param closed2 If true, second contour is closed
 * @return Hausdorff distance
 */
double HausdorffDistance(const std::vector<Point2d>& contour1,
                         const std::vector<Point2d>& contour2,
                         bool closed1 = false, bool closed2 = false);

/**
 * @brief Compute average distance between two contours (one-directional)
 *
 * Computes the average of minimum distances from each point on contour1
 * to contour2.
 *
 * @param contour1 Source contour points
 * @param contour2 Target contour points
 * @param closed2 If true, target contour is closed
 * @return Average distance
 */
double AverageDistanceContourToContour(const std::vector<Point2d>& contour1,
                                        const std::vector<Point2d>& contour2,
                                        bool closed2 = false);

/**
 * @brief Check if a point is inside a closed polygon
 *
 * Uses ray casting algorithm.
 *
 * @param point Query point
 * @param polygon Polygon vertices
 * @return true if point is inside polygon
 */
bool PointInsidePolygon(const Point2d& point, const std::vector<Point2d>& polygon);

} // namespace Qi::Vision::Internal
