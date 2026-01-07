#pragma once

/**
 * @file Intersection.h
 * @brief Geometric intersection calculations between 2D primitives
 *
 * This module provides:
 * - Line-Line, Line-Segment, Segment-Segment intersections
 * - Line/Segment with Circle, Ellipse, Arc intersections
 * - Circle-Circle, Circle-Ellipse intersections
 * - Line/Segment with RotatedRect intersections
 * - Batch intersection operations
 *
 * Used by:
 * - Feature/Measure: Edge intersection calculations
 * - Feature/Metrology: Constraint-based geometry
 * - Internal/Fitting: Clipping operations
 *
 * Design principles:
 * - All functions are pure (no global state)
 * - All coordinates use double for subpixel precision
 * - Return all intersection points when multiple exist
 * - Graceful handling of degenerate and edge cases
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Internal/Geometry2d.h>

#include <cmath>
#include <optional>
#include <vector>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Tolerance for singular matrix detection (parallel/coincident lines)
constexpr double INTERSECTION_SINGULAR_TOLERANCE = 1e-12;

/// Maximum iterations for numerical intersection methods
constexpr int INTERSECTION_MAX_ITERATIONS = 50;

/// Tolerance for numerical convergence
constexpr double INTERSECTION_CONVERGENCE_TOLERANCE = 1e-10;

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief Result of single-point intersection calculation
 */
struct IntersectionResult {
    bool exists = false;        ///< True if intersection exists
    Point2d point;              ///< Intersection point
    double param1 = 0.0;        ///< Parameter on first primitive
    double param2 = 0.0;        ///< Parameter on second primitive

    /// Check if intersection exists
    operator bool() const { return exists; }

    /// Create result for no intersection
    static IntersectionResult None() { return IntersectionResult{}; }

    /// Create result with intersection point
    static IntersectionResult At(const Point2d& p, double t1 = 0.0, double t2 = 0.0) {
        IntersectionResult r;
        r.exists = true;
        r.point = p;
        r.param1 = t1;
        r.param2 = t2;
        return r;
    }
};

/**
 * @brief Result of two-point intersection calculation (e.g., line-circle)
 */
struct IntersectionResult2 {
    int count = 0;              ///< Number of intersections (0, 1, or 2)
    Point2d point1;             ///< First intersection point
    Point2d point2;             ///< Second intersection point
    double param1_1 = 0.0;      ///< First point: parameter on primitive 1
    double param1_2 = 0.0;      ///< First point: parameter on primitive 2
    double param2_1 = 0.0;      ///< Second point: parameter on primitive 1
    double param2_2 = 0.0;      ///< Second point: parameter on primitive 2

    /// Check if any intersection exists
    bool HasIntersection() const { return count > 0; }

    /// Check if exactly one intersection (tangent case)
    bool HasOneIntersection() const { return count == 1; }

    /// Check if two intersections exist
    bool HasTwoIntersections() const { return count == 2; }

    /// Create result for no intersection
    static IntersectionResult2 None() { return IntersectionResult2{}; }

    /// Create result with one intersection point
    static IntersectionResult2 One(const Point2d& p, double t1 = 0.0, double t2 = 0.0) {
        IntersectionResult2 r;
        r.count = 1;
        r.point1 = p;
        r.param1_1 = t1;
        r.param1_2 = t2;
        return r;
    }

    /// Create result with two intersection points
    static IntersectionResult2 Two(const Point2d& p1, const Point2d& p2,
                                    double t1_1 = 0.0, double t1_2 = 0.0,
                                    double t2_1 = 0.0, double t2_2 = 0.0) {
        IntersectionResult2 r;
        r.count = 2;
        r.point1 = p1;
        r.point2 = p2;
        r.param1_1 = t1_1;
        r.param1_2 = t1_2;
        r.param2_1 = t2_1;
        r.param2_2 = t2_2;
        return r;
    }
};

/**
 * @brief Result for multiple intersections (e.g., ellipse-ellipse, up to 4 points)
 */
struct IntersectionResultN {
    std::vector<Point2d> points;    ///< Intersection points
    std::vector<double> params1;    ///< Parameters on primitive 1
    std::vector<double> params2;    ///< Parameters on primitive 2

    /// Number of intersections
    int Count() const { return static_cast<int>(points.size()); }

    /// Check if any intersection exists
    bool HasIntersection() const { return !points.empty(); }

    /// Add an intersection point
    void Add(const Point2d& p, double t1 = 0.0, double t2 = 0.0) {
        points.push_back(p);
        params1.push_back(t1);
        params2.push_back(t2);
    }
};

// =============================================================================
// Line-Line Intersection
// =============================================================================

/**
 * @brief Compute intersection of two infinite lines
 *
 * @param line1 First line (will be normalized internally if needed)
 * @param line2 Second line
 * @return Intersection result (exists=false if lines are parallel or coincident)
 *
 * @note For coincident lines, returns no intersection (use AreCoincident() to check)
 */
IntersectionResult IntersectLineLine(const Line2d& line1, const Line2d& line2);

/**
 * @brief Check if two lines are coincident (same line)
 *
 * @param line1 First line
 * @param line2 Second line
 * @param tolerance Distance tolerance
 * @return true if lines represent the same infinite line
 */
bool AreLinesCoincident(const Line2d& line1, const Line2d& line2,
                        double tolerance = GEOM_TOLERANCE);

// =============================================================================
// Line-Segment and Segment-Segment Intersection
// =============================================================================

/**
 * @brief Compute intersection of infinite line and line segment
 *
 * @param line Infinite line
 * @param segment Line segment
 * @return Intersection result
 *         - param1: Distance along line from foot of perpendicular to origin
 *         - param2: Segment parameter t in [0,1]
 *
 * @note Returns no intersection if parallel or intersection outside segment
 */
IntersectionResult IntersectLineSegment(const Line2d& line, const Segment2d& segment);

/**
 * @brief Compute intersection of two line segments
 *
 * @param seg1 First segment
 * @param seg2 Second segment
 * @return Intersection result with t parameters in [0,1] for both segments
 *
 * @note For overlapping segments, returns one of the overlap endpoints
 */
IntersectionResult IntersectSegmentSegment(const Segment2d& seg1, const Segment2d& seg2);

// =============================================================================
// Line/Segment with Circle
// =============================================================================

/**
 * @brief Compute intersection of infinite line and circle
 *
 * @param line Infinite line
 * @param circle Circle
 * @return 0-2 intersection points
 *         - param1_1/param2_1: Signed distance along line direction from foot
 *         - param1_2/param2_2: Angle on circle in radians [0, 2*PI)
 */
IntersectionResult2 IntersectLineCircle(const Line2d& line, const Circle2d& circle);

/**
 * @brief Compute intersection of segment and circle
 *
 * @param segment Line segment
 * @param circle Circle
 * @return 0-2 intersection points within segment
 *         - param1_1/param2_1: Segment parameter t in [0,1]
 *         - param1_2/param2_2: Angle on circle in radians
 */
IntersectionResult2 IntersectSegmentCircle(const Segment2d& segment, const Circle2d& circle);

/**
 * @brief Check if line intersects circle (fast check without computing points)
 *
 * @param line Infinite line
 * @param circle Circle
 * @return true if line intersects or is tangent to circle
 */
bool LineIntersectsCircle(const Line2d& line, const Circle2d& circle);

/**
 * @brief Check if segment intersects circle (fast check)
 *
 * @param segment Line segment
 * @param circle Circle
 * @return true if segment intersects circle
 */
bool SegmentIntersectsCircle(const Segment2d& segment, const Circle2d& circle);

// =============================================================================
// Line/Segment with Ellipse
// =============================================================================

/**
 * @brief Compute intersection of infinite line and ellipse
 *
 * @param line Infinite line
 * @param ellipse Ellipse (may be rotated)
 * @return 0-2 intersection points
 *         - param1_2/param2_2: Parameter angle theta on ellipse
 */
IntersectionResult2 IntersectLineEllipse(const Line2d& line, const Ellipse2d& ellipse);

/**
 * @brief Compute intersection of segment and ellipse
 *
 * @param segment Line segment
 * @param ellipse Ellipse (may be rotated)
 * @return 0-2 intersection points within segment
 */
IntersectionResult2 IntersectSegmentEllipse(const Segment2d& segment, const Ellipse2d& ellipse);

// =============================================================================
// Line/Segment with Arc
// =============================================================================

/**
 * @brief Compute intersection of infinite line and circular arc
 *
 * @param line Infinite line
 * @param arc Circular arc
 * @return 0-2 intersection points within arc angular range
 */
IntersectionResult2 IntersectLineArc(const Line2d& line, const Arc2d& arc);

/**
 * @brief Compute intersection of segment and circular arc
 *
 * @param segment Line segment
 * @param arc Circular arc
 * @return 0-2 intersection points within segment and arc
 */
IntersectionResult2 IntersectSegmentArc(const Segment2d& segment, const Arc2d& arc);

// =============================================================================
// Circle-Circle Intersection
// =============================================================================

/**
 * @brief Compute intersection of two circles
 *
 * @param circle1 First circle
 * @param circle2 Second circle
 * @return 0-2 intersection points
 *         - param1_1/param2_1: Angle on circle1 in radians
 *         - param1_2/param2_2: Angle on circle2 in radians
 *
 * @note For identical circles, returns no intersection (coincident)
 */
IntersectionResult2 IntersectCircleCircle(const Circle2d& circle1, const Circle2d& circle2);

/**
 * @brief Check if two circles intersect (fast check)
 *
 * @param circle1 First circle
 * @param circle2 Second circle
 * @return true if circles intersect (including tangent)
 */
bool CirclesIntersect(const Circle2d& circle1, const Circle2d& circle2);

/**
 * @brief Determine the relationship between two circles
 *
 * @param circle1 First circle
 * @param circle2 Second circle
 * @return Relationship code:
 *         -2: circle2 contains circle1
 *         -1: circle1 contains circle2
 *          0: circles are separate
 *          1: circles are externally tangent
 *          2: circles intersect at 2 points
 *          3: circles are internally tangent
 *          4: circles are coincident
 */
int CircleRelation(const Circle2d& circle1, const Circle2d& circle2);

// =============================================================================
// Circle-Ellipse Intersection
// =============================================================================

/**
 * @brief Compute intersection of circle and ellipse
 *
 * Uses numerical method (Newton-Raphson) to find up to 4 intersection points.
 *
 * @param circle Circle
 * @param ellipse Ellipse
 * @param maxIterations Maximum Newton iterations
 * @param tolerance Convergence tolerance
 * @return Vector of intersection points (0-4 points)
 */
std::vector<Point2d> IntersectCircleEllipse(
    const Circle2d& circle,
    const Ellipse2d& ellipse,
    int maxIterations = INTERSECTION_MAX_ITERATIONS,
    double tolerance = INTERSECTION_CONVERGENCE_TOLERANCE);

// =============================================================================
// Ellipse-Ellipse Intersection (Optional - Complex)
// =============================================================================

/**
 * @brief Compute intersection of two ellipses
 *
 * Uses numerical method to find up to 4 intersection points.
 * This is a complex computation and may be slower than other intersections.
 *
 * @param ellipse1 First ellipse
 * @param ellipse2 Second ellipse
 * @param maxIterations Maximum iterations
 * @param tolerance Convergence tolerance
 * @return Vector of intersection points (0-4 points)
 */
std::vector<Point2d> IntersectEllipseEllipse(
    const Ellipse2d& ellipse1,
    const Ellipse2d& ellipse2,
    int maxIterations = INTERSECTION_MAX_ITERATIONS,
    double tolerance = INTERSECTION_CONVERGENCE_TOLERANCE);

// =============================================================================
// Line/Segment with RotatedRect
// =============================================================================

/**
 * @brief Compute intersection of infinite line and rotated rectangle
 *
 * @param line Infinite line
 * @param rect Rotated rectangle
 * @return 0-2 intersection points on rectangle boundary
 *         - params indicate which edge (0-3: top, right, bottom, left)
 */
IntersectionResult2 IntersectLineRotatedRect(const Line2d& line, const RotatedRect2d& rect);

/**
 * @brief Compute intersection of segment and rotated rectangle
 *
 * @param segment Line segment
 * @param rect Rotated rectangle
 * @return 0-2 intersection points within segment and on rectangle
 */
IntersectionResult2 IntersectSegmentRotatedRect(const Segment2d& segment, const RotatedRect2d& rect);

// =============================================================================
// Batch Intersection Operations
// =============================================================================

/**
 * @brief Compute intersections of a line with multiple segments
 *
 * @param line Infinite line
 * @param segments Vector of segments
 * @return Vector of intersection results (one per segment, exists=false if none)
 */
std::vector<IntersectionResult> IntersectLineWithSegments(
    const Line2d& line,
    const std::vector<Segment2d>& segments);

/**
 * @brief Compute intersections of a line with a polyline/contour
 *
 * @param line Infinite line
 * @param contourPoints Points defining the contour
 * @param closed If true, includes segment from last to first point
 * @return Vector of intersection results for segments that intersect
 */
std::vector<IntersectionResult> IntersectLineWithContour(
    const Line2d& line,
    const std::vector<Point2d>& contourPoints,
    bool closed = false);

/**
 * @brief Compute intersections of a segment with a polyline/contour
 *
 * @param segment Line segment
 * @param contourPoints Points defining the contour
 * @param closed If true, includes segment from last to first point
 * @return Vector of intersection results for segments that intersect
 */
std::vector<IntersectionResult> IntersectSegmentWithContour(
    const Segment2d& segment,
    const std::vector<Point2d>& contourPoints,
    bool closed = false);

// =============================================================================
// Segment Overlap and Clipping
// =============================================================================

/**
 * @brief Check if two segments overlap (collinear with shared portion)
 *
 * @param seg1 First segment
 * @param seg2 Second segment
 * @param tolerance Distance tolerance for collinearity check
 * @return true if segments are collinear and overlap
 */
bool SegmentsOverlap(const Segment2d& seg1, const Segment2d& seg2,
                     double tolerance = GEOM_TOLERANCE);

/**
 * @brief Get the overlapping portion of two collinear segments
 *
 * @param seg1 First segment
 * @param seg2 Second segment
 * @param tolerance Distance tolerance
 * @return Overlapping segment, or nullopt if no overlap
 */
std::optional<Segment2d> SegmentOverlap(const Segment2d& seg1, const Segment2d& seg2,
                                        double tolerance = GEOM_TOLERANCE);

/**
 * @brief Clip a segment to an axis-aligned rectangle
 *
 * @param segment Input segment
 * @param rect Clipping rectangle (axis-aligned)
 * @return Clipped segment, or nullopt if segment is completely outside
 */
std::optional<Segment2d> ClipSegmentToRect(const Segment2d& segment, const Rect2d& rect);

/**
 * @brief Clip a segment to a rotated rectangle
 *
 * @param segment Input segment
 * @param rect Clipping rotated rectangle
 * @return Clipped segment, or nullopt if segment is completely outside
 */
std::optional<Segment2d> ClipSegmentToRotatedRect(const Segment2d& segment,
                                                   const RotatedRect2d& rect);

// =============================================================================
// Arc Intersection Helpers
// =============================================================================

/**
 * @brief Compute intersection of two circular arcs
 *
 * @param arc1 First arc
 * @param arc2 Second arc
 * @return 0-2 intersection points within both arc ranges
 */
IntersectionResult2 IntersectArcArc(const Arc2d& arc1, const Arc2d& arc2);

/**
 * @brief Check if an angle is within an arc's angular range
 *
 * @param angle Angle to test (radians)
 * @param arc Arc defining the range
 * @return true if angle is within arc's sweep
 */
bool AngleWithinArc(double angle, const Arc2d& arc);

// =============================================================================
// Ray Intersection (for inside/outside tests)
// =============================================================================

/**
 * @brief Count intersections of horizontal ray from point with contour
 *
 * Used for point-in-polygon tests (ray casting algorithm).
 *
 * @param point Starting point of ray (goes to +infinity in x)
 * @param contourPoints Contour points (closed polygon)
 * @return Number of intersections
 */
int CountRayContourIntersections(const Point2d& point,
                                  const std::vector<Point2d>& contourPoints);

} // namespace Qi::Vision::Internal
