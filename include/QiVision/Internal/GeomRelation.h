#pragma once

/**
 * @file GeomRelation.h
 * @brief Geometric relationship detection functions
 *
 * This module provides functions to determine spatial relationships between
 * geometric primitives: containment, tangency, intersection type, etc.
 *
 * Note: Basic functions like AreParallel, ArePerpendicular are in Geometry2d.h
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Internal/Geometry2d.h>

#include <vector>

namespace Qi::Vision {
// Forward declaration
class QContour;
}

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

constexpr double RELATION_TOLERANCE = 1e-9;
constexpr double TANGENT_TOLERANCE = 1e-6;

// =============================================================================
// Enumerations
// =============================================================================

/**
 * @brief Relationship between two circles
 * @note Named CircleCircleRelation to avoid conflict with CircleRelation() function in Intersection.h
 */
enum class CircleCircleRelation {
    Separate,       ///< No intersection, external
    ExternalTangent,///< Touch externally at one point
    Intersecting,   ///< Two intersection points
    InternalTangent,///< Touch internally at one point
    Containing,     ///< First contains second (or vice versa)
    Coincident      ///< Same circle
};

/**
 * @brief Relationship between point and polygon
 */
enum class PointPolygonRelation {
    Inside,         ///< Point is inside polygon
    OnBoundary,     ///< Point is on polygon edge
    Outside         ///< Point is outside polygon
};

/**
 * @brief Relationship between two segments
 */
enum class SegmentRelation {
    Parallel,       ///< Parallel, no intersection
    Collinear,      ///< On same line
    Intersecting,   ///< Cross at one point
    Overlapping,    ///< Collinear with overlap
    Disjoint        ///< Not parallel, no intersection (skew in segment bounds)
};

/**
 * @brief Relationship between line and circle
 */
enum class LineCircleRelation {
    Disjoint,       ///< No intersection
    Tangent,        ///< Touches at one point
    Secant          ///< Crosses at two points
};

// =============================================================================
// Point-Region Containment
// =============================================================================

/**
 * @brief Check if point is inside a circle
 */
inline bool PointInsideCircle(const Point2d& point, const Circle2d& circle) {
    return point.DistanceTo(circle.center) < circle.radius - RELATION_TOLERANCE;
}

/**
 * @brief Check if point is inside or on a circle
 */
inline bool PointInsideOrOnCircle(const Point2d& point, const Circle2d& circle,
                                   double tolerance = RELATION_TOLERANCE) {
    return point.DistanceTo(circle.center) <= circle.radius + tolerance;
}

/**
 * @brief Check if point is inside an ellipse
 */
bool PointInsideEllipse(const Point2d& point, const Ellipse2d& ellipse,
                        double tolerance = RELATION_TOLERANCE);

/**
 * @brief Check if point is inside a rectangle
 */
inline bool PointInsideRect(const Point2d& point, const Rect2d& rect) {
    return point.x > rect.x && point.x < rect.x + rect.width &&
           point.y > rect.y && point.y < rect.y + rect.height;
}

/**
 * @brief Check if point is inside or on a rectangle
 */
inline bool PointInsideOrOnRect(const Point2d& point, const Rect2d& rect,
                                 double tolerance = RELATION_TOLERANCE) {
    return point.x >= rect.x - tolerance && point.x <= rect.x + rect.width + tolerance &&
           point.y >= rect.y - tolerance && point.y <= rect.y + rect.height + tolerance;
}

/**
 * @brief Check if point is inside a rotated rectangle
 */
bool PointInsideRotatedRect(const Point2d& point, const RotatedRect2d& rect,
                            double tolerance = RELATION_TOLERANCE);

/**
 * @brief Check if point is inside a convex polygon
 * @note For general polygons, use PointInPolygon
 */
bool PointInsideConvexPolygon(const Point2d& point, const std::vector<Point2d>& polygon);

/**
 * @brief Check if point is inside a general polygon (convex or concave)
 * @return PointPolygonRelation enum
 */
PointPolygonRelation PointInPolygon(const Point2d& point, const std::vector<Point2d>& polygon,
                                     double tolerance = RELATION_TOLERANCE);

/**
 * @brief Check if point is on polygon boundary
 */
bool PointOnPolygonBoundary(const Point2d& point, const std::vector<Point2d>& polygon,
                            double tolerance = RELATION_TOLERANCE);

// =============================================================================
// Circle-Circle Relationships
// =============================================================================

/**
 * @brief Determine relationship between two circles
 */
CircleCircleRelation GetCircleCircleRelation(const Circle2d& c1, const Circle2d& c2,
                                              double tolerance = RELATION_TOLERANCE);

/**
 * @brief Check if two circles are tangent (internal or external)
 */
inline bool CirclesAreTangent(const Circle2d& c1, const Circle2d& c2,
                               double tolerance = TANGENT_TOLERANCE) {
    CircleCircleRelation rel = GetCircleCircleRelation(c1, c2, tolerance);
    return rel == CircleCircleRelation::ExternalTangent || rel == CircleCircleRelation::InternalTangent;
}

/**
 * @brief Check if first circle contains second
 */
bool CircleContainsCircle(const Circle2d& outer, const Circle2d& inner,
                          double tolerance = RELATION_TOLERANCE);

/**
 * @brief Check if circle contains point
 */
inline bool CircleContainsPoint(const Circle2d& circle, const Point2d& point) {
    return PointInsideCircle(point, circle);
}

// =============================================================================
// Line-Circle Relationships
// =============================================================================

/**
 * @brief Determine relationship between line and circle
 */
LineCircleRelation GetLineCircleRelation(const Line2d& line, const Circle2d& circle,
                                          double tolerance = RELATION_TOLERANCE);

/**
 * @brief Check if line is tangent to circle
 */
bool LineIsTangentToCircle(const Line2d& line, const Circle2d& circle,
                           double tolerance = TANGENT_TOLERANCE);

/**
 * @brief Check if segment is tangent to circle
 */
bool SegmentIsTangentToCircle(const Segment2d& segment, const Circle2d& circle,
                              double tolerance = TANGENT_TOLERANCE);

// =============================================================================
// Segment-Segment Relationships
// =============================================================================

/**
 * @brief Determine relationship between two segments
 */
SegmentRelation GetSegmentRelation(const Segment2d& s1, const Segment2d& s2,
                                    double tolerance = RELATION_TOLERANCE);

// Note: SegmentsOverlap is declared in Intersection.h

/**
 * @brief Check if two segments are connected (share an endpoint)
 */
bool SegmentsConnected(const Segment2d& s1, const Segment2d& s2,
                       double tolerance = RELATION_TOLERANCE);

/**
 * @brief Check if segment endpoints match (same segment, possibly reversed)
 */
bool SegmentsEqual(const Segment2d& s1, const Segment2d& s2,
                   double tolerance = RELATION_TOLERANCE);

// =============================================================================
// Rectangle Relationships
// =============================================================================

/**
 * @brief Check if two rectangles overlap
 */
bool RectsOverlap(const Rect2d& r1, const Rect2d& r2);

/**
 * @brief Check if first rectangle contains second
 */
bool RectContainsRect(const Rect2d& outer, const Rect2d& inner);

/**
 * @brief Check if rectangle contains point
 */
inline bool RectContainsPoint(const Rect2d& rect, const Point2d& point) {
    return PointInsideRect(point, rect);
}

/**
 * @brief Check if two rotated rectangles overlap
 */
bool RotatedRectsOverlap(const RotatedRect2d& r1, const RotatedRect2d& r2);

// =============================================================================
// Polygon Relationships
// =============================================================================

/**
 * @brief Check if polygon is convex
 */
bool IsPolygonConvex(const std::vector<Point2d>& polygon);

/**
 * @brief Check if polygon vertices are ordered counter-clockwise
 */
bool IsPolygonCCW(const std::vector<Point2d>& polygon);

/**
 * @brief Check if two polygons overlap
 */
bool PolygonsOverlap(const std::vector<Point2d>& poly1, const std::vector<Point2d>& poly2);

/**
 * @brief Check if first polygon contains second
 */
bool PolygonContainsPolygon(const std::vector<Point2d>& outer, const std::vector<Point2d>& inner);

// =============================================================================
// Contour Relationships
// =============================================================================

/**
 * @brief Check if point is inside a closed contour
 */
bool PointInsideContour(const Point2d& point, const Qi::Vision::QContour& contour);

/**
 * @brief Check if contour is closed (first and last point are close)
 */
bool IsContourClosed(const Qi::Vision::QContour& contour, double tolerance = RELATION_TOLERANCE);

// =============================================================================
// Angle Relationships
// =============================================================================

/**
 * @brief Check if two angles are equal (considering wrap-around)
 */
bool AnglesEqual(double angle1, double angle2, double tolerance = ANGLE_TOLERANCE);

/**
 * @brief Check if angle is between two angles (in CCW direction from start to end)
 */
bool AngleInRange(double angle, double startAngle, double endAngle);

/**
 * @brief Compute the angular difference (shortest path)
 */
double AngleDifference(double angle1, double angle2);

// =============================================================================
// Coplanarity and Alignment
// =============================================================================

/**
 * @brief Check if multiple points are collinear
 */
bool PointsAreCollinear(const std::vector<Point2d>& points, double tolerance = RELATION_TOLERANCE);

/**
 * @brief Check if multiple points lie on a circle
 */
bool PointsAreConcyclic(const std::vector<Point2d>& points, double tolerance = RELATION_TOLERANCE);

/**
 * @brief Check if three points form a valid triangle (not degenerate)
 */
inline bool IsValidTriangle(const Point2d& p1, const Point2d& p2, const Point2d& p3,
                            double tolerance = RELATION_TOLERANCE) {
    double area = std::abs((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y));
    return area > tolerance;
}

} // namespace Qi::Vision::Internal
