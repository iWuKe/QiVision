#pragma once

/**
 * @file Geometry2d.h
 * @brief 2D geometric primitive operations for QiVision
 *
 * This module provides:
 * - Normalization of geometric primitives
 * - Transformation (translate, rotate, scale, affine)
 * - Property computation (length, area, perimeter, bounding box)
 * - Sampling/discretization (generate point sequences)
 * - Construction helpers
 *
 * Used by:
 * - Internal/Distance: Distance calculations
 * - Internal/Intersection: Intersection calculations
 * - Internal/GeomRelation: Geometric relation tests
 * - Internal/GeomConstruct: Geometric construction
 * - Feature/Measure: Measurement objects
 * - Feature/Matching: Model contour generation
 *
 * Design principles:
 * - All functions are pure (no global state)
 * - All coordinates use double for subpixel precision
 * - Angles in radians
 * - Graceful handling of degenerate cases
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Internal/Matrix.h>

#include <array>
#include <cmath>
#include <optional>
#include <vector>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Tolerance for geometric comparisons (distance)
constexpr double GEOM_TOLERANCE = 1e-9;

/// Tolerance for angle comparisons (radians)
constexpr double ANGLE_TOLERANCE = 1e-9;

/// Default sampling step in pixels
constexpr double DEFAULT_SAMPLING_STEP = 1.0;

/// Maximum number of sampling points to prevent memory issues
constexpr size_t MAX_SAMPLING_POINTS = 1000000;

/// Minimum valid segment length
constexpr double MIN_SEGMENT_LENGTH = 1e-12;

/// Minimum valid radius
constexpr double MIN_RADIUS = 1e-12;

// =============================================================================
// Enumerations
// =============================================================================

/**
 * @brief Arc direction enumeration
 */
enum class ArcDirection {
    CounterClockwise,   ///< Positive angle direction (CCW), default
    Clockwise           ///< Negative angle direction (CW)
};

/**
 * @brief Sampling mode for discretization
 */
enum class SamplingMode {
    ByStep,     ///< Fixed step size (pixels)
    ByCount,    ///< Fixed number of points
    Adaptive    ///< Adaptive based on curvature
};

// =============================================================================
// Normalization Functions
// =============================================================================

/**
 * @brief Normalize a line equation so that a^2 + b^2 = 1
 *
 * @param line Input line (may be unnormalized)
 * @return Normalized line
 *
 * @note If line is degenerate (a = b = 0), returns line unchanged
 */
Line2d NormalizeLine(const Line2d& line);

/**
 * @brief Normalize angle to range [-PI, PI)
 *
 * @param angle Input angle in radians
 * @return Normalized angle in [-PI, PI)
 */
double NormalizeAngle(double angle);

/**
 * @brief Normalize angle to range [0, 2*PI)
 *
 * @param angle Input angle in radians
 * @return Normalized angle in [0, 2*PI)
 */
double NormalizeAngle0To2PI(double angle);

/**
 * @brief Normalize angle difference to range [-PI, PI]
 *
 * @param angleDiff Angle difference in radians
 * @return Normalized angle difference
 */
double NormalizeAngleDiff(double angleDiff);

/**
 * @brief Normalize ellipse parameters so that a >= b
 *
 * If a < b, swaps axes and adjusts angle by PI/2.
 *
 * @param ellipse Input ellipse
 * @return Normalized ellipse with a >= b
 */
Ellipse2d NormalizeEllipse(const Ellipse2d& ellipse);

/**
 * @brief Normalize arc parameters
 *
 * Ensures startAngle is in [0, 2*PI) and sweepAngle is in [-2*PI, 2*PI].
 *
 * @param arc Input arc
 * @return Normalized arc
 */
Arc2d NormalizeArc(const Arc2d& arc);

// =============================================================================
// Point Operations
// =============================================================================

/**
 * @brief Rotate point around origin
 *
 * @param point Input point
 * @param angle Rotation angle in radians (positive = CCW)
 * @return Rotated point
 */
inline Point2d RotatePoint(const Point2d& point, double angle) {
    double cosA = std::cos(angle);
    double sinA = std::sin(angle);
    return {
        point.x * cosA - point.y * sinA,
        point.x * sinA + point.y * cosA
    };
}

/**
 * @brief Rotate point around a specified center
 *
 * @param point Input point
 * @param center Rotation center
 * @param angle Rotation angle in radians
 * @return Rotated point
 */
inline Point2d RotatePointAround(const Point2d& point, const Point2d& center, double angle) {
    Point2d translated = point - center;
    Point2d rotated = RotatePoint(translated, angle);
    return rotated + center;
}

/**
 * @brief Scale point relative to origin
 *
 * @param point Input point
 * @param scaleX Scale factor in X
 * @param scaleY Scale factor in Y
 * @return Scaled point
 */
inline Point2d ScalePoint(const Point2d& point, double scaleX, double scaleY) {
    return {point.x * scaleX, point.y * scaleY};
}

/**
 * @brief Scale point uniformly relative to origin
 *
 * @param point Input point
 * @param scale Uniform scale factor
 * @return Scaled point
 */
inline Point2d ScalePoint(const Point2d& point, double scale) {
    return point * scale;
}

/**
 * @brief Scale point relative to a specified center
 *
 * @param point Input point
 * @param center Scale center
 * @param scaleX Scale factor in X
 * @param scaleY Scale factor in Y
 * @return Scaled point
 */
inline Point2d ScalePointAround(const Point2d& point, const Point2d& center,
                                 double scaleX, double scaleY) {
    return Point2d{
        center.x + (point.x - center.x) * scaleX,
        center.y + (point.y - center.y) * scaleY
    };
}

/**
 * @brief Translate point
 *
 * @param point Input point
 * @param dx Translation in X
 * @param dy Translation in Y
 * @return Translated point
 */
inline Point2d TranslatePoint(const Point2d& point, double dx, double dy) {
    return {point.x + dx, point.y + dy};
}

/**
 * @brief Transform point using affine matrix
 *
 * @param point Input point
 * @param matrix 3x3 affine transformation matrix
 * @return Transformed point
 */
Point2d TransformPoint(const Point2d& point, const Mat33& matrix);

/**
 * @brief Transform multiple points using affine matrix
 *
 * @param points Input points
 * @param matrix 3x3 affine transformation matrix
 * @return Transformed points
 */
std::vector<Point2d> TransformPoints(const std::vector<Point2d>& points, const Mat33& matrix);

// =============================================================================
// Line/Segment Operations
// =============================================================================

/**
 * @brief Create a line perpendicular to given line through a point
 *
 * @param line Reference line
 * @param point Point the perpendicular passes through
 * @return Perpendicular line
 */
Line2d LinePerpendicular(const Line2d& line, const Point2d& point);

/**
 * @brief Create a line parallel to given line through a point
 *
 * @param line Reference line
 * @param point Point the parallel passes through
 * @return Parallel line
 */
Line2d LineParallel(const Line2d& line, const Point2d& point);

/**
 * @brief Create a line from point and angle
 *
 * @param point Point on the line
 * @param angle Line angle in radians (direction)
 * @return Line passing through point at given angle
 */
Line2d LineFromPointAndAngle(const Point2d& point, double angle);

/**
 * @brief Transform line using affine matrix
 *
 * @param line Input line
 * @param matrix 3x3 affine transformation matrix
 * @return Transformed line
 */
Line2d TransformLine(const Line2d& line, const Mat33& matrix);

/**
 * @brief Transform segment using affine matrix
 *
 * @param segment Input segment
 * @param matrix 3x3 affine transformation matrix
 * @return Transformed segment
 */
Segment2d TransformSegment(const Segment2d& segment, const Mat33& matrix);

/**
 * @brief Translate segment
 *
 * @param segment Input segment
 * @param dx Translation in X
 * @param dy Translation in Y
 * @return Translated segment
 */
inline Segment2d TranslateSegment(const Segment2d& segment, double dx, double dy) {
    return Segment2d(
        TranslatePoint(segment.p1, dx, dy),
        TranslatePoint(segment.p2, dx, dy)
    );
}

/**
 * @brief Rotate segment around a center
 *
 * @param segment Input segment
 * @param center Rotation center
 * @param angle Rotation angle in radians
 * @return Rotated segment
 */
inline Segment2d RotateSegment(const Segment2d& segment, const Point2d& center, double angle) {
    return Segment2d(
        RotatePointAround(segment.p1, center, angle),
        RotatePointAround(segment.p2, center, angle)
    );
}

/**
 * @brief Extend segment by distances at both ends
 *
 * @param segment Input segment
 * @param extendStart Distance to extend at start (p1 side), can be negative
 * @param extendEnd Distance to extend at end (p2 side), can be negative
 * @return Extended segment
 */
Segment2d ExtendSegment(const Segment2d& segment, double extendStart, double extendEnd);

/**
 * @brief Clip a line to get a segment within bounds
 *
 * @param line Input line
 * @param bounds Clipping rectangle
 * @return Segment if line intersects bounds, nullopt otherwise
 */
std::optional<Segment2d> ClipLineToRect(const Line2d& line, const Rect2d& bounds);

/**
 * @brief Reverse segment direction (swap p1 and p2)
 *
 * @param segment Input segment
 * @return Reversed segment
 */
inline Segment2d ReverseSegment(const Segment2d& segment) {
    return Segment2d(segment.p2, segment.p1);
}

// =============================================================================
// Circle/Arc Operations
// =============================================================================

/**
 * @brief Translate circle
 *
 * @param circle Input circle
 * @param dx Translation in X
 * @param dy Translation in Y
 * @return Translated circle
 */
inline Circle2d TranslateCircle(const Circle2d& circle, double dx, double dy) {
    return Circle2d(TranslatePoint(circle.center, dx, dy), circle.radius);
}

/**
 * @brief Scale circle uniformly
 *
 * @param circle Input circle
 * @param scale Scale factor
 * @return Scaled circle (both center and radius)
 */
inline Circle2d ScaleCircle(const Circle2d& circle, double scale) {
    return Circle2d(ScalePoint(circle.center, scale), circle.radius * std::abs(scale));
}

/**
 * @brief Scale circle around a center
 *
 * @param circle Input circle
 * @param center Scale center
 * @param scale Scale factor
 * @return Scaled circle
 */
inline Circle2d ScaleCircleAround(const Circle2d& circle, const Point2d& center, double scale) {
    return Circle2d(
        ScalePointAround(circle.center, center, scale, scale),
        circle.radius * std::abs(scale)
    );
}

/**
 * @brief Transform circle using affine matrix
 *
 * @param circle Input circle
 * @param matrix 3x3 affine transformation matrix
 * @return Transformed ellipse (circle becomes ellipse under non-uniform scale/shear)
 *
 * @note If the transform has non-uniform scaling or shear, the result is an ellipse.
 */
Ellipse2d TransformCircle(const Circle2d& circle, const Mat33& matrix);

/**
 * @brief Create arc from three points on the arc
 *
 * @param p1 First point (start of arc)
 * @param p2 Second point (on arc, defines curvature)
 * @param p3 Third point (end of arc)
 * @return Arc passing through all three points, or nullopt if collinear
 */
std::optional<Arc2d> ArcFrom3Points(const Point2d& p1, const Point2d& p2, const Point2d& p3);

/**
 * @brief Create arc from center, radius, and angles
 *
 * @param center Arc center
 * @param radius Arc radius
 * @param startAngle Start angle in radians
 * @param endAngle End angle in radians
 * @param direction Arc direction (CCW or CW)
 * @return Arc
 */
Arc2d ArcFromAngles(const Point2d& center, double radius,
                    double startAngle, double endAngle,
                    ArcDirection direction = ArcDirection::CounterClockwise);

/**
 * @brief Translate arc
 *
 * @param arc Input arc
 * @param dx Translation in X
 * @param dy Translation in Y
 * @return Translated arc
 */
inline Arc2d TranslateArc(const Arc2d& arc, double dx, double dy) {
    return Arc2d(TranslatePoint(arc.center, dx, dy), arc.radius, arc.startAngle, arc.sweepAngle);
}

/**
 * @brief Scale arc uniformly around origin
 *
 * @param arc Input arc
 * @param scale Scale factor
 * @return Scaled arc
 */
inline Arc2d ScaleArc(const Arc2d& arc, double scale) {
    return Arc2d(ScalePoint(arc.center, scale), arc.radius * std::abs(scale),
                 arc.startAngle, arc.sweepAngle);
}

/**
 * @brief Transform arc using affine matrix
 *
 * @param arc Input arc
 * @param matrix 3x3 affine transformation matrix
 * @return Transformed arc (approximation if transform is not similarity)
 *
 * @note For non-uniform scaling or shear, this returns an approximate arc.
 *       For exact transformation, use ellipse arc.
 */
Arc2d TransformArc(const Arc2d& arc, const Mat33& matrix);

/**
 * @brief Get the chord (straight line segment) of an arc
 *
 * @param arc Input arc
 * @return Chord segment from start point to end point
 */
Segment2d ArcToChord(const Arc2d& arc);

/**
 * @brief Split arc at a parameter value
 *
 * @param arc Input arc
 * @param t Split parameter [0, 1], where 0 = start, 1 = end
 * @return Pair of arcs (first: start to t, second: t to end)
 */
std::pair<Arc2d, Arc2d> SplitArc(const Arc2d& arc, double t);

/**
 * @brief Reverse arc direction
 *
 * @param arc Input arc
 * @return Reversed arc (same geometry, opposite direction)
 */
inline Arc2d ReverseArc(const Arc2d& arc) {
    return Arc2d(arc.center, arc.radius, arc.EndAngle(), -arc.sweepAngle);
}

// =============================================================================
// Ellipse Operations
// =============================================================================

/**
 * @brief Translate ellipse
 *
 * @param ellipse Input ellipse
 * @param dx Translation in X
 * @param dy Translation in Y
 * @return Translated ellipse
 */
inline Ellipse2d TranslateEllipse(const Ellipse2d& ellipse, double dx, double dy) {
    return Ellipse2d(TranslatePoint(ellipse.center, dx, dy),
                     ellipse.a, ellipse.b, ellipse.angle);
}

/**
 * @brief Scale ellipse uniformly around origin
 *
 * @param ellipse Input ellipse
 * @param scale Scale factor
 * @return Scaled ellipse
 */
inline Ellipse2d ScaleEllipse(const Ellipse2d& ellipse, double scale) {
    return Ellipse2d(ScalePoint(ellipse.center, scale),
                     ellipse.a * std::abs(scale),
                     ellipse.b * std::abs(scale),
                     ellipse.angle);
}

/**
 * @brief Rotate ellipse around its center
 *
 * @param ellipse Input ellipse
 * @param angle Rotation angle in radians
 * @return Rotated ellipse
 */
Ellipse2d RotateEllipse(const Ellipse2d& ellipse, double angle);

/**
 * @brief Rotate ellipse around a specified center
 *
 * @param ellipse Input ellipse
 * @param center Rotation center
 * @param angle Rotation angle in radians
 * @return Rotated ellipse
 */
Ellipse2d RotateEllipseAround(const Ellipse2d& ellipse, const Point2d& center, double angle);

/**
 * @brief Transform ellipse using affine matrix
 *
 * @param ellipse Input ellipse
 * @param matrix 3x3 affine transformation matrix
 * @return Transformed ellipse
 */
Ellipse2d TransformEllipse(const Ellipse2d& ellipse, const Mat33& matrix);

/**
 * @brief Compute radius of ellipse at a given angle
 *
 * @param ellipse Input ellipse
 * @param theta Angle in ellipse local coordinates (radians)
 * @return Radius at angle theta
 */
double EllipseRadiusAt(const Ellipse2d& ellipse, double theta);

/**
 * @brief Get point on ellipse at a given angle (in ellipse local coordinates)
 *
 * Uses parametric form: x = a*cos(t), y = b*sin(t)
 *
 * @param ellipse Input ellipse
 * @param theta Parameter angle in radians
 * @return Point on ellipse
 */
Point2d EllipsePointAt(const Ellipse2d& ellipse, double theta);

/**
 * @brief Get tangent direction at a point on ellipse
 *
 * @param ellipse Input ellipse
 * @param theta Parameter angle in radians
 * @return Unit tangent vector at the point
 */
Point2d EllipseTangentAt(const Ellipse2d& ellipse, double theta);

/**
 * @brief Get normal direction at a point on ellipse
 *
 * @param ellipse Input ellipse
 * @param theta Parameter angle in radians
 * @return Unit normal vector (outward) at the point
 */
Point2d EllipseNormalAt(const Ellipse2d& ellipse, double theta);

/**
 * @brief Compute approximate arc length of ellipse segment
 *
 * Uses numerical integration (adaptive Simpson's rule).
 *
 * @param ellipse Input ellipse
 * @param thetaStart Start angle (parameter)
 * @param thetaEnd End angle (parameter)
 * @return Arc length
 */
double EllipseArcLength(const Ellipse2d& ellipse, double thetaStart, double thetaEnd);

// =============================================================================
// RotatedRect Operations
// =============================================================================

/**
 * @brief Translate rotated rectangle
 *
 * @param rect Input rectangle
 * @param dx Translation in X
 * @param dy Translation in Y
 * @return Translated rectangle
 */
inline RotatedRect2d TranslateRotatedRect(const RotatedRect2d& rect, double dx, double dy) {
    return RotatedRect2d(TranslatePoint(rect.center, dx, dy),
                         rect.width, rect.height, rect.angle);
}

/**
 * @brief Scale rotated rectangle uniformly around origin
 *
 * @param rect Input rectangle
 * @param scale Scale factor
 * @return Scaled rectangle
 */
inline RotatedRect2d ScaleRotatedRect(const RotatedRect2d& rect, double scale) {
    return RotatedRect2d(ScalePoint(rect.center, scale),
                         rect.width * std::abs(scale),
                         rect.height * std::abs(scale),
                         rect.angle);
}

/**
 * @brief Rotate rotated rectangle around its center
 *
 * @param rect Input rectangle
 * @param angle Rotation angle in radians
 * @return Rotated rectangle
 */
RotatedRect2d RotateRotatedRect(const RotatedRect2d& rect, double angle);

/**
 * @brief Rotate rotated rectangle around a specified center
 *
 * @param rect Input rectangle
 * @param center Rotation center
 * @param angle Rotation angle in radians
 * @return Rotated rectangle
 */
RotatedRect2d RotateRotatedRectAround(const RotatedRect2d& rect, const Point2d& center, double angle);

/**
 * @brief Transform rotated rectangle using affine matrix
 *
 * @param rect Input rectangle
 * @param matrix 3x3 affine transformation matrix
 * @return Transformed rectangle (may not be exact if transform has shear)
 */
RotatedRect2d TransformRotatedRect(const RotatedRect2d& rect, const Mat33& matrix);

/**
 * @brief Get the four corners of a rotated rectangle
 *
 * @param rect Input rectangle
 * @return Array of 4 corners [topLeft, topRight, bottomRight, bottomLeft] in world coordinates
 */
std::array<Point2d, 4> RotatedRectCorners(const RotatedRect2d& rect);

/**
 * @brief Get the four edges of a rotated rectangle as segments
 *
 * @param rect Input rectangle
 * @return Array of 4 edges [top, right, bottom, left]
 */
std::array<Segment2d, 4> RotatedRectEdges(const RotatedRect2d& rect);

// =============================================================================
// Property Computation
// =============================================================================

/**
 * @brief Compute arc sector area
 *
 * @param arc Input arc
 * @return Area of the circular sector defined by the arc
 */
inline double ArcSectorArea(const Arc2d& arc) {
    return 0.5 * arc.radius * arc.radius * std::abs(arc.sweepAngle);
}

/**
 * @brief Compute arc segment (bow) area
 *
 * @param arc Input arc
 * @return Area of the circular segment (region between arc and chord)
 */
double ArcSegmentArea(const Arc2d& arc);

/**
 * @brief Compute bounding box of a circle
 *
 * @param circle Input circle
 * @return Axis-aligned bounding box
 */
inline Rect2d CircleBoundingBox(const Circle2d& circle) {
    return Rect2d(circle.center.x - circle.radius,
                  circle.center.y - circle.radius,
                  2.0 * circle.radius,
                  2.0 * circle.radius);
}

/**
 * @brief Compute bounding box of an arc
 *
 * @param arc Input arc
 * @return Axis-aligned bounding box
 */
Rect2d ArcBoundingBox(const Arc2d& arc);

/**
 * @brief Compute bounding box of an ellipse
 *
 * @param ellipse Input ellipse
 * @return Axis-aligned bounding box
 */
Rect2d EllipseBoundingBox(const Ellipse2d& ellipse);

/**
 * @brief Compute bounding box of a segment
 *
 * @param segment Input segment
 * @return Axis-aligned bounding box
 */
inline Rect2d SegmentBoundingBox(const Segment2d& segment) {
    double minX = std::min(segment.p1.x, segment.p2.x);
    double minY = std::min(segment.p1.y, segment.p2.y);
    double maxX = std::max(segment.p1.x, segment.p2.x);
    double maxY = std::max(segment.p1.y, segment.p2.y);
    return Rect2d(minX, minY, maxX - minX, maxY - minY);
}

/**
 * @brief Compute centroid of an arc (centroid of the arc curve, not sector)
 *
 * @param arc Input arc
 * @return Centroid point
 */
Point2d ArcCentroid(const Arc2d& arc);

/**
 * @brief Compute centroid of an arc sector
 *
 * @param arc Input arc
 * @return Centroid of the sector region
 */
Point2d ArcSectorCentroid(const Arc2d& arc);

// =============================================================================
// Sampling/Discretization
// =============================================================================

/**
 * @brief Sample points along a segment
 *
 * @param segment Input segment
 * @param step Sampling step (pixels)
 * @param includeEndpoints If true, always includes p1 and p2
 * @return Vector of sampled points
 */
std::vector<Point2d> SampleSegment(const Segment2d& segment, double step = DEFAULT_SAMPLING_STEP,
                                    bool includeEndpoints = true);

/**
 * @brief Sample fixed number of points along a segment
 *
 * @param segment Input segment
 * @param numPoints Number of points to sample (minimum 2)
 * @return Vector of sampled points
 */
std::vector<Point2d> SampleSegmentByCount(const Segment2d& segment, size_t numPoints);

/**
 * @brief Sample points along a circle
 *
 * @param circle Input circle
 * @param step Sampling step (arc length in pixels)
 * @return Vector of sampled points (closed, first point repeated at end)
 */
std::vector<Point2d> SampleCircle(const Circle2d& circle, double step = DEFAULT_SAMPLING_STEP);

/**
 * @brief Sample fixed number of points along a circle
 *
 * @param circle Input circle
 * @param numPoints Number of points to sample
 * @param closed If true, last point connects to first
 * @return Vector of sampled points
 */
std::vector<Point2d> SampleCircleByCount(const Circle2d& circle, size_t numPoints, bool closed = true);

/**
 * @brief Sample points along an arc
 *
 * @param arc Input arc
 * @param step Sampling step (arc length in pixels)
 * @param includeEndpoints If true, always includes start and end points
 * @return Vector of sampled points
 */
std::vector<Point2d> SampleArc(const Arc2d& arc, double step = DEFAULT_SAMPLING_STEP,
                                bool includeEndpoints = true);

/**
 * @brief Sample fixed number of points along an arc
 *
 * @param arc Input arc
 * @param numPoints Number of points to sample (minimum 2)
 * @return Vector of sampled points
 */
std::vector<Point2d> SampleArcByCount(const Arc2d& arc, size_t numPoints);

/**
 * @brief Sample points along an ellipse
 *
 * @param ellipse Input ellipse
 * @param step Approximate sampling step (arc length in pixels)
 * @return Vector of sampled points (closed)
 */
std::vector<Point2d> SampleEllipse(const Ellipse2d& ellipse, double step = DEFAULT_SAMPLING_STEP);

/**
 * @brief Sample fixed number of points along an ellipse
 *
 * @param ellipse Input ellipse
 * @param numPoints Number of points to sample
 * @param closed If true, last point connects to first
 * @return Vector of sampled points
 */
std::vector<Point2d> SampleEllipseByCount(const Ellipse2d& ellipse, size_t numPoints, bool closed = true);

/**
 * @brief Sample points along an ellipse arc
 *
 * @param ellipse Base ellipse
 * @param thetaStart Start angle (parameter)
 * @param thetaEnd End angle (parameter)
 * @param step Approximate sampling step
 * @param includeEndpoints If true, always includes start and end points
 * @return Vector of sampled points
 */
std::vector<Point2d> SampleEllipseArc(const Ellipse2d& ellipse,
                                       double thetaStart, double thetaEnd,
                                       double step = DEFAULT_SAMPLING_STEP,
                                       bool includeEndpoints = true);

/**
 * @brief Sample points along the boundary of a rotated rectangle
 *
 * @param rect Input rectangle
 * @param step Sampling step (pixels)
 * @param closed If true, includes closing segment back to first point
 * @return Vector of sampled points
 */
std::vector<Point2d> SampleRotatedRect(const RotatedRect2d& rect,
                                        double step = DEFAULT_SAMPLING_STEP,
                                        bool closed = true);

/**
 * @brief Compute recommended number of sampling points
 *
 * @param arcLength Total arc length
 * @param step Desired step size
 * @param minPoints Minimum number of points
 * @param maxPoints Maximum number of points
 * @return Recommended number of points
 */
inline size_t ComputeSamplingCount(double arcLength, double step,
                                    size_t minPoints = 2, size_t maxPoints = MAX_SAMPLING_POINTS) {
    if (arcLength <= 0 || step <= 0) return minPoints;
    size_t count = static_cast<size_t>(std::ceil(arcLength / step)) + 1;
    return std::clamp(count, minPoints, maxPoints);
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Check if a point lies on a line (within tolerance)
 *
 * @param point Point to check
 * @param line Line to check against
 * @param tolerance Distance tolerance
 * @return true if point is on line within tolerance
 */
inline bool PointOnLine(const Point2d& point, const Line2d& line, double tolerance = GEOM_TOLERANCE) {
    return std::abs(line.SignedDistance(point)) <= tolerance;
}

/**
 * @brief Check if a point lies on a segment (within tolerance)
 *
 * @param point Point to check
 * @param segment Segment to check against
 * @param tolerance Distance tolerance
 * @return true if point is on segment within tolerance
 */
bool PointOnSegment(const Point2d& point, const Segment2d& segment, double tolerance = GEOM_TOLERANCE);

/**
 * @brief Check if a point lies on a circle (within tolerance)
 *
 * @param point Point to check
 * @param circle Circle to check against
 * @param tolerance Distance tolerance
 * @return true if point is on circle circumference within tolerance
 */
inline bool PointOnCircle(const Point2d& point, const Circle2d& circle, double tolerance = GEOM_TOLERANCE) {
    double dist = point.DistanceTo(circle.center);
    return std::abs(dist - circle.radius) <= tolerance;
}

/**
 * @brief Check if a point lies on an arc (within tolerance)
 *
 * @param point Point to check
 * @param arc Arc to check against
 * @param tolerance Distance tolerance
 * @return true if point is on arc within tolerance
 */
bool PointOnArc(const Point2d& point, const Arc2d& arc, double tolerance = GEOM_TOLERANCE);

/**
 * @brief Check if a point lies on an ellipse (within tolerance)
 *
 * @param point Point to check
 * @param ellipse Ellipse to check against
 * @param tolerance Distance tolerance
 * @return true if point is on ellipse circumference within tolerance
 */
bool PointOnEllipse(const Point2d& point, const Ellipse2d& ellipse, double tolerance = GEOM_TOLERANCE);

/**
 * @brief Compute angle between two lines
 *
 * Returns the acute angle between the lines (always positive, <= PI/2).
 *
 * @param line1 First line
 * @param line2 Second line
 * @return Angle in radians [0, PI/2]
 */
double AngleBetweenLines(const Line2d& line1, const Line2d& line2);

/**
 * @brief Compute signed angle from vector1 to vector2
 *
 * @param v1 First vector (or direction)
 * @param v2 Second vector
 * @return Signed angle in radians [-PI, PI], positive = CCW
 */
inline double SignedAngle(const Point2d& v1, const Point2d& v2) {
    return std::atan2(v1.Cross(v2), v1.Dot(v2));
}

/**
 * @brief Check if two lines are parallel
 *
 * @param line1 First line
 * @param line2 Second line
 * @param tolerance Angular tolerance in radians
 * @return true if lines are parallel within tolerance
 */
bool AreParallel(const Line2d& line1, const Line2d& line2, double tolerance = ANGLE_TOLERANCE);

/**
 * @brief Check if two lines are perpendicular
 *
 * @param line1 First line
 * @param line2 Second line
 * @param tolerance Angular tolerance in radians
 * @return true if lines are perpendicular within tolerance
 */
bool ArePerpendicular(const Line2d& line1, const Line2d& line2, double tolerance = ANGLE_TOLERANCE);

/**
 * @brief Check if two segments are collinear
 *
 * @param seg1 First segment
 * @param seg2 Second segment
 * @param tolerance Distance tolerance
 * @return true if segments lie on the same line within tolerance
 */
bool AreCollinear(const Segment2d& seg1, const Segment2d& seg2, double tolerance = GEOM_TOLERANCE);

/**
 * @brief Project point onto line
 *
 * @param point Point to project
 * @param line Line to project onto
 * @return Projection of point onto line
 */
Point2d ProjectPointOnLine(const Point2d& point, const Line2d& line);

/**
 * @brief Project point onto segment (clamped to segment)
 *
 * @param point Point to project
 * @param segment Segment to project onto
 * @return Projection of point onto segment (clamped to endpoints)
 */
Point2d ProjectPointOnSegment(const Point2d& point, const Segment2d& segment);

/**
 * @brief Project point onto circle
 *
 * @param point Point to project
 * @param circle Circle to project onto
 * @return Closest point on circle to the given point
 */
Point2d ProjectPointOnCircle(const Point2d& point, const Circle2d& circle);

/**
 * @brief Compute foot of perpendicular from point to line
 *
 * Same as ProjectPointOnLine.
 *
 * @param point Point
 * @param line Line
 * @return Foot point
 */
inline Point2d FootOfPerpendicular(const Point2d& point, const Line2d& line) {
    return ProjectPointOnLine(point, line);
}

/**
 * @brief Reflect point across a line
 *
 * @param point Point to reflect
 * @param line Mirror line
 * @return Reflected point
 */
Point2d ReflectPointAcrossLine(const Point2d& point, const Line2d& line);

/**
 * @brief Compute midpoint of a segment
 *
 * Same as segment.Midpoint().
 *
 * @param segment Input segment
 * @return Midpoint
 */
inline Point2d SegmentMidpoint(const Segment2d& segment) {
    return segment.Midpoint();
}

/**
 * @brief Check if an angle is within an arc's angular range
 *
 * @param angle Angle to check (radians)
 * @param arc Arc defining the angular range
 * @return true if angle is within arc's sweep
 */
bool AngleInArcRange(double angle, const Arc2d& arc);

/**
 * @brief Convert angle parameter to point on arc
 *
 * Same as arc.PointAt().
 *
 * @param arc Input arc
 * @param t Parameter [0, 1]
 * @return Point on arc
 */
inline Point2d ArcPointAtParameter(const Arc2d& arc, double t) {
    return arc.PointAt(t);
}

} // namespace Qi::Vision::Internal
