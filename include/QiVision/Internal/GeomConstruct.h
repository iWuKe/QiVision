#pragma once

#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Internal/Geometry2d.h>
#include <QiVision/Internal/Fitting.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace Qi::Vision::Internal {

constexpr double CONSTRUCT_TOLERANCE = 1e-9;
constexpr double TANGENT_TOLERANCE = 1e-9;
constexpr int CONSTRUCT_MAX_ITERATIONS = 100;

Line2d PerpendicularLine(const Line2d& line, const Point2d& point);
Line2d PerpendicularLine(const Segment2d& segment, const Point2d& point);
Line2d PerpendicularFromPoint(const Line2d& line, const Point2d& point, Point2d& foot);
Segment2d PerpendicularSegment(const Line2d& line, const Point2d& point);
Segment2d PerpendicularSegment(const Segment2d& segment, const Point2d& point, bool clampToSegment = false);

Line2d ParallelLine(const Line2d& line, const Point2d& point);
Line2d ParallelLineAtDistance(const Line2d& line, double distance);
std::pair<Line2d, Line2d> ParallelLinesAtDistance(const Line2d& line, double distance);
Segment2d ParallelSegmentAtDistance(const Segment2d& segment, double distance);

std::optional<Line2d> AngleBisector(const Line2d& line1, const Line2d& line2);
std::optional<std::pair<Line2d, Line2d>> AngleBisectors(const Line2d& line1, const Line2d& line2);
Line2d AngleBisectorFromPoints(const Point2d& p1, const Point2d& vertex, const Point2d& p3);
Line2d AngleBisector(const Segment2d& seg1, const Segment2d& seg2, const Point2d& vertex);

std::vector<Line2d> TangentLinesToCircle(const Circle2d& circle, const Point2d& point);
std::vector<Point2d> TangentPointsToCircle(const Circle2d& circle, const Point2d& point);
Line2d TangentLineAtAngle(const Circle2d& circle, double angle);
Line2d TangentLineAtClosestPoint(const Circle2d& circle, const Point2d& point);

std::vector<Line2d> TangentLinesToEllipse(const Ellipse2d& ellipse, const Point2d& point);
std::vector<Point2d> TangentPointsToEllipse(const Ellipse2d& ellipse, const Point2d& point);
Line2d TangentLineToEllipseAt(const Ellipse2d& ellipse, double theta);

enum class TangentType { External, Internal };

struct CommonTangentResult {
    std::vector<Line2d> external;
    std::vector<Line2d> internal;
    int TotalCount() const { return static_cast<int>(external.size() + internal.size()); }
};

CommonTangentResult CommonTangents(const Circle2d& circle1, const Circle2d& circle2);
std::vector<Line2d> ExternalCommonTangents(const Circle2d& circle1, const Circle2d& circle2);
std::vector<Line2d> InternalCommonTangents(const Circle2d& circle1, const Circle2d& circle2);

std::optional<Circle2d> CircumscribedCircle(const Point2d& p1, const Point2d& p2, const Point2d& p3);
std::optional<Circle2d> CircumscribedCircle(const std::vector<Point2d>& points, double tolerance = CONSTRUCT_TOLERANCE);

std::optional<Circle2d> InscribedCircle(const Point2d& p1, const Point2d& p2, const Point2d& p3);
std::optional<Circle2d> InscribedCircle(const std::vector<Point2d>& polygon);

std::optional<Circle2d> MinEnclosingCircle(const std::vector<Point2d>& points);
std::optional<Circle2d> MinEnclosingCircleWeighted(const std::vector<Point2d>& points, const std::vector<double>& weights);

std::optional<RotatedRect2d> MinAreaRect(const std::vector<Point2d>& points);
std::optional<Rect2d> MinBoundingRect(const std::vector<Point2d>& points);
RotatedRect2d BoundingRectAtAngle(const std::vector<Point2d>& points, double angle);

std::vector<Point2d> ConvexHull(const std::vector<Point2d>& points);
std::vector<size_t> ConvexHullIndices(const std::vector<Point2d>& points);
bool IsConvex(const std::vector<Point2d>& polygon);
double SignedPolygonArea(const std::vector<Point2d>& polygon);

inline double PolygonArea(const std::vector<Point2d>& polygon) {
    return std::abs(SignedPolygonArea(polygon)) * 0.5;
}

Line2d PerpendicularBisector(const Segment2d& segment);

inline Line2d PerpendicularBisector(const Point2d& p1, const Point2d& p2) {
    return PerpendicularBisector(Segment2d(p1, p2));
}

inline Circle2d CircleFromDiameter(const Point2d& p1, const Point2d& p2) {
    Point2d center = (p1 + p2) * 0.5;
    double radius = p1.DistanceTo(p2) * 0.5;
    return Circle2d(center, radius);
}

inline Circle2d CircleFromCenterAndPoint(const Point2d& center, const Point2d& pointOnCircle) {
    return Circle2d(center, center.DistanceTo(pointOnCircle));
}

Point2d PolygonCentroid(const std::vector<Point2d>& polygon);
double PolygonPerimeter(const std::vector<Point2d>& polygon, bool closed = true);

} // namespace Qi::Vision::Internal
