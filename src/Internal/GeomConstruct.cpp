/**
 * @file GeomConstruct.cpp
 * @brief Implementation of geometric construction algorithms
 */

#include <QiVision/Internal/GeomConstruct.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace Qi::Vision::Internal {

// =============================================================================
// Perpendicular Functions
// =============================================================================

Line2d PerpendicularLine(const Line2d& line, const Point2d& point) {
    Line2d perp;
    perp.a = line.b;
    perp.b = -line.a;
    perp.c = -line.b * point.x + line.a * point.y;
    return NormalizeLine(perp);
}

Line2d PerpendicularLine(const Segment2d& segment, const Point2d& point) {
    return PerpendicularLine(segment.ToLine(), point);
}

Line2d PerpendicularFromPoint(const Line2d& line, const Point2d& point, Point2d& foot) {
    foot = ProjectPointOnLine(point, line);
    return PerpendicularLine(line, point);
}

Segment2d PerpendicularSegment(const Line2d& line, const Point2d& point) {
    Point2d foot = ProjectPointOnLine(point, line);
    return Segment2d(point, foot);
}

Segment2d PerpendicularSegment(const Segment2d& segment, const Point2d& point, bool clampToSegment) {
    Point2d foot;
    if (clampToSegment) {
        foot = ProjectPointOnSegment(point, segment);
    } else {
        foot = ProjectPointOnLine(point, segment.ToLine());
    }
    return Segment2d(point, foot);
}

// =============================================================================
// Parallel Functions
// =============================================================================

Line2d ParallelLine(const Line2d& line, const Point2d& point) {
    Line2d parallel;
    parallel.a = line.a;
    parallel.b = line.b;
    parallel.c = -(line.a * point.x + line.b * point.y);
    return parallel;
}

Line2d ParallelLineAtDistance(const Line2d& line, double distance) {
    Line2d parallel = line;
    parallel.c = line.c - distance;
    return parallel;
}

std::pair<Line2d, Line2d> ParallelLinesAtDistance(const Line2d& line, double distance) {
    double absDistance = std::abs(distance);
    Line2d line1 = line;
    Line2d line2 = line;
    line1.c = line.c - absDistance;
    line2.c = line.c + absDistance;
    return {line1, line2};
}

Segment2d ParallelSegmentAtDistance(const Segment2d& segment, double distance) {
    Point2d dir = segment.UnitDirection();
    Point2d perp(-dir.y, dir.x);
    Point2d offset = perp * distance;
    return Segment2d(segment.p1 + offset, segment.p2 + offset);
}

Line2d PerpendicularBisector(const Segment2d& segment) {
    Point2d midpoint = segment.Midpoint();
    Point2d dir = segment.Direction();
    Point2d perp(-dir.y, dir.x);
    return Line2d::FromPointAngle(midpoint, std::atan2(perp.y, perp.x));
}

// =============================================================================
// Angle Bisector Functions
// =============================================================================

std::optional<Line2d> AngleBisector(const Line2d& line1, const Line2d& line2) {
    double cross = line1.a * line2.b - line1.b * line2.a;
    if (std::abs(cross) < CONSTRUCT_TOLERANCE) {
        return std::nullopt;
    }
    Line2d bisector;
    bisector.a = line1.a - line2.a;
    bisector.b = line1.b - line2.b;
    bisector.c = line1.c - line2.c;
    return NormalizeLine(bisector);
}

std::optional<std::pair<Line2d, Line2d>> AngleBisectors(const Line2d& line1, const Line2d& line2) {
    double cross = line1.a * line2.b - line1.b * line2.a;
    if (std::abs(cross) < CONSTRUCT_TOLERANCE) {
        return std::nullopt;
    }
    Line2d bisector1;
    bisector1.a = line1.a - line2.a;
    bisector1.b = line1.b - line2.b;
    bisector1.c = line1.c - line2.c;
    Line2d bisector2;
    bisector2.a = line1.a + line2.a;
    bisector2.b = line1.b + line2.b;
    bisector2.c = line1.c + line2.c;
    return std::make_pair(NormalizeLine(bisector1), NormalizeLine(bisector2));
}

Line2d AngleBisectorFromPoints(const Point2d& p1, const Point2d& vertex, const Point2d& p3) {
    Point2d v1 = p1 - vertex;
    Point2d v2 = p3 - vertex;
    double len1 = v1.Norm();
    double len2 = v2.Norm();
    if (len1 < CONSTRUCT_TOLERANCE || len2 < CONSTRUCT_TOLERANCE) {
        return Line2d::FromPointAngle(vertex, 0.0);
    }
    v1 = v1 * (1.0 / len1);
    v2 = v2 * (1.0 / len2);
    Point2d bisectorDir = v1 + v2;
    double bisectorLen = bisectorDir.Norm();
    if (bisectorLen < CONSTRUCT_TOLERANCE) {
        bisectorDir = Point2d(-v1.y, v1.x);
    }
    return Line2d::FromPointAngle(vertex, std::atan2(bisectorDir.y, bisectorDir.x));
}

Line2d AngleBisector(const Segment2d& seg1, const Segment2d& seg2, const Point2d& vertex) {
    Point2d dir1 = (seg1.p1.DistanceTo(vertex) < seg1.p2.DistanceTo(vertex))
                   ? seg1.p2 - vertex : seg1.p1 - vertex;
    Point2d dir2 = (seg2.p1.DistanceTo(vertex) < seg2.p2.DistanceTo(vertex))
                   ? seg2.p2 - vertex : seg2.p1 - vertex;
    double len1 = dir1.Norm();
    double len2 = dir2.Norm();
    if (len1 < CONSTRUCT_TOLERANCE || len2 < CONSTRUCT_TOLERANCE) {
        return Line2d::FromPointAngle(vertex, 0.0);
    }
    dir1 = dir1 * (1.0 / len1);
    dir2 = dir2 * (1.0 / len2);
    Point2d bisectorDir = dir1 + dir2;
    double bisectorLen = bisectorDir.Norm();
    if (bisectorLen < CONSTRUCT_TOLERANCE) {
        bisectorDir = Point2d(-dir1.y, dir1.x);
    }
    return Line2d::FromPointAngle(vertex, std::atan2(bisectorDir.y, bisectorDir.x));
}

// =============================================================================
// Circle Tangent Functions
// =============================================================================

std::vector<Line2d> TangentLinesToCircle(const Circle2d& circle, const Point2d& point) {
    std::vector<Line2d> result;
    double dx = point.x - circle.center.x;
    double dy = point.y - circle.center.y;
    double dist = std::sqrt(dx * dx + dy * dy);
    if (dist < circle.radius - TANGENT_TOLERANCE) {
        return result;
    }
    if (dist < circle.radius + TANGENT_TOLERANCE) {
        Point2d normal(dx / dist, dy / dist);
        Line2d tangent;
        tangent.a = normal.x;
        tangent.b = normal.y;
        tangent.c = -(normal.x * point.x + normal.y * point.y);
        result.push_back(tangent);
        return result;
    }
    double baseAngle = std::atan2(dy, dx);
    double alpha = std::acos(circle.radius / dist);
    double theta1 = baseAngle + alpha;
    double theta2 = baseAngle - alpha;
    Point2d tan1(circle.center.x + circle.radius * std::cos(theta1),
                 circle.center.y + circle.radius * std::sin(theta1));
    Point2d tan2(circle.center.x + circle.radius * std::cos(theta2),
                 circle.center.y + circle.radius * std::sin(theta2));
    result.push_back(Line2d::FromPoints(point, tan1));
    result.push_back(Line2d::FromPoints(point, tan2));
    return result;
}

std::vector<Point2d> TangentPointsToCircle(const Circle2d& circle, const Point2d& point) {
    std::vector<Point2d> result;
    double dx = point.x - circle.center.x;
    double dy = point.y - circle.center.y;
    double dist = std::sqrt(dx * dx + dy * dy);
    if (dist < circle.radius - TANGENT_TOLERANCE) {
        return result;
    }
    if (dist < circle.radius + TANGENT_TOLERANCE) {
        result.push_back(point);
        return result;
    }
    double baseAngle = std::atan2(dy, dx);
    double alpha = std::acos(circle.radius / dist);
    double theta1 = baseAngle + alpha;
    double theta2 = baseAngle - alpha;
    result.emplace_back(circle.center.x + circle.radius * std::cos(theta1),
                        circle.center.y + circle.radius * std::sin(theta1));
    result.emplace_back(circle.center.x + circle.radius * std::cos(theta2),
                        circle.center.y + circle.radius * std::sin(theta2));
    return result;
}

Line2d TangentLineAtAngle(const Circle2d& circle, double angle) {
    Point2d pointOnCircle(circle.center.x + circle.radius * std::cos(angle),
                          circle.center.y + circle.radius * std::sin(angle));
    Point2d normal(std::cos(angle), std::sin(angle));
    Line2d tangent;
    tangent.a = normal.x;
    tangent.b = normal.y;
    tangent.c = -(normal.x * pointOnCircle.x + normal.y * pointOnCircle.y);
    return tangent;
}

Line2d TangentLineAtClosestPoint(const Circle2d& circle, const Point2d& point) {
    double dx = point.x - circle.center.x;
    double dy = point.y - circle.center.y;
    double dist = std::sqrt(dx * dx + dy * dy);
    if (dist < TANGENT_TOLERANCE) {
        return TangentLineAtAngle(circle, 0.0);
    }
    double angle = std::atan2(dy, dx);
    return TangentLineAtAngle(circle, angle);
}

// =============================================================================
// Ellipse Tangent Functions
// =============================================================================

Line2d TangentLineToEllipseAt(const Ellipse2d& ellipse, double theta) {
    double cosT = std::cos(theta);
    double sinT = std::sin(theta);
    double px = ellipse.a * cosT;
    double py = ellipse.b * sinT;
    double tx = -ellipse.a * sinT;
    double ty = ellipse.b * cosT;
    double cosA = std::cos(ellipse.angle);
    double sinA = std::sin(ellipse.angle);
    Point2d pointWorld(
        ellipse.center.x + px * cosA - py * sinA,
        ellipse.center.y + px * sinA + py * cosA
    );
    Point2d tangentDir(
        tx * cosA - ty * sinA,
        tx * sinA + ty * cosA
    );
    return Line2d::FromPointAngle(pointWorld, std::atan2(tangentDir.y, tangentDir.x));
}

std::vector<Line2d> TangentLinesToEllipse(const Ellipse2d& ellipse, const Point2d& point) {
    std::vector<Line2d> result;
    double dx = point.x - ellipse.center.x;
    double dy = point.y - ellipse.center.y;
    double cosA = std::cos(-ellipse.angle);
    double sinA = std::sin(-ellipse.angle);
    double localX = dx * cosA - dy * sinA;
    double localY = dx * sinA + dy * cosA;
    double u = localX / ellipse.a;
    double v = localY / ellipse.b;
    double r = std::sqrt(u * u + v * v);
    if (r < 1.0 - TANGENT_TOLERANCE) {
        return result;
    }
    if (r < 1.0 + TANGENT_TOLERANCE) {
        double t = std::atan2(v * ellipse.a, u * ellipse.b);
        result.push_back(TangentLineToEllipseAt(ellipse, t));
        return result;
    }
    double phi = std::atan2(v, u);
    double alpha = std::acos(1.0 / r);
    double t1 = phi + alpha;
    double t2 = phi - alpha;
    result.push_back(TangentLineToEllipseAt(ellipse, t1));
    result.push_back(TangentLineToEllipseAt(ellipse, t2));
    return result;
}

std::vector<Point2d> TangentPointsToEllipse(const Ellipse2d& ellipse, const Point2d& point) {
    std::vector<Point2d> result;
    double dx = point.x - ellipse.center.x;
    double dy = point.y - ellipse.center.y;
    double cosA = std::cos(-ellipse.angle);
    double sinA = std::sin(-ellipse.angle);
    double localX = dx * cosA - dy * sinA;
    double localY = dx * sinA + dy * cosA;
    double u = localX / ellipse.a;
    double v = localY / ellipse.b;
    double r = std::sqrt(u * u + v * v);
    if (r < 1.0 - TANGENT_TOLERANCE) {
        return result;
    }
    if (r < 1.0 + TANGENT_TOLERANCE) {
        result.push_back(point);
        return result;
    }
    double phi = std::atan2(v, u);
    double alpha = std::acos(1.0 / r);
    double t1 = phi + alpha;
    double t2 = phi - alpha;
    result.push_back(EllipsePointAt(ellipse, t1));
    result.push_back(EllipsePointAt(ellipse, t2));
    return result;
}

// =============================================================================
// Common Tangents (Two Circles)
// =============================================================================

std::vector<Line2d> ExternalCommonTangents(const Circle2d& circle1, const Circle2d& circle2) {
    std::vector<Line2d> result;
    double dx = circle2.center.x - circle1.center.x;
    double dy = circle2.center.y - circle1.center.y;
    double d = std::sqrt(dx * dx + dy * dy);
    if (d < TANGENT_TOLERANCE) {
        return result;
    }
    double r1 = circle1.radius;
    double r2 = circle2.radius;
    if (d < std::abs(r1 - r2) - TANGENT_TOLERANCE) {
        return result;
    }
    double baseAngle = std::atan2(dy, dx);
    double sinAlpha = (r1 - r2) / d;
    if (std::abs(sinAlpha) > 1.0 + TANGENT_TOLERANCE) {
        return result;
    }
    sinAlpha = std::clamp(sinAlpha, -1.0, 1.0);
    double alpha = std::asin(sinAlpha);
    double angle1 = baseAngle + M_PI / 2 - alpha;
    double angle2 = baseAngle - M_PI / 2 + alpha;
    Point2d t1_1(circle1.center.x + r1 * std::cos(angle1),
                 circle1.center.y + r1 * std::sin(angle1));
    Point2d t1_2(circle1.center.x + r1 * std::cos(angle2),
                 circle1.center.y + r1 * std::sin(angle2));
    Point2d t2_1(circle2.center.x + r2 * std::cos(angle1),
                 circle2.center.y + r2 * std::sin(angle1));
    Point2d t2_2(circle2.center.x + r2 * std::cos(angle2),
                 circle2.center.y + r2 * std::sin(angle2));
    result.push_back(Line2d::FromPoints(t1_1, t2_1));
    result.push_back(Line2d::FromPoints(t1_2, t2_2));
    return result;
}

std::vector<Line2d> InternalCommonTangents(const Circle2d& circle1, const Circle2d& circle2) {
    std::vector<Line2d> result;
    double dx = circle2.center.x - circle1.center.x;
    double dy = circle2.center.y - circle1.center.y;
    double d = std::sqrt(dx * dx + dy * dy);
    double r1 = circle1.radius;
    double r2 = circle2.radius;
    if (d < r1 + r2 - TANGENT_TOLERANCE) {
        return result;
    }
    if (d < TANGENT_TOLERANCE) {
        return result;
    }
    double baseAngle = std::atan2(dy, dx);
    double sinAlpha = (r1 + r2) / d;
    if (sinAlpha > 1.0 + TANGENT_TOLERANCE) {
        return result;
    }
    sinAlpha = std::clamp(sinAlpha, -1.0, 1.0);
    double alpha = std::asin(sinAlpha);
    double angle1 = baseAngle + M_PI / 2 - alpha;
    double angle2 = baseAngle - M_PI / 2 + alpha;
    Point2d t1_1(circle1.center.x + r1 * std::cos(angle1),
                 circle1.center.y + r1 * std::sin(angle1));
    Point2d t1_2(circle1.center.x + r1 * std::cos(angle2),
                 circle1.center.y + r1 * std::sin(angle2));
    Point2d t2_1(circle2.center.x + r2 * std::cos(angle1 + M_PI),
                 circle2.center.y + r2 * std::sin(angle1 + M_PI));
    Point2d t2_2(circle2.center.x + r2 * std::cos(angle2 + M_PI),
                 circle2.center.y + r2 * std::sin(angle2 + M_PI));
    result.push_back(Line2d::FromPoints(t1_1, t2_1));
    result.push_back(Line2d::FromPoints(t1_2, t2_2));
    return result;
}

CommonTangentResult CommonTangents(const Circle2d& circle1, const Circle2d& circle2) {
    CommonTangentResult result;
    result.external = ExternalCommonTangents(circle1, circle2);
    result.internal = InternalCommonTangents(circle1, circle2);
    return result;
}

// =============================================================================
// Circumscribed Circle
// =============================================================================

std::optional<Circle2d> CircumscribedCircle(const Point2d& p1, const Point2d& p2, const Point2d& p3) {
    return FitCircleExact3Points(p1, p2, p3);
}

std::optional<Circle2d> CircumscribedCircle(const std::vector<Point2d>& points, double tolerance) {
    if (points.size() < 3) {
        return std::nullopt;
    }
    if (points.size() == 3) {
        return CircumscribedCircle(points[0], points[1], points[2]);
    }
    auto result = FitCircleAlgebraic(points);
    if (!result.success) {
        return std::nullopt;
    }
    for (const auto& p : points) {
        double dist = std::abs(p.DistanceTo(result.circle.center) - result.circle.radius);
        if (dist > tolerance) {
            return std::nullopt;
        }
    }
    return result.circle;
}

// =============================================================================
// Inscribed Circle
// =============================================================================

std::optional<Circle2d> InscribedCircle(const Point2d& p1, const Point2d& p2, const Point2d& p3) {
    double area2 = std::abs((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y));
    if (area2 < CONSTRUCT_TOLERANCE) {
        return std::nullopt;
    }
    double a = p2.DistanceTo(p3);
    double b = p1.DistanceTo(p3);
    double c = p1.DistanceTo(p2);
    double s = (a + b + c) / 2.0;
    double totalWeight = a + b + c;
    Point2d incenter(
        (a * p1.x + b * p2.x + c * p3.x) / totalWeight,
        (a * p1.y + b * p2.y + c * p3.y) / totalWeight
    );
    double area = area2 / 2.0;
    double inradius = area / s;
    return Circle2d(incenter, inradius);
}

std::optional<Circle2d> InscribedCircle(const std::vector<Point2d>& polygon) {
    if (polygon.size() < 3) {
        return std::nullopt;
    }
    if (polygon.size() == 3) {
        return InscribedCircle(polygon[0], polygon[1], polygon[2]);
    }
    Point2d centroid = PolygonCentroid(polygon);
    double minDist = std::numeric_limits<double>::max();
    size_t n = polygon.size();
    for (size_t i = 0; i < n; ++i) {
        Segment2d edge(polygon[i], polygon[(i + 1) % n]);
        double dist = edge.DistanceToPoint(centroid);
        minDist = std::min(minDist, dist);
    }
    if (minDist < CONSTRUCT_TOLERANCE) {
        return std::nullopt;
    }
    return Circle2d(centroid, minDist);
}

// =============================================================================
// Minimum Enclosing Circle (Welzl algorithm)
// =============================================================================

namespace {

Circle2d CircleFrom2Points(const Point2d& p1, const Point2d& p2) {
    Point2d center = (p1 + p2) * 0.5;
    double radius = p1.DistanceTo(p2) / 2.0;
    return Circle2d(center, radius);
}

Circle2d CircleFrom3Points(const Point2d& p1, const Point2d& p2, const Point2d& p3) {
    auto result = FitCircleExact3Points(p1, p2, p3);
    if (result.has_value()) {
        return result.value();
    }
    return CircleFrom2Points(p1, p2);
}

bool IsInsideCircle(const Circle2d& circle, const Point2d& p) {
    return p.DistanceTo(circle.center) <= circle.radius + CONSTRUCT_TOLERANCE;
}

Circle2d MinCircleWithPoint(const std::vector<Point2d>& points, size_t n, const Point2d& q,
                            std::mt19937& rng) {
    Circle2d circle(q, 0.0);
    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = i;
    std::shuffle(indices.begin(), indices.end(), rng);
    for (size_t i = 0; i < n; ++i) {
        const Point2d& p = points[indices[i]];
        if (!IsInsideCircle(circle, p)) {
            circle = CircleFrom2Points(q, p);
            for (size_t j = 0; j < i; ++j) {
                const Point2d& pj = points[indices[j]];
                if (!IsInsideCircle(circle, pj)) {
                    circle = CircleFrom3Points(q, p, pj);
                }
            }
        }
    }
    return circle;
}

} // anonymous namespace

std::optional<Circle2d> MinEnclosingCircle(const std::vector<Point2d>& points) {
    if (points.empty()) {
        return std::nullopt;
    }
    if (points.size() == 1) {
        return Circle2d(points[0], 0.0);
    }
    if (points.size() == 2) {
        return CircleFrom2Points(points[0], points[1]);
    }
    std::vector<Point2d> shuffled = points;
    std::mt19937 rng(12345);
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    Circle2d circle(shuffled[0], 0.0);
    for (size_t i = 1; i < shuffled.size(); ++i) {
        if (!IsInsideCircle(circle, shuffled[i])) {
            circle = MinCircleWithPoint(shuffled, i, shuffled[i], rng);
        }
    }
    return circle;
}

std::optional<Circle2d> MinEnclosingCircleWeighted(const std::vector<Point2d>& points,
                                                    const std::vector<double>& weights) {
    if (points.empty() || points.size() != weights.size()) {
        return std::nullopt;
    }
    double totalWeight = 0.0;
    Point2d weightedCenter(0, 0);
    for (size_t i = 0; i < points.size(); ++i) {
        weightedCenter.x += points[i].x * weights[i];
        weightedCenter.y += points[i].y * weights[i];
        totalWeight += weights[i];
    }
    if (totalWeight < CONSTRUCT_TOLERANCE) {
        return std::nullopt;
    }
    weightedCenter = weightedCenter * (1.0 / totalWeight);
    double maxWeightedDist = 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
        double dist = points[i].DistanceTo(weightedCenter) * weights[i];
        maxWeightedDist = std::max(maxWeightedDist, dist);
    }
    return Circle2d(weightedCenter, maxWeightedDist);
}

// =============================================================================
// Convex Hull (Andrew monotone chain)
// =============================================================================

namespace {

double CrossProduct(const Point2d& o, const Point2d& a, const Point2d& b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

} // anonymous namespace

std::vector<Point2d> ConvexHull(const std::vector<Point2d>& points) {
    size_t n = points.size();
    if (n < 3) {
        return points;
    }
    std::vector<Point2d> sorted = points;
    std::sort(sorted.begin(), sorted.end(), [](const Point2d& a, const Point2d& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    sorted.erase(std::unique(sorted.begin(), sorted.end(),
        [](const Point2d& a, const Point2d& b) {
            return std::abs(a.x - b.x) < CONSTRUCT_TOLERANCE &&
                   std::abs(a.y - b.y) < CONSTRUCT_TOLERANCE;
        }), sorted.end());
    n = sorted.size();
    if (n < 3) {
        return sorted;
    }
    std::vector<Point2d> hull;
    hull.reserve(2 * n);
    for (size_t i = 0; i < n; ++i) {
        while (hull.size() >= 2 &&
               CrossProduct(hull[hull.size()-2], hull[hull.size()-1], sorted[i]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(sorted[i]);
    }
    size_t lowerSize = hull.size();
    for (size_t i = n - 1; i > 0; --i) {
        while (hull.size() > lowerSize &&
               CrossProduct(hull[hull.size()-2], hull[hull.size()-1], sorted[i-1]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(sorted[i-1]);
    }
    hull.pop_back();
    return hull;
}

std::vector<size_t> ConvexHullIndices(const std::vector<Point2d>& points) {
    size_t n = points.size();
    if (n < 3) {
        std::vector<size_t> indices(n);
        for (size_t i = 0; i < n; ++i) indices[i] = i;
        return indices;
    }
    std::vector<size_t> sortedIndices(n);
    for (size_t i = 0; i < n; ++i) sortedIndices[i] = i;
    std::sort(sortedIndices.begin(), sortedIndices.end(),
        [&points](size_t a, size_t b) {
            return points[a].x < points[b].x ||
                   (points[a].x == points[b].x && points[a].y < points[b].y);
        });
    std::vector<size_t> hull;
    hull.reserve(2 * n);
    for (size_t i = 0; i < n; ++i) {
        while (hull.size() >= 2 &&
               CrossProduct(points[hull[hull.size()-2]],
                           points[hull[hull.size()-1]],
                           points[sortedIndices[i]]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(sortedIndices[i]);
    }
    size_t lowerSize = hull.size();
    for (size_t i = n - 1; i > 0; --i) {
        while (hull.size() > lowerSize &&
               CrossProduct(points[hull[hull.size()-2]],
                           points[hull[hull.size()-1]],
                           points[sortedIndices[i-1]]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(sortedIndices[i-1]);
    }
    hull.pop_back();
    return hull;
}

bool IsConvex(const std::vector<Point2d>& polygon) {
    size_t n = polygon.size();
    if (n < 3) return true;
    bool hasPositive = false;
    bool hasNegative = false;
    for (size_t i = 0; i < n; ++i) {
        double cross = CrossProduct(polygon[i],
                                   polygon[(i + 1) % n],
                                   polygon[(i + 2) % n]);
        if (cross > CONSTRUCT_TOLERANCE) hasPositive = true;
        if (cross < -CONSTRUCT_TOLERANCE) hasNegative = true;
        if (hasPositive && hasNegative) return false;
    }
    return true;
}

// =============================================================================
// Polygon Area and Centroid
// =============================================================================

double SignedPolygonArea(const std::vector<Point2d>& polygon) {
    if (polygon.size() < 3) return 0.0;
    double area = 0.0;
    size_t n = polygon.size();
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        area += polygon[i].x * polygon[j].y;
        area -= polygon[j].x * polygon[i].y;
    }
    return area;
}

Point2d PolygonCentroid(const std::vector<Point2d>& polygon) {
    if (polygon.empty()) {
        return Point2d(0, 0);
    }
    if (polygon.size() == 1) {
        return polygon[0];
    }
    if (polygon.size() == 2) {
        return (polygon[0] + polygon[1]) * 0.5;
    }
    double signedArea = SignedPolygonArea(polygon) / 2.0;
    if (std::abs(signedArea) < CONSTRUCT_TOLERANCE) {
        Point2d sum(0, 0);
        for (const auto& p : polygon) {
            sum = sum + p;
        }
        return sum * (1.0 / polygon.size());
    }
    double cx = 0.0, cy = 0.0;
    size_t n = polygon.size();
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        double factor = polygon[i].x * polygon[j].y - polygon[j].x * polygon[i].y;
        cx += (polygon[i].x + polygon[j].x) * factor;
        cy += (polygon[i].y + polygon[j].y) * factor;
    }
    double area6 = 6.0 * signedArea;
    return Point2d(cx / area6, cy / area6);
}

double PolygonPerimeter(const std::vector<Point2d>& polygon, bool closed) {
    if (polygon.size() < 2) return 0.0;
    double perimeter = 0.0;
    size_t n = polygon.size();
    size_t limit = closed ? n : n - 1;
    for (size_t i = 0; i < limit; ++i) {
        perimeter += polygon[i].DistanceTo(polygon[(i + 1) % n]);
    }
    return perimeter;
}

// =============================================================================
// Minimum Bounding Rectangle
// =============================================================================

std::optional<Rect2d> MinBoundingRect(const std::vector<Point2d>& points) {
    if (points.empty()) {
        return std::nullopt;
    }
    double minX = std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double maxY = std::numeric_limits<double>::lowest();
    for (const auto& p : points) {
        minX = std::min(minX, p.x);
        minY = std::min(minY, p.y);
        maxX = std::max(maxX, p.x);
        maxY = std::max(maxY, p.y);
    }
    return Rect2d(minX, minY, maxX - minX, maxY - minY);
}

RotatedRect2d BoundingRectAtAngle(const std::vector<Point2d>& points, double angle) {
    if (points.empty()) {
        return RotatedRect2d(Point2d(0, 0), 0, 0, angle);
    }
    double cosA = std::cos(-angle);
    double sinA = std::sin(-angle);
    double minX = std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double maxY = std::numeric_limits<double>::lowest();
    for (const auto& p : points) {
        double rx = p.x * cosA - p.y * sinA;
        double ry = p.x * sinA + p.y * cosA;
        minX = std::min(minX, rx);
        minY = std::min(minY, ry);
        maxX = std::max(maxX, rx);
        maxY = std::max(maxY, ry);
    }
    double cx = (minX + maxX) / 2.0;
    double cy = (minY + maxY) / 2.0;
    double centerX = cx * std::cos(angle) - cy * std::sin(angle);
    double centerY = cx * std::sin(angle) + cy * std::cos(angle);
    return RotatedRect2d(Point2d(centerX, centerY),
                         maxX - minX, maxY - minY, angle);
}

std::optional<RotatedRect2d> MinAreaRect(const std::vector<Point2d>& points) {
    if (points.size() < 3) {
        if (points.empty()) return std::nullopt;
        if (points.size() == 1) {
            return RotatedRect2d(points[0], 0, 0, 0);
        }
        Point2d center = (points[0] + points[1]) * 0.5;
        double length = points[0].DistanceTo(points[1]);
        double angle = std::atan2(points[1].y - points[0].y,
                                  points[1].x - points[0].x);
        return RotatedRect2d(center, length, 0, angle);
    }
    std::vector<Point2d> hull = ConvexHull(points);
    if (hull.size() < 3) {
        return BoundingRectAtAngle(points, 0.0);
    }
    double minArea = std::numeric_limits<double>::max();
    RotatedRect2d minRect;
    size_t n = hull.size();
    for (size_t i = 0; i < n; ++i) {
        Point2d edge = hull[(i + 1) % n] - hull[i];
        double edgeAngle = std::atan2(edge.y, edge.x);
        RotatedRect2d rect = BoundingRectAtAngle(hull, edgeAngle);
        double area = rect.Area();
        if (area < minArea) {
            minArea = area;
            minRect = rect;
        }
    }
    return minRect;
}

} // namespace Qi::Vision::Internal
