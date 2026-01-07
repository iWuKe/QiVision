/**
 * @file Intersection.cpp
 * @brief Implementation of geometric intersection calculations
 */

#include <QiVision/Internal/Intersection.h>
#include <QiVision/Internal/Distance.h>

#include <algorithm>
#include <cmath>

namespace Qi::Vision::Internal {

// =============================================================================
// Line-Line Intersection
// =============================================================================

IntersectionResult IntersectLineLine(const Line2d& line1, const Line2d& line2) {
    // Normalize lines to ensure a^2 + b^2 = 1
    Line2d l1 = NormalizeLine(line1);
    Line2d l2 = NormalizeLine(line2);

    // Compute determinant: det = a1*b2 - a2*b1
    double det = l1.a * l2.b - l2.a * l1.b;

    // Check if lines are parallel
    if (std::abs(det) < INTERSECTION_SINGULAR_TOLERANCE) {
        return IntersectionResult::None();
    }

    // Solve for intersection point
    // x = (b1*c2 - b2*c1) / det
    // y = (a2*c1 - a1*c2) / det
    double x = (l1.b * l2.c - l2.b * l1.c) / det;
    double y = (l2.a * l1.c - l1.a * l2.c) / det;

    return IntersectionResult::At(Point2d(x, y));
}

bool AreLinesCoincident(const Line2d& line1, const Line2d& line2, double tolerance) {
    Line2d l1 = NormalizeLine(line1);
    Line2d l2 = NormalizeLine(line2);

    // Lines are coincident if:
    // 1. They are parallel (a1*b2 - a2*b1 ~ 0)
    // 2. They have the same c value (or opposite signs for same line)

    double det = l1.a * l2.b - l2.a * l1.b;
    if (std::abs(det) > INTERSECTION_SINGULAR_TOLERANCE) {
        return false;  // Not parallel
    }

    // Check if (a1,b1) and (a2,b2) point in same or opposite direction
    double dot = l1.a * l2.a + l1.b * l2.b;

    // c values should match (same direction) or be negatives (opposite)
    if (dot > 0) {
        return std::abs(l1.c - l2.c) <= tolerance;
    } else {
        return std::abs(l1.c + l2.c) <= tolerance;
    }
}

// =============================================================================
// Line-Segment and Segment-Segment Intersection
// =============================================================================

IntersectionResult IntersectLineSegment(const Line2d& line, const Segment2d& segment) {
    // Convert segment to line
    Line2d segLine = segment.ToLine();

    // Find intersection of two lines
    IntersectionResult result = IntersectLineLine(line, segLine);

    if (!result.exists) {
        return IntersectionResult::None();
    }

    // Check if intersection is within segment
    Point2d dir = segment.p2 - segment.p1;
    double lenSq = dir.Dot(dir);

    if (lenSq < MIN_SEGMENT_LENGTH * MIN_SEGMENT_LENGTH) {
        // Degenerate segment (point)
        double dist = line.Distance(segment.p1);
        if (dist <= GEOM_TOLERANCE) {
            return IntersectionResult::At(segment.p1, 0.0, 0.0);
        }
        return IntersectionResult::None();
    }

    // Compute parameter t on segment
    Point2d toPoint = result.point - segment.p1;
    double t = toPoint.Dot(dir) / lenSq;

    // Check if within [0, 1]
    if (t < -GEOM_TOLERANCE || t > 1.0 + GEOM_TOLERANCE) {
        return IntersectionResult::None();
    }

    // Clamp t to [0, 1]
    t = Clamp(t, 0.0, 1.0);

    // Recompute point for numerical stability
    Point2d clampedPoint = segment.p1 + dir * t;

    return IntersectionResult::At(clampedPoint, 0.0, t);
}

IntersectionResult IntersectSegmentSegment(const Segment2d& seg1, const Segment2d& seg2) {
    Point2d d1 = seg1.p2 - seg1.p1;
    Point2d d2 = seg2.p2 - seg2.p1;

    double cross = d1.Cross(d2);

    // Check for parallel segments
    if (std::abs(cross) < INTERSECTION_SINGULAR_TOLERANCE) {
        // Segments are parallel - check for overlap
        // First check if they're collinear
        Point2d toSeg2 = seg2.p1 - seg1.p1;
        double crossToSeg2 = d1.Cross(toSeg2);

        if (std::abs(crossToSeg2) > GEOM_TOLERANCE * d1.Norm()) {
            // Not collinear, no intersection
            return IntersectionResult::None();
        }

        // Collinear - check for overlap using projection
        double len1Sq = d1.Dot(d1);
        if (len1Sq < MIN_SEGMENT_LENGTH * MIN_SEGMENT_LENGTH) {
            // seg1 is degenerate
            double t = seg2.ProjectPoint(seg1.p1);
            if (t >= -GEOM_TOLERANCE && t <= 1.0 + GEOM_TOLERANCE) {
                return IntersectionResult::At(seg1.p1, 0.0, Clamp(t, 0.0, 1.0));
            }
            return IntersectionResult::None();
        }

        // Project seg2 endpoints onto seg1
        double t1 = (seg2.p1 - seg1.p1).Dot(d1) / len1Sq;
        double t2 = (seg2.p2 - seg1.p1).Dot(d1) / len1Sq;

        // Ensure t1 <= t2
        if (t1 > t2) std::swap(t1, t2);

        // Check for overlap with [0, 1]
        if (t2 < -GEOM_TOLERANCE || t1 > 1.0 + GEOM_TOLERANCE) {
            return IntersectionResult::None();
        }

        // Return one endpoint of the overlap
        double tOverlap = Clamp(std::max(t1, 0.0), 0.0, 1.0);
        Point2d overlapPoint = seg1.p1 + d1 * tOverlap;

        // Compute corresponding t on seg2
        double len2Sq = d2.Dot(d2);
        double s = len2Sq > MIN_SEGMENT_LENGTH * MIN_SEGMENT_LENGTH
                       ? (overlapPoint - seg2.p1).Dot(d2) / len2Sq
                       : 0.0;
        s = Clamp(s, 0.0, 1.0);

        return IntersectionResult::At(overlapPoint, tOverlap, s);
    }

    // Non-parallel segments
    Point2d toSeg2 = seg2.p1 - seg1.p1;

    double t = toSeg2.Cross(d2) / cross;
    double s = toSeg2.Cross(d1) / cross;

    // Check if intersection is within both segments
    if (t < -GEOM_TOLERANCE || t > 1.0 + GEOM_TOLERANCE ||
        s < -GEOM_TOLERANCE || s > 1.0 + GEOM_TOLERANCE) {
        return IntersectionResult::None();
    }

    // Clamp parameters
    t = Clamp(t, 0.0, 1.0);
    s = Clamp(s, 0.0, 1.0);

    Point2d intersection = seg1.p1 + d1 * t;

    return IntersectionResult::At(intersection, t, s);
}

// =============================================================================
// Line/Segment with Circle
// =============================================================================

IntersectionResult2 IntersectLineCircle(const Line2d& line, const Circle2d& circle) {
    if (circle.radius <= 0) {
        // Degenerate circle (point)
        if (line.Distance(circle.center) <= GEOM_TOLERANCE) {
            return IntersectionResult2::One(circle.center, 0.0, 0.0);
        }
        return IntersectionResult2::None();
    }

    Line2d l = NormalizeLine(line);

    // Distance from center to line
    double dist = std::abs(l.a * circle.center.x + l.b * circle.center.y + l.c);

    if (dist > circle.radius + GEOM_TOLERANCE) {
        // No intersection
        return IntersectionResult2::None();
    }

    // Foot of perpendicular from center to line
    Point2d foot = ProjectPointOnLine(circle.center, l);

    // Direction along line
    Point2d dir = l.Direction();

    if (dist >= circle.radius - GEOM_TOLERANCE) {
        // Tangent case (1 point)
        double angle = std::atan2(foot.y - circle.center.y, foot.x - circle.center.x);
        return IntersectionResult2::One(foot, 0.0, NormalizeAngle0To2PI(angle));
    }

    // Two intersection points
    double halfChord = std::sqrt(circle.radius * circle.radius - dist * dist);

    Point2d p1 = foot + dir * halfChord;
    Point2d p2 = foot - dir * halfChord;

    double angle1 = std::atan2(p1.y - circle.center.y, p1.x - circle.center.x);
    double angle2 = std::atan2(p2.y - circle.center.y, p2.x - circle.center.x);

    return IntersectionResult2::Two(p1, p2, halfChord, NormalizeAngle0To2PI(angle1),
                                    -halfChord, NormalizeAngle0To2PI(angle2));
}

IntersectionResult2 IntersectSegmentCircle(const Segment2d& segment, const Circle2d& circle) {
    // First intersect the infinite line with circle
    Line2d line = segment.ToLine();
    IntersectionResult2 lineResult = IntersectLineCircle(line, circle);

    if (!lineResult.HasIntersection()) {
        return IntersectionResult2::None();
    }

    // Check which intersection points are within segment
    Point2d dir = segment.p2 - segment.p1;
    double lenSq = dir.Dot(dir);

    if (lenSq < MIN_SEGMENT_LENGTH * MIN_SEGMENT_LENGTH) {
        // Degenerate segment
        double dist = circle.center.DistanceTo(segment.p1);
        if (std::abs(dist - circle.radius) <= GEOM_TOLERANCE) {
            double angle = std::atan2(segment.p1.y - circle.center.y,
                                      segment.p1.x - circle.center.x);
            return IntersectionResult2::One(segment.p1, 0.0, NormalizeAngle0To2PI(angle));
        }
        return IntersectionResult2::None();
    }

    IntersectionResult2 result = IntersectionResult2::None();

    // Helper to check and add point
    auto checkPoint = [&](const Point2d& p, double circleAngle) {
        double t = (p - segment.p1).Dot(dir) / lenSq;
        if (t >= -GEOM_TOLERANCE && t <= 1.0 + GEOM_TOLERANCE) {
            t = Clamp(t, 0.0, 1.0);
            if (result.count == 0) {
                result.count = 1;
                result.point1 = p;
                result.param1_1 = t;
                result.param1_2 = circleAngle;
            } else {
                result.count = 2;
                result.point2 = p;
                result.param2_1 = t;
                result.param2_2 = circleAngle;
            }
        }
    };

    checkPoint(lineResult.point1, lineResult.param1_2);
    if (lineResult.count == 2) {
        checkPoint(lineResult.point2, lineResult.param2_2);
    }

    // Sort by t parameter if two intersections
    if (result.count == 2 && result.param1_1 > result.param2_1) {
        std::swap(result.point1, result.point2);
        std::swap(result.param1_1, result.param2_1);
        std::swap(result.param1_2, result.param2_2);
    }

    return result;
}

bool LineIntersectsCircle(const Line2d& line, const Circle2d& circle) {
    Line2d l = NormalizeLine(line);
    double dist = std::abs(l.a * circle.center.x + l.b * circle.center.y + l.c);
    return dist <= circle.radius + GEOM_TOLERANCE;
}

bool SegmentIntersectsCircle(const Segment2d& segment, const Circle2d& circle) {
    // First check if either endpoint is inside circle
    if (circle.center.DistanceTo(segment.p1) <= circle.radius) return true;
    if (circle.center.DistanceTo(segment.p2) <= circle.radius) return true;

    // Check closest point on segment to circle center
    Point2d closest = ProjectPointOnSegment(circle.center, segment);
    return circle.center.DistanceTo(closest) <= circle.radius;
}

// =============================================================================
// Line/Segment with Ellipse
// =============================================================================

IntersectionResult2 IntersectLineEllipse(const Line2d& line, const Ellipse2d& ellipse) {
    if (ellipse.a <= 0 || ellipse.b <= 0) {
        return IntersectionResult2::None();
    }

    // Transform line to ellipse-local coordinates
    // 1. Translate so ellipse center is at origin
    // 2. Rotate by -ellipse.angle

    double cosA = std::cos(-ellipse.angle);
    double sinA = std::sin(-ellipse.angle);

    // Transform line coefficients
    // Original line: ax + by + c = 0
    // After translation: a(x' + cx) + b(y' + cy) + c = 0
    //                  = ax' + by' + (a*cx + b*cy + c) = 0
    // After rotation: need to transform (a,b) as well

    Line2d l = NormalizeLine(line);

    // The line normal (a, b) rotates as a vector
    double aLocal = l.a * cosA - l.b * sinA;
    double bLocal = l.a * sinA + l.b * cosA;

    // The constant c transforms with the center translation
    double cLocal = l.a * ellipse.center.x + l.b * ellipse.center.y + l.c;

    // Now intersect with axis-aligned ellipse x^2/a^2 + y^2/b^2 = 1
    // Line: aLocal * x + bLocal * y + cLocal = 0

    // If line is nearly horizontal (bLocal ~ 0): solve for y
    // If line is nearly vertical (aLocal ~ 0): solve for x

    double aa = ellipse.a * ellipse.a;
    double bb = ellipse.b * ellipse.b;

    double A, B, C;

    if (std::abs(bLocal) > std::abs(aLocal)) {
        // Express y in terms of x: y = -(aLocal*x + cLocal) / bLocal
        // Substitute into ellipse equation
        // x^2/a^2 + (aLocal*x + cLocal)^2 / (bLocal^2 * b^2) = 1
        double k = aLocal / bLocal;
        double m = cLocal / bLocal;

        // x^2/a^2 + (kx + m)^2/b^2 = 1
        // x^2/a^2 + (k^2*x^2 + 2km*x + m^2)/b^2 = 1
        // x^2*(1/a^2 + k^2/b^2) + x*(2km/b^2) + (m^2/b^2 - 1) = 0

        A = 1.0 / aa + k * k / bb;
        B = 2.0 * k * m / bb;
        C = m * m / bb - 1.0;
    } else {
        // Express x in terms of y: x = -(bLocal*y + cLocal) / aLocal
        double k = bLocal / aLocal;
        double m = cLocal / aLocal;

        A = k * k / aa + 1.0 / bb;
        B = 2.0 * k * m / aa;
        C = m * m / aa - 1.0;
    }

    // Solve quadratic At^2 + Bt + C = 0
    double discriminant = B * B - 4.0 * A * C;

    if (discriminant < -GEOM_TOLERANCE) {
        return IntersectionResult2::None();
    }

    if (discriminant < GEOM_TOLERANCE) {
        discriminant = 0;
    }

    double sqrtD = std::sqrt(discriminant);
    double t1 = (-B + sqrtD) / (2.0 * A);
    double t2 = (-B - sqrtD) / (2.0 * A);

    // Compute local intersection points
    auto computeLocalPoint = [&](double t) -> Point2d {
        if (std::abs(bLocal) > std::abs(aLocal)) {
            double x = t;
            double y = -(aLocal * x + cLocal) / bLocal;
            return Point2d(x, y);
        } else {
            double y = t;
            double x = -(bLocal * y + cLocal) / aLocal;
            return Point2d(x, y);
        }
    };

    // Transform back to world coordinates
    auto toWorld = [&](const Point2d& local) -> Point2d {
        double wx = local.x * std::cos(ellipse.angle) - local.y * std::sin(ellipse.angle);
        double wy = local.x * std::sin(ellipse.angle) + local.y * std::cos(ellipse.angle);
        return Point2d(wx + ellipse.center.x, wy + ellipse.center.y);
    };

    // Compute theta parameter on ellipse (atan2(y/b, x/a) in local coords)
    auto computeTheta = [&](const Point2d& local) -> double {
        return std::atan2(local.y / ellipse.b, local.x / ellipse.a);
    };

    if (std::abs(t1 - t2) < GEOM_TOLERANCE) {
        // Tangent case
        Point2d local = computeLocalPoint(t1);
        Point2d world = toWorld(local);
        double theta = computeTheta(local);
        return IntersectionResult2::One(world, 0.0, theta);
    }

    Point2d local1 = computeLocalPoint(t1);
    Point2d local2 = computeLocalPoint(t2);
    Point2d world1 = toWorld(local1);
    Point2d world2 = toWorld(local2);
    double theta1 = computeTheta(local1);
    double theta2 = computeTheta(local2);

    return IntersectionResult2::Two(world1, world2, 0.0, theta1, 0.0, theta2);
}

IntersectionResult2 IntersectSegmentEllipse(const Segment2d& segment, const Ellipse2d& ellipse) {
    // Intersect line with ellipse
    Line2d line = segment.ToLine();
    IntersectionResult2 lineResult = IntersectLineEllipse(line, ellipse);

    if (!lineResult.HasIntersection()) {
        return IntersectionResult2::None();
    }

    // Filter by segment parameter
    Point2d dir = segment.p2 - segment.p1;
    double lenSq = dir.Dot(dir);

    if (lenSq < MIN_SEGMENT_LENGTH * MIN_SEGMENT_LENGTH) {
        // Degenerate segment
        if (ellipse.Contains(segment.p1)) {
            // Check if on boundary
            Point2d local = segment.p1 - ellipse.center;
            double cosA = std::cos(-ellipse.angle);
            double sinA = std::sin(-ellipse.angle);
            double lx = local.x * cosA - local.y * sinA;
            double ly = local.x * sinA + local.y * cosA;
            double val = (lx * lx) / (ellipse.a * ellipse.a) +
                         (ly * ly) / (ellipse.b * ellipse.b);
            if (std::abs(val - 1.0) < GEOM_TOLERANCE) {
                double theta = std::atan2(ly / ellipse.b, lx / ellipse.a);
                return IntersectionResult2::One(segment.p1, 0.0, theta);
            }
        }
        return IntersectionResult2::None();
    }

    IntersectionResult2 result = IntersectionResult2::None();

    auto checkPoint = [&](const Point2d& p, double theta) {
        double t = (p - segment.p1).Dot(dir) / lenSq;
        if (t >= -GEOM_TOLERANCE && t <= 1.0 + GEOM_TOLERANCE) {
            t = Clamp(t, 0.0, 1.0);
            if (result.count == 0) {
                result.count = 1;
                result.point1 = p;
                result.param1_1 = t;
                result.param1_2 = theta;
            } else {
                result.count = 2;
                result.point2 = p;
                result.param2_1 = t;
                result.param2_2 = theta;
            }
        }
    };

    checkPoint(lineResult.point1, lineResult.param1_2);
    if (lineResult.count == 2) {
        checkPoint(lineResult.point2, lineResult.param2_2);
    }

    // Sort by t parameter
    if (result.count == 2 && result.param1_1 > result.param2_1) {
        std::swap(result.point1, result.point2);
        std::swap(result.param1_1, result.param2_1);
        std::swap(result.param1_2, result.param2_2);
    }

    return result;
}

// =============================================================================
// Line/Segment with Arc
// =============================================================================

bool AngleWithinArc(double angle, const Arc2d& arc) {
    // Normalize angle to [0, 2*PI)
    angle = NormalizeAngle0To2PI(angle);
    double start = NormalizeAngle0To2PI(arc.startAngle);
    double sweep = arc.sweepAngle;

    if (std::abs(sweep) >= TWO_PI - ANGLE_TOLERANCE) {
        return true;  // Full circle
    }

    if (sweep > 0) {
        // CCW arc
        double end = start + sweep;
        if (end <= TWO_PI) {
            return angle >= start - ANGLE_TOLERANCE && angle <= end + ANGLE_TOLERANCE;
        } else {
            // Arc wraps around 2*PI
            return angle >= start - ANGLE_TOLERANCE || angle <= end - TWO_PI + ANGLE_TOLERANCE;
        }
    } else {
        // CW arc (negative sweep)
        double end = start + sweep;  // end < start
        if (end >= 0) {
            return angle <= start + ANGLE_TOLERANCE && angle >= end - ANGLE_TOLERANCE;
        } else {
            // Arc wraps around 0
            return angle <= start + ANGLE_TOLERANCE || angle >= end + TWO_PI - ANGLE_TOLERANCE;
        }
    }
}

IntersectionResult2 IntersectLineArc(const Line2d& line, const Arc2d& arc) {
    // First intersect with full circle
    Circle2d circle = arc.ToCircle();
    IntersectionResult2 circleResult = IntersectLineCircle(line, circle);

    if (!circleResult.HasIntersection()) {
        return IntersectionResult2::None();
    }

    IntersectionResult2 result = IntersectionResult2::None();

    auto checkPoint = [&](const Point2d& p, double circleAngle) {
        if (AngleWithinArc(circleAngle, arc)) {
            if (result.count == 0) {
                result.count = 1;
                result.point1 = p;
                result.param1_2 = circleAngle;
            } else {
                result.count = 2;
                result.point2 = p;
                result.param2_2 = circleAngle;
            }
        }
    };

    checkPoint(circleResult.point1, circleResult.param1_2);
    if (circleResult.count == 2) {
        checkPoint(circleResult.point2, circleResult.param2_2);
    }

    return result;
}

IntersectionResult2 IntersectSegmentArc(const Segment2d& segment, const Arc2d& arc) {
    // First intersect segment with full circle
    IntersectionResult2 circleResult = IntersectSegmentCircle(segment, arc.ToCircle());

    if (!circleResult.HasIntersection()) {
        return IntersectionResult2::None();
    }

    IntersectionResult2 result = IntersectionResult2::None();

    auto checkPoint = [&](const Point2d& p, double t, double circleAngle) {
        if (AngleWithinArc(circleAngle, arc)) {
            if (result.count == 0) {
                result.count = 1;
                result.point1 = p;
                result.param1_1 = t;
                result.param1_2 = circleAngle;
            } else {
                result.count = 2;
                result.point2 = p;
                result.param2_1 = t;
                result.param2_2 = circleAngle;
            }
        }
    };

    checkPoint(circleResult.point1, circleResult.param1_1, circleResult.param1_2);
    if (circleResult.count == 2) {
        checkPoint(circleResult.point2, circleResult.param2_1, circleResult.param2_2);
    }

    // Sort by segment parameter t
    if (result.count == 2 && result.param1_1 > result.param2_1) {
        std::swap(result.point1, result.point2);
        std::swap(result.param1_1, result.param2_1);
        std::swap(result.param1_2, result.param2_2);
    }

    return result;
}

// =============================================================================
// Circle-Circle Intersection
// =============================================================================

IntersectionResult2 IntersectCircleCircle(const Circle2d& circle1, const Circle2d& circle2) {
    double dx = circle2.center.x - circle1.center.x;
    double dy = circle2.center.y - circle1.center.y;
    double d = std::sqrt(dx * dx + dy * dy);

    double r1 = circle1.radius;
    double r2 = circle2.radius;

    // Check for degenerate cases
    if (r1 <= 0 && r2 <= 0) {
        // Both degenerate to points
        if (d < GEOM_TOLERANCE) {
            return IntersectionResult2::One(circle1.center);
        }
        return IntersectionResult2::None();
    }

    if (r1 <= 0) {
        // circle1 is a point
        double dist = circle2.center.DistanceTo(circle1.center);
        if (std::abs(dist - r2) < GEOM_TOLERANCE) {
            double angle = std::atan2(circle1.center.y - circle2.center.y,
                                      circle1.center.x - circle2.center.x);
            return IntersectionResult2::One(circle1.center, 0.0, NormalizeAngle0To2PI(angle));
        }
        return IntersectionResult2::None();
    }

    if (r2 <= 0) {
        // circle2 is a point
        double dist = circle1.center.DistanceTo(circle2.center);
        if (std::abs(dist - r1) < GEOM_TOLERANCE) {
            double angle = std::atan2(circle2.center.y - circle1.center.y,
                                      circle2.center.x - circle1.center.x);
            return IntersectionResult2::One(circle2.center, NormalizeAngle0To2PI(angle), 0.0);
        }
        return IntersectionResult2::None();
    }

    // Check for no intersection
    if (d > r1 + r2 + GEOM_TOLERANCE) {
        // Circles are too far apart
        return IntersectionResult2::None();
    }

    if (d < std::abs(r1 - r2) - GEOM_TOLERANCE) {
        // One circle contains the other
        return IntersectionResult2::None();
    }

    // Check for coincident circles
    if (d < GEOM_TOLERANCE && std::abs(r1 - r2) < GEOM_TOLERANCE) {
        return IntersectionResult2::None();  // Infinite intersections
    }

    // Tangent cases
    if (std::abs(d - (r1 + r2)) < GEOM_TOLERANCE) {
        // External tangent
        double ratio = r1 / (r1 + r2);
        Point2d p(circle1.center.x + dx * ratio, circle1.center.y + dy * ratio);
        double angle1 = std::atan2(p.y - circle1.center.y, p.x - circle1.center.x);
        double angle2 = std::atan2(p.y - circle2.center.y, p.x - circle2.center.x);
        return IntersectionResult2::One(p, NormalizeAngle0To2PI(angle1),
                                        NormalizeAngle0To2PI(angle2));
    }

    if (std::abs(d - std::abs(r1 - r2)) < GEOM_TOLERANCE) {
        // Internal tangent
        double ratio = r1 > r2 ? r1 / d : -r2 / d;
        Point2d p(circle1.center.x + dx * ratio, circle1.center.y + dy * ratio);
        double angle1 = std::atan2(p.y - circle1.center.y, p.x - circle1.center.x);
        double angle2 = std::atan2(p.y - circle2.center.y, p.x - circle2.center.x);
        return IntersectionResult2::One(p, NormalizeAngle0To2PI(angle1),
                                        NormalizeAngle0To2PI(angle2));
    }

    // Two intersection points
    // Using the radical line method
    double a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
    double h = std::sqrt(r1 * r1 - a * a);

    // Point on line between centers at distance a from center1
    double px = circle1.center.x + a * dx / d;
    double py = circle1.center.y + a * dy / d;

    // Perpendicular offset
    double ox = h * dy / d;
    double oy = -h * dx / d;

    Point2d p1(px + ox, py + oy);
    Point2d p2(px - ox, py - oy);

    double angle1_1 = std::atan2(p1.y - circle1.center.y, p1.x - circle1.center.x);
    double angle1_2 = std::atan2(p1.y - circle2.center.y, p1.x - circle2.center.x);
    double angle2_1 = std::atan2(p2.y - circle1.center.y, p2.x - circle1.center.x);
    double angle2_2 = std::atan2(p2.y - circle2.center.y, p2.x - circle2.center.x);

    return IntersectionResult2::Two(p1, p2,
                                    NormalizeAngle0To2PI(angle1_1),
                                    NormalizeAngle0To2PI(angle1_2),
                                    NormalizeAngle0To2PI(angle2_1),
                                    NormalizeAngle0To2PI(angle2_2));
}

bool CirclesIntersect(const Circle2d& circle1, const Circle2d& circle2) {
    double d = circle1.center.DistanceTo(circle2.center);
    double r1 = circle1.radius;
    double r2 = circle2.radius;

    return d <= r1 + r2 + GEOM_TOLERANCE && d >= std::abs(r1 - r2) - GEOM_TOLERANCE;
}

int CircleRelation(const Circle2d& circle1, const Circle2d& circle2) {
    double d = circle1.center.DistanceTo(circle2.center);
    double r1 = circle1.radius;
    double r2 = circle2.radius;

    // Coincident
    if (d < GEOM_TOLERANCE && std::abs(r1 - r2) < GEOM_TOLERANCE) {
        return 4;
    }

    // One contains the other
    if (d < std::abs(r1 - r2) - GEOM_TOLERANCE) {
        return r1 > r2 ? -1 : -2;
    }

    // Internally tangent
    if (std::abs(d - std::abs(r1 - r2)) < GEOM_TOLERANCE) {
        return 3;
    }

    // Intersect at 2 points
    if (d < r1 + r2 - GEOM_TOLERANCE) {
        return 2;
    }

    // Externally tangent
    if (std::abs(d - (r1 + r2)) < GEOM_TOLERANCE) {
        return 1;
    }

    // Separate
    return 0;
}

// =============================================================================
// Circle-Ellipse Intersection
// =============================================================================

std::vector<Point2d> IntersectCircleEllipse(
    const Circle2d& circle,
    const Ellipse2d& ellipse,
    int maxIterations,
    double tolerance) {

    std::vector<Point2d> results;

    if (ellipse.a <= 0 || ellipse.b <= 0 || circle.radius <= 0) {
        return results;
    }

    // Transform to ellipse-local coordinates (center at origin, axes aligned)
    double cosA = std::cos(-ellipse.angle);
    double sinA = std::sin(-ellipse.angle);

    // Transform circle center to ellipse-local
    Point2d localCenter;
    localCenter.x = (circle.center.x - ellipse.center.x) * cosA -
                    (circle.center.y - ellipse.center.y) * sinA;
    localCenter.y = (circle.center.x - ellipse.center.x) * sinA +
                    (circle.center.y - ellipse.center.y) * cosA;

    // In local coords:
    // Ellipse: x^2/a^2 + y^2/b^2 = 1
    // Circle: (x - cx)^2 + (y - cy)^2 = r^2

    double a = ellipse.a;
    double b = ellipse.b;
    double r = circle.radius;
    double cx = localCenter.x;
    double cy = localCenter.y;

    // Sample initial angles and use Newton's method
    // Looking for points where both equations are satisfied
    const int numSamples = 32;
    std::vector<double> initialAngles;

    for (int i = 0; i < numSamples; ++i) {
        double theta = TWO_PI * i / numSamples;
        double x = a * std::cos(theta);
        double y = b * std::sin(theta);

        // Check if close to circle
        double circleEq = (x - cx) * (x - cx) + (y - cy) * (y - cy) - r * r;
        if (std::abs(circleEq) < r * r * 0.5) {
            initialAngles.push_back(theta);
        }
    }

    // Newton-Raphson to refine each candidate
    for (double initTheta : initialAngles) {
        double theta = initTheta;

        for (int iter = 0; iter < maxIterations; ++iter) {
            double x = a * std::cos(theta);
            double y = b * std::sin(theta);

            // f = (x-cx)^2 + (y-cy)^2 - r^2
            double f = (x - cx) * (x - cx) + (y - cy) * (y - cy) - r * r;

            if (std::abs(f) < tolerance * r * r) {
                // Converged - check if unique
                Point2d localPoint(x, y);

                // Transform back to world
                double wx = localPoint.x * std::cos(ellipse.angle) -
                            localPoint.y * std::sin(ellipse.angle) + ellipse.center.x;
                double wy = localPoint.x * std::sin(ellipse.angle) +
                            localPoint.y * std::cos(ellipse.angle) + ellipse.center.y;
                Point2d worldPoint(wx, wy);

                // Check if already found
                bool duplicate = false;
                for (const auto& existing : results) {
                    if (existing.DistanceTo(worldPoint) < tolerance * 10) {
                        duplicate = true;
                        break;
                    }
                }

                if (!duplicate) {
                    results.push_back(worldPoint);
                }
                break;
            }

            // df/dtheta = 2(x-cx)(-a*sin(theta)) + 2(y-cy)(b*cos(theta))
            double dfdTheta = 2.0 * (x - cx) * (-a * std::sin(theta)) +
                              2.0 * (y - cy) * (b * std::cos(theta));

            if (std::abs(dfdTheta) < tolerance) {
                break;  // Can't continue
            }

            theta -= f / dfdTheta;
        }
    }

    return results;
}

// =============================================================================
// Ellipse-Ellipse Intersection
// =============================================================================

std::vector<Point2d> IntersectEllipseEllipse(
    const Ellipse2d& ellipse1,
    const Ellipse2d& ellipse2,
    int maxIterations,
    double tolerance) {

    std::vector<Point2d> results;

    if (ellipse1.a <= 0 || ellipse1.b <= 0 || ellipse2.a <= 0 || ellipse2.b <= 0) {
        return results;
    }

    // Use parametric search with Newton-Raphson refinement
    // Sample both ellipses and look for close points

    const int numSamples = 64;

    for (int i = 0; i < numSamples; ++i) {
        double theta1 = TWO_PI * i / numSamples;
        Point2d p1 = ellipse1.PointAt(theta1);

        // Find closest point on ellipse2
        double bestTheta2 = 0;
        double bestDist = std::numeric_limits<double>::max();

        for (int j = 0; j < numSamples; ++j) {
            double theta2 = TWO_PI * j / numSamples;
            Point2d p2 = ellipse2.PointAt(theta2);
            double dist = p1.DistanceTo(p2);
            if (dist < bestDist) {
                bestDist = dist;
                bestTheta2 = theta2;
            }
        }

        // Skip if too far
        if (bestDist > std::max(ellipse1.a, ellipse2.a) * 0.5) {
            continue;
        }

        // Newton-Raphson refinement
        double t1 = theta1;
        double t2 = bestTheta2;

        for (int iter = 0; iter < maxIterations; ++iter) {
            Point2d p1Current = ellipse1.PointAt(t1);
            Point2d p2Current = ellipse2.PointAt(t2);

            Point2d diff = p1Current - p2Current;
            double dist = diff.Norm();

            if (dist < tolerance) {
                // Converged
                Point2d midpoint = (p1Current + p2Current) * 0.5;

                bool duplicate = false;
                for (const auto& existing : results) {
                    if (existing.DistanceTo(midpoint) < tolerance * 10) {
                        duplicate = true;
                        break;
                    }
                }

                if (!duplicate) {
                    results.push_back(midpoint);
                }
                break;
            }

            // Compute tangent vectors
            Point2d tan1 = EllipseTangentAt(ellipse1, t1) * ellipse1.a;  // Scale for stability
            Point2d tan2 = EllipseTangentAt(ellipse2, t2) * ellipse2.a;

            // Solve: diff + dt1*tan1 - dt2*tan2 = 0 (approximately)
            // [tan1.x, -tan2.x] [dt1]   [-diff.x]
            // [tan1.y, -tan2.y] [dt2] = [-diff.y]

            double det = tan1.x * (-tan2.y) - (-tan2.x) * tan1.y;
            if (std::abs(det) < tolerance) {
                break;
            }

            double dt1 = ((-diff.x) * (-tan2.y) - (-tan2.x) * (-diff.y)) / det;
            double dt2 = (tan1.x * (-diff.y) - (-diff.x) * tan1.y) / det;

            // Damping for stability
            double maxStep = 0.3;
            if (std::abs(dt1) > maxStep) dt1 = maxStep * (dt1 > 0 ? 1 : -1);
            if (std::abs(dt2) > maxStep) dt2 = maxStep * (dt2 > 0 ? 1 : -1);

            t1 += dt1;
            t2 += dt2;
        }
    }

    return results;
}

// =============================================================================
// Line/Segment with RotatedRect
// =============================================================================

IntersectionResult2 IntersectLineRotatedRect(const Line2d& line, const RotatedRect2d& rect) {
    std::array<Segment2d, 4> edges = RotatedRectEdges(rect);

    IntersectionResult2 result = IntersectionResult2::None();

    for (int i = 0; i < 4; ++i) {
        IntersectionResult edgeResult = IntersectLineSegment(line, edges[i]);
        if (edgeResult.exists) {
            if (result.count == 0) {
                result.count = 1;
                result.point1 = edgeResult.point;
                result.param1_2 = static_cast<double>(i);  // Edge index
            } else {
                // Check if duplicate
                if (result.point1.DistanceTo(edgeResult.point) > GEOM_TOLERANCE) {
                    result.count = 2;
                    result.point2 = edgeResult.point;
                    result.param2_2 = static_cast<double>(i);
                    break;  // Maximum 2 intersections
                }
            }
        }
    }

    return result;
}

IntersectionResult2 IntersectSegmentRotatedRect(const Segment2d& segment,
                                                 const RotatedRect2d& rect) {
    std::array<Segment2d, 4> edges = RotatedRectEdges(rect);

    IntersectionResult2 result = IntersectionResult2::None();

    for (int i = 0; i < 4; ++i) {
        IntersectionResult edgeResult = IntersectSegmentSegment(segment, edges[i]);
        if (edgeResult.exists) {
            if (result.count == 0) {
                result.count = 1;
                result.point1 = edgeResult.point;
                result.param1_1 = edgeResult.param1;  // t on segment
                result.param1_2 = static_cast<double>(i);
            } else {
                if (result.point1.DistanceTo(edgeResult.point) > GEOM_TOLERANCE) {
                    result.count = 2;
                    result.point2 = edgeResult.point;
                    result.param2_1 = edgeResult.param1;
                    result.param2_2 = static_cast<double>(i);
                    break;
                }
            }
        }
    }

    // Sort by t parameter on segment
    if (result.count == 2 && result.param1_1 > result.param2_1) {
        std::swap(result.point1, result.point2);
        std::swap(result.param1_1, result.param2_1);
        std::swap(result.param1_2, result.param2_2);
    }

    return result;
}

// =============================================================================
// Batch Intersection Operations
// =============================================================================

std::vector<IntersectionResult> IntersectLineWithSegments(
    const Line2d& line,
    const std::vector<Segment2d>& segments) {

    std::vector<IntersectionResult> results;
    results.reserve(segments.size());

    for (const auto& seg : segments) {
        results.push_back(IntersectLineSegment(line, seg));
    }

    return results;
}

std::vector<IntersectionResult> IntersectLineWithContour(
    const Line2d& line,
    const std::vector<Point2d>& contourPoints,
    bool closed) {

    std::vector<IntersectionResult> results;

    if (contourPoints.size() < 2) {
        return results;
    }

    size_t numSegments = closed ? contourPoints.size() : contourPoints.size() - 1;

    for (size_t i = 0; i < numSegments; ++i) {
        size_t j = (i + 1) % contourPoints.size();
        Segment2d seg(contourPoints[i], contourPoints[j]);

        IntersectionResult ir = IntersectLineSegment(line, seg);
        if (ir.exists) {
            results.push_back(ir);
        }
    }

    return results;
}

std::vector<IntersectionResult> IntersectSegmentWithContour(
    const Segment2d& segment,
    const std::vector<Point2d>& contourPoints,
    bool closed) {

    std::vector<IntersectionResult> results;

    if (contourPoints.size() < 2) {
        return results;
    }

    size_t numSegments = closed ? contourPoints.size() : contourPoints.size() - 1;

    for (size_t i = 0; i < numSegments; ++i) {
        size_t j = (i + 1) % contourPoints.size();
        Segment2d seg(contourPoints[i], contourPoints[j]);

        IntersectionResult ir = IntersectSegmentSegment(segment, seg);
        if (ir.exists) {
            results.push_back(ir);
        }
    }

    return results;
}

// =============================================================================
// Segment Overlap and Clipping
// =============================================================================

bool SegmentsOverlap(const Segment2d& seg1, const Segment2d& seg2, double tolerance) {
    // Check collinearity
    Line2d line1 = seg1.ToLine();
    if (line1.Distance(seg2.p1) > tolerance || line1.Distance(seg2.p2) > tolerance) {
        return false;
    }

    // Project seg2 onto seg1's direction
    Point2d d = seg1.p2 - seg1.p1;
    double lenSq = d.Dot(d);

    if (lenSq < MIN_SEGMENT_LENGTH * MIN_SEGMENT_LENGTH) {
        // seg1 is degenerate - check if seg2 contains it
        return seg2.DistanceToPoint(seg1.p1) <= tolerance;
    }

    double t1 = (seg2.p1 - seg1.p1).Dot(d) / lenSq;
    double t2 = (seg2.p2 - seg1.p1).Dot(d) / lenSq;

    if (t1 > t2) std::swap(t1, t2);

    // Check overlap with [0, 1]
    return t2 >= -tolerance / std::sqrt(lenSq) && t1 <= 1.0 + tolerance / std::sqrt(lenSq);
}

std::optional<Segment2d> SegmentOverlap(const Segment2d& seg1, const Segment2d& seg2,
                                        double tolerance) {
    // Check collinearity
    Line2d line1 = seg1.ToLine();
    if (line1.Distance(seg2.p1) > tolerance || line1.Distance(seg2.p2) > tolerance) {
        return std::nullopt;
    }

    Point2d d = seg1.p2 - seg1.p1;
    double lenSq = d.Dot(d);

    if (lenSq < MIN_SEGMENT_LENGTH * MIN_SEGMENT_LENGTH) {
        // seg1 is degenerate
        if (seg2.DistanceToPoint(seg1.p1) <= tolerance) {
            return Segment2d(seg1.p1, seg1.p1);
        }
        return std::nullopt;
    }

    // Project all four endpoints onto the line
    double tol = tolerance / std::sqrt(lenSq);
    double t1_1 = 0.0;
    double t1_2 = 1.0;
    double t2_1 = (seg2.p1 - seg1.p1).Dot(d) / lenSq;
    double t2_2 = (seg2.p2 - seg1.p1).Dot(d) / lenSq;

    if (t2_1 > t2_2) std::swap(t2_1, t2_2);

    // Compute overlap
    double overlapStart = std::max(t1_1, t2_1);
    double overlapEnd = std::min(t1_2, t2_2);

    if (overlapEnd < overlapStart - tol) {
        return std::nullopt;
    }

    Point2d p1 = seg1.p1 + d * overlapStart;
    Point2d p2 = seg1.p1 + d * overlapEnd;

    return Segment2d(p1, p2);
}

std::optional<Segment2d> ClipSegmentToRect(const Segment2d& segment, const Rect2d& rect) {
    // Cohen-Sutherland algorithm
    const int INSIDE = 0;
    const int LEFT = 1;
    const int RIGHT = 2;
    const int BOTTOM = 4;
    const int TOP = 8;

    auto computeCode = [&](double x, double y) {
        int code = INSIDE;
        if (x < rect.x) code |= LEFT;
        else if (x > rect.x + rect.width) code |= RIGHT;
        if (y < rect.y) code |= BOTTOM;
        else if (y > rect.y + rect.height) code |= TOP;
        return code;
    };

    double x1 = segment.p1.x;
    double y1 = segment.p1.y;
    double x2 = segment.p2.x;
    double y2 = segment.p2.y;

    int code1 = computeCode(x1, y1);
    int code2 = computeCode(x2, y2);

    while (true) {
        if (!(code1 | code2)) {
            // Both inside
            return Segment2d(Point2d(x1, y1), Point2d(x2, y2));
        }

        if (code1 & code2) {
            // Both outside same region
            return std::nullopt;
        }

        // Pick point outside
        int codeOut = code1 ? code1 : code2;
        double x, y;

        if (codeOut & TOP) {
            x = x1 + (x2 - x1) * (rect.y + rect.height - y1) / (y2 - y1);
            y = rect.y + rect.height;
        } else if (codeOut & BOTTOM) {
            x = x1 + (x2 - x1) * (rect.y - y1) / (y2 - y1);
            y = rect.y;
        } else if (codeOut & RIGHT) {
            y = y1 + (y2 - y1) * (rect.x + rect.width - x1) / (x2 - x1);
            x = rect.x + rect.width;
        } else {  // LEFT
            y = y1 + (y2 - y1) * (rect.x - x1) / (x2 - x1);
            x = rect.x;
        }

        if (codeOut == code1) {
            x1 = x;
            y1 = y;
            code1 = computeCode(x1, y1);
        } else {
            x2 = x;
            y2 = y;
            code2 = computeCode(x2, y2);
        }
    }
}

std::optional<Segment2d> ClipSegmentToRotatedRect(const Segment2d& segment,
                                                   const RotatedRect2d& rect) {
    // Find intersections with rotated rect
    IntersectionResult2 intersections = IntersectSegmentRotatedRect(segment, rect);

    // Check if endpoints are inside
    bool p1Inside = rect.Contains(segment.p1);
    bool p2Inside = rect.Contains(segment.p2);

    if (p1Inside && p2Inside) {
        return segment;
    }

    if (!intersections.HasIntersection()) {
        return std::nullopt;
    }

    if (intersections.count == 1) {
        if (p1Inside) {
            return Segment2d(segment.p1, intersections.point1);
        } else if (p2Inside) {
            return Segment2d(intersections.point1, segment.p2);
        }
        // Edge case: tangent
        return std::nullopt;
    }

    // Two intersections
    return Segment2d(intersections.point1, intersections.point2);
}

// =============================================================================
// Arc-Arc Intersection
// =============================================================================

IntersectionResult2 IntersectArcArc(const Arc2d& arc1, const Arc2d& arc2) {
    // First intersect the full circles
    IntersectionResult2 circleResult = IntersectCircleCircle(arc1.ToCircle(), arc2.ToCircle());

    if (!circleResult.HasIntersection()) {
        return IntersectionResult2::None();
    }

    IntersectionResult2 result = IntersectionResult2::None();

    auto checkPoint = [&](const Point2d& p, double angle1, double angle2) {
        if (AngleWithinArc(angle1, arc1) && AngleWithinArc(angle2, arc2)) {
            if (result.count == 0) {
                result.count = 1;
                result.point1 = p;
                result.param1_1 = angle1;
                result.param1_2 = angle2;
            } else {
                result.count = 2;
                result.point2 = p;
                result.param2_1 = angle1;
                result.param2_2 = angle2;
            }
        }
    };

    checkPoint(circleResult.point1, circleResult.param1_1, circleResult.param1_2);
    if (circleResult.count == 2) {
        checkPoint(circleResult.point2, circleResult.param2_1, circleResult.param2_2);
    }

    return result;
}

// =============================================================================
// Ray Intersection for Point-in-Polygon
// =============================================================================

int CountRayContourIntersections(const Point2d& point,
                                  const std::vector<Point2d>& contourPoints) {
    int count = 0;
    size_t n = contourPoints.size();

    if (n < 3) {
        return 0;
    }

    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;

        const Point2d& p1 = contourPoints[i];
        const Point2d& p2 = contourPoints[j];

        // Check if horizontal ray from point intersects segment [p1, p2]
        // Ray goes from (point.x, point.y) to (+inf, point.y)

        // Check if segment straddles the ray's y level
        if ((p1.y <= point.y && p2.y > point.y) ||
            (p2.y <= point.y && p1.y > point.y)) {

            // Compute x-coordinate of intersection
            double t = (point.y - p1.y) / (p2.y - p1.y);
            double xIntersect = p1.x + t * (p2.x - p1.x);

            if (xIntersect > point.x) {
                ++count;
            }
        }
    }

    return count;
}

} // namespace Qi::Vision::Internal
