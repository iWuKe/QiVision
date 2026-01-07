/**
 * @file Distance.cpp
 * @brief Implementation of distance calculation functions
 */

#include <QiVision/Internal/Distance.h>

#include <algorithm>
#include <array>
#include <limits>

namespace Qi::Vision::Internal {

// =============================================================================
// Point-to-Line Distance
// =============================================================================

DistanceResult DistancePointToLine(const Point2d& point, const Line2d& line) {
    // Normalize line if needed
    double norm = std::sqrt(line.a * line.a + line.b * line.b);
    if (norm < GEOM_TOLERANCE) {
        // Degenerate line (a = b = 0)
        return DistanceResult::Invalid();
    }

    double a = line.a / norm;
    double b = line.b / norm;
    double c = line.c / norm;

    // Signed distance: d = a*x + b*y + c
    double signedDist = a * point.x + b * point.y + c;

    // Closest point: P' = P - d * n, where n = (a, b)
    Point2d closestPoint = {
        point.x - signedDist * a,
        point.y - signedDist * b
    };

    return {std::abs(signedDist), closestPoint, 0.0};
}

SignedDistanceResult SignedDistancePointToLine(const Point2d& point, const Line2d& line) {
    // Normalize line if needed
    double norm = std::sqrt(line.a * line.a + line.b * line.b);
    if (norm < GEOM_TOLERANCE) {
        // Degenerate line (a = b = 0)
        SignedDistanceResult result;
        result.signedDistance = 0.0;
        result.closestPoint = point;
        return result;
    }

    double a = line.a / norm;
    double b = line.b / norm;
    double c = line.c / norm;

    // Signed distance: d = a*x + b*y + c
    double signedDist = a * point.x + b * point.y + c;

    // Closest point: P' = P - d * n
    Point2d closestPoint = {
        point.x - signedDist * a,
        point.y - signedDist * b
    };

    return {signedDist, closestPoint, 0.0};
}

// =============================================================================
// Point-to-Segment Distance
// =============================================================================

DistanceResult DistancePointToSegment(const Point2d& point, const Segment2d& segment) {
    Point2d d = segment.p2 - segment.p1;
    double segLengthSq = d.x * d.x + d.y * d.y;

    // Handle degenerate segment (p1 == p2)
    if (segLengthSq < MIN_SEGMENT_LENGTH * MIN_SEGMENT_LENGTH) {
        double dist = point.DistanceTo(segment.p1);
        return {dist, segment.p1, 0.0};
    }

    // Compute projection parameter t
    Point2d v = point - segment.p1;
    double t = v.Dot(d) / segLengthSq;

    // Clamp t to [0, 1]
    t = std::clamp(t, 0.0, 1.0);

    // Compute closest point
    Point2d closestPoint = segment.PointAt(t);
    double distance = point.DistanceTo(closestPoint);

    return {distance, closestPoint, t};
}

SignedDistanceResult SignedDistancePointToSegment(const Point2d& point, const Segment2d& segment) {
    DistanceResult result = DistancePointToSegment(point, segment);

    // Compute sign based on cross product (left = positive)
    Point2d d = segment.p2 - segment.p1;
    Point2d v = point - segment.p1;
    double cross = d.x * v.y - d.y * v.x;

    double sign = (cross >= 0.0) ? 1.0 : -1.0;

    return {sign * result.distance, result.closestPoint, result.parameter};
}

// =============================================================================
// Point-to-Circle Distance
// =============================================================================

DistanceResult DistancePointToCircle(const Point2d& point, const Circle2d& circle) {
    // Handle zero radius circle (point)
    if (circle.radius < MIN_RADIUS) {
        double dist = point.DistanceTo(circle.center);
        return {dist, circle.center, 0.0};
    }

    double distToCenter = point.DistanceTo(circle.center);

    // Handle point at circle center
    if (distToCenter < GEOM_TOLERANCE) {
        // Return arbitrary point on boundary (positive X direction)
        Point2d closestPoint = {circle.center.x + circle.radius, circle.center.y};
        return {circle.radius, closestPoint, 0.0};
    }

    // Compute closest point on circle
    Point2d direction = (point - circle.center) * (1.0 / distToCenter);
    Point2d closestPoint = circle.center + direction * circle.radius;

    // Distance = |distance to center - radius|
    double distance = std::abs(distToCenter - circle.radius);

    // Parameter: angle
    double theta = std::atan2(direction.y, direction.x);

    return {distance, closestPoint, theta};
}

SignedDistanceResult SignedDistancePointToCircle(const Point2d& point, const Circle2d& circle) {
    DistanceResult result = DistancePointToCircle(point, circle);

    double distToCenter = point.DistanceTo(circle.center);

    // Sign: positive outside, negative inside
    double sign = (distToCenter >= circle.radius) ? 1.0 : -1.0;

    return {sign * result.distance, result.closestPoint, result.parameter};
}

// =============================================================================
// Point-to-Ellipse Distance
// =============================================================================

namespace {

// Transform point from world to ellipse local coordinates
Point2d ToEllipseLocal(const Point2d& point, const Ellipse2d& ellipse) {
    // Translate to ellipse center
    double dx = point.x - ellipse.center.x;
    double dy = point.y - ellipse.center.y;

    // Rotate by -angle
    double cosA = std::cos(-ellipse.angle);
    double sinA = std::sin(-ellipse.angle);

    return {
        dx * cosA - dy * sinA,
        dx * sinA + dy * cosA
    };
}

// Transform point from ellipse local to world coordinates
Point2d FromEllipseLocal(const Point2d& localPoint, const Ellipse2d& ellipse) {
    // Rotate by +angle
    double cosA = std::cos(ellipse.angle);
    double sinA = std::sin(ellipse.angle);

    double x = localPoint.x * cosA - localPoint.y * sinA;
    double y = localPoint.x * sinA + localPoint.y * cosA;

    // Translate by center
    return {x + ellipse.center.x, y + ellipse.center.y};
}

// Check if point is inside ellipse (in local coordinates)
bool IsInsideEllipseLocal(double x, double y, double a, double b) {
    if (a <= 0 || b <= 0) return false;
    double val = (x * x) / (a * a) + (y * y) / (b * b);
    return val < 1.0;
}

} // anonymous namespace

DistanceResult DistancePointToEllipse(const Point2d& point, const Ellipse2d& ellipse,
                                       int maxIterations, double tolerance) {
    // Handle degenerate ellipse
    if (ellipse.a < MIN_RADIUS || ellipse.b < MIN_RADIUS) {
        double dist = point.DistanceTo(ellipse.center);
        return {dist, ellipse.center, 0.0};
    }

    // Handle circular ellipse
    if (std::abs(ellipse.a - ellipse.b) < GEOM_TOLERANCE) {
        Circle2d circle(ellipse.center, ellipse.a);
        return DistancePointToCircle(point, circle);
    }

    // Transform point to ellipse local coordinates
    Point2d local = ToEllipseLocal(point, ellipse);

    // Use symmetry: work in first quadrant
    double px = std::abs(local.x);
    double py = std::abs(local.y);

    double a = ellipse.a;
    double b = ellipse.b;

    // Initial estimate for parameter t
    double t = std::atan2(py * a, px * b);

    // Newton iteration to find closest point
    // Minimize: f(t) = (a*cos(t) - px)^2 + (b*sin(t) - py)^2
    // df/dt = 0 => (a*cos(t) - px)*(-a*sin(t)) + (b*sin(t) - py)*(b*cos(t)) = 0
    // Rewrite as: (b^2 - a^2)*sin(t)*cos(t) + a*px*sin(t) - b*py*cos(t) = 0

    for (int iter = 0; iter < maxIterations; ++iter) {
        double cosT = std::cos(t);
        double sinT = std::sin(t);

        // Point on ellipse
        double ex = a * cosT;
        double ey = b * sinT;

        // Tangent vector
        double tx = -a * sinT;
        double ty = b * cosT;

        // Vector from ellipse point to query point
        double dx = px - ex;
        double dy = py - ey;

        // f(t) = tangent dot (point - ellipse) = tx*dx + ty*dy
        double f = tx * dx + ty * dy;

        // f'(t) = -tangent_length^2 + normal dot (point - ellipse)
        // normal = (-b*sin(t), a*cos(t)) normalized direction
        double nx = -a * cosT;  // Second derivative of x
        double ny = -b * sinT;  // Second derivative of y
        double df = tx * (-tx/a) * a + ty * ty / b * (-b) + nx * dx + ny * dy;
        // Simplify: df = -(tx^2 + ty^2) + nx*dx + ny*dy = -|tangent|^2 + normal.diff
        df = -(tx * tx + ty * ty) + nx * dx + ny * dy;

        if (std::abs(df) < tolerance) break;

        double dt = -f / df;
        t += dt;

        if (std::abs(dt) < tolerance) break;
    }

    // Clamp t to valid range
    t = std::fmod(t, 2.0 * PI);
    if (t < 0) t += 2.0 * PI;

    // Compute closest point in first quadrant
    double closestLocalX = a * std::cos(t);
    double closestLocalY = b * std::sin(t);

    // Restore signs
    if (local.x < 0) closestLocalX = -closestLocalX;
    if (local.y < 0) closestLocalY = -closestLocalY;

    // Transform back to world coordinates
    Point2d localClosest = {closestLocalX, closestLocalY};
    Point2d closestPoint = FromEllipseLocal(localClosest, ellipse);

    double distance = point.DistanceTo(closestPoint);

    // Compute parameter angle in world coordinates
    double paramAngle = std::atan2(closestLocalY, closestLocalX);

    return {distance, closestPoint, paramAngle};
}

SignedDistanceResult SignedDistancePointToEllipse(const Point2d& point, const Ellipse2d& ellipse,
                                                   int maxIterations, double tolerance) {
    DistanceResult result = DistancePointToEllipse(point, ellipse, maxIterations, tolerance);

    // Check if point is inside ellipse
    Point2d local = ToEllipseLocal(point, ellipse);
    bool inside = IsInsideEllipseLocal(local.x, local.y, ellipse.a, ellipse.b);

    double sign = inside ? -1.0 : 1.0;

    return {sign * result.distance, result.closestPoint, result.parameter};
}

// =============================================================================
// Point-to-Arc Distance
// =============================================================================

DistanceResult DistancePointToArc(const Point2d& point, const Arc2d& arc) {
    // Handle zero radius arc (point)
    if (arc.radius < MIN_RADIUS) {
        double dist = point.DistanceTo(arc.center);
        return {dist, arc.center, 0.0};
    }

    // Handle zero sweep arc (point)
    if (std::abs(arc.sweepAngle) < ANGLE_TOLERANCE) {
        Point2d startPt = arc.StartPoint();
        double dist = point.DistanceTo(startPt);
        return {dist, startPt, arc.startAngle};
    }

    // First, find the closest point on the full circle
    double distToCenter = point.DistanceTo(arc.center);

    Point2d direction;
    double theta;

    if (distToCenter < GEOM_TOLERANCE) {
        // Point is at center, use arc midpoint direction
        theta = arc.startAngle + arc.sweepAngle * 0.5;
        direction = {std::cos(theta), std::sin(theta)};
    } else {
        direction = (point - arc.center) * (1.0 / distToCenter);
        theta = std::atan2(direction.y, direction.x);
    }

    // Check if closest point angle is within arc range
    if (AngleInArcRange(theta, arc)) {
        // Closest point is on the arc
        Point2d closestPoint = arc.center + direction * arc.radius;
        double distance = std::abs(distToCenter - arc.radius);
        return {distance, closestPoint, theta};
    }

    // Closest point is not on arc; compare endpoints
    Point2d startPt = arc.StartPoint();
    Point2d endPt = arc.EndPoint();

    double distToStart = point.DistanceTo(startPt);
    double distToEnd = point.DistanceTo(endPt);

    if (distToStart <= distToEnd) {
        return {distToStart, startPt, -1.0};  // -1 indicates start endpoint
    } else {
        return {distToEnd, endPt, -2.0};       // -2 indicates end endpoint
    }
}

// =============================================================================
// Point-to-RotatedRect Distance
// =============================================================================

DistanceResult DistancePointToRotatedRect(const Point2d& point, const RotatedRect2d& rect) {
    // Get the four edges
    auto edges = RotatedRectEdges(rect);

    double minDist = std::numeric_limits<double>::max();
    Point2d minClosest;
    int minEdgeIdx = 0;

    for (int i = 0; i < 4; ++i) {
        DistanceResult result = DistancePointToSegment(point, edges[i]);
        if (result.distance < minDist) {
            minDist = result.distance;
            minClosest = result.closestPoint;
            minEdgeIdx = i;
        }
    }

    return {minDist, minClosest, static_cast<double>(minEdgeIdx)};
}

SignedDistanceResult SignedDistancePointToRotatedRect(const Point2d& point, const RotatedRect2d& rect) {
    DistanceResult result = DistancePointToRotatedRect(point, rect);

    // Check if point is inside rectangle
    bool inside = rect.Contains(point);
    double sign = inside ? -1.0 : 1.0;

    return {sign * result.distance, result.closestPoint, result.parameter};
}

// =============================================================================
// Line-to-Line Distance
// =============================================================================

std::optional<double> DistanceLineToLine(const Line2d& line1, const Line2d& line2) {
    // Normalize both lines
    double norm1 = std::sqrt(line1.a * line1.a + line1.b * line1.b);
    double norm2 = std::sqrt(line2.a * line2.a + line2.b * line2.b);

    if (norm1 < GEOM_TOLERANCE || norm2 < GEOM_TOLERANCE) {
        return std::nullopt;
    }

    double a1 = line1.a / norm1, b1 = line1.b / norm1, c1 = line1.c / norm1;
    double a2 = line2.a / norm2, b2 = line2.b / norm2, c2 = line2.c / norm2;

    // Check if parallel: cross product of normals = 0
    double cross = a1 * b2 - a2 * b1;

    if (std::abs(cross) > ANGLE_TOLERANCE) {
        // Lines intersect
        return std::nullopt;
    }

    // Lines are parallel
    // Distance = |c1 - c2| (if normals point same direction)
    // Need to ensure normals point same direction
    double dot = a1 * a2 + b1 * b2;
    if (dot < 0) {
        c2 = -c2;  // Flip sign if normals point opposite
    }

    return std::abs(c1 - c2);
}

std::optional<double> SignedDistanceLineToLine(const Line2d& line1, const Line2d& line2) {
    // Normalize both lines
    double norm1 = std::sqrt(line1.a * line1.a + line1.b * line1.b);
    double norm2 = std::sqrt(line2.a * line2.a + line2.b * line2.b);

    if (norm1 < GEOM_TOLERANCE || norm2 < GEOM_TOLERANCE) {
        return std::nullopt;
    }

    double a1 = line1.a / norm1, b1 = line1.b / norm1, c1 = line1.c / norm1;
    double a2 = line2.a / norm2, b2 = line2.b / norm2, c2 = line2.c / norm2;

    // Check if parallel
    double cross = a1 * b2 - a2 * b1;

    if (std::abs(cross) > ANGLE_TOLERANCE) {
        // Lines intersect
        return std::nullopt;
    }

    // Lines are parallel
    // Ensure normals point same direction
    double dot = a1 * a2 + b1 * b2;
    if (dot < 0) {
        c2 = -c2;
    }

    // Signed distance from line1 to line2 along line1's normal
    return c2 - c1;
}

// =============================================================================
// Segment-to-Segment Distance
// =============================================================================

SegmentDistanceResult DistanceSegmentToSegment(const Segment2d& seg1, const Segment2d& seg2) {
    // Parametric form:
    // P1(s) = seg1.p1 + s * d1, s in [0,1]
    // P2(t) = seg2.p1 + t * d2, t in [0,1]
    // Minimize ||P1(s) - P2(t)||^2

    Point2d d1 = seg1.p2 - seg1.p1;
    Point2d d2 = seg2.p2 - seg2.p1;
    Point2d r = seg1.p1 - seg2.p1;

    double a = d1.Dot(d1);  // |d1|^2
    double b = d1.Dot(d2);  // d1 . d2
    double c = d2.Dot(d2);  // |d2|^2
    double d = d1.Dot(r);   // d1 . r
    double e = d2.Dot(r);   // d2 . r

    double denom = a * c - b * b;

    double sN, sD, tN, tD;
    sD = tD = denom;

    // Handle parallel/collinear segments
    if (std::abs(denom) < GEOM_TOLERANCE * GEOM_TOLERANCE) {
        // Segments are parallel - compute all four endpoint pairs and take minimum
        double minDist = std::numeric_limits<double>::max();
        Point2d minP1, minP2;
        double minS = 0.0, minT = 0.0;

        // Check all four endpoint combinations
        std::array<std::pair<double, double>, 4> endpoints = {{
            {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
        }};

        for (const auto& [sp, tp] : endpoints) {
            Point2d p1 = seg1.PointAt(sp);
            Point2d p2 = seg2.PointAt(tp);
            double dist = p1.DistanceTo(p2);
            if (dist < minDist) {
                minDist = dist;
                minP1 = p1;
                minP2 = p2;
                minS = sp;
                minT = tp;
            }
        }

        // Also check projections of endpoints onto opposite segment
        // Project seg1.p1 onto seg2
        if (c > GEOM_TOLERANCE) {
            double t_proj = -e / c;
            if (t_proj >= 0.0 && t_proj <= 1.0) {
                Point2d p2 = seg2.PointAt(t_proj);
                double dist = seg1.p1.DistanceTo(p2);
                if (dist < minDist) {
                    minDist = dist;
                    minP1 = seg1.p1;
                    minP2 = p2;
                    minS = 0.0;
                    minT = t_proj;
                }
            }
        }

        // Project seg1.p2 onto seg2
        if (c > GEOM_TOLERANCE) {
            double t_proj = (b - e) / c;
            if (t_proj >= 0.0 && t_proj <= 1.0) {
                Point2d p2 = seg2.PointAt(t_proj);
                double dist = seg1.p2.DistanceTo(p2);
                if (dist < minDist) {
                    minDist = dist;
                    minP1 = seg1.p2;
                    minP2 = p2;
                    minS = 1.0;
                    minT = t_proj;
                }
            }
        }

        // Project seg2.p1 onto seg1
        if (a > GEOM_TOLERANCE) {
            double s_proj = -d / a;
            if (s_proj >= 0.0 && s_proj <= 1.0) {
                Point2d p1 = seg1.PointAt(s_proj);
                double dist = p1.DistanceTo(seg2.p1);
                if (dist < minDist) {
                    minDist = dist;
                    minP1 = p1;
                    minP2 = seg2.p1;
                    minS = s_proj;
                    minT = 0.0;
                }
            }
        }

        // Project seg2.p2 onto seg1
        if (a > GEOM_TOLERANCE) {
            double s_proj = (b - d) / a;
            if (s_proj >= 0.0 && s_proj <= 1.0) {
                Point2d p1 = seg1.PointAt(s_proj);
                double dist = p1.DistanceTo(seg2.p2);
                if (dist < minDist) {
                    minDist = dist;
                    minP1 = p1;
                    minP2 = seg2.p2;
                    minS = s_proj;
                    minT = 1.0;
                }
            }
        }

        return {minDist, minP1, minP2, minS, minT};
    }

    // General case: compute s and t
    sN = (b * e - c * d);
    tN = (a * e - b * d);

    // Check s parameter
    if (sN < 0.0) {
        sN = 0.0;
        tN = e;
        tD = c;
    } else if (sN > sD) {
        sN = sD;
        tN = e + b;
        tD = c;
    }

    // Check t parameter
    if (tN < 0.0) {
        tN = 0.0;
        // Recompute s for t = 0
        if (-d < 0.0) {
            sN = 0.0;
        } else if (-d > a) {
            sN = sD;
        } else {
            sN = -d;
            sD = a;
        }
    } else if (tN > tD) {
        tN = tD;
        // Recompute s for t = 1
        if ((-d + b) < 0.0) {
            sN = 0.0;
        } else if ((-d + b) > a) {
            sN = sD;
        } else {
            sN = -d + b;
            sD = a;
        }
    }

    // Compute final parameters
    double s = (std::abs(sN) < GEOM_TOLERANCE) ? 0.0 : sN / sD;
    double t = (std::abs(tN) < GEOM_TOLERANCE) ? 0.0 : tN / tD;

    Point2d closest1 = seg1.PointAt(s);
    Point2d closest2 = seg2.PointAt(t);
    double distance = closest1.DistanceTo(closest2);

    return {distance, closest1, closest2, s, t};
}

// =============================================================================
// Circle-to-Circle Distance
// =============================================================================

CircleDistanceResult DistanceCircleToCircle(const Circle2d& circle1, const Circle2d& circle2) {
    double centerDist = circle1.center.DistanceTo(circle2.center);

    // Handle coincident centers
    if (centerDist < GEOM_TOLERANCE) {
        // Circles are concentric
        double radiusDiff = circle1.radius - circle2.radius;
        Point2d point1 = {circle1.center.x + circle1.radius, circle1.center.y};
        Point2d point2 = {circle2.center.x + circle2.radius, circle2.center.y};
        return {-std::abs(radiusDiff), point1, point2};
    }

    // Direction from center1 to center2
    Point2d direction = (circle2.center - circle1.center) * (1.0 / centerDist);

    // Closest points on each circle (along the line connecting centers)
    Point2d point1 = circle1.center + direction * circle1.radius;
    Point2d point2 = circle2.center - direction * circle2.radius;

    // Distance between boundaries
    double distance = centerDist - circle1.radius - circle2.radius;

    return {distance, point1, point2};
}

// =============================================================================
// Point-to-Contour Distance
// =============================================================================

ContourDistanceResult DistancePointToContour(const Point2d& point,
                                              const std::vector<Point2d>& contourPoints,
                                              bool closed) {
    if (contourPoints.size() < 2) {
        if (contourPoints.size() == 1) {
            double dist = point.DistanceTo(contourPoints[0]);
            ContourDistanceResult result;
            result.distance = dist;
            result.closestPoint = contourPoints[0];
            result.segmentIndex = 0;
            result.segmentParameter = 0.0;
            return result;
        }
        return ContourDistanceResult::Invalid();
    }

    double minDist = std::numeric_limits<double>::max();
    Point2d minClosest;
    size_t minSegIdx = 0;
    double minParam = 0.0;

    size_t numSegs = closed ? contourPoints.size() : contourPoints.size() - 1;

    for (size_t i = 0; i < numSegs; ++i) {
        size_t j = (i + 1) % contourPoints.size();
        Segment2d seg(contourPoints[i], contourPoints[j]);

        DistanceResult result = DistancePointToSegment(point, seg);

        if (result.distance < minDist) {
            minDist = result.distance;
            minClosest = result.closestPoint;
            minSegIdx = i;
            minParam = result.parameter;
        }
    }

    return {minDist, minClosest, minSegIdx, minParam};
}

bool PointInsidePolygon(const Point2d& point, const std::vector<Point2d>& polygon) {
    if (polygon.size() < 3) return false;

    int windingNumber = 0;
    size_t n = polygon.size();

    for (size_t i = 0; i < n; ++i) {
        const Point2d& p1 = polygon[i];
        const Point2d& p2 = polygon[(i + 1) % n];

        if (p1.y <= point.y) {
            if (p2.y > point.y) {
                // Upward crossing
                double cross = (p2.x - p1.x) * (point.y - p1.y) - (point.x - p1.x) * (p2.y - p1.y);
                if (cross > 0) {
                    ++windingNumber;
                }
            }
        } else {
            if (p2.y <= point.y) {
                // Downward crossing
                double cross = (p2.x - p1.x) * (point.y - p1.y) - (point.x - p1.x) * (p2.y - p1.y);
                if (cross < 0) {
                    --windingNumber;
                }
            }
        }
    }

    return windingNumber != 0;
}

SignedDistanceResult SignedDistancePointToContour(const Point2d& point,
                                                   const std::vector<Point2d>& contourPoints) {
    if (contourPoints.size() < 3) {
        SignedDistanceResult result;
        result.signedDistance = 0.0;
        return result;
    }

    // Compute unsigned distance
    ContourDistanceResult distResult = DistancePointToContour(point, contourPoints, true);

    // Determine inside/outside using ray casting
    bool inside = PointInsidePolygon(point, contourPoints);

    double sign = inside ? -1.0 : 1.0;

    return {sign * distResult.distance, distResult.closestPoint, distResult.segmentParameter};
}

// =============================================================================
// Batch Distance Functions
// =============================================================================

std::vector<double> DistancePointsToLine(const std::vector<Point2d>& points, const Line2d& line) {
    std::vector<double> distances;
    distances.reserve(points.size());

    // Normalize line once
    double norm = std::sqrt(line.a * line.a + line.b * line.b);
    if (norm < GEOM_TOLERANCE) {
        distances.resize(points.size(), 0.0);
        return distances;
    }

    double a = line.a / norm;
    double b = line.b / norm;
    double c = line.c / norm;

    for (const auto& point : points) {
        double dist = std::abs(a * point.x + b * point.y + c);
        distances.push_back(dist);
    }

    return distances;
}

std::vector<double> SignedDistancePointsToLine(const std::vector<Point2d>& points, const Line2d& line) {
    std::vector<double> distances;
    distances.reserve(points.size());

    // Normalize line once
    double norm = std::sqrt(line.a * line.a + line.b * line.b);
    if (norm < GEOM_TOLERANCE) {
        distances.resize(points.size(), 0.0);
        return distances;
    }

    double a = line.a / norm;
    double b = line.b / norm;
    double c = line.c / norm;

    for (const auto& point : points) {
        double dist = a * point.x + b * point.y + c;
        distances.push_back(dist);
    }

    return distances;
}

std::vector<double> DistancePointsToCircle(const std::vector<Point2d>& points, const Circle2d& circle) {
    std::vector<double> distances;
    distances.reserve(points.size());

    for (const auto& point : points) {
        double distToCenter = point.DistanceTo(circle.center);
        double dist = std::abs(distToCenter - circle.radius);
        distances.push_back(dist);
    }

    return distances;
}

std::vector<double> SignedDistancePointsToCircle(const std::vector<Point2d>& points, const Circle2d& circle) {
    std::vector<double> distances;
    distances.reserve(points.size());

    for (const auto& point : points) {
        double distToCenter = point.DistanceTo(circle.center);
        double dist = distToCenter - circle.radius;  // Positive outside, negative inside
        distances.push_back(dist);
    }

    return distances;
}

std::vector<double> DistancePointsToEllipse(const std::vector<Point2d>& points, const Ellipse2d& ellipse) {
    std::vector<double> distances;
    distances.reserve(points.size());

    for (const auto& point : points) {
        DistanceResult result = DistancePointToEllipse(point, ellipse);
        distances.push_back(result.distance);
    }

    return distances;
}

std::vector<double> DistancePointsToContour(const std::vector<Point2d>& points,
                                             const std::vector<Point2d>& contourPoints,
                                             bool closed) {
    std::vector<double> distances;
    distances.reserve(points.size());

    for (const auto& point : points) {
        ContourDistanceResult result = DistancePointToContour(point, contourPoints, closed);
        distances.push_back(result.IsValid() ? result.distance : 0.0);
    }

    return distances;
}

// =============================================================================
// Utility Functions
// =============================================================================

std::optional<Point2d> NearestPointOnContour(const Point2d& point,
                                              const std::vector<Point2d>& contourPoints,
                                              bool closed) {
    if (contourPoints.empty()) {
        return std::nullopt;
    }

    if (contourPoints.size() == 1) {
        return contourPoints[0];
    }

    ContourDistanceResult result = DistancePointToContour(point, contourPoints, closed);
    if (result.IsValid()) {
        return result.closestPoint;
    }

    return std::nullopt;
}

double HausdorffDistance(const std::vector<Point2d>& contour1,
                         const std::vector<Point2d>& contour2,
                         bool closed1, bool closed2) {
    if (contour1.empty() || contour2.empty()) {
        return 0.0;
    }

    // Compute max distance from contour1 to contour2
    double maxDist1to2 = 0.0;
    for (const auto& pt : contour1) {
        ContourDistanceResult result = DistancePointToContour(pt, contour2, closed2);
        if (result.IsValid() && result.distance > maxDist1to2) {
            maxDist1to2 = result.distance;
        }
    }

    // Compute max distance from contour2 to contour1
    double maxDist2to1 = 0.0;
    for (const auto& pt : contour2) {
        ContourDistanceResult result = DistancePointToContour(pt, contour1, closed1);
        if (result.IsValid() && result.distance > maxDist2to1) {
            maxDist2to1 = result.distance;
        }
    }

    // Hausdorff distance is max of the two
    return std::max(maxDist1to2, maxDist2to1);
}

double AverageDistanceContourToContour(const std::vector<Point2d>& contour1,
                                        const std::vector<Point2d>& contour2,
                                        bool closed2) {
    if (contour1.empty() || contour2.size() < 2) {
        return 0.0;
    }

    double sumDist = 0.0;
    size_t count = 0;

    for (const auto& pt : contour1) {
        ContourDistanceResult result = DistancePointToContour(pt, contour2, closed2);
        if (result.IsValid()) {
            sumDist += result.distance;
            ++count;
        }
    }

    return (count > 0) ? (sumDist / static_cast<double>(count)) : 0.0;
}

} // namespace Qi::Vision::Internal
