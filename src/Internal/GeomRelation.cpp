/**
 * @file GeomRelation.cpp
 * @brief Implementation of geometric relationship functions
 */

#include <QiVision/Internal/GeomRelation.h>
#include <QiVision/Internal/Distance.h>
#include <QiVision/Internal/Intersection.h>
#include <QiVision/Core/QContour.h>

#include <algorithm>
#include <cmath>

namespace Qi::Vision::Internal {

// =============================================================================
// Point-Region Containment
// =============================================================================

bool PointInsideEllipse(const Point2d& point, const Ellipse2d& ellipse, double tolerance) {
    // Transform point to ellipse-centered coordinates
    double dx = point.x - ellipse.center.x;
    double dy = point.y - ellipse.center.y;

    // Rotate point to align with ellipse axes
    double cosTheta = std::cos(-ellipse.angle);
    double sinTheta = std::sin(-ellipse.angle);
    double x = dx * cosTheta - dy * sinTheta;
    double y = dx * sinTheta + dy * cosTheta;

    // Check if inside ellipse: (x/a)^2 + (y/b)^2 < 1
    double a = ellipse.a;
    double b = ellipse.b;
    double value = (x * x) / (a * a) + (y * y) / (b * b);

    return value < 1.0 - tolerance / std::min(a, b);
}

bool PointInsideRotatedRect(const Point2d& point, const RotatedRect2d& rect, double tolerance) {
    // Transform point to rect-local coordinates
    double dx = point.x - rect.center.x;
    double dy = point.y - rect.center.y;

    double cosTheta = std::cos(-rect.angle);
    double sinTheta = std::sin(-rect.angle);
    double localX = dx * cosTheta - dy * sinTheta;
    double localY = dx * sinTheta + dy * cosTheta;

    double halfW = rect.width / 2.0;
    double halfH = rect.height / 2.0;

    return std::abs(localX) <= halfW + tolerance && std::abs(localY) <= halfH + tolerance;
}

bool PointInsideConvexPolygon(const Point2d& point, const std::vector<Point2d>& polygon) {
    if (polygon.size() < 3) return false;

    // For convex polygon, point is inside if it's on the same side of all edges
    int n = static_cast<int>(polygon.size());
    bool sign = false;
    bool first = true;

    for (int i = 0; i < n; ++i) {
        const Point2d& p1 = polygon[i];
        const Point2d& p2 = polygon[(i + 1) % n];

        // Cross product of edge vector and point-to-p1 vector
        double cross = (p2.x - p1.x) * (point.y - p1.y) - (p2.y - p1.y) * (point.x - p1.x);

        if (first) {
            sign = cross > 0;
            first = false;
        } else {
            if ((cross > 0) != sign) return false;
        }
    }

    return true;
}

PointPolygonRelation PointInPolygon(const Point2d& point, const std::vector<Point2d>& polygon,
                                     double tolerance) {
    if (polygon.size() < 3) return PointPolygonRelation::Outside;

    // First check if on boundary
    if (PointOnPolygonBoundary(point, polygon, tolerance)) {
        return PointPolygonRelation::OnBoundary;
    }

    // Ray casting algorithm
    int n = static_cast<int>(polygon.size());
    int crossings = 0;

    for (int i = 0; i < n; ++i) {
        const Point2d& p1 = polygon[i];
        const Point2d& p2 = polygon[(i + 1) % n];

        // Check if ray from point going right crosses this edge
        if ((p1.y <= point.y && p2.y > point.y) || (p2.y <= point.y && p1.y > point.y)) {
            // Compute x-coordinate of intersection
            double t = (point.y - p1.y) / (p2.y - p1.y);
            double xIntersect = p1.x + t * (p2.x - p1.x);

            if (point.x < xIntersect) {
                crossings++;
            }
        }
    }

    return (crossings % 2 == 1) ? PointPolygonRelation::Inside : PointPolygonRelation::Outside;
}

bool PointOnPolygonBoundary(const Point2d& point, const std::vector<Point2d>& polygon,
                            double tolerance) {
    int n = static_cast<int>(polygon.size());
    for (int i = 0; i < n; ++i) {
        Segment2d edge(polygon[i], polygon[(i + 1) % n]);
        if (PointOnSegment(point, edge, tolerance)) {
            return true;
        }
    }
    return false;
}

// =============================================================================
// Circle-Circle Relationships
// =============================================================================

CircleCircleRelation GetCircleCircleRelation(const Circle2d& c1, const Circle2d& c2, double tolerance) {
    double d = c1.center.DistanceTo(c2.center);
    double r1 = c1.radius;
    double r2 = c2.radius;

    // Check for coincident circles
    if (d < tolerance && std::abs(r1 - r2) < tolerance) {
        return CircleCircleRelation::Coincident;
    }

    double sumR = r1 + r2;
    double diffR = std::abs(r1 - r2);

    // Separate
    if (d > sumR + tolerance) {
        return CircleCircleRelation::Separate;
    }

    // External tangent
    if (std::abs(d - sumR) < tolerance) {
        return CircleCircleRelation::ExternalTangent;
    }

    // One contains the other
    if (d + tolerance < diffR) {
        return CircleCircleRelation::Containing;
    }

    // Internal tangent
    if (std::abs(d - diffR) < tolerance) {
        return CircleCircleRelation::InternalTangent;
    }

    // Intersecting
    return CircleCircleRelation::Intersecting;
}

bool CircleContainsCircle(const Circle2d& outer, const Circle2d& inner, double tolerance) {
    double d = outer.center.DistanceTo(inner.center);
    return d + inner.radius <= outer.radius + tolerance;
}

// =============================================================================
// Line-Circle Relationships
// =============================================================================

LineCircleRelation GetLineCircleRelation(const Line2d& line, const Circle2d& circle,
                                          double tolerance) {
    // Distance from center to line
    double norm = std::sqrt(line.a * line.a + line.b * line.b);
    if (norm < 1e-15) return LineCircleRelation::Disjoint;

    double dist = std::abs(line.a * circle.center.x + line.b * circle.center.y + line.c) / norm;

    if (dist > circle.radius + tolerance) {
        return LineCircleRelation::Disjoint;
    }
    if (std::abs(dist - circle.radius) < tolerance) {
        return LineCircleRelation::Tangent;
    }
    return LineCircleRelation::Secant;
}

bool LineIsTangentToCircle(const Line2d& line, const Circle2d& circle, double tolerance) {
    return GetLineCircleRelation(line, circle, tolerance) == LineCircleRelation::Tangent;
}

bool SegmentIsTangentToCircle(const Segment2d& segment, const Circle2d& circle, double tolerance) {
    Line2d line = Line2d::FromPoints(segment.p1, segment.p2);

    // First check if line is tangent
    if (GetLineCircleRelation(line, circle, tolerance) != LineCircleRelation::Tangent) {
        return false;
    }

    // Find the tangent point and check if it's on the segment
    double norm = std::sqrt(line.a * line.a + line.b * line.b);
    Point2d foot;
    foot.x = circle.center.x - line.a * (line.a * circle.center.x + line.b * circle.center.y + line.c) / (norm * norm);
    foot.y = circle.center.y - line.b * (line.a * circle.center.x + line.b * circle.center.y + line.c) / (norm * norm);

    return PointOnSegment(foot, segment, tolerance);
}

// =============================================================================
// Segment-Segment Relationships
// =============================================================================

SegmentRelation GetSegmentRelation(const Segment2d& s1, const Segment2d& s2, double tolerance) {
    // Direction vectors
    double dx1 = s1.p2.x - s1.p1.x;
    double dy1 = s1.p2.y - s1.p1.y;
    double dx2 = s2.p2.x - s2.p1.x;
    double dy2 = s2.p2.y - s2.p1.y;

    // Cross product of directions
    double cross = dx1 * dy2 - dy1 * dx2;
    double len1 = std::sqrt(dx1 * dx1 + dy1 * dy1);
    double len2 = std::sqrt(dx2 * dx2 + dy2 * dy2);

    // Check parallel
    if (std::abs(cross) < tolerance * std::max(len1, len2)) {
        // Segments are parallel - check if collinear
        double dx = s2.p1.x - s1.p1.x;
        double dy = s2.p1.y - s1.p1.y;
        double crossToPoint = dx1 * dy - dy1 * dx;

        if (std::abs(crossToPoint) < tolerance * len1) {
            // Collinear - check for overlap
            // Project onto s1's direction
            double t1 = 0, t2 = 1;
            double t3, t4;

            if (std::abs(dx1) > std::abs(dy1)) {
                t3 = (s2.p1.x - s1.p1.x) / dx1;
                t4 = (s2.p2.x - s1.p1.x) / dx1;
            } else {
                t3 = (s2.p1.y - s1.p1.y) / dy1;
                t4 = (s2.p2.y - s1.p1.y) / dy1;
            }

            if (t3 > t4) std::swap(t3, t4);

            double overlapStart = std::max(t1, t3);
            double overlapEnd = std::min(t2, t4);

            if (overlapEnd > overlapStart + tolerance / len1) {
                return SegmentRelation::Overlapping;
            }
            return SegmentRelation::Collinear;
        }
        return SegmentRelation::Parallel;
    }

    // Not parallel - check if they actually intersect
    // Solve: s1.p1 + t * (s1.p2 - s1.p1) = s2.p1 + u * (s2.p2 - s2.p1)
    double dx = s2.p1.x - s1.p1.x;
    double dy = s2.p1.y - s1.p1.y;

    double t = (dx * dy2 - dy * dx2) / cross;
    double u = (dx * dy1 - dy * dx1) / cross;

    if (t >= -tolerance && t <= 1 + tolerance && u >= -tolerance && u <= 1 + tolerance) {
        return SegmentRelation::Intersecting;
    }

    return SegmentRelation::Disjoint;
}

// Note: SegmentsOverlap is implemented in Intersection.cpp

bool SegmentsConnected(const Segment2d& s1, const Segment2d& s2, double tolerance) {
    return s1.p1.DistanceTo(s2.p1) < tolerance ||
           s1.p1.DistanceTo(s2.p2) < tolerance ||
           s1.p2.DistanceTo(s2.p1) < tolerance ||
           s1.p2.DistanceTo(s2.p2) < tolerance;
}

bool SegmentsEqual(const Segment2d& s1, const Segment2d& s2, double tolerance) {
    return (s1.p1.DistanceTo(s2.p1) < tolerance && s1.p2.DistanceTo(s2.p2) < tolerance) ||
           (s1.p1.DistanceTo(s2.p2) < tolerance && s1.p2.DistanceTo(s2.p1) < tolerance);
}

// =============================================================================
// Rectangle Relationships
// =============================================================================

bool RectsOverlap(const Rect2d& r1, const Rect2d& r2) {
    return !(r1.x + r1.width < r2.x || r2.x + r2.width < r1.x ||
             r1.y + r1.height < r2.y || r2.y + r2.height < r1.y);
}

bool RectContainsRect(const Rect2d& outer, const Rect2d& inner) {
    return inner.x >= outer.x && inner.x + inner.width <= outer.x + outer.width &&
           inner.y >= outer.y && inner.y + inner.height <= outer.y + outer.height;
}

bool RotatedRectsOverlap(const RotatedRect2d& r1, const RotatedRect2d& r2) {
    // Use Separating Axis Theorem (SAT)
    auto corners1 = RotatedRectCorners(r1);
    auto corners2 = RotatedRectCorners(r2);

    // Axes to test: normals of each edge (4 axes total, but 2 pairs are parallel)
    std::vector<Point2d> axes;

    // Axes from r1
    axes.push_back({std::cos(r1.angle), std::sin(r1.angle)});
    axes.push_back({-std::sin(r1.angle), std::cos(r1.angle)});

    // Axes from r2
    axes.push_back({std::cos(r2.angle), std::sin(r2.angle)});
    axes.push_back({-std::sin(r2.angle), std::cos(r2.angle)});

    for (const auto& axis : axes) {
        // Project both rects onto axis
        double min1 = std::numeric_limits<double>::max();
        double max1 = std::numeric_limits<double>::lowest();
        double min2 = std::numeric_limits<double>::max();
        double max2 = std::numeric_limits<double>::lowest();

        for (const auto& c : corners1) {
            double proj = c.x * axis.x + c.y * axis.y;
            min1 = std::min(min1, proj);
            max1 = std::max(max1, proj);
        }

        for (const auto& c : corners2) {
            double proj = c.x * axis.x + c.y * axis.y;
            min2 = std::min(min2, proj);
            max2 = std::max(max2, proj);
        }

        // Check for separation
        if (max1 < min2 || max2 < min1) {
            return false;  // Separating axis found
        }
    }

    return true;  // No separating axis found, rectangles overlap
}

// =============================================================================
// Polygon Relationships
// =============================================================================

bool IsPolygonConvex(const std::vector<Point2d>& polygon) {
    if (polygon.size() < 3) return false;

    int n = static_cast<int>(polygon.size());
    bool sign = false;
    bool first = true;

    for (int i = 0; i < n; ++i) {
        const Point2d& p0 = polygon[i];
        const Point2d& p1 = polygon[(i + 1) % n];
        const Point2d& p2 = polygon[(i + 2) % n];

        double cross = (p1.x - p0.x) * (p2.y - p1.y) - (p1.y - p0.y) * (p2.x - p1.x);

        if (std::abs(cross) > RELATION_TOLERANCE) {
            if (first) {
                sign = cross > 0;
                first = false;
            } else if ((cross > 0) != sign) {
                return false;
            }
        }
    }

    return true;
}

bool IsPolygonCCW(const std::vector<Point2d>& polygon) {
    if (polygon.size() < 3) return false;

    // Calculate signed area
    double area = 0;
    int n = static_cast<int>(polygon.size());
    for (int i = 0; i < n; ++i) {
        const Point2d& p1 = polygon[i];
        const Point2d& p2 = polygon[(i + 1) % n];
        area += (p2.x - p1.x) * (p2.y + p1.y);
    }

    return area < 0;  // CCW if signed area is negative (standard convention)
}

bool PolygonsOverlap(const std::vector<Point2d>& poly1, const std::vector<Point2d>& poly2) {
    // Check if any vertex of poly1 is inside poly2
    for (const auto& p : poly1) {
        if (PointInPolygon(p, poly2) == PointPolygonRelation::Inside) {
            return true;
        }
    }

    // Check if any vertex of poly2 is inside poly1
    for (const auto& p : poly2) {
        if (PointInPolygon(p, poly1) == PointPolygonRelation::Inside) {
            return true;
        }
    }

    // Check if any edges intersect
    int n1 = static_cast<int>(poly1.size());
    int n2 = static_cast<int>(poly2.size());

    for (int i = 0; i < n1; ++i) {
        Segment2d edge1(poly1[i], poly1[(i + 1) % n1]);
        for (int j = 0; j < n2; ++j) {
            Segment2d edge2(poly2[j], poly2[(j + 1) % n2]);
            if (GetSegmentRelation(edge1, edge2) == SegmentRelation::Intersecting) {
                return true;
            }
        }
    }

    return false;
}

bool PolygonContainsPolygon(const std::vector<Point2d>& outer, const std::vector<Point2d>& inner) {
    // All vertices of inner must be inside outer
    for (const auto& p : inner) {
        PointPolygonRelation rel = PointInPolygon(p, outer);
        if (rel == PointPolygonRelation::Outside) {
            return false;
        }
    }

    // No edges should intersect (except possibly at shared boundaries)
    int nOuter = static_cast<int>(outer.size());
    int nInner = static_cast<int>(inner.size());

    for (int i = 0; i < nInner; ++i) {
        Segment2d innerEdge(inner[i], inner[(i + 1) % nInner]);
        for (int j = 0; j < nOuter; ++j) {
            Segment2d outerEdge(outer[j], outer[(j + 1) % nOuter]);
            SegmentRelation rel = GetSegmentRelation(innerEdge, outerEdge);
            if (rel == SegmentRelation::Intersecting) {
                return false;
            }
        }
    }

    return true;
}

// =============================================================================
// Contour Relationships
// =============================================================================

bool PointInsideContour(const Point2d& point, const Qi::Vision::QContour& contour) {
    if (!contour.IsClosed() || contour.Size() < 3) {
        return false;
    }

    // Convert contour points to polygon
    std::vector<Point2d> polygon;
    polygon.reserve(contour.Size());
    for (size_t i = 0; i < contour.Size(); ++i) {
        polygon.push_back(contour.GetPoint(i));
    }

    return PointInPolygon(point, polygon) == PointPolygonRelation::Inside;
}

bool IsContourClosed(const Qi::Vision::QContour& contour, double tolerance) {
    if (contour.Size() < 2) return false;
    return contour.GetPoint(0).DistanceTo(contour.GetPoint(contour.Size() - 1)) < tolerance;
}

// =============================================================================
// Angle Relationships
// =============================================================================

bool AnglesEqual(double angle1, double angle2, double tolerance) {
    double diff = NormalizeAngle(angle1 - angle2);
    return std::abs(diff) < tolerance || std::abs(diff - 2 * M_PI) < tolerance;
}

bool AngleInRange(double angle, double startAngle, double endAngle) {
    angle = NormalizeAngle0To2PI(angle);
    startAngle = NormalizeAngle0To2PI(startAngle);
    endAngle = NormalizeAngle0To2PI(endAngle);

    if (startAngle <= endAngle) {
        return angle >= startAngle && angle <= endAngle;
    } else {
        // Range crosses 0
        return angle >= startAngle || angle <= endAngle;
    }
}

double AngleDifference(double angle1, double angle2) {
    double diff = NormalizeAngle(angle1 - angle2);
    if (diff > M_PI) diff -= 2 * M_PI;
    if (diff < -M_PI) diff += 2 * M_PI;
    return diff;
}

// =============================================================================
// Coplanarity and Alignment
// =============================================================================

bool PointsAreCollinear(const std::vector<Point2d>& points, double tolerance) {
    if (points.size() < 3) return true;

    // Use first two points to define a line direction
    double dx = points[1].x - points[0].x;
    double dy = points[1].y - points[0].y;
    double len = std::sqrt(dx * dx + dy * dy);

    if (len < tolerance) {
        // First two points are same - check all against first
        for (size_t i = 2; i < points.size(); ++i) {
            if (points[i].DistanceTo(points[0]) > tolerance) {
                // Found a distinct point - use it to define direction
                dx = points[i].x - points[0].x;
                dy = points[i].y - points[0].y;
                len = std::sqrt(dx * dx + dy * dy);
                break;
            }
        }
    }

    if (len < tolerance) return true;  // All points are essentially the same

    // Check all other points
    for (size_t i = 2; i < points.size(); ++i) {
        double pdx = points[i].x - points[0].x;
        double pdy = points[i].y - points[0].y;
        double cross = std::abs(dx * pdy - dy * pdx);
        if (cross > tolerance * len) {
            return false;
        }
    }

    return true;
}

bool PointsAreConcyclic(const std::vector<Point2d>& points, double tolerance) {
    if (points.size() < 4) return true;  // Any 3 points are concyclic

    // Fit a circle to first 3 points
    const Point2d& p1 = points[0];
    const Point2d& p2 = points[1];
    const Point2d& p3 = points[2];

    // Check if collinear
    double area = (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y);
    if (std::abs(area) < tolerance) {
        return false;  // Collinear points, not concyclic
    }

    // Calculate circumcircle
    double d = 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y));
    double cx = ((p1.x * p1.x + p1.y * p1.y) * (p2.y - p3.y) +
                 (p2.x * p2.x + p2.y * p2.y) * (p3.y - p1.y) +
                 (p3.x * p3.x + p3.y * p3.y) * (p1.y - p2.y)) / d;
    double cy = ((p1.x * p1.x + p1.y * p1.y) * (p3.x - p2.x) +
                 (p2.x * p2.x + p2.y * p2.y) * (p1.x - p3.x) +
                 (p3.x * p3.x + p3.y * p3.y) * (p2.x - p1.x)) / d;
    Point2d center(cx, cy);
    double radius = center.DistanceTo(p1);

    // Check all other points
    for (size_t i = 3; i < points.size(); ++i) {
        double dist = points[i].DistanceTo(center);
        if (std::abs(dist - radius) > tolerance) {
            return false;
        }
    }

    return true;
}

} // namespace Qi::Vision::Internal
