/**
 * @file Geometry2d.cpp
 * @brief Implementation of 2D geometric primitive operations
 */

#include <QiVision/Internal/Geometry2d.h>
#include <QiVision/Internal/Fitting.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace Qi::Vision::Internal {

// =============================================================================
// Normalization Functions
// =============================================================================

Line2d NormalizeLine(const Line2d& line) {
    double norm = std::sqrt(line.a * line.a + line.b * line.b);
    if (norm < GEOM_TOLERANCE) {
        return line;  // Degenerate case
    }
    return Line2d(line.a / norm, line.b / norm, line.c / norm);
}

double NormalizeAngle(double angle) {
    // Fast path: most cases are in reasonable range
    if (angle >= -PI && angle < PI) return angle;

    // Use fmod for efficiency
    angle = std::fmod(angle + PI, TWO_PI);
    if (angle < 0) angle += TWO_PI;
    return angle - PI;
}

double NormalizeAngle0To2PI(double angle) {
    // Fast path
    if (angle >= 0 && angle < TWO_PI) return angle;

    angle = std::fmod(angle, TWO_PI);
    if (angle < 0) angle += TWO_PI;
    return angle;
}

double NormalizeAngleDiff(double angleDiff) {
    while (angleDiff > PI) angleDiff -= TWO_PI;
    while (angleDiff < -PI) angleDiff += TWO_PI;
    return angleDiff;
}

Ellipse2d NormalizeEllipse(const Ellipse2d& ellipse) {
    if (ellipse.a >= ellipse.b) {
        return ellipse;
    }
    // Swap axes and adjust angle by PI/2
    return Ellipse2d(ellipse.center, ellipse.b, ellipse.a,
                     NormalizeAngle(ellipse.angle + HALF_PI));
}

Arc2d NormalizeArc(const Arc2d& arc) {
    double startAngle = NormalizeAngle0To2PI(arc.startAngle);
    double sweepAngle = arc.sweepAngle;

    // Clamp sweepAngle to [-2*PI, 2*PI]
    if (sweepAngle > TWO_PI) sweepAngle = TWO_PI;
    else if (sweepAngle < -TWO_PI) sweepAngle = -TWO_PI;

    return Arc2d(arc.center, arc.radius, startAngle, sweepAngle);
}

// =============================================================================
// Point Operations
// =============================================================================

Point2d TransformPoint(const Point2d& point, const Mat33& matrix) {
    // Homogeneous coordinates: [x', y', w'] = matrix * [x, y, 1]
    double x = matrix(0, 0) * point.x + matrix(0, 1) * point.y + matrix(0, 2);
    double y = matrix(1, 0) * point.x + matrix(1, 1) * point.y + matrix(1, 2);
    double w = matrix(2, 0) * point.x + matrix(2, 1) * point.y + matrix(2, 2);

    if (std::abs(w) > GEOM_TOLERANCE) {
        return {x / w, y / w};
    }
    return {x, y};  // Point at infinity, return as-is
}

std::vector<Point2d> TransformPoints(const std::vector<Point2d>& points, const Mat33& matrix) {
    std::vector<Point2d> result;
    result.reserve(points.size());
    for (const auto& p : points) {
        result.push_back(TransformPoint(p, matrix));
    }
    return result;
}

// =============================================================================
// Line/Segment Operations
// =============================================================================

Line2d LinePerpendicular(const Line2d& line, const Point2d& point) {
    // Perpendicular: (b, -a) is the direction
    // Line through point: b*(x - px) - a*(y - py) = 0
    // => b*x - a*y - (b*px - a*py) = 0
    double c = -(line.b * point.x - line.a * point.y);
    return Line2d(line.b, -line.a, c);
}

Line2d LineParallel(const Line2d& line, const Point2d& point) {
    // Same direction coefficients (a, b), just different c
    double c = -(line.a * point.x + line.b * point.y);
    return Line2d(line.a, line.b, c);
}

Line2d LineFromPointAndAngle(const Point2d& point, double angle) {
    // Direction: (cos(angle), sin(angle))
    // Normal: (-sin(angle), cos(angle)) = (a, b)
    double a = -std::sin(angle);
    double b = std::cos(angle);
    double c = -(a * point.x + b * point.y);
    return Line2d(a, b, c);
}

Line2d TransformLine(const Line2d& line, const Mat33& matrix) {
    // For line ax + by + c = 0, under transformation M,
    // the transformed line is: (M^(-T) * [a, b, c]^T)
    Mat33 invT = matrix.Inverse().Transpose();

    double a = invT(0, 0) * line.a + invT(0, 1) * line.b + invT(0, 2) * line.c;
    double b = invT(1, 0) * line.a + invT(1, 1) * line.b + invT(1, 2) * line.c;
    double c = invT(2, 0) * line.a + invT(2, 1) * line.b + invT(2, 2) * line.c;

    return NormalizeLine(Line2d(a, b, c));
}

Segment2d TransformSegment(const Segment2d& segment, const Mat33& matrix) {
    return Segment2d(
        TransformPoint(segment.p1, matrix),
        TransformPoint(segment.p2, matrix)
    );
}

Segment2d ExtendSegment(const Segment2d& segment, double extendStart, double extendEnd) {
    double length = segment.Length();
    if (length < MIN_SEGMENT_LENGTH) {
        return segment;  // Degenerate segment
    }

    Point2d dir = segment.UnitDirection();
    Point2d newP1 = segment.p1 - dir * extendStart;
    Point2d newP2 = segment.p2 + dir * extendEnd;

    return Segment2d(newP1, newP2);
}

std::optional<Segment2d> ClipLineToRect(const Line2d& line, const Rect2d& bounds) {
    // Cohen-Sutherland line clipping algorithm adapted for infinite lines
    double xMin = bounds.x;
    double yMin = bounds.y;
    double xMax = bounds.x + bounds.width;
    double yMax = bounds.y + bounds.height;

    std::vector<Point2d> intersections;

    // Check intersection with each edge
    // For ax + by + c = 0:
    // - Vertical edges (fixed x): y = -(c + ax) / b, need b != 0
    // - Horizontal edges (fixed y): x = -(c + by) / a, need a != 0

    // Left edge: x = xMin
    if (std::abs(line.b) > GEOM_TOLERANCE) {
        double y = -(line.c + line.a * xMin) / line.b;
        if (y >= yMin && y <= yMax) {
            intersections.push_back({xMin, y});
        }
    }

    // Right edge: x = xMax
    if (std::abs(line.b) > GEOM_TOLERANCE) {
        double y = -(line.c + line.a * xMax) / line.b;
        if (y >= yMin && y <= yMax) {
            intersections.push_back({xMax, y});
        }
    }

    // Top edge: y = yMin
    if (std::abs(line.a) > GEOM_TOLERANCE) {
        double x = -(line.c + line.b * yMin) / line.a;
        if (x >= xMin && x <= xMax) {
            intersections.push_back({x, yMin});
        }
    }

    // Bottom edge: y = yMax
    if (std::abs(line.a) > GEOM_TOLERANCE) {
        double x = -(line.c + line.b * yMax) / line.a;
        if (x >= xMin && x <= xMax) {
            intersections.push_back({x, yMax});
        }
    }

    // Remove duplicates (corners)
    if (intersections.size() < 2) {
        return std::nullopt;
    }

    // Sort by x, then y to get consistent ordering
    std::sort(intersections.begin(), intersections.end(),
              [](const Point2d& a, const Point2d& b) {
                  if (std::abs(a.x - b.x) > GEOM_TOLERANCE) return a.x < b.x;
                  return a.y < b.y;
              });

    // Remove near-duplicates
    std::vector<Point2d> unique;
    for (const auto& p : intersections) {
        if (unique.empty() || unique.back().DistanceTo(p) > GEOM_TOLERANCE) {
            unique.push_back(p);
        }
    }

    if (unique.size() < 2) {
        return std::nullopt;
    }

    return Segment2d(unique.front(), unique.back());
}

// =============================================================================
// Circle/Arc Operations
// =============================================================================

Ellipse2d TransformCircle(const Circle2d& circle, const Mat33& matrix) {
    // Transform center
    Point2d newCenter = TransformPoint(circle.center, matrix);

    // Extract 2x2 linear part of the matrix
    double m00 = matrix(0, 0);
    double m01 = matrix(0, 1);
    double m10 = matrix(1, 0);
    double m11 = matrix(1, 1);

    // The circle x^2 + y^2 = r^2 transforms to an ellipse
    // Compute eigenvalues of M^T * M to get semi-axes
    double a = m00 * m00 + m10 * m10;
    double b = m00 * m01 + m10 * m11;
    double c = m01 * m01 + m11 * m11;

    // Eigenvalues of [a b; b c]
    double trace = a + c;
    double det = a * c - b * b;
    double disc = trace * trace - 4.0 * det;

    if (disc < 0) disc = 0;  // Numerical stability

    double sqrtDisc = std::sqrt(disc);
    double lambda1 = (trace + sqrtDisc) * 0.5;
    double lambda2 = (trace - sqrtDisc) * 0.5;

    double semiMajor = circle.radius * std::sqrt(lambda1);
    double semiMinor = circle.radius * std::sqrt(std::max(0.0, lambda2));

    // Compute angle from eigenvector
    double angle = 0.0;
    if (std::abs(b) > GEOM_TOLERANCE) {
        angle = 0.5 * std::atan2(2.0 * b, a - c);
    } else if (a < c) {
        angle = HALF_PI;
    }

    return NormalizeEllipse(Ellipse2d(newCenter, semiMajor, semiMinor, angle));
}

std::optional<Arc2d> ArcFrom3Points(const Point2d& p1, const Point2d& p2, const Point2d& p3) {
    // First, find the circle through the three points using Fitting module
    auto circleOpt = FitCircleExact3Points(p1, p2, p3);
    if (!circleOpt.has_value()) {
        return std::nullopt;  // Points are collinear
    }

    Circle2d circle = circleOpt.value();

    // Compute angles from center to each point
    double angle1 = std::atan2(p1.y - circle.center.y, p1.x - circle.center.x);
    double angle2 = std::atan2(p2.y - circle.center.y, p2.x - circle.center.x);
    double angle3 = std::atan2(p3.y - circle.center.y, p3.x - circle.center.x);

    // Determine the sweep direction
    // p2 should be between p1 and p3 on the arc
    double sweep13 = NormalizeAngleDiff(angle3 - angle1);
    double sweep12 = NormalizeAngleDiff(angle2 - angle1);

    double sweepAngle;
    if ((sweep13 > 0 && sweep12 > 0 && sweep12 < sweep13) ||
        (sweep13 < 0 && (sweep12 < 0 && sweep12 > sweep13))) {
        // Normal direction: p1 -> p2 -> p3
        sweepAngle = sweep13;
    } else {
        // Opposite direction: need to go the long way
        if (sweep13 > 0) {
            sweepAngle = sweep13 - TWO_PI;
        } else {
            sweepAngle = sweep13 + TWO_PI;
        }
    }

    return Arc2d(circle.center, circle.radius, angle1, sweepAngle);
}

Arc2d ArcFromAngles(const Point2d& center, double radius,
                    double startAngle, double endAngle,
                    ArcDirection direction) {
    double sweepAngle;
    if (direction == ArcDirection::CounterClockwise) {
        sweepAngle = NormalizeAngleDiff(endAngle - startAngle);
        if (sweepAngle < 0) sweepAngle += TWO_PI;
    } else {
        sweepAngle = NormalizeAngleDiff(endAngle - startAngle);
        if (sweepAngle > 0) sweepAngle -= TWO_PI;
    }

    return Arc2d(center, radius, startAngle, sweepAngle);
}

Arc2d TransformArc(const Arc2d& arc, const Mat33& matrix) {
    // Transform the arc by transforming its key points and reconstructing
    Point2d startPt = arc.StartPoint();
    Point2d midPt = arc.Midpoint();
    Point2d endPt = arc.EndPoint();

    Point2d newStart = TransformPoint(startPt, matrix);
    Point2d newMid = TransformPoint(midPt, matrix);
    Point2d newEnd = TransformPoint(endPt, matrix);

    // Try to create arc from transformed points
    auto arcOpt = ArcFrom3Points(newStart, newMid, newEnd);
    if (arcOpt.has_value()) {
        return arcOpt.value();
    }

    // Fallback: transform center and approximate
    Point2d newCenter = TransformPoint(arc.center, matrix);
    double scale = std::sqrt(matrix(0, 0) * matrix(0, 0) + matrix(1, 0) * matrix(1, 0));
    double newRadius = arc.radius * scale;
    double newStartAngle = std::atan2(newStart.y - newCenter.y, newStart.x - newCenter.x);
    double newEndAngle = std::atan2(newEnd.y - newCenter.y, newEnd.x - newCenter.x);

    // Preserve direction
    return ArcFromAngles(newCenter, newRadius, newStartAngle, newEndAngle,
                         arc.sweepAngle >= 0 ? ArcDirection::CounterClockwise : ArcDirection::Clockwise);
}

Segment2d ArcToChord(const Arc2d& arc) {
    return Segment2d(arc.StartPoint(), arc.EndPoint());
}

std::pair<Arc2d, Arc2d> SplitArc(const Arc2d& arc, double t) {
    t = Clamp(t, 0.0, 1.0);

    double midAngle = arc.startAngle + arc.sweepAngle * t;
    double sweep1 = arc.sweepAngle * t;
    double sweep2 = arc.sweepAngle * (1.0 - t);

    Arc2d arc1(arc.center, arc.radius, arc.startAngle, sweep1);
    Arc2d arc2(arc.center, arc.radius, midAngle, sweep2);

    return {arc1, arc2};
}

// =============================================================================
// Ellipse Operations
// =============================================================================

Ellipse2d RotateEllipse(const Ellipse2d& ellipse, double angle) {
    return Ellipse2d(ellipse.center, ellipse.a, ellipse.b,
                     NormalizeAngle(ellipse.angle + angle));
}

Ellipse2d RotateEllipseAround(const Ellipse2d& ellipse, const Point2d& center, double angle) {
    Point2d newCenter = RotatePointAround(ellipse.center, center, angle);
    return Ellipse2d(newCenter, ellipse.a, ellipse.b,
                     NormalizeAngle(ellipse.angle + angle));
}

Ellipse2d TransformEllipse(const Ellipse2d& ellipse, const Mat33& matrix) {
    // Transform center
    Point2d newCenter = TransformPoint(ellipse.center, matrix);

    // Build ellipse matrix Q such that x^T Q x = 1 defines the ellipse
    // In local coordinates: (x/a)^2 + (y/b)^2 = 1
    // Q_local = diag(1/a^2, 1/b^2)
    // Rotate to world: Q = R * Q_local * R^T

    double cosA = std::cos(ellipse.angle);
    double sinA = std::sin(ellipse.angle);

    double a2inv = 1.0 / (ellipse.a * ellipse.a);
    double b2inv = 1.0 / (ellipse.b * ellipse.b);

    // Q in world coordinates (2x2 part)
    double q11 = cosA * cosA * a2inv + sinA * sinA * b2inv;
    double q12 = cosA * sinA * (a2inv - b2inv);
    double q22 = sinA * sinA * a2inv + cosA * cosA * b2inv;

    // Extract linear part of transform
    double m00 = matrix(0, 0);
    double m01 = matrix(0, 1);
    double m10 = matrix(1, 0);
    double m11 = matrix(1, 1);

    // Inverse of linear part
    double det = m00 * m11 - m01 * m10;
    if (std::abs(det) < GEOM_TOLERANCE) {
        // Degenerate transform
        return Ellipse2d(newCenter, ellipse.a, ellipse.b, ellipse.angle);
    }

    double invDet = 1.0 / det;
    double i00 = m11 * invDet;
    double i01 = -m01 * invDet;
    double i10 = -m10 * invDet;
    double i11 = m00 * invDet;

    // Transform Q: Q' = M^(-T) * Q * M^(-1)
    // First: T = Q * M^(-1)
    double t11 = q11 * i00 + q12 * i10;
    double t12 = q11 * i01 + q12 * i11;
    double t21 = q12 * i00 + q22 * i10;
    double t22 = q12 * i01 + q22 * i11;

    // Then: Q' = M^(-T) * T
    double qp11 = i00 * t11 + i10 * t21;
    double qp12 = i00 * t12 + i10 * t22;
    double qp22 = i01 * t12 + i11 * t22;

    // Extract ellipse parameters from Q'
    // Q' is symmetric: [qp11 qp12; qp12 qp22]
    // Eigendecomposition gives axes and angle

    double trace = qp11 + qp22;
    double detQ = qp11 * qp22 - qp12 * qp12;
    double disc = trace * trace - 4.0 * detQ;
    if (disc < 0) disc = 0;

    double sqrtDisc = std::sqrt(disc);
    double lambda1 = (trace + sqrtDisc) * 0.5;
    double lambda2 = (trace - sqrtDisc) * 0.5;

    if (lambda1 <= 0 || lambda2 <= 0) {
        // Not a valid ellipse
        return Ellipse2d(newCenter, ellipse.a, ellipse.b, ellipse.angle);
    }

    double newA = 1.0 / std::sqrt(lambda2);  // Larger semi-axis
    double newB = 1.0 / std::sqrt(lambda1);  // Smaller semi-axis

    // Compute angle from eigenvector
    double newAngle = 0.0;
    if (std::abs(qp12) > GEOM_TOLERANCE) {
        newAngle = 0.5 * std::atan2(2.0 * qp12, qp11 - qp22);
    } else if (qp11 > qp22) {
        newAngle = 0.0;
    } else {
        newAngle = HALF_PI;
    }

    return NormalizeEllipse(Ellipse2d(newCenter, newA, newB, newAngle));
}

double EllipseRadiusAt(const Ellipse2d& ellipse, double theta) {
    double cosT = std::cos(theta);
    double sinT = std::sin(theta);
    double a = ellipse.a;
    double b = ellipse.b;

    // r = a*b / sqrt((b*cos(t))^2 + (a*sin(t))^2)
    double denom = std::sqrt(b * b * cosT * cosT + a * a * sinT * sinT);
    if (denom < GEOM_TOLERANCE) return ellipse.a;

    return (a * b) / denom;
}

Point2d EllipsePointAt(const Ellipse2d& ellipse, double theta) {
    // Point in local coordinates
    double localX = ellipse.a * std::cos(theta);
    double localY = ellipse.b * std::sin(theta);

    // Rotate to world coordinates
    double cosA = std::cos(ellipse.angle);
    double sinA = std::sin(ellipse.angle);

    double worldX = ellipse.center.x + localX * cosA - localY * sinA;
    double worldY = ellipse.center.y + localX * sinA + localY * cosA;

    return {worldX, worldY};
}

Point2d EllipseTangentAt(const Ellipse2d& ellipse, double theta) {
    // Derivative of parametric form in local coordinates
    double localTx = -ellipse.a * std::sin(theta);
    double localTy = ellipse.b * std::cos(theta);

    // Rotate to world coordinates (note: only rotation, not translation)
    double cosA = std::cos(ellipse.angle);
    double sinA = std::sin(ellipse.angle);

    double worldTx = localTx * cosA - localTy * sinA;
    double worldTy = localTx * sinA + localTy * cosA;

    // Normalize
    double norm = std::sqrt(worldTx * worldTx + worldTy * worldTy);
    if (norm < GEOM_TOLERANCE) {
        return {1.0, 0.0};
    }

    return {worldTx / norm, worldTy / norm};
}

Point2d EllipseNormalAt(const Ellipse2d& ellipse, double theta) {
    Point2d tangent = EllipseTangentAt(ellipse, theta);
    // Normal is perpendicular to tangent (outward)
    return {tangent.y, -tangent.x};
}

double EllipseArcLength(const Ellipse2d& ellipse, double thetaStart, double thetaEnd) {
    // Numerical integration using adaptive Simpson's rule
    auto integrand = [&](double t) {
        double dx = -ellipse.a * std::sin(t);
        double dy = ellipse.b * std::cos(t);
        return std::sqrt(dx * dx + dy * dy);
    };

    // Simple Simpson's rule with fixed segments
    int n = 64;  // Number of segments
    double h = (thetaEnd - thetaStart) / n;
    double sum = integrand(thetaStart) + integrand(thetaEnd);

    for (int i = 1; i < n; i += 2) {
        sum += 4.0 * integrand(thetaStart + i * h);
    }
    for (int i = 2; i < n; i += 2) {
        sum += 2.0 * integrand(thetaStart + i * h);
    }

    return std::abs(sum * h / 3.0);
}

// =============================================================================
// RotatedRect Operations
// =============================================================================

RotatedRect2d RotateRotatedRect(const RotatedRect2d& rect, double angle) {
    return RotatedRect2d(rect.center, rect.width, rect.height,
                         NormalizeAngle(rect.angle + angle));
}

RotatedRect2d RotateRotatedRectAround(const RotatedRect2d& rect, const Point2d& center, double angle) {
    Point2d newCenter = RotatePointAround(rect.center, center, angle);
    return RotatedRect2d(newCenter, rect.width, rect.height,
                         NormalizeAngle(rect.angle + angle));
}

RotatedRect2d TransformRotatedRect(const RotatedRect2d& rect, const Mat33& matrix) {
    // Transform center
    Point2d newCenter = TransformPoint(rect.center, matrix);

    // Transform the corners and fit a new rotated rect
    auto corners = RotatedRectCorners(rect);
    Point2d c0 = TransformPoint(corners[0], matrix);
    Point2d c1 = TransformPoint(corners[1], matrix);
    Point2d c3 = TransformPoint(corners[3], matrix);

    // New width and height from transformed edges
    double newWidth = c0.DistanceTo(c1);
    double newHeight = c0.DistanceTo(c3);

    // New angle from first edge
    Point2d edge = c1 - c0;
    double newAngle = std::atan2(edge.y, edge.x);

    return RotatedRect2d(newCenter, newWidth, newHeight, newAngle);
}

std::array<Point2d, 4> RotatedRectCorners(const RotatedRect2d& rect) {
    double cosA = std::cos(rect.angle);
    double sinA = std::sin(rect.angle);

    double hw = rect.width * 0.5;
    double hh = rect.height * 0.5;

    // Local corners: (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)
    // Rotate and translate
    std::array<Point2d, 4> corners;

    double dx0 = -hw * cosA + hh * sinA;
    double dy0 = -hw * sinA - hh * cosA;
    corners[0] = {rect.center.x + dx0, rect.center.y + dy0};  // Top-left

    double dx1 = hw * cosA + hh * sinA;
    double dy1 = hw * sinA - hh * cosA;
    corners[1] = {rect.center.x + dx1, rect.center.y + dy1};  // Top-right

    double dx2 = hw * cosA - hh * sinA;
    double dy2 = hw * sinA + hh * cosA;
    corners[2] = {rect.center.x + dx2, rect.center.y + dy2};  // Bottom-right

    double dx3 = -hw * cosA - hh * sinA;
    double dy3 = -hw * sinA + hh * cosA;
    corners[3] = {rect.center.x + dx3, rect.center.y + dy3};  // Bottom-left

    return corners;
}

std::array<Segment2d, 4> RotatedRectEdges(const RotatedRect2d& rect) {
    auto corners = RotatedRectCorners(rect);
    return {
        Segment2d(corners[0], corners[1]),  // Top
        Segment2d(corners[1], corners[2]),  // Right
        Segment2d(corners[2], corners[3]),  // Bottom
        Segment2d(corners[3], corners[0])   // Left
    };
}

// =============================================================================
// Property Computation
// =============================================================================

double ArcSegmentArea(const Arc2d& arc) {
    // Area = (r^2/2) * (theta - sin(theta))
    double theta = std::abs(arc.sweepAngle);
    return 0.5 * arc.radius * arc.radius * (theta - std::sin(theta));
}

Rect2d ArcBoundingBox(const Arc2d& arc) {
    Point2d startPt = arc.StartPoint();
    Point2d endPt = arc.EndPoint();

    double minX = std::min(startPt.x, endPt.x);
    double maxX = std::max(startPt.x, endPt.x);
    double minY = std::min(startPt.y, endPt.y);
    double maxY = std::max(startPt.y, endPt.y);

    // Check if arc passes through axis extrema (0, PI/2, PI, 3*PI/2)
    auto checkAngle = [&](double angle) {
        if (AngleInArcRange(angle, arc)) {
            double x = arc.center.x + arc.radius * std::cos(angle);
            double y = arc.center.y + arc.radius * std::sin(angle);
            minX = std::min(minX, x);
            maxX = std::max(maxX, x);
            minY = std::min(minY, y);
            maxY = std::max(maxY, y);
        }
    };

    checkAngle(0);
    checkAngle(HALF_PI);
    checkAngle(PI);
    checkAngle(-HALF_PI);

    return Rect2d(minX, minY, maxX - minX, maxY - minY);
}

Rect2d EllipseBoundingBox(const Ellipse2d& ellipse) {
    // For a rotated ellipse, the bounding box is found by:
    // x_max = sqrt(a^2 * cos^2(angle) + b^2 * sin^2(angle))
    // y_max = sqrt(a^2 * sin^2(angle) + b^2 * cos^2(angle))

    double cosA = std::cos(ellipse.angle);
    double sinA = std::sin(ellipse.angle);

    double a2 = ellipse.a * ellipse.a;
    double b2 = ellipse.b * ellipse.b;

    double halfWidth = std::sqrt(a2 * cosA * cosA + b2 * sinA * sinA);
    double halfHeight = std::sqrt(a2 * sinA * sinA + b2 * cosA * cosA);

    return Rect2d(ellipse.center.x - halfWidth, ellipse.center.y - halfHeight,
                  2.0 * halfWidth, 2.0 * halfHeight);
}

Point2d ArcCentroid(const Arc2d& arc) {
    // Centroid of arc curve (not sector)
    // For an arc from angle1 to angle2:
    // x_c = r * sin(angle2) - sin(angle1) / (angle2 - angle1)
    // y_c = r * (cos(angle1) - cos(angle2)) / (angle2 - angle1)

    if (std::abs(arc.sweepAngle) < GEOM_TOLERANCE) {
        return arc.StartPoint();
    }

    double angle1 = arc.startAngle;
    double angle2 = arc.EndAngle();
    double dAngle = arc.sweepAngle;

    double xLocal = arc.radius * (std::sin(angle2) - std::sin(angle1)) / dAngle;
    double yLocal = arc.radius * (std::cos(angle1) - std::cos(angle2)) / dAngle;

    return {arc.center.x + xLocal, arc.center.y + yLocal};
}

Point2d ArcSectorCentroid(const Arc2d& arc) {
    // Centroid of sector: 2r/(3*theta) * (sin(theta/2)) from center along bisector
    if (std::abs(arc.sweepAngle) < GEOM_TOLERANCE) {
        return arc.center;
    }

    double theta = std::abs(arc.sweepAngle);
    double midAngle = arc.startAngle + arc.sweepAngle * 0.5;
    double dist = (2.0 * arc.radius * std::sin(theta * 0.5)) / (1.5 * theta);

    return {arc.center.x + dist * std::cos(midAngle),
            arc.center.y + dist * std::sin(midAngle)};
}

// =============================================================================
// Sampling/Discretization
// =============================================================================

std::vector<Point2d> SampleSegment(const Segment2d& segment, double step,
                                    bool includeEndpoints) {
    double length = segment.Length();
    size_t numPoints = ComputeSamplingCount(length, step, 2, MAX_SAMPLING_POINTS);

    std::vector<Point2d> points;
    points.reserve(numPoints);

    if (numPoints <= 2 || length < MIN_SEGMENT_LENGTH) {
        if (includeEndpoints) {
            points.push_back(segment.p1);
            points.push_back(segment.p2);
        }
        return points;
    }

    for (size_t i = 0; i < numPoints; ++i) {
        double t = static_cast<double>(i) / (numPoints - 1);
        points.push_back(segment.PointAt(t));
    }

    // Ensure exact endpoints
    if (includeEndpoints && !points.empty()) {
        points.front() = segment.p1;
        points.back() = segment.p2;
    }

    return points;
}

std::vector<Point2d> SampleSegmentByCount(const Segment2d& segment, size_t numPoints) {
    if (numPoints < 2) numPoints = 2;
    numPoints = std::min(numPoints, MAX_SAMPLING_POINTS);

    std::vector<Point2d> points;
    points.reserve(numPoints);

    for (size_t i = 0; i < numPoints; ++i) {
        double t = static_cast<double>(i) / (numPoints - 1);
        points.push_back(segment.PointAt(t));
    }

    return points;
}

std::vector<Point2d> SampleCircle(const Circle2d& circle, double step) {
    double circumference = circle.Circumference();
    size_t numPoints = ComputeSamplingCount(circumference, step, 4, MAX_SAMPLING_POINTS);

    return SampleCircleByCount(circle, numPoints, true);
}

std::vector<Point2d> SampleCircleByCount(const Circle2d& circle, size_t numPoints, bool closed) {
    if (numPoints < 3) numPoints = 3;
    numPoints = std::min(numPoints, MAX_SAMPLING_POINTS);

    std::vector<Point2d> points;
    points.reserve(closed ? numPoints + 1 : numPoints);

    for (size_t i = 0; i < numPoints; ++i) {
        double angle = TWO_PI * static_cast<double>(i) / numPoints;
        double x = circle.center.x + circle.radius * std::cos(angle);
        double y = circle.center.y + circle.radius * std::sin(angle);
        points.push_back({x, y});
    }

    if (closed && !points.empty()) {
        points.push_back(points.front());
    }

    return points;
}

std::vector<Point2d> SampleArc(const Arc2d& arc, double step, bool /*includeEndpoints*/) {
    // Note: includeEndpoints is always true in current implementation via SampleArcByCount
    double arcLength = arc.Length();
    size_t numPoints = ComputeSamplingCount(arcLength, step, 2, MAX_SAMPLING_POINTS);

    return SampleArcByCount(arc, numPoints);
}

std::vector<Point2d> SampleArcByCount(const Arc2d& arc, size_t numPoints) {
    if (numPoints < 2) numPoints = 2;
    numPoints = std::min(numPoints, MAX_SAMPLING_POINTS);

    std::vector<Point2d> points;
    points.reserve(numPoints);

    for (size_t i = 0; i < numPoints; ++i) {
        double t = static_cast<double>(i) / (numPoints - 1);
        points.push_back(arc.PointAt(t));
    }

    // Ensure exact endpoints
    if (!points.empty()) {
        points.front() = arc.StartPoint();
        points.back() = arc.EndPoint();
    }

    return points;
}

std::vector<Point2d> SampleEllipse(const Ellipse2d& ellipse, double step) {
    double perimeter = ellipse.Perimeter();
    size_t numPoints = ComputeSamplingCount(perimeter, step, 8, MAX_SAMPLING_POINTS);

    return SampleEllipseByCount(ellipse, numPoints, true);
}

std::vector<Point2d> SampleEllipseByCount(const Ellipse2d& ellipse, size_t numPoints, bool closed) {
    if (numPoints < 4) numPoints = 4;
    numPoints = std::min(numPoints, MAX_SAMPLING_POINTS);

    std::vector<Point2d> points;
    points.reserve(closed ? numPoints + 1 : numPoints);

    for (size_t i = 0; i < numPoints; ++i) {
        double theta = TWO_PI * static_cast<double>(i) / numPoints;
        points.push_back(EllipsePointAt(ellipse, theta));
    }

    if (closed && !points.empty()) {
        points.push_back(points.front());
    }

    return points;
}

std::vector<Point2d> SampleEllipseArc(const Ellipse2d& ellipse,
                                       double thetaStart, double thetaEnd,
                                       double step, bool includeEndpoints) {
    double arcLength = EllipseArcLength(ellipse, thetaStart, thetaEnd);
    size_t numPoints = ComputeSamplingCount(arcLength, step, 2, MAX_SAMPLING_POINTS);

    std::vector<Point2d> points;
    points.reserve(numPoints);

    for (size_t i = 0; i < numPoints; ++i) {
        double t = static_cast<double>(i) / (numPoints - 1);
        double theta = thetaStart + t * (thetaEnd - thetaStart);
        points.push_back(EllipsePointAt(ellipse, theta));
    }

    // Ensure exact endpoints
    if (includeEndpoints && !points.empty()) {
        points.front() = EllipsePointAt(ellipse, thetaStart);
        points.back() = EllipsePointAt(ellipse, thetaEnd);
    }

    return points;
}

std::vector<Point2d> SampleRotatedRect(const RotatedRect2d& rect,
                                        double step, bool closed) {
    auto edges = RotatedRectEdges(rect);

    std::vector<Point2d> points;
    double perimeter = 2.0 * (rect.width + rect.height);
    size_t estimatedPoints = ComputeSamplingCount(perimeter, step, 4, MAX_SAMPLING_POINTS);
    points.reserve(estimatedPoints);

    for (int i = 0; i < 4; ++i) {
        auto edgePoints = SampleSegment(edges[i], step, true);
        // Skip first point of subsequent edges (it's the last point of previous)
        size_t startIdx = (i == 0) ? 0 : 1;
        for (size_t j = startIdx; j < edgePoints.size() - 1; ++j) {
            points.push_back(edgePoints[j]);
        }
    }

    // Add last point
    if (!points.empty()) {
        if (closed) {
            points.push_back(points.front());
        } else {
            points.push_back(edges[3].p2);
        }
    }

    return points;
}

// =============================================================================
// Utility Functions
// =============================================================================

bool PointOnSegment(const Point2d& point, const Segment2d& segment, double tolerance) {
    // Check distance to line
    Line2d line = segment.ToLine();
    if (std::abs(line.SignedDistance(point)) > tolerance) {
        return false;
    }

    // Check if projection is within segment
    double t = segment.ProjectPoint(point);
    return t >= -tolerance / segment.Length() && t <= 1.0 + tolerance / segment.Length();
}

bool PointOnArc(const Point2d& point, const Arc2d& arc, double tolerance) {
    // Check if point is on the circle
    double distToCenter = point.DistanceTo(arc.center);
    if (std::abs(distToCenter - arc.radius) > tolerance) {
        return false;
    }

    // Check if angle is within arc range
    double angle = std::atan2(point.y - arc.center.y, point.x - arc.center.x);
    return AngleInArcRange(angle, arc);
}

bool PointOnEllipse(const Point2d& point, const Ellipse2d& ellipse, double tolerance) {
    // Transform point to ellipse local coordinates
    double dx = point.x - ellipse.center.x;
    double dy = point.y - ellipse.center.y;

    double cosA = std::cos(-ellipse.angle);
    double sinA = std::sin(-ellipse.angle);

    double localX = dx * cosA - dy * sinA;
    double localY = dx * sinA + dy * cosA;

    // Check ellipse equation: (x/a)^2 + (y/b)^2 = 1
    double val = (localX * localX) / (ellipse.a * ellipse.a) +
                 (localY * localY) / (ellipse.b * ellipse.b);

    // Approximate distance check
    // For points near ellipse, |val - 1| is approximately proportional to distance
    double avgRadius = (ellipse.a + ellipse.b) * 0.5;
    return std::abs(val - 1.0) * avgRadius * 0.5 <= tolerance;
}

double AngleBetweenLines(const Line2d& line1, const Line2d& line2) {
    // cos(angle) = |dot(n1, n2)| = |a1*a2 + b1*b2|
    double dot = std::abs(line1.a * line2.a + line1.b * line2.b);
    dot = std::min(1.0, dot);  // Clamp for numerical stability
    return std::acos(dot);
}

bool AreParallel(const Line2d& line1, const Line2d& line2, double tolerance) {
    // Lines are parallel if their normal vectors are parallel (or anti-parallel)
    // |cross(n1, n2)| = |a1*b2 - a2*b1| should be near 0
    double cross = std::abs(line1.a * line2.b - line2.a * line1.b);
    return cross < tolerance;
}

bool ArePerpendicular(const Line2d& line1, const Line2d& line2, double tolerance) {
    // Lines are perpendicular if their normal vectors are perpendicular
    // |dot(n1, n2)| = |a1*a2 + b1*b2| should be near 0
    double dot = std::abs(line1.a * line2.a + line1.b * line2.b);
    return dot < tolerance;
}

bool AreCollinear(const Segment2d& seg1, const Segment2d& seg2, double tolerance) {
    // All four endpoints should be on the same line
    Line2d line = seg1.ToLine();
    return std::abs(line.SignedDistance(seg2.p1)) <= tolerance &&
           std::abs(line.SignedDistance(seg2.p2)) <= tolerance;
}

Point2d ProjectPointOnLine(const Point2d& point, const Line2d& line) {
    // Foot of perpendicular from point to line
    // p' = p - (a*px + b*py + c) * (a, b)
    double dist = line.SignedDistance(point);
    return {point.x - dist * line.a, point.y - dist * line.b};
}

Point2d ProjectPointOnSegment(const Point2d& point, const Segment2d& segment) {
    double t = segment.ProjectPoint(point);
    t = Clamp(t, 0.0, 1.0);
    return segment.PointAt(t);
}

Point2d ProjectPointOnCircle(const Point2d& point, const Circle2d& circle) {
    Point2d dir = point - circle.center;
    double dist = dir.Norm();

    if (dist < GEOM_TOLERANCE) {
        // Point is at center, return arbitrary point on circle
        return {circle.center.x + circle.radius, circle.center.y};
    }

    return circle.center + dir * (circle.radius / dist);
}

Point2d ReflectPointAcrossLine(const Point2d& point, const Line2d& line) {
    // Reflect = p - 2 * signed_distance * normal
    double dist = line.SignedDistance(point);
    return {point.x - 2.0 * dist * line.a, point.y - 2.0 * dist * line.b};
}

bool AngleInArcRange(double angle, const Arc2d& arc) {
    // Normalize angle relative to start angle
    double relAngle = NormalizeAngleDiff(angle - arc.startAngle);

    if (arc.sweepAngle >= 0) {
        return relAngle >= -ANGLE_TOLERANCE && relAngle <= arc.sweepAngle + ANGLE_TOLERANCE;
    } else {
        return relAngle <= ANGLE_TOLERANCE && relAngle >= arc.sweepAngle - ANGLE_TOLERANCE;
    }
}

} // namespace Qi::Vision::Internal
