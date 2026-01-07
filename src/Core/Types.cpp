#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>
#include <algorithm>

namespace Qi::Vision {

// =============================================================================
// Line2d Implementation
// =============================================================================

Line2d::Line2d(double a_, double b_, double c_) {
    double norm = std::sqrt(a_ * a_ + b_ * b_);
    if (norm > EPSILON) {
        a = a_ / norm;
        b = b_ / norm;
        c = c_ / norm;
    } else {
        a = 0;
        b = 1;
        c = 0;
    }
}

Line2d Line2d::FromPoints(const Point2d& p1, const Point2d& p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    // Line equation: -dy * x + dx * y + (dy * p1.x - dx * p1.y) = 0
    return Line2d(-dy, dx, dy * p1.x - dx * p1.y);
}

Line2d Line2d::FromPointAngle(const Point2d& point, double angle) {
    double cosA = std::cos(angle);
    double sinA = std::sin(angle);
    // Normal is (-sin, cos), so line is: -sin*x + cos*y + c = 0
    return Line2d(-sinA, cosA, sinA * point.x - cosA * point.y);
}

// =============================================================================
// Circle2d Implementation
// =============================================================================

double Circle2d::Area() const {
    return PI * radius * radius;
}

double Circle2d::Circumference() const {
    return TWO_PI * radius;
}

// =============================================================================
// Segment2d Implementation
// =============================================================================

double Segment2d::DistanceToPoint(const Point2d& p) const {
    double t = ProjectPoint(p);
    t = std::clamp(t, 0.0, 1.0);
    Point2d closest = PointAt(t);
    return p.DistanceTo(closest);
}

double Segment2d::ProjectPoint(const Point2d& p) const {
    Point2d d = Direction();
    double lenSq = d.Dot(d);
    if (lenSq < EPSILON * EPSILON) {
        return 0.0;
    }
    Point2d v = p - p1;
    return v.Dot(d) / lenSq;
}

// =============================================================================
// Ellipse2d Implementation
// =============================================================================

double Ellipse2d::Area() const {
    return PI * a * b;
}

double Ellipse2d::Perimeter() const {
    // Ramanujan's approximation
    double h = Square(a - b) / Square(a + b);
    return PI * (a + b) * (1.0 + 3.0 * h / (10.0 + std::sqrt(4.0 - 3.0 * h)));
}

bool Ellipse2d::Contains(const Point2d& p) const {
    // Transform point to ellipse local coordinates
    double cosA = std::cos(-angle);
    double sinA = std::sin(-angle);
    double dx = p.x - center.x;
    double dy = p.y - center.y;
    double localX = dx * cosA - dy * sinA;
    double localY = dx * sinA + dy * cosA;

    // Check if inside: (x/a)^2 + (y/b)^2 <= 1
    if (a <= 0 || b <= 0) return false;
    double val = Square(localX / a) + Square(localY / b);
    return val <= 1.0;
}

Point2d Ellipse2d::PointAt(double theta) const {
    double cosA = std::cos(angle);
    double sinA = std::sin(angle);
    double localX = a * std::cos(theta);
    double localY = b * std::sin(theta);
    // Rotate to world coordinates
    return {
        center.x + localX * cosA - localY * sinA,
        center.y + localX * sinA + localY * cosA
    };
}

// =============================================================================
// RotatedRect2d Implementation
// =============================================================================

void RotatedRect2d::GetCorners(Point2d corners[4]) const {
    double cosA = std::cos(angle);
    double sinA = std::sin(angle);
    double hw = width * 0.5;
    double hh = height * 0.5;

    // Local corners: (-hw,-hh), (hw,-hh), (hw,hh), (-hw,hh)
    double localX[4] = {-hw,  hw, hw, -hw};
    double localY[4] = {-hh, -hh, hh,  hh};

    for (int i = 0; i < 4; ++i) {
        corners[i].x = center.x + localX[i] * cosA - localY[i] * sinA;
        corners[i].y = center.y + localX[i] * sinA + localY[i] * cosA;
    }
}

Rect2d RotatedRect2d::BoundingBox() const {
    Point2d corners[4];
    GetCorners(corners);

    double minX = corners[0].x, maxX = corners[0].x;
    double minY = corners[0].y, maxY = corners[0].y;

    for (int i = 1; i < 4; ++i) {
        minX = std::min(minX, corners[i].x);
        maxX = std::max(maxX, corners[i].x);
        minY = std::min(minY, corners[i].y);
        maxY = std::max(maxY, corners[i].y);
    }

    return Rect2d(minX, minY, maxX - minX, maxY - minY);
}

bool RotatedRect2d::Contains(const Point2d& p) const {
    // Transform point to local coordinates
    double cosA = std::cos(-angle);
    double sinA = std::sin(-angle);
    double dx = p.x - center.x;
    double dy = p.y - center.y;
    double localX = dx * cosA - dy * sinA;
    double localY = dx * sinA + dy * cosA;

    // Check if inside axis-aligned rect centered at origin
    double hw = width * 0.5;
    double hh = height * 0.5;
    return std::abs(localX) <= hw && std::abs(localY) <= hh;
}

} // namespace Qi::Vision
