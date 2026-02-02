#pragma once

/**
 * @file Types.h
 * @brief Core type definitions for QiVision
 */

#include <cstdint>
#include <QiVision/Core/Export.h>
#include <cmath>
#include <vector>
#include <optional>

namespace Qi::Vision {

// =============================================================================
// Pixel Types
// =============================================================================

/**
 * @brief Supported pixel data types
 */
enum class PixelType {
    UInt8,      ///< 8-bit unsigned [0, 255]
    UInt16,     ///< 16-bit unsigned [0, 65535]
    Int16,      ///< 16-bit signed [-32768, 32767]
    Float32     ///< 32-bit float
};

/**
 * @brief Image channel types
 */
enum class ChannelType {
    Gray,       ///< Single channel grayscale
    RGB,        ///< 3 channels RGB
    BGR,        ///< 3 channels BGR
    RGBA,       ///< 4 channels RGBA
    BGRA        ///< 4 channels BGRA
};

// =============================================================================
// 2D Point Types
// =============================================================================

/**
 * @brief 2D point with integer coordinates
 * @note Uses int32_t to support >32K resolution (line scan cameras)
 */
struct QIVISION_API Point2i {
    int32_t x = 0;
    int32_t y = 0;

    Point2i() = default;
    Point2i(int32_t x_, int32_t y_) : x(x_), y(y_) {}

    bool IsValid() const { return true; }
};

/**
 * @brief 2D point with sub-pixel precision
 */
struct QIVISION_API Point2d {
    double x = 0.0;
    double y = 0.0;

    Point2d() = default;
    Point2d(double x_, double y_) : x(x_), y(y_) {}

    bool IsValid() const { return std::isfinite(x) && std::isfinite(y); }

    /// Vector addition
    Point2d operator+(const Point2d& other) const {
        return {x + other.x, y + other.y};
    }

    /// Vector subtraction
    Point2d operator-(const Point2d& other) const {
        return {x - other.x, y - other.y};
    }

    /// Scalar multiplication
    Point2d operator*(double s) const {
        return {x * s, y * s};
    }

    /// Euclidean norm
    double Norm() const {
        return std::sqrt(x * x + y * y);
    }

    /// Dot product
    double Dot(const Point2d& other) const {
        return x * other.x + y * other.y;
    }

    /// Cross product (2D: returns scalar)
    double Cross(const Point2d& other) const {
        return x * other.y - y * other.x;
    }

    /// Distance to another point
    double DistanceTo(const Point2d& other) const {
        return (*this - other).Norm();
    }
};

// =============================================================================
// 3D Point Type
// =============================================================================

/**
 * @brief 3D point with double precision
 */
struct QIVISION_API Point3d {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    Point3d() = default;
    Point3d(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    bool IsValid() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }
};

// =============================================================================
// Size Type
// =============================================================================

/**
 * @brief 2D size with integer dimensions
 */
struct QIVISION_API Size2i {
    int32_t width = 0;
    int32_t height = 0;

    Size2i() = default;
    Size2i(int32_t w, int32_t h) : width(w), height(h) {}

    int64_t Area() const { return static_cast<int64_t>(width) * height; }
    bool IsValid() const { return width >= 0 && height >= 0; }
};

// =============================================================================
// Rectangle Types
// =============================================================================

/**
 * @brief Axis-aligned rectangle with integer coordinates
 */
struct QIVISION_API Rect2i {
    int32_t x = 0;      ///< Left
    int32_t y = 0;      ///< Top
    int32_t width = 0;
    int32_t height = 0;

    Rect2i() = default;
    Rect2i(int32_t x_, int32_t y_, int32_t w, int32_t h)
        : x(x_), y(y_), width(w), height(h) {}

    int32_t Right() const { return x + width; }
    int32_t Bottom() const { return y + height; }
    int64_t Area() const { return static_cast<int64_t>(width) * height; }
    Point2i Center() const { return {x + width / 2, y + height / 2}; }
    bool IsValid() const { return width >= 0 && height >= 0; }

    bool Contains(int32_t px, int32_t py) const {
        return px >= x && px < Right() && py >= y && py < Bottom();
    }

    bool Contains(const Point2i& p) const {
        return Contains(p.x, p.y);
    }
};

/**
 * @brief Axis-aligned rectangle with double precision
 */
struct QIVISION_API Rect2d {
    double x = 0.0;
    double y = 0.0;
    double width = 0.0;
    double height = 0.0;

    Rect2d() = default;
    Rect2d(double x_, double y_, double w, double h)
        : x(x_), y(y_), width(w), height(h) {}

    double Area() const { return width * height; }
    Point2d Center() const { return {x + width / 2.0, y + height / 2.0}; }
    bool IsValid() const {
        return std::isfinite(x) && std::isfinite(y) &&
               std::isfinite(width) && std::isfinite(height) &&
               width >= 0.0 && height >= 0.0;
    }
};

// =============================================================================
// Geometric Primitives (Forward Declarations)
// =============================================================================

struct Line2d;
struct Segment2d;
struct Circle2d;
struct Ellipse2d;
struct Arc2d;
struct RotatedRect2d;

// =============================================================================
// Line2d
// =============================================================================

/**
 * @brief 2D line in normalized form: ax + by + c = 0
 * @note Normalized such that a² + b² = 1
 */
struct QIVISION_API Line2d {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;

    Line2d() = default;
    Line2d(double a_, double b_, double c_);

    /// Create from two points
    static Line2d FromPoints(const Point2d& p1, const Point2d& p2);

    /// Create from point and angle
    static Line2d FromPointAngle(const Point2d& point, double angle);

    /// Angle of line direction (radians)
    double Angle() const { return std::atan2(-a, b); }

    /// Unit direction vector
    Point2d Direction() const { return {b, -a}; }

    /// Unit normal vector
    Point2d Normal() const { return {a, b}; }

    /// Signed distance from point to line
    double SignedDistance(const Point2d& p) const {
        return a * p.x + b * p.y + c;
    }

    /// Absolute distance from point to line
    double Distance(const Point2d& p) const {
        return std::abs(SignedDistance(p));
    }

    bool IsValid() const {
        return std::isfinite(a) && std::isfinite(b) && std::isfinite(c) &&
               (std::abs(a) + std::abs(b) > 0.0);
    }
};

// =============================================================================
// Circle2d
// =============================================================================

/**
 * @brief 2D circle
 */
struct QIVISION_API Circle2d {
    Point2d center;
    double radius = 0.0;

    Circle2d() = default;
    Circle2d(const Point2d& c, double r) : center(c), radius(r) {}
    Circle2d(double cx, double cy, double r) : center(cx, cy), radius(r) {}

    double Area() const;
    double Circumference() const;

    /// Check if point is inside circle
    bool Contains(const Point2d& p) const {
        return center.DistanceTo(p) <= radius;
    }

    bool IsValid() const {
        return center.IsValid() && std::isfinite(radius) && radius >= 0.0;
    }
};

// =============================================================================
// Segment2d
// =============================================================================

/**
 * @brief 2D line segment defined by two endpoints
 */
struct QIVISION_API Segment2d {
    Point2d p1;
    Point2d p2;

    Segment2d() = default;
    Segment2d(const Point2d& start, const Point2d& end) : p1(start), p2(end) {}
    Segment2d(double x1, double y1, double x2, double y2) : p1(x1, y1), p2(x2, y2) {}

    /// Segment length
    double Length() const { return p1.DistanceTo(p2); }

    /// Midpoint
    Point2d Midpoint() const { return (p1 + p2) * 0.5; }

    /// Direction vector (not normalized)
    Point2d Direction() const { return p2 - p1; }

    /// Unit direction vector
    Point2d UnitDirection() const {
        Point2d d = Direction();
        double len = d.Norm();
        return len > 0 ? d * (1.0 / len) : Point2d(1, 0);
    }

    /// Angle of segment (radians)
    double Angle() const { return std::atan2(p2.y - p1.y, p2.x - p1.x); }

    /// Convert to infinite line
    Line2d ToLine() const { return Line2d::FromPoints(p1, p2); }

    /// Distance from point to segment
    double DistanceToPoint(const Point2d& p) const;

    /// Project point onto segment (returns parameter t in [0,1])
    double ProjectPoint(const Point2d& p) const;

    /// Get point on segment at parameter t (0=p1, 1=p2)
    Point2d PointAt(double t) const {
        return p1 + Direction() * t;
    }

    bool IsValid() const { return p1.IsValid() && p2.IsValid(); }
};

// =============================================================================
// Ellipse2d
// =============================================================================

/**
 * @brief 2D ellipse
 */
struct QIVISION_API Ellipse2d {
    Point2d center;
    double a = 0.0;        ///< Semi-major axis
    double b = 0.0;        ///< Semi-minor axis
    double angle = 0.0;    ///< Rotation angle (radians)

    Ellipse2d() = default;
    Ellipse2d(const Point2d& c, double semiMajor, double semiMinor, double phi = 0.0)
        : center(c), a(semiMajor), b(semiMinor), angle(phi) {}
    Ellipse2d(double cx, double cy, double semiMajor, double semiMinor, double phi = 0.0)
        : center(cx, cy), a(semiMajor), b(semiMinor), angle(phi) {}

    /// Area of ellipse
    double Area() const;

    /// Perimeter (approximate using Ramanujan's formula)
    double Perimeter() const;

    /// Eccentricity
    double Eccentricity() const {
        if (a <= 0) return 0;
        double ratio = b / a;
        return std::sqrt(1.0 - ratio * ratio);
    }

    /// Check if point is inside ellipse
    bool Contains(const Point2d& p) const;

    /// Get point on ellipse at angle theta (in ellipse local coordinates)
    Point2d PointAt(double theta) const;

    bool IsValid() const {
        return center.IsValid() && std::isfinite(a) && std::isfinite(b) &&
               std::isfinite(angle) && a >= 0.0 && b >= 0.0;
    }
};

// =============================================================================
// Arc2d
// =============================================================================

/**
 * @brief 2D circular arc
 */
struct QIVISION_API Arc2d {
    Point2d center;
    double radius = 0.0;
    double startAngle = 0.0;    ///< Start angle (radians)
    double sweepAngle = 0.0;    ///< Sweep angle (radians), positive=CCW

    Arc2d() = default;
    Arc2d(const Point2d& c, double r, double start, double sweep)
        : center(c), radius(r), startAngle(start), sweepAngle(sweep) {}
    Arc2d(double cx, double cy, double r, double start, double sweep)
        : center(cx, cy), radius(r), startAngle(start), sweepAngle(sweep) {}

    /// End angle
    double EndAngle() const { return startAngle + sweepAngle; }

    /// Arc length
    double Length() const { return std::abs(sweepAngle) * radius; }

    /// Start point
    Point2d StartPoint() const {
        return {center.x + radius * std::cos(startAngle),
                center.y + radius * std::sin(startAngle)};
    }

    /// End point
    Point2d EndPoint() const {
        double endAngle = EndAngle();
        return {center.x + radius * std::cos(endAngle),
                center.y + radius * std::sin(endAngle)};
    }

    /// Midpoint on arc
    Point2d Midpoint() const {
        double midAngle = startAngle + sweepAngle * 0.5;
        return {center.x + radius * std::cos(midAngle),
                center.y + radius * std::sin(midAngle)};
    }

    /// Get point on arc at parameter t (0=start, 1=end)
    Point2d PointAt(double t) const {
        double angle = startAngle + sweepAngle * t;
        return {center.x + radius * std::cos(angle),
                center.y + radius * std::sin(angle)};
    }

    /// Convert to full circle
    Circle2d ToCircle() const { return Circle2d(center, radius); }

    bool IsValid() const {
        return center.IsValid() && std::isfinite(radius) &&
               std::isfinite(startAngle) && std::isfinite(sweepAngle) &&
               radius >= 0.0;
    }
};

// =============================================================================
// RotatedRect2d
// =============================================================================

/**
 * @brief 2D rotated rectangle
 */
struct QIVISION_API RotatedRect2d {
    Point2d center;
    double width = 0.0;     ///< Full width (along local X axis)
    double height = 0.0;    ///< Full height (along local Y axis)
    double angle = 0.0;     ///< Rotation angle (radians)

    RotatedRect2d() = default;
    RotatedRect2d(const Point2d& c, double w, double h, double phi = 0.0)
        : center(c), width(w), height(h), angle(phi) {}
    RotatedRect2d(double cx, double cy, double w, double h, double phi = 0.0)
        : center(cx, cy), width(w), height(h), angle(phi) {}

    /// Area
    double Area() const { return width * height; }

    /// Get the 4 corners (in order: top-left, top-right, bottom-right, bottom-left)
    void GetCorners(Point2d corners[4]) const;

    /// Get axis-aligned bounding box
    Rect2d BoundingBox() const;

    /// Check if point is inside
    bool Contains(const Point2d& p) const;

    bool IsValid() const {
        return center.IsValid() && std::isfinite(width) && std::isfinite(height) &&
               std::isfinite(angle) && width >= 0.0 && height >= 0.0;
    }
};

// =============================================================================
// Enumerations
// =============================================================================

/**
 * @brief Border handling types for filtering operations
 */
enum class BorderType {
    Replicate,      ///< aaa|abcd|ddd
    Reflect101,     ///< dcb|abcd|cba (default)
    Reflect,        ///< cba|abcd|dcb
    Wrap,           ///< bcd|abcd|abc
    Constant        ///< 000|abcd|000
};

/**
 * @brief Interpolation methods
 */
enum class InterpolationType {
    Nearest,        ///< Nearest neighbor
    Bilinear,       ///< Bilinear interpolation (default)
    Bicubic,        ///< Bicubic interpolation
    Lanczos,        ///< Lanczos resampling
    Area            ///< Area-based (for downsampling)
};

/**
 * @brief Edge polarity for edge detection
 */
enum class EdgePolarity {
    Positive,       ///< Dark to light transition
    Negative,       ///< Light to dark transition
    Both            ///< Both transitions
};

/**
 * @brief Edge detection mode
 */
enum class EdgeMode {
    Ridge,          ///< Bright line on dark background
    Valley,         ///< Dark line on bright background
    Both            ///< Both ridge and valley
};

/**
 * @brief Connectivity for region operations
 */
enum class Connectivity {
    Four,           ///< 4-connected neighbors
    Eight           ///< 8-connected neighbors
};

} // namespace Qi::Vision
