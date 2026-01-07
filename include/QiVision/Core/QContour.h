#pragma once

/**
 * @file QContour.h
 * @brief XLD (eXtended Line Description) contour for QiVision
 *
 * XLD contours are subpixel-accurate point sequences that represent
 * edges, lines, or object boundaries. Key features:
 * - Subpixel precision (double coordinates)
 * - Local attributes: amplitude, direction, curvature at each point
 * - Hierarchy support (parent/children for holes)
 * - Can be open or closed
 */

#include <QiVision/Core/Types.h>
#include <vector>
#include <memory>

namespace Qi::Vision {

// Forward declaration
class QContour;

/**
 * @brief Contour point with optional local attributes
 */
struct ContourPoint {
    double x = 0.0;           ///< X coordinate (subpixel)
    double y = 0.0;           ///< Y coordinate (subpixel)
    double amplitude = 0.0;   ///< Edge strength/gradient magnitude
    double direction = 0.0;   ///< Edge direction in radians [-π, π]
    double curvature = 0.0;   ///< Local curvature (1/radius)

    ContourPoint() = default;
    ContourPoint(double x_, double y_) : x(x_), y(y_) {}
    ContourPoint(double x_, double y_, double amp, double dir, double curv = 0.0)
        : x(x_), y(y_), amplitude(amp), direction(dir), curvature(curv) {}

    /// Convert to Point2d
    Point2d ToPoint2d() const { return {x, y}; }

    /// Distance to another point
    double DistanceTo(const ContourPoint& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }
};

/**
 * @brief XLD contour - subpixel-accurate point sequence
 *
 * Represents edges, lines, or boundaries extracted from images.
 * Supports hierarchy (parent/children) for representing holes in objects.
 */
class QContour {
public:
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Default constructor (empty contour)
    QContour();

    /// Construct from points (simple coordinates only)
    explicit QContour(const std::vector<Point2d>& points, bool closed = false);

    /// Construct from contour points (with attributes)
    explicit QContour(const std::vector<ContourPoint>& points, bool closed = false);

    /// Construct with reserved capacity
    explicit QContour(size_t capacity);

    // =========================================================================
    // Point Access
    // =========================================================================

    /// Number of points in the contour
    size_t Size() const { return points_.size(); }

    /// Check if contour is empty
    bool Empty() const { return points_.empty(); }

    /// Access point by index
    const ContourPoint& At(size_t index) const;
    ContourPoint& At(size_t index);

    /// Operator[] for const access
    const ContourPoint& operator[](size_t index) const { return points_[index]; }
    ContourPoint& operator[](size_t index) { return points_[index]; }

    /// Get point coordinate as Point2d
    Point2d GetPoint(size_t index) const;

    /// Get all points as Point2d vector
    std::vector<Point2d> GetPoints() const;

    /// Get all contour points with attributes
    const std::vector<ContourPoint>& GetContourPoints() const { return points_; }

    /// Get pointer to raw data (for algorithms)
    const ContourPoint* Data() const { return points_.data(); }
    ContourPoint* Data() { return points_.data(); }

    // =========================================================================
    // Point Modification
    // =========================================================================

    /// Add a point to the end
    void AddPoint(const Point2d& p);
    void AddPoint(double x, double y);
    void AddPoint(const ContourPoint& p);

    /// Insert a point at given index
    void InsertPoint(size_t index, const ContourPoint& p);

    /// Remove point at given index
    void RemovePoint(size_t index);

    /// Clear all points
    void Clear();

    /// Reserve capacity for points
    void Reserve(size_t capacity);

    /// Set points from Point2d vector
    void SetPoints(const std::vector<Point2d>& points);

    /// Set points from ContourPoint vector
    void SetPoints(const std::vector<ContourPoint>& points);

    // =========================================================================
    // Attributes
    // =========================================================================

    /// Is the contour closed?
    bool IsClosed() const { return closed_; }

    /// Set closed state
    void SetClosed(bool closed) { closed_ = closed; }

    /// Get amplitude at a point
    double GetAmplitude(size_t index) const;

    /// Get direction at a point (radians)
    double GetDirection(size_t index) const;

    /// Get curvature at a point
    double GetCurvature(size_t index) const;

    /// Set amplitude at a point
    void SetAmplitude(size_t index, double amplitude);

    /// Set direction at a point
    void SetDirection(size_t index, double direction);

    /// Set curvature at a point
    void SetCurvature(size_t index, double curvature);

    /// Get amplitude array
    std::vector<double> GetAmplitudes() const;

    /// Get direction array
    std::vector<double> GetDirections() const;

    /// Get curvature array
    std::vector<double> GetCurvatures() const;

    // =========================================================================
    // Hierarchy (for holes)
    // =========================================================================

    /// Get parent contour index (-1 if no parent)
    int32_t GetParent() const { return parent_; }

    /// Set parent contour index
    void SetParent(int32_t parent) { parent_ = parent; }

    /// Get children contour indices
    const std::vector<int32_t>& GetChildren() const { return children_; }

    /// Add a child contour index
    void AddChild(int32_t childIndex);

    /// Remove a child contour index
    void RemoveChild(int32_t childIndex);

    /// Clear children
    void ClearChildren();

    /// Has parent? (is this a hole?)
    bool HasParent() const { return parent_ >= 0; }

    /// Has children? (has holes?)
    bool HasChildren() const { return !children_.empty(); }

    // =========================================================================
    // Geometric Properties
    // =========================================================================

    /// Total contour length
    double Length() const;

    /// Area (only meaningful for closed contours, uses shoelace formula)
    double Area() const;

    /// Signed area (positive = counter-clockwise, negative = clockwise)
    double SignedArea() const;

    /// Centroid (center of mass for closed contour, geometric center for open)
    Point2d Centroid() const;

    /// Bounding box
    Rect2d BoundingBox() const;

    /// Circularity = 4π·Area/Perimeter² (1.0 for perfect circle)
    double Circularity() const;

    /// Check if contour is counter-clockwise (positive signed area)
    bool IsCounterClockwise() const { return SignedArea() > 0; }

    /// Reverse the point order
    void Reverse();

    // =========================================================================
    // Point Query
    // =========================================================================

    /// Get point at parameter t ∈ [0, 1] along the contour (interpolated)
    Point2d PointAt(double t) const;

    /// Get tangent direction at parameter t
    double TangentAt(double t) const;

    /// Get normal direction at parameter t
    double NormalAt(double t) const;

    /// Find nearest point on contour to given point
    /// Returns parameter t and distance
    void NearestPoint(const Point2d& p, double& t, double& distance) const;

    /// Distance from point to contour
    double DistanceToPoint(const Point2d& p) const;

    /// Check if a point is inside the contour (for closed contours)
    bool Contains(const Point2d& p) const;
    bool Contains(double x, double y) const;

    // =========================================================================
    // Contour Transformations
    // =========================================================================

    /// Translate the contour
    QContour Translate(double dx, double dy) const;
    QContour Translate(const Point2d& offset) const;

    /// Scale around center
    QContour Scale(double factor) const;
    QContour Scale(double sx, double sy) const;
    QContour Scale(double sx, double sy, const Point2d& center) const;

    /// Rotate around center
    QContour Rotate(double angle) const;
    QContour Rotate(double angle, const Point2d& center) const;

    /// Apply affine transformation matrix
    QContour Transform(const class QMatrix& matrix) const;

    /// Create a deep copy
    QContour Clone() const;

    // =========================================================================
    // Contour Processing
    // =========================================================================

    /// Smooth the contour (Gaussian smoothing)
    QContour Smooth(double sigma) const;

    /// Simplify the contour (Douglas-Peucker algorithm)
    QContour Simplify(double tolerance) const;

    /// Resample to have points at fixed intervals
    QContour Resample(double interval) const;

    /// Resample to have fixed number of points
    QContour ResampleCount(size_t count) const;

    /// Compute curvature at all points (updates internal curvature values)
    void ComputeCurvature(int windowSize = 5);

    /// Close the contour (connect last point to first)
    void Close();

    /// Open the contour (remove connection from last to first)
    void Open();

    // =========================================================================
    // Segment/Arc Extraction
    // =========================================================================

    /// Split into line segments based on curvature
    std::vector<Segment2d> ToSegments(double maxError = 1.0) const;

    /// Split into arcs based on curvature
    std::vector<Arc2d> ToArcs(double maxError = 1.0) const;

    // =========================================================================
    // Static Factory Methods
    // =========================================================================

    /// Create contour from line segment
    static QContour FromSegment(const Segment2d& segment, double interval = 1.0);

    /// Create contour from arc
    static QContour FromArc(const Arc2d& arc, double interval = 1.0);

    /// Create contour from circle
    static QContour FromCircle(const Circle2d& circle, size_t numPoints = 64);

    /// Create contour from ellipse
    static QContour FromEllipse(const Ellipse2d& ellipse, size_t numPoints = 64);

    /// Create contour from rectangle
    static QContour FromRectangle(const Rect2d& rect);

    /// Create contour from rotated rectangle
    static QContour FromRotatedRect(const RotatedRect2d& rect);

    /// Create contour from polygon vertices
    static QContour FromPolygon(const std::vector<Point2d>& vertices, bool closed = true);

private:
    std::vector<ContourPoint> points_;   ///< Contour points with attributes
    bool closed_ = false;                ///< Is the contour closed?

    // Hierarchy
    int32_t parent_ = -1;                ///< Parent contour index (-1 if none)
    std::vector<int32_t> children_;      ///< Child contour indices (holes)

    // Cached values (mutable for lazy computation)
    mutable double cachedLength_ = -1.0;
    mutable double cachedArea_ = -1.0;
    mutable Rect2d cachedBBox_;
    mutable bool bboxValid_ = false;

    /// Invalidate cached values (call when points change)
    void InvalidateCache();

    /// Compute length if not cached
    void EnsureLength() const;

    /// Compute bounding box if not cached
    void EnsureBBox() const;
};

// =============================================================================
// Type Aliases
// =============================================================================

/// Alias for XLD contour
using QXld = QContour;

} // namespace Qi::Vision
