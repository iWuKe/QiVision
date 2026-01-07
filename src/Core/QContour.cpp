#include <QiVision/Core/QContour.h>
#include <QiVision/Core/QMatrix.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Internal/ContourProcess.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <functional>

namespace Qi::Vision {

// =============================================================================
// Constructors
// =============================================================================

QContour::QContour() = default;

QContour::QContour(const std::vector<Point2d>& points, bool closed)
    : closed_(closed) {
    points_.reserve(points.size());
    for (const auto& p : points) {
        points_.emplace_back(p.x, p.y);
    }
}

QContour::QContour(const std::vector<ContourPoint>& points, bool closed)
    : points_(points), closed_(closed) {}

QContour::QContour(size_t capacity) {
    points_.reserve(capacity);
}

// =============================================================================
// Point Access
// =============================================================================

const ContourPoint& QContour::At(size_t index) const {
    if (index >= points_.size()) {
        throw std::out_of_range("Contour point index out of range");
    }
    return points_[index];
}

ContourPoint& QContour::At(size_t index) {
    if (index >= points_.size()) {
        throw std::out_of_range("Contour point index out of range");
    }
    InvalidateCache();
    return points_[index];
}

Point2d QContour::GetPoint(size_t index) const {
    return At(index).ToPoint2d();
}

std::vector<Point2d> QContour::GetPoints() const {
    std::vector<Point2d> result;
    result.reserve(points_.size());
    for (const auto& p : points_) {
        result.push_back(p.ToPoint2d());
    }
    return result;
}

// =============================================================================
// Point Modification
// =============================================================================

void QContour::AddPoint(const Point2d& p) {
    points_.emplace_back(p.x, p.y);
    InvalidateCache();
}

void QContour::AddPoint(double x, double y) {
    points_.emplace_back(x, y);
    InvalidateCache();
}

void QContour::AddPoint(const ContourPoint& p) {
    points_.push_back(p);
    InvalidateCache();
}

void QContour::InsertPoint(size_t index, const ContourPoint& p) {
    if (index > points_.size()) {
        throw std::out_of_range("Insert index out of range");
    }
    points_.insert(points_.begin() + static_cast<ptrdiff_t>(index), p);
    InvalidateCache();
}

void QContour::RemovePoint(size_t index) {
    if (index >= points_.size()) {
        throw std::out_of_range("Remove index out of range");
    }
    points_.erase(points_.begin() + static_cast<ptrdiff_t>(index));
    InvalidateCache();
}

void QContour::Clear() {
    points_.clear();
    InvalidateCache();
}

void QContour::Reserve(size_t capacity) {
    points_.reserve(capacity);
}

void QContour::SetPoints(const std::vector<Point2d>& points) {
    points_.clear();
    points_.reserve(points.size());
    for (const auto& p : points) {
        points_.emplace_back(p.x, p.y);
    }
    InvalidateCache();
}

void QContour::SetPoints(const std::vector<ContourPoint>& points) {
    points_ = points;
    InvalidateCache();
}

// =============================================================================
// Attributes
// =============================================================================

double QContour::GetAmplitude(size_t index) const {
    return At(index).amplitude;
}

double QContour::GetDirection(size_t index) const {
    return At(index).direction;
}

double QContour::GetCurvature(size_t index) const {
    return At(index).curvature;
}

void QContour::SetAmplitude(size_t index, double amplitude) {
    At(index).amplitude = amplitude;
}

void QContour::SetDirection(size_t index, double direction) {
    At(index).direction = direction;
}

void QContour::SetCurvature(size_t index, double curvature) {
    At(index).curvature = curvature;
}

std::vector<double> QContour::GetAmplitudes() const {
    std::vector<double> result;
    result.reserve(points_.size());
    for (const auto& p : points_) {
        result.push_back(p.amplitude);
    }
    return result;
}

std::vector<double> QContour::GetDirections() const {
    std::vector<double> result;
    result.reserve(points_.size());
    for (const auto& p : points_) {
        result.push_back(p.direction);
    }
    return result;
}

std::vector<double> QContour::GetCurvatures() const {
    std::vector<double> result;
    result.reserve(points_.size());
    for (const auto& p : points_) {
        result.push_back(p.curvature);
    }
    return result;
}

// =============================================================================
// Hierarchy
// =============================================================================

void QContour::AddChild(int32_t childIndex) {
    children_.push_back(childIndex);
}

void QContour::RemoveChild(int32_t childIndex) {
    auto it = std::find(children_.begin(), children_.end(), childIndex);
    if (it != children_.end()) {
        children_.erase(it);
    }
}

void QContour::ClearChildren() {
    children_.clear();
}

// =============================================================================
// Geometric Properties
// =============================================================================

double QContour::Length() const {
    EnsureLength();
    return cachedLength_;
}

double QContour::SignedArea() const {
    if (points_.size() < 3) {
        return 0.0;
    }

    // Shoelace formula
    double area = 0.0;
    size_t n = points_.size();

    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        area += points_[i].x * points_[j].y;
        area -= points_[j].x * points_[i].y;
    }

    return area * 0.5;
}

double QContour::Area() const {
    return std::abs(SignedArea());
}

Point2d QContour::Centroid() const {
    if (points_.empty()) {
        return {0.0, 0.0};
    }

    if (points_.size() == 1) {
        return points_[0].ToPoint2d();
    }

    if (!closed_ || points_.size() < 3) {
        // For open contours, just compute the geometric center
        double sumX = 0.0, sumY = 0.0;
        for (const auto& p : points_) {
            sumX += p.x;
            sumY += p.y;
        }
        return {sumX / static_cast<double>(points_.size()),
                sumY / static_cast<double>(points_.size())};
    }

    // For closed contours, compute the centroid of the polygon
    double cx = 0.0, cy = 0.0;
    double signedArea = 0.0;
    size_t n = points_.size();

    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        double cross = points_[i].x * points_[j].y - points_[j].x * points_[i].y;
        signedArea += cross;
        cx += (points_[i].x + points_[j].x) * cross;
        cy += (points_[i].y + points_[j].y) * cross;
    }

    signedArea *= 0.5;
    if (std::abs(signedArea) < EPSILON) {
        // Degenerate case
        double sumX = 0.0, sumY = 0.0;
        for (const auto& p : points_) {
            sumX += p.x;
            sumY += p.y;
        }
        return {sumX / static_cast<double>(n), sumY / static_cast<double>(n)};
    }

    double factor = 1.0 / (6.0 * signedArea);
    return {cx * factor, cy * factor};
}

Rect2d QContour::BoundingBox() const {
    EnsureBBox();
    return cachedBBox_;
}

double QContour::Circularity() const {
    if (!closed_ || points_.size() < 3) {
        return 0.0;
    }

    double area = Area();
    double perimeter = Length();

    if (perimeter < EPSILON) {
        return 0.0;
    }

    // Circularity = 4π·Area / Perimeter²
    return (4.0 * PI * area) / (perimeter * perimeter);
}

void QContour::Reverse() {
    std::reverse(points_.begin(), points_.end());
    // Directions should be flipped
    for (auto& p : points_) {
        p.direction = NormalizeAngle(p.direction + PI);
    }
    InvalidateCache();
}

// =============================================================================
// Point Query
// =============================================================================

Point2d QContour::PointAt(double t) const {
    if (points_.empty()) {
        return {0.0, 0.0};
    }

    if (points_.size() == 1) {
        return points_[0].ToPoint2d();
    }

    t = Clamp(t, 0.0, 1.0);
    double totalLen = Length();

    if (totalLen < EPSILON) {
        return points_[0].ToPoint2d();
    }

    double targetDist = t * totalLen;
    double accDist = 0.0;

    size_t n = points_.size();
    size_t segments = closed_ ? n : n - 1;

    for (size_t i = 0; i < segments; ++i) {
        size_t j = (i + 1) % n;
        double segLen = points_[i].DistanceTo(points_[j]);

        if (accDist + segLen >= targetDist) {
            // Interpolate on this segment
            double localT = (segLen > EPSILON) ? (targetDist - accDist) / segLen : 0.0;
            return {
                points_[i].x + localT * (points_[j].x - points_[i].x),
                points_[i].y + localT * (points_[j].y - points_[i].y)
            };
        }

        accDist += segLen;
    }

    // Should not reach here, return last point
    return points_.back().ToPoint2d();
}

double QContour::TangentAt(double t) const {
    if (points_.size() < 2) {
        return 0.0;
    }

    t = Clamp(t, 0.0, 1.0);
    double totalLen = Length();

    if (totalLen < EPSILON) {
        return 0.0;
    }

    double targetDist = t * totalLen;
    double accDist = 0.0;

    size_t n = points_.size();
    size_t segments = closed_ ? n : n - 1;

    for (size_t i = 0; i < segments; ++i) {
        size_t j = (i + 1) % n;
        double segLen = points_[i].DistanceTo(points_[j]);

        if (accDist + segLen >= targetDist || i == segments - 1) {
            // Return tangent of this segment
            double dx = points_[j].x - points_[i].x;
            double dy = points_[j].y - points_[i].y;
            return std::atan2(dy, dx);
        }

        accDist += segLen;
    }

    return 0.0;
}

double QContour::NormalAt(double t) const {
    return NormalizeAngle(TangentAt(t) + HALF_PI);
}

void QContour::NearestPoint(const Point2d& p, double& t, double& distance) const {
    if (points_.empty()) {
        t = 0.0;
        distance = 0.0;
        return;
    }

    if (points_.size() == 1) {
        t = 0.0;
        distance = p.DistanceTo(points_[0].ToPoint2d());
        return;
    }

    double totalLen = Length();
    double bestDist = std::numeric_limits<double>::max();
    double bestT = 0.0;
    double accDist = 0.0;

    size_t n = points_.size();
    size_t segments = closed_ ? n : n - 1;

    for (size_t i = 0; i < segments; ++i) {
        size_t j = (i + 1) % n;
        Point2d p1 = points_[i].ToPoint2d();
        Point2d p2 = points_[j].ToPoint2d();

        // Project p onto segment
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        double segLenSq = dx * dx + dy * dy;

        double localT = 0.0;
        if (segLenSq > EPSILON * EPSILON) {
            localT = ((p.x - p1.x) * dx + (p.y - p1.y) * dy) / segLenSq;
            localT = Clamp(localT, 0.0, 1.0);
        }

        Point2d closest{p1.x + localT * dx, p1.y + localT * dy};
        double dist = p.DistanceTo(closest);

        if (dist < bestDist) {
            bestDist = dist;
            double segLen = std::sqrt(segLenSq);
            bestT = (accDist + localT * segLen) / totalLen;
        }

        accDist += std::sqrt(segLenSq);
    }

    t = bestT;
    distance = bestDist;
}

double QContour::DistanceToPoint(const Point2d& p) const {
    double t, dist;
    NearestPoint(p, t, dist);
    return dist;
}

bool QContour::Contains(const Point2d& p) const {
    return Contains(p.x, p.y);
}

bool QContour::Contains(double x, double y) const {
    if (!closed_ || points_.size() < 3) {
        return false;
    }

    // Ray casting algorithm
    int crossings = 0;
    size_t n = points_.size();

    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;

        double yi = points_[i].y;
        double yj = points_[j].y;

        if ((yi <= y && yj > y) || (yj <= y && yi > y)) {
            double xi = points_[i].x;
            double xj = points_[j].x;

            double xIntersect = xi + (y - yi) / (yj - yi) * (xj - xi);
            if (x < xIntersect) {
                crossings++;
            }
        }
    }

    return (crossings % 2) == 1;
}

// =============================================================================
// Contour Transformations
// =============================================================================

QContour QContour::Translate(double dx, double dy) const {
    QContour result(*this);
    for (auto& p : result.points_) {
        p.x += dx;
        p.y += dy;
    }
    result.InvalidateCache();
    return result;
}

QContour QContour::Translate(const Point2d& offset) const {
    return Translate(offset.x, offset.y);
}

QContour QContour::Scale(double factor) const {
    return Scale(factor, factor);
}

QContour QContour::Scale(double sx, double sy) const {
    Point2d center = Centroid();
    return Scale(sx, sy, center);
}

QContour QContour::Scale(double sx, double sy, const Point2d& center) const {
    QContour result(*this);
    for (auto& p : result.points_) {
        p.x = center.x + (p.x - center.x) * sx;
        p.y = center.y + (p.y - center.y) * sy;
    }
    result.InvalidateCache();
    return result;
}

QContour QContour::Rotate(double angle) const {
    Point2d center = Centroid();
    return Rotate(angle, center);
}

QContour QContour::Rotate(double angle, const Point2d& center) const {
    double c = std::cos(angle);
    double s = std::sin(angle);

    QContour result(*this);
    for (auto& p : result.points_) {
        double dx = p.x - center.x;
        double dy = p.y - center.y;
        p.x = center.x + dx * c - dy * s;
        p.y = center.y + dx * s + dy * c;
        // Also rotate direction
        p.direction = NormalizeAngle(p.direction + angle);
    }
    result.InvalidateCache();
    return result;
}

QContour QContour::Transform(const QMatrix& matrix) const {
    QContour result(*this);
    for (auto& p : result.points_) {
        Point2d transformed = matrix.Transform(p.x, p.y);
        p.x = transformed.x;
        p.y = transformed.y;
        // Direction needs to be transformed by the linear part
        Point2d dir{std::cos(p.direction), std::sin(p.direction)};
        Point2d transformedDir = matrix.TransformVector(dir);
        p.direction = std::atan2(transformedDir.y, transformedDir.x);
    }
    result.InvalidateCache();
    return result;
}

QContour QContour::Clone() const {
    return QContour(*this);
}

// =============================================================================
// Contour Processing
// =============================================================================

QContour QContour::Smooth(double sigma) const {
    // Delegate to ContourProcess module
    return Internal::SmoothContourGaussian(*this, {sigma});
}

QContour QContour::Simplify(double tolerance) const {
    // Delegate to ContourProcess module
    return Internal::SimplifyContourDouglasPeucker(*this, {tolerance});
}

QContour QContour::Resample(double interval) const {
    // Delegate to ContourProcess module
    return Internal::ResampleContourByDistance(*this, {interval});
}

QContour QContour::ResampleCount(size_t count) const {
    // Delegate to ContourProcess module
    return Internal::ResampleContourByCount(*this, {count});
}

void QContour::ComputeCurvature(int windowSize) {
    if (points_.size() < 3) {
        return;
    }

    int halfWindow = windowSize / 2;
    size_t n = points_.size();

    for (size_t i = 0; i < n; ++i) {
        // Get points before and after
        size_t prevIdx, nextIdx;

        if (closed_) {
            prevIdx = (i + n - static_cast<size_t>(halfWindow)) % n;
            nextIdx = (i + static_cast<size_t>(halfWindow)) % n;
        } else {
            prevIdx = (i >= static_cast<size_t>(halfWindow)) ? i - static_cast<size_t>(halfWindow) : 0;
            nextIdx = (i + static_cast<size_t>(halfWindow) < n) ? i + static_cast<size_t>(halfWindow) : n - 1;
        }

        // Compute curvature using cross product method
        double x0 = points_[prevIdx].x, y0 = points_[prevIdx].y;
        double x1 = points_[i].x, y1 = points_[i].y;
        double x2 = points_[nextIdx].x, y2 = points_[nextIdx].y;

        double dx1 = x1 - x0, dy1 = y1 - y0;
        double dx2 = x2 - x1, dy2 = y2 - y1;

        double cross = dx1 * dy2 - dy1 * dx2;
        double len1 = std::sqrt(dx1 * dx1 + dy1 * dy1);
        double len2 = std::sqrt(dx2 * dx2 + dy2 * dy2);

        if (len1 > EPSILON && len2 > EPSILON) {
            // Curvature = 2 * cross / (len1 * len2 * chord)
            double chord = std::sqrt((x2 - x0) * (x2 - x0) + (y2 - y0) * (y2 - y0));
            if (chord > EPSILON) {
                points_[i].curvature = 2.0 * cross / (len1 * len2 * chord);
            } else {
                points_[i].curvature = 0.0;
            }
        } else {
            points_[i].curvature = 0.0;
        }
    }
}

void QContour::Close() {
    closed_ = true;
    InvalidateCache();
}

void QContour::Open() {
    closed_ = false;
    InvalidateCache();
}

// =============================================================================
// Segment/Arc Extraction
// =============================================================================

std::vector<Segment2d> QContour::ToSegments(double maxError) const {
    std::vector<Segment2d> segments;

    if (points_.size() < 2) {
        return segments;
    }

    // Use Douglas-Peucker simplification and then convert to segments
    QContour simplified = Simplify(maxError);

    size_t n = simplified.Size();
    size_t endIdx = closed_ ? n : n - 1;

    for (size_t i = 0; i < endIdx; ++i) {
        size_t j = (i + 1) % n;
        segments.emplace_back(simplified.GetPoint(i), simplified.GetPoint(j));
    }

    return segments;
}

std::vector<Arc2d> QContour::ToArcs(double maxError) const {
    // Simplified implementation: detect regions of similar curvature
    // and fit arcs to them
    std::vector<Arc2d> arcs;

    if (points_.size() < 3) {
        return arcs;
    }

    // This is a simplified implementation
    // A full implementation would segment based on curvature and fit arcs

    // For now, return empty (to be implemented with proper arc fitting)
    return arcs;
}

// =============================================================================
// Static Factory Methods
// =============================================================================

QContour QContour::FromSegment(const Segment2d& segment, double interval) {
    QContour result;

    double len = segment.Length();
    if (len < EPSILON) {
        result.AddPoint(segment.p1);
        return result;
    }

    size_t numPoints = static_cast<size_t>(std::ceil(len / interval)) + 1;

    for (size_t i = 0; i < numPoints; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(numPoints - 1);
        result.AddPoint(segment.PointAt(t));
    }

    return result;
}

QContour QContour::FromArc(const Arc2d& arc, double interval) {
    QContour result;

    double len = arc.Length();
    if (len < EPSILON) {
        result.AddPoint(arc.StartPoint());
        return result;
    }

    size_t numPoints = static_cast<size_t>(std::ceil(len / interval)) + 1;

    for (size_t i = 0; i < numPoints; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(numPoints - 1);
        double angle = arc.startAngle + t * arc.sweepAngle;
        result.AddPoint(arc.center.x + arc.radius * std::cos(angle),
                        arc.center.y + arc.radius * std::sin(angle));
    }

    return result;
}

QContour QContour::FromCircle(const Circle2d& circle, size_t numPoints) {
    QContour result;
    result.Reserve(numPoints);

    for (size_t i = 0; i < numPoints; ++i) {
        double angle = TWO_PI * static_cast<double>(i) / static_cast<double>(numPoints);
        result.AddPoint(circle.center.x + circle.radius * std::cos(angle),
                        circle.center.y + circle.radius * std::sin(angle));
    }

    result.SetClosed(true);
    return result;
}

QContour QContour::FromEllipse(const Ellipse2d& ellipse, size_t numPoints) {
    QContour result;
    result.Reserve(numPoints);

    double cosA = std::cos(ellipse.angle);
    double sinA = std::sin(ellipse.angle);

    for (size_t i = 0; i < numPoints; ++i) {
        double theta = TWO_PI * static_cast<double>(i) / static_cast<double>(numPoints);
        double localX = ellipse.a * std::cos(theta);
        double localY = ellipse.b * std::sin(theta);

        result.AddPoint(ellipse.center.x + localX * cosA - localY * sinA,
                        ellipse.center.y + localX * sinA + localY * cosA);
    }

    result.SetClosed(true);
    return result;
}

QContour QContour::FromRectangle(const Rect2d& rect) {
    QContour result;
    result.Reserve(4);

    result.AddPoint(rect.x, rect.y);
    result.AddPoint(rect.x + rect.width, rect.y);
    result.AddPoint(rect.x + rect.width, rect.y + rect.height);
    result.AddPoint(rect.x, rect.y + rect.height);

    result.SetClosed(true);
    return result;
}

QContour QContour::FromRotatedRect(const RotatedRect2d& rect) {
    QContour result;
    result.Reserve(4);

    Point2d corners[4];
    rect.GetCorners(corners);

    for (int i = 0; i < 4; ++i) {
        result.AddPoint(corners[i]);
    }

    result.SetClosed(true);
    return result;
}

QContour QContour::FromPolygon(const std::vector<Point2d>& vertices, bool closed) {
    return QContour(vertices, closed);
}

// =============================================================================
// Cache Management
// =============================================================================

void QContour::InvalidateCache() {
    cachedLength_ = -1.0;
    cachedArea_ = -1.0;
    bboxValid_ = false;
}

void QContour::EnsureLength() const {
    if (cachedLength_ >= 0.0) {
        return;
    }

    cachedLength_ = 0.0;

    if (points_.size() < 2) {
        return;
    }

    size_t n = points_.size();
    size_t segments = closed_ ? n : n - 1;

    for (size_t i = 0; i < segments; ++i) {
        size_t j = (i + 1) % n;
        cachedLength_ += points_[i].DistanceTo(points_[j]);
    }
}

void QContour::EnsureBBox() const {
    if (bboxValid_) {
        return;
    }

    if (points_.empty()) {
        cachedBBox_ = Rect2d(0, 0, 0, 0);
        bboxValid_ = true;
        return;
    }

    double minX = points_[0].x, maxX = points_[0].x;
    double minY = points_[0].y, maxY = points_[0].y;

    for (size_t i = 1; i < points_.size(); ++i) {
        minX = std::min(minX, points_[i].x);
        maxX = std::max(maxX, points_[i].x);
        minY = std::min(minY, points_[i].y);
        maxY = std::max(maxY, points_[i].y);
    }

    cachedBBox_ = Rect2d(minX, minY, maxX - minX, maxY - minY);
    bboxValid_ = true;
}

} // namespace Qi::Vision
