/**
 * @file MeasureHandle.cpp
 * @brief Implementation of measurement handles
 */

#include <QiVision/Measure/MeasureHandle.h>
#include <QiVision/Core/Exception.h>

#include <algorithm>
#include <cmath>

namespace Qi::Vision::Measure {

namespace {
    constexpr double PI = 3.14159265358979323846;
    constexpr double TWO_PI = 2.0 * PI;

    // Normalize angle to [0, 2*PI)
    double NormalizeAngle(double angle) {
        while (angle < 0) angle += TWO_PI;
        while (angle >= TWO_PI) angle -= TWO_PI;
        return angle;
    }

    // Normalize angle to [-PI, PI)
    double NormalizeAnglePi(double angle) {
        while (angle < -PI) angle += TWO_PI;
        while (angle >= PI) angle -= TWO_PI;
        return angle;
    }
}

// =============================================================================
// MeasureRectangle2 Implementation
// =============================================================================

MeasureRectangle2::MeasureRectangle2() = default;

MeasureRectangle2::MeasureRectangle2(double row, double column,
                                      double phi, double length1, double length2,
                                      int32_t width, int32_t height,
                                      const std::string& interpolation)
    : row_(row)
    , column_(column)
    , phi_(phi)
    , length1_(length1)
    , length2_(length2)
    , imageWidth_(width)
    , imageHeight_(height)
    , interpolation_(interpolation) {
    if (!std::isfinite(row_) || !std::isfinite(column_) || !std::isfinite(phi_) ||
        !std::isfinite(length1_) || !std::isfinite(length2_) ||
        length1_ <= 0.0 || length2_ < 0.0 || imageWidth_ < 0 || imageHeight_ < 0) {
        throw InvalidArgumentException("MeasureRectangle2: invalid parameters");
    }
    // Compute numLines based on length2 (half-width)
    numLines_ = std::max(1, static_cast<int32_t>(2.0 * length2_));
    samplesPerPixel_ = 1.0;
    ComputeSamplingGeometry();
}

MeasureRectangle2 MeasureRectangle2::FromRotatedRect(const RotatedRect2d& rect,
                                                      int32_t /*numLines*/) {
    if (!rect.IsValid() || rect.width <= 0.0 || rect.height <= 0.0) {
        throw InvalidArgumentException("MeasureRectangle2::FromRotatedRect: invalid rect");
    }
    // RotatedRect: center, width, height, angle
    // Measurement direction is along the longer axis
    double length1 = std::max(rect.width, rect.height) / 2.0;  // Half-length
    double length2 = std::min(rect.width, rect.height) / 2.0;  // Half-width
    double phi = rect.angle;

    // If width is the longer dimension, rotate phi by 90 degrees
    if (rect.width > rect.height) {
        phi += PI / 2.0;
    }

    return MeasureRectangle2(rect.center.y, rect.center.x,
                              phi, length1, length2);
}

MeasureRectangle2 MeasureRectangle2::FromPoints(const Point2d& p1, const Point2d& p2,
                                                 double halfWidth, int32_t /*numLines*/) {
    if (!p1.IsValid() || !p2.IsValid()) {
        throw InvalidArgumentException("MeasureRectangle2::FromPoints: invalid points");
    }
    if (halfWidth <= 0.0) {
        throw InvalidArgumentException("MeasureRectangle2::FromPoints: halfWidth must be > 0");
    }
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double length1 = std::sqrt(dx * dx + dy * dy) / 2.0;  // Half-length
    if (length1 <= 0.0) {
        throw InvalidArgumentException("MeasureRectangle2::FromPoints: points must be distinct");
    }
    double angle = std::atan2(dy, dx);

    // Center is midpoint
    double centerCol = (p1.x + p2.x) / 2.0;
    double centerRow = (p1.y + p2.y) / 2.0;

    // phi is perpendicular to the line direction
    double phi = angle - PI / 2.0;

    return MeasureRectangle2(centerRow, centerCol, phi, length1, halfWidth);
}

bool MeasureRectangle2::IsValid() const {
    return length1_ > 0 && length2_ >= 0 && numLines_ > 0;
}

double MeasureRectangle2::ProfileAngle() const {
    // Profile direction is phi + 90 degrees
    return phi_ + PI / 2.0;
}

Rect2d MeasureRectangle2::BoundingBox() const {
    // Get corners of the rotated rectangle
    double cosP = std::cos(phi_);
    double sinP = std::sin(phi_);

    // Profile direction (perpendicular to phi)
    double profileCos = std::cos(ProfileAngle());
    double profileSin = std::sin(ProfileAngle());

    // Four corners (using half-lengths)
    double dx1 = length1_ * profileCos - length2_ * cosP;
    double dy1 = length1_ * profileSin - length2_ * sinP;
    double dx2 = length1_ * profileCos + length2_ * cosP;
    double dy2 = length1_ * profileSin + length2_ * sinP;
    double dx3 = -length1_ * profileCos - length2_ * cosP;
    double dy3 = -length1_ * profileSin - length2_ * sinP;
    double dx4 = -length1_ * profileCos + length2_ * cosP;
    double dy4 = -length1_ * profileSin + length2_ * sinP;

    double minX = std::min({column_ + dx1, column_ + dx2,
                           column_ + dx3, column_ + dx4});
    double maxX = std::max({column_ + dx1, column_ + dx2,
                           column_ + dx3, column_ + dx4});
    double minY = std::min({row_ + dy1, row_ + dy2,
                           row_ + dy3, row_ + dy4});
    double maxY = std::max({row_ + dy1, row_ + dy2,
                           row_ + dy3, row_ + dy4});

    return Rect2d{minX, minY, maxX - minX, maxY - minY};
}

bool MeasureRectangle2::Contains(const Point2d& point) const {
    if (!point.IsValid()) {
        throw InvalidArgumentException("MeasureRectangle2::Contains: invalid point");
    }
    // Transform point to rectangle-local coordinates
    double dx = point.x - column_;
    double dy = point.y - row_;

    double cosP = std::cos(phi_);
    double sinP = std::sin(phi_);

    // Profile direction components
    double profileCos = std::cos(ProfileAngle());
    double profileSin = std::sin(ProfileAngle());

    // Project onto rectangle axes
    double projProfile = dx * profileCos + dy * profileSin;
    double projWidth = dx * cosP + dy * sinP;

    return std::abs(projProfile) <= length1_ &&
           std::abs(projWidth) <= length2_;
}

RotatedRect2d MeasureRectangle2::ToRotatedRect() const {
    return RotatedRect2d(
        Point2d{column_, row_},
        2.0 * length2_,   // width (full)
        2.0 * length1_,   // height (full length)
        phi_
    );
}

void MeasureRectangle2::GetProfileEndpoints(Point2d& start, Point2d& end) const {
    double profileCos = std::cos(ProfileAngle());
    double profileSin = std::sin(ProfileAngle());

    start.x = column_ - length1_ * profileCos;
    start.y = row_ - length1_ * profileSin;
    end.x = column_ + length1_ * profileCos;
    end.y = row_ + length1_ * profileSin;
}

std::vector<Segment2d> MeasureRectangle2::GetSamplingLines() const {
    std::vector<Segment2d> lines;
    lines.reserve(numLines_);

    Point2d start, end;
    GetProfileEndpoints(start, end);

    double cosP = std::cos(phi_);
    double sinP = std::sin(phi_);

    for (int32_t i = 0; i < numLines_; ++i) {
        double offset = lineOffsets_[i];
        Segment2d seg;
        seg.p1.x = start.x + offset * cosP;
        seg.p1.y = start.y + offset * sinP;
        seg.p2.x = end.x + offset * cosP;
        seg.p2.y = end.y + offset * sinP;
        lines.push_back(seg);
    }

    return lines;
}

void MeasureRectangle2::ComputeSamplingGeometry() {
    lineOffsets_.clear();
    lineOffsets_.reserve(numLines_);

    if (numLines_ == 1) {
        lineOffsets_.push_back(0.0);
    } else {
        // Spread lines across full width (2 * length2_)
        double step = 2.0 * length2_ / (numLines_ - 1);
        for (int32_t i = 0; i < numLines_; ++i) {
            lineOffsets_.push_back(-length2_ + i * step);
        }
    }
}

void MeasureRectangle2::Translate(double deltaRow, double deltaCol) {
    if (!std::isfinite(deltaRow) || !std::isfinite(deltaCol)) {
        throw InvalidArgumentException("MeasureRectangle2::Translate: invalid offset");
    }
    row_ += deltaRow;
    column_ += deltaCol;
    // No need to recompute sampling geometry - offsets are relative
}

void MeasureRectangle2::SetPosition(double row, double column) {
    if (!std::isfinite(row) || !std::isfinite(column)) {
        throw InvalidArgumentException("MeasureRectangle2::SetPosition: invalid position");
    }
    row_ = row;
    column_ = column;
}

// =============================================================================
// MeasureArc Implementation
// =============================================================================

MeasureArc::MeasureArc() = default;

MeasureArc::MeasureArc(double centerRow, double centerCol,
                        double radius, double angleStart, double angleExtent,
                        double annulusRadius, int32_t width, int32_t height,
                        const std::string& interpolation)
    : centerRow_(centerRow)
    , centerCol_(centerCol)
    , radius_(radius)
    , angleStart_(angleStart)
    , angleExtent_(angleExtent)
    , annulusRadius_(annulusRadius) {
    if (!std::isfinite(centerRow_) || !std::isfinite(centerCol_) ||
        !std::isfinite(radius_) || !std::isfinite(angleStart_) ||
        !std::isfinite(angleExtent_) || !std::isfinite(annulusRadius_) ||
        radius_ <= 0.0 || annulusRadius_ < 0.0 || std::abs(angleExtent_) <= 0.0) {
        throw InvalidArgumentException("MeasureArc: invalid parameters");
    }
    (void)width;   // Reserved for future use
    (void)height;
    (void)interpolation;
    // Compute numLines based on annulusRadius
    numLines_ = std::max(1, static_cast<int32_t>(2.0 * annulusRadius_));
    if (numLines_ < 1) numLines_ = 1;
    samplesPerPixel_ = 1.0;
    ComputeSamplingGeometry();
}

MeasureArc MeasureArc::FromArc(const Arc2d& arc, double annulusRadius, int32_t /*numLines*/) {
    if (!arc.IsValid() || arc.radius <= 0.0 || !std::isfinite(annulusRadius) || annulusRadius < 0.0 ||
        std::abs(arc.sweepAngle) <= 0.0) {
        throw InvalidArgumentException("MeasureArc::FromArc: invalid arc");
    }
    return MeasureArc(arc.center.y, arc.center.x,
                       arc.radius, arc.startAngle, arc.sweepAngle,
                       annulusRadius);
}

MeasureArc MeasureArc::FromCircle(const Circle2d& circle, double annulusRadius, int32_t /*numLines*/) {
    if (!circle.IsValid() || circle.radius <= 0.0 ||
        !std::isfinite(annulusRadius) || annulusRadius < 0.0) {
        throw InvalidArgumentException("MeasureArc::FromCircle: invalid circle");
    }
    return MeasureArc(circle.center.y, circle.center.x,
                       circle.radius, 0.0, TWO_PI,
                       annulusRadius);
}

bool MeasureArc::IsValid() const {
    return radius_ > 0 && std::abs(angleExtent_) > 0 && numLines_ > 0;
}

double MeasureArc::ProfileLength() const {
    return std::abs(angleExtent_) * radius_;
}

Rect2d MeasureArc::BoundingBox() const {
    // Sample arc to find bounding box
    double minX = centerCol_, maxX = centerCol_;
    double minY = centerRow_, maxY = centerRow_;

    double rOuter = radius_ + annulusRadius_;
    double rInner = radius_ - annulusRadius_;
    if (rInner < 0) rInner = 0;

    // Sample points along the arc
    int numSamples = std::max(10, static_cast<int>(std::abs(angleExtent_) * 10));
    for (int i = 0; i <= numSamples; ++i) {
        double t = static_cast<double>(i) / numSamples;
        double angle = angleStart_ + t * angleExtent_;
        double cosA = std::cos(angle);
        double sinA = std::sin(angle);

        // Check both inner and outer radius
        for (double r : {rInner, rOuter}) {
            double x = centerCol_ + r * cosA;
            double y = centerRow_ + r * sinA;
            minX = std::min(minX, x);
            maxX = std::max(maxX, x);
            minY = std::min(minY, y);
            maxY = std::max(maxY, y);
        }
    }

    return Rect2d{minX, minY, maxX - minX, maxY - minY};
}

bool MeasureArc::Contains(const Point2d& point) const {
    if (!point.IsValid()) {
        throw InvalidArgumentException("MeasureArc::Contains: invalid point");
    }
    double dx = point.x - centerCol_;
    double dy = point.y - centerRow_;
    double dist = std::sqrt(dx * dx + dy * dy);

    // Check radial extent
    double rMin = radius_ - annulusRadius_;
    double rMax = radius_ + annulusRadius_;
    if (dist < rMin || dist > rMax) return false;

    // Check angular extent
    double angle = std::atan2(dy, dx);
    double startNorm = NormalizeAngle(angleStart_);
    double endNorm = NormalizeAngle(angleStart_ + angleExtent_);
    double angleNorm = NormalizeAngle(angle);

    if (angleExtent_ >= 0) {
        if (startNorm <= endNorm) {
            return angleNorm >= startNorm && angleNorm <= endNorm;
        } else {
            return angleNorm >= startNorm || angleNorm <= endNorm;
        }
    } else {
        if (endNorm <= startNorm) {
            return angleNorm >= endNorm && angleNorm <= startNorm;
        } else {
            return angleNorm >= endNorm || angleNorm <= startNorm;
        }
    }
}

Arc2d MeasureArc::ToArc() const {
    return Arc2d{
        Point2d{centerCol_, centerRow_},
        radius_,
        angleStart_,
        angleExtent_
    };
}

Point2d MeasureArc::PointAt(double t) const {
    if (!std::isfinite(t)) {
        throw InvalidArgumentException("MeasureArc::PointAt: t must be finite");
    }
    double angle = angleStart_ + t * angleExtent_;
    return Point2d{
        centerCol_ + radius_ * std::cos(angle),
        centerRow_ + radius_ * std::sin(angle)
    };
}

double MeasureArc::TangentAt(double t) const {
    double angle = angleStart_ + t * angleExtent_;
    // Tangent is perpendicular to radial direction
    // For CCW arc, tangent is angle + 90 degrees
    if (angleExtent_ >= 0) {
        return angle + PI / 2.0;
    } else {
        return angle - PI / 2.0;
    }
}

double MeasureArc::ProfilePosToAngle(double pos) const {
    if (!std::isfinite(pos)) {
        throw InvalidArgumentException("MeasureArc::ProfilePosToAngle: pos must be finite");
    }
    double arcLen = ProfileLength();
    if (arcLen < 1e-10) return angleStart_;
    double t = pos / arcLen;
    return angleStart_ + t * angleExtent_;
}

double MeasureArc::AngleToProfilePos(double angle) const {
    if (!std::isfinite(angle)) {
        throw InvalidArgumentException("MeasureArc::AngleToProfilePos: angle must be finite");
    }
    double deltaAngle = angle - angleStart_;
    return std::abs(deltaAngle) * radius_;
}

void MeasureArc::ComputeSamplingGeometry() {
    radiusOffsets_.clear();
    radiusOffsets_.reserve(numLines_);

    if (numLines_ == 1 || annulusRadius_ <= 0) {
        radiusOffsets_.push_back(0.0);
    } else {
        double step = 2.0 * annulusRadius_ / (numLines_ - 1);
        for (int32_t i = 0; i < numLines_; ++i) {
            radiusOffsets_.push_back(-annulusRadius_ + i * step);
        }
    }
}

void MeasureArc::Translate(double deltaRow, double deltaCol) {
    if (!std::isfinite(deltaRow) || !std::isfinite(deltaCol)) {
        throw InvalidArgumentException("MeasureArc::Translate: invalid offset");
    }
    centerRow_ += deltaRow;
    centerCol_ += deltaCol;
}

void MeasureArc::SetPosition(double centerRow, double centerCol) {
    if (!std::isfinite(centerRow) || !std::isfinite(centerCol)) {
        throw InvalidArgumentException("MeasureArc::SetPosition: invalid position");
    }
    centerRow_ = centerRow;
    centerCol_ = centerCol;
}

// =============================================================================
// MeasureConcentricCircles Implementation
// =============================================================================

MeasureConcentricCircles::MeasureConcentricCircles() = default;

MeasureConcentricCircles::MeasureConcentricCircles(double centerRow, double centerCol,
                                                    double innerRadius, double outerRadius,
                                                    double angle, double angularWidth,
                                                    int32_t numLines, double samplesPerPixel)
    : centerRow_(centerRow)
    , centerCol_(centerCol)
    , innerRadius_(innerRadius)
    , outerRadius_(outerRadius)
    , angle_(angle)
    , angularWidth_(angularWidth) {
    if (!std::isfinite(centerRow_) || !std::isfinite(centerCol_) ||
        !std::isfinite(innerRadius_) || !std::isfinite(outerRadius_) ||
        !std::isfinite(angle_) || !std::isfinite(angularWidth_) ||
        innerRadius_ < 0.0 || outerRadius_ <= innerRadius_ || angularWidth_ <= 0.0 ||
        !std::isfinite(samplesPerPixel) || samplesPerPixel <= 0.0) {
        throw InvalidArgumentException("MeasureConcentricCircles: invalid parameters");
    }
    numLines_ = (numLines > 0) ? numLines : std::max(1, static_cast<int32_t>(angularWidth_ * 10));
    samplesPerPixel_ = samplesPerPixel;
    ComputeSamplingGeometry();
}

bool MeasureConcentricCircles::IsValid() const {
    return outerRadius_ > innerRadius_ && innerRadius_ >= 0 && numLines_ > 0;
}

Rect2d MeasureConcentricCircles::BoundingBox() const {
    // The bounding box covers the angular sector
    double halfAng = angularWidth_ / 2.0;
    double minX = centerCol_, maxX = centerCol_;
    double minY = centerRow_, maxY = centerRow_;

    // Sample the sector
    int numSamples = 10;
    for (int i = 0; i <= numSamples; ++i) {
        double t = static_cast<double>(i) / numSamples;
        double ang = angle_ - halfAng + t * angularWidth_;
        double cosA = std::cos(ang);
        double sinA = std::sin(ang);

        for (double r : {innerRadius_, outerRadius_}) {
            double x = centerCol_ + r * cosA;
            double y = centerRow_ + r * sinA;
            minX = std::min(minX, x);
            maxX = std::max(maxX, x);
            minY = std::min(minY, y);
            maxY = std::max(maxY, y);
        }
    }

    return Rect2d{minX, minY, maxX - minX, maxY - minY};
}

bool MeasureConcentricCircles::Contains(const Point2d& point) const {
    if (!point.IsValid()) {
        throw InvalidArgumentException("MeasureConcentricCircles::Contains: invalid point");
    }
    double dx = point.x - centerCol_;
    double dy = point.y - centerRow_;
    double dist = std::sqrt(dx * dx + dy * dy);

    // Check radial extent
    if (dist < innerRadius_ || dist > outerRadius_) return false;

    // Check angular extent
    double pointAngle = std::atan2(dy, dx);
    double angleDiff = NormalizeAnglePi(pointAngle - angle_);
    return std::abs(angleDiff) <= angularWidth_ / 2.0;
}

Point2d MeasureConcentricCircles::PointAtRadius(double radius) const {
    if (!std::isfinite(radius) || radius < 0.0) {
        throw InvalidArgumentException("MeasureConcentricCircles::PointAtRadius: invalid radius");
    }
    return Point2d{
        centerCol_ + radius * std::cos(angle_),
        centerRow_ + radius * std::sin(angle_)
    };
}

double MeasureConcentricCircles::ProfilePosToRadius(double pos) const {
    return innerRadius_ + pos;
}

void MeasureConcentricCircles::ComputeSamplingGeometry() {
    angleOffsets_.clear();
    angleOffsets_.reserve(numLines_);

    if (numLines_ == 1 || angularWidth_ <= 0) {
        angleOffsets_.push_back(0.0);
    } else {
        double step = angularWidth_ / (numLines_ - 1);
        for (int32_t i = 0; i < numLines_; ++i) {
            angleOffsets_.push_back(-angularWidth_ / 2.0 + i * step);
        }
    }
}

void MeasureConcentricCircles::Translate(double deltaRow, double deltaCol) {
    if (!std::isfinite(deltaRow) || !std::isfinite(deltaCol)) {
        throw InvalidArgumentException("MeasureConcentricCircles::Translate: invalid offset");
    }
    centerRow_ += deltaRow;
    centerCol_ += deltaCol;
}

void MeasureConcentricCircles::SetPosition(double centerRow, double centerCol) {
    if (!std::isfinite(centerRow) || !std::isfinite(centerCol)) {
        throw InvalidArgumentException("MeasureConcentricCircles::SetPosition: invalid position");
    }
    centerRow_ = centerRow;
    centerCol_ = centerCol;
}

// =============================================================================
// Factory Functions (Halcon compatible)
// =============================================================================

MeasureRectangle2 GenMeasureRectangle2(double row, double column,
                                        double phi, double length1, double length2,
                                        int32_t width, int32_t height,
                                        const std::string& interpolation) {
    return MeasureRectangle2(row, column, phi, length1, length2,
                              width, height, interpolation);
}

MeasureArc GenMeasureArc(double centerRow, double centerCol,
                          double radius, double angleStart, double angleExtent,
                          double annulusRadius, int32_t width, int32_t height,
                          const std::string& interpolation) {
    return MeasureArc(centerRow, centerCol, radius, angleStart, angleExtent,
                       annulusRadius, width, height, interpolation);
}

MeasureConcentricCircles CreateMeasureConcentric(double centerRow, double centerCol,
                                                   double innerRadius, double outerRadius,
                                                   double angle, double angularWidth,
                                                   int32_t width, int32_t height,
                                                   const std::string& interpolation) {
    // Note: width, height, interpolation are ignored for concentric circles
    (void)width;
    (void)height;
    (void)interpolation;
    return MeasureConcentricCircles(centerRow, centerCol, innerRadius, outerRadius,
                                     angle, angularWidth);
}

MeasureRectangle2 CreateMeasureFromSegment(const Point2d& p1, const Point2d& p2,
                                            double halfWidth) {
    return MeasureRectangle2::FromPoints(p1, p2, halfWidth, 0);
}

MeasureRectangle2 CreateMeasureFromRect(const RotatedRect2d& rect) {
    return MeasureRectangle2::FromRotatedRect(rect, 0);
}

} // namespace Qi::Vision::Measure
