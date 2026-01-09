/**
 * @file MeasureHandle.cpp
 * @brief Implementation of measurement handles
 */

#include <QiVision/Measure/MeasureHandle.h>

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

MeasureRectangle2::MeasureRectangle2(double centerRow, double centerCol,
                                      double phi, double length, double width,
                                      int32_t numLines, double samplesPerPixel)
    : centerRow_(centerRow)
    , centerCol_(centerCol)
    , phi_(phi)
    , length_(length)
    , width_(width) {
    numLines_ = std::max(1, numLines);
    samplesPerPixel_ = std::max(0.1, samplesPerPixel);
    ComputeSamplingGeometry();
}

MeasureRectangle2 MeasureRectangle2::FromRotatedRect(const RotatedRect2d& rect,
                                                      int32_t numLines) {
    // RotatedRect: center, width, height, angle
    // Measurement direction is along the longer axis
    double length = std::max(rect.width, rect.height);
    double width = std::min(rect.width, rect.height);
    double phi = rect.angle;

    // If width is the longer dimension, rotate phi by 90 degrees
    if (rect.width > rect.height) {
        phi += PI / 2.0;
    }

    return MeasureRectangle2(rect.center.y, rect.center.x,
                              phi, length, width, numLines);
}

MeasureRectangle2 MeasureRectangle2::FromPoints(const Point2d& p1, const Point2d& p2,
                                                 double width, int32_t numLines) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double length = std::sqrt(dx * dx + dy * dy);
    double angle = std::atan2(dy, dx);

    // Center is midpoint
    double centerCol = (p1.x + p2.x) / 2.0;
    double centerRow = (p1.y + p2.y) / 2.0;

    // phi is perpendicular to the line direction
    double phi = angle - PI / 2.0;

    return MeasureRectangle2(centerRow, centerCol, phi, length, width, numLines);
}

bool MeasureRectangle2::IsValid() const {
    return length_ > 0 && width_ >= 0 && numLines_ > 0;
}

double MeasureRectangle2::ProfileAngle() const {
    // Profile direction is phi + 90 degrees
    return phi_ + PI / 2.0;
}

Rect2d MeasureRectangle2::BoundingBox() const {
    // Get corners of the rotated rectangle
    double cosP = std::cos(phi_);
    double sinP = std::sin(phi_);

    double halfL = length_ / 2.0;
    double halfW = width_ / 2.0;

    // Profile direction (perpendicular to phi)
    double profileCos = std::cos(ProfileAngle());
    double profileSin = std::sin(ProfileAngle());

    // Four corners
    double dx1 = halfL * profileCos - halfW * cosP;
    double dy1 = halfL * profileSin - halfW * sinP;
    double dx2 = halfL * profileCos + halfW * cosP;
    double dy2 = halfL * profileSin + halfW * sinP;
    double dx3 = -halfL * profileCos - halfW * cosP;
    double dy3 = -halfL * profileSin - halfW * sinP;
    double dx4 = -halfL * profileCos + halfW * cosP;
    double dy4 = -halfL * profileSin + halfW * sinP;

    double minX = std::min({centerCol_ + dx1, centerCol_ + dx2,
                           centerCol_ + dx3, centerCol_ + dx4});
    double maxX = std::max({centerCol_ + dx1, centerCol_ + dx2,
                           centerCol_ + dx3, centerCol_ + dx4});
    double minY = std::min({centerRow_ + dy1, centerRow_ + dy2,
                           centerRow_ + dy3, centerRow_ + dy4});
    double maxY = std::max({centerRow_ + dy1, centerRow_ + dy2,
                           centerRow_ + dy3, centerRow_ + dy4});

    return Rect2d{minX, minY, maxX - minX, maxY - minY};
}

bool MeasureRectangle2::Contains(const Point2d& point) const {
    // Transform point to rectangle-local coordinates
    double dx = point.x - centerCol_;
    double dy = point.y - centerRow_;

    double cosP = std::cos(phi_);
    double sinP = std::sin(phi_);

    // Profile direction components
    double profileCos = std::cos(ProfileAngle());
    double profileSin = std::sin(ProfileAngle());

    // Project onto rectangle axes
    double projProfile = dx * profileCos + dy * profileSin;
    double projWidth = dx * cosP + dy * sinP;

    return std::abs(projProfile) <= length_ / 2.0 &&
           std::abs(projWidth) <= width_ / 2.0;
}

RotatedRect2d MeasureRectangle2::ToRotatedRect() const {
    return RotatedRect2d(
        Point2d{centerCol_, centerRow_},
        width_,   // width
        length_,  // height (length)
        phi_
    );
}

void MeasureRectangle2::GetProfileEndpoints(Point2d& start, Point2d& end) const {
    double halfL = length_ / 2.0;
    double profileCos = std::cos(ProfileAngle());
    double profileSin = std::sin(ProfileAngle());

    start.x = centerCol_ - halfL * profileCos;
    start.y = centerRow_ - halfL * profileSin;
    end.x = centerCol_ + halfL * profileCos;
    end.y = centerRow_ + halfL * profileSin;
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
        double halfWidth = width_ / 2.0;
        double step = width_ / (numLines_ - 1);
        for (int32_t i = 0; i < numLines_; ++i) {
            lineOffsets_.push_back(-halfWidth + i * step);
        }
    }
}

// =============================================================================
// MeasureArc Implementation
// =============================================================================

MeasureArc::MeasureArc() = default;

MeasureArc::MeasureArc(double centerRow, double centerCol,
                        double radius, double angleStart, double angleExtent,
                        double annulusRadius, int32_t numLines,
                        double samplesPerPixel)
    : centerRow_(centerRow)
    , centerCol_(centerCol)
    , radius_(radius)
    , angleStart_(angleStart)
    , angleExtent_(angleExtent)
    , annulusRadius_(annulusRadius) {
    numLines_ = std::max(1, numLines);
    samplesPerPixel_ = std::max(0.1, samplesPerPixel);
    ComputeSamplingGeometry();
}

MeasureArc MeasureArc::FromArc(const Arc2d& arc, double annulusRadius, int32_t numLines) {
    return MeasureArc(arc.center.y, arc.center.x,
                       arc.radius, arc.startAngle, arc.sweepAngle,
                       annulusRadius, numLines);
}

MeasureArc MeasureArc::FromCircle(const Circle2d& circle, double annulusRadius, int32_t numLines) {
    return MeasureArc(circle.center.y, circle.center.x,
                       circle.radius, 0.0, TWO_PI,
                       annulusRadius, numLines);
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
    double arcLen = ProfileLength();
    if (arcLen < 1e-10) return angleStart_;
    double t = pos / arcLen;
    return angleStart_ + t * angleExtent_;
}

double MeasureArc::AngleToProfilePos(double angle) const {
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
    numLines_ = std::max(1, numLines);
    samplesPerPixel_ = std::max(0.1, samplesPerPixel);
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

// =============================================================================
// Factory Functions
// =============================================================================

MeasureRectangle2 CreateMeasureRect(double centerRow, double centerCol,
                                     double phi, double length, double width,
                                     int32_t numLines, double samplesPerPixel) {
    return MeasureRectangle2(centerRow, centerCol, phi, length, width,
                              numLines, samplesPerPixel);
}

MeasureArc CreateMeasureArc(double centerRow, double centerCol,
                             double radius, double angleStart, double angleExtent,
                             double annulusRadius, int32_t numLines,
                             double samplesPerPixel) {
    return MeasureArc(centerRow, centerCol, radius, angleStart, angleExtent,
                       annulusRadius, numLines, samplesPerPixel);
}

MeasureConcentricCircles CreateMeasureConcentric(double centerRow, double centerCol,
                                                   double innerRadius, double outerRadius,
                                                   double angle, double angularWidth,
                                                   int32_t numLines, double samplesPerPixel) {
    return MeasureConcentricCircles(centerRow, centerCol, innerRadius, outerRadius,
                                     angle, angularWidth, numLines, samplesPerPixel);
}

MeasureRectangle2 CreateMeasureFromSegment(const Point2d& p1, const Point2d& p2,
                                            double width, int32_t numLines) {
    return MeasureRectangle2::FromPoints(p1, p2, width, numLines);
}

MeasureRectangle2 CreateMeasureFromRect(const RotatedRect2d& rect, int32_t numLines) {
    return MeasureRectangle2::FromRotatedRect(rect, numLines);
}

} // namespace Qi::Vision::Measure
