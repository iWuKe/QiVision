/**
 * @file Morphology.cpp
 * @brief Implementation of morphological operations (public API)
 */

#include <QiVision/Morphology/Morphology.h>
#include <QiVision/Internal/StructElement.h>
#include <QiVision/Internal/MorphBinary.h>
#include <QiVision/Internal/MorphGray.h>
#include <QiVision/Internal/RLEOps.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Core/Validate.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace Qi::Vision::Morphology {

static bool RequireGrayU8Input(const QImage& image, const char* funcName) {
    return Validate::RequireImageU8Gray(image, funcName);
}

// =============================================================================
// StructuringElement Implementation
// =============================================================================

struct StructuringElement::Impl {
    Internal::StructElement se;
    SEShape shape = SEShape::Custom;
};

StructuringElement::StructuringElement() : impl_(new Impl) {
    impl_->se = Internal::StructElement::Square(3);
    impl_->shape = SEShape::Rectangle;
}

StructuringElement::StructuringElement(const StructuringElement& other) : impl_(new Impl) {
    impl_->se = other.impl_->se;
    impl_->shape = other.impl_->shape;
}

StructuringElement::StructuringElement(StructuringElement&& other) noexcept
    : impl_(other.impl_) {
    other.impl_ = nullptr;
}

StructuringElement::~StructuringElement() {
    delete impl_;
}

StructuringElement& StructuringElement::operator=(const StructuringElement& other) {
    if (this != &other) {
        impl_->se = other.impl_->se;
        impl_->shape = other.impl_->shape;
    }
    return *this;
}

StructuringElement& StructuringElement::operator=(StructuringElement&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

StructuringElement StructuringElement::Rectangle(int32_t width, int32_t height) {
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("StructuringElement::Rectangle: width/height must be > 0");
    }
    StructuringElement se;
    se.impl_->se = Internal::StructElement::Rectangle(width, height);
    se.impl_->shape = SEShape::Rectangle;
    return se;
}

StructuringElement StructuringElement::Square(int32_t size) {
    return Rectangle(size, size);
}

StructuringElement StructuringElement::Circle(int32_t radius) {
    if (radius <= 0) {
        throw InvalidArgumentException("StructuringElement::Circle: radius must be > 0");
    }
    StructuringElement se;
    se.impl_->se = Internal::StructElement::Circle(radius);
    se.impl_->shape = SEShape::Circle;
    return se;
}

StructuringElement StructuringElement::Ellipse(int32_t radiusX, int32_t radiusY) {
    if (radiusX <= 0 || radiusY <= 0) {
        throw InvalidArgumentException("StructuringElement::Ellipse: radius must be > 0");
    }
    StructuringElement se;
    se.impl_->se = Internal::StructElement::Ellipse(radiusX, radiusY);
    se.impl_->shape = SEShape::Ellipse;
    return se;
}

StructuringElement StructuringElement::Cross(int32_t armLength, int32_t thickness) {
    if (armLength <= 0 || thickness <= 0) {
        throw InvalidArgumentException("StructuringElement::Cross: armLength/thickness must be > 0");
    }
    StructuringElement se;
    se.impl_->se = Internal::StructElement::Cross(armLength, thickness);
    se.impl_->shape = SEShape::Cross;
    return se;
}

StructuringElement StructuringElement::Diamond(int32_t radius) {
    if (radius <= 0) {
        throw InvalidArgumentException("StructuringElement::Diamond: radius must be > 0");
    }
    StructuringElement se;
    se.impl_->se = Internal::StructElement::Diamond(radius);
    se.impl_->shape = SEShape::Diamond;
    return se;
}

StructuringElement StructuringElement::Line(int32_t length, double angle) {
    if (length <= 0) {
        throw InvalidArgumentException("StructuringElement::Line: length must be > 0");
    }
    if (!std::isfinite(angle)) {
        throw InvalidArgumentException("StructuringElement::Line: invalid angle");
    }
    StructuringElement se;
    se.impl_->se = Internal::StructElement::Line(length, angle);
    se.impl_->shape = SEShape::Line;
    return se;
}

StructuringElement StructuringElement::FromMask(const QImage& mask,
                                                  int32_t anchorX,
                                                  int32_t anchorY) {
    if (!RequireGrayU8Input(mask, "StructuringElement::FromMask")) {
        return StructuringElement();
    }
    StructuringElement se;
    se.impl_->se = Internal::StructElement::FromMask(mask, anchorX, anchorY);
    se.impl_->shape = SEShape::Custom;
    return se;
}

StructuringElement StructuringElement::FromRegion(const QRegion& region,
                                                    int32_t anchorX,
                                                    int32_t anchorY) {
    StructuringElement se;
    se.impl_->se = Internal::StructElement::FromRegion(region, anchorX, anchorY);
    se.impl_->shape = SEShape::Custom;
    return se;
}

bool StructuringElement::Empty() const {
    return impl_ == nullptr || impl_->se.Empty();
}

int32_t StructuringElement::Width() const {
    return impl_ ? impl_->se.Width() : 0;
}

int32_t StructuringElement::Height() const {
    return impl_ ? impl_->se.Height() : 0;
}

int32_t StructuringElement::AnchorX() const {
    return impl_ ? impl_->se.AnchorX() : 0;
}

int32_t StructuringElement::AnchorY() const {
    return impl_ ? impl_->se.AnchorY() : 0;
}

SEShape StructuringElement::Shape() const {
    return impl_ ? impl_->shape : SEShape::Custom;
}

size_t StructuringElement::PixelCount() const {
    return impl_ ? impl_->se.PixelCount() : 0;
}

StructuringElement StructuringElement::Reflect() const {
    StructuringElement se;
    se.impl_->se = impl_->se.Reflect();
    se.impl_->shape = impl_->shape;
    return se;
}

StructuringElement StructuringElement::Rotate(double angle) const {
    StructuringElement se;
    se.impl_->se = impl_->se.Rotate(angle);
    se.impl_->shape = SEShape::Custom;  // Rotation may change shape
    return se;
}

void* StructuringElement::GetInternal() const {
    return impl_ ? const_cast<Internal::StructElement*>(&impl_->se) : nullptr;
}

// =============================================================================
// Binary Morphology (Region Operations)
// =============================================================================

QRegion Dilation(const QRegion& region, const StructuringElement& se) {
    if (region.Empty()) {
        return QRegion();
    }
    if (se.Empty()) {
        throw InvalidArgumentException("Dilation: structuring element is empty");
    }
    const auto* internalSE = static_cast<Internal::StructElement*>(se.GetInternal());
    return Internal::Dilate(region, *internalSE);
}

QRegion DilationCircle(const QRegion& region, int32_t radius) {
    if (region.Empty()) {
        return QRegion();
    }
    if (radius <= 0) {
        throw InvalidArgumentException("DilationCircle: radius must be > 0");
    }
    return Internal::DilateCircle(region, radius);
}

QRegion DilationRectangle(const QRegion& region, int32_t width, int32_t height) {
    if (region.Empty()) {
        return QRegion();
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("DilationRectangle: width/height must be > 0");
    }
    return Internal::DilateRect(region, width, height);
}

QRegion Erosion(const QRegion& region, const StructuringElement& se) {
    if (region.Empty()) {
        return QRegion();
    }
    if (se.Empty()) {
        throw InvalidArgumentException("Erosion: structuring element is empty");
    }
    const auto* internalSE = static_cast<Internal::StructElement*>(se.GetInternal());
    return Internal::Erode(region, *internalSE);
}

QRegion ErosionCircle(const QRegion& region, int32_t radius) {
    if (region.Empty()) {
        return QRegion();
    }
    if (radius <= 0) {
        throw InvalidArgumentException("ErosionCircle: radius must be > 0");
    }
    return Internal::ErodeCircle(region, radius);
}

QRegion ErosionRectangle(const QRegion& region, int32_t width, int32_t height) {
    if (region.Empty()) {
        return QRegion();
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("ErosionRectangle: width/height must be > 0");
    }
    return Internal::ErodeRect(region, width, height);
}

QRegion Opening(const QRegion& region, const StructuringElement& se) {
    if (region.Empty()) {
        return QRegion();
    }
    if (se.Empty()) {
        throw InvalidArgumentException("Opening: structuring element is empty");
    }
    const auto* internalSE = static_cast<Internal::StructElement*>(se.GetInternal());
    return Internal::Opening(region, *internalSE);
}

QRegion OpeningCircle(const QRegion& region, int32_t radius) {
    if (region.Empty()) {
        return QRegion();
    }
    if (radius <= 0) {
        throw InvalidArgumentException("OpeningCircle: radius must be > 0");
    }
    return Internal::OpeningCircle(region, radius);
}

QRegion OpeningRectangle(const QRegion& region, int32_t width, int32_t height) {
    if (region.Empty()) {
        return QRegion();
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("OpeningRectangle: width/height must be > 0");
    }
    return Internal::OpeningRect(region, width, height);
}

QRegion Closing(const QRegion& region, const StructuringElement& se) {
    if (region.Empty()) {
        return QRegion();
    }
    if (se.Empty()) {
        throw InvalidArgumentException("Closing: structuring element is empty");
    }
    const auto* internalSE = static_cast<Internal::StructElement*>(se.GetInternal());
    return Internal::Closing(region, *internalSE);
}

QRegion ClosingCircle(const QRegion& region, int32_t radius) {
    if (region.Empty()) {
        return QRegion();
    }
    if (radius <= 0) {
        throw InvalidArgumentException("ClosingCircle: radius must be > 0");
    }
    return Internal::ClosingCircle(region, radius);
}

QRegion ClosingRectangle(const QRegion& region, int32_t width, int32_t height) {
    if (region.Empty()) {
        return QRegion();
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("ClosingRectangle: width/height must be > 0");
    }
    return Internal::ClosingRect(region, width, height);
}

QRegion Boundary(const QRegion& region, const std::string& type) {
    if (region.Empty()) {
        return QRegion();
    }

    auto se = Internal::SE_Cross3();

    std::string lowerType = type;
    std::transform(lowerType.begin(), lowerType.end(), lowerType.begin(), ::tolower);
    if (lowerType.empty() || lowerType == "both") {
        // Both: Dilate(region) - Erode(region)
        return Internal::MorphGradient(region, se);
    }
    if (lowerType == "inner") {
        // Inner boundary: region - Erode(region)
        return Internal::InternalGradient(region, se);
    } else if (lowerType == "outer") {
        // Outer boundary: Dilate(region) - region
        return Internal::ExternalGradient(region, se);
    }
    throw InvalidArgumentException("Boundary: unknown type: " + type);
}

QRegion Skeleton(const QRegion& region) {
    if (region.Empty()) {
        return QRegion();
    }
    return Internal::Skeleton(region);
}

QRegion Thinning(const QRegion& region, int32_t maxIterations) {
    if (region.Empty()) {
        return QRegion();
    }
    return Internal::Thin(region, maxIterations);
}

QRegion PruneSkeleton(const QRegion& skeleton, int32_t iterations) {
    if (skeleton.Empty()) {
        return QRegion();
    }
    return Internal::PruneSkeleton(skeleton, iterations);
}

QRegion FillUp(const QRegion& region) {
    if (region.Empty()) {
        return QRegion();
    }
    return Internal::FillHolesByReconstruction(region);
}

QRegion ClearBorder(const QRegion& region, const Rect2i& bounds) {
    if (region.Empty()) {
        return QRegion();
    }
    return Internal::ClearBorder(region, bounds);
}

// =============================================================================
// Gray Morphology (Image Operations)
// =============================================================================

void GrayDilation(const QImage& image, QImage& output, const StructuringElement& se) {
    if (!RequireGrayU8Input(image, "GrayDilation")) {
        output = QImage();
        return;
    }
    if (se.Empty()) {
        throw InvalidArgumentException("GrayDilation: structuring element is empty");
    }
    const auto* internalSE = static_cast<Internal::StructElement*>(se.GetInternal());
    output = Internal::GrayDilate(image, *internalSE);
}

void GrayDilationCircle(const QImage& image, QImage& output, int32_t radius) {
    if (!RequireGrayU8Input(image, "GrayDilationCircle")) {
        output = QImage();
        return;
    }
    if (radius <= 0) {
        throw InvalidArgumentException("GrayDilationCircle: radius must be > 0");
    }
    output = Internal::GrayDilateCircle(image, radius);
}

void GrayDilationRectangle(const QImage& image, QImage& output, int32_t width, int32_t height) {
    if (!RequireGrayU8Input(image, "GrayDilationRectangle")) {
        output = QImage();
        return;
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("GrayDilationRectangle: width/height must be > 0");
    }
    output = Internal::GrayDilateRect(image, width, height);
}

void GrayErosion(const QImage& image, QImage& output, const StructuringElement& se) {
    if (!RequireGrayU8Input(image, "GrayErosion")) {
        output = QImage();
        return;
    }
    if (se.Empty()) {
        throw InvalidArgumentException("GrayErosion: structuring element is empty");
    }
    const auto* internalSE = static_cast<Internal::StructElement*>(se.GetInternal());
    output = Internal::GrayErode(image, *internalSE);
}

void GrayErosionCircle(const QImage& image, QImage& output, int32_t radius) {
    if (!RequireGrayU8Input(image, "GrayErosionCircle")) {
        output = QImage();
        return;
    }
    if (radius <= 0) {
        throw InvalidArgumentException("GrayErosionCircle: radius must be > 0");
    }
    output = Internal::GrayErodeCircle(image, radius);
}

void GrayErosionRectangle(const QImage& image, QImage& output, int32_t width, int32_t height) {
    if (!RequireGrayU8Input(image, "GrayErosionRectangle")) {
        output = QImage();
        return;
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("GrayErosionRectangle: width/height must be > 0");
    }
    output = Internal::GrayErodeRect(image, width, height);
}

void GrayOpening(const QImage& image, QImage& output, const StructuringElement& se) {
    if (!RequireGrayU8Input(image, "GrayOpening")) {
        output = QImage();
        return;
    }
    if (se.Empty()) {
        throw InvalidArgumentException("GrayOpening: structuring element is empty");
    }
    const auto* internalSE = static_cast<Internal::StructElement*>(se.GetInternal());
    output = Internal::GrayOpening(image, *internalSE);
}

void GrayOpeningCircle(const QImage& image, QImage& output, int32_t radius) {
    if (!RequireGrayU8Input(image, "GrayOpeningCircle")) {
        output = QImage();
        return;
    }
    if (radius <= 0) {
        throw InvalidArgumentException("GrayOpeningCircle: radius must be > 0");
    }
    output = Internal::GrayOpeningCircle(image, radius);
}

void GrayOpeningRectangle(const QImage& image, QImage& output, int32_t width, int32_t height) {
    if (!RequireGrayU8Input(image, "GrayOpeningRectangle")) {
        output = QImage();
        return;
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("GrayOpeningRectangle: width/height must be > 0");
    }
    output = Internal::GrayOpeningRect(image, width, height);
}

void GrayClosing(const QImage& image, QImage& output, const StructuringElement& se) {
    if (!RequireGrayU8Input(image, "GrayClosing")) {
        output = QImage();
        return;
    }
    if (se.Empty()) {
        throw InvalidArgumentException("GrayClosing: structuring element is empty");
    }
    const auto* internalSE = static_cast<Internal::StructElement*>(se.GetInternal());
    output = Internal::GrayClosing(image, *internalSE);
}

void GrayClosingCircle(const QImage& image, QImage& output, int32_t radius) {
    if (!RequireGrayU8Input(image, "GrayClosingCircle")) {
        output = QImage();
        return;
    }
    if (radius <= 0) {
        throw InvalidArgumentException("GrayClosingCircle: radius must be > 0");
    }
    output = Internal::GrayClosingCircle(image, radius);
}

void GrayClosingRectangle(const QImage& image, QImage& output, int32_t width, int32_t height) {
    if (!RequireGrayU8Input(image, "GrayClosingRectangle")) {
        output = QImage();
        return;
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("GrayClosingRectangle: width/height must be > 0");
    }
    output = Internal::GrayClosingRect(image, width, height);
}

void GrayGradient(const QImage& image, QImage& output, const StructuringElement& se) {
    if (!RequireGrayU8Input(image, "GrayGradient")) {
        output = QImage();
        return;
    }
    if (se.Empty()) {
        throw InvalidArgumentException("GrayGradient: structuring element is empty");
    }
    const auto* internalSE = static_cast<Internal::StructElement*>(se.GetInternal());
    output = Internal::GrayMorphGradient(image, *internalSE);
}

void GrayGradientRectangle(const QImage& image, QImage& output, int32_t width, int32_t height) {
    if (!RequireGrayU8Input(image, "GrayGradientRectangle")) {
        output = QImage();
        return;
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("GrayGradientRectangle: width/height must be > 0");
    }
    output = Internal::GrayRangeRect(image, width, height);
}

void GrayTopHat(const QImage& image, QImage& output, const StructuringElement& se) {
    if (!RequireGrayU8Input(image, "GrayTopHat")) {
        output = QImage();
        return;
    }
    if (se.Empty()) {
        throw InvalidArgumentException("GrayTopHat: structuring element is empty");
    }
    const auto* internalSE = static_cast<Internal::StructElement*>(se.GetInternal());
    output = Internal::GrayTopHat(image, *internalSE);
}

void GrayTopHatCircle(const QImage& image, QImage& output, int32_t radius) {
    if (!RequireGrayU8Input(image, "GrayTopHatCircle")) {
        output = QImage();
        return;
    }
    if (radius <= 0) {
        throw InvalidArgumentException("GrayTopHatCircle: radius must be > 0");
    }
    auto se = Internal::StructElement::Circle(radius);
    output = Internal::GrayTopHat(image, se);
}

void GrayBlackHat(const QImage& image, QImage& output, const StructuringElement& se) {
    if (!RequireGrayU8Input(image, "GrayBlackHat")) {
        output = QImage();
        return;
    }
    if (se.Empty()) {
        throw InvalidArgumentException("GrayBlackHat: structuring element is empty");
    }
    const auto* internalSE = static_cast<Internal::StructElement*>(se.GetInternal());
    output = Internal::GrayBlackHat(image, *internalSE);
}

void GrayBlackHatCircle(const QImage& image, QImage& output, int32_t radius) {
    if (!RequireGrayU8Input(image, "GrayBlackHatCircle")) {
        output = QImage();
        return;
    }
    if (radius <= 0) {
        throw InvalidArgumentException("GrayBlackHatCircle: radius must be > 0");
    }
    auto se = Internal::StructElement::Circle(radius);
    output = Internal::GrayBlackHat(image, se);
}

void GrayRange(const QImage& image, QImage& output, int32_t width, int32_t height) {
    if (!RequireGrayU8Input(image, "GrayRange")) {
        output = QImage();
        return;
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("GrayRange: width/height must be > 0");
    }
    output = Internal::GrayRangeRect(image, width, height);
}

void RollingBall(const QImage& image, QImage& output, int32_t radius) {
    if (!RequireGrayU8Input(image, "RollingBall")) {
        output = QImage();
        return;
    }
    if (radius <= 0) {
        throw InvalidArgumentException("RollingBall: radius must be > 0");
    }
    output = Internal::RollingBallBackground(image, radius);
}

void GrayReconstructDilation(const QImage& marker, const QImage& mask, QImage& output) {
    if (!RequireGrayU8Input(marker, "GrayReconstructDilation") ||
        !RequireGrayU8Input(mask, "GrayReconstructDilation")) {
        output = QImage();
        return;
    }
    if (marker.Width() != mask.Width() || marker.Height() != mask.Height()) {
        throw InvalidArgumentException("GrayReconstructDilation: marker/mask size mismatch");
    }
    output = Internal::GrayReconstructByDilation(marker, mask);
}

void GrayReconstructErosion(const QImage& marker, const QImage& mask, QImage& output) {
    if (!RequireGrayU8Input(marker, "GrayReconstructErosion") ||
        !RequireGrayU8Input(mask, "GrayReconstructErosion")) {
        output = QImage();
        return;
    }
    if (marker.Width() != mask.Width() || marker.Height() != mask.Height()) {
        throw InvalidArgumentException("GrayReconstructErosion: marker/mask size mismatch");
    }
    output = Internal::GrayReconstructByErosion(marker, mask);
}

void GrayFillHoles(const QImage& image, QImage& output) {
    if (!RequireGrayU8Input(image, "GrayFillHoles")) {
        output = QImage();
        return;
    }
    output = Internal::GrayFillHoles(image);
}

// =============================================================================
// Convenience Functions
// =============================================================================

StructuringElement SE_Cross3() {
    return StructuringElement::Cross(1, 1);
}

StructuringElement SE_Square3() {
    return StructuringElement::Square(3);
}

StructuringElement SE_Disk5() {
    return StructuringElement::Circle(2);
}

} // namespace Qi::Vision::Morphology
