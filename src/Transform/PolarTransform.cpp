/**
 * @file PolarTransform.cpp
 * @brief Implementation of public polar transformation API
 */

#include <QiVision/Transform/PolarTransform.h>
#include <QiVision/Internal/PolarTransform.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Core/Validate.h>

#include <cmath>
#include <cstring>

namespace Qi::Vision::Transform {

namespace {

// Transform-specific: sets dst to empty on empty src, requires UInt8
inline bool RequireImageU8(const QImage& src, QImage& dst, const char* funcName) {
    if (!Validate::RequireImageU8(src, funcName)) {
        dst = QImage();
        return false;
    }
    return true;
}

inline void RequirePositiveFinite(double value, const char* name, const char* funcName) {
    if (!std::isfinite(value)) {
        throw InvalidArgumentException(std::string(funcName) + ": " + name + " is invalid");
    }
    Validate::RequirePositive(value, name, funcName);
}

void RequirePointValid(const Point2d& point, const char* funcName) {
    if (!point.IsValid()) {
        throw InvalidArgumentException(std::string(funcName) + ": invalid point");
    }
}

// Convert public enum to internal enum
Internal::PolarMode ToInternalMode(PolarMode mode) {
    return mode == PolarMode::SemiLog ? Internal::PolarMode::SemiLog : Internal::PolarMode::Linear;
}

Internal::InterpolationMethod ToInternalInterp(PolarInterpolation interp) {
    switch (interp) {
        case PolarInterpolation::Nearest:
            return Internal::InterpolationMethod::Nearest;
        case PolarInterpolation::Bicubic:
            return Internal::InterpolationMethod::Bicubic;
        case PolarInterpolation::Bilinear:
        default:
            return Internal::InterpolationMethod::Bilinear;
    }
}

QImage FlipVertical(const QImage& src) {
    if (src.Empty()) {
        return QImage();
    }
    QImage dst(src.Width(), src.Height(), src.Type(), src.GetChannelType());
    int32_t h = src.Height();
    size_t stride = src.Stride();
    for (int32_t y = 0; y < h; ++y) {
        const void* srcRow = src.RowPtr(h - 1 - y);
        void* dstRow = dst.RowPtr(y);
        std::memcpy(dstRow, srcRow, stride);
    }
    return dst;
}

} // anonymous namespace

void CartesianToPolar(
    const QImage& src,
    QImage& dst,
    const Point2d& center,
    double maxRadius,
    int32_t dstWidth,
    int32_t dstHeight,
    PolarMode mode,
    PolarInterpolation interp,
    bool flipRadius)
{
    if (!RequireImageU8(src, dst, "CartesianToPolar")) {
        return;
    }
    RequirePointValid(center, "CartesianToPolar");
    RequirePositiveFinite(maxRadius, "maxRadius", "CartesianToPolar");
    Validate::RequireNonNegative(dstWidth, "dstWidth", "CartesianToPolar");
    Validate::RequireNonNegative(dstHeight, "dstHeight", "CartesianToPolar");
    QImage polar = Internal::WarpPolar(
        src, center, maxRadius,
        dstWidth, dstHeight,
        ToInternalMode(mode),
        false,  // forward transform
        ToInternalInterp(interp),
        Internal::BorderMode::Constant,
        0.0
    );
    dst = flipRadius ? FlipVertical(polar) : std::move(polar);
}

void PolarToCartesian(
    const QImage& src,
    QImage& dst,
    const Point2d& center,
    double maxRadius,
    int32_t dstWidth,
    int32_t dstHeight,
    PolarMode mode,
    PolarInterpolation interp,
    bool flipRadius)
{
    if (!RequireImageU8(src, dst, "PolarToCartesian")) {
        return;
    }
    RequirePointValid(center, "PolarToCartesian");
    RequirePositiveFinite(maxRadius, "maxRadius", "PolarToCartesian");
    Validate::RequireNonNegative(dstWidth, "dstWidth", "PolarToCartesian");
    Validate::RequireNonNegative(dstHeight, "dstHeight", "PolarToCartesian");
    // Default output size = 2 * maxRadius
    if (dstWidth == 0) dstWidth = static_cast<int32_t>(maxRadius * 2);
    if (dstHeight == 0) dstHeight = static_cast<int32_t>(maxRadius * 2);

    QImage srcPolar = flipRadius ? FlipVertical(src) : src;
    dst = Internal::WarpPolar(
        srcPolar, center, maxRadius,
        dstWidth, dstHeight,
        ToInternalMode(mode),
        true,  // inverse transform
        ToInternalInterp(interp),
        Internal::BorderMode::Constant,
        0.0
    );
}

void WarpPolar(
    const QImage& src,
    QImage& dst,
    const Point2d& center,
    double maxRadius,
    int32_t dstWidth,
    int32_t dstHeight,
    PolarMode mode,
    bool inverse,
    PolarInterpolation interp,
    bool flipRadius)
{
    if (!RequireImageU8(src, dst, "WarpPolar")) {
        return;
    }
    RequirePointValid(center, "WarpPolar");
    RequirePositiveFinite(maxRadius, "maxRadius", "WarpPolar");
    Validate::RequireNonNegative(dstWidth, "dstWidth", "WarpPolar");
    Validate::RequireNonNegative(dstHeight, "dstHeight", "WarpPolar");
    if (inverse) {
        PolarToCartesian(src, dst, center, maxRadius, dstWidth, dstHeight, mode, interp, flipRadius);
    } else {
        CartesianToPolar(src, dst, center, maxRadius, dstWidth, dstHeight, mode, interp, flipRadius);
    }
}

Point2d PointCartesianToPolar(const Point2d& pt, const Point2d& center) {
    RequirePointValid(pt, "PointCartesianToPolar");
    RequirePointValid(center, "PointCartesianToPolar");
    return Internal::CartesianToPolar(pt, center);
}

Point2d PointPolarToCartesian(double angle, double radius, const Point2d& center) {
    if (!std::isfinite(angle) || !std::isfinite(radius)) {
        throw InvalidArgumentException("PointPolarToCartesian: invalid angle/radius");
    }
    if (radius < 0.0) {
        throw InvalidArgumentException("PointPolarToCartesian: radius must be >= 0");
    }
    RequirePointValid(center, "PointPolarToCartesian");
    return Internal::PolarToCartesian(radius, angle, center);
}

} // namespace Qi::Vision::Transform
