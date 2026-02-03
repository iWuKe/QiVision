/**
 * @file PolarTransform.cpp
 * @brief Implementation of public polar transformation API
 */

#include <QiVision/Transform/PolarTransform.h>
#include <QiVision/Internal/PolarTransform.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Core/Validate.h>

#include <cmath>

namespace Qi::Vision::Transform {

namespace {

// Transform-specific: sets dst to empty on empty src
inline bool RequireImage(const QImage& src, QImage& dst, const char* funcName) {
    if (src.Empty()) {
        dst = QImage();
        return false;
    }
    if (!src.IsValid()) {
        throw InvalidArgumentException(std::string(funcName) + ": invalid image");
    }
    if (src.Type() != PixelType::UInt8) {
        throw UnsupportedException(std::string(funcName) + ": only UInt8 images are supported");
    }
    return true;
}

inline void RequirePositive(double value, const char* name, const char* funcName) {
    if (!std::isfinite(value)) {
        throw InvalidArgumentException(std::string(funcName) + ": " + name + " is invalid");
    }
    Validate::RequirePositive(value, name, funcName);
}

inline void RequireNonNegativeSize(int32_t value, const char* name, const char* funcName) {
    Validate::RequireNonNegative(value, name, funcName);
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

} // anonymous namespace

void CartesianToPolar(
    const QImage& src,
    QImage& dst,
    const Point2d& center,
    double maxRadius,
    int32_t dstWidth,
    int32_t dstHeight,
    PolarMode mode,
    PolarInterpolation interp)
{
    if (!RequireImage(src, dst, "CartesianToPolar")) {
        return;
    }
    RequirePointValid(center, "CartesianToPolar");
    RequirePositive(maxRadius, "maxRadius", "CartesianToPolar");
    RequireNonNegativeSize(dstWidth, "dstWidth", "CartesianToPolar");
    RequireNonNegativeSize(dstHeight, "dstHeight", "CartesianToPolar");
    dst = Internal::WarpPolar(
        src, center, maxRadius,
        dstWidth, dstHeight,
        ToInternalMode(mode),
        false,  // forward transform
        ToInternalInterp(interp),
        Internal::BorderMode::Constant,
        0.0
    );
}

void PolarToCartesian(
    const QImage& src,
    QImage& dst,
    const Point2d& center,
    double maxRadius,
    int32_t dstWidth,
    int32_t dstHeight,
    PolarMode mode,
    PolarInterpolation interp)
{
    if (!RequireImage(src, dst, "PolarToCartesian")) {
        return;
    }
    RequirePointValid(center, "PolarToCartesian");
    RequirePositive(maxRadius, "maxRadius", "PolarToCartesian");
    RequireNonNegativeSize(dstWidth, "dstWidth", "PolarToCartesian");
    RequireNonNegativeSize(dstHeight, "dstHeight", "PolarToCartesian");
    // Default output size = 2 * maxRadius
    if (dstWidth == 0) dstWidth = static_cast<int32_t>(maxRadius * 2);
    if (dstHeight == 0) dstHeight = static_cast<int32_t>(maxRadius * 2);

    dst = Internal::WarpPolar(
        src, center, maxRadius,
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
    PolarInterpolation interp)
{
    if (!RequireImage(src, dst, "WarpPolar")) {
        return;
    }
    RequirePointValid(center, "WarpPolar");
    RequirePositive(maxRadius, "maxRadius", "WarpPolar");
    RequireNonNegativeSize(dstWidth, "dstWidth", "WarpPolar");
    RequireNonNegativeSize(dstHeight, "dstHeight", "WarpPolar");
    if (inverse) {
        PolarToCartesian(src, dst, center, maxRadius, dstWidth, dstHeight, mode, interp);
    } else {
        CartesianToPolar(src, dst, center, maxRadius, dstWidth, dstHeight, mode, interp);
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
