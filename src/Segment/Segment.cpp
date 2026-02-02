/**
 * @file Segment.cpp
 * @brief Image segmentation and thresholding implementation
 *
 * Wraps Internal/Threshold.h functions for public API.
 */

#include <QiVision/Segment/Segment.h>
#include <QiVision/Internal/Threshold.h>
#include <QiVision/Internal/Histogram.h>
#include <QiVision/Core/Exception.h>

#include <cmath>

namespace Qi::Vision::Segment {

// =============================================================================
// Internal type conversion helpers
// =============================================================================

namespace {

bool RequireGrayU8(const QImage& image, const char* funcName) {
    if (image.Empty()) {
        return false;
    }
    if (!image.IsValid()) {
        throw InvalidArgumentException(std::string(funcName) + ": invalid image");
    }
    if (image.Type() != PixelType::UInt8 || image.Channels() != 1) {
        throw UnsupportedException(std::string(funcName) +
                                   " requires single-channel UInt8 image");
    }
    return true;
}

void RequireFinite(double value, const char* message) {
    if (!std::isfinite(value)) {
        throw InvalidArgumentException(message);
    }
}

Internal::ThresholdType ToInternal(ThresholdType type) {
    switch (type) {
        case ThresholdType::Binary:    return Internal::ThresholdType::Binary;
        case ThresholdType::BinaryInv: return Internal::ThresholdType::BinaryInv;
        case ThresholdType::Truncate:  return Internal::ThresholdType::Truncate;
        case ThresholdType::ToZero:    return Internal::ThresholdType::ToZero;
        case ThresholdType::ToZeroInv: return Internal::ThresholdType::ToZeroInv;
    }
    return Internal::ThresholdType::Binary;
}

Internal::AutoThresholdMethod ToInternal(AutoMethod method) {
    switch (method) {
        case AutoMethod::Otsu:     return Internal::AutoThresholdMethod::Otsu;
        case AutoMethod::Triangle: return Internal::AutoThresholdMethod::Triangle;
        case AutoMethod::MinError: return Internal::AutoThresholdMethod::MinError;
        case AutoMethod::Isodata:  return Internal::AutoThresholdMethod::Isodata;
        case AutoMethod::Median:   return Internal::AutoThresholdMethod::Median;
    }
    return Internal::AutoThresholdMethod::Otsu;
}

Internal::AdaptiveMethod ToInternal(AdaptiveMethod method) {
    switch (method) {
        case AdaptiveMethod::Mean:     return Internal::AdaptiveMethod::Mean;
        case AdaptiveMethod::Gaussian: return Internal::AdaptiveMethod::Gaussian;
        case AdaptiveMethod::Sauvola:  return Internal::AdaptiveMethod::Sauvola;
        case AdaptiveMethod::Niblack:  return Internal::AdaptiveMethod::Niblack;
        case AdaptiveMethod::Wolf:     return Internal::AdaptiveMethod::Wolf;
    }
    return Internal::AdaptiveMethod::Mean;
}

Internal::LightDark ToInternal(LightDark ld) {
    switch (ld) {
        case LightDark::Light:    return Internal::LightDark::Light;
        case LightDark::Dark:     return Internal::LightDark::Dark;
        case LightDark::Equal:    return Internal::LightDark::Equal;
        case LightDark::NotEqual: return Internal::LightDark::NotEqual;
    }
    return Internal::LightDark::Light;
}

} // anonymous namespace

// =============================================================================
// Global Thresholding
// =============================================================================

void Threshold(const QImage& src, QImage& dst,
               double threshold, double maxValue,
               ThresholdType type) {
    if (!RequireGrayU8(src, "Threshold")) {
        dst = QImage();
        return;
    }
    RequireFinite(threshold, "Threshold: invalid threshold");
    RequireFinite(maxValue, "Threshold: invalid maxValue");
    Internal::ThresholdGlobal(src, dst, threshold, maxValue, ToInternal(type));
}

QImage Threshold(const QImage& src, double threshold,
                 double maxValue, ThresholdType type) {
    if (!RequireGrayU8(src, "Threshold")) {
        return QImage();
    }
    RequireFinite(threshold, "Threshold: invalid threshold");
    RequireFinite(maxValue, "Threshold: invalid maxValue");
    return Internal::ThresholdGlobal(src, threshold, maxValue, ToInternal(type));
}

void ThresholdRange(const QImage& src, QImage& dst,
                    double low, double high, double maxValue) {
    if (!RequireGrayU8(src, "ThresholdRange")) {
        dst = QImage();
        return;
    }
    RequireFinite(low, "ThresholdRange: invalid low");
    RequireFinite(high, "ThresholdRange: invalid high");
    RequireFinite(maxValue, "ThresholdRange: invalid maxValue");
    if (low > high) {
        throw InvalidArgumentException("ThresholdRange: low must be <= high");
    }
    Internal::ThresholdRange(src, dst, low, high, maxValue);
}

QImage ThresholdRange(const QImage& src, double low, double high,
                      double maxValue) {
    if (!RequireGrayU8(src, "ThresholdRange")) {
        return QImage();
    }
    RequireFinite(low, "ThresholdRange: invalid low");
    RequireFinite(high, "ThresholdRange: invalid high");
    RequireFinite(maxValue, "ThresholdRange: invalid maxValue");
    if (low > high) {
        throw InvalidArgumentException("ThresholdRange: low must be <= high");
    }
    return Internal::ThresholdRange(src, low, high, maxValue);
}

// =============================================================================
// Auto Thresholding
// =============================================================================

void ThresholdAuto(const QImage& src, QImage& dst,
                   AutoMethod method, double maxValue,
                   double* computedThreshold) {
    if (!RequireGrayU8(src, "ThresholdAuto")) {
        dst = QImage();
        if (computedThreshold) {
            *computedThreshold = 0.0;
        }
        return;
    }
    RequireFinite(maxValue, "ThresholdAuto: invalid maxValue");
    Internal::ThresholdAuto(src, dst, ToInternal(method), maxValue, computedThreshold);
}

void ThresholdOtsu(const QImage& src, QImage& dst,
                   double maxValue, double* computedThreshold) {
    if (!RequireGrayU8(src, "ThresholdOtsu")) {
        dst = QImage();
        if (computedThreshold) {
            *computedThreshold = 0.0;
        }
        return;
    }
    RequireFinite(maxValue, "ThresholdOtsu: invalid maxValue");
    Internal::ThresholdOtsu(src, dst, maxValue, computedThreshold);
}

void ThresholdTriangle(const QImage& src, QImage& dst,
                       double maxValue, double* computedThreshold) {
    if (!RequireGrayU8(src, "ThresholdTriangle")) {
        dst = QImage();
        if (computedThreshold) {
            *computedThreshold = 0.0;
        }
        return;
    }
    RequireFinite(maxValue, "ThresholdTriangle: invalid maxValue");
    Internal::ThresholdTriangle(src, dst, maxValue, computedThreshold);
}

QImage ThresholdOtsu(const QImage& src, double maxValue) {
    if (!RequireGrayU8(src, "ThresholdOtsu")) {
        return QImage();
    }
    RequireFinite(maxValue, "ThresholdOtsu: invalid maxValue");
    return Internal::ThresholdOtsu(src, maxValue);
}

QImage ThresholdTriangle(const QImage& src, double maxValue) {
    if (!RequireGrayU8(src, "ThresholdTriangle")) {
        return QImage();
    }
    RequireFinite(maxValue, "ThresholdTriangle: invalid maxValue");
    return Internal::ThresholdTriangle(src, maxValue);
}

double ComputeAutoThreshold(const QImage& src, AutoMethod method) {
    if (!RequireGrayU8(src, "ComputeAutoThreshold")) {
        return 0.0;
    }
    switch (method) {
        case AutoMethod::Otsu:
            return Internal::ComputeOtsuThreshold(src);
        case AutoMethod::Triangle:
            return Internal::ComputeTriangleThreshold(Internal::ComputeHistogram(src));
        case AutoMethod::MinError:
            return Internal::ComputeMinErrorThreshold(Internal::ComputeHistogram(src));
        case AutoMethod::Isodata:
            return Internal::ComputeIsodataThreshold(Internal::ComputeHistogram(src));
        case AutoMethod::Median: {
            auto hist = Internal::ComputeHistogram(src);
            return Internal::ComputePercentile(hist, 50.0);
        }
    }
    return Internal::ComputeOtsuThreshold(src);
}

// =============================================================================
// Adaptive Thresholding
// =============================================================================

void ThresholdAdaptive(const QImage& src, QImage& dst,
                       AdaptiveMethod method, int32_t blockSize, double C) {
    if (!RequireGrayU8(src, "ThresholdAdaptive")) {
        dst = QImage();
        return;
    }
    if (blockSize <= 0) {
        throw InvalidArgumentException("ThresholdAdaptive: blockSize must be > 0");
    }
    RequireFinite(C, "ThresholdAdaptive: invalid C");
    Internal::AdaptiveThresholdParams params;
    params.method = ToInternal(method);
    params.blockSize = blockSize;
    params.C = C;
    Internal::ThresholdAdaptive(src, dst, params);
}

// =============================================================================
// Multi-level Thresholding
// =============================================================================

void ThresholdMultiLevel(const QImage& src, QImage& dst,
                         const std::vector<double>& thresholds) {
    if (!RequireGrayU8(src, "ThresholdMultiLevel")) {
        dst = QImage();
        return;
    }
    if (thresholds.empty()) {
        throw InvalidArgumentException("ThresholdMultiLevel: thresholds must be non-empty");
    }
    for (double t : thresholds) {
        RequireFinite(t, "ThresholdMultiLevel: invalid threshold");
    }
    Internal::ThresholdMultiLevel(src, dst, thresholds);
}

void ThresholdMultiLevel(const QImage& src, QImage& dst,
                         const std::vector<double>& thresholds,
                         const std::vector<double>& outputValues) {
    if (!RequireGrayU8(src, "ThresholdMultiLevel")) {
        dst = QImage();
        return;
    }
    if (thresholds.empty()) {
        throw InvalidArgumentException("ThresholdMultiLevel: thresholds must be non-empty");
    }
    if (outputValues.size() < thresholds.size() + 1) {
        throw InvalidArgumentException("ThresholdMultiLevel: outputValues size mismatch");
    }
    for (double t : thresholds) {
        RequireFinite(t, "ThresholdMultiLevel: invalid threshold");
    }
    for (double v : outputValues) {
        RequireFinite(v, "ThresholdMultiLevel: invalid output value");
    }
    Internal::ThresholdMultiLevel(src, dst, thresholds, outputValues);
}

// =============================================================================
// Threshold to Region
// =============================================================================

QRegion ThresholdToRegion(const QImage& src, double low, double high) {
    if (!RequireGrayU8(src, "ThresholdToRegion")) {
        return QRegion();
    }
    RequireFinite(low, "ThresholdToRegion: invalid low");
    RequireFinite(high, "ThresholdToRegion: invalid high");
    if (low > high) {
        throw InvalidArgumentException("ThresholdToRegion: low must be <= high");
    }
    return Internal::ThresholdToRegion(src, low, high);
}

QRegion ThresholdToRegion(const QImage& src, double threshold, bool above) {
    if (!RequireGrayU8(src, "ThresholdToRegion")) {
        return QRegion();
    }
    RequireFinite(threshold, "ThresholdToRegion: invalid threshold");
    return Internal::ThresholdToRegion(src, threshold, above);
}

QRegion ThresholdAutoToRegion(const QImage& src,
                               AutoMethod method, bool above,
                               double* computedThreshold) {
    if (!RequireGrayU8(src, "ThresholdAutoToRegion")) {
        if (computedThreshold) {
            *computedThreshold = 0.0;
        }
        return QRegion();
    }
    return Internal::ThresholdAutoToRegion(src, ToInternal(method), above, computedThreshold);
}

// =============================================================================
// Dynamic Threshold
// =============================================================================

QRegion DynThreshold(const QImage& image, const QImage& reference,
                     double offset, LightDark lightDark) {
    if (!RequireGrayU8(image, "DynThreshold") || !RequireGrayU8(reference, "DynThreshold")) {
        return QRegion();
    }
    RequireFinite(offset, "DynThreshold: invalid offset");
    return Internal::DynThreshold(image, reference, offset, ToInternal(lightDark));
}

QRegion DynThreshold(const QImage& image, int32_t filterSize,
                     double offset, LightDark lightDark) {
    if (!RequireGrayU8(image, "DynThreshold")) {
        return QRegion();
    }
    if (filterSize <= 0) {
        throw InvalidArgumentException("DynThreshold: filterSize must be > 0");
    }
    RequireFinite(offset, "DynThreshold: invalid offset");
    return Internal::DynThreshold(image, filterSize, offset, ToInternal(lightDark));
}

DualThresholdResult DualThreshold(const QImage& image,
                                   double lowThreshold, double highThreshold) {
    if (!RequireGrayU8(image, "DualThreshold")) {
        DualThresholdResult result;
        result.lowThreshold = lowThreshold;
        result.highThreshold = highThreshold;
        return result;
    }
    RequireFinite(lowThreshold, "DualThreshold: invalid lowThreshold");
    RequireFinite(highThreshold, "DualThreshold: invalid highThreshold");
    if (lowThreshold > highThreshold) {
        throw InvalidArgumentException("DualThreshold: lowThreshold must be <= highThreshold");
    }
    auto internal = Internal::DualThreshold(image, lowThreshold, highThreshold);
    DualThresholdResult result;
    result.lightRegion = internal.lightRegion;
    result.darkRegion = internal.darkRegion;
    result.middleRegion = internal.middleRegion;
    result.lowThreshold = internal.lowThreshold;
    result.highThreshold = internal.highThreshold;
    return result;
}

QRegion VarThreshold(const QImage& image, int32_t windowSize,
                     double varianceThreshold, LightDark lightDark) {
    if (!RequireGrayU8(image, "VarThreshold")) {
        return QRegion();
    }
    if (windowSize <= 0) {
        throw InvalidArgumentException("VarThreshold: windowSize must be > 0");
    }
    RequireFinite(varianceThreshold, "VarThreshold: invalid varianceThreshold");
    return Internal::VarThreshold(image, windowSize, varianceThreshold, ToInternal(lightDark));
}

QRegion CharThreshold(const QImage& image, double sigma,
                      double percent, LightDark lightDark) {
    if (!RequireGrayU8(image, "CharThreshold")) {
        return QRegion();
    }
    RequireFinite(sigma, "CharThreshold: invalid sigma");
    RequireFinite(percent, "CharThreshold: invalid percent");
    if (sigma <= 0.0) {
        throw InvalidArgumentException("CharThreshold: sigma must be > 0");
    }
    if (percent < 0.0 || percent > 100.0) {
        throw InvalidArgumentException("CharThreshold: percent must be in [0,100]");
    }
    return Internal::CharThreshold(image, sigma, percent, ToInternal(lightDark));
}

QRegion HysteresisThreshold(const QImage& image,
                            double lowThreshold, double highThreshold) {
    if (!RequireGrayU8(image, "HysteresisThreshold")) {
        return QRegion();
    }
    RequireFinite(lowThreshold, "HysteresisThreshold: invalid lowThreshold");
    RequireFinite(highThreshold, "HysteresisThreshold: invalid highThreshold");
    if (lowThreshold > highThreshold) {
        throw InvalidArgumentException("HysteresisThreshold: lowThreshold must be <= highThreshold");
    }
    return Internal::HysteresisThresholdToRegion(image, lowThreshold, highThreshold);
}

// =============================================================================
// Domain-aware Threshold Operations
// =============================================================================

QRegion ThresholdWithDomain(const QImage& image, double low, double high) {
    if (!RequireGrayU8(image, "ThresholdWithDomain")) {
        return QRegion();
    }
    RequireFinite(low, "ThresholdWithDomain: invalid low");
    RequireFinite(high, "ThresholdWithDomain: invalid high");
    if (low > high) {
        throw InvalidArgumentException("ThresholdWithDomain: low must be <= high");
    }
    return Internal::ThresholdWithDomain(image, low, high);
}

QRegion DynThresholdWithDomain(const QImage& image, const QImage& reference,
                               double offset, LightDark lightDark) {
    if (!RequireGrayU8(image, "DynThresholdWithDomain") ||
        !RequireGrayU8(reference, "DynThresholdWithDomain")) {
        return QRegion();
    }
    RequireFinite(offset, "DynThresholdWithDomain: invalid offset");
    return Internal::DynThresholdWithDomain(image, reference, offset, ToInternal(lightDark));
}

QRegion ThresholdAdaptiveToRegion(const QImage& image,
                                 AdaptiveMethod method, int32_t blockSize, double C) {
    if (!RequireGrayU8(image, "ThresholdAdaptiveToRegion")) {
        return QRegion();
    }
    if (blockSize <= 0) {
        throw InvalidArgumentException("ThresholdAdaptiveToRegion: blockSize must be > 0");
    }
    RequireFinite(C, "ThresholdAdaptiveToRegion: invalid C");
    Internal::AdaptiveThresholdParams params;
    params.method = ToInternal(method);
    params.blockSize = blockSize;
    params.C = C;
    return Internal::ThresholdAdaptiveToRegion(image, params);
}

// =============================================================================
// Binary Image Operations
// =============================================================================

void BinaryInvert(const QImage& src, QImage& dst, double maxValue) {
    if (!RequireGrayU8(src, "BinaryInvert")) {
        dst = QImage();
        return;
    }
    RequireFinite(maxValue, "BinaryInvert: invalid maxValue");
    Internal::BinaryInvert(src, dst, maxValue);
}

void BinaryAnd(const QImage& src1, const QImage& src2, QImage& dst,
               double maxValue) {
    if (!RequireGrayU8(src1, "BinaryAnd") || !RequireGrayU8(src2, "BinaryAnd")) {
        dst = QImage();
        return;
    }
    RequireFinite(maxValue, "BinaryAnd: invalid maxValue");
    Internal::BinaryAnd(src1, src2, dst, maxValue);
}

void BinaryOr(const QImage& src1, const QImage& src2, QImage& dst,
              double maxValue) {
    if (!RequireGrayU8(src1, "BinaryOr") || !RequireGrayU8(src2, "BinaryOr")) {
        dst = QImage();
        return;
    }
    RequireFinite(maxValue, "BinaryOr: invalid maxValue");
    Internal::BinaryOr(src1, src2, dst, maxValue);
}

void BinaryXor(const QImage& src1, const QImage& src2, QImage& dst,
               double maxValue) {
    if (!RequireGrayU8(src1, "BinaryXor") || !RequireGrayU8(src2, "BinaryXor")) {
        dst = QImage();
        return;
    }
    RequireFinite(maxValue, "BinaryXor: invalid maxValue");
    Internal::BinaryXor(src1, src2, dst, maxValue);
}

void BinaryDiff(const QImage& src1, const QImage& src2, QImage& dst,
                double maxValue) {
    if (!RequireGrayU8(src1, "BinaryDiff") || !RequireGrayU8(src2, "BinaryDiff")) {
        dst = QImage();
        return;
    }
    RequireFinite(maxValue, "BinaryDiff: invalid maxValue");
    Internal::BinaryDiff(src1, src2, dst, maxValue);
}

// =============================================================================
// Utility Functions
// =============================================================================

bool IsBinaryImage(const QImage& image, double tolerance) {
    if (!RequireGrayU8(image, "IsBinaryImage")) {
        return false;
    }
    RequireFinite(tolerance, "IsBinaryImage: invalid tolerance");
    if (tolerance < 0.0) {
        throw InvalidArgumentException("IsBinaryImage: tolerance must be >= 0");
    }
    return Internal::IsBinaryImage(image, tolerance);
}

uint64_t CountNonZero(const QImage& image) {
    if (!RequireGrayU8(image, "CountNonZero")) {
        return 0;
    }
    return Internal::CountNonZero(image);
}

uint64_t CountInRange(const QImage& image, double low, double high) {
    if (!RequireGrayU8(image, "CountInRange")) {
        return 0;
    }
    RequireFinite(low, "CountInRange: invalid low");
    RequireFinite(high, "CountInRange: invalid high");
    if (low > high) {
        throw InvalidArgumentException("CountInRange: low must be <= high");
    }
    return Internal::CountInRange(image, low, high);
}

double ComputeForegroundRatio(const QImage& image) {
    if (!RequireGrayU8(image, "ComputeForegroundRatio")) {
        return 0.0;
    }
    return Internal::ComputeForegroundRatio(image);
}

void ApplyMask(const QImage& src, const QImage& mask, QImage& dst) {
    if (!RequireGrayU8(src, "ApplyMask") || !RequireGrayU8(mask, "ApplyMask")) {
        dst = QImage();
        return;
    }
    Internal::ApplyMask(src, mask, dst);
}

void RegionToMask(const QRegion& region, QImage& mask) {
    if (region.Empty()) {
        mask = QImage();
        return;
    }
    Internal::RegionToMask(region, mask);
}

QRegion MaskToRegion(const QImage& mask, double threshold) {
    if (!RequireGrayU8(mask, "MaskToRegion")) {
        return QRegion();
    }
    RequireFinite(threshold, "MaskToRegion: invalid threshold");
    return Internal::MaskToRegion(mask, threshold);
}

} // namespace Qi::Vision::Segment
