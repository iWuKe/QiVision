/**
 * @file Segment.cpp
 * @brief Image segmentation and thresholding implementation
 *
 * Wraps Internal/Threshold.h functions for public API.
 */

#include <QiVision/Segment/Segment.h>
#include <QiVision/Internal/Threshold.h>
#include <QiVision/Internal/Histogram.h>

namespace Qi::Vision::Segment {

// =============================================================================
// Internal type conversion helpers
// =============================================================================

namespace {

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
    Internal::ThresholdGlobal(src, dst, threshold, maxValue, ToInternal(type));
}

QImage Threshold(const QImage& src, double threshold,
                 double maxValue, ThresholdType type) {
    return Internal::ThresholdGlobal(src, threshold, maxValue, ToInternal(type));
}

void ThresholdRange(const QImage& src, QImage& dst,
                    double low, double high, double maxValue) {
    Internal::ThresholdRange(src, dst, low, high, maxValue);
}

QImage ThresholdRange(const QImage& src, double low, double high,
                      double maxValue) {
    return Internal::ThresholdRange(src, low, high, maxValue);
}

// =============================================================================
// Auto Thresholding
// =============================================================================

void ThresholdAuto(const QImage& src, QImage& dst,
                   AutoMethod method, double maxValue,
                   double* computedThreshold) {
    Internal::ThresholdAuto(src, dst, ToInternal(method), maxValue, computedThreshold);
}

void ThresholdOtsu(const QImage& src, QImage& dst,
                   double maxValue, double* computedThreshold) {
    Internal::ThresholdOtsu(src, dst, maxValue, computedThreshold);
}

void ThresholdTriangle(const QImage& src, QImage& dst,
                       double maxValue, double* computedThreshold) {
    Internal::ThresholdTriangle(src, dst, maxValue, computedThreshold);
}

QImage ThresholdOtsu(const QImage& src, double maxValue) {
    return Internal::ThresholdOtsu(src, maxValue);
}

QImage ThresholdTriangle(const QImage& src, double maxValue) {
    return Internal::ThresholdTriangle(src, maxValue);
}

double ComputeAutoThreshold(const QImage& src, AutoMethod method) {
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
    Internal::ThresholdMultiLevel(src, dst, thresholds);
}

void ThresholdMultiLevel(const QImage& src, QImage& dst,
                         const std::vector<double>& thresholds,
                         const std::vector<double>& outputValues) {
    Internal::ThresholdMultiLevel(src, dst, thresholds, outputValues);
}

// =============================================================================
// Threshold to Region
// =============================================================================

QRegion ThresholdToRegion(const QImage& src, double low, double high) {
    return Internal::ThresholdToRegion(src, low, high);
}

QRegion ThresholdToRegion(const QImage& src, double threshold, bool above) {
    return Internal::ThresholdToRegion(src, threshold, above);
}

QRegion ThresholdAutoToRegion(const QImage& src,
                               AutoMethod method, bool above,
                               double* computedThreshold) {
    return Internal::ThresholdAutoToRegion(src, ToInternal(method), above, computedThreshold);
}

// =============================================================================
// Dynamic Threshold
// =============================================================================

QRegion DynThreshold(const QImage& image, const QImage& reference,
                     double offset, LightDark lightDark) {
    return Internal::DynThreshold(image, reference, offset, ToInternal(lightDark));
}

QRegion DynThreshold(const QImage& image, int32_t filterSize,
                     double offset, LightDark lightDark) {
    return Internal::DynThreshold(image, filterSize, offset, ToInternal(lightDark));
}

DualThresholdResult DualThreshold(const QImage& image,
                                   double lowThreshold, double highThreshold) {
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
    return Internal::VarThreshold(image, windowSize, varianceThreshold, ToInternal(lightDark));
}

QRegion CharThreshold(const QImage& image, double sigma,
                      double percent, LightDark lightDark) {
    return Internal::CharThreshold(image, sigma, percent, ToInternal(lightDark));
}

QRegion HysteresisThreshold(const QImage& image,
                            double lowThreshold, double highThreshold) {
    return Internal::HysteresisThresholdToRegion(image, lowThreshold, highThreshold);
}

// =============================================================================
// Domain-aware Threshold Operations
// =============================================================================

QRegion ThresholdWithDomain(const QImage& image, double low, double high) {
    return Internal::ThresholdWithDomain(image, low, high);
}

QRegion DynThresholdWithDomain(const QImage& image, const QImage& reference,
                                double offset, LightDark lightDark) {
    return Internal::DynThresholdWithDomain(image, reference, offset, ToInternal(lightDark));
}

QRegion ThresholdAdaptiveToRegion(const QImage& image,
                                   AdaptiveMethod method, int32_t blockSize, double C) {
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
    Internal::BinaryInvert(src, dst, maxValue);
}

void BinaryAnd(const QImage& src1, const QImage& src2, QImage& dst,
               double maxValue) {
    Internal::BinaryAnd(src1, src2, dst, maxValue);
}

void BinaryOr(const QImage& src1, const QImage& src2, QImage& dst,
              double maxValue) {
    Internal::BinaryOr(src1, src2, dst, maxValue);
}

void BinaryXor(const QImage& src1, const QImage& src2, QImage& dst,
               double maxValue) {
    Internal::BinaryXor(src1, src2, dst, maxValue);
}

void BinaryDiff(const QImage& src1, const QImage& src2, QImage& dst,
                double maxValue) {
    Internal::BinaryDiff(src1, src2, dst, maxValue);
}

// =============================================================================
// Utility Functions
// =============================================================================

bool IsBinaryImage(const QImage& image, double tolerance) {
    return Internal::IsBinaryImage(image, tolerance);
}

uint64_t CountNonZero(const QImage& image) {
    return Internal::CountNonZero(image);
}

uint64_t CountInRange(const QImage& image, double low, double high) {
    return Internal::CountInRange(image, low, high);
}

double ComputeForegroundRatio(const QImage& image) {
    return Internal::ComputeForegroundRatio(image);
}

void ApplyMask(const QImage& src, const QImage& mask, QImage& dst) {
    Internal::ApplyMask(src, mask, dst);
}

void RegionToMask(const QRegion& region, QImage& mask) {
    Internal::RegionToMask(region, mask);
}

QRegion MaskToRegion(const QImage& mask, double threshold) {
    return Internal::MaskToRegion(mask, threshold);
}

} // namespace Qi::Vision::Segment
