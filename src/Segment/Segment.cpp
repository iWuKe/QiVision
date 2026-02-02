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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <random>

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

// =============================================================================
// K-Means Clustering Implementation
// =============================================================================

namespace {

// Feature extraction helpers
struct FeatureExtractor {
    KMeansFeature featureType;
    double spatialWeight;
    int32_t width, height;
    int32_t featureDim;

    FeatureExtractor(const QImage& image, KMeansFeature type, double sw)
        : featureType(type), spatialWeight(sw),
          width(image.Width()), height(image.Height()) {
        switch (type) {
            case KMeansFeature::Gray: featureDim = 1; break;
            case KMeansFeature::RGB:
            case KMeansFeature::HSV:
            case KMeansFeature::Lab: featureDim = 3; break;
            case KMeansFeature::GraySpatial: featureDim = 3; break;
            case KMeansFeature::RGBSpatial: featureDim = 5; break;
        }
    }

    void Extract(const QImage& image, std::vector<std::vector<double>>& features) const {
        int64_t numPixels = static_cast<int64_t>(width) * height;
        features.resize(numPixels);

        const uint8_t* data = static_cast<const uint8_t*>(image.Data());
        size_t stride = image.Stride();
        int32_t channels = image.Channels();

        // Normalization factors for spatial coordinates
        double normX = (width > 1) ? 255.0 / (width - 1) : 0.0;
        double normY = (height > 1) ? 255.0 / (height - 1) : 0.0;

        for (int32_t y = 0; y < height; ++y) {
            const uint8_t* row = data + y * stride;
            for (int32_t x = 0; x < width; ++x) {
                int64_t idx = static_cast<int64_t>(y) * width + x;
                features[idx].resize(featureDim);

                switch (featureType) {
                    case KMeansFeature::Gray:
                        features[idx][0] = row[x];
                        break;

                    case KMeansFeature::RGB:
                        if (channels >= 3) {
                            features[idx][0] = row[x * channels + 0];  // R
                            features[idx][1] = row[x * channels + 1];  // G
                            features[idx][2] = row[x * channels + 2];  // B
                        } else {
                            features[idx][0] = features[idx][1] = features[idx][2] = row[x];
                        }
                        break;

                    case KMeansFeature::HSV:
                        if (channels >= 3) {
                            double r = row[x * channels + 0] / 255.0;
                            double g = row[x * channels + 1] / 255.0;
                            double b = row[x * channels + 2] / 255.0;
                            double maxC = std::max({r, g, b});
                            double minC = std::min({r, g, b});
                            double delta = maxC - minC;

                            // H (0-180 like OpenCV)
                            double h = 0;
                            if (delta > 1e-6) {
                                if (maxC == r) h = 60 * std::fmod((g - b) / delta + 6, 6.0);
                                else if (maxC == g) h = 60 * ((b - r) / delta + 2);
                                else h = 60 * ((r - g) / delta + 4);
                            }
                            features[idx][0] = h / 2;  // Scale to 0-90
                            features[idx][1] = (maxC > 1e-6) ? (delta / maxC) * 255 : 0;  // S
                            features[idx][2] = maxC * 255;  // V
                        } else {
                            features[idx][0] = features[idx][1] = 0;
                            features[idx][2] = row[x];
                        }
                        break;

                    case KMeansFeature::Lab:
                        if (channels >= 3) {
                            // Simplified RGB to Lab conversion
                            double r = row[x * channels + 0] / 255.0;
                            double g = row[x * channels + 1] / 255.0;
                            double b = row[x * channels + 2] / 255.0;

                            // RGB to XYZ (D65)
                            double x_ = 0.412453 * r + 0.357580 * g + 0.180423 * b;
                            double y_ = 0.212671 * r + 0.715160 * g + 0.072169 * b;
                            double z_ = 0.019334 * r + 0.119193 * g + 0.950227 * b;

                            // Normalize
                            x_ /= 0.95047;
                            z_ /= 1.08883;

                            auto f = [](double t) {
                                return (t > 0.008856) ? std::cbrt(t) : (7.787 * t + 16.0/116.0);
                            };

                            double L = 116 * f(y_) - 16;
                            double a = 500 * (f(x_) - f(y_));
                            double b_ = 200 * (f(y_) - f(z_));

                            features[idx][0] = L * 2.55;       // L: 0-100 -> 0-255
                            features[idx][1] = a + 128;        // a: -128~127 -> 0-255
                            features[idx][2] = b_ + 128;       // b: -128~127 -> 0-255
                        } else {
                            features[idx][0] = row[x];
                            features[idx][1] = features[idx][2] = 128;
                        }
                        break;

                    case KMeansFeature::GraySpatial:
                        features[idx][0] = row[x];
                        features[idx][1] = x * normX * spatialWeight;
                        features[idx][2] = y * normY * spatialWeight;
                        break;

                    case KMeansFeature::RGBSpatial:
                        if (channels >= 3) {
                            features[idx][0] = row[x * channels + 0];
                            features[idx][1] = row[x * channels + 1];
                            features[idx][2] = row[x * channels + 2];
                        } else {
                            features[idx][0] = features[idx][1] = features[idx][2] = row[x];
                        }
                        features[idx][3] = x * normX * spatialWeight;
                        features[idx][4] = y * normY * spatialWeight;
                        break;
                }
            }
        }
    }
};

// Squared Euclidean distance
double SquaredDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// K-Means++ initialization
void KMeansPPInit(const std::vector<std::vector<double>>& features,
                  int32_t k, std::vector<std::vector<double>>& centers,
                  std::mt19937& rng) {
    size_t n = features.size();
    if (n == 0 || k <= 0) return;

    centers.clear();
    centers.reserve(k);

    // First center: random
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    centers.push_back(features[dist(rng)]);

    // Remaining centers: probability proportional to D(x)^2
    std::vector<double> minDist(n, std::numeric_limits<double>::max());

    for (int32_t c = 1; c < k; ++c) {
        // Update distances to nearest center
        double totalDist = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double d = SquaredDistance(features[i], centers.back());
            if (d < minDist[i]) minDist[i] = d;
            totalDist += minDist[i];
        }

        // Sample next center
        std::uniform_real_distribution<double> realDist(0, totalDist);
        double r = realDist(rng);
        double cumSum = 0.0;
        size_t nextIdx = 0;
        for (size_t i = 0; i < n; ++i) {
            cumSum += minDist[i];
            if (cumSum >= r) {
                nextIdx = i;
                break;
            }
        }
        centers.push_back(features[nextIdx]);
    }
}

// Random initialization
void RandomInit(const std::vector<std::vector<double>>& features,
                int32_t k, std::vector<std::vector<double>>& centers,
                std::mt19937& rng) {
    size_t n = features.size();
    if (n == 0 || k <= 0) return;

    centers.clear();
    centers.reserve(k);

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (int32_t i = 0; i < k && i < static_cast<int32_t>(n); ++i) {
        centers.push_back(features[indices[i]]);
    }
}

// Single K-Means run
KMeansResult RunKMeans(const std::vector<std::vector<double>>& features,
                       int32_t width, int32_t height, int32_t k,
                       const std::vector<std::vector<double>>& initCenters,
                       int32_t maxIterations, double epsilon) {
    KMeansResult result;
    size_t n = features.size();
    int32_t dim = static_cast<int32_t>(initCenters[0].size());

    // Initialize
    result.centers = initCenters;
    std::vector<int32_t> labels(n, 0);

    for (int32_t iter = 0; iter < maxIterations; ++iter) {
        // Assignment step
        result.compactness = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double minDist = std::numeric_limits<double>::max();
            int32_t bestLabel = 0;
            for (int32_t c = 0; c < k; ++c) {
                double d = SquaredDistance(features[i], result.centers[c]);
                if (d < minDist) {
                    minDist = d;
                    bestLabel = c;
                }
            }
            labels[i] = bestLabel;
            result.compactness += minDist;
        }

        // Update step
        std::vector<std::vector<double>> newCenters(k, std::vector<double>(dim, 0.0));
        std::vector<int64_t> counts(k, 0);

        for (size_t i = 0; i < n; ++i) {
            int32_t c = labels[i];
            counts[c]++;
            for (int32_t d = 0; d < dim; ++d) {
                newCenters[c][d] += features[i][d];
            }
        }

        // Compute means and check convergence
        double maxMove = 0.0;
        for (int32_t c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                for (int32_t d = 0; d < dim; ++d) {
                    newCenters[c][d] /= counts[c];
                }
            } else {
                newCenters[c] = result.centers[c];  // Keep old center if empty
            }
            double move = std::sqrt(SquaredDistance(newCenters[c], result.centers[c]));
            if (move > maxMove) maxMove = move;
        }

        result.centers = newCenters;
        result.iterations = iter + 1;

        if (maxMove < epsilon) {
            result.converged = true;
            break;
        }
    }

    // Create label image (use Int16, supports up to 32767 clusters)
    result.labels = QImage(width, height, PixelType::Int16, ChannelType::Gray);
    int16_t* labelData = static_cast<int16_t*>(result.labels.Data());
    size_t labelStride = result.labels.Stride() / sizeof(int16_t);

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            size_t idx = static_cast<size_t>(y) * width + x;
            labelData[y * labelStride + x] = static_cast<int16_t>(labels[idx]);
        }
    }

    // Compute cluster sizes
    result.clusterSizes.assign(k, 0);
    for (size_t i = 0; i < n; ++i) {
        result.clusterSizes[labels[i]]++;
    }

    return result;
}

} // anonymous namespace

KMeansResult KMeans(const QImage& image, const KMeansParams& params) {
    KMeansResult result;

    if (image.Empty()) {
        return result;
    }
    if (!image.IsValid()) {
        throw InvalidArgumentException("KMeans: invalid image");
    }
    if (image.Type() != PixelType::UInt8) {
        throw UnsupportedException("KMeans: requires UInt8 image");
    }
    if (params.k < 1) {
        throw InvalidArgumentException("KMeans: k must be >= 1");
    }

    // Check channel requirements
    bool needsColor = (params.feature == KMeansFeature::RGB ||
                       params.feature == KMeansFeature::HSV ||
                       params.feature == KMeansFeature::Lab ||
                       params.feature == KMeansFeature::RGBSpatial);
    if (needsColor && image.Channels() < 3) {
        throw InvalidArgumentException("KMeans: color feature requires 3-channel image");
    }

    int32_t width = image.Width();
    int32_t height = image.Height();
    int64_t numPixels = static_cast<int64_t>(width) * height;

    if (numPixels == 0) {
        return result;
    }

    // Extract features
    FeatureExtractor extractor(image, params.feature, params.spatialWeight);
    std::vector<std::vector<double>> features;
    extractor.Extract(image, features);

    // Run multiple attempts
    std::random_device rd;
    std::mt19937 rng(rd());

    KMeansResult bestResult;
    bestResult.compactness = std::numeric_limits<double>::max();

    for (int32_t attempt = 0; attempt < params.attempts; ++attempt) {
        std::vector<std::vector<double>> centers;

        if (params.init == KMeansInit::KMeansPP) {
            KMeansPPInit(features, params.k, centers, rng);
        } else {
            RandomInit(features, params.k, centers, rng);
        }

        KMeansResult current = RunKMeans(features, width, height, params.k,
                                         centers, params.maxIterations, params.epsilon);

        if (current.compactness < bestResult.compactness) {
            bestResult = std::move(current);
        }
    }

    return bestResult;
}

KMeansResult KMeans(const QImage& image, int32_t k, KMeansFeature feature) {
    KMeansParams params;
    params.k = k;
    params.feature = feature;
    return KMeans(image, params);
}

QImage KMeansSegment(const QImage& image, int32_t k, KMeansFeature feature) {
    KMeansParams params;
    params.k = k;
    params.feature = feature;
    return KMeansSegment(image, params);
}

QImage KMeansSegment(const QImage& image, const KMeansParams& params) {
    KMeansResult result = KMeans(image, params);

    if (result.labels.Empty()) {
        return QImage();
    }

    int32_t width = image.Width();
    int32_t height = image.Height();
    int32_t channels = image.Channels();
    int32_t k = params.k;

    // Create output image with same format as input
    QImage output(width, height, image.Type(),
                  channels == 1 ? ChannelType::Gray : ChannelType::RGB);

    uint8_t* outData = static_cast<uint8_t*>(output.Data());
    size_t outStride = output.Stride();
    const int16_t* labelData = static_cast<const int16_t*>(result.labels.Data());
    size_t labelStride = result.labels.Stride() / sizeof(int16_t);

    // Map centers back to image space
    bool isGrayFeature = (params.feature == KMeansFeature::Gray ||
                          params.feature == KMeansFeature::GraySpatial);

    if (channels == 1 || isGrayFeature) {
        // Grayscale output
        std::vector<uint8_t> centerValues(k);
        for (int32_t c = 0; c < k; ++c) {
            centerValues[c] = static_cast<uint8_t>(
                std::clamp(result.centers[c][0], 0.0, 255.0));
        }

        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                int32_t label = labelData[y * labelStride + x];
                outData[y * outStride + x] = centerValues[label];
            }
        }
    } else {
        // Color output
        std::vector<std::array<uint8_t, 3>> centerColors(k);
        for (int32_t c = 0; c < k; ++c) {
            if (params.feature == KMeansFeature::HSV) {
                // Convert HSV center back to RGB
                double h = result.centers[c][0] * 2;  // 0-180
                double s = result.centers[c][1] / 255.0;
                double v = result.centers[c][2] / 255.0;

                double c_ = v * s;
                double x_ = c_ * (1 - std::abs(std::fmod(h / 60.0, 2) - 1));
                double m = v - c_;

                double r, g, b;
                if (h < 60) { r = c_; g = x_; b = 0; }
                else if (h < 120) { r = x_; g = c_; b = 0; }
                else if (h < 180) { r = 0; g = c_; b = x_; }
                else if (h < 240) { r = 0; g = x_; b = c_; }
                else if (h < 300) { r = x_; g = 0; b = c_; }
                else { r = c_; g = 0; b = x_; }

                centerColors[c][0] = static_cast<uint8_t>(std::clamp((r + m) * 255, 0.0, 255.0));
                centerColors[c][1] = static_cast<uint8_t>(std::clamp((g + m) * 255, 0.0, 255.0));
                centerColors[c][2] = static_cast<uint8_t>(std::clamp((b + m) * 255, 0.0, 255.0));
            } else if (params.feature == KMeansFeature::Lab) {
                // Convert Lab center back to RGB (simplified)
                double L = result.centers[c][0] / 2.55;
                double a = result.centers[c][1] - 128;
                double b_ = result.centers[c][2] - 128;

                auto fInv = [](double t) {
                    double t3 = t * t * t;
                    return (t3 > 0.008856) ? t3 : (t - 16.0/116.0) / 7.787;
                };

                double fy = (L + 16) / 116.0;
                double fx = a / 500.0 + fy;
                double fz = fy - b_ / 200.0;

                double x_ = fInv(fx) * 0.95047;
                double y_ = fInv(fy);
                double z_ = fInv(fz) * 1.08883;

                double r = 3.2406 * x_ - 1.5372 * y_ - 0.4986 * z_;
                double g = -0.9689 * x_ + 1.8758 * y_ + 0.0415 * z_;
                double b = 0.0557 * x_ - 0.2040 * y_ + 1.0570 * z_;

                centerColors[c][0] = static_cast<uint8_t>(std::clamp(r * 255, 0.0, 255.0));
                centerColors[c][1] = static_cast<uint8_t>(std::clamp(g * 255, 0.0, 255.0));
                centerColors[c][2] = static_cast<uint8_t>(std::clamp(b * 255, 0.0, 255.0));
            } else {
                // RGB or RGBSpatial
                centerColors[c][0] = static_cast<uint8_t>(std::clamp(result.centers[c][0], 0.0, 255.0));
                centerColors[c][1] = static_cast<uint8_t>(std::clamp(result.centers[c][1], 0.0, 255.0));
                centerColors[c][2] = static_cast<uint8_t>(std::clamp(result.centers[c][2], 0.0, 255.0));
            }
        }

        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                int32_t label = labelData[y * labelStride + x];
                outData[y * outStride + x * channels + 0] = centerColors[label][0];
                outData[y * outStride + x * channels + 1] = centerColors[label][1];
                outData[y * outStride + x * channels + 2] = centerColors[label][2];
            }
        }
    }

    return output;
}

void KMeansToRegions(const QImage& image, int32_t k,
                     std::vector<QRegion>& regions, KMeansFeature feature) {
    KMeansParams params;
    params.k = k;
    params.feature = feature;
    KMeansToRegions(image, params, regions);
}

void KMeansToRegions(const QImage& image, const KMeansParams& params,
                     std::vector<QRegion>& regions) {
    KMeansResult result = KMeans(image, params);
    LabelsToRegions(result.labels, params.k, regions);
}

void LabelsToRegions(const QImage& labels, int32_t k,
                     std::vector<QRegion>& regions) {
    regions.clear();
    regions.resize(k);

    if (labels.Empty()) return;
    if (labels.Type() != PixelType::Int16) {
        throw InvalidArgumentException("LabelsToRegions: requires Int16 label image");
    }

    int32_t width = labels.Width();
    int32_t height = labels.Height();
    const int16_t* data = static_cast<const int16_t*>(labels.Data());
    size_t stride = labels.Stride() / sizeof(int16_t);

    // Collect runs for each cluster
    std::vector<std::vector<QRegion::Run>> allRuns(k);

    for (int32_t y = 0; y < height; ++y) {
        const int16_t* row = data + y * stride;

        int32_t runStart = -1;
        int32_t currentLabel = -1;

        for (int32_t x = 0; x <= width; ++x) {
            int32_t label = (x < width) ? static_cast<int32_t>(row[x]) : -1;

            if (label != currentLabel) {
                // End current run
                if (runStart >= 0 && currentLabel >= 0 && currentLabel < k) {
                    allRuns[currentLabel].push_back({y, runStart, x});  // colEnd is exclusive
                }
                // Start new run
                runStart = x;
                currentLabel = label;
            }
        }
    }

    // Create regions
    for (int32_t c = 0; c < k; ++c) {
        if (!allRuns[c].empty()) {
            regions[c] = QRegion(std::move(allRuns[c]));
        }
    }
}

// =============================================================================
// Watershed Segmentation Implementation
// =============================================================================

namespace {

// Priority queue element for watershed
struct WatershedPixel {
    int32_t x, y;
    float value;  // gradient or distance value
    int32_t label;

    bool operator>(const WatershedPixel& other) const {
        return value > other.value;  // Min-heap
    }
};

// 8-connectivity neighbors
const int32_t dx8[] = {-1, 0, 1, -1, 1, -1, 0, 1};
const int32_t dy8[] = {-1, -1, -1, 0, 0, 1, 1, 1};

// Compute distance transform (Euclidean approximation using chamfer 3-4)
void ComputeDistanceTransformInternal(const uint8_t* binary, size_t binaryStride,
                                      float* dist, int32_t width, int32_t height) {
    const float INF = 1e9f;
    const float D1 = 1.0f;      // Orthogonal distance
    const float D2 = 1.414f;    // Diagonal distance

    // Initialize: 0 for background, INF for foreground
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            dist[y * width + x] = (binary[y * binaryStride + x] > 0) ? INF : 0.0f;
        }
    }

    // Forward pass
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            float& d = dist[y * width + x];
            if (x > 0) d = std::min(d, dist[y * width + (x-1)] + D1);
            if (y > 0) d = std::min(d, dist[(y-1) * width + x] + D1);
            if (x > 0 && y > 0) d = std::min(d, dist[(y-1) * width + (x-1)] + D2);
            if (x < width-1 && y > 0) d = std::min(d, dist[(y-1) * width + (x+1)] + D2);
        }
    }

    // Backward pass
    for (int32_t y = height - 1; y >= 0; --y) {
        for (int32_t x = width - 1; x >= 0; --x) {
            float& d = dist[y * width + x];
            if (x < width-1) d = std::min(d, dist[y * width + (x+1)] + D1);
            if (y < height-1) d = std::min(d, dist[(y+1) * width + x] + D1);
            if (x < width-1 && y < height-1) d = std::min(d, dist[(y+1) * width + (x+1)] + D2);
            if (x > 0 && y < height-1) d = std::min(d, dist[(y+1) * width + (x-1)] + D2);
        }
    }
}

// Find local maxima in distance transform
void FindLocalMaxima(const float* dist, int32_t width, int32_t height,
                     double minDistance, double minPeakValue,
                     std::vector<std::pair<int32_t, int32_t>>& peaks) {
    peaks.clear();
    int32_t halfWin = static_cast<int32_t>(minDistance / 2);
    if (halfWin < 1) halfWin = 1;

    // Find all local maxima
    std::vector<std::tuple<float, int32_t, int32_t>> candidates;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            float val = dist[y * width + x];
            if (val < minPeakValue) continue;

            // Check if local maximum in neighborhood
            bool isMax = true;
            for (int32_t dy = -halfWin; dy <= halfWin && isMax; ++dy) {
                for (int32_t dx = -halfWin; dx <= halfWin && isMax; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int32_t nx = x + dx, ny = y + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        if (dist[ny * width + nx] > val) {
                            isMax = false;
                        }
                    }
                }
            }

            if (isMax) {
                candidates.emplace_back(val, x, y);
            }
        }
    }

    // Sort by value (descending)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });

    // Non-maximum suppression
    std::vector<bool> suppressed(candidates.size(), false);
    double minDist2 = minDistance * minDistance;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;

        int32_t x1 = std::get<1>(candidates[i]);
        int32_t y1 = std::get<2>(candidates[i]);
        peaks.emplace_back(x1, y1);

        // Suppress nearby peaks
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;
            int32_t x2 = std::get<1>(candidates[j]);
            int32_t y2 = std::get<2>(candidates[j]);
            double d2 = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
            if (d2 < minDist2) {
                suppressed[j] = true;
            }
        }
    }
}

// Watershed flooding algorithm
void WatershedFlood(const float* gradientOrDist, int32_t width, int32_t height,
                    int16_t* labels, const uint8_t* mask, size_t maskStride) {
    const int16_t WATERSHED = -1;
    const int16_t UNVISITED = -2;

    // Initialize unvisited pixels
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            if (labels[y * width + x] == 0) {
                // Check if in mask (foreground)
                if (mask == nullptr || mask[y * maskStride + x] > 0) {
                    labels[y * width + x] = UNVISITED;
                }
            }
        }
    }

    // Priority queue: process pixels in order of gradient/distance
    std::priority_queue<WatershedPixel, std::vector<WatershedPixel>,
                        std::greater<WatershedPixel>> pq;

    // Initialize queue with neighbors of markers
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            int16_t lbl = labels[y * width + x];
            if (lbl > 0) {
                // Check neighbors
                for (int i = 0; i < 8; ++i) {
                    int32_t nx = x + dx8[i];
                    int32_t ny = y + dy8[i];
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        if (labels[ny * width + nx] == UNVISITED) {
                            labels[ny * width + nx] = 0;  // Mark as in queue
                            pq.push({nx, ny, gradientOrDist[ny * width + nx], lbl});
                        }
                    }
                }
            }
        }
    }

    // Flood
    while (!pq.empty()) {
        WatershedPixel p = pq.top();
        pq.pop();

        // Skip if already labeled
        if (labels[p.y * width + p.x] != 0) continue;

        // Check neighbors for labels
        int16_t neighborLabel = 0;
        bool hasMultipleLabels = false;

        for (int i = 0; i < 8; ++i) {
            int32_t nx = p.x + dx8[i];
            int32_t ny = p.y + dy8[i];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int16_t nlbl = labels[ny * width + nx];
                if (nlbl > 0) {
                    if (neighborLabel == 0) {
                        neighborLabel = nlbl;
                    } else if (neighborLabel != nlbl) {
                        hasMultipleLabels = true;
                    }
                }
            }
        }

        // Assign label or mark as watershed
        if (hasMultipleLabels) {
            labels[p.y * width + p.x] = WATERSHED;
        } else if (neighborLabel > 0) {
            labels[p.y * width + p.x] = neighborLabel;

            // Add unvisited neighbors to queue
            for (int i = 0; i < 8; ++i) {
                int32_t nx = p.x + dx8[i];
                int32_t ny = p.y + dy8[i];
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    if (labels[ny * width + nx] == UNVISITED) {
                        labels[ny * width + nx] = 0;  // Mark as in queue
                        pq.push({nx, ny, gradientOrDist[ny * width + nx], neighborLabel});
                    }
                }
            }
        }
    }

    // Convert remaining UNVISITED to 0
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            if (labels[y * width + x] == UNVISITED) {
                labels[y * width + x] = 0;
            }
        }
    }
}

// Extract regions from label image
void ExtractWatershedRegions(const int16_t* labels, int32_t width, int32_t height,
                             WatershedResult& result) {
    // Find max label
    int16_t maxLabel = 0;
    for (int32_t i = 0; i < width * height; ++i) {
        if (labels[i] > maxLabel) maxLabel = labels[i];
    }

    result.numRegions = maxLabel;
    result.regions.resize(maxLabel);

    // Collect runs for each region
    std::vector<std::vector<QRegion::Run>> allRuns(maxLabel + 1);
    std::vector<QRegion::Run> watershedRuns;

    for (int32_t y = 0; y < height; ++y) {
        int32_t runStart = -1;
        int16_t currentLabel = -2;  // Invalid initial

        for (int32_t x = 0; x <= width; ++x) {
            int16_t lbl = (x < width) ? labels[y * width + x] : -2;

            if (lbl != currentLabel) {
                // End current run
                if (runStart >= 0) {
                    if (currentLabel > 0) {
                        allRuns[currentLabel].push_back({y, runStart, x});  // colEnd is exclusive
                    } else if (currentLabel == -1) {
                        watershedRuns.push_back({y, runStart, x});  // colEnd is exclusive
                    }
                }
                runStart = x;
                currentLabel = lbl;
            }
        }
    }

    // Create regions
    for (int16_t lbl = 1; lbl <= maxLabel; ++lbl) {
        if (!allRuns[lbl].empty()) {
            result.regions[lbl - 1] = QRegion(std::move(allRuns[lbl]));
        }
    }

    if (!watershedRuns.empty()) {
        result.watershedLines = QRegion(std::move(watershedRuns));
    }
}

} // anonymous namespace

QImage DistanceTransform(const QImage& binaryImage) {
    if (binaryImage.Empty()) {
        return QImage();
    }
    if (binaryImage.Type() != PixelType::UInt8 || binaryImage.Channels() != 1) {
        throw UnsupportedException("DistanceTransform: requires single-channel UInt8 image");
    }

    int32_t width = binaryImage.Width();
    int32_t height = binaryImage.Height();

    QImage result(width, height, PixelType::Float32, ChannelType::Gray);
    float* distData = static_cast<float*>(result.Data());
    const uint8_t* binData = static_cast<const uint8_t*>(binaryImage.Data());

    ComputeDistanceTransformInternal(binData, binaryImage.Stride(),
                                     distData, width, height);

    return result;
}

QImage DistanceTransform(const QRegion& region) {
    if (region.Empty()) {
        return QImage();
    }

    // Get bounding box
    auto bbox = region.BoundingBox();
    int32_t width = bbox.width;
    int32_t height = bbox.height;

    // Create binary image from region
    QImage binary(width, height, PixelType::UInt8, ChannelType::Gray);
    std::memset(binary.Data(), 0, binary.Height() * binary.Stride());

    uint8_t* binData = static_cast<uint8_t*>(binary.Data());
    size_t stride = binary.Stride();

    for (const auto& run : region.Runs()) {
        int32_t y = run.row - bbox.y;
        if (y < 0 || y >= height) continue;
        int32_t x0 = std::max(0, run.colBegin - bbox.x);
        int32_t x1 = std::min(width - 1, run.colEnd - bbox.x);
        for (int32_t x = x0; x <= x1; ++x) {
            binData[y * stride + x] = 255;
        }
    }

    return DistanceTransform(binary);
}

QImage CreateWatershedMarkers(const QImage& distanceImage,
                              double minDistance, double minPeakValue) {
    if (distanceImage.Empty()) {
        return QImage();
    }
    if (distanceImage.Type() != PixelType::Float32) {
        throw InvalidArgumentException("CreateWatershedMarkers: requires Float32 distance image");
    }

    int32_t width = distanceImage.Width();
    int32_t height = distanceImage.Height();
    const float* distData = static_cast<const float*>(distanceImage.Data());

    // Find local maxima
    std::vector<std::pair<int32_t, int32_t>> peaks;
    FindLocalMaxima(distData, width, height, minDistance, minPeakValue, peaks);

    // Create marker image
    QImage markers(width, height, PixelType::Int16, ChannelType::Gray);
    std::memset(markers.Data(), 0, markers.Height() * markers.Stride());

    int16_t* markerData = static_cast<int16_t*>(markers.Data());
    size_t markerStride = markers.Stride() / sizeof(int16_t);

    // Label each peak
    for (size_t i = 0; i < peaks.size(); ++i) {
        int32_t x = peaks[i].first;
        int32_t y = peaks[i].second;
        markerData[y * markerStride + x] = static_cast<int16_t>(i + 1);
    }

    return markers;
}

WatershedResult Watershed(const QImage& image, const QImage& markers) {
    WatershedResult result;

    if (image.Empty() || markers.Empty()) {
        return result;
    }
    if (image.Type() != PixelType::UInt8 || image.Channels() != 1) {
        throw UnsupportedException("Watershed: requires single-channel UInt8 image");
    }
    if (markers.Type() != PixelType::Int16) {
        throw InvalidArgumentException("Watershed: markers must be Int16 image");
    }
    if (image.Width() != markers.Width() || image.Height() != markers.Height()) {
        throw InvalidArgumentException("Watershed: image and markers must have same size");
    }

    int32_t width = image.Width();
    int32_t height = image.Height();

    // Compute gradient magnitude for flooding priority
    std::vector<float> gradient(width * height);
    const uint8_t* imgData = static_cast<const uint8_t*>(image.Data());
    size_t imgStride = image.Stride();

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            // Simple Sobel gradient
            float gx = 0, gy = 0;
            if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                gx = -imgData[(y-1)*imgStride + x-1] + imgData[(y-1)*imgStride + x+1]
                     -2*imgData[y*imgStride + x-1] + 2*imgData[y*imgStride + x+1]
                     -imgData[(y+1)*imgStride + x-1] + imgData[(y+1)*imgStride + x+1];
                gy = -imgData[(y-1)*imgStride + x-1] - 2*imgData[(y-1)*imgStride + x] - imgData[(y-1)*imgStride + x+1]
                     +imgData[(y+1)*imgStride + x-1] + 2*imgData[(y+1)*imgStride + x] + imgData[(y+1)*imgStride + x+1];
            }
            gradient[y * width + x] = std::sqrt(gx*gx + gy*gy);
        }
    }

    // Copy markers to result labels
    result.labels = QImage(width, height, PixelType::Int16, ChannelType::Gray);
    int16_t* labelData = static_cast<int16_t*>(result.labels.Data());
    const int16_t* markerData = static_cast<const int16_t*>(markers.Data());
    size_t markerStride = markers.Stride() / sizeof(int16_t);
    size_t labelStride = result.labels.Stride() / sizeof(int16_t);

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            labelData[y * labelStride + x] = markerData[y * markerStride + x];
        }
    }

    // Watershed flooding
    // Use continuous memory for flooding
    std::vector<int16_t> labels(width * height);
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            labels[y * width + x] = labelData[y * labelStride + x];
        }
    }

    WatershedFlood(gradient.data(), width, height, labels.data(), nullptr, 0);

    // Copy back
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            labelData[y * labelStride + x] = labels[y * width + x];
        }
    }

    // Extract regions
    ExtractWatershedRegions(labels.data(), width, height, result);

    return result;
}

WatershedResult WatershedBinary(const QImage& binaryImage, double minDistance) {
    WatershedResult result;

    if (binaryImage.Empty()) {
        return result;
    }
    if (!RequireGrayU8(binaryImage, "WatershedBinary")) {
        return result;
    }

    int32_t width = binaryImage.Width();
    int32_t height = binaryImage.Height();
    const uint8_t* binData = static_cast<const uint8_t*>(binaryImage.Data());
    size_t binStride = binaryImage.Stride();

    // Compute distance transform
    std::vector<float> dist(width * height);
    ComputeDistanceTransformInternal(binData, binStride, dist.data(), width, height);

    // Find markers from distance peaks
    std::vector<std::pair<int32_t, int32_t>> peaks;
    FindLocalMaxima(dist.data(), width, height, minDistance, minDistance / 2.0, peaks);

    if (peaks.empty()) {
        // No peaks found, return single region
        result.labels = QImage(width, height, PixelType::Int16, ChannelType::Gray);
        int16_t* labelData = static_cast<int16_t*>(result.labels.Data());
        size_t labelStride = result.labels.Stride() / sizeof(int16_t);

        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                labelData[y * labelStride + x] = (binData[y * binStride + x] > 0) ? 1 : 0;
            }
        }

        // Create single region from binary
        std::vector<QRegion::Run> runs;
        for (int32_t y = 0; y < height; ++y) {
            int32_t runStart = -1;
            for (int32_t x = 0; x <= width; ++x) {
                bool fg = (x < width) && (binData[y * binStride + x] > 0);
                if (fg && runStart < 0) {
                    runStart = x;
                } else if (!fg && runStart >= 0) {
                    runs.push_back({y, runStart, x});  // colEnd is exclusive
                    runStart = -1;
                }
            }
        }
        if (!runs.empty()) {
            result.regions.push_back(QRegion(std::move(runs)));
        }
        result.numRegions = static_cast<int32_t>(result.regions.size());
        return result;
    }

    // Initialize labels with markers
    std::vector<int16_t> labels(width * height, 0);
    for (size_t i = 0; i < peaks.size(); ++i) {
        labels[peaks[i].second * width + peaks[i].first] = static_cast<int16_t>(i + 1);
    }

    // Use negative distance for flooding (flood from peaks outward)
    std::vector<float> negDist(width * height);
    for (size_t i = 0; i < dist.size(); ++i) {
        negDist[i] = -dist[i];
    }

    // Watershed flooding
    WatershedFlood(negDist.data(), width, height, labels.data(), binData, binStride);

    // Create result
    result.labels = QImage(width, height, PixelType::Int16, ChannelType::Gray);
    int16_t* labelData = static_cast<int16_t*>(result.labels.Data());
    size_t labelStride = result.labels.Stride() / sizeof(int16_t);

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            labelData[y * labelStride + x] = labels[y * width + x];
        }
    }

    ExtractWatershedRegions(labels.data(), width, height, result);

    return result;
}

WatershedResult WatershedRegion(const QRegion& region, double minDistance) {
    WatershedResult result;

    if (region.Empty()) {
        return result;
    }

    // Get bounding box
    auto bbox = region.BoundingBox();
    int32_t width = bbox.width;
    int32_t height = bbox.height;

    // Create binary image from region
    QImage binary(width, height, PixelType::UInt8, ChannelType::Gray);
    std::memset(binary.Data(), 0, binary.Height() * binary.Stride());

    uint8_t* binData = static_cast<uint8_t*>(binary.Data());
    size_t stride = binary.Stride();

    for (const auto& run : region.Runs()) {
        int32_t y = run.row - bbox.y;
        if (y < 0 || y >= height) continue;
        int32_t x0 = std::max(0, run.colBegin - bbox.x);
        int32_t x1 = std::min(width - 1, run.colEnd - bbox.x);
        for (int32_t x = x0; x <= x1; ++x) {
            binData[y * stride + x] = 255;
        }
    }

    // Apply watershed
    WatershedResult localResult = WatershedBinary(binary, minDistance);

    // Offset regions back to original coordinates
    result.labels = std::move(localResult.labels);
    result.numRegions = localResult.numRegions;
    result.regions.resize(localResult.regions.size());

    for (size_t i = 0; i < localResult.regions.size(); ++i) {
        if (!localResult.regions[i].Empty()) {
            result.regions[i] = localResult.regions[i].Translate(bbox.x, bbox.y);
        }
    }

    if (!localResult.watershedLines.Empty()) {
        result.watershedLines = localResult.watershedLines.Translate(bbox.x, bbox.y);
    }

    return result;
}

WatershedResult WatershedGradient(const QImage& image, const QImage* markers,
                                  double gradientThreshold) {
    WatershedResult result;

    if (image.Empty()) {
        return result;
    }
    if (!RequireGrayU8(image, "WatershedGradient")) {
        return result;
    }

    int32_t width = image.Width();
    int32_t height = image.Height();
    const uint8_t* imgData = static_cast<const uint8_t*>(image.Data());
    size_t imgStride = image.Stride();

    // Compute gradient magnitude
    std::vector<float> gradient(width * height);
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            float gx = 0, gy = 0;
            if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                gx = static_cast<float>(imgData[y*imgStride + x+1]) -
                     static_cast<float>(imgData[y*imgStride + x-1]);
                gy = static_cast<float>(imgData[(y+1)*imgStride + x]) -
                     static_cast<float>(imgData[(y-1)*imgStride + x]);
            }
            gradient[y * width + x] = std::sqrt(gx*gx + gy*gy);
        }
    }

    // Initialize labels
    std::vector<int16_t> labels(width * height, 0);

    if (markers != nullptr && !markers->Empty()) {
        // Use provided markers
        if (markers->Type() != PixelType::Int16) {
            throw InvalidArgumentException("WatershedGradient: markers must be Int16");
        }
        const int16_t* markerData = static_cast<const int16_t*>(markers->Data());
        size_t markerStride = markers->Stride() / sizeof(int16_t);
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                labels[y * width + x] = markerData[y * markerStride + x];
            }
        }
    } else {
        // Auto-generate markers from local minima in gradient
        int16_t label = 1;
        for (int32_t y = 1; y < height - 1; ++y) {
            for (int32_t x = 1; x < width - 1; ++x) {
                float val = gradient[y * width + x];
                if (val > gradientThreshold) continue;

                bool isMin = true;
                for (int i = 0; i < 8 && isMin; ++i) {
                    int32_t nx = x + dx8[i];
                    int32_t ny = y + dy8[i];
                    if (gradient[ny * width + nx] < val) {
                        isMin = false;
                    }
                }

                if (isMin) {
                    labels[y * width + x] = label++;
                }
            }
        }
    }

    // Watershed flooding
    WatershedFlood(gradient.data(), width, height, labels.data(), nullptr, 0);

    // Create result
    result.labels = QImage(width, height, PixelType::Int16, ChannelType::Gray);
    int16_t* labelData = static_cast<int16_t*>(result.labels.Data());
    size_t labelStride = result.labels.Stride() / sizeof(int16_t);

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            labelData[y * labelStride + x] = labels[y * width + x];
        }
    }

    ExtractWatershedRegions(labels.data(), width, height, result);

    return result;
}

} // namespace Qi::Vision::Segment
