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
#include <QiVision/Core/Validate.h>

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
    return Validate::RequireImageU8Gray(image, funcName);
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

    if (!Validate::RequireImageU8(image, "KMeans")) {
        return result;
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
    if (!RequireGrayU8(binaryImage, "DistanceTransform")) {
        return QImage();
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

    if (!RequireGrayU8(image, "Watershed") || markers.Empty()) {
        return result;
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

// =============================================================================
// GMM (Gaussian Mixture Model) Implementation
// =============================================================================

namespace {

// GMM Gaussian component
struct GMMComponent {
    std::vector<double> mean;           // Mean vector
    std::vector<double> covariance;     // Covariance (flattened)
    std::vector<double> covInverse;     // Inverse covariance (for full)
    double covDet = 1.0;                // Determinant of covariance
    double weight = 0.0;                // Mixture weight

    void Init(int32_t dim, GMMCovType covType) {
        mean.resize(dim, 0.0);
        switch (covType) {
            case GMMCovType::Full:
                covariance.resize(dim * dim, 0.0);
                covInverse.resize(dim * dim, 0.0);
                // Initialize to identity
                for (int32_t i = 0; i < dim; ++i) {
                    covariance[i * dim + i] = 1.0;
                    covInverse[i * dim + i] = 1.0;
                }
                break;
            case GMMCovType::Diagonal:
                covariance.resize(dim, 1.0);  // Diagonal elements only
                covInverse.resize(dim, 1.0);
                break;
            case GMMCovType::Spherical:
                covariance.resize(1, 1.0);    // Single variance
                covInverse.resize(1, 1.0);
                break;
        }
        covDet = 1.0;
    }
};

// Compute log of Gaussian PDF
double GMMLogPdf(const double* x, const GMMComponent& comp, int32_t dim,
                 GMMCovType covType) {
    // Compute (x - mean)
    std::vector<double> diff(dim);
    for (int32_t i = 0; i < dim; ++i) {
        diff[i] = x[i] - comp.mean[i];
    }

    // Compute Mahalanobis distance: (x-mu)^T * Sigma^-1 * (x-mu)
    double mahal = 0.0;

    switch (covType) {
        case GMMCovType::Full: {
            // Full covariance: diff^T * covInverse * diff
            for (int32_t i = 0; i < dim; ++i) {
                double sum = 0.0;
                for (int32_t j = 0; j < dim; ++j) {
                    sum += comp.covInverse[i * dim + j] * diff[j];
                }
                mahal += diff[i] * sum;
            }
            break;
        }
        case GMMCovType::Diagonal: {
            // Diagonal: sum of diff[i]^2 / var[i]
            for (int32_t i = 0; i < dim; ++i) {
                mahal += diff[i] * diff[i] * comp.covInverse[i];
            }
            break;
        }
        case GMMCovType::Spherical: {
            // Spherical: sum of diff[i]^2 / var
            double invVar = comp.covInverse[0];
            for (int32_t i = 0; i < dim; ++i) {
                mahal += diff[i] * diff[i] * invVar;
            }
            break;
        }
    }

    // log N(x|mu,Sigma) = -0.5 * (d*log(2*pi) + log|Sigma| + mahal)
    static const double LOG_2PI = 1.8378770664093453;  // log(2*pi)
    double logPdf = -0.5 * (dim * LOG_2PI + std::log(std::max(comp.covDet, 1e-300)) + mahal);

    return logPdf;
}

// Compute inverse and determinant of symmetric positive-definite matrix
bool InvertSymmetricMatrix(const double* A, double* Ainv, double& det, int32_t n,
                           double regularization) {
    // Simple Cholesky decomposition for small matrices
    // A = L * L^T, then A^-1 = L^-T * L^-1

    std::vector<double> L(n * n, 0.0);

    // Cholesky decomposition with regularization
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j <= i; ++j) {
            double sum = A[i * n + j];
            for (int32_t k = 0; k < j; ++k) {
                sum -= L[i * n + k] * L[j * n + k];
            }
            if (i == j) {
                // Diagonal element with regularization
                sum += regularization;
                if (sum <= 0) {
                    return false;  // Not positive definite
                }
                L[i * n + j] = std::sqrt(sum);
            } else {
                L[i * n + j] = sum / L[j * n + j];
            }
        }
    }

    // Determinant = product of L[i][i]^2
    det = 1.0;
    for (int32_t i = 0; i < n; ++i) {
        det *= L[i * n + i] * L[i * n + i];
    }

    // Invert L (lower triangular)
    std::vector<double> Linv(n * n, 0.0);
    for (int32_t i = 0; i < n; ++i) {
        Linv[i * n + i] = 1.0 / L[i * n + i];
        for (int32_t j = i + 1; j < n; ++j) {
            double sum = 0.0;
            for (int32_t k = i; k < j; ++k) {
                sum -= L[j * n + k] * Linv[k * n + i];
            }
            Linv[j * n + i] = sum / L[j * n + j];
        }
    }

    // A^-1 = L^-T * L^-1
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int32_t k = std::max(i, j); k < n; ++k) {
                sum += Linv[k * n + i] * Linv[k * n + j];
            }
            Ainv[i * n + j] = sum;
        }
    }

    return true;
}

// Update component covariance and compute inverse/determinant
void UpdateComponentCovariance(GMMComponent& comp, int32_t dim, GMMCovType covType,
                               double regularization) {
    switch (covType) {
        case GMMCovType::Full: {
            // Invert full covariance matrix
            if (!InvertSymmetricMatrix(comp.covariance.data(), comp.covInverse.data(),
                                       comp.covDet, dim, regularization)) {
                // Reset to identity if singular
                comp.covDet = 1.0;
                for (int32_t i = 0; i < dim; ++i) {
                    for (int32_t j = 0; j < dim; ++j) {
                        comp.covariance[i * dim + j] = (i == j) ? 1.0 : 0.0;
                        comp.covInverse[i * dim + j] = (i == j) ? 1.0 : 0.0;
                    }
                }
            }
            break;
        }
        case GMMCovType::Diagonal: {
            // Diagonal: determinant = product, inverse = 1/var
            comp.covDet = 1.0;
            for (int32_t i = 0; i < dim; ++i) {
                double var = std::max(comp.covariance[i], regularization);
                comp.covariance[i] = var;
                comp.covInverse[i] = 1.0 / var;
                comp.covDet *= var;
            }
            break;
        }
        case GMMCovType::Spherical: {
            // Spherical: single variance
            double var = std::max(comp.covariance[0], regularization);
            comp.covariance[0] = var;
            comp.covInverse[0] = 1.0 / var;
            comp.covDet = std::pow(var, dim);
            break;
        }
    }
}

// Extract features from image (reuse from K-Means)
void ExtractGMMFeatures(const QImage& image, GMMFeature feature, double spatialWeight,
                        std::vector<std::vector<double>>& samples, int32_t& featureDim) {
    int32_t width = image.Width();
    int32_t height = image.Height();
    int32_t channels = image.Channels();
    bool isColor = (channels >= 3);

    // Determine feature dimension
    switch (feature) {
        case GMMFeature::Gray:
            featureDim = 1;
            break;
        case GMMFeature::RGB:
        case GMMFeature::HSV:
        case GMMFeature::Lab:
            featureDim = 3;
            break;
        case GMMFeature::GraySpatial:
            featureDim = 3;  // intensity + x + y
            break;
        case GMMFeature::RGBSpatial:
            featureDim = 5;  // RGB + x + y
            break;
    }

    samples.resize(width * height);

    // Spatial normalization factors
    double xNorm = (width > 1) ? 255.0 / (width - 1) : 1.0;
    double yNorm = (height > 1) ? 255.0 / (height - 1) : 1.0;

    const uint8_t* data = static_cast<const uint8_t*>(image.Data());
    size_t stride = image.Stride();

    for (int32_t y = 0; y < height; ++y) {
        const uint8_t* row = data + y * stride;

        for (int32_t x = 0; x < width; ++x) {
            std::vector<double>& sample = samples[y * width + x];
            sample.resize(featureDim);

            switch (feature) {
                case GMMFeature::Gray: {
                    if (isColor) {
                        // Convert RGB to grayscale
                        double r = row[x * channels];
                        double g = row[x * channels + 1];
                        double b = row[x * channels + 2];
                        sample[0] = 0.299 * r + 0.587 * g + 0.114 * b;
                    } else {
                        sample[0] = row[x];
                    }
                    break;
                }
                case GMMFeature::RGB: {
                    if (isColor) {
                        sample[0] = row[x * channels];
                        sample[1] = row[x * channels + 1];
                        sample[2] = row[x * channels + 2];
                    } else {
                        sample[0] = sample[1] = sample[2] = row[x];
                    }
                    break;
                }
                case GMMFeature::HSV: {
                    double r, g, b;
                    if (isColor) {
                        r = row[x * channels] / 255.0;
                        g = row[x * channels + 1] / 255.0;
                        b = row[x * channels + 2] / 255.0;
                    } else {
                        r = g = b = row[x] / 255.0;
                    }
                    double maxC = std::max({r, g, b});
                    double minC = std::min({r, g, b});
                    double delta = maxC - minC;

                    // V
                    double v = maxC;

                    // S
                    double s = (maxC > 0) ? delta / maxC : 0;

                    // H
                    double h = 0;
                    if (delta > 0) {
                        if (maxC == r) {
                            h = 60.0 * std::fmod((g - b) / delta + 6.0, 6.0);
                        } else if (maxC == g) {
                            h = 60.0 * ((b - r) / delta + 2.0);
                        } else {
                            h = 60.0 * ((r - g) / delta + 4.0);
                        }
                    }

                    sample[0] = h / 360.0 * 255.0;
                    sample[1] = s * 255.0;
                    sample[2] = v * 255.0;
                    break;
                }
                case GMMFeature::Lab: {
                    double r, g, b;
                    if (isColor) {
                        r = row[x * channels] / 255.0;
                        g = row[x * channels + 1] / 255.0;
                        b = row[x * channels + 2] / 255.0;
                    } else {
                        r = g = b = row[x] / 255.0;
                    }

                    // Gamma correction
                    auto gamma = [](double c) {
                        return (c > 0.04045) ? std::pow((c + 0.055) / 1.055, 2.4) : c / 12.92;
                    };
                    r = gamma(r);
                    g = gamma(g);
                    b = gamma(b);

                    // RGB to XYZ (D65)
                    double X = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047;
                    double Y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b);
                    double Z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883;

                    // XYZ to Lab
                    auto f = [](double t) {
                        const double delta = 6.0 / 29.0;
                        return (t > delta * delta * delta) ?
                               std::cbrt(t) : t / (3 * delta * delta) + 4.0 / 29.0;
                    };
                    double fX = f(X);
                    double fY = f(Y);
                    double fZ = f(Z);

                    sample[0] = (116.0 * fY - 16.0);       // L: [0, 100]
                    sample[1] = (500.0 * (fX - fY) + 128); // a: [-128, 127] -> [0, 255]
                    sample[2] = (200.0 * (fY - fZ) + 128); // b: [-128, 127] -> [0, 255]
                    break;
                }
                case GMMFeature::GraySpatial: {
                    if (isColor) {
                        double r = row[x * channels];
                        double g = row[x * channels + 1];
                        double b = row[x * channels + 2];
                        sample[0] = 0.299 * r + 0.587 * g + 0.114 * b;
                    } else {
                        sample[0] = row[x];
                    }
                    sample[1] = x * xNorm * spatialWeight;
                    sample[2] = y * yNorm * spatialWeight;
                    break;
                }
                case GMMFeature::RGBSpatial: {
                    if (isColor) {
                        sample[0] = row[x * channels];
                        sample[1] = row[x * channels + 1];
                        sample[2] = row[x * channels + 2];
                    } else {
                        sample[0] = sample[1] = sample[2] = row[x];
                    }
                    sample[3] = x * xNorm * spatialWeight;
                    sample[4] = y * yNorm * spatialWeight;
                    break;
                }
            }
        }
    }
}

// Initialize GMM with K-Means
void InitGMMWithKMeans(const std::vector<std::vector<double>>& samples,
                       std::vector<GMMComponent>& components, int32_t k, int32_t dim,
                       GMMCovType covType, std::mt19937& rng) {
    int64_t n = static_cast<int64_t>(samples.size());
    if (n == 0) return;

    // K-Means++ initialization for means
    std::vector<int32_t> centerIndices;
    centerIndices.reserve(k);

    // First center: random
    std::uniform_int_distribution<int64_t> dist(0, n - 1);
    centerIndices.push_back(static_cast<int32_t>(dist(rng)));

    // Remaining centers
    std::vector<double> minDistSq(n, std::numeric_limits<double>::max());

    for (int32_t c = 1; c < k; ++c) {
        // Update minimum distances to existing centers
        int32_t lastCenter = centerIndices.back();
        for (int64_t i = 0; i < n; ++i) {
            double d = 0;
            for (int32_t j = 0; j < dim; ++j) {
                double diff = samples[i][j] - samples[lastCenter][j];
                d += diff * diff;
            }
            minDistSq[i] = std::min(minDistSq[i], d);
        }

        // Sample proportional to distance squared
        double total = 0;
        for (int64_t i = 0; i < n; ++i) {
            total += minDistSq[i];
        }

        std::uniform_real_distribution<double> fdist(0, total);
        double r = fdist(rng);
        double cumSum = 0;
        int64_t chosen = n - 1;
        for (int64_t i = 0; i < n; ++i) {
            cumSum += minDistSq[i];
            if (cumSum >= r) {
                chosen = i;
                break;
            }
        }
        centerIndices.push_back(static_cast<int32_t>(chosen));
    }

    // Run a few iterations of K-Means to get initial assignments
    std::vector<int32_t> assignments(n);
    std::vector<std::vector<double>> centers(k, std::vector<double>(dim, 0));

    // Initial centers
    for (int32_t c = 0; c < k; ++c) {
        centers[c] = samples[centerIndices[c]];
    }

    // K-Means iterations
    for (int iter = 0; iter < 10; ++iter) {
        // Assign samples to nearest center
        for (int64_t i = 0; i < n; ++i) {
            double bestDist = std::numeric_limits<double>::max();
            int32_t bestC = 0;
            for (int32_t c = 0; c < k; ++c) {
                double d = 0;
                for (int32_t j = 0; j < dim; ++j) {
                    double diff = samples[i][j] - centers[c][j];
                    d += diff * diff;
                }
                if (d < bestDist) {
                    bestDist = d;
                    bestC = c;
                }
            }
            assignments[i] = bestC;
        }

        // Update centers
        std::vector<int64_t> counts(k, 0);
        for (auto& c : centers) {
            std::fill(c.begin(), c.end(), 0.0);
        }
        for (int64_t i = 0; i < n; ++i) {
            int32_t c = assignments[i];
            for (int32_t j = 0; j < dim; ++j) {
                centers[c][j] += samples[i][j];
            }
            counts[c]++;
        }
        for (int32_t c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                for (int32_t j = 0; j < dim; ++j) {
                    centers[c][j] /= counts[c];
                }
            }
        }
    }

    // Initialize GMM components from K-Means result
    for (int32_t c = 0; c < k; ++c) {
        components[c].mean = centers[c];
        components[c].weight = 1.0 / k;

        // Compute initial covariance from assigned samples
        std::vector<double> cov;
        switch (covType) {
            case GMMCovType::Full:
                cov.resize(dim * dim, 0.0);
                break;
            case GMMCovType::Diagonal:
                cov.resize(dim, 0.0);
                break;
            case GMMCovType::Spherical:
                cov.resize(1, 0.0);
                break;
        }

        int64_t count = 0;
        for (int64_t i = 0; i < n; ++i) {
            if (assignments[i] == c) {
                count++;
                switch (covType) {
                    case GMMCovType::Full:
                        for (int32_t p = 0; p < dim; ++p) {
                            for (int32_t q = 0; q < dim; ++q) {
                                cov[p * dim + q] += (samples[i][p] - centers[c][p]) *
                                                    (samples[i][q] - centers[c][q]);
                            }
                        }
                        break;
                    case GMMCovType::Diagonal:
                        for (int32_t p = 0; p < dim; ++p) {
                            double diff = samples[i][p] - centers[c][p];
                            cov[p] += diff * diff;
                        }
                        break;
                    case GMMCovType::Spherical:
                        for (int32_t p = 0; p < dim; ++p) {
                            double diff = samples[i][p] - centers[c][p];
                            cov[0] += diff * diff;
                        }
                        break;
                }
            }
        }

        // Normalize covariance
        if (count > 1) {
            switch (covType) {
                case GMMCovType::Full:
                    for (auto& v : cov) v /= count;
                    break;
                case GMMCovType::Diagonal:
                    for (auto& v : cov) v /= count;
                    break;
                case GMMCovType::Spherical:
                    cov[0] /= (count * dim);
                    break;
            }
        } else {
            // Default variance if no samples
            switch (covType) {
                case GMMCovType::Full:
                    for (int32_t p = 0; p < dim; ++p) {
                        cov[p * dim + p] = 100.0;
                    }
                    break;
                case GMMCovType::Diagonal:
                    std::fill(cov.begin(), cov.end(), 100.0);
                    break;
                case GMMCovType::Spherical:
                    cov[0] = 100.0;
                    break;
            }
        }

        components[c].covariance = cov;
    }
}

// Initialize GMM randomly
void InitGMMRandom(const std::vector<std::vector<double>>& samples,
                   std::vector<GMMComponent>& components, int32_t k, int32_t dim,
                   GMMCovType covType, std::mt19937& rng) {
    int64_t n = static_cast<int64_t>(samples.size());
    if (n == 0) return;

    // Compute overall mean and variance
    std::vector<double> globalMean(dim, 0.0);
    for (int64_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < dim; ++j) {
            globalMean[j] += samples[i][j];
        }
    }
    for (int32_t j = 0; j < dim; ++j) {
        globalMean[j] /= n;
    }

    std::vector<double> globalVar(dim, 0.0);
    for (int64_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < dim; ++j) {
            double diff = samples[i][j] - globalMean[j];
            globalVar[j] += diff * diff;
        }
    }
    for (int32_t j = 0; j < dim; ++j) {
        globalVar[j] /= n;
    }

    // Random initialization
    std::uniform_int_distribution<int64_t> dist(0, n - 1);

    for (int32_t c = 0; c < k; ++c) {
        // Random sample as mean
        components[c].mean = samples[dist(rng)];
        components[c].weight = 1.0 / k;

        // Initial covariance from global variance
        switch (covType) {
            case GMMCovType::Full:
                for (int32_t p = 0; p < dim; ++p) {
                    components[c].covariance[p * dim + p] = globalVar[p];
                }
                break;
            case GMMCovType::Diagonal:
                for (int32_t p = 0; p < dim; ++p) {
                    components[c].covariance[p] = globalVar[p];
                }
                break;
            case GMMCovType::Spherical: {
                double avgVar = 0;
                for (int32_t p = 0; p < dim; ++p) avgVar += globalVar[p];
                components[c].covariance[0] = avgVar / dim;
                break;
            }
        }
    }
}

} // anonymous namespace

// Main GMM function
GMMResult GMM(const QImage& image, const GMMParams& params) {
    GMMResult result;

    if (image.Empty()) {
        return result;
    }
    if (!image.IsValid()) {
        throw InvalidArgumentException("GMM: invalid image");
    }
    if (params.k < 1) {
        throw InvalidArgumentException("GMM: k must be >= 1");
    }

    int32_t width = image.Width();
    int32_t height = image.Height();
    int64_t n = static_cast<int64_t>(width) * height;
    int32_t k = params.k;

    // Extract features
    std::vector<std::vector<double>> samples;
    int32_t dim;
    ExtractGMMFeatures(image, params.feature, params.spatialWeight, samples, dim);

    // Initialize random generator
    std::random_device rd;
    std::mt19937 rng(rd());

    // Initialize components
    std::vector<GMMComponent> components(k);
    for (int32_t c = 0; c < k; ++c) {
        components[c].Init(dim, params.covType);
    }

    // Initialize using K-Means or random
    if (params.init == GMMInit::KMeans) {
        InitGMMWithKMeans(samples, components, k, dim, params.covType, rng);
    } else {
        InitGMMRandom(samples, components, k, dim, params.covType, rng);
    }

    // Update covariance inverses and determinants
    for (int32_t c = 0; c < k; ++c) {
        UpdateComponentCovariance(components[c], dim, params.covType, params.regularization);
    }

    // EM algorithm
    std::vector<std::vector<double>> responsibilities(n, std::vector<double>(k, 0.0));
    double prevLogLik = -std::numeric_limits<double>::max();

    for (int32_t iter = 0; iter < params.maxIterations; ++iter) {
        // E-step: compute responsibilities
        double logLik = 0.0;

        #pragma omp parallel for reduction(+:logLik) schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            // Compute log P(x|k) + log P(k) for each component
            std::vector<double> logProbs(k);
            double maxLogProb = -std::numeric_limits<double>::max();

            for (int32_t c = 0; c < k; ++c) {
                logProbs[c] = GMMLogPdf(samples[i].data(), components[c], dim, params.covType)
                              + std::log(std::max(components[c].weight, 1e-300));
                maxLogProb = std::max(maxLogProb, logProbs[c]);
            }

            // Log-sum-exp trick for numerical stability
            double sumExp = 0.0;
            for (int32_t c = 0; c < k; ++c) {
                sumExp += std::exp(logProbs[c] - maxLogProb);
            }
            double logSumExp = maxLogProb + std::log(sumExp);

            // Responsibilities: P(k|x) = P(x|k)*P(k) / sum_k P(x|k)*P(k)
            for (int32_t c = 0; c < k; ++c) {
                responsibilities[i][c] = std::exp(logProbs[c] - logSumExp);
            }

            logLik += logSumExp;
        }

        // Check convergence
        result.iterations = iter + 1;
        if (iter > 0 && std::abs(logLik - prevLogLik) < params.epsilon) {
            result.converged = true;
            result.logLikelihood = logLik;
            break;
        }
        prevLogLik = logLik;
        result.logLikelihood = logLik;

        // M-step: update parameters
        for (int32_t c = 0; c < k; ++c) {
            double Nk = 0.0;  // Effective number of samples for component c

            // Sum responsibilities
            for (int64_t i = 0; i < n; ++i) {
                Nk += responsibilities[i][c];
            }

            if (Nk < 1e-10) {
                // Component has negligible responsibility, reset
                std::uniform_int_distribution<int64_t> dist(0, n - 1);
                components[c].mean = samples[dist(rng)];
                components[c].weight = 1e-10;
                continue;
            }

            // Update weight
            components[c].weight = Nk / n;

            // Update mean
            std::fill(components[c].mean.begin(), components[c].mean.end(), 0.0);
            for (int64_t i = 0; i < n; ++i) {
                double r = responsibilities[i][c];
                for (int32_t j = 0; j < dim; ++j) {
                    components[c].mean[j] += r * samples[i][j];
                }
            }
            for (int32_t j = 0; j < dim; ++j) {
                components[c].mean[j] /= Nk;
            }

            // Update covariance
            switch (params.covType) {
                case GMMCovType::Full: {
                    std::fill(components[c].covariance.begin(),
                              components[c].covariance.end(), 0.0);
                    for (int64_t i = 0; i < n; ++i) {
                        double r = responsibilities[i][c];
                        for (int32_t p = 0; p < dim; ++p) {
                            double dp = samples[i][p] - components[c].mean[p];
                            for (int32_t q = 0; q < dim; ++q) {
                                double dq = samples[i][q] - components[c].mean[q];
                                components[c].covariance[p * dim + q] += r * dp * dq;
                            }
                        }
                    }
                    for (auto& v : components[c].covariance) {
                        v /= Nk;
                    }
                    break;
                }
                case GMMCovType::Diagonal: {
                    std::fill(components[c].covariance.begin(),
                              components[c].covariance.end(), 0.0);
                    for (int64_t i = 0; i < n; ++i) {
                        double r = responsibilities[i][c];
                        for (int32_t p = 0; p < dim; ++p) {
                            double d = samples[i][p] - components[c].mean[p];
                            components[c].covariance[p] += r * d * d;
                        }
                    }
                    for (auto& v : components[c].covariance) {
                        v /= Nk;
                    }
                    break;
                }
                case GMMCovType::Spherical: {
                    components[c].covariance[0] = 0.0;
                    for (int64_t i = 0; i < n; ++i) {
                        double r = responsibilities[i][c];
                        for (int32_t p = 0; p < dim; ++p) {
                            double d = samples[i][p] - components[c].mean[p];
                            components[c].covariance[0] += r * d * d;
                        }
                    }
                    components[c].covariance[0] /= (Nk * dim);
                    break;
                }
            }

            // Update covariance inverse and determinant
            UpdateComponentCovariance(components[c], dim, params.covType, params.regularization);
        }
    }

    // Create output labels (hard assignment)
    result.labels = QImage(width, height, PixelType::Int16, ChannelType::Gray);
    int16_t* labelData = static_cast<int16_t*>(result.labels.Data());
    size_t labelStride = result.labels.Stride() / sizeof(int16_t);

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            int64_t i = y * width + x;
            int16_t bestC = 0;
            double bestResp = responsibilities[i][0];
            for (int32_t c = 1; c < k; ++c) {
                if (responsibilities[i][c] > bestResp) {
                    bestResp = responsibilities[i][c];
                    bestC = static_cast<int16_t>(c);
                }
            }
            labelData[y * labelStride + x] = bestC;
        }
    }

    // Create probability maps
    result.probabilities.resize(k);
    for (int32_t c = 0; c < k; ++c) {
        result.probabilities[c] = QImage(width, height, PixelType::UInt8, ChannelType::Gray);
        uint8_t* probData = static_cast<uint8_t*>(result.probabilities[c].Data());
        size_t probStride = result.probabilities[c].Stride();

        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                int64_t i = y * width + x;
                probData[y * probStride + x] = static_cast<uint8_t>(
                    std::round(responsibilities[i][c] * 255.0));
            }
        }
    }

    // Copy model parameters
    result.weights.resize(k);
    result.means.resize(k);
    result.covariances.resize(k);
    for (int32_t c = 0; c < k; ++c) {
        result.weights[c] = components[c].weight;
        result.means[c] = components[c].mean;
        result.covariances[c] = components[c].covariance;
    }

    return result;
}

// Simple interface
GMMResult GMM(const QImage& image, int32_t k, GMMFeature feature) {
    GMMParams params;
    params.k = k;
    params.feature = feature;
    return GMM(image, params);
}

// GMM segment (recolored image)
QImage GMMSegment(const QImage& image, int32_t k, GMMFeature feature) {
    GMMParams params;
    params.k = k;
    params.feature = feature;
    return GMMSegment(image, params);
}

QImage GMMSegment(const QImage& image, const GMMParams& params) {
    GMMResult result = GMM(image, params);

    if (result.labels.Empty()) {
        return QImage();
    }

    int32_t width = result.labels.Width();
    int32_t height = result.labels.Height();
    int32_t k = params.k;

    // Generate colors for each component
    std::vector<std::array<uint8_t, 3>> colors(k);
    for (int32_t c = 0; c < k; ++c) {
        // HSV to RGB with evenly spaced hues
        double h = c * 360.0 / k;
        double s = 0.8;
        double v = 0.9;

        int hi = static_cast<int>(h / 60.0) % 6;
        double f = h / 60.0 - hi;
        double p = v * (1 - s);
        double q = v * (1 - f * s);
        double t = v * (1 - (1 - f) * s);

        double r, g, b;
        switch (hi) {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            default: r = v; g = p; b = q; break;
        }
        colors[c] = {static_cast<uint8_t>(r * 255),
                     static_cast<uint8_t>(g * 255),
                     static_cast<uint8_t>(b * 255)};
    }

    // Create output image
    QImage output(width, height, PixelType::UInt8, ChannelType::RGB);
    const int16_t* labelData = static_cast<const int16_t*>(result.labels.Data());
    size_t labelStride = result.labels.Stride() / sizeof(int16_t);
    uint8_t* outData = static_cast<uint8_t*>(output.Data());
    size_t outStride = output.Stride();

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            int16_t label = labelData[y * labelStride + x];
            if (label >= 0 && label < k) {
                outData[y * outStride + x * 3] = colors[label][0];
                outData[y * outStride + x * 3 + 1] = colors[label][1];
                outData[y * outStride + x * 3 + 2] = colors[label][2];
            }
        }
    }

    return output;
}

// GMM to regions
void GMMToRegions(const QImage& image, int32_t k, std::vector<QRegion>& regions,
                  GMMFeature feature) {
    GMMParams params;
    params.k = k;
    params.feature = feature;
    GMMToRegions(image, params, regions);
}

void GMMToRegions(const QImage& image, const GMMParams& params,
                  std::vector<QRegion>& regions) {
    GMMResult result = GMM(image, params);
    LabelsToRegions(result.labels, params.k, regions);
}

// Get probability maps
void GMMProbabilities(const QImage& image, int32_t k, std::vector<QImage>& probMaps,
                      GMMFeature feature) {
    GMMParams params;
    params.k = k;
    params.feature = feature;
    GMMProbabilities(image, params, probMaps);
}

void GMMProbabilities(const QImage& image, const GMMParams& params,
                      std::vector<QImage>& probMaps) {
    GMMResult result = GMM(image, params);
    probMaps = std::move(result.probabilities);
}

// Classify using trained model
QImage GMMClassify(const QImage& image, const GMMResult& model, GMMFeature feature) {
    if (image.Empty() || model.means.empty()) {
        return QImage();
    }

    int32_t width = image.Width();
    int32_t height = image.Height();
    int32_t k = static_cast<int32_t>(model.means.size());
    int32_t dim = static_cast<int32_t>(model.means[0].size());

    // Determine spatial weight (default 0.5 for spatial features)
    double spatialWeight = 0.5;

    // Extract features
    std::vector<std::vector<double>> samples;
    int32_t extractedDim;
    ExtractGMMFeatures(image, feature, spatialWeight, samples, extractedDim);

    if (extractedDim != dim) {
        throw InvalidArgumentException("GMMClassify: feature dimension mismatch");
    }

    // Determine covariance type from model
    GMMCovType covType;
    if (model.covariances.empty() || model.covariances[0].empty()) {
        covType = GMMCovType::Spherical;
    } else if (model.covariances[0].size() == 1) {
        covType = GMMCovType::Spherical;
    } else if (model.covariances[0].size() == static_cast<size_t>(dim)) {
        covType = GMMCovType::Diagonal;
    } else {
        covType = GMMCovType::Full;
    }

    // Reconstruct components
    std::vector<GMMComponent> components(k);
    for (int32_t c = 0; c < k; ++c) {
        components[c].Init(dim, covType);
        components[c].mean = model.means[c];
        components[c].weight = model.weights[c];
        components[c].covariance = model.covariances[c];
        UpdateComponentCovariance(components[c], dim, covType, 1e-6);
    }

    // Classify each pixel
    QImage labels(width, height, PixelType::Int16, ChannelType::Gray);
    int16_t* labelData = static_cast<int16_t*>(labels.Data());
    size_t labelStride = labels.Stride() / sizeof(int16_t);

    int64_t n = static_cast<int64_t>(width) * height;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        double bestLogProb = -std::numeric_limits<double>::max();
        int16_t bestC = 0;

        for (int32_t c = 0; c < k; ++c) {
            double logProb = GMMLogPdf(samples[i].data(), components[c], dim, covType)
                            + std::log(std::max(components[c].weight, 1e-300));
            if (logProb > bestLogProb) {
                bestLogProb = logProb;
                bestC = static_cast<int16_t>(c);
            }
        }

        int32_t y = static_cast<int32_t>(i / width);
        int32_t x = static_cast<int32_t>(i % width);
        labelData[y * labelStride + x] = bestC;
    }

    return labels;
}

} // namespace Qi::Vision::Segment
