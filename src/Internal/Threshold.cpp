/**
 * @file Threshold.cpp
 * @brief Image thresholding implementation
 */

#include <QiVision/Internal/Threshold.h>
#include <QiVision/Internal/Histogram.h>
#include <QiVision/Internal/Gaussian.h>
#include <QiVision/Internal/Convolution.h>
#include <QiVision/Core/Exception.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <queue>

namespace Qi::Vision::Internal {

// ============================================================================
// Utility Functions Implementation
// ============================================================================

void ComputeIntegralImages(const QImage& src,
                           std::vector<double>& integralSum,
                           std::vector<double>& integralSqSum) {
    if (src.Empty()) {
        integralSum.clear();
        integralSqSum.clear();
        return;
    }

    int32_t width = src.Width();
    int32_t height = src.Height();
    int32_t size = (width + 1) * (height + 1);

    integralSum.resize(size, 0.0);
    integralSqSum.resize(size, 0.0);

    const uint8_t* data = static_cast<const uint8_t*>(src.Data());

    // Build integral images with padding for easy boundary handling
    for (int32_t y = 1; y <= height; ++y) {
        double rowSum = 0.0;
        double rowSqSum = 0.0;
        for (int32_t x = 1; x <= width; ++x) {
            double val = static_cast<double>(data[(y - 1) * width + (x - 1)]);
            rowSum += val;
            rowSqSum += val * val;

            int32_t idx = y * (width + 1) + x;
            int32_t idxAbove = (y - 1) * (width + 1) + x;

            integralSum[idx] = rowSum + integralSum[idxAbove];
            integralSqSum[idx] = rowSqSum + integralSqSum[idxAbove];
        }
    }
}

void GetBlockStats(const std::vector<double>& integralSum,
                   const std::vector<double>& integralSqSum,
                   int32_t width, int32_t height,
                   int32_t x, int32_t y, int32_t halfSize,
                   double& mean, double& stddev) {
    // Compute block bounds (clamped to image)
    int32_t x0 = std::max(0, x - halfSize);
    int32_t y0 = std::max(0, y - halfSize);
    int32_t x1 = std::min(width, x + halfSize + 1);
    int32_t y1 = std::min(height, y + halfSize + 1);

    int32_t count = (x1 - x0) * (y1 - y0);
    if (count <= 0) {
        mean = 0;
        stddev = 0;
        return;
    }

    // Integral image has padding of 1
    int32_t stride = width + 1;

    // Get sum using integral image
    double sum = integralSum[y1 * stride + x1]
               - integralSum[y0 * stride + x1]
               - integralSum[y1 * stride + x0]
               + integralSum[y0 * stride + x0];

    double sqSum = integralSqSum[y1 * stride + x1]
                 - integralSqSum[y0 * stride + x1]
                 - integralSqSum[y1 * stride + x0]
                 + integralSqSum[y0 * stride + x0];

    mean = sum / count;
    double variance = (sqSum / count) - (mean * mean);
    stddev = variance > 0 ? std::sqrt(variance) : 0;
}

// ============================================================================
// Global Thresholding Implementation
// ============================================================================

void ThresholdGlobal(const QImage& src, QImage& dst,
                     double threshold, double maxValue,
                     ThresholdType type) {
    if (src.Empty()) {
        dst = QImage();
        return;
    }

    int32_t width = src.Width();
    int32_t height = src.Height();

    if (dst.Empty() || dst.Width() != width || dst.Height() != height) {
        dst = QImage(width, height, PixelType::UInt8, ChannelType::Gray);
    }

    const uint8_t* srcData = static_cast<const uint8_t*>(src.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());
    size_t count = static_cast<size_t>(width * height);

    ThresholdGlobal(srcData, dstData, count, threshold, maxValue, type);
}

QImage ThresholdGlobal(const QImage& src, double threshold,
                       double maxValue, ThresholdType type) {
    QImage dst;
    ThresholdGlobal(src, dst, threshold, maxValue, type);
    return dst;
}

void ThresholdAbove(const QImage& src, QImage& dst,
                    double threshold, double maxValue) {
    ThresholdGlobal(src, dst, threshold, maxValue, ThresholdType::Binary);
}

void ThresholdBelow(const QImage& src, QImage& dst,
                    double threshold, double maxValue) {
    // Pixels < threshold become maxValue
    ThresholdGlobal(src, dst, threshold - 1, maxValue, ThresholdType::BinaryInv);
}

void ThresholdRange(const QImage& src, QImage& dst,
                    double low, double high, double maxValue) {
    if (src.Empty()) {
        dst = QImage();
        return;
    }

    int32_t width = src.Width();
    int32_t height = src.Height();

    if (dst.Empty() || dst.Width() != width || dst.Height() != height) {
        dst = QImage(width, height, PixelType::UInt8, ChannelType::Gray);
    }

    const uint8_t* srcData = static_cast<const uint8_t*>(src.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

    uint8_t lowThresh = static_cast<uint8_t>(std::max(0.0, std::min(255.0, low)));
    uint8_t highThresh = static_cast<uint8_t>(std::max(0.0, std::min(255.0, high)));
    uint8_t maxVal = static_cast<uint8_t>(std::max(0.0, std::min(255.0, maxValue)));

    for (int32_t i = 0; i < width * height; ++i) {
        uint8_t val = srcData[i];
        dstData[i] = (val >= lowThresh && val <= highThresh) ? maxVal : 0;
    }
}

QImage ThresholdRange(const QImage& src, double low, double high, double maxValue) {
    QImage dst;
    ThresholdRange(src, dst, low, high, maxValue);
    return dst;
}

// ============================================================================
// Adaptive Thresholding Implementation
// ============================================================================

void ThresholdAdaptive(const QImage& src, QImage& dst,
                       const AdaptiveThresholdParams& params) {
    if (src.Empty()) {
        dst = QImage();
        return;
    }

    int32_t width = src.Width();
    int32_t height = src.Height();

    if (dst.Empty() || dst.Width() != width || dst.Height() != height) {
        dst = QImage(width, height, PixelType::UInt8, ChannelType::Gray);
    }

    const uint8_t* srcData = static_cast<const uint8_t*>(src.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

    int32_t blockSize = params.blockSize | 1;  // Ensure odd
    int32_t halfSize = blockSize / 2;

    // Compute integral images
    std::vector<double> integralSum, integralSqSum;
    ComputeIntegralImages(src, integralSum, integralSqSum);

    uint8_t maxVal = static_cast<uint8_t>(std::max(0.0, std::min(255.0, params.maxValue)));

    // For Gaussian method, compute Gaussian blurred image
    std::vector<float> gaussianBlurred;
    if (params.method == AdaptiveMethod::Gaussian) {
        // Compute Gaussian blur using separable convolution
        double sigma = blockSize / 6.0;  // Approximate sigma from block size
        int32_t kernelSize = blockSize;
        std::vector<double> kernel = Gaussian::Kernel1D(sigma, kernelSize);

        gaussianBlurred.resize(width * height);
        std::vector<float> temp(width * height);
        std::vector<float> srcFloat(width * height);

        // Convert to float
        for (int32_t i = 0; i < width * height; ++i) {
            srcFloat[i] = static_cast<float>(srcData[i]);
        }

        // Apply separable Gaussian convolution
        ConvolveSeparable(srcFloat.data(), gaussianBlurred.data(), width, height,
                          kernel.data(), kernelSize, kernel.data(), kernelSize,
                          BorderMode::Reflect101);
    }

    // Apply adaptive threshold for each pixel
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            int32_t idx = y * width + x;
            uint8_t pixelValue = srcData[idx];

            double localMean, localStddev;
            GetBlockStats(integralSum, integralSqSum, width, height,
                          x, y, halfSize, localMean, localStddev);

            double threshold;

            switch (params.method) {
                case AdaptiveMethod::Mean:
                    threshold = localMean - params.C;
                    break;

                case AdaptiveMethod::Gaussian:
                    threshold = gaussianBlurred[idx] - params.C;
                    break;

                case AdaptiveMethod::Sauvola:
                    // T = mean * (1 + k * (stddev/R - 1))
                    threshold = localMean * (1.0 + params.k * (localStddev / params.R - 1.0));
                    break;

                case AdaptiveMethod::Niblack:
                    // T = mean + k * stddev
                    threshold = localMean + params.k * localStddev;
                    break;

                default:
                    threshold = localMean - params.C;
                    break;
            }

            dstData[idx] = (pixelValue > threshold) ? maxVal : 0;
        }
    }
}

QImage ThresholdAdaptive(const QImage& src,
                         const AdaptiveThresholdParams& params) {
    QImage dst;
    ThresholdAdaptive(src, dst, params);
    return dst;
}

QImage ComputeLocalThresholdMap(const QImage& src,
                                 const AdaptiveThresholdParams& params) {
    if (src.Empty()) {
        return QImage();
    }

    int32_t width = src.Width();
    int32_t height = src.Height();

    QImage thresholdMap(width, height, PixelType::Float32, ChannelType::Gray);
    float* mapData = static_cast<float*>(thresholdMap.Data());

    int32_t blockSize = params.blockSize | 1;
    int32_t halfSize = blockSize / 2;

    std::vector<double> integralSum, integralSqSum;
    ComputeIntegralImages(src, integralSum, integralSqSum);

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            int32_t idx = y * width + x;

            double localMean, localStddev;
            GetBlockStats(integralSum, integralSqSum, width, height,
                          x, y, halfSize, localMean, localStddev);

            double threshold;

            switch (params.method) {
                case AdaptiveMethod::Mean:
                    threshold = localMean - params.C;
                    break;

                case AdaptiveMethod::Gaussian:
                    // For map, use mean as approximation
                    threshold = localMean - params.C;
                    break;

                case AdaptiveMethod::Sauvola:
                    threshold = localMean * (1.0 + params.k * (localStddev / params.R - 1.0));
                    break;

                case AdaptiveMethod::Niblack:
                    threshold = localMean + params.k * localStddev;
                    break;

                default:
                    threshold = localMean - params.C;
                    break;
            }

            mapData[idx] = static_cast<float>(threshold);
        }
    }

    return thresholdMap;
}

// ============================================================================
// Multi-level Thresholding Implementation
// ============================================================================

void ThresholdMultiLevel(const QImage& src, QImage& dst,
                         const std::vector<double>& thresholds) {
    if (src.Empty() || thresholds.empty()) {
        dst = src;
        return;
    }

    int32_t numLevels = static_cast<int32_t>(thresholds.size()) + 1;
    std::vector<double> outputValues(numLevels);

    // Default output values: evenly spaced from 0 to 255
    for (int32_t i = 0; i < numLevels; ++i) {
        outputValues[i] = static_cast<double>(i * 255 / (numLevels - 1));
    }

    ThresholdMultiLevel(src, dst, thresholds, outputValues);
}

void ThresholdMultiLevel(const QImage& src, QImage& dst,
                         const std::vector<double>& thresholds,
                         const std::vector<double>& outputValues) {
    if (src.Empty()) {
        dst = QImage();
        return;
    }

    int32_t width = src.Width();
    int32_t height = src.Height();

    if (dst.Empty() || dst.Width() != width || dst.Height() != height) {
        dst = QImage(width, height, PixelType::UInt8, ChannelType::Gray);
    }

    if (thresholds.empty() || outputValues.size() < thresholds.size() + 1) {
        std::memcpy(dst.Data(), src.Data(), width * height);
        return;
    }

    // Sort thresholds
    std::vector<double> sortedThresh = thresholds;
    std::sort(sortedThresh.begin(), sortedThresh.end());

    const uint8_t* srcData = static_cast<const uint8_t*>(src.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

    for (int32_t i = 0; i < width * height; ++i) {
        double val = static_cast<double>(srcData[i]);

        // Find which level this pixel belongs to
        int32_t level = 0;
        for (size_t t = 0; t < sortedThresh.size(); ++t) {
            if (val > sortedThresh[t]) {
                level = static_cast<int32_t>(t + 1);
            } else {
                break;
            }
        }

        dstData[i] = static_cast<uint8_t>(std::max(0.0, std::min(255.0, outputValues[level])));
    }
}

QImage ThresholdMultiLevel(const QImage& src,
                           const std::vector<double>& thresholds) {
    QImage dst;
    ThresholdMultiLevel(src, dst, thresholds);
    return dst;
}

// ============================================================================
// Auto Threshold Implementation
// ============================================================================

void ThresholdAuto(const QImage& src, QImage& dst,
                   AutoThresholdMethod method,
                   double maxValue,
                   double* computedThreshold) {
    if (src.Empty()) {
        dst = QImage();
        if (computedThreshold) *computedThreshold = 0;
        return;
    }

    Histogram hist = ComputeHistogram(src);
    double threshold = 128;

    switch (method) {
        case AutoThresholdMethod::Otsu:
            threshold = ComputeOtsuThreshold(hist);
            break;

        case AutoThresholdMethod::Triangle:
            threshold = ComputeTriangleThreshold(hist);
            break;

        case AutoThresholdMethod::MinError:
            threshold = ComputeMinErrorThreshold(hist);
            break;

        case AutoThresholdMethod::Isodata:
            threshold = ComputeIsodataThreshold(hist);
            break;

        case AutoThresholdMethod::Median:
            threshold = ComputePercentile(hist, 50.0);
            break;

        default:
            threshold = ComputeOtsuThreshold(hist);
            break;
    }

    if (computedThreshold) {
        *computedThreshold = threshold;
    }

    ThresholdGlobal(src, dst, threshold, maxValue, ThresholdType::Binary);
}

void ThresholdOtsu(const QImage& src, QImage& dst,
                   double maxValue, double* computedThreshold) {
    ThresholdAuto(src, dst, AutoThresholdMethod::Otsu, maxValue, computedThreshold);
}

void ThresholdTriangle(const QImage& src, QImage& dst,
                       double maxValue, double* computedThreshold) {
    ThresholdAuto(src, dst, AutoThresholdMethod::Triangle, maxValue, computedThreshold);
}

QImage ThresholdOtsu(const QImage& src, double maxValue) {
    QImage dst;
    ThresholdOtsu(src, dst, maxValue, nullptr);
    return dst;
}

QImage ThresholdTriangle(const QImage& src, double maxValue) {
    QImage dst;
    ThresholdTriangle(src, dst, maxValue, nullptr);
    return dst;
}

// ============================================================================
// Threshold to Region Implementation
// ============================================================================

QRegion ThresholdToRegion(const QImage& src, double low, double high) {
    if (src.Empty()) {
        return QRegion();
    }

    int32_t width = src.Width();
    int32_t height = src.Height();

    const uint8_t* data = static_cast<const uint8_t*>(src.Data());

    uint8_t lowThresh = static_cast<uint8_t>(std::max(0.0, std::min(255.0, low)));
    uint8_t highThresh = static_cast<uint8_t>(std::max(0.0, std::min(255.0, high)));

    std::vector<QRegion::Run> runs;

    for (int32_t y = 0; y < height; ++y) {
        int32_t runStart = -1;

        for (int32_t x = 0; x < width; ++x) {
            uint8_t val = data[y * width + x];
            bool inRange = (val >= lowThresh && val <= highThresh);

            if (inRange && runStart < 0) {
                // Start new run
                runStart = x;
            } else if (!inRange && runStart >= 0) {
                // End current run
                runs.emplace_back(y, runStart, x);
                runStart = -1;
            }
        }

        // Handle run that extends to end of row
        if (runStart >= 0) {
            runs.emplace_back(y, runStart, width);
        }
    }

    return QRegion(runs);
}

QRegion ThresholdToRegion(const QImage& src, double threshold, bool above) {
    if (above) {
        return ThresholdToRegion(src, threshold + 1, 255.0);
    } else {
        return ThresholdToRegion(src, 0.0, threshold - 1);
    }
}

QRegion ThresholdAutoToRegion(const QImage& src,
                               AutoThresholdMethod method,
                               bool above,
                               double* computedThreshold) {
    if (src.Empty()) {
        if (computedThreshold) *computedThreshold = 0;
        return QRegion();
    }

    Histogram hist = ComputeHistogram(src);
    double threshold = 128;

    switch (method) {
        case AutoThresholdMethod::Otsu:
            threshold = ComputeOtsuThreshold(hist);
            break;

        case AutoThresholdMethod::Triangle:
            threshold = ComputeTriangleThreshold(hist);
            break;

        case AutoThresholdMethod::MinError:
            threshold = ComputeMinErrorThreshold(hist);
            break;

        case AutoThresholdMethod::Isodata:
            threshold = ComputeIsodataThreshold(hist);
            break;

        case AutoThresholdMethod::Median:
            threshold = ComputePercentile(hist, 50.0);
            break;

        default:
            threshold = ComputeOtsuThreshold(hist);
            break;
    }

    if (computedThreshold) {
        *computedThreshold = threshold;
    }

    return ThresholdToRegion(src, threshold, above);
}

// ============================================================================
// Binary Image Operations Implementation
// ============================================================================

void BinaryInvert(const QImage& src, QImage& dst, double maxValue) {
    if (src.Empty()) {
        dst = QImage();
        return;
    }

    int32_t width = src.Width();
    int32_t height = src.Height();

    if (dst.Empty() || dst.Width() != width || dst.Height() != height) {
        dst = QImage(width, height, PixelType::UInt8, ChannelType::Gray);
    }

    const uint8_t* srcData = static_cast<const uint8_t*>(src.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

    uint8_t maxVal = static_cast<uint8_t>(std::max(0.0, std::min(255.0, maxValue)));

    for (int32_t i = 0; i < width * height; ++i) {
        dstData[i] = maxVal - srcData[i];
    }
}

void BinaryAnd(const QImage& src1, const QImage& src2, QImage& dst,
               double maxValue) {
    if (src1.Empty() || src2.Empty()) {
        dst = QImage();
        return;
    }

    int32_t width = src1.Width();
    int32_t height = src1.Height();

    if (src2.Width() != width || src2.Height() != height) {
        throw InvalidArgumentException("BinaryAnd: image sizes must match");
    }

    if (dst.Empty() || dst.Width() != width || dst.Height() != height) {
        dst = QImage(width, height, PixelType::UInt8, ChannelType::Gray);
    }

    const uint8_t* src1Data = static_cast<const uint8_t*>(src1.Data());
    const uint8_t* src2Data = static_cast<const uint8_t*>(src2.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

    uint8_t maxVal = static_cast<uint8_t>(std::max(0.0, std::min(255.0, maxValue)));

    for (int32_t i = 0; i < width * height; ++i) {
        dstData[i] = (src1Data[i] > 0 && src2Data[i] > 0) ? maxVal : 0;
    }
}

void BinaryOr(const QImage& src1, const QImage& src2, QImage& dst,
              double maxValue) {
    if (src1.Empty() || src2.Empty()) {
        dst = QImage();
        return;
    }

    int32_t width = src1.Width();
    int32_t height = src1.Height();

    if (src2.Width() != width || src2.Height() != height) {
        throw InvalidArgumentException("BinaryOr: image sizes must match");
    }

    if (dst.Empty() || dst.Width() != width || dst.Height() != height) {
        dst = QImage(width, height, PixelType::UInt8, ChannelType::Gray);
    }

    const uint8_t* src1Data = static_cast<const uint8_t*>(src1.Data());
    const uint8_t* src2Data = static_cast<const uint8_t*>(src2.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

    uint8_t maxVal = static_cast<uint8_t>(std::max(0.0, std::min(255.0, maxValue)));

    for (int32_t i = 0; i < width * height; ++i) {
        dstData[i] = (src1Data[i] > 0 || src2Data[i] > 0) ? maxVal : 0;
    }
}

void BinaryXor(const QImage& src1, const QImage& src2, QImage& dst,
               double maxValue) {
    if (src1.Empty() || src2.Empty()) {
        dst = QImage();
        return;
    }

    int32_t width = src1.Width();
    int32_t height = src1.Height();

    if (src2.Width() != width || src2.Height() != height) {
        throw InvalidArgumentException("BinaryXor: image sizes must match");
    }

    if (dst.Empty() || dst.Width() != width || dst.Height() != height) {
        dst = QImage(width, height, PixelType::UInt8, ChannelType::Gray);
    }

    const uint8_t* src1Data = static_cast<const uint8_t*>(src1.Data());
    const uint8_t* src2Data = static_cast<const uint8_t*>(src2.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

    uint8_t maxVal = static_cast<uint8_t>(std::max(0.0, std::min(255.0, maxValue)));

    for (int32_t i = 0; i < width * height; ++i) {
        bool a = src1Data[i] > 0;
        bool b = src2Data[i] > 0;
        dstData[i] = (a != b) ? maxVal : 0;
    }
}

void BinaryDiff(const QImage& src1, const QImage& src2, QImage& dst,
                double maxValue) {
    if (src1.Empty() || src2.Empty()) {
        dst = QImage();
        return;
    }

    int32_t width = src1.Width();
    int32_t height = src1.Height();

    if (src2.Width() != width || src2.Height() != height) {
        throw InvalidArgumentException("BinaryDiff: image sizes must match");
    }

    if (dst.Empty() || dst.Width() != width || dst.Height() != height) {
        dst = QImage(width, height, PixelType::UInt8, ChannelType::Gray);
    }

    const uint8_t* src1Data = static_cast<const uint8_t*>(src1.Data());
    const uint8_t* src2Data = static_cast<const uint8_t*>(src2.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

    uint8_t maxVal = static_cast<uint8_t>(std::max(0.0, std::min(255.0, maxValue)));

    for (int32_t i = 0; i < width * height; ++i) {
        dstData[i] = (src1Data[i] > 0 && src2Data[i] == 0) ? maxVal : 0;
    }
}

// ============================================================================
// Utility Functions Implementation
// ============================================================================

bool IsBinaryImage(const QImage& image, double tolerance) {
    if (image.Empty()) {
        return true;
    }

    const uint8_t* data = static_cast<const uint8_t*>(image.Data());
    int32_t size = image.Width() * image.Height();

    uint8_t tol = static_cast<uint8_t>(tolerance);

    for (int32_t i = 0; i < size; ++i) {
        uint8_t val = data[i];
        // Check if value is near 0 or near 255
        if (val > tol && val < (255 - tol)) {
            return false;
        }
    }

    return true;
}

uint64_t CountNonZero(const QImage& image) {
    if (image.Empty()) {
        return 0;
    }

    const uint8_t* data = static_cast<const uint8_t*>(image.Data());
    int32_t size = image.Width() * image.Height();

    uint64_t count = 0;
    for (int32_t i = 0; i < size; ++i) {
        if (data[i] > 0) {
            ++count;
        }
    }

    return count;
}

uint64_t CountInRange(const QImage& image, double low, double high) {
    if (image.Empty()) {
        return 0;
    }

    const uint8_t* data = static_cast<const uint8_t*>(image.Data());
    int32_t size = image.Width() * image.Height();

    uint8_t lowThresh = static_cast<uint8_t>(std::max(0.0, std::min(255.0, low)));
    uint8_t highThresh = static_cast<uint8_t>(std::max(0.0, std::min(255.0, high)));

    uint64_t count = 0;
    for (int32_t i = 0; i < size; ++i) {
        uint8_t val = data[i];
        if (val >= lowThresh && val <= highThresh) {
            ++count;
        }
    }

    return count;
}

double ComputeForegroundRatio(const QImage& image) {
    if (image.Empty()) {
        return 0;
    }

    uint64_t nonZero = CountNonZero(image);
    uint64_t total = static_cast<uint64_t>(image.Width()) * image.Height();

    return total > 0 ? static_cast<double>(nonZero) / total : 0;
}

void ApplyMask(const QImage& src, const QImage& mask, QImage& dst) {
    if (src.Empty() || mask.Empty()) {
        dst = QImage();
        return;
    }

    int32_t width = src.Width();
    int32_t height = src.Height();

    if (mask.Width() != width || mask.Height() != height) {
        throw InvalidArgumentException("ApplyMask: image and mask sizes must match");
    }

    if (dst.Empty() || dst.Width() != width || dst.Height() != height) {
        dst = QImage(width, height, PixelType::UInt8, ChannelType::Gray);
    }

    const uint8_t* srcData = static_cast<const uint8_t*>(src.Data());
    const uint8_t* maskData = static_cast<const uint8_t*>(mask.Data());
    uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

    for (int32_t i = 0; i < width * height; ++i) {
        dstData[i] = (maskData[i] > 0) ? srcData[i] : 0;
    }
}

void RegionToMask(const QRegion& region, QImage& mask) {
    if (region.Empty()) {
        mask = QImage();
        return;
    }

    Rect2i bbox = region.BoundingBox();
    int32_t width = bbox.x + bbox.width;
    int32_t height = bbox.y + bbox.height;

    if (mask.Empty() || mask.Width() < width || mask.Height() < height) {
        mask = QImage(width, height, PixelType::UInt8, ChannelType::Gray);
    }

    // Clear mask
    std::memset(mask.Data(), 0, static_cast<size_t>(mask.Width()) * mask.Height());

    uint8_t* maskData = static_cast<uint8_t*>(mask.Data());
    int32_t stride = mask.Width();

    const auto& runs = region.Runs();
    for (const auto& run : runs) {
        for (int32_t x = run.colBegin; x < run.colEnd; ++x) {
            maskData[run.row * stride + x] = 255;
        }
    }
}

QRegion MaskToRegion(const QImage& mask, double threshold) {
    if (mask.Empty()) {
        return QRegion();
    }

    // Use ThresholdToRegion with threshold to max value
    return ThresholdToRegion(mask, threshold + 1, 255.0);
}

// ============================================================================
// Halcon-style Threshold Operations Implementation
// ============================================================================

QRegion DynThreshold(const QImage& image, const QImage& reference,
                     double offset, LightDark lightDark) {
    if (image.Empty() || reference.Empty()) {
        return QRegion();
    }

    int32_t width = image.Width();
    int32_t height = image.Height();

    if (reference.Width() != width || reference.Height() != height) {
        throw InvalidArgumentException("DynThreshold: image and reference sizes must match");
    }

    const uint8_t* imgData = static_cast<const uint8_t*>(image.Data());
    const uint8_t* refData = static_cast<const uint8_t*>(reference.Data());

    std::vector<QRegion::Run> runs;

    for (int32_t y = 0; y < height; ++y) {
        int32_t runStart = -1;

        for (int32_t x = 0; x < width; ++x) {
            int32_t idx = y * width + x;
            double imgVal = static_cast<double>(imgData[idx]);
            double refVal = static_cast<double>(refData[idx]);

            bool selected = false;

            switch (lightDark) {
                case LightDark::Light:
                    selected = (imgVal > refVal + offset);
                    break;
                case LightDark::Dark:
                    selected = (imgVal < refVal - offset);
                    break;
                case LightDark::Equal:
                    selected = (std::abs(imgVal - refVal) <= offset);
                    break;
                case LightDark::NotEqual:
                    selected = (std::abs(imgVal - refVal) > offset);
                    break;
            }

            if (selected && runStart < 0) {
                runStart = x;
            } else if (!selected && runStart >= 0) {
                runs.emplace_back(y, runStart, x);
                runStart = -1;
            }
        }

        if (runStart >= 0) {
            runs.emplace_back(y, runStart, width);
        }
    }

    return QRegion(runs);
}

QRegion DynThreshold(const QImage& image, int32_t filterSize,
                     double offset, LightDark lightDark) {
    if (image.Empty()) {
        return QRegion();
    }

    // Create smoothed reference using Gaussian blur
    int32_t width = image.Width();
    int32_t height = image.Height();

    // Compute Gaussian kernel
    double sigma = filterSize / 6.0;
    std::vector<double> kernel = Gaussian::Kernel1D(sigma, filterSize | 1);

    // Apply separable Gaussian convolution
    std::vector<float> srcFloat(width * height);
    std::vector<float> blurred(width * height);

    const uint8_t* srcData = static_cast<const uint8_t*>(image.Data());
    for (int32_t i = 0; i < width * height; ++i) {
        srcFloat[i] = static_cast<float>(srcData[i]);
    }

    ConvolveSeparable(srcFloat.data(), blurred.data(), width, height,
                      kernel.data(), static_cast<int32_t>(kernel.size()),
                      kernel.data(), static_cast<int32_t>(kernel.size()),
                      BorderMode::Reflect101);

    // Create reference image
    QImage reference(width, height, PixelType::UInt8, ChannelType::Gray);
    uint8_t* refData = static_cast<uint8_t*>(reference.Data());
    for (int32_t i = 0; i < width * height; ++i) {
        refData[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, blurred[i])));
    }

    return DynThreshold(image, reference, offset, lightDark);
}

DualThresholdResult DualThreshold(const QImage& image,
                                   double lowThreshold, double highThreshold) {
    DualThresholdResult result;
    result.lowThreshold = lowThreshold;
    result.highThreshold = highThreshold;

    if (image.Empty()) {
        return result;
    }

    int32_t width = image.Width();
    int32_t height = image.Height();

    const uint8_t* data = static_cast<const uint8_t*>(image.Data());

    uint8_t lowThresh = static_cast<uint8_t>(std::max(0.0, std::min(255.0, lowThreshold)));
    uint8_t highThresh = static_cast<uint8_t>(std::max(0.0, std::min(255.0, highThreshold)));

    std::vector<QRegion::Run> lightRuns, darkRuns, middleRuns;

    for (int32_t y = 0; y < height; ++y) {
        int32_t lightStart = -1, darkStart = -1, middleStart = -1;

        for (int32_t x = 0; x < width; ++x) {
            uint8_t val = data[y * width + x];

            bool isLight = (val > highThresh);
            bool isDark = (val < lowThresh);
            bool isMiddle = !isLight && !isDark;

            // Light region
            if (isLight && lightStart < 0) {
                lightStart = x;
            } else if (!isLight && lightStart >= 0) {
                lightRuns.emplace_back(y, lightStart, x);
                lightStart = -1;
            }

            // Dark region
            if (isDark && darkStart < 0) {
                darkStart = x;
            } else if (!isDark && darkStart >= 0) {
                darkRuns.emplace_back(y, darkStart, x);
                darkStart = -1;
            }

            // Middle region
            if (isMiddle && middleStart < 0) {
                middleStart = x;
            } else if (!isMiddle && middleStart >= 0) {
                middleRuns.emplace_back(y, middleStart, x);
                middleStart = -1;
            }
        }

        // End of row
        if (lightStart >= 0) lightRuns.emplace_back(y, lightStart, width);
        if (darkStart >= 0) darkRuns.emplace_back(y, darkStart, width);
        if (middleStart >= 0) middleRuns.emplace_back(y, middleStart, width);
    }

    result.lightRegion = QRegion(lightRuns);
    result.darkRegion = QRegion(darkRuns);
    result.middleRegion = QRegion(middleRuns);

    return result;
}

DualThresholdResult DualThresholdAuto(const QImage& image, const std::string& method) {
    DualThresholdResult result;

    if (image.Empty()) {
        return result;
    }

    std::string lower = method;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower.empty()) {
        lower = "otsu";
    }

    Histogram hist = ComputeHistogram(image);

    if (lower == "otsu" || lower == "max_separability") {
        // Use multi-Otsu to get two thresholds
        auto thresholds = ComputeMultiOtsuThresholds(hist, 2);
        if (thresholds.size() >= 2 &&
            thresholds[0] > 1.0 && thresholds[1] > 1.0 &&  // Valid range check
            thresholds[1] > thresholds[0] + 10.0) {        // Meaningful separation
            result.lowThreshold = thresholds[0];
            result.highThreshold = thresholds[1];
        } else {
            // Fall back to single Otsu with scaling
            double t = ComputeOtsuThreshold(hist);
            result.lowThreshold = t * 0.7;
            result.highThreshold = t * 1.3;
        }
    } else if (lower == "histogram_valley") {
        // Find valleys in histogram
        auto valleys = FindHistogramValleys(hist, 10);
        if (valleys.size() >= 2) {
            result.lowThreshold = hist.GetBinValue(valleys[0]);
            result.highThreshold = hist.GetBinValue(valleys[1]);
        } else if (valleys.size() == 1) {
            double t = hist.GetBinValue(valleys[0]);
            result.lowThreshold = t * 0.7;
            result.highThreshold = t * 1.3;
        } else {
            // Fallback to Otsu
            double t = ComputeOtsuThreshold(hist);
            result.lowThreshold = t * 0.7;
            result.highThreshold = t * 1.3;
        }
    } else {
        throw InvalidArgumentException("DualThresholdAuto: unknown method: " + method);
    }

    return DualThreshold(image, result.lowThreshold, result.highThreshold);
}

QRegion VarThreshold(const QImage& image, int32_t windowSize,
                     double varianceThreshold, LightDark lightDark) {
    if (image.Empty()) {
        return QRegion();
    }

    int32_t width = image.Width();
    int32_t height = image.Height();

    // Compute integral images for fast variance calculation
    std::vector<double> integralSum, integralSqSum;
    ComputeIntegralImages(image, integralSum, integralSqSum);

    int32_t halfSize = (windowSize | 1) / 2;

    std::vector<QRegion::Run> runs;

    for (int32_t y = 0; y < height; ++y) {
        int32_t runStart = -1;

        for (int32_t x = 0; x < width; ++x) {
            double mean, stddev;
            GetBlockStats(integralSum, integralSqSum, width, height,
                          x, y, halfSize, mean, stddev);

            double variance = stddev * stddev;
            bool selected = false;

            switch (lightDark) {
                case LightDark::Light:
                    selected = (variance > varianceThreshold);
                    break;
                case LightDark::Dark:
                    selected = (variance < varianceThreshold);
                    break;
                case LightDark::Equal:
                    selected = (std::abs(variance - varianceThreshold) < varianceThreshold * 0.1);
                    break;
                case LightDark::NotEqual:
                    selected = (std::abs(variance - varianceThreshold) >= varianceThreshold * 0.1);
                    break;
            }

            if (selected && runStart < 0) {
                runStart = x;
            } else if (!selected && runStart >= 0) {
                runs.emplace_back(y, runStart, x);
                runStart = -1;
            }
        }

        if (runStart >= 0) {
            runs.emplace_back(y, runStart, width);
        }
    }

    return QRegion(runs);
}

QRegion CharThreshold(const QImage& image, double sigma, double percent, LightDark lightDark) {
    if (image.Empty()) {
        return QRegion();
    }

    int32_t width = image.Width();
    int32_t height = image.Height();

    // Compute Gaussian smoothed image
    int32_t kernelSize = static_cast<int32_t>(sigma * 6) | 1;
    std::vector<double> kernel = Gaussian::Kernel1D(sigma, kernelSize);

    std::vector<float> srcFloat(width * height);
    std::vector<float> smoothed(width * height);

    const uint8_t* srcData = static_cast<const uint8_t*>(image.Data());
    for (int32_t i = 0; i < width * height; ++i) {
        srcFloat[i] = static_cast<float>(srcData[i]);
    }

    ConvolveSeparable(srcFloat.data(), smoothed.data(), width, height,
                      kernel.data(), kernelSize, kernel.data(), kernelSize,
                      BorderMode::Reflect101);

    // Use dynamic threshold with smoothed image
    QImage reference(width, height, PixelType::UInt8, ChannelType::Gray);
    uint8_t* refData = static_cast<uint8_t*>(reference.Data());
    for (int32_t i = 0; i < width * height; ++i) {
        refData[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, smoothed[i])));
    }

    // Offset based on percentile difference
    double offset = (100.0 - percent) * 2.55 * 0.1;  // Scale offset

    return DynThreshold(image, reference, offset, lightDark);
}

QRegion HysteresisThresholdToRegion(const QImage& image,
                                     double lowThreshold, double highThreshold) {
    if (image.Empty()) {
        return QRegion();
    }

    int32_t width = image.Width();
    int32_t height = image.Height();

    const uint8_t* data = static_cast<const uint8_t*>(image.Data());

    // Create binary images for strong and weak edges
    std::vector<uint8_t> strong(width * height, 0);
    std::vector<uint8_t> weak(width * height, 0);

    uint8_t lowThresh = static_cast<uint8_t>(std::max(0.0, std::min(255.0, lowThreshold)));
    uint8_t highThresh = static_cast<uint8_t>(std::max(0.0, std::min(255.0, highThreshold)));

    for (int32_t i = 0; i < width * height; ++i) {
        if (data[i] > highThresh) {
            strong[i] = 255;
        } else if (data[i] > lowThresh) {
            weak[i] = 255;
        }
    }

    // Use hysteresis from NonMaxSuppression module
    std::vector<uint8_t> result(width * height, 0);

    // BFS to connect weak edges to strong edges
    std::queue<std::pair<int32_t, int32_t>> queue;

    // Initialize queue with strong edge pixels
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            if (strong[y * width + x] > 0) {
                result[y * width + x] = 255;
                queue.push({x, y});
            }
        }
    }

    // 8-connected neighbors
    const int32_t dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int32_t dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};

    while (!queue.empty()) {
        auto [x, y] = queue.front();
        queue.pop();

        for (int32_t i = 0; i < 8; ++i) {
            int32_t nx = x + dx[i];
            int32_t ny = y + dy[i];

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int32_t idx = ny * width + nx;
                if (weak[idx] > 0 && result[idx] == 0) {
                    result[idx] = 255;
                    queue.push({nx, ny});
                }
            }
        }
    }

    // Convert result to region
    std::vector<QRegion::Run> runs;
    for (int32_t y = 0; y < height; ++y) {
        int32_t runStart = -1;

        for (int32_t x = 0; x < width; ++x) {
            if (result[y * width + x] > 0 && runStart < 0) {
                runStart = x;
            } else if (result[y * width + x] == 0 && runStart >= 0) {
                runs.emplace_back(y, runStart, x);
                runStart = -1;
            }
        }

        if (runStart >= 0) {
            runs.emplace_back(y, runStart, width);
        }
    }

    return QRegion(runs);
}

// ============================================================================
// Domain-aware Threshold Operations Implementation
// ============================================================================

QRegion ThresholdWithDomain(const QImage& image, double low, double high) {
    if (image.Empty()) {
        return QRegion();
    }

    // Check if image has a non-full domain
    const QRegion* domain = image.GetDomain();
    if (domain == nullptr || image.IsFullDomain()) {
        // No domain restriction, use standard threshold
        return ThresholdToRegion(image, low, high);
    }

    int32_t width = image.Width();

    const uint8_t* data = static_cast<const uint8_t*>(image.Data());

    uint8_t lowThresh = static_cast<uint8_t>(std::max(0.0, std::min(255.0, low)));
    uint8_t highThresh = static_cast<uint8_t>(std::max(0.0, std::min(255.0, high)));

    std::vector<QRegion::Run> runs;

    // Iterate over domain runs
    const auto& domainRuns = domain->Runs();
    for (const auto& run : domainRuns) {
        int32_t y = run.row;
        int32_t runStart = -1;

        for (int32_t x = run.colBegin; x < run.colEnd; ++x) {
            uint8_t val = data[y * width + x];
            bool inRange = (val >= lowThresh && val <= highThresh);

            if (inRange && runStart < 0) {
                runStart = x;
            } else if (!inRange && runStart >= 0) {
                runs.emplace_back(y, runStart, x);
                runStart = -1;
            }
        }

        if (runStart >= 0) {
            runs.emplace_back(y, runStart, run.colEnd);
        }
    }

    return QRegion(runs);
}

QRegion DynThresholdWithDomain(const QImage& image, const QImage& reference,
                                double offset, LightDark lightDark) {
    if (image.Empty() || reference.Empty()) {
        return QRegion();
    }

    // Check if image has a non-full domain
    const QRegion* domain = image.GetDomain();
    if (domain == nullptr || image.IsFullDomain()) {
        return DynThreshold(image, reference, offset, lightDark);
    }

    int32_t width = image.Width();
    int32_t height = image.Height();

    if (reference.Width() != width || reference.Height() != height) {
        throw InvalidArgumentException("DynThresholdWithDomain: image and reference sizes must match");
    }

    const uint8_t* imgData = static_cast<const uint8_t*>(image.Data());
    const uint8_t* refData = static_cast<const uint8_t*>(reference.Data());

    std::vector<QRegion::Run> runs;

    const auto& domainRuns = domain->Runs();
    for (const auto& run : domainRuns) {
        int32_t y = run.row;
        int32_t runStart = -1;

        for (int32_t x = run.colBegin; x < run.colEnd; ++x) {
            int32_t idx = y * width + x;
            double imgVal = static_cast<double>(imgData[idx]);
            double refVal = static_cast<double>(refData[idx]);

            bool selected = false;

            switch (lightDark) {
                case LightDark::Light:
                    selected = (imgVal > refVal + offset);
                    break;
                case LightDark::Dark:
                    selected = (imgVal < refVal - offset);
                    break;
                case LightDark::Equal:
                    selected = (std::abs(imgVal - refVal) <= offset);
                    break;
                case LightDark::NotEqual:
                    selected = (std::abs(imgVal - refVal) > offset);
                    break;
            }

            if (selected && runStart < 0) {
                runStart = x;
            } else if (!selected && runStart >= 0) {
                runs.emplace_back(y, runStart, x);
                runStart = -1;
            }
        }

        if (runStart >= 0) {
            runs.emplace_back(y, runStart, run.colEnd);
        }
    }

    return QRegion(runs);
}

// Helper function for adaptive threshold value computation
static double ComputeAdaptiveThresholdValue(AdaptiveMethod method,
                                             double mean, double stddev,
                                             const AdaptiveThresholdParams& params) {
    switch (method) {
        case AdaptiveMethod::Mean:
            return mean - params.C;

        case AdaptiveMethod::Gaussian:
            return mean - params.C;

        case AdaptiveMethod::Sauvola:
            return mean * (1.0 + params.k * (stddev / params.R - 1.0));

        case AdaptiveMethod::Niblack:
            return mean + params.k * stddev;

        case AdaptiveMethod::Wolf:
            // Wolf's method uses global min and max stddev
            // Simplified version: use local approximation
            return mean - params.k * (1.0 - stddev / 128.0) * (mean - 0);

        default:
            return mean - params.C;
    }
}

QRegion ThresholdAdaptiveToRegion(const QImage& image,
                                   const AdaptiveThresholdParams& params) {
    if (image.Empty()) {
        return QRegion();
    }

    int32_t width = image.Width();
    int32_t height = image.Height();

    const uint8_t* srcData = static_cast<const uint8_t*>(image.Data());

    int32_t blockSize = params.blockSize | 1;  // Ensure odd
    int32_t halfSize = blockSize / 2;

    // Compute integral images
    std::vector<double> integralSum, integralSqSum;
    ComputeIntegralImages(image, integralSum, integralSqSum);

    // Check for domain
    const QRegion* domain = image.GetDomain();
    bool hasDomain = domain != nullptr && !image.IsFullDomain();

    std::vector<QRegion::Run> runs;

    if (hasDomain) {
        const auto& domainRuns = domain->Runs();

        for (const auto& run : domainRuns) {
            int32_t y = run.row;
            int32_t runStart = -1;

            for (int32_t x = run.colBegin; x < run.colEnd; ++x) {
                int32_t idx = y * width + x;
                uint8_t pixelValue = srcData[idx];

                double localMean, localStddev;
                GetBlockStats(integralSum, integralSqSum, width, height,
                              x, y, halfSize, localMean, localStddev);

                double threshold = ComputeAdaptiveThresholdValue(
                    params.method, localMean, localStddev, params);

                bool selected = (pixelValue > threshold);

                if (selected && runStart < 0) {
                    runStart = x;
                } else if (!selected && runStart >= 0) {
                    runs.emplace_back(y, runStart, x);
                    runStart = -1;
                }
            }

            if (runStart >= 0) {
                runs.emplace_back(y, runStart, run.colEnd);
            }
        }
    } else {
        // Full image
        for (int32_t y = 0; y < height; ++y) {
            int32_t runStart = -1;

            for (int32_t x = 0; x < width; ++x) {
                int32_t idx = y * width + x;
                uint8_t pixelValue = srcData[idx];

                double localMean, localStddev;
                GetBlockStats(integralSum, integralSqSum, width, height,
                              x, y, halfSize, localMean, localStddev);

                double threshold = ComputeAdaptiveThresholdValue(
                    params.method, localMean, localStddev, params);

                bool selected = (pixelValue > threshold);

                if (selected && runStart < 0) {
                    runStart = x;
                } else if (!selected && runStart >= 0) {
                    runs.emplace_back(y, runStart, x);
                    runStart = -1;
                }
            }

            if (runStart >= 0) {
                runs.emplace_back(y, runStart, width);
            }
        }
    }

    return QRegion(runs);
}

}  // namespace Qi::Vision::Internal
