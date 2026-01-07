/**
 * @file Histogram.cpp
 * @brief Image histogram computation and enhancement implementation
 */

#include <QiVision/Internal/Histogram.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace Qi::Vision::Internal {

// ============================================================================
// Histogram Computation
// ============================================================================

Histogram ComputeHistogram(const QImage& image, int32_t numBins) {
    Histogram hist(numBins, 0, 255);

    if (image.Empty()) {
        return hist;
    }

    int32_t width = image.Width();
    int32_t height = image.Height();

    switch (image.Type()) {
        case PixelType::UInt8: {
            const uint8_t* data = static_cast<const uint8_t*>(image.Data());
            return ComputeHistogram(data, width, height, numBins, 0, 255);
        }
        case PixelType::UInt16: {
            const uint16_t* data = static_cast<const uint16_t*>(image.Data());
            return ComputeHistogram(data, width, height, numBins, 0, 65535);
        }
        case PixelType::Float32: {
            const float* data = static_cast<const float*>(image.Data());
            // Find min/max for float images
            float minVal = data[0], maxVal = data[0];
            for (int32_t i = 1; i < width * height; ++i) {
                if (data[i] < minVal) minVal = data[i];
                if (data[i] > maxVal) maxVal = data[i];
            }
            return ComputeHistogram(data, width, height, numBins, minVal, maxVal);
        }
        default:
            return hist;
    }
}

Histogram ComputeHistogramMasked(const QImage& image, const QImage& mask, int32_t numBins) {
    Histogram hist(numBins, 0, 255);

    if (image.Empty() || mask.Empty()) {
        return hist;
    }

    if (image.Width() != mask.Width() || image.Height() != mask.Height()) {
        return hist;
    }

    int32_t width = image.Width();
    int32_t height = image.Height();

    const uint8_t* maskData = static_cast<const uint8_t*>(mask.Data());

    if (image.Type() == PixelType::UInt8) {
        const uint8_t* imgData = static_cast<const uint8_t*>(image.Data());

        for (int32_t i = 0; i < width * height; ++i) {
            if (maskData[i] != 0) {
                int32_t bin = static_cast<int32_t>(imgData[i] * numBins / 256);
                bin = std::min(bin, numBins - 1);
                hist.bins[bin]++;
                hist.totalCount++;
            }
        }
    }

    return hist;
}

Histogram ComputeHistogramROI(const QImage& image, const Rect2i& roi, int32_t numBins) {
    Histogram hist(numBins, 0, 255);

    if (image.Empty()) {
        return hist;
    }

    int32_t width = image.Width();
    int32_t height = image.Height();

    // Clamp ROI to image bounds
    int32_t x0 = std::max(0, roi.x);
    int32_t y0 = std::max(0, roi.y);
    int32_t x1 = std::min(width, roi.x + roi.width);
    int32_t y1 = std::min(height, roi.y + roi.height);

    if (x1 <= x0 || y1 <= y0) {
        return hist;
    }

    if (image.Type() == PixelType::UInt8) {
        const uint8_t* data = static_cast<const uint8_t*>(image.Data());

        for (int32_t y = y0; y < y1; ++y) {
            for (int32_t x = x0; x < x1; ++x) {
                uint8_t value = data[y * width + x];
                int32_t bin = static_cast<int32_t>(value * numBins / 256);
                bin = std::min(bin, numBins - 1);
                hist.bins[bin]++;
                hist.totalCount++;
            }
        }
    }

    return hist;
}

std::vector<double> ComputeCumulativeHistogram(const Histogram& hist) {
    std::vector<double> cdf(hist.numBins, 0);

    if (hist.totalCount == 0) {
        return cdf;
    }

    double sum = 0;
    double total = static_cast<double>(hist.totalCount);

    for (int32_t i = 0; i < hist.numBins; ++i) {
        sum += hist.bins[i];
        cdf[i] = sum / total;
    }

    return cdf;
}

std::vector<double> NormalizeHistogram(const Histogram& hist) {
    std::vector<double> normalized(hist.numBins, 0);

    if (hist.totalCount == 0) {
        return normalized;
    }

    double total = static_cast<double>(hist.totalCount);
    for (int32_t i = 0; i < hist.numBins; ++i) {
        normalized[i] = hist.bins[i] / total;
    }

    return normalized;
}

// ============================================================================
// Histogram Statistics
// ============================================================================

HistogramStats ComputeHistogramStats(const Histogram& hist) {
    HistogramStats stats;
    stats.totalCount = hist.totalCount;

    if (hist.totalCount == 0) {
        return stats;
    }

    // Find min and max
    int32_t minIdx = -1, maxIdx = -1;
    uint32_t maxCount = 0;

    for (int32_t i = 0; i < hist.numBins; ++i) {
        if (hist.bins[i] > 0) {
            if (minIdx < 0) minIdx = i;
            maxIdx = i;
            if (hist.bins[i] > maxCount) {
                maxCount = hist.bins[i];
                stats.mode = hist.GetBinValue(i);
            }
        }
    }

    if (minIdx < 0) {
        return stats;
    }

    stats.min = hist.GetBinValue(minIdx);
    stats.max = hist.GetBinValue(maxIdx);
    stats.contrast = stats.max - stats.min;

    // Compute mean
    double sum = 0;
    for (int32_t i = 0; i < hist.numBins; ++i) {
        double value = hist.GetBinValue(i);
        sum += value * hist.bins[i];
    }
    stats.mean = sum / hist.totalCount;

    // Compute variance and stddev
    double sumSq = 0;
    for (int32_t i = 0; i < hist.numBins; ++i) {
        double value = hist.GetBinValue(i);
        double diff = value - stats.mean;
        sumSq += diff * diff * hist.bins[i];
    }
    stats.variance = sumSq / hist.totalCount;
    stats.stddev = std::sqrt(stats.variance);

    // Compute median (50th percentile)
    stats.median = ComputePercentile(hist, 50.0);

    // Compute entropy
    stats.entropy = ComputeEntropy(hist);

    return stats;
}

double ComputePercentile(const Histogram& hist, double percentile) {
    if (hist.totalCount == 0 || percentile < 0 || percentile > 100) {
        return 0;
    }

    uint64_t targetCount = static_cast<uint64_t>(hist.totalCount * percentile / 100.0);
    uint64_t cumSum = 0;

    for (int32_t i = 0; i < hist.numBins; ++i) {
        cumSum += hist.bins[i];
        if (cumSum >= targetCount) {
            return hist.GetBinValue(i);
        }
    }

    return hist.maxValue;
}

std::vector<double> ComputePercentiles(const Histogram& hist,
                                        const std::vector<double>& percentiles) {
    std::vector<double> result(percentiles.size());

    for (size_t i = 0; i < percentiles.size(); ++i) {
        result[i] = ComputePercentile(hist, percentiles[i]);
    }

    return result;
}

double ComputeEntropy(const Histogram& hist) {
    if (hist.totalCount == 0) {
        return 0;
    }

    double entropy = 0;
    double total = static_cast<double>(hist.totalCount);

    for (int32_t i = 0; i < hist.numBins; ++i) {
        if (hist.bins[i] > 0) {
            double p = hist.bins[i] / total;
            entropy -= p * std::log2(p);
        }
    }

    return entropy;
}

// ============================================================================
// Histogram Equalization
// ============================================================================

std::vector<uint8_t> ComputeEqualizationLUT(const Histogram& hist,
                                             double outputMin,
                                             double outputMax) {
    std::vector<uint8_t> lut(256, 0);

    if (hist.totalCount == 0) {
        for (int i = 0; i < 256; ++i) {
            lut[i] = static_cast<uint8_t>(i);
        }
        return lut;
    }

    // Compute CDF
    auto cdf = ComputeCumulativeHistogram(hist);

    // Find minimum non-zero CDF value
    double cdfMin = 1.0;
    for (int32_t i = 0; i < hist.numBins; ++i) {
        if (cdf[i] > 0) {
            cdfMin = cdf[i];
            break;
        }
    }

    // Scale CDF to output range
    double range = outputMax - outputMin;

    for (int32_t i = 0; i < 256; ++i) {
        int32_t bin = i * hist.numBins / 256;
        double normalized = (cdf[bin] - cdfMin) / (1.0 - cdfMin);
        int32_t value = static_cast<int32_t>(outputMin + normalized * range + 0.5);
        lut[i] = static_cast<uint8_t>(std::max(0, std::min(255, value)));
    }

    return lut;
}

QImage ApplyLUT(const QImage& image, const std::vector<uint8_t>& lut) {
    if (image.Empty() || lut.size() < 256) {
        return QImage();
    }

    QImage result(image.Width(), image.Height(), PixelType::UInt8, ChannelType::Gray);

    const uint8_t* src = static_cast<const uint8_t*>(image.Data());
    uint8_t* dst = static_cast<uint8_t*>(result.Data());

    int32_t size = image.Width() * image.Height();
    for (int32_t i = 0; i < size; ++i) {
        dst[i] = lut[src[i]];
    }

    return result;
}

void ApplyLUTInPlace(QImage& image, const std::vector<uint8_t>& lut) {
    if (image.Empty() || lut.size() < 256) {
        return;
    }

    uint8_t* data = static_cast<uint8_t*>(image.Data());
    int32_t size = image.Width() * image.Height();

    for (int32_t i = 0; i < size; ++i) {
        data[i] = lut[data[i]];
    }
}

QImage HistogramEqualize(const QImage& image) {
    if (image.Empty()) {
        return QImage();
    }

    Histogram hist = ComputeHistogram(image);
    auto lut = ComputeEqualizationLUT(hist);
    return ApplyLUT(image, lut);
}

void HistogramEqualizeInPlace(QImage& image) {
    if (image.Empty()) {
        return;
    }

    Histogram hist = ComputeHistogram(image);
    auto lut = ComputeEqualizationLUT(hist);
    ApplyLUTInPlace(image, lut);
}

// ============================================================================
// CLAHE Implementation
// ============================================================================

QImage ApplyCLAHE(const QImage& image, const CLAHEParams& params) {
    if (image.Empty()) {
        return QImage();
    }

    int32_t width = image.Width();
    int32_t height = image.Height();

    QImage result(width, height, PixelType::UInt8, ChannelType::Gray);

    const uint8_t* src = static_cast<const uint8_t*>(image.Data());
    uint8_t* dst = static_cast<uint8_t*>(result.Data());

    int32_t tilesX = params.tileGridSizeX;
    int32_t tilesY = params.tileGridSizeY;
    int32_t numBins = params.numBins;

    // Compute tile dimensions
    int32_t tileWidth = (width + tilesX - 1) / tilesX;
    int32_t tileHeight = (height + tilesY - 1) / tilesY;

    // Compute histograms for each tile
    std::vector<std::vector<uint32_t>> tileHists(tilesX * tilesY,
                                                   std::vector<uint32_t>(numBins, 0));
    std::vector<std::vector<uint8_t>> tileLUTs(tilesX * tilesY);

    // Build histograms for each tile
    for (int32_t ty = 0; ty < tilesY; ++ty) {
        for (int32_t tx = 0; tx < tilesX; ++tx) {
            int32_t x0 = tx * tileWidth;
            int32_t y0 = ty * tileHeight;
            int32_t x1 = std::min(x0 + tileWidth, width);
            int32_t y1 = std::min(y0 + tileHeight, height);

            auto& hist = tileHists[ty * tilesX + tx];
            int32_t tilePixels = 0;

            for (int32_t y = y0; y < y1; ++y) {
                for (int32_t x = x0; x < x1; ++x) {
                    int32_t bin = src[y * width + x] * numBins / 256;
                    bin = std::min(bin, numBins - 1);
                    hist[bin]++;
                    tilePixels++;
                }
            }

            // Clip histogram
            if (params.clipLimit > 0 && tilePixels > 0) {
                int32_t clipValue = static_cast<int32_t>(
                    params.clipLimit * tilePixels / numBins);
                clipValue = std::max(1, clipValue);

                int32_t excess = 0;
                for (int32_t i = 0; i < numBins; ++i) {
                    if (static_cast<int32_t>(hist[i]) > clipValue) {
                        excess += hist[i] - clipValue;
                        hist[i] = clipValue;
                    }
                }

                // Redistribute excess
                int32_t perBin = excess / numBins;
                int32_t remainder = excess % numBins;

                for (int32_t i = 0; i < numBins; ++i) {
                    hist[i] += perBin;
                    if (i < remainder) {
                        hist[i]++;
                    }
                }
            }

            // Compute LUT from clipped histogram
            auto& lut = tileLUTs[ty * tilesX + tx];
            lut.resize(256);

            uint64_t cumSum = 0;
            uint64_t total = 0;
            for (int32_t i = 0; i < numBins; ++i) {
                total += hist[i];
            }

            if (total > 0) {
                for (int32_t i = 0; i < 256; ++i) {
                    int32_t bin = i * numBins / 256;
                    bin = std::min(bin, numBins - 1);

                    // Sum up to this bin
                    uint64_t sum = 0;
                    for (int32_t j = 0; j <= bin; ++j) {
                        sum += hist[j];
                    }

                    lut[i] = static_cast<uint8_t>(
                        std::min(255.0, (sum * 255.0) / total));
                }
            } else {
                for (int32_t i = 0; i < 256; ++i) {
                    lut[i] = static_cast<uint8_t>(i);
                }
            }
        }
    }

    // Apply bilinear interpolation between tiles
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            // Find tile position
            double tileX = (x + 0.5) * tilesX / width - 0.5;
            double tileY = (y + 0.5) * tilesY / height - 0.5;

            int32_t tx0 = static_cast<int32_t>(std::floor(tileX));
            int32_t ty0 = static_cast<int32_t>(std::floor(tileY));
            int32_t tx1 = tx0 + 1;
            int32_t ty1 = ty0 + 1;

            // Clamp to valid range
            tx0 = std::max(0, std::min(tx0, tilesX - 1));
            ty0 = std::max(0, std::min(ty0, tilesY - 1));
            tx1 = std::max(0, std::min(tx1, tilesX - 1));
            ty1 = std::max(0, std::min(ty1, tilesY - 1));

            double fx = tileX - std::floor(tileX);
            double fy = tileY - std::floor(tileY);

            // Handle boundary cases
            if (tx0 == tx1) fx = 0;
            if (ty0 == ty1) fy = 0;

            uint8_t pixel = src[y * width + x];

            // Bilinear interpolation of LUT values
            double v00 = tileLUTs[ty0 * tilesX + tx0][pixel];
            double v01 = tileLUTs[ty0 * tilesX + tx1][pixel];
            double v10 = tileLUTs[ty1 * tilesX + tx0][pixel];
            double v11 = tileLUTs[ty1 * tilesX + tx1][pixel];

            double v0 = v00 * (1 - fx) + v01 * fx;
            double v1 = v10 * (1 - fx) + v11 * fx;
            double value = v0 * (1 - fy) + v1 * fy;

            dst[y * width + x] = static_cast<uint8_t>(
                std::max(0.0, std::min(255.0, value + 0.5)));
        }
    }

    return result;
}

void ApplyCLAHEInPlace(QImage& image, const CLAHEParams& params) {
    QImage result = ApplyCLAHE(image, params);
    if (!result.Empty()) {
        image = std::move(result);
    }
}

// ============================================================================
// Histogram Matching
// ============================================================================

std::vector<uint8_t> ComputeMatchingLUT(const Histogram& sourceHist,
                                         const Histogram& targetHist) {
    std::vector<uint8_t> lut(256, 0);

    auto sourceCdf = ComputeCumulativeHistogram(sourceHist);
    auto targetCdf = ComputeCumulativeHistogram(targetHist);

    for (int32_t i = 0; i < 256; ++i) {
        int32_t srcBin = i * sourceHist.numBins / 256;
        double srcCdfVal = sourceCdf[std::min(srcBin, sourceHist.numBins - 1)];

        // Find closest CDF value in target
        int32_t bestBin = 0;
        double minDiff = std::abs(targetCdf[0] - srcCdfVal);

        for (int32_t j = 1; j < targetHist.numBins; ++j) {
            double diff = std::abs(targetCdf[j] - srcCdfVal);
            if (diff < minDiff) {
                minDiff = diff;
                bestBin = j;
            }
        }

        lut[i] = static_cast<uint8_t>(bestBin * 255 / (targetHist.numBins - 1));
    }

    return lut;
}

QImage HistogramMatch(const QImage& image, const Histogram& targetHist) {
    if (image.Empty()) {
        return QImage();
    }

    Histogram sourceHist = ComputeHistogram(image);
    auto lut = ComputeMatchingLUT(sourceHist, targetHist);
    return ApplyLUT(image, lut);
}

QImage HistogramMatchToImage(const QImage& image, const QImage& reference) {
    if (image.Empty() || reference.Empty()) {
        return QImage();
    }

    Histogram targetHist = ComputeHistogram(reference);
    return HistogramMatch(image, targetHist);
}

// ============================================================================
// Contrast Stretching
// ============================================================================

QImage ContrastStretch(const QImage& image,
                        double lowPercentile,
                        double highPercentile,
                        double outputMin,
                        double outputMax) {
    if (image.Empty()) {
        return QImage();
    }

    Histogram hist = ComputeHistogram(image);

    double inputMin = ComputePercentile(hist, lowPercentile);
    double inputMax = ComputePercentile(hist, highPercentile);

    if (inputMax <= inputMin) {
        return image;  // No stretch needed
    }

    std::vector<uint8_t> lut(256);
    double scale = (outputMax - outputMin) / (inputMax - inputMin);

    for (int32_t i = 0; i < 256; ++i) {
        double value = outputMin + (i - inputMin) * scale;
        value = std::max(outputMin, std::min(outputMax, value));
        lut[i] = static_cast<uint8_t>(value + 0.5);
    }

    return ApplyLUT(image, lut);
}

QImage AutoContrast(const QImage& image) {
    return ContrastStretch(image, 0, 100, 0, 255);
}

QImage NormalizeImage(const QImage& image, double outputMin, double outputMax) {
    return ContrastStretch(image, 0, 100, outputMin, outputMax);
}

// ============================================================================
// Automatic Thresholding
// ============================================================================

double ComputeOtsuThreshold(const Histogram& hist) {
    if (hist.totalCount == 0) {
        return 128;
    }

    // Compute total mean
    double totalMean = 0;
    for (int32_t i = 0; i < hist.numBins; ++i) {
        totalMean += i * hist.bins[i];
    }
    totalMean /= hist.totalCount;

    double maxVariance = 0;
    int32_t optimalThreshold = 0;

    double w0 = 0;  // Weight of background
    double sum0 = 0;  // Sum for background

    for (int32_t t = 0; t < hist.numBins - 1; ++t) {
        w0 += hist.bins[t];
        if (w0 == 0) continue;

        double w1 = hist.totalCount - w0;
        if (w1 == 0) break;

        sum0 += t * hist.bins[t];

        double mean0 = sum0 / w0;
        double mean1 = (totalMean * hist.totalCount - sum0) / w1;

        double variance = w0 * w1 * (mean0 - mean1) * (mean0 - mean1);

        if (variance > maxVariance) {
            maxVariance = variance;
            optimalThreshold = t;
        }
    }

    return hist.GetBinValue(optimalThreshold);
}

double ComputeOtsuThreshold(const QImage& image) {
    Histogram hist = ComputeHistogram(image);
    return ComputeOtsuThreshold(hist);
}

std::vector<double> ComputeMultiOtsuThresholds(const Histogram& hist, int32_t numThresholds) {
    std::vector<double> thresholds;

    if (numThresholds <= 0 || hist.totalCount == 0) {
        return thresholds;
    }

    if (numThresholds == 1) {
        thresholds.push_back(ComputeOtsuThreshold(hist));
        return thresholds;
    }

    // For simplicity, use recursive Otsu for 2 thresholds
    // Full multi-level Otsu would require dynamic programming

    if (numThresholds == 2) {
        double maxVariance = 0;
        int32_t t1Opt = 0, t2Opt = 0;

        auto normalizedHist = NormalizeHistogram(hist);

        for (int32_t t1 = 1; t1 < hist.numBins - 2; ++t1) {
            for (int32_t t2 = t1 + 1; t2 < hist.numBins - 1; ++t2) {
                // Compute weights and means for 3 classes
                double w0 = 0, w1 = 0, w2 = 0;
                double sum0 = 0, sum1 = 0, sum2 = 0;

                for (int32_t i = 0; i < t1; ++i) {
                    w0 += normalizedHist[i];
                    sum0 += i * normalizedHist[i];
                }
                for (int32_t i = t1; i < t2; ++i) {
                    w1 += normalizedHist[i];
                    sum1 += i * normalizedHist[i];
                }
                for (int32_t i = t2; i < hist.numBins; ++i) {
                    w2 += normalizedHist[i];
                    sum2 += i * normalizedHist[i];
                }

                if (w0 < 1e-10 || w1 < 1e-10 || w2 < 1e-10) continue;

                double mean0 = sum0 / w0;
                double mean1 = sum1 / w1;
                double mean2 = sum2 / w2;
                double meanT = sum0 + sum1 + sum2;

                double variance = w0 * (mean0 - meanT) * (mean0 - meanT) +
                                  w1 * (mean1 - meanT) * (mean1 - meanT) +
                                  w2 * (mean2 - meanT) * (mean2 - meanT);

                if (variance > maxVariance) {
                    maxVariance = variance;
                    t1Opt = t1;
                    t2Opt = t2;
                }
            }
        }

        thresholds.push_back(hist.GetBinValue(t1Opt));
        thresholds.push_back(hist.GetBinValue(t2Opt));
    }

    return thresholds;
}

double ComputeTriangleThreshold(const Histogram& hist) {
    if (hist.totalCount == 0) {
        return 128;
    }

    // Find the peak
    int32_t peakIdx = 0;
    uint32_t maxCount = 0;
    for (int32_t i = 0; i < hist.numBins; ++i) {
        if (hist.bins[i] > maxCount) {
            maxCount = hist.bins[i];
            peakIdx = i;
        }
    }

    // Find the farthest end from peak
    int32_t endIdx = (peakIdx < hist.numBins / 2) ? (hist.numBins - 1) : 0;

    // Line from peak to end
    double x1 = peakIdx;
    double y1 = hist.bins[peakIdx];
    double x2 = endIdx;
    double y2 = hist.bins[endIdx];

    double lineLen = std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    if (lineLen < 1e-10) {
        return hist.GetBinValue(peakIdx);
    }

    // Find point with maximum distance to line
    double maxDist = 0;
    int32_t threshold = peakIdx;

    int32_t start = std::min(peakIdx, endIdx);
    int32_t end = std::max(peakIdx, endIdx);

    for (int32_t i = start; i <= end; ++i) {
        double px = i;
        double py = hist.bins[i];

        // Distance from point to line
        double dist = std::abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / lineLen;

        if (dist > maxDist) {
            maxDist = dist;
            threshold = i;
        }
    }

    return hist.GetBinValue(threshold);
}

double ComputeMinErrorThreshold(const Histogram& hist) {
    if (hist.totalCount == 0) {
        return 128;
    }

    auto normalizedHist = NormalizeHistogram(hist);

    double minError = std::numeric_limits<double>::max();
    int32_t optThreshold = 0;

    for (int32_t t = 1; t < hist.numBins - 1; ++t) {
        // Compute weights
        double w0 = 0, w1 = 0;
        for (int32_t i = 0; i < t; ++i) w0 += normalizedHist[i];
        for (int32_t i = t; i < hist.numBins; ++i) w1 += normalizedHist[i];

        if (w0 < 1e-10 || w1 < 1e-10) continue;

        // Compute means
        double mean0 = 0, mean1 = 0;
        for (int32_t i = 0; i < t; ++i) mean0 += i * normalizedHist[i];
        for (int32_t i = t; i < hist.numBins; ++i) mean1 += i * normalizedHist[i];
        mean0 /= w0;
        mean1 /= w1;

        // Compute variances
        double var0 = 0, var1 = 0;
        for (int32_t i = 0; i < t; ++i) {
            double diff = i - mean0;
            var0 += diff * diff * normalizedHist[i];
        }
        for (int32_t i = t; i < hist.numBins; ++i) {
            double diff = i - mean1;
            var1 += diff * diff * normalizedHist[i];
        }
        var0 /= w0;
        var1 /= w1;

        // Minimum error criterion (Kittler-Illingworth)
        double sigma0 = std::sqrt(var0 + 1e-10);
        double sigma1 = std::sqrt(var1 + 1e-10);

        double error = 1 + 2 * (w0 * std::log(sigma0) + w1 * std::log(sigma1)) -
                       2 * (w0 * std::log(w0) + w1 * std::log(w1));

        if (error < minError) {
            minError = error;
            optThreshold = t;
        }
    }

    return hist.GetBinValue(optThreshold);
}

double ComputeIsodataThreshold(const Histogram& hist, int32_t maxIterations) {
    if (hist.totalCount == 0) {
        return 128;
    }

    // Initial threshold is the mean
    double threshold = 0;
    for (int32_t i = 0; i < hist.numBins; ++i) {
        threshold += i * hist.bins[i];
    }
    threshold /= hist.totalCount;

    for (int32_t iter = 0; iter < maxIterations; ++iter) {
        int32_t t = static_cast<int32_t>(threshold);

        // Compute means of two classes
        double sum0 = 0, count0 = 0;
        double sum1 = 0, count1 = 0;

        for (int32_t i = 0; i < t; ++i) {
            sum0 += i * hist.bins[i];
            count0 += hist.bins[i];
        }
        for (int32_t i = t; i < hist.numBins; ++i) {
            sum1 += i * hist.bins[i];
            count1 += hist.bins[i];
        }

        if (count0 < 1 || count1 < 1) break;

        double mean0 = sum0 / count0;
        double mean1 = sum1 / count1;

        double newThreshold = (mean0 + mean1) / 2.0;

        if (std::abs(newThreshold - threshold) < 0.5) {
            break;
        }

        threshold = newThreshold;
    }

    return hist.GetBinValue(static_cast<int32_t>(threshold));
}

// ============================================================================
// Utility Functions
// ============================================================================

int32_t FindHistogramPeak(const Histogram& hist) {
    int32_t peakIdx = 0;
    uint32_t maxCount = 0;

    for (int32_t i = 0; i < hist.numBins; ++i) {
        if (hist.bins[i] > maxCount) {
            maxCount = hist.bins[i];
            peakIdx = i;
        }
    }

    return peakIdx;
}

std::vector<int32_t> FindHistogramPeaks(const Histogram& hist,
                                         double minHeight,
                                         int32_t minDistance) {
    std::vector<int32_t> peaks;

    if (hist.numBins < 3) {
        return peaks;
    }

    uint32_t maxCount = *std::max_element(hist.bins.begin(), hist.bins.end());
    uint32_t threshold = static_cast<uint32_t>(maxCount * minHeight);

    for (int32_t i = 1; i < hist.numBins - 1; ++i) {
        if (hist.bins[i] >= threshold &&
            hist.bins[i] > hist.bins[i - 1] &&
            hist.bins[i] > hist.bins[i + 1]) {

            // Check minimum distance from existing peaks
            bool tooClose = false;
            for (int32_t peak : peaks) {
                if (std::abs(i - peak) < minDistance) {
                    tooClose = true;
                    break;
                }
            }

            if (!tooClose) {
                peaks.push_back(i);
            }
        }
    }

    return peaks;
}

std::vector<int32_t> FindHistogramValleys(const Histogram& hist, int32_t minDistance) {
    std::vector<int32_t> valleys;

    if (hist.numBins < 3) {
        return valleys;
    }

    for (int32_t i = 1; i < hist.numBins - 1; ++i) {
        if (hist.bins[i] < hist.bins[i - 1] &&
            hist.bins[i] < hist.bins[i + 1]) {

            bool tooClose = false;
            for (int32_t valley : valleys) {
                if (std::abs(i - valley) < minDistance) {
                    tooClose = true;
                    break;
                }
            }

            if (!tooClose) {
                valleys.push_back(i);
            }
        }
    }

    return valleys;
}

Histogram SmoothHistogram(const Histogram& hist, int32_t kernelSize) {
    Histogram smoothed = hist;

    if (kernelSize < 3 || hist.numBins < kernelSize) {
        return smoothed;
    }

    // Make kernel size odd
    if (kernelSize % 2 == 0) {
        kernelSize++;
    }

    int32_t halfKernel = kernelSize / 2;

    for (int32_t i = 0; i < hist.numBins; ++i) {
        uint64_t sum = 0;
        int32_t count = 0;

        for (int32_t k = -halfKernel; k <= halfKernel; ++k) {
            int32_t idx = i + k;
            if (idx >= 0 && idx < hist.numBins) {
                sum += hist.bins[idx];
                count++;
            }
        }

        smoothed.bins[i] = static_cast<uint32_t>(sum / count);
    }

    return smoothed;
}

double CompareHistograms(const Histogram& hist1, const Histogram& hist2,
                          HistogramCompareMethod method) {
    auto norm1 = NormalizeHistogram(hist1);
    auto norm2 = NormalizeHistogram(hist2);

    int32_t numBins = std::min(hist1.numBins, hist2.numBins);

    switch (method) {
        case HistogramCompareMethod::Correlation: {
            double mean1 = 0, mean2 = 0;
            for (int32_t i = 0; i < numBins; ++i) {
                mean1 += norm1[i];
                mean2 += norm2[i];
            }
            mean1 /= numBins;
            mean2 /= numBins;

            double num = 0, den1 = 0, den2 = 0;
            for (int32_t i = 0; i < numBins; ++i) {
                double d1 = norm1[i] - mean1;
                double d2 = norm2[i] - mean2;
                num += d1 * d2;
                den1 += d1 * d1;
                den2 += d2 * d2;
            }

            double den = std::sqrt(den1 * den2);
            return (den > 1e-10) ? (num / den) : 0;
        }

        case HistogramCompareMethod::ChiSquare: {
            double result = 0;
            for (int32_t i = 0; i < numBins; ++i) {
                double sum = norm1[i] + norm2[i];
                if (sum > 1e-10) {
                    double diff = norm1[i] - norm2[i];
                    result += diff * diff / sum;
                }
            }
            return result;
        }

        case HistogramCompareMethod::Intersection: {
            double result = 0;
            for (int32_t i = 0; i < numBins; ++i) {
                result += std::min(norm1[i], norm2[i]);
            }
            return result;
        }

        case HistogramCompareMethod::Bhattacharyya: {
            double bc = 0;
            for (int32_t i = 0; i < numBins; ++i) {
                bc += std::sqrt(norm1[i] * norm2[i]);
            }
            return std::sqrt(std::max(0.0, 1.0 - bc));
        }

        case HistogramCompareMethod::KLDivergence: {
            double result = 0;
            for (int32_t i = 0; i < numBins; ++i) {
                if (norm1[i] > 1e-10 && norm2[i] > 1e-10) {
                    result += norm1[i] * std::log(norm1[i] / norm2[i]);
                }
            }
            return result;
        }

        default:
            return 0;
    }
}

} // namespace Qi::Vision::Internal
