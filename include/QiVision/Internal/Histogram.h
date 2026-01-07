#pragma once

/**
 * @file Histogram.h
 * @brief Image histogram computation and enhancement
 *
 * Provides:
 * - Histogram computation (1D grayscale, multi-channel)
 * - Histogram statistics (min, max, percentiles, entropy)
 * - Histogram equalization (global and local)
 * - CLAHE (Contrast Limited Adaptive Histogram Equalization)
 * - Histogram matching/specification
 *
 * Used by:
 * - Image preprocessing
 * - Contrast enhancement
 * - Automatic thresholding (Otsu, etc.)
 * - Feature normalization
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QImage.h>

#include <array>
#include <cstdint>
#include <vector>

namespace Qi::Vision::Internal {

// ============================================================================
// Constants
// ============================================================================

/// Standard histogram bin count for 8-bit images
constexpr int32_t HISTOGRAM_BINS_8BIT = 256;

/// Default CLAHE clip limit
constexpr double DEFAULT_CLAHE_CLIP_LIMIT = 40.0;

/// Default CLAHE tile grid size
constexpr int32_t DEFAULT_CLAHE_TILE_SIZE = 8;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief 1D Histogram data
 */
struct Histogram {
    std::vector<uint32_t> bins;     ///< Bin counts
    int32_t numBins = 256;          ///< Number of bins
    double minValue = 0;            ///< Minimum value in range
    double maxValue = 255;          ///< Maximum value in range
    uint64_t totalCount = 0;        ///< Total pixel count

    /// Default constructor
    Histogram() : bins(HISTOGRAM_BINS_8BIT, 0) {}

    /// Constructor with custom bin count
    explicit Histogram(int32_t nBins, double minVal = 0, double maxVal = 255)
        : bins(nBins, 0), numBins(nBins), minValue(minVal), maxValue(maxVal) {}

    /// Get bin count at index
    uint32_t At(int32_t idx) const {
        if (idx < 0 || idx >= static_cast<int32_t>(bins.size())) return 0;
        return bins[idx];
    }

    /// Get bin index for a value
    int32_t GetBinIndex(double value) const {
        if (maxValue <= minValue) return 0;
        double normalized = (value - minValue) / (maxValue - minValue);
        int32_t idx = static_cast<int32_t>(normalized * numBins);
        return std::max(0, std::min(idx, numBins - 1));
    }

    /// Get value at bin center
    double GetBinValue(int32_t idx) const {
        double binWidth = (maxValue - minValue) / numBins;
        return minValue + (idx + 0.5) * binWidth;
    }

    /// Check if empty
    bool Empty() const { return totalCount == 0; }

    /// Clear histogram
    void Clear() {
        std::fill(bins.begin(), bins.end(), 0);
        totalCount = 0;
    }
};

/**
 * @brief Histogram statistics
 */
struct HistogramStats {
    double min = 0;             ///< Minimum value with non-zero count
    double max = 0;             ///< Maximum value with non-zero count
    double mean = 0;            ///< Mean value
    double median = 0;          ///< Median value (50th percentile)
    double mode = 0;            ///< Mode (most frequent value)
    double stddev = 0;          ///< Standard deviation
    double variance = 0;        ///< Variance
    double entropy = 0;         ///< Shannon entropy (bits)
    double contrast = 0;        ///< Contrast (max - min)
    uint64_t totalCount = 0;    ///< Total pixel count
};

/**
 * @brief CLAHE parameters
 */
struct CLAHEParams {
    int32_t tileGridSizeX = DEFAULT_CLAHE_TILE_SIZE;    ///< Tile grid width
    int32_t tileGridSizeY = DEFAULT_CLAHE_TILE_SIZE;    ///< Tile grid height
    double clipLimit = DEFAULT_CLAHE_CLIP_LIMIT;        ///< Clip limit (contrast limit)
    int32_t numBins = HISTOGRAM_BINS_8BIT;              ///< Number of histogram bins

    /// Create default params
    static CLAHEParams Default() { return CLAHEParams(); }

    /// Create params with custom tile size
    static CLAHEParams WithTileSize(int32_t tileSize, double clipLimit = DEFAULT_CLAHE_CLIP_LIMIT) {
        CLAHEParams p;
        p.tileGridSizeX = tileSize;
        p.tileGridSizeY = tileSize;
        p.clipLimit = clipLimit;
        return p;
    }
};

// ============================================================================
// Histogram Computation
// ============================================================================

/**
 * @brief Compute histogram from image
 *
 * @param image Input image (grayscale)
 * @param numBins Number of histogram bins
 * @return Histogram
 */
Histogram ComputeHistogram(const QImage& image, int32_t numBins = HISTOGRAM_BINS_8BIT);

/**
 * @brief Compute histogram from raw data
 */
template<typename T>
Histogram ComputeHistogram(const T* data, int32_t width, int32_t height,
                           int32_t numBins = HISTOGRAM_BINS_8BIT,
                           double minValue = 0, double maxValue = 255);

/**
 * @brief Compute histogram with mask
 *
 * @param image Input image
 * @param mask Mask image (non-zero = include)
 * @param numBins Number of bins
 * @return Histogram
 */
Histogram ComputeHistogramMasked(const QImage& image, const QImage& mask,
                                  int32_t numBins = HISTOGRAM_BINS_8BIT);

/**
 * @brief Compute histogram in region of interest
 */
Histogram ComputeHistogramROI(const QImage& image, const Rect2i& roi,
                               int32_t numBins = HISTOGRAM_BINS_8BIT);

/**
 * @brief Compute cumulative histogram (CDF)
 *
 * @param hist Input histogram
 * @return Cumulative histogram (normalized to [0, 1])
 */
std::vector<double> ComputeCumulativeHistogram(const Histogram& hist);

/**
 * @brief Normalize histogram to probability distribution
 *
 * @param hist Histogram
 * @return Normalized histogram (sums to 1)
 */
std::vector<double> NormalizeHistogram(const Histogram& hist);

// ============================================================================
// Histogram Statistics
// ============================================================================

/**
 * @brief Compute histogram statistics
 */
HistogramStats ComputeHistogramStats(const Histogram& hist);

/**
 * @brief Compute percentile value from histogram
 *
 * @param hist Histogram
 * @param percentile Percentile (0-100)
 * @return Value at percentile
 */
double ComputePercentile(const Histogram& hist, double percentile);

/**
 * @brief Compute multiple percentiles
 *
 * @param hist Histogram
 * @param percentiles Vector of percentiles (0-100)
 * @return Vector of values at percentiles
 */
std::vector<double> ComputePercentiles(const Histogram& hist,
                                        const std::vector<double>& percentiles);

/**
 * @brief Compute Shannon entropy from histogram
 *
 * @param hist Histogram
 * @return Entropy in bits
 */
double ComputeEntropy(const Histogram& hist);

// ============================================================================
// Histogram Equalization
// ============================================================================

/**
 * @brief Apply histogram equalization to image
 *
 * @param image Input image
 * @return Equalized image
 */
QImage HistogramEqualize(const QImage& image);

/**
 * @brief Apply histogram equalization in-place
 *
 * @param image Image to equalize (modified in-place)
 */
void HistogramEqualizeInPlace(QImage& image);

/**
 * @brief Compute equalization lookup table
 *
 * @param hist Input histogram
 * @param outputMin Output minimum value
 * @param outputMax Output maximum value
 * @return Lookup table (256 entries for 8-bit)
 */
std::vector<uint8_t> ComputeEqualizationLUT(const Histogram& hist,
                                             double outputMin = 0,
                                             double outputMax = 255);

/**
 * @brief Apply lookup table to image
 *
 * @param image Input image
 * @param lut Lookup table (256 entries)
 * @return Transformed image
 */
QImage ApplyLUT(const QImage& image, const std::vector<uint8_t>& lut);

/**
 * @brief Apply lookup table in-place
 */
void ApplyLUTInPlace(QImage& image, const std::vector<uint8_t>& lut);

// ============================================================================
// CLAHE (Contrast Limited Adaptive Histogram Equalization)
// ============================================================================

/**
 * @brief Apply CLAHE to image
 *
 * @param image Input image
 * @param params CLAHE parameters
 * @return Enhanced image
 */
QImage ApplyCLAHE(const QImage& image, const CLAHEParams& params = CLAHEParams());

/**
 * @brief Apply CLAHE in-place
 */
void ApplyCLAHEInPlace(QImage& image, const CLAHEParams& params = CLAHEParams());

// ============================================================================
// Histogram Matching / Specification
// ============================================================================

/**
 * @brief Match image histogram to target histogram
 *
 * @param image Input image
 * @param targetHist Target histogram to match
 * @return Image with matched histogram
 */
QImage HistogramMatch(const QImage& image, const Histogram& targetHist);

/**
 * @brief Match image histogram to reference image
 *
 * @param image Input image
 * @param reference Reference image
 * @return Image with histogram matching reference
 */
QImage HistogramMatchToImage(const QImage& image, const QImage& reference);

/**
 * @brief Compute histogram matching lookup table
 *
 * @param sourceHist Source histogram
 * @param targetHist Target histogram
 * @return Lookup table
 */
std::vector<uint8_t> ComputeMatchingLUT(const Histogram& sourceHist,
                                         const Histogram& targetHist);

// ============================================================================
// Contrast Stretching
// ============================================================================

/**
 * @brief Linear contrast stretching
 *
 * @param image Input image
 * @param lowPercentile Lower percentile for clipping (0-100)
 * @param highPercentile Upper percentile for clipping (0-100)
 * @param outputMin Output minimum value
 * @param outputMax Output maximum value
 * @return Stretched image
 */
QImage ContrastStretch(const QImage& image,
                        double lowPercentile = 1.0,
                        double highPercentile = 99.0,
                        double outputMin = 0,
                        double outputMax = 255);

/**
 * @brief Auto contrast (full range stretch)
 *
 * @param image Input image
 * @return Contrast-enhanced image
 */
QImage AutoContrast(const QImage& image);

/**
 * @brief Normalize image to specified range
 *
 * @param image Input image
 * @param outputMin Output minimum value
 * @param outputMax Output maximum value
 * @return Normalized image
 */
QImage NormalizeImage(const QImage& image, double outputMin = 0, double outputMax = 255);

// ============================================================================
// Automatic Thresholding
// ============================================================================

/**
 * @brief Compute Otsu threshold
 *
 * @param hist Histogram
 * @return Optimal threshold value
 */
double ComputeOtsuThreshold(const Histogram& hist);

/**
 * @brief Compute Otsu threshold from image
 */
double ComputeOtsuThreshold(const QImage& image);

/**
 * @brief Compute multi-level Otsu thresholds
 *
 * @param hist Histogram
 * @param numThresholds Number of thresholds
 * @return Vector of threshold values
 */
std::vector<double> ComputeMultiOtsuThresholds(const Histogram& hist, int32_t numThresholds);

/**
 * @brief Compute triangle threshold (Zack method)
 *
 * @param hist Histogram
 * @return Threshold value
 */
double ComputeTriangleThreshold(const Histogram& hist);

/**
 * @brief Compute minimum error threshold (Kittler-Illingworth)
 *
 * @param hist Histogram
 * @return Threshold value
 */
double ComputeMinErrorThreshold(const Histogram& hist);

/**
 * @brief Compute Isodata (iterative) threshold
 *
 * @param hist Histogram
 * @param maxIterations Maximum iterations
 * @return Threshold value
 */
double ComputeIsodataThreshold(const Histogram& hist, int32_t maxIterations = 100);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Get histogram peak (mode) index
 */
int32_t FindHistogramPeak(const Histogram& hist);

/**
 * @brief Find all peaks in histogram
 *
 * @param hist Histogram
 * @param minHeight Minimum peak height (relative to max)
 * @param minDistance Minimum distance between peaks
 * @return Vector of peak indices
 */
std::vector<int32_t> FindHistogramPeaks(const Histogram& hist,
                                         double minHeight = 0.1,
                                         int32_t minDistance = 5);

/**
 * @brief Find all valleys in histogram
 */
std::vector<int32_t> FindHistogramValleys(const Histogram& hist,
                                           int32_t minDistance = 5);

/**
 * @brief Smooth histogram
 *
 * @param hist Histogram
 * @param kernelSize Smoothing kernel size (odd number)
 * @return Smoothed histogram
 */
Histogram SmoothHistogram(const Histogram& hist, int32_t kernelSize = 5);

/**
 * @brief Compare histograms using various metrics
 */
enum class HistogramCompareMethod {
    Correlation,        ///< Correlation coefficient
    ChiSquare,          ///< Chi-square distance
    Intersection,       ///< Histogram intersection
    Bhattacharyya,      ///< Bhattacharyya distance
    KLDivergence        ///< Kullback-Leibler divergence
};

/**
 * @brief Compare two histograms
 *
 * @param hist1 First histogram
 * @param hist2 Second histogram
 * @param method Comparison method
 * @return Comparison result (interpretation depends on method)
 */
double CompareHistograms(const Histogram& hist1, const Histogram& hist2,
                          HistogramCompareMethod method);

// ============================================================================
// Template Implementations
// ============================================================================

template<typename T>
Histogram ComputeHistogram(const T* data, int32_t width, int32_t height,
                           int32_t numBins, double minValue, double maxValue) {
    Histogram hist(numBins, minValue, maxValue);

    if (data == nullptr || width <= 0 || height <= 0) {
        return hist;
    }

    double range = maxValue - minValue;
    if (range <= 0) {
        return hist;
    }

    double scale = numBins / range;

    for (int32_t i = 0; i < width * height; ++i) {
        double value = static_cast<double>(data[i]);
        int32_t bin = static_cast<int32_t>((value - minValue) * scale);
        bin = std::max(0, std::min(bin, numBins - 1));
        hist.bins[bin]++;
        hist.totalCount++;
    }

    return hist;
}

} // namespace Qi::Vision::Internal
