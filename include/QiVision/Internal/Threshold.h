#pragma once

/**
 * @file Threshold.h
 * @brief Image thresholding operations
 *
 * Provides:
 * - Global thresholding (binary, truncate, to-zero)
 * - Adaptive thresholding (mean, Gaussian, Sauvola, Niblack)
 * - Multi-level thresholding
 * - Range thresholding
 * - Auto-threshold wrappers (Otsu, Triangle)
 *
 * Note: Threshold value computation (Otsu, Triangle, etc.) is in Histogram.h
 * This module focuses on applying thresholds to images.
 *
 * Used by:
 * - Blob analysis (preprocessing)
 * - OCR/Barcode (binarization)
 * - Defect detection
 * - Region extraction
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>

#include <cstdint>
#include <vector>

namespace Qi::Vision::Internal {

// ============================================================================
// Constants
// ============================================================================

/// Default adaptive threshold block size
constexpr int32_t DEFAULT_ADAPTIVE_BLOCK_SIZE = 11;

/// Default constant subtracted from mean in adaptive thresholding
constexpr double DEFAULT_ADAPTIVE_C = 2.0;

/// Default k parameter for Sauvola thresholding
constexpr double DEFAULT_SAUVOLA_K = 0.5;

/// Default k parameter for Niblack thresholding
constexpr double DEFAULT_NIBLACK_K = -0.2;

/// Default R parameter (dynamic range) for Sauvola
constexpr double DEFAULT_SAUVOLA_R = 128.0;

// ============================================================================
// Enums
// ============================================================================

/**
 * @brief Global threshold operation types
 */
enum class ThresholdType {
    Binary,         ///< dst = (src > thresh) ? maxVal : 0
    BinaryInv,      ///< dst = (src > thresh) ? 0 : maxVal
    Truncate,       ///< dst = (src > thresh) ? thresh : src
    ToZero,         ///< dst = (src > thresh) ? src : 0
    ToZeroInv       ///< dst = (src > thresh) ? 0 : src
};

/**
 * @brief Adaptive threshold methods
 */
enum class AdaptiveMethod {
    Mean,           ///< T(x,y) = mean(blockSize) - C
    Gaussian,       ///< T(x,y) = Gaussian-weighted mean - C
    Sauvola,        ///< T(x,y) = mean * (1 + k * (stddev/R - 1))
    Niblack,        ///< T(x,y) = mean + k * stddev
    Wolf            ///< T(x,y) = mean - k * (1 - stddev/max_stddev) * (mean - min_val)
};

/**
 * @brief Light/Dark mode for dynamic threshold
 *
 * Corresponds to Halcon's dyn_threshold LightDark parameter
 */
enum class LightDark {
    Light,          ///< Select pixels brighter than reference
    Dark,           ///< Select pixels darker than reference
    Equal,          ///< Select pixels equal to reference (within offset)
    NotEqual        ///< Select pixels different from reference
};

/**
 * @brief Auto threshold method selection
 */
enum class AutoThresholdMethod {
    Otsu,           ///< Otsu's method (maximizes between-class variance)
    Triangle,       ///< Triangle algorithm
    MinError,       ///< Minimum error method
    Isodata,        ///< Iterative isodata method
    Median          ///< Simple median-based threshold
};

// ============================================================================
// Parameters
// ============================================================================

/**
 * @brief Adaptive thresholding parameters
 */
struct AdaptiveThresholdParams {
    AdaptiveMethod method = AdaptiveMethod::Mean;   ///< Method
    int32_t blockSize = DEFAULT_ADAPTIVE_BLOCK_SIZE; ///< Block size (must be odd)
    double C = DEFAULT_ADAPTIVE_C;                   ///< Constant to subtract
    double k = DEFAULT_SAUVOLA_K;                    ///< k for Sauvola/Niblack
    double R = DEFAULT_SAUVOLA_R;                    ///< R for Sauvola
    double maxValue = 255.0;                         ///< Max value for binary output

    /// Factory for mean adaptive threshold
    static AdaptiveThresholdParams Mean(int32_t blockSz = DEFAULT_ADAPTIVE_BLOCK_SIZE,
                                        double c = DEFAULT_ADAPTIVE_C) {
        AdaptiveThresholdParams p;
        p.method = AdaptiveMethod::Mean;
        p.blockSize = blockSz;
        p.C = c;
        return p;
    }

    /// Factory for Gaussian adaptive threshold
    static AdaptiveThresholdParams Gaussian(int32_t blockSz = DEFAULT_ADAPTIVE_BLOCK_SIZE,
                                            double c = DEFAULT_ADAPTIVE_C) {
        AdaptiveThresholdParams p;
        p.method = AdaptiveMethod::Gaussian;
        p.blockSize = blockSz;
        p.C = c;
        return p;
    }

    /// Factory for Sauvola threshold
    static AdaptiveThresholdParams Sauvola(int32_t blockSz = DEFAULT_ADAPTIVE_BLOCK_SIZE,
                                            double k = DEFAULT_SAUVOLA_K,
                                            double R = DEFAULT_SAUVOLA_R) {
        AdaptiveThresholdParams p;
        p.method = AdaptiveMethod::Sauvola;
        p.blockSize = blockSz;
        p.k = k;
        p.R = R;
        p.C = 0;  // Sauvola doesn't use C
        return p;
    }

    /// Factory for Niblack threshold
    static AdaptiveThresholdParams Niblack(int32_t blockSz = DEFAULT_ADAPTIVE_BLOCK_SIZE,
                                            double k = DEFAULT_NIBLACK_K) {
        AdaptiveThresholdParams p;
        p.method = AdaptiveMethod::Niblack;
        p.blockSize = blockSz;
        p.k = k;
        p.C = 0;  // Niblack doesn't use C
        return p;
    }

    /// Factory for Wolf threshold (improved Sauvola for uneven illumination)
    static AdaptiveThresholdParams Wolf(int32_t blockSz = DEFAULT_ADAPTIVE_BLOCK_SIZE,
                                         double k = 0.5) {
        AdaptiveThresholdParams p;
        p.method = AdaptiveMethod::Wolf;
        p.blockSize = blockSz;
        p.k = k;
        p.C = 0;
        return p;
    }
};

/**
 * @brief Result of dual threshold operation
 */
struct DualThresholdResult {
    QRegion lightRegion;        ///< Pixels > highThreshold
    QRegion darkRegion;         ///< Pixels < lowThreshold
    QRegion middleRegion;       ///< Pixels in [lowThreshold, highThreshold]
    double lowThreshold = 0;    ///< Used low threshold
    double highThreshold = 0;   ///< Used high threshold
};

// ============================================================================
// Global Thresholding
// ============================================================================

/**
 * @brief Apply global threshold to image
 *
 * @param src Input image (grayscale)
 * @param dst Output image (same size as src)
 * @param threshold Threshold value
 * @param maxValue Maximum value for binary output
 * @param type Threshold operation type
 */
void ThresholdGlobal(const QImage& src, QImage& dst,
                     double threshold, double maxValue = 255.0,
                     ThresholdType type = ThresholdType::Binary);

/**
 * @brief Apply global threshold and return result
 *
 * @param src Input image (grayscale)
 * @param threshold Threshold value
 * @param maxValue Maximum value for binary output
 * @param type Threshold operation type
 * @return Thresholded image
 */
QImage ThresholdGlobal(const QImage& src, double threshold,
                       double maxValue = 255.0,
                       ThresholdType type = ThresholdType::Binary);

/**
 * @brief Apply threshold to raw data
 *
 * @param src Source data
 * @param dst Destination data (output)
 * @param count Number of elements
 * @param threshold Threshold value
 * @param maxValue Max output value
 * @param type Threshold type
 */
template<typename T>
void ThresholdGlobal(const T* src, T* dst, size_t count,
                     double threshold, double maxValue = 255.0,
                     ThresholdType type = ThresholdType::Binary);

/**
 * @brief Threshold above value (convenience function)
 * Pixels > threshold become maxValue, others become 0
 */
void ThresholdAbove(const QImage& src, QImage& dst,
                    double threshold, double maxValue = 255.0);

/**
 * @brief Threshold below value (convenience function)
 * Pixels < threshold become maxValue, others become 0
 */
void ThresholdBelow(const QImage& src, QImage& dst,
                    double threshold, double maxValue = 255.0);

/**
 * @brief Range thresholding
 * Pixels in [low, high] become maxValue, others become 0
 *
 * @param src Input image
 * @param dst Output binary image
 * @param low Lower threshold (inclusive)
 * @param high Upper threshold (inclusive)
 * @param maxValue Output value for in-range pixels
 */
void ThresholdRange(const QImage& src, QImage& dst,
                    double low, double high, double maxValue = 255.0);

/**
 * @brief Range thresholding (return version)
 */
QImage ThresholdRange(const QImage& src, double low, double high,
                      double maxValue = 255.0);

// ============================================================================
// Adaptive Thresholding
// ============================================================================

/**
 * @brief Apply adaptive threshold to image
 *
 * For each pixel, computes local threshold based on neighborhood:
 * - Mean: T = mean - C
 * - Gaussian: T = gaussian_weighted_mean - C
 * - Sauvola: T = mean * (1 + k * (stddev/R - 1))
 * - Niblack: T = mean + k * stddev
 *
 * @param src Input grayscale image
 * @param dst Output binary image
 * @param params Adaptive threshold parameters
 */
void ThresholdAdaptive(const QImage& src, QImage& dst,
                       const AdaptiveThresholdParams& params);

/**
 * @brief Apply adaptive threshold (return version)
 */
QImage ThresholdAdaptive(const QImage& src,
                         const AdaptiveThresholdParams& params);

/**
 * @brief Compute local threshold map
 *
 * Returns the threshold value for each pixel (useful for debugging)
 *
 * @param src Input image
 * @param params Adaptive threshold parameters
 * @return Float image with threshold values
 */
QImage ComputeLocalThresholdMap(const QImage& src,
                                 const AdaptiveThresholdParams& params);

/**
 * @brief Compute integral image for fast mean computation
 *
 * @param src Input image
 * @param integralSum Output integral sum image
 * @param integralSqSum Output integral squared sum image (for stddev)
 */
void ComputeIntegralImages(const QImage& src,
                           std::vector<double>& integralSum,
                           std::vector<double>& integralSqSum);

/**
 * @brief Get mean and stddev from integral images
 */
void GetBlockStats(const std::vector<double>& integralSum,
                   const std::vector<double>& integralSqSum,
                   int32_t width, int32_t height,
                   int32_t x, int32_t y, int32_t halfSize,
                   double& mean, double& stddev);

// ============================================================================
// Multi-level Thresholding
// ============================================================================

/**
 * @brief Apply multi-level thresholding
 *
 * Segments image into (N+1) levels using N thresholds.
 * Output values are 0, 255/(N), 2*255/(N), ..., 255
 *
 * @param src Input image
 * @param dst Output labeled image
 * @param thresholds Sorted threshold values
 */
void ThresholdMultiLevel(const QImage& src, QImage& dst,
                         const std::vector<double>& thresholds);

/**
 * @brief Multi-level threshold with custom output values
 *
 * @param src Input image
 * @param dst Output labeled image
 * @param thresholds N threshold values
 * @param outputValues N+1 output values
 */
void ThresholdMultiLevel(const QImage& src, QImage& dst,
                         const std::vector<double>& thresholds,
                         const std::vector<double>& outputValues);

/**
 * @brief Apply multi-level threshold (return version)
 */
QImage ThresholdMultiLevel(const QImage& src,
                           const std::vector<double>& thresholds);

// ============================================================================
// Auto Threshold (convenience wrappers using Histogram.h)
// ============================================================================

/**
 * @brief Automatic thresholding using selected method
 *
 * @param src Input image
 * @param dst Output binary image
 * @param method Auto threshold method
 * @param maxValue Max output value
 * @param computedThreshold Output: computed threshold value
 */
void ThresholdAuto(const QImage& src, QImage& dst,
                   AutoThresholdMethod method = AutoThresholdMethod::Otsu,
                   double maxValue = 255.0,
                   double* computedThreshold = nullptr);

/**
 * @brief Otsu auto-threshold (convenience)
 */
void ThresholdOtsu(const QImage& src, QImage& dst,
                   double maxValue = 255.0,
                   double* computedThreshold = nullptr);

/**
 * @brief Triangle auto-threshold (convenience)
 */
void ThresholdTriangle(const QImage& src, QImage& dst,
                       double maxValue = 255.0,
                       double* computedThreshold = nullptr);

/**
 * @brief Otsu threshold (return version)
 */
QImage ThresholdOtsu(const QImage& src, double maxValue = 255.0);

/**
 * @brief Triangle threshold (return version)
 */
QImage ThresholdTriangle(const QImage& src, double maxValue = 255.0);

// ============================================================================
// Threshold to Region
// ============================================================================

/**
 * @brief Create region from thresholding
 *
 * @param src Input image
 * @param low Lower threshold (inclusive)
 * @param high Upper threshold (inclusive)
 * @return Region containing pixels in range [low, high]
 */
QRegion ThresholdToRegion(const QImage& src, double low, double high);

/**
 * @brief Create region from binary threshold
 *
 * @param src Input image
 * @param threshold Threshold value
 * @param above If true, region contains pixels > threshold; else pixels < threshold
 * @return Region
 */
QRegion ThresholdToRegion(const QImage& src, double threshold, bool above = true);

/**
 * @brief Create region from auto-threshold
 *
 * @param src Input image
 * @param method Auto threshold method
 * @param above If true, region contains pixels > threshold
 * @param computedThreshold Output: computed threshold value
 * @return Region
 */
QRegion ThresholdAutoToRegion(const QImage& src,
                               AutoThresholdMethod method = AutoThresholdMethod::Otsu,
                               bool above = true,
                               double* computedThreshold = nullptr);

// ============================================================================
// Halcon-style Threshold Operations
// ============================================================================

/**
 * @brief Dynamic threshold (compare with smoothed/reference image)
 *
 * Corresponds to Halcon's dyn_threshold operator.
 * Compares original image with a smoothed version to handle uneven illumination.
 *
 * @param image Original image
 * @param reference Reference/smoothed image (typically Gaussian blur of original)
 * @param offset Offset value for comparison
 * @param lightDark Selection mode: Light, Dark, Equal, NotEqual
 * @return Region of selected pixels
 *
 * @note For Light mode: selects pixels where image > reference + offset
 * @note For Dark mode: selects pixels where image < reference - offset
 * @note For Equal mode: selects pixels where |image - reference| <= offset
 * @note For NotEqual mode: selects pixels where |image - reference| > offset
 */
QRegion DynThreshold(const QImage& image, const QImage& reference,
                     double offset, LightDark lightDark = LightDark::Light);

/**
 * @brief Dynamic threshold with auto-generated reference
 *
 * Convenience function that automatically computes Gaussian blur.
 *
 * @param image Input image
 * @param filterSize Size of Gaussian filter for smoothing
 * @param offset Offset value
 * @param lightDark Selection mode
 * @return Region of selected pixels
 */
QRegion DynThreshold(const QImage& image, int32_t filterSize,
                     double offset, LightDark lightDark = LightDark::Light);

/**
 * @brief Dual threshold - separate light and dark regions
 *
 * Corresponds to Halcon's dual_threshold operator.
 * Separates image into light (above high threshold), dark (below low threshold),
 * and middle regions.
 *
 * @param image Input image
 * @param lowThreshold Low threshold value
 * @param highThreshold High threshold value
 * @return DualThresholdResult containing light, dark, and middle regions
 */
DualThresholdResult DualThreshold(const QImage& image,
                                   double lowThreshold, double highThreshold);

/**
 * @brief Dual threshold with automatic threshold computation
 *
 * Uses histogram analysis to find optimal thresholds.
 *
 * @param image Input image
 * @param method Method for computing thresholds ("otsu", "histogram_valley")
 * @return DualThresholdResult
 */
DualThresholdResult DualThresholdAuto(const QImage& image,
                                       const std::string& method = "otsu");

/**
 * @brief Variance-based threshold
 *
 * Corresponds to Halcon's var_threshold operator.
 * Selects regions based on local variance (useful for texture detection).
 *
 * @param image Input image
 * @param windowSize Window size for variance computation
 * @param varianceThreshold Variance threshold
 * @param lightDark Selection mode (Light = high variance, Dark = low variance)
 * @return Region of selected pixels
 */
QRegion VarThreshold(const QImage& image, int32_t windowSize,
                     double varianceThreshold, LightDark lightDark = LightDark::Light);

/**
 * @brief Character threshold (optimized for text/document images)
 *
 * Corresponds to Halcon's char_threshold operator.
 * Specialized threshold for binarizing text in documents.
 *
 * @param image Input image
 * @param sigma Smoothing sigma (typical: 1-3)
 * @param percent Percentile for dark pixels (typical: 90-99)
 * @param lightDark "light" for dark text on light background, "dark" for light text
 * @return Region containing text
 */
QRegion CharThreshold(const QImage& image, double sigma = 2.0,
                      double percent = 95.0, LightDark lightDark = LightDark::Dark);

/**
 * @brief Hysteresis threshold
 *
 * Dual threshold with connectivity constraint.
 * Pixels above highThreshold are definite foreground.
 * Pixels between lowThreshold and highThreshold are foreground
 * only if connected to definite foreground.
 *
 * @param image Input image
 * @param lowThreshold Low threshold
 * @param highThreshold High threshold
 * @return Region with hysteresis-filtered foreground
 */
QRegion HysteresisThresholdToRegion(const QImage& image,
                                     double lowThreshold, double highThreshold);

// ============================================================================
// Domain-aware Threshold Operations
// ============================================================================

/**
 * @brief Threshold with Domain support
 *
 * Only processes pixels within the image's Domain.
 * Returns region relative to full image but only contains Domain pixels.
 *
 * @param image Input image with Domain
 * @param low Lower threshold
 * @param high Upper threshold
 * @return Region containing thresholded pixels within Domain
 */
QRegion ThresholdWithDomain(const QImage& image, double low, double high);

/**
 * @brief Dynamic threshold with Domain support
 *
 * @param image Original image with Domain
 * @param reference Reference image
 * @param offset Offset value
 * @param lightDark Selection mode
 * @return Region within Domain
 */
QRegion DynThresholdWithDomain(const QImage& image, const QImage& reference,
                                double offset, LightDark lightDark = LightDark::Light);

/**
 * @brief Adaptive threshold with Domain support
 *
 * @param image Input image with Domain
 * @param params Adaptive threshold parameters
 * @return Region within Domain
 */
QRegion ThresholdAdaptiveToRegion(const QImage& image,
                                   const AdaptiveThresholdParams& params);

// ============================================================================
// Binary Image Operations
// ============================================================================

/**
 * @brief Invert binary image
 * dst = maxValue - src
 */
void BinaryInvert(const QImage& src, QImage& dst, double maxValue = 255.0);

/**
 * @brief Binary AND of two images
 * dst = (src1 > 0) && (src2 > 0) ? maxValue : 0
 */
void BinaryAnd(const QImage& src1, const QImage& src2, QImage& dst,
               double maxValue = 255.0);

/**
 * @brief Binary OR of two images
 * dst = (src1 > 0) || (src2 > 0) ? maxValue : 0
 */
void BinaryOr(const QImage& src1, const QImage& src2, QImage& dst,
              double maxValue = 255.0);

/**
 * @brief Binary XOR of two images
 * dst = ((src1 > 0) != (src2 > 0)) ? maxValue : 0
 */
void BinaryXor(const QImage& src1, const QImage& src2, QImage& dst,
               double maxValue = 255.0);

/**
 * @brief Binary difference
 * dst = (src1 > 0) && !(src2 > 0) ? maxValue : 0
 */
void BinaryDiff(const QImage& src1, const QImage& src2, QImage& dst,
                double maxValue = 255.0);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Check if image appears to be binary (only 0 and maxValue)
 */
bool IsBinaryImage(const QImage& image, double tolerance = 1.0);

/**
 * @brief Count non-zero pixels in image
 */
uint64_t CountNonZero(const QImage& image);

/**
 * @brief Count pixels in range
 */
uint64_t CountInRange(const QImage& image, double low, double high);

/**
 * @brief Compute percentage of foreground pixels (non-zero)
 */
double ComputeForegroundRatio(const QImage& image);

/**
 * @brief Apply mask to image
 * dst = (mask > 0) ? src : 0
 */
void ApplyMask(const QImage& src, const QImage& mask, QImage& dst);

/**
 * @brief Create mask from region
 */
void RegionToMask(const QRegion& region, QImage& mask);

/**
 * @brief Create region from mask
 */
QRegion MaskToRegion(const QImage& mask, double threshold = 0);

// ============================================================================
// Template Implementation
// ============================================================================

template<typename T>
void ThresholdGlobal(const T* src, T* dst, size_t count,
                     double threshold, double maxValue,
                     ThresholdType type) {
    T thresh = static_cast<T>(threshold);
    T maxVal = static_cast<T>(maxValue);

    switch (type) {
        case ThresholdType::Binary:
            for (size_t i = 0; i < count; ++i) {
                dst[i] = (src[i] > thresh) ? maxVal : static_cast<T>(0);
            }
            break;

        case ThresholdType::BinaryInv:
            for (size_t i = 0; i < count; ++i) {
                dst[i] = (src[i] > thresh) ? static_cast<T>(0) : maxVal;
            }
            break;

        case ThresholdType::Truncate:
            for (size_t i = 0; i < count; ++i) {
                dst[i] = (src[i] > thresh) ? thresh : src[i];
            }
            break;

        case ThresholdType::ToZero:
            for (size_t i = 0; i < count; ++i) {
                dst[i] = (src[i] > thresh) ? src[i] : static_cast<T>(0);
            }
            break;

        case ThresholdType::ToZeroInv:
            for (size_t i = 0; i < count; ++i) {
                dst[i] = (src[i] > thresh) ? static_cast<T>(0) : src[i];
            }
            break;
    }
}

}  // namespace Qi::Vision::Internal
