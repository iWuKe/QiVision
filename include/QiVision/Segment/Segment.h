#pragma once

#include <QiVision/Core/Export.h>

/**
 * @file Segment.h
 * @brief Image segmentation and thresholding (Halcon-style API)
 *
 * Provides:
 * - Global thresholding (binary, truncate, to-zero)
 * - Auto thresholding (Otsu, Triangle, MinError, Isodata)
 * - Adaptive thresholding (Mean, Gaussian, Sauvola, Niblack)
 * - Dynamic thresholding (compare with reference)
 * - Threshold to region conversion
 *
 * Halcon reference operators:
 * - threshold, auto_threshold, binary_threshold
 * - dyn_threshold, var_threshold, char_threshold
 * - dual_threshold, hysteresis_threshold
 *
 * API Style: QRegion Func(const QImage& image, params...)
 *            void Func(const QImage& src, QImage& dst, params...)
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>
#include <QiVision/Core/Types.h>

#include <cstdint>
#include <string>
#include <vector>

namespace Qi::Vision::Segment {

// =============================================================================
// Enums
// =============================================================================

/**
 * @brief Threshold operation types
 */
enum class ThresholdType {
    Binary,         ///< dst = (src > thresh) ? maxVal : 0
    BinaryInv,      ///< dst = (src > thresh) ? 0 : maxVal
    Truncate,       ///< dst = (src > thresh) ? thresh : src
    ToZero,         ///< dst = (src > thresh) ? src : 0
    ToZeroInv       ///< dst = (src > thresh) ? 0 : src
};

/**
 * @brief Auto threshold method
 */
enum class AutoMethod {
    Otsu,           ///< Otsu's method (maximizes between-class variance)
    Triangle,       ///< Triangle algorithm (Zack method)
    MinError,       ///< Minimum error method (Kittler-Illingworth)
    Isodata,        ///< Iterative isodata method
    Median          ///< Simple median-based threshold
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
 */
enum class LightDark {
    Light,          ///< Select pixels brighter than reference
    Dark,           ///< Select pixels darker than reference
    Equal,          ///< Select pixels equal to reference (within offset)
    NotEqual        ///< Select pixels different from reference
};

/**
 * @brief Result of dual threshold operation
 */
struct QIVISION_API DualThresholdResult {
    QRegion lightRegion;        ///< Pixels > highThreshold
    QRegion darkRegion;         ///< Pixels < lowThreshold
    QRegion middleRegion;       ///< Pixels in [lowThreshold, highThreshold]
    double lowThreshold = 0;    ///< Used low threshold
    double highThreshold = 0;   ///< Used high threshold
};

// =============================================================================
// Global Thresholding
// =============================================================================

/**
 * @brief Apply global threshold to image
 *
 * Equivalent to Halcon's threshold operator.
 *
 * @param src Input image (grayscale)
 * @param dst Output image (same size as src)
 * @param threshold Threshold value
 * @param maxValue Maximum value for binary output
 * @param type Threshold operation type
 *
 * @code
 * QImage binary;
 * Threshold(image, binary, 128);  // Binary threshold at 128
 * @endcode
 */
QIVISION_API void Threshold(const QImage& src, QImage& dst,
               double threshold, double maxValue = 255.0,
               ThresholdType type = ThresholdType::Binary);

/**
 * @brief Apply global threshold (return version)
 */
QIVISION_API QImage Threshold(const QImage& src, double threshold,
                 double maxValue = 255.0,
                 ThresholdType type = ThresholdType::Binary);

/**
 * @brief Range thresholding
 *
 * Pixels in [low, high] become maxValue, others become 0.
 *
 * @param src Input image
 * @param dst Output binary image
 * @param low Lower threshold (inclusive)
 * @param high Upper threshold (inclusive)
 * @param maxValue Output value for in-range pixels
 *
 * @code
 * QImage binary;
 * ThresholdRange(image, binary, 100, 200);  // Keep pixels in [100, 200]
 * @endcode
 */
QIVISION_API void ThresholdRange(const QImage& src, QImage& dst,
                    double low, double high, double maxValue = 255.0);

/**
 * @brief Range thresholding (return version)
 */
QIVISION_API QImage ThresholdRange(const QImage& src, double low, double high,
                      double maxValue = 255.0);

// =============================================================================
// Auto Thresholding
// =============================================================================

/**
 * @brief Automatic thresholding using selected method
 *
 * Equivalent to Halcon's auto_threshold / binary_threshold operator.
 *
 * @param src Input image
 * @param dst Output binary image
 * @param method Auto threshold method
 * @param maxValue Max output value
 * @param computedThreshold Output: computed threshold value (optional)
 *
 * @code
 * QImage binary;
 * double thresh;
 * ThresholdAuto(image, binary, AutoMethod::Otsu, 255.0, &thresh);
 * @endcode
 */
QIVISION_API void ThresholdAuto(const QImage& src, QImage& dst,
                   AutoMethod method = AutoMethod::Otsu,
                   double maxValue = 255.0,
                   double* computedThreshold = nullptr);

/**
 * @brief Otsu auto-threshold (convenience)
 */
QIVISION_API void ThresholdOtsu(const QImage& src, QImage& dst,
                   double maxValue = 255.0,
                   double* computedThreshold = nullptr);

/**
 * @brief Triangle auto-threshold (convenience)
 */
QIVISION_API void ThresholdTriangle(const QImage& src, QImage& dst,
                       double maxValue = 255.0,
                       double* computedThreshold = nullptr);

/**
 * @brief Otsu threshold (return version)
 */
QIVISION_API QImage ThresholdOtsu(const QImage& src, double maxValue = 255.0);

/**
 * @brief Triangle threshold (return version)
 */
QIVISION_API QImage ThresholdTriangle(const QImage& src, double maxValue = 255.0);

/**
 * @brief Compute auto threshold value without applying
 *
 * @param src Input image
 * @param method Auto threshold method
 * @return Computed threshold value
 */
QIVISION_API double ComputeAutoThreshold(const QImage& src, AutoMethod method = AutoMethod::Otsu);

// =============================================================================
// Adaptive Thresholding
// =============================================================================

/**
 * @brief Apply adaptive threshold to image
 *
 * For each pixel, computes local threshold based on neighborhood.
 *
 * @param src Input grayscale image
 * @param dst Output binary image
 * @param method Adaptive method
 * @param blockSize Block size (odd number)
 * @param C Constant to subtract from mean
 */
QIVISION_API void ThresholdAdaptive(const QImage& src, QImage& dst,
                       AdaptiveMethod method, int32_t blockSize, double C);

// =============================================================================
// Multi-level Thresholding
// =============================================================================

/**
 * @brief Apply multi-level thresholding
 *
 * Segments image into (N+1) levels using N thresholds.
 *
 * @param src Input image
 * @param dst Output labeled image
 * @param thresholds Sorted threshold values
 *
 * @code
 * QImage labeled;
 * ThresholdMultiLevel(image, labeled, {85, 170});  // 3 levels
 * @endcode
 */
QIVISION_API void ThresholdMultiLevel(const QImage& src, QImage& dst,
                         const std::vector<double>& thresholds);

/**
 * @brief Multi-level threshold with custom output values
 *
 * @param src Input image
 * @param dst Output labeled image
 * @param thresholds N threshold values
 * @param outputValues N+1 output values
 */
QIVISION_API void ThresholdMultiLevel(const QImage& src, QImage& dst,
                         const std::vector<double>& thresholds,
                         const std::vector<double>& outputValues);

// =============================================================================
// Threshold to Region
// =============================================================================

/**
 * @brief Create region from thresholding
 *
 * Equivalent to Halcon's threshold operator returning region.
 *
 * @param src Input image
 * @param low Lower threshold (inclusive)
 * @param high Upper threshold (inclusive)
 * @return Region containing pixels in range [low, high]
 *
 * @code
 * QRegion region = ThresholdToRegion(image, 100, 255);
 * @endcode
 */
QIVISION_API QRegion ThresholdToRegion(const QImage& src, double low, double high);

/**
 * @brief Create region from binary threshold
 *
 * @param src Input image
 * @param threshold Threshold value
 * @param above If true, region contains pixels > threshold; else pixels < threshold
 * @return Region
 */
QIVISION_API QRegion ThresholdToRegion(const QImage& src, double threshold, bool above = true);

/**
 * @brief Create region from auto-threshold
 *
 * @param src Input image
 * @param method Auto threshold method
 * @param above If true, region contains pixels > threshold
 * @param computedThreshold Output: computed threshold value
 * @return Region
 */
QIVISION_API QRegion ThresholdAutoToRegion(const QImage& src,
                               AutoMethod method = AutoMethod::Otsu,
                               bool above = true,
                               double* computedThreshold = nullptr);

// =============================================================================
// Halcon-style Dynamic Threshold
// =============================================================================

/**
 * @brief Dynamic threshold (compare with reference image)
 *
 * Equivalent to Halcon's dyn_threshold operator.
 * Compares original image with a reference to handle uneven illumination.
 *
 * @param image Original image
 * @param reference Reference/smoothed image
 * @param offset Offset value for comparison
 * @param lightDark Selection mode
 * @return Region of selected pixels
 *
 * @code
 * QImage smooth;
 * GaussFilter(image, smooth, 15.0);
 * QRegion defects = DynThreshold(image, smooth, 10, LightDark::Dark);
 * @endcode
 */
QIVISION_API QRegion DynThreshold(const QImage& image, const QImage& reference,
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
QIVISION_API QRegion DynThreshold(const QImage& image, int32_t filterSize,
                     double offset, LightDark lightDark = LightDark::Light);

/**
 * @brief Dual threshold - separate light and dark regions
 *
 * Equivalent to Halcon's dual_threshold operator.
 *
 * @param image Input image
 * @param lowThreshold Low threshold value
 * @param highThreshold High threshold value
 * @return DualThresholdResult containing light, dark, and middle regions
 */
QIVISION_API DualThresholdResult DualThreshold(const QImage& image,
                                   double lowThreshold, double highThreshold);

/**
 * @brief Variance-based threshold
 *
 * Equivalent to Halcon's var_threshold operator.
 *
 * @param image Input image
 * @param windowSize Window size for variance computation
 * @param varianceThreshold Variance threshold
 * @param lightDark Selection mode (Light = high variance, Dark = low variance)
 * @return Region of selected pixels
 */
QIVISION_API QRegion VarThreshold(const QImage& image, int32_t windowSize,
                     double varianceThreshold, LightDark lightDark = LightDark::Light);

/**
 * @brief Character threshold (optimized for text/documents)
 *
 * Equivalent to Halcon's char_threshold operator.
 *
 * @param image Input image
 * @param sigma Smoothing sigma (typical: 1-3)
 * @param percent Percentile for dark pixels (typical: 90-99)
 * @param lightDark "light" for dark text on light background
 * @return Region containing text
 */
QIVISION_API QRegion CharThreshold(const QImage& image, double sigma = 2.0,
                      double percent = 95.0, LightDark lightDark = LightDark::Dark);

/**
 * @brief Hysteresis threshold
 *
 * Dual threshold with connectivity constraint.
 *
 * @param image Input image
 * @param lowThreshold Low threshold
 * @param highThreshold High threshold
 * @return Region with hysteresis-filtered foreground
 */
QIVISION_API QRegion HysteresisThreshold(const QImage& image,
                            double lowThreshold, double highThreshold);

// =============================================================================
// Domain-aware Threshold Operations
// =============================================================================

/**
 * @brief Threshold with Domain support
 *
 * Only processes pixels within the image's Domain.
 *
 * @param image Input image with Domain
 * @param low Lower threshold
 * @param high Upper threshold
 * @return Region containing thresholded pixels within Domain
 */
QIVISION_API QRegion ThresholdWithDomain(const QImage& image, double low, double high);

/**
 * @brief Dynamic threshold with Domain support
 */
QIVISION_API QRegion DynThresholdWithDomain(const QImage& image, const QImage& reference,
                                double offset, LightDark lightDark = LightDark::Light);

/**
 * @brief Adaptive threshold to region with Domain support
 *
 * @param image Input image with Domain
 * @param method Adaptive method
 * @param blockSize Block size (odd number)
 * @param C Constant to subtract from mean
 */
QIVISION_API QRegion ThresholdAdaptiveToRegion(const QImage& image,
                                   AdaptiveMethod method, int32_t blockSize, double C);

// =============================================================================
// Binary Image Operations
// =============================================================================

/**
 * @brief Invert binary image
 */
QIVISION_API void BinaryInvert(const QImage& src, QImage& dst, double maxValue = 255.0);

/**
 * @brief Binary AND of two images
 */
QIVISION_API void BinaryAnd(const QImage& src1, const QImage& src2, QImage& dst,
               double maxValue = 255.0);

/**
 * @brief Binary OR of two images
 */
QIVISION_API void BinaryOr(const QImage& src1, const QImage& src2, QImage& dst,
              double maxValue = 255.0);

/**
 * @brief Binary XOR of two images
 */
QIVISION_API void BinaryXor(const QImage& src1, const QImage& src2, QImage& dst,
               double maxValue = 255.0);

/**
 * @brief Binary difference (src1 AND NOT src2)
 */
QIVISION_API void BinaryDiff(const QImage& src1, const QImage& src2, QImage& dst,
                double maxValue = 255.0);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Check if image appears to be binary
 */
QIVISION_API bool IsBinaryImage(const QImage& image, double tolerance = 1.0);

/**
 * @brief Count non-zero pixels in image
 */
QIVISION_API uint64_t CountNonZero(const QImage& image);

/**
 * @brief Count pixels in range
 */
QIVISION_API uint64_t CountInRange(const QImage& image, double low, double high);

/**
 * @brief Compute percentage of foreground pixels
 */
QIVISION_API double ComputeForegroundRatio(const QImage& image);

/**
 * @brief Apply mask to image
 */
QIVISION_API void ApplyMask(const QImage& src, const QImage& mask, QImage& dst);

/**
 * @brief Create mask from region
 */
QIVISION_API void RegionToMask(const QRegion& region, QImage& mask);

/**
 * @brief Create region from mask
 */
QIVISION_API QRegion MaskToRegion(const QImage& mask, double threshold = 0);

// =============================================================================
// K-Means Clustering Segmentation
// =============================================================================

/**
 * @brief K-Means initialization method
 */
enum class KMeansInit {
    Random,     ///< Random center selection
    KMeansPP    ///< K-Means++ (better initialization, recommended)
};

/**
 * @brief K-Means feature space
 */
enum class KMeansFeature {
    Gray,           ///< Grayscale intensity only
    RGB,            ///< RGB color space
    HSV,            ///< HSV color space (better for color segmentation)
    Lab,            ///< CIE Lab color space (perceptually uniform)
    GraySpatial,    ///< Grayscale + spatial coordinates (x, y)
    RGBSpatial      ///< RGB + spatial coordinates
};

/**
 * @brief K-Means clustering result
 */
struct QIVISION_API KMeansResult {
    QImage labels;                              ///< Label image (Int16, values 0 to k-1)
    std::vector<std::vector<double>> centers;   ///< Cluster centers (k x feature_dim)
    std::vector<int64_t> clusterSizes;          ///< Pixel count per cluster
    double compactness = 0.0;                   ///< Sum of squared distances to centers
    int32_t iterations = 0;                     ///< Actual iterations performed
    bool converged = false;                     ///< True if converged before max iterations
};

/**
 * @brief K-Means clustering parameters
 */
struct QIVISION_API KMeansParams {
    int32_t k = 2;                              ///< Number of clusters
    KMeansFeature feature = KMeansFeature::Gray;///< Feature space
    KMeansInit init = KMeansInit::KMeansPP;     ///< Initialization method
    int32_t maxIterations = 100;                ///< Maximum iterations
    double epsilon = 1.0;                       ///< Convergence threshold (center movement)
    int32_t attempts = 3;                       ///< Number of attempts with different seeds
    double spatialWeight = 0.5;                 ///< Weight for spatial features (0-1)

    static KMeansParams Default(int32_t k = 2) {
        KMeansParams p;
        p.k = k;
        return p;
    }

    static KMeansParams Color(int32_t k = 3) {
        KMeansParams p;
        p.k = k;
        p.feature = KMeansFeature::HSV;
        return p;
    }

    static KMeansParams Spatial(int32_t k = 2, double spatialWeight = 0.5) {
        KMeansParams p;
        p.k = k;
        p.feature = KMeansFeature::GraySpatial;
        p.spatialWeight = spatialWeight;
        return p;
    }
};

/**
 * @brief K-Means clustering segmentation
 *
 * Segments image into k clusters using K-Means algorithm.
 *
 * @param image Input image (grayscale or color)
 * @param params K-Means parameters
 * @return KMeansResult containing labels, centers, and statistics
 *
 * @code
 * // Simple grayscale segmentation into 3 levels
 * auto result = KMeans(grayImage, KMeansParams::Default(3));
 *
 * // Color segmentation using HSV
 * auto result = KMeans(colorImage, KMeansParams::Color(5));
 * @endcode
 */
QIVISION_API KMeansResult KMeans(const QImage& image, const KMeansParams& params);

/**
 * @brief K-Means clustering (simple interface)
 *
 * @param image Input image
 * @param k Number of clusters
 * @param feature Feature space
 * @return KMeansResult
 */
QIVISION_API KMeansResult KMeans(const QImage& image, int32_t k,
                    KMeansFeature feature = KMeansFeature::Gray);

/**
 * @brief Segment image using K-Means and return recolored image
 *
 * Each pixel is replaced with its cluster center value.
 *
 * @param image Input image
 * @param k Number of clusters
 * @param feature Feature space
 * @return Segmented image with cluster colors
 *
 * @code
 * QImage segmented = KMeansSegment(image, 4);  // Posterize to 4 colors
 * @endcode
 */
QIVISION_API QImage KMeansSegment(const QImage& image, int32_t k,
                     KMeansFeature feature = KMeansFeature::Gray);

/**
 * @brief Segment image using K-Means and return recolored image
 */
QIVISION_API QImage KMeansSegment(const QImage& image, const KMeansParams& params);

/**
 * @brief K-Means clustering to regions
 *
 * Segments image into k regions, one per cluster.
 *
 * @param image Input image
 * @param k Number of clusters
 * @param regions Output regions (size = k)
 * @param feature Feature space
 *
 * @code
 * std::vector<QRegion> regions;
 * KMeansToRegions(image, 3, regions);
 * // regions[0], regions[1], regions[2] are the 3 segmented regions
 * @endcode
 */
QIVISION_API void KMeansToRegions(const QImage& image, int32_t k,
                     std::vector<QRegion>& regions,
                     KMeansFeature feature = KMeansFeature::Gray);

/**
 * @brief K-Means clustering to regions with full parameters
 */
QIVISION_API void KMeansToRegions(const QImage& image, const KMeansParams& params,
                     std::vector<QRegion>& regions);

/**
 * @brief Convert K-Means label image to regions
 *
 * @param labels Label image from KMeansResult
 * @param k Number of clusters
 * @param regions Output regions
 */
QIVISION_API void LabelsToRegions(const QImage& labels, int32_t k,
                     std::vector<QRegion>& regions);

// =============================================================================
// Watershed Segmentation
// =============================================================================

/**
 * @brief Watershed segmentation result
 */
struct QIVISION_API WatershedResult {
    QImage labels;                      ///< Label image (Int16, 0=background, -1=watershed, >0=regions)
    std::vector<QRegion> regions;       ///< Segmented regions (excluding background)
    QRegion watershedLines;             ///< Watershed boundary lines
    int32_t numRegions = 0;             ///< Number of segmented regions
};

/**
 * @brief Watershed segmentation (marker-based)
 *
 * Performs marker-controlled watershed segmentation.
 * Markers define the initial "seeds" for flooding.
 *
 * @param image Input grayscale image
 * @param markers Marker image (Int16):
 *                - 0: Unknown/to be determined
 *                - >0: Foreground markers (each connected component has unique label)
 *                - Typically created from distance transform peaks
 * @return WatershedResult containing labels and regions
 *
 * @code
 * // Separate touching objects
 * QRegion foreground = ThresholdToRegion(image, 128, 255);
 * QImage dist = DistanceTransform(foreground);
 * QImage markers = FindDistancePeaks(dist, 10);  // Find local maxima
 * auto result = Watershed(image, markers);
 * @endcode
 */
QIVISION_API WatershedResult Watershed(const QImage& image, const QImage& markers);

/**
 * @brief Watershed segmentation from binary image
 *
 * Automatically generates markers using distance transform.
 * Useful for separating touching/overlapping objects.
 *
 * @param binaryImage Binary image (foreground objects to separate)
 * @param minDistance Minimum distance between object centers (for marker detection)
 * @return WatershedResult containing separated regions
 *
 * @code
 * // Separate touching cells
 * QRegion cells = ThresholdToRegion(image, 100, 255);
 * QImage binary;
 * RegionToMask(cells, binary);
 * auto result = WatershedBinary(binary, 20);  // Min 20px between centers
 * @endcode
 */
QIVISION_API WatershedResult WatershedBinary(const QImage& binaryImage, double minDistance = 10.0);

/**
 * @brief Watershed segmentation from region
 *
 * Separates connected components within a region using distance-based watershed.
 *
 * @param region Input region containing touching objects
 * @param minDistance Minimum distance between object centers
 * @return WatershedResult containing separated regions
 */
QIVISION_API WatershedResult WatershedRegion(const QRegion& region, double minDistance = 10.0);

/**
 * @brief Gradient-based watershed segmentation
 *
 * Performs watershed on gradient magnitude image.
 * Watershed lines form at edges (high gradient).
 *
 * @param image Input grayscale image
 * @param markers Optional marker image (nullptr for automatic)
 * @param gradientThreshold Minimum gradient for watershed lines
 * @return WatershedResult
 */
QIVISION_API WatershedResult WatershedGradient(const QImage& image,
                                   const QImage* markers = nullptr,
                                   double gradientThreshold = 0.0);

/**
 * @brief Create markers from distance transform
 *
 * Finds local maxima in distance transform as watershed markers.
 *
 * @param distanceImage Distance transform image (Float32)
 * @param minDistance Minimum distance between markers
 * @param minPeakValue Minimum distance value for a peak
 * @return Marker image (Int16, each marker has unique label)
 */
QIVISION_API QImage CreateWatershedMarkers(const QImage& distanceImage,
                              double minDistance = 10.0,
                              double minPeakValue = 5.0);

/**
 * @brief Compute distance transform for watershed
 *
 * @param binaryImage Binary foreground image
 * @return Distance transform image (Float32)
 */
QIVISION_API QImage DistanceTransform(const QImage& binaryImage);

/**
 * @brief Compute distance transform from region
 *
 * @param region Input region
 * @return Distance transform image (Float32)
 */
QIVISION_API QImage DistanceTransform(const QRegion& region);

// =============================================================================
// GMM (Gaussian Mixture Model) Segmentation
// =============================================================================

/**
 * @brief GMM feature space (same as K-Means for consistency)
 */
using GMMFeature = KMeansFeature;

/**
 * @brief GMM initialization method
 */
enum class GMMInit {
    Random,     ///< Random initialization
    KMeans      ///< Initialize with K-Means (recommended, more stable)
};

/**
 * @brief Covariance type for GMM
 */
enum class GMMCovType {
    Full,       ///< Full covariance matrix (most flexible, slower)
    Diagonal,   ///< Diagonal covariance (faster, assumes feature independence)
    Spherical   ///< Spherical covariance (single variance per component)
};

/**
 * @brief Gaussian Mixture Model result
 */
struct QIVISION_API GMMResult {
    QImage labels;                                  ///< Hard labels (most probable component)
    std::vector<QImage> probabilities;              ///< Soft labels: P(k|x) for each component k
    std::vector<double> weights;                    ///< Mixture weights (pi_k), sum to 1
    std::vector<std::vector<double>> means;         ///< Component means (k x dim)
    std::vector<std::vector<double>> covariances;   ///< Covariances (flattened, depends on cov_type)
    double logLikelihood = 0.0;                     ///< Final log-likelihood
    int32_t iterations = 0;                         ///< Number of iterations
    bool converged = false;                         ///< Whether EM converged
};

/**
 * @brief GMM parameters
 */
struct QIVISION_API GMMParams {
    int32_t k = 2;                                  ///< Number of Gaussian components
    GMMFeature feature = GMMFeature::Gray;          ///< Feature space
    GMMInit init = GMMInit::KMeans;                 ///< Initialization method
    GMMCovType covType = GMMCovType::Full;          ///< Covariance type
    int32_t maxIterations = 100;                    ///< Maximum EM iterations
    double epsilon = 1e-4;                          ///< Convergence threshold (log-likelihood change)
    double regularization = 1e-6;                   ///< Covariance regularization (prevent singularity)
    double spatialWeight = 0.5;                     ///< Weight for spatial features (0-1)

    /// Default parameters
    static GMMParams Default(int32_t k = 2) {
        GMMParams p;
        p.k = k;
        return p;
    }

    /// Parameters for color segmentation
    static GMMParams Color(int32_t k = 3) {
        GMMParams p;
        p.k = k;
        p.feature = GMMFeature::HSV;
        return p;
    }

    /// Parameters with spatial features
    static GMMParams Spatial(int32_t k = 2, double spatialWeight = 0.5) {
        GMMParams p;
        p.k = k;
        p.feature = GMMFeature::GraySpatial;
        p.spatialWeight = spatialWeight;
        return p;
    }

    /// Fast parameters (diagonal covariance)
    static GMMParams Fast(int32_t k = 2) {
        GMMParams p;
        p.k = k;
        p.covType = GMMCovType::Diagonal;
        return p;
    }
};

/**
 * @brief Gaussian Mixture Model segmentation (EM algorithm)
 *
 * GMM provides soft clustering where each pixel has a probability
 * of belonging to each Gaussian component. This is useful for:
 * - Overlapping distributions (foreground/background modeling)
 * - Probabilistic segmentation
 * - Color/texture modeling with uncertainty
 *
 * @param image Input image (grayscale or color)
 * @param params GMM parameters
 * @return GMMResult containing labels, probabilities, and model parameters
 *
 * @code
 * // Basic grayscale GMM segmentation
 * auto result = GMM(grayImage, GMMParams::Default(3));
 *
 * // Color segmentation with soft labels
 * auto result = GMM(colorImage, GMMParams::Color(5));
 * // Access probability map for component 0
 * QImage prob0 = result.probabilities[0];
 * @endcode
 */
QIVISION_API GMMResult GMM(const QImage& image, const GMMParams& params);

/**
 * @brief GMM segmentation (simple interface)
 *
 * @param image Input image
 * @param k Number of Gaussian components
 * @param feature Feature space
 * @return GMMResult
 */
QIVISION_API GMMResult GMM(const QImage& image, int32_t k,
                           GMMFeature feature = GMMFeature::Gray);

/**
 * @brief Segment image using GMM and return hard-labeled image
 *
 * Returns an image where each pixel is colored according to its
 * most probable Gaussian component.
 *
 * @param image Input image
 * @param k Number of components
 * @param feature Feature space
 * @return Segmented image with component colors
 */
QIVISION_API QImage GMMSegment(const QImage& image, int32_t k,
                               GMMFeature feature = GMMFeature::Gray);

/**
 * @brief Segment image using GMM (full parameters)
 */
QIVISION_API QImage GMMSegment(const QImage& image, const GMMParams& params);

/**
 * @brief GMM segmentation to regions (hard assignment)
 *
 * @param image Input image
 * @param k Number of components
 * @param regions Output regions (one per component)
 * @param feature Feature space
 */
QIVISION_API void GMMToRegions(const QImage& image, int32_t k,
                               std::vector<QRegion>& regions,
                               GMMFeature feature = GMMFeature::Gray);

/**
 * @brief GMM segmentation to regions (full parameters)
 */
QIVISION_API void GMMToRegions(const QImage& image, const GMMParams& params,
                               std::vector<QRegion>& regions);

/**
 * @brief Get probability maps from GMM
 *
 * Returns probability images where each pixel value represents
 * the probability P(component k | pixel) scaled to [0, 255].
 *
 * @param image Input image
 * @param k Number of components
 * @param probMaps Output probability maps (k images, UInt8)
 * @param feature Feature space
 *
 * @code
 * std::vector<QImage> probs;
 * GMMProbabilities(image, 2, probs);
 * // probs[0]: probability of foreground
 * // probs[1]: probability of background
 * @endcode
 */
QIVISION_API void GMMProbabilities(const QImage& image, int32_t k,
                                   std::vector<QImage>& probMaps,
                                   GMMFeature feature = GMMFeature::Gray);

/**
 * @brief Get probability maps from GMM (full parameters)
 */
QIVISION_API void GMMProbabilities(const QImage& image, const GMMParams& params,
                                   std::vector<QImage>& probMaps);

/**
 * @brief Classify pixels using trained GMM model
 *
 * Applies a pre-trained GMM model to a new image.
 *
 * @param image Input image to classify
 * @param model Pre-trained GMMResult containing model parameters
 * @param feature Feature space (must match training)
 * @return Labels image with component assignments
 */
QIVISION_API QImage GMMClassify(const QImage& image, const GMMResult& model,
                                GMMFeature feature = GMMFeature::Gray);

} // namespace Qi::Vision::Segment
