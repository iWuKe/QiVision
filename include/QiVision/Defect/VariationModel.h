#pragma once

/**
 * @file VariationModel.h
 * @brief Variation Model for defect detection (Halcon-style)
 *
 * The variation model compares test images against a statistical model
 * built from "golden" (good) samples. Regions where the test image
 * deviates significantly from the expected mean (beyond the allowed
 * variance) are flagged as defects.
 *
 * Two modes:
 * 1. Multi-image training: Train with multiple good samples
 * 2. Single-image creation: Generate model from one image + edge analysis
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>
#include <QiVision/Core/Export.h>
#include <QiVision/Segment/Segment.h>  // For LightDark enum
#include <string>
#include <memory>

namespace Qi::Vision::Defect {

// Reuse LightDark enum from Segment module
using Segment::LightDark;

/**
 * @brief Variation Model for defect detection
 *
 * Example usage (multi-image training):
 * @code
 * VariationModel model(width, height);
 * for (const auto& goodImage : goodImages) {
 *     model.Train(goodImage);
 * }
 * model.Prepare();
 * QRegion defects = model.Compare(testImage);
 * @endcode
 *
 * Example usage (single image):
 * @code
 * VariationModel model;
 * model.CreateFromSingleImage(goldenImage, 30.0, 10.0);
 * QRegion defects = model.Compare(testImage);
 * @endcode
 */
class QIVISION_API VariationModel {
public:
    /**
     * @brief Create empty model (for multi-image training)
     * @param width Image width (0 = auto from first training image)
     * @param height Image height (0 = auto from first training image)
     */
    VariationModel(int32_t width = 0, int32_t height = 0);

    ~VariationModel();

    // Move semantics
    VariationModel(VariationModel&& other) noexcept;
    VariationModel& operator=(VariationModel&& other) noexcept;

    // Non-copyable (use Clone() if needed)
    VariationModel(const VariationModel&) = delete;
    VariationModel& operator=(const VariationModel&) = delete;

    // =========================================================================
    // Multi-image training mode
    // =========================================================================

    /**
     * @brief Add a good sample for training
     * @param goodImage Good (defect-free) image
     *
     * Call multiple times with different good samples, then call Prepare().
     */
    void Train(const QImage& goodImage);

    /**
     * @brief Finalize training and compute statistics
     *
     * Computes mean and variance from all trained images.
     * Must be called after Train() and before Compare().
     */
    void Prepare();

    // =========================================================================
    // Single-image mode (with edge-aware variance)
    // =========================================================================

    /**
     * @brief Create model from a single golden image
     * @param golden The reference (golden) image
     * @param edgeTolerance Tolerance for edge regions (default 30.0)
     * @param flatTolerance Tolerance for flat regions (default 10.0)
     * @param edgeSigma Sigma for edge detection (default 1.5)
     * @param edgeDilateRadius Radius to dilate edge regions (default 2)
     *
     * Automatically detects edges and assigns larger variance to edge regions
     * (where slight position shifts are expected) and smaller variance to
     * flat regions (where intensity should be stable).
     */
    void CreateFromSingleImage(
        const QImage& golden,
        double edgeTolerance = 30.0,
        double flatTolerance = 10.0,
        double edgeSigma = 1.5,
        int32_t edgeDilateRadius = 2
    );

    /**
     * @brief Create model from golden image with custom variance image
     * @param golden The reference (mean) image
     * @param varImage Custom variance image (same size, Float32)
     */
    void CreateFromImages(const QImage& golden, const QImage& varImage);

    // =========================================================================
    // Comparison / Detection
    // =========================================================================

    /**
     * @brief Compare test image against model and find defects
     * @param testImage Image to test
     * @param threshold Detection threshold in standard deviations (default 3.0)
     * @return Region containing defect pixels
     *
     * A pixel is flagged as defect if: |test - mean| > threshold * sqrt(var)
     */
    QRegion Compare(const QImage& testImage, double threshold = 3.0) const;

    /**
     * @brief Compare with ROI (only check within region)
     * @param testImage Image to test
     * @param roi Region of interest
     * @param threshold Detection threshold
     * @return Region containing defect pixels (within ROI)
     */
    QRegion Compare(const QImage& testImage, const QRegion& roi,
                    double threshold = 3.0) const;

    /**
     * @brief Get difference image (|test - mean| / sqrt(var))
     * @param testImage Image to test
     * @param diffImage Output: normalized difference image
     */
    void GetDiffImage(const QImage& testImage, QImage& diffImage) const;

    // =========================================================================
    // Model access
    // =========================================================================

    /**
     * @brief Get mean (average) image
     */
    QImage GetMeanImage() const;

    /**
     * @brief Get variance image
     */
    QImage GetVarImage() const;

    /**
     * @brief Set custom variance image
     * @param varImage Variance image (Float32, same size as mean)
     */
    void SetVarImage(const QImage& varImage);

    /**
     * @brief Set minimum variance (floor) to avoid division by zero
     * @param minVar Minimum variance value (default 1.0)
     */
    void SetMinVariance(double minVar);

    /**
     * @brief Get model dimensions
     */
    int32_t Width() const;
    int32_t Height() const;

    /**
     * @brief Check if model is ready for comparison
     */
    bool IsReady() const;

    /**
     * @brief Get number of training images used
     */
    int32_t TrainingCount() const;

    // =========================================================================
    // Serialization
    // =========================================================================

    /**
     * @brief Save model to file
     * @param filename Output filename
     */
    void Write(const std::string& filename) const;

    /**
     * @brief Load model from file
     * @param filename Input filename
     * @return Loaded model
     */
    static VariationModel Read(const std::string& filename);

    /**
     * @brief Create a deep copy
     */
    VariationModel Clone() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// =============================================================================
// Convenience functions
// =============================================================================

/**
 * @brief Quick comparison using single golden image
 * @param golden Reference image
 * @param test Test image
 * @param tolerance Tolerance value (applied uniformly)
 * @return Defect region
 */
QIVISION_API QRegion CompareImages(
    const QImage& golden,
    const QImage& test,
    double tolerance = 10.0
);

/**
 * @brief Quick comparison with edge-aware tolerance
 * @param golden Reference image
 * @param test Test image
 * @param edgeTolerance Tolerance for edges
 * @param flatTolerance Tolerance for flat areas
 * @return Defect region
 */
QIVISION_API QRegion CompareImagesEdgeAware(
    const QImage& golden,
    const QImage& test,
    double edgeTolerance = 30.0,
    double flatTolerance = 10.0
);

/**
 * @brief Simple absolute difference thresholding
 * @param image1 First image
 * @param image2 Second image
 * @param threshold Difference threshold
 * @return Region where |image1 - image2| > threshold
 */
QIVISION_API QRegion AbsDiffThreshold(
    const QImage& image1,
    const QImage& image2,
    double threshold
);

/**
 * @brief Compute absolute difference image
 * @param image1 First image
 * @param image2 Second image
 * @param diffImage Output difference image
 */
QIVISION_API void AbsDiffImage(
    const QImage& image1,
    const QImage& image2,
    QImage& diffImage
);

// =============================================================================
// Local adaptive defect detection
// =============================================================================

/**
 * @brief Local adaptive defect detection using local statistics on diff image
 * @param golden Reference (golden) image (UInt8 only)
 * @param test Test image (UInt8 only)
 * @param blockSize Window size for local statistics (must be odd, default 21)
 * @param k Threshold multiplier in standard deviations (default 3.0)
 * @param lightDark Detection mode (Light/Dark/NotEqual; Equal not supported)
 * @return Region containing defect pixels
 *
 * Algorithm:
 * 1. Compute diff = |test - golden|
 * 2. Compute local mean and stddev of diff within blockSize x blockSize window
 * 3. Mark pixels where diff > localMean + k * localStdDev as defects
 *
 * @note Local statistics are computed on the **difference image**, not on
 * the original images. This approach better suppresses texture/noise but
 * may reduce sensitivity to small stable defects. For scenarios requiring
 * higher sensitivity, consider using VariationModel with trained statistics.
 *
 * @note Only UInt8 images are supported. LightDark::Equal is not supported
 * (throws UnsupportedException).
 *
 * This is more robust to non-uniform illumination compared to global thresholding.
 * The local statistics adapt to the local noise/texture level.
 */
QIVISION_API QRegion LocalAdaptiveCompare(
    const QImage& golden,
    const QImage& test,
    int32_t blockSize = 21,
    double k = 3.0,
    LightDark lightDark = LightDark::NotEqual
);

/**
 * @brief Local adaptive defect detection with separate diff image output
 * @param golden Reference (golden) image
 * @param test Test image
 * @param diffImage Output: absolute difference image
 * @param blockSize Window size for local statistics (must be odd, default 21)
 * @param k Threshold multiplier in standard deviations (default 3.0)
 * @param lightDark Detection mode (default NotEqual = both bright and dark)
 * @return Region containing defect pixels
 */
QIVISION_API QRegion LocalAdaptiveCompare(
    const QImage& golden,
    const QImage& test,
    QImage& diffImage,
    int32_t blockSize = 21,
    double k = 3.0,
    LightDark lightDark = LightDark::NotEqual
);

/**
 * @brief Dynamic threshold defect detection (Halcon dyn_threshold style)
 * @param image Original image (UInt8 only)
 * @param filterSize Size for smoothing filter to generate reference (must be odd)
 * @param offset Threshold offset
 * @param lightDark Detection mode (Light/Dark/NotEqual; Equal not supported)
 * @return Region containing defect pixels
 *
 * Uses a smoothed version of the image as reference:
 * - Light: returns pixels where image > smoothed + offset
 * - Dark: returns pixels where image < smoothed - offset
 * - NotEqual: returns pixels where |image - smoothed| > offset
 *
 * @note Only UInt8 images are supported. LightDark::Equal is not supported
 * (throws UnsupportedException).
 */
QIVISION_API QRegion DynThresholdDefect(
    const QImage& image,
    int32_t filterSize,
    double offset,
    LightDark lightDark = LightDark::NotEqual
);

} // namespace Qi::Vision::Defect
