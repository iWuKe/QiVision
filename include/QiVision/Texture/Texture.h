#pragma once

/**
 * @file Texture.h
 * @brief Texture analysis module
 *
 * Provides texture feature extraction and analysis:
 * - LBP (Local Binary Pattern): Fast texture descriptor
 * - GLCM (Gray Level Co-occurrence Matrix): Statistical texture features
 * - Gabor filters: Multi-scale, multi-orientation texture analysis
 *
 * Applications:
 * - Surface defect detection
 * - Material classification
 * - Fabric inspection
 * - Medical image analysis
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>
#include <vector>
#include <array>

namespace Qi::Vision::Texture {

// =============================================================================
// LBP (Local Binary Pattern)
// =============================================================================

/**
 * @brief LBP variants
 */
enum class LBPType {
    Standard,       ///< Basic 8-neighbor LBP
    Uniform,        ///< Uniform patterns only (58 + 1 bins)
    RotationInvariant,  ///< Rotation invariant LBP
    UniformRI       ///< Uniform + Rotation Invariant (9 + 1 bins)
};

/**
 * @brief Compute LBP (Local Binary Pattern) image
 * @param image Input grayscale image
 * @param lbpImage Output LBP image (UInt8)
 * @param type LBP variant
 *
 * LBP compares each pixel with its 8 neighbors:
 * - Neighbor >= center: bit = 1
 * - Neighbor < center: bit = 0
 * - 8 bits form the LBP code (0-255)
 */
void ComputeLBP(const QImage& image, QImage& lbpImage,
                LBPType type = LBPType::Standard);

/**
 * @brief Compute LBP with radius and sample points
 * @param image Input grayscale image
 * @param lbpImage Output LBP image
 * @param radius Radius for circular neighborhood (default 1)
 * @param numPoints Number of sample points (default 8)
 * @param type LBP variant
 *
 * Extended LBP with configurable radius and points.
 * Uses bilinear interpolation for non-integer positions.
 */
void ComputeLBPExtended(const QImage& image, QImage& lbpImage,
                        int32_t radius = 1, int32_t numPoints = 8,
                        LBPType type = LBPType::Standard);

/**
 * @brief Compute LBP histogram
 * @param lbpImage LBP image (from ComputeLBP)
 * @param histogram Output histogram (normalized)
 * @param type LBP type used (affects bin count)
 * @return Number of bins
 *
 * Histogram bins:
 * - Standard: 256 bins
 * - Uniform: 59 bins (58 uniform + 1 non-uniform)
 * - RotationInvariant: 36 bins
 * - UniformRI: 10 bins (9 uniform + 1 non-uniform)
 */
int32_t ComputeLBPHistogram(const QImage& lbpImage,
                            std::vector<double>& histogram,
                            LBPType type = LBPType::Standard);

/**
 * @brief Compute LBP histogram for a region
 * @param lbpImage LBP image
 * @param region Region of interest
 * @param histogram Output histogram (normalized)
 * @param type LBP type
 * @return Number of bins
 */
int32_t ComputeLBPHistogram(const QImage& lbpImage,
                            const QRegion& region,
                            std::vector<double>& histogram,
                            LBPType type = LBPType::Standard);

// =============================================================================
// GLCM (Gray Level Co-occurrence Matrix)
// =============================================================================

/**
 * @brief GLCM feature set
 */
struct GLCMFeatures {
    double contrast = 0;        ///< Local intensity variation
    double dissimilarity = 0;   ///< Similar to contrast
    double homogeneity = 0;     ///< Closeness of distribution to diagonal
    double energy = 0;          ///< Sum of squared elements (uniformity)
    double entropy = 0;         ///< Randomness of intensity distribution
    double correlation = 0;     ///< Linear dependency of gray levels
    double mean = 0;            ///< Mean gray level
    double variance = 0;        ///< Variance of gray levels
    double asm_ = 0;            ///< Angular Second Moment (same as energy)
    double maxProbability = 0;  ///< Maximum probability in GLCM
};

/**
 * @brief GLCM direction
 */
enum class GLCMDirection {
    Horizontal,     ///< 0° (right)
    Vertical,       ///< 90° (down)
    Diagonal45,     ///< 45° (up-right)
    Diagonal135,    ///< 135° (up-left)
    Average         ///< Average of all 4 directions
};

/**
 * @brief Compute GLCM (Gray Level Co-occurrence Matrix)
 * @param image Input grayscale image
 * @param glcm Output GLCM matrix (numLevels x numLevels)
 * @param distance Pixel distance (default 1)
 * @param direction Direction for co-occurrence
 * @param numLevels Number of gray levels (default 256, can reduce for speed)
 *
 * GLCM[i,j] = probability of pixel with value i having a neighbor
 * at given distance/direction with value j.
 */
void ComputeGLCM(const QImage& image,
                 std::vector<std::vector<double>>& glcm,
                 int32_t distance = 1,
                 GLCMDirection direction = GLCMDirection::Horizontal,
                 int32_t numLevels = 256);

/**
 * @brief Compute GLCM for a region
 */
void ComputeGLCM(const QImage& image,
                 const QRegion& region,
                 std::vector<std::vector<double>>& glcm,
                 int32_t distance = 1,
                 GLCMDirection direction = GLCMDirection::Horizontal,
                 int32_t numLevels = 256);

/**
 * @brief Extract features from GLCM
 * @param glcm GLCM matrix (from ComputeGLCM)
 * @return Feature set
 */
GLCMFeatures ExtractGLCMFeatures(const std::vector<std::vector<double>>& glcm);

/**
 * @brief Compute GLCM features directly from image
 * @param image Input grayscale image
 * @param distance Pixel distance
 * @param direction Direction (or Average for all)
 * @param numLevels Number of gray levels
 * @return Feature set
 *
 * Convenience function that combines ComputeGLCM and ExtractGLCMFeatures.
 */
GLCMFeatures ComputeGLCMFeatures(const QImage& image,
                                  int32_t distance = 1,
                                  GLCMDirection direction = GLCMDirection::Average,
                                  int32_t numLevels = 64);

/**
 * @brief Compute GLCM features for a region
 */
GLCMFeatures ComputeGLCMFeatures(const QImage& image,
                                  const QRegion& region,
                                  int32_t distance = 1,
                                  GLCMDirection direction = GLCMDirection::Average,
                                  int32_t numLevels = 64);

// =============================================================================
// Gabor Filters
// =============================================================================

/**
 * @brief Gabor filter parameters
 */
struct GaborParams {
    double sigma = 3.0;         ///< Gaussian envelope sigma
    double theta = 0.0;         ///< Orientation (radians)
    double lambda = 8.0;        ///< Wavelength of sinusoid
    double gamma = 0.5;         ///< Spatial aspect ratio
    double psi = 0.0;           ///< Phase offset
    int32_t kernelSize = 0;     ///< Kernel size (0 = auto from sigma)

    /// Create with specific orientation (degrees)
    static GaborParams WithOrientation(double degrees) {
        GaborParams p;
        p.theta = degrees * 3.14159265358979323846 / 180.0;
        return p;
    }
};

/**
 * @brief Create Gabor filter kernel
 * @param params Gabor parameters
 * @param kernel Output kernel (Float32 image)
 */
void CreateGaborKernel(const GaborParams& params, QImage& kernel);

/**
 * @brief Apply Gabor filter
 * @param image Input grayscale image
 * @param output Filtered image (Float32)
 * @param params Gabor parameters
 */
void ApplyGaborFilter(const QImage& image, QImage& output,
                      const GaborParams& params);

/**
 * @brief Apply Gabor filter bank (multiple orientations)
 * @param image Input grayscale image
 * @param responses Output: one response image per orientation
 * @param numOrientations Number of orientations (evenly spaced 0-180°)
 * @param sigma Gaussian sigma
 * @param lambda Wavelength
 *
 * Common setup: 4-8 orientations for texture analysis.
 */
void ApplyGaborFilterBank(const QImage& image,
                          std::vector<QImage>& responses,
                          int32_t numOrientations = 8,
                          double sigma = 3.0,
                          double lambda = 8.0);

/**
 * @brief Compute Gabor energy (magnitude response)
 * @param image Input grayscale image
 * @param energy Output energy image
 * @param params Gabor parameters
 *
 * Computes sqrt(real² + imag²) using Gabor pair (0° and 90° phase).
 */
void ComputeGaborEnergy(const QImage& image, QImage& energy,
                        const GaborParams& params);

/**
 * @brief Gabor texture features
 */
struct GaborFeatures {
    std::vector<double> meanEnergy;     ///< Mean energy per orientation
    std::vector<double> stdEnergy;      ///< Std dev of energy per orientation
    double dominantOrientation = 0;     ///< Dominant texture orientation (degrees)
    double orientationStrength = 0;     ///< Strength of dominant orientation
};

/**
 * @brief Extract Gabor texture features
 * @param image Input grayscale image
 * @param numOrientations Number of orientations
 * @param sigma Gaussian sigma
 * @param lambda Wavelength
 * @return Gabor feature set
 */
GaborFeatures ExtractGaborFeatures(const QImage& image,
                                    int32_t numOrientations = 8,
                                    double sigma = 3.0,
                                    double lambda = 8.0);

/**
 * @brief Extract Gabor features for a region
 */
GaborFeatures ExtractGaborFeatures(const QImage& image,
                                    const QRegion& region,
                                    int32_t numOrientations = 8,
                                    double sigma = 3.0,
                                    double lambda = 8.0);

// =============================================================================
// Texture Comparison
// =============================================================================

/**
 * @brief Compare two LBP histograms
 * @param hist1 First histogram
 * @param hist2 Second histogram
 * @return Chi-square distance (0 = identical)
 */
double CompareLBPHistograms(const std::vector<double>& hist1,
                            const std::vector<double>& hist2);

/**
 * @brief Compare two GLCM feature sets
 * @param f1 First feature set
 * @param f2 Second feature set
 * @return Euclidean distance in feature space
 */
double CompareGLCMFeatures(const GLCMFeatures& f1, const GLCMFeatures& f2);

/**
 * @brief Compare two Gabor feature sets
 * @param f1 First feature set
 * @param f2 Second feature set
 * @return Euclidean distance in feature space
 */
double CompareGaborFeatures(const GaborFeatures& f1, const GaborFeatures& f2);

// =============================================================================
// Texture Segmentation
// =============================================================================

/**
 * @brief Segment image by texture using LBP
 * @param image Input grayscale image
 * @param labels Output label image (Int32)
 * @param numClusters Number of texture clusters
 * @param windowSize Window size for local histogram (default 16)
 * @return Actual number of clusters found
 */
int32_t SegmentByTextureLBP(const QImage& image, QImage& labels,
                            int32_t numClusters = 2,
                            int32_t windowSize = 16);

/**
 * @brief Detect texture anomalies using LBP
 * @param image Input grayscale image
 * @param referenceHist Reference texture histogram
 * @param anomalyMap Output anomaly map (Float32, higher = more anomalous)
 * @param windowSize Window size for local analysis
 * @param type LBP type
 */
void DetectTextureAnomalies(const QImage& image,
                            const std::vector<double>& referenceHist,
                            QImage& anomalyMap,
                            int32_t windowSize = 16,
                            LBPType type = LBPType::Uniform);

} // namespace Qi::Vision::Texture
