#pragma once

/**
 * @file Caliper.h
 * @brief Caliper measurement functions
 *
 * Provides:
 * - Edge position measurement (MeasurePos)
 * - Edge pair/width measurement (MeasurePairs)
 * - Fuzzy (robust) variants with scoring
 * - Profile extraction and analysis
 *
 * Precision targets (standard conditions: contrast>=50, noise sigma<=5):
 * - Position: < 0.03 px (1 sigma)
 * - Width: < 0.05 px (1 sigma)
 *
 * Thread safety:
 * - All functions are thread-safe for const handles
 * - Multiple threads can measure with same handle simultaneously
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Measure/MeasureTypes.h>
#include <QiVision/Measure/MeasureHandle.h>

#include <vector>

namespace Qi::Vision::Measure {

// =============================================================================
// Edge Position Measurement
// =============================================================================

/**
 * @brief Measure edge positions in rectangular region
 *
 * Extracts profile along the measurement direction, detects edges using
 * 1D edge detection with optional Gaussian smoothing.
 *
 * @param image Input image (grayscale)
 * @param handle Measurement handle
 * @param params Measurement parameters
 * @return Vector of edge results (may be empty if no edges found)
 *
 * @note Results are sorted by profile position (ascending)
 * @note Use higher sigma for noisy images
 * @note Use higher numLines for averaging across texture
 *
 * Example:
 * @code
 * auto handle = CreateMeasureRect(100, 200, 0, 50, 10);
 * auto params = MeasureParams().SetSigma(1.5).SetMinAmplitude(30);
 * auto edges = MeasurePos(image, handle, params);
 * for (const auto& e : edges) {
 *     std::cout << "Edge at (" << e.column << ", " << e.row << ")\n";
 * }
 * @endcode
 */
std::vector<EdgeResult> MeasurePos(const QImage& image,
                                    const MeasureRectangle2& handle,
                                    const MeasureParams& params = MeasureParams());

/**
 * @brief Measure edge positions along arc
 */
std::vector<EdgeResult> MeasurePos(const QImage& image,
                                    const MeasureArc& handle,
                                    const MeasureParams& params = MeasureParams());

/**
 * @brief Measure edge positions along concentric circles (radial)
 */
std::vector<EdgeResult> MeasurePos(const QImage& image,
                                    const MeasureConcentricCircles& handle,
                                    const MeasureParams& params = MeasureParams());

// =============================================================================
// Edge Pair (Width) Measurement
// =============================================================================

/**
 * @brief Measure edge pairs (width) in rectangular region
 *
 * Detects pairs of edges for width measurement. By default, finds
 * positive (rising) followed by negative (falling) edges.
 *
 * @param image Input image (grayscale)
 * @param handle Measurement handle
 * @param params Pair measurement parameters
 * @return Vector of pair results (may be empty if no pairs found)
 *
 * @note firstTransition and secondTransition control pairing
 * @note Use minWidth/maxWidth to filter by expected width
 *
 * Example:
 * @code
 * auto handle = CreateMeasureRect(100, 200, 0, 100, 10);
 * auto params = PairParams()
 *     .SetMinAmplitude(25)
 *     .SetWidthRange(5, 50)
 *     .SetPairSelectMode(PairSelectMode::Strongest);
 * auto pairs = MeasurePairs(image, handle, params);
 * if (!pairs.empty()) {
 *     std::cout << "Width: " << pairs[0].width << " pixels\n";
 * }
 * @endcode
 */
std::vector<PairResult> MeasurePairs(const QImage& image,
                                      const MeasureRectangle2& handle,
                                      const PairParams& params = PairParams());

/**
 * @brief Measure edge pairs along arc
 */
std::vector<PairResult> MeasurePairs(const QImage& image,
                                      const MeasureArc& handle,
                                      const PairParams& params = PairParams());

/**
 * @brief Measure edge pairs along concentric circles
 */
std::vector<PairResult> MeasurePairs(const QImage& image,
                                      const MeasureConcentricCircles& handle,
                                      const PairParams& params = PairParams());

// =============================================================================
// Fuzzy (Robust) Measurement
// =============================================================================

/**
 * @brief Fuzzy edge position measurement with scoring
 *
 * Similar to MeasurePos but computes quality scores for each edge.
 * Scores are based on amplitude, local contrast, and consistency.
 * Better for uncertain edge conditions.
 *
 * @param image Input image
 * @param handle Measurement handle
 * @param params Fuzzy measurement parameters
 * @param stats Optional output statistics
 * @return Vector of edge results with scores
 *
 * @note Scores in [0, 1], higher is better
 * @note Low-score edges can be filtered by minScore
 */
std::vector<EdgeResult> FuzzyMeasurePos(const QImage& image,
                                         const MeasureRectangle2& handle,
                                         const FuzzyParams& params = FuzzyParams(),
                                         MeasureStats* stats = nullptr);

std::vector<EdgeResult> FuzzyMeasurePos(const QImage& image,
                                         const MeasureArc& handle,
                                         const FuzzyParams& params = FuzzyParams(),
                                         MeasureStats* stats = nullptr);

std::vector<EdgeResult> FuzzyMeasurePos(const QImage& image,
                                         const MeasureConcentricCircles& handle,
                                         const FuzzyParams& params = FuzzyParams(),
                                         MeasureStats* stats = nullptr);

/**
 * @brief Fuzzy edge pair measurement with scoring
 *
 * @param image Input image
 * @param handle Measurement handle
 * @param params Fuzzy measurement parameters
 * @param stats Optional output statistics
 * @return Vector of pair results with scores
 */
std::vector<PairResult> FuzzyMeasurePairs(const QImage& image,
                                           const MeasureRectangle2& handle,
                                           const FuzzyParams& params = FuzzyParams(),
                                           MeasureStats* stats = nullptr);

std::vector<PairResult> FuzzyMeasurePairs(const QImage& image,
                                           const MeasureArc& handle,
                                           const FuzzyParams& params = FuzzyParams(),
                                           MeasureStats* stats = nullptr);

std::vector<PairResult> FuzzyMeasurePairs(const QImage& image,
                                           const MeasureConcentricCircles& handle,
                                           const FuzzyParams& params = FuzzyParams(),
                                           MeasureStats* stats = nullptr);

// =============================================================================
// Profile Extraction (for debugging/visualization)
// =============================================================================

/**
 * @brief Extract measurement profile from image using rectangle handle
 *
 * @param image Input image
 * @param handle Measurement handle
 * @param interp Interpolation method
 * @return Profile data (gray values along profile)
 */
std::vector<double> ExtractMeasureProfile(const QImage& image,
                                           const MeasureRectangle2& handle,
                                           ProfileInterpolation interp = ProfileInterpolation::Bilinear);

/**
 * @brief Extract measurement profile from image using arc handle
 */
std::vector<double> ExtractMeasureProfile(const QImage& image,
                                           const MeasureArc& handle,
                                           ProfileInterpolation interp = ProfileInterpolation::Bilinear);

/**
 * @brief Extract measurement profile from image using concentric handle
 */
std::vector<double> ExtractMeasureProfile(const QImage& image,
                                           const MeasureConcentricCircles& handle,
                                           ProfileInterpolation interp = ProfileInterpolation::Bilinear);

// =============================================================================
// Coordinate Transformation
// =============================================================================

/**
 * @brief Convert profile position to image coordinates for rectangle handle
 *
 * @param handle Measurement handle
 * @param profilePos Position along profile [0, ProfileLength]
 * @return Image coordinates (x=column, y=row)
 */
Point2d ProfileToImage(const MeasureRectangle2& handle, double profilePos);

/**
 * @brief Convert profile position to image coordinates for arc handle
 */
Point2d ProfileToImage(const MeasureArc& handle, double profilePos);

/**
 * @brief Convert profile position to image coordinates for concentric handle
 */
Point2d ProfileToImage(const MeasureConcentricCircles& handle, double profilePos);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Compute expected number of samples for handle
 */
int32_t GetNumSamples(const MeasureRectangle2& handle);
int32_t GetNumSamples(const MeasureArc& handle);
int32_t GetNumSamples(const MeasureConcentricCircles& handle);

/**
 * @brief Filter edges by selection mode
 */
std::vector<EdgeResult> SelectEdges(const std::vector<EdgeResult>& edges,
                                     EdgeSelectMode mode,
                                     int32_t maxCount = MAX_EDGES);

/**
 * @brief Filter pairs by selection mode
 */
std::vector<PairResult> SelectPairs(const std::vector<PairResult>& pairs,
                                     PairSelectMode mode,
                                     int32_t maxCount = MAX_EDGES);

/**
 * @brief Sort edges by various criteria
 */
enum class EdgeSortBy { Position, Amplitude, Score };
void SortEdges(std::vector<EdgeResult>& edges, EdgeSortBy criterion, bool ascending = true);

/**
 * @brief Sort pairs by various criteria
 */
enum class PairSortBy { Position, Width, Score, Symmetry };
void SortPairs(std::vector<PairResult>& pairs, PairSortBy criterion, bool ascending = true);

} // namespace Qi::Vision::Measure
