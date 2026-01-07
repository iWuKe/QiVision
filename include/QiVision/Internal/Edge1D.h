#pragma once

/**
 * @file Edge1D.h
 * @brief 1D edge detection for Caliper measurement
 *
 * Provides:
 * - 1D edge detection in profiles
 * - Subpixel edge localization
 * - Edge polarity (positive/negative/any)
 * - Edge pairing for width measurement
 * - Multiple edge detection
 *
 * Used by:
 * - Caliper measurement
 * - Metrology
 * - Edge-based positioning
 *
 * Precision:
 * - Subpixel accuracy < 0.02 pixel (standard conditions)
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Internal/Interpolate.h>

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>

namespace Qi::Vision::Internal {

// Use EdgePolarity from Core/Types.h (Positive, Negative, Both)
// Both = Any direction
using Qi::Vision::EdgePolarity;

/**
 * @brief Edge selection mode
 */
enum class EdgeSelect {
    All,        ///< Return all edges
    First,      ///< First edge only
    Last,       ///< Last edge only
    Strongest,  ///< Strongest edge (highest amplitude)
    Best        ///< Best match (for edge pairs)
};

/**
 * @brief Edge pair selection mode
 */
enum class EdgePairSelect {
    All,        ///< All valid pairs
    FirstLast,  ///< First positive, last negative
    BestPair,   ///< Best matching pair by amplitude
    Closest     ///< Pair with smallest gap
};

/**
 * @brief 1D edge detection result
 */
struct Edge1DResult {
    double position;    ///< Subpixel position along profile
    double amplitude;   ///< Edge strength (gradient magnitude)
    EdgePolarity polarity; ///< Edge direction

    Edge1DResult() : position(0), amplitude(0), polarity(EdgePolarity::Positive) {}
    Edge1DResult(double pos, double amp, EdgePolarity pol)
        : position(pos), amplitude(amp), polarity(pol) {}
};

/**
 * @brief Edge pair result (for width measurement)
 */
struct EdgePairResult {
    Edge1DResult first;     ///< First edge
    Edge1DResult second;    ///< Second edge
    double distance;        ///< Distance between edges (width)

    EdgePairResult() : distance(0) {}
    EdgePairResult(const Edge1DResult& e1, const Edge1DResult& e2)
        : first(e1), second(e2), distance(std::abs(e2.position - e1.position)) {}
};

// ============================================================================
// Profile Gradient Computation
// ============================================================================

/**
 * @brief Compute gradient of 1D profile
 * @param profile Input profile values
 * @param gradient Output gradient (must be pre-allocated)
 * @param length Profile length
 * @param sigma Gaussian smoothing sigma (0 = no smoothing)
 *
 * Gradient is computed as central difference after optional smoothing.
 */
void ComputeProfileGradient(const double* profile, double* gradient,
                            size_t length, double sigma = 0.0);

/**
 * @brief Compute smoothed gradient using Gaussian derivative
 * @param profile Input profile
 * @param gradient Output gradient
 * @param length Profile length
 * @param sigma Gaussian sigma
 *
 * Uses convolution with Gaussian derivative kernel.
 */
void ComputeProfileGradientSmooth(const double* profile, double* gradient,
                                   size_t length, double sigma);

// ============================================================================
// Single Edge Detection
// ============================================================================

/**
 * @brief Detect edges in 1D profile
 * @param profile Input profile values
 * @param length Profile length
 * @param minAmplitude Minimum edge amplitude
 * @param polarity Edge polarity to detect
 * @param sigma Gaussian smoothing sigma (0 = no smoothing)
 * @return Vector of detected edges sorted by position
 */
std::vector<Edge1DResult> DetectEdges1D(const double* profile, size_t length,
                                         double minAmplitude,
                                         EdgePolarity polarity = EdgePolarity::Both,
                                         double sigma = 1.0);

/**
 * @brief Detect single edge (first, last, or strongest)
 * @param profile Input profile
 * @param length Profile length
 * @param minAmplitude Minimum amplitude
 * @param polarity Edge polarity
 * @param select Edge selection mode
 * @param sigma Smoothing sigma
 * @param[out] found True if edge was found
 * @return Edge result (valid only if found=true)
 */
Edge1DResult DetectSingleEdge1D(const double* profile, size_t length,
                                 double minAmplitude,
                                 EdgePolarity polarity,
                                 EdgeSelect select,
                                 double sigma,
                                 bool& found);

// ============================================================================
// Subpixel Refinement
// ============================================================================

/**
 * @brief Refine edge position to subpixel accuracy
 * @param gradient Gradient profile
 * @param length Profile length
 * @param roughPos Rough edge position (integer or near-integer)
 * @return Subpixel position
 *
 * Uses parabolic interpolation around the peak.
 */
double RefineEdgeSubpixel(const double* gradient, size_t length, double roughPos);

/**
 * @brief Refine edge using zero-crossing of second derivative
 * @param profile Original profile
 * @param length Profile length
 * @param roughPos Rough position
 * @param sigma Smoothing sigma
 * @return Subpixel position
 */
double RefineEdgeZeroCrossing(const double* profile, size_t length,
                               double roughPos, double sigma = 1.0);

// ============================================================================
// Edge Pair Detection (for width measurement)
// ============================================================================

/**
 * @brief Detect edge pairs for width measurement
 * @param profile Input profile
 * @param length Profile length
 * @param minAmplitude Minimum edge amplitude
 * @param selection Pair selection mode
 * @param sigma Smoothing sigma
 * @return Vector of edge pairs
 *
 * Pairs consist of positive followed by negative edges (or vice versa).
 */
std::vector<EdgePairResult> DetectEdgePairs1D(const double* profile, size_t length,
                                               double minAmplitude,
                                               EdgePairSelect selection = EdgePairSelect::All,
                                               double sigma = 1.0);

/**
 * @brief Detect single edge pair
 * @param profile Input profile
 * @param length Profile length
 * @param minAmplitude Minimum amplitude
 * @param selection Selection mode
 * @param sigma Smoothing sigma
 * @param[out] found True if pair was found
 * @return Edge pair result
 */
EdgePairResult DetectSinglePair1D(const double* profile, size_t length,
                                   double minAmplitude,
                                   EdgePairSelect selection,
                                   double sigma,
                                   bool& found);

// ============================================================================
// Profile Extraction (from image)
// ============================================================================

/**
 * @brief Extract profile from image along a line
 * @param data Image data
 * @param width Image width
 * @param height Image height
 * @param x0, y0 Start point
 * @param x1, y1 End point
 * @param[out] profile Output profile
 * @param numSamples Number of samples (0 = auto based on line length)
 * @param method Interpolation method
 *
 * Note: Profile is extracted using subpixel interpolation.
 */
template<typename T>
void ExtractProfile(const T* data, int32_t width, int32_t height,
                    double x0, double y0, double x1, double y1,
                    std::vector<double>& profile,
                    size_t numSamples = 0,
                    InterpolationMethod method = InterpolationMethod::Bilinear);

/**
 * @brief Extract profile perpendicular to line at given position
 * @param data Image data
 * @param width Image width
 * @param height Image height
 * @param centerX, centerY Center position on the line
 * @param angle Line angle in radians
 * @param profileLength Profile length (perpendicular extent)
 * @param[out] profile Output profile
 * @param numSamples Number of samples
 * @param method Interpolation method
 */
template<typename T>
void ExtractPerpendicularProfile(const T* data, int32_t width, int32_t height,
                                  double centerX, double centerY,
                                  double angle, double profileLength,
                                  std::vector<double>& profile,
                                  size_t numSamples = 0,
                                  InterpolationMethod method = InterpolationMethod::Bilinear);

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Find local maxima in gradient (candidate edges)
 * @param gradient Gradient profile
 * @param length Profile length
 * @param minAmplitude Minimum amplitude
 * @return Indices of local maxima
 */
std::vector<size_t> FindGradientPeaks(const double* gradient, size_t length,
                                       double minAmplitude);

/**
 * @brief Classify edge polarity from gradient
 */
inline EdgePolarity ClassifyPolarity(double gradientValue) {
    return (gradientValue > 0) ? EdgePolarity::Positive : EdgePolarity::Negative;
}

/**
 * @brief Check if polarity matches filter
 */
inline bool MatchesPolarity(EdgePolarity actual, EdgePolarity filter) {
    return filter == EdgePolarity::Both || actual == filter;
}

// ============================================================================
// Template Implementations
// ============================================================================

template<typename T>
void ExtractProfile(const T* data, int32_t width, int32_t height,
                    double x0, double y0, double x1, double y1,
                    std::vector<double>& profile,
                    size_t numSamples,
                    InterpolationMethod method) {
    // Compute line length
    double dx = x1 - x0;
    double dy = y1 - y0;
    double length = std::sqrt(dx * dx + dy * dy);

    // Auto-determine sample count if not specified
    if (numSamples == 0) {
        numSamples = static_cast<size_t>(std::ceil(length)) + 1;
    }
    if (numSamples < 2) numSamples = 2;

    profile.resize(numSamples);

    if (numSamples == 1) {
        profile[0] = Interpolate(data, width, height, x0, y0, method);
        return;
    }

    // Sample along line
    double stepX = dx / (numSamples - 1);
    double stepY = dy / (numSamples - 1);

    for (size_t i = 0; i < numSamples; ++i) {
        double x = x0 + i * stepX;
        double y = y0 + i * stepY;
        profile[i] = Interpolate(data, width, height, x, y, method);
    }
}

template<typename T>
void ExtractPerpendicularProfile(const T* data, int32_t width, int32_t height,
                                  double centerX, double centerY,
                                  double angle, double profileLength,
                                  std::vector<double>& profile,
                                  size_t numSamples,
                                  InterpolationMethod method) {
    // Perpendicular direction: angle - π/2 points to the right of the line direction
    // This convention: looking along the line direction (angle), perpendicular points right
    double perpAngle = angle - M_PI / 2.0;
    double dx = std::cos(perpAngle);
    double dy = std::sin(perpAngle);

    // Start and end points
    double halfLen = profileLength / 2.0;
    double x0 = centerX - halfLen * dx;
    double y0 = centerY - halfLen * dy;
    double x1 = centerX + halfLen * dx;
    double y1 = centerY + halfLen * dy;

    ExtractProfile(data, width, height, x0, y0, x1, y1, profile, numSamples, method);
}

} // namespace Qi::Vision::Internal
