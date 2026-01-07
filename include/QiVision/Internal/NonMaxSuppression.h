#pragma once

/**
 * @file NonMaxSuppression.h
 * @brief Non-Maximum Suppression (NMS) algorithms
 *
 * Provides:
 * - 1D NMS for profile edge detection (Caliper, Edge1D)
 * - 2D Gradient NMS for Canny edge detection
 * - 2D Feature NMS for keypoint detection (Harris, ShapeModel)
 * - Box NMS for bounding box suppression
 *
 * Used by:
 * - Internal/Edge1D: 1D edge peak detection
 * - Internal/Canny: Edge thinning
 * - Internal/Steger: Ridge/valley detection
 * - Matching/ShapeModel: Response peak detection
 * - Feature detectors: Harris, FAST, etc.
 */

#include <QiVision/Core/Types.h>

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>

namespace Qi::Vision::Internal {

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief A peak detected in 1D signal
 */
struct Peak1D {
    int32_t index = 0;        ///< Index in the signal
    double value = 0.0;       ///< Peak value
    double subpixelIndex = 0.0; ///< Subpixel refined index (if computed)

    bool operator<(const Peak1D& other) const {
        return value > other.value;  // Sort by descending value
    }
};

/**
 * @brief A peak detected in 2D image
 */
struct Peak2D {
    int32_t x = 0;            ///< X coordinate
    int32_t y = 0;            ///< Y coordinate
    double value = 0.0;       ///< Peak value
    double subpixelX = 0.0;   ///< Subpixel refined X (if computed)
    double subpixelY = 0.0;   ///< Subpixel refined Y (if computed)

    bool operator<(const Peak2D& other) const {
        return value > other.value;  // Sort by descending value
    }
};

/**
 * @brief Bounding box for Box NMS
 */
struct BoundingBox {
    double x1 = 0.0;          ///< Left
    double y1 = 0.0;          ///< Top
    double x2 = 0.0;          ///< Right
    double y2 = 0.0;          ///< Bottom
    double score = 0.0;       ///< Confidence score
    int32_t classId = 0;      ///< Class ID (for multi-class NMS)

    double Area() const {
        return std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
    }

    bool operator<(const BoundingBox& other) const {
        return score > other.score;  // Sort by descending score
    }
};

// ============================================================================
// 1D Non-Maximum Suppression
// ============================================================================

/**
 * @brief Find local maxima in 1D signal with simple comparison
 *
 * A point is a local maximum if it's greater than its immediate neighbors.
 *
 * @param signal Input signal
 * @param size Signal length
 * @param minValue Minimum value threshold for peaks
 * @return Vector of peak indices
 */
std::vector<int32_t> FindLocalMaxima1D(const double* signal, size_t size,
                                        double minValue = 0.0);

/**
 * @brief Find local maxima in 1D signal with configurable neighborhood
 *
 * A point is a local maximum if it's greater than all points within
 * the neighborhood radius.
 *
 * @param signal Input signal
 * @param size Signal length
 * @param radius Neighborhood radius (half-window size)
 * @param minValue Minimum value threshold
 * @return Vector of peaks with values
 */
std::vector<Peak1D> FindLocalMaxima1DRadius(const double* signal, size_t size,
                                             int32_t radius,
                                             double minValue = 0.0);

/**
 * @brief Find local maxima (peaks) with subpixel refinement
 *
 * Uses parabolic interpolation to refine peak position.
 *
 * @param signal Input signal
 * @param size Signal length
 * @param radius Neighborhood radius
 * @param minValue Minimum value threshold
 * @param refineSubpixel Whether to compute subpixel position
 * @return Vector of peaks with subpixel positions
 */
std::vector<Peak1D> FindPeaks1D(const double* signal, size_t size,
                                 int32_t radius = 1,
                                 double minValue = 0.0,
                                 bool refineSubpixel = true);

/**
 * @brief Find local minima (valleys) in 1D signal
 *
 * @param signal Input signal
 * @param size Signal length
 * @param radius Neighborhood radius
 * @param maxValue Maximum value threshold (peaks must be below this)
 * @param refineSubpixel Whether to compute subpixel position
 * @return Vector of valleys (value will be negative of actual)
 */
std::vector<Peak1D> FindValleys1D(const double* signal, size_t size,
                                   int32_t radius = 1,
                                   double maxValue = 0.0,
                                   bool refineSubpixel = true);

/**
 * @brief Apply NMS to keep only the strongest peaks
 *
 * Keeps at most N peaks, with minimum distance constraint.
 *
 * @param peaks Input peaks (will be sorted by value)
 * @param maxCount Maximum number of peaks to keep
 * @param minDistance Minimum distance between peaks
 * @return Filtered peaks
 */
std::vector<Peak1D> SuppressPeaks1D(std::vector<Peak1D> peaks,
                                     int32_t maxCount,
                                     double minDistance = 0.0);

// ============================================================================
// 1D Subpixel Refinement
// ============================================================================

/**
 * @brief Refine peak position using parabolic (quadratic) interpolation
 *
 * Fits a parabola to 3 points: (index-1, v0), (index, v1), (index+1, v2)
 * and finds the maximum.
 *
 * @param v0 Value at index - 1
 * @param v1 Value at index (the peak)
 * @param v2 Value at index + 1
 * @return Subpixel offset from index (in range [-0.5, 0.5])
 */
inline double RefineSubpixelParabolic(double v0, double v1, double v2) {
    // Parabola: y = a*x^2 + b*x + c
    // At x=-1: v0 = a - b + c
    // At x=0:  v1 = c
    // At x=1:  v2 = a + b + c
    // Maximum at x = -b/(2a)
    // a = (v0 + v2)/2 - v1
    // b = (v2 - v0)/2
    double denom = 2.0 * (v0 - 2.0 * v1 + v2);
    if (std::abs(denom) < 1e-10) return 0.0;
    double offset = (v0 - v2) / denom;
    // Clamp to prevent runaway extrapolation
    return std::max(-0.5, std::min(0.5, offset));
}

/**
 * @brief Compute interpolated peak value at subpixel position
 *
 * @param v0 Value at index - 1
 * @param v1 Value at index
 * @param v2 Value at index + 1
 * @param offset Subpixel offset from RefineSubpixelParabolic
 * @return Interpolated value at peak
 */
inline double InterpolatedPeakValue(double v0, double v1, double v2, double offset) {
    // Using the parabola coefficients
    double a = (v0 + v2) * 0.5 - v1;
    double b = (v2 - v0) * 0.5;
    double c = v1;
    return a * offset * offset + b * offset + c;
}

// ============================================================================
// 2D Non-Maximum Suppression (Gradient-based for Canny)
// ============================================================================

/**
 * @brief Apply gradient-direction NMS for Canny edge detection
 *
 * For each pixel, checks if the magnitude is greater than the
 * interpolated neighbors along the gradient direction.
 *
 * @param magnitude Gradient magnitude image
 * @param direction Gradient direction image (radians)
 * @param output Output edge image (same size as input)
 * @param width Image width
 * @param height Image height
 * @param lowThreshold Low threshold for hysteresis (optional here)
 */
void NMS2DGradient(const float* magnitude, const float* direction,
                   float* output,
                   int32_t width, int32_t height,
                   float lowThreshold = 0.0f);

/**
 * @brief Apply gradient-direction NMS with quantized directions
 *
 * Faster version that quantizes gradient direction to 4 directions
 * (0°, 45°, 90°, 135°) and uses direct neighbor lookup.
 *
 * @param magnitude Gradient magnitude image
 * @param direction Gradient direction image (radians)
 * @param output Output edge image
 * @param width Image width
 * @param height Image height
 */
void NMS2DGradientQuantized(const float* magnitude, const float* direction,
                            float* output,
                            int32_t width, int32_t height);

/**
 * @brief Quantize gradient direction to 4 main directions
 *
 * @param angle Angle in radians [-PI, PI]
 * @return Quantized direction: 0=horizontal, 1=45°, 2=vertical, 3=135°
 */
inline int32_t QuantizeDirection(float angle) {
    // Normalize to [0, PI]
    if (angle < 0) angle += static_cast<float>(M_PI);

    // Divide into 4 sectors
    // Sector 0: [-22.5°, 22.5°) and [157.5°, 180°) -> horizontal
    // Sector 1: [22.5°, 67.5°) -> 45°
    // Sector 2: [67.5°, 112.5°) -> vertical
    // Sector 3: [112.5°, 157.5°) -> 135°

    constexpr float sector = static_cast<float>(M_PI / 4.0);  // 45°
    constexpr float halfSector = static_cast<float>(M_PI / 8.0);  // 22.5°

    if (angle < halfSector || angle >= static_cast<float>(M_PI) - halfSector) {
        return 0;  // Horizontal
    } else if (angle < halfSector + sector) {
        return 1;  // 45°
    } else if (angle < halfSector + 2 * sector) {
        return 2;  // Vertical
    } else {
        return 3;  // 135°
    }
}

// ============================================================================
// 2D Non-Maximum Suppression (Feature Point Detection)
// ============================================================================

/**
 * @brief Find local maxima in 2D response image
 *
 * A pixel is a local maximum if it's greater than all neighbors
 * within the specified radius.
 *
 * @param response Response/score image
 * @param width Image width
 * @param height Image height
 * @param radius Neighborhood radius (3x3 for radius=1, 5x5 for radius=2, etc.)
 * @param minValue Minimum response threshold
 * @return Vector of 2D peaks
 */
std::vector<Peak2D> FindLocalMaxima2D(const float* response,
                                       int32_t width, int32_t height,
                                       int32_t radius = 1,
                                       double minValue = 0.0);

/**
 * @brief Find local maxima with subpixel refinement
 *
 * Uses 2D parabolic fitting for subpixel accuracy.
 *
 * @param response Response image
 * @param width Image width
 * @param height Image height
 * @param radius Neighborhood radius
 * @param minValue Minimum threshold
 * @param refineSubpixel Whether to compute subpixel positions
 * @return Vector of peaks
 */
std::vector<Peak2D> FindPeaks2D(const float* response,
                                 int32_t width, int32_t height,
                                 int32_t radius = 1,
                                 double minValue = 0.0,
                                 bool refineSubpixel = true);

/**
 * @brief Apply NMS to keep only the strongest 2D peaks
 *
 * @param peaks Input peaks
 * @param maxCount Maximum number of peaks to keep
 * @param minDistance Minimum Euclidean distance between peaks
 * @return Filtered peaks
 */
std::vector<Peak2D> SuppressPeaks2D(std::vector<Peak2D> peaks,
                                     int32_t maxCount,
                                     double minDistance = 0.0);

/**
 * @brief Apply grid-based NMS for uniform peak distribution
 *
 * Divides image into cells and keeps only the best peak per cell.
 *
 * @param peaks Input peaks
 * @param width Image width
 * @param height Image height
 * @param cellSize Grid cell size
 * @return Filtered peaks (one per cell)
 */
std::vector<Peak2D> SuppressPeaks2DGrid(const std::vector<Peak2D>& peaks,
                                         int32_t width, int32_t height,
                                         int32_t cellSize);

// ============================================================================
// 2D Subpixel Refinement
// ============================================================================

/**
 * @brief Refine 2D peak position using parabolic interpolation
 *
 * Fits a 2D paraboloid to the 3x3 neighborhood and finds the maximum.
 *
 * @param response Response image
 * @param width Image width
 * @param height Image height
 * @param x Peak x coordinate
 * @param y Peak y coordinate
 * @param[out] subX Refined x coordinate
 * @param[out] subY Refined y coordinate
 * @return Interpolated peak value
 */
double RefineSubpixel2D(const float* response,
                        int32_t width, int32_t height,
                        int32_t x, int32_t y,
                        double& subX, double& subY);

/**
 * @brief Refine 2D peak using Taylor expansion (more robust)
 *
 * Uses first and second derivatives for refinement.
 *
 * @param response Response image
 * @param width Image width
 * @param height Image height
 * @param x Peak x coordinate
 * @param y Peak y coordinate
 * @param[out] subX Refined x coordinate
 * @param[out] subY Refined y coordinate
 * @return Interpolated peak value
 */
double RefineSubpixel2DTaylor(const float* response,
                              int32_t width, int32_t height,
                              int32_t x, int32_t y,
                              double& subX, double& subY);

// ============================================================================
// Box NMS (for object detection bounding boxes)
// ============================================================================

/**
 * @brief Compute Intersection over Union (IoU) between two boxes
 */
inline double ComputeIoU(const BoundingBox& a, const BoundingBox& b) {
    double interX1 = std::max(a.x1, b.x1);
    double interY1 = std::max(a.y1, b.y1);
    double interX2 = std::min(a.x2, b.x2);
    double interY2 = std::min(a.y2, b.y2);

    double interArea = std::max(0.0, interX2 - interX1) *
                       std::max(0.0, interY2 - interY1);

    double unionArea = a.Area() + b.Area() - interArea;
    if (unionArea <= 0.0) return 0.0;

    return interArea / unionArea;
}

/**
 * @brief Apply standard NMS to bounding boxes
 *
 * @param boxes Input boxes (sorted by score descending)
 * @param iouThreshold IoU threshold for suppression (typically 0.5)
 * @return Indices of kept boxes
 */
std::vector<int32_t> NMSBoxes(const std::vector<BoundingBox>& boxes,
                               double iouThreshold = 0.5);

/**
 * @brief Apply NMS with class awareness
 *
 * Only suppresses boxes of the same class.
 *
 * @param boxes Input boxes
 * @param iouThreshold IoU threshold
 * @return Indices of kept boxes
 */
std::vector<int32_t> NMSBoxesMultiClass(const std::vector<BoundingBox>& boxes,
                                         double iouThreshold = 0.5);

/**
 * @brief Apply Soft-NMS (decay score instead of removing)
 *
 * Uses Gaussian decay: score *= exp(-IoU^2 / sigma)
 *
 * @param boxes Input boxes (will be modified in place)
 * @param sigma Decay parameter
 * @param scoreThreshold Minimum score to keep
 * @return Indices of kept boxes (score >= threshold)
 */
std::vector<int32_t> SoftNMSBoxes(std::vector<BoundingBox>& boxes,
                                   double sigma = 0.5,
                                   double scoreThreshold = 0.001);

// ============================================================================
// Hysteresis Thresholding (commonly used with NMS)
// ============================================================================

/**
 * @brief Apply hysteresis thresholding to edge image
 *
 * Strong edges (>= highThreshold) are kept.
 * Weak edges (>= lowThreshold) are kept if connected to strong edges.
 *
 * @param edges Input edge magnitude image
 * @param output Output binary edge image (same size)
 * @param width Image width
 * @param height Image height
 * @param lowThreshold Low threshold
 * @param highThreshold High threshold
 */
void HysteresisThreshold(const float* edges, uint8_t* output,
                         int32_t width, int32_t height,
                         float lowThreshold, float highThreshold);

/**
 * @brief Apply hysteresis thresholding in-place
 *
 * @param edges Edge image (modified in place: edges become 0 or 255)
 * @param width Image width
 * @param height Image height
 * @param lowThreshold Low threshold
 * @param highThreshold High threshold
 */
void HysteresisThresholdInPlace(float* edges,
                                 int32_t width, int32_t height,
                                 float lowThreshold, float highThreshold);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Sort peaks by value (descending)
 */
template<typename PeakType>
void SortPeaksByValue(std::vector<PeakType>& peaks) {
    std::sort(peaks.begin(), peaks.end());
}

/**
 * @brief Sort peaks by position (for consistent output)
 */
inline void SortPeaks1DByPosition(std::vector<Peak1D>& peaks) {
    std::sort(peaks.begin(), peaks.end(),
              [](const Peak1D& a, const Peak1D& b) {
                  return a.index < b.index;
              });
}

/**
 * @brief Sort 2D peaks by position (row-major order)
 */
inline void SortPeaks2DByPosition(std::vector<Peak2D>& peaks) {
    std::sort(peaks.begin(), peaks.end(),
              [](const Peak2D& a, const Peak2D& b) {
                  if (a.y != b.y) return a.y < b.y;
                  return a.x < b.x;
              });
}

/**
 * @brief Compute Euclidean distance between two 2D peaks
 */
inline double PeakDistance(const Peak2D& a, const Peak2D& b) {
    double dx = a.subpixelX - b.subpixelX;
    double dy = a.subpixelY - b.subpixelY;
    return std::sqrt(dx * dx + dy * dy);
}

} // namespace Qi::Vision::Internal
