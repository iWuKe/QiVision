#pragma once

/**
 * @file Canny.h
 * @brief Canny edge detection algorithm
 *
 * Implements the classic Canny edge detector with optional subpixel refinement:
 * 1. Gaussian smoothing (noise reduction)
 * 2. Gradient computation (Sobel/Scharr)
 * 3. Non-maximum suppression (edge thinning)
 * 4. Hysteresis thresholding (edge selection)
 * 5. Edge linking (contour extraction)
 *
 * Corresponds to Halcon's `edges_sub_pix` operator.
 *
 * Reference:
 * - Canny, "A Computational Approach to Edge Detection" (1986)
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QContour.h>
#include <QiVision/Internal/Gradient.h>

#include <cstdint>
#include <vector>

namespace Qi::Vision::Internal {

// ============================================================================
// Enumerations
// ============================================================================

/**
 * @brief Gradient operator for Canny
 */
enum class CannyGradientOp {
    Sobel,      ///< Sobel 3x3 (default, good balance)
    Scharr,     ///< Scharr 3x3 (more accurate, slightly slower)
    Sobel5x5    ///< Sobel 5x5 (smoother, better for noisy images)
};

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief Parameters for Canny edge detection
 */
struct CannyParams {
    // Smoothing
    double sigma = 1.0;             ///< Gaussian sigma for smoothing (0 = no smoothing)

    // Thresholds
    double lowThreshold = 20.0;     ///< Low threshold for hysteresis
    double highThreshold = 60.0;    ///< High threshold for hysteresis
    bool autoThreshold = false;     ///< Automatically compute thresholds from image

    // Gradient
    CannyGradientOp gradientOp = CannyGradientOp::Sobel;

    // Output options
    bool subpixelRefinement = true; ///< Refine edge positions to subpixel accuracy
    bool linkEdges = true;          ///< Link edge pixels into contours
    double minContourLength = 5.0;  ///< Minimum contour length (pixels)
    int32_t minContourPoints = 3;   ///< Minimum points in a contour

    /**
     * @brief Create params with automatic thresholding
     */
    static CannyParams Auto(double sigma = 1.0) {
        CannyParams p;
        p.sigma = sigma;
        p.autoThreshold = true;
        return p;
    }

    /**
     * @brief Create params with specified thresholds
     */
    static CannyParams WithThresholds(double low, double high, double sigma = 1.0) {
        CannyParams p;
        p.sigma = sigma;
        p.lowThreshold = low;
        p.highThreshold = high;
        return p;
    }
};

/**
 * @brief A Canny edge point with subpixel position
 */
struct CannyEdgePoint {
    double x = 0.0;             ///< Subpixel x position
    double y = 0.0;             ///< Subpixel y position
    double magnitude = 0.0;     ///< Edge magnitude (gradient strength)
    double direction = 0.0;     ///< Edge direction in radians

    CannyEdgePoint() = default;
    CannyEdgePoint(double px, double py, double mag = 0.0, double dir = 0.0)
        : x(px), y(py), magnitude(mag), direction(dir) {}
};

/**
 * @brief Result of Canny edge detection
 */
struct CannyResult {
    std::vector<CannyEdgePoint> edgePoints;  ///< All detected edge points
    std::vector<QContour> contours;          ///< Linked edge contours
    QImage edgeImage;                        ///< Binary edge image (optional)

    // Statistics
    int32_t numEdgePixels = 0;               ///< Number of edge pixels
    double avgMagnitude = 0.0;               ///< Average edge magnitude
};

// ============================================================================
// Main Detection Functions
// ============================================================================

/**
 * @brief Detect edges using Canny algorithm
 *
 * @param image Input grayscale image
 * @param params Detection parameters
 * @return Vector of edge contours
 */
std::vector<QContour> DetectEdgesCanny(const QImage& image,
                                        const CannyParams& params = CannyParams());

/**
 * @brief Detect edges with full result
 *
 * @param image Input grayscale image
 * @param params Detection parameters
 * @return Full result including edge points and statistics
 */
CannyResult DetectEdgesCannyFull(const QImage& image,
                                  const CannyParams& params = CannyParams());

/**
 * @brief Detect edges and return binary edge image
 *
 * @param image Input grayscale image
 * @param params Detection parameters
 * @return Binary edge image (255 = edge, 0 = non-edge)
 */
QImage DetectEdgesCannyImage(const QImage& image,
                              const CannyParams& params = CannyParams());

// ============================================================================
// Pipeline Steps (for custom workflows)
// ============================================================================

/**
 * @brief Apply Gaussian smoothing
 *
 * @param src Source image data
 * @param dst Destination (must be pre-allocated, same size as src)
 * @param width Image width
 * @param height Image height
 * @param sigma Gaussian sigma
 */
void CannySmooth(const uint8_t* src, float* dst,
                 int32_t width, int32_t height,
                 double sigma);

/**
 * @brief Compute gradient magnitude and direction
 *
 * @param src Smoothed image (float)
 * @param magnitude Output magnitude image
 * @param direction Output direction image (radians)
 * @param width Image width
 * @param height Image height
 * @param op Gradient operator
 */
void CannyGradient(const float* src, float* magnitude, float* direction,
                   int32_t width, int32_t height,
                   CannyGradientOp op = CannyGradientOp::Sobel);

/**
 * @brief Apply non-maximum suppression
 *
 * Thins edges to 1-pixel width by suppressing non-maximum pixels
 * along the gradient direction.
 *
 * @param magnitude Gradient magnitude
 * @param direction Gradient direction
 * @param output Output NMS result
 * @param width Image width
 * @param height Image height
 */
void CannyNMS(const float* magnitude, const float* direction,
              float* output,
              int32_t width, int32_t height);

/**
 * @brief Apply hysteresis thresholding
 *
 * Strong edges (>= high) are kept. Weak edges (>= low) are kept
 * only if connected to strong edges.
 *
 * @param nmsOutput NMS output image
 * @param output Binary output (255 or 0)
 * @param width Image width
 * @param height Image height
 * @param lowThreshold Low threshold
 * @param highThreshold High threshold
 */
void CannyHysteresis(const float* nmsOutput, uint8_t* output,
                     int32_t width, int32_t height,
                     double lowThreshold, double highThreshold);

/**
 * @brief Compute automatic thresholds using Otsu or median-based method
 *
 * @param magnitude Gradient magnitude image
 * @param width Image width
 * @param height Image height
 * @param[out] lowThreshold Computed low threshold
 * @param[out] highThreshold Computed high threshold
 */
void ComputeAutoThresholds(const float* magnitude,
                           int32_t width, int32_t height,
                           double& lowThreshold, double& highThreshold);

// ============================================================================
// Subpixel Refinement
// ============================================================================

/**
 * @brief Refine edge position to subpixel accuracy
 *
 * Uses parabolic interpolation along the gradient direction.
 *
 * @param magnitude Gradient magnitude
 * @param direction Gradient direction
 * @param width Image width
 * @param height Image height
 * @param x Pixel x coordinate
 * @param y Pixel y coordinate
 * @param[out] subX Refined x coordinate
 * @param[out] subY Refined y coordinate
 * @return Interpolated magnitude at subpixel position
 */
double RefineEdgeSubpixel(const float* magnitude, const float* direction,
                          int32_t width, int32_t height,
                          int32_t x, int32_t y,
                          double& subX, double& subY);

/**
 * @brief Extract edge points from binary edge image with subpixel refinement
 *
 * @param edgeImage Binary edge image
 * @param magnitude Gradient magnitude
 * @param direction Gradient direction
 * @param width Image width
 * @param height Image height
 * @param refineSubpixel Whether to refine to subpixel positions
 * @return Vector of edge points
 */
std::vector<CannyEdgePoint> ExtractEdgePoints(const uint8_t* edgeImage,
                                               const float* magnitude,
                                               const float* direction,
                                               int32_t width, int32_t height,
                                               bool refineSubpixel = true);

// ============================================================================
// Edge Linking
// ============================================================================

/**
 * @brief Link edge points into contours
 *
 * @param edgePoints Edge points to link
 * @param width Image width (for spatial indexing)
 * @param height Image height
 * @param minLength Minimum contour length
 * @param minPoints Minimum number of points
 * @return Vector of contours
 */
std::vector<QContour> LinkCannyEdges(const std::vector<CannyEdgePoint>& edgePoints,
                                      int32_t width, int32_t height,
                                      double minLength = 5.0,
                                      int32_t minPoints = 3);

/**
 * @brief Link edge pixels directly from binary edge image
 *
 * Uses 8-connectivity chain following.
 *
 * @param edgeImage Binary edge image
 * @param magnitude Gradient magnitude (for edge attributes)
 * @param direction Gradient direction
 * @param width Image width
 * @param height Image height
 * @param minLength Minimum contour length
 * @return Vector of contours
 */
std::vector<QContour> LinkEdgePixels(const uint8_t* edgeImage,
                                      const float* magnitude,
                                      const float* direction,
                                      int32_t width, int32_t height,
                                      double minLength = 5.0);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Convert gradient operator enum to Gradient.h enum
 */
inline GradientOperator ToGradientOperator(CannyGradientOp op) {
    switch (op) {
        case CannyGradientOp::Scharr: return GradientOperator::Scharr;
        case CannyGradientOp::Sobel5x5: return GradientOperator::Sobel5x5;
        default: return GradientOperator::Sobel3x3;
    }
}

/**
 * @brief Get edge normal direction (perpendicular to edge)
 *
 * The gradient direction points from dark to bright.
 * The edge direction is perpendicular to the gradient.
 */
inline double EdgeDirection(double gradientDirection) {
    return gradientDirection + M_PI / 2.0;
}

} // namespace Qi::Vision::Internal
