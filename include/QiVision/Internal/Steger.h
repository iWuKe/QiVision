#pragma once

/**
 * @file Steger.h
 * @brief Steger subpixel edge detection (ridge/valley detection)
 *
 * Implementation of Steger's algorithm for curvilinear structure detection.
 * Used extensively in industrial machine vision for high-precision edge detection.
 *
 * Algorithm flow:
 * 1. Gaussian smoothing
 * 2. Hessian matrix computation
 * 3. Eigenvalue decomposition
 * 4. Ridge/valley classification
 * 5. Subpixel position refinement
 * 6. Threshold filtering
 * 7. Edge point linking
 * 8. Contour output
 *
 * Reference:
 * - Steger, "An Unbiased Detector of Curvilinear Structures" (1998)
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QContour.h>
#include <QiVision/Internal/Hessian.h>

#include <cstdint>
#include <vector>

namespace Qi::Vision::Internal {

// ============================================================================
// Enumerations
// ============================================================================

/**
 * @brief Type of line to detect
 */
enum class LineType {
    Ridge,   ///< Ridge (bright line on dark background): λ1 < 0, |λ1| >> |λ2|
    Valley,  ///< Valley (dark line on bright background): λ1 > 0, |λ1| >> |λ2|
    Both     ///< Detect both ridge and valley lines
};

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief Parameters for Steger edge detection
 */
struct StegerParams {
    double sigma = 1.0;             ///< Gaussian sigma for smoothing (affects line width)
    double lowThreshold = 5.0;      ///< Low threshold for hysteresis (weak edges)
    double highThreshold = 15.0;    ///< High threshold for hysteresis (strong edges)
    LineType lineType = LineType::Both;  ///< Type of lines to detect
    double minLength = 5.0;         ///< Minimum contour length in pixels
    double maxGap = 2.0;            ///< Maximum gap for edge linking
    double maxAngleDiff = 0.5;      ///< Maximum angle difference for linking (radians)
    bool subPixelRefinement = true; ///< Enable subpixel position refinement
};

/**
 * @brief A detected Steger edge point with subpixel position
 */
struct StegerPoint {
    double x = 0.0;             ///< Subpixel x position
    double y = 0.0;             ///< Subpixel y position
    double nx = 0.0;            ///< Normal vector x-component
    double ny = 0.0;            ///< Normal vector y-component
    double tx = 0.0;            ///< Tangent vector x-component
    double ty = 0.0;            ///< Tangent vector y-component
    double response = 0.0;      ///< Response strength (|λ1|)
    double amplitude = 0.0;     ///< Amplitude at the edge point
    bool isRidge = true;        ///< true = ridge (bright line), false = valley (dark line)
    int32_t pixelX = 0;         ///< Original pixel x coordinate
    int32_t pixelY = 0;         ///< Original pixel y coordinate

    /**
     * @brief Get the subpixel offset from the pixel center
     */
    Point2d SubpixelOffset() const {
        return Point2d(x - pixelX, y - pixelY);
    }
};

/**
 * @brief Result of Steger detection
 */
struct StegerResult {
    std::vector<StegerPoint> points;     ///< All detected edge points
    std::vector<QContour> contours;      ///< Linked contours
    int32_t numRidgePoints = 0;          ///< Number of ridge points
    int32_t numValleyPoints = 0;         ///< Number of valley points
};

// ============================================================================
// Main Detection Functions
// ============================================================================

/**
 * @brief Detect Steger edges in an image
 *
 * Performs the complete Steger edge detection algorithm:
 * 1. Compute Hessian matrix at each pixel
 * 2. Compute eigenvalues/eigenvectors
 * 3. Find ridge/valley points based on threshold
 * 4. Refine to subpixel positions
 * 5. Link edge points into contours
 *
 * @param image Input image (grayscale)
 * @param params Detection parameters
 * @return Vector of detected contours
 */
std::vector<QContour> DetectStegerEdges(const QImage& image,
                                         const StegerParams& params);

/**
 * @brief Detect Steger edges with full result
 *
 * @param image Input image (grayscale)
 * @param params Detection parameters
 * @return StegerResult containing both points and contours
 */
StegerResult DetectStegerEdgesFull(const QImage& image,
                                    const StegerParams& params);

// ============================================================================
// Detection Steps (for custom pipelines)
// ============================================================================

/**
 * @brief Detect edge points at pixel level (before subpixel refinement)
 *
 * Uses Hessian eigenvalue analysis to find candidate edge points.
 *
 * @param lambda1 Principal eigenvalue image
 * @param lambda2 Secondary eigenvalue image
 * @param nx Principal direction x-component image
 * @param ny Principal direction y-component image
 * @param width Image width
 * @param height Image height
 * @param params Detection parameters
 * @return Vector of candidate edge points
 */
std::vector<StegerPoint> DetectCandidatePoints(
    const float* lambda1, const float* lambda2,
    const float* nx, const float* ny,
    int32_t width, int32_t height,
    const StegerParams& params);

/**
 * @brief Refine edge point position to subpixel accuracy
 *
 * Uses Taylor expansion along the principal direction to find the exact
 * zero-crossing of the gradient (edge position).
 *
 * @param dxx Second derivative xx image
 * @param dxy Second derivative xy image
 * @param dyy Second derivative yy image
 * @param width Image width
 * @param height Image height
 * @param x Pixel x coordinate
 * @param y Pixel y coordinate
 * @param nx Principal direction x-component
 * @param ny Principal direction y-component
 * @return Subpixel offset from (x, y)
 */
Point2d RefineSubpixelSteger(const float* dxx, const float* dxy, const float* dyy,
                              int32_t width, int32_t height,
                              int32_t x, int32_t y,
                              double nx, double ny);

/**
 * @brief Refine all edge points to subpixel positions
 *
 * @param points Edge points to refine (modified in place)
 * @param dxx Second derivative xx image
 * @param dxy Second derivative xy image
 * @param dyy Second derivative yy image
 * @param width Image width
 * @param height Image height
 */
void RefineAllSubpixel(std::vector<StegerPoint>& points,
                       const float* dxx, const float* dxy, const float* dyy,
                       int32_t width, int32_t height);

// ============================================================================
// Edge Linking
// ============================================================================

/**
 * @brief Link edge points into contours
 *
 * Uses direction and proximity to connect edge points.
 *
 * @param points Edge points to link
 * @param maxGap Maximum distance between points to link
 * @param maxAngleDiff Maximum angle difference between tangent vectors (radians)
 * @return Vector of linked contours
 */
std::vector<QContour> LinkEdgePoints(const std::vector<StegerPoint>& points,
                                      double maxGap,
                                      double maxAngleDiff);

/**
 * @brief Build spatial index for efficient point lookup
 *
 * Creates a grid-based spatial index for fast neighbor queries.
 *
 * @param points Edge points
 * @param width Image width
 * @param height Image height
 * @param cellSize Size of grid cells
 * @return Grid of point indices
 */
std::vector<std::vector<int32_t>> BuildSpatialIndex(
    const std::vector<StegerPoint>& points,
    int32_t width, int32_t height,
    int32_t cellSize = 4);

// ============================================================================
// Filtering Functions
// ============================================================================

/**
 * @brief Filter points by response threshold using hysteresis
 *
 * Points with response >= highThreshold are kept.
 * Points with lowThreshold <= response < highThreshold are kept
 * only if connected to a strong point.
 *
 * @param points Edge points
 * @param lowThreshold Low threshold
 * @param highThreshold High threshold
 * @return Filtered points
 */
std::vector<StegerPoint> FilterByHysteresis(
    const std::vector<StegerPoint>& points,
    double lowThreshold,
    double highThreshold);

/**
 * @brief Filter contours by minimum length
 *
 * @param contours Input contours
 * @param minLength Minimum length in pixels
 * @return Filtered contours
 */
std::vector<QContour> FilterByLength(const std::vector<QContour>& contours,
                                      double minLength);

// ============================================================================
// Non-Maximum Suppression
// ============================================================================

/**
 * @brief Perform non-maximum suppression along principal direction
 *
 * Keeps only points that are local maxima of response along the
 * principal eigenvector direction.
 *
 * @param points Edge points
 * @param lambda1 Principal eigenvalue image
 * @param width Image width
 * @param height Image height
 * @return Points after NMS
 */
std::vector<StegerPoint> NonMaxSuppressionSteger(
    const std::vector<StegerPoint>& points,
    const float* lambda1,
    int32_t width, int32_t height);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Check if a point is a valid edge candidate
 *
 * @param lambda1 Principal eigenvalue
 * @param lambda2 Secondary eigenvalue
 * @param threshold Response threshold
 * @param lineType Type of lines to detect
 * @return true if valid candidate
 */
inline bool IsEdgeCandidate(double lambda1, double lambda2,
                            double threshold, LineType lineType) {
    double response = std::abs(lambda1);
    if (response < threshold) return false;

    // Check anisotropy (line-like structure)
    double anisotropy = (std::abs(lambda2) > 1e-10)
                        ? response / std::abs(lambda2)
                        : 1e10;
    if (anisotropy < 2.0) return false;  // Too blob-like

    switch (lineType) {
        case LineType::Ridge:
            return lambda1 < 0;
        case LineType::Valley:
            return lambda1 > 0;
        case LineType::Both:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Compute the angle between two tangent vectors
 *
 * @param tx1 First tangent x
 * @param ty1 First tangent y
 * @param tx2 Second tangent x
 * @param ty2 Second tangent y
 * @return Angle in radians [0, π]
 */
inline double TangentAngleDiff(double tx1, double ty1, double tx2, double ty2) {
    // Tangent vectors can point in either direction, so we compare
    // the absolute dot product
    double dot = tx1 * tx2 + ty1 * ty2;
    double absDot = std::abs(dot);
    if (absDot > 1.0) absDot = 1.0;
    return std::acos(absDot);
}

/**
 * @brief Compute distance between two edge points
 */
inline double PointDistance(const StegerPoint& p1, const StegerPoint& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

/**
 * @brief Convert StegerPoints to ContourPoints for QContour
 */
std::vector<ContourPoint> ToContourPoints(const std::vector<StegerPoint>& points);

/**
 * @brief Create QContour from StegerPoints
 */
QContour CreateContour(const std::vector<StegerPoint>& points);

} // namespace Qi::Vision::Internal
