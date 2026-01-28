#pragma once

/**
 * @file CornerRefine.h
 * @brief Subpixel corner refinement algorithms
 *
 * Internal module for corner detection and refinement.
 *
 * Algorithms:
 * - Gradient-based corner refinement (Forstner-style)
 * - Harris corner detection
 * - Shi-Tomasi corner detection (Good Features to Track)
 *
 * Reference:
 * - Forstner W. "A Feature Based Correspondence Algorithm for Image Matching" 1986
 * - Harris C. & Stephens M. "A Combined Corner and Edge Detector" 1988
 * - Shi J. & Tomasi C. "Good Features to Track" 1994
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QImage.h>

#include <vector>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Default Harris corner detector parameter
constexpr double HARRIS_K = 0.04;

/// Default minimum distance between corners
constexpr double CORNER_MIN_DISTANCE = 10.0;

/// Default quality level for corner detection
constexpr double CORNER_QUALITY_LEVEL = 0.01;

/// Default block size for corner detection
constexpr int32_t CORNER_BLOCK_SIZE = 3;

/// Default refinement window half-size
constexpr int32_t CORNER_REFINE_WIN_SIZE = 5;

/// Default refinement max iterations
constexpr int32_t CORNER_REFINE_MAX_ITER = 30;

/// Default refinement convergence threshold
constexpr double CORNER_REFINE_EPSILON = 0.001;

// =============================================================================
// Corner Refinement
// =============================================================================

/**
 * @brief Refine corner to subpixel accuracy using gradient method
 *
 * Uses the equation: sum(gradient * gradient^T) * delta = sum(gradient * (q - p))
 * where q is the current estimate and p is the corner location.
 *
 * The algorithm iteratively refines the corner position by solving a 2x2 linear
 * system that minimizes the sum of squared dot products between the gradient
 * vectors and the vectors from each pixel to the corner.
 *
 * @param image Grayscale image
 * @param corner Initial corner estimate (modified in place)
 * @param winSize Half window size
 * @param maxIterations Maximum iterations
 * @param epsilon Convergence threshold (pixel movement)
 * @return true if converged within epsilon
 */
bool RefineCornerGradient(
    const QImage& image,
    Point2d& corner,
    int32_t winSize = CORNER_REFINE_WIN_SIZE,
    int32_t maxIterations = CORNER_REFINE_MAX_ITER,
    double epsilon = CORNER_REFINE_EPSILON
);

/**
 * @brief Refine multiple corners to subpixel accuracy
 *
 * Batch version of RefineCornerGradient.
 *
 * @param image Grayscale image
 * @param corners Input/Output corner positions
 * @param winSize Half window size
 * @param maxIterations Maximum iterations
 * @param epsilon Convergence threshold
 */
void RefineCorners(
    const QImage& image,
    std::vector<Point2d>& corners,
    int32_t winSize = CORNER_REFINE_WIN_SIZE,
    int32_t maxIterations = CORNER_REFINE_MAX_ITER,
    double epsilon = CORNER_REFINE_EPSILON
);

// =============================================================================
// Harris Corner Detection
// =============================================================================

/**
 * @brief Compute Harris corner response at a single point
 *
 * Harris response: R = det(M) - k * trace(M)^2
 * where M is the structure tensor (sum of outer products of gradients)
 *
 * @param image Grayscale image
 * @param x Point x coordinate
 * @param y Point y coordinate
 * @param blockSize Neighborhood size for gradient averaging
 * @param k Harris detector free parameter (typically 0.04-0.06)
 * @return Corner response value (positive for corners)
 */
double HarrisResponse(
    const QImage& image,
    int32_t x,
    int32_t y,
    int32_t blockSize = CORNER_BLOCK_SIZE,
    double k = HARRIS_K
);

/**
 * @brief Compute Harris corner response image
 *
 * @param image Input grayscale image
 * @param response Output response image (Float32)
 * @param blockSize Neighborhood size
 * @param k Harris parameter
 */
void HarrisResponseImage(
    const QImage& image,
    QImage& response,
    int32_t blockSize = CORNER_BLOCK_SIZE,
    double k = HARRIS_K
);

/**
 * @brief Detect corners using Harris corner response
 *
 * Algorithm:
 * 1. Compute Harris response image
 * 2. Apply non-maximum suppression
 * 3. Filter by quality level (relative to max response)
 * 4. Filter by minimum distance
 * 5. Sort by response (strongest first)
 *
 * @param image Grayscale image
 * @param maxCorners Maximum number of corners to return (0 = unlimited)
 * @param qualityLevel Minimum quality factor relative to best corner (0-1)
 * @param minDistance Minimum Euclidean distance between corners
 * @param blockSize Neighborhood size for corner detection
 * @param k Harris detector free parameter
 * @return Detected corner positions sorted by response (strongest first)
 */
std::vector<Point2d> DetectHarrisCorners(
    const QImage& image,
    int32_t maxCorners = 1000,
    double qualityLevel = CORNER_QUALITY_LEVEL,
    double minDistance = CORNER_MIN_DISTANCE,
    int32_t blockSize = CORNER_BLOCK_SIZE,
    double k = HARRIS_K
);

// =============================================================================
// Shi-Tomasi Corner Detection (Good Features to Track)
// =============================================================================

/**
 * @brief Compute Shi-Tomasi corner response at a single point
 *
 * Shi-Tomasi response: min(lambda1, lambda2)
 * where lambda1, lambda2 are eigenvalues of the structure tensor
 *
 * This is more stable than Harris for corner tracking.
 *
 * @param image Grayscale image
 * @param x Point x coordinate
 * @param y Point y coordinate
 * @param blockSize Neighborhood size
 * @return Corner response value (minimum eigenvalue)
 */
double ShiTomasiResponse(
    const QImage& image,
    int32_t x,
    int32_t y,
    int32_t blockSize = CORNER_BLOCK_SIZE
);

/**
 * @brief Detect corners using Shi-Tomasi (Good Features to Track)
 *
 * @param image Grayscale image
 * @param maxCorners Maximum number of corners to return
 * @param qualityLevel Minimum quality factor (0-1)
 * @param minDistance Minimum distance between corners
 * @param blockSize Neighborhood size
 * @return Detected corner positions
 */
std::vector<Point2d> DetectShiTomasiCorners(
    const QImage& image,
    int32_t maxCorners = 1000,
    double qualityLevel = CORNER_QUALITY_LEVEL,
    double minDistance = CORNER_MIN_DISTANCE,
    int32_t blockSize = CORNER_BLOCK_SIZE
);

// =============================================================================
// Structure Tensor
// =============================================================================

/**
 * @brief Compute structure tensor at a point
 *
 * Structure tensor M = [ sum(Ix*Ix)  sum(Ix*Iy) ]
 *                      [ sum(Ix*Iy)  sum(Iy*Iy) ]
 *
 * @param image Grayscale image
 * @param x Point x coordinate
 * @param y Point y coordinate
 * @param blockSize Neighborhood size
 * @param[out] Ixx Sum of Ix*Ix
 * @param[out] Ixy Sum of Ix*Iy
 * @param[out] Iyy Sum of Iy*Iy
 */
void ComputeStructureTensor(
    const QImage& image,
    int32_t x,
    int32_t y,
    int32_t blockSize,
    double& Ixx,
    double& Ixy,
    double& Iyy
);

/**
 * @brief Compute eigenvalues of 2x2 symmetric matrix
 *
 * For matrix [ a  b ]
 *            [ b  c ]
 *
 * @param a Matrix element (0,0)
 * @param b Matrix element (0,1) and (1,0)
 * @param c Matrix element (1,1)
 * @param[out] lambda1 Larger eigenvalue
 * @param[out] lambda2 Smaller eigenvalue
 */
void Eigenvalues2x2(double a, double b, double c, double& lambda1, double& lambda2);

// =============================================================================
// Non-Maximum Suppression for Corners
// =============================================================================

/**
 * @brief Apply non-maximum suppression to corner response image
 *
 * @param response Corner response image
 * @param winSize Window size for local maximum check
 * @param threshold Minimum response threshold
 * @return Positions of local maxima
 */
std::vector<Point2d> NonMaximumSuppressionCorners(
    const QImage& response,
    int32_t winSize = 3,
    double threshold = 0.0
);

/**
 * @brief Filter corners by minimum distance
 *
 * Keeps corners sorted by strength, removing those too close to stronger ones.
 *
 * @param corners Input corners (assumed sorted by strength, strongest first)
 * @param strengths Corner strengths (parallel to corners)
 * @param minDistance Minimum distance between kept corners
 * @return Filtered corners
 */
std::vector<Point2d> FilterByDistance(
    const std::vector<Point2d>& corners,
    const std::vector<double>& strengths,
    double minDistance
);

} // namespace Qi::Vision::Internal
