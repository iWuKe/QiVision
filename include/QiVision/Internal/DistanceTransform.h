#pragma once

/**
 * @file DistanceTransform.h
 * @brief Distance transform algorithms for binary images and regions
 *
 * This module provides distance transform algorithms that compute the
 * distance from each foreground pixel to the nearest background pixel.
 *
 * Supported distance metrics:
 * - L1 (Manhattan/City-block): |dx| + |dy|
 * - L2 (Euclidean): sqrt(dx^2 + dy^2)
 * - LInf (Chessboard): max(|dx|, |dy|)
 * - Chamfer 3-4: Approximation with weights 3 and 4
 * - Chamfer 5-7-11: Better approximation
 *
 * Algorithms:
 * - Two-pass scanning for L1/LInf
 * - Chamfer distance transform
 * - Exact Euclidean distance transform (Meijster et al.)
 *
 * Reference Halcon operators:
 * - distance_transform
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>
#include <QiVision/Core/Types.h>

#include <vector>

namespace Qi::Vision::Internal {

// =============================================================================
// Types
// =============================================================================

/**
 * @brief Distance metric types
 */
enum class DistanceType {
    L1,         ///< Manhattan distance: |dx| + |dy|
    L2,         ///< Euclidean distance: sqrt(dx^2 + dy^2)
    LInf,       ///< Chessboard distance: max(|dx|, |dy|)
    Chamfer3_4, ///< Chamfer approximation with 3-4 weights
    Chamfer5_7  ///< Chamfer approximation with 5-7-11 weights
};

/**
 * @brief Output type for distance transform
 */
enum class DistanceOutputType {
    Float32,    ///< 32-bit float (recommended for L2)
    UInt8,      ///< 8-bit unsigned (clamped to 255)
    UInt16,     ///< 16-bit unsigned (clamped to 65535)
    Int16       ///< 16-bit signed integer
};

// =============================================================================
// Distance Transform - Image Based
// =============================================================================

/**
 * @brief Compute distance transform of a binary image
 *
 * Computes the distance from each foreground pixel to the nearest
 * background pixel. Background pixels have distance 0.
 *
 * @param binary Input binary image (non-zero = foreground)
 * @param distType Distance metric to use
 * @param outputType Output pixel type
 * @return Distance transform image
 */
QImage DistanceTransform(const QImage& binary,
                         DistanceType distType = DistanceType::L2,
                         DistanceOutputType outputType = DistanceOutputType::Float32);

/**
 * @brief Compute distance transform with normalization
 *
 * @param binary Input binary image
 * @param distType Distance metric to use
 * @param normalize If true, normalize to [0, 1] range
 * @return Distance transform image (Float32)
 */
QImage DistanceTransformNormalized(const QImage& binary,
                                   DistanceType distType = DistanceType::L2);

/**
 * @brief Compute L1 (Manhattan) distance transform
 *
 * Fast implementation using two-pass scanning.
 *
 * @param binary Input binary image
 * @return Distance transform image (Float32)
 */
QImage DistanceTransformL1(const QImage& binary);

/**
 * @brief Compute L2 (Euclidean) distance transform
 *
 * Uses Meijster's algorithm for exact Euclidean distance.
 *
 * @param binary Input binary image
 * @return Distance transform image (Float32)
 */
QImage DistanceTransformL2(const QImage& binary);

/**
 * @brief Compute LInf (Chessboard) distance transform
 *
 * Fast implementation using two-pass scanning.
 *
 * @param binary Input binary image
 * @return Distance transform image (Float32)
 */
QImage DistanceTransformLInf(const QImage& binary);

/**
 * @brief Compute Chamfer distance transform
 *
 * Approximates Euclidean distance using integer weights.
 *
 * @param binary Input binary image
 * @param use5_7 If true, use 5-7-11 weights; otherwise use 3-4
 * @return Distance transform image (Float32, scaled to approximate L2)
 */
QImage DistanceTransformChamfer(const QImage& binary, bool use5_7 = true);

// =============================================================================
// Distance Transform - Region Based (RLE)
// =============================================================================

/**
 * @brief Compute distance transform for a region
 *
 * @param region Input region
 * @param bounds Bounding rectangle for output
 * @param distType Distance metric to use
 * @return Distance transform image within bounds
 */
QImage DistanceTransformRegion(const QRegion& region,
                               const Rect2i& bounds,
                               DistanceType distType = DistanceType::L2);

/**
 * @brief Compute distance to region boundary (signed distance)
 *
 * Positive values inside the region, negative outside.
 *
 * @param region Input region
 * @param bounds Bounding rectangle for output
 * @param distType Distance metric to use
 * @return Signed distance transform image
 */
QImage SignedDistanceTransform(const QRegion& region,
                               const Rect2i& bounds,
                               DistanceType distType = DistanceType::L2);

// =============================================================================
// Distance to Specific Points/Features
// =============================================================================

/**
 * @brief Compute distance from each pixel to nearest seed point
 *
 * @param width Image width
 * @param height Image height
 * @param seedPoints Vector of seed point coordinates
 * @param distType Distance metric to use
 * @return Distance image
 */
QImage DistanceToPoints(int32_t width, int32_t height,
                        const std::vector<Point2i>& seedPoints,
                        DistanceType distType = DistanceType::L2);

/**
 * @brief Compute distance from each pixel to nearest edge pixel
 *
 * @param edges Binary edge image (non-zero = edge)
 * @param distType Distance metric to use
 * @return Distance to nearest edge
 */
QImage DistanceToEdges(const QImage& edges,
                       DistanceType distType = DistanceType::L2);

// =============================================================================
// Voronoi Diagram from Distance Transform
// =============================================================================

/**
 * @brief Compute Voronoi diagram from seed points
 *
 * Each pixel is labeled with the index of the nearest seed point.
 *
 * @param width Image width
 * @param height Image height
 * @param seedPoints Vector of seed point coordinates
 * @param distType Distance metric to use
 * @return Label image (0 to numSeeds-1)
 */
QImage VoronoiDiagram(int32_t width, int32_t height,
                      const std::vector<Point2i>& seedPoints,
                      DistanceType distType = DistanceType::L2);

/**
 * @brief Compute Voronoi diagram from labeled regions
 *
 * Extends each labeled region to fill the entire image.
 *
 * @param labels Input label image
 * @param distType Distance metric to use
 * @return Extended label image
 */
QImage VoronoiFromLabels(const QImage& labels,
                         DistanceType distType = DistanceType::L2);

// =============================================================================
// Skeleton from Distance Transform
// =============================================================================

/**
 * @brief Extract skeleton using distance transform
 *
 * Finds ridge lines in the distance transform.
 *
 * @param binary Input binary image
 * @param distType Distance metric for transform
 * @return Binary skeleton image
 */
QImage SkeletonFromDistance(const QImage& binary,
                            DistanceType distType = DistanceType::L2);

/**
 * @brief Compute medial axis transform
 *
 * Returns the distance transform values only at skeleton pixels.
 *
 * @param binary Input binary image
 * @param[out] skeleton Output skeleton image
 * @return Distance values at skeleton pixels (Float32)
 */
QImage MedialAxisTransform(const QImage& binary, QImage& skeleton);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Get maximum distance value in distance transform
 *
 * @param distanceImage Distance transform image
 * @return Maximum distance value
 */
double GetMaxDistance(const QImage& distanceImage);

/**
 * @brief Threshold distance transform to create binary mask
 *
 * @param distanceImage Distance transform image
 * @param threshold Distance threshold
 * @param invert If true, select pixels below threshold
 * @return Binary mask
 */
QImage ThresholdDistance(const QImage& distanceImage,
                         double threshold,
                         bool invert = false);

/**
 * @brief Find pixels at specific distance
 *
 * @param distanceImage Distance transform image
 * @param distance Target distance
 * @param tolerance Tolerance around target distance
 * @return Points at the specified distance
 */
std::vector<Point2i> FindPixelsAtDistance(const QImage& distanceImage,
                                          double distance,
                                          double tolerance = 0.5);

/**
 * @brief Find local maxima in distance transform (skeleton candidates)
 *
 * @param distanceImage Distance transform image
 * @param minDistance Minimum distance value to consider
 * @return Points at local maxima
 */
std::vector<Point2i> FindDistanceMaxima(const QImage& distanceImage,
                                        double minDistance = 1.0);

} // namespace Qi::Vision::Internal
