#pragma once

/**
 * @file RegionFeatures.h
 * @brief Region feature extraction and shape analysis
 *
 * This module provides algorithms for computing various features
 * of binary regions (QRegion), including geometric properties,
 * shape descriptors, and moment-based features.
 *
 * Feature categories:
 * - Basic: area, perimeter, bounding box, centroid
 * - Shape: circularity, compactness, elongation, rectangularity
 * - Moments: raw, central, normalized, Hu invariant moments
 * - Convex hull: convexity, solidity, convex defects
 * - Orientation: principal axes, angle, eccentricity
 *
 * Reference Halcon operators:
 * - area_center, smallest_rectangle1/2, circularity
 * - compactness, convexity, moments_region_*
 * - orientation_region, elliptic_axis
 */

#include <QiVision/Core/QRegion.h>
#include <QiVision/Core/Types.h>

#include <vector>
#include <array>

namespace Qi::Vision::Internal {

// =============================================================================
// Feature Structures
// =============================================================================

/**
 * @brief Basic geometric features of a region
 */
struct RegionBasicFeatures {
    int64_t area = 0;           ///< Area in pixels
    double perimeter = 0.0;     ///< Perimeter length
    double centroidX = 0.0;     ///< Centroid X coordinate
    double centroidY = 0.0;     ///< Centroid Y coordinate
    Rect2i boundingBox;         ///< Axis-aligned bounding box
};

/**
 * @brief Shape descriptor features
 */
struct RegionShapeFeatures {
    double circularity = 0.0;      ///< 4*pi*area/perimeter^2, 1.0 for circle
    double compactness = 0.0;      ///< perimeter^2/area, minimum for circle
    double elongation = 0.0;       ///< Major/minor axis ratio
    double rectangularity = 0.0;   ///< area / bounding_box_area
    double convexity = 0.0;        ///< perimeter_convex_hull / perimeter
    double solidity = 0.0;         ///< area / convex_hull_area
    double roundness = 0.0;        ///< 4*area / (pi*major_axis^2)
    double aspectRatio = 0.0;      ///< bbox_width / bbox_height
};

/**
 * @brief Moment-based features
 */
struct RegionMoments {
    // Raw moments m_pq = sum(x^p * y^q)
    double m00 = 0.0;  ///< Area
    double m10 = 0.0;  ///< First moment X
    double m01 = 0.0;  ///< First moment Y
    double m20 = 0.0;  ///< Second moment XX
    double m11 = 0.0;  ///< Second moment XY
    double m02 = 0.0;  ///< Second moment YY
    double m30 = 0.0;  ///< Third moment XXX
    double m21 = 0.0;  ///< Third moment XXY
    double m12 = 0.0;  ///< Third moment XYY
    double m03 = 0.0;  ///< Third moment YYY

    // Central moments mu_pq = sum((x-cx)^p * (y-cy)^q)
    double mu20 = 0.0;
    double mu11 = 0.0;
    double mu02 = 0.0;
    double mu30 = 0.0;
    double mu21 = 0.0;
    double mu12 = 0.0;
    double mu03 = 0.0;

    // Normalized central moments nu_pq = mu_pq / m00^((p+q)/2+1)
    double nu20 = 0.0;
    double nu11 = 0.0;
    double nu02 = 0.0;
    double nu30 = 0.0;
    double nu21 = 0.0;
    double nu12 = 0.0;
    double nu03 = 0.0;

    // Hu moments (rotation, scale, translation invariant)
    std::array<double, 7> hu = {0};
};

/**
 * @brief Ellipse fitting features (equivalent ellipse)
 */
struct RegionEllipseFeatures {
    double centerX = 0.0;       ///< Ellipse center X
    double centerY = 0.0;       ///< Ellipse center Y
    double majorAxis = 0.0;     ///< Semi-major axis length
    double minorAxis = 0.0;     ///< Semi-minor axis length
    double angle = 0.0;         ///< Orientation angle in radians
    double eccentricity = 0.0;  ///< sqrt(1 - (minor/major)^2)
};

/**
 * @brief Smallest enclosing rectangle (rotated)
 */
struct MinAreaRect {
    Point2d center;             ///< Rectangle center
    double width = 0.0;         ///< Rectangle width
    double height = 0.0;        ///< Rectangle height
    double angle = 0.0;         ///< Rotation angle in radians
    std::array<Point2d, 4> corners;  ///< Corner points
};

/**
 * @brief Smallest enclosing circle
 */
struct MinEnclosingCircle {
    Point2d center;             ///< Circle center
    double radius = 0.0;        ///< Circle radius
};

/**
 * @brief Comprehensive region features
 */
struct RegionFeatures {
    RegionBasicFeatures basic;
    RegionShapeFeatures shape;
    RegionMoments moments;
    RegionEllipseFeatures ellipse;
};

// =============================================================================
// Basic Features
// =============================================================================

/**
 * @brief Compute basic geometric features
 *
 * @param region Input region
 * @return Basic features (area, perimeter, centroid, bounding box)
 */
RegionBasicFeatures ComputeBasicFeatures(const QRegion& region);

/**
 * @brief Compute region area
 *
 * @param region Input region
 * @return Area in pixels
 */
int64_t ComputeArea(const QRegion& region);

// Note: ComputePerimeter is declared in RLEOps.h

/**
 * @brief Compute region centroid
 *
 * @param region Input region
 * @return Centroid point
 */
Point2d ComputeRegionCentroid(const QRegion& region);

/**
 * @brief Compute axis-aligned bounding box
 *
 * @param region Input region
 * @return Bounding box
 */
Rect2i ComputeBoundingBox(const QRegion& region);

// =============================================================================
// Shape Features
// =============================================================================

/**
 * @brief Compute shape descriptor features
 *
 * @param region Input region
 * @return Shape features
 */
RegionShapeFeatures ComputeShapeFeatures(const QRegion& region);

// Note: ComputeCircularity, ComputeCompactness, ComputeRectangularity
// are declared in RLEOps.h

/**
 * @brief Compute elongation (aspect ratio of equivalent ellipse)
 *
 * @param region Input region
 * @return Elongation value >= 1.0
 */
double ComputeElongation(const QRegion& region);

/**
 * @brief Compute convexity
 *
 * Convexity = perimeter_convex_hull / perimeter
 *
 * @param region Input region
 * @return Convexity value [0, 1]
 */
double ComputeConvexity(const QRegion& region);

/**
 * @brief Compute solidity
 *
 * Solidity = area / convex_hull_area
 *
 * @param region Input region
 * @return Solidity value [0, 1]
 */
double ComputeSolidity(const QRegion& region);

// =============================================================================
// Moment Features
// =============================================================================

/**
 * @brief Compute all moment-based features
 *
 * Includes raw, central, normalized, and Hu moments.
 *
 * @param region Input region
 * @return Moment features
 */
RegionMoments ComputeMoments(const QRegion& region);

/**
 * @brief Compute raw spatial moments
 *
 * @param region Input region
 * @param p X exponent (0-3)
 * @param q Y exponent (0-3)
 * @return Raw moment m_pq
 */
double ComputeRawMoment(const QRegion& region, int32_t p, int32_t q);

/**
 * @brief Compute central moments
 *
 * @param region Input region
 * @param p X exponent
 * @param q Y exponent
 * @return Central moment mu_pq
 */
double ComputeCentralMoment(const QRegion& region, int32_t p, int32_t q);

/**
 * @brief Compute Hu invariant moments
 *
 * @param region Input region
 * @return Array of 7 Hu moments
 */
std::array<double, 7> ComputeHuMoments(const QRegion& region);

// =============================================================================
// Ellipse and Orientation Features
// =============================================================================

/**
 * @brief Compute equivalent ellipse features
 *
 * Finds the ellipse with the same second moments as the region.
 *
 * @param region Input region
 * @return Ellipse features
 */
RegionEllipseFeatures ComputeEllipseFeatures(const QRegion& region);

/**
 * @brief Compute principal orientation angle
 *
 * @param region Input region
 * @return Orientation angle in radians [-pi/2, pi/2]
 */
double ComputeOrientation(const QRegion& region);

/**
 * @brief Compute principal axes lengths
 *
 * @param region Input region
 * @param[out] majorAxis Length of major axis
 * @param[out] minorAxis Length of minor axis
 */
void ComputePrincipalAxes(const QRegion& region,
                          double& majorAxis,
                          double& minorAxis);

// =============================================================================
// Enclosing Shapes
// =============================================================================

/**
 * @brief Compute minimum area enclosing rectangle
 *
 * @param region Input region
 * @return Minimum area rectangle
 */
MinAreaRect ComputeMinAreaRect(const QRegion& region);

/**
 * @brief Compute minimum enclosing circle
 *
 * @param region Input region
 * @return Minimum enclosing circle
 */
MinEnclosingCircle ComputeMinEnclosingCircle(const QRegion& region);

/**
 * @brief Compute convex hull of region
 *
 * @param region Input region
 * @return Convex hull as vector of points
 */
std::vector<Point2i> ComputeConvexHull(const QRegion& region);

/**
 * @brief Compute convex hull area
 *
 * @param region Input region
 * @return Convex hull area
 */
double ComputeConvexHullArea(const QRegion& region);

/**
 * @brief Compute convex hull perimeter
 *
 * @param region Input region
 * @return Convex hull perimeter
 */
double ComputeConvexHullPerimeter(const QRegion& region);

// =============================================================================
// Comprehensive Feature Extraction
// =============================================================================

/**
 * @brief Compute all region features
 *
 * @param region Input region
 * @return Comprehensive features structure
 */
RegionFeatures ComputeAllFeatures(const QRegion& region);

/**
 * @brief Compute features for multiple regions
 *
 * @param regions Input regions
 * @return Vector of feature structures
 */
std::vector<RegionFeatures> ComputeAllFeatures(const std::vector<QRegion>& regions);

// =============================================================================
// Feature-based Selection
// =============================================================================

/**
 * @brief Select regions by circularity
 *
 * @param regions Input regions
 * @param minCirc Minimum circularity
 * @param maxCirc Maximum circularity
 * @return Selected regions
 */
std::vector<QRegion> SelectByCircularity(const std::vector<QRegion>& regions,
                                          double minCirc,
                                          double maxCirc);

/**
 * @brief Select regions by compactness
 *
 * @param regions Input regions
 * @param minComp Minimum compactness
 * @param maxComp Maximum compactness
 * @return Selected regions
 */
std::vector<QRegion> SelectByCompactness(const std::vector<QRegion>& regions,
                                          double minComp,
                                          double maxComp);

/**
 * @brief Select regions by elongation
 *
 * @param regions Input regions
 * @param minElong Minimum elongation
 * @param maxElong Maximum elongation
 * @return Selected regions
 */
std::vector<QRegion> SelectByElongation(const std::vector<QRegion>& regions,
                                         double minElong,
                                         double maxElong);

/**
 * @brief Select regions by orientation
 *
 * @param regions Input regions
 * @param minAngle Minimum angle (radians)
 * @param maxAngle Maximum angle (radians)
 * @return Selected regions
 */
std::vector<QRegion> SelectByOrientation(const std::vector<QRegion>& regions,
                                          double minAngle,
                                          double maxAngle);

} // namespace Qi::Vision::Internal
