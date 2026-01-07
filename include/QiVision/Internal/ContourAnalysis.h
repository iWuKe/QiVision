#pragma once

/**
 * @file ContourAnalysis.h
 * @brief Contour geometric property analysis for QiVision
 *
 * This module provides:
 * - Basic properties: length, area, centroid, perimeter
 * - Curvature analysis: point curvature, statistics, histogram
 * - Orientation: principal axis direction, ellipse fitting direction
 * - Moments: geometric, central, normalized, Hu invariant moments
 * - Shape descriptors: circularity, compactness, convexity, eccentricity, etc.
 * - Bounding geometry: AABB, min area rect, min enclosing circle/ellipse
 * - Convexity analysis: convex hull, convexity defects
 *
 * Used by:
 * - Internal/ContourSelect: Filter contours by properties
 * - Feature/Blob: Blob analysis
 * - Feature/Edge: Edge analysis
 *
 * Design principles:
 * - All functions are pure (input not modified)
 * - All coordinates use double for subpixel precision
 * - Distinguish between open and closed contours
 * - Area-based properties require closed contours
 */

#include <QiVision/Core/QContour.h>
#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>

#include <array>
#include <optional>
#include <vector>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Minimum points for area calculation
constexpr size_t MIN_POINTS_FOR_AREA = 3;

/// Minimum points for curvature calculation
constexpr size_t MIN_POINTS_FOR_CURVATURE = 3;

/// Minimum points for moment calculation
constexpr size_t MIN_POINTS_FOR_MOMENTS = 3;

/// Minimum points for convex hull
constexpr size_t MIN_POINTS_FOR_CONVEX_HULL = 3;

/// Default window size for curvature calculation
constexpr int32_t DEFAULT_CURVATURE_WINDOW = 5;

/// Curvature computation tolerance
constexpr double CURVATURE_TOLERANCE = 1e-10;

/// Moment computation tolerance
constexpr double MOMENT_TOLERANCE = 1e-15;

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief Area and centroid result
 */
struct AreaCenterResult {
    double area = 0.0;          ///< Signed area (positive=CCW)
    Point2d centroid;           ///< Geometric centroid
    bool valid = false;         ///< Whether result is valid
};

/**
 * @brief Curvature statistics
 */
struct CurvatureStats {
    double mean = 0.0;          ///< Mean curvature
    double stddev = 0.0;        ///< Standard deviation
    double min = 0.0;           ///< Minimum curvature
    double max = 0.0;           ///< Maximum curvature
    double median = 0.0;        ///< Median curvature
    size_t minIndex = 0;        ///< Index of minimum curvature point
    size_t maxIndex = 0;        ///< Index of maximum curvature point
};

/**
 * @brief Principal axes result
 */
struct PrincipalAxesResult {
    Point2d centroid;           ///< Centroid
    double angle = 0.0;         ///< Principal axis angle (radians)
    double majorLength = 0.0;   ///< Length along major axis
    double minorLength = 0.0;   ///< Length along minor axis
    Point2d majorAxis;          ///< Unit vector of major axis
    Point2d minorAxis;          ///< Unit vector of minor axis
    bool valid = false;
};

/**
 * @brief Geometric moments result (up to order 3)
 */
struct MomentsResult {
    // Raw moments m_pq = sum(x^p * y^q)
    double m00 = 0.0;           ///< Zeroth moment (area)
    double m10 = 0.0, m01 = 0.0;          ///< First moments
    double m20 = 0.0, m11 = 0.0, m02 = 0.0;   ///< Second moments
    double m30 = 0.0, m21 = 0.0, m12 = 0.0, m03 = 0.0; ///< Third moments

    /// Get centroid from moments
    Point2d Centroid() const {
        if (m00 < MOMENT_TOLERANCE) return {0, 0};
        return {m10 / m00, m01 / m00};
    }
};

/**
 * @brief Central moments result
 */
struct CentralMomentsResult {
    // Central moments mu_pq = sum((x-cx)^p * (y-cy)^q)
    double mu00 = 0.0;          ///< = m00
    double mu20 = 0.0, mu11 = 0.0, mu02 = 0.0;   ///< Second central moments
    double mu30 = 0.0, mu21 = 0.0, mu12 = 0.0, mu03 = 0.0; ///< Third central moments

    Point2d centroid;           ///< Centroid used
};

/**
 * @brief Normalized central moments result
 */
struct NormalizedMomentsResult {
    // Normalized moments eta_pq = mu_pq / mu00^((p+q)/2 + 1)
    double eta20 = 0.0, eta11 = 0.0, eta02 = 0.0;
    double eta30 = 0.0, eta21 = 0.0, eta12 = 0.0, eta03 = 0.0;
};

/**
 * @brief Hu invariant moments (7 values)
 */
struct HuMomentsResult {
    std::array<double, 7> hu = {};  ///< Hu moments h1-h7

    double& operator[](size_t i) { return hu[i]; }
    const double& operator[](size_t i) const { return hu[i]; }
};

/**
 * @brief All shape descriptors
 */
struct ShapeDescriptors {
    double circularity = 0.0;       ///< 4*pi*A/P^2, 1.0 for circle
    double compactness = 0.0;       ///< P^2/A
    double convexity = 0.0;         ///< Convex hull perimeter / contour perimeter
    double solidity = 0.0;          ///< Area / convex hull area
    double eccentricity = 0.0;      ///< sqrt(1 - (b/a)^2), 0 for circle
    double elongation = 0.0;        ///< 1 - minorAxis/majorAxis
    double rectangularity = 0.0;    ///< Area / min bounding rect area
    double extent = 0.0;            ///< Area / AABB area
    double aspectRatio = 0.0;       ///< Major axis / minor axis

    bool valid = false;
};

/**
 * @brief Convexity defect
 */
struct ConvexityDefect {
    size_t startIndex = 0;      ///< Start point index on contour
    size_t endIndex = 0;        ///< End point index on contour
    size_t deepestIndex = 0;    ///< Deepest point index on contour
    Point2d startPoint;         ///< Start point (on convex hull)
    Point2d endPoint;           ///< End point (on convex hull)
    Point2d deepestPoint;       ///< Deepest defect point
    double depth = 0.0;         ///< Defect depth (perpendicular distance)
};

// =============================================================================
// Curvature Methods
// =============================================================================

/**
 * @brief Method for curvature calculation
 */
enum class CurvatureMethod {
    ThreePoint,         ///< 3-point circle fitting (default)
    FivePoint,          ///< 5-point circle fitting (smoother)
    Derivative,         ///< Based on derivatives (k = |x'y'' - x''y'| / (x'^2+y'^2)^1.5)
    Regression          ///< Local polynomial regression
};

// =============================================================================
// Basic Property Functions
// =============================================================================

/**
 * @brief Compute contour length (arc length)
 *
 * For closed contours, includes the closing segment.
 *
 * @param contour Input contour
 * @return Total arc length (0 if empty or single point)
 */
double ContourLength(const QContour& contour);

/**
 * @brief Compute signed area of a closed contour
 *
 * Uses the shoelace formula: A = 0.5 * sum(x_i * y_{i+1} - x_{i+1} * y_i)
 * Positive area indicates counter-clockwise orientation.
 *
 * @param contour Input contour (should be closed)
 * @return Signed area (positive=CCW, negative=CW)
 *
 * @note For open contours, treats as closed by connecting last to first.
 *       Returns 0 for contours with fewer than 3 points.
 */
double ContourSignedArea(const QContour& contour);

/**
 * @brief Compute absolute area of a closed contour
 *
 * @param contour Input contour
 * @return Absolute area (always non-negative)
 */
double ContourArea(const QContour& contour);

/**
 * @brief Compute perimeter of contour
 *
 * For closed contours, same as length.
 * For open contours, same as length.
 *
 * @param contour Input contour
 * @return Perimeter
 */
double ContourPerimeter(const QContour& contour);

/**
 * @brief Compute geometric centroid of contour
 *
 * For closed contours: uses area-weighted centroid formula.
 * For open contours: uses simple average of points.
 *
 * @param contour Input contour
 * @return Centroid point
 */
Point2d ContourCentroid(const QContour& contour);

/**
 * @brief Compute area and centroid together (more efficient)
 *
 * @param contour Input contour
 * @return AreaCenterResult with area, centroid, and validity flag
 */
AreaCenterResult ContourAreaCenter(const QContour& contour);

// =============================================================================
// Curvature Analysis Functions
// =============================================================================

/**
 * @brief Compute curvature at each point of the contour
 *
 * Curvature k = 1/R where R is the radius of the osculating circle.
 * Positive curvature = left turn, negative = right turn.
 *
 * @param contour Input contour
 * @param method Curvature calculation method
 * @param windowSize Window size for smoothing (used by some methods)
 * @return Vector of curvatures (same size as contour points)
 *
 * @note Window size affects smoothness: larger = smoother but less local.
 */
std::vector<double> ComputeContourCurvature(const QContour& contour,
                                             CurvatureMethod method = CurvatureMethod::ThreePoint,
                                             int32_t windowSize = DEFAULT_CURVATURE_WINDOW);

/**
 * @brief Compute mean curvature of contour
 *
 * @param contour Input contour
 * @param method Curvature method
 * @return Mean absolute curvature
 */
double ContourMeanCurvature(const QContour& contour,
                            CurvatureMethod method = CurvatureMethod::ThreePoint);

/**
 * @brief Compute maximum absolute curvature
 *
 * @param contour Input contour
 * @param method Curvature method
 * @return Maximum absolute curvature value
 */
double ContourMaxCurvature(const QContour& contour,
                           CurvatureMethod method = CurvatureMethod::ThreePoint);

/**
 * @brief Compute minimum absolute curvature
 *
 * @param contour Input contour
 * @param method Curvature method
 * @return Minimum absolute curvature value (often near 0 for straight segments)
 */
double ContourMinCurvature(const QContour& contour,
                           CurvatureMethod method = CurvatureMethod::ThreePoint);

/**
 * @brief Compute comprehensive curvature statistics
 *
 * @param contour Input contour
 * @param method Curvature method
 * @return CurvatureStats with mean, stddev, min, max, median
 */
CurvatureStats ContourCurvatureStats(const QContour& contour,
                                      CurvatureMethod method = CurvatureMethod::ThreePoint);

/**
 * @brief Compute curvature histogram
 *
 * @param contour Input contour
 * @param numBins Number of histogram bins
 * @param minCurvature Minimum curvature for histogram range (auto if >= maxCurvature)
 * @param maxCurvature Maximum curvature for histogram range
 * @param method Curvature method
 * @return Histogram as vector of counts
 */
std::vector<int32_t> ContourCurvatureHistogram(const QContour& contour,
                                                int32_t numBins = 32,
                                                double minCurvature = 0.0,
                                                double maxCurvature = 0.0,
                                                CurvatureMethod method = CurvatureMethod::ThreePoint);

// =============================================================================
// Orientation Functions
// =============================================================================

/**
 * @brief Compute principal axis orientation
 *
 * Based on second-order central moments (covariance matrix eigenanalysis).
 * Returns angle of major axis from positive X-axis.
 *
 * @param contour Input contour
 * @return Angle in radians [-PI/2, PI/2]
 */
double ContourOrientation(const QContour& contour);

/**
 * @brief Compute orientation using ellipse fitting
 *
 * Fits an ellipse to the contour points and returns its orientation.
 *
 * @param contour Input contour
 * @return Angle in radians [-PI/2, PI/2], or 0 if fitting fails
 */
double ContourOrientationEllipse(const QContour& contour);

/**
 * @brief Compute principal axes with full information
 *
 * @param contour Input contour
 * @return PrincipalAxesResult with centroid, angle, axis lengths
 */
PrincipalAxesResult ContourPrincipalAxes(const QContour& contour);

// =============================================================================
// Moment Functions
// =============================================================================

/**
 * @brief Compute raw geometric moments
 *
 * m_pq = sum(x^p * y^q) for all contour points.
 * For closed contours, uses Green's theorem for exact area integration.
 *
 * @param contour Input contour
 * @return MomentsResult with m00, m10, m01, m20, m11, m02, m30, m21, m12, m03
 */
MomentsResult ContourMoments(const QContour& contour);

/**
 * @brief Compute central moments
 *
 * mu_pq = sum((x-cx)^p * (y-cy)^q) where (cx,cy) is centroid.
 * Translation invariant.
 *
 * @param contour Input contour
 * @return CentralMomentsResult
 */
CentralMomentsResult ContourCentralMoments(const QContour& contour);

/**
 * @brief Compute normalized central moments
 *
 * eta_pq = mu_pq / mu00^((p+q)/2 + 1)
 * Translation and scale invariant.
 *
 * @param contour Input contour
 * @return NormalizedMomentsResult
 */
NormalizedMomentsResult ContourNormalizedMoments(const QContour& contour);

/**
 * @brief Compute Hu invariant moments
 *
 * Seven moments invariant to translation, scale, and rotation.
 * h7 also has sign flip invariance under reflection.
 *
 * @param contour Input contour
 * @return HuMomentsResult with 7 Hu moments
 *
 * @note Formula based on Hu (1962):
 * h1 = eta20 + eta02
 * h2 = (eta20 - eta02)^2 + 4*eta11^2
 * h3 = (eta30 - 3*eta12)^2 + (3*eta21 - eta03)^2
 * h4 = (eta30 + eta12)^2 + (eta21 + eta03)^2
 * h5 = (eta30 - 3*eta12)(eta30 + eta12)[(eta30 + eta12)^2 - 3(eta21 + eta03)^2]
 *      + (3*eta21 - eta03)(eta21 + eta03)[3(eta30 + eta12)^2 - (eta21 + eta03)^2]
 * h6 = (eta20 - eta02)[(eta30 + eta12)^2 - (eta21 + eta03)^2]
 *      + 4*eta11*(eta30 + eta12)(eta21 + eta03)
 * h7 = (3*eta21 - eta03)(eta30 + eta12)[(eta30 + eta12)^2 - 3(eta21 + eta03)^2]
 *      - (eta30 - 3*eta12)(eta21 + eta03)[3(eta30 + eta12)^2 - (eta21 + eta03)^2]
 */
HuMomentsResult ContourHuMoments(const QContour& contour);

// =============================================================================
// Shape Descriptor Functions
// =============================================================================

/**
 * @brief Compute circularity (isoperimetric quotient)
 *
 * Circularity = 4 * PI * Area / Perimeter^2
 * Equals 1.0 for a perfect circle, < 1.0 for other shapes.
 *
 * @param contour Input contour (should be closed)
 * @return Circularity in [0, 1], or 0 if invalid
 */
double ContourCircularity(const QContour& contour);

/**
 * @brief Compute compactness
 *
 * Compactness = Perimeter^2 / Area
 * Minimum for circle (4*PI), larger for elongated/irregular shapes.
 *
 * @param contour Input contour
 * @return Compactness (>= 4*PI)
 */
double ContourCompactness(const QContour& contour);

/**
 * @brief Compute convexity
 *
 * Convexity = Convex hull perimeter / Contour perimeter
 * Equals 1.0 for convex shapes, < 1.0 for concave shapes.
 *
 * @param contour Input contour
 * @return Convexity in [0, 1]
 */
double ContourConvexity(const QContour& contour);

/**
 * @brief Compute solidity
 *
 * Solidity = Contour area / Convex hull area
 * Equals 1.0 for convex shapes, < 1.0 for concave shapes.
 *
 * @param contour Input contour
 * @return Solidity in [0, 1]
 */
double ContourSolidity(const QContour& contour);

/**
 * @brief Compute eccentricity
 *
 * Eccentricity = sqrt(1 - (minorAxis/majorAxis)^2)
 * Equals 0 for circle, approaches 1 for elongated shapes.
 *
 * @param contour Input contour
 * @return Eccentricity in [0, 1)
 */
double ContourEccentricity(const QContour& contour);

/**
 * @brief Compute elongation
 *
 * Elongation = 1 - minorAxis / majorAxis
 * Equals 0 for circle, approaches 1 for line.
 *
 * @param contour Input contour
 * @return Elongation in [0, 1)
 */
double ContourElongation(const QContour& contour);

/**
 * @brief Compute rectangularity
 *
 * Rectangularity = Area / MinAreaRect.Area
 * Equals 1.0 for rectangle, < 1.0 for other shapes.
 *
 * @param contour Input contour
 * @return Rectangularity in [0, 1]
 */
double ContourRectangularity(const QContour& contour);

/**
 * @brief Compute extent
 *
 * Extent = Area / BoundingBox.Area
 * Equals PI/4 for circle in AABB, 1.0 for axis-aligned rectangle.
 *
 * @param contour Input contour
 * @return Extent in [0, 1]
 */
double ContourExtent(const QContour& contour);

/**
 * @brief Compute aspect ratio
 *
 * AspectRatio = majorAxisLength / minorAxisLength
 * Equals 1.0 for circle, > 1.0 for elongated shapes.
 *
 * @param contour Input contour
 * @return Aspect ratio (>= 1.0)
 */
double ContourAspectRatio(const QContour& contour);

/**
 * @brief Compute all shape descriptors at once (more efficient)
 *
 * @param contour Input contour
 * @return ShapeDescriptors with all values
 */
ShapeDescriptors ContourAllDescriptors(const QContour& contour);

// =============================================================================
// Bounding Geometry Functions
// =============================================================================

/**
 * @brief Compute axis-aligned bounding box
 *
 * @param contour Input contour
 * @return Rect2d bounding box
 */
Rect2d ContourBoundingBox(const QContour& contour);

/**
 * @brief Compute minimum area enclosing rectangle
 *
 * Uses rotating calipers algorithm on convex hull.
 *
 * @param contour Input contour
 * @return RotatedRect2d, or nullopt if fewer than 3 points
 */
std::optional<RotatedRect2d> ContourMinAreaRect(const QContour& contour);

/**
 * @brief Compute minimum enclosing circle
 *
 * Uses Welzl's algorithm (expected O(n) time).
 *
 * @param contour Input contour
 * @return Circle2d, or nullopt if empty
 */
std::optional<Circle2d> ContourMinEnclosingCircle(const QContour& contour);

/**
 * @brief Compute minimum enclosing ellipse (approximate)
 *
 * Fits ellipse to contour using least squares.
 *
 * @param contour Input contour
 * @return Ellipse2d, or nullopt if fewer than 5 points
 */
std::optional<Ellipse2d> ContourMinEnclosingEllipse(const QContour& contour);

// =============================================================================
// Convexity Analysis Functions
// =============================================================================

/**
 * @brief Compute convex hull of contour points
 *
 * @param contour Input contour
 * @return QContour representing the convex hull (closed)
 */
QContour ContourConvexHull(const QContour& contour);

/**
 * @brief Compute convex hull area
 *
 * @param contour Input contour
 * @return Convex hull area
 */
double ContourConvexHullArea(const QContour& contour);

/**
 * @brief Check if contour is convex
 *
 * @param contour Input contour
 * @return true if all points form a convex polygon
 */
bool IsContourConvex(const QContour& contour);

/**
 * @brief Find convexity defects
 *
 * Convexity defects are regions where the contour deviates from its convex hull.
 *
 * @param contour Input contour
 * @param minDepth Minimum defect depth to report (pixels)
 * @return Vector of ConvexityDefect
 */
std::vector<ConvexityDefect> ContourConvexityDefects(const QContour& contour,
                                                      double minDepth = 1.0);

// =============================================================================
// Shape Comparison Functions
// =============================================================================

/**
 * @brief Compare two shapes using Hu moments
 *
 * Lower values indicate more similar shapes.
 *
 * @param contour1 First contour
 * @param contour2 Second contour
 * @param method Comparison method (1, 2, or 3)
 *               1: sum(|1/m1_i - 1/m2_i|)
 *               2: sum(|m1_i - m2_i|)
 *               3: max(|m1_i - m2_i| / |m1_i|)
 * @return Similarity measure (lower = more similar)
 */
double MatchShapesHu(const QContour& contour1, const QContour& contour2, int method = 1);

/**
 * @brief Compare two contours using shape context or Fourier descriptors
 *
 * @param contour1 First contour
 * @param contour2 Second contour
 * @return Similarity score in [0, 1] (1 = identical)
 */
double MatchShapesContour(const QContour& contour1, const QContour& contour2);

} // namespace Qi::Vision::Internal
