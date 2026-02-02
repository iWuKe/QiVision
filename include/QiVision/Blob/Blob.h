#pragma once

#include <QiVision/Core/Export.h>

/**
 * @file Blob.h
 * @brief Blob analysis and region selection (Halcon-style API)
 *
 * Halcon reference operators:
 * - connection, select_shape, count_obj, select_obj
 * - area_center, smallest_rectangle1/2, smallest_circle
 * - circularity, compactness, convexity, orientation_region
 * - moments_region_2nd, elliptic_axis
 *
 * Precision: sub-pixel features computed from RLE region representation
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>
#include <QiVision/Core/Types.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace Qi::Vision::Blob {

// =============================================================================
// Types
// =============================================================================

/**
 * @brief Shape features for select_shape
 */
enum class ShapeFeature {
    Area,               ///< Region area in pixels
    Row,                ///< Centroid row
    Column,             ///< Centroid column
    Width,              ///< Bounding box width
    Height,             ///< Bounding box height
    Circularity,        ///< 4*pi*area/perimeter^2
    Compactness,        ///< perimeter^2/area
    Convexity,          ///< perimeter_convex/perimeter
    Rectangularity,     ///< area/bbox_area
    Elongation,         ///< major_axis/minor_axis
    Orientation,        ///< Principal axis angle (radians)
    Ra,                 ///< Equivalent ellipse major radius
    Rb,                 ///< Equivalent ellipse minor radius
    Phi,                ///< Equivalent ellipse orientation
    Anisometry,         ///< Ra/Rb
    Bulkiness,          ///< pi*Ra*Rb/area
    StructureFactor,    ///< Anisometry*Bulkiness-1
    OuterRadius,        ///< Smallest enclosing circle radius
    InnerRadius,        ///< Largest inscribed circle radius
    Holes               ///< Number of holes
};

/**
 * @brief Selection operation mode
 */
enum class SelectOperation {
    And,    ///< All features must be in range
    Or      ///< Any feature in range
};

/**
 * @brief Region sorting criteria
 */
enum class SortMode {
    None,               ///< No sorting
    Area,               ///< Sort by area
    Row,                ///< Sort by centroid row
    Column,             ///< Sort by centroid column
    FirstPoint,         ///< Sort by first point
    LastPoint           ///< Sort by last point
};

// =============================================================================
// Connection (Connected Component Analysis)
// =============================================================================

/**
 * @brief Extract connected components from a region
 *
 * Equivalent to Halcon's connection operator.
 *
 * @param region Input region
 * @param[out] regions Output connected regions
 *
 * @code
 * QRegion binaryRegion = ThresholdToRegion(image, 128, 255);
 * std::vector<QRegion> blobs;
 * Connection(binaryRegion, blobs);
 * std::cout << "Found " << blobs.size() << " blobs\n";
 * @endcode
 */
QIVISION_API void Connection(const QRegion& region, std::vector<QRegion>& regions);

/**
 * @brief Extract connected components from binary image
 *
 * @param binaryImage Binary image (non-zero = foreground)
 * @param[out] regions Output connected regions
 * @param connectivity 4 or 8 connectivity (default: 8)
 */
QIVISION_API void Connection(const QImage& binaryImage,
                std::vector<QRegion>& regions,
                Connectivity connectivity = Connectivity::Eight);

/**
 * @brief Count number of objects
 *
 * Equivalent to Halcon's count_obj operator.
 */
inline int32_t CountObj(const std::vector<QRegion>& regions) {
    return static_cast<int32_t>(regions.size());
}

/**
 * @brief Select object by index (1-based like Halcon)
 *
 * Equivalent to Halcon's select_obj operator.
 *
 * @param regions Input regions
 * @param index 1-based index
 * @return Selected region (empty if out of range)
 */
QIVISION_API QRegion SelectObj(const std::vector<QRegion>& regions, int32_t index);

// =============================================================================
// Region Features
// =============================================================================

/**
 * @brief Get area and centroid of region
 *
 * Equivalent to Halcon's area_center operator.
 *
 * @param region Input region
 * @param[out] area Area in pixels
 * @param[out] row Centroid row
 * @param[out] column Centroid column
 */
QIVISION_API void AreaCenter(const QRegion& region, int64_t& area, double& row, double& column);

/**
 * @brief Get area and centroid for multiple regions
 */
QIVISION_API void AreaCenter(const std::vector<QRegion>& regions,
                std::vector<int64_t>& areas,
                std::vector<double>& rows,
                std::vector<double>& columns);

/**
 * @brief Get axis-aligned bounding box
 *
 * Equivalent to Halcon's smallest_rectangle1 operator.
 *
 * @param region Input region
 * @param[out] row1 Top row
 * @param[out] column1 Left column
 * @param[out] row2 Bottom row
 * @param[out] column2 Right column
 */
QIVISION_API void SmallestRectangle1(const QRegion& region,
                         int32_t& row1, int32_t& column1,
                         int32_t& row2, int32_t& column2);

/**
 * @brief Get minimum area rotated bounding box
 *
 * Equivalent to Halcon's smallest_rectangle2 operator.
 *
 * @param region Input region
 * @param[out] row Center row
 * @param[out] column Center column
 * @param[out] phi Rotation angle (radians)
 * @param[out] length1 Half-length along major axis
 * @param[out] length2 Half-length along minor axis
 */
QIVISION_API void SmallestRectangle2(const QRegion& region,
                         double& row, double& column, double& phi,
                         double& length1, double& length2);

/**
 * @brief Get smallest enclosing circle
 *
 * Equivalent to Halcon's smallest_circle operator.
 *
 * @param region Input region
 * @param[out] row Circle center row
 * @param[out] column Circle center column
 * @param[out] radius Circle radius
 */
QIVISION_API void SmallestCircle(const QRegion& region,
                     double& row, double& column, double& radius);

/**
 * @brief Get largest inscribed circle (inner circle)
 *
 * Equivalent to Halcon's inner_circle operator.
 * Uses distance transform to find the point furthest from the boundary.
 *
 * @param region Input region
 * @param[out] row Circle center row
 * @param[out] column Circle center column
 * @param[out] radius Circle radius
 */
QIVISION_API void InnerCircle(const QRegion& region,
                  double& row, double& column, double& radius);

/**
 * @brief Compute region contour length (perimeter)
 *
 * Equivalent to Halcon's contlength operator.
 *
 * @param region Input region
 * @return Contour length in pixels
 */
QIVISION_API double ContourLength(const QRegion& region);

/**
 * @brief Count number of holes in a region
 *
 * Equivalent to Halcon's connect_and_holes / euler_number.
 *
 * @param region Input region
 * @return Number of holes (0 if no holes)
 */
QIVISION_API int32_t CountHoles(const QRegion& region);

/**
 * @brief Get the Euler number of a region
 *
 * Euler number = #connected_components - #holes
 * For a single region without holes, Euler = 1
 *
 * @param region Input region
 * @return Euler number
 */
QIVISION_API int32_t EulerNumber(const QRegion& region);

/**
 * @brief Fill holes in a region
 *
 * Equivalent to Halcon's fill_up operator.
 *
 * @param region Input region
 * @param[out] filled Region with holes filled
 */
QIVISION_API void FillUp(const QRegion& region, QRegion& filled);

/**
 * @brief Get the holes of a region
 *
 * @param region Input region
 * @param[out] holes Vector of hole regions
 */
QIVISION_API void GetHoles(const QRegion& region, std::vector<QRegion>& holes);

/**
 * @brief Compute region circularity
 *
 * Equivalent to Halcon's circularity operator.
 * Circularity = 4*pi*area/perimeter^2, 1.0 for perfect circle.
 *
 * @param region Input region
 * @return Circularity value [0, 1]
 */
QIVISION_API double Circularity(const QRegion& region);

/**
 * @brief Compute region compactness
 *
 * Equivalent to Halcon's compactness operator.
 * Compactness = perimeter^2/area, minimum for circle.
 *
 * @param region Input region
 * @return Compactness value (>= 4*pi)
 */
QIVISION_API double Compactness(const QRegion& region);

/**
 * @brief Compute region convexity
 *
 * Equivalent to Halcon's convexity operator.
 *
 * @param region Input region
 * @return Convexity value [0, 1]
 */
QIVISION_API double Convexity(const QRegion& region);

/**
 * @brief Compute region rectangularity
 *
 * @param region Input region
 * @return Rectangularity = area/bbox_area [0, 1]
 */
QIVISION_API double Rectangularity(const QRegion& region);

/**
 * @brief Compute equivalent ellipse parameters
 *
 * Equivalent to Halcon's elliptic_axis operator.
 *
 * @param region Input region
 * @param[out] ra Major radius (semi-major axis)
 * @param[out] rb Minor radius (semi-minor axis)
 * @param[out] phi Orientation angle (radians)
 */
QIVISION_API void EllipticAxis(const QRegion& region, double& ra, double& rb, double& phi);

/**
 * @brief Compute region orientation
 *
 * Equivalent to Halcon's orientation_region operator.
 *
 * @param region Input region
 * @return Orientation angle in radians [-pi/2, pi/2]
 */
QIVISION_API double OrientationRegion(const QRegion& region);

/**
 * @brief Compute second-order central moments
 *
 * Equivalent to Halcon's moments_region_2nd operator.
 *
 * @param region Input region
 * @param[out] m11 Mixed moment
 * @param[out] m20 Second moment in X
 * @param[out] m02 Second moment in Y
 * @param[out] ia Major moment of inertia
 * @param[out] ib Minor moment of inertia
 */
QIVISION_API void MomentsRegion2nd(const QRegion& region,
                       double& m11, double& m20, double& m02,
                       double& ia, double& ib);

/**
 * @brief Compute Hu invariant moments
 *
 * Hu moments are rotation, scale, and translation invariant shape descriptors.
 * They are useful for shape recognition and matching.
 *
 * The 7 Hu moments are derived from normalized central moments:
 * - hu[0]: nu20 + nu02
 * - hu[1]: (nu20 - nu02)^2 + 4*nu11^2
 * - hu[2-6]: Higher-order combinations for more discriminative power
 *
 * @param region Input region
 * @return Array of 7 Hu moments
 *
 * @code
 * QRegion blobRegion = ...;
 * auto hu = HuMoments(blobRegion);
 * // Compare with template region for shape matching
 * double similarity = ComputeHuSimilarity(hu, templateHu);
 * @endcode
 */
QIVISION_API std::array<double, 7> HuMoments(const QRegion& region);

/**
 * @brief Compute Hu invariant moments (Halcon-style output parameters)
 *
 * @param region Input region
 * @param[out] hu1 First Hu moment (most significant)
 * @param[out] hu2 Second Hu moment
 * @param[out] hu3 Third Hu moment
 * @param[out] hu4 Fourth Hu moment
 * @param[out] hu5 Fifth Hu moment
 * @param[out] hu6 Sixth Hu moment
 * @param[out] hu7 Seventh Hu moment (sign can distinguish mirror images)
 */
QIVISION_API void HuMoments(const QRegion& region,
                double& hu1, double& hu2, double& hu3, double& hu4,
                double& hu5, double& hu6, double& hu7);

/**
 * @brief Compute Hu moments for multiple regions
 *
 * @param regions Input regions
 * @param[out] huMoments Output Hu moments for each region
 */
QIVISION_API void HuMoments(const std::vector<QRegion>& regions,
                std::vector<std::array<double, 7>>& huMoments);

/**
 * @brief Compute eccentricity features
 *
 * Equivalent to Halcon's eccentricity operator.
 *
 * @param region Input region
 * @param[out] anisometry Ra/Rb
 * @param[out] bulkiness pi*Ra*Rb/Area
 * @param[out] structureFactor Anisometry*Bulkiness-1
 */
QIVISION_API void Eccentricity(const QRegion& region,
                   double& anisometry, double& bulkiness, double& structureFactor);

// =============================================================================
// Region Selection
// =============================================================================

/**
 * @brief Select regions by shape features
 *
 * Equivalent to Halcon's select_shape operator.
 *
 * @param regions Input regions
 * @param[out] selected Output selected regions
 * @param feature Feature to select by
 * @param operation Selection operation (And/Or)
 * @param minValue Minimum feature value
 * @param maxValue Maximum feature value
 *
 * @code
 * // Select circular blobs with area > 100
 * std::vector<QRegion> selected;
 * SelectShape(blobs, selected, ShapeFeature::Area,
 *             SelectOperation::And, 100, 999999);
 * SelectShape(selected, selected, ShapeFeature::Circularity,
 *             SelectOperation::And, 0.8, 1.0);
 * @endcode
 */
QIVISION_API void SelectShape(const std::vector<QRegion>& regions,
                 std::vector<QRegion>& selected,
                 ShapeFeature feature,
                 SelectOperation operation,
                 double minValue,
                 double maxValue);

/**
 * @brief Select regions by string feature name (Halcon compatible)
 *
 * @param regions Input regions
 * @param[out] selected Output selected regions
 * @param features Feature name: "area", "circularity", "compactness", etc.
 * @param operation "and" or "or"
 * @param minValue Minimum value
 * @param maxValue Maximum value
 */
QIVISION_API void SelectShape(const std::vector<QRegion>& regions,
                 std::vector<QRegion>& selected,
                 const std::string& features,
                 const std::string& operation,
                 double minValue,
                 double maxValue);

/**
 * @brief Select regions by area
 *
 * @param regions Input regions
 * @param[out] selected Output selected regions
 * @param minArea Minimum area
 * @param maxArea Maximum area
 */
QIVISION_API void SelectShapeArea(const std::vector<QRegion>& regions,
                     std::vector<QRegion>& selected,
                     int64_t minArea,
                     int64_t maxArea);

/**
 * @brief Select regions by circularity
 */
QIVISION_API void SelectShapeCircularity(const std::vector<QRegion>& regions,
                            std::vector<QRegion>& selected,
                            double minCirc,
                            double maxCirc);

/**
 * @brief Select regions by rectangularity
 */
QIVISION_API void SelectShapeRectangularity(const std::vector<QRegion>& regions,
                               std::vector<QRegion>& selected,
                               double minRect,
                               double maxRect);

/**
 * @brief Select regions by standard deviation of shape features
 *
 * Equivalent to Halcon's select_shape_std operator.
 * Selects regions whose feature value is within N standard deviations of the mean.
 *
 * @param regions Input regions
 * @param[out] selected Output selected regions
 * @param feature Feature to analyze
 * @param deviationFactor Number of standard deviations (e.g., 1.0, 2.0)
 */
QIVISION_API void SelectShapeStd(const std::vector<QRegion>& regions,
                    std::vector<QRegion>& selected,
                    ShapeFeature feature,
                    double deviationFactor);

/**
 * @brief Select regions by multiple features
 *
 * @param regions Input regions
 * @param[out] selected Output selected regions
 * @param features Vector of features
 * @param operation "and" = all must match, "or" = any must match
 * @param minValues Minimum values for each feature
 * @param maxValues Maximum values for each feature
 */
QIVISION_API void SelectShapeMulti(const std::vector<QRegion>& regions,
                      std::vector<QRegion>& selected,
                      const std::vector<ShapeFeature>& features,
                      SelectOperation operation,
                      const std::vector<double>& minValues,
                      const std::vector<double>& maxValues);

/**
 * @brief Select regions by convexity
 */
QIVISION_API void SelectShapeConvexity(const std::vector<QRegion>& regions,
                          std::vector<QRegion>& selected,
                          double minConvex,
                          double maxConvex);

/**
 * @brief Select regions by elongation
 */
QIVISION_API void SelectShapeElongation(const std::vector<QRegion>& regions,
                           std::vector<QRegion>& selected,
                           double minElong,
                           double maxElong);

/**
 * @brief Select the N largest/smallest regions by area
 *
 * @param regions Input regions
 * @param[out] selected Output selected regions
 * @param n Number of regions to select
 * @param largest If true, select largest; if false, select smallest
 */
QIVISION_API void SelectShapeProto(const std::vector<QRegion>& regions,
                      std::vector<QRegion>& selected,
                      int32_t n,
                      bool largest = true);

// =============================================================================
// Region Sorting
// =============================================================================

/**
 * @brief Sort regions by specified criteria
 *
 * Equivalent to Halcon's sort_region operator.
 *
 * @param regions Input regions
 * @param[out] sorted Output sorted regions
 * @param mode Sort mode
 * @param ascending true = ascending, false = descending
 */
QIVISION_API void SortRegion(const std::vector<QRegion>& regions,
                std::vector<QRegion>& sorted,
                SortMode mode,
                bool ascending = true);

/**
 * @brief Sort regions by string mode (Halcon compatible)
 */
QIVISION_API void SortRegion(const std::vector<QRegion>& regions,
                std::vector<QRegion>& sorted,
                const std::string& sortMode,
                const std::string& order,
                const std::string& rowOrCol);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Get single feature value for a region
 *
 * @param region Input region
 * @param feature Feature to compute
 * @return Feature value
 */
QIVISION_API double GetRegionFeature(const QRegion& region, ShapeFeature feature);

/**
 * @brief Get feature values for multiple regions
 *
 * @param regions Input regions
 * @param feature Feature to compute
 * @return Vector of feature values
 */
QIVISION_API std::vector<double> GetRegionFeatures(const std::vector<QRegion>& regions,
                                       ShapeFeature feature);

/**
 * @brief Parse feature name string to enum
 */
QIVISION_API ShapeFeature ParseShapeFeature(const std::string& name);

/**
 * @brief Get feature name string
 */
QIVISION_API std::string GetShapeFeatureName(ShapeFeature feature);

} // namespace Qi::Vision::Blob
