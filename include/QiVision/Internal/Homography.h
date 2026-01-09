#pragma once

/**
 * @file Homography.h
 * @brief Homography (projective transformation) operations
 *
 * This module provides:
 * - Homography estimation from point correspondences
 * - Perspective image warping
 * - Homography decomposition and analysis
 * - Utility functions for projective geometry
 *
 * Reference Halcon operators:
 * - vector_to_proj_hom_mat2d, proj_match_points_ransac
 * - projective_trans_image, projective_trans_point_2d
 * - hom_mat2d_to_affine_par, proj_hom_mat2d_compose
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QMatrix.h>
#include <QiVision/Core/QContour.h>
#include <QiVision/Internal/Matrix.h>
#include <QiVision/Internal/Interpolate.h>

#include <vector>
#include <optional>
#include <array>

namespace Qi::Vision::Internal {

// =============================================================================
// Homography Matrix Type
// =============================================================================

/**
 * @brief 3x3 Homography matrix for projective transformations
 *
 * Represents the transformation:
 *   [x']   [h00 h01 h02] [x]
 *   [y'] = [h10 h11 h12] [y]
 *   [w']   [h20 h21 h22] [1]
 *
 * Homogeneous coordinates: x_dst = x'/w', y_dst = y'/w'
 */
class Homography {
public:
    /// Default constructor (identity)
    Homography();

    /// Construct from 3x3 matrix elements (row-major)
    Homography(double h00, double h01, double h02,
               double h10, double h11, double h12,
               double h20, double h21, double h22);

    /// Construct from Mat33
    explicit Homography(const Mat33& mat);

    /// Construct from array (row-major)
    explicit Homography(const double* data);

    // =========================================================================
    // Element Access
    // =========================================================================

    /// Access element at (row, col)
    double& operator()(int row, int col) { return data_[row * 3 + col]; }
    double operator()(int row, int col) const { return data_[row * 3 + col]; }

    /// Get raw data pointer
    double* Data() { return data_; }
    const double* Data() const { return data_; }

    /// Convert to Mat33
    Mat33 ToMat33() const;

    // =========================================================================
    // Static Constructors
    // =========================================================================

    /// Identity homography
    static Homography Identity();

    /// From affine transform (embeds QMatrix into homography)
    static Homography FromAffine(const QMatrix& affine);

    /// From 4 point correspondences (exact solution)
    static std::optional<Homography> From4Points(
        const std::array<Point2d, 4>& srcPoints,
        const std::array<Point2d, 4>& dstPoints);

    // =========================================================================
    // Point Transformation
    // =========================================================================

    /// Transform a single point
    Point2d Transform(const Point2d& p) const;
    Point2d Transform(double x, double y) const;

    /// Transform multiple points
    std::vector<Point2d> Transform(const std::vector<Point2d>& points) const;

    // =========================================================================
    // Matrix Operations
    // =========================================================================

    /// Inverse homography
    Homography Inverse() const;

    /// Check if invertible
    bool IsInvertible(double tolerance = 1e-10) const;

    /// Determinant
    double Determinant() const;

    /// Normalize (make h22 = 1 if nonzero)
    Homography Normalized() const;

    /// Compose with another homography: result = this * other
    Homography operator*(const Homography& other) const;

    // =========================================================================
    // Type Checking
    // =========================================================================

    /// Check if this is actually an affine transform (h20 = h21 = 0)
    bool IsAffine(double tolerance = 1e-10) const;

    /// Convert to QMatrix if affine (returns nullopt if not affine)
    std::optional<QMatrix> ToAffine(double tolerance = 1e-10) const;

private:
    double data_[9];  // Row-major storage
};

// =============================================================================
// Homography Estimation
// =============================================================================

/**
 * @brief Estimate homography from point correspondences using DLT
 *
 * Uses Direct Linear Transform with normalization.
 * Requires at least 4 point pairs (more for overdetermined solution).
 *
 * @param srcPoints Source points
 * @param dstPoints Destination points
 * @return Estimated homography, or nullopt if failed
 */
std::optional<Homography> EstimateHomography(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints);

/**
 * @brief Estimate homography with RANSAC
 *
 * Robust to outliers. Requires at least 4 point pairs.
 *
 * @param srcPoints Source points
 * @param dstPoints Destination points
 * @param threshold Distance threshold for inlier classification (in pixels)
 * @param confidence Desired confidence level (0-1)
 * @param maxIterations Maximum RANSAC iterations
 * @param[out] inlierMask Optional mask indicating inliers
 * @return Estimated homography, or nullopt if failed
 */
std::optional<Homography> EstimateHomographyRANSAC(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints,
    double threshold = 3.0,
    double confidence = 0.99,
    int32_t maxIterations = 2000,
    std::vector<bool>* inlierMask = nullptr);

/**
 * @brief Compute homography error (symmetric transfer error)
 *
 * @param srcPoints Source points
 * @param dstPoints Destination points
 * @param H Homography matrix
 * @return RMS symmetric transfer error
 */
double ComputeHomographyError(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints,
    const Homography& H);

/**
 * @brief Compute per-point homography errors
 *
 * @param srcPoints Source points
 * @param dstPoints Destination points
 * @param H Homography matrix
 * @return Vector of per-point errors (forward transfer error)
 */
std::vector<double> ComputePointHomographyErrors(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints,
    const Homography& H);

// =============================================================================
// Image Warping
// =============================================================================

/**
 * @brief Apply perspective transformation to an image
 *
 * @param src Source image
 * @param H Homography matrix
 * @param dstWidth Output width (0 = auto-calculate)
 * @param dstHeight Output height (0 = auto-calculate)
 * @param method Interpolation method
 * @param borderMode Border handling mode
 * @param borderValue Border value for Constant mode
 * @return Transformed image
 */
QImage WarpPerspective(const QImage& src,
                       const Homography& H,
                       int32_t dstWidth = 0,
                       int32_t dstHeight = 0,
                       InterpolationMethod method = InterpolationMethod::Bilinear,
                       BorderMode borderMode = BorderMode::Constant,
                       double borderValue = 0.0);

/**
 * @brief Compute output size for perspective transform
 *
 * @param srcWidth Source width
 * @param srcHeight Source height
 * @param H Homography matrix
 * @param[out] dstWidth Output width
 * @param[out] dstHeight Output height
 * @param[out] offsetX X offset to keep all content visible
 * @param[out] offsetY Y offset to keep all content visible
 */
void ComputePerspectiveOutputSize(int32_t srcWidth, int32_t srcHeight,
                                   const Homography& H,
                                   int32_t& dstWidth, int32_t& dstHeight,
                                   double& offsetX, double& offsetY);

// =============================================================================
// Contour Transformation
// =============================================================================

/**
 * @brief Apply perspective transformation to a contour
 *
 * @param contour Source contour
 * @param H Homography matrix
 * @return Transformed contour
 */
QContour PerspectiveTransformContour(const QContour& contour, const Homography& H);

/**
 * @brief Apply perspective transformation to multiple contours
 *
 * @param contours Source contours
 * @param H Homography matrix
 * @return Transformed contours
 */
std::vector<QContour> PerspectiveTransformContours(
    const std::vector<QContour>& contours,
    const Homography& H);

// =============================================================================
// Homography Decomposition
// =============================================================================

/**
 * @brief Decomposition result for homography induced by plane
 */
struct HomographyDecomposition {
    Mat33 R;           ///< Rotation matrix
    Vec3 t;            ///< Translation vector (normalized)
    Vec3 n;            ///< Plane normal (in first camera frame)
    double d;          ///< Distance to plane (normalized, always positive)
    bool valid;        ///< Whether decomposition is valid
};

/**
 * @brief Decompose homography into rotation, translation, and plane normal
 *
 * Decomposes H = R + (1/d) * t * n^T
 * where R is rotation, t is translation, n is plane normal, d is distance.
 *
 * Note: This decomposition assumes a calibrated camera (H is between
 * normalized image coordinates). For pixel coordinates, apply K^-1 * H * K.
 *
 * @param H Homography matrix
 * @return Up to 4 possible decompositions (due to sign ambiguity)
 */
std::vector<HomographyDecomposition> DecomposeHomography(const Homography& H);

/**
 * @brief Filter decompositions by positive depth constraint
 *
 * @param decompositions All decompositions from DecomposeHomography
 * @param srcPoints Source points visible in first view
 * @return Decompositions where all points have positive depth in both views
 */
std::vector<HomographyDecomposition> FilterDecompositionsByVisibility(
    const std::vector<HomographyDecomposition>& decompositions,
    const std::vector<Point2d>& srcPoints);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Create homography that maps a quadrilateral to a rectangle
 *
 * @param quad Four corners of quadrilateral (in order: TL, TR, BR, BL)
 * @param width Target rectangle width
 * @param height Target rectangle height
 * @return Homography that rectifies the quadrilateral
 */
std::optional<Homography> RectifyQuadrilateral(
    const std::array<Point2d, 4>& quad,
    double width,
    double height);

/**
 * @brief Create homography that maps a rectangle to a quadrilateral
 *
 * @param width Source rectangle width
 * @param height Source rectangle height
 * @param quad Target quadrilateral corners (TL, TR, BR, BL)
 * @return Homography for the mapping
 */
std::optional<Homography> RectangleToQuadrilateral(
    double width,
    double height,
    const std::array<Point2d, 4>& quad);

/**
 * @brief Compute bounding box after perspective transformation
 *
 * @param bbox Input bounding box
 * @param H Homography matrix
 * @return Transformed bounding box (axis-aligned)
 */
Rect2d TransformBoundingBoxPerspective(const Rect2d& bbox, const Homography& H);

/**
 * @brief Check if a homography produces a valid (non-degenerate) mapping
 *
 * Checks for:
 * - Invertibility
 * - No sign reversal (all corners map to same side)
 * - Reasonable aspect ratio preservation
 *
 * @param H Homography matrix
 * @param srcWidth Source image width
 * @param srcHeight Source image height
 * @return true if the mapping is valid
 */
bool IsValidHomography(const Homography& H, int32_t srcWidth, int32_t srcHeight);

/**
 * @brief Compute the Sampson error for homography
 *
 * More accurate than algebraic error, used for refinement.
 *
 * @param src Source point
 * @param dst Destination point
 * @param H Homography matrix
 * @return Sampson error
 */
double SampsonError(const Point2d& src, const Point2d& dst, const Homography& H);

/**
 * @brief Refine homography using Levenberg-Marquardt
 *
 * @param srcPoints Source points
 * @param dstPoints Destination points
 * @param H Initial homography estimate
 * @param maxIterations Maximum iterations
 * @return Refined homography
 */
Homography RefineHomographyLM(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints,
    const Homography& H,
    int32_t maxIterations = 10);

} // namespace Qi::Vision::Internal
