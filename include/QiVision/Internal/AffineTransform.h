#pragma once

/**
 * @file AffineTransform.h
 * @brief Affine transformation operations for images, regions, and points
 *
 * This module provides:
 * - Image warping (affine, rotate, scale, crop)
 * - Transform estimation from point correspondences
 * - Region transformation
 * - Transform analysis and decomposition
 *
 * Reference Halcon operators:
 * - affine_trans_image, rotate_image, zoom_image_size
 * - affine_trans_region, affine_trans_contour_xld
 * - vector_to_rigid, vector_to_similarity, vector_to_hom_mat2d
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QMatrix.h>
#include <QiVision/Core/QRegion.h>
#include <QiVision/Core/QContour.h>
#include <QiVision/Internal/Interpolate.h>

#include <vector>
#include <optional>

namespace Qi::Vision::Internal {

// =============================================================================
// Image Warping
// =============================================================================

/**
 * @brief Apply affine transformation to an image
 *
 * @param src Source image
 * @param matrix Affine transformation matrix
 * @param dstWidth Output width (0 = auto-calculate to fit)
 * @param dstHeight Output height (0 = auto-calculate to fit)
 * @param method Interpolation method
 * @param borderMode Border handling mode
 * @param borderValue Border value for Constant mode
 * @return Transformed image
 */
QImage WarpAffine(const QImage& src,
                  const QMatrix& matrix,
                  int32_t dstWidth = 0,
                  int32_t dstHeight = 0,
                  InterpolationMethod method = InterpolationMethod::Bilinear,
                  BorderMode borderMode = BorderMode::Constant,
                  double borderValue = 0.0);

/**
 * @brief Rotate image around its center
 *
 * @param src Source image
 * @param angle Rotation angle in radians
 * @param resize If true, resize output to fit entire rotated image
 * @param method Interpolation method
 * @param borderMode Border handling mode
 * @param borderValue Border value for Constant mode
 * @return Rotated image
 */
QImage RotateImage(const QImage& src,
                   double angle,
                   bool resize = true,
                   InterpolationMethod method = InterpolationMethod::Bilinear,
                   BorderMode borderMode = BorderMode::Constant,
                   double borderValue = 0.0);

/**
 * @brief Rotate image around a specified center
 *
 * @param src Source image
 * @param angle Rotation angle in radians
 * @param centerX Rotation center X
 * @param centerY Rotation center Y
 * @param resize If true, resize output to fit entire rotated image
 * @param method Interpolation method
 * @param borderMode Border handling mode
 * @param borderValue Border value for Constant mode
 * @return Rotated image
 */
QImage RotateImage(const QImage& src,
                   double angle,
                   double centerX,
                   double centerY,
                   bool resize = true,
                   InterpolationMethod method = InterpolationMethod::Bilinear,
                   BorderMode borderMode = BorderMode::Constant,
                   double borderValue = 0.0);

/**
 * @brief Scale image to specified size
 *
 * @param src Source image
 * @param dstWidth Output width
 * @param dstHeight Output height
 * @param method Interpolation method
 * @return Scaled image
 */
QImage ScaleImage(const QImage& src,
                  int32_t dstWidth,
                  int32_t dstHeight,
                  InterpolationMethod method = InterpolationMethod::Bilinear);

/**
 * @brief Scale image by factors
 *
 * @param src Source image
 * @param scaleX Horizontal scale factor
 * @param scaleY Vertical scale factor
 * @param method Interpolation method
 * @return Scaled image
 */
QImage ScaleImageFactor(const QImage& src,
                        double scaleX,
                        double scaleY,
                        InterpolationMethod method = InterpolationMethod::Bilinear);

/**
 * @brief Extract a rotated rectangular region from image
 *
 * @param src Source image
 * @param rect Rotated rectangle to extract
 * @param method Interpolation method
 * @return Axis-aligned image containing the rotated rectangle content
 */
QImage CropRotatedRect(const QImage& src,
                       const RotatedRect2d& rect,
                       InterpolationMethod method = InterpolationMethod::Bilinear);

/**
 * @brief Compute output size for affine transform
 *
 * @param srcWidth Source width
 * @param srcHeight Source height
 * @param matrix Transformation matrix
 * @param[out] dstWidth Output width
 * @param[out] dstHeight Output height
 * @param[out] offsetX X offset to keep all content visible
 * @param[out] offsetY Y offset to keep all content visible
 */
void ComputeAffineOutputSize(int32_t srcWidth, int32_t srcHeight,
                             const QMatrix& matrix,
                             int32_t& dstWidth, int32_t& dstHeight,
                             double& offsetX, double& offsetY);

// =============================================================================
// Transform Estimation
// =============================================================================

/**
 * @brief Estimate affine transform from point correspondences (least squares)
 *
 * Requires at least 3 point pairs. Uses SVD for robust solution.
 *
 * @param srcPoints Source points
 * @param dstPoints Destination points
 * @return Estimated transformation matrix, or nullopt if failed
 */
std::optional<QMatrix> EstimateAffine(const std::vector<Point2d>& srcPoints,
                                       const std::vector<Point2d>& dstPoints);

/**
 * @brief Estimate affine transform with RANSAC
 *
 * Robust to outliers. Requires at least 3 point pairs.
 *
 * @param srcPoints Source points
 * @param dstPoints Destination points
 * @param threshold Distance threshold for inlier classification
 * @param confidence Desired confidence level (0-1)
 * @param maxIterations Maximum RANSAC iterations
 * @param[out] inlierMask Optional mask indicating inliers
 * @return Estimated transformation matrix, or nullopt if failed
 */
std::optional<QMatrix> EstimateAffineRANSAC(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints,
    double threshold = 3.0,
    double confidence = 0.99,
    int32_t maxIterations = 1000,
    std::vector<bool>* inlierMask = nullptr);

/**
 * @brief Estimate rigid transform (rotation + translation only)
 *
 * Uses Procrustes analysis. Requires at least 2 point pairs.
 *
 * @param srcPoints Source points
 * @param dstPoints Destination points
 * @return Rigid transformation matrix, or nullopt if failed
 */
std::optional<QMatrix> EstimateRigid(const std::vector<Point2d>& srcPoints,
                                      const std::vector<Point2d>& dstPoints);

/**
 * @brief Estimate rigid transform with RANSAC
 *
 * @param srcPoints Source points
 * @param dstPoints Destination points
 * @param threshold Distance threshold for inlier classification
 * @param confidence Desired confidence level (0-1)
 * @param maxIterations Maximum RANSAC iterations
 * @param[out] inlierMask Optional mask indicating inliers
 * @return Rigid transformation matrix, or nullopt if failed
 */
std::optional<QMatrix> EstimateRigidRANSAC(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints,
    double threshold = 3.0,
    double confidence = 0.99,
    int32_t maxIterations = 1000,
    std::vector<bool>* inlierMask = nullptr);

/**
 * @brief Estimate similarity transform (rotation + translation + uniform scale)
 *
 * Requires at least 2 point pairs.
 *
 * @param srcPoints Source points
 * @param dstPoints Destination points
 * @return Similarity transformation matrix, or nullopt if failed
 */
std::optional<QMatrix> EstimateSimilarity(const std::vector<Point2d>& srcPoints,
                                           const std::vector<Point2d>& dstPoints);

/**
 * @brief Estimate similarity transform with RANSAC
 *
 * @param srcPoints Source points
 * @param dstPoints Destination points
 * @param threshold Distance threshold for inlier classification
 * @param confidence Desired confidence level (0-1)
 * @param maxIterations Maximum RANSAC iterations
 * @param[out] inlierMask Optional mask indicating inliers
 * @return Similarity transformation matrix, or nullopt if failed
 */
std::optional<QMatrix> EstimateSimilarityRANSAC(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints,
    double threshold = 3.0,
    double confidence = 0.99,
    int32_t maxIterations = 1000,
    std::vector<bool>* inlierMask = nullptr);

/**
 * @brief Compute transformation error (RMS of distances)
 *
 * @param srcPoints Source points
 * @param dstPoints Destination points
 * @param matrix Transformation matrix
 * @return RMS error
 */
double ComputeTransformError(const std::vector<Point2d>& srcPoints,
                              const std::vector<Point2d>& dstPoints,
                              const QMatrix& matrix);

/**
 * @brief Compute per-point transformation errors
 *
 * @param srcPoints Source points
 * @param dstPoints Destination points
 * @param matrix Transformation matrix
 * @return Vector of per-point errors
 */
std::vector<double> ComputePointErrors(const std::vector<Point2d>& srcPoints,
                                        const std::vector<Point2d>& dstPoints,
                                        const QMatrix& matrix);

// =============================================================================
// Region Transformation
// =============================================================================

/**
 * @brief Apply affine transformation to a region
 *
 * @param region Source region
 * @param matrix Transformation matrix
 * @return Transformed region
 */
QRegion AffineTransformRegion(const QRegion& region, const QMatrix& matrix);

/**
 * @brief Apply affine transformation to multiple regions
 *
 * @param regions Source regions
 * @param matrix Transformation matrix
 * @return Transformed regions
 */
std::vector<QRegion> AffineTransformRegions(const std::vector<QRegion>& regions,
                                             const QMatrix& matrix);

// =============================================================================
// Contour Transformation (uses QContour::Transform internally)
// =============================================================================

/**
 * @brief Apply affine transformation to a contour
 *
 * @param contour Source contour
 * @param matrix Transformation matrix
 * @return Transformed contour
 */
QContour AffineTransformContour(const QContour& contour, const QMatrix& matrix);

/**
 * @brief Apply affine transformation to multiple contours
 *
 * @param contours Source contours
 * @param matrix Transformation matrix
 * @return Transformed contours
 */
std::vector<QContour> AffineTransformContours(const std::vector<QContour>& contours,
                                               const QMatrix& matrix);

// =============================================================================
// Transform Analysis
// =============================================================================

/**
 * @brief Decompose affine transform into components
 *
 * Decomposes into: Translation * Rotation * Scale * Shear
 *
 * @param matrix Input transformation matrix
 * @param[out] tx Translation X
 * @param[out] ty Translation Y
 * @param[out] angle Rotation angle (radians)
 * @param[out] scaleX Scale X
 * @param[out] scaleY Scale Y
 * @param[out] shear Shear factor
 * @return true if decomposition succeeded
 */
bool DecomposeAffine(const QMatrix& matrix,
                     double& tx, double& ty,
                     double& angle,
                     double& scaleX, double& scaleY,
                     double& shear);

/**
 * @brief Check if matrix represents a rigid transform
 *
 * A rigid transform preserves distances (rotation + translation only).
 *
 * @param matrix Input transformation matrix
 * @param tolerance Numerical tolerance
 * @return true if rigid
 */
bool IsRigidTransform(const QMatrix& matrix, double tolerance = 1e-6);

/**
 * @brief Check if matrix represents a similarity transform
 *
 * A similarity transform preserves angles (rotation + translation + uniform scale).
 *
 * @param matrix Input transformation matrix
 * @param tolerance Numerical tolerance
 * @return true if similarity
 */
bool IsSimilarityTransform(const QMatrix& matrix, double tolerance = 1e-6);

/**
 * @brief Interpolate between two transforms
 *
 * @param m1 First transform
 * @param m2 Second transform
 * @param t Interpolation parameter (0=m1, 1=m2)
 * @return Interpolated transform
 */
QMatrix InterpolateTransform(const QMatrix& m1, const QMatrix& m2, double t);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Create matrix to map from source rect to destination rect
 *
 * @param srcRect Source rectangle
 * @param dstRect Destination rectangle
 * @return Transformation matrix
 */
QMatrix RectToRectTransform(const Rect2d& srcRect, const Rect2d& dstRect);

/**
 * @brief Create matrix to map rotated rect to axis-aligned rect
 *
 * @param rotRect Rotated rectangle
 * @return Transformation matrix that maps rotRect to axis-aligned [0,w] x [0,h]
 */
QMatrix RotatedRectToAxisAligned(const RotatedRect2d& rotRect);

/**
 * @brief Compute bounding box after transformation
 *
 * @param bbox Input bounding box
 * @param matrix Transformation matrix
 * @return Transformed bounding box (axis-aligned)
 */
Rect2d TransformBoundingBox(const Rect2d& bbox, const QMatrix& matrix);

/**
 * @brief Compute bounding box of transformed points
 *
 * @param points Input points
 * @param matrix Transformation matrix
 * @return Bounding box of transformed points
 */
Rect2d TransformPointsBoundingBox(const std::vector<Point2d>& points,
                                   const QMatrix& matrix);

} // namespace Qi::Vision::Internal
