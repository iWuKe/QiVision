#pragma once

/**
 * @file Hessian.h
 * @brief Hessian matrix computation for ridge/valley detection
 *
 * The Hessian matrix is the matrix of second-order partial derivatives:
 * H = [Gxx  Gxy]
 *     [Gxy  Gyy]
 *
 * Eigenvalue analysis:
 * - Ridge (bright line): λ1 < 0, |λ1| >> |λ2|
 * - Valley (dark line):  λ1 > 0, |λ1| >> |λ2|
 * - Blob:               |λ1| ≈ |λ2|, same sign
 * - Saddle:             opposite signs
 *
 * Used by:
 * - Steger subpixel edge detection
 * - Blob detection (LoG, DoG)
 * - Feature point detection (Hessian affine)
 * - Ridge/valley enhancement
 *
 * References:
 * - Steger, "An Unbiased Detector of Curvilinear Structures" (1998)
 * - Lindeberg, "Edge detection and ridge detection with automatic scale selection"
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Internal/Interpolate.h>

#include <cstdint>
#include <cmath>
#include <vector>

namespace Qi::Vision::Internal {

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief Result of Hessian computation at a single point
 */
struct HessianResult {
    // Second-order partial derivatives
    double dxx = 0.0;    ///< ∂²I/∂x²
    double dxy = 0.0;    ///< ∂²I/∂x∂y
    double dyy = 0.0;    ///< ∂²I/∂y²

    // Eigenvalues (sorted by absolute value: |λ1| >= |λ2|)
    double lambda1 = 0.0;    ///< Principal eigenvalue (larger |λ|)
    double lambda2 = 0.0;    ///< Secondary eigenvalue (smaller |λ|)

    // Eigenvectors
    double nx = 0.0, ny = 0.0;    ///< Principal direction (normal to ridge/valley)
    double tx = 0.0, ty = 0.0;    ///< Tangent direction (along ridge/valley)

    /**
     * @brief Check if this is a ridge point (bright line on dark background)
     * Ridge: λ1 < 0 (concave in principal direction)
     */
    bool IsRidge() const { return lambda1 < 0; }

    /**
     * @brief Check if this is a valley point (dark line on bright background)
     * Valley: λ1 > 0 (convex in principal direction)
     */
    bool IsValley() const { return lambda1 > 0; }

    /**
     * @brief Get the response strength (|λ1|)
     * Higher value = stronger ridge/valley
     */
    double Response() const { return std::abs(lambda1); }

    /**
     * @brief Get the anisotropy ratio (|λ1| / |λ2|)
     * Higher value = more line-like (less blob-like)
     */
    double Anisotropy() const {
        if (std::abs(lambda2) < 1e-10) return 1e10;
        return std::abs(lambda1) / std::abs(lambda2);
    }
};

/**
 * @brief Parameters for Hessian computation
 */
struct HessianParams {
    double sigma = 1.0;           ///< Gaussian sigma for smoothing
    BorderMode border = BorderMode::Reflect101;  ///< Border handling
};

// ============================================================================
// 2x2 Eigenvalue Decomposition
// ============================================================================

/**
 * @brief Eigenvalue decomposition for 2x2 symmetric matrix
 *
 * Given matrix A = [a  b]
 *                  [b  c]
 *
 * Computes eigenvalues and eigenvectors:
 * - λ1, λ2 where |λ1| >= |λ2|
 * - (nx, ny) eigenvector for λ1 (normalized)
 * - (tx, ty) eigenvector for λ2 (normalized, perpendicular to (nx,ny))
 *
 * @param a Top-left element (Gxx)
 * @param b Off-diagonal element (Gxy)
 * @param c Bottom-right element (Gyy)
 * @param[out] lambda1 Principal eigenvalue (|λ1| >= |λ2|)
 * @param[out] lambda2 Secondary eigenvalue
 * @param[out] nx Principal eigenvector x-component
 * @param[out] ny Principal eigenvector y-component
 */
void EigenDecompose2x2(double a, double b, double c,
                        double& lambda1, double& lambda2,
                        double& nx, double& ny);

/**
 * @brief Compute both eigenvectors for 2x2 symmetric matrix
 *
 * @param a Top-left element (Gxx)
 * @param b Off-diagonal element (Gxy)
 * @param c Bottom-right element (Gyy)
 * @param[out] lambda1 Principal eigenvalue (|λ1| >= |λ2|)
 * @param[out] lambda2 Secondary eigenvalue
 * @param[out] nx Principal eigenvector x-component
 * @param[out] ny Principal eigenvector y-component
 * @param[out] tx Tangent eigenvector x-component
 * @param[out] ty Tangent eigenvector y-component
 */
void EigenDecompose2x2Full(double a, double b, double c,
                            double& lambda1, double& lambda2,
                            double& nx, double& ny,
                            double& tx, double& ty);

// ============================================================================
// Single Point Hessian
// ============================================================================

/**
 * @brief Compute Hessian at a single pixel location
 *
 * Uses Gaussian second derivatives for robust computation.
 *
 * @param image Input image (uint8_t, uint16_t, or float)
 * @param x X coordinate
 * @param y Y coordinate
 * @param sigma Gaussian sigma for smoothing
 * @param border Border handling mode
 * @return HessianResult with derivatives, eigenvalues, and eigenvectors
 */
template<typename T>
HessianResult ComputeHessianAt(const T* data, int32_t width, int32_t height,
                                int32_t x, int32_t y, double sigma,
                                BorderMode border = BorderMode::Reflect101);

/**
 * @brief Compute Hessian at a subpixel location
 *
 * Uses bicubic interpolation for subpixel accuracy.
 *
 * @param data Image data pointer
 * @param width Image width
 * @param height Image height
 * @param x Subpixel x coordinate
 * @param y Subpixel y coordinate
 * @param sigma Gaussian sigma
 * @param border Border handling mode
 * @return HessianResult at subpixel location
 */
template<typename T>
HessianResult ComputeHessianAtSubpixel(const T* data, int32_t width, int32_t height,
                                        double x, double y, double sigma,
                                        BorderMode border = BorderMode::Reflect101);

// ============================================================================
// Whole Image Hessian
// ============================================================================

/**
 * @brief Compute Hessian components for entire image
 *
 * Efficiently computes all second derivatives using separable convolution.
 * Output images are float32 regardless of input type.
 *
 * @param data Input image data
 * @param width Image width
 * @param height Image height
 * @param sigma Gaussian sigma
 * @param[out] dxx Second derivative in x (float image, same size)
 * @param[out] dxy Mixed partial derivative (float image, same size)
 * @param[out] dyy Second derivative in y (float image, same size)
 * @param border Border handling mode
 */
template<typename T>
void ComputeHessianImage(const T* data, int32_t width, int32_t height,
                          double sigma,
                          std::vector<float>& dxx,
                          std::vector<float>& dxy,
                          std::vector<float>& dyy,
                          BorderMode border = BorderMode::Reflect101);

/**
 * @brief Compute Hessian components using QImage
 *
 * @param image Input image
 * @param sigma Gaussian sigma
 * @param[out] dxxImage Second derivative in x
 * @param[out] dxyImage Mixed partial derivative
 * @param[out] dyyImage Second derivative in y
 * @param border Border handling mode
 */
void ComputeHessianImage(const QImage& image, double sigma,
                          QImage& dxxImage, QImage& dxyImage, QImage& dyyImage,
                          BorderMode border = BorderMode::Reflect101);

// ============================================================================
// Eigenvalue Images
// ============================================================================

/**
 * @brief Compute eigenvalue images from Hessian components
 *
 * @param dxx Second derivative xx image
 * @param dxy Mixed derivative xy image
 * @param dyy Second derivative yy image
 * @param width Image width
 * @param height Image height
 * @param[out] lambda1 Principal eigenvalue image
 * @param[out] lambda2 Secondary eigenvalue image
 */
void ComputeEigenvalueImages(const float* dxx, const float* dxy, const float* dyy,
                              int32_t width, int32_t height,
                              std::vector<float>& lambda1,
                              std::vector<float>& lambda2);

/**
 * @brief Compute eigenvector images from Hessian components
 *
 * @param dxx Second derivative xx image
 * @param dxy Mixed derivative xy image
 * @param dyy Second derivative yy image
 * @param width Image width
 * @param height Image height
 * @param[out] nx Principal eigenvector x-component
 * @param[out] ny Principal eigenvector y-component
 */
void ComputeEigenvectorImages(const float* dxx, const float* dxy, const float* dyy,
                               int32_t width, int32_t height,
                               std::vector<float>& nx,
                               std::vector<float>& ny);

// ============================================================================
// Ridge/Valley Response
// ============================================================================

/**
 * @brief Compute ridge response image
 *
 * Ridge response = max(0, -λ1) where λ1 < 0 indicates ridge
 *
 * @param lambda1 Principal eigenvalue image
 * @param width Image width
 * @param height Image height
 * @param[out] response Ridge response image
 */
void ComputeRidgeResponse(const float* lambda1, int32_t width, int32_t height,
                           std::vector<float>& response);

/**
 * @brief Compute valley response image
 *
 * Valley response = max(0, λ1) where λ1 > 0 indicates valley
 *
 * @param lambda1 Principal eigenvalue image
 * @param width Image width
 * @param height Image height
 * @param[out] response Valley response image
 */
void ComputeValleyResponse(const float* lambda1, int32_t width, int32_t height,
                            std::vector<float>& response);

/**
 * @brief Compute combined ridge/valley response image
 *
 * Response = |λ1| regardless of sign
 *
 * @param lambda1 Principal eigenvalue image
 * @param width Image width
 * @param height Image height
 * @param[out] response Combined response image
 */
void ComputeLineResponse(const float* lambda1, int32_t width, int32_t height,
                          std::vector<float>& response);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Compute determinant of Hessian
 * det(H) = Gxx * Gyy - Gxy²
 *
 * @param dxx Second derivative xx
 * @param dxy Mixed derivative xy
 * @param dyy Second derivative yy
 * @return Determinant value
 */
inline double HessianDeterminant(double dxx, double dxy, double dyy) {
    return dxx * dyy - dxy * dxy;
}

/**
 * @brief Compute trace of Hessian (Laplacian)
 * trace(H) = Gxx + Gyy
 *
 * @param dxx Second derivative xx
 * @param dyy Second derivative yy
 * @return Trace value
 */
inline double HessianTrace(double dxx, double dyy) {
    return dxx + dyy;
}

/**
 * @brief Compute Frobenius norm of Hessian
 * ||H||_F = sqrt(Gxx² + 2*Gxy² + Gyy²)
 *
 * @param dxx Second derivative xx
 * @param dxy Mixed derivative xy
 * @param dyy Second derivative yy
 * @return Frobenius norm
 */
inline double HessianNorm(double dxx, double dxy, double dyy) {
    return std::sqrt(dxx * dxx + 2.0 * dxy * dxy + dyy * dyy);
}

/**
 * @brief Check if point is a local maximum of ridge response
 *
 * Non-maximum suppression along the principal eigenvector direction.
 *
 * @param lambda1 Principal eigenvalue image
 * @param nx Principal eigenvector x image
 * @param ny Principal eigenvector y image
 * @param width Image width
 * @param height Image height
 * @param x Pixel x coordinate
 * @param y Pixel y coordinate
 * @return true if local maximum, false otherwise
 */
bool IsRidgeMaximum(const float* lambda1, const float* nx, const float* ny,
                     int32_t width, int32_t height, int32_t x, int32_t y);

// ============================================================================
// Template Implementations
// ============================================================================

template<typename T>
HessianResult ComputeHessianAt(const T* data, int32_t width, int32_t height,
                                int32_t x, int32_t y, double sigma,
                                BorderMode border) {
    HessianResult result;

    // Compute kernel size
    int32_t ksize = static_cast<int32_t>(std::ceil(3.0 * sigma)) * 2 + 1;
    if (ksize < 3) ksize = 3;
    int32_t halfK = ksize / 2;

    double sigma2 = sigma * sigma;
    double sigma4 = sigma2 * sigma2;
    double twoSigma2 = 2.0 * sigma2;

    // First pass: compute kernel weights and their sums (for zero-sum normalization)
    // This ensures constant images produce zero second derivatives
    double sumWxx = 0.0, sumWyy = 0.0;
    int32_t numPixels = 0;

    for (int32_t ky = -halfK; ky <= halfK; ++ky) {
        for (int32_t kx = -halfK; kx <= halfK; ++kx) {
            double kx2 = static_cast<double>(kx * kx);
            double ky2 = static_cast<double>(ky * ky);
            double r2 = kx2 + ky2;
            double g = std::exp(-r2 / twoSigma2);

            sumWxx += (kx2 - sigma2) * g / sigma4;
            sumWyy += (ky2 - sigma2) * g / sigma4;
            ++numPixels;
        }
    }

    // Compute correction to make weights zero-sum
    double corrWxx = sumWxx / numPixels;
    double corrWyy = sumWyy / numPixels;

    // Second pass: compute weighted sums with zero-sum corrected weights
    double sumGxx = 0.0, sumGxy = 0.0, sumGyy = 0.0;

    for (int32_t ky = -halfK; ky <= halfK; ++ky) {
        for (int32_t kx = -halfK; kx <= halfK; ++kx) {
            // Get pixel value with border handling
            int32_t px = HandleBorder(x + kx, width, border);
            int32_t py = HandleBorder(y + ky, height, border);

            if (px < 0 || py < 0) continue;  // Constant border with 0

            double val = static_cast<double>(data[py * width + px]);

            // Gaussian weights and derivatives
            double kx2 = static_cast<double>(kx * kx);
            double ky2 = static_cast<double>(ky * ky);
            double r2 = kx2 + ky2;
            double g = std::exp(-r2 / twoSigma2);

            // Second derivative weights with zero-sum correction
            double wxx = (kx2 - sigma2) * g / sigma4 - corrWxx;
            double wxy = static_cast<double>(kx * ky) * g / sigma4;  // Already zero-sum
            double wyy = (ky2 - sigma2) * g / sigma4 - corrWyy;

            sumGxx += val * wxx;
            sumGxy += val * wxy;
            sumGyy += val * wyy;
        }
    }

    result.dxx = sumGxx;
    result.dxy = sumGxy;
    result.dyy = sumGyy;

    // Compute eigenvalues and eigenvectors
    EigenDecompose2x2Full(result.dxx, result.dxy, result.dyy,
                           result.lambda1, result.lambda2,
                           result.nx, result.ny,
                           result.tx, result.ty);

    return result;
}

template<typename T>
HessianResult ComputeHessianAtSubpixel(const T* data, int32_t width, int32_t height,
                                        double x, double y, double sigma,
                                        BorderMode border) {
    // For subpixel, we compute at the 4 nearest integer positions and interpolate
    int32_t x0 = static_cast<int32_t>(std::floor(x));
    int32_t y0 = static_cast<int32_t>(std::floor(y));
    int32_t x1 = x0 + 1;
    int32_t y1 = y0 + 1;

    double fx = x - x0;
    double fy = y - y0;

    // Clamp to valid range
    x0 = std::max(0, std::min(x0, width - 1));
    x1 = std::max(0, std::min(x1, width - 1));
    y0 = std::max(0, std::min(y0, height - 1));
    y1 = std::max(0, std::min(y1, height - 1));

    // Compute Hessian at 4 corners
    auto h00 = ComputeHessianAt(data, width, height, x0, y0, sigma, border);
    auto h10 = ComputeHessianAt(data, width, height, x1, y0, sigma, border);
    auto h01 = ComputeHessianAt(data, width, height, x0, y1, sigma, border);
    auto h11 = ComputeHessianAt(data, width, height, x1, y1, sigma, border);

    // Bilinear interpolation of derivatives
    HessianResult result;
    result.dxx = (1-fx)*(1-fy)*h00.dxx + fx*(1-fy)*h10.dxx + (1-fx)*fy*h01.dxx + fx*fy*h11.dxx;
    result.dxy = (1-fx)*(1-fy)*h00.dxy + fx*(1-fy)*h10.dxy + (1-fx)*fy*h01.dxy + fx*fy*h11.dxy;
    result.dyy = (1-fx)*(1-fy)*h00.dyy + fx*(1-fy)*h10.dyy + (1-fx)*fy*h01.dyy + fx*fy*h11.dyy;

    // Recompute eigenvalues/eigenvectors from interpolated derivatives
    EigenDecompose2x2Full(result.dxx, result.dxy, result.dyy,
                           result.lambda1, result.lambda2,
                           result.nx, result.ny,
                           result.tx, result.ty);

    return result;
}

template<typename T>
void ComputeHessianImage(const T* data, int32_t width, int32_t height,
                          double sigma,
                          std::vector<float>& dxx,
                          std::vector<float>& dxy,
                          std::vector<float>& dyy,
                          BorderMode border) {
    size_t size = static_cast<size_t>(width * height);
    dxx.resize(size);
    dxy.resize(size);
    dyy.resize(size);

    // For efficiency, use separable convolution with Gaussian derivative kernels
    // Gxx = G'' * G (x then y)
    // Gyy = G * G'' (x then y)
    // Gxy = G' * G' (x then y)

    // For now, compute point by point (can be optimized with separable kernels)
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            auto h = ComputeHessianAt(data, width, height, x, y, sigma, border);
            size_t idx = static_cast<size_t>(y * width + x);
            dxx[idx] = static_cast<float>(h.dxx);
            dxy[idx] = static_cast<float>(h.dxy);
            dyy[idx] = static_cast<float>(h.dyy);
        }
    }
}

} // namespace Qi::Vision::Internal
