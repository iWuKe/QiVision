#pragma once

/**
 * @file Gaussian.h
 * @brief Gaussian kernel generation for filtering and edge detection
 *
 * This module provides:
 * - 1D Gaussian kernels
 * - 1D Gaussian derivative kernels (1st, 2nd order)
 * - 2D Gaussian kernels
 * - Kernel size computation from sigma
 *
 * Used by:
 * - Filter module (Gaussian blur)
 * - Edge detection (gradient computation)
 * - Steger algorithm (ridge/valley detection)
 * - Scale-space analysis
 */

#include <QiVision/Core/Constants.h>

#include <vector>
#include <cstdint>
#include <cmath>

namespace Qi::Vision::Internal {

/**
 * @brief Gaussian kernel generator
 *
 * Provides static methods for generating various Gaussian-based kernels.
 * All kernels are returned as std::vector<double> for flexibility.
 */
class Gaussian {
public:
    // =========================================================================
    // Kernel Size Computation
    // =========================================================================

    /**
     * @brief Compute recommended kernel size for given sigma
     * @param sigma Standard deviation
     * @param cutoff Number of standard deviations to include (default: 3.0 for 99.7%)
     * @return Kernel size (always odd)
     *
     * Returns size = 2 * ceil(cutoff * sigma) + 1, minimum 3.
     */
    static int32_t ComputeKernelSize(double sigma, double cutoff = 3.0);

    /**
     * @brief Compute sigma from kernel size
     * @param kernelSize Kernel size (should be odd)
     * @param cutoff Number of standard deviations (default: 3.0)
     * @return Sigma value
     */
    static double ComputeSigma(int32_t kernelSize, double cutoff = 3.0);

    // =========================================================================
    // 1D Gaussian Kernels
    // =========================================================================

    /**
     * @brief Generate 1D Gaussian kernel (smoothing)
     * @param sigma Standard deviation
     * @param size Kernel size (0 = auto-compute from sigma)
     * @param normalize If true, kernel sums to 1.0
     * @return 1D kernel values
     *
     * G(x) = exp(-x² / (2σ²)) / (√(2π) * σ)
     */
    static std::vector<double> Kernel1D(double sigma, int32_t size = 0, bool normalize = true);

    /**
     * @brief Generate 1D first derivative of Gaussian
     * @param sigma Standard deviation
     * @param size Kernel size (0 = auto-compute)
     * @param normalize If true, normalize for gradient magnitude preservation
     * @return 1D derivative kernel
     *
     * G'(x) = -x * G(x) / σ²
     *
     * This kernel gives the gradient when convolved with the image.
     * For edge detection: convolve with Gaussian in one direction,
     * convolve with derivative in perpendicular direction.
     */
    static std::vector<double> Derivative1D(double sigma, int32_t size = 0, bool normalize = true);

    /**
     * @brief Generate 1D second derivative of Gaussian
     * @param sigma Standard deviation
     * @param size Kernel size (0 = auto-compute)
     * @param normalize If true, normalize
     * @return 1D second derivative kernel
     *
     * G''(x) = (x² - σ²) * G(x) / σ⁴
     *
     * Used for Laplacian of Gaussian (LoG) and ridge detection.
     */
    static std::vector<double> SecondDerivative1D(double sigma, int32_t size = 0, bool normalize = true);

    // =========================================================================
    // 2D Gaussian Kernels
    // =========================================================================

    /**
     * @brief Generate 2D Gaussian kernel
     * @param sigma Standard deviation (same for x and y)
     * @param size Kernel size (0 = auto-compute, size × size)
     * @param normalize If true, kernel sums to 1.0
     * @return 2D kernel as row-major vector (size × size)
     *
     * G(x,y) = exp(-(x² + y²) / (2σ²)) / (2π * σ²)
     */
    static std::vector<double> Kernel2D(double sigma, int32_t size = 0, bool normalize = true);

    /**
     * @brief Generate 2D Gaussian kernel (anisotropic)
     * @param sigmaX Standard deviation in X
     * @param sigmaY Standard deviation in Y
     * @param sizeX Kernel size in X (0 = auto)
     * @param sizeY Kernel size in Y (0 = auto)
     * @param normalize If true, kernel sums to 1.0
     * @return 2D kernel as row-major vector (sizeY × sizeX)
     */
    static std::vector<double> Kernel2DAnisotropic(
        double sigmaX, double sigmaY,
        int32_t sizeX = 0, int32_t sizeY = 0,
        bool normalize = true);

    // =========================================================================
    // Separable Kernel Pairs
    // =========================================================================

    /**
     * @brief Kernel pair for separable 2D convolution
     */
    struct SeparableKernel {
        std::vector<double> horizontal;  ///< Horizontal (row) kernel
        std::vector<double> vertical;    ///< Vertical (column) kernel
    };

    /**
     * @brief Get separable kernels for 2D Gaussian smoothing
     * @param sigma Standard deviation
     * @param size Kernel size (0 = auto)
     * @return Pair of 1D kernels for separable convolution
     *
     * For 2D Gaussian, both kernels are identical (isotropic case).
     */
    static SeparableKernel SeparableSmooth(double sigma, int32_t size = 0);

    /**
     * @brief Get separable kernels for X gradient (Gx)
     * @param sigma Standard deviation
     * @param size Kernel size (0 = auto)
     * @return Pair: horizontal = derivative, vertical = smooth
     *
     * Computes ∂G/∂x by convolving:
     * - Vertically with Gaussian
     * - Horizontally with Gaussian derivative
     */
    static SeparableKernel SeparableGradientX(double sigma, int32_t size = 0);

    /**
     * @brief Get separable kernels for Y gradient (Gy)
     * @param sigma Standard deviation
     * @param size Kernel size (0 = auto)
     * @return Pair: horizontal = smooth, vertical = derivative
     *
     * Computes ∂G/∂y by convolving:
     * - Horizontally with Gaussian
     * - Vertically with Gaussian derivative
     */
    static SeparableKernel SeparableGradientY(double sigma, int32_t size = 0);

    /**
     * @brief Get separable kernels for Gxx (second derivative in X)
     * @param sigma Standard deviation
     * @param size Kernel size (0 = auto)
     * @return Pair: horizontal = 2nd derivative, vertical = smooth
     */
    static SeparableKernel SeparableGxx(double sigma, int32_t size = 0);

    /**
     * @brief Get separable kernels for Gyy (second derivative in Y)
     * @param sigma Standard deviation
     * @param size Kernel size (0 = auto)
     * @return Pair: horizontal = smooth, vertical = 2nd derivative
     */
    static SeparableKernel SeparableGyy(double sigma, int32_t size = 0);

    /**
     * @brief Get separable kernels for Gxy (mixed partial derivative)
     * @param sigma Standard deviation
     * @param size Kernel size (0 = auto)
     * @return Pair: horizontal = 1st derivative, vertical = 1st derivative
     *
     * Computes ∂²G/∂x∂y
     */
    static SeparableKernel SeparableGxy(double sigma, int32_t size = 0);

    // =========================================================================
    // Special Purpose Kernels
    // =========================================================================

    /**
     * @brief Generate Laplacian of Gaussian (LoG) kernel
     * @param sigma Standard deviation
     * @param size Kernel size (0 = auto)
     * @param normalize If true, normalize
     * @return 2D LoG kernel
     *
     * LoG(x,y) = (x² + y² - 2σ²) * G(x,y) / σ⁴
     *
     * Used for blob detection and edge detection.
     * Zero-crossings indicate edges.
     */
    static std::vector<double> LaplacianOfGaussian(double sigma, int32_t size = 0, bool normalize = true);

    /**
     * @brief Generate Difference of Gaussians (DoG) kernel
     * @param sigma1 Smaller sigma
     * @param sigma2 Larger sigma
     * @param size Kernel size (0 = auto from sigma2)
     * @param normalize If true, normalize
     * @return 2D DoG kernel
     *
     * DoG = G(sigma1) - G(sigma2)
     *
     * Approximates LoG, used in SIFT.
     */
    static std::vector<double> DifferenceOfGaussians(
        double sigma1, double sigma2,
        int32_t size = 0, bool normalize = true);

    // =========================================================================
    // Utility Functions
    // =========================================================================

    /**
     * @brief Compute Gaussian value at a point
     * @param x Distance from center
     * @param sigma Standard deviation
     * @return Unnormalized Gaussian value exp(-x²/(2σ²))
     */
    static double GaussianValue(double x, double sigma);

    /**
     * @brief Compute 2D Gaussian value at a point
     * @param x X distance from center
     * @param y Y distance from center
     * @param sigma Standard deviation
     * @return Unnormalized 2D Gaussian value
     */
    static double GaussianValue2D(double x, double y, double sigma);

    /**
     * @brief Normalize a kernel to sum to 1.0
     * @param kernel Kernel to normalize (modified in place)
     */
    static void Normalize(std::vector<double>& kernel);

    /**
     * @brief Normalize a derivative kernel for gradient preservation
     * @param kernel Derivative kernel (modified in place)
     *
     * Normalizes so that convolution with a step edge gives magnitude 1.
     */
    static void NormalizeDerivative(std::vector<double>& kernel);

private:
    // Minimum kernel size
    static constexpr int32_t MIN_KERNEL_SIZE = 3;
};

} // namespace Qi::Vision::Internal
