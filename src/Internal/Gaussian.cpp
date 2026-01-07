/**
 * @file Gaussian.cpp
 * @brief Gaussian kernel implementation
 */

#include <QiVision/Internal/Gaussian.h>

#include <cmath>
#include <algorithm>
#include <numeric>

namespace Qi::Vision::Internal {

// ============================================================================
// Kernel Size Computation
// ============================================================================

int32_t Gaussian::ComputeKernelSize(double sigma, double cutoff) {
    if (sigma <= 0.0) {
        return MIN_KERNEL_SIZE;
    }

    // Size = 2 * ceil(cutoff * sigma) + 1 (ensures odd)
    int32_t halfSize = static_cast<int32_t>(std::ceil(cutoff * sigma));
    int32_t size = 2 * halfSize + 1;

    return std::max(size, MIN_KERNEL_SIZE);
}

double Gaussian::ComputeSigma(int32_t kernelSize, double cutoff) {
    if (kernelSize < MIN_KERNEL_SIZE) {
        kernelSize = MIN_KERNEL_SIZE;
    }

    // Ensure odd
    if (kernelSize % 2 == 0) {
        ++kernelSize;
    }

    int32_t halfSize = kernelSize / 2;
    return static_cast<double>(halfSize) / cutoff;
}

// ============================================================================
// 1D Gaussian Kernels
// ============================================================================

std::vector<double> Gaussian::Kernel1D(double sigma, int32_t size, bool normalize) {
    if (sigma <= 0.0) {
        // Return delta function
        if (size <= 0) size = 3;
        if (size % 2 == 0) ++size;
        std::vector<double> kernel(size, 0.0);
        kernel[size / 2] = 1.0;
        return kernel;
    }

    // Compute size if not specified
    if (size <= 0) {
        size = ComputeKernelSize(sigma);
    }

    // Ensure odd
    if (size % 2 == 0) {
        ++size;
    }

    std::vector<double> kernel(size);
    int32_t center = size / 2;

    // Precompute coefficient
    double twoSigmaSq = 2.0 * sigma * sigma;

    // Fill kernel
    for (int32_t i = 0; i < size; ++i) {
        double x = static_cast<double>(i - center);
        kernel[i] = std::exp(-x * x / twoSigmaSq);
    }

    // Normalize if requested
    if (normalize) {
        Normalize(kernel);
    }

    return kernel;
}

std::vector<double> Gaussian::Derivative1D(double sigma, int32_t size, bool normalize) {
    if (sigma <= 0.0) {
        // Return simple difference kernel [-1, 0, 1]
        return {-0.5, 0.0, 0.5};
    }

    // Compute size if not specified
    if (size <= 0) {
        size = ComputeKernelSize(sigma);
    }

    // Ensure odd
    if (size % 2 == 0) {
        ++size;
    }

    std::vector<double> kernel(size);
    int32_t center = size / 2;

    double twoSigmaSq = 2.0 * sigma * sigma;
    double sigmaSq = sigma * sigma;

    // G'(x) = x * G(x) / σ²
    // Sign convention: positive gradient for rising edge (dark → bright)
    // This differs from the mathematical derivative of Gaussian (-x*G/σ²)
    // but matches the intuitive gradient direction convention
    for (int32_t i = 0; i < size; ++i) {
        double x = static_cast<double>(i - center);
        double g = std::exp(-x * x / twoSigmaSq);
        kernel[i] = x * g / sigmaSq;
    }

    // Normalize for gradient preservation
    if (normalize) {
        NormalizeDerivative(kernel);
    }

    return kernel;
}

std::vector<double> Gaussian::SecondDerivative1D(double sigma, int32_t size, bool normalize) {
    if (sigma <= 0.0) {
        // Return simple second difference kernel [1, -2, 1]
        return {1.0, -2.0, 1.0};
    }

    // Compute size if not specified
    if (size <= 0) {
        size = ComputeKernelSize(sigma);
    }

    // Ensure odd
    if (size % 2 == 0) {
        ++size;
    }

    std::vector<double> kernel(size);
    int32_t center = size / 2;

    double twoSigmaSq = 2.0 * sigma * sigma;
    double sigmaSq = sigma * sigma;
    double sigmaFourth = sigmaSq * sigmaSq;

    // G''(x) = (x² - σ²) * G(x) / σ⁴
    for (int32_t i = 0; i < size; ++i) {
        double x = static_cast<double>(i - center);
        double xSq = x * x;
        double g = std::exp(-xSq / twoSigmaSq);
        kernel[i] = (xSq - sigmaSq) * g / sigmaFourth;
    }

    if (normalize) {
        // Normalize so that response to unit step edge is consistent
        double sum = 0.0;
        for (int32_t i = center + 1; i < size; ++i) {
            sum += kernel[i];
        }
        if (std::abs(sum) > 1e-10) {
            double scale = 1.0 / sum;
            for (auto& v : kernel) {
                v *= scale;
            }
        }
    }

    return kernel;
}

// ============================================================================
// 2D Gaussian Kernels
// ============================================================================

std::vector<double> Gaussian::Kernel2D(double sigma, int32_t size, bool normalize) {
    return Kernel2DAnisotropic(sigma, sigma, size, size, normalize);
}

std::vector<double> Gaussian::Kernel2DAnisotropic(
    double sigmaX, double sigmaY,
    int32_t sizeX, int32_t sizeY,
    bool normalize) {

    // Handle zero/negative sigma
    if (sigmaX <= 0.0) sigmaX = 0.5;
    if (sigmaY <= 0.0) sigmaY = 0.5;

    // Compute sizes if not specified
    if (sizeX <= 0) {
        sizeX = ComputeKernelSize(sigmaX);
    }
    if (sizeY <= 0) {
        sizeY = ComputeKernelSize(sigmaY);
    }

    // Ensure odd
    if (sizeX % 2 == 0) ++sizeX;
    if (sizeY % 2 == 0) ++sizeY;

    std::vector<double> kernel(sizeY * sizeX);
    int32_t centerX = sizeX / 2;
    int32_t centerY = sizeY / 2;

    double twoSigmaXSq = 2.0 * sigmaX * sigmaX;
    double twoSigmaYSq = 2.0 * sigmaY * sigmaY;

    // Fill 2D kernel
    for (int32_t y = 0; y < sizeY; ++y) {
        double dy = static_cast<double>(y - centerY);
        double expY = std::exp(-dy * dy / twoSigmaYSq);

        for (int32_t x = 0; x < sizeX; ++x) {
            double dx = static_cast<double>(x - centerX);
            double expX = std::exp(-dx * dx / twoSigmaXSq);
            kernel[y * sizeX + x] = expX * expY;
        }
    }

    if (normalize) {
        Normalize(kernel);
    }

    return kernel;
}

// ============================================================================
// Separable Kernel Pairs
// ============================================================================

Gaussian::SeparableKernel Gaussian::SeparableSmooth(double sigma, int32_t size) {
    auto kernel = Kernel1D(sigma, size, true);
    return {kernel, kernel};  // Same for both directions
}

Gaussian::SeparableKernel Gaussian::SeparableGradientX(double sigma, int32_t size) {
    return {
        Derivative1D(sigma, size, true),   // Horizontal: derivative
        Kernel1D(sigma, size, true)        // Vertical: smooth
    };
}

Gaussian::SeparableKernel Gaussian::SeparableGradientY(double sigma, int32_t size) {
    return {
        Kernel1D(sigma, size, true),       // Horizontal: smooth
        Derivative1D(sigma, size, true)    // Vertical: derivative
    };
}

Gaussian::SeparableKernel Gaussian::SeparableGxx(double sigma, int32_t size) {
    return {
        SecondDerivative1D(sigma, size, true),  // Horizontal: 2nd derivative
        Kernel1D(sigma, size, true)             // Vertical: smooth
    };
}

Gaussian::SeparableKernel Gaussian::SeparableGyy(double sigma, int32_t size) {
    return {
        Kernel1D(sigma, size, true),            // Horizontal: smooth
        SecondDerivative1D(sigma, size, true)   // Vertical: 2nd derivative
    };
}

Gaussian::SeparableKernel Gaussian::SeparableGxy(double sigma, int32_t size) {
    auto deriv = Derivative1D(sigma, size, true);
    return {deriv, deriv};  // Both directions: 1st derivative
}

// ============================================================================
// Special Purpose Kernels
// ============================================================================

std::vector<double> Gaussian::LaplacianOfGaussian(double sigma, int32_t size, bool normalize) {
    if (sigma <= 0.0) sigma = 1.0;

    if (size <= 0) {
        size = ComputeKernelSize(sigma);
    }
    if (size % 2 == 0) ++size;

    std::vector<double> kernel(size * size);
    int32_t center = size / 2;

    double twoSigmaSq = 2.0 * sigma * sigma;
    double sigmaSq = sigma * sigma;
    double sigmaFourth = sigmaSq * sigmaSq;

    // LoG(x,y) = (x² + y² - 2σ²) * G(x,y) / σ⁴
    // = ((x² + y²) / σ⁴ - 2/σ²) * G(x,y)
    for (int32_t y = 0; y < size; ++y) {
        double dy = static_cast<double>(y - center);
        double dySq = dy * dy;

        for (int32_t x = 0; x < size; ++x) {
            double dx = static_cast<double>(x - center);
            double dxSq = dx * dx;

            double rSq = dxSq + dySq;
            double g = std::exp(-rSq / twoSigmaSq);
            kernel[y * size + x] = (rSq - 2.0 * sigmaSq) * g / sigmaFourth;
        }
    }

    if (normalize) {
        // Normalize so that the sum of positive values equals 1
        // (LoG sums to 0 by design)
        double posSum = 0.0;
        for (double v : kernel) {
            if (v > 0.0) posSum += v;
        }
        if (posSum > 1e-10) {
            for (auto& v : kernel) {
                v /= posSum;
            }
        }
    }

    return kernel;
}

std::vector<double> Gaussian::DifferenceOfGaussians(
    double sigma1, double sigma2,
    int32_t size, bool normalize) {

    // Ensure sigma1 < sigma2
    if (sigma1 > sigma2) {
        std::swap(sigma1, sigma2);
    }

    if (size <= 0) {
        size = ComputeKernelSize(sigma2);  // Use larger sigma
    }
    if (size % 2 == 0) ++size;

    // Generate both Gaussians
    auto g1 = Kernel2D(sigma1, size, true);
    auto g2 = Kernel2D(sigma2, size, true);

    // Compute difference
    std::vector<double> kernel(size * size);
    for (size_t i = 0; i < kernel.size(); ++i) {
        kernel[i] = g1[i] - g2[i];
    }

    if (normalize) {
        // Normalize like LoG
        double posSum = 0.0;
        for (double v : kernel) {
            if (v > 0.0) posSum += v;
        }
        if (posSum > 1e-10) {
            for (auto& v : kernel) {
                v /= posSum;
            }
        }
    }

    return kernel;
}

// ============================================================================
// Utility Functions
// ============================================================================

double Gaussian::GaussianValue(double x, double sigma) {
    if (sigma <= 0.0) {
        return (std::abs(x) < 1e-10) ? 1.0 : 0.0;
    }
    return std::exp(-x * x / (2.0 * sigma * sigma));
}

double Gaussian::GaussianValue2D(double x, double y, double sigma) {
    if (sigma <= 0.0) {
        return (std::abs(x) < 1e-10 && std::abs(y) < 1e-10) ? 1.0 : 0.0;
    }
    double rSq = x * x + y * y;
    return std::exp(-rSq / (2.0 * sigma * sigma));
}

void Gaussian::Normalize(std::vector<double>& kernel) {
    if (kernel.empty()) return;

    double sum = std::accumulate(kernel.begin(), kernel.end(), 0.0);
    if (std::abs(sum) > 1e-10) {
        for (auto& v : kernel) {
            v /= sum;
        }
    }
}

void Gaussian::NormalizeDerivative(std::vector<double>& kernel) {
    if (kernel.empty()) return;

    // For derivative kernels, normalize so that the response to
    // a step edge [0,0,...,0,1,1,...,1] gives magnitude 1.
    // This is the sum of positive (or negative) values.
    double posSum = 0.0;
    double negSum = 0.0;

    for (double v : kernel) {
        if (v > 0.0) posSum += v;
        else negSum += v;
    }

    // Use the larger absolute value for normalization
    double maxAbs = std::max(posSum, std::abs(negSum));
    if (maxAbs > 1e-10) {
        for (auto& v : kernel) {
            v /= maxAbs;
        }
    }
}

} // namespace Qi::Vision::Internal
