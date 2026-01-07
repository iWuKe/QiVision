/**
 * @file Convolution.cpp
 * @brief Convolution operations implementation
 */

#include <QiVision/Internal/Convolution.h>

#include <cmath>

namespace Qi::Vision::Internal {

std::vector<double> GenerateGaussianKernel1D(double sigma, int32_t size) {
    if (size <= 0) {
        size = KernelSizeFromSigma(sigma);
    }
    size = MakeOdd(size);

    std::vector<double> kernel(size);
    int32_t halfK = size / 2;
    double sum = 0.0;
    double sigma2 = 2.0 * sigma * sigma;

    for (int32_t i = 0; i < size; ++i) {
        double x = static_cast<double>(i - halfK);
        kernel[i] = std::exp(-x * x / sigma2);
        sum += kernel[i];
    }

    // Normalize
    for (int32_t i = 0; i < size; ++i) {
        kernel[i] /= sum;
    }

    return kernel;
}

std::vector<double> GenerateBoxKernel1D(int32_t size) {
    size = MakeOdd(size);
    double val = 1.0 / size;
    return std::vector<double>(size, val);
}

// Explicit template instantiations

// ConvolveRow
template void ConvolveRow<uint8_t, float>(const uint8_t*, float*, int32_t, int32_t, const double*, int32_t, BorderMode, double);
template void ConvolveRow<uint8_t, double>(const uint8_t*, double*, int32_t, int32_t, const double*, int32_t, BorderMode, double);
template void ConvolveRow<float, float>(const float*, float*, int32_t, int32_t, const double*, int32_t, BorderMode, double);
template void ConvolveRow<double, double>(const double*, double*, int32_t, int32_t, const double*, int32_t, BorderMode, double);

// ConvolveCol
template void ConvolveCol<uint8_t, float>(const uint8_t*, float*, int32_t, int32_t, const double*, int32_t, BorderMode, double);
template void ConvolveCol<uint8_t, double>(const uint8_t*, double*, int32_t, int32_t, const double*, int32_t, BorderMode, double);
template void ConvolveCol<float, float>(const float*, float*, int32_t, int32_t, const double*, int32_t, BorderMode, double);
template void ConvolveCol<double, double>(const double*, double*, int32_t, int32_t, const double*, int32_t, BorderMode, double);

// ConvolveSeparable
template void ConvolveSeparable<uint8_t, float>(const uint8_t*, float*, int32_t, int32_t, const double*, int32_t, const double*, int32_t, BorderMode, double);
template void ConvolveSeparable<uint8_t, double>(const uint8_t*, double*, int32_t, int32_t, const double*, int32_t, const double*, int32_t, BorderMode, double);
template void ConvolveSeparable<float, float>(const float*, float*, int32_t, int32_t, const double*, int32_t, const double*, int32_t, BorderMode, double);
template void ConvolveSeparable<double, double>(const double*, double*, int32_t, int32_t, const double*, int32_t, const double*, int32_t, BorderMode, double);

// ConvolveSeparableSymmetric
template void ConvolveSeparableSymmetric<uint8_t, float>(const uint8_t*, float*, int32_t, int32_t, const double*, int32_t, BorderMode);
template void ConvolveSeparableSymmetric<float, float>(const float*, float*, int32_t, int32_t, const double*, int32_t, BorderMode);

// Convolve2D
template void Convolve2D<uint8_t, float>(const uint8_t*, float*, int32_t, int32_t, const double*, int32_t, int32_t, BorderMode, double);
template void Convolve2D<uint8_t, double>(const uint8_t*, double*, int32_t, int32_t, const double*, int32_t, int32_t, BorderMode, double);
template void Convolve2D<float, float>(const float*, float*, int32_t, int32_t, const double*, int32_t, int32_t, BorderMode, double);
template void Convolve2D<double, double>(const double*, double*, int32_t, int32_t, const double*, int32_t, int32_t, BorderMode, double);

// ComputeIntegralImage
template void ComputeIntegralImage<uint8_t>(const uint8_t*, double*, int32_t, int32_t);
template void ComputeIntegralImage<float>(const float*, double*, int32_t, int32_t);

// BoxFilter
template void BoxFilter<uint8_t, float>(const uint8_t*, float*, int32_t, int32_t, int32_t, int32_t, BorderMode);
template void BoxFilter<float, float>(const float*, float*, int32_t, int32_t, int32_t, int32_t, BorderMode);

// GaussianBlur
template void GaussianBlur<uint8_t, float>(const uint8_t*, float*, int32_t, int32_t, double, double, BorderMode);
template void GaussianBlur<uint8_t, uint8_t>(const uint8_t*, uint8_t*, int32_t, int32_t, double, double, BorderMode);
template void GaussianBlur<float, float>(const float*, float*, int32_t, int32_t, double, double, BorderMode);

// GaussianBlurFixed
template void GaussianBlurFixed<uint8_t, float>(const uint8_t*, float*, int32_t, int32_t, int32_t, double, BorderMode);
template void GaussianBlurFixed<float, float>(const float*, float*, int32_t, int32_t, int32_t, double, BorderMode);

// ConvolveNormalized
template void ConvolveNormalized<uint8_t, float>(const uint8_t*, const uint8_t*, float*, int32_t, int32_t, const double*, int32_t, int32_t);
template void ConvolveNormalized<float, float>(const float*, const uint8_t*, float*, int32_t, int32_t, const double*, int32_t, int32_t);

} // namespace Qi::Vision::Internal
