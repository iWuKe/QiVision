/**
 * @file Gradient.cpp
 * @brief Image gradient computation implementation
 */

#include <QiVision/Internal/Gradient.h>

#include <algorithm>
#include <cstring>

namespace Qi::Vision::Internal {

// ============================================================================
// Kernel Functions
// ============================================================================

std::vector<double> SobelDerivativeKernel(int32_t size) {
    switch (size) {
        case 3:
            return {-1.0, 0.0, 1.0};
        case 5:
            return {-1.0, -2.0, 0.0, 2.0, 1.0};
        case 7:
            return {-1.0, -4.0, -5.0, 0.0, 5.0, 4.0, 1.0};
        default:
            return {-1.0, 0.0, 1.0};
    }
}

std::vector<double> SobelSmoothingKernel(int32_t size) {
    switch (size) {
        case 3:
            return {1.0, 2.0, 1.0};
        case 5:
            return {1.0, 4.0, 6.0, 4.0, 1.0};
        case 7:
            return {1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0};
        default:
            return {1.0, 2.0, 1.0};
    }
}

std::vector<double> ScharrDerivativeKernel() {
    return {-1.0, 0.0, 1.0};
}

std::vector<double> ScharrSmoothingKernel() {
    // Scharr uses [3, 10, 3] / 16 for smoothing (better isotropy)
    return {3.0 / 16.0, 10.0 / 16.0, 3.0 / 16.0};
}

std::vector<double> PrewittDerivativeKernel() {
    return {-1.0, 0.0, 1.0};
}

std::vector<double> PrewittSmoothingKernel() {
    return {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
}

// ============================================================================
// 1D Convolution Helpers
// ============================================================================

template<typename SrcT, typename DstT>
void Convolve1DRow(const SrcT* src, DstT* dst,
                   int32_t width, int32_t height,
                   const double* kernel, int32_t kernelSize,
                   BorderMode borderMode) {
    int32_t halfK = kernelSize / 2;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            double sum = 0.0;
            for (int32_t k = -halfK; k <= halfK; ++k) {
                int32_t srcX = x + k;
                if (srcX < 0 || srcX >= width) {
                    srcX = HandleBorder(srcX, width, borderMode);
                }
                sum += static_cast<double>(src[y * width + srcX]) *
                       kernel[k + halfK];
            }
            dst[y * width + x] = static_cast<DstT>(sum);
        }
    }
}

template<typename SrcT, typename DstT>
void Convolve1DCol(const SrcT* src, DstT* dst,
                   int32_t width, int32_t height,
                   const double* kernel, int32_t kernelSize,
                   BorderMode borderMode) {
    int32_t halfK = kernelSize / 2;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            double sum = 0.0;
            for (int32_t k = -halfK; k <= halfK; ++k) {
                int32_t srcY = y + k;
                if (srcY < 0 || srcY >= height) {
                    srcY = HandleBorder(srcY, height, borderMode);
                }
                sum += static_cast<double>(src[srcY * width + x]) *
                       kernel[k + halfK];
            }
            dst[y * width + x] = static_cast<DstT>(sum);
        }
    }
}

// ============================================================================
// Gradient X Implementation
// ============================================================================

template<typename SrcT, typename DstT>
void GradientX(const SrcT* src, DstT* dst,
               int32_t width, int32_t height,
               GradientOperator op, BorderMode borderMode) {

    std::vector<double> smoothKernel, derivKernel;
    int32_t kernelSize;

    switch (op) {
        case GradientOperator::Sobel3x3:
            smoothKernel = SobelSmoothingKernel(3);
            derivKernel = SobelDerivativeKernel(3);
            kernelSize = 3;
            break;
        case GradientOperator::Sobel5x5:
            smoothKernel = SobelSmoothingKernel(5);
            derivKernel = SobelDerivativeKernel(5);
            kernelSize = 5;
            break;
        case GradientOperator::Sobel7x7:
            smoothKernel = SobelSmoothingKernel(7);
            derivKernel = SobelDerivativeKernel(7);
            kernelSize = 7;
            break;
        case GradientOperator::Scharr:
            smoothKernel = ScharrSmoothingKernel();
            derivKernel = ScharrDerivativeKernel();
            kernelSize = 3;
            break;
        case GradientOperator::Prewitt:
            smoothKernel = PrewittSmoothingKernel();
            derivKernel = PrewittDerivativeKernel();
            kernelSize = 3;
            break;
        case GradientOperator::Central:
            // Simple central difference: (I(x+1) - I(x-1)) / 2
            for (int32_t y = 0; y < height; ++y) {
                for (int32_t x = 0; x < width; ++x) {
                    int32_t xm1 = (x > 0) ? x - 1 : HandleBorder(-1, width, borderMode);
                    int32_t xp1 = (x < width - 1) ? x + 1 : HandleBorder(width, width, borderMode);
                    dst[y * width + x] = static_cast<DstT>(
                        (static_cast<double>(src[y * width + xp1]) -
                         static_cast<double>(src[y * width + xm1])) * 0.5);
                }
            }
            return;
        default:
            smoothKernel = SobelSmoothingKernel(3);
            derivKernel = SobelDerivativeKernel(3);
            kernelSize = 3;
    }

    // Separable convolution: Gx = deriv_x * smooth_y
    // Step 1: Smooth in Y direction
    std::vector<double> temp(width * height);
    Convolve1DCol(src, temp.data(), width, height,
                  smoothKernel.data(), kernelSize, borderMode);

    // Step 2: Derivative in X direction
    Convolve1DRow(temp.data(), dst, width, height,
                  derivKernel.data(), kernelSize, borderMode);
}

// ============================================================================
// Gradient Y Implementation
// ============================================================================

template<typename SrcT, typename DstT>
void GradientY(const SrcT* src, DstT* dst,
               int32_t width, int32_t height,
               GradientOperator op, BorderMode borderMode) {

    std::vector<double> smoothKernel, derivKernel;
    int32_t kernelSize;

    switch (op) {
        case GradientOperator::Sobel3x3:
            smoothKernel = SobelSmoothingKernel(3);
            derivKernel = SobelDerivativeKernel(3);
            kernelSize = 3;
            break;
        case GradientOperator::Sobel5x5:
            smoothKernel = SobelSmoothingKernel(5);
            derivKernel = SobelDerivativeKernel(5);
            kernelSize = 5;
            break;
        case GradientOperator::Sobel7x7:
            smoothKernel = SobelSmoothingKernel(7);
            derivKernel = SobelDerivativeKernel(7);
            kernelSize = 7;
            break;
        case GradientOperator::Scharr:
            smoothKernel = ScharrSmoothingKernel();
            derivKernel = ScharrDerivativeKernel();
            kernelSize = 3;
            break;
        case GradientOperator::Prewitt:
            smoothKernel = PrewittSmoothingKernel();
            derivKernel = PrewittDerivativeKernel();
            kernelSize = 3;
            break;
        case GradientOperator::Central:
            // Simple central difference: (I(y+1) - I(y-1)) / 2
            for (int32_t y = 0; y < height; ++y) {
                for (int32_t x = 0; x < width; ++x) {
                    int32_t ym1 = (y > 0) ? y - 1 : HandleBorder(-1, height, borderMode);
                    int32_t yp1 = (y < height - 1) ? y + 1 : HandleBorder(height, height, borderMode);
                    dst[y * width + x] = static_cast<DstT>(
                        (static_cast<double>(src[yp1 * width + x]) -
                         static_cast<double>(src[ym1 * width + x])) * 0.5);
                }
            }
            return;
        default:
            smoothKernel = SobelSmoothingKernel(3);
            derivKernel = SobelDerivativeKernel(3);
            kernelSize = 3;
    }

    // Separable convolution: Gy = smooth_x * deriv_y
    // Step 1: Smooth in X direction
    std::vector<double> temp(width * height);
    Convolve1DRow(src, temp.data(), width, height,
                  smoothKernel.data(), kernelSize, borderMode);

    // Step 2: Derivative in Y direction
    Convolve1DCol(temp.data(), dst, width, height,
                  derivKernel.data(), kernelSize, borderMode);
}

// ============================================================================
// Combined Gradient
// ============================================================================

template<typename SrcT, typename DstT>
void Gradient(const SrcT* src, DstT* gx, DstT* gy,
              int32_t width, int32_t height,
              GradientOperator op, BorderMode borderMode) {

    std::vector<double> smoothKernel, derivKernel;
    int32_t kernelSize;

    switch (op) {
        case GradientOperator::Sobel3x3:
            smoothKernel = SobelSmoothingKernel(3);
            derivKernel = SobelDerivativeKernel(3);
            kernelSize = 3;
            break;
        case GradientOperator::Sobel5x5:
            smoothKernel = SobelSmoothingKernel(5);
            derivKernel = SobelDerivativeKernel(5);
            kernelSize = 5;
            break;
        case GradientOperator::Sobel7x7:
            smoothKernel = SobelSmoothingKernel(7);
            derivKernel = SobelDerivativeKernel(7);
            kernelSize = 7;
            break;
        case GradientOperator::Scharr:
            smoothKernel = ScharrSmoothingKernel();
            derivKernel = ScharrDerivativeKernel();
            kernelSize = 3;
            break;
        case GradientOperator::Prewitt:
            smoothKernel = PrewittSmoothingKernel();
            derivKernel = PrewittDerivativeKernel();
            kernelSize = 3;
            break;
        case GradientOperator::Central:
            // For central difference, compute separately
            GradientX<SrcT, DstT>(src, gx, width, height, op, borderMode);
            GradientY<SrcT, DstT>(src, gy, width, height, op, borderMode);
            return;
        default:
            smoothKernel = SobelSmoothingKernel(3);
            derivKernel = SobelDerivativeKernel(3);
            kernelSize = 3;
    }

    // Compute intermediates once
    std::vector<double> smoothX(width * height);
    std::vector<double> smoothY(width * height);

    // Smooth in X direction (for Gy)
    Convolve1DRow(src, smoothX.data(), width, height,
                  smoothKernel.data(), kernelSize, borderMode);

    // Smooth in Y direction (for Gx)
    Convolve1DCol(src, smoothY.data(), width, height,
                  smoothKernel.data(), kernelSize, borderMode);

    // Derivative in X direction (for Gx)
    Convolve1DRow(smoothY.data(), gx, width, height,
                  derivKernel.data(), kernelSize, borderMode);

    // Derivative in Y direction (for Gy)
    Convolve1DCol(smoothX.data(), gy, width, height,
                  derivKernel.data(), kernelSize, borderMode);
}

// ============================================================================
// Gradient Magnitude and Direction
// ============================================================================

template<typename SrcT>
void GradientMagDir(const SrcT* src, float* mag, float* dir,
                    int32_t width, int32_t height,
                    GradientOperator op, BorderMode borderMode) {
    size_t size = static_cast<size_t>(width) * height;

    std::vector<float> gx(size), gy(size);
    Gradient<SrcT, float>(src, gx.data(), gy.data(), width, height, op, borderMode);

    if (mag != nullptr) {
        GradientMagnitude(gx.data(), gy.data(), mag, size, false);
    }

    if (dir != nullptr) {
        GradientDirection(gx.data(), gy.data(), dir, size);
    }
}

// ============================================================================
// Second Derivatives
// ============================================================================

template<typename SrcT, typename DstT>
void GradientXX(const SrcT* src, DstT* dst,
                int32_t width, int32_t height,
                BorderMode borderMode) {
    // Second derivative in X: d²I/dx² = I(x+1) - 2*I(x) + I(x-1)
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            int32_t xm1 = (x > 0) ? x - 1 : HandleBorder(-1, width, borderMode);
            int32_t xp1 = (x < width - 1) ? x + 1 : HandleBorder(width, width, borderMode);

            double val = static_cast<double>(src[y * width + xp1]) -
                         2.0 * static_cast<double>(src[y * width + x]) +
                         static_cast<double>(src[y * width + xm1]);
            dst[y * width + x] = static_cast<DstT>(val);
        }
    }
}

template<typename SrcT, typename DstT>
void GradientYY(const SrcT* src, DstT* dst,
                int32_t width, int32_t height,
                BorderMode borderMode) {
    // Second derivative in Y: d²I/dy² = I(y+1) - 2*I(y) + I(y-1)
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            int32_t ym1 = (y > 0) ? y - 1 : HandleBorder(-1, height, borderMode);
            int32_t yp1 = (y < height - 1) ? y + 1 : HandleBorder(height, height, borderMode);

            double val = static_cast<double>(src[yp1 * width + x]) -
                         2.0 * static_cast<double>(src[y * width + x]) +
                         static_cast<double>(src[ym1 * width + x]);
            dst[y * width + x] = static_cast<DstT>(val);
        }
    }
}

template<typename SrcT, typename DstT>
void GradientXY(const SrcT* src, DstT* dst,
                int32_t width, int32_t height,
                BorderMode borderMode) {
    // Mixed derivative: d²I/dxdy
    // = (I(x+1,y+1) - I(x-1,y+1) - I(x+1,y-1) + I(x-1,y-1)) / 4
    auto getPixel = [&](int32_t px, int32_t py) -> double {
        if (px < 0 || px >= width) {
            px = HandleBorder(px, width, borderMode);
        }
        if (py < 0 || py >= height) {
            py = HandleBorder(py, height, borderMode);
        }
        return static_cast<double>(src[py * width + px]);
    };

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            double val = (getPixel(x + 1, y + 1) - getPixel(x - 1, y + 1) -
                          getPixel(x + 1, y - 1) + getPixel(x - 1, y - 1)) * 0.25;
            dst[y * width + x] = static_cast<DstT>(val);
        }
    }
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

// Convolve1D
template void Convolve1DRow<uint8_t, double>(const uint8_t*, double*, int32_t, int32_t, const double*, int32_t, BorderMode);
template void Convolve1DRow<double, double>(const double*, double*, int32_t, int32_t, const double*, int32_t, BorderMode);
template void Convolve1DRow<double, float>(const double*, float*, int32_t, int32_t, const double*, int32_t, BorderMode);
template void Convolve1DCol<uint8_t, double>(const uint8_t*, double*, int32_t, int32_t, const double*, int32_t, BorderMode);
template void Convolve1DCol<double, double>(const double*, double*, int32_t, int32_t, const double*, int32_t, BorderMode);
template void Convolve1DCol<double, float>(const double*, float*, int32_t, int32_t, const double*, int32_t, BorderMode);

// GradientX
template void GradientX<uint8_t, float>(const uint8_t*, float*, int32_t, int32_t, GradientOperator, BorderMode);
template void GradientX<uint16_t, float>(const uint16_t*, float*, int32_t, int32_t, GradientOperator, BorderMode);
template void GradientX<int16_t, float>(const int16_t*, float*, int32_t, int32_t, GradientOperator, BorderMode);
template void GradientX<float, float>(const float*, float*, int32_t, int32_t, GradientOperator, BorderMode);
template void GradientX<double, double>(const double*, double*, int32_t, int32_t, GradientOperator, BorderMode);

// GradientY
template void GradientY<uint8_t, float>(const uint8_t*, float*, int32_t, int32_t, GradientOperator, BorderMode);
template void GradientY<uint16_t, float>(const uint16_t*, float*, int32_t, int32_t, GradientOperator, BorderMode);
template void GradientY<int16_t, float>(const int16_t*, float*, int32_t, int32_t, GradientOperator, BorderMode);
template void GradientY<float, float>(const float*, float*, int32_t, int32_t, GradientOperator, BorderMode);
template void GradientY<double, double>(const double*, double*, int32_t, int32_t, GradientOperator, BorderMode);

// Gradient
template void Gradient<uint8_t, float>(const uint8_t*, float*, float*, int32_t, int32_t, GradientOperator, BorderMode);
template void Gradient<uint16_t, float>(const uint16_t*, float*, float*, int32_t, int32_t, GradientOperator, BorderMode);
template void Gradient<int16_t, float>(const int16_t*, float*, float*, int32_t, int32_t, GradientOperator, BorderMode);
template void Gradient<float, float>(const float*, float*, float*, int32_t, int32_t, GradientOperator, BorderMode);
template void Gradient<double, double>(const double*, double*, double*, int32_t, int32_t, GradientOperator, BorderMode);

// GradientMagDir
template void GradientMagDir<uint8_t>(const uint8_t*, float*, float*, int32_t, int32_t, GradientOperator, BorderMode);
template void GradientMagDir<uint16_t>(const uint16_t*, float*, float*, int32_t, int32_t, GradientOperator, BorderMode);
template void GradientMagDir<float>(const float*, float*, float*, int32_t, int32_t, GradientOperator, BorderMode);

// Second derivatives
template void GradientXX<uint8_t, float>(const uint8_t*, float*, int32_t, int32_t, BorderMode);
template void GradientXX<float, float>(const float*, float*, int32_t, int32_t, BorderMode);
template void GradientYY<uint8_t, float>(const uint8_t*, float*, int32_t, int32_t, BorderMode);
template void GradientYY<float, float>(const float*, float*, int32_t, int32_t, BorderMode);
template void GradientXY<uint8_t, float>(const uint8_t*, float*, int32_t, int32_t, BorderMode);
template void GradientXY<float, float>(const float*, float*, int32_t, int32_t, BorderMode);

} // namespace Qi::Vision::Internal
