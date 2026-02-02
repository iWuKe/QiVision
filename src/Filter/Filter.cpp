/**
 * @file Filter.cpp
 * @brief Image filtering operations implementation
 *
 * Wraps Internal layer functions with Halcon-style API
 */

#include <QiVision/Filter/Filter.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Internal/Convolution.h>
#include <QiVision/Internal/Gradient.h>
#include <QiVision/Internal/Gaussian.h>
#include <QiVision/Internal/MorphGray.h>
#include <QiVision/Internal/Histogram.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace Qi::Vision::Filter {

// Use Internal types
using Internal::BorderMode;

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

BorderMode ParseBorderMode(const std::string& mode) {
    std::string lower = mode;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "reflect" || lower == "reflect101" || lower == "mirrored") {
        return BorderMode::Reflect101;
    } else if (lower == "replicate" || lower == "continued") {
        return BorderMode::Replicate;
    } else if (lower == "constant" || lower == "zero") {
        return BorderMode::Constant;
    } else if (lower == "wrap" || lower == "cyclic") {
        return BorderMode::Wrap;
    }

    if (!lower.empty()) {
        throw InvalidArgumentException("Unknown border mode: " + mode);
    }
    return BorderMode::Reflect101;
}

int32_t ParseKernelSize(const std::string& size) {
    if (size.empty()) {
        return 3;
    }
    std::string lower = size;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "3x3" || lower == "3") return 3;
    if (lower == "5x5" || lower == "5") return 5;
    if (lower == "7x7" || lower == "7") return 7;
    if (lower == "9x9" || lower == "9") return 9;
    if (lower == "11x11" || lower == "11") return 11;
    throw InvalidArgumentException("Unknown kernel size: " + size);
}

inline double Clamp(double val, double minVal, double maxVal) {
    return std::max(minVal, std::min(maxVal, val));
}

inline uint8_t ClampU8(double val) {
    return static_cast<uint8_t>(Clamp(val, 0.0, 255.0));
}

bool RequireGrayU8(const QImage& image, const char* funcName) {
    if (image.Empty()) {
        return false;
    }
    if (!image.IsValid()) {
        throw InvalidArgumentException(std::string(funcName) + ": invalid image");
    }
    if (image.Type() != PixelType::UInt8 || image.Channels() != 1) {
        throw UnsupportedException(std::string(funcName) +
                                   " requires single-channel UInt8 image");
    }
    return true;
}

bool RequireValidImage(const QImage& image, const char* funcName) {
    if (image.Empty()) {
        return false;
    }
    if (!image.IsValid()) {
        throw InvalidArgumentException(std::string(funcName) + ": invalid image");
    }
    return true;
}

} // anonymous namespace

// =============================================================================
// Utility Functions
// =============================================================================

int32_t OptimalKernelSize(double sigma) {
    // Halcon convention: kernel size = 2 * ceil(3 * sigma) + 1
    int32_t size = 2 * static_cast<int32_t>(std::ceil(3.0 * sigma)) + 1;
    return std::max(3, size);
}

std::vector<double> GenGaussKernel(double sigma, int32_t size) {
    if (size <= 0) {
        size = OptimalKernelSize(sigma);
    }

    // Ensure odd size
    if (size % 2 == 0) size++;

    std::vector<double> kernel(size);
    int32_t center = size / 2;
    double sum = 0.0;
    double sigma2 = 2.0 * sigma * sigma;

    for (int32_t i = 0; i < size; ++i) {
        double x = i - center;
        kernel[i] = std::exp(-x * x / sigma2);
        sum += kernel[i];
    }

    // Normalize
    for (double& k : kernel) {
        k /= sum;
    }

    return kernel;
}

std::vector<double> GenGaussDerivKernel(double sigma, int32_t order, int32_t size) {
    if (size <= 0) {
        size = OptimalKernelSize(sigma);
    }

    if (size % 2 == 0) size++;

    std::vector<double> kernel(size);
    int32_t center = size / 2;
    double sigma2 = sigma * sigma;
    double sigma4 = sigma2 * sigma2;

    for (int32_t i = 0; i < size; ++i) {
        double x = i - center;

        if (order == 1) {
            // First derivative: -x / sigma^2 * G(x)
            kernel[i] = -x / sigma2 * std::exp(-x * x / (2.0 * sigma2));
        } else if (order == 2) {
            // Second derivative: (x^2 - sigma^2) / sigma^4 * G(x)
            kernel[i] = (x * x - sigma2) / sigma4 * std::exp(-x * x / (2.0 * sigma2));
        } else {
            kernel[i] = std::exp(-x * x / (2.0 * sigma2));
        }
    }

    return kernel;
}

// =============================================================================
// Smoothing Filters
// =============================================================================

void GaussFilter(const QImage& image, QImage& output, double sigma) {
    GaussFilter(image, output, sigma, sigma, "reflect");
}

void GaussFilter(const QImage& image, QImage& output, double sigmaX, double sigmaY,
                  const std::string& borderMode) {
    if (!RequireValidImage(image, "GaussFilter")) {
        output = QImage();
        return;
    }
    if (!std::isfinite(sigmaX) || !std::isfinite(sigmaY) || sigmaX <= 0.0 || sigmaY <= 0.0) {
        throw InvalidArgumentException("GaussFilter: sigma must be > 0");
    }

    if (image.Type() != PixelType::UInt8) {
        throw UnsupportedException("GaussFilter only supports UInt8 images");
    }

    BorderMode border = ParseBorderMode(borderMode);

    // Generate Gaussian kernels
    auto kernelX = GenGaussKernel(sigmaX);
    auto kernelY = GenGaussKernel(sigmaY);

    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();

    output = QImage(w, h, image.Type(), image.GetChannelType());

    // Process each channel
    std::vector<float> temp(w * h);
    std::vector<float> src(w * h);

    for (int c = 0; c < channels; ++c) {
        // Extract channel to float
        for (int32_t y = 0; y < h; ++y) {
            const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < w; ++x) {
                src[y * w + x] = static_cast<float>(row[x * channels + c]);
            }
        }

        // Apply separable convolution
        Internal::ConvolveSeparable<float, float>(
            src.data(), temp.data(), w, h,
            kernelX.data(), static_cast<int32_t>(kernelX.size()),
            kernelY.data(), static_cast<int32_t>(kernelY.size()),
            border);

        // Write back to result
        for (int32_t y = 0; y < h; ++y) {
            uint8_t* row = static_cast<uint8_t*>(output.RowPtr(y));
            for (int32_t x = 0; x < w; ++x) {
                row[x * channels + c] = ClampU8(temp[y * w + x]);
            }
        }
    }
}

void GaussImage(const QImage& image, QImage& output, const std::string& size) {
    int32_t kernelSize = ParseKernelSize(size);
    double sigma = kernelSize / 6.0;  // Approximate sigma from size
    GaussFilter(image, output, sigma);
}

void MeanImage(const QImage& image, QImage& output, int32_t width, int32_t height,
                const std::string& borderMode) {
    if (!RequireValidImage(image, "MeanImage")) {
        output = QImage();
        return;
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("MeanImage: width/height must be > 0");
    }

    if (image.Type() != PixelType::UInt8) {
        throw UnsupportedException("MeanImage only supports UInt8 images");
    }

    // Create uniform kernel
    std::vector<double> kernelX(width, 1.0 / width);
    std::vector<double> kernelY(height, 1.0 / height);

    BorderMode border = ParseBorderMode(borderMode);

    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();

    output = QImage(w, h, image.Type(), image.GetChannelType());

    std::vector<float> temp(w * h);
    std::vector<float> src(w * h);

    for (int c = 0; c < channels; ++c) {
        for (int32_t y = 0; y < h; ++y) {
            const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < w; ++x) {
                src[y * w + x] = static_cast<float>(row[x * channels + c]);
            }
        }

        Internal::ConvolveSeparable<float, float>(
            src.data(), temp.data(), w, h,
            kernelX.data(), width,
            kernelY.data(), height,
            border);

        for (int32_t y = 0; y < h; ++y) {
            uint8_t* row = static_cast<uint8_t*>(output.RowPtr(y));
            for (int32_t x = 0; x < w; ++x) {
                row[x * channels + c] = ClampU8(temp[y * w + x]);
            }
        }
    }
}

void MeanImage(const QImage& image, QImage& output, int32_t size,
                const std::string& borderMode) {
    MeanImage(image, output, size, size, borderMode);
}

void MedianImage(const QImage& image, QImage& output, const std::string& maskType,
                  int32_t radius, const std::string& marginMode) {
    (void)maskType;
    (void)marginMode;
    if (!RequireValidImage(image, "MedianImage")) {
        output = QImage();
        return;
    }
    if (radius <= 0) {
        throw InvalidArgumentException("MedianImage: radius must be > 0");
    }

    if (image.Type() != PixelType::UInt8) {
        throw UnsupportedException("MedianImage only supports UInt8 images");
    }

    int32_t size = 2 * radius + 1;
    MedianRect(image, output, size, size);
}

void MedianRect(const QImage& image, QImage& output, int32_t width, int32_t height) {
    if (!RequireValidImage(image, "MedianRect")) {
        output = QImage();
        return;
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("MedianRect: width/height must be > 0");
    }

    if (image.Type() != PixelType::UInt8) {
        throw UnsupportedException("MedianRect only supports UInt8 images");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();
    int32_t halfW = width / 2;
    int32_t halfH = height / 2;

    output = QImage(w, h, image.Type(), image.GetChannelType());
    std::vector<uint8_t> neighborhood(width * height);

    for (int c = 0; c < channels; ++c) {
        for (int32_t y = 0; y < h; ++y) {
            uint8_t* dstRow = static_cast<uint8_t*>(output.RowPtr(y));

            for (int32_t x = 0; x < w; ++x) {
                // Collect neighborhood
                int count = 0;
                for (int32_t ky = -halfH; ky <= halfH; ++ky) {
                    int32_t sy = std::max(0, std::min(h - 1, y + ky));
                    const uint8_t* srcRow = static_cast<const uint8_t*>(image.RowPtr(sy));

                    for (int32_t kx = -halfW; kx <= halfW; ++kx) {
                        int32_t sx = std::max(0, std::min(w - 1, x + kx));
                        neighborhood[count++] = srcRow[sx * channels + c];
                    }
                }

                // Find median (partial sort)
                std::nth_element(neighborhood.begin(), neighborhood.begin() + count / 2,
                                neighborhood.begin() + count);
                dstRow[x * channels + c] = neighborhood[count / 2];
            }
        }
    }
}

void BilateralFilter(const QImage& image, QImage& output,
                      double sigmaSpatial, double sigmaIntensity) {
    if (!std::isfinite(sigmaSpatial) || !std::isfinite(sigmaIntensity) ||
        sigmaSpatial <= 0.0 || sigmaIntensity <= 0.0) {
        throw InvalidArgumentException("BilateralFilter: sigma must be > 0");
    }
    int32_t size = OptimalKernelSize(sigmaSpatial);
    BilateralFilter(image, output, size, sigmaSpatial, sigmaIntensity);
}

void BilateralFilter(const QImage& image, QImage& output, int32_t size,
                      double sigmaSpatial, double sigmaIntensity) {
    if (!RequireValidImage(image, "BilateralFilter")) {
        output = QImage();
        return;
    }
    if (size <= 0) {
        throw InvalidArgumentException("BilateralFilter: size must be > 0");
    }
    if (!std::isfinite(sigmaSpatial) || !std::isfinite(sigmaIntensity) ||
        sigmaSpatial <= 0.0 || sigmaIntensity <= 0.0) {
        throw InvalidArgumentException("BilateralFilter: sigma must be > 0");
    }

    if (image.Type() != PixelType::UInt8) {
        throw UnsupportedException("BilateralFilter only supports UInt8 images");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();
    int32_t halfSize = size / 2;

    double spatialCoeff = -0.5 / (sigmaSpatial * sigmaSpatial);
    double intensityCoeff = -0.5 / (sigmaIntensity * sigmaIntensity);

    output = QImage(w, h, image.Type(), image.GetChannelType());

    // Precompute spatial weights
    std::vector<double> spatialWeight(size * size);
    for (int ky = 0; ky < size; ++ky) {
        for (int kx = 0; kx < size; ++kx) {
            double dy = ky - halfSize;
            double dx = kx - halfSize;
            spatialWeight[ky * size + kx] = std::exp((dx * dx + dy * dy) * spatialCoeff);
        }
    }

    for (int32_t y = 0; y < h; ++y) {
        uint8_t* dstRow = static_cast<uint8_t*>(output.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            for (int c = 0; c < channels; ++c) {
                const uint8_t* centerRow = static_cast<const uint8_t*>(image.RowPtr(y));
                double centerVal = centerRow[x * channels + c];

                double sum = 0.0;
                double weightSum = 0.0;

                for (int32_t ky = -halfSize; ky <= halfSize; ++ky) {
                    int32_t sy = std::max(0, std::min(h - 1, y + ky));
                    const uint8_t* srcRow = static_cast<const uint8_t*>(image.RowPtr(sy));

                    for (int32_t kx = -halfSize; kx <= halfSize; ++kx) {
                        int32_t sx = std::max(0, std::min(w - 1, x + kx));
                        double pixelVal = srcRow[sx * channels + c];

                        double intensityDiff = pixelVal - centerVal;
                        double intensityWeight = std::exp(intensityDiff * intensityDiff * intensityCoeff);

                        int kidx = (ky + halfSize) * size + (kx + halfSize);
                        double weight = spatialWeight[kidx] * intensityWeight;

                        sum += pixelVal * weight;
                        weightSum += weight;
                    }
                }

                dstRow[x * channels + c] = ClampU8(sum / weightSum);
            }
        }
    }
}

void BinomialFilter(const QImage& image, QImage& output, int32_t width, int32_t height,
                     const std::string& borderMode) {
    if (!RequireValidImage(image, "BinomialFilter")) {
        output = QImage();
        return;
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("BinomialFilter: width/height must be > 0");
    }
    // Binomial coefficients for different sizes
    auto getBinomial = [](int32_t n) -> std::vector<double> {
        std::vector<double> coeffs(n);
        double sum = 0;
        for (int i = 0; i < n; ++i) {
            coeffs[i] = std::tgamma(n) / (std::tgamma(i + 1) * std::tgamma(n - i));
            sum += coeffs[i];
        }
        for (double& c : coeffs) c /= sum;
        return coeffs;
    };

    auto kernelX = getBinomial(width);
    auto kernelY = getBinomial(height);

    BorderMode border = ParseBorderMode(borderMode);

    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();

    output = QImage(w, h, image.Type(), image.GetChannelType());

    std::vector<float> temp(w * h);
    std::vector<float> src(w * h);

    for (int c = 0; c < channels; ++c) {
        for (int32_t y = 0; y < h; ++y) {
            const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < w; ++x) {
                src[y * w + x] = static_cast<float>(row[x * channels + c]);
            }
        }

        Internal::ConvolveSeparable<float, float>(
            src.data(), temp.data(), w, h,
            kernelX.data(), width,
            kernelY.data(), height,
            border);

        for (int32_t y = 0; y < h; ++y) {
            uint8_t* row = static_cast<uint8_t*>(output.RowPtr(y));
            for (int32_t x = 0; x < w; ++x) {
                row[x * channels + c] = ClampU8(temp[y * w + x]);
            }
        }
    }
}

// =============================================================================
// Derivative Filters
// =============================================================================

void SobelAmp(const QImage& image, QImage& output,
               const std::string& filterType, int32_t size) {
    if (!RequireValidImage(image, "SobelAmp")) {
        output = QImage();
        return;
    }

    if (image.Type() != PixelType::UInt8 || image.Channels() != 1) {
        throw UnsupportedException("SobelAmp requires single-channel UInt8 image");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();

    // Get Sobel kernels
    auto deriv = Internal::SobelDerivativeKernel(size);
    auto smooth = Internal::SobelSmoothingKernel(size);

    std::vector<float> src(w * h);
    std::vector<float> gx(w * h);
    std::vector<float> gy(w * h);

    // Convert to float
    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
        for (int32_t x = 0; x < w; ++x) {
            src[y * w + x] = static_cast<float>(row[x]);
        }
    }

    // Compute Gx and Gy using separable convolution
    Internal::ConvolveSeparable<float, float>(
        src.data(), gx.data(), w, h,
        deriv.data(), static_cast<int32_t>(deriv.size()),
        smooth.data(), static_cast<int32_t>(smooth.size()),
        BorderMode::Reflect101);

    Internal::ConvolveSeparable<float, float>(
        src.data(), gy.data(), w, h,
        smooth.data(), static_cast<int32_t>(smooth.size()),
        deriv.data(), static_cast<int32_t>(deriv.size()),
        BorderMode::Reflect101);

    // Create result
    output = QImage(w, h, PixelType::UInt8, ChannelType::Gray);

    std::string lowerType = filterType;
    std::transform(lowerType.begin(), lowerType.end(), lowerType.begin(), ::tolower);
    if (lowerType.empty()) {
        lowerType = "sum_abs";
    }
    bool useSqrt = false;
    if (lowerType == "sum_abs") {
        useSqrt = false;
    } else if (lowerType == "sum_sqrt" || lowerType == "sqrt") {
        useSqrt = true;
    } else {
        throw InvalidArgumentException("SobelAmp: unknown filterType: " + filterType);
    }

    for (int32_t y = 0; y < h; ++y) {
        uint8_t* row = static_cast<uint8_t*>(output.RowPtr(y));
        for (int32_t x = 0; x < w; ++x) {
            float gxVal = gx[y * w + x];
            float gyVal = gy[y * w + x];

            double mag;
            if (useSqrt) {
                mag = std::sqrt(gxVal * gxVal + gyVal * gyVal);
            } else {
                mag = std::abs(gxVal) + std::abs(gyVal);
            }

            row[x] = ClampU8(mag);
        }
    }
}

void SobelDir(const QImage& image, QImage& output,
               const std::string& dirType, int32_t size) {
    std::string lowerType = dirType;
    std::transform(lowerType.begin(), lowerType.end(), lowerType.begin(), ::tolower);
    if (lowerType.empty()) {
        lowerType = "gradient";
    }
    if (lowerType != "gradient" && lowerType != "tangent") {
        throw InvalidArgumentException("SobelDir: unknown dirType: " + dirType);
    }
    bool tangent = (lowerType == "tangent");
    if (!RequireValidImage(image, "SobelDir")) {
        output = QImage();
        return;
    }

    if (image.Type() != PixelType::UInt8 || image.Channels() != 1) {
        throw UnsupportedException("SobelDir requires single-channel UInt8 image");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();

    auto deriv = Internal::SobelDerivativeKernel(size);
    auto smooth = Internal::SobelSmoothingKernel(size);

    std::vector<float> src(w * h);
    std::vector<float> gx(w * h);
    std::vector<float> gy(w * h);

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
        for (int32_t x = 0; x < w; ++x) {
            src[y * w + x] = static_cast<float>(row[x]);
        }
    }

    Internal::ConvolveSeparable<float, float>(
        src.data(), gx.data(), w, h,
        deriv.data(), static_cast<int32_t>(deriv.size()),
        smooth.data(), static_cast<int32_t>(smooth.size()),
        BorderMode::Reflect101);

    Internal::ConvolveSeparable<float, float>(
        src.data(), gy.data(), w, h,
        smooth.data(), static_cast<int32_t>(smooth.size()),
        deriv.data(), static_cast<int32_t>(deriv.size()),
        BorderMode::Reflect101);

    // Return as float image with direction in radians
    output = QImage(w, h, PixelType::Float32, ChannelType::Gray);

    for (int32_t y = 0; y < h; ++y) {
        float* row = static_cast<float*>(output.RowPtr(y));
        for (int32_t x = 0; x < w; ++x) {
            double angle = std::atan2(gy[y * w + x], gx[y * w + x]);
            if (tangent) {
                angle = NormalizeAngle(angle + HALF_PI);
            }
            row[x] = static_cast<float>(angle);
        }
    }
}

void PrewittAmp(const QImage& image, QImage& output, const std::string& filterType) {
    if (!RequireValidImage(image, "PrewittAmp")) {
        output = QImage();
        return;
    }

    if (image.Type() != PixelType::UInt8 || image.Channels() != 1) {
        throw UnsupportedException("PrewittAmp requires single-channel UInt8 image");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();

    // Prewitt uses uniform smoothing [1,1,1]/3 and derivative [-1,0,1]
    auto deriv = Internal::PrewittDerivativeKernel();
    auto smooth = Internal::PrewittSmoothingKernel();

    std::vector<float> src(w * h);
    std::vector<float> gx(w * h);
    std::vector<float> gy(w * h);

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
        for (int32_t x = 0; x < w; ++x) {
            src[y * w + x] = static_cast<float>(row[x]);
        }
    }

    // Gx: derivative in X, smooth in Y
    Internal::ConvolveSeparable<float, float>(
        src.data(), gx.data(), w, h,
        deriv.data(), static_cast<int32_t>(deriv.size()),
        smooth.data(), static_cast<int32_t>(smooth.size()),
        BorderMode::Reflect101);

    // Gy: smooth in X, derivative in Y
    Internal::ConvolveSeparable<float, float>(
        src.data(), gy.data(), w, h,
        smooth.data(), static_cast<int32_t>(smooth.size()),
        deriv.data(), static_cast<int32_t>(deriv.size()),
        BorderMode::Reflect101);

    output = QImage(w, h, PixelType::UInt8, ChannelType::Gray);

    std::string lowerType = filterType;
    std::transform(lowerType.begin(), lowerType.end(), lowerType.begin(), ::tolower);
    if (lowerType.empty()) {
        lowerType = "sum_abs";
    }
    bool useSqrt = false;
    if (lowerType == "sum_abs") {
        useSqrt = false;
    } else if (lowerType == "sum_sqrt" || lowerType == "sqrt") {
        useSqrt = true;
    } else {
        throw InvalidArgumentException("PrewittAmp: unknown filterType: " + filterType);
    }

    for (int32_t y = 0; y < h; ++y) {
        uint8_t* row = static_cast<uint8_t*>(output.RowPtr(y));
        for (int32_t x = 0; x < w; ++x) {
            float gxVal = gx[y * w + x];
            float gyVal = gy[y * w + x];
            double mag = useSqrt ? std::sqrt(gxVal * gxVal + gyVal * gyVal)
                                 : std::abs(gxVal) + std::abs(gyVal);
            row[x] = ClampU8(mag);
        }
    }
}

void RobertsAmp(const QImage& image, QImage& output, const std::string& filterType) {
    if (!RequireValidImage(image, "RobertsAmp")) {
        output = QImage();
        return;
    }

    if (image.Type() != PixelType::UInt8 || image.Channels() != 1) {
        throw UnsupportedException("RobertsAmp requires single-channel UInt8 image");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();

    output = QImage(w, h, PixelType::UInt8, ChannelType::Gray);

    std::string lowerType = filterType;
    std::transform(lowerType.begin(), lowerType.end(), lowerType.begin(), ::tolower);
    if (lowerType.empty()) {
        lowerType = "sum_abs";
    }
    bool useSqrt = false;
    if (lowerType == "sum_abs") {
        useSqrt = false;
    } else if (lowerType == "sum_sqrt" || lowerType == "sqrt") {
        useSqrt = true;
    } else {
        throw InvalidArgumentException("RobertsAmp: unknown filterType: " + filterType);
    }

    // Roberts cross kernels (2x2):
    // Gx: [[1, 0], [0, -1]]  (diagonal difference)
    // Gy: [[0, 1], [-1, 0]]  (anti-diagonal difference)

    for (int32_t y = 0; y < h - 1; ++y) {
        const uint8_t* row0 = static_cast<const uint8_t*>(image.RowPtr(y));
        const uint8_t* row1 = static_cast<const uint8_t*>(image.RowPtr(y + 1));
        uint8_t* dstRow = static_cast<uint8_t*>(output.RowPtr(y));

        for (int32_t x = 0; x < w - 1; ++x) {
            // Gx = I(x,y) - I(x+1,y+1)
            double gx = static_cast<double>(row0[x]) - static_cast<double>(row1[x + 1]);
            // Gy = I(x+1,y) - I(x,y+1)
            double gy = static_cast<double>(row0[x + 1]) - static_cast<double>(row1[x]);

            double mag = useSqrt ? std::sqrt(gx * gx + gy * gy)
                                 : std::abs(gx) + std::abs(gy);
            dstRow[x] = ClampU8(mag);
        }
        // Last column
        dstRow[w - 1] = 0;
    }
    // Last row
    uint8_t* lastRow = static_cast<uint8_t*>(output.RowPtr(h - 1));
    std::memset(lastRow, 0, w);
}

void DerivateGauss(const QImage& image, QImage& output,
                    double sigma, const std::string& component) {
    if (!std::isfinite(sigma) || sigma <= 0.0) {
        throw InvalidArgumentException("DerivateGauss: sigma must be > 0");
    }
    if (!RequireValidImage(image, "DerivateGauss")) {
        output = QImage();
        return;
    }

    if (image.Type() != PixelType::UInt8 || image.Channels() != 1) {
        throw UnsupportedException("DerivateGauss requires single-channel UInt8 image");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();

    std::string lowerComp = component;
    std::transform(lowerComp.begin(), lowerComp.end(), lowerComp.begin(), ::tolower);
    if (lowerComp.empty()) {
        throw InvalidArgumentException("DerivateGauss: component must be non-empty");
    }

    std::vector<double> gaussKernel = GenGaussKernel(sigma);
    std::vector<double> derivKernel = GenGaussDerivKernel(sigma, 1);
    std::vector<double> deriv2Kernel = GenGaussDerivKernel(sigma, 2);

    std::vector<float> src(w * h);
    std::vector<float> dst(w * h);

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
        for (int32_t x = 0; x < w; ++x) {
            src[y * w + x] = static_cast<float>(row[x]);
        }
    }

    if (lowerComp == "x") {
        Internal::ConvolveSeparable<float, float>(
            src.data(), dst.data(), w, h,
            derivKernel.data(), static_cast<int32_t>(derivKernel.size()),
            gaussKernel.data(), static_cast<int32_t>(gaussKernel.size()),
            BorderMode::Reflect101);
    } else if (lowerComp == "y") {
        Internal::ConvolveSeparable<float, float>(
            src.data(), dst.data(), w, h,
            gaussKernel.data(), static_cast<int32_t>(gaussKernel.size()),
            derivKernel.data(), static_cast<int32_t>(derivKernel.size()),
            BorderMode::Reflect101);
    } else if (lowerComp == "xx") {
        Internal::ConvolveSeparable<float, float>(
            src.data(), dst.data(), w, h,
            deriv2Kernel.data(), static_cast<int32_t>(deriv2Kernel.size()),
            gaussKernel.data(), static_cast<int32_t>(gaussKernel.size()),
            BorderMode::Reflect101);
    } else if (lowerComp == "yy") {
        Internal::ConvolveSeparable<float, float>(
            src.data(), dst.data(), w, h,
            gaussKernel.data(), static_cast<int32_t>(gaussKernel.size()),
            deriv2Kernel.data(), static_cast<int32_t>(deriv2Kernel.size()),
            BorderMode::Reflect101);
    } else if (lowerComp == "xy") {
        Internal::ConvolveSeparable<float, float>(
            src.data(), dst.data(), w, h,
            derivKernel.data(), static_cast<int32_t>(derivKernel.size()),
            derivKernel.data(), static_cast<int32_t>(derivKernel.size()),
            BorderMode::Reflect101);
    } else {
        throw InvalidArgumentException("DerivateGauss: unknown component: " + component);
    }

    output = QImage(w, h, PixelType::Float32, ChannelType::Gray);
    std::memcpy(output.Data(), dst.data(), w * h * sizeof(float));
}

void GradientMagnitude(const QImage& image, QImage& output, double sigma) {
    if (!std::isfinite(sigma) || sigma <= 0.0) {
        throw InvalidArgumentException("GradientMagnitude: sigma must be > 0");
    }
    if (!RequireValidImage(image, "GradientMagnitude")) {
        output = QImage();
        return;
    }

    QImage gx, gy;
    DerivateGauss(image, gx, sigma, "x");
    DerivateGauss(image, gy, sigma, "y");

    int32_t w = image.Width();
    int32_t h = image.Height();

    output = QImage(w, h, PixelType::Float32, ChannelType::Gray);

    for (int32_t y = 0; y < h; ++y) {
        const float* gxRow = static_cast<const float*>(gx.RowPtr(y));
        const float* gyRow = static_cast<const float*>(gy.RowPtr(y));
        float* dstRow = static_cast<float*>(output.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            dstRow[x] = std::sqrt(gxRow[x] * gxRow[x] + gyRow[x] * gyRow[x]);
        }
    }
}

void GradientDirection(const QImage& image, QImage& output, double sigma) {
    if (!std::isfinite(sigma) || sigma <= 0.0) {
        throw InvalidArgumentException("GradientDirection: sigma must be > 0");
    }
    if (!RequireValidImage(image, "GradientDirection")) {
        output = QImage();
        return;
    }

    QImage gx, gy;
    DerivateGauss(image, gx, sigma, "x");
    DerivateGauss(image, gy, sigma, "y");

    int32_t w = image.Width();
    int32_t h = image.Height();

    output = QImage(w, h, PixelType::Float32, ChannelType::Gray);

    for (int32_t y = 0; y < h; ++y) {
        const float* gxRow = static_cast<const float*>(gx.RowPtr(y));
        const float* gyRow = static_cast<const float*>(gy.RowPtr(y));
        float* dstRow = static_cast<float*>(output.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            dstRow[x] = std::atan2(gyRow[x], gxRow[x]);
        }
    }
}

void Laplace(const QImage& image, QImage& output, const std::string& filterType) {
    if (!RequireValidImage(image, "Laplace")) {
        output = QImage();
        return;
    }

    if (image.Type() != PixelType::UInt8 || image.Channels() != 1) {
        throw UnsupportedException("Laplace requires single-channel UInt8 image");
    }

    // Laplacian kernels
    std::vector<double> kernel;

    std::string lower = filterType;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower.empty()) {
        lower = "n4";
    }

    if (lower == "n4" || lower == "4" || lower == "3x3") {
        kernel = {0, 1, 0, 1, -4, 1, 0, 1, 0};
    } else if (lower == "n8" || lower == "8") {
        kernel = {1, 1, 1, 1, -8, 1, 1, 1, 1};
    } else {
        throw InvalidArgumentException("Laplace: unknown filterType: " + filterType);
    }

    ConvolImage(image, output, kernel, 3, 3, false, "reflect");
}

void LaplacianOfGaussian(const QImage& image, QImage& output, double sigma) {
    if (!std::isfinite(sigma) || sigma <= 0.0) {
        throw InvalidArgumentException("LaplacianOfGaussian: sigma must be > 0");
    }
    if (!RequireValidImage(image, "LaplacianOfGaussian")) {
        output = QImage();
        return;
    }

    // LoG = Gxx + Gyy
    QImage gxx, gyy;
    DerivateGauss(image, gxx, sigma, "xx");
    DerivateGauss(image, gyy, sigma, "yy");

    int32_t w = image.Width();
    int32_t h = image.Height();

    output = QImage(w, h, PixelType::Float32, ChannelType::Gray);

    for (int32_t y = 0; y < h; ++y) {
        const float* gxxRow = static_cast<const float*>(gxx.RowPtr(y));
        const float* gyyRow = static_cast<const float*>(gyy.RowPtr(y));
        float* dstRow = static_cast<float*>(output.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            dstRow[x] = gxxRow[x] + gyyRow[x];
        }
    }
}

// =============================================================================
// Frequency Domain Filters
// =============================================================================

void HighpassImage(const QImage& image, QImage& output, int32_t width, int32_t height) {
    if (!RequireValidImage(image, "HighpassImage")) {
        output = QImage();
        return;
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("HighpassImage: width/height must be > 0");
    }

    // Highpass = Original - Lowpass
    QImage lowpass;
    MeanImage(image, lowpass, width, height);

    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();

    output = QImage(w, h, image.Type(), image.GetChannelType());

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* srcRow = static_cast<const uint8_t*>(image.RowPtr(y));
        const uint8_t* lpRow = static_cast<const uint8_t*>(lowpass.RowPtr(y));
        uint8_t* dstRow = static_cast<uint8_t*>(output.RowPtr(y));

        for (int32_t x = 0; x < w * channels; ++x) {
            int diff = srcRow[x] - lpRow[x] + 128;  // Add offset to keep in range
            dstRow[x] = ClampU8(diff);
        }
    }
}

void LowpassImage(const QImage& image, QImage& output, int32_t width, int32_t height) {
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("LowpassImage: width/height must be > 0");
    }
    MeanImage(image, output, width, height);
}

// =============================================================================
// Enhancement Filters
// =============================================================================

void EmphasizeImage(const QImage& image, QImage& output,
                     int32_t width, int32_t height, double factor) {
    if (!RequireValidImage(image, "EmphasizeImage")) {
        output = QImage();
        return;
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("EmphasizeImage: width/height must be > 0");
    }
    if (!std::isfinite(factor)) {
        throw InvalidArgumentException("EmphasizeImage: factor must be finite");
    }

    QImage lowpass;
    MeanImage(image, lowpass, width, height);

    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();

    output = QImage(w, h, image.Type(), image.GetChannelType());

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* srcRow = static_cast<const uint8_t*>(image.RowPtr(y));
        const uint8_t* lpRow = static_cast<const uint8_t*>(lowpass.RowPtr(y));
        uint8_t* dstRow = static_cast<uint8_t*>(output.RowPtr(y));

        for (int32_t x = 0; x < w * channels; ++x) {
            double detail = srcRow[x] - lpRow[x];
            double enhanced = srcRow[x] + factor * detail;
            dstRow[x] = ClampU8(enhanced);
        }
    }
}

void UnsharpMask(const QImage& image, QImage& output,
                  double sigma, double amount, double threshold) {
    if (!RequireValidImage(image, "UnsharpMask")) {
        output = QImage();
        return;
    }
    if (!std::isfinite(sigma) || sigma <= 0.0) {
        throw InvalidArgumentException("UnsharpMask: sigma must be > 0");
    }
    if (!std::isfinite(amount) || amount < 0.0) {
        throw InvalidArgumentException("UnsharpMask: amount must be >= 0");
    }
    if (!std::isfinite(threshold) || threshold < 0.0) {
        throw InvalidArgumentException("UnsharpMask: threshold must be >= 0");
    }

    QImage blurred;
    GaussFilter(image, blurred, sigma);

    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();

    output = QImage(w, h, image.Type(), image.GetChannelType());

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* srcRow = static_cast<const uint8_t*>(image.RowPtr(y));
        const uint8_t* blurRow = static_cast<const uint8_t*>(blurred.RowPtr(y));
        uint8_t* dstRow = static_cast<uint8_t*>(output.RowPtr(y));

        for (int32_t x = 0; x < w * channels; ++x) {
            double diff = srcRow[x] - blurRow[x];

            if (std::abs(diff) >= threshold) {
                double sharpened = srcRow[x] + amount * diff;
                dstRow[x] = ClampU8(sharpened);
            } else {
                dstRow[x] = srcRow[x];
            }
        }
    }
}

void ShockFilter(const QImage& image, QImage& output, int32_t iterations, double dt) {
    if (!RequireValidImage(image, "ShockFilter")) {
        output = QImage();
        return;
    }
    if (iterations <= 0) {
        throw InvalidArgumentException("ShockFilter: iterations must be > 0");
    }
    if (!std::isfinite(dt) || dt <= 0.0) {
        throw InvalidArgumentException("ShockFilter: dt must be > 0");
    }

    QImage current = image.Clone();

    for (int32_t iter = 0; iter < iterations; ++iter) {
        QImage lap;
        Laplace(current, lap, "n4");

        int32_t w = current.Width();
        int32_t h = current.Height();

        for (int32_t y = 0; y < h; ++y) {
            const uint8_t* lapRow = static_cast<const uint8_t*>(lap.RowPtr(y));
            uint8_t* curRow = static_cast<uint8_t*>(current.RowPtr(y));

            for (int32_t x = 0; x < w; ++x) {
                double lapVal = lapRow[x] - 128;  // Remove offset
                double update = dt * (lapVal < 0 ? 1.0 : -1.0);
                curRow[x] = ClampU8(curRow[x] + update);
            }
        }
    }

    output = std::move(current);
}

// =============================================================================
// Anisotropic Diffusion
// =============================================================================

void AnisoDiff(const QImage& image, QImage& output, const std::string& mode,
                double contrast, double theta, int32_t iterations) {
    if (!RequireValidImage(image, "AnisoDiff")) {
        output = QImage();
        return;
    }
    if (!std::isfinite(contrast) || contrast <= 0.0) {
        throw InvalidArgumentException("AnisoDiff: contrast must be > 0");
    }
    if (!std::isfinite(theta) || theta <= 0.0) {
        throw InvalidArgumentException("AnisoDiff: theta must be > 0");
    }
    if (iterations <= 0) {
        throw InvalidArgumentException("AnisoDiff: iterations must be > 0");
    }

    if (image.Type() != PixelType::UInt8 || image.Channels() != 1) {
        throw UnsupportedException("AnisoDiff requires single-channel UInt8 image");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();

    std::string lowerMode = mode;
    std::transform(lowerMode.begin(), lowerMode.end(), lowerMode.begin(), ::tolower);
    if (lowerMode.empty()) {
        throw InvalidArgumentException("AnisoDiff: mode must be non-empty");
    }
    bool usePM1 = false;
    if (lowerMode == "pm1") {
        usePM1 = true;
    } else if (lowerMode == "pm2") {
        usePM1 = false;
    } else {
        throw InvalidArgumentException("AnisoDiff: unknown mode: " + mode);
    }

    double k2 = contrast * contrast;

    // Work with float
    std::vector<float> current(w * h);
    std::vector<float> next(w * h);

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
        for (int32_t x = 0; x < w; ++x) {
            current[y * w + x] = static_cast<float>(row[x]);
        }
    }

    for (int32_t iter = 0; iter < iterations; ++iter) {
        for (int32_t y = 0; y < h; ++y) {
            for (int32_t x = 0; x < w; ++x) {
                float center = current[y * w + x];

                // Compute gradients (4-connected)
                float gN = (y > 0) ? current[(y-1) * w + x] - center : 0;
                float gS = (y < h-1) ? current[(y+1) * w + x] - center : 0;
                float gE = (x < w-1) ? current[y * w + x + 1] - center : 0;
                float gW = (x > 0) ? current[y * w + x - 1] - center : 0;

                // Compute diffusion coefficients
                auto diffCoeff = [&](float g) -> float {
                    double g2 = g * g;
                    if (usePM1) {
                        // PM1: c(g) = exp(-g^2/k^2)
                        return static_cast<float>(std::exp(-g2 / k2));
                    } else {
                        // PM2: c(g) = 1 / (1 + g^2/k^2)
                        return static_cast<float>(1.0 / (1.0 + g2 / k2));
                    }
                };

                float cN = diffCoeff(gN);
                float cS = diffCoeff(gS);
                float cE = diffCoeff(gE);
                float cW = diffCoeff(gW);

                // Update
                float update = theta * (cN * gN + cS * gS + cE * gE + cW * gW);
                next[y * w + x] = center + update;
            }
        }

        std::swap(current, next);
    }

    // Convert back to image
    output = QImage(w, h, PixelType::UInt8, ChannelType::Gray);

    for (int32_t y = 0; y < h; ++y) {
        uint8_t* row = static_cast<uint8_t*>(output.RowPtr(y));
        for (int32_t x = 0; x < w; ++x) {
            row[x] = ClampU8(current[y * w + x]);
        }
    }
}

// =============================================================================
// Custom Convolution
// =============================================================================

void ConvolImage(const QImage& image, QImage& output,
                  const std::vector<double>& kernel,
                  int32_t kernelWidth, int32_t kernelHeight,
                  bool normalize,
                  const std::string& borderMode) {
    if (!RequireValidImage(image, "ConvolImage")) {
        output = QImage();
        return;
    }
    if (kernelWidth <= 0 || kernelHeight <= 0) {
        throw InvalidArgumentException("ConvolImage: kernelWidth/Height must be > 0");
    }
    if (kernel.empty() || static_cast<int32_t>(kernel.size()) != kernelWidth * kernelHeight) {
        throw InvalidArgumentException("ConvolImage: kernel size mismatch");
    }

    if (image.Type() != PixelType::UInt8) {
        throw UnsupportedException("ConvolImage only supports UInt8 images");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();

    BorderMode border = ParseBorderMode(borderMode);

    std::vector<double> normKernel = kernel;
    if (normalize) {
        double sum = std::accumulate(normKernel.begin(), normKernel.end(), 0.0);
        if (sum != 0) {
            for (double& k : normKernel) k /= sum;
        }
    }

    output = QImage(w, h, image.Type(), image.GetChannelType());

    std::vector<float> src(w * h);
    std::vector<float> dst(w * h);

    for (int c = 0; c < channels; ++c) {
        for (int32_t y = 0; y < h; ++y) {
            const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < w; ++x) {
                src[y * w + x] = static_cast<float>(row[x * channels + c]);
            }
        }

        Internal::Convolve2D<float, float>(
            src.data(), dst.data(), w, h,
            normKernel.data(), kernelWidth, kernelHeight,
            border);

        for (int32_t y = 0; y < h; ++y) {
            uint8_t* row = static_cast<uint8_t*>(output.RowPtr(y));
            for (int32_t x = 0; x < w; ++x) {
                row[x * channels + c] = ClampU8(dst[y * w + x]);
            }
        }
    }
}

void ConvolSeparable(const QImage& image, QImage& output,
                      const std::vector<double>& kernelX,
                      const std::vector<double>& kernelY,
                      const std::string& borderMode) {
    if (!RequireValidImage(image, "ConvolSeparable")) {
        output = QImage();
        return;
    }
    if (kernelX.empty() || kernelY.empty()) {
        throw InvalidArgumentException("ConvolSeparable: kernels must be non-empty");
    }

    if (image.Type() != PixelType::UInt8) {
        throw UnsupportedException("ConvolSeparable only supports UInt8 images");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    int channels = image.Channels();

    BorderMode border = ParseBorderMode(borderMode);

    output = QImage(w, h, image.Type(), image.GetChannelType());

    std::vector<float> src(w * h);
    std::vector<float> dst(w * h);

    for (int c = 0; c < channels; ++c) {
        for (int32_t y = 0; y < h; ++y) {
            const uint8_t* row = static_cast<const uint8_t*>(image.RowPtr(y));
            for (int32_t x = 0; x < w; ++x) {
                src[y * w + x] = static_cast<float>(row[x * channels + c]);
            }
        }

        Internal::ConvolveSeparable<float, float>(
            src.data(), dst.data(), w, h,
            kernelX.data(), static_cast<int32_t>(kernelX.size()),
            kernelY.data(), static_cast<int32_t>(kernelY.size()),
            border);

        for (int32_t y = 0; y < h; ++y) {
            uint8_t* row = static_cast<uint8_t*>(output.RowPtr(y));
            for (int32_t x = 0; x < w; ++x) {
                row[x * channels + c] = ClampU8(dst[y * w + x]);
            }
        }
    }
}

// =============================================================================
// Rank Filters
// =============================================================================

void RankImage(const QImage& image, QImage& output,
                int32_t width, int32_t height, int32_t rank) {
    if (!RequireValidImage(image, "RankImage")) {
        output = QImage();
        return;
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("RankImage: width/height must be > 0");
    }
    if (rank < 0) {
        throw InvalidArgumentException("RankImage: rank must be >= 0");
    }

    if (image.Type() != PixelType::UInt8 || image.Channels() != 1) {
        throw UnsupportedException("RankImage requires single-channel UInt8 image");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    int32_t halfW = width / 2;
    int32_t halfH = height / 2;

    output = QImage(w, h, PixelType::UInt8, ChannelType::Gray);
    std::vector<uint8_t> neighborhood(width * height);

    for (int32_t y = 0; y < h; ++y) {
        uint8_t* dstRow = static_cast<uint8_t*>(output.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            int count = 0;
            for (int32_t ky = -halfH; ky <= halfH; ++ky) {
                int32_t sy = std::max(0, std::min(h - 1, y + ky));
                const uint8_t* srcRow = static_cast<const uint8_t*>(image.RowPtr(sy));

                for (int32_t kx = -halfW; kx <= halfW; ++kx) {
                    int32_t sx = std::max(0, std::min(w - 1, x + kx));
                    neighborhood[count++] = srcRow[sx];
                }
            }

            int32_t actualRank = std::min(rank, count - 1);
            std::nth_element(neighborhood.begin(), neighborhood.begin() + actualRank,
                            neighborhood.begin() + count);
            dstRow[x] = neighborhood[actualRank];
        }
    }
}

void MinImage(const QImage& image, QImage& output, int32_t width, int32_t height) {
    RankImage(image, output, width, height, 0);
}

void MaxImage(const QImage& image, QImage& output, int32_t width, int32_t height) {
    RankImage(image, output, width, height, width * height - 1);
}

// =============================================================================
// Texture Filters
// =============================================================================

void StdDevImage(const QImage& image, QImage& output, int32_t width, int32_t height) {
    if (!RequireValidImage(image, "StdDevImage")) {
        output = QImage();
        return;
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("StdDevImage: width/height must be > 0");
    }

    QImage variance;
    VarianceImage(image, variance, width, height);

    int32_t w = variance.Width();
    int32_t h = variance.Height();

    output = QImage(w, h, PixelType::Float32, ChannelType::Gray);

    for (int32_t y = 0; y < h; ++y) {
        const float* varRow = static_cast<const float*>(variance.RowPtr(y));
        float* dstRow = static_cast<float*>(output.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            dstRow[x] = std::sqrt(varRow[x]);
        }
    }
}

void VarianceImage(const QImage& image, QImage& output, int32_t width, int32_t height) {
    if (!RequireValidImage(image, "VarianceImage")) {
        output = QImage();
        return;
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("VarianceImage: width/height must be > 0");
    }

    if (image.Type() != PixelType::UInt8 || image.Channels() != 1) {
        throw UnsupportedException("VarianceImage requires single-channel UInt8 image");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    int32_t halfW = width / 2;
    int32_t halfH = height / 2;

    output = QImage(w, h, PixelType::Float32, ChannelType::Gray);

    for (int32_t y = 0; y < h; ++y) {
        float* dstRow = static_cast<float*>(output.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            double sum = 0;
            double sumSq = 0;
            int count = 0;

            for (int32_t ky = -halfH; ky <= halfH; ++ky) {
                int32_t sy = std::max(0, std::min(h - 1, y + ky));
                const uint8_t* srcRow = static_cast<const uint8_t*>(image.RowPtr(sy));

                for (int32_t kx = -halfW; kx <= halfW; ++kx) {
                    int32_t sx = std::max(0, std::min(w - 1, x + kx));
                    double val = srcRow[sx];
                    sum += val;
                    sumSq += val * val;
                    count++;
                }
            }

            double mean = sum / count;
            double variance = (sumSq / count) - (mean * mean);
            dstRow[x] = static_cast<float>(std::max(0.0, variance));
        }
    }
}

void EntropyImage(const QImage& image, QImage& output,
                   int32_t width, int32_t height, int32_t numBins) {
    if (!RequireValidImage(image, "EntropyImage")) {
        output = QImage();
        return;
    }
    if (width <= 0 || height <= 0) {
        throw InvalidArgumentException("EntropyImage: width/height must be > 0");
    }
    if (numBins <= 0) {
        throw InvalidArgumentException("EntropyImage: numBins must be > 0");
    }

    if (image.Type() != PixelType::UInt8 || image.Channels() != 1) {
        throw UnsupportedException("EntropyImage requires single-channel UInt8 image");
    }

    int32_t w = image.Width();
    int32_t h = image.Height();
    int32_t halfW = width / 2;
    int32_t halfH = height / 2;

    output = QImage(w, h, PixelType::Float32, ChannelType::Gray);
    std::vector<int> histogram(numBins);

    for (int32_t y = 0; y < h; ++y) {
        float* dstRow = static_cast<float*>(output.RowPtr(y));

        for (int32_t x = 0; x < w; ++x) {
            std::fill(histogram.begin(), histogram.end(), 0);
            int count = 0;

            for (int32_t ky = -halfH; ky <= halfH; ++ky) {
                int32_t sy = std::max(0, std::min(h - 1, y + ky));
                const uint8_t* srcRow = static_cast<const uint8_t*>(image.RowPtr(sy));

                for (int32_t kx = -halfW; kx <= halfW; ++kx) {
                    int32_t sx = std::max(0, std::min(w - 1, x + kx));
                    int bin = srcRow[sx] * numBins / 256;
                    histogram[bin]++;
                    count++;
                }
            }

            double entropy = 0;
            for (int i = 0; i < numBins; ++i) {
                if (histogram[i] > 0) {
                    double p = static_cast<double>(histogram[i]) / count;
                    entropy -= p * std::log2(p);
                }
            }

            dstRow[x] = static_cast<float>(entropy);
        }
    }
}

// =============================================================================
// Histogram Enhancement
// =============================================================================

void HistogramEqualize(const QImage& image, QImage& output) {
    if (!RequireGrayU8(image, "HistogramEqualize")) {
        output = QImage();
        return;
    }
    output = Internal::HistogramEqualize(image);
}

QImage HistogramEqualize(const QImage& image) {
    if (!RequireGrayU8(image, "HistogramEqualize")) {
        return QImage();
    }
    return Internal::HistogramEqualize(image);
}

void ApplyCLAHE(const QImage& image, QImage& output,
                int32_t tileSize, double clipLimit) {
    if (!RequireGrayU8(image, "ApplyCLAHE")) {
        output = QImage();
        return;
    }
    if (tileSize <= 0) {
        throw InvalidArgumentException("ApplyCLAHE: tileSize must be > 0");
    }
    if (!std::isfinite(clipLimit) || clipLimit <= 0.0) {
        throw InvalidArgumentException("ApplyCLAHE: clipLimit must be > 0");
    }
    Internal::CLAHEParams params;
    params.tileGridSizeX = tileSize;
    params.tileGridSizeY = tileSize;
    params.clipLimit = clipLimit;
    output = Internal::ApplyCLAHE(image, params);
}

QImage ApplyCLAHE(const QImage& image, int32_t tileSize, double clipLimit) {
    if (!RequireGrayU8(image, "ApplyCLAHE")) {
        return QImage();
    }
    if (tileSize <= 0) {
        throw InvalidArgumentException("ApplyCLAHE: tileSize must be > 0");
    }
    if (!std::isfinite(clipLimit) || clipLimit <= 0.0) {
        throw InvalidArgumentException("ApplyCLAHE: clipLimit must be > 0");
    }
    Internal::CLAHEParams params;
    params.tileGridSizeX = tileSize;
    params.tileGridSizeY = tileSize;
    params.clipLimit = clipLimit;
    return Internal::ApplyCLAHE(image, params);
}

void ContrastStretch(const QImage& image, QImage& output,
                     double lowPercentile, double highPercentile,
                     double outputMin, double outputMax) {
    if (!RequireGrayU8(image, "ContrastStretch")) {
        output = QImage();
        return;
    }
    if (!std::isfinite(lowPercentile) || !std::isfinite(highPercentile) ||
        lowPercentile < 0.0 || highPercentile > 100.0 || lowPercentile > highPercentile) {
        throw InvalidArgumentException("ContrastStretch: invalid percentile range");
    }
    if (!std::isfinite(outputMin) || !std::isfinite(outputMax) || outputMax <= outputMin) {
        throw InvalidArgumentException("ContrastStretch: outputMax must be > outputMin");
    }
    output = Internal::ContrastStretch(image, lowPercentile, highPercentile,
                                       outputMin, outputMax);
}

QImage ContrastStretch(const QImage& image,
                       double lowPercentile, double highPercentile,
                       double outputMin, double outputMax) {
    if (!RequireGrayU8(image, "ContrastStretch")) {
        return QImage();
    }
    if (!std::isfinite(lowPercentile) || !std::isfinite(highPercentile) ||
        lowPercentile < 0.0 || highPercentile > 100.0 || lowPercentile > highPercentile) {
        throw InvalidArgumentException("ContrastStretch: invalid percentile range");
    }
    if (!std::isfinite(outputMin) || !std::isfinite(outputMax) || outputMax <= outputMin) {
        throw InvalidArgumentException("ContrastStretch: outputMax must be > outputMin");
    }
    return Internal::ContrastStretch(image, lowPercentile, highPercentile,
                                     outputMin, outputMax);
}

void AutoContrast(const QImage& image, QImage& output) {
    if (!RequireGrayU8(image, "AutoContrast")) {
        output = QImage();
        return;
    }
    output = Internal::AutoContrast(image);
}

QImage AutoContrast(const QImage& image) {
    if (!RequireGrayU8(image, "AutoContrast")) {
        return QImage();
    }
    return Internal::AutoContrast(image);
}

void NormalizeImage(const QImage& image, QImage& output,
                    double outputMin, double outputMax) {
    if (!RequireGrayU8(image, "NormalizeImage")) {
        output = QImage();
        return;
    }
    if (!std::isfinite(outputMin) || !std::isfinite(outputMax) || outputMax <= outputMin) {
        throw InvalidArgumentException("NormalizeImage: outputMax must be > outputMin");
    }
    output = Internal::NormalizeImage(image, outputMin, outputMax);
}

QImage NormalizeImage(const QImage& image, double outputMin, double outputMax) {
    if (!RequireGrayU8(image, "NormalizeImage")) {
        return QImage();
    }
    if (!std::isfinite(outputMin) || !std::isfinite(outputMax) || outputMax <= outputMin) {
        throw InvalidArgumentException("NormalizeImage: outputMax must be > outputMin");
    }
    return Internal::NormalizeImage(image, outputMin, outputMax);
}

void HistogramMatch(const QImage& image, QImage& output,
                    const QImage& reference) {
    if (!RequireGrayU8(image, "HistogramMatch") ||
        !RequireGrayU8(reference, "HistogramMatch")) {
        output = QImage();
        return;
    }
    output = Internal::HistogramMatchToImage(image, reference);
}

QImage HistogramMatch(const QImage& image, const QImage& reference) {
    if (!RequireGrayU8(image, "HistogramMatch") ||
        !RequireGrayU8(reference, "HistogramMatch")) {
        return QImage();
    }
    return Internal::HistogramMatchToImage(image, reference);
}

} // namespace Qi::Vision::Filter
