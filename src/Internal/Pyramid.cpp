/**
 * @file Pyramid.cpp
 * @brief Image pyramid implementation
 */

#include <QiVision/Internal/Pyramid.h>
#include <QiVision/Internal/Gaussian.h>
#include <QiVision/Internal/Gradient.h>
#include <QiVision/Internal/Interpolate.h>

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace Qi::Vision::Internal {

// ============================================================================
// ImagePyramid Implementation
// ============================================================================

const PyramidLevel& ImagePyramid::GetLevel(int32_t level) const {
    if (level < 0 || level >= static_cast<int32_t>(levels_.size())) {
        throw std::out_of_range("Pyramid level index out of range");
    }
    return levels_[level];
}

PyramidLevel& ImagePyramid::GetLevel(int32_t level) {
    if (level < 0 || level >= static_cast<int32_t>(levels_.size())) {
        throw std::out_of_range("Pyramid level index out of range");
    }
    return levels_[level];
}

const PyramidLevel& ImagePyramid::GetLevelByScale(double scale) const {
    if (levels_.empty()) {
        throw std::runtime_error("Pyramid is empty");
    }

    // Find the level with closest scale
    int32_t bestLevel = 0;
    double bestDiff = std::abs(levels_[0].scale - scale);

    for (size_t i = 1; i < levels_.size(); ++i) {
        double diff = std::abs(levels_[i].scale - scale);
        if (diff < bestDiff) {
            bestDiff = diff;
            bestLevel = static_cast<int32_t>(i);
        }
    }

    return levels_[bestLevel];
}

// ============================================================================
// GradientPyramid Implementation
// ============================================================================

const GradientPyramidLevel& GradientPyramid::GetLevel(int32_t level) const {
    if (level < 0 || level >= static_cast<int32_t>(levels_.size())) {
        throw std::out_of_range("Gradient pyramid level index out of range");
    }
    return levels_[level];
}

GradientPyramidLevel& GradientPyramid::GetLevel(int32_t level) {
    if (level < 0 || level >= static_cast<int32_t>(levels_.size())) {
        throw std::out_of_range("Gradient pyramid level index out of range");
    }
    return levels_[level];
}

// ============================================================================
// Helper Functions
// ============================================================================

namespace {

// Apply 1D Gaussian filter
void GaussianFilter1D(const float* src, float* dst, int32_t length,
                      const std::vector<double>& kernel) {
    const int32_t radius = static_cast<int32_t>(kernel.size()) / 2;

    for (int32_t i = 0; i < length; ++i) {
        double sum = 0.0;
        for (int32_t k = -radius; k <= radius; ++k) {
            int32_t idx = i + k;
            // Reflect101 border handling
            if (idx < 0) idx = -idx;
            if (idx >= length) idx = 2 * length - 2 - idx;
            idx = std::clamp(idx, 0, length - 1);
            sum += src[idx] * kernel[k + radius];
        }
        dst[i] = static_cast<float>(sum);
    }
}

// Separable Gaussian blur
void GaussianBlur(const float* src, float* dst,
                  int32_t width, int32_t height,
                  double sigma) {
    auto kernel = Gaussian::Kernel1D(sigma);

    std::vector<float> temp(width * height);

    // Horizontal pass
    for (int32_t y = 0; y < height; ++y) {
        GaussianFilter1D(src + y * width, temp.data() + y * width, width, kernel);
    }

    // Vertical pass - need to work column by column
    std::vector<float> col(height);
    std::vector<float> colOut(height);

    for (int32_t x = 0; x < width; ++x) {
        // Extract column
        for (int32_t y = 0; y < height; ++y) {
            col[y] = temp[y * width + x];
        }

        // Filter column
        GaussianFilter1D(col.data(), colOut.data(), height, kernel);

        // Store result
        for (int32_t y = 0; y < height; ++y) {
            dst[y * width + x] = colOut[y];
        }
    }
}

} // anonymous namespace

// ============================================================================
// Core Functions
// ============================================================================

int32_t ComputeNumLevels(int32_t width, int32_t height,
                         double scaleFactor,
                         int32_t minDimension) {
    if (scaleFactor <= 0.0 || scaleFactor >= 1.0) {
        return 1;
    }

    int32_t levels = 1;
    int32_t w = width;
    int32_t h = height;

    while (levels < MAX_PYRAMID_LEVELS) {
        w = static_cast<int32_t>(w * scaleFactor);
        h = static_cast<int32_t>(h * scaleFactor);

        if (w < minDimension || h < minDimension) {
            break;
        }
        levels++;
    }

    return levels;
}

void GetLevelDimensions(int32_t originalWidth, int32_t originalHeight,
                        int32_t level, double scaleFactor,
                        int32_t& levelWidth, int32_t& levelHeight) {
    double scale = std::pow(scaleFactor, level);
    levelWidth = std::max(1, static_cast<int32_t>(originalWidth * scale));
    levelHeight = std::max(1, static_cast<int32_t>(originalHeight * scale));
}

void ConvertCoordinates(double x, double y,
                        int32_t srcLevel, int32_t dstLevel,
                        double scaleFactor,
                        double& dstX, double& dstY) {
    double relativeScale = std::pow(scaleFactor, dstLevel - srcLevel);
    dstX = x * relativeScale;
    dstY = y * relativeScale;
}

// ============================================================================
// Downsampling
// ============================================================================

void DownsampleBy2(const float* src, int32_t srcWidth, int32_t srcHeight,
                   float* dst,
                   double sigma,
                   DownsampleMethod method) {
    const int32_t dstWidth = srcWidth / 2;
    const int32_t dstHeight = srcHeight / 2;

    if (dstWidth <= 0 || dstHeight <= 0) {
        return;
    }

    switch (method) {
        case DownsampleMethod::Skip: {
            // Take every other pixel
            for (int32_t y = 0; y < dstHeight; ++y) {
                for (int32_t x = 0; x < dstWidth; ++x) {
                    dst[y * dstWidth + x] = src[(y * 2) * srcWidth + (x * 2)];
                }
            }
            break;
        }

        case DownsampleMethod::Average: {
            // Average 2x2 block
            for (int32_t y = 0; y < dstHeight; ++y) {
                for (int32_t x = 0; x < dstWidth; ++x) {
                    int32_t sx = x * 2;
                    int32_t sy = y * 2;
                    float sum = src[sy * srcWidth + sx] +
                                src[sy * srcWidth + sx + 1] +
                                src[(sy + 1) * srcWidth + sx] +
                                src[(sy + 1) * srcWidth + sx + 1];
                    dst[y * dstWidth + x] = sum * 0.25f;
                }
            }
            break;
        }

        case DownsampleMethod::Gaussian: {
            // Gaussian blur then subsample
            std::vector<float> blurred(srcWidth * srcHeight);
            GaussianBlur(src, blurred.data(), srcWidth, srcHeight, sigma);

            // Subsample
            for (int32_t y = 0; y < dstHeight; ++y) {
                for (int32_t x = 0; x < dstWidth; ++x) {
                    dst[y * dstWidth + x] = blurred[(y * 2) * srcWidth + (x * 2)];
                }
            }
            break;
        }
    }
}

// ============================================================================
// Upsampling
// ============================================================================

void UpsampleBy2(const float* src, int32_t srcWidth, int32_t srcHeight,
                 float* dst,
                 UpsampleMethod method) {
    const int32_t dstWidth = srcWidth * 2;
    const int32_t dstHeight = srcHeight * 2;

    switch (method) {
        case UpsampleMethod::NearestNeighbor: {
            for (int32_t y = 0; y < dstHeight; ++y) {
                for (int32_t x = 0; x < dstWidth; ++x) {
                    int32_t sx = x / 2;
                    int32_t sy = y / 2;
                    dst[y * dstWidth + x] = src[sy * srcWidth + sx];
                }
            }
            break;
        }

        case UpsampleMethod::Bilinear: {
            for (int32_t y = 0; y < dstHeight; ++y) {
                for (int32_t x = 0; x < dstWidth; ++x) {
                    double sx = (x + 0.5) * 0.5 - 0.5;
                    double sy = (y + 0.5) * 0.5 - 0.5;
                    dst[y * dstWidth + x] = InterpolateBilinear(
                        src, srcWidth, srcHeight, sx, sy);
                }
            }
            break;
        }

        case UpsampleMethod::Bicubic: {
            for (int32_t y = 0; y < dstHeight; ++y) {
                for (int32_t x = 0; x < dstWidth; ++x) {
                    double sx = (x + 0.5) * 0.5 - 0.5;
                    double sy = (y + 0.5) * 0.5 - 0.5;
                    dst[y * dstWidth + x] = InterpolateBicubic(
                        src, srcWidth, srcHeight, sx, sy);
                }
            }
            break;
        }
    }
}

// ============================================================================
// Gaussian Pyramid
// ============================================================================

ImagePyramid BuildGaussianPyramid(const QImage& image,
                                   const PyramidParams& params) {
    if (image.Empty()) {
        return ImagePyramid();
    }

    const int32_t width = image.Width();
    const int32_t height = image.Height();
    const int32_t stride = image.Stride();  // May differ from width due to alignment

    // Convert image to float, handling stride correctly
    std::vector<float> floatData(width * height);
    const uint8_t* srcData = static_cast<const uint8_t*>(image.Data());

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            floatData[y * width + x] = static_cast<float>(srcData[y * stride + x]);
        }
    }

    return BuildGaussianPyramid(floatData.data(), width, height, params);
}

ImagePyramid BuildGaussianPyramid(const float* src, int32_t width, int32_t height,
                                   const PyramidParams& params) {
    ImagePyramid pyramid;

    if (!src || width <= 0 || height <= 0) {
        return pyramid;
    }

    // Determine number of levels
    int32_t numLevels = params.numLevels;
    if (numLevels <= 0) {
        numLevels = ComputeNumLevels(width, height,
                                     params.scaleFactor,
                                     params.minDimension);
    }
    numLevels = std::min(numLevels, MAX_PYRAMID_LEVELS);

    // Level 0 is the original image
    PyramidLevel level0;
    level0.width = width;
    level0.height = height;
    level0.scale = 1.0;
    level0.level = 0;
    level0.data.assign(src, src + width * height);
    pyramid.AddLevel(std::move(level0));

    // Build subsequent levels
    int32_t currentWidth = width;
    int32_t currentHeight = height;
    const float* currentData = src;
    std::vector<float> tempBuffer;

    for (int32_t lvl = 1; lvl < numLevels; ++lvl) {
        int32_t nextWidth = currentWidth / 2;
        int32_t nextHeight = currentHeight / 2;

        if (nextWidth < params.minDimension || nextHeight < params.minDimension) {
            break;
        }

        PyramidLevel level;
        level.width = nextWidth;
        level.height = nextHeight;
        level.scale = std::pow(params.scaleFactor, lvl);
        level.level = lvl;
        level.data.resize(nextWidth * nextHeight);

        // Downsample from previous level
        DownsampleBy2(pyramid.GetLevel(lvl - 1).data.data(),
                      currentWidth, currentHeight,
                      level.data.data(),
                      params.sigma,
                      params.downsample);

        currentWidth = nextWidth;
        currentHeight = nextHeight;

        pyramid.AddLevel(std::move(level));
    }

    return pyramid;
}

// ============================================================================
// Laplacian Pyramid
// ============================================================================

ImagePyramid BuildLaplacianPyramid(const QImage& image,
                                    const PyramidParams& params) {
    // First build Gaussian pyramid
    ImagePyramid gaussian = BuildGaussianPyramid(image, params);

    // Convert to Laplacian
    return GaussianToLaplacian(gaussian);
}

ImagePyramid GaussianToLaplacian(const ImagePyramid& gaussian) {
    ImagePyramid laplacian;

    if (gaussian.Empty()) {
        return laplacian;
    }

    const int32_t numLevels = gaussian.NumLevels();

    // For each level except the last, compute difference
    for (int32_t lvl = 0; lvl < numLevels - 1; ++lvl) {
        const PyramidLevel& current = gaussian.GetLevel(lvl);
        const PyramidLevel& next = gaussian.GetLevel(lvl + 1);

        // Upsample the next (coarser) level
        std::vector<float> upsampled(current.width * current.height);
        UpsampleBy2(next.data.data(), next.width, next.height,
                    upsampled.data(), UpsampleMethod::Bilinear);

        // Compute difference (Laplacian = Gaussian - Upsampled(Gaussian_next))
        PyramidLevel lapLevel;
        lapLevel.width = current.width;
        lapLevel.height = current.height;
        lapLevel.scale = current.scale;
        lapLevel.level = lvl;
        lapLevel.data.resize(current.width * current.height);

        for (size_t i = 0; i < current.data.size(); ++i) {
            lapLevel.data[i] = current.data[i] - upsampled[i];
        }

        laplacian.AddLevel(std::move(lapLevel));
    }

    // Last level is just the low-pass residual (copy from Gaussian)
    const PyramidLevel& lastGaussian = gaussian.GetLevel(numLevels - 1);
    PyramidLevel lastLevel;
    lastLevel.width = lastGaussian.width;
    lastLevel.height = lastGaussian.height;
    lastLevel.scale = lastGaussian.scale;
    lastLevel.level = numLevels - 1;
    lastLevel.data = lastGaussian.data;
    laplacian.AddLevel(std::move(lastLevel));

    return laplacian;
}

PyramidLevel ReconstructFromLaplacian(const ImagePyramid& laplacian,
                                       UpsampleMethod method) {
    if (laplacian.Empty()) {
        return PyramidLevel();
    }

    const int32_t numLevels = laplacian.NumLevels();

    // Start with the coarsest level (low-pass residual)
    std::vector<float> current = laplacian.GetLevel(numLevels - 1).data;
    int32_t currentWidth = laplacian.GetLevel(numLevels - 1).width;
    int32_t currentHeight = laplacian.GetLevel(numLevels - 1).height;

    // Reconstruct from coarse to fine
    for (int32_t lvl = numLevels - 2; lvl >= 0; --lvl) {
        const PyramidLevel& lapLevel = laplacian.GetLevel(lvl);

        // Upsample current reconstruction
        std::vector<float> upsampled(lapLevel.width * lapLevel.height);
        UpsampleBy2(current.data(), currentWidth, currentHeight,
                    upsampled.data(), method);

        // Add Laplacian detail
        current.resize(lapLevel.width * lapLevel.height);
        for (size_t i = 0; i < current.size(); ++i) {
            current[i] = upsampled[i] + lapLevel.data[i];
        }

        currentWidth = lapLevel.width;
        currentHeight = lapLevel.height;
    }

    // Return reconstructed level
    PyramidLevel result;
    result.width = currentWidth;
    result.height = currentHeight;
    result.scale = 1.0;
    result.level = 0;
    result.data = std::move(current);

    return result;
}

QImage BlendLaplacian(const QImage& img1, const QImage& img2,
                      const QImage& mask, int32_t numLevels) {
    if (img1.Empty() || img2.Empty() || mask.Empty()) {
        return QImage();
    }

    const int32_t width = img1.Width();
    const int32_t height = img1.Height();

    // Build Laplacian pyramids for both images
    PyramidParams params = PyramidParams::WithLevels(numLevels);
    ImagePyramid lap1 = BuildLaplacianPyramid(img1, params);
    ImagePyramid lap2 = BuildLaplacianPyramid(img2, params);

    // Build Gaussian pyramid for mask
    ImagePyramid maskPyr = BuildGaussianPyramid(mask, params);

    // Blend each level
    ImagePyramid blendedLap;
    for (int32_t lvl = 0; lvl < lap1.NumLevels(); ++lvl) {
        const PyramidLevel& l1 = lap1.GetLevel(lvl);
        const PyramidLevel& l2 = lap2.GetLevel(lvl);
        const PyramidLevel& m = maskPyr.GetLevel(lvl);

        PyramidLevel blended;
        blended.width = l1.width;
        blended.height = l1.height;
        blended.scale = l1.scale;
        blended.level = lvl;
        blended.data.resize(l1.width * l1.height);

        for (size_t i = 0; i < blended.data.size(); ++i) {
            float alpha = m.data[i] / 255.0f;
            blended.data[i] = l1.data[i] * (1.0f - alpha) + l2.data[i] * alpha;
        }

        blendedLap.AddLevel(std::move(blended));
    }

    // Reconstruct from blended Laplacian
    PyramidLevel result = ReconstructFromLaplacian(blendedLap);

    // Convert to QImage
    return PyramidLevelToImage(result, true);
}

// ============================================================================
// Gradient Pyramid
// ============================================================================

GradientPyramid BuildGradientPyramid(const QImage& image,
                                      const PyramidParams& params) {
    // First build Gaussian pyramid
    ImagePyramid gaussian = BuildGaussianPyramid(image, params);

    // Convert to gradient pyramid
    return GaussianToGradient(gaussian);
}

GradientPyramid GaussianToGradient(const ImagePyramid& gaussian) {
    GradientPyramid gradPyr;

    for (int32_t lvl = 0; lvl < gaussian.NumLevels(); ++lvl) {
        const PyramidLevel& level = gaussian.GetLevel(lvl);

        GradientPyramidLevel gradLevel;
        gradLevel.width = level.width;
        gradLevel.height = level.height;
        gradLevel.scale = level.scale;
        gradLevel.level = lvl;
        gradLevel.magnitude.resize(level.width * level.height);
        gradLevel.direction.resize(level.width * level.height);

        // Compute gradient using existing gradient functions
        std::vector<float> gx(level.width * level.height);
        std::vector<float> gy(level.width * level.height);

        Gradient<float, float>(level.data.data(), gx.data(), gy.data(),
                               level.width, level.height,
                               GradientOperator::Sobel3x3);

        // Compute magnitude and direction
        for (size_t i = 0; i < gradLevel.magnitude.size(); ++i) {
            float dx = gx[i];
            float dy = gy[i];
            gradLevel.magnitude[i] = std::sqrt(dx * dx + dy * dy);
            gradLevel.direction[i] = std::atan2(dy, dx);
        }

        gradPyr.AddLevel(std::move(gradLevel));
    }

    return gradPyr;
}

// ============================================================================
// Utility Functions
// ============================================================================

QImage PyramidLevelToImage(const PyramidLevel& level, bool normalize) {
    if (!level.IsValid()) {
        return QImage();
    }

    QImage result(level.width, level.height, PixelType::UInt8);
    uint8_t* dst = static_cast<uint8_t*>(result.Data());
    const int32_t stride = result.Stride();

    if (normalize) {
        // Find min/max
        float minVal = level.data[0];
        float maxVal = level.data[0];
        for (float v : level.data) {
            minVal = std::min(minVal, v);
            maxVal = std::max(maxVal, v);
        }

        float range = maxVal - minVal;
        if (range < 1e-6f) {
            range = 1.0f;
        }

        // Handle stride correctly - QImage may have 64-byte row alignment
        for (int32_t y = 0; y < level.height; ++y) {
            for (int32_t x = 0; x < level.width; ++x) {
                float normalized = (level.data[y * level.width + x] - minVal) / range * 255.0f;
                dst[y * stride + x] = static_cast<uint8_t>(std::clamp(normalized, 0.0f, 255.0f));
            }
        }
    } else {
        // Handle stride correctly - QImage may have 64-byte row alignment
        for (int32_t y = 0; y < level.height; ++y) {
            for (int32_t x = 0; x < level.width; ++x) {
                dst[y * stride + x] = static_cast<uint8_t>(
                    std::clamp(level.data[y * level.width + x], 0.0f, 255.0f));
            }
        }
    }

    return result;
}

PyramidLevel ImageToPyramidLevel(const QImage& image, int32_t levelIndex,
                                  double scale) {
    PyramidLevel level;

    if (image.Empty()) {
        return level;
    }

    level.width = image.Width();
    level.height = image.Height();
    level.scale = scale;
    level.level = levelIndex;
    level.data.resize(level.width * level.height);

    const uint8_t* src = static_cast<const uint8_t*>(image.Data());
    const int32_t stride = image.Stride();

    // Handle stride correctly - QImage may have 64-byte row alignment
    for (int32_t y = 0; y < level.height; ++y) {
        for (int32_t x = 0; x < level.width; ++x) {
            level.data[y * level.width + x] = static_cast<float>(src[y * stride + x]);
        }
    }

    return level;
}

float SamplePyramidAtScale(const ImagePyramid& pyramid,
                           double x, double y, double scale) {
    if (pyramid.Empty()) {
        return 0.0f;
    }

    // Find the two levels that bracket the desired scale
    int32_t lowerLevel = -1;
    int32_t upperLevel = -1;

    for (int32_t lvl = 0; lvl < pyramid.NumLevels(); ++lvl) {
        if (pyramid.GetLevel(lvl).scale >= scale) {
            lowerLevel = lvl;
        }
        if (pyramid.GetLevel(lvl).scale <= scale && upperLevel < 0) {
            upperLevel = lvl;
        }
    }

    if (lowerLevel < 0) lowerLevel = 0;
    if (upperLevel < 0) upperLevel = pyramid.NumLevels() - 1;

    if (lowerLevel == upperLevel) {
        // Just sample from this level
        const PyramidLevel& level = pyramid.GetLevel(lowerLevel);
        double lx = x * level.scale;
        double ly = y * level.scale;
        return InterpolateBilinear(level.data.data(), level.width, level.height,
                                   lx, ly);
    }

    // Interpolate between levels
    const PyramidLevel& lower = pyramid.GetLevel(lowerLevel);
    const PyramidLevel& upper = pyramid.GetLevel(upperLevel);

    double lx1 = x * lower.scale;
    double ly1 = y * lower.scale;
    float val1 = InterpolateBilinear(lower.data.data(), lower.width, lower.height,
                                     lx1, ly1);

    double lx2 = x * upper.scale;
    double ly2 = y * upper.scale;
    float val2 = InterpolateBilinear(upper.data.data(), upper.width, upper.height,
                                     lx2, ly2);

    // Linear interpolation between levels
    double t = (scale - lower.scale) / (upper.scale - lower.scale);
    return static_cast<float>(val1 * (1.0 - t) + val2 * t);
}

std::vector<double> ComputeSearchScales(int32_t modelWidth, int32_t modelHeight,
                                         double minScale, double maxScale,
                                         double scaleStep) {
    std::vector<double> scales;

    if (scaleStep <= 0.0 || minScale > maxScale) {
        scales.push_back(1.0);
        return scales;
    }

    for (double s = minScale; s <= maxScale; s += scaleStep) {
        scales.push_back(s);
    }

    // Ensure we include maxScale
    if (scales.empty() || scales.back() < maxScale - scaleStep * 0.5) {
        scales.push_back(maxScale);
    }

    return scales;
}

} // namespace Qi::Vision::Internal
