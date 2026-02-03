/**
 * @file Texture.cpp
 * @brief Implementation of texture analysis module
 */

#include <QiVision/Texture/Texture.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Core/Validate.h>
#include <QiVision/Internal/Convolution.h>

#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace Qi::Vision::Texture {

// =============================================================================
// LBP Implementation
// =============================================================================

namespace {

int32_t LbpNumBins(LBPType type) {
    switch (type) {
        case LBPType::Standard: return 256;
        case LBPType::Uniform: return 59;
        case LBPType::RotationInvariant: return 36;
        case LBPType::UniformRI: return 10;
        default: return 256;
    }
}

} // anonymous namespace

// LBP lookup table for uniform patterns
static const int32_t LBP_UNIFORM_TABLE[256] = {
    0, 1, 1, 2, 1, 58, 2, 3, 1, 58, 58, 58, 2, 58, 3, 4,
    1, 58, 58, 58, 58, 58, 58, 58, 2, 58, 58, 58, 3, 58, 4, 5,
    1, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    2, 58, 58, 58, 58, 58, 58, 58, 3, 58, 58, 58, 4, 58, 5, 6,
    1, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    2, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    3, 58, 58, 58, 58, 58, 58, 58, 4, 58, 58, 58, 5, 58, 6, 7,
    1, 2, 58, 3, 58, 58, 58, 4, 58, 58, 58, 58, 58, 58, 58, 5,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 6,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 7,
    2, 3, 58, 4, 58, 58, 58, 5, 58, 58, 58, 58, 58, 58, 58, 6,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 7,
    3, 4, 58, 5, 58, 58, 58, 6, 58, 58, 58, 58, 58, 58, 58, 7,
    4, 5, 58, 6, 58, 58, 58, 7, 5, 6, 58, 7, 6, 7, 7, 8
};

// Count number of 1->0 or 0->1 transitions in circular bit pattern
static int32_t CountTransitions(uint8_t pattern) {
    int32_t transitions = 0;
    uint8_t prev = pattern & 0x80;
    for (int i = 0; i < 8; ++i) {
        uint8_t curr = (pattern << i) & 0x80;
        if (curr != prev) transitions++;
        prev = curr;
    }
    // Check wrap-around
    if (((pattern & 1) != 0) != ((pattern & 0x80) != 0)) transitions++;
    return transitions;
}

// Rotation invariant: minimum value of all rotations
static uint8_t RotationInvariant(uint8_t pattern) {
    uint8_t minVal = pattern;
    for (int i = 1; i < 8; ++i) {
        uint8_t rotated = ((pattern >> i) | (pattern << (8 - i))) & 0xFF;
        minVal = std::min(minVal, rotated);
    }
    return minVal;
}

// Rotation invariant mapping table
static int32_t GetRIMapping(uint8_t pattern) {
    static bool initialized = false;
    static int32_t riTable[256];

    if (!initialized) {
        // Map each pattern to its rotation-invariant class
        std::vector<uint8_t> representatives;
        std::fill(riTable, riTable + 256, -1);

        for (int p = 0; p < 256; ++p) {
            uint8_t ri = RotationInvariant(static_cast<uint8_t>(p));

            // Find or create class
            auto it = std::find(representatives.begin(), representatives.end(), ri);
            if (it == representatives.end()) {
                riTable[p] = static_cast<int32_t>(representatives.size());
                representatives.push_back(ri);
            } else {
                riTable[p] = static_cast<int32_t>(it - representatives.begin());
            }
        }
        initialized = true;
    }

    return riTable[pattern];
}

void ComputeLBP(const QImage& image, QImage& lbpImage, LBPType type) {
    if (!Validate::RequireImageU8Gray(image, "ComputeLBP")) {
        lbpImage = QImage();
        return;
    }

    int32_t width = image.Width();
    int32_t height = image.Height();

    lbpImage = QImage(width, height, PixelType::UInt8);

    const uint8_t* src = static_cast<const uint8_t*>(image.Data());
    uint8_t* dst = static_cast<uint8_t*>(lbpImage.Data());
    int32_t srcStride = image.Stride();
    int32_t dstStride = lbpImage.Stride();

    // Set border pixels to 0
    for (int32_t x = 0; x < width; ++x) {
        dst[x] = 0;
        dst[(height - 1) * dstStride + x] = 0;
    }
    for (int32_t y = 0; y < height; ++y) {
        dst[y * dstStride] = 0;
        dst[y * dstStride + width - 1] = 0;
    }

    // Compute LBP for interior pixels
    for (int32_t y = 1; y < height - 1; ++y) {
        for (int32_t x = 1; x < width - 1; ++x) {
            uint8_t center = src[y * srcStride + x];
            uint8_t code = 0;

            // 8-connectivity pattern (clockwise from top-left)
            // 7 0 1
            // 6 c 2
            // 5 4 3
            if (src[(y - 1) * srcStride + x] >= center) code |= (1 << 0);
            if (src[(y - 1) * srcStride + x + 1] >= center) code |= (1 << 1);
            if (src[y * srcStride + x + 1] >= center) code |= (1 << 2);
            if (src[(y + 1) * srcStride + x + 1] >= center) code |= (1 << 3);
            if (src[(y + 1) * srcStride + x] >= center) code |= (1 << 4);
            if (src[(y + 1) * srcStride + x - 1] >= center) code |= (1 << 5);
            if (src[y * srcStride + x - 1] >= center) code |= (1 << 6);
            if (src[(y - 1) * srcStride + x - 1] >= center) code |= (1 << 7);

            // Apply variant
            switch (type) {
                case LBPType::Standard:
                    break;
                case LBPType::Uniform:
                    code = static_cast<uint8_t>(LBP_UNIFORM_TABLE[code]);
                    break;
                case LBPType::RotationInvariant:
                    code = static_cast<uint8_t>(GetRIMapping(code));
                    break;
                case LBPType::UniformRI:
                    // Uniform + RI: only 10 patterns
                    if (CountTransitions(code) <= 2) {
                        // Count number of 1s
                        code = static_cast<uint8_t>(__builtin_popcount(code));
                    } else {
                        code = 9; // Non-uniform
                    }
                    break;
            }

            dst[y * dstStride + x] = code;
        }
    }
}

void ComputeLBPExtended(const QImage& image, QImage& lbpImage,
                        int32_t radius, int32_t numPoints, LBPType type) {
    if (!Validate::RequireImageU8Gray(image, "ComputeLBPExtended")) {
        lbpImage = QImage();
        return;
    }
    if (radius < 1 || numPoints < 4) {
        throw InvalidArgumentException("ComputeLBPExtended: invalid radius or numPoints");
    }

    // For now, use simple version for radius=1, numPoints=8
    if (radius == 1 && numPoints == 8) {
        ComputeLBP(image, lbpImage, type);
        return;
    }

    int32_t width = image.Width();
    int32_t height = image.Height();

    lbpImage = QImage(width, height, PixelType::UInt8);

    const uint8_t* src = static_cast<const uint8_t*>(image.Data());
    uint8_t* dst = static_cast<uint8_t*>(lbpImage.Data());
    int32_t srcStride = image.Stride();
    int32_t dstStride = lbpImage.Stride();

    // Precompute sample positions
    std::vector<double> sampleX(numPoints), sampleY(numPoints);
    for (int32_t i = 0; i < numPoints; ++i) {
        double angle = 2.0 * 3.14159265358979323846 * i / numPoints;
        sampleX[i] = radius * std::cos(angle);
        sampleY[i] = -radius * std::sin(angle);  // -sin for image coords
    }

    // Clear border
    std::memset(dst, 0, dstStride * height);

    // Process interior
    for (int32_t y = radius; y < height - radius; ++y) {
        for (int32_t x = radius; x < width - radius; ++x) {
            double center = src[y * srcStride + x];
            uint32_t code = 0;

            for (int32_t i = 0; i < numPoints; ++i) {
                double sx = x + sampleX[i];
                double sy = y + sampleY[i];

                // Bilinear interpolation
                int32_t x0 = static_cast<int32_t>(std::floor(sx));
                int32_t y0 = static_cast<int32_t>(std::floor(sy));
                double fx = sx - x0;
                double fy = sy - y0;

                double val = (1 - fx) * (1 - fy) * src[y0 * srcStride + x0] +
                             fx * (1 - fy) * src[y0 * srcStride + x0 + 1] +
                             (1 - fx) * fy * src[(y0 + 1) * srcStride + x0] +
                             fx * fy * src[(y0 + 1) * srcStride + x0 + 1];

                if (val >= center) {
                    code |= (1u << i);
                }
            }

            // Map to uint8 (truncate if numPoints > 8)
            dst[y * dstStride + x] = static_cast<uint8_t>(code & 0xFF);
        }
    }
}

int32_t ComputeLBPHistogram(const QImage& lbpImage,
                            std::vector<double>& histogram,
                            LBPType type) {
    int32_t numBins = LbpNumBins(type);
    histogram.assign(numBins, 0.0);

    if (!Validate::RequireImageU8Gray(lbpImage, "ComputeLBPHistogram")) {
        return numBins;
    }

    const uint8_t* data = static_cast<const uint8_t*>(lbpImage.Data());
    int32_t stride = lbpImage.Stride();
    int32_t width = lbpImage.Width();
    int32_t height = lbpImage.Height();
    int64_t count = 0;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            int32_t bin = data[y * stride + x];
            if (bin < numBins) {
                histogram[bin] += 1.0;
                count++;
            }
        }
    }

    // Normalize
    if (count > 0) {
        for (auto& h : histogram) {
            h /= count;
        }
    }

    return numBins;
}

int32_t ComputeLBPHistogram(const QImage& lbpImage,
                            const QRegion& region,
                            std::vector<double>& histogram,
                            LBPType type) {
    int32_t numBins = LbpNumBins(type);
    histogram.assign(numBins, 0.0);

    if (!Validate::RequireImageU8Gray(lbpImage, "ComputeLBPHistogram")) {
        return numBins;
    }
    if (region.Empty()) {
        return numBins;
    }

    histogram.assign(numBins, 0.0);

    const uint8_t* data = static_cast<const uint8_t*>(lbpImage.Data());
    int32_t stride = lbpImage.Stride();
    int64_t count = 0;

    const auto& runs = region.Runs();
    for (const auto& run : runs) {
        for (int32_t x = run.colBegin; x < run.colEnd; ++x) {
            int32_t bin = data[run.row * stride + x];
            if (bin < numBins) {
                histogram[bin] += 1.0;
                count++;
            }
        }
    }

    if (count > 0) {
        for (auto& h : histogram) {
            h /= count;
        }
    }

    return numBins;
}

// =============================================================================
// GLCM Implementation
// =============================================================================

void ComputeGLCM(const QImage& image,
                 std::vector<std::vector<double>>& glcm,
                 int32_t distance,
                 GLCMDirection direction,
                 int32_t numLevels) {
    if (!Validate::RequireImageU8Gray(image, "ComputeGLCM")) {
        glcm.clear();
        return;
    }
    if (distance < 1) {
        throw InvalidArgumentException("ComputeGLCM: distance must be > 0");
    }
    if (numLevels < 2 || numLevels > 256) {
        throw InvalidArgumentException("ComputeGLCM: numLevels must be between 2 and 256");
    }

    int32_t width = image.Width();
    int32_t height = image.Height();
    const uint8_t* data = static_cast<const uint8_t*>(image.Data());
    int32_t stride = image.Stride();

    // Initialize GLCM
    glcm.assign(numLevels, std::vector<double>(numLevels, 0.0));

    // Direction offsets
    int32_t dx = 0, dy = 0;
    switch (direction) {
        case GLCMDirection::Horizontal: dx = distance; dy = 0; break;
        case GLCMDirection::Vertical: dx = 0; dy = distance; break;
        case GLCMDirection::Diagonal45: dx = distance; dy = -distance; break;
        case GLCMDirection::Diagonal135: dx = -distance; dy = -distance; break;
        case GLCMDirection::Average: break; // Handle separately
    }

    if (direction == GLCMDirection::Average) {
        // Compute average of all 4 directions
        std::vector<std::vector<double>> tempGlcm;
        ComputeGLCM(image, tempGlcm, distance, GLCMDirection::Horizontal, numLevels);
        for (int i = 0; i < numLevels; ++i)
            for (int j = 0; j < numLevels; ++j)
                glcm[i][j] += tempGlcm[i][j];

        ComputeGLCM(image, tempGlcm, distance, GLCMDirection::Vertical, numLevels);
        for (int i = 0; i < numLevels; ++i)
            for (int j = 0; j < numLevels; ++j)
                glcm[i][j] += tempGlcm[i][j];

        ComputeGLCM(image, tempGlcm, distance, GLCMDirection::Diagonal45, numLevels);
        for (int i = 0; i < numLevels; ++i)
            for (int j = 0; j < numLevels; ++j)
                glcm[i][j] += tempGlcm[i][j];

        ComputeGLCM(image, tempGlcm, distance, GLCMDirection::Diagonal135, numLevels);
        for (int i = 0; i < numLevels; ++i)
            for (int j = 0; j < numLevels; ++j)
                glcm[i][j] += tempGlcm[i][j];

        // Normalize
        double sum = 0;
        for (int i = 0; i < numLevels; ++i)
            for (int j = 0; j < numLevels; ++j)
                sum += glcm[i][j];
        if (sum > 0) {
            for (int i = 0; i < numLevels; ++i)
                for (int j = 0; j < numLevels; ++j)
                    glcm[i][j] /= sum;
        }
        return;
    }

    // Quantization factor
    double scale = numLevels / 256.0;
    int64_t count = 0;

    // Compute co-occurrences
    for (int32_t y = std::max(0, -dy); y < height - std::max(0, dy); ++y) {
        for (int32_t x = std::max(0, -dx); x < width - std::max(0, dx); ++x) {
            int32_t i = static_cast<int32_t>(data[y * stride + x] * scale);
            int32_t j = static_cast<int32_t>(data[(y + dy) * stride + (x + dx)] * scale);
            i = std::min(i, numLevels - 1);
            j = std::min(j, numLevels - 1);

            glcm[i][j] += 1.0;
            glcm[j][i] += 1.0;  // Symmetric
            count += 2;
        }
    }

    // Normalize
    if (count > 0) {
        for (int i = 0; i < numLevels; ++i) {
            for (int j = 0; j < numLevels; ++j) {
                glcm[i][j] /= count;
            }
        }
    }
}

void ComputeGLCM(const QImage& image,
                 const QRegion& region,
                 std::vector<std::vector<double>>& glcm,
                 int32_t distance,
                 GLCMDirection direction,
                 int32_t numLevels) {
    if (!Validate::RequireImageU8Gray(image, "ComputeGLCM")) {
        glcm.clear();
        return;
    }
    if (distance < 1) {
        throw InvalidArgumentException("ComputeGLCM: distance must be > 0");
    }
    if (numLevels < 2 || numLevels > 256) {
        throw InvalidArgumentException("ComputeGLCM: numLevels must be between 2 and 256");
    }
    if (region.Empty()) {
        glcm.assign(numLevels, std::vector<double>(numLevels, 0.0));
        return;
    }
    // For simplicity, use bounding box of region
    // Full implementation would check region membership
    Rect2i bbox = region.BoundingBox();
    if (bbox.width <= 0 || bbox.height <= 0) {
        glcm.assign(numLevels, std::vector<double>(numLevels, 0.0));
        return;
    }
    QImage subImage = image.SubImage(bbox.x, bbox.y, bbox.width, bbox.height);
    ComputeGLCM(subImage, glcm, distance, direction, numLevels);
}

GLCMFeatures ExtractGLCMFeatures(const std::vector<std::vector<double>>& glcm) {
    GLCMFeatures f;
    int32_t numLevels = static_cast<int32_t>(glcm.size());
    if (numLevels == 0) return f;

    // Compute marginals and mean
    std::vector<double> px(numLevels, 0), py(numLevels, 0);
    for (int i = 0; i < numLevels; ++i) {
        for (int j = 0; j < numLevels; ++j) {
            px[i] += glcm[i][j];
            py[j] += glcm[i][j];
        }
    }

    double meanX = 0, meanY = 0;
    for (int i = 0; i < numLevels; ++i) {
        meanX += i * px[i];
        meanY += i * py[i];
    }

    double varX = 0, varY = 0;
    for (int i = 0; i < numLevels; ++i) {
        varX += (i - meanX) * (i - meanX) * px[i];
        varY += (i - meanY) * (i - meanY) * py[i];
    }

    f.mean = (meanX + meanY) / 2;
    f.variance = (varX + varY) / 2;

    // Compute features
    for (int i = 0; i < numLevels; ++i) {
        for (int j = 0; j < numLevels; ++j) {
            double p = glcm[i][j];
            if (p > 0) {
                int diff = std::abs(i - j);

                f.contrast += diff * diff * p;
                f.dissimilarity += diff * p;
                f.homogeneity += p / (1 + diff * diff);
                f.energy += p * p;
                f.entropy -= p * std::log2(p + 1e-10);
                f.maxProbability = std::max(f.maxProbability, p);

                if (varX > 0 && varY > 0) {
                    f.correlation += (i - meanX) * (j - meanY) * p /
                                     (std::sqrt(varX) * std::sqrt(varY));
                }
            }
        }
    }

    f.asm_ = f.energy;  // Angular Second Moment = Energy

    return f;
}

GLCMFeatures ComputeGLCMFeatures(const QImage& image,
                                  int32_t distance,
                                  GLCMDirection direction,
                                  int32_t numLevels) {
    std::vector<std::vector<double>> glcm;
    ComputeGLCM(image, glcm, distance, direction, numLevels);
    return ExtractGLCMFeatures(glcm);
}

GLCMFeatures ComputeGLCMFeatures(const QImage& image,
                                  const QRegion& region,
                                  int32_t distance,
                                  GLCMDirection direction,
                                  int32_t numLevels) {
    std::vector<std::vector<double>> glcm;
    ComputeGLCM(image, region, glcm, distance, direction, numLevels);
    return ExtractGLCMFeatures(glcm);
}

// =============================================================================
// Gabor Implementation
// =============================================================================

void CreateGaborKernel(const GaborParams& params, QImage& kernel) {
    if (!std::isfinite(params.sigma) || !std::isfinite(params.lambda) ||
        !std::isfinite(params.theta) || !std::isfinite(params.psi) ||
        !std::isfinite(params.gamma) ||
        params.sigma <= 0.0 || params.lambda <= 0.0 || params.gamma <= 0.0) {
        throw InvalidArgumentException("CreateGaborKernel: sigma and lambda must be > 0");
    }
    int32_t size = params.kernelSize;
    if (size <= 0) {
        size = static_cast<int32_t>(std::ceil(params.sigma * 6)) | 1;  // Odd
    }
    if (size < 3) size = 3;

    kernel = QImage(size, size, PixelType::Float32);
    float* data = static_cast<float*>(kernel.Data());
    int32_t stride = kernel.Stride() / sizeof(float);

    int32_t half = size / 2;
    double sigma2 = params.sigma * params.sigma;
    double cosT = std::cos(params.theta);
    double sinT = std::sin(params.theta);

    for (int32_t y = 0; y < size; ++y) {
        for (int32_t x = 0; x < size; ++x) {
            double px = x - half;
            double py = y - half;

            // Rotate
            double xTheta = px * cosT + py * sinT;
            double yTheta = -px * sinT + py * cosT;

            // Gabor function
            double gaussian = std::exp(-(xTheta * xTheta +
                              params.gamma * params.gamma * yTheta * yTheta) /
                              (2 * sigma2));
            double sinusoid = std::cos(2 * 3.14159265358979323846 * xTheta /
                              params.lambda + params.psi);

            data[y * stride + x] = static_cast<float>(gaussian * sinusoid);
        }
    }
}

void ApplyGaborFilter(const QImage& image, QImage& output,
                      const GaborParams& params) {
    if (!Validate::RequireImageU8Gray(image, "ApplyGaborFilter")) {
        output = QImage();
        return;
    }
    QImage kernel;
    CreateGaborKernel(params, kernel);

    // Use convolution (simplified - should use separable if possible)
    int32_t width = image.Width();
    int32_t height = image.Height();
    int32_t kSize = kernel.Width();
    int32_t kHalf = kSize / 2;

    output = QImage(width, height, PixelType::Float32);

    const uint8_t* src = static_cast<const uint8_t*>(image.Data());
    const float* kData = static_cast<const float*>(kernel.Data());
    float* dst = static_cast<float*>(output.Data());

    int32_t srcStride = image.Stride();
    int32_t kStride = kernel.Stride() / sizeof(float);
    int32_t dstStride = output.Stride() / sizeof(float);

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            double sum = 0;

            for (int32_t ky = 0; ky < kSize; ++ky) {
                for (int32_t kx = 0; kx < kSize; ++kx) {
                    int32_t sy = y + ky - kHalf;
                    int32_t sx = x + kx - kHalf;

                    // Reflect border
                    sy = std::max(0, std::min(height - 1, sy));
                    sx = std::max(0, std::min(width - 1, sx));

                    sum += src[sy * srcStride + sx] * kData[ky * kStride + kx];
                }
            }

            dst[y * dstStride + x] = static_cast<float>(sum);
        }
    }
}

void ApplyGaborFilterBank(const QImage& image,
                          std::vector<QImage>& responses,
                          int32_t numOrientations,
                          double sigma,
                          double lambda) {
    Validate::RequirePositive(numOrientations, "numOrientations", "ApplyGaborFilterBank");
    if (!std::isfinite(sigma) || !std::isfinite(lambda) || sigma <= 0.0 || lambda <= 0.0) {
        throw InvalidArgumentException("ApplyGaborFilterBank: sigma/lambda must be > 0");
    }
    if (!Validate::RequireImageU8Gray(image, "ApplyGaborFilterBank")) {
        responses.clear();
        return;
    }
    responses.resize(numOrientations);

    for (int32_t i = 0; i < numOrientations; ++i) {
        GaborParams params;
        params.sigma = sigma;
        params.lambda = lambda;
        params.theta = 3.14159265358979323846 * i / numOrientations;

        ApplyGaborFilter(image, responses[i], params);
    }
}

void ComputeGaborEnergy(const QImage& image, QImage& energy,
                        const GaborParams& params) {
    if (!Validate::RequireImageU8Gray(image, "ComputeGaborEnergy")) {
        energy = QImage();
        return;
    }
    // Compute real (0° phase) and imaginary (90° phase) responses
    QImage realResp, imagResp;

    GaborParams realParams = params;
    realParams.psi = 0;
    ApplyGaborFilter(image, realResp, realParams);

    GaborParams imagParams = params;
    imagParams.psi = 3.14159265358979323846 / 2;
    ApplyGaborFilter(image, imagResp, imagParams);

    // Compute energy = sqrt(real² + imag²)
    int32_t width = image.Width();
    int32_t height = image.Height();
    energy = QImage(width, height, PixelType::Float32);

    const float* realData = static_cast<const float*>(realResp.Data());
    const float* imagData = static_cast<const float*>(imagResp.Data());
    float* outData = static_cast<float*>(energy.Data());

    int32_t stride = energy.Stride() / sizeof(float);

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            float r = realData[y * stride + x];
            float i = imagData[y * stride + x];
            outData[y * stride + x] = std::sqrt(r * r + i * i);
        }
    }
}

GaborFeatures ExtractGaborFeatures(const QImage& image,
                                    int32_t numOrientations,
                                    double sigma,
                                    double lambda) {
    GaborFeatures f;
    Validate::RequirePositive(numOrientations, "numOrientations", "ExtractGaborFeatures");
    if (!std::isfinite(sigma) || !std::isfinite(lambda) || sigma <= 0.0 || lambda <= 0.0) {
        throw InvalidArgumentException("ExtractGaborFeatures: sigma/lambda must be > 0");
    }
    f.meanEnergy.resize(numOrientations);
    f.stdEnergy.resize(numOrientations);

    if (!Validate::RequireImageU8Gray(image, "ExtractGaborFeatures")) {
        f.dominantOrientation = 0.0;
        f.orientationStrength = 0.0;
        return f;
    }

    int32_t width = image.Width();
    int32_t height = image.Height();
    int64_t numPixels = static_cast<int64_t>(width) * height;

    double maxMean = 0;
    int32_t maxIdx = 0;

    for (int32_t i = 0; i < numOrientations; ++i) {
        GaborParams params;
        params.sigma = sigma;
        params.lambda = lambda;
        params.theta = 3.14159265358979323846 * i / numOrientations;

        QImage energy;
        ComputeGaborEnergy(image, energy, params);

        // Compute statistics
        const float* data = static_cast<const float*>(energy.Data());
        int32_t stride = energy.Stride() / sizeof(float);

        double sum = 0, sumSq = 0;
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                double val = data[y * stride + x];
                sum += val;
                sumSq += val * val;
            }
        }

        double mean = sum / numPixels;
        double variance = (sumSq / numPixels) - (mean * mean);

        f.meanEnergy[i] = mean;
        f.stdEnergy[i] = std::sqrt(std::max(0.0, variance));

        if (mean > maxMean) {
            maxMean = mean;
            maxIdx = i;
        }
    }

    f.dominantOrientation = 180.0 * maxIdx / numOrientations;
    f.orientationStrength = maxMean;

    return f;
}

GaborFeatures ExtractGaborFeatures(const QImage& image,
                                    const QRegion& region,
                                    int32_t numOrientations,
                                    double sigma,
                                    double lambda) {
    Validate::RequirePositive(numOrientations, "numOrientations", "ExtractGaborFeatures");
    if (!std::isfinite(sigma) || !std::isfinite(lambda) || sigma <= 0.0 || lambda <= 0.0) {
        throw InvalidArgumentException("ExtractGaborFeatures: sigma/lambda must be > 0");
    }
    GaborFeatures f;
    f.meanEnergy.resize(numOrientations);
    f.stdEnergy.resize(numOrientations);

    if (!Validate::RequireImageU8Gray(image, "ExtractGaborFeatures") || region.Empty()) {
        return f;
    }
    // Use bounding box for simplicity
    Rect2i bbox = region.BoundingBox();
    if (bbox.width <= 0 || bbox.height <= 0) {
        GaborFeatures f;
        f.meanEnergy.resize(numOrientations);
        f.stdEnergy.resize(numOrientations);
        return f;
    }
    QImage subImage = image.SubImage(bbox.x, bbox.y, bbox.width, bbox.height);
    return ExtractGaborFeatures(subImage, numOrientations, sigma, lambda);
}

// =============================================================================
// Texture Comparison
// =============================================================================

double CompareLBPHistograms(const std::vector<double>& hist1,
                            const std::vector<double>& hist2) {
    if (hist1.size() != hist2.size()) {
        throw InvalidArgumentException("CompareLBPHistograms: histogram sizes don't match");
    }

    // Chi-square distance
    double chiSq = 0;
    for (size_t i = 0; i < hist1.size(); ++i) {
        double sum = hist1[i] + hist2[i];
        if (sum > 0) {
            double diff = hist1[i] - hist2[i];
            chiSq += (diff * diff) / sum;
        }
    }
    return chiSq / 2;
}

double CompareGLCMFeatures(const GLCMFeatures& f1, const GLCMFeatures& f2) {
    // Euclidean distance in normalized feature space
    auto norm = [](double v, double scale) { return v / scale; };

    double d = 0;
    d += std::pow(norm(f1.contrast - f2.contrast, 100), 2);
    d += std::pow(norm(f1.dissimilarity - f2.dissimilarity, 50), 2);
    d += std::pow(f1.homogeneity - f2.homogeneity, 2);
    d += std::pow(f1.energy - f2.energy, 2);
    d += std::pow(norm(f1.entropy - f2.entropy, 8), 2);
    d += std::pow(f1.correlation - f2.correlation, 2);

    return std::sqrt(d);
}

double CompareGaborFeatures(const GaborFeatures& f1, const GaborFeatures& f2) {
    if (f1.meanEnergy.size() != f2.meanEnergy.size()) {
        throw InvalidArgumentException("CompareGaborFeatures: feature sizes don't match");
    }

    double d = 0;
    for (size_t i = 0; i < f1.meanEnergy.size(); ++i) {
        d += std::pow(f1.meanEnergy[i] - f2.meanEnergy[i], 2);
        d += std::pow(f1.stdEnergy[i] - f2.stdEnergy[i], 2);
    }

    return std::sqrt(d);
}

// =============================================================================
// Texture Segmentation
// =============================================================================

int32_t SegmentByTextureLBP(const QImage& image, QImage& labels,
                            int32_t numClusters, int32_t windowSize) {
    Validate::RequirePositive(numClusters, "numClusters", "SegmentByTextureLBP");
    Validate::RequirePositive(windowSize, "windowSize", "SegmentByTextureLBP");
    if (!Validate::RequireImageU8Gray(image, "SegmentByTextureLBP")) {
        labels = QImage();
        return 0;
    }

    // Compute LBP
    QImage lbpImage;
    ComputeLBP(image, lbpImage, LBPType::Uniform);

    int32_t width = image.Width();
    int32_t height = image.Height();

    // Compute local histograms and cluster
    // Simplified k-means on histogram features
    int32_t gridX = (width + windowSize - 1) / windowSize;
    int32_t gridY = (height + windowSize - 1) / windowSize;

    // Compute histograms for each cell
    std::vector<std::vector<double>> histograms(gridX * gridY);

    for (int32_t gy = 0; gy < gridY; ++gy) {
        for (int32_t gx = 0; gx < gridX; ++gx) {
            int32_t x0 = gx * windowSize;
            int32_t y0 = gy * windowSize;
            int32_t x1 = std::min(x0 + windowSize, width);
            int32_t y1 = std::min(y0 + windowSize, height);

            std::vector<QRegion::Run> runs;
            for (int32_t y = y0; y < y1; ++y) {
                runs.push_back({y, x0, x1});
            }
            QRegion region(std::move(runs));

            ComputeLBPHistogram(lbpImage, region, histograms[gy * gridX + gx],
                               LBPType::Uniform);
        }
    }

    // Simple k-means clustering
    std::vector<int32_t> cellLabels(gridX * gridY, 0);

    // Initialize centroids (first k cells)
    std::vector<std::vector<double>> centroids(numClusters);
    for (int32_t k = 0; k < numClusters && k < gridX * gridY; ++k) {
        centroids[k] = histograms[k * gridX * gridY / numClusters];
    }

    // Iterate
    for (int iter = 0; iter < 10; ++iter) {
        // Assign
        for (size_t i = 0; i < histograms.size(); ++i) {
            double minDist = 1e10;
            for (int32_t k = 0; k < numClusters; ++k) {
                double dist = CompareLBPHistograms(histograms[i], centroids[k]);
                if (dist < minDist) {
                    minDist = dist;
                    cellLabels[i] = k;
                }
            }
        }

        // Update centroids
        for (int32_t k = 0; k < numClusters; ++k) {
            std::vector<double> sum(59, 0);
            int32_t count = 0;
            for (size_t i = 0; i < histograms.size(); ++i) {
                if (cellLabels[i] == k) {
                    for (size_t j = 0; j < 59; ++j) {
                        sum[j] += histograms[i][j];
                    }
                    count++;
                }
            }
            if (count > 0) {
                for (size_t j = 0; j < 59; ++j) {
                    centroids[k][j] = sum[j] / count;
                }
            }
        }
    }

    // Create label image
    labels = QImage(width, height, PixelType::UInt8);
    uint8_t* labelData = static_cast<uint8_t*>(labels.Data());
    int32_t labelStride = labels.Stride();

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            int32_t gx = x / windowSize;
            int32_t gy = y / windowSize;
            gx = std::min(gx, gridX - 1);
            gy = std::min(gy, gridY - 1);
            labelData[y * labelStride + x] =
                static_cast<uint8_t>(cellLabels[gy * gridX + gx]);
        }
    }

    return numClusters;
}

void DetectTextureAnomalies(const QImage& image,
                            const std::vector<double>& referenceHist,
                            QImage& anomalyMap,
                            int32_t windowSize,
                            LBPType type) {
    Validate::RequirePositive(windowSize, "windowSize", "DetectTextureAnomalies");
    if (!Validate::RequireImageU8Gray(image, "DetectTextureAnomalies")) {
        anomalyMap = QImage();
        return;
    }
    if (referenceHist.empty()) {
        throw InvalidArgumentException("DetectTextureAnomalies: referenceHist is empty");
    }
    if (static_cast<int32_t>(referenceHist.size()) != LbpNumBins(type)) {
        throw InvalidArgumentException("DetectTextureAnomalies: referenceHist size mismatch");
    }

    // Compute LBP
    QImage lbpImage;
    ComputeLBP(image, lbpImage, type);

    int32_t width = image.Width();
    int32_t height = image.Height();
    int32_t halfWin = windowSize / 2;

    anomalyMap = QImage(width, height, PixelType::Float32);
    float* outData = static_cast<float*>(anomalyMap.Data());
    int32_t outStride = anomalyMap.Stride() / sizeof(float);

    // For each pixel, compute local histogram and compare
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            int32_t x0 = std::max(0, x - halfWin);
            int32_t y0 = std::max(0, y - halfWin);
            int32_t x1 = std::min(width, x + halfWin + 1);
            int32_t y1 = std::min(height, y + halfWin + 1);

            std::vector<QRegion::Run> runs;
            for (int32_t yy = y0; yy < y1; ++yy) {
                runs.push_back({yy, x0, x1});
            }
            QRegion region(std::move(runs));

            std::vector<double> localHist;
            ComputeLBPHistogram(lbpImage, region, localHist, type);

            double dist = CompareLBPHistograms(localHist, referenceHist);
            outData[y * outStride + x] = static_cast<float>(dist);
        }
    }
}

} // namespace Qi::Vision::Texture
