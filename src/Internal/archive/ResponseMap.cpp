/**
 * @file ResponseMap.cpp
 * @brief Implementation of Response Map for fast shape-based matching
 *
 * Based on LINE-MOD algorithm with spreading for robustness.
 * Optimized with AVX2 SIMD for Halcon-like performance.
 */

#include <QiVision/Internal/ResponseMap.h>
#include <QiVision/Platform/SIMD.h>

#include <algorithm>
#include <cmath>
#include <cstring>

// SIMD intrinsics
#if defined(__AVX2__)
#include <immintrin.h>
#define HAVE_AVX2 1
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#define HAVE_SSE4 1
#endif

namespace Qi::Vision::Internal {

// =============================================================================
// ResponseMap Implementation
// =============================================================================

bool ResponseMap::Build(const AnglePyramid& pyramid, int32_t spreadRadius)
{
    Clear();

    if (!pyramid.IsValid()) {
        return false;
    }

    numLevels_ = pyramid.NumLevels();
    spreadRadius_ = std::max(1, spreadRadius);
    levels_.resize(numLevels_);

    for (int32_t level = 0; level < numLevels_; ++level) {
        if (!BuildLevel(pyramid, level)) {
            Clear();
            return false;
        }
    }

    valid_ = true;
    return true;
}

bool ResponseMap::BuildLevel(const AnglePyramid& pyramid, int32_t levelIdx)
{
    auto& levelData = levels_[levelIdx];

    const auto& pyramidLevel = pyramid.GetLevel(levelIdx);
    const int32_t W = pyramidLevel.width;
    const int32_t H = pyramidLevel.height;

    if (W <= 0 || H <= 0) {
        return false;
    }

    // 64-byte aligned stride for SIMD
    const int32_t stride = (W + 63) & ~63;

    levelData.width = W;
    levelData.height = H;
    levelData.stride = stride;

    const size_t bufferSize = static_cast<size_t>(stride) * H;

    // Allocate all 8 bins
    for (int32_t bin = 0; bin < RESPONSE_NUM_BINS; ++bin) {
        levelData.bins[bin].resize(bufferSize, 0);
    }

    // Get gradient data from pyramid level
    const QImage& gradMag = pyramidLevel.gradMag;
    const QImage& gradDir = pyramidLevel.gradDir;

    if (gradMag.Empty() || gradDir.Empty()) {
        return false;
    }

    const float* magData = static_cast<const float*>(gradMag.Data());
    const float* dirData = static_cast<const float*>(gradDir.Data());

    const int32_t magStride = static_cast<int32_t>(gradMag.Stride() / sizeof(float));
    const int32_t dirStride = static_cast<int32_t>(gradDir.Stride() / sizeof(float));

    // Count strong gradient pixels per bin (for debugging)
    std::array<int32_t, RESPONSE_NUM_BINS> binCounts = {0};
    int32_t totalStrongPixels = 0;

    // Temporary buffer for each bin before spreading
    std::vector<uint8_t> tempMask(bufferSize);

    // First pass: count pixels and populate all bins simultaneously
    // This is more efficient than iterating per-bin
    std::array<std::vector<uint8_t>, RESPONSE_NUM_BINS> tempMasks;
    for (int32_t bin = 0; bin < RESPONSE_NUM_BINS; ++bin) {
        tempMasks[bin].resize(bufferSize, 0);
    }

    for (int32_t y = 0; y < H; ++y) {
        for (int32_t x = 0; x < W; ++x) {
            float mag = magData[y * magStride + x];

            if (mag < MIN_RESPONSE_MAGNITUDE) continue;

            totalStrongPixels++;

            float dir = dirData[y * dirStride + x];
            int32_t pixelBin = AngleToBin(static_cast<double>(dir));

            binCounts[pixelBin]++;

            // Scale magnitude to [0, 255]
            uint8_t value = static_cast<uint8_t>(std::min(255.0f, mag * 2.0f));
            if (value < 32) value = 32;  // Minimum response value
            tempMasks[pixelBin][y * stride + x] = value;
        }
    }

    // Apply spreading for each bin
    for (int32_t bin = 0; bin < RESPONSE_NUM_BINS; ++bin) {
        SpreadBinMask(tempMasks[bin].data(), levelData.bins[bin].data(),
                      W, H, stride);
    }

#ifdef QIVISION_DEBUG
    // Debug output
    fprintf(stderr, "[ResponseMap] Level %d: %dx%d, strong pixels=%d\n",
            levelIdx, W, H, totalStrongPixels);
    for (int32_t bin = 0; bin < RESPONSE_NUM_BINS; ++bin) {
        fprintf(stderr, "  Bin %d: %d pixels\n", bin, binCounts[bin]);
    }
#endif

    return true;
}

void ResponseMap::SpreadBinMask(const uint8_t* input, uint8_t* output,
                                 int32_t width, int32_t height, int32_t stride)
{
    const int32_t R = spreadRadius_;

    // Two-pass separable max filter for efficiency
    // Total complexity: O(W*H*R) instead of O(W*H*R^2)

    std::vector<uint8_t> temp(static_cast<size_t>(stride) * height);

    // Horizontal pass: max filter along rows
    for (int32_t y = 0; y < height; ++y) {
        const uint8_t* inRow = input + y * stride;
        uint8_t* outRow = temp.data() + y * stride;

        for (int32_t x = 0; x < width; ++x) {
            uint8_t maxVal = 0;
            const int32_t x0 = std::max(0, x - R);
            const int32_t x1 = std::min(width - 1, x + R);

            for (int32_t xx = x0; xx <= x1; ++xx) {
                if (inRow[xx] > maxVal) maxVal = inRow[xx];
            }
            outRow[x] = maxVal;
        }
    }

    // Vertical pass: max filter along columns
    for (int32_t y = 0; y < height; ++y) {
        uint8_t* outRow = output + y * stride;
        const int32_t y0 = std::max(0, y - R);
        const int32_t y1 = std::min(height - 1, y + R);

        for (int32_t x = 0; x < width; ++x) {
            uint8_t maxVal = 0;
            for (int32_t yy = y0; yy <= y1; ++yy) {
                uint8_t val = temp[yy * stride + x];
                if (val > maxVal) maxVal = val;
            }
            outRow[x] = maxVal;
        }
    }
}

void ResponseMap::SpreadBinMaskOR(uint8_t* data, int32_t width, int32_t height, int32_t stride)
{
    // Alternative spreading using OR operation (LINE-MOD style)
    // Spreads by T pixels in each direction using OR

    const int32_t T = SPREAD_T;

    std::vector<uint8_t> temp(static_cast<size_t>(stride) * height);
    std::memcpy(temp.data(), data, static_cast<size_t>(stride) * height);

    // Spread in T×T neighborhood using OR
    for (int32_t r = 0; r < T; ++r) {
        for (int32_t c = 0; c < T; ++c) {
            if (r == 0 && c == 0) continue;

            for (int32_t y = 0; y < height - r; ++y) {
                for (int32_t x = 0; x < width - c; ++x) {
                    uint8_t srcVal = temp[(y + r) * stride + (x + c)];
                    if (srcVal > data[y * stride + x]) {
                        data[y * stride + x] = srcVal;
                    }
                }
            }
        }
    }
}

void ResponseMap::Clear()
{
    levels_.clear();
    numLevels_ = 0;
    valid_ = false;
}

double ResponseMap::ComputeScore(const RotatedResponseModel& model,
                                  int32_t level,
                                  int32_t posX, int32_t posY,
                                  double* outCoverage) const
{
    if (!valid_ || level < 0 || level >= numLevels_) {
        if (outCoverage) *outCoverage = 0.0;
        return 0.0;
    }

    const auto& levelData = levels_[level];
    const int32_t W = levelData.width;
    const int32_t H = levelData.height;
    const int32_t stride = levelData.stride;

    // Early bounding box check
    if (!model.IsValidPosition(posX, posY, W, H)) {
        if (outCoverage) *outCoverage = 0.0;
        return 0.0;
    }

    uint32_t totalScore = 0;
    uint32_t totalWeight = 0;
    int32_t validCount = 0;

    const auto& points = model.points;
    const int32_t numPoints = static_cast<int32_t>(points.size());

#ifdef QIVISION_DEBUG
    static bool debugOnce = true;
    std::array<int32_t, RESPONSE_NUM_BINS> modelBinCounts = {0};
    std::array<int32_t, RESPONSE_NUM_BINS> responseCounts = {0};
    int32_t anyResponseCount = 0;
#endif

    // Main scoring loop - optimized for cache locality
    for (int32_t i = 0; i < numPoints; ++i) {
        const auto& pt = points[i];

        const int32_t imgX = posX + pt.offsetX;
        const int32_t imgY = posY + pt.offsetY;

#ifdef QIVISION_DEBUG
        modelBinCounts[pt.angleBin]++;
#endif

        // Get response from precomputed map (O(1) lookup)
        // This is the key optimization: direct array access instead of
        // bilinear interpolation + trigonometric functions
        const uint8_t response = levelData.bins[pt.angleBin][imgY * stride + imgX];

#ifdef QIVISION_DEBUG
        // Check if ANY bin has response at this position
        for (int32_t b = 0; b < RESPONSE_NUM_BINS; ++b) {
            if (levelData.bins[b][imgY * stride + imgX] > 0) {
                responseCounts[b]++;
                anyResponseCount++;
                break;
            }
        }
#endif

        if (response > 0) {
            totalScore += static_cast<uint32_t>(response) * pt.weight;
            validCount++;
        }
        totalWeight += pt.weight;
    }

#ifdef QIVISION_DEBUG
    if (debugOnce && posX == W/2 && posY == H/2) {
        debugOnce = false;
        fprintf(stderr, "[ComputeScore] Debug at center (%d, %d):\n", posX, posY);
        fprintf(stderr, "  Model points per bin: ");
        for (int b = 0; b < RESPONSE_NUM_BINS; ++b) {
            fprintf(stderr, "%d ", modelBinCounts[b]);
        }
        fprintf(stderr, "\n  Positions with ANY response: %d/%d\n", anyResponseCount, numPoints);
        fprintf(stderr, "  validCount (exact bin match): %d\n", validCount);
        fprintf(stderr, "  totalScore=%u, totalWeight=%u\n", totalScore, totalWeight);
    }
#endif

    if (outCoverage) {
        *outCoverage = (numPoints > 0) ?
            static_cast<double>(validCount) / numPoints : 0.0;
    }

    if (totalWeight == 0) {
        return 0.0;
    }

    // Normalize to [0, 1]
    // Max possible score: 255 (response) * 255 (weight) per point
    return static_cast<double>(totalScore) / (static_cast<double>(totalWeight) * 255.0);
}

size_t ResponseMap::MemoryBytes() const
{
    size_t total = 0;
    for (const auto& level : levels_) {
        for (int32_t bin = 0; bin < RESPONSE_NUM_BINS; ++bin) {
            total += level.bins[bin].capacity();
        }
    }
    return total;
}

} // namespace Qi::Vision::Internal
