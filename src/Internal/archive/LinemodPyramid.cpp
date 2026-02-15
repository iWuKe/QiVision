/**
 * @file LinemodPyramid.cpp
 * @brief Implementation of LINEMOD-style gradient pyramid
 *
 * Implements the exact algorithm from:
 * - Hinterstoisser et al. "Gradient Response Maps for Real-Time Detection
 *   of Texture-Less Objects" (TPAMI 2012)
 *
 * Key implementation details:
 * - 8-bin quantization using bit flags (not index)
 * - 3x3 neighbor histogram with threshold voting
 * - OR spreading (not max filter)
 * - SIMILARITY_LUT[8][256] for O(1) scoring
 */

#include <QiVision/Internal/LinemodPyramid.h>
#include <QiVision/Internal/Gradient.h>
#include <QiVision/Internal/Gaussian.h>
#include <QiVision/Internal/Convolution.h>
#include <QiVision/Internal/Pyramid.h>
#include <QiVision/Core/Constants.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace Qi::Vision::Internal {

// =============================================================================
// SIMILARITY_LUT Implementation
// =============================================================================

SimilarityLUT::SimilarityLUT() {
    // Precompute LUT[ori][mask]
    // LUT[i][L] = max_{j in L} |cos((i-j) * 45 degrees)| * 4
    //
    // Cosine values for angle differences:
    // 0°:  cos(0)   = 1.000 -> 4
    // 45°: cos(45)  = 0.707 -> 3
    // 90°: cos(90)  = 0.000 -> 0
    // 135°: cos(135) = -0.707 -> 3 (absolute value)
    // 180°: cos(180) = -1.000 -> 4 (absolute value)

    // Similarity by orientation-bin distance using round(|cos(diff * 22.5°)| * 4).
    // Uses 16→8 bin folding (bin16 & 7), so each bin covers 22.5° effective range.
    // More tolerant than floor-based scoring, improving recall for small angle diffs.
    static const uint8_t cosTable[8] = {
        4,  // 0 bins (0°):    cos(0)    = 1.000 → round(4.00) = 4
        4,  // 1 bin  (22.5°): cos(22.5) = 0.924 → round(3.70) = 4
        3,  // 2 bins (45°):   cos(45)   = 0.707 → round(2.83) = 3
        2,  // 3 bins (67.5°): cos(67.5) = 0.383 → round(1.53) = 2
        0,  // 4 bins (90°):   cos(90)   = 0.000 → round(0.00) = 0
        2,  // 5 bins (112.5°): symmetric to 3
        3,  // 6 bins (135°):   symmetric to 2
        4   // 7 bins (157.5°): symmetric to 1
    };

    for (int32_t ori = 0; ori < 8; ++ori) {
        for (int32_t mask = 0; mask < 256; ++mask) {
            uint8_t maxSim = 0;

            // Check each bit in the mask
            for (int32_t j = 0; j < 8; ++j) {
                if (mask & (1 << j)) {
                    // Compute angle difference
                    int32_t diff = (ori - j + 8) % 8;
                    uint8_t sim = cosTable[diff];
                    if (sim > maxSim) {
                        maxSim = sim;
                    }
                }
            }

            lut_[ori][mask] = maxSim;
        }
    }
}

// Global instance
const SimilarityLUT g_SimilarityLUT;

// =============================================================================
// LinemodPyramid Implementation
// =============================================================================

LinemodPyramid::LinemodPyramid() = default;
LinemodPyramid::~LinemodPyramid() = default;

LinemodPyramid::LinemodPyramid(LinemodPyramid&&) noexcept = default;
LinemodPyramid& LinemodPyramid::operator=(LinemodPyramid&&) noexcept = default;

void LinemodPyramid::Clear() {
    levels_.clear();
    numLevels_ = 0;
    valid_ = false;
}

bool LinemodPyramid::Build(const QImage& image, const LinemodPyramidParams& params) {
    Clear();

    if (image.Empty() || image.Channels() != 1) {
        return false;
    }

    params_ = params;
    numLevels_ = std::max(1, std::min(params.numLevels, 10));
    if (!params_.spreadTAtLevel.empty()) {
        for (auto& t : params_.spreadTAtLevel) {
            t = std::max(1, t);
        }
        if (static_cast<int32_t>(params_.spreadTAtLevel.size()) < numLevels_) {
            int32_t fillT = params_.spreadTAtLevel.back();
            params_.spreadTAtLevel.resize(static_cast<size_t>(numLevels_), fillT);
        }
    }

    PyramidParams pyrParams;
    pyrParams.numLevels = numLevels_;
    pyrParams.sigma = params_.smoothSigma;
    pyrParams.downsample = DownsampleMethod::Gaussian;
    pyrParams.minDimension = 8;
    ImagePyramid gaussianPyr = BuildGaussianPyramid(image, pyrParams);
    if (gaussianPyr.Empty()) {
        return false;
    }

    numLevels_ = std::min(numLevels_, gaussianPyr.NumLevels());
    levels_.resize(numLevels_);

    for (int32_t level = 0; level < numLevels_; ++level) {
        const auto& pyrLevel = gaussianPyr.GetLevel(level);
        if (!pyrLevel.IsValid() || pyrLevel.width < 8 || pyrLevel.height < 8) {
            numLevels_ = level;
            levels_.resize(numLevels_);
            break;
        }

        QImage levelImage(pyrLevel.width, pyrLevel.height, PixelType::Float32, ChannelType::Gray);
        float* dst = static_cast<float*>(levelImage.Data());
        const int32_t dstStride = static_cast<int32_t>(levelImage.Stride() / sizeof(float));

        for (int32_t y = 0; y < pyrLevel.height; ++y) {
            std::memcpy(dst + y * dstStride,
                        pyrLevel.data.data() + static_cast<size_t>(y) * pyrLevel.width,
                        static_cast<size_t>(pyrLevel.width) * sizeof(float));
        }

        if (!BuildLevel(levelImage, level)) {
            Clear();
            return false;
        }
    }

    valid_ = true;
    return true;
}

bool LinemodPyramid::BuildLevel(const QImage& image, int32_t levelIdx) {
    auto& level = levels_[levelIdx];

    level.width = image.Width();
    level.height = image.Height();
    level.scale = std::pow(0.5, levelIdx);
    level.stride = (level.width + 63) & ~63;  // 64-byte alignment

    const int32_t W = level.width;
    const int32_t H = level.height;

    // =========================================================================
    // 优化流程: 快速内联 3x3 高斯 + Sobel (单次分配，两个 pass)
    // Pass 1: 行方向 [1,2,1]/4 高斯
    // Pass 2: 列方向 [1,2,1]/4 高斯 + Sobel + 量化 (融合)
    // =========================================================================

    const float* srcData = static_cast<const float*>(image.Data());
    const int32_t srcStride = static_cast<int32_t>(image.Stride() / sizeof(float));

    // 只在需要提取特征时分配 gradMag
    const bool needGradMag = params_.extractFeatures;
    if (needGradMag) {
        level.gradMag = QImage(W, H, PixelType::Float32, ChannelType::Gray);
    }
    level.quantized = QImage(W, H, PixelType::UInt8, ChannelType::Gray);

    float* magData = needGradMag ? static_cast<float*>(level.gradMag.Data()) : nullptr;
    uint8_t* quantData = static_cast<uint8_t*>(level.quantized.Data());
    const int32_t magStride = needGradMag ? static_cast<int32_t>(level.gradMag.Stride() / sizeof(float)) : 0;
    const int32_t quantStride = static_cast<int32_t>(level.quantized.Stride());

    const float minMag = params_.minMagnitude;
    const float minMag2 = minMag * minMag;

    // Pass 1: 行方向高斯 [1,2,1]/4 - 使用复用缓冲区 (SIMD 优化)
    const size_t bufSize = static_cast<size_t>(W) * H;
    EnsureBuffer(rowSmoothed_, bufSize);

#pragma omp parallel for schedule(static)
    for (int32_t y = 0; y < H; ++y) {
        const float* srcRow = srcData + y * srcStride;
        float* dstRow = rowSmoothed_.data() + y * W;

        // 边界: x=0
        dstRow[0] = (srcRow[0] * 3.0f + srcRow[1]) * 0.25f;

        int32_t x = 1;

#ifdef __AVX__
        // AVX: 一次处理 8 个 float
        // dst[x] = (src[x-1] + 2*src[x] + src[x+1]) * 0.25
        const __m256 vTwo = _mm256_set1_ps(2.0f);
        const __m256 vQuarter = _mm256_set1_ps(0.25f);

        // 处理到 W-9 确保 src[x+1] 不越界 (需要读取 x+8)
        for (; x + 8 <= W - 1; x += 8) {
            __m256 left   = _mm256_loadu_ps(srcRow + x - 1);
            __m256 center = _mm256_loadu_ps(srcRow + x);
            __m256 right  = _mm256_loadu_ps(srcRow + x + 1);

            // (left + 2*center + right) * 0.25
            __m256 result = _mm256_mul_ps(center, vTwo);
            result = _mm256_add_ps(result, left);
            result = _mm256_add_ps(result, right);
            result = _mm256_mul_ps(result, vQuarter);

            _mm256_storeu_ps(dstRow + x, result);
        }
#endif

#ifdef __SSE2__
        // SSE2 fallback: 一次处理 4 个 float
        const __m128 vTwo4 = _mm_set1_ps(2.0f);
        const __m128 vQuarter4 = _mm_set1_ps(0.25f);

        for (; x + 4 <= W - 1; x += 4) {
            __m128 left   = _mm_loadu_ps(srcRow + x - 1);
            __m128 center = _mm_loadu_ps(srcRow + x);
            __m128 right  = _mm_loadu_ps(srcRow + x + 1);

            __m128 result = _mm_mul_ps(center, vTwo4);
            result = _mm_add_ps(result, left);
            result = _mm_add_ps(result, right);
            result = _mm_mul_ps(result, vQuarter4);

            _mm_storeu_ps(dstRow + x, result);
        }
#endif

        // 标量处理剩余部分
        for (; x < W - 1; ++x) {
            dstRow[x] = (srcRow[x - 1] + 2.0f * srcRow[x] + srcRow[x + 1]) * 0.25f;
        }

        // 边界: x=W-1
        if (W > 1) {
            dstRow[W - 1] = (srcRow[W - 2] + srcRow[W - 1] * 3.0f) * 0.25f;
        }
    }

    // Pass 2: 列方向高斯 [1,2,1]/4 (SIMD 优化)
    // 使用复用缓冲区
    EnsureBuffer(smoothed_, bufSize);

#pragma omp parallel for schedule(static)
    for (int32_t y = 0; y < H; ++y) {
        const float* row0 = rowSmoothed_.data() + ((y > 0) ? (y - 1) : 0) * W;
        const float* row1 = rowSmoothed_.data() + y * W;
        const float* row2 = rowSmoothed_.data() + ((y < H - 1) ? (y + 1) : (H - 1)) * W;
        float* dstRow = smoothed_.data() + y * W;

        int32_t x = 0;

#ifdef __AVX__
        const __m256 vTwo = _mm256_set1_ps(2.0f);
        const __m256 vQuarter = _mm256_set1_ps(0.25f);

        for (; x + 8 <= W; x += 8) {
            __m256 r0 = _mm256_loadu_ps(row0 + x);
            __m256 r1 = _mm256_loadu_ps(row1 + x);
            __m256 r2 = _mm256_loadu_ps(row2 + x);

            __m256 result = _mm256_mul_ps(r1, vTwo);
            result = _mm256_add_ps(result, r0);
            result = _mm256_add_ps(result, r2);
            result = _mm256_mul_ps(result, vQuarter);

            _mm256_storeu_ps(dstRow + x, result);
        }
#endif

#ifdef __SSE2__
        const __m128 vTwo4 = _mm_set1_ps(2.0f);
        const __m128 vQuarter4 = _mm_set1_ps(0.25f);

        for (; x + 4 <= W; x += 4) {
            __m128 r0 = _mm_loadu_ps(row0 + x);
            __m128 r1 = _mm_loadu_ps(row1 + x);
            __m128 r2 = _mm_loadu_ps(row2 + x);

            __m128 result = _mm_mul_ps(r1, vTwo4);
            result = _mm_add_ps(result, r0);
            result = _mm_add_ps(result, r2);
            result = _mm_mul_ps(result, vQuarter4);

            _mm_storeu_ps(dstRow + x, result);
        }
#endif

        for (; x < W; ++x) {
            dstRow[x] = (row0[x] + 2.0f * row1[x] + row2[x]) * 0.25f;
        }
    }

    // Pass 3: Sobel + 量化 (在完全平滑后的图像上)
    // line2Dup 风格: 先量化到 16 个方向，再折叠到 8 个方向。
    auto quantizeOri = [](float gx, float gy) -> uint8_t {
        double angle = std::atan2(static_cast<double>(gy), static_cast<double>(gx));
        if (angle < 0.0) {
            angle += 2.0 * PI;
        }
        int32_t bin16 = static_cast<int32_t>(angle * 16.0 / (2.0 * PI));
        bin16 = std::clamp(bin16, 0, 15);
        int32_t bin = bin16 & 7;
        return static_cast<uint8_t>(1 << bin);
    };

#pragma omp parallel for schedule(static)
    for (int32_t y = 0; y < H; ++y) {
        const float* row0 = smoothed_.data() + ((y > 0) ? (y - 1) : 0) * W;
        const float* row1 = smoothed_.data() + y * W;
        const float* row2 = smoothed_.data() + ((y < H - 1) ? (y + 1) : (H - 1)) * W;
        float* magRow = magData ? magData + y * magStride : nullptr;
        uint8_t* quantRow = quantData + y * quantStride;

        // 边界 x=0
        {
            float p00 = row0[0], p01 = row0[0], p02 = row0[1];
            float p10 = row1[0], p12 = row1[1];
            float p20 = row2[0], p21 = row2[0], p22 = row2[1];

            float gx = -p00 + p02 - 2.0f * p10 + 2.0f * p12 - p20 + p22;
            float gy = -p00 - 2.0f * p01 - p02 + p20 + 2.0f * p21 + p22;
            const float mag2 = gx * gx + gy * gy;
            if (magRow) {
                magRow[0] = std::sqrt(mag2);
            }
            quantRow[0] = (mag2 < minMag2) ? 0 : quantizeOri(gx, gy);
        }

        // 内部像素
        for (int32_t x = 1; x < W - 1; ++x) {
            float p00 = row0[x - 1], p01 = row0[x], p02 = row0[x + 1];
            float p10 = row1[x - 1], p12 = row1[x + 1];
            float p20 = row2[x - 1], p21 = row2[x], p22 = row2[x + 1];

            float gx = -p00 + p02 - 2.0f * p10 + 2.0f * p12 - p20 + p22;
            float gy = -p00 - 2.0f * p01 - p02 + p20 + 2.0f * p21 + p22;
            const float mag2 = gx * gx + gy * gy;
            if (magRow) {
                magRow[x] = std::sqrt(mag2);
            }
            quantRow[x] = (mag2 < minMag2) ? 0 : quantizeOri(gx, gy);
        }

        // 边界 x=W-1
        if (W > 1) {
            int32_t x = W - 1;
            float p00 = row0[x - 1], p01 = row0[x], p02 = row0[x];
            float p10 = row1[x - 1], p12 = row1[x];
            float p20 = row2[x - 1], p21 = row2[x], p22 = row2[x];

            float gx = -p00 + p02 - 2.0f * p10 + 2.0f * p12 - p20 + p22;
            float gy = -p00 - 2.0f * p01 - p02 + p20 + 2.0f * p21 + p22;
            const float mag2 = gx * gx + gy * gy;
            if (magRow) {
                magRow[x] = std::sqrt(mag2);
            }
            quantRow[x] = (mag2 < minMag2) ? 0 : quantizeOri(gx, gy);
        }
    }

    // Keep a 1-pixel border empty like reference implementations.
    if (W > 1 && H > 1) {
        std::memset(quantData, 0, static_cast<size_t>(quantStride));
        std::memset(quantData + (H - 1) * quantStride, 0, static_cast<size_t>(quantStride));
        for (int32_t y = 0; y < H; ++y) {
            quantData[y * quantStride] = 0;
            quantData[y * quantStride + (W - 1)] = 0;
        }
    }

    // Pass 3: 投票 + OR扩散 (保持原有实现)
    ApplyNeighborVoting(level);
    ApplyORSpreading(level, levelIdx);
    BuildResponseMaps(level);

    // Linearize response maps for cache-friendly search access
    {
        int32_t spreadT = params_.spreadT;
        if (!params_.spreadTAtLevel.empty() &&
            levelIdx >= 0 &&
            levelIdx < static_cast<int32_t>(params_.spreadTAtLevel.size())) {
            spreadT = params_.spreadTAtLevel[static_cast<size_t>(levelIdx)];
        }
        int32_t linT = std::max(1, spreadT);
        Linearize(level, linT);
    }

    // 提取特征 (模板创建时)
    if (params_.extractFeatures) {
        ExtractLevelFeatures(level);
    }

    return true;
}

void LinemodPyramid::ApplyNeighborVoting(LinemodLevelData& level) {
    // Paper: "Assign to each location the quantized value that occurs most
    //         often in a 3x3 (or 5x5) neighborhood"
    // "Only keep pixels where max_votes >= NEIGHBOR_THRESHOLD"

    const int32_t W = level.width;
    const int32_t H = level.height;
    const int32_t stride = static_cast<int32_t>(level.quantized.Stride());
    const uint8_t* src = static_cast<const uint8_t*>(level.quantized.Data());

    // Output to spread (will be replaced by OR spreading later)
    QImage voted(W, H, PixelType::UInt8, ChannelType::Gray);
    uint8_t* dst = static_cast<uint8_t*>(voted.Data());
    const int32_t dstStride = static_cast<int32_t>(voted.Stride());

    std::memset(dst, 0, voted.Height() * voted.Stride());

    const int32_t threshold = params_.neighborThreshold;

#pragma omp parallel for if(W * H > 100000)
    for (int32_t y = 1; y < H - 1; ++y) {
        const uint8_t* srcRow0 = src + (y - 1) * stride;
        const uint8_t* srcRow1 = src + y * stride;
        const uint8_t* srcRow2 = src + (y + 1) * stride;
        uint8_t* dstRow = dst + y * dstStride;

        for (int32_t x = 1; x < W - 1; ++x) {
            // 读取 3x3 邻域的 9 个值
            uint8_t n0 = srcRow0[x - 1];
            uint8_t n1 = srcRow0[x];
            uint8_t n2 = srcRow0[x + 1];
            uint8_t n3 = srcRow1[x - 1];
            uint8_t n4 = srcRow1[x];
            uint8_t n5 = srcRow1[x + 1];
            uint8_t n6 = srcRow2[x - 1];
            uint8_t n7 = srcRow2[x];
            uint8_t n8 = srcRow2[x + 1];

            // 快速检查：如果所有邻域都是 0，跳过
            uint8_t anySet = n0 | n1 | n2 | n3 | n4 | n5 | n6 | n7 | n8;
            if (anySet == 0) {
                dstRow[x] = 0;
                continue;
            }

            // 统计每个 bin 的投票数
            // 由于每个像素只有 0 或 1 位被设置，可以用位运算优化
            int32_t votes[8] = {0, 0, 0, 0, 0, 0, 0, 0};

            // 展开循环统计每个 bit 的投票
            #define COUNT_BIT(n, bit) if (n & (1 << bit)) votes[bit]++
            #define COUNT_ALL_BITS(n) \
                COUNT_BIT(n, 0); COUNT_BIT(n, 1); COUNT_BIT(n, 2); COUNT_BIT(n, 3); \
                COUNT_BIT(n, 4); COUNT_BIT(n, 5); COUNT_BIT(n, 6); COUNT_BIT(n, 7)

            COUNT_ALL_BITS(n0);
            COUNT_ALL_BITS(n1);
            COUNT_ALL_BITS(n2);
            COUNT_ALL_BITS(n3);
            COUNT_ALL_BITS(n4);
            COUNT_ALL_BITS(n5);
            COUNT_ALL_BITS(n6);
            COUNT_ALL_BITS(n7);
            COUNT_ALL_BITS(n8);

            #undef COUNT_BIT
            #undef COUNT_ALL_BITS

            // 找最大投票的 bin
            int32_t maxVotes = votes[0];
            int32_t maxBin = 0;
            for (int32_t b = 1; b < 8; ++b) {
                if (votes[b] > maxVotes) {
                    maxVotes = votes[b];
                    maxBin = b;
                }
            }

            // 阈值判断
            dstRow[x] = (maxVotes >= threshold)
                ? static_cast<uint8_t>(1 << maxBin)
                : 0;
        }
    }

    // Copy voted back to quantized
    level.quantized = std::move(voted);
}

void LinemodPyramid::ApplyORSpreading(LinemodLevelData& level, int32_t levelIdx) {
    // Paper: "Spread binary labels using OR operation"
    // spread[x] |= quantized[x + offset] for all offsets in T×T region

    const int32_t W = level.width;
    const int32_t H = level.height;
    int32_t T = params_.spreadT;
    if (!params_.spreadTAtLevel.empty() &&
        levelIdx >= 0 &&
        levelIdx < static_cast<int32_t>(params_.spreadTAtLevel.size())) {
        T = params_.spreadTAtLevel[static_cast<size_t>(levelIdx)];
    }
    T = std::max(1, T);
    const int32_t srcStride = static_cast<int32_t>(level.quantized.Stride());
    const uint8_t* src = static_cast<const uint8_t*>(level.quantized.Data());

    level.spread = QImage(W, H, PixelType::UInt8, ChannelType::Gray);
    level.stride = static_cast<int32_t>(level.spread.Stride());
    uint8_t* dst = static_cast<uint8_t*>(level.spread.Data());
    const int32_t dstStride = level.stride;

    // Initialize with source
    std::memcpy(dst, src, H * dstStride);

    // OR spreading (single direction, line2Dup-style):
    // spread[y][x] |= quantized[y + dy][x + dx], dy/dx in [0, T).
    // Inner x loop uses SIMD for batch OR operations.
    for (int32_t dy = 0; dy < T; ++dy) {
        for (int32_t dx = 0; dx < T; ++dx) {
            if (dy == 0 && dx == 0) continue;

            const int32_t xLimit = W - dx;
#pragma omp parallel for schedule(static) if(W * H > 100000)
            for (int32_t y = 0; y < H - dy; ++y) {
                uint8_t* dstRow = dst + y * dstStride;
                const uint8_t* srcRow = src + (y + dy) * srcStride + dx;
                int32_t x = 0;

#ifdef __AVX2__
                for (; x + 32 <= xLimit; x += 32) {
                    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dstRow + x));
                    __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(srcRow + x));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstRow + x), _mm256_or_si256(a, b));
                }
#endif

#ifdef __SSE2__
                for (; x + 16 <= xLimit; x += 16) {
                    __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstRow + x));
                    __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcRow + x));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(dstRow + x), _mm_or_si128(a, b));
                }
#endif

                for (; x < xLimit; ++x) {
                    dstRow[x] |= srcRow[x];
                }
            }
        }
    }
}

void LinemodPyramid::BuildResponseMaps(LinemodLevelData& level) {
    const int32_t W = level.width;
    const int32_t H = level.height;
    const int32_t stride = static_cast<int32_t>(level.spread.Stride());
    const uint8_t* spreadData = static_cast<const uint8_t*>(level.spread.Data());
    if (W <= 0 || H <= 0 || spreadData == nullptr) {
        for (auto& map : level.responseMaps) {
            map.clear();
        }
        return;
    }

    const size_t bufSize = static_cast<size_t>(stride) * static_cast<size_t>(H);
    for (int32_t bin = 0; bin < LINEMOD_NUM_BINS; ++bin) {
        auto& response = level.responseMaps[static_cast<size_t>(bin)];
        response.assign(bufSize, 0);

#pragma omp parallel for schedule(static) if(W * H > 100000)
        for (int32_t y = 0; y < H; ++y) {
            const uint8_t* srcRow = spreadData + y * stride;
            uint8_t* dstRow = response.data() + y * stride;
            for (int32_t x = 0; x < W; ++x) {
                dstRow[x] = g_SimilarityLUT.Get(bin, srcRow[x]);
            }
        }
    }
}

void LinemodPyramid::Linearize(LinemodLevelData& level, int32_t T) {
    const int32_t W = level.width;
    const int32_t H = level.height;
    const int32_t stride = static_cast<int32_t>(level.spread.Stride());

    if (T <= 0 || W <= 0 || H <= 0 || level.responseMaps[0].empty()) {
        level.linearT = 0;
        return;
    }

    const int32_t BW = (W + T - 1) / T;
    const int32_t BH = (H + T - 1) / T;
    const int32_t numBlocks = BW * BH;
    const int32_t numPhases = T * T;
    const size_t totalSize = static_cast<size_t>(numPhases) * numBlocks;

    level.linearT = T;
    level.linearBW = BW;
    level.linearBH = BH;

    for (int32_t bin = 0; bin < LINEMOD_NUM_BINS; ++bin) {
        auto& lin = level.linearized[static_cast<size_t>(bin)];
        lin.assign(totalSize, 0);

        const auto& resp = level.responseMaps[static_cast<size_t>(bin)];

        for (int32_t dy = 0; dy < T; ++dy) {
            for (int32_t dx = 0; dx < T; ++dx) {
                const int32_t phaseOffset = (dy * T + dx) * numBlocks;

                for (int32_t by = 0; by < BH; ++by) {
                    const int32_t srcY = by * T + dy;
                    if (srcY >= H) break;

                    for (int32_t bx = 0; bx < BW; ++bx) {
                        const int32_t srcX = bx * T + dx;
                        if (srcX >= W) continue;

                        lin[static_cast<size_t>(phaseOffset + by * BW + bx)] =
                            resp[static_cast<size_t>(srcY * stride + srcX)];
                    }
                }
            }
        }
    }
}

void LinemodPyramid::ExtractLevelFeatures(LinemodLevelData& level) {
    // Extract all candidate features from this level
    const int32_t W = level.width;
    const int32_t H = level.height;
    const uint8_t* quantData = static_cast<const uint8_t*>(level.quantized.Data());
    const float* magData = static_cast<const float*>(level.gradMag.Data());
    const int32_t quantStride = static_cast<int32_t>(level.quantized.Stride());
    const int32_t magStride = static_cast<int32_t>(level.gradMag.Stride() / sizeof(float));

    level.features.clear();
    level.features.reserve(W * H / 16);  // Rough estimate

    for (int32_t y = 1; y < H - 1; ++y) {
        for (int32_t x = 1; x < W - 1; ++x) {
            uint8_t q = quantData[y * quantStride + x];
            if (q == 0) continue;

            float mag = magData[y * magStride + x];
            if (mag < params_.minMagnitude) continue;

            // Find the set bit (orientation)
            int32_t ori = 0;
            for (int32_t b = 0; b < 8; ++b) {
                if (q & (1 << b)) {
                    ori = b;
                    break;
                }
            }

            level.features.emplace_back(
                static_cast<int16_t>(x),
                static_cast<int16_t>(y),
                static_cast<uint8_t>(ori)
            );
        }
    }
}

std::vector<LinemodFeature> LinemodPyramid::ExtractFeatures(
    int32_t level, const Rect2i& roi, int32_t maxFeatures, float minDistance) const
{
    if (!valid_ || level < 0 || level >= numLevels_) {
        return {};
    }

    const auto& levelData = levels_[level];
    const int32_t W = levelData.width;
    const int32_t H = levelData.height;

    // Determine ROI
    Rect2i actualRoi = roi;
    if (actualRoi.width <= 0 || actualRoi.height <= 0) {
        actualRoi = Rect2i(0, 0, W, H);
    }

    // Clamp ROI
    actualRoi.x = std::max(0, actualRoi.x);
    actualRoi.y = std::max(0, actualRoi.y);
    actualRoi.width = std::min(actualRoi.width, W - actualRoi.x);
    actualRoi.height = std::min(actualRoi.height, H - actualRoi.y);

    // Center of ROI (features will be relative to this)
    float centerX = actualRoi.x + actualRoi.width / 2.0f;
    float centerY = actualRoi.y + actualRoi.height / 2.0f;

    // Collect candidates with score (line2Dup-style).
    struct Candidate {
        int16_t x, y;
        uint8_t ori;
        float score;
    };
    std::vector<Candidate> candidates;

    const uint8_t* quantData = static_cast<const uint8_t*>(levelData.quantized.Data());
    const int32_t quantStride = static_cast<int32_t>(levelData.quantized.Stride());

    // gradMag may be empty for search pyramids when feature extraction is disabled.
    const bool hasGradMag = !levelData.gradMag.Empty();
    const float* magData = hasGradMag ? static_cast<const float*>(levelData.gradMag.Data()) : nullptr;
    const int32_t magStride = hasGradMag ? static_cast<int32_t>(levelData.gradMag.Stride() / sizeof(float)) : 0;

    // 5x5 NMS suppresses clustered points before scattered selection.
    constexpr int32_t NMS_KERNEL = 5;
    constexpr int32_t NMS_HALF = NMS_KERNEL / 2;
    std::vector<uint8_t> nmsMask(static_cast<size_t>(W * H), 255);

    const int32_t roiYBegin = std::max(actualRoi.y, NMS_HALF);
    const int32_t roiYEnd = std::min(actualRoi.y + actualRoi.height, H - NMS_HALF);
    const int32_t roiXBegin = std::max(actualRoi.x, NMS_HALF);
    const int32_t roiXEnd = std::min(actualRoi.x + actualRoi.width, W - NMS_HALF);

    for (int32_t y = roiYBegin; y < roiYEnd; ++y) {
        for (int32_t x = roiXBegin; x < roiXEnd; ++x) {
            if (!nmsMask[static_cast<size_t>(y * W + x)]) {
                continue;
            }

            uint8_t q = quantData[y * quantStride + x];
            if (q == 0) {
                continue;
            }

            float mag = hasGradMag ? magData[y * magStride + x] : (params_.minMagnitude + 1.0f);
            if (hasGradMag && mag < params_.minMagnitude) {
                continue;
            }

            if (hasGradMag) {
                bool isLocalMax = true;
                for (int32_t ny = y - NMS_HALF; ny <= y + NMS_HALF && isLocalMax; ++ny) {
                    for (int32_t nx = x - NMS_HALF; nx <= x + NMS_HALF; ++nx) {
                        if (nx == x && ny == y) {
                            continue;
                        }
                        if (magData[ny * magStride + nx] > mag) {
                            isLocalMax = false;
                            break;
                        }
                    }
                }
                if (isLocalMax) {
                    for (int32_t ny = y - NMS_HALF; ny <= y + NMS_HALF; ++ny) {
                        for (int32_t nx = x - NMS_HALF; nx <= x + NMS_HALF; ++nx) {
                            if (nx == x && ny == y) {
                                continue;
                            }
                            nmsMask[static_cast<size_t>(ny * W + nx)] = 0;
                        }
                    }
                }
            }

            int32_t ori = 0;
            for (int32_t b = 0; b < 8; ++b) {
                if (q & (1 << b)) {
                    ori = b;
                    break;
                }
            }

            candidates.push_back({
                static_cast<int16_t>(x),
                static_cast<int16_t>(y),
                static_cast<uint8_t>(ori),
                mag * mag
            });
        }
    }

    if (candidates.empty() || maxFeatures <= 0) {
        return {};
    }

    maxFeatures = std::min<int32_t>(maxFeatures, static_cast<int32_t>(candidates.size()));

    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    // line2Dup-style scattered selection with adaptive distance relaxation.
    std::vector<LinemodFeature> result;
    result.reserve(maxFeatures);
    float distance = std::max(minDistance, static_cast<float>(candidates.size() / maxFeatures + 1));
    size_t candidateIndex = 0;
    while (static_cast<int32_t>(result.size()) < maxFeatures) {
        const auto& c = candidates[candidateIndex];
        bool keep = true;
        const float distSq = distance * distance;
        for (const auto& f : result) {
            float dx = (c.x - centerX) - f.x;
            float dy = (c.y - centerY) - f.y;
            if (dx * dx + dy * dy < distSq) {
                keep = false;
                break;
            }
        }
        if (keep) {
            result.emplace_back(
                static_cast<int16_t>(c.x - centerX),
                static_cast<int16_t>(c.y - centerY),
                c.ori
            );
        }

        candidateIndex++;
        if (candidateIndex >= candidates.size()) {
            candidateIndex = 0;
            distance -= 1.0f;
            if (distance < 0.0f) {
                distance = 0.0f;
            }
        }
    }

    return result;
}

const std::vector<LinemodFeature>& LinemodPyramid::GetAllFeatures(int32_t level) const {
    static const std::vector<LinemodFeature> empty;
    if (!valid_ || level < 0 || level >= numLevels_) {
        return empty;
    }
    return levels_[level].features;
}

double LinemodPyramid::ComputeScore(const std::vector<LinemodFeature>& features,
                                     int32_t level, int32_t x, int32_t y) const
{
    if (!valid_ || level < 0 || level >= numLevels_ || features.empty()) {
        return 0.0;
    }

    const auto& levelData = levels_[level];
    const int32_t W = levelData.width;
    const int32_t H = levelData.height;
    const int32_t spreadStride = static_cast<int32_t>(levelData.spread.Stride());
    const bool hasResponseMap = !levelData.responseMaps[0].empty();
    const uint8_t* spreadData = hasResponseMap ? nullptr
                                               : static_cast<const uint8_t*>(levelData.spread.Data());

    int32_t totalScore = 0;

    for (const auto& f : features) {
        int32_t fx = x + f.x;
        int32_t fy = y + f.y;

        // Bounds check
        if (fx < 0 || fx >= W || fy < 0 || fy >= H) {
            continue;
        }

        const int32_t idx = fy * spreadStride + fx;
        if (hasResponseMap) {
            totalScore += levelData.responseMaps[f.ori][idx];
        } else {
            totalScore += g_SimilarityLUT.Get(f.ori, spreadData[idx]);
        }
    }

    // Normalize to [0, 1]
    // Max score per feature is LINEMOD_MAX_RESPONSE (4)
    return static_cast<double>(totalScore) /
           (static_cast<double>(features.size()) * LINEMOD_MAX_RESPONSE);
}

double LinemodPyramid::ComputeScoreRotated(const std::vector<LinemodFeature>& features,
                                            int32_t level, int32_t x, int32_t y,
                                            double angle) const
{
    if (!valid_ || level < 0 || level >= numLevels_ || features.empty()) {
        return 0.0;
    }

    const auto& levelData = levels_[level];
    const int32_t W = levelData.width;
    const int32_t H = levelData.height;
    const int32_t spreadStride = static_cast<int32_t>(levelData.spread.Stride());
    const bool hasResponseMap = !levelData.responseMaps[0].empty();
    const uint8_t* spreadData = hasResponseMap ? nullptr
                                               : static_cast<const uint8_t*>(levelData.spread.Data());

    const double cosA = std::cos(angle);
    const double sinA = std::sin(angle);

    // Precompute rotation bin offset (how many 45° steps)
    // angle in radians -> bins
    int32_t rotBinOffset = static_cast<int32_t>(std::round(angle * 8.0 / (2.0 * PI)));
    rotBinOffset = ((rotBinOffset % 8) + 8) % 8;  // Normalize to [0, 7]

    int32_t totalScore = 0;

    for (const auto& f : features) {
        // Rotate feature coordinates
        double rx = cosA * f.x - sinA * f.y;
        double ry = sinA * f.x + cosA * f.y;

        int32_t fx = x + static_cast<int32_t>(std::round(rx));
        int32_t fy = y + static_cast<int32_t>(std::round(ry));

        // Bounds check
        if (fx < 0 || fx >= W || fy < 0 || fy >= H) {
            continue;
        }

        // Rotate orientation bin
        int32_t rotatedOri = (f.ori + rotBinOffset) & 7;

        const int32_t idx = fy * spreadStride + fx;
        if (hasResponseMap) {
            totalScore += levelData.responseMaps[rotatedOri][idx];
        } else {
            totalScore += g_SimilarityLUT.Get(rotatedOri, spreadData[idx]);
        }
    }

    // Normalize to [0, 1]
    return static_cast<double>(totalScore) /
           (static_cast<double>(features.size()) * LINEMOD_MAX_RESPONSE);
}

double LinemodPyramid::ComputeScorePrecomputed(const std::vector<LinemodFeature>& rotatedFeatures,
                                               int32_t level, int32_t x, int32_t y,
                                               double earlyRejectThreshold) const
{
    if (!valid_ || level < 0 || level >= numLevels_ || rotatedFeatures.empty()) {
        return 0.0;
    }

    const auto& levelData = levels_[level];
    const int32_t W = levelData.width;
    const int32_t H = levelData.height;
    const int32_t spreadStride = static_cast<int32_t>(levelData.spread.Stride());
    const bool hasResponseMap = !levelData.responseMaps[0].empty();
    const uint8_t* spreadData = hasResponseMap ? nullptr
                                               : static_cast<const uint8_t*>(levelData.spread.Data());

    const int32_t N = static_cast<int32_t>(rotatedFeatures.size());
    const int32_t earlyRejectScore = (earlyRejectThreshold > 0.0)
        ? static_cast<int32_t>(earlyRejectThreshold * N * LINEMOD_MAX_RESPONSE)
        : -1;

    int32_t totalScore = 0;

    for (int32_t i = 0; i < N; ++i) {
        const auto& f = rotatedFeatures[static_cast<size_t>(i)];
        int32_t fx = x + f.x;
        int32_t fy = y + f.y;

        // Bounds check
        if (fx < 0 || fx >= W || fy < 0 || fy >= H) {
            continue;
        }

        const int32_t idx = fy * spreadStride + fx;
        if (hasResponseMap) {
            totalScore += levelData.responseMaps[f.ori][idx];
        } else {
            totalScore += g_SimilarityLUT.Get(f.ori, spreadData[idx]);
        }

        // Early termination: accumulated + remaining max possible < threshold
        if (earlyRejectScore > 0) {
            int32_t remaining = (N - i - 1) * LINEMOD_MAX_RESPONSE;
            if (totalScore + remaining < earlyRejectScore) {
                return 0.0;
            }
        }
    }

    return static_cast<double>(totalScore) /
           (static_cast<double>(N) * LINEMOD_MAX_RESPONSE);
}

void LinemodPyramid::ComputeScoresBatch8(const std::vector<LinemodFeature>& rotatedFeatures,
                                          int32_t level, int32_t x, int32_t y,
                                          double* scoresOut,
                                          double earlyRejectThreshold) const
{
    for (int32_t i = 0; i < 8; ++i) {
        scoresOut[i] = 0.0;
    }

    if (!valid_ || level < 0 || level >= numLevels_ || rotatedFeatures.empty()) {
        return;
    }

    const auto& levelData = levels_[level];
    const int32_t W = levelData.width;
    const int32_t H = levelData.height;
    const int32_t stride = static_cast<int32_t>(levelData.spread.Stride());
    const bool hasResponseMap = !levelData.responseMaps[0].empty();
    const uint8_t* spreadData = hasResponseMap ? nullptr
                                               : static_cast<const uint8_t*>(levelData.spread.Data());

    const int32_t N = static_cast<int32_t>(rotatedFeatures.size());
    const int32_t earlyRejectScore = (earlyRejectThreshold > 0.0)
        ? static_cast<int32_t>(earlyRejectThreshold * N * LINEMOD_MAX_RESPONSE)
        : -1;
    constexpr int32_t EARLY_CHECK_INTERVAL = 16;

#ifdef __AVX2__
    __m256i vTotals = _mm256_setzero_si256();
    alignas(32) int32_t totals[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int32_t fi = 0; fi < N; ++fi) {
        const auto& f = rotatedFeatures[static_cast<size_t>(fi)];
        const int32_t fy = y + f.y;
        const int32_t fx = x + f.x;
        if (fy < 0 || fy >= H || fx < 0 || fx + 7 >= W) {
            continue;
        }

        const uint8_t* ptr = nullptr;
        if (hasResponseMap) {
            ptr = levelData.responseMaps[f.ori].data() + fy * stride + fx;
        } else {
            ptr = spreadData + fy * stride + fx;
        }

        if (hasResponseMap) {
            __m128i v8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr));
            __m256i v32 = _mm256_cvtepu8_epi32(v8);
            vTotals = _mm256_add_epi32(vTotals, v32);
        } else {
            for (int32_t i = 0; i < 8; ++i) {
                totals[i] += g_SimilarityLUT.Get(f.ori, ptr[i]);
            }
        }

        // Early termination check every EARLY_CHECK_INTERVAL features
        if (earlyRejectScore > 0 && ((fi + 1) % EARLY_CHECK_INTERVAL == 0)) {
            int32_t remaining = (N - fi - 1) * LINEMOD_MAX_RESPONSE;
            if (hasResponseMap) {
                // Extract current max from vTotals
                alignas(32) int32_t tmpTotals[8];
                _mm256_store_si256(reinterpret_cast<__m256i*>(tmpTotals), vTotals);
                int32_t maxVal = tmpTotals[0];
                for (int32_t i = 1; i < 8; ++i) {
                    if (tmpTotals[i] > maxVal) maxVal = tmpTotals[i];
                }
                if (maxVal + remaining < earlyRejectScore) {
                    return;  // All 8 positions will be below threshold
                }
            } else {
                int32_t maxVal = totals[0];
                for (int32_t i = 1; i < 8; ++i) {
                    if (totals[i] > maxVal) maxVal = totals[i];
                }
                if (maxVal + remaining < earlyRejectScore) {
                    return;
                }
            }
        }
    }

    if (hasResponseMap) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(totals), vTotals);
    }

    const double denom = static_cast<double>(N * LINEMOD_MAX_RESPONSE);
    for (int32_t i = 0; i < 8; ++i) {
        scoresOut[i] = static_cast<double>(totals[i]) / denom;
    }
#else
    int32_t totals[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int32_t fi = 0; fi < N; ++fi) {
        const auto& f = rotatedFeatures[static_cast<size_t>(fi)];
        const int32_t fy = y + f.y;
        const int32_t fx = x + f.x;
        if (fy < 0 || fy >= H || fx < 0 || fx + 7 >= W) {
            continue;
        }

        if (hasResponseMap) {
            const uint8_t* ptr = levelData.responseMaps[f.ori].data() + fy * stride + fx;
            for (int32_t i = 0; i < 8; ++i) {
                totals[i] += ptr[i];
            }
        } else {
            const uint8_t* ptr = spreadData + fy * stride + fx;
            for (int32_t i = 0; i < 8; ++i) {
                totals[i] += g_SimilarityLUT.Get(f.ori, ptr[i]);
            }
        }

        // Early termination check every EARLY_CHECK_INTERVAL features
        if (earlyRejectScore > 0 && ((fi + 1) % EARLY_CHECK_INTERVAL == 0)) {
            int32_t remaining = (N - fi - 1) * LINEMOD_MAX_RESPONSE;
            int32_t maxVal = totals[0];
            for (int32_t i = 1; i < 8; ++i) {
                if (totals[i] > maxVal) maxVal = totals[i];
            }
            if (maxVal + remaining < earlyRejectScore) {
                return;
            }
        }
    }

    const double denom = static_cast<double>(N * LINEMOD_MAX_RESPONSE);
    for (int32_t i = 0; i < 8; ++i) {
        scoresOut[i] = static_cast<double>(totals[i]) / denom;
    }
#endif
}

LinemodFeature LinemodPyramid::RotateFeature(const LinemodFeature& f, double angle) {
    const double cosA = std::cos(angle);
    const double sinA = std::sin(angle);

    // Rotate coordinates
    double rx = cosA * f.x - sinA * f.y;
    double ry = sinA * f.x + cosA * f.y;

    // Rotate orientation bin
    int32_t rotBinOffset = static_cast<int32_t>(std::round(angle * 8.0 / (2.0 * PI)));
    int32_t rotatedOri = ((f.ori + rotBinOffset) % 8 + 8) % 8;

    return LinemodFeature(
        static_cast<int16_t>(std::round(rx)),
        static_cast<int16_t>(std::round(ry)),
        static_cast<uint8_t>(rotatedOri)
    );
}

std::vector<LinemodFeature> LinemodPyramid::RotateFeatures(
    const std::vector<LinemodFeature>& features, double angle)
{
    std::vector<LinemodFeature> result;
    result.reserve(features.size());

    for (const auto& f : features) {
        result.push_back(RotateFeature(f, angle));
    }

    return result;
}

const uint8_t* LinemodPyramid::GetSpreadData(int32_t level) const {
    if (!valid_ || level < 0 || level >= numLevels_) {
        return nullptr;
    }
    return static_cast<const uint8_t*>(levels_[level].spread.Data());
}

int32_t LinemodPyramid::GetSpreadStride(int32_t level) const {
    if (!valid_ || level < 0 || level >= numLevels_) {
        return 0;
    }
    return static_cast<int32_t>(levels_[level].spread.Stride());
}

int32_t LinemodPyramid::GetWidth(int32_t level) const {
    if (!valid_ || level < 0 || level >= numLevels_) return 0;
    return levels_[level].width;
}

int32_t LinemodPyramid::GetHeight(int32_t level) const {
    if (!valid_ || level < 0 || level >= numLevels_) return 0;
    return levels_[level].height;
}

double LinemodPyramid::GetScale(int32_t level) const {
    if (!valid_ || level < 0 || level >= numLevels_) return 1.0;
    return levels_[level].scale;
}

const LinemodLevelData& LinemodPyramid::GetLevel(int32_t level) const {
    static const LinemodLevelData empty;
    if (!valid_ || level < 0 || level >= numLevels_) return empty;
    return levels_[level];
}

} // namespace Qi::Vision::Internal
