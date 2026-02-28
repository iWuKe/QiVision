/**
 * @file AnglePyramid.cpp
 * @brief Implementation of angle pyramid for shape-based matching
 *
 * Optimized with OpenMP parallelization and AVX2 SIMD
 */

#include <QiVision/Internal/AnglePyramid.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Internal/Gradient.h>
#include <QiVision/Internal/Gaussian.h>
#include <QiVision/Internal/Convolution.h>
#include <QiVision/Internal/Pyramid.h>
#include <QiVision/Internal/Interpolate.h>
#include <QiVision/Internal/Geometry2d.h>
#include <QiVision/Internal/NonMaxSuppression.h>
#include <QiVision/Core/Constants.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace Qi::Vision::Internal {

// =============================================================================
// OpenMP 初始化
// =============================================================================

#ifdef _OPENMP
namespace {
/// 自动设置 OpenMP 线程数（只执行一次）
struct OpenMPInitializer {
    OpenMPInitializer() {
        // 获取可用核心数，使用一半（避免超线程开销）
        int numProcs = omp_get_num_procs();
        int optimalThreads = std::max(1, std::min(numProcs / 2, 8));
        omp_set_num_threads(optimalThreads);
    }
};
static OpenMPInitializer g_ompInit;
}
#endif

// =============================================================================
// OpenMP 阈值控制
// =============================================================================

/// 启用 OpenMP 的最小像素数阈值 (约 700×700)
constexpr int32_t OPENMP_MIN_PIXELS = 200000;

/// 判断是否使用 OpenMP 并行
/// @param width 图像宽度
/// @param height 图像高度
/// @param forceDisable 强制禁用并行（优先级最高）
inline bool ShouldUseOpenMP(int32_t width, int32_t height, bool forceDisable = false) {
    if (forceDisable) return false;
    return static_cast<int64_t>(width) * height >= OPENMP_MIN_PIXELS;
}

// =============================================================================
// Threshold-based 16-bin direction quantization (matches decompiled)
// =============================================================================

/// Threshold-based 16-bin direction quantization (matches decompiled).
/// Returns 0-based bin (0..15). Caller adds +1 for 1-based output.
/// Uses only comparison and multiplication, no atan.
/// Uses dword_1800D6A60 = 0.19891236f ~ tan(11.25 deg) as threshold.
static inline int32_t quantize_bin_threshold_16(float gx, float gy) {
    const float t = 0.19891236f; // tan(11.25 deg)
    float abs_gx = std::abs(gx);
    float abs_gy = std::abs(gy);

    if (abs_gx >= abs_gy) {
        // Closer to x-axis
        if (abs_gy < abs_gx * t) {
            // Within 11.25 deg of x-axis
            return (gx >= 0) ? 0 : 8;
        } else {
            // Between 11.25 deg and 45 deg from x-axis
            if (gx >= 0)
                return (gy >= 0) ? 1 : 15;
            else
                return (gy >= 0) ? 7 : 9;
        }
    } else {
        // Closer to y-axis
        if (abs_gx < abs_gy * t) {
            // Within 11.25 deg of y-axis
            return (gy >= 0) ? 4 : 12;
        } else {
            // Between 11.25 deg and 45 deg from y-axis
            if (gy >= 0)
                return (gx >= 0) ? 3 : 5;
            else
                return (gx >= 0) ? 13 : 11;
        }
    }
}

/// Map a 16-bin result to numBins. For numBins==16, returns bin16 directly.
static inline int32_t quantize_bin_for_nbins(float gx, float gy, int32_t numBins) {
    int32_t bin16 = quantize_bin_threshold_16(gx, gy);
    if (numBins == 16) return bin16;
    return (bin16 * numBins) >> 4; // bin16 * numBins / 16
}

// =============================================================================
// SIMD Helper Functions
// =============================================================================

#ifdef __AVX2__

// Fast reciprocal using rcp_ps + Newton-Raphson iteration
// More accurate than raw rcp_ps (~0.0003% error vs ~0.4% error)
// Faster than _mm256_div_ps by ~2-4x
static inline __m256 fast_rcp_avx2(__m256 x) {
    // Initial estimate: rcp_ps has ~12 bits of precision
    __m256 rcp = _mm256_rcp_ps(x);

    // Newton-Raphson iteration: rcp' = rcp * (2 - x * rcp)
    // This gives ~23 bits of precision (enough for float)
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 x_rcp = _mm256_mul_ps(x, rcp);
    rcp = _mm256_mul_ps(rcp, _mm256_sub_ps(two, x_rcp));

    return rcp;
}

// Fast division: a / b = a * rcp(b) with Newton-Raphson
static inline __m256 fast_div_avx2(__m256 a, __m256 b) {
    return _mm256_mul_ps(a, fast_rcp_avx2(b));
}

/// AVX2 threshold-based 16-bin direction quantization (matches decompiled).
/// Returns 8x int32 bins (0..15). Uses scalar fallback per lane
/// to avoid complex SIMD branch logic. Performance impact is minimal
/// since quantization is a small fraction of overall computation.
static inline __m256i quantize_bin_threshold_16_avx2(__m256 gy, __m256 gx) {
    alignas(32) float gx_arr[8], gy_arr[8];
    alignas(32) int32_t bin_arr[8];
    _mm256_store_ps(gx_arr, gx);
    _mm256_store_ps(gy_arr, gy);

    for (int i = 0; i < 8; ++i) {
        bin_arr[i] = quantize_bin_threshold_16(gx_arr[i], gy_arr[i]);
    }

    return _mm256_load_si256(reinterpret_cast<const __m256i*>(bin_arr));
}

/// AVX2 threshold-based quantization with numBins support.
/// For numBins==16, returns bin16 directly. Otherwise scales.
static inline __m256i quantize_bin_avx2_nbins(__m256 gy, __m256 gx, int32_t numBins) {
    __m256i vBin16 = quantize_bin_threshold_16_avx2(gy, gx);
    if (numBins == 16) {
        return vBin16;
    }
    // Scale from 16 bins to numBins: bin = bin16 * numBins / 16
    __m256i vNum = _mm256_set1_epi32(numBins);
    __m256i vScaled = _mm256_mullo_epi32(vBin16, vNum);
    return _mm256_srli_epi32(vScaled, 4); // /16
}

// Fast atan2 approximation for 8 float values
// Using polynomial approximation, accurate to ~0.01 radians
static inline __m256 atan2_avx2(__m256 y, __m256 x) {
    // Constants
    const __m256 pi = _mm256_set1_ps(static_cast<float>(PI));
    const __m256 pi_2 = _mm256_set1_ps(static_cast<float>(PI / 2.0));
    const __m256 zero = _mm256_setzero_ps();

    // Polynomial coefficients for atan approximation
    const __m256 c1 = _mm256_set1_ps(0.9998660f);
    const __m256 c3 = _mm256_set1_ps(-0.3302995f);
    const __m256 c5 = _mm256_set1_ps(0.1801410f);
    const __m256 c7 = _mm256_set1_ps(-0.0851330f);

    // Compute |y| and |x|
    __m256 abs_y = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), y);
    __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);

    // Determine which octant we're in
    __m256 swap_mask = _mm256_cmp_ps(abs_y, abs_x, _CMP_GT_OQ);
    __m256 num = _mm256_blendv_ps(abs_y, abs_x, swap_mask);
    __m256 den = _mm256_blendv_ps(abs_x, abs_y, swap_mask);

    // Avoid division by zero
    den = _mm256_max_ps(den, _mm256_set1_ps(1e-10f));

    // t = num / den (in range [0, 1])
    __m256 t = fast_div_avx2(num, den);  // rcp + Newton-Raphson
    __m256 t2 = _mm256_mul_ps(t, t);

    // Polynomial: atan(t) ≈ t * (c1 + t² * (c3 + t² * (c5 + t² * c7)))
    __m256 poly = _mm256_fmadd_ps(t2, c7, c5);
    poly = _mm256_fmadd_ps(t2, poly, c3);
    poly = _mm256_fmadd_ps(t2, poly, c1);
    __m256 atan_val = _mm256_mul_ps(t, poly);

    // Adjust for octant: if swapped, result = π/2 - atan
    atan_val = _mm256_blendv_ps(atan_val, _mm256_sub_ps(pi_2, atan_val), swap_mask);

    // Adjust for quadrant
    __m256 x_neg = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
    __m256 y_neg = _mm256_cmp_ps(y, zero, _CMP_LT_OQ);

    // If x < 0: result = π - result
    atan_val = _mm256_blendv_ps(atan_val, _mm256_sub_ps(pi, atan_val), x_neg);

    // If y < 0: result = -result
    atan_val = _mm256_blendv_ps(atan_val, _mm256_sub_ps(zero, atan_val), y_neg);

    // Normalize to [0, 2π]
    __m256 neg_mask = _mm256_cmp_ps(atan_val, zero, _CMP_LT_OQ);
    __m256 two_pi = _mm256_set1_ps(static_cast<float>(2.0 * PI));
    atan_val = _mm256_blendv_ps(atan_val, _mm256_add_ps(atan_val, two_pi), neg_mask);

    return atan_val;
}

// Fast sqrt for 8 floats using rsqrt + Newton-Raphson
static inline __m256 fast_sqrt_avx2(__m256 x) {
    // sqrt(x) = x * rsqrt(x)
    // With one Newton-Raphson iteration for better accuracy
    __m256 zero = _mm256_setzero_ps();
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 three = _mm256_set1_ps(3.0f);

    // Initial estimate
    __m256 rsqrt_x = _mm256_rsqrt_ps(x);

    // Newton-Raphson: rsqrt' = 0.5 * rsqrt * (3 - x * rsqrt^2)
    __m256 x_rsqrt2 = _mm256_mul_ps(x, _mm256_mul_ps(rsqrt_x, rsqrt_x));
    rsqrt_x = _mm256_mul_ps(_mm256_mul_ps(half, rsqrt_x), _mm256_sub_ps(three, x_rsqrt2));

    // sqrt(x) = x * rsqrt(x), but handle zero case
    __m256 sqrt_x = _mm256_mul_ps(x, rsqrt_x);
    sqrt_x = _mm256_blendv_ps(sqrt_x, zero, _mm256_cmp_ps(x, zero, _CMP_EQ_OQ));

    return sqrt_x;
}
#endif

// =============================================================================
// Implementation Class
// =============================================================================

class AnglePyramid::Impl {
public:
    AnglePyramidParams params_;
    std::vector<PyramidLevelData> levels_;
    std::vector<PyramidLevel> gaussLevels_;  // Reuse Gaussian pyramid buffers
    std::vector<float> nmsMagBuf_;
    std::vector<float> nmsDirBuf_;
    std::vector<float> nmsOutBuf_;
    int32_t originalWidth_ = 0;
    int32_t originalHeight_ = 0;
    bool valid_ = false;
    AnglePyramidTiming timing_;  // Timing statistics

    bool BuildLevel(const std::vector<float>& srcData, int32_t width, int32_t height,
                    int32_t level, double scale);
    bool BuildLevelFused(const std::vector<float>& srcData, int32_t width, int32_t height,
                         int32_t level, double scale);
    bool BuildLevelFusedInto(PyramidLevelData& levelData, const std::vector<float>& srcData,
                             int32_t width, int32_t height, int32_t level, double scale);
    int32_t BuildGaussianPyramidReuse(const QImage& grayImage, const PyramidParams& params);
    void ExtractEdgePointsForLevel(int32_t level);

    // Optimized functions
    void ComputeGradientMagnitudeDirectionOpt(const float* gx, const float* gy,
                                               float* mag, float* dir,
                                               int32_t width, int32_t height,
                                               int32_t gxStride, int32_t gyStride,
                                               int32_t magStride, int32_t dirStride);

    void QuantizeDirectionOpt(const float* dir, int16_t* bins,
                              int32_t width, int32_t height,
                              int32_t dirStride, int32_t binStride,
                              int32_t numBins);

    // Fused: Sobel + Mag + Dir + Quantize in one pass
    void FusedSobelMagDirBin(const float* src, int32_t width, int32_t height,
                              float* gx, float* gy, float* mag, float* dir, int16_t* bins,
                              int32_t numBins);

    // Optimized: Compute Gx/Gy + Mag + Bins (skip Dir for search mode)
    void FusedSobelGradMagBin(const float* src, int32_t width, int32_t height,
                               float* gx, float* gy, float* mag, int16_t* bins,
                               int32_t numBins);

    // Direct write versions: write directly to QImage with stride (no copy)
    void FusedSobelGradMagBinDirect(const float* src, int32_t width, int32_t height,
                                    float* gx, int32_t gxStride, float* gy, int32_t gyStride,
                                    float* mag, int32_t magStride, int16_t* bins, int32_t binStride,
                                    int32_t numBins);

    void FusedSobelGradDirect(const float* src, int32_t width, int32_t height,
                              float* gx, int32_t gxStride, float* gy, int32_t gyStride);

    void FusedSobelMagDirBinDirect(const float* src, int32_t width, int32_t height,
                                   float* gx, int32_t gxStride, float* gy, int32_t gyStride,
                                   float* mag, int32_t magStride, float* dir, int32_t dirStride,
                                   int16_t* bins, int32_t binStride, int32_t numBins);
};

void AnglePyramid::Impl::ComputeGradientMagnitudeDirectionOpt(
    const float* gx, const float* gy,
    float* mag, float* dir,
    int32_t width, int32_t height,
    int32_t gxStride, int32_t gyStride,
    int32_t magStride, int32_t dirStride)
{
    const bool useParallel = ShouldUseOpenMP(width, height);

#ifdef __AVX2__
    const float twoPi = static_cast<float>(2.0 * PI);

    auto processRow = [&](int32_t y) {
        const float* gxRow = gx + y * gxStride;
        const float* gyRow = gy + y * gyStride;
        float* magRow = mag + y * magStride;
        float* dirRow = dir + y * dirStride;

        int32_t x = 0;
        // AVX2: process 8 pixels at a time
        for (; x + 8 <= width; x += 8) {
            __m256 vgx = _mm256_loadu_ps(gxRow + x);
            __m256 vgy = _mm256_loadu_ps(gyRow + x);

            // Magnitude: sqrt(gx^2 + gy^2)
            __m256 gx2 = _mm256_mul_ps(vgx, vgx);
            __m256 gy2 = _mm256_mul_ps(vgy, vgy);
            __m256 vmag = fast_sqrt_avx2(_mm256_add_ps(gx2, gy2));
            _mm256_storeu_ps(magRow + x, vmag);

            // Direction: atan2(gy, gx) normalized to [0, 2π]
            __m256 vdir = atan2_avx2(vgy, vgx);
            _mm256_storeu_ps(dirRow + x, vdir);
        }

        // Scalar fallback for remaining pixels
        for (; x < width; ++x) {
            float gxVal = gxRow[x];
            float gyVal = gyRow[x];
            magRow[x] = std::sqrt(gxVal * gxVal + gyVal * gyVal);
            float angle = std::atan2(gyVal, gxVal);
            if (angle < 0) angle += twoPi;
            dirRow[x] = angle;
        }
    };

#ifdef _OPENMP
    if (useParallel) {
        #pragma omp parallel for schedule(static)
        for (int32_t y = 0; y < height; ++y) {
            processRow(y);
        }
    } else
#endif
    {
        (void)useParallel;
        for (int32_t y = 0; y < height; ++y) {
            processRow(y);
        }
    }
#else
    // Non-SIMD version
    const float twoPi = static_cast<float>(2.0 * PI);

    auto processRow = [&](int32_t y) {
        const float* gxRow = gx + y * gxStride;
        const float* gyRow = gy + y * gyStride;
        float* magRow = mag + y * magStride;
        float* dirRow = dir + y * dirStride;

        for (int32_t x = 0; x < width; ++x) {
            float gxVal = gxRow[x];
            float gyVal = gyRow[x];
            magRow[x] = std::sqrt(gxVal * gxVal + gyVal * gyVal);
            float angle = std::atan2(gyVal, gxVal);
            if (angle < 0) angle += twoPi;
            dirRow[x] = angle;
        }
    };

#ifdef _OPENMP
    if (useParallel) {
        #pragma omp parallel for schedule(static)
        for (int32_t y = 0; y < height; ++y) {
            processRow(y);
        }
    } else
#endif
    {
        (void)useParallel;
        for (int32_t y = 0; y < height; ++y) {
            processRow(y);
        }
    }
#endif
}

void AnglePyramid::Impl::QuantizeDirectionOpt(
    const float* dir, int16_t* bins,
    int32_t width, int32_t height,
    int32_t dirStride, int32_t binStride,
    int32_t numBins)
{
    const float binScale = static_cast<float>(numBins / (2.0 * PI));
    const int16_t maxBin = static_cast<int16_t>(numBins - 1);
    const bool useParallel = ShouldUseOpenMP(width, height);

    // NOTE: This function only has direction data, not gx/gy.
    // All bins are output as 1-based (valid = 1..numBins).
    // Zero-gradient pixels (dir=0 from atan2(0,0)) will get bin=1,
    // which is not ideal but acceptable since callers should check magnitude.

#ifdef __AVX2__
    const __m256 vBinScale = _mm256_set1_ps(binScale);
    const __m256i vMaxBin = _mm256_set1_epi32(numBins - 1);
    const __m256i vZero = _mm256_setzero_si256();
    const __m256i vOne = _mm256_set1_epi32(1);

    auto processRow = [&](int32_t y) {
        const float* dirRow = dir + y * dirStride;
        int16_t* binRow = bins + y * binStride;

        int32_t x = 0;
        // AVX2: process 8 pixels at a time
        for (; x + 8 <= width; x += 8) {
            __m256 vdir = _mm256_loadu_ps(dirRow + x);
            __m256 vbin_f = _mm256_mul_ps(vdir, vBinScale);
            __m256i vbin = _mm256_cvttps_epi32(vbin_f);

            // Clamp to [0, numBins-1]
            vbin = _mm256_max_epi32(vbin, vZero);
            vbin = _mm256_min_epi32(vbin, vMaxBin);
            // +1 to make 1-based (valid = 1..numBins)
            vbin = _mm256_add_epi32(vbin, vOne);

            // Pack 32-bit integers to 16-bit
            __m128i lo = _mm256_castsi256_si128(vbin);
            __m128i hi = _mm256_extracti128_si256(vbin, 1);
            __m128i packed = _mm_packs_epi32(lo, hi);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(binRow + x), packed);
        }

        // Scalar fallback
        for (; x < width; ++x) {
            int32_t bin = static_cast<int32_t>(dirRow[x] * binScale);
            binRow[x] = static_cast<int16_t>(std::clamp(bin, 0, static_cast<int32_t>(maxBin)) + 1);  // 1-based
        }
    };

#ifdef _OPENMP
    if (useParallel) {
        #pragma omp parallel for schedule(static)
        for (int32_t y = 0; y < height; ++y) {
            processRow(y);
        }
    } else
#endif
    {
        (void)useParallel;
        for (int32_t y = 0; y < height; ++y) {
            processRow(y);
        }
    }
#else
    auto processRow = [&](int32_t y) {
        const float* dirRow = dir + y * dirStride;
        int16_t* binRow = bins + y * binStride;

        for (int32_t x = 0; x < width; ++x) {
            int32_t bin = static_cast<int32_t>(dirRow[x] * binScale);
            binRow[x] = static_cast<int16_t>(std::clamp(bin, 0, static_cast<int32_t>(maxBin)) + 1);  // 1-based
        }
    };

#ifdef _OPENMP
    if (useParallel) {
        #pragma omp parallel for schedule(static)
        for (int32_t y = 0; y < height; ++y) {
            processRow(y);
        }
    } else
#endif
    {
        (void)useParallel;
        for (int32_t y = 0; y < height; ++y) {
            processRow(y);
        }
    }
#endif
}

// =============================================================================
// Fused Sobel + Magnitude + Direction + Quantize (single pass)
// =============================================================================

void AnglePyramid::Impl::FusedSobelMagDirBin(
    const float* src, int32_t width, int32_t height,
    float* gx, float* gy, float* mag, float* dir, int16_t* bins,
    int32_t numBins)
{
    const float twoPi = static_cast<float>(2.0 * PI);
    const bool useParallel = ShouldUseOpenMP(width, height);

    // Sobel 3x3 kernel weights: [-1,0,1], [1,2,1]^T for Gx
    // For center pixels (y=1..height-2, x=1..width-2)

#ifdef __AVX2__

    auto processRow = [&](int32_t y) {
        const float* row0 = src + (y - 1) * width;
        const float* row1 = src + y * width;
        const float* row2 = src + (y + 1) * width;

        float* gxRow = gx + y * width;
        float* gyRow = gy + y * width;
        float* magRow = mag + y * width;
        float* dirRow = dir + y * width;
        int16_t* binRow = bins + y * width;

        // Border: x=0
        gxRow[0] = 0; gyRow[0] = 0; magRow[0] = 0; dirRow[0] = 0; binRow[0] = 0;

        int32_t x = 1;

        // AVX2 vectorized: process 8 pixels at a time
        for (; x + 8 < width; x += 8) {
            // Load 10 values for Sobel (need x-1 to x+8)
            // Gx = (p02 - p00) + 2*(p12 - p10) + (p22 - p20)
            // Gy = (p20 - p00) + 2*(p21 - p01) + (p22 - p02)

            __m256 p00 = _mm256_loadu_ps(row0 + x - 1);
            __m256 p01 = _mm256_loadu_ps(row0 + x);
            __m256 p02 = _mm256_loadu_ps(row0 + x + 1);

            __m256 p10 = _mm256_loadu_ps(row1 + x - 1);
            __m256 p12 = _mm256_loadu_ps(row1 + x + 1);

            __m256 p20 = _mm256_loadu_ps(row2 + x - 1);
            __m256 p21 = _mm256_loadu_ps(row2 + x);
            __m256 p22 = _mm256_loadu_ps(row2 + x + 1);

            // Gx = (p02 - p00) + 2*(p12 - p10) + (p22 - p20)
            __m256 diff_top = _mm256_sub_ps(p02, p00);
            __m256 diff_mid = _mm256_sub_ps(p12, p10);
            __m256 diff_bot = _mm256_sub_ps(p22, p20);
            __m256 vGx = _mm256_add_ps(diff_top, _mm256_add_ps(_mm256_add_ps(diff_mid, diff_mid), diff_bot));

            // Gy = (p20 - p00) + 2*(p21 - p01) + (p22 - p02)
            __m256 vert_left = _mm256_sub_ps(p20, p00);
            __m256 vert_mid = _mm256_sub_ps(p21, p01);
            __m256 vert_right = _mm256_sub_ps(p22, p02);
            __m256 vGy = _mm256_add_ps(vert_left, _mm256_add_ps(_mm256_add_ps(vert_mid, vert_mid), vert_right));

            // Store gradients
            _mm256_storeu_ps(gxRow + x, vGx);
            _mm256_storeu_ps(gyRow + x, vGy);

            // Magnitude = sqrt(gx^2 + gy^2)
            __m256 gx2 = _mm256_mul_ps(vGx, vGx);
            __m256 gy2 = _mm256_mul_ps(vGy, vGy);
            __m256 vMag = fast_sqrt_avx2(_mm256_add_ps(gx2, gy2));
            _mm256_storeu_ps(magRow + x, vMag);

            // Direction = atan2(gy, gx) in [0, 2π]
            __m256 vDir = atan2_avx2(vGy, vGx);
            _mm256_storeu_ps(dirRow + x, vDir);

            // Threshold-based bin quantization from gx/gy (matches decompiled)
            __m256i vBin = quantize_bin_avx2_nbins(vGy, vGx, numBins);
            // +1 to make 1-based (valid = 1..numBins, 0 = invalid/no gradient)
            vBin = _mm256_add_epi32(vBin, _mm256_set1_epi32(1));
            // Zero out bins where gradient is negligible
            {
                __m256 sign_mask_local = _mm256_set1_ps(-0.0f);
                __m256 abs_gx_local = _mm256_andnot_ps(sign_mask_local, vGx);
                __m256 abs_gy_local = _mm256_andnot_ps(sign_mask_local, vGy);
                __m256 maxGrad = _mm256_max_ps(abs_gx_local, abs_gy_local);
                __m256 gradThresh = _mm256_set1_ps(1e-6f);
                __m256 gradValid = _mm256_cmp_ps(maxGrad, gradThresh, _CMP_GT_OQ);
                vBin = _mm256_and_si256(vBin, _mm256_castps_si256(gradValid));
            }

            // Pack to int16
            __m128i lo = _mm256_castsi256_si128(vBin);
            __m128i hi = _mm256_extracti128_si256(vBin, 1);
            __m128i packed = _mm_packs_epi32(lo, hi);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(binRow + x), packed);
        }

        // Scalar fallback for remaining pixels
        for (; x < width - 1; ++x) {
            float p00 = row0[x - 1], p01 = row0[x], p02 = row0[x + 1];
            float p10 = row1[x - 1],               p12 = row1[x + 1];
            float p20 = row2[x - 1], p21 = row2[x], p22 = row2[x + 1];

            float gxVal = (p02 - p00) + 2.0f * (p12 - p10) + (p22 - p20);
            float gyVal = (p20 - p00) + 2.0f * (p21 - p01) + (p22 - p02);

            gxRow[x] = gxVal;
            gyRow[x] = gyVal;
            magRow[x] = std::sqrt(gxVal * gxVal + gyVal * gyVal);

            float angle = std::atan2(gyVal, gxVal);
            if (angle < 0) angle += twoPi;
            dirRow[x] = angle;

            float abs_gx_s = std::abs(gxVal);
            float abs_gy_s = std::abs(gyVal);
            if (std::max(abs_gx_s, abs_gy_s) < 1e-6f) {
                binRow[x] = 0;  // Invalid: no gradient
            } else {
                int32_t bin = quantize_bin_for_nbins(gxVal, gyVal, numBins);
                binRow[x] = static_cast<int16_t>(std::clamp(bin, 0, numBins - 1) + 1);  // 1-based
            }
        }

        // Border: x=width-1
        gxRow[width-1] = 0; gyRow[width-1] = 0; magRow[width-1] = 0;
        dirRow[width-1] = 0; binRow[width-1] = 0;
    };

#else
    // Scalar version
    auto processRow = [&](int32_t y) {
        const float* row0 = src + (y - 1) * width;
        const float* row1 = src + y * width;
        const float* row2 = src + (y + 1) * width;

        float* gxRow = gx + y * width;
        float* gyRow = gy + y * width;
        float* magRow = mag + y * width;
        float* dirRow = dir + y * width;
        int16_t* binRow = bins + y * width;

        // Border: x=0
        gxRow[0] = 0; gyRow[0] = 0; magRow[0] = 0; dirRow[0] = 0; binRow[0] = 0;

        for (int32_t x = 1; x < width - 1; ++x) {
            float p00 = row0[x - 1], p01 = row0[x], p02 = row0[x + 1];
            float p10 = row1[x - 1],               p12 = row1[x + 1];
            float p20 = row2[x - 1], p21 = row2[x], p22 = row2[x + 1];

            float gxVal = (p02 - p00) + 2.0f * (p12 - p10) + (p22 - p20);
            float gyVal = (p20 - p00) + 2.0f * (p21 - p01) + (p22 - p02);

            gxRow[x] = gxVal;
            gyRow[x] = gyVal;
            magRow[x] = std::sqrt(gxVal * gxVal + gyVal * gyVal);

            float angle = std::atan2(gyVal, gxVal);
            if (angle < 0) angle += twoPi;
            dirRow[x] = angle;

            float abs_gx_s = std::abs(gxVal);
            float abs_gy_s = std::abs(gyVal);
            if (std::max(abs_gx_s, abs_gy_s) < 1e-6f) {
                binRow[x] = 0;  // Invalid: no gradient
            } else {
                int32_t bin = quantize_bin_for_nbins(gxVal, gyVal, numBins);
                binRow[x] = static_cast<int16_t>(std::clamp(bin, 0, numBins - 1) + 1);  // 1-based
            }
        }

        // Border: x=width-1
        gxRow[width-1] = 0; gyRow[width-1] = 0; magRow[width-1] = 0;
        dirRow[width-1] = 0; binRow[width-1] = 0;
    };
#endif

    // First row (y=0): set to zero
    std::memset(gx, 0, width * sizeof(float));
    std::memset(gy, 0, width * sizeof(float));
    std::memset(mag, 0, width * sizeof(float));
    std::memset(dir, 0, width * sizeof(float));
    std::memset(bins, 0, width * sizeof(int16_t));

    // Process interior rows
#ifdef _OPENMP
    if (useParallel) {
        #pragma omp parallel for schedule(static)
        for (int32_t y = 1; y < height - 1; ++y) {
            processRow(y);
        }
    } else
#endif
    {
        (void)useParallel;
        for (int32_t y = 1; y < height - 1; ++y) {
            processRow(y);
        }
    }

    // Last row (y=height-1): set to zero
    int32_t lastRow = height - 1;
    std::memset(gx + lastRow * width, 0, width * sizeof(float));
    std::memset(gy + lastRow * width, 0, width * sizeof(float));
    std::memset(mag + lastRow * width, 0, width * sizeof(float));
    std::memset(dir + lastRow * width, 0, width * sizeof(float));
    std::memset(bins + lastRow * width, 0, width * sizeof(int16_t));
}

// =============================================================================
// FusedSobelGradMagBin: Optimized for search (gx/gy + mag + bins, skip dir)
// =============================================================================

void AnglePyramid::Impl::FusedSobelGradMagBin(
    const float* src, int32_t width, int32_t height,
    float* gx, float* gy, float* mag, int16_t* bins, int32_t numBins)
{
    const bool useParallel = ShouldUseOpenMP(width, height);

#ifdef __AVX2__
    auto processRow = [&](int32_t y) {
        const float* row0 = src + (y - 1) * width;
        const float* row1 = src + y * width;
        const float* row2 = src + (y + 1) * width;

        float* gxRow = gx + y * width;
        float* gyRow = gy + y * width;
        float* magRow = mag + y * width;
        int16_t* binRow = bins + y * width;

        // Border: x=0
        gxRow[0] = 0; gyRow[0] = 0; magRow[0] = 0; binRow[0] = 0;

        int32_t x = 1;

        // AVX2 vectorized: process 8 pixels at a time
        for (; x + 8 < width; x += 8) {
            __m256 p00 = _mm256_loadu_ps(row0 + x - 1);
            __m256 p01 = _mm256_loadu_ps(row0 + x);
            __m256 p02 = _mm256_loadu_ps(row0 + x + 1);

            __m256 p10 = _mm256_loadu_ps(row1 + x - 1);
            __m256 p12 = _mm256_loadu_ps(row1 + x + 1);

            __m256 p20 = _mm256_loadu_ps(row2 + x - 1);
            __m256 p21 = _mm256_loadu_ps(row2 + x);
            __m256 p22 = _mm256_loadu_ps(row2 + x + 1);

            // Gx = (p02 - p00) + 2*(p12 - p10) + (p22 - p20)
            __m256 diff_top = _mm256_sub_ps(p02, p00);
            __m256 diff_mid = _mm256_sub_ps(p12, p10);
            __m256 diff_bot = _mm256_sub_ps(p22, p20);
            __m256 vGx = _mm256_add_ps(diff_top, _mm256_add_ps(_mm256_add_ps(diff_mid, diff_mid), diff_bot));

            // Gy = (p20 - p00) + 2*(p21 - p01) + (p22 - p02)
            __m256 vert_left = _mm256_sub_ps(p20, p00);
            __m256 vert_mid = _mm256_sub_ps(p21, p01);
            __m256 vert_right = _mm256_sub_ps(p22, p02);
            __m256 vGy = _mm256_add_ps(vert_left, _mm256_add_ps(_mm256_add_ps(vert_mid, vert_mid), vert_right));

            // Store Gx/Gy
            _mm256_storeu_ps(gxRow + x, vGx);
            _mm256_storeu_ps(gyRow + x, vGy);

            // Magnitude = sqrt(gx^2 + gy^2)
            __m256 gx2 = _mm256_mul_ps(vGx, vGx);
            __m256 gy2 = _mm256_mul_ps(vGy, vGy);
            __m256 vMag = fast_sqrt_avx2(_mm256_add_ps(gx2, gy2));
            _mm256_storeu_ps(magRow + x, vMag);

            // Threshold-based bin quantization from gx/gy (matches decompiled)
            __m256i vBin = quantize_bin_avx2_nbins(vGy, vGx, numBins);
            // +1 to make 1-based (valid = 1..numBins, 0 = invalid/no gradient)
            vBin = _mm256_add_epi32(vBin, _mm256_set1_epi32(1));
            // Zero out bins where gradient is negligible
            {
                __m256 sign_mask_local = _mm256_set1_ps(-0.0f);
                __m256 abs_gx_local = _mm256_andnot_ps(sign_mask_local, vGx);
                __m256 abs_gy_local = _mm256_andnot_ps(sign_mask_local, vGy);
                __m256 maxGrad = _mm256_max_ps(abs_gx_local, abs_gy_local);
                __m256 gradThresh = _mm256_set1_ps(1e-6f);
                __m256 gradValid = _mm256_cmp_ps(maxGrad, gradThresh, _CMP_GT_OQ);
                vBin = _mm256_and_si256(vBin, _mm256_castps_si256(gradValid));
            }

            // Pack to int16
            __m128i lo = _mm256_castsi256_si128(vBin);
            __m128i hi = _mm256_extracti128_si256(vBin, 1);
            __m128i packed = _mm_packs_epi32(lo, hi);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(binRow + x), packed);
        }

        // Scalar fallback for remaining pixels (threshold-based, matches decompiled)
        for (; x < width - 1; ++x) {
            float p00 = row0[x - 1], p01 = row0[x], p02 = row0[x + 1];
            float p10 = row1[x - 1],               p12 = row1[x + 1];
            float p20 = row2[x - 1], p21 = row2[x], p22 = row2[x + 1];

            float gxVal = (p02 - p00) + 2.0f * (p12 - p10) + (p22 - p20);
            float gyVal = (p20 - p00) + 2.0f * (p21 - p01) + (p22 - p02);

            gxRow[x] = gxVal;
            gyRow[x] = gyVal;
            magRow[x] = std::sqrt(gxVal * gxVal + gyVal * gyVal);

            // Threshold-based bin quantization (matches decompiled)
            float abs_gx = std::abs(gxVal);
            float abs_gy = std::abs(gyVal);
            if (std::max(abs_gx, abs_gy) < 1e-6f) {
                binRow[x] = 0;  // Invalid: no gradient
            } else {
                int32_t bin = quantize_bin_for_nbins(gxVal, gyVal, numBins);
                binRow[x] = static_cast<int16_t>(std::clamp(bin, 0, numBins - 1) + 1);  // 1-based
            }
        }

        // Border: x=width-1
        gxRow[width-1] = 0; gyRow[width-1] = 0; magRow[width-1] = 0; binRow[width-1] = 0;
    };

#else
    // Non-AVX2 scalar version
    auto processRow = [&](int32_t y) {
        const float* row0 = src + (y - 1) * width;
        const float* row1 = src + y * width;
        const float* row2 = src + (y + 1) * width;

        float* gxRow = gx + y * width;
        float* gyRow = gy + y * width;
        float* magRow = mag + y * width;
        int16_t* binRow = bins + y * width;

        gxRow[0] = 0; gyRow[0] = 0; magRow[0] = 0; binRow[0] = 0;

        for (int32_t x = 1; x < width - 1; ++x) {
            float p00 = row0[x - 1], p01 = row0[x], p02 = row0[x + 1];
            float p10 = row1[x - 1],               p12 = row1[x + 1];
            float p20 = row2[x - 1], p21 = row2[x], p22 = row2[x + 1];

            float gxVal = (p02 - p00) + 2.0f * (p12 - p10) + (p22 - p20);
            float gyVal = (p20 - p00) + 2.0f * (p21 - p01) + (p22 - p02);

            gxRow[x] = gxVal;
            gyRow[x] = gyVal;
            magRow[x] = std::sqrt(gxVal * gxVal + gyVal * gyVal);

            float abs_gx_s = std::abs(gxVal);
            float abs_gy_s = std::abs(gyVal);
            if (std::max(abs_gx_s, abs_gy_s) < 1e-6f) {
                binRow[x] = 0;  // Invalid: no gradient
            } else {
                int32_t bin = quantize_bin_for_nbins(gxVal, gyVal, numBins);
                binRow[x] = static_cast<int16_t>(std::clamp(bin, 0, numBins - 1) + 1);  // 1-based
            }
        }

        gxRow[width-1] = 0; gyRow[width-1] = 0; magRow[width-1] = 0; binRow[width-1] = 0;
    };
#endif

    // First row: set to zero
    std::memset(gx, 0, width * sizeof(float));
    std::memset(gy, 0, width * sizeof(float));
    std::memset(mag, 0, width * sizeof(float));
    std::memset(bins, 0, width * sizeof(int16_t));

    // Process interior rows
#ifdef _OPENMP
    if (useParallel) {
        #pragma omp parallel for schedule(static)
        for (int32_t y = 1; y < height - 1; ++y) {
            processRow(y);
        }
    } else
#endif
    {
        (void)useParallel;
        for (int32_t y = 1; y < height - 1; ++y) {
            processRow(y);
        }
    }

    // Last row: set to zero
    int32_t lastRow = height - 1;
    std::memset(mag + lastRow * width, 0, width * sizeof(float));
    std::memset(bins + lastRow * width, 0, width * sizeof(int16_t));
}

// =============================================================================
// FusedSobelGradMagBinDirect: Direct write to QImage with stride (no copy)
// =============================================================================

void AnglePyramid::Impl::FusedSobelGradMagBinDirect(
    const float* src, int32_t width, int32_t height,
    float* gx, int32_t gxStride, float* gy, int32_t gyStride,
    float* mag, int32_t magStride, int16_t* bins, int32_t binStride,
    int32_t numBins)
{
    const bool useParallel = ShouldUseOpenMP(width, height);

#ifdef __AVX2__
    auto processRow = [&](int32_t y) {
        const float* row0 = src + (y - 1) * width;
        const float* row1 = src + y * width;
        const float* row2 = src + (y + 1) * width;

        float* gxRow = gx + y * gxStride;
        float* gyRow = gy + y * gyStride;
        float* magRow = mag + y * magStride;
        int16_t* binRow = bins + y * binStride;

        // Border: x=0
        gxRow[0] = 0; gyRow[0] = 0; magRow[0] = 0; binRow[0] = 0;

        int32_t x = 1;

        // AVX2 vectorized: process 8 pixels at a time
        for (; x + 8 < width; x += 8) {
            __m256 p00 = _mm256_loadu_ps(row0 + x - 1);
            __m256 p01 = _mm256_loadu_ps(row0 + x);
            __m256 p02 = _mm256_loadu_ps(row0 + x + 1);

            __m256 p10 = _mm256_loadu_ps(row1 + x - 1);
            __m256 p12 = _mm256_loadu_ps(row1 + x + 1);

            __m256 p20 = _mm256_loadu_ps(row2 + x - 1);
            __m256 p21 = _mm256_loadu_ps(row2 + x);
            __m256 p22 = _mm256_loadu_ps(row2 + x + 1);

            // Gx = (p02 - p00) + 2*(p12 - p10) + (p22 - p20)
            __m256 diff_top = _mm256_sub_ps(p02, p00);
            __m256 diff_mid = _mm256_sub_ps(p12, p10);
            __m256 diff_bot = _mm256_sub_ps(p22, p20);
            __m256 vGx = _mm256_add_ps(diff_top, _mm256_add_ps(_mm256_add_ps(diff_mid, diff_mid), diff_bot));

            // Gy = (p20 - p00) + 2*(p21 - p01) + (p22 - p02)
            __m256 vert_left = _mm256_sub_ps(p20, p00);
            __m256 vert_mid = _mm256_sub_ps(p21, p01);
            __m256 vert_right = _mm256_sub_ps(p22, p02);
            __m256 vGy = _mm256_add_ps(vert_left, _mm256_add_ps(_mm256_add_ps(vert_mid, vert_mid), vert_right));

            // Store Gx/Gy
            _mm256_storeu_ps(gxRow + x, vGx);
            _mm256_storeu_ps(gyRow + x, vGy);

            // Magnitude = sqrt(gx^2 + gy^2)
            __m256 gx2 = _mm256_mul_ps(vGx, vGx);
            __m256 gy2 = _mm256_mul_ps(vGy, vGy);
            __m256 vMag = fast_sqrt_avx2(_mm256_add_ps(gx2, gy2));
            _mm256_storeu_ps(magRow + x, vMag);

            // Threshold-based bin quantization from gx/gy (matches decompiled)
            __m256i vBin = quantize_bin_avx2_nbins(vGy, vGx, numBins);
            // +1 to make 1-based (valid = 1..numBins, 0 = invalid/no gradient)
            vBin = _mm256_add_epi32(vBin, _mm256_set1_epi32(1));
            // Zero out bins where gradient is negligible
            {
                __m256 sign_mask_local = _mm256_set1_ps(-0.0f);
                __m256 abs_gx_local = _mm256_andnot_ps(sign_mask_local, vGx);
                __m256 abs_gy_local = _mm256_andnot_ps(sign_mask_local, vGy);
                __m256 maxGrad = _mm256_max_ps(abs_gx_local, abs_gy_local);
                __m256 gradThresh = _mm256_set1_ps(1e-6f);
                __m256 gradValid = _mm256_cmp_ps(maxGrad, gradThresh, _CMP_GT_OQ);
                vBin = _mm256_and_si256(vBin, _mm256_castps_si256(gradValid));
            }

            // Pack to int16
            __m128i lo = _mm256_castsi256_si128(vBin);
            __m128i hi = _mm256_extracti128_si256(vBin, 1);
            __m128i packed = _mm_packs_epi32(lo, hi);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(binRow + x), packed);
        }

        // Scalar fallback for remaining pixels
        for (; x < width - 1; ++x) {
            float p00 = row0[x - 1], p01 = row0[x], p02 = row0[x + 1];
            float p10 = row1[x - 1],               p12 = row1[x + 1];
            float p20 = row2[x - 1], p21 = row2[x], p22 = row2[x + 1];

            float gxVal = (p02 - p00) + 2.0f * (p12 - p10) + (p22 - p20);
            float gyVal = (p20 - p00) + 2.0f * (p21 - p01) + (p22 - p02);

            gxRow[x] = gxVal;
            gyRow[x] = gyVal;
            magRow[x] = std::sqrt(gxVal * gxVal + gyVal * gyVal);

            // Threshold-based bin quantization (matches decompiled)
            float abs_gx = std::abs(gxVal);
            float abs_gy = std::abs(gyVal);
            if (std::max(abs_gx, abs_gy) < 1e-6f) {
                binRow[x] = 0;  // Invalid: no gradient
            } else {
                int32_t bin = quantize_bin_for_nbins(gxVal, gyVal, numBins);
                binRow[x] = static_cast<int16_t>(std::clamp(bin, 0, numBins - 1) + 1);  // 1-based
            }
        }

        // Border: x=width-1
        gxRow[width-1] = 0; gyRow[width-1] = 0; magRow[width-1] = 0; binRow[width-1] = 0;
    };

#else
    auto processRow = [&](int32_t y) {
        const float* row0 = src + (y - 1) * width;
        const float* row1 = src + y * width;
        const float* row2 = src + (y + 1) * width;

        float* gxRow = gx + y * gxStride;
        float* gyRow = gy + y * gyStride;
        float* magRow = mag + y * magStride;
        int16_t* binRow = bins + y * binStride;

        gxRow[0] = 0; gyRow[0] = 0; magRow[0] = 0; binRow[0] = 0;

        for (int32_t x = 1; x < width - 1; ++x) {
            float p00 = row0[x - 1], p01 = row0[x], p02 = row0[x + 1];
            float p10 = row1[x - 1],               p12 = row1[x + 1];
            float p20 = row2[x - 1], p21 = row2[x], p22 = row2[x + 1];

            float gxVal = (p02 - p00) + 2.0f * (p12 - p10) + (p22 - p20);
            float gyVal = (p20 - p00) + 2.0f * (p21 - p01) + (p22 - p02);

            gxRow[x] = gxVal;
            gyRow[x] = gyVal;
            magRow[x] = std::sqrt(gxVal * gxVal + gyVal * gyVal);

            float abs_gx_s = std::abs(gxVal);
            float abs_gy_s = std::abs(gyVal);
            if (std::max(abs_gx_s, abs_gy_s) < 1e-6f) {
                binRow[x] = 0;  // Invalid: no gradient
            } else {
                int32_t bin = quantize_bin_for_nbins(gxVal, gyVal, numBins);
                binRow[x] = static_cast<int16_t>(std::clamp(bin, 0, numBins - 1) + 1);  // 1-based
            }
        }

        gxRow[width-1] = 0; gyRow[width-1] = 0; magRow[width-1] = 0; binRow[width-1] = 0;
    };
#endif

    // First row: set to zero (with stride)
    for (int32_t x = 0; x < width; ++x) {
        gx[x] = 0; gy[x] = 0; mag[x] = 0; bins[x] = 0;
    }

    // Process interior rows with schedule(static, 64) for reduced overhead
#ifdef _OPENMP
    if (useParallel) {
        #pragma omp parallel for schedule(static, 64)
        for (int32_t y = 1; y < height - 1; ++y) {
            processRow(y);
        }
    } else
#endif
    {
        (void)useParallel;
        for (int32_t y = 1; y < height - 1; ++y) {
            processRow(y);
        }
    }

    // Last row: set to zero (with stride)
    int32_t lastRow = height - 1;
    float* lastGx = gx + lastRow * gxStride;
    float* lastGy = gy + lastRow * gyStride;
    float* lastMag = mag + lastRow * magStride;
    int16_t* lastBins = bins + lastRow * binStride;
    for (int32_t x = 0; x < width; ++x) {
        lastGx[x] = 0; lastGy[x] = 0; lastMag[x] = 0; lastBins[x] = 0;
    }
}


// =============================================================================
// FusedSobelGradDirect: Direct write of Gx/Gy only (fast search mode)
// =============================================================================

void AnglePyramid::Impl::FusedSobelGradDirect(
    const float* src, int32_t width, int32_t height,
    float* gx, int32_t gxStride, float* gy, int32_t gyStride)
{
    const bool useParallel = ShouldUseOpenMP(width, height);

#ifdef __AVX2__
    auto processRow = [&](int32_t y) {
        const float* row0 = src + (y - 1) * width;
        const float* row1 = src + y * width;
        const float* row2 = src + (y + 1) * width;

        float* gxRow = gx + y * gxStride;
        float* gyRow = gy + y * gyStride;

        gxRow[0] = 0; gyRow[0] = 0;

        int32_t x = 1;
        for (; x + 8 < width; x += 8) {
            __m256 p00 = _mm256_loadu_ps(row0 + x - 1);
            __m256 p01 = _mm256_loadu_ps(row0 + x);
            __m256 p02 = _mm256_loadu_ps(row0 + x + 1);

            __m256 p10 = _mm256_loadu_ps(row1 + x - 1);
            __m256 p12 = _mm256_loadu_ps(row1 + x + 1);

            __m256 p20 = _mm256_loadu_ps(row2 + x - 1);
            __m256 p21 = _mm256_loadu_ps(row2 + x);
            __m256 p22 = _mm256_loadu_ps(row2 + x + 1);

            __m256 diff_top = _mm256_sub_ps(p02, p00);
            __m256 diff_mid = _mm256_sub_ps(p12, p10);
            __m256 diff_bot = _mm256_sub_ps(p22, p20);
            __m256 vGx = _mm256_add_ps(diff_top, _mm256_add_ps(_mm256_add_ps(diff_mid, diff_mid), diff_bot));

            __m256 vert_left = _mm256_sub_ps(p20, p00);
            __m256 vert_mid = _mm256_sub_ps(p21, p01);
            __m256 vert_right = _mm256_sub_ps(p22, p02);
            __m256 vGy = _mm256_add_ps(vert_left, _mm256_add_ps(_mm256_add_ps(vert_mid, vert_mid), vert_right));

            _mm256_storeu_ps(gxRow + x, vGx);
            _mm256_storeu_ps(gyRow + x, vGy);
        }

        for (; x < width - 1; ++x) {
            float p00 = row0[x - 1], p01 = row0[x], p02 = row0[x + 1];
            float p10 = row1[x - 1],               p12 = row1[x + 1];
            float p20 = row2[x - 1], p21 = row2[x], p22 = row2[x + 1];

            float gxVal = (p02 - p00) + 2.0f * (p12 - p10) + (p22 - p20);
            float gyVal = (p20 - p00) + 2.0f * (p21 - p01) + (p22 - p02);

            gxRow[x] = gxVal;
            gyRow[x] = gyVal;
        }

        gxRow[width-1] = 0; gyRow[width-1] = 0;
    };
#else
    auto processRow = [&](int32_t y) {
        const float* row0 = src + (y - 1) * width;
        const float* row1 = src + y * width;
        const float* row2 = src + (y + 1) * width;

        float* gxRow = gx + y * gxStride;
        float* gyRow = gy + y * gyStride;

        gxRow[0] = 0; gyRow[0] = 0;
        for (int32_t x = 1; x < width - 1; ++x) {
            float p00 = row0[x - 1], p01 = row0[x], p02 = row0[x + 1];
            float p10 = row1[x - 1],               p12 = row1[x + 1];
            float p20 = row2[x - 1], p21 = row2[x], p22 = row2[x + 1];

            float gxVal = (p02 - p00) + 2.0f * (p12 - p10) + (p22 - p20);
            float gyVal = (p20 - p00) + 2.0f * (p21 - p01) + (p22 - p02);

            gxRow[x] = gxVal;
            gyRow[x] = gyVal;
        }
        gxRow[width-1] = 0; gyRow[width-1] = 0;
    };
#endif

#ifdef _OPENMP
    if (useParallel) {
        #pragma omp parallel for schedule(static, 64)
        for (int32_t y = 1; y < height - 1; ++y) {
            processRow(y);
        }
    } else
#endif
    {
        (void)useParallel;
        for (int32_t y = 1; y < height - 1; ++y) {
            processRow(y);
        }
    }

    // First/last row zero
    for (int32_t x = 0; x < width; ++x) {
        gx[x] = 0; gy[x] = 0;
    }
    int32_t lastRow = height - 1;
    float* lastGx = gx + lastRow * gxStride;
    float* lastGy = gy + lastRow * gyStride;
    for (int32_t x = 0; x < width; ++x) {
        lastGx[x] = 0; lastGy[x] = 0;
    }
}

// =============================================================================
// FusedSobelMagDirBinDirect: Direct write with direction storage
// =============================================================================

void AnglePyramid::Impl::FusedSobelMagDirBinDirect(
    const float* src, int32_t width, int32_t height,
    float* gx, int32_t gxStride, float* gy, int32_t gyStride,
    float* mag, int32_t magStride, float* dir, int32_t dirStride,
    int16_t* bins, int32_t binStride, int32_t numBins)
{
    const bool useParallel = ShouldUseOpenMP(width, height);
    const double twoPi = 2.0 * M_PI;

    auto processRow = [&](int32_t y) {
        const float* row0 = src + (y - 1) * width;
        const float* row1 = src + y * width;
        const float* row2 = src + (y + 1) * width;

        float* gxRow = gx + y * gxStride;
        float* gyRow = gy + y * gyStride;
        float* magRow = mag + y * magStride;
        float* dirRow = dir + y * dirStride;
        int16_t* binRow = bins + y * binStride;

        gxRow[0] = 0; gyRow[0] = 0; magRow[0] = 0; dirRow[0] = 0; binRow[0] = 0;

        for (int32_t x = 1; x < width - 1; ++x) {
            float p00 = row0[x - 1], p01 = row0[x], p02 = row0[x + 1];
            float p10 = row1[x - 1],               p12 = row1[x + 1];
            float p20 = row2[x - 1], p21 = row2[x], p22 = row2[x + 1];

            float gxVal = (p02 - p00) + 2.0f * (p12 - p10) + (p22 - p20);
            float gyVal = (p20 - p00) + 2.0f * (p21 - p01) + (p22 - p02);

            gxRow[x] = gxVal;
            gyRow[x] = gyVal;
            magRow[x] = std::sqrt(gxVal * gxVal + gyVal * gyVal);

            float angle = std::atan2(gyVal, gxVal);
            if (angle < 0) angle += twoPi;
            dirRow[x] = angle;

            float abs_gx_s = std::abs(gxVal);
            float abs_gy_s = std::abs(gyVal);
            if (std::max(abs_gx_s, abs_gy_s) < 1e-6f) {
                binRow[x] = 0;  // Invalid: no gradient
            } else {
                int32_t bin = quantize_bin_for_nbins(gxVal, gyVal, numBins);
                binRow[x] = static_cast<int16_t>(std::clamp(bin, 0, numBins - 1) + 1);  // 1-based
            }
        }

        gxRow[width-1] = 0; gyRow[width-1] = 0; magRow[width-1] = 0;
        dirRow[width-1] = 0; binRow[width-1] = 0;
    };

    // First row: set to zero (with stride)
    for (int32_t x = 0; x < width; ++x) {
        gx[x] = 0; gy[x] = 0; mag[x] = 0; dir[x] = 0; bins[x] = 0;
    }

    // Process interior rows with schedule(static, 64)
#ifdef _OPENMP
    if (useParallel) {
        #pragma omp parallel for schedule(static, 64)
        for (int32_t y = 1; y < height - 1; ++y) {
            processRow(y);
        }
    } else
#endif
    {
        (void)useParallel;
        for (int32_t y = 1; y < height - 1; ++y) {
            processRow(y);
        }
    }

    // Last row: set to zero (with stride)
    int32_t lastRow = height - 1;
    float* lastGx = gx + lastRow * gxStride;
    float* lastGy = gy + lastRow * gyStride;
    float* lastMag = mag + lastRow * magStride;
    float* lastDir = dir + lastRow * dirStride;
    int16_t* lastBins = bins + lastRow * binStride;
    for (int32_t x = 0; x < width; ++x) {
        lastGx[x] = 0; lastGy[x] = 0; lastMag[x] = 0; lastDir[x] = 0; lastBins[x] = 0;
    }
}


int32_t AnglePyramid::Impl::BuildGaussianPyramidReuse(const QImage& grayImage, const PyramidParams& params) {
    if (grayImage.Empty()) return 0;

    int32_t w = grayImage.Width();
    int32_t h = grayImage.Height();

    int32_t numLevels = params.numLevels;
    if (numLevels <= 0) {
        numLevels = ComputeNumLevels(w, h, params.scaleFactor, params.minDimension);
    }
    numLevels = std::max(1, numLevels);

    gaussLevels_.resize(numLevels);

    // Level 0: convert to float directly into reused buffer
    PyramidLevel& level0 = gaussLevels_[0];
    level0.width = w;
    level0.height = h;
    level0.scale = 1.0;
    level0.level = 0;
    level0.data.resize(static_cast<size_t>(w) * h);

    if (grayImage.Type() == PixelType::Float32) {
        int32_t srcStride = grayImage.Stride() / sizeof(float);
        const float* srcData = static_cast<const float*>(grayImage.Data());
        float* dstData = level0.data.data();
        if (srcStride == w) {
            std::memcpy(dstData, srcData, static_cast<size_t>(w) * h * sizeof(float));
        } else {
            for (int32_t y = 0; y < h; ++y) {
                std::memcpy(dstData + y * w, srcData + y * srcStride, w * sizeof(float));
            }
        }
    } else {
        const uint8_t* srcData = static_cast<const uint8_t*>(grayImage.Data());
        int32_t srcStride = grayImage.Stride();
        float* dstData = level0.data.data();

        const bool useParallel = ShouldUseOpenMP(w, h);

#ifdef __AVX2__
        auto convertRowAVX2 = [&](int32_t y) {
            const uint8_t* srcRow = srcData + y * srcStride;
            float* dstRow = dstData + y * w;
            int32_t x = 0;
            for (; x + 8 <= w; x += 8) {
                __m128i v8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcRow + x));
                __m256i v32 = _mm256_cvtepu8_epi32(v8);
                __m256 vf = _mm256_cvtepi32_ps(v32);
                _mm256_storeu_ps(dstRow + x, vf);
            }
            for (; x < w; ++x) {
                dstRow[x] = static_cast<float>(srcRow[x]);
            }
        };
#ifdef _OPENMP
        if (useParallel) {
            #pragma omp parallel for schedule(static)
            for (int32_t y = 0; y < h; ++y) { convertRowAVX2(y); }
        } else
#endif
        { for (int32_t y = 0; y < h; ++y) { convertRowAVX2(y); } }
#else
#ifdef _OPENMP
        if (useParallel) {
            #pragma omp parallel for schedule(static)
            for (int32_t y = 0; y < h; ++y) {
                const uint8_t* srcRow = srcData + y * srcStride;
                float* dstRow = dstData + y * w;
                for (int32_t x = 0; x < w; ++x) {
                    dstRow[x] = static_cast<float>(srcRow[x]);
                }
            }
        } else
#endif
        {
            (void)useParallel;
            for (int32_t y = 0; y < h; ++y) {
                const uint8_t* srcRow = srcData + y * srcStride;
                float* dstRow = dstData + y * w;
                for (int32_t x = 0; x < w; ++x) {
                    dstRow[x] = static_cast<float>(srcRow[x]);
                }
            }
        }
#endif
    }

    int32_t actualLevels = 1;

    for (int32_t lvl = 1; lvl < numLevels; ++lvl) {
        const PyramidLevel& prevLevel = gaussLevels_[lvl - 1];
        int32_t newWidth = prevLevel.width / 2;
        int32_t newHeight = prevLevel.height / 2;

        if (newWidth < params.minDimension || newHeight < params.minDimension) break;

        PyramidLevel& newLevel = gaussLevels_[lvl];
        newLevel.width = newWidth;
        newLevel.height = newHeight;
        newLevel.scale = prevLevel.scale * params.scaleFactor;
        newLevel.level = lvl;
        newLevel.data.resize(static_cast<size_t>(newWidth) * newHeight);

        DownsampleBy2(prevLevel.data.data(), prevLevel.width, prevLevel.height,
                      newLevel.data.data(), params.sigma, params.downsample);

        actualLevels = lvl + 1;
    }

    gaussLevels_.resize(actualLevels);
    return actualLevels;
}

// =============================================================================
// BuildLevelFused: Use fused computation
// =============================================================================


bool AnglePyramid::Impl::BuildLevelFusedInto(PyramidLevelData& levelData, const std::vector<float>& srcData,
                                             int32_t width, int32_t height,
                                             int32_t level, double scale) {
    levelData.level = level;
    levelData.width = width;
    levelData.height = height;
    levelData.scale = scale;

    size_t pixelCount = static_cast<size_t>(width) * height;
    const bool storeDir = params_.storeDirection;

    if (srcData.size() < pixelCount) {
        fprintf(stderr, "[BuildLevelFused] ERROR: srcData size %zu < pixelCount %zu\n",
                srcData.size(), pixelCount);
        return false;
    }
    if (width < 3 || height < 3) {
        fprintf(stderr, "[BuildLevelFused] ERROR: width=%d, height=%d too small\n", width, height);
        return false;
    }

    auto ensureImage = [&](QImage& img, int32_t w, int32_t h, PixelType type) {
        if (img.Width() != w || img.Height() != h || img.Type() != type ||
            img.GetChannelType() != ChannelType::Gray) {
            img = QImage(w, h, type, ChannelType::Gray);
        }
    };

    const bool minimalSearchMode = (!params_.extractEdgePoints && !params_.storeDirection);

    ensureImage(levelData.gradX, width, height, PixelType::Float32);
    ensureImage(levelData.gradY, width, height, PixelType::Float32);

    if (minimalSearchMode && !params_.storeAngleBins) {
        // Fully minimal mode: only Gx, Gy (no angleBin needed)
        levelData.gradMag = QImage();
        levelData.angleBinImage = QImage();
        levelData.gradDir = QImage();

        float* gxDst = static_cast<float*>(levelData.gradX.Data());
        float* gyDst = static_cast<float*>(levelData.gradY.Data());
        int32_t gxStride = levelData.gradX.Stride() / sizeof(float);
        int32_t gyStride = levelData.gradY.Stride() / sizeof(float);

        FusedSobelGradDirect(srcData.data(), width, height,
                             gxDst, gxStride, gyDst, gyStride);
        return true;
    }

    if (minimalSearchMode) {
        // Search mode + angleBin: compute Gx, Gy + angleBinImage (skip gradMag/gradDir storage)
        levelData.gradDir = QImage();

        float* gxDst = static_cast<float*>(levelData.gradX.Data());
        float* gyDst = static_cast<float*>(levelData.gradY.Data());
        int32_t gxStride = levelData.gradX.Stride() / sizeof(float);
        int32_t gyStride = levelData.gradY.Stride() / sizeof(float);

        // We need angleBinImage but can discard magnitude
        ensureImage(levelData.angleBinImage, width, height, PixelType::Int16);
        int16_t* binDst = static_cast<int16_t*>(levelData.angleBinImage.Data());
        int32_t binStride = levelData.angleBinImage.Stride() / sizeof(int16_t);

        // Use temporary buffer for magnitude (needed by FusedSobelGradMagBinDirect but not stored)
        ensureImage(levelData.gradMag, width, height, PixelType::Float32);
        float* magDst = static_cast<float*>(levelData.gradMag.Data());
        int32_t magStride = levelData.gradMag.Stride() / sizeof(float);

        FusedSobelGradMagBinDirect(srcData.data(), width, height,
                                    gxDst, gxStride, gyDst, gyStride,
                                    magDst, magStride, binDst, binStride,
                                    params_.angleBins);

        // We keep gradMag allocated (small overhead) since it's useful for threshold checking
        return true;
    }

    ensureImage(levelData.gradMag, width, height, PixelType::Float32);
    ensureImage(levelData.angleBinImage, width, height, PixelType::Int16);
    if (storeDir) {
        ensureImage(levelData.gradDir, width, height, PixelType::Float32);
    } else {
        levelData.gradDir = QImage();
    }

    float* gxDst = static_cast<float*>(levelData.gradX.Data());
    float* gyDst = static_cast<float*>(levelData.gradY.Data());
    float* magDst = static_cast<float*>(levelData.gradMag.Data());
    int16_t* binDst = static_cast<int16_t*>(levelData.angleBinImage.Data());
    int32_t gxStride = levelData.gradX.Stride() / sizeof(float);
    int32_t gyStride = levelData.gradY.Stride() / sizeof(float);
    int32_t magStride = levelData.gradMag.Stride() / sizeof(float);
    int32_t binStride = levelData.angleBinImage.Stride() / sizeof(int16_t);

    if (storeDir) {
        float* dirDst = static_cast<float*>(levelData.gradDir.Data());
        int32_t dirStride = levelData.gradDir.Stride() / sizeof(float);
        FusedSobelMagDirBinDirect(srcData.data(), width, height,
                                  gxDst, gxStride, gyDst, gyStride,
                                  magDst, magStride, dirDst, dirStride,
                                  binDst, binStride, params_.angleBins);
    } else {
        FusedSobelGradMagBinDirect(srcData.data(), width, height,
                                   gxDst, gxStride, gyDst, gyStride,
                                   magDst, magStride, binDst, binStride,
                                   params_.angleBins);
    }

    return true;
}

bool AnglePyramid::Impl::BuildLevelFused(const std::vector<float>& srcData, int32_t width, int32_t height,
                                          int32_t level, double scale) {
    PyramidLevelData levelData;
    if (!BuildLevelFusedInto(levelData, srcData, width, height, level, scale)) {
        return false;
    }
    levels_.push_back(std::move(levelData));
    return true;
}

bool AnglePyramid::Impl::BuildLevel(const std::vector<float>& srcData, int32_t width, int32_t height,
                                    int32_t level, double scale) {
    PyramidLevelData levelData;
    levelData.level = level;
    levelData.width = width;
    levelData.height = height;
    levelData.scale = scale;

    // Allocate contiguous buffers for gradients
    std::vector<float> gxBuffer(static_cast<size_t>(width) * height);
    std::vector<float> gyBuffer(static_cast<size_t>(width) * height);

    // Use Sobel operator for gradient computation (contiguous memory)
    Gradient<float, float>(srcData.data(), gxBuffer.data(), gyBuffer.data(),
                           width, height,
                           GradientOperator::Sobel3x3, BorderMode::Reflect101);

    // Create QImage for storage
    QImage gx(width, height, PixelType::Float32, ChannelType::Gray);
    QImage gy(width, height, PixelType::Float32, ChannelType::Gray);

    float* gxDst = static_cast<float*>(gx.Data());
    float* gyDst = static_cast<float*>(gy.Data());
    int32_t gxStride = gx.Stride() / sizeof(float);
    int32_t gyStride = gy.Stride() / sizeof(float);

    // Copy with conditional OpenMP parallelization
    const bool useParallel = ShouldUseOpenMP(width, height);

    auto copyRow = [&](int32_t y) {
        for (int32_t x = 0; x < width; ++x) {
            gxDst[y * gxStride + x] = gxBuffer[y * width + x];
            gyDst[y * gyStride + x] = gyBuffer[y * width + x];
        }
    };

#ifdef _OPENMP
    if (useParallel) {
        #pragma omp parallel for schedule(static)
        for (int32_t y = 0; y < height; ++y) {
            copyRow(y);
        }
    } else
#endif
    {
        (void)useParallel;
        for (int32_t y = 0; y < height; ++y) {
            copyRow(y);
        }
    }

    if (gx.Empty() || gy.Empty()) {
        return false;
    }

    levelData.gradX = std::move(gx);
    levelData.gradY = std::move(gy);

    // Allocate magnitude and direction images
    levelData.gradMag = QImage(width, height, PixelType::Float32, ChannelType::Gray);
    levelData.gradDir = QImage(width, height, PixelType::Float32, ChannelType::Gray);

    // Compute magnitude and direction with optimized SIMD + OpenMP
    const float* gxData = static_cast<const float*>(levelData.gradX.Data());
    const float* gyData = static_cast<const float*>(levelData.gradY.Data());
    float* magData = static_cast<float*>(levelData.gradMag.Data());
    float* dirData = static_cast<float*>(levelData.gradDir.Data());

    int32_t gxStr = levelData.gradX.Stride() / sizeof(float);
    int32_t gyStr = levelData.gradY.Stride() / sizeof(float);
    int32_t magStr = levelData.gradMag.Stride() / sizeof(float);
    int32_t dirStr = levelData.gradDir.Stride() / sizeof(float);

    ComputeGradientMagnitudeDirectionOpt(gxData, gyData, magData, dirData,
                                          width, height,
                                          gxStr, gyStr, magStr, dirStr);

    // Allocate and quantize direction to angle bins
    levelData.angleBinImage = QImage(width, height, PixelType::Int16, ChannelType::Gray);
    int16_t* binData = static_cast<int16_t*>(levelData.angleBinImage.Data());
    int32_t binStr = levelData.angleBinImage.Stride() / sizeof(int16_t);

    QuantizeDirectionOpt(dirData, binData, width, height, dirStr, binStr, params_.angleBins);

    levels_.push_back(std::move(levelData));
    return true;
}

void AnglePyramid::Impl::ExtractEdgePointsForLevel(int32_t level) {
    if (level < 0 || level >= static_cast<int32_t>(levels_.size())) {
        return;
    }

    auto& levelData = levels_[level];
    levelData.edgePoints.clear();

    const int32_t width = levelData.width;
    const int32_t height = levelData.height;

    const float* magData = static_cast<const float*>(levelData.gradMag.Data());
    const float* dirData = static_cast<const float*>(levelData.gradDir.Data());
    const int16_t* binData = static_cast<const int16_t*>(levelData.angleBinImage.Data());

    int32_t magStride = levelData.gradMag.Stride() / sizeof(float);
    int32_t dirStride = levelData.gradDir.Stride() / sizeof(float);
    int32_t binStride = levelData.angleBinImage.Stride() / sizeof(int16_t);

    const float minContrast = static_cast<float>(params_.minContrast);

    if (params_.useNMS) {
        // Original behavior: Apply Non-Maximum Suppression along gradient direction
        // This thins edges from "bands" to single-pixel-wide "ridges" (Canny-style)
        // NMS2DGradient expects contiguous data, so copy if stride != width
        const size_t nmsSize = static_cast<size_t>(width) * height;
        const float* magForNms = magData;
        const float* dirForNms = dirData;

        if (magStride != width) {
            if (nmsMagBuf_.size() < nmsSize) nmsMagBuf_.resize(nmsSize);
            for (int32_t y = 0; y < height; ++y) {
                std::memcpy(nmsMagBuf_.data() + y * width,
                            magData + y * magStride, width * sizeof(float));
            }
            magForNms = nmsMagBuf_.data();
        }
        if (dirStride != width) {
            if (nmsDirBuf_.size() < nmsSize) nmsDirBuf_.resize(nmsSize);
            for (int32_t y = 0; y < height; ++y) {
                std::memcpy(nmsDirBuf_.data() + y * width,
                            dirData + y * dirStride, width * sizeof(float));
            }
            dirForNms = nmsDirBuf_.data();
        }

        if (nmsOutBuf_.size() < nmsSize) nmsOutBuf_.resize(nmsSize);
        NMS2DGradient(magForNms, dirForNms, nmsOutBuf_.data(), width, height, minContrast);

        // Extract edge points from NMS output with subpixel refinement
        levelData.edgePoints.reserve(width * height / 20);

        for (int32_t y = 2; y < height - 2; ++y) {
            for (int32_t x = 2; x < width - 2; ++x) {
                float nmsMag = nmsOutBuf_[y * width + x];

                if (nmsMag >= minContrast) {
                    float dir = dirData[y * dirStride + x];
                    double cosDir = std::cos(dir);
                    double sinDir = std::sin(dir);

                    auto sampleMag = [&](double ox, double oy) -> double {
                        int32_t ix = x + static_cast<int32_t>(std::round(ox));
                        int32_t iy = y + static_cast<int32_t>(std::round(oy));
                        ix = std::clamp(ix, 0, width - 1);
                        iy = std::clamp(iy, 0, height - 1);
                        return magData[iy * magStride + ix];
                    };

                    double p0 = sampleMag(-cosDir, -sinDir);
                    double p1 = magData[y * magStride + x];
                    double p2 = sampleMag(cosDir, sinDir);

                    double denom = 2.0 * (p0 - 2.0 * p1 + p2);
                    double offset = 0.0;
                    if (std::abs(denom) > 1e-10) {
                        offset = (p0 - p2) / denom;
                        offset = std::clamp(offset, -0.5, 0.5);
                    }

                    double subX = x + offset * cosDir;
                    double subY = y + offset * sinDir;

                    int16_t bin = binData[y * binStride + x];

                    levelData.edgePoints.emplace_back(
                        subX, subY,
                        static_cast<double>(dir),
                        static_cast<double>(nmsMag),
                        static_cast<int32_t>(bin)
                    );
                }
            }
        }
    } else {
        // Shape-based matching mode (LINEMOD/Halcon style):
        // 1. NO Canny-style NMS
        // 2. Apply 3x3 neighborhood mode filter for robust orientation quantization
        //    IMPORTANT: Replace orientation with neighborhood mode, don't filter pixels!
        //
        // Reference: LINEMOD paper (Hinterstoisser et al. ICCV 2011)
        // "Quantized orientations are made more robust by taking the mode in a 3x3 neighborhood"
        //
        // Paper pseudo-code:
        //   for each pixel location x:
        //       quantized_value[x] = mode(quantized values in 3×3 neighborhood of x)

        levelData.edgePoints.reserve(width * height / 10);

        // Helper: compute mode (most frequent value) in 3x3 neighborhood
        // Returns both mode bin and the corresponding direction
        auto computeNeighborhoodMode = [&](int32_t cx, int32_t cy, float& modeDir) -> int16_t {
            // Count occurrences of each bin in 3x3 neighborhood
            // Only count pixels above contrast threshold
            std::array<int32_t, 64> binCounts = {};
            std::array<float, 64> binDirSum = {};  // Sum of directions for each bin
            int32_t maxCount = 0;
            int16_t modeBin = binData[cy * binStride + cx];

            for (int32_t dy = -1; dy <= 1; ++dy) {
                for (int32_t dx = -1; dx <= 1; ++dx) {
                    int32_t nx = cx + dx;
                    int32_t ny = cy + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        float neighborMag = magData[ny * magStride + nx];
                        if (neighborMag >= minContrast) {
                            int16_t neighborBin = binData[ny * binStride + nx];
                            if (neighborBin >= 0 && neighborBin < 64) {
                                binCounts[neighborBin]++;
                                binDirSum[neighborBin] += dirData[ny * dirStride + nx];
                                if (binCounts[neighborBin] > maxCount) {
                                    maxCount = binCounts[neighborBin];
                                    modeBin = neighborBin;
                                }
                            }
                        }
                    }
                }
            }

            // Compute average direction for the mode bin
            if (binCounts[modeBin] > 0) {
                modeDir = binDirSum[modeBin] / binCounts[modeBin];
            } else {
                modeDir = dirData[cy * dirStride + cx];
            }

            return modeBin;
        };

        for (int32_t y = 1; y < height - 1; ++y) {
            for (int32_t x = 1; x < width - 1; ++x) {
                float mag = magData[y * magStride + x];

                if (mag >= minContrast) {
                    // Robust quantization: REPLACE orientation with neighborhood mode
                    // (Don't filter pixels - keep all pixels above threshold)
                    float modeDir;
                    int16_t modeBin = computeNeighborhoodMode(x, y, modeDir);

                    levelData.edgePoints.emplace_back(
                        static_cast<double>(x),
                        static_cast<double>(y),
                        static_cast<double>(modeDir),  // Use mode direction
                        static_cast<double>(mag),
                        static_cast<int32_t>(modeBin)  // Use mode bin
                    );
                }
            }
        }
    }
}

// =============================================================================
// AnglePyramid Implementation
// =============================================================================

AnglePyramid::AnglePyramid() : impl_(std::make_unique<Impl>()) {}

AnglePyramid::~AnglePyramid() = default;

AnglePyramid::AnglePyramid(const AnglePyramid& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

AnglePyramid::AnglePyramid(AnglePyramid&& other) noexcept = default;

AnglePyramid& AnglePyramid::operator=(const AnglePyramid& other) {
    if (this != &other) {
        impl_ = std::make_unique<Impl>(*other.impl_);
    }
    return *this;
}

AnglePyramid& AnglePyramid::operator=(AnglePyramid&& other) noexcept = default;

bool AnglePyramid::Build(const QImage& image, const AnglePyramidParams& params) {
    Clear();

    if (image.Empty()) {
        return false;
    }

    // Reset timing
    impl_->timing_ = AnglePyramidTiming();
    auto tTotal = std::chrono::high_resolution_clock::now();

    // Validate parameters
    impl_->params_ = params;
    impl_->params_.numLevels = std::clamp(params.numLevels, 1, ANGLE_PYRAMID_MAX_LEVELS);
    impl_->params_.angleBins = std::clamp(params.angleBins, MIN_ANGLE_BINS, MAX_ANGLE_BINS);

    impl_->originalWidth_ = image.Width();
    impl_->originalHeight_ = image.Height();

    // Convert to grayscale if needed
    QImage grayImage = image;
    if (image.Channels() > 1) {
        // TODO: Convert to grayscale
        return false;
    }

#ifdef _OPENMP
    // RAII guard: set optimal thread count (12) for Search on exit
    // Pyramid builds with single thread for efficiency, Search uses 12 threads
    struct ThreadGuard {
        int optimal;
        ThreadGuard() : optimal(std::min(omp_get_num_procs(), 12)) {}
        ~ThreadGuard() { omp_set_num_threads(optimal); }
    } threadGuard;

    // Pyramid uses single thread (avoids OpenMP overhead)
    if (!params.useParallel) {
        omp_set_num_threads(1);
    }
#endif

    // Build Gaussian pyramid with buffer reuse
    PyramidParams pyramidParams;
    pyramidParams.numLevels = impl_->params_.numLevels;
    pyramidParams.sigma = std::max(1.0, params.smoothSigma);
    pyramidParams.minDimension = 8;  // Minimum 8 pixels at coarsest level

    auto tGaussPyramid = std::chrono::high_resolution_clock::now();
    int32_t actualLevels = impl_->BuildGaussianPyramidReuse(grayImage, pyramidParams);
    if (actualLevels <= 0) {
        return false;
    }

    if (params.enableTiming) {
        impl_->timing_.toFloatMs = 0.0;  // Conversion is fused into pyramid build
        impl_->timing_.copyMs = 0.0;
        impl_->timing_.gaussPyramidMs = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - tGaussPyramid).count();
    }

    // Update actual number of levels
    impl_->params_.numLevels = actualLevels;

    // Build angle pyramid for each level
    auto tSobel = std::chrono::high_resolution_clock::now();
    impl_->levels_.resize(actualLevels);

    bool ok = true;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(&&:ok)
#endif
    for (int32_t level = 0; level < actualLevels; ++level) {
        const auto& pyramidLevel = impl_->gaussLevels_[level];
        if (!impl_->BuildLevelFusedInto(impl_->levels_[level], pyramidLevel.data,
                                        pyramidLevel.width, pyramidLevel.height,
                                        level, pyramidLevel.scale)) {
            ok = false;
        }
    }

    if (!ok) {
        Clear();
        return false;
    }

    // Validate build results (fallback if any level is invalid)
    for (int32_t level = 0; level < actualLevels; ++level) {
        const auto& lvl = impl_->levels_[level];
        if (lvl.gradX.Empty() || lvl.gradY.Empty()) {
            Clear();
            return false;
        }
    }
    if (params.enableTiming) {
        impl_->timing_.sobelMs = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - tSobel).count();
    }

    // Extract edge points for each level (only if needed, e.g., for model creation)
    auto tExtractEdge = std::chrono::high_resolution_clock::now();
    if (params.extractEdgePoints) {
        // NOTE: ExtractEdgePointsForLevel reuses shared buffers; keep single-threaded.
        for (int32_t level = 0; level < static_cast<int32_t>(impl_->levels_.size()); ++level) {
            impl_->ExtractEdgePointsForLevel(level);
        }
    }
    if (params.enableTiming) {
        impl_->timing_.extractEdgeMs = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - tExtractEdge).count();
        impl_->timing_.totalMs = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - tTotal).count();
    }

    impl_->valid_ = true;
    return true;
}

void AnglePyramid::Clear() {
    impl_->levels_.clear();
    impl_->originalWidth_ = 0;
    impl_->originalHeight_ = 0;
    impl_->valid_ = false;
}

bool AnglePyramid::IsValid() const {
    return impl_->valid_;
}

int32_t AnglePyramid::NumLevels() const {
    return static_cast<int32_t>(impl_->levels_.size());
}

int32_t AnglePyramid::AngleBins() const {
    return impl_->params_.angleBins;
}

int32_t AnglePyramid::OriginalWidth() const {
    return impl_->originalWidth_;
}

int32_t AnglePyramid::OriginalHeight() const {
    return impl_->originalHeight_;
}

const AnglePyramidParams& AnglePyramid::GetParams() const {
    return impl_->params_;
}

const AnglePyramidTiming& AnglePyramid::GetTiming() const {
    return impl_->timing_;
}

const PyramidLevelData& AnglePyramid::GetLevel(int32_t level) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        throw OutOfRangeException("AnglePyramid::GetLevel: level out of range");
    }
    return impl_->levels_[level];
}

int32_t AnglePyramid::GetWidth(int32_t level) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return 0;
    }
    return impl_->levels_[level].width;
}

int32_t AnglePyramid::GetHeight(int32_t level) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return 0;
    }
    return impl_->levels_[level].height;
}

double AnglePyramid::GetScale(int32_t level) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return 0.0;
    }
    return impl_->levels_[level].scale;
}

double AnglePyramid::GetAngleAt(int32_t level, double x, double y) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return -1.0;
    }

    const auto& levelData = impl_->levels_[level];

    if (x < 0 || x >= levelData.width - 1 || y < 0 || y >= levelData.height - 1) {
        return -1.0;
    }

    // Bilinear interpolation of angle (careful with wrap-around)
    int32_t x0 = static_cast<int32_t>(x);
    int32_t y0 = static_cast<int32_t>(y);
    double fx = x - x0;
    double fy = y - y0;

    const float* dirData = static_cast<const float*>(levelData.gradDir.Data());
    int32_t stride = levelData.gradDir.Stride() / sizeof(float);

    // Get four corner angles
    double a00 = dirData[y0 * stride + x0];
    double a10 = dirData[y0 * stride + x0 + 1];
    double a01 = dirData[(y0 + 1) * stride + x0];
    double a11 = dirData[(y0 + 1) * stride + x0 + 1];

    // Handle angle wrap-around: convert to unit vectors and interpolate
    double cx = (1 - fx) * (1 - fy) * std::cos(a00) +
                fx * (1 - fy) * std::cos(a10) +
                (1 - fx) * fy * std::cos(a01) +
                fx * fy * std::cos(a11);

    double cy = (1 - fx) * (1 - fy) * std::sin(a00) +
                fx * (1 - fy) * std::sin(a10) +
                (1 - fx) * fy * std::sin(a01) +
                fx * fy * std::sin(a11);

    return NormalizeAngle0To2PI(std::atan2(cy, cx));
}

double AnglePyramid::GetMagnitudeAt(int32_t level, double x, double y) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return 0.0;
    }

    const auto& levelData = impl_->levels_[level];

    if (x < 0 || x >= levelData.width - 1 || y < 0 || y >= levelData.height - 1) {
        return 0.0;
    }

    // Bilinear interpolation
    int32_t x0 = static_cast<int32_t>(x);
    int32_t y0 = static_cast<int32_t>(y);
    double fx = x - x0;
    double fy = y - y0;

    const float* magData = static_cast<const float*>(levelData.gradMag.Data());
    int32_t stride = levelData.gradMag.Stride() / sizeof(float);

    double m00 = magData[y0 * stride + x0];
    double m10 = magData[y0 * stride + x0 + 1];
    double m01 = magData[(y0 + 1) * stride + x0];
    double m11 = magData[(y0 + 1) * stride + x0 + 1];

    return (1 - fx) * (1 - fy) * m00 +
           fx * (1 - fy) * m10 +
           (1 - fx) * fy * m01 +
           fx * fy * m11;
}

int32_t AnglePyramid::GetAngleBinAt(int32_t level, int32_t x, int32_t y) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return -1;
    }

    const auto& levelData = impl_->levels_[level];

    if (x < 0 || x >= levelData.width || y < 0 || y >= levelData.height) {
        return -1;
    }

    const int16_t* binData = static_cast<const int16_t*>(levelData.angleBinImage.Data());
    int32_t stride = levelData.angleBinImage.Stride() / sizeof(int16_t);

    return binData[y * stride + x];
}

bool AnglePyramid::GetGradientAt(int32_t level, double x, double y,
                                  double& gx, double& gy) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return false;
    }

    const auto& levelData = impl_->levels_[level];

    if (x < 0 || x >= levelData.width - 1 || y < 0 || y >= levelData.height - 1) {
        return false;
    }

    // Bilinear interpolation
    int32_t x0 = static_cast<int32_t>(x);
    int32_t y0 = static_cast<int32_t>(y);
    double fx = x - x0;
    double fy = y - y0;

    const float* gxData = static_cast<const float*>(levelData.gradX.Data());
    const float* gyData = static_cast<const float*>(levelData.gradY.Data());
    int32_t strideX = levelData.gradX.Stride() / sizeof(float);
    int32_t strideY = levelData.gradY.Stride() / sizeof(float);

    gx = (1 - fx) * (1 - fy) * gxData[y0 * strideX + x0] +
         fx * (1 - fy) * gxData[y0 * strideX + x0 + 1] +
         (1 - fx) * fy * gxData[(y0 + 1) * strideX + x0] +
         fx * fy * gxData[(y0 + 1) * strideX + x0 + 1];

    gy = (1 - fx) * (1 - fy) * gyData[y0 * strideY + x0] +
         fx * (1 - fy) * gyData[y0 * strideY + x0 + 1] +
         (1 - fx) * fy * gyData[(y0 + 1) * strideY + x0] +
         fx * fy * gyData[(y0 + 1) * strideY + x0 + 1];

    return true;
}

bool AnglePyramid::GetGradientAtFast(int32_t level, int32_t x, int32_t y,
                                      float& gx, float& gy) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return false;
    }

    const auto& levelData = impl_->levels_[level];

    if (x < 0 || x >= levelData.width || y < 0 || y >= levelData.height) {
        return false;
    }

    const float* gxData = static_cast<const float*>(levelData.gradX.Data());
    const float* gyData = static_cast<const float*>(levelData.gradY.Data());
    int32_t stride = levelData.gradX.Stride() / sizeof(float);

    gx = gxData[y * stride + x];
    gy = gyData[y * stride + x];

    return true;
}

bool AnglePyramid::GetGradientData(int32_t level, const float*& gxData, const float*& gyData,
                                    int32_t& width, int32_t& height, int32_t& stride) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return false;
    }

    const auto& levelData = impl_->levels_[level];
    gxData = static_cast<const float*>(levelData.gradX.Data());
    gyData = static_cast<const float*>(levelData.gradY.Data());
    width = levelData.width;
    height = levelData.height;
    stride = levelData.gradX.Stride() / sizeof(float);

    return true;
}

bool AnglePyramid::GetAngleBinData(int32_t level, const int16_t*& binData,
                                    int32_t& width, int32_t& height, int32_t& stride,
                                    int32_t& numBins) const {
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return false;
    }

    const auto& levelData = impl_->levels_[level];
    if (levelData.angleBinImage.Empty() || levelData.angleBinImage.Data() == nullptr) {
        return false;
    }
    binData = static_cast<const int16_t*>(levelData.angleBinImage.Data());
    width = levelData.width;
    height = levelData.height;
    stride = levelData.angleBinImage.Stride() / sizeof(int16_t);
    numBins = impl_->params_.angleBins;

    return true;
}

std::vector<EdgePoint> AnglePyramid::ExtractEdgePoints(int32_t level,
                                                        const Rect2i& roi,
                                                        double minContrast) const {
    std::vector<EdgePoint> result;

    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return result;
    }

    const auto& levelData = impl_->levels_[level];

    // Determine ROI
    int32_t x0 = 1, y0 = 1;
    int32_t x1 = levelData.width - 1;
    int32_t y1 = levelData.height - 1;

    if (roi.width > 0 && roi.height > 0) {
        x0 = std::max(1, roi.x);
        y0 = std::max(1, roi.y);
        x1 = std::min(levelData.width - 1, roi.x + roi.width);
        y1 = std::min(levelData.height - 1, roi.y + roi.height);
    }

    double threshold = (minContrast > 0) ? minContrast : impl_->params_.minContrast;

    const float* magData = static_cast<const float*>(levelData.gradMag.Data());
    const float* dirData = static_cast<const float*>(levelData.gradDir.Data());
    const int16_t* binData = static_cast<const int16_t*>(levelData.angleBinImage.Data());

    int32_t magStride = levelData.gradMag.Stride() / sizeof(float);
    int32_t dirStride = levelData.gradDir.Stride() / sizeof(float);
    int32_t binStride = levelData.angleBinImage.Stride() / sizeof(int16_t);

    const bool useParallel = ShouldUseOpenMP(levelData.width, levelData.height);

#ifdef _OPENMP
    if (useParallel) {
        // Use OpenMP with thread-local vectors
        #pragma omp parallel
        {
            std::vector<EdgePoint> localPoints;
            localPoints.reserve(1000);

            #pragma omp for schedule(static) nowait
            for (int32_t y = y0; y < y1; ++y) {
                for (int32_t x = x0; x < x1; ++x) {
                    float mag = magData[y * magStride + x];

                    if (mag >= threshold) {
                        float dir = dirData[y * dirStride + x];
                        int16_t bin = binData[y * binStride + x];

                        localPoints.emplace_back(
                            static_cast<double>(x),
                            static_cast<double>(y),
                            static_cast<double>(dir),
                            static_cast<double>(mag),
                            static_cast<int32_t>(bin)
                        );
                    }
                }
            }

            #pragma omp critical
            {
                result.insert(result.end(), localPoints.begin(), localPoints.end());
            }
        }
    } else
#endif
    {
        // 小图: 单线程执行
        (void)useParallel;
        result.reserve(1000);

        for (int32_t y = y0; y < y1; ++y) {
            for (int32_t x = x0; x < x1; ++x) {
                float mag = magData[y * magStride + x];

                if (mag >= threshold) {
                    float dir = dirData[y * dirStride + x];
                    int16_t bin = binData[y * binStride + x];

                    result.emplace_back(
                        static_cast<double>(x),
                        static_cast<double>(y),
                        static_cast<double>(dir),
                        static_cast<double>(mag),
                        static_cast<int32_t>(bin)
                    );
                }
            }
        }
    }

    return result;
}

std::vector<EdgePoint> AnglePyramid::ExtractEdgePoints(int32_t level,
                                                        const QRegion& region,
                                                        double minContrast) const {
    std::vector<EdgePoint> result;

    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return result;
    }

    if (region.Empty()) {
        // Empty region = full image, delegate to Rect2i version
        return ExtractEdgePoints(level, Rect2i(), minContrast);
    }

    const auto& levelData = impl_->levels_[level];

    // Get region bounding box for fast culling
    Rect2i bbox = region.BoundingBox();
    int32_t x0 = std::max(1, bbox.x);
    int32_t y0 = std::max(1, bbox.y);
    int32_t x1 = std::min(levelData.width - 1, bbox.x + bbox.width);
    int32_t y1 = std::min(levelData.height - 1, bbox.y + bbox.height);

    double threshold = (minContrast > 0) ? minContrast : impl_->params_.minContrast;

    const float* magData = static_cast<const float*>(levelData.gradMag.Data());
    const float* dirData = static_cast<const float*>(levelData.gradDir.Data());
    const int16_t* binData = static_cast<const int16_t*>(levelData.angleBinImage.Data());

    int32_t magStride = levelData.gradMag.Stride() / sizeof(float);
    int32_t dirStride = levelData.gradDir.Stride() / sizeof(float);
    int32_t binStride = levelData.angleBinImage.Stride() / sizeof(int16_t);

    const bool useParallel = ShouldUseOpenMP(levelData.width, levelData.height);

#ifdef _OPENMP
    if (useParallel) {
        // Use OpenMP with thread-local vectors
        #pragma omp parallel
        {
            std::vector<EdgePoint> localPoints;
            localPoints.reserve(1000);

            #pragma omp for schedule(static) nowait
            for (int32_t y = y0; y < y1; ++y) {
                for (int32_t x = x0; x < x1; ++x) {
                    // Check if point is within region
                    if (!region.Contains(x, y)) {
                        continue;
                    }

                    float mag = magData[y * magStride + x];

                    if (mag >= threshold) {
                        float dir = dirData[y * dirStride + x];
                        int16_t bin = binData[y * binStride + x];

                        localPoints.emplace_back(
                            static_cast<double>(x),
                            static_cast<double>(y),
                            static_cast<double>(dir),
                            static_cast<double>(mag),
                            static_cast<int32_t>(bin)
                        );
                    }
                }
            }

            #pragma omp critical
            {
                result.insert(result.end(), localPoints.begin(), localPoints.end());
            }
        }
    } else
#endif
    {
        // Small image: single-threaded execution
        (void)useParallel;
        result.reserve(1000);

        for (int32_t y = y0; y < y1; ++y) {
            for (int32_t x = x0; x < x1; ++x) {
                // Check if point is within region
                if (!region.Contains(x, y)) {
                    continue;
                }

                float mag = magData[y * magStride + x];

                if (mag >= threshold) {
                    float dir = dirData[y * dirStride + x];
                    int16_t bin = binData[y * binStride + x];

                    result.emplace_back(
                        static_cast<double>(x),
                        static_cast<double>(y),
                        static_cast<double>(dir),
                        static_cast<double>(mag),
                        static_cast<int32_t>(bin)
                    );
                }
            }
        }
    }

    return result;
}

const std::vector<EdgePoint>& AnglePyramid::GetEdgePoints(int32_t level) const {
    static const std::vector<EdgePoint> empty;
    if (level < 0 || level >= static_cast<int32_t>(impl_->levels_.size())) {
        return empty;
    }
    return impl_->levels_[level].edgePoints;
}

Point2d AnglePyramid::ToLevelCoords(int32_t level, const Point2d& original) const {
    double scale = GetScale(level);
    if (scale <= 0) return original;
    return Point2d{original.x * scale, original.y * scale};
}

Point2d AnglePyramid::ToOriginalCoords(int32_t level, const Point2d& levelCoords) const {
    double scale = GetScale(level);
    if (scale <= 0) return levelCoords;
    return Point2d{levelCoords.x / scale, levelCoords.y / scale};
}

int32_t AnglePyramid::AngleToBin(double angle) const {
    double normalized = NormalizeAngle0To2PI(angle);
    int32_t bin = static_cast<int32_t>(normalized * impl_->params_.angleBins / (2.0 * PI));
    return std::clamp(bin, 0, impl_->params_.angleBins - 1);
}

double AnglePyramid::BinToAngle(int32_t bin) const {
    return (bin + 0.5) * 2.0 * PI / impl_->params_.angleBins;
}

double AnglePyramid::AngleDifference(double angle1, double angle2) {
    double diff = std::abs(angle1 - angle2);
    if (diff > PI) {
        diff = 2.0 * PI - diff;
    }
    return diff;
}

double AnglePyramid::AngleSimilarity(double angle1, double angle2) {
    return std::cos(angle1 - angle2);
}

// =============================================================================
// Utility Functions (also optimized)
// =============================================================================

QImage ComputeGradientDirection(const QImage& gradX, const QImage& gradY) {
    if (gradX.Empty() || gradY.Empty()) {
        return QImage();
    }

    if (gradX.Width() != gradY.Width() || gradX.Height() != gradY.Height()) {
        return QImage();
    }

    int32_t width = gradX.Width();
    int32_t height = gradX.Height();

    QImage result(width, height, PixelType::Float32, ChannelType::Gray);

    const float* gxData = static_cast<const float*>(gradX.Data());
    const float* gyData = static_cast<const float*>(gradY.Data());
    float* outData = static_cast<float*>(result.Data());

    int32_t gxStride = gradX.Stride() / sizeof(float);
    int32_t gyStride = gradY.Stride() / sizeof(float);
    int32_t outStride = result.Stride() / sizeof(float);

    const float twoPi = static_cast<float>(2.0 * PI);
    const bool useParallel = ShouldUseOpenMP(width, height);

    auto processRow = [&](int32_t y) {
        for (int32_t x = 0; x < width; ++x) {
            float gx = gxData[y * gxStride + x];
            float gy = gyData[y * gyStride + x];
            float angle = std::atan2(gy, gx);
            if (angle < 0) angle += twoPi;
            outData[y * outStride + x] = angle;
        }
    };

#ifdef _OPENMP
    if (useParallel) {
        #pragma omp parallel for schedule(static)
        for (int32_t y = 0; y < height; ++y) {
            processRow(y);
        }
    } else
#endif
    {
        (void)useParallel;
        for (int32_t y = 0; y < height; ++y) {
            processRow(y);
        }
    }

    return result;
}

QImage ComputeGradientMagnitude(const QImage& gradX, const QImage& gradY) {
    if (gradX.Empty() || gradY.Empty()) {
        return QImage();
    }

    if (gradX.Width() != gradY.Width() || gradX.Height() != gradY.Height()) {
        return QImage();
    }

    int32_t width = gradX.Width();
    int32_t height = gradX.Height();

    QImage result(width, height, PixelType::Float32, ChannelType::Gray);

    const float* gxData = static_cast<const float*>(gradX.Data());
    const float* gyData = static_cast<const float*>(gradY.Data());
    float* outData = static_cast<float*>(result.Data());

    int32_t gxStride = gradX.Stride() / sizeof(float);
    int32_t gyStride = gradY.Stride() / sizeof(float);
    int32_t outStride = result.Stride() / sizeof(float);

    const bool useParallel = ShouldUseOpenMP(width, height);

#ifdef __AVX2__
    auto processRow = [&](int32_t y) {
        const float* gxRow = gxData + y * gxStride;
        const float* gyRow = gyData + y * gyStride;
        float* outRow = outData + y * outStride;

        int32_t x = 0;
        for (; x + 8 <= width; x += 8) {
            __m256 vgx = _mm256_loadu_ps(gxRow + x);
            __m256 vgy = _mm256_loadu_ps(gyRow + x);
            __m256 gx2 = _mm256_mul_ps(vgx, vgx);
            __m256 gy2 = _mm256_mul_ps(vgy, vgy);
            __m256 mag = _mm256_sqrt_ps(_mm256_add_ps(gx2, gy2));
            _mm256_storeu_ps(outRow + x, mag);
        }
        for (; x < width; ++x) {
            float gx = gxRow[x];
            float gy = gyRow[x];
            outRow[x] = std::sqrt(gx * gx + gy * gy);
        }
    };

#ifdef _OPENMP
    if (useParallel) {
        #pragma omp parallel for schedule(static)
        for (int32_t y = 0; y < height; ++y) {
            processRow(y);
        }
    } else
#endif
    {
        (void)useParallel;
        for (int32_t y = 0; y < height; ++y) {
            processRow(y);
        }
    }
#else
    auto processRow = [&](int32_t y) {
        for (int32_t x = 0; x < width; ++x) {
            float gx = gxData[y * gxStride + x];
            float gy = gyData[y * gyStride + x];
            outData[y * outStride + x] = std::sqrt(gx * gx + gy * gy);
        }
    };

#ifdef _OPENMP
    if (useParallel) {
        #pragma omp parallel for schedule(static)
        for (int32_t y = 0; y < height; ++y) {
            processRow(y);
        }
    } else
#endif
    {
        (void)useParallel;
        for (int32_t y = 0; y < height; ++y) {
            processRow(y);
        }
    }
#endif

    return result;
}

QImage QuantizeGradientDirection(const QImage& gradDir, int32_t numBins) {
    if (gradDir.Empty() || numBins <= 0) {
        return QImage();
    }

    int32_t width = gradDir.Width();
    int32_t height = gradDir.Height();

    QImage result(width, height, PixelType::Int16, ChannelType::Gray);

    const float* dirData = static_cast<const float*>(gradDir.Data());
    int16_t* outData = static_cast<int16_t*>(result.Data());

    int32_t dirStride = gradDir.Stride() / sizeof(float);
    int32_t outStride = result.Stride() / sizeof(int16_t);

    double binScale = numBins / (2.0 * PI);
    const bool useParallel = ShouldUseOpenMP(width, height);

    auto processRow = [&](int32_t y) {
        for (int32_t x = 0; x < width; ++x) {
            float angle = dirData[y * dirStride + x];
            int32_t bin = static_cast<int32_t>(angle * binScale);
            bin = std::clamp(bin, 0, numBins - 1);
            outData[y * outStride + x] = static_cast<int16_t>(bin);
        }
    };

#ifdef _OPENMP
    if (useParallel) {
        #pragma omp parallel for schedule(static)
        for (int32_t y = 0; y < height; ++y) {
            processRow(y);
        }
    } else
#endif
    {
        (void)useParallel;
        for (int32_t y = 0; y < height; ++y) {
            processRow(y);
        }
    }

    return result;
}

int32_t ComputeOptimalPyramidLevels(int32_t width, int32_t height, int32_t minSize) {
    int32_t minDim = std::min(width, height);
    int32_t levels = 1;

    while (minDim >= minSize * 2) {
        minDim /= 2;
        levels++;
    }

    return std::min(levels, ANGLE_PYRAMID_MAX_LEVELS);
}

} // namespace Qi::Vision::Internal
