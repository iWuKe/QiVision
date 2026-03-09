/**
 * @file ShapeModelResponseMap.h
 * @brief Response Map + LUT infrastructure for coarse shape matching
 *
 * Implements the Halcon-style response map scoring:
 * - 16x16 cos-based LUT (matches decompiled byte_180108E70)
 * - int16 response map accumulation per rotation angle
 * - 3x3 NMS candidate extraction
 *
 * Score = sum(LUT[imageBin][modelBin]) / (numPoints * 127)
 */

#pragma once

#include <QiVision/Core/Constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// SIMD intrinsics for BuildResponseMap acceleration
#if defined(__AVX2__) && !defined(HAVE_AVX2)
#include <immintrin.h>
#define HAVE_AVX2 1
#elif defined(__SSE4_1__) && !defined(HAVE_SSE4)
#include <smmintrin.h>
#define HAVE_SSE4 1
#endif

namespace Qi::Vision::Matching::Internal {

// =============================================================================
// 16x16 cos-based LUT (matches decompiled byte_180108E70)
// =============================================================================

struct ResponseMapLUT {
    int8_t table[16][16];  ///< table[imageBin][modelBin]

    ResponseMapLUT() {
        for (int ib = 0; ib < 16; ++ib)
            for (int mb = 0; mb < 16; ++mb)
                table[ib][mb] = static_cast<int8_t>(
                    std::round(127.0 * std::cos(TWO_PI * (ib - mb) / 16.0)));
    }
};

// =============================================================================
// 2D int16 response map
// =============================================================================

struct ResponseMap {
    std::vector<int16_t> data;
    int32_t width = 0, height = 0, stride = 0;

    void Allocate(int32_t w, int32_t h) {
        width = w; height = h;
        stride = (w + 15) & ~15;  // 16-element alignment for SIMD
        data.resize(static_cast<size_t>(stride) * h, 0);
    }

    void Clear() {
        std::memset(data.data(), 0, data.size() * sizeof(int16_t));
    }

    int16_t& At(int32_t x, int32_t y) {
        return data[static_cast<size_t>(y) * stride + x];
    }

    int16_t At(int32_t x, int32_t y) const {
        return data[static_cast<size_t>(y) * stride + x];
    }
};

// =============================================================================
// Rotated model point for response map accumulation
// =============================================================================

struct ResponsePoint {
    int32_t offsetX, offsetY;  ///< Rotated integer offset from search position
    int32_t angleBin16;        ///< Model direction quantized to 16 bins (after rotation)
};

// =============================================================================
// NMS candidate from response map
// =============================================================================

struct ResponseCandidate {
    int32_t x, y;
    int16_t response;
};

// =============================================================================
// Core functions
// =============================================================================

/**
 * @brief Accumulate model contributions to response map
 *
 * For each search position (x,y) in the search region, sums LUT[imageBin][modelBin]
 * across all model points. The result is a 2D map of raw response values.
 *
 * @param map         Output response map (must be pre-allocated and cleared)
 * @param modelPoints Rotated model points with integer offsets and 16-bin angle
 * @param angleBinImage Search image angle bin data (int16_t, 16 bins)
 * @param binWidth    Angle bin image width
 * @param binHeight   Angle bin image height
 * @param binStride   Angle bin image stride (in elements)
 * @param lut         16x16 cos-based LUT
 * @param searchXMin/Max/YMin/Max  Search region bounds (inclusive)
 */
/// Max search width for stack-allocated contribution buffer (matches decompiled 0x2000)
constexpr int32_t MAX_RESPONSE_MAP_WIDTH = 8192;

inline void BuildResponseMap(
    ResponseMap& map,
    const std::vector<ResponsePoint>& modelPoints,
    const int16_t* angleBinImage, int32_t binWidth, int32_t binHeight, int32_t binStride,
    const ResponseMapLUT& lut,
    int32_t searchXMin, int32_t searchXMax,
    int32_t searchYMin, int32_t searchYMax,
    bool ignorePolarity = false)
{
    const int32_t searchW = searchXMax - searchXMin + 1;
    if (searchW <= 0) return;

    // Heap fallback for ultra-wide images
    std::vector<int8_t> heapBuf;
    if (searchW > MAX_RESPONSE_MAP_WIDTH) {
        heapBuf.resize(searchW);
    }

    for (const auto& pt : modelPoints) {
        const int32_t mb = pt.angleBin16;

        // Pre-compute per-model-bin LUT row (decompiled pattern)
        int8_t mbLut[16];
        if (ignorePolarity) {
            for (int ib = 0; ib < 16; ++ib)
                mbLut[ib] = static_cast<int8_t>(
                    lut.table[ib][mb] < 0 ? -lut.table[ib][mb] : lut.table[ib][mb]);
        } else {
            for (int ib = 0; ib < 16; ++ib)
                mbLut[ib] = lut.table[ib][mb];
        }

        for (int32_t y = searchYMin; y <= searchYMax; ++y) {
            int32_t imgY = y + pt.offsetY;
            if (imgY < 0 || imgY >= binHeight) continue;

            const int16_t* binRow = angleBinImage + static_cast<size_t>(imgY) * binStride;
            int16_t* mapRow = map.data.data() + static_cast<size_t>(y) * map.stride;

            // Phase 1: Pre-compute int8 contribution buffer for this row
            int8_t stackBuf[MAX_RESPONSE_MAP_WIDTH];
            int8_t* contribBuf = (searchW <= MAX_RESPONSE_MAP_WIDTH) ? stackBuf : heapBuf.data();

            for (int32_t x = searchXMin; x <= searchXMax; ++x) {
                int32_t imgX = x + pt.offsetX;
                int32_t xi = x - searchXMin;
                if (imgX < 0 || imgX >= binWidth) {
                    contribBuf[xi] = 0;
                    continue;
                }
                int16_t imageBin = binRow[imgX];
                contribBuf[xi] = (imageBin >= 1 && imageBin <= 16)
                    ? mbLut[imageBin - 1] : 0;
            }

            // Phase 2: SIMD accumulation — vpmovsxbw + vpaddsw
            int16_t* mapPtr = mapRow + searchXMin;
            int32_t xi = 0;

#if HAVE_AVX2
            // Process 16 int8 → 16 int16 per iteration (matches decompiled pattern)
            for (; xi + 16 <= searchW; xi += 16) {
                __m128i vContrib8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(contribBuf + xi));
                __m256i vContrib16 = _mm256_cvtepi8_epi16(vContrib8);
                __m256i vMap = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mapPtr + xi));
                vMap = _mm256_adds_epi16(vMap, vContrib16);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(mapPtr + xi), vMap);
            }
#endif
            // Scalar tail
            for (; xi < searchW; ++xi) {
                int32_t sum = static_cast<int32_t>(mapPtr[xi]) + contribBuf[xi];
                mapPtr[xi] = static_cast<int16_t>(std::clamp(sum, -32768, 32767));
            }
        }
    }
}

/**
 * @brief Extract local maxima via 3x3 NMS on response map
 *
 * A position is a local maximum if its response >= all 8 neighbors
 * and exceeds the minimum threshold.
 *
 * @param map           Response map
 * @param searchXMin/Max/YMin/Max  Search region bounds
 * @param minResponse   Minimum response threshold
 * @return Vector of candidate positions with response values
 */
inline std::vector<ResponseCandidate> ExtractCandidatesNMS3x3(
    const ResponseMap& map,
    int32_t searchXMin, int32_t searchXMax,
    int32_t searchYMin, int32_t searchYMax,
    int16_t minResponse)
{
    std::vector<ResponseCandidate> candidates;

    // NMS needs 1-pixel border
    int32_t yStart = std::max(searchYMin, 1);
    int32_t yEnd   = std::min(searchYMax, map.height - 2);
    int32_t xStart = std::max(searchXMin, 1);
    int32_t xEnd   = std::min(searchXMax, map.width - 2);

    for (int32_t y = yStart; y <= yEnd; ++y) {
        const int16_t* rowPrev = map.data.data() + static_cast<size_t>(y - 1) * map.stride;
        const int16_t* rowCurr = map.data.data() + static_cast<size_t>(y)     * map.stride;
        const int16_t* rowNext = map.data.data() + static_cast<size_t>(y + 1) * map.stride;

        for (int32_t x = xStart; x <= xEnd; ++x) {
            int16_t val = rowCurr[x];
            if (val < minResponse) continue;

            // 8-neighbor check: center >= all neighbors
            if (val >= rowPrev[x - 1] && val >= rowPrev[x] && val >= rowPrev[x + 1] &&
                val >= rowCurr[x - 1] &&                      val >= rowCurr[x + 1] &&
                val >= rowNext[x - 1] && val >= rowNext[x] && val >= rowNext[x + 1]) {
                candidates.push_back({x, y, val});
            }
        }
    }

    return candidates;
}

/**
 * @brief Apply IoU-based NMS on candidates (decompiled cv::dnn::NMSBoxes equivalent)
 *
 * Decompiled sub_1800497F0 calls cv::dnn::NMSBoxes after 3x3 local max extraction.
 * All candidates within one (angle,scale) work item share the same bounding box size,
 * so IoU depends only on center distance.
 *
 * @param candidates    Input candidates (from ExtractCandidatesNMS3x3)
 * @param boxWidth      Template bounding box width at current level/scale
 * @param boxHeight     Template bounding box height at current level/scale
 * @param nmsThreshold  IoU threshold for suppression (0.5 = standard)
 * @return Filtered candidates
 */
inline std::vector<ResponseCandidate> IoUNMSCandidates(
    std::vector<ResponseCandidate>& candidates,
    int32_t boxWidth, int32_t boxHeight,
    float nmsThreshold = 0.5f)
{
    if (candidates.size() <= 1 || boxWidth <= 0 || boxHeight <= 0) {
        return candidates;
    }

    // Sort by response descending (greedy NMS: keep highest score first)
    std::sort(candidates.begin(), candidates.end(),
        [](const ResponseCandidate& a, const ResponseCandidate& b) {
            return a.response > b.response;
        });

    const float bw = static_cast<float>(boxWidth);
    const float bh = static_cast<float>(boxHeight);
    const float boxArea = bw * bh;

    std::vector<ResponseCandidate> kept;
    kept.reserve(candidates.size());

    for (const auto& cand : candidates) {
        bool suppressed = false;
        for (const auto& k : kept) {
            // AABB IoU for same-size boxes centered at (x, y)
            float dx = static_cast<float>(std::abs(cand.x - k.x));
            float dy = static_cast<float>(std::abs(cand.y - k.y));
            if (dx >= bw || dy >= bh) continue;  // no overlap

            float interW = bw - dx;
            float interH = bh - dy;
            float inter = interW * interH;
            float unionArea = 2.0f * boxArea - inter;
            float iou = inter / unionArea;

            if (iou > nmsThreshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(cand);
        }
    }

    return kept;
}

// =============================================================================
// Float-direction model point for scaled response map (decompiled sub_1800B72F0)
// =============================================================================

struct ResponsePointFloat {
    int32_t offsetX, offsetY;  ///< Rotated+scaled integer offset from search position
    float cosDir, sinDir;      ///< Float gradient direction (not quantized to bins)
};

// =============================================================================
// Float dot-product response map (decompiled sub_180039480 scaled path)
// =============================================================================

/**
 * @brief Accumulate model contributions using float dot-product scoring
 *
 * Decompiled scaled worker (sub_18005B460) uses sub_1800B72F0 to generate
 * float cosDir/sinDir vectors, then scores via:
 *   dot = cosDir * (Gx/|G|) + sinDir * (Gy/|G|) → cos(angle_diff) in [-1, 1]
 *   contrib = round(dot * 127) → [-127, 127] (matches LUT output range)
 *
 * This provides continuous direction scoring without 16-bin quantization loss,
 * critical for scaled matching where rotated+scaled model directions don't
 * align to bin boundaries.
 *
 * @param map         Output response map (must be pre-allocated and cleared)
 * @param modelPoints Rotated+scaled model points with float cosDir/sinDir
 * @param gxData      Gradient X data (raw Sobel float values)
 * @param gyData      Gradient Y data (raw Sobel float values)
 * @param gradWidth   Gradient image width
 * @param gradHeight  Gradient image height
 * @param gradStride  Gradient image stride (in float elements)
 * @param searchXMin/Max/YMin/Max  Search region bounds (inclusive)
 * @param ignorePolarity  If true, use |dot| (ignore 180-degree ambiguity)
 */
inline void BuildResponseMapFloat(
    ResponseMap& map,
    const std::vector<ResponsePointFloat>& modelPoints,
    const float* gxData, const float* gyData,
    int32_t gradWidth, int32_t gradHeight, int32_t gradStride,
    int32_t searchXMin, int32_t searchXMax,
    int32_t searchYMin, int32_t searchYMax,
    bool ignorePolarity = false)
{
    const int32_t searchW = searchXMax - searchXMin + 1;
    if (searchW <= 0) return;

    // Heap fallback for ultra-wide images
    std::vector<int8_t> heapBuf;
    if (searchW > MAX_RESPONSE_MAP_WIDTH) {
        heapBuf.resize(searchW);
    }

    for (const auto& pt : modelPoints) {
        for (int32_t y = searchYMin; y <= searchYMax; ++y) {
            int32_t imgY = y + pt.offsetY;
            if (imgY < 0 || imgY >= gradHeight) continue;

            const float* gxRow = gxData + static_cast<size_t>(imgY) * gradStride;
            const float* gyRow = gyData + static_cast<size_t>(imgY) * gradStride;
            int16_t* mapRow = map.data.data() + static_cast<size_t>(y) * map.stride;

            // Phase 1: Pre-compute int8 contribution buffer for this row
            int8_t stackBuf[MAX_RESPONSE_MAP_WIDTH];
            int8_t* contribBuf = (searchW <= MAX_RESPONSE_MAP_WIDTH) ? stackBuf : heapBuf.data();

            for (int32_t x = searchXMin; x <= searchXMax; ++x) {
                int32_t imgX = x + pt.offsetX;
                int32_t xi = x - searchXMin;
                if (imgX < 0 || imgX >= gradWidth) {
                    contribBuf[xi] = 0;
                    continue;
                }

                float gx = gxRow[imgX];
                float gy = gyRow[imgX];
                float mag = std::sqrt(gx * gx + gy * gy);
                if (mag < 1e-6f) {
                    contribBuf[xi] = 0;
                    continue;
                }

                // Normalize gradient direction: nx, ny are unit vectors
                float invMag = 1.0f / mag;
                float nx = gx * invMag;
                float ny = gy * invMag;

                // dot = cos(angle_diff) in [-1, 1]
                float dot = pt.cosDir * nx + pt.sinDir * ny;
                if (ignorePolarity) dot = std::fabs(dot);

                // Scale to [-127, 127] to match LUT output range
                int32_t contrib = static_cast<int32_t>(std::round(dot * 127.0f));
                contribBuf[xi] = static_cast<int8_t>(std::clamp(contrib, -127, 127));
            }

            // Phase 2: SIMD accumulation — vpmovsxbw + vpaddsw (same as LUT path)
            int16_t* mapPtr = mapRow + searchXMin;
            int32_t xi = 0;

#if HAVE_AVX2
            for (; xi + 16 <= searchW; xi += 16) {
                __m128i vContrib8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(contribBuf + xi));
                __m256i vContrib16 = _mm256_cvtepi8_epi16(vContrib8);
                __m256i vMap = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mapPtr + xi));
                vMap = _mm256_adds_epi16(vMap, vContrib16);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(mapPtr + xi), vMap);
            }
#endif
            // Scalar tail
            for (; xi < searchW; ++xi) {
                int32_t sum = static_cast<int32_t>(mapPtr[xi]) + contribBuf[xi];
                mapPtr[xi] = static_cast<int16_t>(std::clamp(sum, -32768, 32767));
            }
        }
    }
}

} // namespace Qi::Vision::Matching::Internal
