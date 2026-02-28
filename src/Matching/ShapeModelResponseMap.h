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
inline void BuildResponseMap(
    ResponseMap& map,
    const std::vector<ResponsePoint>& modelPoints,
    const int16_t* angleBinImage, int32_t binWidth, int32_t binHeight, int32_t binStride,
    const ResponseMapLUT& lut,
    int32_t searchXMin, int32_t searchXMax,
    int32_t searchYMin, int32_t searchYMax,
    bool ignorePolarity = false)
{
    for (const auto& pt : modelPoints) {
        const int32_t mb = pt.angleBin16;

        for (int32_t y = searchYMin; y <= searchYMax; ++y) {
            int32_t imgY = y + pt.offsetY;
            if (imgY < 0 || imgY >= binHeight) continue;

            const int16_t* binRow = angleBinImage + static_cast<size_t>(imgY) * binStride;
            int16_t* mapRow = map.data.data() + static_cast<size_t>(y) * map.stride;

            for (int32_t x = searchXMin; x <= searchXMax; ++x) {
                int32_t imgX = x + pt.offsetX;
                if (imgX < 0 || imgX >= binWidth) continue;

                int16_t imageBin = binRow[imgX];
                if (imageBin <= 0 || imageBin > 16) continue;  // 0 = invalid, valid = 1..16

                // Decompiled: sub_180007BA0 (polarity) uses signed LUT value directly;
                //             sub_180007F60 (no polarity) uses abs of LUT value.
                int8_t rawContrib = lut.table[imageBin - 1][mb];  // Convert to 0-based for LUT
                int8_t contrib = ignorePolarity ?
                    static_cast<int8_t>(rawContrib < 0 ? -rawContrib : rawContrib) : rawContrib;
                // Saturating add to int16
                int32_t sum = static_cast<int32_t>(mapRow[x]) + contrib;
                mapRow[x] = static_cast<int16_t>(std::clamp(sum, -32768, 32767));
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

} // namespace Qi::Vision::Matching::Internal
