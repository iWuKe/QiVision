/**
 * @file ShapeModelCreate.cpp
 * @brief Model creation functions for ShapeModel
 *
 * Contains:
 * - CreateModel (QRegion unified version)
 * - FinalizeModel (shared post-processing)
 * - OptimizeModel
 * - BuildCosLUT
 * - BuildSearchAngleCache
 * - BuildScaledModels
 * - ComputeModelBounds / ComputeMinCoverage
 * - ComputeRotatedBounds
 * - LevelModel methods
 *
 * Uses EdgesSubPixGray for sub-pixel edge extraction (replaces AnglePyramid-based flow).
 */

#include "ShapeModelImpl.h"
#include <QiVision/Internal/Pyramid.h>
#include <QiVision/Internal/EdgesSubPix.h>
#include <QiVision/Internal/GeomConstruct.h>
#include <QiVision/Core/Exception.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <set>

namespace Qi::Vision::Matching {

// =============================================================================
// Anonymous namespace helpers
// =============================================================================

namespace {

/**
 * @brief Simple bilinear scaling for QImage with correct stride handling
 *
 * This function correctly handles QImage's row-padded memory layout,
 * unlike the generic ScaleImage function which has stride bugs.
 *
 * @param src Source grayscale image
 * @param scale Scale factor (< 1 for downsampling)
 * @return Scaled QImage
 */
QImage ScaleImageBilinear(const QImage& src, double scale) {
    if (src.Empty() || scale <= 0) return QImage();

    int32_t srcW = src.Width();
    int32_t srcH = src.Height();
    int32_t dstW = static_cast<int32_t>(std::round(srcW * scale));
    int32_t dstH = static_cast<int32_t>(std::round(srcH * scale));

    if (dstW <= 0 || dstH <= 0) return QImage();

    QImage dst(dstW, dstH, src.Type(), src.GetChannelType());

    // Get strides (bytes per row)
    int32_t srcStride = src.Stride();
    int32_t dstStride = dst.Stride();

    // Process based on pixel type
    if (src.Type() == PixelType::UInt8) {
        const uint8_t* srcData = static_cast<const uint8_t*>(src.Data());
        uint8_t* dstData = static_cast<uint8_t*>(dst.Data());

        for (int32_t dy = 0; dy < dstH; ++dy) {
            double sy = (dy + 0.5) / scale - 0.5;
            int32_t y0 = static_cast<int32_t>(std::floor(sy));
            int32_t y1 = y0 + 1;
            double fy = sy - y0;

            // Clamp to valid range
            y0 = std::max(0, std::min(y0, srcH - 1));
            y1 = std::max(0, std::min(y1, srcH - 1));

            const uint8_t* row0 = srcData + y0 * srcStride;
            const uint8_t* row1 = srcData + y1 * srcStride;
            uint8_t* dstRow = dstData + dy * dstStride;

            for (int32_t dx = 0; dx < dstW; ++dx) {
                double sx = (dx + 0.5) / scale - 0.5;
                int32_t x0 = static_cast<int32_t>(std::floor(sx));
                int32_t x1 = x0 + 1;
                double fx = sx - x0;

                // Clamp to valid range
                x0 = std::max(0, std::min(x0, srcW - 1));
                x1 = std::max(0, std::min(x1, srcW - 1));

                // Bilinear interpolation
                double v00 = row0[x0];
                double v10 = row0[x1];
                double v01 = row1[x0];
                double v11 = row1[x1];

                double v = v00 * (1 - fx) * (1 - fy) +
                          v10 * fx * (1 - fy) +
                          v01 * (1 - fx) * fy +
                          v11 * fx * fy;

                dstRow[dx] = static_cast<uint8_t>(std::clamp(std::round(v), 0.0, 255.0));
            }
        }
    } else if (src.Type() == PixelType::Float32) {
        const float* srcData = static_cast<const float*>(src.Data());
        float* dstData = static_cast<float*>(dst.Data());
        int32_t srcStrideF = srcStride / sizeof(float);
        int32_t dstStrideF = dstStride / sizeof(float);

        for (int32_t dy = 0; dy < dstH; ++dy) {
            double sy = (dy + 0.5) / scale - 0.5;
            int32_t y0 = static_cast<int32_t>(std::floor(sy));
            int32_t y1 = y0 + 1;
            double fy = sy - y0;

            y0 = std::max(0, std::min(y0, srcH - 1));
            y1 = std::max(0, std::min(y1, srcH - 1));

            const float* row0 = srcData + y0 * srcStrideF;
            const float* row1 = srcData + y1 * srcStrideF;
            float* dstRow = dstData + dy * dstStrideF;

            for (int32_t dx = 0; dx < dstW; ++dx) {
                double sx = (dx + 0.5) / scale - 0.5;
                int32_t x0 = static_cast<int32_t>(std::floor(sx));
                int32_t x1 = x0 + 1;
                double fx = sx - x0;

                x0 = std::max(0, std::min(x0, srcW - 1));
                x1 = std::max(0, std::min(x1, srcW - 1));

                double v00 = row0[x0];
                double v10 = row0[x1];
                double v01 = row1[x0];
                double v11 = row1[x1];

                double v = v00 * (1 - fx) * (1 - fy) +
                          v10 * fx * (1 - fy) +
                          v01 * (1 - fx) * fy +
                          v11 * fx * fy;

                dstRow[dx] = static_cast<float>(v);
            }
        }
    }

    return dst;
}

/**
 * @brief Convert QImage to contiguous float buffer (row-major, no padding)
 *
 * EdgesSubPixGray requires a contiguous float buffer without stride padding.
 *
 * @param img Source image (UInt8 or Float32)
 * @param[out] floatBuf Output buffer (resized to width*height)
 * @return true if conversion succeeded
 */
bool ImageToFloatBuffer(const QImage& img, std::vector<float>& floatBuf) {
    if (img.Empty()) return false;

    int32_t w = img.Width();
    int32_t h = img.Height();
    floatBuf.resize(static_cast<size_t>(w) * h);

    if (img.Type() == PixelType::Float32) {
        int32_t stride = img.Stride() / static_cast<int32_t>(sizeof(float));
        const float* src = static_cast<const float*>(img.Data());
        if (stride == w) {
            std::memcpy(floatBuf.data(), src, sizeof(float) * w * h);
        } else {
            for (int32_t y = 0; y < h; ++y) {
                std::memcpy(floatBuf.data() + y * w, src + y * stride, sizeof(float) * w);
            }
        }
    } else if (img.Type() == PixelType::UInt8) {
        int32_t stride = img.Stride();
        const uint8_t* src = static_cast<const uint8_t*>(img.Data());
        for (int32_t y = 0; y < h; ++y) {
            const uint8_t* row = src + y * stride;
            float* dst = floatBuf.data() + y * w;
            for (int32_t x = 0; x < w; ++x) {
                dst[x] = static_cast<float>(row[x]);
            }
        }
    } else {
        return false;
    }

    return true;
}

/**
 * @brief Downsample uint8 mask by 2x using 5x5 [1,4,6,4,1]/256 kernel.
 * Equivalent to cv::pyrDown for mask images. BORDER_REFLECT_101.
 */
void DownsampleMaskPyrDown(const std::vector<uint8_t>& src,
                           int32_t srcW, int32_t srcH,
                           std::vector<uint8_t>& dst,
                           int32_t dstW, int32_t dstH) {
    // reflect101: gfedcb|abcdefgh|gfedcba  (d = dimension)
    auto reflect101 = [](int32_t i, int32_t d) -> int32_t {
        if (i < 0) i = -i;
        if (i >= d) i = 2 * (d - 1) - i;
        return i;
    };
    auto clampW = [&](int32_t x) { return reflect101(x, srcW); };
    auto clampH = [&](int32_t y) { return reflect101(y, srcH); };

    // Horizontal pass: src (srcW x srcH) → tmp (srcW x srcH)
    std::vector<int32_t> tmp(static_cast<size_t>(srcW) * srcH);
    for (int32_t y = 0; y < srcH; ++y) {
        for (int32_t x = 0; x < srcW; ++x) {
            int32_t v = 1 * src[y * srcW + clampW(x - 2)]
                      + 4 * src[y * srcW + clampW(x - 1)]
                      + 6 * src[y * srcW + x]
                      + 4 * src[y * srcW + clampW(x + 1)]
                      + 1 * src[y * srcW + clampW(x + 2)];
            tmp[y * srcW + x] = v;
        }
    }
    // Vertical pass + subsample: tmp → dst (dstW x dstH)
    dst.resize(static_cast<size_t>(dstW) * dstH);
    for (int32_t dy = 0; dy < dstH; ++dy) {
        int32_t sy = dy * 2;
        for (int32_t dx = 0; dx < dstW; ++dx) {
            int32_t sx = dx * 2;
            int32_t v = 1 * tmp[clampH(sy - 2) * srcW + sx]
                      + 4 * tmp[clampH(sy - 1) * srcW + sx]
                      + 6 * tmp[sy * srcW + sx]
                      + 4 * tmp[clampH(sy + 1) * srcW + sx]
                      + 1 * tmp[clampH(sy + 2) * srcW + sx];
            // v / 256: exact cv::pyrDown normalization for uint8
            dst[dy * dstW + dx] = static_cast<uint8_t>((v + 128) / 256);
        }
    }
}

/**
 * @brief Convert EdgePoints to ModelPoints centered at image center.
 * Re-quantizes angle bins to per-level count. Returns maxRadiusSq and pts2d for alignment.
 */
void BuildLevelModelPoints(const std::vector<Qi::Vision::Internal::EdgePoint>& edges,
                           int32_t imgW, int32_t imgH,
                           int32_t levelAngleBins,
                           std::vector<ModelPoint>& outPoints,
                           std::vector<Point2d>& outPts2d,
                           double& outMaxRadiusSq) {
    double centerX = imgW * 0.5;
    double centerY = imgH * 0.5;
    outPoints.clear();
    outPoints.reserve(edges.size());
    outPts2d.clear();
    outPts2d.reserve(edges.size());
    outMaxRadiusSq = 0.0;

    for (const auto& ep : edges) {
        double relX = ep.x - centerX;
        double relY = ep.y - centerY;

        outPts2d.emplace_back(relX, relY);
        outMaxRadiusSq = std::max(outMaxRadiusSq, relX * relX + relY * relY);

        // Re-quantize angle to per-level angleBins
        int32_t modelBin = static_cast<int32_t>(ep.angle / (2.0 * PI) * levelAngleBins);
        modelBin = modelBin % levelAngleBins;
        if (modelBin < 0) modelBin += levelAngleBins;

        outPoints.emplace_back(relX, relY, ep.angle, ep.magnitude, modelBin, 1.0);
    }
}

/**
 * @brief Generate unique integer-coordinate grid points from subpixel model points.
 * Sorted by (Y, X) for cache-friendly access during coarse search.
 */
std::vector<ModelPoint> GenerateGridPoints(const std::vector<ModelPoint>& points) {
    std::set<std::pair<int32_t, int32_t>> uniqueCoords;
    std::vector<ModelPoint> gridPts;
    gridPts.reserve(points.size());

    for (const auto& pt : points) {
        int32_t gx = static_cast<int32_t>(std::round(pt.x));
        int32_t gy = static_cast<int32_t>(std::round(pt.y));

        auto key = std::make_pair(gx, gy);
        if (uniqueCoords.find(key) == uniqueCoords.end()) {
            uniqueCoords.insert(key);
            gridPts.emplace_back(static_cast<double>(gx), static_cast<double>(gy),
                                pt.angle, pt.magnitude, pt.angleBin, pt.weight);
        }
    }

    std::sort(gridPts.begin(), gridPts.end(),
        [](const ModelPoint& a, const ModelPoint& b) {
            if (static_cast<int32_t>(a.y) != static_cast<int32_t>(b.y))
                return a.y < b.y;
            return a.x < b.x;
        });

    return gridPts;
}

/**
 * @brief Compute alignment angle and rotated AABB from MinAreaRect of model points.
 *
 * Decompiled from sub_18004E770.
 */
void ComputeAlignmentAndBBox(const std::vector<Point2d>& pts2d,
                              Internal::LevelCreateData& lcd) {
    auto optRect = Qi::Vision::Internal::MinAreaRect(pts2d);
    if (!optRect) return;

    Point2d corners[4];
    optRect->GetCorners(corners);

    // Get two edges
    double edge1x = corners[1].x - corners[0].x;
    double edge1y = corners[1].y - corners[0].y;
    double edge2x = corners[2].x - corners[1].x;
    double edge2y = corners[2].y - corners[1].y;

    double len1sq = edge1x * edge1x + edge1y * edge1y;
    double len2sq = edge2x * edge2x + edge2y * edge2y;

    // Choose shorter edge for alignment
    double alignDeg;
    if (len1sq <= len2sq) {
        alignDeg = std::atan2(edge1y, edge1x) * 180.0 / PI;
    } else {
        alignDeg = std::atan2(edge2y, edge2x) * 180.0 / PI;
    }

    // Normalize to [-90, 90)
    while (alignDeg < -90.0) alignDeg += 180.0;
    while (alignDeg >= 90.0) alignDeg -= 180.0;

    // Also check +90 variant, pick smaller |abs|
    double altDeg = alignDeg + 90.0;
    while (altDeg < -90.0) altDeg += 180.0;
    while (altDeg >= 90.0) altDeg -= 180.0;

    if (std::abs(altDeg) < std::abs(alignDeg)) {
        alignDeg = altDeg;
    }

    // Convert to radians (negate per decompiled convention)
    double alignRad = alignDeg * (-PI) / 180.0;

    lcd.alignmentAngle = alignRad;

    // Rotate corners to compute AABB
    double cosA = std::cos(alignRad);
    double sinA = std::sin(alignRad);

    double rMinX = 1e30, rMaxX = -1e30, rMinY = 1e30, rMaxY = -1e30;
    for (int k = 0; k < 4; ++k) {
        double rx = cosA * corners[k].x - sinA * corners[k].y;
        double ry = sinA * corners[k].x + cosA * corners[k].y;
        rMinX = std::min(rMinX, rx);
        rMaxX = std::max(rMaxX, rx);
        rMinY = std::min(rMinY, ry);
        rMaxY = std::max(rMaxY, ry);
    }

    lcd.bboxMinX = rMinX;
    lcd.bboxMaxX = rMaxX;
    lcd.bboxMinY = rMinY;
    lcd.bboxMaxY = rMaxY;
}

} // anonymous namespace

namespace Internal {

// =============================================================================
// LevelModel Implementation
// =============================================================================

void LevelModel::BuildSoA() {
    // Regenerate gridPoints from points if empty (e.g., after loading from file)
    if (gridPoints.empty() && !points.empty()) {
        RegenerateGridPoints();
    }

    // Build SoA for Block 1 (subpixel points)
    BuildSoAForPoints(points, soaX, soaY, soaCosAngle, soaSinAngle, soaWeight, soaAngleBin);

    // Build SoA for Block 2 (grid points)
    BuildSoAForPoints(gridPoints, gridSoaX, gridSoaY, gridSoaCosAngle, gridSoaSinAngle, gridSoaWeight, gridSoaAngleBin);

    // Pre-compute fixed 16-bin angleBin for response map LUT
    // (independent of per-level numAngleBins which halves at each pyramid level)
    const size_t n = gridPoints.size();
    const size_t paddedN = (n + 7) & ~7;
    gridSoaAngleBin16.resize(paddedN, 0);
    for (size_t i = 0; i < n; ++i) {
        double angle = std::atan2(
            static_cast<double>(gridSoaSinAngle[i]),
            static_cast<double>(gridSoaCosAngle[i]));
        if (angle < 0) angle += TWO_PI;
        int bin = static_cast<int>(angle * 16.0 / TWO_PI);
        gridSoaAngleBin16[i] = static_cast<int16_t>(std::clamp(bin, 0, 15));
    }
}

void LevelModel::RegenerateGridPoints() {
    std::set<std::pair<int32_t, int32_t>> uniqueGridCoords;
    gridPoints.clear();
    gridPoints.reserve(points.size());

    for (const auto& pt : points) {
        int32_t gx = static_cast<int32_t>(std::round(pt.x));
        int32_t gy = static_cast<int32_t>(std::round(pt.y));

        auto key = std::make_pair(gx, gy);
        if (uniqueGridCoords.find(key) == uniqueGridCoords.end()) {
            uniqueGridCoords.insert(key);
            gridPoints.emplace_back(static_cast<double>(gx), static_cast<double>(gy),
                                   pt.angle, pt.magnitude, pt.angleBin, pt.weight);
        }
    }

    // Sort by Y then X
    std::sort(gridPoints.begin(), gridPoints.end(),
        [](const ModelPoint& a, const ModelPoint& b) {
            if (static_cast<int32_t>(a.y) != static_cast<int32_t>(b.y))
                return a.y < b.y;
            return a.x < b.x;
        });
}

void LevelModel::BuildSoAForPoints(const std::vector<ModelPoint>& pts,
                                    std::vector<float>& x, std::vector<float>& y,
                                    std::vector<float>& cosA, std::vector<float>& sinA,
                                    std::vector<float>& w, std::vector<int16_t>& bins) {
    const size_t n = pts.size();
    const size_t paddedN = (n + 7) & ~7;  // Pad to multiple of 8 for AVX2

    x.resize(paddedN, 0.0f);
    y.resize(paddedN, 0.0f);
    cosA.resize(paddedN, 1.0f);
    sinA.resize(paddedN, 0.0f);
    w.resize(paddedN, 0.0f);
    bins.resize(paddedN, 0);

    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(pts[i].x);
        y[i] = static_cast<float>(pts[i].y);
        cosA[i] = static_cast<float>(pts[i].cosAngle);
        sinA[i] = static_cast<float>(pts[i].sinAngle);
        w[i] = static_cast<float>(pts[i].weight);
        bins[i] = static_cast<int16_t>(pts[i].angleBin);
    }
}

// =============================================================================
// Helpers for scaled model cache
// =============================================================================

static void ComputeBoundsForLevels(const std::vector<LevelModel>& levels,
                                   double& minX, double& maxX,
                                   double& minY, double& maxY) {
    if (levels.empty() || levels[0].points.empty()) {
        minX = maxX = minY = maxY = 0.0;
        return;
    }

    minX = minY = std::numeric_limits<double>::max();
    maxX = maxY = std::numeric_limits<double>::lowest();

    for (const auto& pt : levels[0].points) {
        minX = std::min(minX, pt.x);
        maxX = std::max(maxX, pt.x);
        minY = std::min(minY, pt.y);
        maxY = std::max(maxY, pt.y);
    }
}

static double ComputeMinCoverageForLevels(const std::vector<LevelModel>& /*levels*/) {
    // Halcon-aligned: no coverage gate at coarse search.
    // The response map + LUT scoring and float dot-product scoring already use
    // fixed denominator (numPoints), so low-coverage matches naturally get low scores.
    // Coverage filtering is disabled to avoid extra heuristic suppression.
    return 0.0;
}

static void BuildSearchAngleCacheForLevels(const std::vector<LevelModel>& levels,
                                           const Size2i& templateSize,
                                           double angleStart, double angleExtent, double angleStep,
                                           std::vector<SearchAngleData>& outCache,
                                           double& outStep) {
    outCache.clear();

    // Auto-compute angle step if not specified (Halcon: AngleStep = atan(1/R_max))
    if (angleStep <= 0) {
        int32_t modelSize = std::max(templateSize.width, templateSize.height);
        angleStep = EstimateAngleStep(modelSize);
    }
    outStep = angleStep;

    int32_t numAngles = static_cast<int32_t>(std::ceil(angleExtent / angleStep)) + 1;
    outCache.resize(numAngles);

    const size_t numLevels = levels.size();

    for (int32_t i = 0; i < numAngles; ++i) {
        SearchAngleData& data = outCache[i];
        data.angle = angleStart + i * angleStep;
        data.cosA = static_cast<float>(std::cos(data.angle));
        data.sinA = static_cast<float>(std::sin(data.angle));

        data.levelBounds.resize(numLevels);
        for (size_t level = 0; level < numLevels; ++level) {
            const auto& levelModel = levels[level];
            if (levelModel.points.empty()) {
                data.levelBounds[level] = {0, 0, 0, 0};
                continue;
            }

            double minX = std::numeric_limits<double>::max();
            double maxX = std::numeric_limits<double>::lowest();
            double minY = std::numeric_limits<double>::max();
            double maxY = std::numeric_limits<double>::lowest();

            const double cosA = data.cosA;
            const double sinA = data.sinA;

            for (const auto& pt : levelModel.points) {
                double rx = cosA * pt.x - sinA * pt.y;
                double ry = sinA * pt.x + cosA * pt.y;
                minX = std::min(minX, rx);
                maxX = std::max(maxX, rx);
                minY = std::min(minY, ry);
                maxY = std::max(maxY, ry);
            }

            data.levelBounds[level].minX = static_cast<int32_t>(std::floor(minX));
            data.levelBounds[level].maxX = static_cast<int32_t>(std::ceil(maxX));
            data.levelBounds[level].minY = static_cast<int32_t>(std::floor(minY));
            data.levelBounds[level].maxY = static_cast<int32_t>(std::ceil(maxY));
        }
    }
}

// =============================================================================
// Helper: Extract model levels from float image using EdgesSubPixGray
// =============================================================================

/**
 * @brief Core per-level edge extraction loop used by CreateModel and BuildScaledModels.
 *
 * Matches decompiled sub_18004E770:
 *   - Center = image center (cols*0.5, rows*0.5) at each pyramid level
 *   - angleBinsPerLevel[] with per-level halving (min=2)
 *   - contrastPerLevel[] with per-level halving (min=1)
 *   - Small-image stop only when auto-detecting levels (if (!v211))
 *   - Timeout support (a8 > 0 && elapsed > timeLimit)
 *
 * @param floatData Contiguous float image buffer (row-major)
 * @param imgWidth Image width
 * @param imgHeight Image height
 * @param numAngleBins Number of angle bins (from params, clamped [1,128])
 * @param contrastHigh High threshold for edge detection
 * @param contrastLow Low threshold (0 = auto = 0.5*high)
 * @param maxLevels Maximum number of pyramid levels to build (clamped [1,10] by caller)
 * @param[out] outLevels Output level models
 * @param[out] outLevelCreateData Output per-level creation metadata (optional, can be nullptr)
 * @param debugPrint If true, print debug info
 * @param region Optional region for edge filtering (nullptr = no filtering)
 * @param autoLevels If true, auto-stop when image becomes too small (decompiled: if (!v211))
 * @param timeoutMs Timeout in milliseconds (0 = no timeout; decompiled: a8 > 0 && elapsed > timeLimit)
 * @return Number of valid levels created
 */
static int32_t ExtractEdgeLevels(const std::vector<float>& floatData,
                                  int32_t imgWidth, int32_t imgHeight,
                                  int32_t numAngleBins,
                                  double contrastHigh, double contrastLow,
                                  int32_t maxLevels,
                                  std::vector<LevelModel>& outLevels,
                                  std::vector<LevelCreateData>* outLevelCreateData,
                                  bool debugPrint,
                                  const QRegion* region = nullptr,
                                  bool autoLevels = true,
                                  int32_t timeoutMs = 0) {
    using Qi::Vision::Internal::EdgesSubPixGray;
    using Qi::Vision::Internal::DownsampleBy2;
    using Qi::Vision::Internal::DownsampleMethod;

    outLevels.clear();
    if (outLevelCreateData) outLevelCreateData->clear();

    // Timeout tracking (decompiled: a8 > 0 && elapsed > timeLimit → return -2)
    auto startTime = std::chrono::high_resolution_clock::now();

    // Edge thresholds: constant across all levels (decompiled behavior)
    // Constant table: dword_1800D6AA8=1.0f (high), dword_1800D6B38=2.0f (low)
    // When user specifies contrast (origin >= 1.0): high=origin, low=origin
    // When auto (origin < 1.0): high=1.0, low=2.0 (decompiled §4 line 1549)
    double edgeHigh, edgeLow;
    if (contrastHigh >= 1.0) {
        edgeHigh = contrastHigh;
        edgeLow = (contrastLow > 0) ? contrastLow : contrastHigh;
    } else {
        edgeHigh = 1.0;
        edgeLow = 2.0;  // dword_1800D6B38 = 2.0f
    }

    // Per-level arrays (matching decompiled sub_18004E770):
    // angleBinsPerLevel: [0]=numAngleBins, [i]=max(prev/2, 2)
    // contrastLevelPerLevel: [0]=1, [i]=max(prev/2, 1) (integer division → always 1)
    std::vector<int32_t> angleBinsPerLevel(maxLevels);
    angleBinsPerLevel[0] = numAngleBins;
    for (int32_t i = 1; i < maxLevels; ++i) {
        angleBinsPerLevel[i] = std::max(2, angleBinsPerLevel[i - 1] / 2);
    }

    // Current image buffer (will be downsampled each level)
    std::vector<float> currentImage = floatData;
    int32_t currentW = imgWidth;
    int32_t currentH = imgHeight;

    // Mask buffer: convert QRegion to uint8_t mask, pyrDown per level
    std::vector<uint8_t> currentMask;
    if (region) {
        currentMask.resize(static_cast<size_t>(currentW) * currentH, 0);
        for (int32_t y = 0; y < currentH; ++y) {
            for (int32_t x = 0; x < currentW; ++x) {
                if (region->Contains(x, y)) {
                    currentMask[y * currentW + x] = 255;
                }
            }
        }
    }

    for (int32_t level = 0; level < maxLevels; ++level) {
        // Downsample if not level 0
        if (level > 0) {
            int32_t newW = currentW / 2;
            int32_t newH = currentH / 2;
            // Minimum size guard: EdgesSubPixGray requires width >= 3 && height >= 3
            // Only auto-levels has the aggressive <8/<12 stop (handled below);
            // non-auto just prevents illegal dimensions.
            if (newW < 3 || newH < 3) break;

            // Downsample image
            std::vector<float> downsampled(static_cast<size_t>(newW) * newH);
            DownsampleBy2(currentImage.data(), currentW, currentH,
                          downsampled.data(), 1.0, DownsampleMethod::Gaussian);
            currentImage = std::move(downsampled);

            // Downsample mask: cv::pyrDown equivalent (5x5 [1,4,6,4,1]/256, BORDER_REFLECT_101)
            if (!currentMask.empty()) {
                std::vector<uint8_t> downMask;
                DownsampleMaskPyrDown(currentMask, currentW, currentH, downMask, newW, newH);
                currentMask = std::move(downMask);
            }

            currentW = newW;
            currentH = newH;
        }

        // Auto-level check: only when auto-detecting levels (decompiled: if (!v211))
        if (autoLevels) {
            if (currentH < 8 || currentW < 8) break;
            if (currentH < 12 && currentW < 12) break;
        }

        // Level scale
        double levelScale = 1.0 / static_cast<double>(1 << level);

        // Angle bins for this level
        int32_t levelAngleBins = angleBinsPerLevel[level];

        // Run EdgesSubPixGray with constant thresholds and optional mask
        // Decompiled: a4=0 (no Gaussian pre-smoothing) — pyrDown already smooths
        // Decompiled: a7=levelAngleBins, a8=1.0 (contrastScale, no-op)
        const uint8_t* maskPtr = currentMask.empty() ? nullptr : currentMask.data();
        int32_t edgeStatus = 0;
        auto edges = EdgesSubPixGray(currentImage.data(), currentW, currentH,
                                     edgeHigh, edgeLow, 0.0,
                                     maskPtr, currentW,
                                     levelAngleBins, 1.0, &edgeStatus);

        if (debugPrint) {
            std::printf("[ExtractEdgeLevels] Level %d (%dx%d, scale=%.3f): %zu edges, "
                        "threshold=[%.1f, %.1f], angleBins=%d%s%s\n",
                        level, currentW, currentH, levelScale,
                        edges.size(), edgeLow, edgeHigh,
                        levelAngleBins, maskPtr ? " (masked)" : "",
                        edgeStatus != 0 ? " [EDGE ERROR]" : "");
            std::fflush(stdout);
        }

        // Distinguish "algorithm error" (edgeStatus!=0) from "normal stop" (too few edges)
        // Decompiled: non-zero rc → error path, model creation fails
        if (edgeStatus != 0) {
            if (debugPrint) {
                std::printf("[ExtractEdgeLevels] Level %d: EdgesSubPix returned error %d\n",
                            level, edgeStatus);
            }
            return -3;  // algorithm error (hard failure)
        }

        // Decompiled: numEdgePoints < 20 stops pyramid expansion for this model.
        if (edges.size() < 20) break;

        // Build model points (centered at image center, angle re-quantized)
        std::vector<ModelPoint> modelPoints;
        std::vector<Point2d> pts2d;
        double maxRadiusSq = 0.0;
        BuildLevelModelPoints(edges, currentW, currentH, levelAngleBins,
                              modelPoints, pts2d, maxRadiusSq);

        LevelModel levelModel;
        levelModel.width = currentW;
        levelModel.height = currentH;
        levelModel.scale = levelScale;
        levelModel.numAngleBins = levelAngleBins;
        levelModel.points = std::move(modelPoints);

        // Generate grid points (unique integer coordinates, sorted by Y,X)
        levelModel.gridPoints = GenerateGridPoints(levelModel.points);

        outLevels.push_back(std::move(levelModel));

        // Compute alignment and AABB from MinAreaRect
        if (outLevelCreateData) {
            LevelCreateData lcd;
            lcd.maxRadius = std::sqrt(maxRadiusSq);

            if (!pts2d.empty()) {
                ComputeAlignmentAndBBox(pts2d, lcd);
            }

            outLevelCreateData->push_back(lcd);
        }

        // Timeout check (decompiled: a8 > 0 && elapsed > timeLimit → return -2)
        if (timeoutMs > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - startTime).count();
            if (elapsed > timeoutMs) {
                if (debugPrint) {
                    std::printf("[ExtractEdgeLevels] Timeout after %lld ms (limit=%d)\n",
                                (long long)elapsed, timeoutMs);
                    std::fflush(stdout);
                }
                return -2;  // timeout error (decompiled: v64 = -2; goto cleanup)
            }
        }
    }

    // Decompiled: numStoredLevels==0 → return -1 (error)
    if (outLevels.empty()) return -1;
    return static_cast<int32_t>(outLevels.size());
}

// =============================================================================
// ShapeModelImpl::CreateModel (QRegion version — unified path)
// =============================================================================

bool ShapeModelImpl::CreateModel(const QImage& image, const QRegion& region, const Point2d& origin) {
    if (!origin.IsValid()) {
        throw InvalidArgumentException("CreateShapeModel: invalid origin");
    }

    // Validate and fix contrast parameters (HALCON-style hard checks)
    if (!params_.ValidateAndFixContrast()) {
        if (timingParams_.debugCreateModel) {
            std::printf("[CreateModel] Warning: Contrast parameters were invalid and auto-fixed.\n");
            std::printf("  contrastLow=%.1f, contrastHigh=%.1f, minContrast=%.1f, minComponentSize=%d\n",
                        params_.contrastLow, params_.contrastHigh,
                        params_.minContrast, params_.minComponentSize);
        }
    }

    // Reset timing
    createTiming_ = ShapeModelCreateTiming();
    auto tTotal = std::chrono::high_resolution_clock::now();
    auto tStep = tTotal;

    auto elapsedMs = [](auto start) {
        return std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count();
    };

    // Extract template: if region is empty, use full image; otherwise use region bbox
    QImage templateImg;
    const QRegion* regionPtr = nullptr;
    QRegion localRegion;

    if (region.Empty()) {
        // No region: use full image (equivalent to old Rect2i{} path)
        templateImg = image;
        templateSize_ = Size2i{image.Width(), image.Height()};
    } else {
        // Region: extract bounding box, translate region to local coordinates
        Rect2i bbox = region.BoundingBox();
        templateImg = image.SubImage(bbox.x, bbox.y, bbox.width, bbox.height);
        templateSize_ = Size2i{bbox.width, bbox.height};
        localRegion = region.Translate(-bbox.x, -bbox.y);
        regionPtr = &localRegion;
    }

    if (templateImg.Empty()) return false;

    // Convert to float buffer for EdgesSubPixGray
    std::vector<float> floatBuf;
    if (!ImageToFloatBuffer(templateImg, floatBuf)) return false;

    int32_t imgW = templateImg.Width();
    int32_t imgH = templateImg.Height();

    // Max levels: decompiled uses 10 as max, auto-detect via pyramid loop
    int32_t numLevels = (params_.numLevels <= 0) ? 10 : std::min(params_.numLevels, 10);

    if (timingParams_.enableTiming) {
        createTiming_.pyramidBuildMs = elapsedMs(tStep);
    }

    // Store origin directly (decompiled: no auto-centering, caller provides real origin)
    origin_ = origin;
    // numAngleBins: clamped [1, 128] (decompiled: if v19>0 { if v19>128 v19=128 } else v19=1)
    numAngleBins_ = std::clamp(params_.numAngleBins, 1, 128);

    // Extract model points using EdgesSubPixGray pyramid loop
    tStep = std::chrono::high_resolution_clock::now();
    double contrastLow = (params_.contrastLow > 0) ? params_.contrastLow : 0.0;
    bool autoLevels = (params_.numLevels <= 0);
    int32_t actualLevels = ExtractEdgeLevels(
        floatBuf, imgW, imgH, numAngleBins_,
        params_.contrastHigh, contrastLow, numLevels,
        levels_, &levelCreateData_, timingParams_.debugCreateModel,
        regionPtr,
        autoLevels,
        1000);      // default timeout 1000ms (decompiled: v74[18] = 1000)

    // Timeout error (decompiled: return -2 → model creation fails)
    if (actualLevels < 0) {
        levels_.clear();
        return false;
    }
    params_.numLevels = actualLevels;

    if (timingParams_.enableTiming) {
        createTiming_.extractPointsMs = elapsedMs(tStep);
    }

    tStep = std::chrono::high_resolution_clock::now();
    bool result = FinalizeModel();

    if (timingParams_.enableTiming) {
        createTiming_.optimizeMs = elapsedMs(tStep);
        createTiming_.totalMs = elapsedMs(tTotal);

        if (timingParams_.printTiming) {
            createTiming_.Print();
        }
    }

    return result;
}

// =============================================================================
// ShapeModelImpl::ComputeMinCoverage
// =============================================================================

void ShapeModelImpl::ComputeMinCoverage() {
    minCoverage_ = ComputeMinCoverageForLevels(levels_);
}

// =============================================================================
// ShapeModelImpl::FinalizeModel
// =============================================================================

bool ShapeModelImpl::FinalizeModel() {
    if (levels_.empty() || levels_[0].points.empty()) {
        return false;
    }

    // Apply point reduction + weight normalization
    OptimizeModel(levels_);

    // Compute model bounding box
    ComputeModelBounds();

    // Compute dynamic coverage threshold
    ComputeMinCoverage();

    // Build SoA data for SIMD optimization
    for (auto& level : levels_) {
        level.BuildSoA();
    }

    // Build cosine lookup table for direction-quantized scoring
    BuildCosLUT(numAngleBins_);

    // Build pregenerated search angle cache
    double angleExtent = (params_.angleExtent > 0) ? params_.angleExtent : 2.0 * PI;
    BuildSearchAngleCache(params_.angleStart, angleExtent, params_.angleStep);

    // Decompiled find_shape_model L633-646: populate search radius per level
    // Assign kAngleBinSizeTable values from coarsest level downward
    searchRadiusPerLevel_.resize(levels_.size(), 0);
    int32_t tableIdx = 0;
    for (int32_t lvl = static_cast<int32_t>(levels_.size()) - 1; lvl >= 0; --lvl) {
        if (levels_[lvl].numAngleBins > 1 && tableIdx < 21) {
            searchRadiusPerLevel_[lvl] = kAngleBinSizeTable[tableIdx++];
        }
    }

    valid_ = true;
    return true;
}

// =============================================================================
// ShapeModelImpl::OptimizeModel
// =============================================================================

void ShapeModelImpl::OptimizeModel(std::vector<LevelModel>& levels) {
    // Set all weights to 1.0 unconditionally (Halcon-aligned: fixed weight)
    for (auto& level : levels) {
        for (auto& pt : level.points) {
            pt.weight = 1.0;
        }
    }

    const int32_t numLevels = static_cast<int32_t>(levels.size());

    // =========================================================================
    // Decompiled alignment (sub_1800B72F0):
    //   step = 1 << (numLevels - level)
    //   Coarsest level → step=1: ALL points used for coarse search
    //   Finer levels → step increases: fewer points (applied at SEARCH time)
    //
    // QiVision approach: model stores ALL points at all levels.
    // Stride subsampling is NOT applied during creation (decompiled does it
    // in GetModelTransformScaled at search time). Instead, we only skip
    // spatial filtering at the coarsest level to preserve all points there.
    // =========================================================================
    if (params_.optimization == OptimizationMode::None) {
        return;
    }

    double minSpacing = 1.0;
    switch (params_.optimization) {
        case OptimizationMode::PointReductionLow:
            minSpacing = 2.0;
            break;
        case OptimizationMode::PointReductionMedium:
            minSpacing = 3.0;
            break;
        case OptimizationMode::PointReductionHigh:
            minSpacing = 4.0;
            break;
        case OptimizationMode::Auto:
        default:
            {
                int32_t templateDim = std::max(templateSize_.width, templateSize_.height);
                if (templateDim <= 100) {
                    minSpacing = 2.0;
                } else if (templateDim <= 300) {
                    minSpacing = 2.5;
                } else {
                    minSpacing = 3.0;
                }
            }
            break;
        case OptimizationMode::None:
            return;
    }

    if (minSpacing > 0.5) {
        for (int32_t levelIdx = 0; levelIdx < numLevels; ++levelIdx) {
            auto& level = levels[levelIdx];
            if (level.points.empty()) continue;

            // Skip spatial filtering at coarsest level — keep all points
            // (decompiled: top level always uses step=1, no additional reduction)
            if (levelIdx == numLevels - 1) continue;

            // Spatial distance filtering for finer levels
            double minDistSq = minSpacing * minSpacing;

            bool hasValidTopology = !level.contourStarts.empty() &&
                                    level.contourStarts.size() > 1;

            if (hasValidTopology) {
                std::vector<ModelPoint> filteredAll;
                std::vector<int32_t> newContourStarts;
                std::vector<bool> newContourClosed;

                filteredAll.reserve(level.points.size());
                newContourStarts.reserve(level.contourStarts.size());
                newContourClosed.reserve(level.contourClosed.size());

                size_t numContours = level.contourStarts.size() - 1;
                for (size_t c = 0; c < numContours; ++c) {
                    int32_t startIdx = level.contourStarts[c];
                    int32_t endIdx = level.contourStarts[c + 1];
                    if (endIdx <= startIdx) continue;

                    std::vector<ModelPoint> filtered;
                    filtered.reserve(endIdx - startIdx);

                    double accumulatedDist = 0.0;
                    filtered.push_back(level.points[startIdx]);

                    for (int32_t i = startIdx + 1; i < endIdx; ++i) {
                        double dx = level.points[i].x - level.points[i-1].x;
                        double dy = level.points[i].y - level.points[i-1].y;
                        accumulatedDist += std::sqrt(dx*dx + dy*dy);
                        if (accumulatedDist >= minSpacing) {
                            filtered.push_back(level.points[i]);
                            accumulatedDist = 0.0;
                        }
                    }

                    bool isClosed = (c < level.contourClosed.size()) ? level.contourClosed[c] : false;
                    if (filtered.size() >= 2 && !isClosed) {
                        const auto& lastPt = level.points[endIdx - 1];
                        if (filtered.back().x != lastPt.x || filtered.back().y != lastPt.y) {
                            filtered.push_back(lastPt);
                        }
                    }

                    if (filtered.size() >= 2) {
                        newContourStarts.push_back(static_cast<int32_t>(filteredAll.size()));
                        newContourClosed.push_back(isClosed);
                        for (auto& pt : filtered) {
                            filteredAll.push_back(std::move(pt));
                        }
                    }
                }

                newContourStarts.push_back(static_cast<int32_t>(filteredAll.size()));
                level.points = std::move(filteredAll);
                level.contourStarts = std::move(newContourStarts);
                level.contourClosed = std::move(newContourClosed);
            } else {
                std::vector<ModelPoint> filtered;
                filtered.reserve(level.points.size());

                std::sort(level.points.begin(), level.points.end(),
                    [](const ModelPoint& a, const ModelPoint& b) {
                        return a.magnitude > b.magnitude;
                    });

                for (const auto& pt : level.points) {
                    bool tooClose = false;
                    for (const auto& kept : filtered) {
                        double dx = pt.x - kept.x;
                        double dy = pt.y - kept.y;
                        if (dx * dx + dy * dy < minDistSq) {
                            tooClose = true;
                            break;
                        }
                    }
                    if (!tooClose) {
                        filtered.push_back(pt);
                    }
                }

                level.points = std::move(filtered);
                level.contourStarts.clear();
                level.contourClosed.clear();
            }
        }
    }
}

// =============================================================================
// ShapeModelImpl::BuildCosLUT
// =============================================================================

void ShapeModelImpl::BuildCosLUT(int32_t numBins) {
    numAngleBins_ = numBins;
    cosLUT_.resize(numBins);

    const double step = 2.0 * PI / numBins;
    for (int32_t i = 0; i < numBins; ++i) {
        cosLUT_[i] = static_cast<float>(std::fabs(std::cos(i * step)));
    }
}

// =============================================================================
// ShapeModelImpl::BuildSearchAngleCache (Halcon pregeneration strategy)
// =============================================================================

void ShapeModelImpl::BuildSearchAngleCache(double angleStart, double angleExtent, double angleStep) {
    searchAngleCache_.clear();

    // Store search parameters
    searchAngleStart_ = angleStart;
    searchAngleExtent_ = angleExtent;

    // Auto-compute angle step if not specified (Halcon: AngleStep = atan(1/R_max))
    if (angleStep <= 0) {
        int32_t modelSize = std::max(templateSize_.width, templateSize_.height);
        angleStep = EstimateAngleStep(modelSize);
    }
    searchAngleStep_ = angleStep;

    // Calculate number of angles
    int32_t numAngles = static_cast<int32_t>(std::ceil(angleExtent / angleStep)) + 1;
    searchAngleCache_.resize(numAngles);

    const size_t numLevels = levels_.size();

    // Precompute all angle data
    for (int32_t i = 0; i < numAngles; ++i) {
        SearchAngleData& data = searchAngleCache_[i];
        data.angle = angleStart + i * angleStep;
        data.cosA = static_cast<float>(std::cos(data.angle));
        data.sinA = static_cast<float>(std::sin(data.angle));

        // Precompute bounds for each pyramid level
        data.levelBounds.resize(numLevels);

        for (size_t level = 0; level < numLevels; ++level) {
            const auto& levelModel = levels_[level];
            if (levelModel.points.empty()) {
                data.levelBounds[level] = {0, 0, 0, 0};
                continue;
            }

            // Compute rotated bounds for this level
            double minX = std::numeric_limits<double>::max();
            double maxX = std::numeric_limits<double>::lowest();
            double minY = std::numeric_limits<double>::max();
            double maxY = std::numeric_limits<double>::lowest();

            const double cosA = data.cosA;
            const double sinA = data.sinA;

            for (const auto& pt : levelModel.points) {
                double rx = cosA * pt.x - sinA * pt.y;
                double ry = sinA * pt.x + cosA * pt.y;
                minX = std::min(minX, rx);
                maxX = std::max(maxX, rx);
                minY = std::min(minY, ry);
                maxY = std::max(maxY, ry);
            }

            // Store as integer bounds (floor/ceil for safety margin)
            data.levelBounds[level].minX = static_cast<int32_t>(std::floor(minX));
            data.levelBounds[level].maxX = static_cast<int32_t>(std::ceil(maxX));
            data.levelBounds[level].minY = static_cast<int32_t>(std::floor(minY));
            data.levelBounds[level].maxY = static_cast<int32_t>(std::ceil(maxY));
        }
    }
}

// =============================================================================
// ShapeModelImpl::ComputeModelBounds
// =============================================================================

void ShapeModelImpl::ComputeModelBounds() {
    if (!levelCreateData_.empty()) {
        modelMinX_ = levelCreateData_[0].bboxMinX;
        modelMaxX_ = levelCreateData_[0].bboxMaxX;
        modelMinY_ = levelCreateData_[0].bboxMinY;
        modelMaxY_ = levelCreateData_[0].bboxMaxY;
    } else if (!levels_.empty() && !levels_[0].points.empty()) {
        // Fallback: compute from points
        modelMinX_ = modelMinY_ = std::numeric_limits<double>::max();
        modelMaxX_ = modelMaxY_ = std::numeric_limits<double>::lowest();

        for (const auto& pt : levels_[0].points) {
            modelMinX_ = std::min(modelMinX_, pt.x);
            modelMaxX_ = std::max(modelMaxX_, pt.x);
            modelMinY_ = std::min(modelMinY_, pt.y);
            modelMaxY_ = std::max(modelMaxY_, pt.y);
        }
    } else {
        modelMinX_ = modelMaxX_ = modelMinY_ = modelMaxY_ = 0;
    }
}

// =============================================================================
// ShapeModelImpl::ComputeRotatedBounds
// =============================================================================

void ShapeModelImpl::ComputeRotatedBounds(const std::vector<ModelPoint>& points, double angle,
                                          double& minX, double& maxX, double& minY, double& maxY) {
    if (points.empty()) {
        minX = maxX = minY = maxY = 0;
        return;
    }

    double cosA = std::cos(angle);
    double sinA = std::sin(angle);

    minX = minY = std::numeric_limits<double>::max();
    maxX = maxY = std::numeric_limits<double>::lowest();

    for (const auto& pt : points) {
        double rx = cosA * pt.x - sinA * pt.y;
        double ry = sinA * pt.x + cosA * pt.y;
        minX = std::min(minX, rx);
        maxX = std::max(maxX, rx);
        minY = std::min(minY, ry);
        maxY = std::max(maxY, ry);
    }
}

} // namespace Internal
} // namespace Qi::Vision::Matching
