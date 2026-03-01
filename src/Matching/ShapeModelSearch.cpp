/**
 * @file ShapeModelSearch.cpp
 * @brief 4-stage search pipeline for ShapeModel
 *
 * Architecture:
 *   SearchPyramid()          — Main entry (~30 lines)
 *     ├── CoarseSearch()     — Stage 1: top-level grid search
 *     ├── PyramidRefine()    — Stage 2: per-level candidate refinement
 *     ├── SubPixelRefine()   — Stage 3: subpixel position/angle refinement
 *     └── FinalizeResults()  — Stage 4: final scoring + NMS
 *
 * Each stage is completely independent, no cross-calling.
 *
 * BUG fixes included:
 *   #1 — Angle step: min(11.25°, acos(1 - s²/(2R²)))
 *   #3 — Greediness: Halcon progressive threshold (in ScoreCore.h)
 *   #4 — Layer thresholds: [0.8, 0.9, 0.9, ...] not all 0.7
 *   #5 — Coverage penalty removed (direct similarity)
 *   #6 — Coordinate scaling: actual pyramid scale ratio, not hardcoded 2.0
 *   DIFF #2 — Angle range: acos formula per level, not fixed 6° halving
 *   DIFF #8 — Point set: coarse=gridPoints, level0=subpixel points
 */

#include "ShapeModelImpl.h"
#include "ShapeModelScoreCore.h"
#include "ShapeModelResponseMap.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace Qi::Vision::Matching {
namespace Internal {

// =============================================================================
// Halcon-style angle step computation (BUG #1 fix)
// =============================================================================

double ShapeModelImpl::ComputeHalconAngleStep(double maxRadius, double safety) {
    // Formula: min(11.25°, acos(1 - safety² / (2 * R²)))
    // safety = 1.5 for coarse search, 2.0 for refinement
    constexpr double MAX_STEP = 11.25 * PI / 180.0;  // 11.25°

    if (maxRadius < 1.0) return MAX_STEP;

    double arg = 1.0 - (safety * safety) / (2.0 * maxRadius * maxRadius);
    arg = std::max(-1.0, std::min(1.0, arg));  // Clamp for acos domain
    double step = std::acos(arg);

    return std::min(MAX_STEP, step);
}

// =============================================================================
// CollectCandidatesNMS — Spatial hash NMS + angle distance suppression
// Aligned with decompiled sub_18004A5A0
// =============================================================================

std::vector<MatchResult> ShapeModelImpl::CollectCandidatesNMS(
    std::vector<MatchResult> candidates,
    int32_t imageWidth,
    double angleStep)
{
    if (candidates.empty()) return {};

    // Phase 1: Build spatial hash + max score per position
    std::unordered_map<int64_t, std::vector<MatchResult>> spatialBuckets;
    std::unordered_map<int64_t, double> maxScoreMap;

    for (auto& c : candidates) {
        int64_t key = static_cast<int64_t>(static_cast<int32_t>(c.x))
                    + static_cast<int64_t>(static_cast<int32_t>(c.y)) * imageWidth;
        spatialBuckets[key].push_back(c);
        auto it = maxScoreMap.find(key);
        if (it == maxScoreMap.end() || c.score > it->second) {
            maxScoreMap[key] = c.score;
        }
    }

    // Phase 2: 3x3 neighborhood NMS
    // Neighbor offsets (center + 8 neighbors)
    const int64_t offsets[9] = {
        0,
        -imageWidth,                  // top
        1 - imageWidth,               // top-right
        1,                            // right
        static_cast<int64_t>(imageWidth) + 1, // bottom-right
        imageWidth,                   // bottom
        static_cast<int64_t>(imageWidth) - 1, // bottom-left
        -1,                           // left
        -(static_cast<int64_t>(imageWidth) + 1) // top-left
    };

    // Collect keys that survive NMS, paired with their max score
    std::vector<std::pair<int64_t, double>> nmsPassedKeys;
    for (const auto& [key, maxScore] : maxScoreMap) {
        bool isLocalMax = true;
        for (int d = 1; d <= 8; ++d) {
            auto it = maxScoreMap.find(key + offsets[d]);
            if (it != maxScoreMap.end() && it->second > maxScore) {
                isLocalMax = false;
                break;
            }
        }
        if (isLocalMax) {
            nmsPassedKeys.emplace_back(key, maxScore);
        }
    }

    // Sort NMS survivors by score descending
    std::sort(nmsPassedKeys.begin(), nmsPassedKeys.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Phase 3: Collect survivors + angle distance suppression
    const double angleThreshold = angleStep * 2.5;
    std::vector<MatchResult> output;

    for (const auto& [centerKey, _] : nmsPassedKeys) {
        // Collect candidates from 3x3 neighborhood
        std::vector<MatchResult> localCands;
        for (int d = 0; d <= 8; ++d) {
            int64_t neighborKey = centerKey + offsets[d];
            auto it = spatialBuckets.find(neighborKey);
            if (it != spatialBuckets.end()) {
                localCands.insert(localCands.end(), it->second.begin(), it->second.end());
                it->second.clear();  // Remove collected candidates
            }
        }

        if (localCands.empty()) continue;

        // Sort by score descending
        std::sort(localCands.begin(), localCands.end());

        // Angle distance suppression: remove lower-score candidates
        // whose angle is within angleStep * 2.5 of a higher-score candidate
        std::vector<MatchResult> filtered;
        for (size_t j = 0; j < localCands.size(); ++j) {
            bool suppressed = false;
            for (size_t k = 0; k < filtered.size(); ++k) {
                double diff = localCands[j].angle - filtered[k].angle;
                // Normalize to [-π, π)
                while (diff > PI) diff -= 2.0 * PI;
                while (diff < -PI) diff += 2.0 * PI;
                if (std::fabs(diff) < angleThreshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                filtered.push_back(localCands[j]);
            }
        }

        output.insert(output.end(), filtered.begin(), filtered.end());
    }

    // Final sort by score descending
    std::sort(output.begin(), output.end());

    return output;
}

// =============================================================================
// Stage 1: CoarseSearch — Response Map + LUT (primary path)
// =============================================================================

std::vector<MatchResult> ShapeModelImpl::CoarseSearch(
    const AnglePyramid& targetPyramid,
    int32_t startLevel,
    const SearchParams& params) const
{
    const auto& topLevel = levels_[startLevel];
    int32_t targetWidth = targetPyramid.GetWidth(startLevel);
    int32_t targetHeight = targetPyramid.GetHeight(startLevel);

    // Get angleBinImage — fall back to float scoring if unavailable
    const int16_t* angleBinData = nullptr;
    int32_t binW = 0, binH = 0, binStride = 0, numBins = 0;
    if (!targetPyramid.GetAngleBinData(startLevel,
            angleBinData, binW, binH, binStride, numBins)) {
        return CoarseSearchFloat(targetPyramid, startLevel, params);
    }

    // Step 2: Use levelCreateData_ maxRadius for angle step (decompiled behavior)
    double maxRadius = (startLevel < static_cast<int32_t>(levelCreateData_.size()))
        ? levelCreateData_[startLevel].maxRadius : 0.0;

    // BUG #1 fix: Halcon angle step formula with safety=1.5
    double coarseAngleStep = ComputeHalconAngleStep(maxRadius, 1.5);

    // Response threshold: score = response / (numPts * 127), so
    // minResponse = minScore * 0.8 * numPts * 127
    const int32_t numGridPts = static_cast<int32_t>(topLevel.gridPoints.size());
    if (numGridPts == 0) return {};
    const double rawMinResp = params.minScore * 0.8 * numGridPts * 127.0;
    const int16_t minResponse = static_cast<int16_t>(std::clamp(rawMinResp, 1.0, 32767.0));

    // Search ROI
    double levelScale = targetPyramid.GetScale(startLevel);
    Rect2i levelROI;
    bool hasSearchROI = (params.searchROI.width > 0 && params.searchROI.height > 0);
    if (hasSearchROI) {
        levelROI.x = static_cast<int32_t>(params.searchROI.x * levelScale);
        levelROI.y = static_cast<int32_t>(params.searchROI.y * levelScale);
        levelROI.width = static_cast<int32_t>(params.searchROI.width * levelScale);
        levelROI.height = static_cast<int32_t>(params.searchROI.height * levelScale);
    }

    // Build angle list
    std::vector<double> angles;
    const bool usePregenCache = !searchAngleCache_.empty() &&
                                 static_cast<size_t>(startLevel) < levels_.size() &&
                                 searchAngleStep_ > 0;
    if (usePregenCache) {
        const int32_t coarseStride = std::max(1, static_cast<int32_t>(
            coarseAngleStep / searchAngleStep_));
        for (size_t ai = 0; ai < searchAngleCache_.size(); ai += coarseStride) {
            double a = searchAngleCache_[ai].angle;
            if (a >= params.angleStart - 0.001 &&
                a <= params.angleStart + params.angleExtent + 0.001) {
                angles.push_back(a);
            }
        }
    } else {
        // Step 4: Uniform angle coverage (decompiled sub_1800B7FB0)
        double extent = params.angleExtent;
        int32_t numSteps = std::max(1, static_cast<int32_t>(std::round(extent / coarseAngleStep)));
        double adjustedStep = extent / numSteps;
        for (int32_t i = 0; i <= numSteps; ++i) {
            angles.push_back(params.angleStart + i * adjustedStep);
        }
    }

    std::vector<MatchResult> candidates;

    #pragma omp parallel
    {
        ResponseMap map;
        map.Allocate(targetWidth, targetHeight);
        std::vector<MatchResult> localCandidates;

        #pragma omp for schedule(dynamic)
        for (size_t ai = 0; ai < angles.size(); ++ai) {
            double angle = angles[ai];
            float cosR = static_cast<float>(std::cos(angle));
            float sinR = static_cast<float>(std::sin(angle));

            // 1. Rotate model points → ResponsePoint (integer offsets + rotated 16-bin)
            std::vector<ResponsePoint> rpts(numGridPts);
            for (int32_t i = 0; i < numGridPts; ++i) {
                float rx = cosR * topLevel.gridSoaX[i] - sinR * topLevel.gridSoaY[i];
                float ry = sinR * topLevel.gridSoaX[i] + cosR * topLevel.gridSoaY[i];
                rpts[i].offsetX = static_cast<int32_t>(std::round(rx));
                rpts[i].offsetY = static_cast<int32_t>(std::round(ry));

                // Rotate model direction and re-quantize to 16 bins
                // Uses threshold-based quantization (matches decompiled dword_1800D6A60)
                // for consistency with image bin quantization in AnglePyramid
                float rotCos = topLevel.gridSoaCosAngle[i] * cosR - topLevel.gridSoaSinAngle[i] * sinR;
                float rotSin = topLevel.gridSoaSinAngle[i] * cosR + topLevel.gridSoaCosAngle[i] * sinR;
                rpts[i].angleBin16 = Qi::Vision::Internal::GradientToBin16(rotCos, rotSin);
            }

            // 2. Compute search bounds (ensure all model point offsets stay in image)
            int32_t sxMin = 0, sxMax = targetWidth - 1;
            int32_t syMin = 0, syMax = targetHeight - 1;
            for (const auto& rp : rpts) {
                sxMin = std::max(sxMin, -rp.offsetX);
                sxMax = std::min(sxMax, targetWidth - 1 - rp.offsetX);
                syMin = std::max(syMin, -rp.offsetY);
                syMax = std::min(syMax, targetHeight - 1 - rp.offsetY);
            }
            if (sxMin > sxMax || syMin > syMax) continue;

            // Apply searchROI
            if (hasSearchROI) {
                sxMin = std::max(sxMin, levelROI.x);
                sxMax = std::min(sxMax, levelROI.x + levelROI.width - 1);
                syMin = std::max(syMin, levelROI.y);
                syMax = std::min(syMax, levelROI.y + levelROI.height - 1);
            }
            if (sxMin > sxMax || syMin > syMax) continue;

            // 3. Build response map
            // Decompiled: polarity mode uses signed LUT (sub_180007BA0),
            //             ignore-polarity modes use abs of LUT (sub_180007F60).
            const bool ignorePolarity = (params_.metric == MetricMode::IgnoreLocalPolarity ||
                                         params_.metric == MetricMode::IgnoreGlobalPolarity ||
                                         params_.metric == MetricMode::IgnoreColorPolarity);
            map.Clear();
            BuildResponseMap(map, rpts, angleBinData, binW, binH, binStride,
                             responseMapLUT_, sxMin, sxMax, syMin, syMax, ignorePolarity);

            // 4. 3x3 NMS to extract candidates
            auto resCands = ExtractCandidatesNMS3x3(
                map, sxMin, sxMax, syMin, syMax, minResponse);

            // 5. Convert to MatchResult
            for (const auto& rc : resCands) {
                MatchResult m;
                m.x = rc.x;
                m.y = rc.y;
                m.angle = angle;
                m.score = static_cast<double>(rc.response) / (numGridPts * 127.0);
                m.pyramidLevel = startLevel;
                localCandidates.push_back(m);
            }
        }

        #pragma omp critical
        {
            candidates.insert(candidates.end(), localCandidates.begin(), localCandidates.end());
        }
    }

    // A-2: Spatial hash NMS + angle distance suppression (replaces naive sort+truncate)
    candidates = CollectCandidatesNMS(std::move(candidates), targetWidth, coarseAngleStep);

    return candidates;
}

// =============================================================================
// Stage 1 fallback: CoarseSearchFloat — float dot-product (original)
// =============================================================================

std::vector<MatchResult> ShapeModelImpl::CoarseSearchFloat(
    const AnglePyramid& targetPyramid,
    int32_t startLevel,
    const SearchParams& params) const
{
    std::vector<MatchResult> candidates;

    const auto& topLevel = levels_[startLevel];
    int32_t targetWidth = targetPyramid.GetWidth(startLevel);
    int32_t targetHeight = targetPyramid.GetHeight(startLevel);

    // Step 2: Use levelCreateData_ maxRadius for angle step (decompiled behavior)
    double maxRadius = (startLevel < static_cast<int32_t>(levelCreateData_.size()))
        ? levelCreateData_[startLevel].maxRadius : 0.0;

    double coarseAngleStep = ComputeHalconAngleStep(maxRadius, 1.5);
    const double candidateThreshold = params.minScore * 0.8;
    constexpr bool useGridPoints = true;
    int32_t stepSize = 1;

    double levelScale = targetPyramid.GetScale(startLevel);
    Rect2i levelROI;
    bool hasSearchROI = (params.searchROI.width > 0 && params.searchROI.height > 0);
    if (hasSearchROI) {
        levelROI.x = static_cast<int32_t>(params.searchROI.x * levelScale);
        levelROI.y = static_cast<int32_t>(params.searchROI.y * levelScale);
        levelROI.width = static_cast<int32_t>(params.searchROI.width * levelScale);
        levelROI.height = static_cast<int32_t>(params.searchROI.height * levelScale);
    }

    const bool usePregenCache = !searchAngleCache_.empty() &&
                                 static_cast<size_t>(startLevel) < levels_.size() &&
                                 searchAngleStep_ > 0;

    if (usePregenCache) {
        const int32_t coarseStride = std::max(1, static_cast<int32_t>(
            coarseAngleStep / searchAngleStep_));

        std::vector<size_t> angleIndices;
        for (size_t ai = 0; ai < searchAngleCache_.size(); ai += coarseStride) {
            const double angle = searchAngleCache_[ai].angle;
            if (angle >= params.angleStart - 0.001 &&
                angle <= params.angleStart + params.angleExtent + 0.001) {
                angleIndices.push_back(ai);
            }
        }

        #pragma omp parallel
        {
            std::vector<MatchResult> localCandidates;

            #pragma omp for schedule(dynamic)
            for (size_t ii = 0; ii < angleIndices.size(); ++ii) {
                const size_t ai = angleIndices[ii];
                const SearchAngleData& angleData = searchAngleCache_[ai];
                const float cosR = angleData.cosA;
                const float sinR = angleData.sinA;
                const double angle = angleData.angle;

                const auto& bounds = angleData.levelBounds[startLevel];
                double scaleFactor = params.scaleMin;
                int32_t scaledMinX = static_cast<int32_t>(bounds.minX * scaleFactor);
                int32_t scaledMaxX = static_cast<int32_t>(bounds.maxX * scaleFactor);
                int32_t scaledMinY = static_cast<int32_t>(bounds.minY * scaleFactor);
                int32_t scaledMaxY = static_cast<int32_t>(bounds.maxY * scaleFactor);

                int32_t searchXMin = std::max(0, -scaledMinX);
                int32_t searchXMax = std::min(targetWidth - 1, targetWidth - 1 - scaledMaxX);
                int32_t searchYMin = std::max(0, -scaledMinY);
                int32_t searchYMax = std::min(targetHeight - 1, targetHeight - 1 - scaledMaxY);

                if (hasSearchROI) {
                    searchXMin = std::max(searchXMin, levelROI.x);
                    searchXMax = std::min(searchXMax, levelROI.x + levelROI.width - 1);
                    searchYMin = std::max(searchYMin, levelROI.y);
                    searchYMax = std::min(searchYMax, levelROI.y + levelROI.height - 1);
                }

                if (searchXMin > searchXMax || searchYMin > searchYMax) continue;

                for (int32_t y = searchYMin; y <= searchYMax; y += stepSize) {
                    for (int32_t x = searchXMin; x <= searchXMax; x += stepSize) {
                        double coverage = 0.0;
                        double score = ComputeScore(targetPyramid, startLevel,
                            x, y, cosR, sinR, params.scaleMin,
                            params.greediness, candidateThreshold,
                            &coverage, useGridPoints);

                        if (score >= candidateThreshold) {
                            MatchResult result;
                            result.x = x;
                            result.y = y;
                            result.angle = angle;
                            result.score = score;
                            result.pyramidLevel = startLevel;
                            localCandidates.push_back(result);
                        }
                    }
                }
            }

            #pragma omp critical
            {
                candidates.insert(candidates.end(), localCandidates.begin(), localCandidates.end());
            }
        }
    } else {
        std::vector<double> angles;
        {
            // Step 4: Uniform angle coverage (decompiled sub_1800B7FB0)
            double extent = params.angleExtent;
            int32_t numSteps = std::max(1, static_cast<int32_t>(std::round(extent / coarseAngleStep)));
            double adjustedStep = extent / numSteps;
            for (int32_t i = 0; i <= numSteps; ++i) {
                angles.push_back(params.angleStart + i * adjustedStep);
            }
        }

        #pragma omp parallel
        {
            std::vector<MatchResult> localCandidates;

            #pragma omp for schedule(dynamic)
            for (size_t ai = 0; ai < angles.size(); ++ai) {
                const double angle = angles[ai];
                const float cosR = static_cast<float>(std::cos(angle));
                const float sinR = static_cast<float>(std::sin(angle));

                double rMinX, rMaxX, rMinY, rMaxY;
                ComputeRotatedBounds(topLevel.points, angle, rMinX, rMaxX, rMinY, rMaxY);

                double scaleFactor = params.scaleMin;
                rMinX *= scaleFactor;
                rMaxX *= scaleFactor;
                rMinY *= scaleFactor;
                rMaxY *= scaleFactor;

                int32_t searchXMin = static_cast<int32_t>(std::ceil(-rMinX));
                int32_t searchXMax = static_cast<int32_t>(std::floor(targetWidth - 1 - rMaxX));
                int32_t searchYMin = static_cast<int32_t>(std::ceil(-rMinY));
                int32_t searchYMax = static_cast<int32_t>(std::floor(targetHeight - 1 - rMaxY));

                searchXMin = std::max(0, searchXMin);
                searchYMin = std::max(0, searchYMin);
                searchXMax = std::min(targetWidth - 1, searchXMax);
                searchYMax = std::min(targetHeight - 1, searchYMax);

                if (hasSearchROI) {
                    searchXMin = std::max(searchXMin, levelROI.x);
                    searchXMax = std::min(searchXMax, levelROI.x + levelROI.width - 1);
                    searchYMin = std::max(searchYMin, levelROI.y);
                    searchYMax = std::min(searchYMax, levelROI.y + levelROI.height - 1);
                }

                if (searchXMin > searchXMax || searchYMin > searchYMax) continue;

                for (int32_t y = searchYMin; y <= searchYMax; y += stepSize) {
                    for (int32_t x = searchXMin; x <= searchXMax; x += stepSize) {
                        double coverage = 0.0;
                        double score = ComputeScore(targetPyramid, startLevel,
                            x, y, cosR, sinR, params.scaleMin,
                            params.greediness, candidateThreshold,
                            &coverage, useGridPoints);

                        if (score >= candidateThreshold) {
                            MatchResult result;
                            result.x = x;
                            result.y = y;
                            result.angle = angle;
                            result.score = score;
                            result.pyramidLevel = startLevel;
                            localCandidates.push_back(result);
                        }
                    }
                }
            }

            #pragma omp critical
            {
                candidates.insert(candidates.end(), localCandidates.begin(), localCandidates.end());
            }
        }
    }

    // A-2: Spatial hash NMS + angle distance suppression (replaces naive sort+truncate)
    candidates = CollectCandidatesNMS(std::move(candidates), targetWidth, coarseAngleStep);

    return candidates;
}

// =============================================================================
// Stage 2: PyramidRefine — per-level dispatcher
// =============================================================================

std::vector<MatchResult> ShapeModelImpl::PyramidRefine(
    const AnglePyramid& targetPyramid,
    int32_t startLevel,
    std::vector<MatchResult> candidates,
    const SearchParams& params) const
{
    for (int32_t level = startLevel - 1; level >= 0; --level) {
        candidates = RefineAtLevel(targetPyramid, level, startLevel,
                                    std::move(candidates), params);
    }
    return candidates;
}

// =============================================================================
// RefineAtLevel — unified per-level refinement (position + angle)
// Decompiled sub_18003C7B0: all levels do position grid + angle iteration
//   level > 0: gridPoints,  level 0: subpixel points
// =============================================================================

std::vector<MatchResult> ShapeModelImpl::RefineAtLevel(
    const AnglePyramid& pyramid, int32_t level, int32_t startLevel,
    std::vector<MatchResult> candidates, const SearchParams& params) const
{
    // BUG #6 fix: use actual pyramid scale ratio
    double scaleFactor = pyramid.GetScale(level) / pyramid.GetScale(level + 1);

    // Angle step from level's maxRadius (all levels use safety=2.0)
    double levelRadius = (level < static_cast<int32_t>(levelCreateData_.size()))
        ? levelCreateData_[level].maxRadius : 0.0;
    double angleRadius = ComputeHalconAngleStep(levelRadius, 2.0);
    double angleStep = std::max(0.005, angleRadius / 3.0);

    double levelThreshold = params.minScore * 0.9;

    // DIFF #8: level 0 uses subpixel points, others use grid points
    bool useGridPoints = (level > 0);

    // Search radius from table lookup
    static constexpr int32_t MAX_SEARCH_RADIUS = 16;
    int32_t tableVal = (level < static_cast<int32_t>(searchRadiusPerLevel_.size()))
        ? searchRadiusPerLevel_[level] : 4;
    int32_t searchRadius = std::min(std::max(4, tableVal), MAX_SEARCH_RADIUS);

    // Get gradient data and model SoA
    detail::GradientView grad;
    if (!detail::GetGradientView(pyramid, level, grad)) {
        return {};
    }
    auto soa = detail::SelectSoA(levels_[level], useGridPoints);
    if (soa.count == 0) return {};

    const bool ignorePolarity = (params_.metric == MetricMode::IgnoreLocalPolarity ||
                                 params_.metric == MetricMode::IgnoreColorPolarity);
    const bool isGlobalPolarity = (params_.metric == MetricMode::IgnoreGlobalPolarity);

    const int32_t gridSize = 2 * searchRadius + 1;
    const int32_t gridArea = gridSize * gridSize;

    // Decompiled dword_1800D6A3C: convergence threshold
    // 0.1° balances precision vs performance at all levels
    constexpr double ANGLE_CONV_RAD = 0.1 * PI / 180.0;

    const int32_t numCandidates = static_cast<int32_t>(candidates.size());
    std::vector<MatchResult> allResults(numCandidates);
    std::vector<bool> validResults(numCandidates, false);

    #pragma omp parallel for schedule(dynamic)
    for (int32_t ci = 0; ci < numCandidates; ++ci) {
        const auto& candidate = candidates[ci];
        int32_t bx = static_cast<int32_t>(std::round(candidate.x * scaleFactor));
        int32_t by = static_cast<int32_t>(std::round(candidate.y * scaleFactor));
        double baseAngle = candidate.angle;

        // Stack-allocated score grids
        static constexpr int32_t MAX_GRID_AREA = (2 * MAX_SEARCH_RADIUS + 1) * (2 * MAX_SEARCH_RADIUS + 1);
        float scoreGrid[MAX_GRID_AREA];
        float scoreGridInv[MAX_GRID_AREA];

        double curAngle = baseAngle;
        double aStep = angleStep;

        // Helper lambda: build score grid at given angle+center, find peak with boundary retry
        // Returns peak position (outPx, outPy) in image coords
        auto buildGridFindPeak = [&](double testAngle, int32_t cx, int32_t cy,
                                     int32_t& outPx, int32_t& outPy) {
            float tCosR = static_cast<float>(std::cos(testAngle));
            float tSinR = static_cast<float>(std::sin(testAngle));

            int32_t curCX = cx, curCY = cy;
            int32_t retry = 0;

            while (true) {
                std::memset(scoreGrid, 0, gridArea * sizeof(float));
                if (isGlobalPolarity) std::memset(scoreGridInv, 0, gridArea * sizeof(float));

                // Pre-compute safe bounds
                int32_t safeGyMin = 0, safeGyMax = gridSize;
                {
                    int32_t minOffY = 0, maxOffY = 0, minOffX = 0, maxOffX = 0;
                    for (int32_t i = 0; i < soa.count; ++i) {
                        int32_t oX = static_cast<int32_t>(std::round(tCosR * soa.x[i] - tSinR * soa.y[i]));
                        int32_t oY = static_cast<int32_t>(std::round(tSinR * soa.x[i] + tCosR * soa.y[i]));
                        if (oX < minOffX) minOffX = oX;
                        if (oX > maxOffX) maxOffX = oX;
                        if (oY < minOffY) minOffY = oY;
                        if (oY > maxOffY) maxOffY = oY;
                    }
                    int32_t baseY = curCY - searchRadius;
                    safeGyMin = std::max(0, -baseY - minOffY);
                    safeGyMax = std::min(gridSize, grad.height - baseY - maxOffY);
                    int32_t baseX = curCX - searchRadius;
                    int32_t safeGxMin = std::max(0, -baseX - minOffX);
                    int32_t safeGxMax = std::min(gridSize, grad.width - baseX - maxOffX);
                    (void)safeGxMin; (void)safeGxMax;
                }

                const bool fullySafe = (safeGyMin == 0 && safeGyMax == gridSize);

                for (int32_t i = 0; i < soa.count; ++i) {
                    int32_t offX = static_cast<int32_t>(std::round(tCosR * soa.x[i] - tSinR * soa.y[i]));
                    int32_t offY = static_cast<int32_t>(std::round(tSinR * soa.x[i] + tCosR * soa.y[i]));

                    float rc = soa.cosAngle[i] * tCosR - soa.sinAngle[i] * tSinR;
                    float rs = soa.sinAngle[i] * tCosR + soa.cosAngle[i] * tSinR;

                    if (fullySafe) {
                        for (int32_t gy = 0; gy < gridSize; ++gy) {
                            int32_t imgY = curCY - searchRadius + gy + offY;
                            const float* gxRow = grad.gxData + imgY * grad.stride;
                            const float* gyRow = grad.gyData + imgY * grad.stride;
                            float* gRow = scoreGrid + gy * gridSize;
                            int32_t imgXBase = curCX - searchRadius + offX;
                            const float* gxPtr = gxRow + imgXBase;
                            const float* gyPtr = gyRow + imgXBase;

                            if (ignorePolarity) {
                                // Decompiled: vandps with abs mask after dot product
                                int32_t gx = 0;
#if HAVE_AVX2
                                const __m256 vRc = _mm256_set1_ps(rc);
                                const __m256 vRs = _mm256_set1_ps(rs);
                                const __m256 vAbsMask = _mm256_castsi256_ps(
                                    _mm256_set1_epi32(0x7FFFFFFF));
                                for (; gx + 8 <= gridSize; gx += 8) {
                                    __m256 vGx = _mm256_loadu_ps(gxPtr + gx);
                                    __m256 vGy = _mm256_loadu_ps(gyPtr + gx);
                                    __m256 vDot = _mm256_fmadd_ps(vRc, vGx,
                                        _mm256_mul_ps(vRs, vGy));
                                    vDot = _mm256_and_ps(vDot, vAbsMask);
                                    __m256 vAcc = _mm256_loadu_ps(gRow + gx);
                                    _mm256_storeu_ps(gRow + gx, _mm256_add_ps(vAcc, vDot));
                                }
#endif
                                for (; gx < gridSize; ++gx) {
                                    float dot = rc * gxPtr[gx] + rs * gyPtr[gx];
                                    gRow[gx] += std::fabs(dot);
                                }
                            } else if (isGlobalPolarity) {
                                float* gRowInv = scoreGridInv + gy * gridSize;
                                int32_t gx = 0;
#if HAVE_AVX2
                                const __m256 vRc = _mm256_set1_ps(rc);
                                const __m256 vRs = _mm256_set1_ps(rs);
                                const __m256 vZero = _mm256_setzero_ps();
                                for (; gx + 8 <= gridSize; gx += 8) {
                                    __m256 vGx = _mm256_loadu_ps(gxPtr + gx);
                                    __m256 vGy = _mm256_loadu_ps(gyPtr + gx);
                                    __m256 vDot = _mm256_fmadd_ps(vRc, vGx,
                                        _mm256_mul_ps(vRs, vGy));
                                    __m256 vPos = _mm256_max_ps(vDot, vZero);
                                    __m256 vNeg = _mm256_max_ps(
                                        _mm256_sub_ps(vZero, vDot), vZero);
                                    __m256 vAcc = _mm256_loadu_ps(gRow + gx);
                                    _mm256_storeu_ps(gRow + gx, _mm256_add_ps(vAcc, vPos));
                                    __m256 vAccInv = _mm256_loadu_ps(gRowInv + gx);
                                    _mm256_storeu_ps(gRowInv + gx,
                                        _mm256_add_ps(vAccInv, vNeg));
                                }
#endif
                                for (; gx < gridSize; ++gx) {
                                    float dot = rc * gxPtr[gx] + rs * gyPtr[gx];
                                    gRow[gx] += std::max(0.0f, dot);
                                    gRowInv[gx] += std::max(0.0f, -dot);
                                }
                            } else {
                                // Decompiled: vmulps + vfmadd231ps + vaddps (8-wide)
                                int32_t gx = 0;
#if HAVE_AVX2
                                const __m256 vRc = _mm256_set1_ps(rc);
                                const __m256 vRs = _mm256_set1_ps(rs);
                                for (; gx + 8 <= gridSize; gx += 8) {
                                    __m256 vGx = _mm256_loadu_ps(gxPtr + gx);
                                    __m256 vGy = _mm256_loadu_ps(gyPtr + gx);
                                    __m256 vDot = _mm256_fmadd_ps(vRc, vGx,
                                        _mm256_mul_ps(vRs, vGy));
                                    __m256 vAcc = _mm256_loadu_ps(gRow + gx);
                                    _mm256_storeu_ps(gRow + gx, _mm256_add_ps(vAcc, vDot));
                                }
#endif
                                for (; gx < gridSize; ++gx) {
                                    float dot = rc * gxPtr[gx] + rs * gyPtr[gx];
                                    gRow[gx] += dot;
                                }
                            }
                        }
                    } else {
                        for (int32_t gy = 0; gy < gridSize; ++gy) {
                            int32_t imgY = curCY - searchRadius + gy + offY;
                            if (imgY < 0 || imgY >= grad.height) continue;

                            const float* gxRow = grad.gxData + imgY * grad.stride;
                            const float* gyRow = grad.gyData + imgY * grad.stride;
                            float* gRow = scoreGrid + gy * gridSize;

                            int32_t imgXBase = curCX - searchRadius + offX;
                            int32_t xStart = std::max(0, -imgXBase);
                            int32_t xEnd = std::min(gridSize, grad.width - imgXBase);

                            if (ignorePolarity) {
                                for (int32_t gx = xStart; gx < xEnd; ++gx) {
                                    float dot = rc * gxRow[imgXBase + gx] + rs * gyRow[imgXBase + gx];
                                    gRow[gx] += std::fabs(dot);
                                }
                            } else if (isGlobalPolarity) {
                                float* gRowInv = scoreGridInv + gy * gridSize;
                                for (int32_t gx = xStart; gx < xEnd; ++gx) {
                                    float dot = rc * gxRow[imgXBase + gx] + rs * gyRow[imgXBase + gx];
                                    gRow[gx] += std::max(0.0f, dot);
                                    gRowInv[gx] += std::max(0.0f, -dot);
                                }
                            } else {
                                for (int32_t gx = xStart; gx < xEnd; ++gx) {
                                    float dot = rc * gxRow[imgXBase + gx] + rs * gyRow[imgXBase + gx];
                                    gRow[gx] += dot;
                                }
                            }
                        }
                    }
                }

                if (isGlobalPolarity) {
                    for (int32_t i = 0; i < gridArea; ++i) {
                        scoreGrid[i] = std::max(scoreGrid[i], scoreGridInv[i]);
                    }
                }

                // Find peak (scalar scan — grid is small, overhead of AVX2 not worthwhile)
                float bestVal = -1.0f;
                int32_t bestIdx = 0;
                for (int32_t gi = 0; gi < gridArea; ++gi) {
                    if (scoreGrid[gi] > bestVal) {
                        bestVal = scoreGrid[gi];
                        bestIdx = gi;
                    }
                }

                int32_t peakGx = bestIdx % gridSize;
                int32_t peakGy = bestIdx / gridSize;

                // Boundary retry (decompiled: max 1 retry when peak on edge)
                bool onBoundary = (peakGx == 0 || peakGx == gridSize - 1 ||
                                   peakGy == 0 || peakGy == gridSize - 1);
                if (onBoundary && retry < 1) {
                    curCX += peakGx - searchRadius;
                    curCY += peakGy - searchRadius;
                    retry++;
                    continue;
                }

                outPx = curCX + peakGx - searchRadius;
                outPy = curCY + peakGy - searchRadius;
                break;
            }
        };

        // Step 1: Grid search at current angle → find best position
        int32_t peakX = bx, peakY = by;
        buildGridFindPeak(curAngle, bx, by, peakX, peakY);

        // Step 2: Angle iteration using ComputeScore at peak position (O(N) per eval, NOT O(N×G²))
        // Decompiled sub_18003C7B0: evaluate angle candidates at the found position
        struct AngleEval { double angle = 0.0; double score = 0.0; };
        AngleEval evalCenter, evalLeft, evalRight;

        auto scoreAtAngle = [&](double testAngle) -> double {
            float tCosR = static_cast<float>(std::cos(testAngle));
            float tSinR = static_cast<float>(std::sin(testAngle));
            return ComputeScore(pyramid, level,
                static_cast<double>(peakX), static_cast<double>(peakY),
                tCosR, tSinR, params.scaleMin,
                0.0, 0.0, nullptr, useGridPoints);
        };

        double centerScore = scoreAtAngle(curAngle);
        evalCenter = {curAngle, centerScore};
        evalLeft = evalCenter;
        evalRight = evalCenter;

        while (std::fabs(aStep) >= ANGLE_CONV_RAD) {
            double sL = scoreAtAngle(curAngle - aStep);
            double sR = scoreAtAngle(curAngle + aStep);

            evalLeft = {curAngle - aStep, sL};
            evalRight = {curAngle + aStep, sR};

            if (sL > centerScore && sL >= sR) {
                evalCenter = {curAngle, centerScore};
                curAngle -= aStep;
                centerScore = sL;
            } else if (sR > centerScore) {
                evalCenter = {curAngle, centerScore};
                curAngle += aStep;
                centerScore = sR;
            } else {
                evalCenter = {curAngle, centerScore};
                aStep *= 0.5;
            }
        }

        // 3-point parabolic interpolation (decompiled divided-difference form)
        {
            double a1 = evalLeft.angle, s1 = evalLeft.score;
            double a2 = evalCenter.angle, s2 = evalCenter.score;
            double a3 = evalRight.angle, s3 = evalRight.score;

            if (std::fabs(a1 - a2) > 1e-12 && std::fabs(a2 - a3) > 1e-12 &&
                std::fabs(a1 - a3) > 1e-12) {
                double d12 = (s1 - s2) / (a1 - a2);
                double d23 = (s2 - s3) / (a2 - a3);
                double dd = (d12 - d23) / (a1 - a3);
                if (std::fabs(dd) > 1e-10) {
                    double peak = 0.5 * (a1 + a2 - d12 / dd);
                    double aMin = std::min({a1, a2, a3});
                    double aMax = std::max({a1, a2, a3});
                    if (peak >= aMin && peak <= aMax) {
                        curAngle = peak;
                    }
                }
            }
        }

        // Step 3: Final grid search at converged angle to get refined position
        int32_t finalPx = peakX, finalPy = peakY;
        buildGridFindPeak(curAngle, peakX, peakY, finalPx, finalPy);

        // Final ComputeScore evaluation
        MatchResult bestMatch;
        bestMatch.score = -1.0;
        bestMatch.pyramidLevel = level;
        {
            float cosR_f = static_cast<float>(std::cos(curAngle));
            float sinR_f = static_cast<float>(std::sin(curAngle));
            double coverage = 0.0;
            double score = ComputeScore(pyramid, level,
                static_cast<double>(finalPx), static_cast<double>(finalPy),
                cosR_f, sinR_f, params.scaleMin,
                0.0, 0.0, &coverage, useGridPoints);

            bestMatch.x = static_cast<double>(finalPx);
            bestMatch.y = static_cast<double>(finalPy);
            bestMatch.angle = curAngle;
            bestMatch.score = score;
        }

        allResults[ci] = bestMatch;
        if (bestMatch.score >= levelThreshold) {
            validResults[ci] = true;
        }
    }

    // Collect valid results
    std::vector<MatchResult> refined;
    refined.reserve(numCandidates);
    for (int32_t ci = 0; ci < numCandidates; ++ci) {
        if (validResults[ci]) {
            refined.push_back(allResults[ci]);
        }
    }

    // Angle normalization + range filtering (decompiled L1237-1303)
    double angleFilterThreshold = 2.0 * angleRadius;
    for (auto it = refined.begin(); it != refined.end(); ) {
        double a = it->angle;
        while (a > PI) a -= 2.0 * PI;
        while (a < -PI) a += 2.0 * PI;
        it->angle = a;

        double angleDiff = a - params.angleStart;
        while (angleDiff < 0.0) angleDiff += 2.0 * PI;
        while (angleDiff >= 2.0 * PI) angleDiff -= 2.0 * PI;

        if (angleDiff > params.angleExtent + angleFilterThreshold) {
            it = refined.erase(it);
        } else {
            ++it;
        }
    }

    std::sort(refined.begin(), refined.end());
    return refined;
}

// =============================================================================
// Stage 3: SubPixelRefine — subpixel position/angle refinement
// =============================================================================

std::vector<MatchResult> ShapeModelImpl::SubPixelRefine(
    const AnglePyramid& targetPyramid,
    std::vector<MatchResult> candidates,
    const SearchParams& params) const
{
    const double scale = levels_[0].scale;
    const double invScale = (scale != 1.0) ? (1.0 / scale) : 1.0;
    const int32_t numCandidates = static_cast<int32_t>(candidates.size());

    // Decompiled uses PPL parallel_for across candidates.
    // Each candidate is independent: gradient data is read-only,
    // RefinePosition uses only stack-local variables.
    #pragma omp parallel for schedule(dynamic)
    for (int32_t ci = 0; ci < numCandidates; ++ci) {
        auto& match = candidates[ci];
        if (params.subpixelMethod != SubpixelMethod::None) {
            RefinePosition(targetPyramid, match, params.subpixelMethod, params.scaleMin);
        }

        // Scale back from level 0 coordinates to original image coordinates
        if (scale != 1.0) {
            match.x *= invScale;
            match.y *= invScale;
        }
    }

    return candidates;
}

// =============================================================================
// Stage 4: FinalizeResults — final scoring + NMS
// =============================================================================

std::vector<MatchResult> ShapeModelImpl::FinalizeResults(
    const AnglePyramid& targetPyramid,
    std::vector<MatchResult> candidates,
    const SearchParams& params,
    bool applyNMS) const
{
    std::vector<MatchResult> results;
    results.reserve(candidates.size());

    // Step 6: Use SubPixelRefine scores directly, no re-scoring
    // SubPixelRefine already evaluates score at refined position (level 0 coords)
    // Re-scoring after coordinate scale-back would use wrong coordinate space
    for (auto match : candidates) {
        // Normalize angle to [-π, π]
        while (match.angle > PI) match.angle -= 2.0 * PI;
        while (match.angle < -PI) match.angle += 2.0 * PI;

        if (match.score >= params.minScore) {
            results.push_back(match);
        }
    }

    // Sort by score (descending)
    std::sort(results.begin(), results.end());

    // Non-maximum suppression (overlap-based, Halcon-compatible)
    if (applyNMS && params.maxOverlap < 1.0) {
        results = NonMaxSuppressionOverlap(results, params.maxOverlap,
                                            templateSize_.width, templateSize_.height);
    }

    // Limit results
    if (params.maxMatches > 0 && static_cast<int32_t>(results.size()) > params.maxMatches) {
        results.resize(params.maxMatches);
    }

    return results;
}

// =============================================================================
// SearchPyramid — Main entry point (4-stage pipeline)
// =============================================================================

std::vector<MatchResult> ShapeModelImpl::SearchPyramid(
    const AnglePyramid& targetPyramid,
    const SearchParams& params,
    bool applyNMS) const
{
    if (!valid_ || levels_.empty()) {
        return {};
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    int32_t startLevel = std::min(static_cast<int32_t>(levels_.size()) - 1,
                                   targetPyramid.NumLevels() - 1);

    // Respect numLevels parameter (0 = use all available)
    if (params.numLevels > 0) {
        startLevel = std::min(startLevel, params.numLevels - 1);
    }

    // Stage 1: Coarse search at top level
    auto candidates = CoarseSearch(targetPyramid, startLevel, params);

    auto t1 = std::chrono::high_resolution_clock::now();
    size_t coarseCandidates = candidates.size();

    if (timingParams_.enableTiming) {
        findTiming_.coarseSearchMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        findTiming_.numCoarseCandidates = static_cast<int32_t>(coarseCandidates);
    }

    // Stage 2: Pyramid refinement (coarse → fine)
    candidates = PyramidRefine(targetPyramid, startLevel, std::move(candidates), params);

    auto t2 = std::chrono::high_resolution_clock::now();
    size_t refinedCandidates = candidates.size();
    if (timingParams_.enableTiming) {
        findTiming_.pyramidRefineMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
    }

    // Stage 3: Subpixel refinement at level 0
    candidates = SubPixelRefine(targetPyramid, std::move(candidates), params);

    auto t3 = std::chrono::high_resolution_clock::now();
    if (timingParams_.enableTiming) {
        findTiming_.subpixelRefineMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
    }

    // Stage 4: Final scoring + NMS
    auto results = FinalizeResults(targetPyramid, std::move(candidates), params, applyNMS);

    auto t4 = std::chrono::high_resolution_clock::now();
    if (timingParams_.enableTiming) {
        findTiming_.nmsMs = std::chrono::duration<double, std::milli>(t4 - t3).count();
    }

    if (timingParams_.printTiming) {
        auto ms = [](auto start, auto end) {
            return std::chrono::duration<double, std::milli>(end - start).count();
        };
        fprintf(stderr, "[Timing] Coarse: %.1fms (%zu cands), Refine: %.1fms (%zu→%zu), SubPix: %.1fms, Final: %.1fms | Total: %.1fms\n",
                ms(t0, t1), coarseCandidates, ms(t1, t2), coarseCandidates, refinedCandidates,
                ms(t2, t3), ms(t3, t4), ms(t0, t4));
    }

    return results;
}

} // namespace Internal
} // namespace Qi::Vision::Matching
