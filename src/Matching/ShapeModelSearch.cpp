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

    // Compute model max radius for angle step
    double maxRadius = 0.0;
    for (const auto& pt : topLevel.points) {
        double r = std::sqrt(pt.x * pt.x + pt.y * pt.y);
        maxRadius = std::max(maxRadius, r);
    }

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
        for (double a = params.angleStart;
             a <= params.angleStart + params.angleExtent;
             a += coarseAngleStep) {
            angles.push_back(a);
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

    std::sort(candidates.begin(), candidates.end());

    if (candidates.size() > 1000) {
        candidates.resize(1000);
    }

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

    // Compute model max radius for angle step
    double maxRadius = 0.0;
    const auto& pts = topLevel.points;
    for (const auto& pt : pts) {
        double r = std::sqrt(pt.x * pt.x + pt.y * pt.y);
        maxRadius = std::max(maxRadius, r);
    }

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
        for (double angle = params.angleStart;
             angle <= params.angleStart + params.angleExtent;
             angle += coarseAngleStep) {
            angles.push_back(angle);
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

    std::sort(candidates.begin(), candidates.end());

    if (candidates.size() > 1000) {
        candidates.resize(1000);
    }

    return candidates;
}

// =============================================================================
// Stage 2: PyramidRefine — per-level candidate refinement
// =============================================================================

std::vector<MatchResult> ShapeModelImpl::PyramidRefine(
    const AnglePyramid& targetPyramid,
    int32_t startLevel,
    std::vector<MatchResult> candidates,
    const SearchParams& params) const
{
    // Compute model max radius for angle range computation
    double maxRadius = 0.0;
    if (!levels_.empty() && !levels_[0].points.empty()) {
        for (const auto& pt : levels_[0].points) {
            double r = std::sqrt(pt.x * pt.x + pt.y * pt.y);
            maxRadius = std::max(maxRadius, r);
        }
    }

    for (int32_t level = startLevel - 1; level >= 0; --level) {
        // BUG #6 fix: use actual pyramid scale ratio, not hardcoded 2.0
        // Going from coarser level (level+1) to finer level (level):
        // coordinates must be MULTIPLIED by finer/coarser ratio = 2.0
        double scaleFactor = targetPyramid.GetScale(level) / targetPyramid.GetScale(level + 1);

        // DIFF #2 fix: angle range from acos formula
        // A-6 fix: all refine levels use safety=2.0 (matches decompiled)
        double refineSafety = 2.0;
        double levelRadius = maxRadius * targetPyramid.GetScale(level);
        double angleRadius = ComputeHalconAngleStep(levelRadius, refineSafety);
        double angleStep = std::max(0.005, angleRadius / 3.0);

        // BUG #4 fix: level thresholds from decompiled constants {0.8, 0.9, 0.9, 0.9}
        double levelThreshold = (level == 0) ? (params.minScore * 0.8) : (params.minScore * 0.9);

        // DIFF #8 fix: level 0 uses subpixel points, others use grid points
        bool useGridPoints = (level > 0);

        int32_t searchRadius = 4;

        int32_t targetWidth = targetPyramid.GetWidth(level);
        int32_t targetHeight = targetPyramid.GetHeight(level);

        const int32_t numCandidates = static_cast<int32_t>(candidates.size());
        std::vector<MatchResult> allResults(numCandidates);
        std::vector<bool> validResults(numCandidates, false);

        #pragma omp parallel for schedule(dynamic)
        for (int32_t ci = 0; ci < numCandidates; ++ci) {
            const auto& candidate = candidates[ci];
            double baseX = candidate.x * scaleFactor;
            double baseY = candidate.y * scaleFactor;
            double baseAngle = candidate.angle;

            MatchResult bestMatch;
            bestMatch.score = -1.0;
            bestMatch.x = baseX;
            bestMatch.y = baseY;
            bestMatch.pyramidLevel = level;

            for (int32_t dy = -searchRadius; dy <= searchRadius; ++dy) {
                for (int32_t dx = -searchRadius; dx <= searchRadius; ++dx) {
                    double px = baseX + dx;
                    double py = baseY + dy;

                    if (px < 0 || px >= targetWidth || py < 0 || py >= targetHeight) continue;

                    for (double dAngle = -angleRadius; dAngle <= angleRadius; dAngle += angleStep) {
                        double angle = baseAngle + dAngle;
                        float cosR = static_cast<float>(std::cos(angle));
                        float sinR = static_cast<float>(std::sin(angle));

                        double coverage = 0.0;
                        // PyramidRefine uses greediness=0 to avoid early termination
                        // with many model points (early termination formula requires
                        // checking ~22% of points before passing, which fails at i=7)
                        double score = ComputeScore(targetPyramid, level,
                            px, py, cosR, sinR, params.scaleMin,
                            0.0, levelThreshold,
                            &coverage, useGridPoints);

                        if (score > bestMatch.score) {
                            bestMatch.x = px;
                            bestMatch.y = py;
                            bestMatch.angle = angle;
                            bestMatch.score = score;
                        }
                    }
                }
            }

            // Always store bestMatch so statistics reflect actual evaluation scores
            allResults[ci] = bestMatch;
            if (bestMatch.score >= levelThreshold) {
                validResults[ci] = true;
            }
        }

        // Collect valid results
        std::vector<MatchResult> refined;
        refined.reserve(numCandidates);
        double bestScoreAll = -1.0;
        for (int32_t ci = 0; ci < numCandidates; ++ci) {
            if (allResults[ci].score > bestScoreAll) bestScoreAll = allResults[ci].score;
            if (validResults[ci]) {
                refined.push_back(allResults[ci]);
            }
        }
        fprintf(stderr, "[DEBUG] PyramidRefine level %d: %d candidates → %zu valid (threshold=%.3f, bestScore=%.4f, angleRadius=%.4f)\n",
                level, numCandidates, refined.size(), levelThreshold, bestScoreAll, angleRadius);

        // Sort and limit
        std::sort(refined.begin(), refined.end());
        size_t limit = (level == 0) ? 50 : 500;
        if (refined.size() > limit) {
            refined.resize(limit);
        }

        candidates = std::move(refined);
    }

    return candidates;
}

// =============================================================================
// Stage 3: SubPixelRefine — subpixel position/angle refinement
// =============================================================================

std::vector<MatchResult> ShapeModelImpl::SubPixelRefine(
    const AnglePyramid& targetPyramid,
    std::vector<MatchResult> candidates,
    const SearchParams& params) const
{
    for (auto& match : candidates) {
        if (params.subpixelMethod != SubpixelMethod::None) {
            RefinePosition(targetPyramid, match, params.subpixelMethod, params.scaleMin);
        }

        // Scale back from level 0 coordinates to original image coordinates
        double scale = levels_[0].scale;
        if (scale != 1.0) {
            match.x /= scale;
            match.y /= scale;
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

    for (auto match : candidates) {
        // E-2 fix: Normalize angle to [-π, π]
        while (match.angle > PI) match.angle -= 2.0 * PI;
        while (match.angle < -PI) match.angle += 2.0 * PI;

        // Recompute precise score at level 0 (no greediness, no early stop)
        double coverage = 0.0;
        float cosR = static_cast<float>(std::cos(match.angle));
        float sinR = static_cast<float>(std::sin(match.angle));
        double similarity = ComputeScore(targetPyramid, 0,
            match.x, match.y, cosR, sinR, params.scaleMin,
            0.0, 0.0, &coverage, false);

        // BUG #5 fix: NO coverage penalty. Direct similarity score.
        // Halcon does NOT apply score *= coverage^0.75
        match.score = similarity;

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
    fprintf(stderr, "[DEBUG] Stage1 CoarseSearch (level %d): %zu candidates\n", startLevel, coarseCandidates);
    for (size_t i = 0; i < std::min(coarseCandidates, size_t(5)); ++i)
        fprintf(stderr, "  [%zu] x=%.1f y=%.1f angle=%.2f score=%.4f\n",
                i, candidates[i].x, candidates[i].y, candidates[i].angle, candidates[i].score);

    if (timingParams_.enableTiming) {
        findTiming_.coarseSearchMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        findTiming_.numCoarseCandidates = static_cast<int32_t>(coarseCandidates);
    }

    // Stage 2: Pyramid refinement (coarse → fine)
    candidates = PyramidRefine(targetPyramid, startLevel, std::move(candidates), params);
    fprintf(stderr, "[DEBUG] Stage2 PyramidRefine: %zu candidates\n", candidates.size());
    for (size_t i = 0; i < std::min(candidates.size(), size_t(5)); ++i)
        fprintf(stderr, "  [%zu] x=%.1f y=%.1f angle=%.2f score=%.4f\n",
                i, candidates[i].x, candidates[i].y, candidates[i].angle, candidates[i].score);

    auto t2 = std::chrono::high_resolution_clock::now();
    if (timingParams_.enableTiming) {
        findTiming_.pyramidRefineMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
    }

    // Stage 3: Subpixel refinement at level 0
    candidates = SubPixelRefine(targetPyramid, std::move(candidates), params);
    fprintf(stderr, "[DEBUG] Stage3 SubPixelRefine: %zu candidates\n", candidates.size());
    for (size_t i = 0; i < std::min(candidates.size(), size_t(5)); ++i)
        fprintf(stderr, "  [%zu] x=%.2f y=%.2f angle=%.4f score=%.4f\n",
                i, candidates[i].x, candidates[i].y, candidates[i].angle, candidates[i].score);

    auto t3 = std::chrono::high_resolution_clock::now();
    if (timingParams_.enableTiming) {
        findTiming_.subpixelRefineMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
    }

    // Stage 4: Final scoring + NMS
    auto results = FinalizeResults(targetPyramid, std::move(candidates), params, applyNMS);
    fprintf(stderr, "[DEBUG] Stage4 FinalizeResults: %zu results\n", results.size());

    auto t4 = std::chrono::high_resolution_clock::now();
    if (timingParams_.enableTiming) {
        findTiming_.nmsMs = std::chrono::duration<double, std::milli>(t4 - t3).count();
    }

    if (timingParams_.printTiming) {
        auto ms = [](auto start, auto end) {
            return std::chrono::duration<double, std::milli>(end - start).count();
        };
        fprintf(stderr, "[Timing] Coarse: %.1fms (%zu candidates), Refine: %.1fms, SubPix: %.1fms, NMS: %.1fms | Total: %.1fms\n",
                ms(t0, t1), coarseCandidates, ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t0, t4));
    }

    return results;
}

} // namespace Internal
} // namespace Qi::Vision::Matching
