/**
 * @file ShapeModelSearch.cpp
 * @brief 4-stage search pipeline for ShapeModel
 *
 * Architecture:
 *   SearchPyramid()                — Entry point
 *     ├── CoarseSearch()           — Stage 1: response map grid search (shared)
 *     ├── PyramidRefine()          — Stage 2: per-level dispatcher
 *     │   ├── level > startLevel:  — RefineAtLevel() always (angle-only, scale passthrough)
 *     │   └── level == startLevel: — Dispatch:
 *     │       ├── fixed scale:     — RefineAtLevel()       (sub_18003C7B0, 0.1°)
 *     │       └── scale range:     — RefineAtLevelScaled() (sub_180040150, 0.01°)
 *     ├── SubPixelRefine()         — Stage 3: subpixel refinement (shared)
 *     └── FinalizeResults()        — Stage 4: NMS + filtering (shared)
 *
 * Path branching (Stage 2):
 *   Decompiled architecture: intermediate levels always do angle-only refinement,
 *   joint angle+scale search only at the final level (level==startLevel) when scale range exists.
 *
 *   RefineAtLevel (decompiled sub_18003C7B0 / sub_18004C8C0):
 *     - Used for all intermediate levels AND final level with fixed scale
 *     - Angle-only binary search: eval 3 angles, halve step, converge at 0.1°
 *     - 27-point polyfit: joint (dx, dy, dAngle) subpixel
 *     - Uses candidate.scaleX for scoring (passthrough from CoarseSearch)
 *
 *   RefineAtLevelScaled (decompiled sub_180040150):
 *     - Used only at level==startLevel when scaleMin != scaleMax
 *     - 5x5 joint angle-scale grid: 24 evals/iter, halve both steps, converge at 0.01°
 *     - Newton divided-difference parabolic angle interpolation
 *     - 27-point polyfit: joint (dx, dy, dScale) subpixel
 *
 *   Shared between both paths:
 *     - BuildScoreGridFindPeak(): score grid accumulation + peak search + boundary retry
 *     - PolyFit27SubPixel(): 27-point 3D polynomial subpixel fit
 *     - EigenPositionRefine(): 2D eigendecomp fallback for position
 *     - Angle normalization + range filtering at level exit
 */

#include "ShapeModelImpl.h"
#include "ShapeModelScoreCore.h"
#include "ShapeModelResponseMap.h"

#include <QiVision/Internal/Solver.h>
#include <QiVision/Internal/Eigen.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace Qi::Vision::Matching {

namespace {

// Decompiled sub_1800B8160/sub_1800B7FB0: uniform range with roundHalfUp + remainder check
std::vector<double> GenerateUniformRange(double minVal, double maxVal, double step) {
    double range = maxVal - minVal;
    if (range <= 0 || step <= 0) return {minVal};

    double numF = range / step;
    // roundHalfUp: round to nearest, but 0.5 rounds up
    int num = static_cast<int>(numF + 0.5);
    if (num == 0) num = 1;

    // Remainder check: if range isn't evenly divisible, recompute step
    double remainder = std::fabs(range - num * step);
    if (remainder > step * 0.01) {
        num = static_cast<int>(std::floor(numF)) + 1;
        step = range / num;
    }

    std::vector<double> result(num + 1);
    for (int i = 0; i <= num; ++i) {
        result[i] = minVal + i * step;
    }
    return result;
}

// 27-point 3D polynomial fit for joint (dx, dy, dz) subpixel refinement
// scores[iz*9 + iy*3 + ix]: 3×3×3 neighborhood scores
// dzStep: z-dimension step size (angleStep or scaleStep)
// Returns {subDx, subDy, subDz, valid}
struct PolyFit27Result {
    double subDx = 0.0, subDy = 0.0, subDz = 0.0;
    bool valid = false;
};

PolyFit27Result PolyFit27SubPixel(const double scores[27], double dzStep) {
    using namespace Qi::Vision::Internal;

    // Build 27×10 design matrix with basis: [1, dx, dy, dz, dx², dy², dz², dx·dy, dx·dz, dy·dz]
    MatX A(27, 10);
    VecX b(27);

    int idx = 0;
    for (int iz = 0; iz < 3; ++iz) {
        double dz = (iz - 1) * dzStep;
        for (int iy = 0; iy < 3; ++iy) {
            double dy = static_cast<double>(iy - 1);
            for (int ix = 0; ix < 3; ++ix) {
                double dx = static_cast<double>(ix - 1);
                A(idx, 0) = 1.0;
                A(idx, 1) = dx;
                A(idx, 2) = dy;
                A(idx, 3) = dz;
                A(idx, 4) = dx * dx;
                A(idx, 5) = dy * dy;
                A(idx, 6) = dz * dz;
                A(idx, 7) = dx * dy;
                A(idx, 8) = dx * dz;
                A(idx, 9) = dy * dz;
                b[idx] = scores[idx];
                ++idx;
            }
        }
    }

    // Solve least squares: A * c = b
    VecX c = SolveLeastSquares(A, b);
    if (c.Size() != 10) return {};

    // Gradient: g = [c1, c2, c3]
    Vec3 g;
    g[0] = c[1];
    g[1] = c[2];
    g[2] = c[3];

    // Hessian: H = [[2c4, c7, c8], [c7, 2c5, c9], [c8, c9, 2c6]]
    Mat33 H;
    H(0, 0) = 2.0 * c[4];  H(0, 1) = c[7];        H(0, 2) = c[8];
    H(1, 0) = c[7];        H(1, 1) = 2.0 * c[5];  H(1, 2) = c[9];
    H(2, 0) = c[8];        H(2, 1) = c[9];        H(2, 2) = 2.0 * c[6];

    // Newton step: delta = -H^{-1} * g
    if (!IsSolvable3x3(H)) return {};
    Vec3 delta = Solve3x3(H, g);
    // delta = -H^{-1} * g, but Solve3x3 solves H*x = g, so we negate
    double subDx = -delta[0];
    double subDy = -delta[1];
    double subDz = -delta[2];

    // Decompiled: position within ±1.0 px (dword_1800D6AA8), z within ±dzStep
    PolyFit27Result result;
    if (std::fabs(subDx) < 1.0 && std::fabs(subDy) < 1.0 &&
        std::fabs(subDz) < dzStep) {
        result.subDx = subDx;
        result.subDy = subDy;
        result.subDz = subDz;
        result.valid = true;
    }
    return result;
}

// 2D Eigen position refinement using 3×3 spatial neighborhood
// scores[iy*3 + ix]: 3×3 neighborhood scores (row-major)
// Returns {subDx, subDy, valid}
struct EigenPosResult {
    double subDx = 0.0, subDy = 0.0;
    bool valid = false;
};

EigenPosResult EigenPositionRefine(const double scores[9]) {
    using namespace Qi::Vision::Internal;

    // Decompiled sub_18003C7B0: uniform [1,1,1] weights for row/column sums
    // s00 s01 s02
    // s10 s11 s12
    // s20 s21 s22
    double s00 = scores[0], s01 = scores[1], s02 = scores[2];
    double s10 = scores[3], s11 = scores[4], s12 = scores[5];
    double s20 = scores[6], s21 = scores[7], s22 = scores[8];

    // Column sums (uniform weight)
    double colLeft   = s00 + s10 + s20;  // x=-1
    double colCenter = s01 + s11 + s21;  // x=0
    double colRight  = s02 + s12 + s22;  // x=+1

    // Row sums (uniform weight)
    double rowTop    = s00 + s01 + s02;  // y=-1
    double rowCenter = s10 + s11 + s12;  // y=0
    double rowBot    = s20 + s21 + s22;  // y=+1

    // First-order derivatives (uniform kernel / 6.0)
    double gx = (colRight - colLeft) / 6.0;
    double gy = (rowBot - rowTop) / 6.0;

    // Second-order derivatives (uniform kernel / 6.0)
    double Hxx = (colRight + colLeft - 2.0 * colCenter) / 6.0;
    double Hyy = (rowBot + rowTop - 2.0 * rowCenter) / 6.0;
    double Hxy = (s00 - s02 - s20 + s22) * 0.25;

    // Decompiled: 2x2 eigendecomp matrix uses 2*Hyy and 2*Hxx on diagonal
    Mat22 eigMat;
    eigMat(0, 0) = 2.0 * Hyy;  eigMat(0, 1) = Hxy;
    eigMat(1, 0) = Hxy;        eigMat(1, 1) = 2.0 * Hxx;

    // Eigendecomposition
    auto eig = EigenSymmetric2x2(eigMat);
    if (!eig.valid) return {};

    // Both eigenvalues should be negative (score is a maximum)
    if (eig.lambda1 >= 0.0 && eig.lambda2 >= 0.0) return {};

    // Newton step along principal eigenvector (largest magnitude eigenvalue)
    double lambda = eig.lambda1;
    Vec2 v = eig.v1;
    if (std::fabs(eig.lambda2) > std::fabs(eig.lambda1)) {
        lambda = eig.lambda2;
        v = eig.v2;
    }

    if (std::fabs(lambda) < 1e-10) return {};

    // Decompiled: gradient projection uses gx/gy (note: v[0] is y-component, v[1] is x-component
    // since eigMat diagonal is [2*Hyy, 2*Hxx])
    double vg = v[0] * gy + v[1] * gx;

    // Decompiled: v^T * H * v uses ORIGINAL Hyy/Hxx (not 2x versions)
    // H_raw = [[Hyy, Hxy], [Hxy, Hxx]] matching eigenvector ordering
    double vHv = v[0] * v[0] * Hyy + v[0] * v[1] * Hxy + v[1] * v[1] * Hxx;

    if (std::fabs(vHv) < 1e-10) return {};

    double t = -0.5 * vg / vHv;

    // v[0] corresponds to y-direction, v[1] to x-direction (matching eigMat layout)
    double subDy = t * v[0];
    double subDx = t * v[1];

    // Decompiled dword_1800D6A8C: displacement threshold = 0.5
    EigenPosResult result;
    if (std::fabs(subDx) < 0.5 && std::fabs(subDy) < 0.5) {
        result.subDx = subDx;
        result.subDy = subDy;
        result.valid = true;
    }
    return result;
}

} // anonymous namespace

namespace Internal {

// =============================================================================
// BuildScoreGridFindPeak -- shared position grid search
// Used by both RefineAtLevel (candidate.scaleX passthrough) and RefineAtLevelScaled.
// Accumulates rotated+scaled model-target gradient dot products into a score grid,
// finds the peak position with boundary retry (max 1 attempt).
// =============================================================================

static void BuildScoreGridFindPeak(
    const detail::GradientView& grad,
    const detail::SoAView& soa,
    double testAngle, float scale,
    int32_t cx, int32_t cy,
    int32_t searchRadius,
    bool ignorePolarity, bool isGlobalPolarity,
    float* scoreGrid, float* scoreGridInv,
    int32_t& outPx, int32_t& outPy)
{
    const float tCosR = static_cast<float>(std::cos(testAngle));
    const float tSinR = static_cast<float>(std::sin(testAngle));
    const int32_t gridSize = 2 * searchRadius + 1;
    const int32_t gridArea = gridSize * gridSize;

    int32_t curCX = cx, curCY = cy;
    int32_t retry = 0;

    while (true) {
        std::memset(scoreGrid, 0, gridArea * sizeof(float));
        if (isGlobalPolarity) std::memset(scoreGridInv, 0, gridArea * sizeof(float));

        // Pre-compute safe bounds (Y-direction check for fullySafe fast path)
        int32_t safeGyMin = 0, safeGyMax = gridSize;
        {
            int32_t minOffY = 0, maxOffY = 0;
            for (int32_t i = 0; i < soa.count; ++i) {
                float sx = soa.x[i] * scale;
                float sy = soa.y[i] * scale;
                int32_t oY = static_cast<int32_t>(std::round(tSinR * sx + tCosR * sy));
                if (oY < minOffY) minOffY = oY;
                if (oY > maxOffY) maxOffY = oY;
            }
            int32_t baseY = curCY - searchRadius;
            safeGyMin = std::max(0, -baseY - minOffY);
            safeGyMax = std::min(gridSize, grad.height - baseY - maxOffY);
        }

        const bool fullySafe = (safeGyMin == 0 && safeGyMax == gridSize);

        // Accumulate gradient dot products
        for (int32_t i = 0; i < soa.count; ++i) {
            float sx = soa.x[i] * scale;
            float sy = soa.y[i] * scale;
            int32_t offX = static_cast<int32_t>(std::round(tCosR * sx - tSinR * sy));
            int32_t offY = static_cast<int32_t>(std::round(tSinR * sx + tCosR * sy));

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
                // Boundary-safe path: per-pixel clipping
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

        // Find peak
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
}

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

// Decompiled sub_1800B82C0: geometry-based scale step
// step = STEP_CONSTANT / ceil(maxRadius)
double ShapeModelImpl::ComputeScaleStep(double maxRadius) {
    constexpr double STEP_CONSTANT = 1.5;  // dword_1800D6B38
    if (maxRadius < 1.0) return 0.1;
    return STEP_CONSTANT / std::ceil(maxRadius);
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
        // Decompiled sub_1800B7FB0: uniform angle coverage with roundHalfUp
        angles = GenerateUniformRange(params.angleStart, params.angleStart + params.angleExtent,
                                       coarseAngleStep);
    }

    // Build scale list for coarse search (decompiled sub_1800B8160)
    std::vector<double> scaleList;
    if (params.scaleMin == params.scaleMax) {
        scaleList.push_back(params.scaleMin);
    } else {
        double coarseScaleStep = ComputeScaleStep(maxRadius);
        scaleList = GenerateUniformRange(params.scaleMin, params.scaleMax, coarseScaleStep);
    }

    // Build (scale, angle) work items for parallel iteration
    struct ScaleAnglePair { double scale; double angle; };
    std::vector<ScaleAnglePair> workItems;
    workItems.reserve(scaleList.size() * angles.size());
    for (double scale : scaleList) {
        for (double angle : angles) {
            workItems.push_back({scale, angle});
        }
    }

    std::vector<MatchResult> candidates;

    #pragma omp parallel
    {
        ResponseMap map;
        map.Allocate(targetWidth, targetHeight);
        std::vector<MatchResult> localCandidates;

        #pragma omp for schedule(dynamic)
        for (size_t wi = 0; wi < workItems.size(); ++wi) {
            double scale = workItems[wi].scale;
            double angle = workItems[wi].angle;
            float cosR = static_cast<float>(std::cos(angle));
            float sinR = static_cast<float>(std::sin(angle));
            float sf = static_cast<float>(scale);

            // 1. Rotate+scale model points → ResponsePoint (integer offsets + rotated 16-bin)
            std::vector<ResponsePoint> rpts(numGridPts);
            for (int32_t i = 0; i < numGridPts; ++i) {
                float sx = topLevel.gridSoaX[i] * sf;
                float sy = topLevel.gridSoaY[i] * sf;
                float rx = cosR * sx - sinR * sy;
                float ry = sinR * sx + cosR * sy;
                rpts[i].offsetX = static_cast<int32_t>(std::round(rx));
                rpts[i].offsetY = static_cast<int32_t>(std::round(ry));

                // Rotate model direction and re-quantize to 16 bins
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
            const bool ignorePolarity = (params_.metric == MetricMode::IgnoreLocalPolarity ||
                                         params_.metric == MetricMode::IgnoreGlobalPolarity ||
                                         params_.metric == MetricMode::IgnoreColorPolarity);
            map.Clear();
            BuildResponseMap(map, rpts, angleBinData, binW, binH, binStride,
                             responseMapLUT_, sxMin, sxMax, syMin, syMax, ignorePolarity);

            // 4. 3x3 NMS to extract candidates
            auto resCands = ExtractCandidatesNMS3x3(
                map, sxMin, sxMax, syMin, syMax, minResponse);

            // 4b. IoU NMS (decompiled sub_1800497F0 calls cv::dnn::NMSBoxes)
            // Box size = template size at current pyramid level × scale
            int32_t levelFactor = 1 << startLevel;
            int32_t boxW = std::max(1, static_cast<int32_t>(
                templateSize_.width * scale / levelFactor));
            int32_t boxH = std::max(1, static_cast<int32_t>(
                templateSize_.height * scale / levelFactor));
            resCands = IoUNMSCandidates(resCands, boxW, boxH, 0.5f);

            // 5. Convert to MatchResult
            for (const auto& rc : resCands) {
                MatchResult m;
                m.x = rc.x;
                m.y = rc.y;
                m.angle = angle;
                m.score = static_cast<double>(rc.response) / (numGridPts * 127.0);
                m.pyramidLevel = startLevel;
                m.scaleX = scale;
                m.scaleY = scale;
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

    // Build scale list (decompiled sub_1800B8160)
    std::vector<double> scaleList;
    if (params.scaleMin == params.scaleMax) {
        scaleList.push_back(params.scaleMin);
    } else {
        double coarseScaleStep = ComputeScaleStep(maxRadius);
        scaleList = GenerateUniformRange(params.scaleMin, params.scaleMax, coarseScaleStep);
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

        // Build (scale, angleIndex) work items
        struct WorkItem { double scale; size_t angleIdx; };
        std::vector<WorkItem> workItems;
        workItems.reserve(scaleList.size() * angleIndices.size());
        for (double scale : scaleList) {
            for (size_t idx : angleIndices) {
                workItems.push_back({scale, idx});
            }
        }

        #pragma omp parallel
        {
            std::vector<MatchResult> localCandidates;

            #pragma omp for schedule(dynamic)
            for (size_t wi = 0; wi < workItems.size(); ++wi) {
                double scaleFactor = workItems[wi].scale;
                const size_t ai = workItems[wi].angleIdx;
                const SearchAngleData& angleData = searchAngleCache_[ai];
                const float cosR = angleData.cosA;
                const float sinR = angleData.sinA;
                const double angle = angleData.angle;

                const auto& bounds = angleData.levelBounds[startLevel];
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
                            x, y, cosR, sinR, scaleFactor,
                            params.greediness, candidateThreshold,
                            &coverage, useGridPoints);

                        if (score >= candidateThreshold) {
                            MatchResult result;
                            result.x = x;
                            result.y = y;
                            result.angle = angle;
                            result.score = score;
                            result.pyramidLevel = startLevel;
                            result.scaleX = scaleFactor;
                            result.scaleY = scaleFactor;
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
        // Decompiled sub_1800B7FB0: uniform angle coverage with roundHalfUp
        std::vector<double> angles = GenerateUniformRange(
            params.angleStart, params.angleStart + params.angleExtent, coarseAngleStep);

        // Build (scale, angle) work items
        struct WorkItem { double scale; double angle; };
        std::vector<WorkItem> workItems;
        workItems.reserve(scaleList.size() * angles.size());
        for (double scale : scaleList) {
            for (double angle : angles) {
                workItems.push_back({scale, angle});
            }
        }

        #pragma omp parallel
        {
            std::vector<MatchResult> localCandidates;

            #pragma omp for schedule(dynamic)
            for (size_t wi = 0; wi < workItems.size(); ++wi) {
                double scaleFactor = workItems[wi].scale;
                const double angle = workItems[wi].angle;
                const float cosR = static_cast<float>(std::cos(angle));
                const float sinR = static_cast<float>(std::sin(angle));

                double rMinX, rMaxX, rMinY, rMaxY;
                ComputeRotatedBounds(topLevel.points, angle, rMinX, rMaxX, rMinY, rMaxY);

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
                            x, y, cosR, sinR, scaleFactor,
                            params.greediness, candidateThreshold,
                            &coverage, useGridPoints);

                        if (score >= candidateThreshold) {
                            MatchResult result;
                            result.x = x;
                            result.y = y;
                            result.angle = angle;
                            result.score = score;
                            result.pyramidLevel = startLevel;
                            result.scaleX = scaleFactor;
                            result.scaleY = scaleFactor;
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
// Stage 2: PyramidRefine -- per-level dispatcher
//
// Decompiled architecture (§1 line 117-134, §3.11):
//   level > startLevel:  always RefineAtLevel (angle-only, candidate scale passthrough)
//   level == startLevel: dispatch based on scale range:
//     - fixed scale (scaleMin == scaleMax): RefineAtLevel  (sub_18003C7B0)
//     - scale range  (scaleMin != scaleMax): RefineAtLevelScaled (sub_180040150)
//   startLevel defaults to 0 (params.startLevel, decompiled a13[1])
//
// Key differences:
//   | Aspect          | Path A (non-scaled)     | Path B (scaled)           |
//   |-----------------|-------------------------|---------------------------|
//   | Angle iteration | Binary search (3-point) | 5x5 joint grid (24-point) |
//   | Convergence     | 0.1deg (ANGLE_CONV_RAD) | 0.01deg (ANGLE_CONV_RAD)  |
//   | Level-0 radius  | No special limit        | Clamped to 4 (9x9 grid)   |
//   | Polyfit dz      | angleStep               | scaleStep                 |
//   | Post-iteration  | --                      | Newton angle interpolation|
//   | Decompiled ref  | sub_18003C7B0           | sub_180040150             |
// =============================================================================

std::vector<MatchResult> ShapeModelImpl::PyramidRefine(
    const AnglePyramid& targetPyramid,
    int32_t startLevel,
    std::vector<MatchResult> candidates,
    const SearchParams& params) const
{
    const bool hasScale = (params.scaleMin != params.scaleMax);
    const int32_t refineStopLevel = params.startLevel;  // decompiled a13[1], default 0

    for (int32_t level = startLevel - 1; level >= refineStopLevel; --level) {
        if (level == refineStopLevel && hasScale) {
            // Final level + scale range: joint angle+scale refinement (sub_180040150)
            candidates = RefineAtLevelScaled(targetPyramid, level, startLevel,
                                              std::move(candidates), params);
        } else {
            // Intermediate levels (angle-only, scale passthrough) or final level fixed-scale
            // (sub_18004C8C0 for level > refineStopLevel, sub_18003C7B0 for level == refineStopLevel)
            candidates = RefineAtLevel(targetPyramid, level, startLevel,
                                        std::move(candidates), params);
        }
    }
    return candidates;
}

// =============================================================================
// RefineAtLevel — per-level refinement (position + angle, scale passthrough)
// Decompiled sub_18003C7B0 (level==startLevel) / sub_18004C8C0 (level > startLevel)
// Uses candidate.scaleX from CoarseSearch; does NOT search over scale.
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

    // Search radius: use searchRadiusBase halving chain when available (scaled pipeline),
    // otherwise fall back to model table (non-scaled pipeline)
    static constexpr int32_t MAX_SEARCH_RADIUS = 16;
    int32_t searchRadius;
    if (params.searchRadiusBase > 0) {
        // Decompiled: searchRadiusBase from top, halve per level
        int32_t topLevel = startLevel - 1;
        int32_t topBase = std::min(params.searchRadiusBase, 32);
        int32_t scaledRadius = topBase;
        for (int32_t l = topLevel; l > level; --l) {
            scaledRadius = std::max(1, scaledRadius / 2);
        }
        searchRadius = std::min(std::max(1, scaledRadius), MAX_SEARCH_RADIUS);
    } else {
        // Non-scaled default: model table lookup
        int32_t tableVal = (level < static_cast<int32_t>(searchRadiusPerLevel_.size()))
            ? searchRadiusPerLevel_[level] : 4;
        searchRadius = std::min(std::max(4, tableVal), MAX_SEARCH_RADIUS);
    }

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

    // Non-scaled path: convergence threshold 0.1° (sub_18003C7B0 uses dynamic candidate list,
    // not the 0.01° from sub_180040150 which is scaled-only)
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

        // Step 1: Position grid search at current angle
        const float candidateScale = static_cast<float>(candidate.scaleX);
        int32_t peakX = bx, peakY = by;
        BuildScoreGridFindPeak(grad, soa, curAngle, candidateScale, bx, by, searchRadius,
            ignorePolarity, isGlobalPolarity, scoreGrid, scoreGridInv, peakX, peakY);

        // Step 2: Angle iteration using ComputeScore at peak position (O(N) per eval, NOT O(N×G²))
        // Decompiled sub_18003C7B0: evaluate angle candidates at the found position
        struct AngleEval { double angle = 0.0; double score = 0.0; };
        AngleEval evalCenter, evalLeft, evalRight;

        auto scoreAtAngle = [&](double testAngle) -> double {
            float tCosR = static_cast<float>(std::cos(testAngle));
            float tSinR = static_cast<float>(std::sin(testAngle));
            return ComputeScore(pyramid, level,
                static_cast<double>(peakX), static_cast<double>(peakY),
                tCosR, tSinR, static_cast<double>(candidateScale),
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

        // Step 3: 27-point polynomial fit for joint (dx, dy, dAngle) subpixel
        // Decompiled sub_18003C7B0: polyfit outputs (subDx, subDy, subDz) jointly.
        // If polyfit succeeds -> use its position AND angle. Eigen is only a fallback.
        MatchResult bestMatch;
        bestMatch.score = -1.0;
        bestMatch.pyramidLevel = level;
        {
            double angles[3] = { curAngle - angleStep, curAngle, curAngle + angleStep };
            double scores27[27];

            for (int iz = 0; iz < 3; ++iz) {
                float tCos = static_cast<float>(std::cos(angles[iz]));
                float tSin = static_cast<float>(std::sin(angles[iz]));
                for (int iy = 0; iy < 3; ++iy) {
                    double py = static_cast<double>(peakY + iy - 1);
                    for (int ix = 0; ix < 3; ++ix) {
                        double px = static_cast<double>(peakX + ix - 1);
                        scores27[iz * 9 + iy * 3 + ix] = ComputeScore(
                            pyramid, level, px, py, tCos, tSin,
                            static_cast<double>(candidateScale), 0.0, 0.0, nullptr, useGridPoints);
                    }
                }
            }

            double finalX, finalY;
            auto polyResult = PolyFit27SubPixel(scores27, angleStep);
            if (polyResult.valid) {
                // Decompiled: polyfit success -> use all three outputs directly
                finalX = static_cast<double>(peakX) + polyResult.subDx;
                finalY = static_cast<double>(peakY) + polyResult.subDy;
                curAngle += polyResult.subDz;
            } else {
                // Decompiled: polyfit failed -> fallback to Eigen for position only
                double scores9[9];
                for (int i = 0; i < 9; ++i) {
                    scores9[i] = scores27[9 + i];  // iz=1 (center angle) slice
                }

                auto eigenResult = EigenPositionRefine(scores9);
                if (eigenResult.valid) {
                    finalX = static_cast<double>(peakX) + eigenResult.subDx;
                    finalY = static_cast<double>(peakY) + eigenResult.subDy;
                } else {
                    finalX = static_cast<double>(peakX);
                    finalY = static_cast<double>(peakY);
                }
            }

            // Final score at refined position
            float cosR_f = static_cast<float>(std::cos(curAngle));
            float sinR_f = static_cast<float>(std::sin(curAngle));
            double score = ComputeScore(pyramid, level, finalX, finalY,
                cosR_f, sinR_f, static_cast<double>(candidateScale),
                0.0, 0.0, nullptr, useGridPoints);

            bestMatch.x = finalX;
            bestMatch.y = finalY;
            bestMatch.angle = curAngle;
            bestMatch.score = score;
            bestMatch.scaleX = candidate.scaleX;
            bestMatch.scaleY = candidate.scaleY;
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
// RefineAtLevelScaled — Path B: position + angle + scale refinement
// Decompiled sub_180040150: adds scale iteration on top of RefineAtLevel
// =============================================================================

std::vector<MatchResult> ShapeModelImpl::RefineAtLevelScaled(
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

    // Search radius: decompiled a12/10 → searchRadiusBase, halving per level
    // level[top] = searchRadiusBase, level[i] = max(1, level[i+1] / 2)
    // Final level (refineStopLevel) clamped to max 4 (9x9 grid)
    static constexpr int32_t MAX_SEARCH_RADIUS = 16;
    int32_t topLevel = startLevel - 1;  // highest level entering refinement
    // Priority: params.searchRadiusBase (decompiled a12/10) > model table fallback
    int32_t topBase;
    if (params.searchRadiusBase > 0) {
        topBase = std::min(params.searchRadiusBase, 32);  // decompiled clamp to 32
    } else {
        topBase = (topLevel >= 0 && topLevel < static_cast<int32_t>(searchRadiusPerLevel_.size()))
            ? searchRadiusPerLevel_[topLevel] : 4;
    }
    int32_t scaledRadius = topBase;
    for (int32_t l = topLevel; l > level; --l) {
        scaledRadius = std::max(1, scaledRadius / 2);
    }
    // Decompiled sub_180040150: if (a7 >= 4) v33 = a7; → minimum radius is 4
    int32_t refineStopLevel = params.startLevel;
    int32_t searchRadius = (level == refineStopLevel)
        ? std::max(4, scaledRadius)
        : std::min(std::max(1, scaledRadius), MAX_SEARCH_RADIUS);

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

    // Decompiled dword_1800D6A3C: convergence threshold ≈ 0.01°
    constexpr double ANGLE_CONV_RAD = 0.01 * PI / 180.0;

    // Decompiled sub_1800B82C0: geometry-based scale step
    double initialScaleStep = ComputeScaleStep(levelRadius);
    initialScaleStep = std::max(initialScaleStep, 0.001);

    const int32_t numCandidates = static_cast<int32_t>(candidates.size());
    std::vector<MatchResult> allResults(numCandidates);
    std::vector<bool> validResults(numCandidates, false);

    #pragma omp parallel for schedule(dynamic)
    for (int32_t ci = 0; ci < numCandidates; ++ci) {
        const auto& candidate = candidates[ci];
        int32_t bx = static_cast<int32_t>(std::round(candidate.x * scaleFactor));
        int32_t by = static_cast<int32_t>(std::round(candidate.y * scaleFactor));
        double baseAngle = candidate.angle;
        double curScale = candidate.scaleX;
        if (curScale <= 0.0) curScale = params.scaleMin;

        // Stack-allocated score grids
        static constexpr int32_t MAX_GRID_AREA = (2 * MAX_SEARCH_RADIUS + 1) * (2 * MAX_SEARCH_RADIUS + 1);
        float scoreGrid[MAX_GRID_AREA];
        float scoreGridInv[MAX_GRID_AREA];

        double curAngle = baseAngle;
        double aStep = angleStep;

        // Step 1: Position grid search at current angle+scale (shared helper)
        int32_t peakX = bx, peakY = by;
        BuildScoreGridFindPeak(grad, soa, curAngle, static_cast<float>(curScale), bx, by,
            searchRadius, ignorePolarity, isGlobalPolarity,
            scoreGrid, scoreGridInv, peakX, peakY);

        // Score helper
        auto scoreAtAngleScale = [&](double testAngle, double testScale) -> double {
            float tCosR = static_cast<float>(std::cos(testAngle));
            float tSinR = static_cast<float>(std::sin(testAngle));
            return ComputeScore(pyramid, level,
                static_cast<double>(peakX), static_cast<double>(peakY),
                tCosR, tSinR, testScale,
                0.0, 0.0, nullptr, useGridPoints);
        };

        // Step 2+3: Joint angle+scale 5×5 grid iteration (decompiled sub_180040150 §3.5)
        // Each iteration evaluates 5×5 = 24 angle-scale combinations (skip center),
        // picks best, halves both steps unconditionally
        double sStep = initialScaleStep;
        double centerScore = scoreAtAngleScale(curAngle, curScale);

        while (std::fabs(aStep) >= ANGLE_CONV_RAD || sStep >= 0.001) {
            double bestScore = centerScore;
            double bestAngle = curAngle;
            double bestScale = curScale;
            bool improved = false;

            // 5×5 joint search (decompiled sub_180040150 §3.5)
            for (int ai = -2; ai <= 2; ++ai) {
                double testA = curAngle + ai * aStep;
                for (int si = -2; si <= 2; ++si) {
                    if (ai == 0 && si == 0) continue;  // skip center
                    double testS = std::clamp(curScale + si * sStep,
                                               params.scaleMin, params.scaleMax);
                    double s = scoreAtAngleScale(testA, testS);
                    if (s > bestScore) {
                        bestScore = s;
                        bestAngle = testA;
                        bestScale = testS;
                        improved = true;
                    }
                }
            }

            if (improved) {
                curAngle = bestAngle;
                curScale = bestScale;
                centerScore = bestScore;
                // Re-find peak position after angle/scale update
                BuildScoreGridFindPeak(grad, soa, curAngle, static_cast<float>(curScale),
                    peakX, peakY, searchRadius, ignorePolarity, isGlobalPolarity,
                    scoreGrid, scoreGridInv, peakX, peakY);
                centerScore = scoreAtAngleScale(curAngle, curScale);
            }

            aStep *= 0.5;
            sStep *= 0.5;
        }

        // Parabolic angle interpolation — Newton divided-difference form (§3.5 step 4)
        {
            double finalAStep = aStep * 2.0;  // last step before final halving
            double sL = scoreAtAngleScale(curAngle - finalAStep, curScale);
            double sC = scoreAtAngleScale(curAngle, curScale);
            double sR = scoreAtAngleScale(curAngle + finalAStep, curScale);

            double x0 = curAngle - finalAStep;
            double x1 = curAngle;
            double x2 = curAngle + finalAStep;
            double d1 = (sR - sC) / (x2 - x1);
            double d2 = (sC - sL) / (x1 - x0);
            double dd = (d1 - d2) / (x2 - x0);
            if (std::fabs(dd) > 1e-10) {
                double peak = 0.5 * (x0 + x1) - d2 / dd;  // parabola vertex
                if (std::fabs(peak - curAngle) <= finalAStep) {
                    curAngle = peak;
                }
            }
        }

        // Step 4: 27-point polynomial fit for joint (dx, dy, dScale) subpixel
        // Decompiled: polyfit outputs (subDx, subDy, subDz) jointly.
        // If polyfit succeeds -> use its position AND scale. Eigen is only a fallback.
        MatchResult bestMatch;
        bestMatch.score = -1.0;
        bestMatch.pyramidLevel = level;
        {
            double scales[3] = {
                std::max(params.scaleMin, curScale - initialScaleStep),
                curScale,
                std::min(params.scaleMax, curScale + initialScaleStep)
            };
            double scores27[27];

            for (int iz = 0; iz < 3; ++iz) {
                float tCos = static_cast<float>(std::cos(curAngle));
                float tSin = static_cast<float>(std::sin(curAngle));
                for (int iy = 0; iy < 3; ++iy) {
                    double py = static_cast<double>(peakY + iy - 1);
                    for (int ix = 0; ix < 3; ++ix) {
                        double px = static_cast<double>(peakX + ix - 1);
                        scores27[iz * 9 + iy * 3 + ix] = ComputeScore(
                            pyramid, level, px, py, tCos, tSin,
                            scales[iz], 0.0, 0.0, nullptr, useGridPoints);
                    }
                }
            }

            double finalX, finalY;
            auto polyResult = PolyFit27SubPixel(scores27, initialScaleStep);
            if (polyResult.valid) {
                // Decompiled: polyfit success -> use all three outputs directly
                finalX = static_cast<double>(peakX) + polyResult.subDx;
                finalY = static_cast<double>(peakY) + polyResult.subDy;
                curScale = std::clamp(curScale + polyResult.subDz,
                                      params.scaleMin, params.scaleMax);
            } else {
                // Decompiled: polyfit failed -> fallback to Eigen for position only
                double scores9[9];
                for (int i = 0; i < 9; ++i) {
                    scores9[i] = scores27[9 + i];  // iz=1 (center scale) slice
                }

                auto eigenResult = EigenPositionRefine(scores9);
                if (eigenResult.valid) {
                    finalX = static_cast<double>(peakX) + eigenResult.subDx;
                    finalY = static_cast<double>(peakY) + eigenResult.subDy;
                } else {
                    finalX = static_cast<double>(peakX);
                    finalY = static_cast<double>(peakY);
                }
            }

            // Final score at refined position
            float cosR_f = static_cast<float>(std::cos(curAngle));
            float sinR_f = static_cast<float>(std::sin(curAngle));
            double score = ComputeScore(pyramid, level, finalX, finalY,
                cosR_f, sinR_f, curScale,
                0.0, 0.0, nullptr, useGridPoints);

            bestMatch.x = finalX;
            bestMatch.y = finalY;
            bestMatch.angle = curAngle;
            bestMatch.score = score;
            bestMatch.scaleX = curScale;
            bestMatch.scaleY = curScale;
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

    // Angle normalization + range filtering
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
            RefinePosition(targetPyramid, match, params.subpixelMethod, match.scaleX);
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
