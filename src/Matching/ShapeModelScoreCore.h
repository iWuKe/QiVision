/**
 * @file ShapeModelScoreCore.h
 * @brief Unified template scoring core for ShapeModel
 *
 * Architecture: Template (compile-time) + enum (runtime) dispatch.
 * - Compile-time: InterpMode (Bilinear / NearestNeighbor), SimdMode (Scalar / AVX2)
 * - Runtime: MetricMode (UsePolarity / IgnoreGlobalPolarity / IgnoreLocal)
 *
 * Only 3 template instantiations: Bilinear+Scalar, NN+Scalar, NN+AVX2.
 * All BUG fixes are localized in helper functions, not scattered across 6 copies.
 */

#pragma once

#include <QiVision/Internal/AnglePyramid.h>
#include <QiVision/Matching/MatchTypes.h>

#include <cmath>
#include <cstdint>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace Qi::Vision::Matching {
namespace Internal {

// Forward declarations
struct LevelModel;

namespace detail {

// =============================================================================
// Enums for template dispatch
// =============================================================================

enum class InterpMode { Bilinear, NearestNeighbor };
enum class SimdMode { Scalar, AVX2 };

// =============================================================================
// (A) SoA Data View — eliminates repeated grid/points selection
// =============================================================================

struct SoAView {
    const float* x;
    const float* y;
    const float* cosAngle;
    const float* sinAngle;
    const float* weight;
    const int16_t* angleBin;
    int32_t count;
    float totalWeight;  // Fixed normalizer = numPoints (all weights=1.0, do NOT use dynamic denominator)
};

// Implemented in ShapeModelScore.cpp (needs LevelModel full definition)
SoAView SelectSoA(const LevelModel& model, bool useGridPoints);

// =============================================================================
// (B) Gradient Map View
// =============================================================================

struct GradientView {
    const float* gxData;
    const float* gyData;
    int32_t width;
    int32_t height;
    int32_t stride;
};

inline bool GetGradientView(const Qi::Vision::Internal::AnglePyramid& pyramid,
                            int32_t level, GradientView& out) {
    return pyramid.GetGradientData(level, out.gxData, out.gyData,
                                   out.width, out.height, out.stride);
}

// =============================================================================
// (C) Minimum gradient threshold — fixes NN hardcoded 25.0f
// =============================================================================

inline float ComputeMinMagSq(double minContrast, double pyramidScale) {
    float baseMin = (minContrast > 0.0) ? static_cast<float>(minContrast) : 5.0f;
    float minMag = std::max(2.0f, baseMin * static_cast<float>(pyramidScale));
    return minMag * minMag;
}

// =============================================================================
// (D) Polarity-aware similarity — 3 modes unified
// =============================================================================

inline float ComputeSimilarity(float dot, float invMag,
                               MetricMode metric, MatchPolarity polarity) {
    float similarity = dot * invMag;
    float simPos = std::max(0.0f, similarity);
    float simNeg = std::max(0.0f, -similarity);

    if (metric == MetricMode::IgnoreLocalPolarity ||
        metric == MetricMode::IgnoreColorPolarity) {
        return std::fabs(similarity);
    }

    // UsePolarity and IgnoreGlobalPolarity both use polarity per-point
    // (IgnoreGlobalPolarity picks max of normal/inverted at the END)
    if (polarity == MatchPolarity::Opposite) {
        return simNeg;
    } else if (polarity == MatchPolarity::Any) {
        return std::max(simPos, simNeg);
    } else {
        return simPos;
    }
}

// =============================================================================
// (E) Early termination — BUG #3 fix (Halcon progressive threshold)
// =============================================================================
//
// Halcon formula: partialAvg < minScore - (1 - greediness) * (1 - checkedFraction)
// When greediness=1: never terminates early (tolerance = minScore - 0)
// When greediness=0: only terminates if partialAvg < minScore - (1 - fraction)
//   i.e., very conservative, allows low partial scores early

inline bool ShouldTerminateEarly(float totalScore, float fixedTotalWeight,
                                 int32_t checkedCount, int32_t numPoints,
                                 float minScore, float greediness) {
    if (greediness <= 0.0f || fixedTotalWeight <= 0.0f) return false;

    // Use fixed normalizer for consistent early termination
    float partialAvg = totalScore / fixedTotalWeight;
    float checkedFraction = static_cast<float>(checkedCount) / static_cast<float>(numPoints);
    float threshold = minScore - (1.0f - greediness) * (1.0f - checkedFraction);

    return partialAvg < threshold;
}

// =============================================================================
// (F) Score finalization
// =============================================================================

struct ScoreResult {
    double score;
    double coverage;
};

inline ScoreResult FinalizeScore(float totalScore, float fixedTotalWeight,
                                 int32_t matchedCount, int32_t numPoints) {
    ScoreResult r;
    r.coverage = static_cast<double>(matchedCount) / std::max(1, numPoints);
    // IMPORTANT: Fixed denominator = total model points (all weights = 1.0).
    // Halcon-aligned: unmatched points contribute 0 to numerator but ARE counted
    // in denominator. Do NOT change to "matched-only" dynamic denominator.
    r.score = (fixedTotalWeight > 0.0f)
        ? static_cast<double>(totalScore) / fixedTotalWeight
        : 0.0;
    return r;
}

// =============================================================================
// (G) Unified scoring template — ONE loop, written ONCE
// =============================================================================

template <InterpMode INTERP, SimdMode SIMD = SimdMode::Scalar>
ScoreResult ComputeScoreUnified(
    SoAView soa, GradientView grad,
    float posX, float posY, float cosR, float sinR, float scale,
    float minMagSq, float minScore, float greediness,
    MetricMode metric, MatchPolarity polarity);

// ---------------------------------------------------------------------------
// Scalar implementation (Bilinear or NN, selected by INTERP)
// ---------------------------------------------------------------------------

template <InterpMode INTERP>
inline ScoreResult ComputeScoreUnifiedScalar(
    SoAView soa, GradientView grad,
    float posX, float posY, float cosR, float sinR, float scale,
    float minMagSq, float minScore, float greediness,
    MetricMode metric, MatchPolarity polarity)
{
    const int32_t numPoints = soa.count;
    if (numPoints == 0) return {0.0, 0.0};

    // Bilinear needs w-2, h-2; NN needs w-1, h-1
    const int32_t maxX = (INTERP == InterpMode::Bilinear) ? grad.width - 2 : grad.width - 1;
    const int32_t maxY = (INTERP == InterpMode::Bilinear) ? grad.height - 2 : grad.height - 1;

    // Fixed normalizer: total weight of all model points (Halcon-compatible)
    const float fixedWeight = soa.totalWeight;

    float totalScore = 0.0f;
    int32_t matchedCount = 0;

    // For IgnoreGlobalPolarity: track inverted scores separately
    float totalScoreInv = 0.0f;
    int32_t matchedCountInv = 0;
    const bool isGlobalPolarity = (metric == MetricMode::IgnoreGlobalPolarity);

    for (int32_t i = 0; i < numPoints; ++i) {
        // 1. Rotate model point
        float rotX = cosR * soa.x[i] - sinR * soa.y[i];
        float rotY = sinR * soa.x[i] + cosR * soa.y[i];

        // 2. Image coordinate
        float imgX = posX + scale * rotX;
        float imgY = posY + scale * rotY;

        float gx, gy;

        if constexpr (INTERP == InterpMode::Bilinear) {
            // 3a. Bilinear interpolation
            int32_t ix = static_cast<int32_t>(imgX);
            int32_t iy = static_cast<int32_t>(imgY);
            if (imgX < 0) ix--;
            if (imgY < 0) iy--;

            if (ix < 0 || ix > maxX || iy < 0 || iy > maxY) continue;

            float fx = imgX - ix;
            float fy = imgY - iy;
            float w00 = (1.0f - fx) * (1.0f - fy);
            float w10 = fx * (1.0f - fy);
            float w01 = (1.0f - fx) * fy;
            float w11 = fx * fy;

            int32_t idx = iy * grad.stride + ix;
            gx = w00 * grad.gxData[idx] + w10 * grad.gxData[idx + 1] +
                 w01 * grad.gxData[idx + grad.stride] + w11 * grad.gxData[idx + grad.stride + 1];
            gy = w00 * grad.gyData[idx] + w10 * grad.gyData[idx + 1] +
                 w01 * grad.gyData[idx + grad.stride] + w11 * grad.gyData[idx + grad.stride + 1];
        } else {
            // 3b. Nearest neighbor
            int32_t ix = static_cast<int32_t>(imgX + 0.5f);
            int32_t iy = static_cast<int32_t>(imgY + 0.5f);

            if (ix < 0 || ix > maxX || iy < 0 || iy > maxY) continue;

            int32_t idx = iy * grad.stride + ix;
            gx = grad.gxData[idx];
            gy = grad.gyData[idx];
        }

        // 4. Gradient magnitude check
        float magSq = gx * gx + gy * gy;
        if (magSq < minMagSq) continue;

        // 5. Rotate model direction + dot product
        float rotCos = soa.cosAngle[i] * cosR - soa.sinAngle[i] * sinR;
        float rotSin = soa.sinAngle[i] * cosR + soa.cosAngle[i] * sinR;
        float dot = rotCos * gx + rotSin * gy;
        float invMag = 1.0f / std::sqrt(magSq);

        // 6. Similarity — accumulate score, use fixed weight for normalization
        if (isGlobalPolarity) {
            float similarity = dot * invMag;
            totalScore += soa.weight[i] * std::max(0.0f, similarity);
            totalScoreInv += soa.weight[i] * std::max(0.0f, -similarity);
            matchedCount++;
        } else {
            float score = ComputeSimilarity(dot, invMag, metric, polarity);
            if (score > 0.0f) {
                totalScore += soa.weight[i] * score;
                matchedCount++;
            }
        }

        // 7. Early termination check every 8 points (BUG #3 fix)
        if (((i + 1) & 7) == 0) {
            if (isGlobalPolarity) {
                bool termNormal = ShouldTerminateEarly(totalScore, fixedWeight,
                    i + 1, numPoints, minScore, greediness);
                bool termInv = ShouldTerminateEarly(totalScoreInv, fixedWeight,
                    i + 1, numPoints, minScore, greediness);
                if (termNormal && termInv) {
                    return {0.0, static_cast<double>(matchedCount) / numPoints};
                }
            } else {
                if (ShouldTerminateEarly(totalScore, fixedWeight,
                        i + 1, numPoints, minScore, greediness)) {
                    return {0.0, static_cast<double>(matchedCount) / numPoints};
                }
            }
        }
    }

    // Pick best polarity for IgnoreGlobalPolarity
    if (isGlobalPolarity) {
        if (totalScoreInv > totalScore) {
            return FinalizeScore(totalScoreInv, fixedWeight, matchedCount, numPoints);
        }
    }

    return FinalizeScore(totalScore, fixedWeight, matchedCount, numPoints);
}

// ---------------------------------------------------------------------------
// AVX2 implementation (Nearest Neighbor only)
// ---------------------------------------------------------------------------

#ifdef __AVX2__

static inline float HorizontalSumAVX2(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    vlow = _mm_hadd_ps(vlow, vlow);
    vlow = _mm_hadd_ps(vlow, vlow);
    return _mm_cvtss_f32(vlow);
}

inline ScoreResult ComputeScoreUnifiedAVX2(
    SoAView soa, GradientView grad,
    float posX, float posY, float cosR, float sinR, float scale,
    float minMagSq, float minScore, float greediness,
    MetricMode metric, MatchPolarity polarity)
{
    // AVX2 fast path: no early termination — used only for top-level coarse screening
    (void)minScore;
    (void)greediness;

    const int32_t numPoints = soa.count;
    if (numPoints == 0) return {0.0, 0.0};

    // Fixed normalizer
    const float fixedWeight = soa.totalWeight;

    const __m256 vCosR = _mm256_set1_ps(cosR);
    const __m256 vSinR = _mm256_set1_ps(sinR);
    const __m256 vXf = _mm256_set1_ps(posX);
    const __m256 vYf = _mm256_set1_ps(posY);
    const __m256 vScale = _mm256_set1_ps(scale);
    const __m256 vHalf = _mm256_set1_ps(0.5f);
    const __m256 vZero = _mm256_setzero_ps();
    const __m256 vMinMagSq = _mm256_set1_ps(minMagSq);
    const __m256 vMaxXf = _mm256_set1_ps(static_cast<float>(grad.width - 1));
    const __m256 vMaxYf = _mm256_set1_ps(static_cast<float>(grad.height - 1));
    const __m256i vStride = _mm256_set1_epi32(grad.stride);
    const __m256 vAbsMask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 vOne = _mm256_set1_ps(1.0f);

    __m256 vTotalScore = _mm256_setzero_ps();
    __m256 vMatchedCount = _mm256_setzero_ps();

    // For IgnoreGlobalPolarity
    const bool isGlobalPolarity = (metric == MetricMode::IgnoreGlobalPolarity);
    __m256 vTotalScoreInv = _mm256_setzero_ps();

    const bool isIgnoreLocal = (metric == MetricMode::IgnoreLocalPolarity ||
                                metric == MetricMode::IgnoreColorPolarity);

    int32_t i = 0;
    const int32_t numVec = numPoints & ~7;

    for (; i < numVec; i += 8) {
        __m256 vSoaX = _mm256_loadu_ps(&soa.x[i]);
        __m256 vSoaY = _mm256_loadu_ps(&soa.y[i]);
        __m256 vSoaCos = _mm256_loadu_ps(&soa.cosAngle[i]);
        __m256 vSoaSin = _mm256_loadu_ps(&soa.sinAngle[i]);
        __m256 vWeight = _mm256_loadu_ps(&soa.weight[i]);

        // Rotate: rotX = cos*x - sin*y, rotY = sin*x + cos*y
        __m256 vRotX = _mm256_fmsub_ps(vCosR, vSoaX, _mm256_mul_ps(vSinR, vSoaY));
        __m256 vRotY = _mm256_fmadd_ps(vSinR, vSoaX, _mm256_mul_ps(vCosR, vSoaY));

        // Image coordinates
        __m256 vImgX = _mm256_fmadd_ps(vScale, vRotX, vXf);
        __m256 vImgY = _mm256_fmadd_ps(vScale, vRotY, vYf);

        // Nearest neighbor
        __m256i vIx = _mm256_cvtps_epi32(_mm256_add_ps(vImgX, vHalf));
        __m256i vIy = _mm256_cvtps_epi32(_mm256_add_ps(vImgY, vHalf));

        // Linear index
        __m256i vIdx = _mm256_add_epi32(_mm256_mullo_epi32(vIy, vStride), vIx);

        // Boundary check
        __m256 vRoundedX = _mm256_cvtepi32_ps(vIx);
        __m256 vRoundedY = _mm256_cvtepi32_ps(vIy);
        __m256 vBoundsMask = _mm256_and_ps(
            _mm256_and_ps(
                _mm256_cmp_ps(vRoundedX, vZero, _CMP_GE_OS),
                _mm256_cmp_ps(vRoundedX, vMaxXf, _CMP_LE_OS)),
            _mm256_and_ps(
                _mm256_cmp_ps(vRoundedY, vZero, _CMP_GE_OS),
                _mm256_cmp_ps(vRoundedY, vMaxYf, _CMP_LE_OS)));

        // Safe gather
        __m256i vGatherMask = _mm256_castps_si256(vBoundsMask);
        __m256i vSafeIdx = _mm256_blendv_epi8(_mm256_setzero_si256(), vIdx, vGatherMask);

        __m256 vGx = _mm256_and_ps(_mm256_i32gather_ps(grad.gxData, vSafeIdx, 4), vBoundsMask);
        __m256 vGy = _mm256_and_ps(_mm256_i32gather_ps(grad.gyData, vSafeIdx, 4), vBoundsMask);

        // Magnitude check
        __m256 vMagSq = _mm256_fmadd_ps(vGx, vGx, _mm256_mul_ps(vGy, vGy));
        __m256 vMagMask = _mm256_cmp_ps(vMagSq, vMinMagSq, _CMP_GE_OS);
        __m256 vValidMask = _mm256_and_ps(vBoundsMask, vMagMask);

        // Rotate direction
        __m256 vRotCos = _mm256_fmsub_ps(vSoaCos, vCosR, _mm256_mul_ps(vSoaSin, vSinR));
        __m256 vRotSin = _mm256_fmadd_ps(vSoaSin, vCosR, _mm256_mul_ps(vSoaCos, vSinR));

        // Dot product
        __m256 vDot = _mm256_fmadd_ps(vRotCos, vGx, _mm256_mul_ps(vRotSin, vGy));

        // Inverse magnitude
        __m256 vMagSqSafe = _mm256_max_ps(vMagSq, _mm256_set1_ps(1e-10f));
        __m256 vInvMag = _mm256_rsqrt_ps(vMagSqSafe);

        // Score based on metric
        __m256 vScore;
        if (isIgnoreLocal) {
            vScore = _mm256_mul_ps(_mm256_and_ps(vDot, vAbsMask), vInvMag);
        } else if (isGlobalPolarity) {
            __m256 vSim = _mm256_mul_ps(vDot, vInvMag);
            vScore = _mm256_max_ps(vSim, vZero);  // positive
            __m256 vScoreInv = _mm256_max_ps(_mm256_sub_ps(vZero, vSim), vZero);
            vTotalScoreInv = _mm256_add_ps(vTotalScoreInv,
                _mm256_and_ps(_mm256_mul_ps(vWeight, vScoreInv), vValidMask));
        } else {
            // UsePolarity
            __m256 vSim = _mm256_mul_ps(vDot, vInvMag);
            if (polarity == MatchPolarity::Opposite) {
                vScore = _mm256_max_ps(_mm256_sub_ps(vZero, vSim), vZero);
            } else if (polarity == MatchPolarity::Any) {
                vScore = _mm256_max_ps(_mm256_and_ps(vSim, vAbsMask), vZero);
            } else {
                vScore = _mm256_max_ps(vSim, vZero);
            }
        }

        __m256 vContrib = _mm256_mul_ps(vWeight, vScore);
        vTotalScore = _mm256_add_ps(vTotalScore, _mm256_and_ps(vContrib, vValidMask));
        vMatchedCount = _mm256_add_ps(vMatchedCount, _mm256_and_ps(vOne, vValidMask));
    }

    // Horizontal reduction
    float totalScore = HorizontalSumAVX2(vTotalScore);
    int32_t matchedCount = static_cast<int32_t>(HorizontalSumAVX2(vMatchedCount));
    float totalScoreInv = isGlobalPolarity ? HorizontalSumAVX2(vTotalScoreInv) : 0.0f;

    // Scalar cleanup for remaining points
    const int32_t maxCoord = grad.width - 1;
    const int32_t maxCoordY = grad.height - 1;

    for (; i < numPoints; ++i) {
        float rotX = cosR * soa.x[i] - sinR * soa.y[i];
        float rotY = sinR * soa.x[i] + cosR * soa.y[i];
        int32_t imgX = static_cast<int32_t>(posX + scale * rotX + 0.5f);
        int32_t imgY = static_cast<int32_t>(posY + scale * rotY + 0.5f);

        if (imgX < 0 || imgX > maxCoord || imgY < 0 || imgY > maxCoordY) continue;

        int32_t idx = imgY * grad.stride + imgX;
        float gx = grad.gxData[idx];
        float gy = grad.gyData[idx];

        float magSq_s = gx * gx + gy * gy;
        if (magSq_s < minMagSq) continue;

        float rotCos = soa.cosAngle[i] * cosR - soa.sinAngle[i] * sinR;
        float rotSin = soa.sinAngle[i] * cosR + soa.cosAngle[i] * sinR;
        float dot = rotCos * gx + rotSin * gy;
        float inv = 1.0f / std::sqrt(magSq_s);

        if (isIgnoreLocal) {
            float score = std::fabs(dot) * inv;
            totalScore += soa.weight[i] * score;
            matchedCount++;
        } else if (isGlobalPolarity) {
            float sim = dot * inv;
            totalScore += soa.weight[i] * std::max(0.0f, sim);
            totalScoreInv += soa.weight[i] * std::max(0.0f, -sim);
            matchedCount++;
        } else {
            float score = ComputeSimilarity(dot, inv, metric, polarity);
            if (score > 0.0f) {
                totalScore += soa.weight[i] * score;
                matchedCount++;
            }
        }
    }

    if (isGlobalPolarity) {
        if (totalScoreInv > totalScore) {
            return FinalizeScore(totalScoreInv, fixedWeight, matchedCount, numPoints);
        }
    }

    return FinalizeScore(totalScore, fixedWeight, matchedCount, numPoints);
}

#endif  // __AVX2__

// ---------------------------------------------------------------------------
// Template specializations — dispatch to correct implementation
// ---------------------------------------------------------------------------

template <>
inline ScoreResult ComputeScoreUnified<InterpMode::Bilinear, SimdMode::Scalar>(
    SoAView soa, GradientView grad,
    float posX, float posY, float cosR, float sinR, float scale,
    float minMagSq, float minScore, float greediness,
    MetricMode metric, MatchPolarity polarity)
{
    return ComputeScoreUnifiedScalar<InterpMode::Bilinear>(
        soa, grad, posX, posY, cosR, sinR, scale,
        minMagSq, minScore, greediness, metric, polarity);
}

template <>
inline ScoreResult ComputeScoreUnified<InterpMode::NearestNeighbor, SimdMode::Scalar>(
    SoAView soa, GradientView grad,
    float posX, float posY, float cosR, float sinR, float scale,
    float minMagSq, float minScore, float greediness,
    MetricMode metric, MatchPolarity polarity)
{
    return ComputeScoreUnifiedScalar<InterpMode::NearestNeighbor>(
        soa, grad, posX, posY, cosR, sinR, scale,
        minMagSq, minScore, greediness, metric, polarity);
}

#ifdef __AVX2__
template <>
inline ScoreResult ComputeScoreUnified<InterpMode::NearestNeighbor, SimdMode::AVX2>(
    SoAView soa, GradientView grad,
    float posX, float posY, float cosR, float sinR, float scale,
    float minMagSq, float minScore, float greediness,
    MetricMode metric, MatchPolarity polarity)
{
    return ComputeScoreUnifiedAVX2(
        soa, grad, posX, posY, cosR, sinR, scale,
        minMagSq, minScore, greediness, metric, polarity);
}
#endif

} // namespace detail
} // namespace Internal
} // namespace Qi::Vision::Matching
