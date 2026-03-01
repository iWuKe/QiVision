/**
 * @file ShapeModelScore.cpp
 * @brief Unified score computation for ShapeModel
 *
 * Architecture after refactor:
 * - ComputeScore(): Single entry point, dispatches to template instantiations
 * - RefinePosition(): Subpixel refinement (Parabolic / Gauss-Newton gradient profile)
 * - RefineGaussNewton(): Per-point gradient direction profile + 3x3 linear solve
 * - FastCosTable: Cosine lookup table
 * - detail::SelectSoA(): SoA data view factory
 *
 * All scoring logic lives in ShapeModelScoreCore.h as templates.
 */

#include "ShapeModelImpl.h"
#include "ShapeModelScoreCore.h"

#include <QiVision/Internal/Solver.h>

#include <cmath>
#include <algorithm>

namespace Qi::Vision::Matching {
namespace Internal {

// =============================================================================
// FastCosTable Implementation
// =============================================================================

FastCosTable::FastCosTable() {
    for (int i = 0; i < TABLE_SIZE; ++i) {
        double angle = (i * 2.0 * PI) / TABLE_SIZE;
        table_[i] = static_cast<float>(std::cos(angle));
    }
}

float FastCosTable::FastCos(double angle) const {
    angle = angle - std::floor(angle * INV_2PI) * (2.0 * PI);
    int idx = static_cast<int>(angle * TABLE_SCALE) & TABLE_MASK;
    return table_[idx];
}

float FastCosTable::FastAbsCos(double angleDiff) const {
    angleDiff = std::fabs(angleDiff);
    angleDiff = angleDiff - std::floor(angleDiff * INV_PI) * PI;
    int idx = static_cast<int>(angleDiff * TABLE_SCALE) & TABLE_MASK;
    return std::fabs(table_[idx]);
}

const FastCosTable g_cosTable;

// =============================================================================
// detail::SelectSoA — SoA data view factory
// =============================================================================

namespace detail {

SoAView SelectSoA(const LevelModel& model, bool useGridPoints) {
    SoAView v;
    if (useGridPoints && !model.gridSoaX.empty()) {
        v.x = model.gridSoaX.data();
        v.y = model.gridSoaY.data();
        v.cosAngle = model.gridSoaCosAngle.data();
        v.sinAngle = model.gridSoaSinAngle.data();
        v.weight = model.gridSoaWeight.data();
        v.angleBin = model.gridSoaAngleBin.data();
        v.count = static_cast<int32_t>(model.gridPoints.size());
    } else {
        v.x = model.soaX.data();
        v.y = model.soaY.data();
        v.cosAngle = model.soaCosAngle.data();
        v.sinAngle = model.soaSinAngle.data();
        v.weight = model.soaWeight.data();
        v.angleBin = model.soaAngleBin.data();
        v.count = static_cast<int32_t>(model.points.size());
    }
    // Pre-compute fixed normalizer: sum of all weights
    v.totalWeight = 0.0f;
    for (int32_t i = 0; i < v.count; ++i) {
        v.totalWeight += v.weight[i];
    }
    return v;
}

} // namespace detail

// =============================================================================
// ShapeModelImpl::ComputeScore — SINGLE entry point
// =============================================================================

double ShapeModelImpl::ComputeScore(
    const AnglePyramid& pyramid, int32_t level,
    double x, double y, float cosR, float sinR, double scale,
    double greediness, double minScore,
    double* outCoverage, bool useGridPoints) const
{
    if (level < 0 || level >= static_cast<int32_t>(levels_.size())) {
        if (outCoverage) *outCoverage = 0.0;
        return 0.0;
    }

    auto soa = detail::SelectSoA(levels_[level], useGridPoints);
    if (soa.count == 0) {
        if (outCoverage) *outCoverage = 0.0;
        return 0.0;
    }

    detail::GradientView grad;
    if (!detail::GetGradientView(pyramid, level, grad)) {
        if (outCoverage) *outCoverage = 0.0;
        return 0.0;
    }

    float minMagSq = detail::ComputeMinMagSq(params_.minContrast, pyramid.GetScale(level));
    float posX = static_cast<float>(x);
    float posY = static_cast<float>(y);
    float scalef = static_cast<float>(scale);
    float greed = static_cast<float>(greediness);
    float minSc = static_cast<float>(minScore);

    // Select template instantiation based on level and platform
    // Default: always bilinear (accurate).
    // AVX2 NN fast path: only for top level + IgnoreLocalPolarity + enough points.
    const int32_t numLevels = static_cast<int32_t>(levels_.size());
    const bool isTopLevel = (level == numLevels - 1);

    detail::ScoreResult result;

#ifdef __AVX2__
    {
        const bool isIgnoreLocal = (params_.metric == MetricMode::IgnoreLocalPolarity ||
                                     params_.metric == MetricMode::IgnoreColorPolarity);
        if (isTopLevel && isIgnoreLocal && soa.count >= 32) {
            result = detail::ComputeScoreUnified<detail::InterpMode::NearestNeighbor,
                                                  detail::SimdMode::AVX2>(
                soa, grad, posX, posY, cosR, sinR, scalef,
                minMagSq, minSc, greed, params_.metric, params_.polarity);
            if (outCoverage) *outCoverage = result.coverage;
            return result.score;
        }
    }
#endif
    (void)isTopLevel;
    {
        result = detail::ComputeScoreUnified<detail::InterpMode::Bilinear,
                                              detail::SimdMode::Scalar>(
            soa, grad, posX, posY, cosR, sinR, scalef,
            minMagSq, minSc, greed, params_.metric, params_.polarity);
    }

    if (outCoverage) *outCoverage = result.coverage;
    return result.score;
}

// =============================================================================
// ShapeModelImpl::RefinePosition
// =============================================================================

void ShapeModelImpl::RefinePosition(
    const AnglePyramid& pyramid, MatchResult& match,
    SubpixelMethod method, double scale) const
{
    if (method == SubpixelMethod::None) {
        return;
    }

    constexpr int32_t level = 0;  // Always refine at finest level

    // Helper lambda for score evaluation (always bilinear, no greediness)
    auto evalScore = [&](double ex, double ey, double eangle) -> double {
        float cosR = static_cast<float>(std::cos(eangle));
        float sinR = static_cast<float>(std::sin(eangle));
        return ComputeScore(pyramid, level, ex, ey, cosR, sinR, scale,
                           0.0, 0.0, nullptr, false);
    };

    if (method == SubpixelMethod::Parabolic) {
        // =================================================================
        // HALCON 'interpolation' style: Parabolic fitting (position only)
        // 9 Score evaluations for 3x3 grid, then re-evaluate final score
        // Halcon mode 1 does NOT have independent angle refinement
        // =================================================================

        constexpr double delta = 0.5;

        // Position refinement: 3x3 sampling grid
        double scores[3][3];
        for (int32_t dy = -1; dy <= 1; ++dy) {
            for (int32_t dx = -1; dx <= 1; ++dx) {
                scores[dy + 1][dx + 1] = evalScore(
                    match.x + dx * delta, match.y + dy * delta, match.angle);
            }
        }

        // X-direction subpixel (middle row)
        double denom_x = 2.0 * (scores[1][0] - 2.0 * scores[1][1] + scores[1][2]);
        if (std::abs(denom_x) > 1e-10) {
            double dx = delta * (scores[1][0] - scores[1][2]) / denom_x;
            dx = std::max(-delta, std::min(delta, dx));
            match.x += dx;
        }

        // Y-direction subpixel (middle column)
        double denom_y = 2.0 * (scores[0][1] - 2.0 * scores[1][1] + scores[2][1]);
        if (std::abs(denom_y) > 1e-10) {
            double dy = delta * (scores[0][1] - scores[2][1]) / denom_y;
            dy = std::max(-delta, std::min(delta, dy));
            match.y += dy;
        }

        // Re-evaluate final score at refined position (original angle)
        match.score = evalScore(match.x, match.y, match.angle);
    }
    else if (method == SubpixelMethod::LeastSquares ||
             method == SubpixelMethod::LeastSquaresHigh ||
             method == SubpixelMethod::LeastSquaresVeryHigh) {
        // =================================================================
        // HALCON 'least_squares' style: Per-point gradient profile +
        // Gauss-Newton 3x3 linear solve
        // - LeastSquares: 1 iteration
        // - LeastSquaresHigh / LeastSquaresVeryHigh: 2 iterations
        // =================================================================

        int32_t numIter = (method == SubpixelMethod::LeastSquares) ? 1 : 2;
        RefineGaussNewton(pyramid, match, scale, numIter);

        // Re-evaluate final score at refined position
        match.score = evalScore(match.x, match.y, match.angle);
    }

    match.refined = true;
}

// =============================================================================
// ShapeModelImpl::RefineGaussNewton
//
// Per-point gradient direction profile + 3x3 linear least squares solve.
// Matches decompiled Halcon 'least_squares' refinement (sub_180038660):
//   For each model point, Bresenham-step along gradient normal ±5 pixels,
//   find best forward/backward scores, fit parabola for sub-pixel displacement,
//   then solve ICP-style normal equations for (dx, dy, dtheta).
// =============================================================================

void ShapeModelImpl::RefineGaussNewton(
    const AnglePyramid& pyramid, MatchResult& match,
    double scale, int32_t numIterations) const
{
    constexpr int32_t level = 0;

    if (level >= static_cast<int32_t>(levels_.size())) return;

    // Get gradient data for level 0
    detail::GradientView grad;
    if (!detail::GetGradientView(pyramid, level, grad)) return;

    // Use subpixel points (Block 1) at level 0
    auto soa = detail::SelectSoA(levels_[level], false);
    if (soa.count == 0) return;

    float minMagSq = detail::ComputeMinMagSq(params_.minContrast, pyramid.GetScale(level));

    const int32_t maxIx = grad.width - 2;   // bilinear needs +1 neighbor
    const int32_t maxIy = grad.height - 2;
    const float scalef = static_cast<float>(scale);

    double cx = match.x;
    double cy = match.y;
    double angle = match.angle;

    for (int32_t iter = 0; iter < numIterations; ++iter) {
        float cosR = static_cast<float>(std::cos(angle));
        float sinR = static_cast<float>(std::sin(angle));

        // Normal equation accumulators: ATA (3x3 symmetric) and ATb (3x1)
        double ata00 = 0, ata01 = 0, ata02 = 0;
        double ata11 = 0, ata12 = 0;
        double ata22 = 0;
        double atb0 = 0, atb1 = 0, atb2 = 0;
        int32_t validCount = 0;

        for (int32_t i = 0; i < soa.count; ++i) {
            // 1. Rotate model point
            float rotX = cosR * soa.x[i] - sinR * soa.y[i];
            float rotY = sinR * soa.x[i] + cosR * soa.y[i];

            // 2. Image position
            float imgX = static_cast<float>(cx) + scalef * rotX;
            float imgY = static_cast<float>(cy) + scalef * rotY;

            // 3. Bilinear interpolation of gradient at center
            int32_t ix = static_cast<int32_t>(imgX);
            int32_t iy = static_cast<int32_t>(imgY);
            if (imgX < 0) ix--;
            if (imgY < 0) iy--;

            if (ix < 0 || ix > maxIx || iy < 0 || iy > maxIy) continue;

            float fx = imgX - ix;
            float fy = imgY - iy;
            float w00 = (1.0f - fx) * (1.0f - fy);
            float w10 = fx * (1.0f - fy);
            float w01 = (1.0f - fx) * fy;
            float w11 = fx * fy;

            int32_t idx = iy * grad.stride + ix;
            float gx0 = w00 * grad.gxData[idx] + w10 * grad.gxData[idx + 1] +
                         w01 * grad.gxData[idx + grad.stride] + w11 * grad.gxData[idx + grad.stride + 1];
            float gy0 = w00 * grad.gyData[idx] + w10 * grad.gyData[idx + 1] +
                         w01 * grad.gyData[idx + grad.stride] + w11 * grad.gyData[idx + grad.stride + 1];

            // 4. Gradient magnitude check
            float magSq = gx0 * gx0 + gy0 * gy0;
            if (magSq < minMagSq) continue;

            // 5. Gradient normal direction (unit vector)
            float invMag = 1.0f / std::sqrt(magSq);
            float nx = gx0 * invMag;
            float ny = gy0 * invMag;

            // 6. Rotated model direction
            float mcx = soa.cosAngle[i] * cosR - soa.sinAngle[i] * sinR;
            float mcy = soa.sinAngle[i] * cosR + soa.cosAngle[i] * sinR;

            // 7. Bresenham direction profile (decompiled sub_180038660)
            //    Step along gradient normal using Bresenham, ±5 steps max
            float score_center = mcx * gx0 + mcy * gy0;

            // Bresenham setup: main axis = larger |component|
            bool mainIsX = std::fabs(nx) >= std::fabs(ny);
            int32_t mainStep, crossStep;
            float errorInc;
            if (mainIsX) {
                mainStep = (nx >= 0.0f) ? 1 : -1;
                crossStep = (ny >= 0.0f) ? 1 : -1;
                errorInc = (std::fabs(nx) > 1e-10f) ? std::fabs(ny / nx) : 0.0f;
            } else {
                mainStep = (ny >= 0.0f) ? 1 : -1;
                crossStep = (nx >= 0.0f) ? 1 : -1;
                errorInc = (std::fabs(ny) > 1e-10f) ? std::fabs(nx / ny) : 0.0f;
            }

            // Forward scan (up to 5 steps along normal)
            float bestFwdScore = -1e30f;
            int32_t bestFwdStep = 0;
            {
                int32_t px = 0, py = 0;
                float err = 0.0f;
                for (int32_t s = 1; s <= 5; ++s) {
                    if (mainIsX) px += mainStep; else py += mainStep;
                    err += errorInc;
                    if (err >= 0.5f) {
                        err -= 1.0f;
                        if (mainIsX) py += crossStep; else px += crossStep;
                    }
                    float spx = imgX + px, spy = imgY + py;
                    int32_t six = static_cast<int32_t>(spx);
                    int32_t siy = static_cast<int32_t>(spy);
                    if (spx < 0) six--;
                    if (spy < 0) siy--;
                    if (six < 0 || six > maxIx || siy < 0 || siy > maxIy) break;
                    float sfx = spx - six, sfy = spy - siy;
                    int32_t sidx = siy * grad.stride + six;
                    float sgx = (1.0f-sfx)*(1.0f-sfy)*grad.gxData[sidx] + sfx*(1.0f-sfy)*grad.gxData[sidx+1] +
                                (1.0f-sfx)*sfy*grad.gxData[sidx+grad.stride] + sfx*sfy*grad.gxData[sidx+grad.stride+1];
                    float sgy = (1.0f-sfx)*(1.0f-sfy)*grad.gyData[sidx] + sfx*(1.0f-sfy)*grad.gyData[sidx+1] +
                                (1.0f-sfx)*sfy*grad.gyData[sidx+grad.stride] + sfx*sfy*grad.gyData[sidx+grad.stride+1];
                    float ss = mcx * sgx + mcy * sgy;
                    if (ss > bestFwdScore) { bestFwdScore = ss; bestFwdStep = s; }
                }
            }

            // Backward scan (up to 5 steps, negated direction)
            float bestBwdScore = -1e30f;
            int32_t bestBwdStep = 0;
            {
                int32_t px = 0, py = 0;
                float err = 0.0f;
                for (int32_t s = 1; s <= 5; ++s) {
                    if (mainIsX) px -= mainStep; else py -= mainStep;
                    err += errorInc;
                    if (err >= 0.5f) {
                        err -= 1.0f;
                        if (mainIsX) py -= crossStep; else px -= crossStep;
                    }
                    float spx = imgX + px, spy = imgY + py;
                    int32_t six = static_cast<int32_t>(spx);
                    int32_t siy = static_cast<int32_t>(spy);
                    if (spx < 0) six--;
                    if (spy < 0) siy--;
                    if (six < 0 || six > maxIx || siy < 0 || siy > maxIy) break;
                    float sfx = spx - six, sfy = spy - siy;
                    int32_t sidx = siy * grad.stride + six;
                    float sgx = (1.0f-sfx)*(1.0f-sfy)*grad.gxData[sidx] + sfx*(1.0f-sfy)*grad.gxData[sidx+1] +
                                (1.0f-sfx)*sfy*grad.gxData[sidx+grad.stride] + sfx*sfy*grad.gxData[sidx+grad.stride+1];
                    float sgy = (1.0f-sfx)*(1.0f-sfy)*grad.gyData[sidx] + sfx*(1.0f-sfy)*grad.gyData[sidx+1] +
                                (1.0f-sfx)*sfy*grad.gyData[sidx+grad.stride] + sfx*sfy*grad.gyData[sidx+grad.stride+1];
                    float ss = mcx * sgx + mcy * sgy;
                    if (ss > bestBwdScore) { bestBwdScore = ss; bestBwdStep = s; }
                }
            }

            if (bestFwdStep == 0 || bestBwdStep == 0) continue;

            // 8. Parabolic fit with non-uniform spacing
            // Points: (-bestBwdStep, bestBwdScore), (0, score_center), (bestFwdStep, bestFwdScore)
            double sf = static_cast<double>(bestFwdStep);
            double sb = static_cast<double>(bestBwdStep);
            double dfScore = static_cast<double>(bestFwdScore) - static_cast<double>(score_center);
            double dbScore = static_cast<double>(bestBwdScore) - static_cast<double>(score_center);
            double denomFit = 2.0 * (dbScore * sf + dfScore * sb);
            if (std::abs(denomFit) < 1e-10) continue;
            double d_raw = -(dfScore * sb * sb - dbScore * sf * sf) / denomFit;
            float d_i = static_cast<float>(std::max(-sb, std::min(sf, d_raw)));

            // 9. Jacobian row: [nx, ny, scale*(-rotY*nx + rotX*ny)]
            double Jx = static_cast<double>(nx);
            double Jy = static_cast<double>(ny);
            double Jtheta = static_cast<double>(scalef) *
                (static_cast<double>(-rotY) * Jx + static_cast<double>(rotX) * Jy);

            // 10. Accumulate normal equations: ATA += J*J^T, ATb += J*(-d_i)
            //     (negative sign: decompiled uses position -= solve_result)
            double neg_d = static_cast<double>(-d_i);

            ata00 += Jx * Jx;
            ata01 += Jx * Jy;
            ata02 += Jx * Jtheta;
            ata11 += Jy * Jy;
            ata12 += Jy * Jtheta;
            ata22 += Jtheta * Jtheta;

            atb0 += Jx * neg_d;
            atb1 += Jy * neg_d;
            atb2 += Jtheta * neg_d;

            validCount++;
        }

        // Need at least 3 valid points for a 3-DOF solve
        if (validCount < 3) break;

        // Solve 3x3 normal equation: ATA * x = ATb
        Qi::Vision::Internal::Mat33 ATA{
            ata00, ata01, ata02,
            ata01, ata11, ata12,
            ata02, ata12, ata22
        };
        Qi::Vision::Internal::Vec3 ATb{atb0, atb1, atb2};

        auto result = Qi::Vision::Internal::Solve3x3(ATA, ATb);

        // Clamp corrections (decompiled limits)
        double dx = std::max(-2.0, std::min(2.0, result[0]));
        double dy = std::max(-2.0, std::min(2.0, result[1]));
        double dtheta = std::max(-0.1, std::min(0.1, result[2]));

        // Update position: position -= solve_result (decompiled convention)
        cx -= dx;
        cy -= dy;
        angle -= dtheta;
    }

    match.x = cx;
    match.y = cy;
    match.angle = angle;
}

} // namespace Internal
} // namespace Qi::Vision::Matching
