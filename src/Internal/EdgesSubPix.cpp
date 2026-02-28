/**
 * @file EdgesSubPix.cpp
 * @brief Sub-pixel edge detection — HALCON-compatible edges_sub_pix_gray
 *
 * Architecture (reverse-engineered from decompiled library):
 *
 *   Subsystem A — Sobel gradient field ("where are edges"):
 *     Input → GaussianBlur(opt) → Sobel dx,dy (×0.25) → magnitude = √(dx²+dy²)
 *
 *   Subsystem B — Facet Model + Hessian eigensystem ("exact position"):
 *     3×3 magnitude window → quadratic surface fit →
 *     Hessian eigendecomposition → simultaneously:
 *       ① NMS:   λ_n < 0 (magnitude ridge = local max along normal)
 *       ② SubPx: t = -(b·nx+f·ny)/λ_n,  |t| ≤ 0.5
 *       ③ Flag:  magnitude vs thresholds → flag 2 (strong) or 1 (weak)
 *     No separate NMS pass — the Hessian check IS the NMS.
 *
 *   Subsystem C — Greedy chain linking ("ordered edge chains"):
 *     Sort candidates by magnitude descending →
 *     For each unvisited flag==2 seed: extend bidirectionally along tangent,
 *     using direction-based neighbor generation (2-3 candidates along edge
 *     tangent), accepting flag≥1 neighbors with angle consistency (< 67.5°).
 *
 * Coordinate convention: x = column, y = row.
 * Angle convention:      Hessian eigenvector atan2(ny, nx), gradient-adjusted, [0, 2π).
 * AngleBin convention:   4 bins on tangent_mod_π ∈ [0, π), bin = floor(t*4/π+0.5) % 4.
 */

#include <QiVision/Internal/EdgesSubPix.h>
#include <QiVision/Internal/Convolution.h>  // GaussianBlurFixed<>
#include <QiVision/Internal/Gradient.h>     // Gradient<>
#include <QiVision/Core/Types.h>            // BorderMode

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Qi::Vision::Internal {

// =============================================================================
// Internal helpers
// =============================================================================

namespace {

/**
 * @brief Compute 2D quadratic surface coefficients from a 3×3 window.
 *
 * Fits  f(x,y) = a + b·x + f_·y + e·x² + c·y² + d·x·y
 * over the 3×3 neighbourhood of gradient magnitude values by the
 * closed-form least-squares solution for a uniform 3×3 grid.
 *
 * Naming follows the HALCON technical document:
 *   b  = ∂f/∂x   (first-order x)
 *   f_ = ∂f/∂y   (first-order y)
 *   e  = ∂²f/∂x² (second-order x)
 *   c  = ∂²f/∂y² (second-order y)
 *   d  = ∂²f/∂x∂y (cross term)
 *
 * Grid layout (rows = y-offset, cols = x-offset):
 *   m00 m01 m02        dy=-1
 *   m10 m11 m12        dy= 0
 *   m20 m21 m22        dy=+1
 * where mij = f(dy=i-1, dx=j-1).
 */
inline void QuadraticFit3x3(const double m[9],
                             double& b_, double& f_,
                             double& e_, double& c_, double& d_)
{
    const double m00=m[0], m01=m[1], m02=m[2];
    const double m10=m[3], m11=m[4], m12=m[5];
    const double m20=m[6], m21=m[7], m22=m[8];

    b_ = (m02 + m12 + m22 - m00 - m10 - m20) / 6.0;
    f_ = (m20 + m21 + m22 - m00 - m01 - m02) / 6.0;
    e_ = (m00 + m10 + m20 + m02 + m12 + m22
          - 2.0 * (m01 + m11 + m21)) / 6.0;
    c_ = (m00 + m01 + m02 + m20 + m21 + m22
          - 2.0 * (m10 + m11 + m12)) / 6.0;
    d_ = (m22 + m00 - m02 - m20) / 4.0;
}

/**
 * @brief Hessian eigensystem on magnitude surface → NMS + sub-pixel in one step.
 *
 * Computes the eigenvector (nx, ny) of H = [[2e, d], [d, 2c]] corresponding to
 * the largest absolute eigenvalue λ_n (= edge normal direction on the magnitude
 * ridge), then finds the 1D extremum along that direction:
 *   t = -(b·nx + f·ny) / λ_n,  (sub_x, sub_y) = t·(nx, ny).
 *
 * Caller should accept the point only when:
 *   λ_n < 0                       (ridge = magnitude local max = NMS equivalent)
 *   |sub_x| ≤ 0.5 AND |sub_y| ≤ 0.5  (extremum within ½ pixel, L∞ norm)
 *
 * @param[out] lambda_n  Largest-|eigenvalue| of the Hessian.
 * @return false if |λ_n| < 1e-12 (degenerate patch).
 */
inline bool StegerSubPixel(double b_, double f_,
                            double e_, double c_, double d_,
                            double& sub_x, double& sub_y,
                            double& lambda_n,
                            double& nx_out, double& ny_out)
{
    const double fxx = 2.0 * e_;
    const double fxy = d_;
    const double fyy = 2.0 * c_;

    const double th   = 0.5 * (fxx + fyy);
    const double dh   = 0.5 * (fxx - fyy);
    const double disc = std::sqrt(dh * dh + fxy * fxy);
    const double lam1 = th + disc;
    const double lam2 = th - disc;

    lambda_n = (std::abs(lam1) >= std::abs(lam2)) ? lam1 : lam2;
    if (std::abs(lambda_n) < 1e-12) return false;

    double nx, ny;
    if (std::abs(fxy) > 1e-10) {
        const double vx  = fxy;
        const double vy  = lambda_n - fxx;
        const double len = std::sqrt(vx * vx + vy * vy);
        if (len < 1e-12) return false;
        nx = vx / len;
        ny = vy / len;
    } else {
        nx = (std::abs(fxx) >= std::abs(fyy)) ? 1.0 : 0.0;
        ny = 1.0 - std::abs(nx);
    }

    // Decompiled uses n^T H n (recomputed from quadratic coefficients) as the
    // sub-pixel divisor instead of lambda_n directly.  Mathematically equivalent
    // for an exact eigenvector, but more self-consistent in floating point.
    const double curvature = fxx * nx * nx + 2.0 * fxy * nx * ny + fyy * ny * ny;
    if (curvature == 0.0) return false;

    const double t = -(b_ * nx + f_ * ny) / curvature;
    sub_x = t * nx;
    sub_y = t * ny;
    nx_out = nx;
    ny_out = ny;
    return true;
}

struct Candidate {
    int32_t pixelX = 0;
    int32_t pixelY = 0;
    double subX = 0.0;
    double subY = 0.0;
    double angle = 0.0;       // eigenvector angle (edge normal), [0, 2π)
    double angle90 = 0.0;     // tangent direction = (angle mod π) + π/2, for neighbor generation
    double magnitude = 0.0;
    int32_t angleBin = 0;     // 4-bin tangent quantization
    int8_t flag = 0;  // 2 = strong (mag >= high), 1 = weak (mag >= low)
};

inline double WrapAngleDiff(double a, double b) {
    double d = std::fabs(a - b);
    if (d > M_PI) d = 2.0 * M_PI - d;
    return d;
}

inline int32_t Reflect101Index(int32_t v, int32_t limit) {
    if (limit <= 1) return 0;
    while (v < 0 || v >= limit) {
        if (v < 0) {
            v = -v;
        } else {
            v = 2 * limit - v - 2;
        }
    }
    return v;
}

} // anonymous namespace

int32_t EdgesSubPixGrayLegacyCore(
    const float* src, int32_t width, int32_t height,
    int32_t gaussianKernel,
    int8_t edgeType,
    int8_t linkMode,
    double highThreshold,
    double lowThreshold,
    std::vector<EdgePoint>& out,
    const uint8_t* mask,
    int32_t maskStride);

// =============================================================================
// EdgesSubPixGray
// =============================================================================

std::vector<EdgePoint> EdgesSubPixGray(
    const float* src, int32_t width, int32_t height,
    double highThreshold,
    double lowThreshold,
    double smoothSigma,
    const uint8_t* mask,
    int32_t maskStride,
    int32_t angleBins,
    double contrastScale)
{
    int32_t gaussianKernel = 0;
    if (smoothSigma > 0.0) {
        gaussianKernel = static_cast<int32_t>(std::ceil(6.0 * smoothSigma));
        if ((gaussianKernel & 1) == 0) ++gaussianKernel;
        gaussianKernel = std::max(3, gaussianKernel);
    }

    // Apply per-level contrast scale to thresholds (decompiled a8, typically 1.0)
    double scaledHigh = highThreshold * contrastScale;
    double scaledLow  = lowThreshold  * contrastScale;

    // angleBins: passed for API alignment with decompiled signature.
    // Internally angle quantization is fixed to 4 bins over [0, π).
    (void)angleBins;

    std::vector<EdgePoint> result;
    const int32_t rc = EdgesSubPixGrayLegacyCore(
        src, width, height,
        gaussianKernel,
        0,  // edgeType: both
        1,  // linkMode: orientation-aware
        scaledHigh, scaledLow,
        result,
        mask, maskStride);

    if (rc != 0) return {};
    return result;
}

std::vector<EdgePoint> EdgesSubPixGray(
    const float* src, int32_t width, int32_t height,
    double highThreshold,
    double lowThreshold,
    double smoothSigma,
    const uint8_t* mask,
    int32_t maskStride,
    int32_t angleBins,
    double contrastScale,
    int32_t* outStatus)
{
    int32_t gaussianKernel = 0;
    if (smoothSigma > 0.0) {
        gaussianKernel = static_cast<int32_t>(std::ceil(6.0 * smoothSigma));
        if ((gaussianKernel & 1) == 0) ++gaussianKernel;
        gaussianKernel = std::max(3, gaussianKernel);
    }

    double scaledHigh = highThreshold * contrastScale;
    double scaledLow  = lowThreshold  * contrastScale;
    (void)angleBins;

    std::vector<EdgePoint> result;
    const int32_t rc = EdgesSubPixGrayLegacyCore(
        src, width, height,
        gaussianKernel,
        0, 1,
        scaledHigh, scaledLow,
        result,
        mask, maskStride);

    if (outStatus) *outStatus = rc;
    if (rc != 0) return {};
    return result;
}

// =============================================================================
// EdgesSubPixGrayLegacyCore
// =============================================================================

int32_t EdgesSubPixGrayLegacyCore(
    const float* src, int32_t width, int32_t height,
    int32_t gaussianKernel,
    int8_t edgeType,
    int8_t linkMode,
    double highThreshold,
    double lowThreshold,
    std::vector<EdgePoint>& out,
    const uint8_t* mask,
    int32_t maskStride)
{
    out.clear();
    if (!src || width < 3 || height < 3 || highThreshold <= 0.0) {
        return -1;
    }

    if (lowThreshold <= 0.0 || lowThreshold > highThreshold) {
        lowThreshold = highThreshold * 0.5;
    }
    if (mask && maskStride <= 0) {
        maskStride = width;
    }

    constexpr int32_t MAX_POINTS = 65536;
    const double LINK_ANGLE_RAD = 67.5 * M_PI / 180.0;

    const int32_t N = width * height;

    // =========================================================================
    // Subsystem A: Sobel gradient field → magnitude terrain
    // =========================================================================

    // A1) Gaussian pre-smoothing (optional)
    std::vector<float> blurred(N);
    if (gaussianKernel > 0) {
        int32_t k = std::max(3, gaussianKernel | 1);
        GaussianBlurFixed<float, float>(src, blurred.data(),
                                        width, height,
                                        k, 0.0,
                                        BorderMode::Reflect101);
    } else {
        std::copy(src, src + N, blurred.data());
    }

    // A2) Sobel 3×3 gradient, legacy scale 0.25
    std::vector<float> gx(N), gy(N);
    Gradient<float, float>(blurred.data(), gx.data(), gy.data(),
                           width, height,
                           GradientOperator::Sobel3x3,
                           BorderMode::Reflect101);
    for (int32_t i = 0; i < N; ++i) {
        gx[i] *= 0.25f;
        gy[i] *= 0.25f;
    }

    // A3) Magnitude terrain (no direction array, no NMS pass)
    std::vector<float> mag(N);
    for (int32_t i = 0; i < N; ++i) {
        mag[i] = std::sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
    }

    // =========================================================================
    // Subsystem B: Facet Model + Hessian → NMS + sub-pixel + dual-threshold
    // All three done per pixel in one pass. No separate NMS step needed:
    //   λ_n < 0 ≡ magnitude ridge ≡ gradient-direction NMS.
    // =========================================================================

    std::vector<Candidate> candidates;
    candidates.reserve(1024);
    std::vector<int32_t> pixelToCand(N, -1);  // pixel → winning candidate index

    for (int32_t y = 1; y < height - 1; ++y) {
        for (int32_t x = 1; x < width - 1; ++x) {
            const int32_t idx = y * width + x;

            if (mask && mask[y * maskStride + x] == 0) continue;

            const double magVal = mag[idx];
            if (magVal < lowThreshold) continue;

            // B1) 3×3 magnitude window → quadratic surface
            double win[9];
            win[0] = mag[(y-1)*width + (x-1)];
            win[1] = mag[(y-1)*width +  x   ];
            win[2] = mag[(y-1)*width + (x+1)];
            win[3] = mag[ y   *width + (x-1)];
            win[4] = mag[ y   *width +  x   ];
            win[5] = mag[ y   *width + (x+1)];
            win[6] = mag[(y+1)*width + (x-1)];
            win[7] = mag[(y+1)*width +  x   ];
            win[8] = mag[(y+1)*width + (x+1)];

            double b_, f_, e_, c_, d_;
            QuadraticFit3x3(win, b_, f_, e_, c_, d_);

            // B2) Hessian eigensystem → sub-pixel offset
            double sub_x, sub_y, lambda_n, nx, ny;
            if (!StegerSubPixel(b_, f_, e_, c_, d_, sub_x, sub_y, lambda_n, nx, ny))
                continue;

            // B3) λ_n < 0: magnitude ridge (≡ NMS along normal direction)
            if (lambda_n >= 0.0) continue;

            // B3.5) Gradient-eigenvector alignment pre-check (decompiled filter):
            //   |dot(eigvec, gradient)| / magnitude must exceed 0.5, otherwise the
            //   Hessian eigenvector is unreliable at this pixel.
            {
                const double gradDot = nx * gx[idx] + ny * gy[idx];
                if (std::fabs(gradDot) <= 0.5 * magVal) continue;
            }

            // B4) Per-axis |sub| ≤ 0.5: extremum within half a pixel (L∞ norm)
            if (std::fabs(sub_x) > 0.5 || std::fabs(sub_y) > 0.5) continue;

            // B5) Polarity filter (optional)
            if (edgeType != 0) {
                const double proj = sub_x * gx[idx] + sub_y * gy[idx];
                if (edgeType == 1 && proj <= 0.0) continue;
                if (edgeType == 2 && proj >= 0.0) continue;
            }

            // B6) Angle from Hessian eigenvector, adjusted by gradient consistency
            //     Step 1: eigenvector angle = atan2(ny, nx), wrap to [0, 2π)
            double eigAngle = std::atan2(ny, nx);
            if (eigAngle < 0.0) eigAngle += 2.0 * M_PI;

            //     Step 2: gradient angle = atan2(gy, gx), wrap to [0, 2π)
            double gradAngle = std::atan2(static_cast<double>(gy[idx]),
                                          static_cast<double>(gx[idx]));
            if (gradAngle < 0.0) gradAngle += 2.0 * M_PI;

            //     Step 3: if eigenvector and gradient point in opposite hemispheres,
            //             flip eigenvector by adding π
            double angleDiff = std::fabs(eigAngle - gradAngle);
            double cosCheck = std::cos(angleDiff);
            if (cosCheck < 0.0) {
                eigAngle += M_PI;
            }

            //     Step 4: wrap angle to [0, 2π)
            if (eigAngle >= 2.0 * M_PI) eigAngle -= 2.0 * M_PI;

            double angle = eigAngle;

            //     Step 5: angle_90 = angle mod π  (edge normal in [0, π))
            double angle_90 = angle;
            if (angle_90 >= M_PI) angle_90 -= M_PI;

            //     Step 6: tangent = angle_90 + π/2  (edge tangent direction)
            double tangent = angle_90 + M_PI * 0.5;
            if (tangent >= 2.0 * M_PI) tangent -= 2.0 * M_PI;

            //     Step 7: tangent_mod_pi = tangent mod π (tangent in [0, π))
            double tangent_mod_pi = tangent;
            if (tangent_mod_pi >= M_PI) tangent_mod_pi -= M_PI;

            //     Step 8: angleBin = floor(tangent_mod_pi * 4/π + 0.5) % 4
            int32_t angleBin = static_cast<int32_t>(
                std::floor(tangent_mod_pi * (4.0 / M_PI) + 0.5)) % 4;
            if (angleBin < 0) angleBin += 4;

            // B7) Dual-threshold flag: 2 = strong, 1 = weak
            Candidate cnd;
            cnd.pixelX = x;
            cnd.pixelY = y;
            cnd.subX = static_cast<double>(x) + sub_x;
            cnd.subY = static_cast<double>(y) + sub_y;
            cnd.angle = angle;
            cnd.angle90 = tangent;  // tangent direction for neighbor generation
            cnd.magnitude = magVal;
            cnd.angleBin = angleBin;
            cnd.flag = (magVal >= highThreshold) ? 2 : 1;

            const int32_t cndIdx = static_cast<int32_t>(candidates.size());
            candidates.push_back(cnd);

            const int32_t prev = pixelToCand[idx];
            if (prev < 0 || candidates[prev].magnitude < cnd.magnitude) {
                pixelToCand[idx] = cndIdx;
            }
        }
    }

    // Deduplicate: keep only the winning candidate per pixel.
    std::vector<int32_t> active;
    active.reserve(candidates.size());
    {
        std::vector<int8_t> seen(candidates.size(), 0);
        for (int32_t idx : pixelToCand) {
            if (idx < 0 || seen[idx]) continue;
            seen[idx] = 1;
            active.push_back(idx);
        }
    }

    // =========================================================================
    // Subsystem C: Greedy chain linking (direction-aware)
    //   Sort strong seeds by magnitude descending.
    //   Seeds = flag==2 (strong). Each seed starts a chain, extending
    //   bidirectionally along tangent, accepting flag≥1 neighbors with
    //   angle consistency (< 67.5°).
    // =========================================================================

    // Sort all active candidates by magnitude (deterministic traversal order).
    std::sort(active.begin(), active.end(), [&](int32_t a, int32_t b) {
        return candidates[a].magnitude > candidates[b].magnitude;
    });

    // Decompiled behavior applies the 65536 guard to strong seeds only.
    std::vector<int32_t> strongSeeds;
    strongSeeds.reserve(active.size());
    for (int32_t rawIdx : active) {
        if (candidates[rawIdx].flag > 1) {
            strongSeeds.push_back(rawIdx);
        }
    }
    if (static_cast<int32_t>(strongSeeds.size()) > MAX_POINTS) {
        out.clear();
        return -2;
    }

    // Strong seeds are globally consumed; weak points are only chain-local.
    std::vector<uint8_t> strongUsed(candidates.size(), 0);
    std::vector<uint32_t> chainVisitStamp(candidates.size(), 0);
    uint32_t stamp = 1;
    std::vector<uint8_t> strongSeedAllowed(N, 0);
    for (int32_t rawIdx : strongSeeds) {
        const Candidate& c = candidates[rawIdx];
        strongSeedAllowed[c.pixelY * width + c.pixelX] = 1;
    }

    // Direction-based neighbor generation: compute 3-4 candidate pixel offsets
    // along the tangent direction using quadrant-aware asymmetric neighbors.
    // Matches the decompiled sub_18002E940 logic with:
    //   - Diagonal: primary + two axis-aligned secondaries (symmetric)
    //   - Horizontal: primary + two neighbors leaning toward sin direction,
    //     plus optional proximity neighbor in opposite direction
    //   - Vertical: primary + two neighbors leaning toward cos direction,
    //     plus optional proximity neighbor in opposite direction
    // tangentAngle: the edge tangent direction in radians
    // sign: +1 for forward, -1 for backward (flip tangent direction)
    auto generateDirectionalNeighbors =
        [](int32_t pixelX, int32_t pixelY, double subX, double subY,
           double tangentAngle, int32_t sign,
           int32_t neighbors[][2]) -> int32_t
    {
        // If backward, reverse the tangent direction
        double dir = tangentAngle;
        if (sign < 0) dir += M_PI;
        const double dx = std::cos(dir);
        const double dy = std::sin(dir);

        // Fractional offset from pixel center
        const double fracX = subX - pixelX;
        const double fracY = subY - pixelY;

        // Effective displacement including fractional position
        const double nx = dx + fracX;
        const double ny = dy + fracY;

        // Round half away from zero: sign(v) * floor(|v| + 0.5)
        auto roundHalfAwayFromZero = [](double v) -> int32_t {
            return (v >= 0.0)
                ? static_cast<int32_t>(std::floor(v + 0.5))
                : -static_cast<int32_t>(std::floor(-v + 0.5));
        };

        int32_t ix = roundHalfAwayFromZero(nx);
        int32_t iy = roundHalfAwayFromZero(ny);

        // If both offsets are zero (very small step), double the step
        if (ix == 0 && iy == 0) {
            ix = roundHalfAwayFromZero(nx + dx);
            iy = roundHalfAwayFromZero(ny + dy);
        }

        int32_t count = 0;

        // Primary candidate
        neighbors[count][0] = ix;
        neighbors[count][1] = iy;
        ++count;

        if (ix != 0 && iy != 0) {
            // Diagonal step: add the two axis-aligned neighbors (symmetric)
            neighbors[count][0] = 0;
            neighbors[count][1] = iy;
            ++count;
            neighbors[count][0] = ix;
            neighbors[count][1] = 0;
            ++count;
        } else if (ix == 0) {
            // Vertical step: lean toward cos direction (sign of dx)
            const int32_t lean_x = (dx >= 0.0) ? 1 : -1;
            neighbors[count][0] = lean_x;
            neighbors[count][1] = iy;
            ++count;
            neighbors[count][0] = lean_x;
            neighbors[count][1] = 0;
            ++count;
            // Proximity check: if sub-pixel position is close to opposite edge
            if (lean_x * nx < -0.25) {
                neighbors[count][0] = -lean_x;
                neighbors[count][1] = iy;
                ++count;
            }
        } else {
            // Horizontal step: lean toward sin direction (sign of dy)
            const int32_t lean_y = (dy >= 0.0) ? 1 : -1;
            neighbors[count][0] = ix;
            neighbors[count][1] = lean_y;
            ++count;
            neighbors[count][0] = 0;
            neighbors[count][1] = lean_y;
            ++count;
            // Proximity check: if sub-pixel position is close to opposite edge
            if (lean_y * ny < -0.25) {
                neighbors[count][0] = ix;
                neighbors[count][1] = -lean_y;
                ++count;
            }
        }

        return count;
    };

    // Find best directional neighbor along the given tangent side.
    // sign: +1 for forward (angle + π/2), -1 for backward (angle - π/2)
    //
    // Decompiled flow:
    //   1. Generate 3-4 neighbor offsets via sub_18002E940
    //   2. For each valid neighbor, compute (cosDiff, angleDeg, distMetric, cost)
    //      and push into parallel arrays
    //   3. While count > 2: iteratively remove the entry with max cost
    //   4. Single candidate: accept if angleDeg < 67.5
    //   5. Two candidates: if direction angle < 5°, use distance; else use angleDeg
    constexpr double COS_5DEG = 0.99619469809;  // cos(5°)

    auto bestDirectionalNeighbor =
        [&](int32_t currentRawIdx,
            int32_t /* prevRawIdx (unused in decompiled) */,
            int32_t sign,
            uint32_t currentStamp) -> int32_t
    {
        const Candidate& cur = candidates[currentRawIdx];

        int32_t neighborOffsets[4][2];
        const int32_t numNeighbors = generateDirectionalNeighbors(
            cur.pixelX, cur.pixelY, cur.subX, cur.subY, cur.angle90, sign, neighborOffsets);

        // Parallel arrays matching decompiled v382/v389/v387/v384/v373
        struct CandEntry {
            int32_t raw;
            float cosDiff;    // v382: cos(angle_diff)
            float angleDeg;   // v389: |angle_diff| in degrees
            float distMetric; // v387: distance/alignment metric
            float cost;       // v384: pruning cost (higher = worse)
        };
        CandEntry entries[8];
        int32_t numEntries = 0;

        for (int32_t n = 0; n < numNeighbors; ++n) {
            const int32_t nx = cur.pixelX + neighborOffsets[n][0];
            const int32_t ny = cur.pixelY + neighborOffsets[n][1];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

            const int32_t raw = pixelToCand[ny * width + nx];
            if (raw < 0) continue;
            if (chainVisitStamp[raw] == currentStamp) continue;
            if (candidates[raw].flag <= 0) continue;

            // Angle difference
            double ad = 0.0;
            double cd = 1.0;
            if (linkMode != 0) {
                ad = WrapAngleDiff(cur.angle, candidates[raw].angle);
                cd = std::cos(ad);
                if (cd < 0.0) continue;       // > 90°: reject
                if (ad > LINK_ANGLE_RAD) continue;  // > 67.5°: reject
            }

            // Tangent forward-consistency
            const double tx = std::cos(cur.angle90 + (sign < 0 ? M_PI : 0.0));
            const double ty = std::sin(cur.angle90 + (sign < 0 ? M_PI : 0.0));
            const double dx = candidates[raw].subX - cur.subX;
            const double dy = candidates[raw].subY - cur.subY;
            const double len = std::sqrt(dx * dx + dy * dy);
            if (len <= 1e-12) continue;
            const double align = (dx * tx + dy * ty) / len;
            if (align <= 0.0) continue;

            const float adDeg = static_cast<float>(ad * 180.0 / M_PI);
            // Cost: angle deviation + inverse alignment penalty
            const float cost = static_cast<float>(ad + (1.0 - align) * 0.25);

            if (numEntries < 8) {
                entries[numEntries++] = {
                    raw,
                    static_cast<float>(cd),
                    adDeg,
                    static_cast<float>(len),
                    cost
                };
            }
        }

        if (numEntries == 0) return -1;

        // Iterative pruning: remove max-cost entry until <= 2 remain
        while (numEntries > 2) {
            int32_t worstIdx = 0;
            for (int32_t i = 1; i < numEntries; ++i) {
                if (entries[i].cost > entries[worstIdx].cost) {
                    worstIdx = i;
                }
            }
            // Remove by shifting
            for (int32_t i = worstIdx; i < numEntries - 1; ++i) {
                entries[i] = entries[i + 1];
            }
            --numEntries;
        }

        if (numEntries == 1) {
            // Single candidate: check |cos| sanity + angle < 67.5°
            if (std::fabs(entries[0].cosDiff) > 1.00001f) return -1;
            if (entries[0].angleDeg >= 67.5f) return -1;
            return entries[0].raw;
        }

        // Two candidates: disambiguate
        // Compute direction cosine between the two candidate displacement vectors
        const double dx0 = candidates[entries[0].raw].subX - cur.subX;
        const double dy0 = candidates[entries[0].raw].subY - cur.subY;
        const double dx1 = candidates[entries[1].raw].subX - cur.subX;
        const double dy1 = candidates[entries[1].raw].subY - cur.subY;
        const double len0 = std::sqrt(dx0 * dx0 + dy0 * dy0);
        const double len1 = std::sqrt(dx1 * dx1 + dy1 * dy1);

        int32_t chosen = 0;
        if (len0 > 1e-12 && len1 > 1e-12) {
            const double dot = dx0 * dx1 + dy0 * dy1;
            const double cosDir = dot / (len0 * len1);

            if (cosDir >= COS_5DEG) {
                // Near-parallel candidates (< 5°): use distMetric (shorter = better)
                chosen = (entries[1].distMetric < entries[0].distMetric) ? 1 : 0;
            } else {
                // Divergent candidates: pick smaller angle difference
                chosen = (entries[1].angleDeg < entries[0].angleDeg) ? 1 : 0;
            }
        }

        // Final 67.5° gate on the chosen candidate
        if (entries[chosen].angleDeg >= 67.5f) return -1;
        return entries[chosen].raw;
    };

    // Decompiled: 3×3 grid suppression centered on the sub-pixel position.
    // For the most recently added chain point, clear strongSeedAllowed in
    // a 3×3 pixel neighborhood using round-to-nearest + Reflect101.
    auto suppressStrongNeighborhood =
        [&](int32_t curRaw) {
            const Candidate& cur = candidates[curRaw];
            for (int32_t dy = -1; dy <= 1; ++dy) {
                for (int32_t dx = -1; dx <= 1; ++dx) {
                    int32_t sy = static_cast<int32_t>(std::trunc(cur.subY + dy + 0.5));
                    int32_t sx = static_cast<int32_t>(std::trunc(cur.subX + dx + 0.5));
                    sy = Reflect101Index(sy, height);
                    sx = Reflect101Index(sx, width);
                    strongSeedAllowed[sy * width + sx] = 0;
                }
            }
        };

    // Grow chains from strong seeds (flag==2), sorted strongest-first.
    // Weak points without a strong neighbor remain unchained (dropped).
    for (int32_t seedRaw : strongSeeds) {
        if (strongUsed[seedRaw]) continue;
        const Candidate& seed = candidates[seedRaw];
        if (strongSeedAllowed[seed.pixelY * width + seed.pixelX] == 0) continue;

        if (++stamp == 0) {
            std::fill(chainVisitStamp.begin(), chainVisitStamp.end(), 0);
            stamp = 1;
        }

        std::deque<int32_t> chain;
        chain.push_back(seedRaw);
        chainVisitStamp[seedRaw] = stamp;
        strongUsed[seedRaw] = 1;

        // Forward extension along tangent direction (angle + π/2)
        int32_t prevFwdRaw = -1;
        for (int32_t curRaw = seedRaw;;) {
            const int32_t nextRaw = bestDirectionalNeighbor(curRaw, prevFwdRaw, +1, stamp);
            if (nextRaw < 0) break;
            chainVisitStamp[nextRaw] = stamp;
            chain.push_back(nextRaw);
            if (candidates[nextRaw].flag > 1) {
                strongUsed[nextRaw] = 1;
            }
            if (chain.size() > 2) {
                suppressStrongNeighborhood(nextRaw);
            }
            prevFwdRaw = curRaw;
            curRaw = nextRaw;
        }

        // Backward extension along opposite tangent (angle - π/2)
        int32_t prevBwdRaw = -1;
        for (int32_t curRaw = seedRaw;;) {
            const int32_t nextRaw = bestDirectionalNeighbor(curRaw, prevBwdRaw, -1, stamp);
            if (nextRaw < 0) break;
            chainVisitStamp[nextRaw] = stamp;
            chain.push_front(nextRaw);
            if (candidates[nextRaw].flag > 1) {
                strongUsed[nextRaw] = 1;
            }
            if (chain.size() > 2) {
                suppressStrongNeighborhood(nextRaw);
            }
            prevBwdRaw = curRaw;
            curRaw = nextRaw;
        }

        for (int32_t rawIdx : chain) {
            const Candidate& c = candidates[rawIdx];
            out.emplace_back(c.subX, c.subY, c.angle, c.magnitude, c.angleBin);
        }
    }

    // Final sort by magnitude descending
    std::sort(out.begin(), out.end(),
              [](const EdgePoint& a, const EdgePoint& b) {
                  return a.magnitude > b.magnitude;
              });

    return 0;
}

} // namespace Qi::Vision::Internal
