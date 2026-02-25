/**
 * @file EdgesSubPix.cpp
 * @brief Sub-pixel edge detection — HALCON-compatible edges_sub_pix_gray
 *
 * Algorithm reference:
 *   Steger, C. (1998). An Unbiased Detector of Curvilinear Structures.
 *   IEEE TPAMI 20(2): 113-125.
 *   + 2D quadratic surface fitting as used in HALCON edges_sub_pix_gray.
 *
 * Coordinate convention: x = column, y = row.
 * Angle convention:      atan2(Gy, Gx), wrapped to [0, 2π).
 */

#include <QiVision/Internal/EdgesSubPix.h>
#include <QiVision/Internal/Convolution.h>       // GaussianBlur<>
#include <QiVision/Internal/Gradient.h>           // Gradient<>
#include <QiVision/Internal/NonMaxSuppression.h>  // NMS2DGradient
#include <QiVision/Core/Types.h>                  // BorderMode

#include <cmath>
#include <cstddef>
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
 *
 * @param[in]  m   Row-major 3×3 array: m[row*3+col]
 * @param[out] b_  first-order x coefficient
 * @param[out] f_  first-order y coefficient
 * @param[out] e_  second-order x coefficient
 * @param[out] c_  second-order y coefficient
 * @param[out] d_  cross-term coefficient
 */
inline void QuadraticFit3x3(const double m[9],
                             double& b_, double& f_,
                             double& e_, double& c_, double& d_)
{
    // m[row*3+col], row/col ∈ {0,1,2}
    const double m00=m[0], m01=m[1], m02=m[2];
    const double m10=m[3], m11=m[4], m12=m[5];
    const double m20=m[6], m21=m[7], m22=m[8];

    // ∂f/∂x: right column sum − left column sum, divided by 6
    b_ = (m02 + m12 + m22 - m00 - m10 - m20) / 6.0;

    // ∂f/∂y: bottom row sum − top row sum, divided by 6
    f_ = (m20 + m21 + m22 - m00 - m01 - m02) / 6.0;

    // ∂²f/∂x²: left+right columns − 2×centre column, divided by 6
    e_ = (m00 + m10 + m20 + m02 + m12 + m22
          - 2.0 * (m01 + m11 + m21)) / 6.0;

    // ∂²f/∂y²: top+bottom rows − 2×centre row, divided by 6
    c_ = (m00 + m01 + m02 + m20 + m21 + m22
          - 2.0 * (m10 + m11 + m12)) / 6.0;

    // ∂²f/∂x∂y: diagonal differences, divided by 4
    d_ = (m22 + m00 - m02 - m20) / 4.0;
}

/**
 * @brief Find the extremum of the 2D quadratic surface.
 *
 * Solves:  [2e  d] [sub_x]   [-b ]
 *          [d  2c] [sub_y] = [-f_]
 *
 * Returns false if the system is (near-)singular.
 */
inline bool QuadraticExtremum(double b_, double f_,
                               double e_, double c_, double d_,
                               double& sub_x, double& sub_y)
{
    const double det = 4.0 * c_ * e_ - d_ * d_;
    if (std::abs(det) < 1e-10) return false;

    sub_x = -(2.0 * c_ * b_ - d_ * f_) / det;
    sub_y = -(2.0 * e_ * f_ - d_ * b_) / det;
    return true;
}

} // anonymous namespace

// =============================================================================
// EdgesSubPixGray
// =============================================================================

std::vector<EdgePoint> EdgesSubPixGray(
    const float* src, int32_t width, int32_t height,
    double highThreshold,
    double lowThreshold,
    double smoothSigma,
    int32_t numAngleBins)
{
    if (!src || width < 3 || height < 3 || highThreshold <= 0.0)
        return {};

    if (numAngleBins < 1) numAngleBins = 16;
    if (lowThreshold <= 0.0 || lowThreshold > highThreshold)
        lowThreshold = highThreshold * 0.5;

    const int32_t N = width * height;

    // -------------------------------------------------------------------------
    // 1. Gaussian pre-smoothing
    // -------------------------------------------------------------------------
    std::vector<float> blurred(N);
    if (smoothSigma > 0.0) {
        GaussianBlur<float, float>(src, blurred.data(),
                                   width, height,
                                   smoothSigma,
                                   smoothSigma,
                                   BorderMode::Reflect101);
    } else {
        std::copy(src, src + N, blurred.data());
    }

    // -------------------------------------------------------------------------
    // 2. Sobel gradient  Gx, Gy  (float)
    // -------------------------------------------------------------------------
    std::vector<float> gx(N), gy(N);
    Gradient<float, float>(blurred.data(), gx.data(), gy.data(),
                           width, height,
                           GradientOperator::Sobel3x3,
                           BorderMode::Reflect101);

    // -------------------------------------------------------------------------
    // 3. Gradient magnitude and direction
    // -------------------------------------------------------------------------
    std::vector<float> mag(N), dir(N);
    for (int32_t i = 0; i < N; ++i) {
        mag[i] = std::sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
        dir[i] = std::atan2(gy[i], gx[i]);   // [-π, π]
    }

    // -------------------------------------------------------------------------
    // 4. Gradient-direction NMS (thin edges to 1-pixel width)
    // -------------------------------------------------------------------------
    std::vector<float> nms(N, 0.0f);
    NMS2DGradient(mag.data(), dir.data(), nms.data(),
                  width, height,
                  static_cast<float>(lowThreshold));

    // -------------------------------------------------------------------------
    // 5. Per-pixel: quadratic fitting + projection + convexity + threshold
    // -------------------------------------------------------------------------
    std::vector<EdgePoint> result;
    result.reserve(512);

    const double angleStep = (2.0 * M_PI) / numAngleBins;

    for (int32_t y = 1; y < height - 1; ++y) {
        for (int32_t x = 1; x < width - 1; ++x) {

            // Skip NMS-suppressed pixels
            if (nms[y * width + x] == 0.0f) continue;

            const double magVal = mag[y * width + x];
            if (magVal < lowThreshold) continue;

            // -----------------------------------------------------------------
            // 5a. Build 3×3 magnitude window
            // -----------------------------------------------------------------
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

            // -----------------------------------------------------------------
            // 5b. Quadratic surface fit + find extremum (sub_x, sub_y)
            // -----------------------------------------------------------------
            double b_, f_, e_, c_, d_;
            QuadraticFit3x3(win, b_, f_, e_, c_, d_);

            double sub_x, sub_y;
            if (!QuadraticExtremum(b_, f_, e_, c_, d_, sub_x, sub_y))
                continue;

            // Reject if offset is unreasonably large
            if (std::abs(sub_x) > 1.5 || std::abs(sub_y) > 1.5) continue;

            // -----------------------------------------------------------------
            // 5c. Projection check
            //     proj = (sub_x·Gx + sub_y·Gy) / mag
            //     The sub-pixel offset must be primarily in the gradient dir.
            //     Reject if |proj| < 0.5.
            // -----------------------------------------------------------------
            const double gxVal = gx[y * width + x];
            const double gyVal = gy[y * width + x];
            const double proj  = (sub_x * gxVal + sub_y * gyVal) / magVal;
            if (std::abs(proj) < 0.5) continue;

            // -----------------------------------------------------------------
            // 5d. Convexity check
            //     The surface must be concave (true maximum) in gradient dir.
            //     2e·cos²θ + 2d·cosθ·sinθ + 2c·sin²θ < 0
            // -----------------------------------------------------------------
            const double cosT = gxVal / magVal;
            const double sinT = gyVal / magVal;
            const double d2   = 2.0 * e_ * cosT * cosT
                              + 2.0 * d_ * cosT * sinT
                              + 2.0 * c_ * sinT * sinT;
            if (d2 >= 0.0) continue;

            // -----------------------------------------------------------------
            // 5e. High-threshold filter
            // -----------------------------------------------------------------
            if (magVal < highThreshold) continue;

            // -----------------------------------------------------------------
            // 5f. Compute angle and angle bin, store result
            // -----------------------------------------------------------------
            double angle = std::atan2(gyVal, gxVal);
            if (angle < 0.0) angle += 2.0 * M_PI;

            const int32_t bin = static_cast<int32_t>(angle / angleStep)
                                % numAngleBins;

            result.emplace_back(x + sub_x, y + sub_y, angle, magVal, bin);
        }
    }

    return result;
}

} // namespace Qi::Vision::Internal
