/**
 * @file test_hessian.cpp
 * @brief Unit tests for Internal/Hessian.h
 */

#include <QiVision/Internal/Hessian.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

class HessianTest : public ::testing::Test {
protected:
    // Test image size
    int32_t width_ = 32;
    int32_t height_ = 32;

    // Constant image (all derivatives should be zero)
    std::vector<float> constImage_;

    // Quadratic image: f(x,y) = x^2 + y^2
    // First derivatives: fx = 2x, fy = 2y
    // Second derivatives: fxx = 2, fxy = 0, fyy = 2
    std::vector<float> quadImage_;

    // Linear image: f(x,y) = x + y
    // All second derivatives should be zero
    std::vector<float> linearImage_;

    // Ridge image: Gaussian ridge along y-axis
    // f(x,y) = exp(-(x-16)^2 / (2*sigma^2))
    std::vector<float> ridgeImage_;

    void SetUp() override {
        size_t size = static_cast<size_t>(width_ * height_);

        // Constant image
        constImage_.resize(size, 128.0f);

        // Quadratic image: (x-cx)^2 + (y-cy)^2
        double cx = width_ / 2.0;
        double cy = height_ / 2.0;
        quadImage_.resize(size);
        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                double dx = x - cx;
                double dy = y - cy;
                quadImage_[y * width_ + x] = static_cast<float>(dx * dx + dy * dy);
            }
        }

        // Linear image
        linearImage_.resize(size);
        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                linearImage_[y * width_ + x] = static_cast<float>(x + y);
            }
        }

        // Ridge image (Gaussian ridge along y-axis at x=16)
        double sigma = 3.0;
        double cx_ridge = width_ / 2.0;
        ridgeImage_.resize(size);
        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                double dx = x - cx_ridge;
                ridgeImage_[y * width_ + x] = static_cast<float>(
                    255.0 * std::exp(-dx * dx / (2.0 * sigma * sigma))
                );
            }
        }
    }
};

// ============================================================================
// Eigenvalue Decomposition Tests
// ============================================================================

TEST_F(HessianTest, EigenDecompose2x2_Diagonal) {
    // Diagonal matrix: [3 0; 0 1]
    double lambda1, lambda2, nx, ny;
    EigenDecompose2x2(3.0, 0.0, 1.0, lambda1, lambda2, nx, ny);

    // Eigenvalues should be 3 and 1 (sorted by abs)
    EXPECT_NEAR(lambda1, 3.0, 1e-10);
    EXPECT_NEAR(lambda2, 1.0, 1e-10);

    // Eigenvector for λ1=3 should be (1, 0)
    EXPECT_NEAR(std::abs(nx), 1.0, 1e-10);
    EXPECT_NEAR(ny, 0.0, 1e-10);
}

TEST_F(HessianTest, EigenDecompose2x2_Symmetric) {
    // Symmetric matrix: [2 1; 1 2]
    // Eigenvalues: 3, 1
    // Eigenvectors: (1,1)/sqrt(2), (-1,1)/sqrt(2)
    double lambda1, lambda2, nx, ny;
    EigenDecompose2x2(2.0, 1.0, 2.0, lambda1, lambda2, nx, ny);

    EXPECT_NEAR(lambda1, 3.0, 1e-10);
    EXPECT_NEAR(lambda2, 1.0, 1e-10);

    // Eigenvector should be normalized and along (1,1) direction
    double len = std::sqrt(nx * nx + ny * ny);
    EXPECT_NEAR(len, 1.0, 1e-10);
    EXPECT_NEAR(std::abs(nx), std::abs(ny), 1e-10);  // |nx| = |ny| for (1,1)
}

TEST_F(HessianTest, EigenDecompose2x2_NegativeEigenvalues) {
    // Matrix with negative eigenvalues: [-3 0; 0 -1]
    double lambda1, lambda2, nx, ny;
    EigenDecompose2x2(-3.0, 0.0, -1.0, lambda1, lambda2, nx, ny);

    // Sorted by absolute value: |-3| > |-1|
    EXPECT_NEAR(lambda1, -3.0, 1e-10);
    EXPECT_NEAR(lambda2, -1.0, 1e-10);
}

TEST_F(HessianTest, EigenDecompose2x2Full_Orthogonal) {
    // Test that eigenvectors are orthogonal
    double lambda1, lambda2, nx, ny, tx, ty;
    EigenDecompose2x2Full(2.0, 1.0, 2.0, lambda1, lambda2, nx, ny, tx, ty);

    // Eigenvectors should be orthogonal: n · t = 0
    double dot = nx * tx + ny * ty;
    EXPECT_NEAR(dot, 0.0, 1e-10);

    // Both should be normalized
    EXPECT_NEAR(nx * nx + ny * ny, 1.0, 1e-10);
    EXPECT_NEAR(tx * tx + ty * ty, 1.0, 1e-10);
}

// ============================================================================
// Hessian Result Tests
// ============================================================================

TEST_F(HessianTest, HessianResult_IsRidge) {
    HessianResult r;
    r.lambda1 = -5.0;
    r.lambda2 = -0.1;
    EXPECT_TRUE(r.IsRidge());
    EXPECT_FALSE(r.IsValley());
}

TEST_F(HessianTest, HessianResult_IsValley) {
    HessianResult r;
    r.lambda1 = 5.0;
    r.lambda2 = 0.1;
    EXPECT_TRUE(r.IsValley());
    EXPECT_FALSE(r.IsRidge());
}

TEST_F(HessianTest, HessianResult_Response) {
    HessianResult r;
    r.lambda1 = -5.0;
    EXPECT_DOUBLE_EQ(r.Response(), 5.0);

    r.lambda1 = 3.0;
    EXPECT_DOUBLE_EQ(r.Response(), 3.0);
}

TEST_F(HessianTest, HessianResult_Anisotropy) {
    HessianResult r;
    r.lambda1 = -10.0;
    r.lambda2 = -2.0;
    EXPECT_DOUBLE_EQ(r.Anisotropy(), 5.0);  // |10| / |2| = 5
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST_F(HessianTest, HessianDeterminant) {
    // det([2 1; 1 3]) = 2*3 - 1*1 = 5
    EXPECT_DOUBLE_EQ(HessianDeterminant(2.0, 1.0, 3.0), 5.0);

    // det([4 0; 0 4]) = 16
    EXPECT_DOUBLE_EQ(HessianDeterminant(4.0, 0.0, 4.0), 16.0);
}

TEST_F(HessianTest, HessianTrace) {
    // trace([2 x; x 3]) = 5
    EXPECT_DOUBLE_EQ(HessianTrace(2.0, 3.0), 5.0);
}

TEST_F(HessianTest, HessianNorm) {
    // ||[1 0; 0 1]||_F = sqrt(1 + 0 + 1) = sqrt(2)
    EXPECT_NEAR(HessianNorm(1.0, 0.0, 1.0), std::sqrt(2.0), 1e-10);

    // ||[1 1; 1 1]||_F = sqrt(1 + 2*1 + 1) = 2
    EXPECT_NEAR(HessianNorm(1.0, 1.0, 1.0), 2.0, 1e-10);
}

// ============================================================================
// Point Hessian Tests
// ============================================================================

TEST_F(HessianTest, ComputeHessianAt_ConstantImage) {
    // All second derivatives of constant should be ~0
    auto h = ComputeHessianAt(constImage_.data(), width_, height_,
                               width_ / 2, height_ / 2, 1.0);

    EXPECT_NEAR(h.dxx, 0.0, 1e-5);
    EXPECT_NEAR(h.dxy, 0.0, 1e-5);
    EXPECT_NEAR(h.dyy, 0.0, 1e-5);
}

TEST_F(HessianTest, ComputeHessianAt_LinearImage) {
    // All second derivatives of linear function should be ~0
    auto h = ComputeHessianAt(linearImage_.data(), width_, height_,
                               width_ / 2, height_ / 2, 1.0);

    // With some tolerance due to discretization
    EXPECT_NEAR(h.dxx, 0.0, 0.1);
    EXPECT_NEAR(h.dxy, 0.0, 0.1);
    EXPECT_NEAR(h.dyy, 0.0, 0.1);
}

TEST_F(HessianTest, ComputeHessianAt_QuadraticImage) {
    // For f = x^2 + y^2: fxx = 2, fyy = 2, fxy = 0
    // Due to Gaussian smoothing, values will be attenuated
    auto h = ComputeHessianAt(quadImage_.data(), width_, height_,
                               width_ / 2, height_ / 2, 1.0);

    // fxx and fyy should be approximately equal (isotropic)
    EXPECT_NEAR(h.dxx, h.dyy, 0.5);

    // fxy should be small
    EXPECT_NEAR(h.dxy, 0.0, 0.1);

    // Both should be positive (convex function)
    EXPECT_GT(h.dxx, 0.0);
    EXPECT_GT(h.dyy, 0.0);
}

TEST_F(HessianTest, ComputeHessianAt_RidgeImage) {
    // At the center of the ridge, we expect:
    // - Strong negative dxx (concave in x direction)
    // - Near zero dyy (flat along y direction)
    // - Near zero dxy
    auto h = ComputeHessianAt(ridgeImage_.data(), width_, height_,
                               width_ / 2, height_ / 2, 1.5);

    // Ridge should have negative principal eigenvalue
    EXPECT_TRUE(h.IsRidge());

    // Principal direction should be roughly along x-axis (perpendicular to ridge)
    // Normal vector should have |nx| > |ny|
    EXPECT_GT(std::abs(h.nx), std::abs(h.ny));
}

// ============================================================================
// Whole Image Hessian Tests
// ============================================================================

TEST_F(HessianTest, ComputeHessianImage_Size) {
    std::vector<float> dxx, dxy, dyy;
    ComputeHessianImage(constImage_.data(), width_, height_, 1.0, dxx, dxy, dyy);

    size_t expected = static_cast<size_t>(width_ * height_);
    EXPECT_EQ(dxx.size(), expected);
    EXPECT_EQ(dxy.size(), expected);
    EXPECT_EQ(dyy.size(), expected);
}

TEST_F(HessianTest, ComputeHessianImage_ConstantAllZero) {
    std::vector<float> dxx, dxy, dyy;
    ComputeHessianImage(constImage_.data(), width_, height_, 1.0, dxx, dxy, dyy);

    // All values should be near zero for constant image
    for (size_t i = 0; i < dxx.size(); ++i) {
        EXPECT_NEAR(dxx[i], 0.0f, 1e-5f);
        EXPECT_NEAR(dxy[i], 0.0f, 1e-5f);
        EXPECT_NEAR(dyy[i], 0.0f, 1e-5f);
    }
}

// ============================================================================
// Eigenvalue Image Tests
// ============================================================================

TEST_F(HessianTest, ComputeEigenvalueImages_Size) {
    std::vector<float> dxx, dxy, dyy;
    ComputeHessianImage(constImage_.data(), width_, height_, 1.0, dxx, dxy, dyy);

    std::vector<float> lambda1, lambda2;
    ComputeEigenvalueImages(dxx.data(), dxy.data(), dyy.data(),
                             width_, height_, lambda1, lambda2);

    size_t expected = static_cast<size_t>(width_ * height_);
    EXPECT_EQ(lambda1.size(), expected);
    EXPECT_EQ(lambda2.size(), expected);
}

TEST_F(HessianTest, ComputeEigenvalueImages_Ordering) {
    std::vector<float> dxx, dxy, dyy;
    ComputeHessianImage(quadImage_.data(), width_, height_, 1.0, dxx, dxy, dyy);

    std::vector<float> lambda1, lambda2;
    ComputeEigenvalueImages(dxx.data(), dxy.data(), dyy.data(),
                             width_, height_, lambda1, lambda2);

    // |lambda1| >= |lambda2| everywhere
    for (size_t i = 0; i < lambda1.size(); ++i) {
        EXPECT_GE(std::abs(lambda1[i]), std::abs(lambda2[i]));
    }
}

// ============================================================================
// Ridge/Valley Response Tests
// ============================================================================

TEST_F(HessianTest, ComputeRidgeResponse_Ridge) {
    std::vector<float> dxx, dxy, dyy;
    ComputeHessianImage(ridgeImage_.data(), width_, height_, 1.5, dxx, dxy, dyy);

    std::vector<float> lambda1, lambda2;
    ComputeEigenvalueImages(dxx.data(), dxy.data(), dyy.data(),
                             width_, height_, lambda1, lambda2);

    std::vector<float> response;
    ComputeRidgeResponse(lambda1.data(), width_, height_, response);

    // Ridge center should have positive response
    int32_t cx = width_ / 2;
    int32_t cy = height_ / 2;
    size_t centerIdx = static_cast<size_t>(cy * width_ + cx);
    EXPECT_GT(response[centerIdx], 0.0f);
}

TEST_F(HessianTest, ComputeValleyResponse_NonNegative) {
    std::vector<float> dxx, dxy, dyy;
    ComputeHessianImage(ridgeImage_.data(), width_, height_, 1.5, dxx, dxy, dyy);

    std::vector<float> lambda1, lambda2;
    ComputeEigenvalueImages(dxx.data(), dxy.data(), dyy.data(),
                             width_, height_, lambda1, lambda2);

    std::vector<float> response;
    ComputeValleyResponse(lambda1.data(), width_, height_, response);

    // All responses should be non-negative
    for (const auto& r : response) {
        EXPECT_GE(r, 0.0f);
    }
}

TEST_F(HessianTest, ComputeLineResponse_AbsoluteValue) {
    std::vector<float> dxx, dxy, dyy;
    ComputeHessianImage(ridgeImage_.data(), width_, height_, 1.5, dxx, dxy, dyy);

    std::vector<float> lambda1, lambda2;
    ComputeEigenvalueImages(dxx.data(), dxy.data(), dyy.data(),
                             width_, height_, lambda1, lambda2);

    std::vector<float> response;
    ComputeLineResponse(lambda1.data(), width_, height_, response);

    // All responses should be |λ1|
    for (size_t i = 0; i < response.size(); ++i) {
        EXPECT_FLOAT_EQ(response[i], std::abs(lambda1[i]));
    }
}

// ============================================================================
// Subpixel Hessian Tests
// ============================================================================

TEST_F(HessianTest, ComputeHessianAtSubpixel_IntegerCoord) {
    // At integer coordinates, subpixel should match regular
    int32_t x = width_ / 2;
    int32_t y = height_ / 2;

    auto h1 = ComputeHessianAt(quadImage_.data(), width_, height_,
                                x, y, 1.0);
    auto h2 = ComputeHessianAtSubpixel(quadImage_.data(), width_, height_,
                                        static_cast<double>(x),
                                        static_cast<double>(y), 1.0);

    EXPECT_NEAR(h1.dxx, h2.dxx, 1e-5);
    EXPECT_NEAR(h1.dxy, h2.dxy, 1e-5);
    EXPECT_NEAR(h1.dyy, h2.dyy, 1e-5);
}

TEST_F(HessianTest, ComputeHessianAtSubpixel_Interpolation) {
    // Subpixel position should give interpolated result
    double x = width_ / 2.0 + 0.5;
    double y = height_ / 2.0 + 0.5;

    auto h = ComputeHessianAtSubpixel(quadImage_.data(), width_, height_,
                                       x, y, 1.0);

    // Should still detect the quadratic nature
    EXPECT_GT(h.dxx, 0.0);
    EXPECT_GT(h.dyy, 0.0);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(HessianTest, ComputeHessianAt_SmallSigma) {
    // Small sigma should still work
    auto h = ComputeHessianAt(quadImage_.data(), width_, height_,
                               width_ / 2, height_ / 2, 0.5);

    // Just check it doesn't crash and returns valid values
    EXPECT_FALSE(std::isnan(h.dxx));
    EXPECT_FALSE(std::isnan(h.dxy));
    EXPECT_FALSE(std::isnan(h.dyy));
}

TEST_F(HessianTest, ComputeHessianAt_LargeSigma) {
    // Large sigma should still work (more smoothing)
    auto h = ComputeHessianAt(quadImage_.data(), width_, height_,
                               width_ / 2, height_ / 2, 5.0);

    EXPECT_FALSE(std::isnan(h.dxx));
    EXPECT_FALSE(std::isnan(h.dxy));
    EXPECT_FALSE(std::isnan(h.dyy));
}

TEST_F(HessianTest, ComputeHessianAt_BorderPixels) {
    // Should handle border pixels without crashing
    auto h1 = ComputeHessianAt(quadImage_.data(), width_, height_, 0, 0, 1.0);
    auto h2 = ComputeHessianAt(quadImage_.data(), width_, height_,
                                width_ - 1, height_ - 1, 1.0);

    EXPECT_FALSE(std::isnan(h1.dxx));
    EXPECT_FALSE(std::isnan(h2.dxx));
}
