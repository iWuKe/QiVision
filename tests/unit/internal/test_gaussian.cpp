/**
 * @file test_gaussian.cpp
 * @brief Unit tests for Internal/Gaussian.h
 */

#include <QiVision/Internal/Gaussian.h>
#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

using namespace Qi::Vision::Internal;

class GaussianTest : public ::testing::Test {
protected:
    static constexpr double TOLERANCE = 1e-10;

    // Check that a vector sums to approximately target
    bool SumsTo(const std::vector<double>& v, double target) {
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        return std::abs(sum - target) < TOLERANCE;
    }

    // Check symmetry around center
    bool IsSymmetric(const std::vector<double>& v) {
        size_t n = v.size();
        for (size_t i = 0; i < n / 2; ++i) {
            if (std::abs(v[i] - v[n - 1 - i]) > TOLERANCE) {
                return false;
            }
        }
        return true;
    }

    // Check antisymmetry around center (for derivatives)
    bool IsAntisymmetric(const std::vector<double>& v) {
        size_t n = v.size();
        for (size_t i = 0; i < n / 2; ++i) {
            if (std::abs(v[i] + v[n - 1 - i]) > TOLERANCE) {
                return false;
            }
        }
        // Center should be zero
        if (n % 2 == 1) {
            if (std::abs(v[n / 2]) > TOLERANCE) {
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// Kernel Size Computation Tests
// ============================================================================

TEST_F(GaussianTest, ComputeKernelSizeBasic) {
    // sigma=1.0, cutoff=3.0 -> size = 2*ceil(3) + 1 = 7
    EXPECT_EQ(Gaussian::ComputeKernelSize(1.0), 7);
}

TEST_F(GaussianTest, ComputeKernelSizeLargeSigma) {
    // sigma=2.5, cutoff=3.0 -> size = 2*ceil(7.5) + 1 = 17
    EXPECT_EQ(Gaussian::ComputeKernelSize(2.5), 17);
}

TEST_F(GaussianTest, ComputeKernelSizeSmallSigma) {
    // sigma=0.5 -> size = 2*ceil(1.5) + 1 = 5
    EXPECT_EQ(Gaussian::ComputeKernelSize(0.5), 5);
}

TEST_F(GaussianTest, ComputeKernelSizeMinimum) {
    // Very small sigma should return minimum size 3
    EXPECT_EQ(Gaussian::ComputeKernelSize(0.0), 3);
    EXPECT_EQ(Gaussian::ComputeKernelSize(0.1), 3);
}

TEST_F(GaussianTest, ComputeKernelSizeAlwaysOdd) {
    for (double sigma = 0.5; sigma <= 5.0; sigma += 0.3) {
        int32_t size = Gaussian::ComputeKernelSize(sigma);
        EXPECT_EQ(size % 2, 1) << "Size should be odd for sigma=" << sigma;
    }
}

TEST_F(GaussianTest, ComputeSigmaRoundTrip) {
    // ComputeSigma should be approximate inverse of ComputeKernelSize
    for (double sigma = 1.0; sigma <= 5.0; sigma += 0.5) {
        int32_t size = Gaussian::ComputeKernelSize(sigma);
        double recovered = Gaussian::ComputeSigma(size);
        // Should be approximately equal (not exact due to ceiling)
        EXPECT_NEAR(recovered, sigma, 0.5);
    }
}

// ============================================================================
// 1D Gaussian Kernel Tests
// ============================================================================

TEST_F(GaussianTest, Kernel1DCorrectSize) {
    auto k = Gaussian::Kernel1D(1.0);
    EXPECT_EQ(k.size(), 7u);  // Auto-computed

    auto k2 = Gaussian::Kernel1D(1.0, 11);
    EXPECT_EQ(k2.size(), 11u);  // Specified size
}

TEST_F(GaussianTest, Kernel1DNormalizedSumsToOne) {
    auto k = Gaussian::Kernel1D(1.5, 0, true);
    EXPECT_TRUE(SumsTo(k, 1.0));
}

TEST_F(GaussianTest, Kernel1DSymmetric) {
    auto k = Gaussian::Kernel1D(2.0);
    EXPECT_TRUE(IsSymmetric(k));
}

TEST_F(GaussianTest, Kernel1DPeakAtCenter) {
    auto k = Gaussian::Kernel1D(1.5);
    size_t center = k.size() / 2;

    // Center should have maximum value
    for (size_t i = 0; i < k.size(); ++i) {
        EXPECT_LE(k[i], k[center] + TOLERANCE);
    }
}

TEST_F(GaussianTest, Kernel1DDecreaseFromCenter) {
    auto k = Gaussian::Kernel1D(1.5);
    size_t center = k.size() / 2;

    // Values should decrease monotonically from center
    for (size_t i = 0; i < center; ++i) {
        EXPECT_LT(k[i], k[i + 1]);
    }
    for (size_t i = center; i < k.size() - 1; ++i) {
        EXPECT_GT(k[i], k[i + 1]);
    }
}

TEST_F(GaussianTest, Kernel1DZeroSigma) {
    // Zero sigma should give delta function
    auto k = Gaussian::Kernel1D(0.0, 5);
    EXPECT_EQ(k[2], 1.0);  // Center
    EXPECT_EQ(k[0], 0.0);
    EXPECT_EQ(k[4], 0.0);
}

TEST_F(GaussianTest, Kernel1DUnnormalized) {
    auto k = Gaussian::Kernel1D(1.0, 0, false);
    // Center value should be 1.0 (unnormalized Gaussian at x=0)
    size_t center = k.size() / 2;
    EXPECT_NEAR(k[center], 1.0, TOLERANCE);
}

// ============================================================================
// 1D Derivative Kernel Tests
// ============================================================================

TEST_F(GaussianTest, Derivative1DCorrectSize) {
    auto k = Gaussian::Derivative1D(1.0);
    EXPECT_EQ(k.size(), 7u);
}

TEST_F(GaussianTest, Derivative1DAntisymmetric) {
    auto k = Gaussian::Derivative1D(1.5);
    EXPECT_TRUE(IsAntisymmetric(k));
}

TEST_F(GaussianTest, Derivative1DSumsToZero) {
    auto k = Gaussian::Derivative1D(1.5);
    double sum = std::accumulate(k.begin(), k.end(), 0.0);
    EXPECT_NEAR(sum, 0.0, TOLERANCE);
}

TEST_F(GaussianTest, Derivative1DSign) {
    auto k = Gaussian::Derivative1D(1.5);
    size_t center = k.size() / 2;

    // Left half should be positive (or negative, depending on sign convention)
    // Right half should have opposite sign
    for (size_t i = 0; i < center; ++i) {
        EXPECT_GT(std::abs(k[i]), TOLERANCE);  // Non-zero
    }
}

TEST_F(GaussianTest, Derivative1DZeroSigma) {
    auto k = Gaussian::Derivative1D(0.0);
    // Should return simple difference kernel
    EXPECT_EQ(k.size(), 3u);
    EXPECT_NEAR(k[0], -0.5, TOLERANCE);
    EXPECT_NEAR(k[1], 0.0, TOLERANCE);
    EXPECT_NEAR(k[2], 0.5, TOLERANCE);
}

// ============================================================================
// 1D Second Derivative Kernel Tests
// ============================================================================

TEST_F(GaussianTest, SecondDerivative1DSymmetric) {
    auto k = Gaussian::SecondDerivative1D(1.5);
    EXPECT_TRUE(IsSymmetric(k));
}

TEST_F(GaussianTest, SecondDerivative1DSumsToZero) {
    // Second derivative of Gaussian sums to zero
    auto k = Gaussian::SecondDerivative1D(1.5, 0, false);
    double sum = std::accumulate(k.begin(), k.end(), 0.0);
    EXPECT_NEAR(sum, 0.0, 0.01);  // Looser tolerance due to truncation
}

TEST_F(GaussianTest, SecondDerivative1DCenterNegative) {
    // Center value should be negative (Mexican hat shape)
    auto k = Gaussian::SecondDerivative1D(1.5, 0, false);
    size_t center = k.size() / 2;
    EXPECT_LT(k[center], 0.0);
}

TEST_F(GaussianTest, SecondDerivative1DZeroSigma) {
    auto k = Gaussian::SecondDerivative1D(0.0);
    EXPECT_EQ(k.size(), 3u);
    EXPECT_NEAR(k[0], 1.0, TOLERANCE);
    EXPECT_NEAR(k[1], -2.0, TOLERANCE);
    EXPECT_NEAR(k[2], 1.0, TOLERANCE);
}

// ============================================================================
// 2D Gaussian Kernel Tests
// ============================================================================

TEST_F(GaussianTest, Kernel2DCorrectSize) {
    auto k = Gaussian::Kernel2D(1.0);
    EXPECT_EQ(k.size(), 7u * 7u);

    auto k2 = Gaussian::Kernel2D(1.0, 5);
    EXPECT_EQ(k2.size(), 5u * 5u);
}

TEST_F(GaussianTest, Kernel2DNormalizedSumsToOne) {
    auto k = Gaussian::Kernel2D(1.5, 0, true);
    EXPECT_TRUE(SumsTo(k, 1.0));
}

TEST_F(GaussianTest, Kernel2DSymmetricHorizontally) {
    auto k = Gaussian::Kernel2D(1.5, 5);
    // Check row symmetry
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 2; ++x) {
            double left = k[y * 5 + x];
            double right = k[y * 5 + (4 - x)];
            EXPECT_NEAR(left, right, TOLERANCE);
        }
    }
}

TEST_F(GaussianTest, Kernel2DSymmetricVertically) {
    auto k = Gaussian::Kernel2D(1.5, 5);
    // Check column symmetry
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 2; ++y) {
            double top = k[y * 5 + x];
            double bottom = k[(4 - y) * 5 + x];
            EXPECT_NEAR(top, bottom, TOLERANCE);
        }
    }
}

TEST_F(GaussianTest, Kernel2DCircularlySymmetric) {
    // Points at same distance from center should have same value
    auto k = Gaussian::Kernel2D(1.5, 7);
    // Corners (distance sqrt(9+9)=~4.24)
    EXPECT_NEAR(k[0], k[6], TOLERANCE);      // Top-left vs top-right
    EXPECT_NEAR(k[42], k[48], TOLERANCE);    // Bottom-left vs bottom-right
    EXPECT_NEAR(k[0], k[42], TOLERANCE);     // Top-left vs bottom-left
}

TEST_F(GaussianTest, Kernel2DPeakAtCenter) {
    auto k = Gaussian::Kernel2D(1.5, 5);
    double center = k[2 * 5 + 2];

    for (size_t i = 0; i < k.size(); ++i) {
        EXPECT_LE(k[i], center + TOLERANCE);
    }
}

// ============================================================================
// Anisotropic 2D Gaussian Tests
// ============================================================================

TEST_F(GaussianTest, Kernel2DAnisotropicCorrectSize) {
    auto k = Gaussian::Kernel2DAnisotropic(1.0, 2.0, 5, 9);
    EXPECT_EQ(k.size(), 9u * 5u);
}

TEST_F(GaussianTest, Kernel2DAnisotropicNormalizedSumsToOne) {
    auto k = Gaussian::Kernel2DAnisotropic(1.0, 2.0, 0, 0, true);
    EXPECT_TRUE(SumsTo(k, 1.0));
}

TEST_F(GaussianTest, Kernel2DAnisotropicShape) {
    auto k = Gaussian::Kernel2DAnisotropic(1.0, 2.0, 7, 11, false);

    // With larger sigmaY, kernel should be more elongated vertically
    // Compare decay in X vs Y direction from center
    int centerX = 3, centerY = 5;
    double center = k[centerY * 7 + centerX];

    // At distance 2 in X direction
    double distX = k[centerY * 7 + (centerX + 2)];
    // At distance 2 in Y direction
    double distY = k[(centerY + 2) * 7 + centerX];

    // With sigmaY > sigmaX, distY should be larger (slower decay)
    EXPECT_GT(distY, distX);
}

// ============================================================================
// Separable Kernel Tests
// ============================================================================

TEST_F(GaussianTest, SeparableSmoothBothSame) {
    auto sep = Gaussian::SeparableSmooth(1.5);
    EXPECT_EQ(sep.horizontal, sep.vertical);
}

TEST_F(GaussianTest, SeparableSmoothNormalized) {
    auto sep = Gaussian::SeparableSmooth(1.5);
    EXPECT_TRUE(SumsTo(sep.horizontal, 1.0));
    EXPECT_TRUE(SumsTo(sep.vertical, 1.0));
}

TEST_F(GaussianTest, SeparableGradientXCorrect) {
    auto sep = Gaussian::SeparableGradientX(1.5);

    // Horizontal should be antisymmetric (derivative)
    EXPECT_TRUE(IsAntisymmetric(sep.horizontal));
    // Vertical should be symmetric (smooth)
    EXPECT_TRUE(IsSymmetric(sep.vertical));
}

TEST_F(GaussianTest, SeparableGradientYCorrect) {
    auto sep = Gaussian::SeparableGradientY(1.5);

    // Horizontal should be symmetric (smooth)
    EXPECT_TRUE(IsSymmetric(sep.horizontal));
    // Vertical should be antisymmetric (derivative)
    EXPECT_TRUE(IsAntisymmetric(sep.vertical));
}

TEST_F(GaussianTest, SeparableGxxCorrect) {
    auto sep = Gaussian::SeparableGxx(1.5);

    // Horizontal should be symmetric (2nd derivative is symmetric)
    EXPECT_TRUE(IsSymmetric(sep.horizontal));
    // Vertical should be symmetric (smooth)
    EXPECT_TRUE(IsSymmetric(sep.vertical));
}

TEST_F(GaussianTest, SeparableGyyCorrect) {
    auto sep = Gaussian::SeparableGyy(1.5);

    // Horizontal should be symmetric (smooth)
    EXPECT_TRUE(IsSymmetric(sep.horizontal));
    // Vertical should be symmetric (2nd derivative)
    EXPECT_TRUE(IsSymmetric(sep.vertical));
}

TEST_F(GaussianTest, SeparableGxyBothDerivative) {
    auto sep = Gaussian::SeparableGxy(1.5);

    // Both should be antisymmetric (1st derivative)
    EXPECT_TRUE(IsAntisymmetric(sep.horizontal));
    EXPECT_TRUE(IsAntisymmetric(sep.vertical));
}

// ============================================================================
// Laplacian of Gaussian Tests
// ============================================================================

TEST_F(GaussianTest, LaplacianOfGaussianCorrectSize) {
    auto k = Gaussian::LaplacianOfGaussian(1.5);
    int size = static_cast<int>(std::sqrt(k.size()));
    EXPECT_EQ(size % 2, 1);  // Odd
    EXPECT_EQ(k.size(), static_cast<size_t>(size * size));
}

TEST_F(GaussianTest, LaplacianOfGaussianSumsToZero) {
    // LoG should sum to zero (approximately)
    // Use larger kernel size for better accuracy
    auto k = Gaussian::LaplacianOfGaussian(1.5, 15, false);
    double sum = std::accumulate(k.begin(), k.end(), 0.0);
    // Relaxed tolerance due to kernel truncation
    EXPECT_NEAR(sum, 0.0, 0.05);
}

TEST_F(GaussianTest, LaplacianOfGaussianMexicanHat) {
    auto k = Gaussian::LaplacianOfGaussian(1.5, 9, false);

    // Center should be negative
    int center = 4 * 9 + 4;
    EXPECT_LT(k[center], 0.0);

    // Corners should be positive (or nearly zero)
    EXPECT_GT(k[0], k[center]);
}

TEST_F(GaussianTest, LaplacianOfGaussianSymmetric) {
    auto k = Gaussian::LaplacianOfGaussian(1.5, 7);
    int size = 7;

    // Check 8-way symmetry
    EXPECT_NEAR(k[0], k[6], TOLERANCE);                  // Top corners
    EXPECT_NEAR(k[42], k[48], TOLERANCE);                // Bottom corners
    EXPECT_NEAR(k[0], k[42], TOLERANCE);                 // Left corners
    EXPECT_NEAR(k[3], k[3 + 42], TOLERANCE);             // Top/bottom mid
}

// ============================================================================
// Difference of Gaussians Tests
// ============================================================================

TEST_F(GaussianTest, DifferenceOfGaussiansCorrectSize) {
    auto k = Gaussian::DifferenceOfGaussians(1.0, 1.6, 9);
    EXPECT_EQ(k.size(), 9u * 9u);
}

TEST_F(GaussianTest, DifferenceOfGaussiansSumsToZero) {
    auto k = Gaussian::DifferenceOfGaussians(1.0, 1.6, 0, false);
    double sum = std::accumulate(k.begin(), k.end(), 0.0);
    EXPECT_NEAR(sum, 0.0, 0.01);
}

TEST_F(GaussianTest, DifferenceOfGaussiansApproximatesLoG) {
    // DoG = G(sigma1) - G(sigma2) approximates LoG
    // Note: DoG center is POSITIVE (smaller sigma has higher peak)
    // LoG center is NEGATIVE (inverted Mexican hat)
    // They approximate each other up to a sign flip and scaling
    auto dog = Gaussian::DifferenceOfGaussians(1.0, 1.6, 9, false);
    auto log = Gaussian::LaplacianOfGaussian(1.0, 9, false);

    int center = 4 * 9 + 4;
    // DoG center is positive (smaller Gaussian has higher peak)
    EXPECT_GT(dog[center], 0.0);
    // LoG center is negative
    EXPECT_LT(log[center], 0.0);

    // Both should have similar structure (edges opposite sign from center)
    // Check corners have opposite sign from center
    EXPECT_LT(dog[0] * dog[center], 0.0);  // Opposite signs
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST_F(GaussianTest, GaussianValueAtZero) {
    EXPECT_NEAR(Gaussian::GaussianValue(0.0, 1.0), 1.0, TOLERANCE);
    EXPECT_NEAR(Gaussian::GaussianValue(0.0, 2.0), 1.0, TOLERANCE);
}

TEST_F(GaussianTest, GaussianValueSymmetric) {
    EXPECT_NEAR(
        Gaussian::GaussianValue(1.5, 1.0),
        Gaussian::GaussianValue(-1.5, 1.0),
        TOLERANCE);
}

TEST_F(GaussianTest, GaussianValueDecays) {
    double sigma = 1.5;
    double g0 = Gaussian::GaussianValue(0.0, sigma);
    double g1 = Gaussian::GaussianValue(1.0, sigma);
    double g2 = Gaussian::GaussianValue(2.0, sigma);

    EXPECT_GT(g0, g1);
    EXPECT_GT(g1, g2);
}

TEST_F(GaussianTest, GaussianValue2DAtZero) {
    EXPECT_NEAR(Gaussian::GaussianValue2D(0.0, 0.0, 1.0), 1.0, TOLERANCE);
}

TEST_F(GaussianTest, GaussianValue2DSymmetric) {
    double sigma = 1.5;
    EXPECT_NEAR(
        Gaussian::GaussianValue2D(1.0, 1.0, sigma),
        Gaussian::GaussianValue2D(-1.0, -1.0, sigma),
        TOLERANCE);
    EXPECT_NEAR(
        Gaussian::GaussianValue2D(1.0, -1.0, sigma),
        Gaussian::GaussianValue2D(-1.0, 1.0, sigma),
        TOLERANCE);
}

TEST_F(GaussianTest, NormalizeBasic) {
    std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
    Gaussian::Normalize(v);
    EXPECT_TRUE(SumsTo(v, 1.0));
}

TEST_F(GaussianTest, NormalizeEmpty) {
    std::vector<double> v;
    Gaussian::Normalize(v);  // Should not crash
    EXPECT_TRUE(v.empty());
}

TEST_F(GaussianTest, NormalizeDerivativeBasic) {
    std::vector<double> v = {-2.0, -1.0, 0.0, 1.0, 2.0};
    Gaussian::NormalizeDerivative(v);

    // Max absolute value of positive or negative sum should be ~1
    double posSum = 0.0, negSum = 0.0;
    for (double x : v) {
        if (x > 0) posSum += x;
        else negSum += x;
    }
    double maxAbs = std::max(posSum, std::abs(negSum));
    EXPECT_NEAR(maxAbs, 1.0, TOLERANCE);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(GaussianTest, VerySmallSigma) {
    auto k = Gaussian::Kernel1D(0.001);
    // Should still produce valid kernel
    EXPECT_GE(k.size(), 3u);
    EXPECT_TRUE(SumsTo(k, 1.0));
}

TEST_F(GaussianTest, LargeSigma) {
    auto k = Gaussian::Kernel1D(10.0);
    // Should produce larger kernel
    EXPECT_GE(k.size(), 61u);  // 6*10+1 = 61
    EXPECT_TRUE(SumsTo(k, 1.0));
}

TEST_F(GaussianTest, SpecifiedEvenSize) {
    // Even sizes should be adjusted to odd
    auto k = Gaussian::Kernel1D(1.0, 6);
    EXPECT_EQ(k.size(), 7u);  // Adjusted to 7

    auto k2d = Gaussian::Kernel2D(1.0, 8);
    EXPECT_EQ(k2d.size(), 81u);  // 9x9
}

TEST_F(GaussianTest, NegativeSigma) {
    // Should handle gracefully (treat as zero or small positive)
    auto k = Gaussian::Kernel1D(-1.0, 5);
    EXPECT_EQ(k.size(), 5u);
}
