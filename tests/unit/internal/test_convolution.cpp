/**
 * @file test_convolution.cpp
 * @brief Unit tests for Internal/Convolution.h
 */

#include <QiVision/Internal/Convolution.h>
#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

class ConvolutionTest : public ::testing::Test {
protected:
    // 8x8 constant image
    std::vector<uint8_t> constant8x8_;

    // 8x8 gradient image
    std::vector<uint8_t> gradient8x8_;

    // 16x16 test image
    std::vector<float> test16x16_;

    void SetUp() override {
        // Constant image (all 100)
        constant8x8_.resize(64, 100);

        // Horizontal gradient (0-224)
        gradient8x8_.resize(64);
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                gradient8x8_[y * 8 + x] = static_cast<uint8_t>(x * 32);
            }
        }

        // Test image with known values
        test16x16_.resize(256);
        for (int y = 0; y < 16; ++y) {
            for (int x = 0; x < 16; ++x) {
                test16x16_[y * 16 + x] = static_cast<float>(x + y * 0.5f);
            }
        }
    }
};

// ============================================================================
// Helper Function Tests
// ============================================================================

TEST_F(ConvolutionTest, MakeOdd) {
    EXPECT_EQ(MakeOdd(3), 3);
    EXPECT_EQ(MakeOdd(4), 5);
    EXPECT_EQ(MakeOdd(5), 5);
    EXPECT_EQ(MakeOdd(6), 7);
    EXPECT_EQ(MakeOdd(1), 1);
}

TEST_F(ConvolutionTest, KernelSizeFromSigma) {
    // sigma * 6 rounded to odd
    EXPECT_EQ(KernelSizeFromSigma(1.0), 7);   // 6 -> 7
    EXPECT_EQ(KernelSizeFromSigma(2.0), 13);  // 12 -> 13
    EXPECT_EQ(KernelSizeFromSigma(0.5), 3);   // 3 -> 3
}

TEST_F(ConvolutionTest, SigmaFromKernelSize) {
    EXPECT_NEAR(SigmaFromKernelSize(7), 7.0 / 6.0, 1e-10);
    EXPECT_NEAR(SigmaFromKernelSize(13), 13.0 / 6.0, 1e-10);
}

TEST_F(ConvolutionTest, GenerateGaussianKernel1D) {
    auto kernel = GenerateGaussianKernel1D(1.0);

    // Should be odd size
    EXPECT_EQ(kernel.size() % 2, 1u);

    // Should be normalized (sum to 1)
    double sum = std::accumulate(kernel.begin(), kernel.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-10);

    // Center should be maximum
    size_t center = kernel.size() / 2;
    for (size_t i = 0; i < kernel.size(); ++i) {
        EXPECT_LE(kernel[i], kernel[center] + 1e-10);
    }

    // Should be symmetric
    for (size_t i = 0; i < kernel.size() / 2; ++i) {
        EXPECT_NEAR(kernel[i], kernel[kernel.size() - 1 - i], 1e-10);
    }
}

TEST_F(ConvolutionTest, GenerateGaussianKernel1DFixedSize) {
    auto kernel = GenerateGaussianKernel1D(1.0, 5);

    EXPECT_EQ(kernel.size(), 5u);

    double sum = std::accumulate(kernel.begin(), kernel.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST_F(ConvolutionTest, GenerateBoxKernel1D) {
    auto kernel = GenerateBoxKernel1D(5);

    EXPECT_EQ(kernel.size(), 5u);

    // All values should be 1/5
    for (double val : kernel) {
        EXPECT_NEAR(val, 0.2, 1e-10);
    }

    // Sum should be 1
    double sum = std::accumulate(kernel.begin(), kernel.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

// ============================================================================
// ConvolveRow/Col Tests
// ============================================================================

TEST_F(ConvolutionTest, ConvolveRowIdentity) {
    // Identity kernel [0, 1, 0]
    std::vector<double> kernel = {0.0, 1.0, 0.0};
    std::vector<float> dst(64);

    ConvolveRow(constant8x8_.data(), dst.data(), 8, 8,
                kernel.data(), 3);

    for (size_t i = 0; i < 64; ++i) {
        EXPECT_NEAR(dst[i], 100.0f, 1e-5);
    }
}

TEST_F(ConvolutionTest, ConvolveRowBoxFilter) {
    // Box kernel [1/3, 1/3, 1/3]
    std::vector<double> kernel = {1.0/3.0, 1.0/3.0, 1.0/3.0};
    std::vector<float> dst(64);

    ConvolveRow(constant8x8_.data(), dst.data(), 8, 8,
                kernel.data(), 3);

    // Constant image should remain constant
    for (size_t i = 0; i < 64; ++i) {
        EXPECT_NEAR(dst[i], 100.0f, 1e-5);
    }
}

TEST_F(ConvolutionTest, ConvolveColIdentity) {
    std::vector<double> kernel = {0.0, 1.0, 0.0};
    std::vector<float> dst(64);

    ConvolveCol(constant8x8_.data(), dst.data(), 8, 8,
                kernel.data(), 3);

    for (size_t i = 0; i < 64; ++i) {
        EXPECT_NEAR(dst[i], 100.0f, 1e-5);
    }
}

// ============================================================================
// ConvolveSeparable Tests
// ============================================================================

TEST_F(ConvolutionTest, ConvolveSeparableIdentity) {
    std::vector<double> kernelX = {0.0, 1.0, 0.0};
    std::vector<double> kernelY = {0.0, 1.0, 0.0};
    std::vector<float> dst(64);

    ConvolveSeparable(constant8x8_.data(), dst.data(), 8, 8,
                      kernelX.data(), 3, kernelY.data(), 3);

    for (size_t i = 0; i < 64; ++i) {
        EXPECT_NEAR(dst[i], 100.0f, 1e-5);
    }
}

TEST_F(ConvolutionTest, ConvolveSeparableSymmetric) {
    std::vector<double> kernel = {0.25, 0.5, 0.25};
    std::vector<float> dst(64);

    ConvolveSeparableSymmetric(constant8x8_.data(), dst.data(), 8, 8,
                                kernel.data(), 3);

    // Constant image stays constant
    for (int y = 1; y < 7; ++y) {
        for (int x = 1; x < 7; ++x) {
            EXPECT_NEAR(dst[y * 8 + x], 100.0f, 1e-3);
        }
    }
}

// ============================================================================
// Convolve2D Tests
// ============================================================================

TEST_F(ConvolutionTest, Convolve2DIdentity) {
    // 3x3 identity kernel
    std::vector<double> kernel = {
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0
    };
    std::vector<float> dst(64);

    Convolve2D(constant8x8_.data(), dst.data(), 8, 8,
               kernel.data(), 3, 3);

    for (size_t i = 0; i < 64; ++i) {
        EXPECT_NEAR(dst[i], 100.0f, 1e-5);
    }
}

TEST_F(ConvolutionTest, Convolve2DBoxFilter) {
    // 3x3 box kernel
    std::vector<double> kernel(9, 1.0 / 9.0);
    std::vector<float> dst(64);

    Convolve2D(constant8x8_.data(), dst.data(), 8, 8,
               kernel.data(), 3, 3);

    // Constant image stays constant
    for (int y = 1; y < 7; ++y) {
        for (int x = 1; x < 7; ++x) {
            EXPECT_NEAR(dst[y * 8 + x], 100.0f, 1e-3);
        }
    }
}

// ============================================================================
// Integral Image Tests
// ============================================================================

TEST_F(ConvolutionTest, IntegralImageConstant) {
    std::vector<double> integral(9 * 9);  // (8+1) x (8+1)

    ComputeIntegralImage(constant8x8_.data(), integral.data(), 8, 8);

    // First row/col should be 0
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(integral[i], 0.0);
        EXPECT_DOUBLE_EQ(integral[i * 9], 0.0);
    }

    // integral[y+1][x+1] should be 100 * (x+1) * (y+1)
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            double expected = 100.0 * (x + 1) * (y + 1);
            EXPECT_NEAR(integral[(y + 1) * 9 + (x + 1)], expected, 1e-10)
                << "at (" << x << ", " << y << ")";
        }
    }
}

TEST_F(ConvolutionTest, GetRectSum) {
    std::vector<double> integral(9 * 9);
    ComputeIntegralImage(constant8x8_.data(), integral.data(), 8, 8);

    // Sum of 3x3 region starting at (2,2)
    double sum = GetRectSum(integral.data(), 9, 2, 2, 5, 5);
    EXPECT_NEAR(sum, 100.0 * 9, 1e-10);

    // Sum of entire image
    sum = GetRectSum(integral.data(), 9, 0, 0, 8, 8);
    EXPECT_NEAR(sum, 100.0 * 64, 1e-10);

    // Sum of single pixel
    sum = GetRectSum(integral.data(), 9, 3, 3, 4, 4);
    EXPECT_NEAR(sum, 100.0, 1e-10);
}

// ============================================================================
// BoxFilter Tests
// ============================================================================

TEST_F(ConvolutionTest, BoxFilterConstant) {
    std::vector<float> dst(64);

    BoxFilter(constant8x8_.data(), dst.data(), 8, 8, 3, 3);

    // Should preserve constant value
    for (size_t i = 0; i < 64; ++i) {
        EXPECT_NEAR(dst[i], 100.0f, 1.0f);
    }
}

TEST_F(ConvolutionTest, BoxFilterSmall) {
    std::vector<float> dst(64);

    // Small kernel uses direct convolution
    BoxFilter(constant8x8_.data(), dst.data(), 8, 8, 3, 3);

    for (int y = 1; y < 7; ++y) {
        for (int x = 1; x < 7; ++x) {
            EXPECT_NEAR(dst[y * 8 + x], 100.0f, 1e-3);
        }
    }
}

TEST_F(ConvolutionTest, BoxFilterLarge) {
    // Create larger image
    std::vector<uint8_t> large(32 * 32, 100);
    std::vector<float> dst(32 * 32);

    // Large kernel uses integral image
    BoxFilter(large.data(), dst.data(), 32, 32, 7, 7);

    // Center should be accurate
    for (int y = 5; y < 27; ++y) {
        for (int x = 5; x < 27; ++x) {
            EXPECT_NEAR(dst[y * 32 + x], 100.0f, 1e-3);
        }
    }
}

// ============================================================================
// GaussianBlur Tests
// ============================================================================

TEST_F(ConvolutionTest, GaussianBlurConstant) {
    std::vector<float> dst(64);

    GaussianBlur(constant8x8_.data(), dst.data(), 8, 8, 1.0);

    // Constant image stays constant
    for (int y = 2; y < 6; ++y) {
        for (int x = 2; x < 6; ++x) {
            EXPECT_NEAR(dst[y * 8 + x], 100.0f, 0.5f);
        }
    }
}

TEST_F(ConvolutionTest, GaussianBlurSmoothing) {
    // Create image with sharp edge
    std::vector<uint8_t> edge(64, 0);
    for (int y = 0; y < 8; ++y) {
        for (int x = 4; x < 8; ++x) {
            edge[y * 8 + x] = 255;
        }
    }

    std::vector<float> dst(64);
    GaussianBlur(edge.data(), dst.data(), 8, 8, 1.0);

    // Edge should be smoothed
    // At x=3 (just before edge), value should be between 0 and 255
    EXPECT_GT(dst[4 * 8 + 3], 0.0f);
    EXPECT_LT(dst[4 * 8 + 3], 128.0f);

    // At x=4 (start of edge), value should be between 0 and 255
    EXPECT_GT(dst[4 * 8 + 4], 128.0f);
    EXPECT_LT(dst[4 * 8 + 4], 255.0f);
}

TEST_F(ConvolutionTest, GaussianBlurAnisotropic) {
    std::vector<float> dst(64);

    // Different sigmas for X and Y
    GaussianBlur(constant8x8_.data(), dst.data(), 8, 8, 1.0, 2.0);

    // Should still preserve constant
    for (int y = 3; y < 5; ++y) {
        for (int x = 2; x < 6; ++x) {
            EXPECT_NEAR(dst[y * 8 + x], 100.0f, 1.0f);
        }
    }
}

TEST_F(ConvolutionTest, GaussianBlurFixed) {
    std::vector<float> dst(64);

    GaussianBlurFixed(constant8x8_.data(), dst.data(), 8, 8, 5, 1.0);

    for (int y = 2; y < 6; ++y) {
        for (int x = 2; x < 6; ++x) {
            EXPECT_NEAR(dst[y * 8 + x], 100.0f, 1.0f);
        }
    }
}

TEST_F(ConvolutionTest, GaussianBlurFixedAutoSigma) {
    std::vector<float> dst(64);

    // sigma = 0 means compute from kernel size
    GaussianBlurFixed(constant8x8_.data(), dst.data(), 8, 8, 5, 0.0);

    for (int y = 2; y < 6; ++y) {
        for (int x = 2; x < 6; ++x) {
            EXPECT_NEAR(dst[y * 8 + x], 100.0f, 1.0f);
        }
    }
}

// ============================================================================
// ConvolveNormalized Tests
// ============================================================================

TEST_F(ConvolutionTest, ConvolveNormalizedNoMask) {
    std::vector<double> kernel(9, 1.0);  // All 1s (will be normalized)
    std::vector<float> dst(64);

    ConvolveNormalized(constant8x8_.data(), nullptr, dst.data(), 8, 8,
                       kernel.data(), 3, 3);

    // Interior should be average = 100
    for (int y = 1; y < 7; ++y) {
        for (int x = 1; x < 7; ++x) {
            EXPECT_NEAR(dst[y * 8 + x], 100.0f, 1e-3);
        }
    }
}

TEST_F(ConvolutionTest, ConvolveNormalizedWithMask) {
    // Create mask with some invalid pixels
    std::vector<uint8_t> mask(64, 1);
    mask[0] = 0;  // Top-left corner invalid

    std::vector<double> kernel(9, 1.0);
    std::vector<float> dst(64);

    ConvolveNormalized(constant8x8_.data(), mask.data(), dst.data(), 8, 8,
                       kernel.data(), 3, 3);

    // At (1,1), one neighbor is invalid, but result should still be ~100
    EXPECT_NEAR(dst[1 * 8 + 1], 100.0f, 1.0f);
}

TEST_F(ConvolutionTest, ConvolveNormalizedAllMasked) {
    // All pixels masked
    std::vector<uint8_t> mask(64, 0);
    std::vector<double> kernel(9, 1.0);
    std::vector<float> dst(64);

    ConvolveNormalized(constant8x8_.data(), mask.data(), dst.data(), 8, 8,
                       kernel.data(), 3, 3);

    // Result should be 0 when no valid neighbors
    for (size_t i = 0; i < 64; ++i) {
        EXPECT_NEAR(dst[i], 0.0f, 1e-5);
    }
}

// ============================================================================
// Border Mode Tests
// ============================================================================

TEST_F(ConvolutionTest, ConvolveBorderConstant) {
    std::vector<double> kernel = {1.0/3.0, 1.0/3.0, 1.0/3.0};
    std::vector<float> dst(64);

    ConvolveRow(constant8x8_.data(), dst.data(), 8, 8,
                kernel.data(), 3, BorderMode::Constant, 0.0);

    // Edge pixels should be affected by border value
    // At x=0: (0 + 100 + 100) / 3 = 66.67
    EXPECT_NEAR(dst[0], 200.0f / 3.0f, 1e-3);
}

TEST_F(ConvolutionTest, ConvolveBorderReplicate) {
    std::vector<double> kernel = {1.0/3.0, 1.0/3.0, 1.0/3.0};
    std::vector<float> dst(64);

    ConvolveRow(constant8x8_.data(), dst.data(), 8, 8,
                kernel.data(), 3, BorderMode::Replicate);

    // Edge pixels should replicate edge value
    // At x=0: (100 + 100 + 100) / 3 = 100
    EXPECT_NEAR(dst[0], 100.0f, 1e-3);
}

TEST_F(ConvolutionTest, ConvolveBorderReflect101) {
    std::vector<double> kernel = {1.0/3.0, 1.0/3.0, 1.0/3.0};
    std::vector<float> dst(64);

    ConvolveRow(constant8x8_.data(), dst.data(), 8, 8,
                kernel.data(), 3, BorderMode::Reflect101);

    // Constant image, so all border modes give same result
    EXPECT_NEAR(dst[0], 100.0f, 1e-3);
}

// ============================================================================
// Type Tests
// ============================================================================

TEST_F(ConvolutionTest, ConvolveFloat) {
    std::vector<float> src(64, 100.0f);
    std::vector<float> dst(64);
    std::vector<double> kernel = {0.25, 0.5, 0.25};

    ConvolveSeparableSymmetric(src.data(), dst.data(), 8, 8,
                                kernel.data(), 3);

    for (int y = 1; y < 7; ++y) {
        for (int x = 1; x < 7; ++x) {
            EXPECT_NEAR(dst[y * 8 + x], 100.0f, 1e-3);
        }
    }
}

TEST_F(ConvolutionTest, ConvolveDouble) {
    std::vector<double> src(64, 100.0);
    std::vector<double> dst(64);
    std::vector<double> kernel = {0.25, 0.5, 0.25};

    ConvolveSeparable(src.data(), dst.data(), 8, 8,
                      kernel.data(), 3, kernel.data(), 3);

    for (int y = 1; y < 7; ++y) {
        for (int x = 1; x < 7; ++x) {
            EXPECT_NEAR(dst[y * 8 + x], 100.0, 1e-10);
        }
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(ConvolutionTest, SmallImage1x1) {
    std::vector<uint8_t> tiny = {100};
    std::vector<float> dst(1);
    std::vector<double> kernel = {0.25, 0.5, 0.25};

    ConvolveSeparableSymmetric(tiny.data(), dst.data(), 1, 1,
                                kernel.data(), 3);

    EXPECT_NEAR(dst[0], 100.0f, 1e-3);
}

TEST_F(ConvolutionTest, SmallImage2x2) {
    std::vector<uint8_t> small = {100, 100, 100, 100};
    std::vector<float> dst(4);
    std::vector<double> kernel = {0.25, 0.5, 0.25};

    ConvolveSeparableSymmetric(small.data(), dst.data(), 2, 2,
                                kernel.data(), 3);

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(dst[i], 100.0f, 1e-3);
    }
}

TEST_F(ConvolutionTest, LargeKernel) {
    std::vector<float> dst(64);

    // Kernel size 7
    GaussianBlurFixed(constant8x8_.data(), dst.data(), 8, 8, 7, 1.5);

    // Center should still be ~100
    EXPECT_NEAR(dst[4 * 8 + 4], 100.0f, 2.0f);
}

