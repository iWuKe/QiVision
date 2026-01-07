/**
 * @file test_interpolate.cpp
 * @brief Unit tests for Internal/Interpolate.h
 */

#include <QiVision/Internal/Interpolate.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

class InterpolateTest : public ::testing::Test {
protected:
    // Simple 4x4 test image with known values
    std::vector<uint8_t> img4x4_;
    // Gradient image for precision tests
    std::vector<float> gradientImg_;
    int32_t gradientSize_ = 100;

    void SetUp() override {
        // 4x4 image:
        //  0  1  2  3
        //  4  5  6  7
        //  8  9 10 11
        // 12 13 14 15
        img4x4_ = {
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
            12, 13, 14, 15
        };

        // Create gradient image: f(x,y) = x + y*0.5
        gradientImg_.resize(gradientSize_ * gradientSize_);
        for (int y = 0; y < gradientSize_; ++y) {
            for (int x = 0; x < gradientSize_; ++x) {
                gradientImg_[y * gradientSize_ + x] =
                    static_cast<float>(x) + static_cast<float>(y) * 0.5f;
            }
        }
    }
};

// ============================================================================
// HandleBorder Tests
// ============================================================================

TEST_F(InterpolateTest, HandleBorderReplicate) {
    EXPECT_EQ(HandleBorder(-1, 4, BorderMode::Replicate), 0);
    EXPECT_EQ(HandleBorder(-5, 4, BorderMode::Replicate), 0);
    EXPECT_EQ(HandleBorder(4, 4, BorderMode::Replicate), 3);
    EXPECT_EQ(HandleBorder(10, 4, BorderMode::Replicate), 3);
    EXPECT_EQ(HandleBorder(2, 4, BorderMode::Replicate), 2);
}

TEST_F(InterpolateTest, HandleBorderReflect) {
    // Reflect (includes edge): cba|abcd|dcb
    // size=4: -1 -> 0, -2 -> 1, 4 -> 3, 5 -> 2
    EXPECT_EQ(HandleBorder(-1, 4, BorderMode::Reflect), 0);
    EXPECT_EQ(HandleBorder(-2, 4, BorderMode::Reflect), 1);
    EXPECT_EQ(HandleBorder(4, 4, BorderMode::Reflect), 3);
    EXPECT_EQ(HandleBorder(5, 4, BorderMode::Reflect), 2);
    EXPECT_EQ(HandleBorder(2, 4, BorderMode::Reflect), 2);
}

TEST_F(InterpolateTest, HandleBorderReflect101) {
    // Reflect101 (excludes edge): dcb|abcd|cba
    // size=4: -1 -> 1, -2 -> 2, 4 -> 2, 5 -> 1
    EXPECT_EQ(HandleBorder(-1, 4, BorderMode::Reflect101), 1);
    EXPECT_EQ(HandleBorder(-2, 4, BorderMode::Reflect101), 2);
    EXPECT_EQ(HandleBorder(4, 4, BorderMode::Reflect101), 2);
    EXPECT_EQ(HandleBorder(5, 4, BorderMode::Reflect101), 1);
    EXPECT_EQ(HandleBorder(2, 4, BorderMode::Reflect101), 2);
}

TEST_F(InterpolateTest, HandleBorderWrap) {
    // Wrap: bcd|abcd|abc
    EXPECT_EQ(HandleBorder(-1, 4, BorderMode::Wrap), 3);
    EXPECT_EQ(HandleBorder(-2, 4, BorderMode::Wrap), 2);
    EXPECT_EQ(HandleBorder(4, 4, BorderMode::Wrap), 0);
    EXPECT_EQ(HandleBorder(5, 4, BorderMode::Wrap), 1);
    EXPECT_EQ(HandleBorder(2, 4, BorderMode::Wrap), 2);
}

TEST_F(InterpolateTest, HandleBorderReflect101Size1) {
    // Edge case: size=1
    EXPECT_EQ(HandleBorder(-1, 1, BorderMode::Reflect101), 0);
    EXPECT_EQ(HandleBorder(1, 1, BorderMode::Reflect101), 0);
}

// ============================================================================
// CubicWeight Tests
// ============================================================================

TEST_F(InterpolateTest, CubicWeightAtInteger) {
    // At t=0, weight should be 1
    EXPECT_NEAR(CubicWeight(0.0), 1.0, 1e-10);
    // At t=1, weight should be 0
    EXPECT_NEAR(CubicWeight(1.0), 0.0, 1e-10);
    EXPECT_NEAR(CubicWeight(-1.0), 0.0, 1e-10);
    // At t=2 and beyond, weight should be 0
    EXPECT_NEAR(CubicWeight(2.0), 0.0, 1e-10);
    EXPECT_NEAR(CubicWeight(-2.0), 0.0, 1e-10);
    EXPECT_NEAR(CubicWeight(3.0), 0.0, 1e-10);
}

TEST_F(InterpolateTest, CubicWeightSymmetric) {
    EXPECT_NEAR(CubicWeight(0.5), CubicWeight(-0.5), 1e-10);
    EXPECT_NEAR(CubicWeight(1.5), CubicWeight(-1.5), 1e-10);
}

TEST_F(InterpolateTest, CubicWeightSumToOne) {
    // For integer interpolation points, sum of weights should be ~1
    for (double f = 0.0; f <= 1.0; f += 0.1) {
        double sum = 0.0;
        for (int i = 0; i < 4; ++i) {
            sum += CubicWeight(f - (i - 1));
        }
        EXPECT_NEAR(sum, 1.0, 1e-10);
    }
}

// ============================================================================
// Nearest Neighbor Tests
// ============================================================================

TEST_F(InterpolateTest, NearestAtPixelCenter) {
    // At exact pixel centers
    EXPECT_DOUBLE_EQ(InterpolateNearest(img4x4_.data(), 4, 4, 0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(InterpolateNearest(img4x4_.data(), 4, 4, 1.0, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(InterpolateNearest(img4x4_.data(), 4, 4, 1.0, 1.0), 5.0);
    EXPECT_DOUBLE_EQ(InterpolateNearest(img4x4_.data(), 4, 4, 3.0, 3.0), 15.0);
}

TEST_F(InterpolateTest, NearestRoundsCorrectly) {
    // Should round to nearest
    EXPECT_DOUBLE_EQ(InterpolateNearest(img4x4_.data(), 4, 4, 0.4, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(InterpolateNearest(img4x4_.data(), 4, 4, 0.6, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(InterpolateNearest(img4x4_.data(), 4, 4, 0.4, 0.6), 4.0);
}

TEST_F(InterpolateTest, NearestConstantBorder) {
    double val = InterpolateNearest(img4x4_.data(), 4, 4, -1.0, 0.0,
                                     BorderMode::Constant, 255.0);
    EXPECT_DOUBLE_EQ(val, 255.0);
}

// ============================================================================
// Bilinear Tests
// ============================================================================

TEST_F(InterpolateTest, BilinearAtPixelCenter) {
    // At pixel centers, bilinear should return exact pixel values
    EXPECT_DOUBLE_EQ(InterpolateBilinear(img4x4_.data(), 4, 4, 0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(InterpolateBilinear(img4x4_.data(), 4, 4, 1.0, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(InterpolateBilinear(img4x4_.data(), 4, 4, 1.0, 1.0), 5.0);
}

TEST_F(InterpolateTest, BilinearMidpoint) {
    // At midpoint between pixels
    // Between (0,0)=0 and (1,0)=1: should be 0.5
    EXPECT_DOUBLE_EQ(InterpolateBilinear(img4x4_.data(), 4, 4, 0.5, 0.0), 0.5);

    // Between (0,0)=0 and (0,1)=4: should be 2.0
    EXPECT_DOUBLE_EQ(InterpolateBilinear(img4x4_.data(), 4, 4, 0.0, 0.5), 2.0);

    // At center of first 4 pixels: (0+1+4+5)/4 = 2.5
    EXPECT_DOUBLE_EQ(InterpolateBilinear(img4x4_.data(), 4, 4, 0.5, 0.5), 2.5);
}

TEST_F(InterpolateTest, BilinearPrecision) {
    // Test on gradient image where f(x,y) = x + 0.5*y
    // Bilinear should be exact for linear gradients
    for (double y = 1.0; y < gradientSize_ - 2; y += 0.37) {
        for (double x = 1.0; x < gradientSize_ - 2; x += 0.37) {
            double expected = x + 0.5 * y;
            double actual = InterpolateBilinear(gradientImg_.data(),
                                                 gradientSize_, gradientSize_, x, y);
            EXPECT_NEAR(actual, expected, 1e-5);
        }
    }
}

TEST_F(InterpolateTest, BilinearConstantBorder) {
    double val = InterpolateBilinear(img4x4_.data(), 4, 4, -0.5, 0.0,
                                      BorderMode::Constant, 100.0);
    // Interpolating between border value (100) and pixel (0)
    EXPECT_NEAR(val, 50.0, 1e-10);
}

TEST_F(InterpolateTest, BilinearReflect101Border) {
    // Test that Reflect101 works correctly at edges
    double val = InterpolateBilinear(img4x4_.data(), 4, 4, -0.5, 0.0,
                                      BorderMode::Reflect101);
    // With Reflect101, -0.5 rounds to coordinate 0 or 1
    // Should use reflected pixel values
    EXPECT_GE(val, 0.0);
    EXPECT_LE(val, 4.0);
}

// ============================================================================
// Bicubic Tests
// ============================================================================

TEST_F(InterpolateTest, BicubicAtPixelCenter) {
    // At pixel centers, bicubic should return exact pixel values
    EXPECT_NEAR(InterpolateBicubic(img4x4_.data(), 4, 4, 1.0, 1.0), 5.0, 1e-10);
    EXPECT_NEAR(InterpolateBicubic(img4x4_.data(), 4, 4, 2.0, 2.0), 10.0, 1e-10);
}

TEST_F(InterpolateTest, BicubicPrecision) {
    // Test on gradient image where f(x,y) = x + 0.5*y
    // For linear gradients, bicubic should also be exact
    for (double y = 2.0; y < gradientSize_ - 3; y += 0.37) {
        for (double x = 2.0; x < gradientSize_ - 3; x += 0.37) {
            double expected = x + 0.5 * y;
            double actual = InterpolateBicubic(gradientImg_.data(),
                                                gradientSize_, gradientSize_, x, y);
            EXPECT_NEAR(actual, expected, 0.01);
        }
    }
}

TEST_F(InterpolateTest, BicubicSmoother) {
    // Create a step function image
    std::vector<uint8_t> step(16 * 16, 0);
    for (int y = 0; y < 16; ++y) {
        for (int x = 8; x < 16; ++x) {
            step[y * 16 + x] = 255;
        }
    }

    // At the edge, bicubic should show overshoot (ringing)
    double bilinear = InterpolateBilinear(step.data(), 16, 16, 7.5, 8.0);
    double bicubic = InterpolateBicubic(step.data(), 16, 16, 7.5, 8.0);

    // Both should be around 127.5 for midpoint
    EXPECT_NEAR(bilinear, 127.5, 1.0);
    // Bicubic may have slight over/undershoot
    EXPECT_NEAR(bicubic, 127.5, 20.0);
}

// ============================================================================
// Gradient Interpolation Tests
// ============================================================================

TEST_F(InterpolateTest, BilinearGradientDirection) {
    // On gradient image f(x,y) = x + 0.5*y
    // df/dx = 1, df/dy = 0.5
    double dx, dy;
    double val = InterpolateBilinearWithGradient(gradientImg_.data(),
                                                  gradientSize_, gradientSize_,
                                                  50.0, 50.0, dx, dy);

    EXPECT_NEAR(val, 50.0 + 0.5 * 50.0, 1e-5);
    EXPECT_NEAR(dx, 1.0, 0.01);
    EXPECT_NEAR(dy, 0.5, 0.01);
}

TEST_F(InterpolateTest, BilinearGradientSubpixel) {
    double dx, dy;
    InterpolateBilinearWithGradient(gradientImg_.data(),
                                     gradientSize_, gradientSize_,
                                     50.3, 50.7, dx, dy);

    EXPECT_NEAR(dx, 1.0, 0.01);
    EXPECT_NEAR(dy, 0.5, 0.01);
}

TEST_F(InterpolateTest, BicubicGradientDirection) {
    double dx, dy;
    double val = InterpolateBicubicWithGradient(gradientImg_.data(),
                                                 gradientSize_, gradientSize_,
                                                 50.0, 50.0, dx, dy);

    EXPECT_NEAR(val, 50.0 + 0.5 * 50.0, 0.01);
    EXPECT_NEAR(dx, 1.0, 0.05);
    EXPECT_NEAR(dy, 0.5, 0.05);
}

TEST_F(InterpolateTest, BicubicGradientSubpixel) {
    double dx, dy;
    InterpolateBicubicWithGradient(gradientImg_.data(),
                                    gradientSize_, gradientSize_,
                                    50.3, 50.7, dx, dy);

    EXPECT_NEAR(dx, 1.0, 0.05);
    EXPECT_NEAR(dy, 0.5, 0.05);
}

// ============================================================================
// Batch Interpolation Tests
// ============================================================================

TEST_F(InterpolateTest, BatchInterpolation) {
    std::vector<Point2d> points = {
        {0.0, 0.0}, {1.0, 0.0}, {0.5, 0.5}, {2.0, 2.0}
    };
    std::vector<double> results(points.size());

    InterpolateBatch(img4x4_.data(), 4, 4, points.data(), points.size(),
                      results.data(), InterpolationMethod::Bilinear);

    EXPECT_DOUBLE_EQ(results[0], 0.0);
    EXPECT_DOUBLE_EQ(results[1], 1.0);
    EXPECT_DOUBLE_EQ(results[2], 2.5);
    EXPECT_DOUBLE_EQ(results[3], 10.0);
}

TEST_F(InterpolateTest, BatchEmpty) {
    std::vector<double> results;
    InterpolateBatch<uint8_t>(img4x4_.data(), 4, 4, nullptr, 0,
                               results.data(), InterpolationMethod::Bilinear);
    // Should not crash
}

// ============================================================================
// Line Interpolation Tests
// ============================================================================

TEST_F(InterpolateTest, InterpolateAlongLineHorizontal) {
    // Horizontal line at y=0
    std::vector<double> results(4);
    InterpolateAlongLine(img4x4_.data(), 4, 4, 0.0, 0.0, 3.0, 0.0,
                          4, results.data());

    EXPECT_DOUBLE_EQ(results[0], 0.0);
    EXPECT_DOUBLE_EQ(results[1], 1.0);
    EXPECT_DOUBLE_EQ(results[2], 2.0);
    EXPECT_DOUBLE_EQ(results[3], 3.0);
}

TEST_F(InterpolateTest, InterpolateAlongLineVertical) {
    // Vertical line at x=0
    std::vector<double> results(4);
    InterpolateAlongLine(img4x4_.data(), 4, 4, 0.0, 0.0, 0.0, 3.0,
                          4, results.data());

    EXPECT_DOUBLE_EQ(results[0], 0.0);
    EXPECT_DOUBLE_EQ(results[1], 4.0);
    EXPECT_DOUBLE_EQ(results[2], 8.0);
    EXPECT_DOUBLE_EQ(results[3], 12.0);
}

TEST_F(InterpolateTest, InterpolateAlongLineDiagonal) {
    // Diagonal line from (0,0) to (3,3)
    std::vector<double> results(4);
    InterpolateAlongLine(img4x4_.data(), 4, 4, 0.0, 0.0, 3.0, 3.0,
                          4, results.data());

    EXPECT_DOUBLE_EQ(results[0], 0.0);   // (0,0) = 0
    EXPECT_DOUBLE_EQ(results[1], 5.0);   // (1,1) = 5
    EXPECT_DOUBLE_EQ(results[2], 10.0);  // (2,2) = 10
    EXPECT_DOUBLE_EQ(results[3], 15.0);  // (3,3) = 15
}

TEST_F(InterpolateTest, InterpolateAlongLineSingleSample) {
    std::vector<double> results(1);
    InterpolateAlongLine(img4x4_.data(), 4, 4, 1.5, 1.5, 2.5, 2.5,
                          1, results.data());
    // Should return value at start point
    EXPECT_NEAR(results[0], 7.5, 1e-10);
}

TEST_F(InterpolateTest, InterpolateAlongLineZeroSamples) {
    std::vector<double> results;
    InterpolateAlongLine(img4x4_.data(), 4, 4, 0.0, 0.0, 3.0, 3.0,
                          0, results.data());
    // Should not crash
}

// ============================================================================
// Generic Interpolate Function Tests
// ============================================================================

TEST_F(InterpolateTest, GenericInterpolateDispatch) {
    double nearest = Interpolate(img4x4_.data(), 4, 4, 0.3, 0.3,
                                  InterpolationMethod::Nearest);
    double bilinear = Interpolate(img4x4_.data(), 4, 4, 0.3, 0.3,
                                   InterpolationMethod::Bilinear);
    double bicubic = Interpolate(img4x4_.data(), 4, 4, 0.3, 0.3,
                                  InterpolationMethod::Bicubic);

    // Nearest should round to (0,0) = 0
    EXPECT_DOUBLE_EQ(nearest, 0.0);
    // Bilinear should be between 0 and 5
    EXPECT_GT(bilinear, 0.0);
    EXPECT_LT(bilinear, 5.0);
    // Bicubic should also be reasonable
    EXPECT_GT(bicubic, -1.0);
    EXPECT_LT(bicubic, 6.0);
}

// ============================================================================
// Type Tests (uint8_t, uint16_t, float)
// ============================================================================

TEST_F(InterpolateTest, InterpolateUInt16) {
    std::vector<uint16_t> img16 = {0, 1000, 2000, 3000,
                                    4000, 5000, 6000, 7000,
                                    8000, 9000, 10000, 11000,
                                    12000, 13000, 14000, 15000};

    double val = InterpolateBilinear(img16.data(), 4, 4, 0.5, 0.5);
    EXPECT_DOUBLE_EQ(val, 2500.0);
}

TEST_F(InterpolateTest, InterpolateFloat) {
    std::vector<float> imgF = {0.0f, 0.1f, 0.2f, 0.3f,
                                0.4f, 0.5f, 0.6f, 0.7f,
                                0.8f, 0.9f, 1.0f, 1.1f,
                                1.2f, 1.3f, 1.4f, 1.5f};

    double val = InterpolateBilinear(imgF.data(), 4, 4, 0.5, 0.5);
    EXPECT_NEAR(val, 0.25, 1e-6);
}

TEST_F(InterpolateTest, InterpolateInt16) {
    std::vector<int16_t> imgS = {-100, -50, 0, 50,
                                  100, 150, 200, 250,
                                  300, 350, 400, 450,
                                  500, 550, 600, 650};

    double val = InterpolateBilinear(imgS.data(), 4, 4, 0.5, 0.5);
    EXPECT_NEAR(val, 25.0, 1e-10);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(InterpolateTest, InterpolateAtCorners) {
    // Test at all four corners
    EXPECT_DOUBLE_EQ(InterpolateBilinear(img4x4_.data(), 4, 4, 0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(InterpolateBilinear(img4x4_.data(), 4, 4, 3.0, 0.0), 3.0);
    EXPECT_DOUBLE_EQ(InterpolateBilinear(img4x4_.data(), 4, 4, 0.0, 3.0), 12.0);
    EXPECT_DOUBLE_EQ(InterpolateBilinear(img4x4_.data(), 4, 4, 3.0, 3.0), 15.0);
}

TEST_F(InterpolateTest, InterpolateOutOfBoundsReflect101) {
    // Test outside image with reflection
    double val = InterpolateBilinear(img4x4_.data(), 4, 4, -0.5, 0.0,
                                      BorderMode::Reflect101);
    // Should be interpolation of reflected values
    EXPECT_GE(val, 0.0);
    EXPECT_LE(val, 4.0);
}

TEST_F(InterpolateTest, InterpolateFarOutOfBounds) {
    // Very far outside
    double val = InterpolateBilinear(img4x4_.data(), 4, 4, -100.0, -100.0,
                                      BorderMode::Reflect101);
    // Should still produce a valid result
    EXPECT_GE(val, 0.0);
    EXPECT_LE(val, 15.0);
}

TEST_F(InterpolateTest, Interpolate1x1Image) {
    std::vector<uint8_t> tiny = {42};

    // All positions should return 42 with replicate
    EXPECT_DOUBLE_EQ(InterpolateBilinear(tiny.data(), 1, 1, 0.0, 0.0), 42.0);
    EXPECT_DOUBLE_EQ(InterpolateBilinear(tiny.data(), 1, 1, 0.5, 0.5,
                                          BorderMode::Replicate), 42.0);
}

// ============================================================================
// Precision Tests (for subpixel accuracy claims)
// ============================================================================

TEST_F(InterpolateTest, BilinearSubpixelPrecision) {
    // Create a quadratic surface: f(x,y) = 100 + 10*x + 5*y
    // Linear function should be exactly reproduced by bilinear
    const int size = 50;
    std::vector<float> quadImg(size * size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            quadImg[y * size + x] = 100.0f + 10.0f * x + 5.0f * y;
        }
    }

    // Test at many subpixel locations
    double maxError = 0.0;
    for (double y = 5.0; y < size - 5; y += 0.17) {
        for (double x = 5.0; x < size - 5; x += 0.17) {
            double expected = 100.0 + 10.0 * x + 5.0 * y;
            double actual = InterpolateBilinear(quadImg.data(), size, size, x, y);
            double error = std::abs(actual - expected);
            maxError = std::max(maxError, error);
        }
    }

    // For linear functions, bilinear should be exact
    EXPECT_LT(maxError, 1e-4);
}

TEST_F(InterpolateTest, BicubicSubpixelPrecision) {
    // For linear functions, bicubic should also be exact
    const int size = 50;
    std::vector<float> linearImg(size * size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            linearImg[y * size + x] = 100.0f + 10.0f * x + 5.0f * y;
        }
    }

    double maxError = 0.0;
    for (double y = 5.0; y < size - 5; y += 0.17) {
        for (double x = 5.0; x < size - 5; x += 0.17) {
            double expected = 100.0 + 10.0 * x + 5.0 * y;
            double actual = InterpolateBicubic(linearImg.data(), size, size, x, y);
            double error = std::abs(actual - expected);
            maxError = std::max(maxError, error);
        }
    }

    // For linear functions, bicubic should be very accurate
    EXPECT_LT(maxError, 0.01);
}

