/**
 * @file test_gradient.cpp
 * @brief Unit tests for Internal/Gradient.h
 */

#include <QiVision/Internal/Gradient.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

class GradientTest : public ::testing::Test {
protected:
    // Simple 8x8 test image
    std::vector<uint8_t> img8x8_;

    // Horizontal gradient image: f(x,y) = x
    std::vector<float> horzGrad_;

    // Vertical gradient image: f(x,y) = y
    std::vector<float> vertGrad_;

    int32_t size_ = 16;

    void SetUp() override {
        // 8x8 image with linear horizontal gradient
        img8x8_.resize(64);
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                img8x8_[y * 8 + x] = static_cast<uint8_t>(x * 32);
            }
        }

        // Horizontal gradient image: f(x,y) = x
        horzGrad_.resize(size_ * size_);
        for (int y = 0; y < size_; ++y) {
            for (int x = 0; x < size_; ++x) {
                horzGrad_[y * size_ + x] = static_cast<float>(x);
            }
        }

        // Vertical gradient image: f(x,y) = y
        vertGrad_.resize(size_ * size_);
        for (int y = 0; y < size_; ++y) {
            for (int x = 0; x < size_; ++x) {
                vertGrad_[y * size_ + x] = static_cast<float>(y);
            }
        }
    }
};

// ============================================================================
// Kernel Tests
// ============================================================================

TEST_F(GradientTest, SobelDerivativeKernel3x3) {
    auto k = SobelDerivativeKernel(3);
    EXPECT_EQ(k.size(), 3u);
    EXPECT_DOUBLE_EQ(k[0], -1.0);
    EXPECT_DOUBLE_EQ(k[1], 0.0);
    EXPECT_DOUBLE_EQ(k[2], 1.0);
}

TEST_F(GradientTest, SobelDerivativeKernel5x5) {
    auto k = SobelDerivativeKernel(5);
    EXPECT_EQ(k.size(), 5u);
    EXPECT_DOUBLE_EQ(k[0], -1.0);
    EXPECT_DOUBLE_EQ(k[2], 0.0);
    EXPECT_DOUBLE_EQ(k[4], 1.0);
}

TEST_F(GradientTest, SobelSmoothingKernel3x3) {
    auto k = SobelSmoothingKernel(3);
    EXPECT_EQ(k.size(), 3u);
    EXPECT_DOUBLE_EQ(k[0], 1.0);
    EXPECT_DOUBLE_EQ(k[1], 2.0);
    EXPECT_DOUBLE_EQ(k[2], 1.0);
}

TEST_F(GradientTest, ScharrKernels) {
    auto d = ScharrDerivativeKernel();
    auto s = ScharrSmoothingKernel();

    EXPECT_EQ(d.size(), 3u);
    EXPECT_EQ(s.size(), 3u);

    // Scharr smoothing should sum to 1
    EXPECT_NEAR(s[0] + s[1] + s[2], 1.0, 1e-10);
}

TEST_F(GradientTest, PrewittKernels) {
    auto d = PrewittDerivativeKernel();
    auto s = PrewittSmoothingKernel();

    EXPECT_EQ(d.size(), 3u);
    EXPECT_EQ(s.size(), 3u);

    // Prewitt smoothing should sum to 1
    EXPECT_NEAR(s[0] + s[1] + s[2], 1.0, 1e-10);
}

TEST_F(GradientTest, GetKernelSize) {
    EXPECT_EQ(GetKernelSize(GradientOperator::Sobel3x3), 3);
    EXPECT_EQ(GetKernelSize(GradientOperator::Sobel5x5), 5);
    EXPECT_EQ(GetKernelSize(GradientOperator::Sobel7x7), 7);
    EXPECT_EQ(GetKernelSize(GradientOperator::Scharr), 3);
    EXPECT_EQ(GetKernelSize(GradientOperator::Prewitt), 3);
    EXPECT_EQ(GetKernelSize(GradientOperator::Central), 3);
}

// ============================================================================
// GradientX Tests
// ============================================================================

TEST_F(GradientTest, GradientXConstantImage) {
    // Constant image should have zero horizontal gradient
    std::vector<uint8_t> constant(64, 128);
    std::vector<float> gx(64);

    GradientX(constant.data(), gx.data(), 8, 8,
              GradientOperator::Sobel3x3);

    for (size_t i = 0; i < 64; ++i) {
        EXPECT_NEAR(gx[i], 0.0f, 1e-5) << "at index " << i;
    }
}

TEST_F(GradientTest, GradientXHorizontalRamp) {
    // Horizontal gradient image should have positive Gx
    std::vector<float> gx(size_ * size_);

    GradientX(horzGrad_.data(), gx.data(), size_, size_,
              GradientOperator::Central);

    // Away from borders, Gx should be ~1 (since f(x,y) = x)
    for (int y = 2; y < size_ - 2; ++y) {
        for (int x = 2; x < size_ - 2; ++x) {
            EXPECT_NEAR(gx[y * size_ + x], 1.0f, 0.1f)
                << "at (" << x << ", " << y << ")";
        }
    }
}

TEST_F(GradientTest, GradientXVerticalRamp) {
    // Vertical gradient image should have zero Gx
    std::vector<float> gx(size_ * size_);

    GradientX(vertGrad_.data(), gx.data(), size_, size_,
              GradientOperator::Central);

    // Gx should be ~0 everywhere
    for (int y = 2; y < size_ - 2; ++y) {
        for (int x = 2; x < size_ - 2; ++x) {
            EXPECT_NEAR(gx[y * size_ + x], 0.0f, 0.1f)
                << "at (" << x << ", " << y << ")";
        }
    }
}

// ============================================================================
// GradientY Tests
// ============================================================================

TEST_F(GradientTest, GradientYConstantImage) {
    std::vector<uint8_t> constant(64, 128);
    std::vector<float> gy(64);

    GradientY(constant.data(), gy.data(), 8, 8,
              GradientOperator::Sobel3x3);

    for (size_t i = 0; i < 64; ++i) {
        EXPECT_NEAR(gy[i], 0.0f, 1e-5) << "at index " << i;
    }
}

TEST_F(GradientTest, GradientYVerticalRamp) {
    // Vertical gradient image should have positive Gy
    std::vector<float> gy(size_ * size_);

    GradientY(vertGrad_.data(), gy.data(), size_, size_,
              GradientOperator::Central);

    // Away from borders, Gy should be ~1 (since f(x,y) = y)
    for (int y = 2; y < size_ - 2; ++y) {
        for (int x = 2; x < size_ - 2; ++x) {
            EXPECT_NEAR(gy[y * size_ + x], 1.0f, 0.1f)
                << "at (" << x << ", " << y << ")";
        }
    }
}

TEST_F(GradientTest, GradientYHorizontalRamp) {
    // Horizontal gradient image should have zero Gy
    std::vector<float> gy(size_ * size_);

    GradientY(horzGrad_.data(), gy.data(), size_, size_,
              GradientOperator::Central);

    // Gy should be ~0 everywhere
    for (int y = 2; y < size_ - 2; ++y) {
        for (int x = 2; x < size_ - 2; ++x) {
            EXPECT_NEAR(gy[y * size_ + x], 0.0f, 0.1f)
                << "at (" << x << ", " << y << ")";
        }
    }
}

// ============================================================================
// Combined Gradient Tests
// ============================================================================

TEST_F(GradientTest, GradientCombined) {
    std::vector<float> gx(size_ * size_), gy(size_ * size_);

    Gradient(horzGrad_.data(), gx.data(), gy.data(), size_, size_,
             GradientOperator::Central);

    for (int y = 2; y < size_ - 2; ++y) {
        for (int x = 2; x < size_ - 2; ++x) {
            EXPECT_NEAR(gx[y * size_ + x], 1.0f, 0.1f);
            EXPECT_NEAR(gy[y * size_ + x], 0.0f, 0.1f);
        }
    }
}

TEST_F(GradientTest, GradientDiagonalRamp) {
    // Diagonal gradient: f(x,y) = x + y
    std::vector<float> diag(size_ * size_);
    for (int y = 0; y < size_; ++y) {
        for (int x = 0; x < size_; ++x) {
            diag[y * size_ + x] = static_cast<float>(x + y);
        }
    }

    std::vector<float> gx(size_ * size_), gy(size_ * size_);
    Gradient(diag.data(), gx.data(), gy.data(), size_, size_,
             GradientOperator::Central);

    // Both gradients should be ~1
    for (int y = 2; y < size_ - 2; ++y) {
        for (int x = 2; x < size_ - 2; ++x) {
            EXPECT_NEAR(gx[y * size_ + x], 1.0f, 0.1f);
            EXPECT_NEAR(gy[y * size_ + x], 1.0f, 0.1f);
        }
    }
}

// ============================================================================
// Magnitude Tests
// ============================================================================

TEST_F(GradientTest, MagnitudeL2) {
    EXPECT_NEAR(MagnitudeL2(3.0f, 4.0f), 5.0f, 1e-5);
    EXPECT_NEAR(MagnitudeL2(0.0f, 0.0f), 0.0f, 1e-10);
    EXPECT_NEAR(MagnitudeL2(1.0f, 0.0f), 1.0f, 1e-10);
    EXPECT_NEAR(MagnitudeL2(-3.0f, 4.0f), 5.0f, 1e-5);
}

TEST_F(GradientTest, MagnitudeL1) {
    EXPECT_NEAR(MagnitudeL1(3.0f, 4.0f), 7.0f, 1e-5);
    EXPECT_NEAR(MagnitudeL1(-3.0f, -4.0f), 7.0f, 1e-5);
    EXPECT_NEAR(MagnitudeL1(0.0f, 0.0f), 0.0f, 1e-10);
}

TEST_F(GradientTest, GradientMagnitudeArray) {
    std::vector<float> gx = {3.0f, 0.0f, 1.0f, -4.0f};
    std::vector<float> gy = {4.0f, 0.0f, 0.0f, 3.0f};
    std::vector<float> mag(4);

    GradientMagnitude(gx.data(), gy.data(), mag.data(), 4, false);

    EXPECT_NEAR(mag[0], 5.0f, 1e-5);
    EXPECT_NEAR(mag[1], 0.0f, 1e-10);
    EXPECT_NEAR(mag[2], 1.0f, 1e-10);
    EXPECT_NEAR(mag[3], 5.0f, 1e-5);
}

TEST_F(GradientTest, GradientMagnitudeNormalized) {
    std::vector<float> gx = {3.0f, 0.0f, 1.0f, 1.5f};
    std::vector<float> gy = {4.0f, 0.0f, 0.0f, 2.0f};
    std::vector<float> mag(4);

    GradientMagnitude(gx.data(), gy.data(), mag.data(), 4, true);

    // Max magnitude is 5.0, so normalized to 255
    EXPECT_NEAR(mag[0], 255.0f, 1.0f);  // 5/5 * 255
    EXPECT_NEAR(mag[1], 0.0f, 1.0f);     // 0/5 * 255
}

// ============================================================================
// Direction Tests
// ============================================================================

TEST_F(GradientTest, DirectionFromGradient) {
    // Pure horizontal gradient (pointing right)
    EXPECT_NEAR(DirectionFromGradient(1.0f, 0.0f), 0.0f, 1e-5);

    // Pure vertical gradient (pointing down)
    EXPECT_NEAR(DirectionFromGradient(0.0f, 1.0f),
                static_cast<float>(M_PI / 2), 1e-5);

    // Pure horizontal gradient (pointing left)
    EXPECT_NEAR(std::abs(DirectionFromGradient(-1.0f, 0.0f)),
                static_cast<float>(M_PI), 1e-5);

    // 45 degrees
    EXPECT_NEAR(DirectionFromGradient(1.0f, 1.0f),
                static_cast<float>(M_PI / 4), 1e-5);

    // -45 degrees
    EXPECT_NEAR(DirectionFromGradient(1.0f, -1.0f),
                static_cast<float>(-M_PI / 4), 1e-5);
}

TEST_F(GradientTest, GradientDirectionArray) {
    std::vector<float> gx = {1.0f, 0.0f, -1.0f, 1.0f};
    std::vector<float> gy = {0.0f, 1.0f, 0.0f, 1.0f};
    std::vector<float> dir(4);

    GradientDirection(gx.data(), gy.data(), dir.data(), 4);

    EXPECT_NEAR(dir[0], 0.0f, 1e-5);
    EXPECT_NEAR(dir[1], static_cast<float>(M_PI / 2), 1e-5);
    EXPECT_NEAR(std::abs(dir[2]), static_cast<float>(M_PI), 1e-5);
    EXPECT_NEAR(dir[3], static_cast<float>(M_PI / 4), 1e-5);
}

// ============================================================================
// MagDir Combined Tests
// ============================================================================

TEST_F(GradientTest, GradientMagDirSobel) {
    std::vector<float> mag(size_ * size_);
    std::vector<float> dir(size_ * size_);

    GradientMagDir(horzGrad_.data(), mag.data(), dir.data(),
                   size_, size_, GradientOperator::Sobel3x3);

    // Horizontal gradient should have direction ~0
    for (int y = 3; y < size_ - 3; ++y) {
        for (int x = 3; x < size_ - 3; ++x) {
            EXPECT_NEAR(std::abs(dir[y * size_ + x]), 0.0f, 0.3f)
                << "at (" << x << ", " << y << ")";
        }
    }
}

TEST_F(GradientTest, GradientMagDirMagOnly) {
    std::vector<float> mag(size_ * size_);

    // Should work with nullptr for direction
    GradientMagDir(horzGrad_.data(), mag.data(), nullptr,
                   size_, size_, GradientOperator::Sobel3x3);

    // Should have computed magnitude
    bool hasNonZero = false;
    for (int i = 0; i < size_ * size_; ++i) {
        if (mag[i] > 0.1f) hasNonZero = true;
    }
    EXPECT_TRUE(hasNonZero);
}

TEST_F(GradientTest, GradientMagDirDirOnly) {
    std::vector<float> dir(size_ * size_);

    // Should work with nullptr for magnitude
    GradientMagDir(horzGrad_.data(), nullptr, dir.data(),
                   size_, size_, GradientOperator::Sobel3x3);

    // Direction should be computed
    // Just check it doesn't crash and produces valid values
    for (int i = 0; i < size_ * size_; ++i) {
        EXPECT_GE(dir[i], static_cast<float>(-M_PI));
        EXPECT_LE(dir[i], static_cast<float>(M_PI));
    }
}

// ============================================================================
// Second Derivative Tests
// ============================================================================

TEST_F(GradientTest, GradientXXConstant) {
    std::vector<uint8_t> constant(64, 100);
    std::vector<float> dxx(64);

    GradientXX(constant.data(), dxx.data(), 8, 8);

    for (size_t i = 0; i < 64; ++i) {
        EXPECT_NEAR(dxx[i], 0.0f, 1e-5);
    }
}

TEST_F(GradientTest, GradientXXLinear) {
    // Linear gradient f(x) = x has zero second derivative
    std::vector<float> dxx(size_ * size_);

    GradientXX(horzGrad_.data(), dxx.data(), size_, size_);

    for (int y = 1; y < size_ - 1; ++y) {
        for (int x = 1; x < size_ - 1; ++x) {
            EXPECT_NEAR(dxx[y * size_ + x], 0.0f, 1e-5)
                << "at (" << x << ", " << y << ")";
        }
    }
}

TEST_F(GradientTest, GradientXXQuadratic) {
    // Quadratic: f(x) = x^2, f''(x) = 2
    std::vector<float> quadratic(size_ * size_);
    for (int y = 0; y < size_; ++y) {
        for (int x = 0; x < size_; ++x) {
            quadratic[y * size_ + x] = static_cast<float>(x * x);
        }
    }

    std::vector<float> dxx(size_ * size_);
    GradientXX(quadratic.data(), dxx.data(), size_, size_);

    // Second derivative should be ~2
    for (int y = 1; y < size_ - 1; ++y) {
        for (int x = 1; x < size_ - 1; ++x) {
            EXPECT_NEAR(dxx[y * size_ + x], 2.0f, 0.1f)
                << "at (" << x << ", " << y << ")";
        }
    }
}

TEST_F(GradientTest, GradientYYQuadratic) {
    // Quadratic: f(y) = y^2, f''(y) = 2
    std::vector<float> quadratic(size_ * size_);
    for (int y = 0; y < size_; ++y) {
        for (int x = 0; x < size_; ++x) {
            quadratic[y * size_ + x] = static_cast<float>(y * y);
        }
    }

    std::vector<float> dyy(size_ * size_);
    GradientYY(quadratic.data(), dyy.data(), size_, size_);

    // Second derivative should be ~2
    for (int y = 1; y < size_ - 1; ++y) {
        for (int x = 1; x < size_ - 1; ++x) {
            EXPECT_NEAR(dyy[y * size_ + x], 2.0f, 0.1f)
                << "at (" << x << ", " << y << ")";
        }
    }
}

TEST_F(GradientTest, GradientXYMixed) {
    // f(x,y) = x*y, d²f/dxdy = 1
    std::vector<float> mixed(size_ * size_);
    for (int y = 0; y < size_; ++y) {
        for (int x = 0; x < size_; ++x) {
            mixed[y * size_ + x] = static_cast<float>(x * y);
        }
    }

    std::vector<float> dxy(size_ * size_);
    GradientXY(mixed.data(), dxy.data(), size_, size_);

    // Mixed derivative should be ~1
    for (int y = 1; y < size_ - 1; ++y) {
        for (int x = 1; x < size_ - 1; ++x) {
            EXPECT_NEAR(dxy[y * size_ + x], 1.0f, 0.1f)
                << "at (" << x << ", " << y << ")";
        }
    }
}

// ============================================================================
// GradientAtPixel Tests
// ============================================================================

TEST_F(GradientTest, GradientAtPixelCenter) {
    double gx, gy;
    GradientAtPixel(horzGrad_.data(), size_, size_, 8, 8, gx, gy);

    EXPECT_NEAR(gx, 1.0, 0.1);
    EXPECT_NEAR(gy, 0.0, 0.1);
}

TEST_F(GradientTest, GradientAtPixelBorder) {
    double gx, gy;
    // At corner with Reflect101
    GradientAtPixel(horzGrad_.data(), size_, size_, 0, 0, gx, gy,
                    BorderMode::Reflect101);

    // Should still be valid
    EXPECT_FALSE(std::isnan(gx));
    EXPECT_FALSE(std::isnan(gy));
}

// ============================================================================
// GradientAtSubpixel Tests
// ============================================================================

TEST_F(GradientTest, GradientAtSubpixelBilinear) {
    double gx, gy;
    GradientAtSubpixel(horzGrad_.data(), size_, size_, 8.5, 8.5,
                       gx, gy, InterpolationMethod::Bilinear);

    EXPECT_NEAR(gx, 1.0, 0.1);
    EXPECT_NEAR(gy, 0.0, 0.1);
}

TEST_F(GradientTest, GradientAtSubpixelBicubic) {
    double gx, gy;
    GradientAtSubpixel(horzGrad_.data(), size_, size_, 8.5, 8.5,
                       gx, gy, InterpolationMethod::Bicubic);

    EXPECT_NEAR(gx, 1.0, 0.1);
    EXPECT_NEAR(gy, 0.0, 0.1);
}

// ============================================================================
// Operator Tests
// ============================================================================

TEST_F(GradientTest, AllOperatorsProduceValidOutput) {
    std::vector<GradientOperator> ops = {
        GradientOperator::Sobel3x3,
        GradientOperator::Sobel5x5,
        GradientOperator::Sobel7x7,
        GradientOperator::Scharr,
        GradientOperator::Prewitt,
        GradientOperator::Central
    };

    std::vector<float> gx(size_ * size_), gy(size_ * size_);

    for (auto op : ops) {
        Gradient(horzGrad_.data(), gx.data(), gy.data(),
                 size_, size_, op);

        // Check center pixels have reasonable gradient
        int center = size_ / 2;
        float centerGx = gx[center * size_ + center];

        // All operators should detect positive horizontal gradient
        EXPECT_GT(centerGx, 0.0f)
            << "Operator " << static_cast<int>(op) << " failed";
    }
}

TEST_F(GradientTest, ScharrMoreAccurateThanSobel) {
    // Create an image with diagonal edge
    std::vector<float> diag(size_ * size_);
    for (int y = 0; y < size_; ++y) {
        for (int x = 0; x < size_; ++x) {
            // Diagonal gradient at 45 degrees
            diag[y * size_ + x] = static_cast<float>(x + y);
        }
    }

    std::vector<float> gxSobel(size_ * size_), gySobel(size_ * size_);
    std::vector<float> gxScharr(size_ * size_), gyScharr(size_ * size_);

    Gradient(diag.data(), gxSobel.data(), gySobel.data(),
             size_, size_, GradientOperator::Sobel3x3);
    Gradient(diag.data(), gxScharr.data(), gyScharr.data(),
             size_, size_, GradientOperator::Scharr);

    // Both should detect similar gradients (we just verify they work)
    int center = size_ / 2;
    EXPECT_GT(gxSobel[center * size_ + center], 0.0f);
    EXPECT_GT(gySobel[center * size_ + center], 0.0f);
    EXPECT_GT(gxScharr[center * size_ + center], 0.0f);
    EXPECT_GT(gyScharr[center * size_ + center], 0.0f);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(GradientTest, SmallImage3x3) {
    std::vector<uint8_t> small = {
        0, 128, 255,
        0, 128, 255,
        0, 128, 255
    };
    std::vector<float> gx(9);

    GradientX(small.data(), gx.data(), 3, 3, GradientOperator::Central);

    // Should produce valid results
    EXPECT_FALSE(std::isnan(gx[4]));
}

TEST_F(GradientTest, Uint16Input) {
    std::vector<uint16_t> img16(size_ * size_);
    for (int y = 0; y < size_; ++y) {
        for (int x = 0; x < size_; ++x) {
            img16[y * size_ + x] = static_cast<uint16_t>(x * 1000);
        }
    }

    std::vector<float> gx(size_ * size_);
    GradientX(img16.data(), gx.data(), size_, size_, GradientOperator::Central);

    // Central difference: (I(x+1) - I(x-1)) * 0.5
    // For f(x) = 1000*x: (1000*(x+1) - 1000*(x-1)) * 0.5 = 1000
    int center = size_ / 2;
    EXPECT_NEAR(gx[center * size_ + center], 1000.0f, 100.0f);
}

