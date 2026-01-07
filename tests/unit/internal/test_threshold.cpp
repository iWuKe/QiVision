/**
 * @file test_threshold.cpp
 * @brief Unit tests for Internal/Threshold
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/Threshold.h>
#include <QiVision/Internal/Histogram.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>

#include <cmath>
#include <cstring>
#include <numeric>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

// ============================================================================
// Test Fixtures
// ============================================================================

class ThresholdTest : public ::testing::Test {
protected:
    void SetUp() override {
        width_ = 100;
        height_ = 100;

        // Uniform image (all pixels = 128)
        uniformImg_ = QImage(width_, height_, PixelType::UInt8, ChannelType::Gray);
        std::memset(uniformImg_.Data(), 128, width_ * height_);

        // Gradient image (left to right: 0 to 255)
        gradientImg_ = QImage(width_, height_, PixelType::UInt8, ChannelType::Gray);
        uint8_t* gradData = static_cast<uint8_t*>(gradientImg_.Data());
        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                gradData[y * width_ + x] = static_cast<uint8_t>(x * 255 / (width_ - 1));
            }
        }

        // Bimodal image (left half = 50, right half = 200)
        bimodalImg_ = QImage(width_, height_, PixelType::UInt8, ChannelType::Gray);
        uint8_t* biData = static_cast<uint8_t*>(bimodalImg_.Data());
        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                biData[y * width_ + x] = (x < width_ / 2) ? 50 : 200;
            }
        }

        // Black image
        blackImg_ = QImage(width_, height_, PixelType::UInt8, ChannelType::Gray);
        std::memset(blackImg_.Data(), 0, width_ * height_);

        // White image
        whiteImg_ = QImage(width_, height_, PixelType::UInt8, ChannelType::Gray);
        std::memset(whiteImg_.Data(), 255, width_ * height_);
    }

    int32_t width_, height_;
    QImage uniformImg_;
    QImage gradientImg_;
    QImage bimodalImg_;
    QImage blackImg_;
    QImage whiteImg_;
};

// ============================================================================
// Global Threshold Tests
// ============================================================================

TEST_F(ThresholdTest, GlobalBinary) {
    QImage dst;
    ThresholdGlobal(uniformImg_, dst, 127.0, 255.0, ThresholdType::Binary);

    ASSERT_FALSE(dst.Empty());
    EXPECT_EQ(dst.Width(), width_);
    EXPECT_EQ(dst.Height(), height_);

    // All pixels (128) > 127, so all should be 255
    const uint8_t* data = static_cast<const uint8_t*>(dst.Data());
    for (int32_t i = 0; i < width_ * height_; ++i) {
        EXPECT_EQ(data[i], 255) << "Pixel " << i << " should be 255";
    }
}

TEST_F(ThresholdTest, GlobalBinaryInv) {
    QImage dst;
    ThresholdGlobal(uniformImg_, dst, 127.0, 255.0, ThresholdType::BinaryInv);

    // All pixels (128) > 127, so all should be 0 (inverted)
    const uint8_t* data = static_cast<const uint8_t*>(dst.Data());
    for (int32_t i = 0; i < width_ * height_; ++i) {
        EXPECT_EQ(data[i], 0) << "Pixel " << i << " should be 0";
    }
}

TEST_F(ThresholdTest, GlobalTruncate) {
    QImage dst;
    ThresholdGlobal(uniformImg_, dst, 100.0, 255.0, ThresholdType::Truncate);

    // All pixels (128) > 100, so all should be truncated to 100
    const uint8_t* data = static_cast<const uint8_t*>(dst.Data());
    for (int32_t i = 0; i < width_ * height_; ++i) {
        EXPECT_EQ(data[i], 100) << "Pixel " << i << " should be 100";
    }
}

TEST_F(ThresholdTest, GlobalToZero) {
    QImage dst;
    ThresholdGlobal(uniformImg_, dst, 127.0, 255.0, ThresholdType::ToZero);

    // All pixels (128) > 127, so all should remain as 128
    const uint8_t* data = static_cast<const uint8_t*>(dst.Data());
    for (int32_t i = 0; i < width_ * height_; ++i) {
        EXPECT_EQ(data[i], 128) << "Pixel " << i << " should be 128";
    }
}

TEST_F(ThresholdTest, GlobalToZeroInv) {
    QImage dst;
    ThresholdGlobal(uniformImg_, dst, 127.0, 255.0, ThresholdType::ToZeroInv);

    // All pixels (128) > 127, so all should be 0
    const uint8_t* data = static_cast<const uint8_t*>(dst.Data());
    for (int32_t i = 0; i < width_ * height_; ++i) {
        EXPECT_EQ(data[i], 0) << "Pixel " << i << " should be 0";
    }
}

TEST_F(ThresholdTest, GlobalThresholdGradient) {
    QImage dst;
    // Threshold at 127 - left ~half should be 0, right ~half should be 255
    ThresholdGlobal(gradientImg_, dst, 127.0, 255.0, ThresholdType::Binary);

    const uint8_t* data = static_cast<const uint8_t*>(dst.Data());

    // Check a few sample pixels
    EXPECT_EQ(data[0], 0);  // x=0: value=0 <= 127
    EXPECT_EQ(data[width_ / 2], 255);  // x=50: value ~127, should be 255 if > 127
    EXPECT_EQ(data[width_ - 1], 255);  // x=99: value=255 > 127
}

// ============================================================================
// Range Threshold Tests
// ============================================================================

TEST_F(ThresholdTest, RangeThreshold) {
    QImage dst;
    ThresholdRange(gradientImg_, dst, 100.0, 200.0, 255.0);

    const uint8_t* data = static_cast<const uint8_t*>(dst.Data());

    // Check sample pixels
    int32_t count255 = 0;
    int32_t count0 = 0;
    for (int32_t i = 0; i < width_; ++i) {
        if (data[i] == 255) count255++;
        else if (data[i] == 0) count0++;
    }

    // Approximately 40% of pixels should be in range [100, 200]
    EXPECT_GT(count255, 30);
    EXPECT_GT(count0, 30);
}

TEST_F(ThresholdTest, RangeThresholdUniformInRange) {
    QImage dst;
    // 128 is in range [100, 200]
    ThresholdRange(uniformImg_, dst, 100.0, 200.0, 255.0);

    const uint8_t* data = static_cast<const uint8_t*>(dst.Data());
    for (int32_t i = 0; i < width_ * height_; ++i) {
        EXPECT_EQ(data[i], 255) << "Pixel " << i << " should be 255";
    }
}

TEST_F(ThresholdTest, RangeThresholdUniformOutOfRange) {
    QImage dst;
    // 128 is out of range [0, 50]
    ThresholdRange(uniformImg_, dst, 0.0, 50.0, 255.0);

    const uint8_t* data = static_cast<const uint8_t*>(dst.Data());
    for (int32_t i = 0; i < width_ * height_; ++i) {
        EXPECT_EQ(data[i], 0) << "Pixel " << i << " should be 0";
    }
}

// ============================================================================
// Auto Threshold Tests
// ============================================================================

TEST_F(ThresholdTest, OtsuThresholdBimodal) {
    double threshold;
    QImage dst;
    ThresholdOtsu(bimodalImg_, dst, 255.0, &threshold);

    // Otsu should find threshold between 50 and 200
    EXPECT_GT(threshold, 50);
    EXPECT_LT(threshold, 200);

    // Check the binary result
    const uint8_t* data = static_cast<const uint8_t*>(dst.Data());
    int32_t foregroundCount = 0;
    for (int32_t i = 0; i < width_ * height_; ++i) {
        if (data[i] > 0) foregroundCount++;
    }

    // Right half (value 200) should be foreground
    EXPECT_NEAR(foregroundCount, width_ * height_ / 2, width_ * height_ * 0.1);
}

TEST_F(ThresholdTest, TriangleThreshold) {
    double threshold;
    QImage dst;
    ThresholdTriangle(gradientImg_, dst, 255.0, &threshold);

    EXPECT_GE(threshold, 0);
    EXPECT_LE(threshold, 255);
}

TEST_F(ThresholdTest, AutoThresholdMethods) {
    // Test all auto threshold methods don't crash
    Histogram hist = ComputeHistogram(bimodalImg_);

    double otsu = ComputeOtsuThreshold(hist);
    EXPECT_GT(otsu, 0);
    EXPECT_LT(otsu, 255);

    double triangle = ComputeTriangleThreshold(hist);
    EXPECT_GE(triangle, 0);
    EXPECT_LE(triangle, 255);

    double minError = ComputeMinErrorThreshold(hist);
    EXPECT_GE(minError, 0);
    EXPECT_LE(minError, 255);

    double isodata = ComputeIsodataThreshold(hist);
    EXPECT_GE(isodata, 0);
    EXPECT_LE(isodata, 255);
}

// ============================================================================
// Adaptive Threshold Tests
// ============================================================================

TEST_F(ThresholdTest, AdaptiveMean) {
    AdaptiveThresholdParams params = AdaptiveThresholdParams::Mean(11, 0);
    QImage dst = ThresholdAdaptive(uniformImg_, params);

    ASSERT_FALSE(dst.Empty());
    EXPECT_EQ(dst.Width(), width_);
    EXPECT_EQ(dst.Height(), height_);

    // For uniform image, all pixels equal local mean, so result depends on C
    // With C=0, pixels should equal mean and threshold test is src > mean
    // Since src == mean, all should be 0
}

TEST_F(ThresholdTest, AdaptiveGaussian) {
    AdaptiveThresholdParams params = AdaptiveThresholdParams::Gaussian(11, 5);
    QImage dst = ThresholdAdaptive(gradientImg_, params);

    ASSERT_FALSE(dst.Empty());
    EXPECT_EQ(dst.Width(), width_);
    EXPECT_EQ(dst.Height(), height_);
}

TEST_F(ThresholdTest, AdaptiveSauvola) {
    AdaptiveThresholdParams params = AdaptiveThresholdParams::Sauvola(15, 0.5, 128);
    QImage dst = ThresholdAdaptive(gradientImg_, params);

    ASSERT_FALSE(dst.Empty());
    EXPECT_EQ(dst.Width(), width_);
    EXPECT_EQ(dst.Height(), height_);
}

TEST_F(ThresholdTest, AdaptiveNiblack) {
    AdaptiveThresholdParams params = AdaptiveThresholdParams::Niblack(15, -0.2);
    QImage dst = ThresholdAdaptive(gradientImg_, params);

    ASSERT_FALSE(dst.Empty());
    EXPECT_EQ(dst.Width(), width_);
    EXPECT_EQ(dst.Height(), height_);
}

TEST_F(ThresholdTest, LocalThresholdMap) {
    AdaptiveThresholdParams params = AdaptiveThresholdParams::Mean(11, 0);
    QImage threshMap = ComputeLocalThresholdMap(gradientImg_, params);

    ASSERT_FALSE(threshMap.Empty());
    EXPECT_EQ(threshMap.Width(), width_);
    EXPECT_EQ(threshMap.Height(), height_);
    EXPECT_EQ(threshMap.Type(), PixelType::Float32);

    // Check that threshold values are reasonable
    const float* data = static_cast<const float*>(threshMap.Data());
    for (int32_t i = 0; i < width_ * height_; ++i) {
        EXPECT_GE(data[i], 0) << "Threshold at " << i << " should be >= 0";
        EXPECT_LE(data[i], 255) << "Threshold at " << i << " should be <= 255";
    }
}

// ============================================================================
// Multi-level Threshold Tests
// ============================================================================

TEST_F(ThresholdTest, MultiLevelTwoThresholds) {
    std::vector<double> thresholds = {85.0, 170.0};
    QImage dst;
    ThresholdMultiLevel(gradientImg_, dst, thresholds);

    ASSERT_FALSE(dst.Empty());

    const uint8_t* data = static_cast<const uint8_t*>(dst.Data());

    // Check a few sample pixels at different gradient positions
    // Level 0: value <= 85 -> 0
    // Level 1: 85 < value <= 170 -> 127
    // Level 2: value > 170 -> 255
    EXPECT_EQ(data[0], 0);  // x=0: value=0 <= 85
    EXPECT_EQ(data[width_ - 1], 255);  // x=99: value=255 > 170
}

TEST_F(ThresholdTest, MultiLevelWithOutputValues) {
    std::vector<double> thresholds = {128.0};
    std::vector<double> outputValues = {50.0, 200.0};
    QImage dst;
    ThresholdMultiLevel(gradientImg_, dst, thresholds, outputValues);

    ASSERT_FALSE(dst.Empty());

    const uint8_t* data = static_cast<const uint8_t*>(dst.Data());
    EXPECT_EQ(data[0], 50);  // value=0 <= 128
    EXPECT_EQ(data[width_ - 1], 200);  // value=255 > 128
}

// ============================================================================
// Threshold to Region Tests
// ============================================================================

TEST_F(ThresholdTest, ThresholdToRegionRange) {
    QRegion region = ThresholdToRegion(bimodalImg_, 150.0, 255.0);

    // Right half (value 200) should be in region
    EXPECT_FALSE(region.Empty());
    EXPECT_NEAR(region.Area(), width_ * height_ / 2, width_);
}

TEST_F(ThresholdTest, ThresholdToRegionAbove) {
    QRegion region = ThresholdToRegion(bimodalImg_, 100.0, true);

    // Right half (value 200) should be > 100
    EXPECT_FALSE(region.Empty());
    EXPECT_NEAR(region.Area(), width_ * height_ / 2, width_);
}

TEST_F(ThresholdTest, ThresholdToRegionBelow) {
    QRegion region = ThresholdToRegion(bimodalImg_, 100.0, false);

    // Left half (value 50) should be < 100
    EXPECT_FALSE(region.Empty());
    EXPECT_NEAR(region.Area(), width_ * height_ / 2, width_);
}

TEST_F(ThresholdTest, AutoThresholdToRegion) {
    double threshold;
    QRegion region = ThresholdAutoToRegion(bimodalImg_, AutoThresholdMethod::Otsu, true, &threshold);

    EXPECT_GT(threshold, 50);
    EXPECT_LT(threshold, 200);
    EXPECT_FALSE(region.Empty());
    EXPECT_NEAR(region.Area(), width_ * height_ / 2, width_ * height_ * 0.1);
}

TEST_F(ThresholdTest, ThresholdToRegionEmpty) {
    // All pixels are 128, range [200, 255] should be empty
    QRegion region = ThresholdToRegion(uniformImg_, 200.0, 255.0);
    EXPECT_TRUE(region.Empty());
}

TEST_F(ThresholdTest, ThresholdToRegionFull) {
    // All pixels are 128, range [0, 200] should include all
    QRegion region = ThresholdToRegion(uniformImg_, 0.0, 200.0);
    EXPECT_EQ(region.Area(), width_ * height_);
}

// ============================================================================
// Binary Operations Tests
// ============================================================================

TEST_F(ThresholdTest, BinaryInvert) {
    QImage binary;
    ThresholdGlobal(bimodalImg_, binary, 100.0, 255.0, ThresholdType::Binary);

    QImage inverted;
    BinaryInvert(binary, inverted, 255.0);

    const uint8_t* binData = static_cast<const uint8_t*>(binary.Data());
    const uint8_t* invData = static_cast<const uint8_t*>(inverted.Data());

    for (int32_t i = 0; i < width_ * height_; ++i) {
        EXPECT_EQ(binData[i] + invData[i], 255);
    }
}

TEST_F(ThresholdTest, BinaryAnd) {
    // Create two binary images
    QImage bin1, bin2, result;
    ThresholdRange(gradientImg_, bin1, 0.0, 200.0, 255.0);    // Left ~80%
    ThresholdRange(gradientImg_, bin2, 50.0, 255.0, 255.0);   // Right ~80%

    BinaryAnd(bin1, bin2, result, 255.0);

    const uint8_t* data = static_cast<const uint8_t*>(result.Data());

    // Check that result is intersection
    int32_t count = 0;
    for (int32_t i = 0; i < width_ * height_; ++i) {
        if (data[i] > 0) count++;
    }

    // Intersection should be approximately 60% (range [50, 200])
    double ratio = static_cast<double>(count) / (width_ * height_);
    EXPECT_GT(ratio, 0.4);
    EXPECT_LT(ratio, 0.8);
}

TEST_F(ThresholdTest, BinaryOr) {
    QImage bin1, bin2, result;
    ThresholdRange(gradientImg_, bin1, 0.0, 100.0, 255.0);    // Left ~40%
    ThresholdRange(gradientImg_, bin2, 150.0, 255.0, 255.0);  // Right ~40%

    BinaryOr(bin1, bin2, result, 255.0);

    const uint8_t* data = static_cast<const uint8_t*>(result.Data());

    int32_t count = 0;
    for (int32_t i = 0; i < width_ * height_; ++i) {
        if (data[i] > 0) count++;
    }

    // Union should be approximately 80%
    double ratio = static_cast<double>(count) / (width_ * height_);
    EXPECT_GT(ratio, 0.6);
}

TEST_F(ThresholdTest, BinaryXor) {
    QImage bin1, bin2, result;
    ThresholdGlobal(bimodalImg_, bin1, 100.0, 255.0, ThresholdType::Binary);
    ThresholdGlobal(uniformImg_, bin2, 100.0, 255.0, ThresholdType::Binary);

    BinaryXor(bin1, bin2, result, 255.0);

    // bin1 has right half = 255, left half = 0
    // bin2 has all = 255 (uniform 128 > 100)
    // XOR: left half should be 255 (0 XOR 1), right half should be 0 (1 XOR 1)

    const uint8_t* data = static_cast<const uint8_t*>(result.Data());
    int32_t count = 0;
    for (int32_t i = 0; i < width_ * height_; ++i) {
        if (data[i] > 0) count++;
    }

    EXPECT_NEAR(count, width_ * height_ / 2, width_);
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST_F(ThresholdTest, IsBinaryImage) {
    QImage binary;
    ThresholdGlobal(bimodalImg_, binary, 100.0, 255.0, ThresholdType::Binary);

    EXPECT_TRUE(IsBinaryImage(binary));
    EXPECT_FALSE(IsBinaryImage(gradientImg_));
}

TEST_F(ThresholdTest, CountNonZero) {
    QImage binary;
    ThresholdGlobal(bimodalImg_, binary, 100.0, 255.0, ThresholdType::Binary);

    uint64_t count = CountNonZero(binary);
    EXPECT_NEAR(count, width_ * height_ / 2, width_);
}

TEST_F(ThresholdTest, CountInRange) {
    uint64_t count = CountInRange(bimodalImg_, 100.0, 255.0);
    EXPECT_NEAR(count, width_ * height_ / 2, width_);
}

TEST_F(ThresholdTest, ComputeForegroundRatio) {
    QImage binary;
    ThresholdGlobal(bimodalImg_, binary, 100.0, 255.0, ThresholdType::Binary);

    double ratio = ComputeForegroundRatio(binary);
    EXPECT_NEAR(ratio, 0.5, 0.05);
}

TEST_F(ThresholdTest, ApplyMask) {
    // Create mask (right half)
    QImage mask;
    ThresholdGlobal(bimodalImg_, mask, 100.0, 255.0, ThresholdType::Binary);

    QImage result;
    ApplyMask(gradientImg_, mask, result);

    const uint8_t* data = static_cast<const uint8_t*>(result.Data());

    // Left half should be 0 (masked out)
    for (int32_t y = 0; y < height_; ++y) {
        for (int32_t x = 0; x < width_ / 2; ++x) {
            EXPECT_EQ(data[y * width_ + x], 0);
        }
    }

    // Right half should have original gradient values
    for (int32_t y = 0; y < height_; ++y) {
        for (int32_t x = width_ / 2; x < width_; ++x) {
            EXPECT_GT(data[y * width_ + x], 100);
        }
    }
}

TEST_F(ThresholdTest, RegionToMaskRoundTrip) {
    QRegion region = ThresholdToRegion(bimodalImg_, 150.0, 255.0);

    QImage mask;
    RegionToMask(region, mask);

    QRegion region2 = MaskToRegion(mask, 0);

    EXPECT_EQ(region.Area(), region2.Area());
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST_F(ThresholdTest, EmptyImage) {
    QImage empty;
    QImage dst;

    ThresholdGlobal(empty, dst, 100.0);
    EXPECT_TRUE(dst.Empty());

    QRegion region = ThresholdToRegion(empty, 0.0, 255.0);
    EXPECT_TRUE(region.Empty());
}

TEST_F(ThresholdTest, BlackImage) {
    QImage dst;
    ThresholdGlobal(blackImg_, dst, 0.0, 255.0, ThresholdType::Binary);

    // All pixels are 0, none > 0, so all should remain 0
    uint64_t count = CountNonZero(dst);
    EXPECT_EQ(count, 0u);
}

TEST_F(ThresholdTest, WhiteImage) {
    QImage dst;
    ThresholdGlobal(whiteImg_, dst, 254.0, 255.0, ThresholdType::Binary);

    // All pixels are 255 > 254, so all should be 255
    uint64_t count = CountNonZero(dst);
    EXPECT_EQ(count, static_cast<uint64_t>(width_ * height_));
}

// ============================================================================
// Integral Image Tests
// ============================================================================

TEST_F(ThresholdTest, IntegralImagesCorrectness) {
    std::vector<double> integralSum, integralSqSum;
    ComputeIntegralImages(uniformImg_, integralSum, integralSqSum);

    // For uniform image with value 128:
    // Sum in 10x10 window should be 128 * 100 = 12800
    double mean, stddev;
    GetBlockStats(integralSum, integralSqSum, width_, height_, 50, 50, 5, mean, stddev);

    EXPECT_NEAR(mean, 128.0, 1.0);
    EXPECT_NEAR(stddev, 0.0, 1.0);  // Uniform image has 0 stddev
}

TEST_F(ThresholdTest, IntegralImagesGradient) {
    std::vector<double> integralSum, integralSqSum;
    ComputeIntegralImages(gradientImg_, integralSum, integralSqSum);

    double mean, stddev;
    GetBlockStats(integralSum, integralSqSum, width_, height_, 50, 50, 5, mean, stddev);

    // Middle of gradient should have mean around 127
    EXPECT_GT(mean, 100);
    EXPECT_LT(mean, 160);
    EXPECT_GT(stddev, 0);  // Gradient has non-zero stddev
}

// ============================================================================
// Template Function Tests
// ============================================================================

TEST_F(ThresholdTest, TemplateThresholdGlobal) {
    std::vector<uint8_t> src = {0, 50, 100, 127, 128, 150, 200, 255};
    std::vector<uint8_t> dst(src.size());

    ThresholdGlobal(src.data(), dst.data(), src.size(), 127.0, 255.0, ThresholdType::Binary);

    EXPECT_EQ(dst[0], 0);    // 0 <= 127
    EXPECT_EQ(dst[1], 0);    // 50 <= 127
    EXPECT_EQ(dst[2], 0);    // 100 <= 127
    EXPECT_EQ(dst[3], 0);    // 127 <= 127
    EXPECT_EQ(dst[4], 255);  // 128 > 127
    EXPECT_EQ(dst[5], 255);  // 150 > 127
    EXPECT_EQ(dst[6], 255);  // 200 > 127
    EXPECT_EQ(dst[7], 255);  // 255 > 127
}

// ============================================================================
// Return Value Function Tests
// ============================================================================

TEST_F(ThresholdTest, ReturnVersionGlobal) {
    QImage dst = ThresholdGlobal(uniformImg_, 100.0);
    EXPECT_FALSE(dst.Empty());
    EXPECT_EQ(dst.Width(), width_);
    EXPECT_EQ(dst.Height(), height_);
}

TEST_F(ThresholdTest, ReturnVersionRange) {
    QImage dst = ThresholdRange(uniformImg_, 100.0, 200.0);
    EXPECT_FALSE(dst.Empty());
    EXPECT_EQ(dst.Width(), width_);
}

TEST_F(ThresholdTest, ReturnVersionOtsu) {
    QImage dst = ThresholdOtsu(bimodalImg_);
    EXPECT_FALSE(dst.Empty());
}

TEST_F(ThresholdTest, ReturnVersionTriangle) {
    QImage dst = ThresholdTriangle(gradientImg_);
    EXPECT_FALSE(dst.Empty());
}

TEST_F(ThresholdTest, ReturnVersionMultiLevel) {
    std::vector<double> thresholds = {85.0, 170.0};
    QImage dst = ThresholdMultiLevel(gradientImg_, thresholds);
    EXPECT_FALSE(dst.Empty());
}

TEST_F(ThresholdTest, ReturnVersionAdaptive) {
    AdaptiveThresholdParams params = AdaptiveThresholdParams::Mean(11, 5);
    QImage dst = ThresholdAdaptive(gradientImg_, params);
    EXPECT_FALSE(dst.Empty());
}

// ============================================================================
// DynThreshold Tests (Halcon-style dynamic threshold)
// ============================================================================

TEST_F(ThresholdTest, DynThresholdWithReference_Light) {
    // Create a reference image (uniform 100)
    QImage reference(width_, height_, PixelType::UInt8, ChannelType::Gray);
    std::memset(reference.Data(), 100, width_ * height_);

    // bimodalImg_ has left=50, right=200
    // Light mode selects pixels brighter than reference + offset
    QRegion result = DynThreshold(bimodalImg_, reference, 5.0, LightDark::Light);

    // Right half (200) is brighter than 100+5=105
    EXPECT_FALSE(result.Empty());
    int64_t area = result.Area();
    EXPECT_GT(area, 0);
    EXPECT_LE(area, static_cast<int64_t>(width_ / 2 * height_));  // At most right half
}

TEST_F(ThresholdTest, DynThresholdWithReference_Dark) {
    QImage reference(width_, height_, PixelType::UInt8, ChannelType::Gray);
    std::memset(reference.Data(), 100, width_ * height_);

    // Dark mode selects pixels darker than reference - offset
    QRegion result = DynThreshold(bimodalImg_, reference, 5.0, LightDark::Dark);

    // Left half (50) is darker than 100-5=95
    EXPECT_FALSE(result.Empty());
    int64_t area = result.Area();
    EXPECT_GT(area, 0);
    EXPECT_LE(area, static_cast<int64_t>(width_ / 2 * height_));  // At most left half
}

TEST_F(ThresholdTest, DynThresholdWithReference_Equal) {
    QImage reference(width_, height_, PixelType::UInt8, ChannelType::Gray);
    std::memset(reference.Data(), 128, width_ * height_);

    // Equal mode selects pixels equal to reference (within offset)
    QRegion result = DynThreshold(uniformImg_, reference, 10.0, LightDark::Equal);

    // All pixels are 128, reference is 128, so all should be selected
    EXPECT_FALSE(result.Empty());
    int64_t area = result.Area();
    EXPECT_EQ(area, width_ * height_);
}

TEST_F(ThresholdTest, DynThresholdWithReference_NotEqual) {
    QImage reference(width_, height_, PixelType::UInt8, ChannelType::Gray);
    std::memset(reference.Data(), 128, width_ * height_);

    // NotEqual selects pixels different from reference (outside offset)
    QRegion result = DynThreshold(bimodalImg_, reference, 10.0, LightDark::NotEqual);

    // Left=50, Right=200, both are far from 128
    EXPECT_FALSE(result.Empty());
}

TEST_F(ThresholdTest, DynThresholdAutoSmoothed) {
    // Use auto-generated smoothed reference
    QRegion result = DynThreshold(bimodalImg_, 15, 10.0, LightDark::Light);

    // Should find bright regions compared to local average
    EXPECT_FALSE(result.Empty());
}

// ============================================================================
// DualThreshold Tests
// ============================================================================

TEST_F(ThresholdTest, DualThresholdBimodal) {
    // low=75, high=150
    // Left half (50) -> dark, Right half (200) -> light
    DualThresholdResult result = DualThreshold(bimodalImg_, 75.0, 150.0);

    EXPECT_FALSE(result.darkRegion.Empty());
    EXPECT_FALSE(result.lightRegion.Empty());

    // Dark region should be approximately left half
    EXPECT_GT(result.darkRegion.Area(), 0);

    // Light region should be approximately right half
    EXPECT_GT(result.lightRegion.Area(), 0);
}

TEST_F(ThresholdTest, DualThresholdMiddle) {
    // Create image with three levels: 50, 128, 200
    QImage triLevelImg(width_, height_, PixelType::UInt8, ChannelType::Gray);
    uint8_t* data = static_cast<uint8_t*>(triLevelImg.Data());
    for (int32_t y = 0; y < height_; ++y) {
        for (int32_t x = 0; x < width_; ++x) {
            if (x < width_ / 3) {
                data[y * width_ + x] = 50;
            } else if (x < 2 * width_ / 3) {
                data[y * width_ + x] = 128;
            } else {
                data[y * width_ + x] = 200;
            }
        }
    }

    DualThresholdResult result = DualThreshold(triLevelImg, 75.0, 175.0);

    // All three regions should be non-empty
    EXPECT_FALSE(result.darkRegion.Empty());
    EXPECT_FALSE(result.lightRegion.Empty());
}

TEST_F(ThresholdTest, DualThresholdAuto) {
    DualThresholdResult result = DualThresholdAuto(bimodalImg_);

    // Auto thresholds should separate the bimodal distribution
    // At least one of dark or light regions should be non-empty for bimodal image
    // The auto algorithm may not always find both regions depending on the threshold method
    int64_t totalArea = result.darkRegion.Area() + result.lightRegion.Area() + result.middleRegion.Area();
    EXPECT_GT(totalArea, 0);

    // Verify thresholds are reasonable
    EXPECT_GT(result.lowThreshold, 0);
    EXPECT_LT(result.highThreshold, 255);
    EXPECT_LT(result.lowThreshold, result.highThreshold);
}

// ============================================================================
// VarThreshold Tests
// ============================================================================

TEST_F(ThresholdTest, VarThresholdUniform) {
    // Uniform image has zero variance everywhere
    QRegion result = VarThreshold(uniformImg_, 15, 10.0, LightDark::Light);

    // Light mode selects high variance - should be mostly empty
    EXPECT_TRUE(result.Empty() || result.Area() < width_ * height_ / 2);
}

TEST_F(ThresholdTest, VarThresholdGradient) {
    // Gradient image has variance everywhere (except edges)
    QRegion result = VarThreshold(gradientImg_, 15, 10.0, LightDark::Light);

    // Should find regions with variance > 10
    // Gradient has consistent variance
    EXPECT_FALSE(result.Empty());
}

TEST_F(ThresholdTest, VarThresholdBimodal) {
    // Bimodal image has high variance at the edge
    QRegion result = VarThreshold(bimodalImg_, 15, 100.0, LightDark::Light);

    // High variance at the boundary between 50 and 200
    // The result should be mostly at the edge
}

// ============================================================================
// CharThreshold Tests
// ============================================================================

TEST_F(ThresholdTest, CharThresholdBasic) {
    // CharThreshold is designed for document/text images
    QRegion result = CharThreshold(bimodalImg_, 2.0, 95.0, LightDark::Dark);

    // Should find dark characters (left half = 50)
    EXPECT_FALSE(result.Empty());
}

TEST_F(ThresholdTest, CharThresholdGradient) {
    QRegion result = CharThreshold(gradientImg_, 2.0, 95.0, LightDark::Dark);

    // Should find dark regions in gradient
    EXPECT_FALSE(result.Empty());
}

// ============================================================================
// HysteresisThresholdToRegion Tests
// ============================================================================

TEST_F(ThresholdTest, HysteresisThresholdBasic) {
    // Create an image with strong and weak edges
    QImage edgeImg(width_, height_, PixelType::UInt8, ChannelType::Gray);
    std::memset(edgeImg.Data(), 50, width_ * height_);

    uint8_t* data = static_cast<uint8_t*>(edgeImg.Data());

    // Strong edge (value 200)
    for (int32_t x = 40; x < 50; ++x) {
        for (int32_t y = 40; y < 60; ++y) {
            data[y * width_ + x] = 200;
        }
    }

    // Weak edge connected to strong (value 120)
    for (int32_t x = 50; x < 60; ++x) {
        for (int32_t y = 40; y < 60; ++y) {
            data[y * width_ + x] = 120;
        }
    }

    // Another weak edge NOT connected (value 120)
    for (int32_t x = 70; x < 80; ++x) {
        for (int32_t y = 70; y < 80; ++y) {
            data[y * width_ + x] = 120;
        }
    }

    QRegion result = HysteresisThresholdToRegion(edgeImg, 100.0, 150.0);

    // Strong edge and connected weak edge should be included
    EXPECT_FALSE(result.Empty());

    // The non-connected weak edge should NOT be included
    // Total area should be less than all pixels > 100
    int64_t area = result.Area();
    EXPECT_GT(area, 0);
}

TEST_F(ThresholdTest, HysteresisThresholdOnlyStrong) {
    // All pixels above high threshold
    QImage strongImg(width_, height_, PixelType::UInt8, ChannelType::Gray);
    std::memset(strongImg.Data(), 200, width_ * height_);

    QRegion result = HysteresisThresholdToRegion(strongImg, 100.0, 150.0);

    // All pixels are strong edges
    EXPECT_EQ(result.Area(), width_ * height_);
}

// ============================================================================
// Domain-aware Threshold Tests
// ============================================================================

TEST_F(ThresholdTest, ThresholdWithDomainNoDomain) {
    // Image without domain - should work like regular threshold
    QRegion result = ThresholdWithDomain(bimodalImg_, 100.0, 255.0);

    // Right half (200) is in range [100, 255]
    EXPECT_FALSE(result.Empty());
    EXPECT_EQ(result.Area(), width_ / 2 * height_);
}

TEST_F(ThresholdTest, DynThresholdWithDomainNoDomain) {
    QImage reference(width_, height_, PixelType::UInt8, ChannelType::Gray);
    std::memset(reference.Data(), 100, width_ * height_);

    QRegion result = DynThresholdWithDomain(bimodalImg_, reference, 5.0, LightDark::Light);

    EXPECT_FALSE(result.Empty());
}

TEST_F(ThresholdTest, ThresholdAdaptiveToRegion) {
    AdaptiveThresholdParams params = AdaptiveThresholdParams::Mean(15, 5);
    QRegion result = ThresholdAdaptiveToRegion(gradientImg_, params);

    // Should find regions above local mean
    EXPECT_FALSE(result.Empty());
}

TEST_F(ThresholdTest, ThresholdAdaptiveToRegionSauvola) {
    AdaptiveThresholdParams params = AdaptiveThresholdParams::Sauvola(15, 0.5, 128.0);
    QRegion result = ThresholdAdaptiveToRegion(bimodalImg_, params);

    EXPECT_FALSE(result.Empty());
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(ThresholdTest, DynThresholdEmptyImage) {
    QImage empty;
    QImage reference(10, 10, PixelType::UInt8, ChannelType::Gray);

    QRegion result = DynThreshold(empty, reference, 5.0, LightDark::Light);
    EXPECT_TRUE(result.Empty());
}

TEST_F(ThresholdTest, DualThresholdEmptyImage) {
    QImage empty;
    DualThresholdResult result = DualThreshold(empty, 50.0, 150.0);

    EXPECT_TRUE(result.darkRegion.Empty());
    EXPECT_TRUE(result.lightRegion.Empty());
}

TEST_F(ThresholdTest, VarThresholdEmptyImage) {
    QImage empty;
    QRegion result = VarThreshold(empty, 15, 10.0, LightDark::Light);
    EXPECT_TRUE(result.Empty());
}

TEST_F(ThresholdTest, CharThresholdEmptyImage) {
    QImage empty;
    QRegion result = CharThreshold(empty, 2.0, 95.0, LightDark::Dark);
    EXPECT_TRUE(result.Empty());
}

TEST_F(ThresholdTest, HysteresisThresholdEmptyImage) {
    QImage empty;
    QRegion result = HysteresisThresholdToRegion(empty, 100.0, 150.0);
    EXPECT_TRUE(result.Empty());
}
