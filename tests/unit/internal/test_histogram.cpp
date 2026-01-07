/**
 * @file test_histogram.cpp
 * @brief Unit tests for Internal/Histogram
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/Histogram.h>
#include <QiVision/Core/QImage.h>

#include <cmath>
#include <cstring>
#include <numeric>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

// ============================================================================
// Test Fixtures
// ============================================================================

class HistogramTest : public ::testing::Test {
protected:
    void SetUp() override {
        width_ = 100;
        height_ = 100;

        // Uniform image (all pixels = 128)
        uniformImg_ = QImage(width_, height_, PixelType::UInt8, ChannelType::Gray);
        std::memset(uniformImg_.Data(), 128, width_ * height_);

        // Gradient image (0 to 255)
        gradientImg_ = QImage(width_, height_, PixelType::UInt8, ChannelType::Gray);
        uint8_t* gradData = static_cast<uint8_t*>(gradientImg_.Data());
        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                gradData[y * width_ + x] = static_cast<uint8_t>(x * 255 / (width_ - 1));
            }
        }

        // Bimodal image (50% black, 50% white)
        bimodalImg_ = QImage(width_, height_, PixelType::UInt8, ChannelType::Gray);
        uint8_t* biData = static_cast<uint8_t*>(bimodalImg_.Data());
        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                biData[y * width_ + x] = (x < width_ / 2) ? 50 : 200;
            }
        }

        // Low contrast image
        lowContrastImg_ = QImage(width_, height_, PixelType::UInt8, ChannelType::Gray);
        uint8_t* lcData = static_cast<uint8_t*>(lowContrastImg_.Data());
        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                lcData[y * width_ + x] = static_cast<uint8_t>(100 + x * 55 / (width_ - 1));
            }
        }
    }

    int32_t width_, height_;
    QImage uniformImg_;
    QImage gradientImg_;
    QImage bimodalImg_;
    QImage lowContrastImg_;
};

// ============================================================================
// Histogram Structure Tests
// ============================================================================

TEST(HistogramStructTest, DefaultConstruction) {
    Histogram hist;
    EXPECT_EQ(hist.numBins, 256);
    EXPECT_EQ(hist.bins.size(), 256u);
    EXPECT_TRUE(hist.Empty());
    EXPECT_EQ(hist.totalCount, 0u);
}

TEST(HistogramStructTest, CustomBins) {
    Histogram hist(64, 0, 100);
    EXPECT_EQ(hist.numBins, 64);
    EXPECT_EQ(hist.bins.size(), 64u);
    EXPECT_DOUBLE_EQ(hist.minValue, 0);
    EXPECT_DOUBLE_EQ(hist.maxValue, 100);
}

TEST(HistogramStructTest, GetBinIndex) {
    Histogram hist(256, 0, 255);
    EXPECT_EQ(hist.GetBinIndex(0), 0);
    EXPECT_EQ(hist.GetBinIndex(127), 127);
    EXPECT_EQ(hist.GetBinIndex(255), 255);
}

TEST(HistogramStructTest, GetBinValue) {
    Histogram hist(256, 0, 255);
    // Bin width = 255/256 ≈ 0.996, bin center = minValue + (idx + 0.5) * binWidth
    double binWidth = 255.0 / 256.0;
    EXPECT_NEAR(hist.GetBinValue(0), 0.5 * binWidth, 0.5);
    EXPECT_NEAR(hist.GetBinValue(127), 127.5 * binWidth, 0.5);
    EXPECT_NEAR(hist.GetBinValue(255), 255.5 * binWidth, 1.0);
}

TEST(HistogramStructTest, Clear) {
    Histogram hist;
    hist.bins[100] = 50;
    hist.totalCount = 50;

    hist.Clear();

    EXPECT_EQ(hist.totalCount, 0u);
    EXPECT_EQ(hist.bins[100], 0u);
}

// ============================================================================
// Histogram Computation Tests
// ============================================================================

TEST_F(HistogramTest, ComputeHistogram_Uniform) {
    auto hist = ComputeHistogram(uniformImg_);

    EXPECT_FALSE(hist.Empty());
    EXPECT_EQ(hist.totalCount, static_cast<uint64_t>(width_ * height_));

    // All pixels are 128
    EXPECT_EQ(hist.bins[128], static_cast<uint32_t>(width_ * height_));

    // Other bins should be 0
    uint32_t sum = 0;
    for (int i = 0; i < 256; ++i) {
        if (i != 128) sum += hist.bins[i];
    }
    EXPECT_EQ(sum, 0u);
}

TEST_F(HistogramTest, ComputeHistogram_Gradient) {
    auto hist = ComputeHistogram(gradientImg_);

    EXPECT_FALSE(hist.Empty());
    EXPECT_EQ(hist.totalCount, static_cast<uint64_t>(width_ * height_));

    // Gradient should have multiple non-zero bins
    int nonZeroBins = 0;
    for (int i = 0; i < 256; ++i) {
        if (hist.bins[i] > 0) nonZeroBins++;
    }
    EXPECT_GT(nonZeroBins, 50);
}

TEST_F(HistogramTest, ComputeHistogram_Bimodal) {
    auto hist = ComputeHistogram(bimodalImg_);

    // Should have two peaks at 50 and 200
    EXPECT_GT(hist.bins[50], 0u);
    EXPECT_GT(hist.bins[200], 0u);

    // Each should have about half the pixels
    EXPECT_NEAR(hist.bins[50], width_ * height_ / 2, width_);
    EXPECT_NEAR(hist.bins[200], width_ * height_ / 2, width_);
}

TEST_F(HistogramTest, ComputeHistogramROI) {
    Rect2i roi(25, 25, 50, 50);
    auto hist = ComputeHistogramROI(uniformImg_, roi);

    EXPECT_EQ(hist.totalCount, 50u * 50u);
    EXPECT_EQ(hist.bins[128], 50u * 50u);
}

TEST_F(HistogramTest, ComputeCumulativeHistogram) {
    auto hist = ComputeHistogram(uniformImg_);
    auto cdf = ComputeCumulativeHistogram(hist);

    EXPECT_EQ(cdf.size(), 256u);

    // CDF should be 0 before 128, 1 at and after 128
    EXPECT_NEAR(cdf[127], 0.0, 1e-10);
    EXPECT_NEAR(cdf[128], 1.0, 1e-10);
    EXPECT_NEAR(cdf[255], 1.0, 1e-10);
}

TEST_F(HistogramTest, NormalizeHistogram) {
    auto hist = ComputeHistogram(gradientImg_);
    auto normalized = NormalizeHistogram(hist);

    double sum = std::accumulate(normalized.begin(), normalized.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

// ============================================================================
// Histogram Statistics Tests
// ============================================================================

TEST_F(HistogramTest, ComputeHistogramStats_Uniform) {
    auto hist = ComputeHistogram(uniformImg_);
    auto stats = ComputeHistogramStats(hist);

    EXPECT_NEAR(stats.mean, 128.0, 1.0);
    EXPECT_NEAR(stats.median, 128.0, 1.0);
    EXPECT_NEAR(stats.mode, 128.0, 1.0);
    EXPECT_NEAR(stats.stddev, 0.0, 1.0);
    EXPECT_NEAR(stats.contrast, 0.0, 2.0);
}

TEST_F(HistogramTest, ComputeHistogramStats_Gradient) {
    auto hist = ComputeHistogram(gradientImg_);
    auto stats = ComputeHistogramStats(hist);

    // Mean should be around 127
    EXPECT_NEAR(stats.mean, 127.5, 10.0);

    // Stddev should be significant
    EXPECT_GT(stats.stddev, 50.0);

    // Contrast should be high
    EXPECT_GT(stats.contrast, 200.0);
}

TEST_F(HistogramTest, ComputePercentile) {
    auto hist = ComputeHistogram(gradientImg_);

    double p0 = ComputePercentile(hist, 0);
    double p50 = ComputePercentile(hist, 50);
    double p100 = ComputePercentile(hist, 100);

    EXPECT_LT(p0, p50);
    EXPECT_LT(p50, p100);
    EXPECT_NEAR(p50, 127.5, 10.0);
}

TEST_F(HistogramTest, ComputeEntropy_Uniform) {
    auto hist = ComputeHistogram(uniformImg_);
    double entropy = ComputeEntropy(hist);

    // Uniform image has 0 entropy (all same value)
    EXPECT_NEAR(entropy, 0.0, 0.1);
}

TEST_F(HistogramTest, ComputeEntropy_Gradient) {
    auto hist = ComputeHistogram(gradientImg_);
    double entropy = ComputeEntropy(hist);

    // Gradient should have high entropy
    EXPECT_GT(entropy, 5.0);
}

// ============================================================================
// Histogram Equalization Tests
// ============================================================================

TEST_F(HistogramTest, HistogramEqualize_LowContrast) {
    auto equalized = HistogramEqualize(lowContrastImg_);

    EXPECT_FALSE(equalized.Empty());
    EXPECT_EQ(equalized.Width(), lowContrastImg_.Width());
    EXPECT_EQ(equalized.Height(), lowContrastImg_.Height());

    // Equalized histogram should have more contrast
    auto origStats = ComputeHistogramStats(ComputeHistogram(lowContrastImg_));
    auto eqStats = ComputeHistogramStats(ComputeHistogram(equalized));

    EXPECT_GT(eqStats.contrast, origStats.contrast);
}

TEST_F(HistogramTest, ComputeEqualizationLUT) {
    auto hist = ComputeHistogram(lowContrastImg_);
    auto lut = ComputeEqualizationLUT(hist);

    EXPECT_EQ(lut.size(), 256u);

    // LUT should be monotonically increasing
    for (int i = 1; i < 256; ++i) {
        EXPECT_GE(lut[i], lut[i - 1]);
    }
}

TEST_F(HistogramTest, ApplyLUT) {
    std::vector<uint8_t> lut(256);
    for (int i = 0; i < 256; ++i) {
        lut[i] = static_cast<uint8_t>(255 - i);  // Invert
    }

    auto inverted = ApplyLUT(uniformImg_, lut);

    const uint8_t* data = static_cast<const uint8_t*>(inverted.Data());
    EXPECT_EQ(data[0], 127);  // 255 - 128 = 127
}

TEST_F(HistogramTest, ApplyLUTInPlace) {
    QImage img = uniformImg_;  // Copy

    std::vector<uint8_t> lut(256);
    for (int i = 0; i < 256; ++i) {
        lut[i] = static_cast<uint8_t>(std::min(255, i * 2));  // Brighten
    }

    ApplyLUTInPlace(img, lut);

    const uint8_t* data = static_cast<const uint8_t*>(img.Data());
    EXPECT_EQ(data[0], 255);  // 128 * 2 clamped to 255
}

// ============================================================================
// CLAHE Tests
// ============================================================================

TEST_F(HistogramTest, ApplyCLAHE_Basic) {
    auto enhanced = ApplyCLAHE(lowContrastImg_);

    EXPECT_FALSE(enhanced.Empty());
    EXPECT_EQ(enhanced.Width(), lowContrastImg_.Width());
    EXPECT_EQ(enhanced.Height(), lowContrastImg_.Height());
}

TEST_F(HistogramTest, ApplyCLAHE_ContrastImproved) {
    CLAHEParams params;
    params.tileGridSizeX = 8;
    params.tileGridSizeY = 8;
    params.clipLimit = 40.0;

    auto enhanced = ApplyCLAHE(lowContrastImg_, params);

    auto origStats = ComputeHistogramStats(ComputeHistogram(lowContrastImg_));
    auto enhStats = ComputeHistogramStats(ComputeHistogram(enhanced));

    // CLAHE should improve contrast
    EXPECT_GT(enhStats.contrast, origStats.contrast);
}

TEST_F(HistogramTest, CLAHEParams_Factory) {
    auto params = CLAHEParams::WithTileSize(16, 30.0);

    EXPECT_EQ(params.tileGridSizeX, 16);
    EXPECT_EQ(params.tileGridSizeY, 16);
    EXPECT_DOUBLE_EQ(params.clipLimit, 30.0);
}

// ============================================================================
// Histogram Matching Tests
// ============================================================================

TEST_F(HistogramTest, ComputeMatchingLUT) {
    auto sourceHist = ComputeHistogram(lowContrastImg_);
    auto targetHist = ComputeHistogram(gradientImg_);

    auto lut = ComputeMatchingLUT(sourceHist, targetHist);

    EXPECT_EQ(lut.size(), 256u);

    // LUT should be monotonically increasing
    for (int i = 1; i < 256; ++i) {
        EXPECT_GE(lut[i], lut[i - 1]);
    }
}

TEST_F(HistogramTest, HistogramMatch) {
    auto matched = HistogramMatch(lowContrastImg_, ComputeHistogram(gradientImg_));

    EXPECT_FALSE(matched.Empty());

    // Matched histogram should be more similar to target
    auto matchedHist = ComputeHistogram(matched);
    auto targetHist = ComputeHistogram(gradientImg_);

    // Contrast should be more like target
    auto matchedStats = ComputeHistogramStats(matchedHist);
    auto targetStats = ComputeHistogramStats(targetHist);

    EXPECT_GT(matchedStats.contrast, 100.0);
}

TEST_F(HistogramTest, HistogramMatchToImage) {
    auto matched = HistogramMatchToImage(lowContrastImg_, gradientImg_);

    EXPECT_FALSE(matched.Empty());
    EXPECT_EQ(matched.Width(), lowContrastImg_.Width());
}

// ============================================================================
// Contrast Stretching Tests
// ============================================================================

TEST_F(HistogramTest, ContrastStretch) {
    auto stretched = ContrastStretch(lowContrastImg_, 1.0, 99.0, 0, 255);

    EXPECT_FALSE(stretched.Empty());

    auto stats = ComputeHistogramStats(ComputeHistogram(stretched));
    EXPECT_GT(stats.contrast, 200.0);
}

TEST_F(HistogramTest, AutoContrast) {
    auto enhanced = AutoContrast(lowContrastImg_);

    EXPECT_FALSE(enhanced.Empty());

    auto origStats = ComputeHistogramStats(ComputeHistogram(lowContrastImg_));
    auto stats = ComputeHistogramStats(ComputeHistogram(enhanced));

    // Auto contrast should improve or maintain contrast
    // The low contrast image has range [100, 155], after stretching it should be wider
    EXPECT_GE(stats.contrast, origStats.contrast);
}

TEST_F(HistogramTest, NormalizeImage) {
    auto normalized = NormalizeImage(lowContrastImg_, 50, 200);

    EXPECT_FALSE(normalized.Empty());

    auto hist = ComputeHistogram(normalized);
    auto stats = ComputeHistogramStats(hist);

    EXPECT_GE(stats.min, 50.0);
    EXPECT_LE(stats.max, 200.0);
}

// ============================================================================
// Automatic Thresholding Tests
// ============================================================================

TEST_F(HistogramTest, ComputeOtsuThreshold_Bimodal) {
    auto hist = ComputeHistogram(bimodalImg_);
    double threshold = ComputeOtsuThreshold(hist);

    // Threshold should be between the two modes
    EXPECT_GT(threshold, 50.0);
    EXPECT_LT(threshold, 200.0);
}

TEST_F(HistogramTest, ComputeOtsuThreshold_FromImage) {
    double threshold = ComputeOtsuThreshold(bimodalImg_);

    EXPECT_GT(threshold, 50.0);
    EXPECT_LT(threshold, 200.0);
}

TEST_F(HistogramTest, ComputeMultiOtsuThresholds) {
    auto hist = ComputeHistogram(gradientImg_);
    auto thresholds = ComputeMultiOtsuThresholds(hist, 2);

    EXPECT_EQ(thresholds.size(), 2u);
    EXPECT_LT(thresholds[0], thresholds[1]);
}

TEST_F(HistogramTest, ComputeTriangleThreshold) {
    auto hist = ComputeHistogram(bimodalImg_);
    double threshold = ComputeTriangleThreshold(hist);

    // Should find a reasonable threshold
    EXPECT_GT(threshold, 0.0);
    EXPECT_LT(threshold, 255.0);
}

TEST_F(HistogramTest, ComputeIsodataThreshold) {
    auto hist = ComputeHistogram(bimodalImg_);
    double threshold = ComputeIsodataThreshold(hist);

    EXPECT_GT(threshold, 50.0);
    EXPECT_LT(threshold, 200.0);
}

// ============================================================================
// Utility Functions Tests
// ============================================================================

TEST_F(HistogramTest, FindHistogramPeak) {
    auto hist = ComputeHistogram(uniformImg_);
    int32_t peak = FindHistogramPeak(hist);

    EXPECT_EQ(peak, 128);
}

TEST_F(HistogramTest, FindHistogramPeaks_Bimodal) {
    auto hist = ComputeHistogram(bimodalImg_);
    auto peaks = FindHistogramPeaks(hist, 0.1, 20);

    EXPECT_GE(peaks.size(), 2u);
}

TEST_F(HistogramTest, FindHistogramValleys) {
    // Create a histogram with a clear valley: high-low-high pattern
    Histogram hist(256, 0, 255);

    // Fill with a shape that has distinct peaks and valleys
    for (int i = 0; i < 256; ++i) {
        if (i < 50) {
            hist.bins[i] = 100 + i * 10;  // Rising to peak at 50
        } else if (i < 100) {
            hist.bins[i] = 600 - (i - 50) * 10;  // Falling to valley at 100
        } else if (i < 150) {
            hist.bins[i] = 100 + (i - 100) * 10;  // Rising to peak at 150
        } else if (i < 200) {
            hist.bins[i] = 600 - (i - 150) * 10;  // Falling to valley at 200
        } else {
            hist.bins[i] = 100 + (i - 200) * 5;  // Rising
        }
        hist.totalCount += hist.bins[i];
    }

    auto valleys = FindHistogramValleys(hist, 20);

    // Should find valleys around 100 and 200
    EXPECT_GE(valleys.size(), 1u);

    // Verify a valley is found in expected range
    bool foundValley = false;
    for (int32_t v : valleys) {
        if ((v >= 95 && v <= 105) || (v >= 195 && v <= 205)) {
            foundValley = true;
            break;
        }
    }
    EXPECT_TRUE(foundValley);
}

TEST_F(HistogramTest, SmoothHistogram) {
    auto hist = ComputeHistogram(gradientImg_);
    auto smoothed = SmoothHistogram(hist, 5);

    EXPECT_EQ(smoothed.numBins, hist.numBins);

    // Smoothed histogram should be less "spiky"
    // (This is hard to test precisely, just verify it runs)
    EXPECT_FALSE(smoothed.Empty());
}

// ============================================================================
// Histogram Comparison Tests
// ============================================================================

TEST_F(HistogramTest, CompareHistograms_Same) {
    auto hist = ComputeHistogram(uniformImg_);

    double correlation = CompareHistograms(hist, hist, HistogramCompareMethod::Correlation);
    double intersection = CompareHistograms(hist, hist, HistogramCompareMethod::Intersection);
    double bhattacharyya = CompareHistograms(hist, hist, HistogramCompareMethod::Bhattacharyya);

    EXPECT_NEAR(correlation, 1.0, 0.01);
    EXPECT_NEAR(intersection, 1.0, 0.01);
    EXPECT_NEAR(bhattacharyya, 0.0, 0.01);
}

TEST_F(HistogramTest, CompareHistograms_Different) {
    auto hist1 = ComputeHistogram(uniformImg_);
    auto hist2 = ComputeHistogram(gradientImg_);

    double correlation = CompareHistograms(hist1, hist2, HistogramCompareMethod::Correlation);
    double bhattacharyya = CompareHistograms(hist1, hist2, HistogramCompareMethod::Bhattacharyya);
    double chiSquare = CompareHistograms(hist1, hist2, HistogramCompareMethod::ChiSquare);

    // Different histograms should have lower correlation, higher distance
    EXPECT_LT(correlation, 0.5);
    EXPECT_GT(bhattacharyya, 0.5);
    EXPECT_GT(chiSquare, 0.0);
}

TEST_F(HistogramTest, CompareHistograms_Intersection) {
    auto hist1 = ComputeHistogram(uniformImg_);
    auto hist2 = ComputeHistogram(gradientImg_);

    double intersection = CompareHistograms(hist1, hist2, HistogramCompareMethod::Intersection);

    // Intersection should be between 0 and 1
    EXPECT_GE(intersection, 0.0);
    EXPECT_LE(intersection, 1.0);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(HistogramEdgeCaseTest, EmptyImage) {
    QImage empty;
    auto hist = ComputeHistogram(empty);

    EXPECT_TRUE(hist.Empty());
}

TEST(HistogramEdgeCaseTest, SinglePixelImage) {
    QImage img(1, 1, PixelType::UInt8, ChannelType::Gray);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    data[0] = 100;

    auto hist = ComputeHistogram(img);

    EXPECT_EQ(hist.totalCount, 1u);
    EXPECT_EQ(hist.bins[100], 1u);
}

TEST(HistogramEdgeCaseTest, EqualizeEmptyImage) {
    QImage empty;
    auto result = HistogramEqualize(empty);
    EXPECT_TRUE(result.Empty());
}

TEST(HistogramEdgeCaseTest, CLAHESmallImage) {
    QImage small(10, 10, PixelType::UInt8, ChannelType::Gray);
    std::memset(small.Data(), 100, 100);

    // Should not crash on small image
    auto result = ApplyCLAHE(small);
    EXPECT_FALSE(result.Empty());
}

// ============================================================================
// HistogramStats Structure Tests
// ============================================================================

TEST(HistogramStatsTest, DefaultValues) {
    HistogramStats stats;

    EXPECT_DOUBLE_EQ(stats.min, 0);
    EXPECT_DOUBLE_EQ(stats.max, 0);
    EXPECT_DOUBLE_EQ(stats.mean, 0);
    EXPECT_DOUBLE_EQ(stats.stddev, 0);
    EXPECT_EQ(stats.totalCount, 0u);
}

// ============================================================================
// Different Pixel Types
// ============================================================================

TEST(HistogramPixelTypeTest, UInt16Image) {
    QImage img16(50, 50, PixelType::UInt16, ChannelType::Gray);
    uint16_t* data = static_cast<uint16_t*>(img16.Data());
    for (int i = 0; i < 50 * 50; ++i) {
        data[i] = static_cast<uint16_t>(i * 26);  // 0 to ~65000
    }

    auto hist = ComputeHistogram(img16);

    EXPECT_FALSE(hist.Empty());
    EXPECT_EQ(hist.totalCount, 50u * 50u);
}

TEST(HistogramPixelTypeTest, FloatImage) {
    QImage imgFloat(50, 50, PixelType::Float32, ChannelType::Gray);
    float* data = static_cast<float*>(imgFloat.Data());
    for (int i = 0; i < 50 * 50; ++i) {
        data[i] = static_cast<float>(i) / (50 * 50);  // 0 to 1
    }

    auto hist = ComputeHistogram(imgFloat);

    EXPECT_FALSE(hist.Empty());
    EXPECT_EQ(hist.totalCount, 50u * 50u);
}
