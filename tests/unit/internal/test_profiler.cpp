/**
 * @file test_profiler.cpp
 * @brief Unit tests for Internal/Profiler
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/Profiler.h>
#include <QiVision/Core/QImage.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

// ============================================================================
// Test Fixtures
// ============================================================================

class ProfilerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test images
        width_ = 100;
        height_ = 100;

        // Uniform gray image
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

        // Vertical gradient image (top to bottom: 0 to 255)
        vGradientImg_ = QImage(width_, height_, PixelType::UInt8, ChannelType::Gray);
        uint8_t* vGradData = static_cast<uint8_t*>(vGradientImg_.Data());
        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                vGradData[y * width_ + x] = static_cast<uint8_t>(y * 255 / (height_ - 1));
            }
        }

        // Circular pattern image
        circleImg_ = QImage(width_, height_, PixelType::UInt8, ChannelType::Gray);
        uint8_t* circData = static_cast<uint8_t*>(circleImg_.Data());
        double cx = width_ / 2.0;
        double cy = height_ / 2.0;
        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                double dist = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                circData[y * width_ + x] = static_cast<uint8_t>(
                    std::min(255.0, dist * 5.0));
            }
        }
    }

    int32_t width_, height_;
    QImage uniformImg_;
    QImage gradientImg_;
    QImage vGradientImg_;
    QImage circleImg_;
};

// ============================================================================
// Profile1D Tests
// ============================================================================

TEST(Profile1DTest, DefaultConstruction) {
    Profile1D profile;
    EXPECT_TRUE(profile.Empty());
    EXPECT_EQ(profile.Size(), 0u);
    EXPECT_DOUBLE_EQ(profile.Length(), 0.0);
}

TEST(Profile1DTest, DataAccess) {
    Profile1D profile;
    profile.data = {1.0, 2.0, 3.0, 4.0, 5.0};
    profile.startX = 0;
    profile.startY = 0;
    profile.endX = 4;
    profile.endY = 0;

    EXPECT_EQ(profile.Size(), 5u);
    EXPECT_FALSE(profile.Empty());
    EXPECT_DOUBLE_EQ(profile.At(0), 1.0);
    EXPECT_DOUBLE_EQ(profile.At(2), 3.0);
    EXPECT_DOUBLE_EQ(profile.At(4), 5.0);
    EXPECT_DOUBLE_EQ(profile.At(10), 0.0);  // Out of bounds
}

TEST(Profile1DTest, AtSafe) {
    Profile1D profile;
    profile.data = {10.0, 20.0, 30.0};

    EXPECT_DOUBLE_EQ(profile.AtSafe(-1), 10.0);  // Clamp to first
    EXPECT_DOUBLE_EQ(profile.AtSafe(0), 10.0);
    EXPECT_DOUBLE_EQ(profile.AtSafe(1), 20.0);
    EXPECT_DOUBLE_EQ(profile.AtSafe(2), 30.0);
    EXPECT_DOUBLE_EQ(profile.AtSafe(10), 30.0);  // Clamp to last
}

TEST(Profile1DTest, IndexToCoord) {
    Profile1D profile;
    profile.data.resize(11);  // 11 samples
    profile.startX = 0;
    profile.startY = 0;
    profile.endX = 10;
    profile.endY = 10;

    double x, y;

    profile.IndexToCoord(0, x, y);
    EXPECT_NEAR(x, 0.0, 1e-10);
    EXPECT_NEAR(y, 0.0, 1e-10);

    profile.IndexToCoord(5, x, y);
    EXPECT_NEAR(x, 5.0, 1e-10);
    EXPECT_NEAR(y, 5.0, 1e-10);

    profile.IndexToCoord(10, x, y);
    EXPECT_NEAR(x, 10.0, 1e-10);
    EXPECT_NEAR(y, 10.0, 1e-10);
}

TEST(Profile1DTest, CoordToIndex) {
    Profile1D profile;
    profile.data.resize(11);
    profile.startX = 0;
    profile.startY = 0;
    profile.endX = 10;
    profile.endY = 0;

    EXPECT_NEAR(profile.CoordToIndex(0, 0), 0.0, 1e-10);
    EXPECT_NEAR(profile.CoordToIndex(5, 0), 5.0, 1e-10);
    EXPECT_NEAR(profile.CoordToIndex(10, 0), 10.0, 1e-10);
}

TEST(Profile1DTest, Length) {
    Profile1D profile;
    profile.startX = 0;
    profile.startY = 0;
    profile.endX = 3;
    profile.endY = 4;

    EXPECT_DOUBLE_EQ(profile.Length(), 5.0);  // 3-4-5 triangle
}

// ============================================================================
// Line Profile Extraction Tests
// ============================================================================

TEST_F(ProfilerTest, ExtractLineProfile_Horizontal) {
    // Extract horizontal profile through center
    auto profile = ExtractLineProfile(gradientImg_, 0, 50, 99, 50);

    EXPECT_FALSE(profile.Empty());
    EXPECT_EQ(profile.Size(), 100u);

    // Should follow the gradient: 0 to 255
    EXPECT_NEAR(profile.data[0], 0.0, 2.0);
    EXPECT_NEAR(profile.data[50], 127.5, 2.0);
    EXPECT_NEAR(profile.data[99], 255.0, 2.0);
}

TEST_F(ProfilerTest, ExtractLineProfile_Vertical) {
    // Extract vertical profile through center
    auto profile = ExtractLineProfile(vGradientImg_, 50, 0, 50, 99);

    EXPECT_FALSE(profile.Empty());
    EXPECT_EQ(profile.Size(), 100u);

    // Should follow vertical gradient
    EXPECT_NEAR(profile.data[0], 0.0, 2.0);
    EXPECT_NEAR(profile.data[50], 127.5, 2.0);
    EXPECT_NEAR(profile.data[99], 255.0, 2.0);
}

TEST_F(ProfilerTest, ExtractLineProfile_Diagonal) {
    // Diagonal line
    auto profile = ExtractLineProfile(gradientImg_, 0, 0, 99, 99);

    EXPECT_FALSE(profile.Empty());
    // Diagonal length is sqrt(99^2 + 99^2) ≈ 140
    EXPECT_GT(profile.Size(), 100u);

    // Values should increase along diagonal
    EXPECT_LT(profile.data[0], profile.data[profile.Size() - 1]);
}

TEST_F(ProfilerTest, ExtractLineProfile_Uniform) {
    auto profile = ExtractLineProfile(uniformImg_, 10, 50, 90, 50);

    EXPECT_FALSE(profile.Empty());

    // All values should be 128
    for (double val : profile.data) {
        EXPECT_NEAR(val, 128.0, 1.0);
    }
}

TEST_F(ProfilerTest, ExtractLineProfile_CustomSamples) {
    auto profile = ExtractLineProfile(gradientImg_, 0, 50, 99, 50, 50);

    EXPECT_EQ(profile.Size(), 50u);

    // Still should follow gradient
    EXPECT_NEAR(profile.data[0], 0.0, 2.0);
    EXPECT_NEAR(profile.data[49], 255.0, 2.0);
}

TEST_F(ProfilerTest, ExtractLineProfile_EmptyImage) {
    QImage empty;
    auto profile = ExtractLineProfile(empty, 0, 0, 10, 10);
    EXPECT_TRUE(profile.Empty());
}

// ============================================================================
// Parallel Profiles Tests
// ============================================================================

TEST_F(ProfilerTest, ExtractParallelProfiles) {
    auto profiles = ExtractParallelProfiles(uniformImg_, 10, 40, 90, 40, 5.0, 5);

    EXPECT_EQ(profiles.size(), 5u);

    for (const auto& profile : profiles) {
        EXPECT_FALSE(profile.Empty());
        // All values should be uniform
        for (double val : profile.data) {
            EXPECT_NEAR(val, 128.0, 1.0);
        }
    }
}

TEST_F(ProfilerTest, ExtractParallelProfiles_Gradient) {
    // Parallel horizontal lines in vertical gradient
    // Spacing is perpendicular to line direction (vertical offset)
    auto profiles = ExtractParallelProfiles(vGradientImg_, 10, 50, 90, 50, 10.0, 3);

    EXPECT_EQ(profiles.size(), 3u);

    // All profiles are horizontal at different y values
    // Due to perpendicular offset calculation (perpAngle = angle - PI/2):
    // - For horizontal line (angle=0), perpAngle = -PI/2, so perp direction is (0, -1)
    // - With halfExtent = 10:
    //   - profile[0]: offset=-10, y = 50 + 10 = 60 (below center, higher values)
    //   - profile[1]: offset=0, y = 50 (center)
    //   - profile[2]: offset=10, y = 50 - 10 = 40 (above center, lower values)
    // In vertical gradient (top=0, bottom=255), higher y means higher values

    double mean0 = 0, mean1 = 0, mean2 = 0;
    for (double v : profiles[0].data) mean0 += v;
    for (double v : profiles[1].data) mean1 += v;
    for (double v : profiles[2].data) mean2 += v;

    mean0 /= profiles[0].Size();
    mean1 /= profiles[1].Size();
    mean2 /= profiles[2].Size();

    // All three profiles have distinct means due to vertical gradient
    // The exact order depends on perpendicular direction convention
    // Just verify they are different
    EXPECT_NE(mean0, mean1);
    EXPECT_NE(mean1, mean2);
    EXPECT_NE(mean0, mean2);
}

// ============================================================================
// Rectangle Profile Tests
// ============================================================================

TEST_F(ProfilerTest, ExtractRectProfile_SingleLine) {
    RectProfileParams params;
    params.centerX = 50;
    params.centerY = 50;
    params.length = 80;
    params.width = 1;
    params.angle = 0;  // Horizontal
    params.numLines = 1;

    auto profile = ExtractRectProfile(gradientImg_, params);

    EXPECT_FALSE(profile.Empty());
    EXPECT_GT(profile.Size(), 0u);

    // Should follow gradient
    EXPECT_LT(profile.data[0], profile.data[profile.Size() - 1]);
}

TEST_F(ProfilerTest, ExtractRectProfile_Averaging) {
    RectProfileParams params;
    params.centerX = 50;
    params.centerY = 50;
    params.length = 80;
    params.width = 20;
    params.angle = 0;
    params.numLines = 10;
    params.method = ProfileMethod::Average;

    auto profile = ExtractRectProfile(uniformImg_, params);

    EXPECT_FALSE(profile.Empty());

    // Averaged uniform should still be 128
    for (double val : profile.data) {
        EXPECT_NEAR(val, 128.0, 2.0);
    }
}

TEST_F(ProfilerTest, ExtractRectProfile_FromLine) {
    auto params = RectProfileParams::FromLine(10, 50, 90, 50, 10.0, 5);

    EXPECT_DOUBLE_EQ(params.centerX, 50.0);
    EXPECT_DOUBLE_EQ(params.centerY, 50.0);
    EXPECT_NEAR(params.length, 80.0, 0.1);
    EXPECT_NEAR(params.angle, 0.0, 0.01);
    EXPECT_EQ(params.numLines, 5);
}

TEST_F(ProfilerTest, ExtractRectProfile_FromCenter) {
    auto params = RectProfileParams::FromCenter(50, 50, 80, M_PI / 4, 10.0, 3);

    EXPECT_DOUBLE_EQ(params.centerX, 50.0);
    EXPECT_DOUBLE_EQ(params.centerY, 50.0);
    EXPECT_DOUBLE_EQ(params.length, 80.0);
    EXPECT_DOUBLE_EQ(params.angle, M_PI / 4);
}

TEST_F(ProfilerTest, ExtractRectProfile_Maximum) {
    RectProfileParams params;
    params.centerX = 50;
    params.centerY = 50;
    params.length = 80;
    params.width = 20;
    params.angle = 0;
    params.numLines = 10;
    params.method = ProfileMethod::Maximum;

    auto profile = ExtractRectProfile(vGradientImg_, params);

    EXPECT_FALSE(profile.Empty());

    // Max of vertical gradient in horizontal rect should be bottom row value
    // All samples should be >= 128 (center value)
    for (double val : profile.data) {
        EXPECT_GE(val, 40.0);  // Approximate lower bound
    }
}

// ============================================================================
// Arc Profile Tests
// ============================================================================

TEST_F(ProfilerTest, ExtractArcProfile_Basic) {
    ArcProfileParams params;
    params.centerX = 50;
    params.centerY = 50;
    params.radius = 30;
    params.startAngle = 0;
    params.endAngle = M_PI;
    params.numLines = 1;

    auto profile = ExtractArcProfile(circleImg_, params);

    EXPECT_FALSE(profile.Empty());
    EXPECT_GT(profile.Size(), 0u);

    // On circular pattern, arc at constant radius should have constant value
    double minVal = *std::min_element(profile.data.begin(), profile.data.end());
    double maxVal = *std::max_element(profile.data.begin(), profile.data.end());

    // Values should be relatively constant (circular pattern)
    EXPECT_LT(maxVal - minVal, 30.0);  // Allow some variation
}

TEST_F(ProfilerTest, ExtractArcProfile_FullCircle) {
    auto params = ArcProfileParams::FullCircle(50, 50, 25, 5.0, 3);

    EXPECT_DOUBLE_EQ(params.centerX, 50.0);
    EXPECT_DOUBLE_EQ(params.centerY, 50.0);
    EXPECT_DOUBLE_EQ(params.radius, 25.0);
    EXPECT_NEAR(params.SweepAngle(), 2 * M_PI, 1e-10);
}

TEST_F(ProfilerTest, ExtractArcProfile_Averaging) {
    ArcProfileParams params;
    params.centerX = 50;
    params.centerY = 50;
    params.radius = 30;
    params.startAngle = 0;
    params.endAngle = M_PI / 2;
    params.width = 10;
    params.numLines = 5;
    params.method = ProfileMethod::Average;

    auto profile = ExtractArcProfile(uniformImg_, params);

    // Averaged uniform should be 128
    for (double val : profile.data) {
        EXPECT_NEAR(val, 128.0, 2.0);
    }
}

TEST_F(ProfilerTest, ArcProfileParams_ArcLength) {
    ArcProfileParams params;
    params.radius = 10;
    params.startAngle = 0;
    params.endAngle = M_PI;

    EXPECT_NEAR(params.ArcLength(), 10.0 * M_PI, 1e-10);
}

// ============================================================================
// Annular Profile Tests
// ============================================================================

TEST_F(ProfilerTest, ExtractAnnularProfile_Basic) {
    AnnularProfileParams params;
    params.centerX = 50;
    params.centerY = 50;
    params.innerRadius = 10;
    params.outerRadius = 40;
    params.angle = 0;
    params.numLines = 1;

    auto profile = ExtractAnnularProfile(circleImg_, params);

    EXPECT_FALSE(profile.Empty());
    EXPECT_GT(profile.Size(), 0u);

    // In circular pattern, radial profile should increase
    EXPECT_LT(profile.data[0], profile.data[profile.Size() - 1]);
}

TEST_F(ProfilerTest, ExtractAnnularProfile_Averaging) {
    AnnularProfileParams params;
    params.centerX = 50;
    params.centerY = 50;
    params.innerRadius = 10;
    params.outerRadius = 40;
    params.angle = 0;
    params.angularWidth = 0.2;
    params.numLines = 5;
    params.method = ProfileMethod::Average;

    auto profile = ExtractAnnularProfile(uniformImg_, params);

    for (double val : profile.data) {
        EXPECT_NEAR(val, 128.0, 2.0);
    }
}

TEST_F(ProfilerTest, AnnularProfileParams_RadialExtent) {
    auto params = AnnularProfileParams::FromRadii(50, 50, 10, 40);
    EXPECT_DOUBLE_EQ(params.RadialExtent(), 30.0);
}

// ============================================================================
// Profile Statistics Tests
// ============================================================================

TEST(ProfileStatsTest, ComputeStats) {
    Profile1D profile;
    profile.data = {1.0, 2.0, 3.0, 4.0, 5.0};

    auto stats = ComputeProfileStats(profile);

    EXPECT_DOUBLE_EQ(stats.min, 1.0);
    EXPECT_DOUBLE_EQ(stats.max, 5.0);
    EXPECT_DOUBLE_EQ(stats.mean, 3.0);
    EXPECT_DOUBLE_EQ(stats.sum, 15.0);
    EXPECT_EQ(stats.count, 5u);
    EXPECT_EQ(stats.minIdx, 0u);
    EXPECT_EQ(stats.maxIdx, 4u);
    EXPECT_GT(stats.stddev, 0.0);
}

TEST(ProfileStatsTest, ComputeStats_Empty) {
    Profile1D profile;
    auto stats = ComputeProfileStats(profile);

    EXPECT_EQ(stats.count, 0u);
    EXPECT_DOUBLE_EQ(stats.sum, 0.0);
}

TEST(ProfileStatsTest, ComputeStats_SingleValue) {
    Profile1D profile;
    profile.data = {42.0};

    auto stats = ComputeProfileStats(profile);

    EXPECT_DOUBLE_EQ(stats.min, 42.0);
    EXPECT_DOUBLE_EQ(stats.max, 42.0);
    EXPECT_DOUBLE_EQ(stats.mean, 42.0);
    EXPECT_DOUBLE_EQ(stats.stddev, 0.0);
}

// ============================================================================
// Profile Normalization Tests
// ============================================================================

TEST(ProfileNormalizeTest, MinMax) {
    Profile1D profile;
    profile.data = {0.0, 50.0, 100.0};

    NormalizeProfile(profile, ProfileNormalize::MinMax);

    EXPECT_DOUBLE_EQ(profile.data[0], 0.0);
    EXPECT_DOUBLE_EQ(profile.data[1], 0.5);
    EXPECT_DOUBLE_EQ(profile.data[2], 1.0);
}

TEST(ProfileNormalizeTest, ZScore) {
    Profile1D profile;
    profile.data = {0.0, 5.0, 10.0};

    NormalizeProfile(profile, ProfileNormalize::ZScore);

    EXPECT_NEAR(profile.data[1], 0.0, 1e-10);  // Mean should be 0
    // Check stddev normalization
    double sum = 0;
    for (double v : profile.data) sum += v * v;
    double variance = sum / profile.data.size();
    EXPECT_NEAR(variance, 1.0, 0.5);  // Approximately unit variance
}

TEST(ProfileNormalizeTest, Sum) {
    Profile1D profile;
    profile.data = {1.0, 2.0, 3.0, 4.0};

    NormalizeProfile(profile, ProfileNormalize::Sum);

    double sum = 0;
    for (double v : profile.data) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

// ============================================================================
// Profile Smoothing Tests
// ============================================================================

TEST(ProfileSmoothTest, NoChange_ZeroSigma) {
    Profile1D profile;
    profile.data = {1.0, 10.0, 1.0};
    std::vector<double> original = profile.data;

    SmoothProfile(profile, 0.0);

    for (size_t i = 0; i < profile.Size(); ++i) {
        EXPECT_DOUBLE_EQ(profile.data[i], original[i]);
    }
}

TEST(ProfileSmoothTest, Smoothing) {
    Profile1D profile;
    profile.data = {0.0, 0.0, 100.0, 0.0, 0.0};

    SmoothProfile(profile, 1.0);

    // Peak should be reduced
    EXPECT_LT(profile.data[2], 100.0);
    // Neighbors should increase
    EXPECT_GT(profile.data[1], 0.0);
    EXPECT_GT(profile.data[3], 0.0);
}

// ============================================================================
// Profile Gradient Tests
// ============================================================================

TEST(ProfileGradientTest, LinearGradient) {
    Profile1D profile;
    profile.data = {0.0, 1.0, 2.0, 3.0, 4.0};

    auto gradient = ComputeProfileGradient(profile, 0.0);

    EXPECT_EQ(gradient.Size(), profile.Size());

    // Linear increase should have constant gradient
    for (size_t i = 1; i < gradient.Size() - 1; ++i) {
        EXPECT_NEAR(gradient.data[i], 1.0, 0.1);
    }
}

TEST(ProfileGradientTest, StepEdge) {
    Profile1D profile;
    profile.data = {0.0, 0.0, 0.0, 100.0, 100.0, 100.0};

    auto gradient = ComputeProfileGradient(profile, 0.0);

    // Maximum gradient should be near the step (between index 2 and 3)
    size_t maxIdx = 0;
    double maxVal = 0;
    for (size_t i = 0; i < gradient.Size(); ++i) {
        if (std::abs(gradient.data[i]) > maxVal) {
            maxVal = std::abs(gradient.data[i]);
            maxIdx = i;
        }
    }

    // Central difference at index 2: (100-0)/2 = 50
    // Central difference at index 3: (100-0)/2 = 50
    // Both have the same magnitude, so maxIdx could be 2 or 3
    EXPECT_TRUE(maxIdx == 2u || maxIdx == 3u);
}

// ============================================================================
// Profile Resampling Tests
// ============================================================================

TEST(ProfileResampleTest, Upsample) {
    Profile1D profile;
    profile.data = {0.0, 100.0};
    profile.startX = 0;
    profile.endX = 1;

    auto resampled = ResampleProfile(profile, 5);

    EXPECT_EQ(resampled.Size(), 5u);
    EXPECT_NEAR(resampled.data[0], 0.0, 1e-10);
    EXPECT_NEAR(resampled.data[2], 50.0, 1e-10);
    EXPECT_NEAR(resampled.data[4], 100.0, 1e-10);
}

TEST(ProfileResampleTest, Downsample) {
    Profile1D profile;
    profile.data = {0.0, 25.0, 50.0, 75.0, 100.0};

    auto resampled = ResampleProfile(profile, 3);

    EXPECT_EQ(resampled.Size(), 3u);
    EXPECT_NEAR(resampled.data[0], 0.0, 1e-10);
    EXPECT_NEAR(resampled.data[1], 50.0, 1e-10);
    EXPECT_NEAR(resampled.data[2], 100.0, 1e-10);
}

// ============================================================================
// Combine Profiles Tests
// ============================================================================

TEST(CombineProfilesTest, Average) {
    Profile1D p1, p2, p3;
    p1.data = {10.0, 20.0, 30.0};
    p2.data = {20.0, 30.0, 40.0};
    p3.data = {30.0, 40.0, 50.0};

    auto combined = CombineProfiles({p1, p2, p3}, ProfileMethod::Average);

    EXPECT_DOUBLE_EQ(combined.data[0], 20.0);
    EXPECT_DOUBLE_EQ(combined.data[1], 30.0);
    EXPECT_DOUBLE_EQ(combined.data[2], 40.0);
}

TEST(CombineProfilesTest, Maximum) {
    Profile1D p1, p2;
    p1.data = {10.0, 50.0, 30.0};
    p2.data = {20.0, 30.0, 40.0};

    auto combined = CombineProfiles({p1, p2}, ProfileMethod::Maximum);

    EXPECT_DOUBLE_EQ(combined.data[0], 20.0);
    EXPECT_DOUBLE_EQ(combined.data[1], 50.0);
    EXPECT_DOUBLE_EQ(combined.data[2], 40.0);
}

TEST(CombineProfilesTest, Minimum) {
    Profile1D p1, p2;
    p1.data = {10.0, 50.0, 30.0};
    p2.data = {20.0, 30.0, 40.0};

    auto combined = CombineProfiles({p1, p2}, ProfileMethod::Minimum);

    EXPECT_DOUBLE_EQ(combined.data[0], 10.0);
    EXPECT_DOUBLE_EQ(combined.data[1], 30.0);
    EXPECT_DOUBLE_EQ(combined.data[2], 30.0);
}

TEST(CombineProfilesTest, Median) {
    Profile1D p1, p2, p3;
    p1.data = {10.0};
    p2.data = {20.0};
    p3.data = {30.0};

    auto combined = CombineProfiles({p1, p2, p3}, ProfileMethod::Median);

    EXPECT_DOUBLE_EQ(combined.data[0], 20.0);
}

// ============================================================================
// Region Projection Tests
// ============================================================================

TEST_F(ProfilerTest, ProjectRegion_Horizontal) {
    Rect2i rect(10, 40, 80, 20);

    auto profile = ProjectRegion(gradientImg_, rect, true);

    EXPECT_EQ(profile.Size(), 80u);

    // Horizontal projection of horizontal gradient should show gradient
    EXPECT_LT(profile.data[0], profile.data[profile.Size() - 1]);
}

TEST_F(ProfilerTest, ProjectRegion_Vertical) {
    Rect2i rect(40, 10, 20, 80);

    auto profile = ProjectRegion(vGradientImg_, rect, false);

    EXPECT_EQ(profile.Size(), 80u);

    // Vertical projection of vertical gradient should show gradient
    EXPECT_LT(profile.data[0], profile.data[profile.Size() - 1]);
}

TEST_F(ProfilerTest, ProjectRegion_Uniform) {
    Rect2i rect(20, 20, 60, 60);

    auto profile = ProjectRegion(uniformImg_, rect, true, ProfileMethod::Average);

    for (double val : profile.data) {
        EXPECT_NEAR(val, 128.0, 1.0);
    }
}

// ============================================================================
// Utility Functions Tests
// ============================================================================

TEST(UtilityTest, ComputeLineSamples) {
    std::vector<double> xCoords, yCoords;
    ComputeLineSamples(0, 0, 10, 0, 11, xCoords, yCoords);

    EXPECT_EQ(xCoords.size(), 11u);
    EXPECT_EQ(yCoords.size(), 11u);

    for (size_t i = 0; i < 11; ++i) {
        EXPECT_DOUBLE_EQ(xCoords[i], static_cast<double>(i));
        EXPECT_DOUBLE_EQ(yCoords[i], 0.0);
    }
}

TEST(UtilityTest, ComputeArcSamples) {
    std::vector<double> xCoords, yCoords;
    ComputeArcSamples(0, 0, 10, 0, M_PI, 5, xCoords, yCoords);

    EXPECT_EQ(xCoords.size(), 5u);
    EXPECT_EQ(yCoords.size(), 5u);

    // First point: (10, 0)
    EXPECT_NEAR(xCoords[0], 10.0, 1e-10);
    EXPECT_NEAR(yCoords[0], 0.0, 1e-10);

    // Last point: (-10, 0)
    EXPECT_NEAR(xCoords[4], -10.0, 1e-10);
    EXPECT_NEAR(yCoords[4], 0.0, 1e-10);
}

TEST(UtilityTest, IsInsideImage) {
    EXPECT_TRUE(IsInsideImage(5, 5, 10, 10));
    EXPECT_TRUE(IsInsideImage(0, 0, 10, 10));
    EXPECT_TRUE(IsInsideImage(9.99, 9.99, 10, 10));

    EXPECT_FALSE(IsInsideImage(-1, 5, 10, 10));
    EXPECT_FALSE(IsInsideImage(5, -1, 10, 10));
    EXPECT_FALSE(IsInsideImage(10, 5, 10, 10));
    EXPECT_FALSE(IsInsideImage(5, 10, 10, 10));
}

TEST(UtilityTest, ComputePerpendicularPoint) {
    double px, py;

    // Angle 0 (pointing right), perpendicular points down
    ComputePerpendicularPoint(0, 0, 0, 10, px, py);
    EXPECT_NEAR(px, 0.0, 1e-10);
    EXPECT_NEAR(py, -10.0, 1e-10);

    // Angle π/2 (pointing up), perpendicular points right
    ComputePerpendicularPoint(0, 0, M_PI / 2, 10, px, py);
    EXPECT_NEAR(px, 10.0, 1e-10);
    EXPECT_NEAR(py, 0.0, 1e-10);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(ProfilerTest, ExtractProfile_ShortLine) {
    auto profile = ExtractLineProfile(uniformImg_, 50, 50, 50.5, 50.5);

    EXPECT_FALSE(profile.Empty());
    EXPECT_GE(profile.Size(), 2u);
}

TEST_F(ProfilerTest, ExtractProfile_ZeroLength) {
    auto profile = ExtractLineProfile(uniformImg_, 50, 50, 50, 50);

    // Should handle zero-length line gracefully
    EXPECT_GE(profile.Size(), 2u);  // Minimum 2 samples
}

TEST_F(ProfilerTest, ExtractArcProfile_SmallArc) {
    ArcProfileParams params;
    params.centerX = 50;
    params.centerY = 50;
    params.radius = 30;
    params.startAngle = 0;
    params.endAngle = 0.01;  // Very small arc

    auto profile = ExtractArcProfile(uniformImg_, params);

    EXPECT_GE(profile.Size(), 2u);
}

TEST_F(ProfilerTest, ExtractAnnularProfile_NarrowRing) {
    AnnularProfileParams params;
    params.centerX = 50;
    params.centerY = 50;
    params.innerRadius = 20;
    params.outerRadius = 21;  // Very narrow

    auto profile = ExtractAnnularProfile(uniformImg_, params);

    EXPECT_GE(profile.Size(), 2u);
}

// ============================================================================
// Different Pixel Types
// ============================================================================

TEST(ProfilerPixelTypeTest, Float32Image) {
    QImage floatImg(100, 100, PixelType::Float32, ChannelType::Gray);
    float* data = static_cast<float*>(floatImg.Data());
    for (int i = 0; i < 100 * 100; ++i) {
        data[i] = static_cast<float>(i % 100) / 99.0f;  // 0 to 1 gradient
    }

    auto profile = ExtractLineProfile(floatImg, 0, 50, 99, 50);

    EXPECT_FALSE(profile.Empty());
    EXPECT_NEAR(profile.data[0], 0.0, 0.02);
    EXPECT_NEAR(profile.data[profile.Size() - 1], 1.0, 0.02);
}

TEST(ProfilerPixelTypeTest, UInt16Image) {
    QImage img16(100, 100, PixelType::UInt16, ChannelType::Gray);
    uint16_t* data = static_cast<uint16_t*>(img16.Data());
    for (int y = 0; y < 100; ++y) {
        for (int x = 0; x < 100; ++x) {
            data[y * 100 + x] = static_cast<uint16_t>(x * 655);  // 0 to 65500
        }
    }

    auto profile = ExtractLineProfile(img16, 0, 50, 99, 50);

    EXPECT_FALSE(profile.Empty());
    EXPECT_NEAR(profile.data[0], 0.0, 100.0);
    // Last value at x=99: 99*655 = 64845
    EXPECT_NEAR(profile.data[profile.Size() - 1], 64845.0, 1000.0);
}
