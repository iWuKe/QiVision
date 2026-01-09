/**
 * @file test_morph_gray.cpp
 * @brief Unit tests for MorphGray module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/MorphGray.h>
#include <QiVision/Internal/StructElement.h>
#include <QiVision/Core/QImage.h>

#include <cstring>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

// Create a test image with known values
QImage CreateTestImage(int32_t width, int32_t height, uint8_t value) {
    QImage img(width, height, PixelType::UInt8, ChannelType::Gray);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());

    for (int32_t r = 0; r < height; ++r) {
        std::memset(data + r * stride, value, width);
    }
    return img;
}

// Create image with a bright spot
QImage CreateBrightSpotImage(int32_t width, int32_t height,
                              int32_t cx, int32_t cy, int32_t radius,
                              uint8_t bg, uint8_t fg) {
    QImage img = CreateTestImage(width, height, bg);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());

    for (int32_t r = cy - radius; r <= cy + radius; ++r) {
        for (int32_t c = cx - radius; c <= cx + radius; ++c) {
            if (r >= 0 && r < height && c >= 0 && c < width) {
                int32_t dr = r - cy;
                int32_t dc = c - cx;
                if (dr * dr + dc * dc <= radius * radius) {
                    data[r * stride + c] = fg;
                }
            }
        }
    }
    return img;
}

// Create image with a dark spot
QImage CreateDarkSpotImage(int32_t width, int32_t height,
                            int32_t cx, int32_t cy, int32_t radius,
                            uint8_t bg, uint8_t fg) {
    return CreateBrightSpotImage(width, height, cx, cy, radius, bg, fg);
}

// Get pixel value
uint8_t GetPixel(const QImage& img, int32_t row, int32_t col) {
    const uint8_t* data = static_cast<const uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());
    return data[row * stride + col];
}

// Find max pixel value
uint8_t MaxPixel(const QImage& img) {
    uint8_t maxVal = 0;
    const uint8_t* data = static_cast<const uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());
    for (int32_t r = 0; r < img.Height(); ++r) {
        for (int32_t c = 0; c < img.Width(); ++c) {
            maxVal = std::max(maxVal, data[r * stride + c]);
        }
    }
    return maxVal;
}

// Find min pixel value
uint8_t MinPixel(const QImage& img) {
    uint8_t minVal = 255;
    const uint8_t* data = static_cast<const uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());
    for (int32_t r = 0; r < img.Height(); ++r) {
        for (int32_t c = 0; c < img.Width(); ++c) {
            minVal = std::min(minVal, data[r * stride + c]);
        }
    }
    return minVal;
}

} // anonymous namespace

// =============================================================================
// Basic Dilation Tests
// =============================================================================

TEST(MorphGrayTest, GrayDilate_Empty) {
    QImage empty;
    auto se = StructElement::Square(3);
    QImage result = GrayDilate(empty, se);
    EXPECT_TRUE(result.Empty());
}

TEST(MorphGrayTest, GrayDilate_Uniform) {
    QImage img = CreateTestImage(50, 50, 128);
    auto se = StructElement::Square(3);
    QImage result = GrayDilate(img, se);

    // Uniform image should stay uniform after dilation
    EXPECT_EQ(GetPixel(result, 25, 25), 128);
}

TEST(MorphGrayTest, GrayDilate_BrightSpot) {
    // Dark background with bright spot - dilation should expand bright area
    QImage img = CreateBrightSpotImage(50, 50, 25, 25, 3, 50, 200);

    auto se = StructElement::Square(3);
    QImage result = GrayDilate(img, se);

    // Bright region should expand
    EXPECT_EQ(GetPixel(result, 25, 25), 200);  // Center stays bright
    // Near neighbors should become bright
    EXPECT_EQ(GetPixel(result, 25 + 4, 25), 200);  // Expanded
}

TEST(MorphGrayTest, GrayDilate_SinglePixel) {
    QImage img = CreateTestImage(20, 20, 0);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());
    data[10 * stride + 10] = 255;  // Single bright pixel

    auto se = StructElement::Square(3);
    QImage result = GrayDilate(img, se);

    // 3x3 neighborhood should be bright
    for (int32_t r = 9; r <= 11; ++r) {
        for (int32_t c = 9; c <= 11; ++c) {
            EXPECT_EQ(GetPixel(result, r, c), 255);
        }
    }
}

TEST(MorphGrayTest, GrayDilateRect_Basic) {
    QImage img = CreateTestImage(50, 50, 100);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());
    data[25 * stride + 25] = 200;

    QImage result = GrayDilateRect(img, 5, 5);

    // Center should be bright
    EXPECT_EQ(GetPixel(result, 25, 25), 200);
    // 5x5 neighborhood should be affected
    EXPECT_EQ(GetPixel(result, 25 + 2, 25 + 2), 200);
}

TEST(MorphGrayTest, GrayDilateCircle_Basic) {
    QImage img = CreateTestImage(50, 50, 100);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());
    data[25 * stride + 25] = 200;

    QImage result = GrayDilateCircle(img, 2);

    EXPECT_EQ(GetPixel(result, 25, 25), 200);
}

// =============================================================================
// Basic Erosion Tests
// =============================================================================

TEST(MorphGrayTest, GrayErode_Empty) {
    QImage empty;
    auto se = StructElement::Square(3);
    QImage result = GrayErode(empty, se);
    EXPECT_TRUE(result.Empty());
}

TEST(MorphGrayTest, GrayErode_Uniform) {
    QImage img = CreateTestImage(50, 50, 128);
    auto se = StructElement::Square(3);
    QImage result = GrayErode(img, se);

    EXPECT_EQ(GetPixel(result, 25, 25), 128);
}

TEST(MorphGrayTest, GrayErode_DarkSpot) {
    // Bright background with dark spot - erosion should expand dark area
    QImage img = CreateDarkSpotImage(50, 50, 25, 25, 3, 200, 50);

    auto se = StructElement::Square(3);
    QImage result = GrayErode(img, se);

    // Dark region should expand
    EXPECT_EQ(GetPixel(result, 25, 25), 50);  // Center stays dark
}

TEST(MorphGrayTest, GrayErode_SingleDarkPixel) {
    QImage img = CreateTestImage(20, 20, 255);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());
    data[10 * stride + 10] = 0;  // Single dark pixel

    auto se = StructElement::Square(3);
    QImage result = GrayErode(img, se);

    // 3x3 neighborhood should be dark
    for (int32_t r = 9; r <= 11; ++r) {
        for (int32_t c = 9; c <= 11; ++c) {
            EXPECT_EQ(GetPixel(result, r, c), 0);
        }
    }
}

TEST(MorphGrayTest, GrayErodeRect_Basic) {
    QImage img = CreateTestImage(50, 50, 200);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());
    data[25 * stride + 25] = 50;

    QImage result = GrayErodeRect(img, 5, 5);

    EXPECT_EQ(GetPixel(result, 25, 25), 50);
}

// =============================================================================
// Opening Tests
// =============================================================================

TEST(MorphGrayTest, GrayOpening_Empty) {
    QImage empty;
    auto se = StructElement::Square(3);
    QImage result = GrayOpening(empty, se);
    EXPECT_TRUE(result.Empty());
}

TEST(MorphGrayTest, GrayOpening_RemovesBrightSpots) {
    // Opening removes bright spots smaller than SE
    QImage img = CreateTestImage(50, 50, 100);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());
    // Single bright pixel
    data[25 * stride + 25] = 200;

    auto se = StructElement::Square(3);
    QImage result = GrayOpening(img, se);

    // Single bright pixel should be removed
    EXPECT_LT(GetPixel(result, 25, 25), 200);
}

TEST(MorphGrayTest, GrayOpening_PreservesLargeBrightRegion) {
    // Large bright region should survive opening
    QImage img = CreateBrightSpotImage(50, 50, 25, 25, 5, 100, 200);

    auto se = StructElement::Square(3);
    QImage result = GrayOpening(img, se);

    // Center should still be bright
    EXPECT_EQ(GetPixel(result, 25, 25), 200);
}

TEST(MorphGrayTest, GrayOpeningRect_Basic) {
    QImage img = CreateTestImage(50, 50, 100);
    QImage result = GrayOpeningRect(img, 3, 3);
    EXPECT_EQ(GetPixel(result, 25, 25), 100);
}

// =============================================================================
// Closing Tests
// =============================================================================

TEST(MorphGrayTest, GrayClosing_Empty) {
    QImage empty;
    auto se = StructElement::Square(3);
    QImage result = GrayClosing(empty, se);
    EXPECT_TRUE(result.Empty());
}

TEST(MorphGrayTest, GrayClosing_FillsDarkSpots) {
    // Closing fills dark spots smaller than SE
    QImage img = CreateTestImage(50, 50, 200);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());
    // Single dark pixel
    data[25 * stride + 25] = 50;

    auto se = StructElement::Square(3);
    QImage result = GrayClosing(img, se);

    // Single dark pixel should be filled
    EXPECT_GT(GetPixel(result, 25, 25), 50);
}

TEST(MorphGrayTest, GrayClosingRect_Basic) {
    QImage img = CreateTestImage(50, 50, 100);
    QImage result = GrayClosingRect(img, 3, 3);
    EXPECT_EQ(GetPixel(result, 25, 25), 100);
}

// =============================================================================
// Gradient Tests
// =============================================================================

TEST(MorphGrayTest, GrayMorphGradient_Uniform) {
    QImage img = CreateTestImage(50, 50, 128);
    auto se = StructElement::Square(3);
    QImage result = GrayMorphGradient(img, se);

    // Uniform image has zero gradient
    EXPECT_EQ(GetPixel(result, 25, 25), 0);
}

TEST(MorphGrayTest, GrayMorphGradient_Edge) {
    // Create image with edge
    QImage img = CreateTestImage(50, 50, 50);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());
    for (int32_t r = 0; r < 50; ++r) {
        for (int32_t c = 25; c < 50; ++c) {
            data[r * stride + c] = 200;
        }
    }

    auto se = StructElement::Square(3);
    QImage result = GrayMorphGradient(img, se);

    // Should have high gradient at edge
    EXPECT_GT(GetPixel(result, 25, 25), 100);
}

TEST(MorphGrayTest, GrayInternalGradient_Basic) {
    QImage img = CreateBrightSpotImage(50, 50, 25, 25, 5, 50, 200);

    auto se = StructElement::Square(3);
    QImage result = GrayInternalGradient(img, se);

    // Non-zero result expected
    EXPECT_GT(MaxPixel(result), 0);
}

TEST(MorphGrayTest, GrayExternalGradient_Basic) {
    QImage img = CreateBrightSpotImage(50, 50, 25, 25, 5, 50, 200);

    auto se = StructElement::Square(3);
    QImage result = GrayExternalGradient(img, se);

    EXPECT_GT(MaxPixel(result), 0);
}

// =============================================================================
// TopHat/BlackHat Tests
// =============================================================================

TEST(MorphGrayTest, GrayTopHat_ExtractsBrightSpots) {
    // TopHat extracts small bright features
    QImage img = CreateTestImage(50, 50, 100);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());
    // Small bright spot
    data[25 * stride + 25] = 200;

    auto se = StructElement::Square(5);
    QImage result = GrayTopHat(img, se);

    // Bright spot should appear in tophat
    EXPECT_GT(GetPixel(result, 25, 25), 50);
}

TEST(MorphGrayTest, GrayBlackHat_ExtractsDarkSpots) {
    // BlackHat extracts small dark features
    QImage img = CreateTestImage(50, 50, 200);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());
    // Small dark spot
    data[25 * stride + 25] = 50;

    auto se = StructElement::Square(5);
    QImage result = GrayBlackHat(img, se);

    // Dark spot should appear in blackhat
    EXPECT_GT(GetPixel(result, 25, 25), 50);
}

// =============================================================================
// Range Tests
// =============================================================================

TEST(MorphGrayTest, GrayRangeRect_Uniform) {
    QImage img = CreateTestImage(50, 50, 128);
    QImage result = GrayRangeRect(img, 5, 5);

    // Uniform image has zero range
    EXPECT_EQ(GetPixel(result, 25, 25), 0);
}

TEST(MorphGrayTest, GrayRangeRect_Edge) {
    // Create image with edge
    QImage img = CreateTestImage(50, 50, 50);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());
    for (int32_t r = 0; r < 50; ++r) {
        for (int32_t c = 25; c < 50; ++c) {
            data[r * stride + c] = 200;
        }
    }

    QImage result = GrayRangeRect(img, 5, 5);

    // High range at edge
    EXPECT_GT(GetPixel(result, 25, 25), 100);
}

TEST(MorphGrayTest, GrayRangeCircle_Basic) {
    QImage img = CreateBrightSpotImage(50, 50, 25, 25, 5, 50, 200);
    QImage result = GrayRangeCircle(img, 3);

    EXPECT_GT(MaxPixel(result), 0);
}

// =============================================================================
// Iterative Operations Tests
// =============================================================================

TEST(MorphGrayTest, GrayDilateN_Basic) {
    QImage img = CreateTestImage(50, 50, 100);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());
    data[25 * stride + 25] = 200;

    auto se = StructElement::Square(3);
    QImage result1 = GrayDilate(img, se);
    QImage result2 = GrayDilateN(img, se, 2);

    // Two dilations should expand more
    // Count bright pixels
    int count1 = 0, count2 = 0;
    for (int32_t r = 0; r < 50; ++r) {
        for (int32_t c = 0; c < 50; ++c) {
            if (GetPixel(result1, r, c) == 200) count1++;
            if (GetPixel(result2, r, c) == 200) count2++;
        }
    }
    EXPECT_GT(count2, count1);
}

TEST(MorphGrayTest, GrayErodeN_Basic) {
    QImage img = CreateTestImage(50, 50, 200);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());
    data[25 * stride + 25] = 50;

    auto se = StructElement::Square(3);
    QImage result1 = GrayErode(img, se);
    QImage result2 = GrayErodeN(img, se, 2);

    // Two erosions should expand dark region more
    int count1 = 0, count2 = 0;
    for (int32_t r = 0; r < 50; ++r) {
        for (int32_t c = 0; c < 50; ++c) {
            if (GetPixel(result1, r, c) == 50) count1++;
            if (GetPixel(result2, r, c) == 50) count2++;
        }
    }
    EXPECT_GT(count2, count1);
}

// =============================================================================
// Geodesic Operations Tests
// =============================================================================

TEST(MorphGrayTest, GrayGeodesicDilate_Basic) {
    // Marker: small bright area
    QImage marker = CreateTestImage(50, 50, 50);
    uint8_t* markerData = static_cast<uint8_t*>(marker.Data());
    int32_t markerStride = static_cast<int32_t>(marker.Stride());
    markerData[25 * markerStride + 25] = 150;

    // Mask: larger bright region
    QImage mask = CreateBrightSpotImage(50, 50, 25, 25, 10, 50, 200);

    auto se = StructElement::Square(3);
    QImage result = GrayGeodesicDilate(marker, mask, se);

    // Result should be >= marker and <= mask
    EXPECT_GE(GetPixel(result, 25, 25), 150);
    EXPECT_LE(GetPixel(result, 25, 25), 200);
}

TEST(MorphGrayTest, GrayReconstructByDilation_Basic) {
    // Marker: seed point at mask value
    QImage marker = CreateTestImage(50, 50, 50);
    uint8_t* markerData = static_cast<uint8_t*>(marker.Data());
    int32_t markerStride = static_cast<int32_t>(marker.Stride());
    markerData[25 * markerStride + 25] = 200;  // Same as mask peak

    // Mask: connected bright region
    QImage mask = CreateBrightSpotImage(50, 50, 25, 25, 5, 50, 200);

    QImage result = GrayReconstructByDilation(marker, mask);

    // Result should reconstruct the bright region from seed
    EXPECT_EQ(GetPixel(result, 25, 25), 200);
    // Nearby pixels in mask should also be reconstructed
    EXPECT_EQ(GetPixel(result, 25 + 2, 25), 200);
}

TEST(MorphGrayTest, GrayOpeningByReconstruction_Basic) {
    QImage img = CreateBrightSpotImage(50, 50, 25, 25, 5, 50, 200);

    auto se = StructElement::Square(3);
    QImage result = GrayOpeningByReconstruction(img, se);

    // Should preserve shape better than regular opening
    EXPECT_GT(GetPixel(result, 25, 25), 100);
}

TEST(MorphGrayTest, GrayFillHoles_Basic) {
    // Create image with a hole (dark region inside bright region)
    QImage img = CreateTestImage(50, 50, 200);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());
    // Create hole in center
    for (int32_t r = 23; r <= 27; ++r) {
        for (int32_t c = 23; c <= 27; ++c) {
            data[r * stride + c] = 50;
        }
    }

    QImage result = GrayFillHoles(img);

    // Hole should be filled
    EXPECT_GT(GetPixel(result, 25, 25), 50);
}

// =============================================================================
// Background Correction Tests
// =============================================================================

TEST(MorphGrayTest, RollingBallBackground_Basic) {
    // Create image with uneven background + features
    QImage img = CreateTestImage(50, 50, 100);
    uint8_t* data = static_cast<uint8_t*>(img.Data());
    int32_t stride = static_cast<int32_t>(img.Stride());

    // Add gradient background
    for (int32_t r = 0; r < 50; ++r) {
        for (int32_t c = 0; c < 50; ++c) {
            data[r * stride + c] = static_cast<uint8_t>(50 + r);
        }
    }
    // Add small feature
    data[25 * stride + 25] = 200;

    QImage result = RollingBallBackground(img, 5);

    // Background should be flattened
    // The result should have less variation
    EXPECT_LT(MaxPixel(result) - MinPixel(result), 200);
}

TEST(MorphGrayTest, EstimateBackground_Basic) {
    QImage img = CreateBrightSpotImage(50, 50, 25, 25, 3, 100, 200);

    auto se = StructElement::Circle(5);
    QImage background = EstimateBackground(img, se);

    // Background should not contain the small bright spot
    EXPECT_LT(GetPixel(background, 25, 25), 200);
}

TEST(MorphGrayTest, SubtractBackground_Basic) {
    QImage img = CreateTestImage(50, 50, 150);
    QImage bg = CreateTestImage(50, 50, 100);

    QImage result = SubtractBackground(img, bg, 0);

    EXPECT_EQ(GetPixel(result, 25, 25), 50);
}

TEST(MorphGrayTest, SubtractBackground_WithOffset) {
    QImage img = CreateTestImage(50, 50, 150);
    QImage bg = CreateTestImage(50, 50, 100);

    QImage result = SubtractBackground(img, bg, 50);

    EXPECT_EQ(GetPixel(result, 25, 25), 100);  // 150 - 100 + 50 = 100
}

TEST(MorphGrayTest, SubtractBackground_Saturate) {
    QImage img = CreateTestImage(50, 50, 50);
    QImage bg = CreateTestImage(50, 50, 100);

    QImage result = SubtractBackground(img, bg, 0);

    // Should saturate at 0
    EXPECT_EQ(GetPixel(result, 25, 25), 0);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(MorphGrayTest, GrayDilate_EmptySE) {
    QImage img = CreateTestImage(50, 50, 128);
    StructElement emptySE;

    QImage result = GrayDilate(img, emptySE);

    // Empty SE returns original
    EXPECT_FALSE(result.Empty());
}

TEST(MorphGrayTest, GrayDilateRect_InvalidSize) {
    QImage img = CreateTestImage(50, 50, 128);

    QImage result = GrayDilateRect(img, 0, 0);

    // Invalid size returns original
    EXPECT_FALSE(result.Empty());
}

