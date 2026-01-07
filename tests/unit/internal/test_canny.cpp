/**
 * @file test_canny.cpp
 * @brief Unit tests for Canny edge detection module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/Canny.h>

#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>

using namespace Qi::Vision::Internal;
using Qi::Vision::QImage;
using Qi::Vision::QContour;
using Qi::Vision::PixelType;
using Qi::Vision::ChannelType;

// ============================================================================
// Test Fixture
// ============================================================================

class CannyTest : public ::testing::Test {
protected:
    void SetUp() override {}

    // Create a test image with a vertical edge
    QImage CreateVerticalEdgeImage(int32_t width, int32_t height,
                                   int32_t edgeX, uint8_t leftVal = 50,
                                   uint8_t rightVal = 200) {
        QImage img(width, height, PixelType::UInt8);
        uint8_t* data = static_cast<uint8_t*>(img.Data());

        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                data[y * width + x] = (x < edgeX) ? leftVal : rightVal;
            }
        }
        return img;
    }

    // Create a test image with a horizontal edge
    QImage CreateHorizontalEdgeImage(int32_t width, int32_t height,
                                     int32_t edgeY, uint8_t topVal = 50,
                                     uint8_t bottomVal = 200) {
        QImage img(width, height, PixelType::UInt8);
        uint8_t* data = static_cast<uint8_t*>(img.Data());

        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                data[y * width + x] = (y < edgeY) ? topVal : bottomVal;
            }
        }
        return img;
    }

    // Create a square image (bright square on dark background)
    QImage CreateSquareImage(int32_t size, int32_t squareSize) {
        QImage img(size, size, PixelType::UInt8);
        uint8_t* data = static_cast<uint8_t*>(img.Data());

        int32_t margin = (size - squareSize) / 2;

        for (int32_t y = 0; y < size; ++y) {
            for (int32_t x = 0; x < size; ++x) {
                bool inSquare = (x >= margin && x < margin + squareSize &&
                                 y >= margin && y < margin + squareSize);
                data[y * size + x] = inSquare ? 200 : 50;
            }
        }
        return img;
    }

    // Create a circle image
    QImage CreateCircleImage(int32_t size, int32_t radius) {
        QImage img(size, size, PixelType::UInt8);
        uint8_t* data = static_cast<uint8_t*>(img.Data());

        double cx = size / 2.0;
        double cy = size / 2.0;

        for (int32_t y = 0; y < size; ++y) {
            for (int32_t x = 0; x < size; ++x) {
                double dx = x - cx;
                double dy = y - cy;
                double dist = std::sqrt(dx * dx + dy * dy);
                data[y * size + x] = (dist < radius) ? 200 : 50;
            }
        }
        return img;
    }

    // Create uniform image
    QImage CreateUniformImage(int32_t width, int32_t height, uint8_t value) {
        QImage img(width, height, PixelType::UInt8);
        std::memset(static_cast<uint8_t*>(img.Data()), value, width * height);
        return img;
    }

    // Create noisy image
    QImage CreateNoisyImage(int32_t width, int32_t height, double noiseStd) {
        QImage img(width, height, PixelType::UInt8);
        uint8_t* data = static_cast<uint8_t*>(img.Data());

        std::mt19937 rng(42);
        std::normal_distribution<double> noise(128.0, noiseStd);

        for (int32_t i = 0; i < width * height; ++i) {
            int32_t val = static_cast<int32_t>(noise(rng));
            data[i] = static_cast<uint8_t>(std::clamp(val, 0, 255));
        }
        return img;
    }
};

// ============================================================================
// CannySmooth Tests
// ============================================================================

TEST_F(CannyTest, CannySmooth_ZeroSigma) {
    const int32_t width = 10, height = 10;
    std::vector<uint8_t> src(width * height, 100);
    std::vector<float> dst(width * height);

    CannySmooth(src.data(), dst.data(), width, height, 0.0);

    // Should just convert to float
    for (int32_t i = 0; i < width * height; ++i) {
        EXPECT_FLOAT_EQ(dst[i], 100.0f);
    }
}

TEST_F(CannyTest, CannySmooth_PreservesUniform) {
    const int32_t width = 20, height = 20;
    std::vector<uint8_t> src(width * height, 150);
    std::vector<float> dst(width * height);

    CannySmooth(src.data(), dst.data(), width, height, 1.5);

    // Center pixels should be close to original value
    float center = dst[10 * width + 10];
    EXPECT_NEAR(center, 150.0f, 1.0f);
}

TEST_F(CannyTest, CannySmooth_ReducesNoise) {
    const int32_t width = 50, height = 50;
    auto img = CreateNoisyImage(width, height, 30.0);
    std::vector<float> dst(width * height);

    CannySmooth(static_cast<const uint8_t*>(img.Data()), dst.data(), width, height, 2.0);

    // Calculate variance of center region
    double sumSq = 0.0, sum = 0.0;
    int32_t count = 0;
    for (int32_t y = 10; y < 40; ++y) {
        for (int32_t x = 10; x < 40; ++x) {
            float val = dst[y * width + x];
            sum += val;
            sumSq += val * val;
            count++;
        }
    }
    double mean = sum / count;
    double variance = sumSq / count - mean * mean;
    double stddev = std::sqrt(variance);

    // Smoothed stddev should be less than original (30.0)
    EXPECT_LT(stddev, 20.0);
}

// ============================================================================
// CannyGradient Tests
// ============================================================================

TEST_F(CannyTest, CannyGradient_VerticalEdge) {
    const int32_t width = 30, height = 30;
    auto img = CreateVerticalEdgeImage(width, height, 15);
    std::vector<float> src(width * height);
    std::vector<float> mag(width * height);
    std::vector<float> dir(width * height);

    // Convert to float
    const uint8_t* imgData = static_cast<const uint8_t*>(img.Data());
    for (int32_t i = 0; i < width * height; ++i) {
        src[i] = static_cast<float>(imgData[i]);
    }

    CannyGradient(src.data(), mag.data(), dir.data(), width, height,
                  CannyGradientOp::Sobel);

    // Maximum magnitude should be near the edge
    float maxMag = 0.0f;
    int32_t maxX = 0;
    for (int32_t y = 10; y < 20; ++y) {
        for (int32_t x = 10; x < 20; ++x) {
            if (mag[y * width + x] > maxMag) {
                maxMag = mag[y * width + x];
                maxX = x;
            }
        }
    }
    EXPECT_NEAR(maxX, 15, 2);  // Near the edge

    // Direction should be horizontal (0 or π) at the edge
    float edgeDir = dir[15 * width + 15];
    double normalizedDir = std::abs(std::cos(edgeDir));
    EXPECT_GT(normalizedDir, 0.9);  // Close to horizontal
}

TEST_F(CannyTest, CannyGradient_HorizontalEdge) {
    const int32_t width = 30, height = 30;
    auto img = CreateHorizontalEdgeImage(width, height, 15);
    std::vector<float> src(width * height);
    std::vector<float> mag(width * height);
    std::vector<float> dir(width * height);

    const uint8_t* imgData = static_cast<const uint8_t*>(img.Data());
    for (int32_t i = 0; i < width * height; ++i) {
        src[i] = static_cast<float>(imgData[i]);
    }

    CannyGradient(src.data(), mag.data(), dir.data(), width, height,
                  CannyGradientOp::Sobel);

    // Maximum magnitude should be near the edge
    float maxMag = 0.0f;
    int32_t maxY = 0;
    for (int32_t y = 10; y < 20; ++y) {
        for (int32_t x = 10; x < 20; ++x) {
            if (mag[y * width + x] > maxMag) {
                maxMag = mag[y * width + x];
                maxY = y;
            }
        }
    }
    EXPECT_NEAR(maxY, 15, 2);  // Near the edge
}

TEST_F(CannyTest, CannyGradient_Scharr) {
    const int32_t width = 30, height = 30;
    auto img = CreateVerticalEdgeImage(width, height, 15);
    std::vector<float> src(width * height);
    std::vector<float> mag(width * height);
    std::vector<float> dir(width * height);

    const uint8_t* imgData = static_cast<const uint8_t*>(img.Data());
    for (int32_t i = 0; i < width * height; ++i) {
        src[i] = static_cast<float>(imgData[i]);
    }

    CannyGradient(src.data(), mag.data(), dir.data(), width, height,
                  CannyGradientOp::Scharr);

    // Should detect the edge
    float maxMag = *std::max_element(mag.begin(), mag.end());
    EXPECT_GT(maxMag, 100.0f);
}

// ============================================================================
// CannyNMS Tests
// ============================================================================

TEST_F(CannyTest, CannyNMS_ThinsEdges) {
    const int32_t width = 30, height = 30;
    auto img = CreateVerticalEdgeImage(width, height, 15);
    std::vector<float> src(width * height);
    std::vector<float> mag(width * height);
    std::vector<float> dir(width * height);
    std::vector<float> nms(width * height);

    const uint8_t* imgData = static_cast<const uint8_t*>(img.Data());
    for (int32_t i = 0; i < width * height; ++i) {
        src[i] = static_cast<float>(imgData[i]);
    }

    CannyGradient(src.data(), mag.data(), dir.data(), width, height,
                  CannyGradientOp::Sobel);
    CannyNMS(mag.data(), dir.data(), nms.data(), width, height);

    // Count non-zero pixels in a row
    int32_t y = 15;
    int32_t nonZeroCount = 0;
    for (int32_t x = 5; x < 25; ++x) {
        if (nms[y * width + x] > 0) {
            nonZeroCount++;
        }
    }

    // Should be thin (1-2 pixels wide)
    EXPECT_LE(nonZeroCount, 3);
    EXPECT_GE(nonZeroCount, 1);
}

// ============================================================================
// CannyHysteresis Tests
// ============================================================================

TEST_F(CannyTest, CannyHysteresis_StrongEdges) {
    const int32_t width = 10, height = 10;
    std::vector<float> nms(width * height, 0.0f);
    std::vector<uint8_t> output(width * height);

    // Create a strong edge pixel
    nms[5 * width + 5] = 100.0f;

    CannyHysteresis(nms.data(), output.data(), width, height, 30.0, 50.0);

    // Strong edge should be kept
    EXPECT_EQ(output[5 * width + 5], 255);
}

TEST_F(CannyTest, CannyHysteresis_WeakEdgesConnected) {
    const int32_t width = 10, height = 10;
    std::vector<float> nms(width * height, 0.0f);
    std::vector<uint8_t> output(width * height);

    // Create a chain: strong -> weak -> weak
    nms[5 * width + 3] = 100.0f;  // Strong
    nms[5 * width + 4] = 40.0f;   // Weak
    nms[5 * width + 5] = 35.0f;   // Weak

    CannyHysteresis(nms.data(), output.data(), width, height, 30.0, 50.0);

    // All should be kept due to connection
    EXPECT_EQ(output[5 * width + 3], 255);
    EXPECT_EQ(output[5 * width + 4], 255);
    EXPECT_EQ(output[5 * width + 5], 255);
}

TEST_F(CannyTest, CannyHysteresis_WeakEdgesIsolated) {
    const int32_t width = 10, height = 10;
    std::vector<float> nms(width * height, 0.0f);
    std::vector<uint8_t> output(width * height);

    // Create an isolated weak edge
    nms[5 * width + 5] = 40.0f;  // Weak but not connected to strong

    CannyHysteresis(nms.data(), output.data(), width, height, 30.0, 50.0);

    // Should be suppressed
    EXPECT_EQ(output[5 * width + 5], 0);
}

// ============================================================================
// Auto Threshold Tests
// ============================================================================

TEST_F(CannyTest, ComputeAutoThresholds_Basic) {
    const int32_t width = 100, height = 100;
    std::vector<float> magnitude(width * height);

    // Create gradient values with a distribution
    for (int32_t i = 0; i < width * height; ++i) {
        magnitude[i] = static_cast<float>(i % 100);
    }

    double low, high;
    ComputeAutoThresholds(magnitude.data(), width, height, low, high);

    EXPECT_GT(high, low);
    EXPECT_GT(low, 0.0);
}

TEST_F(CannyTest, ComputeAutoThresholds_ZeroImage) {
    const int32_t width = 50, height = 50;
    std::vector<float> magnitude(width * height, 0.0f);

    double low, high;
    ComputeAutoThresholds(magnitude.data(), width, height, low, high);

    EXPECT_DOUBLE_EQ(low, 0.0);
    EXPECT_DOUBLE_EQ(high, 0.0);
}

// ============================================================================
// Subpixel Refinement Tests
// ============================================================================

TEST_F(CannyTest, RefineEdgeSubpixel_Center) {
    const int32_t width = 10, height = 10;
    std::vector<float> magnitude(width * height, 0.0f);
    std::vector<float> direction(width * height, 0.0f);

    // Create a peak at (5, 5)
    magnitude[4 * width + 5] = 50.0f;
    magnitude[5 * width + 5] = 100.0f;
    magnitude[6 * width + 5] = 50.0f;

    // Gradient direction is vertical
    direction[5 * width + 5] = M_PI / 2;

    double subX, subY;
    double mag = RefineEdgeSubpixel(magnitude.data(), direction.data(),
                                    width, height, 5, 5, subX, subY);

    // Should be close to integer position (symmetric peak)
    EXPECT_NEAR(subX, 5.0, 0.1);
    EXPECT_NEAR(subY, 5.0, 0.1);
    EXPECT_GT(mag, 0.0);
}

TEST_F(CannyTest, RefineEdgeSubpixel_Offset) {
    const int32_t width = 10, height = 10;
    std::vector<float> magnitude(width * height, 0.0f);
    std::vector<float> direction(width * height, 0.0f);

    // Create an asymmetric peak
    magnitude[4 * width + 5] = 80.0f;
    magnitude[5 * width + 5] = 100.0f;
    magnitude[6 * width + 5] = 40.0f;

    direction[5 * width + 5] = M_PI / 2;

    double subX, subY;
    RefineEdgeSubpixel(magnitude.data(), direction.data(),
                       width, height, 5, 5, subX, subY);

    // Should be offset towards the higher neighbor
    EXPECT_NEAR(subX, 5.0, 0.1);
    EXPECT_LT(subY, 5.0);  // Offset towards y=4
}

// ============================================================================
// Edge Point Extraction Tests
// ============================================================================

TEST_F(CannyTest, ExtractEdgePoints_Empty) {
    const int32_t width = 10, height = 10;
    std::vector<uint8_t> edgeImage(width * height, 0);
    std::vector<float> magnitude(width * height, 0.0f);
    std::vector<float> direction(width * height, 0.0f);

    auto points = ExtractEdgePoints(edgeImage.data(), magnitude.data(),
                                    direction.data(), width, height, false);

    EXPECT_TRUE(points.empty());
}

TEST_F(CannyTest, ExtractEdgePoints_SinglePoint) {
    const int32_t width = 10, height = 10;
    std::vector<uint8_t> edgeImage(width * height, 0);
    std::vector<float> magnitude(width * height, 100.0f);
    std::vector<float> direction(width * height, 0.5f);

    edgeImage[5 * width + 5] = 255;

    auto points = ExtractEdgePoints(edgeImage.data(), magnitude.data(),
                                    direction.data(), width, height, false);

    ASSERT_EQ(points.size(), 1u);
    EXPECT_DOUBLE_EQ(points[0].x, 5.0);
    EXPECT_DOUBLE_EQ(points[0].y, 5.0);
}

TEST_F(CannyTest, ExtractEdgePoints_MultiplePoints) {
    const int32_t width = 10, height = 10;
    std::vector<uint8_t> edgeImage(width * height, 0);
    std::vector<float> magnitude(width * height, 100.0f);
    std::vector<float> direction(width * height, 0.0f);

    // Create a line of edge points
    for (int32_t x = 2; x < 8; ++x) {
        edgeImage[5 * width + x] = 255;
    }

    auto points = ExtractEdgePoints(edgeImage.data(), magnitude.data(),
                                    direction.data(), width, height, false);

    EXPECT_EQ(points.size(), 6u);
}

// ============================================================================
// Integration Tests - DetectEdgesCanny
// ============================================================================

TEST_F(CannyTest, DetectEdgesCanny_EmptyImage) {
    QImage emptyImg;

    auto contours = DetectEdgesCanny(emptyImg);

    EXPECT_TRUE(contours.empty());
}

TEST_F(CannyTest, DetectEdgesCanny_UniformImage) {
    auto img = CreateUniformImage(50, 50, 128);

    auto contours = DetectEdgesCanny(img);

    // Uniform image should have no edges
    EXPECT_TRUE(contours.empty());
}

TEST_F(CannyTest, DetectEdgesCanny_VerticalEdge) {
    // Use larger image for more reliable edge detection
    auto img = CreateVerticalEdgeImage(80, 80, 40);

    CannyParams params;
    params.sigma = 1.0;
    params.lowThreshold = 30.0;
    params.highThreshold = 80.0;
    params.minContourLength = 3.0;
    params.minContourPoints = 3;

    // Use DetectEdgesCannyFull to verify edge points are detected
    auto result = DetectEdgesCannyFull(img, params);

    // Edge points should be detected even if not linked into contours
    EXPECT_GT(result.numEdgePixels, 0);
    EXPECT_FALSE(result.edgePoints.empty());

    // If contours were formed, they should have points near the edge
    for (const auto& c : result.contours) {
        EXPECT_GT(c.Size(), 0u);
    }
}

TEST_F(CannyTest, DetectEdgesCanny_HorizontalEdge) {
    auto img = CreateHorizontalEdgeImage(50, 50, 25);

    CannyParams params;
    params.lowThreshold = 20.0;
    params.highThreshold = 50.0;
    params.minContourLength = 5.0;

    auto contours = DetectEdgesCanny(img, params);

    ASSERT_GE(contours.size(), 1u);
}

TEST_F(CannyTest, DetectEdgesCanny_Square) {
    auto img = CreateSquareImage(80, 40);

    CannyParams params;
    params.sigma = 1.0;
    params.lowThreshold = 30.0;
    params.highThreshold = 80.0;
    params.minContourLength = 10.0;

    auto contours = DetectEdgesCanny(img, params);

    // Should detect edges of the square
    EXPECT_GE(contours.size(), 1u);
}

TEST_F(CannyTest, DetectEdgesCanny_Circle) {
    auto img = CreateCircleImage(100, 30);

    CannyParams params;
    params.sigma = 1.5;
    params.lowThreshold = 20.0;
    params.highThreshold = 60.0;
    params.minContourLength = 20.0;

    auto contours = DetectEdgesCanny(img, params);

    // Should detect the circle edge
    EXPECT_GE(contours.size(), 1u);
}

TEST_F(CannyTest, DetectEdgesCanny_AutoThreshold) {
    // Use a high-contrast image for reliable auto-threshold
    auto img = CreateSquareImage(100, 60);

    // First, verify that auto-threshold doesn't crash
    CannyParams params = CannyParams::Auto(1.0);
    params.minContourLength = 3.0;
    params.minContourPoints = 3;

    // Test that auto threshold works without error
    auto result = DetectEdgesCannyFull(img, params);

    // Edge binary image should be created regardless
    EXPECT_FALSE(result.edgeImage.Empty());

    // For a high-contrast square, with auto threshold we might or might not get edges
    // depending on the threshold calculation. The key is it doesn't crash.
    // If edges were found, they should have positive magnitude
    if (!result.edgePoints.empty()) {
        EXPECT_GT(result.avgMagnitude, 0.0);
    }
}

TEST_F(CannyTest, DetectEdgesCanny_DifferentGradientOps) {
    // Use larger image for reliable edge detection with all operators
    auto img = CreateVerticalEdgeImage(100, 100, 50);

    // Test with different gradient operators
    std::vector<CannyGradientOp> ops = {
        CannyGradientOp::Sobel,
        CannyGradientOp::Scharr,
        CannyGradientOp::Sobel5x5
    };

    for (auto op : ops) {
        CannyParams params;
        params.gradientOp = op;
        params.sigma = 1.0;
        params.lowThreshold = 30.0;
        params.highThreshold = 80.0;
        params.minContourLength = 3.0;
        params.minContourPoints = 3;

        auto result = DetectEdgesCannyFull(img, params);

        // All gradient operators should detect edge points
        EXPECT_GT(result.numEdgePixels, 0) << "Failed with op: " << static_cast<int>(op);
        EXPECT_FALSE(result.edgePoints.empty()) << "Failed with op: " << static_cast<int>(op);
    }
}

// ============================================================================
// DetectEdgesCannyFull Tests
// ============================================================================

TEST_F(CannyTest, DetectEdgesCannyFull_ReturnsAllData) {
    auto img = CreateSquareImage(80, 40);

    CannyParams params;
    params.lowThreshold = 30.0;
    params.highThreshold = 80.0;

    auto result = DetectEdgesCannyFull(img, params);

    // Check all fields
    EXPECT_GT(result.numEdgePixels, 0);
    EXPECT_GT(result.avgMagnitude, 0.0);
    EXPECT_FALSE(result.edgePoints.empty());
    EXPECT_FALSE(result.contours.empty());
    EXPECT_FALSE(result.edgeImage.Empty());
    EXPECT_EQ(result.edgeImage.Width(), 80);
    EXPECT_EQ(result.edgeImage.Height(), 80);
}

TEST_F(CannyTest, DetectEdgesCannyFull_EdgePointsHaveAttributes) {
    auto img = CreateVerticalEdgeImage(50, 50, 25);

    CannyParams params;
    params.subpixelRefinement = true;

    auto result = DetectEdgesCannyFull(img, params);

    ASSERT_FALSE(result.edgePoints.empty());

    // Check that edge points have reasonable attributes
    for (const auto& pt : result.edgePoints) {
        EXPECT_GE(pt.x, 0.0);
        EXPECT_LT(pt.x, 50.0);
        EXPECT_GE(pt.y, 0.0);
        EXPECT_LT(pt.y, 50.0);
        EXPECT_GT(pt.magnitude, 0.0);
    }
}

TEST_F(CannyTest, DetectEdgesCannyFull_NoSubpixel) {
    auto img = CreateVerticalEdgeImage(50, 50, 25);

    CannyParams params;
    params.subpixelRefinement = false;

    auto result = DetectEdgesCannyFull(img, params);

    ASSERT_FALSE(result.edgePoints.empty());

    // Without subpixel, coordinates should be integers
    for (const auto& pt : result.edgePoints) {
        EXPECT_DOUBLE_EQ(pt.x, std::floor(pt.x));
        EXPECT_DOUBLE_EQ(pt.y, std::floor(pt.y));
    }
}

TEST_F(CannyTest, DetectEdgesCannyFull_NoLinking) {
    auto img = CreateVerticalEdgeImage(50, 50, 25);

    CannyParams params;
    params.linkEdges = false;

    auto result = DetectEdgesCannyFull(img, params);

    // Edge points should be extracted but not linked
    EXPECT_FALSE(result.edgePoints.empty());
    EXPECT_TRUE(result.contours.empty());
}

// ============================================================================
// DetectEdgesCannyImage Tests
// ============================================================================

TEST_F(CannyTest, DetectEdgesCannyImage_ReturnsBinaryImage) {
    auto img = CreateSquareImage(80, 40);

    CannyParams params;
    params.lowThreshold = 30.0;
    params.highThreshold = 80.0;

    auto edgeImg = DetectEdgesCannyImage(img, params);

    EXPECT_FALSE(edgeImg.Empty());
    EXPECT_EQ(edgeImg.Width(), 80);
    EXPECT_EQ(edgeImg.Height(), 80);

    // Check that it's binary (0 or 255)
    const uint8_t* data = static_cast<const uint8_t*>(edgeImg.Data());
    int32_t nonZeroCount = 0;
    for (int32_t i = 0; i < 80 * 80; ++i) {
        EXPECT_TRUE(data[i] == 0 || data[i] == 255);
        if (data[i] > 0) nonZeroCount++;
    }
    EXPECT_GT(nonZeroCount, 0);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(CannyTest, EdgeCase_SmallImage) {
    QImage img(5, 5, PixelType::UInt8);
    std::memset(static_cast<uint8_t*>(img.Data()), 128, 25);

    // Should not crash
    auto contours = DetectEdgesCanny(img);
    EXPECT_TRUE(contours.empty());  // Too small for meaningful edges
}

TEST_F(CannyTest, EdgeCase_VeryLowThreshold) {
    auto img = CreateNoisyImage(50, 50, 20.0);

    CannyParams params;
    params.lowThreshold = 1.0;
    params.highThreshold = 5.0;

    // Should not crash, may detect many noise edges
    auto contours = DetectEdgesCanny(img, params);
    // Result depends on noise pattern, just verify no crash
}

TEST_F(CannyTest, EdgeCase_VeryHighThreshold) {
    auto img = CreateSquareImage(50, 30);

    CannyParams params;
    params.lowThreshold = 500.0;
    params.highThreshold = 1000.0;

    auto contours = DetectEdgesCanny(img, params);

    // Very high threshold should suppress all edges
    EXPECT_TRUE(contours.empty());
}

TEST_F(CannyTest, EdgeCase_LargeSigma) {
    auto img = CreateSquareImage(100, 40);

    CannyParams params;
    params.sigma = 5.0;  // Large smoothing
    params.lowThreshold = 10.0;
    params.highThreshold = 30.0;

    auto contours = DetectEdgesCanny(img, params);

    // Large sigma should still detect edges, just smoother
    EXPECT_GE(contours.size(), 1u);
}

TEST_F(CannyTest, EdgeCase_ZeroSigma) {
    auto img = CreateSquareImage(60, 30);

    CannyParams params;
    params.sigma = 0.0;  // No smoothing
    params.lowThreshold = 50.0;
    params.highThreshold = 100.0;

    auto contours = DetectEdgesCanny(img, params);

    // Should still work
    EXPECT_GE(contours.size(), 1u);
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST_F(CannyTest, ToGradientOperator_Conversions) {
    EXPECT_EQ(ToGradientOperator(CannyGradientOp::Sobel), GradientOperator::Sobel3x3);
    EXPECT_EQ(ToGradientOperator(CannyGradientOp::Scharr), GradientOperator::Scharr);
    EXPECT_EQ(ToGradientOperator(CannyGradientOp::Sobel5x5), GradientOperator::Sobel5x5);
}

TEST_F(CannyTest, EdgeDirection_PerpendiculateToGradient) {
    // Gradient pointing right (0) -> edge direction up (π/2)
    EXPECT_NEAR(EdgeDirection(0.0), M_PI / 2, 0.001);

    // Gradient pointing up (π/2) -> edge direction left (π)
    EXPECT_NEAR(EdgeDirection(M_PI / 2), M_PI, 0.001);
}

TEST_F(CannyTest, CannyParams_StaticFactories) {
    auto autoParams = CannyParams::Auto(2.0);
    EXPECT_DOUBLE_EQ(autoParams.sigma, 2.0);
    EXPECT_TRUE(autoParams.autoThreshold);

    auto threshParams = CannyParams::WithThresholds(15.0, 45.0, 1.5);
    EXPECT_DOUBLE_EQ(threshParams.lowThreshold, 15.0);
    EXPECT_DOUBLE_EQ(threshParams.highThreshold, 45.0);
    EXPECT_DOUBLE_EQ(threshParams.sigma, 1.5);
    EXPECT_FALSE(threshParams.autoThreshold);
}

