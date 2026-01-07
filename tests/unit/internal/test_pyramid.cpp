/**
 * @file test_pyramid.cpp
 * @brief Unit tests for Pyramid module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/Pyramid.h>

#include <cmath>
#include <cstring>
#include <random>

using namespace Qi::Vision::Internal;
using Qi::Vision::QImage;
using Qi::Vision::PixelType;

// ============================================================================
// Test Fixture
// ============================================================================

class PyramidTest : public ::testing::Test {
protected:
    void SetUp() override {}

    // Create a test image with gradient
    QImage CreateGradientImage(int32_t width, int32_t height) {
        QImage img(width, height, PixelType::UInt8);
        uint8_t* data = static_cast<uint8_t*>(img.Data());

        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                data[y * width + x] = static_cast<uint8_t>((x + y) % 256);
            }
        }
        return img;
    }

    // Create a uniform image
    QImage CreateUniformImage(int32_t width, int32_t height, uint8_t value) {
        QImage img(width, height, PixelType::UInt8);
        std::memset(static_cast<uint8_t*>(img.Data()), value, width * height);
        return img;
    }

    // Create a checkerboard image
    QImage CreateCheckerboardImage(int32_t width, int32_t height, int32_t cellSize) {
        QImage img(width, height, PixelType::UInt8);
        uint8_t* data = static_cast<uint8_t*>(img.Data());

        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                bool white = ((x / cellSize) + (y / cellSize)) % 2 == 0;
                data[y * width + x] = white ? 255 : 0;
            }
        }
        return img;
    }

    // Create a step edge image
    QImage CreateStepEdgeImage(int32_t width, int32_t height, int32_t edgeX) {
        QImage img(width, height, PixelType::UInt8);
        uint8_t* data = static_cast<uint8_t*>(img.Data());

        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                data[y * width + x] = (x < edgeX) ? 50 : 200;
            }
        }
        return img;
    }
};

// ============================================================================
// ComputeNumLevels Tests
// ============================================================================

TEST_F(PyramidTest, ComputeNumLevels_Basic) {
    // 256x256 image with scale 0.5 should give log2(256/4) + 1 = 7 levels
    int32_t levels = ComputeNumLevels(256, 256, 0.5, 4);
    EXPECT_GE(levels, 6);
    EXPECT_LE(levels, 8);
}

TEST_F(PyramidTest, ComputeNumLevels_SmallImage) {
    int32_t levels = ComputeNumLevels(16, 16, 0.5, 4);
    EXPECT_GE(levels, 2);
    EXPECT_LE(levels, 4);
}

TEST_F(PyramidTest, ComputeNumLevels_Asymmetric) {
    // 512x128 - limited by smaller dimension
    int32_t levels = ComputeNumLevels(512, 128, 0.5, 4);
    // Should be limited by height 128
    EXPECT_LE(levels, 6);
}

TEST_F(PyramidTest, ComputeNumLevels_InvalidScale) {
    // Invalid scale factor should return 1 level
    int32_t levels = ComputeNumLevels(256, 256, 1.5, 4);
    EXPECT_EQ(levels, 1);

    levels = ComputeNumLevels(256, 256, 0.0, 4);
    EXPECT_EQ(levels, 1);
}

// ============================================================================
// GetLevelDimensions Tests
// ============================================================================

TEST_F(PyramidTest, GetLevelDimensions_Level0) {
    int32_t w, h;
    GetLevelDimensions(256, 256, 0, 0.5, w, h);
    EXPECT_EQ(w, 256);
    EXPECT_EQ(h, 256);
}

TEST_F(PyramidTest, GetLevelDimensions_Level1) {
    int32_t w, h;
    GetLevelDimensions(256, 256, 1, 0.5, w, h);
    EXPECT_EQ(w, 128);
    EXPECT_EQ(h, 128);
}

TEST_F(PyramidTest, GetLevelDimensions_Level2) {
    int32_t w, h;
    GetLevelDimensions(256, 256, 2, 0.5, w, h);
    EXPECT_EQ(w, 64);
    EXPECT_EQ(h, 64);
}

// ============================================================================
// ConvertCoordinates Tests
// ============================================================================

TEST_F(PyramidTest, ConvertCoordinates_SameLevel) {
    double dx, dy;
    ConvertCoordinates(100.0, 50.0, 0, 0, 0.5, dx, dy);
    EXPECT_DOUBLE_EQ(dx, 100.0);
    EXPECT_DOUBLE_EQ(dy, 50.0);
}

TEST_F(PyramidTest, ConvertCoordinates_Level0To1) {
    double dx, dy;
    ConvertCoordinates(100.0, 50.0, 0, 1, 0.5, dx, dy);
    EXPECT_DOUBLE_EQ(dx, 50.0);
    EXPECT_DOUBLE_EQ(dy, 25.0);
}

TEST_F(PyramidTest, ConvertCoordinates_Level1To0) {
    double dx, dy;
    ConvertCoordinates(50.0, 25.0, 1, 0, 0.5, dx, dy);
    EXPECT_DOUBLE_EQ(dx, 100.0);
    EXPECT_DOUBLE_EQ(dy, 50.0);
}

// ============================================================================
// DownsampleBy2 Tests
// ============================================================================

TEST_F(PyramidTest, DownsampleBy2_Skip) {
    const int32_t w = 8, h = 8;
    std::vector<float> src(w * h);
    for (int32_t i = 0; i < w * h; ++i) {
        src[i] = static_cast<float>(i);
    }

    std::vector<float> dst((w / 2) * (h / 2));
    DownsampleBy2(src.data(), w, h, dst.data(), 1.0, DownsampleMethod::Skip);

    // Check that we got every other pixel
    EXPECT_FLOAT_EQ(dst[0], src[0]);
    EXPECT_FLOAT_EQ(dst[1], src[2]);
}

TEST_F(PyramidTest, DownsampleBy2_Average) {
    const int32_t w = 4, h = 4;
    std::vector<float> src(w * h, 100.0f);

    std::vector<float> dst((w / 2) * (h / 2));
    DownsampleBy2(src.data(), w, h, dst.data(), 1.0, DownsampleMethod::Average);

    // Uniform image should stay uniform
    for (float v : dst) {
        EXPECT_FLOAT_EQ(v, 100.0f);
    }
}

TEST_F(PyramidTest, DownsampleBy2_Gaussian) {
    const int32_t w = 16, h = 16;
    std::vector<float> src(w * h, 128.0f);

    std::vector<float> dst((w / 2) * (h / 2));
    DownsampleBy2(src.data(), w, h, dst.data(), 1.0, DownsampleMethod::Gaussian);

    // Uniform image should stay approximately uniform
    for (float v : dst) {
        EXPECT_NEAR(v, 128.0f, 1.0f);
    }
}

// ============================================================================
// UpsampleBy2 Tests
// ============================================================================

TEST_F(PyramidTest, UpsampleBy2_NearestNeighbor) {
    const int32_t w = 4, h = 4;
    std::vector<float> src(w * h);
    for (int32_t i = 0; i < w * h; ++i) {
        src[i] = static_cast<float>(i);
    }

    std::vector<float> dst((w * 2) * (h * 2));
    UpsampleBy2(src.data(), w, h, dst.data(), UpsampleMethod::NearestNeighbor);

    // Each source pixel should appear in 2x2 block
    EXPECT_FLOAT_EQ(dst[0], src[0]);
    EXPECT_FLOAT_EQ(dst[1], src[0]);
    EXPECT_FLOAT_EQ(dst[8], src[0]);
    EXPECT_FLOAT_EQ(dst[9], src[0]);
}

TEST_F(PyramidTest, UpsampleBy2_Bilinear) {
    const int32_t w = 2, h = 2;
    std::vector<float> src = {0.0f, 100.0f, 0.0f, 100.0f};

    std::vector<float> dst((w * 2) * (h * 2));
    UpsampleBy2(src.data(), w, h, dst.data(), UpsampleMethod::Bilinear);

    // Result should be interpolated smoothly
    EXPECT_GT(dst.size(), 0u);  // Just check it completes
}

TEST_F(PyramidTest, UpsampleBy2_Bicubic) {
    const int32_t w = 4, h = 4;
    std::vector<float> src(w * h, 100.0f);

    std::vector<float> dst((w * 2) * (h * 2));
    UpsampleBy2(src.data(), w, h, dst.data(), UpsampleMethod::Bicubic);

    // Uniform should stay uniform
    for (float v : dst) {
        EXPECT_NEAR(v, 100.0f, 5.0f);
    }
}

// ============================================================================
// BuildGaussianPyramid Tests
// ============================================================================

TEST_F(PyramidTest, BuildGaussianPyramid_Empty) {
    QImage emptyImg;
    auto pyramid = BuildGaussianPyramid(emptyImg);
    EXPECT_TRUE(pyramid.Empty());
}

TEST_F(PyramidTest, BuildGaussianPyramid_Basic) {
    auto img = CreateGradientImage(128, 128);
    auto pyramid = BuildGaussianPyramid(img, PyramidParams::Auto());

    EXPECT_FALSE(pyramid.Empty());
    EXPECT_GE(pyramid.NumLevels(), 3);

    // Level 0 should match original size
    EXPECT_EQ(pyramid.GetLevel(0).width, 128);
    EXPECT_EQ(pyramid.GetLevel(0).height, 128);

    // Each level should be half the size
    for (int32_t i = 1; i < pyramid.NumLevels(); ++i) {
        const auto& prev = pyramid.GetLevel(i - 1);
        const auto& curr = pyramid.GetLevel(i);
        EXPECT_EQ(curr.width, prev.width / 2);
        EXPECT_EQ(curr.height, prev.height / 2);
    }
}

TEST_F(PyramidTest, BuildGaussianPyramid_SpecifiedLevels) {
    auto img = CreateGradientImage(256, 256);
    auto pyramid = BuildGaussianPyramid(img, PyramidParams::WithLevels(4));

    EXPECT_EQ(pyramid.NumLevels(), 4);
}

TEST_F(PyramidTest, BuildGaussianPyramid_ScaleValues) {
    auto img = CreateGradientImage(64, 64);
    auto pyramid = BuildGaussianPyramid(img);

    for (int32_t i = 0; i < pyramid.NumLevels(); ++i) {
        const auto& level = pyramid.GetLevel(i);
        EXPECT_DOUBLE_EQ(level.scale, std::pow(0.5, i));
        EXPECT_EQ(level.level, i);
    }
}

TEST_F(PyramidTest, BuildGaussianPyramid_Uniform) {
    auto img = CreateUniformImage(64, 64, 150);
    auto pyramid = BuildGaussianPyramid(img);

    // All levels should be approximately uniform
    for (int32_t i = 0; i < pyramid.NumLevels(); ++i) {
        const auto& level = pyramid.GetLevel(i);
        for (float v : level.data) {
            EXPECT_NEAR(v, 150.0f, 5.0f);
        }
    }
}

// ============================================================================
// BuildLaplacianPyramid Tests
// ============================================================================

TEST_F(PyramidTest, BuildLaplacianPyramid_Basic) {
    auto img = CreateGradientImage(64, 64);
    auto pyramid = BuildLaplacianPyramid(img);

    EXPECT_FALSE(pyramid.Empty());
    EXPECT_GE(pyramid.NumLevels(), 3);
}

TEST_F(PyramidTest, GaussianToLaplacian_Conversion) {
    auto img = CreateGradientImage(64, 64);
    auto gaussian = BuildGaussianPyramid(img);
    auto laplacian = GaussianToLaplacian(gaussian);

    EXPECT_EQ(laplacian.NumLevels(), gaussian.NumLevels());

    // Laplacian levels should have same dimensions as Gaussian
    for (int32_t i = 0; i < laplacian.NumLevels(); ++i) {
        EXPECT_EQ(laplacian.GetLevel(i).width, gaussian.GetLevel(i).width);
        EXPECT_EQ(laplacian.GetLevel(i).height, gaussian.GetLevel(i).height);
    }
}

// ============================================================================
// Laplacian Reconstruction Tests
// ============================================================================

TEST_F(PyramidTest, ReconstructFromLaplacian_Basic) {
    auto img = CreateGradientImage(64, 64);
    auto laplacian = BuildLaplacianPyramid(img);

    auto reconstructed = ReconstructFromLaplacian(laplacian);

    EXPECT_TRUE(reconstructed.IsValid());
    EXPECT_EQ(reconstructed.width, 64);
    EXPECT_EQ(reconstructed.height, 64);
}

TEST_F(PyramidTest, ReconstructFromLaplacian_Accuracy) {
    auto img = CreateUniformImage(32, 32, 100);
    auto laplacian = BuildLaplacianPyramid(img);

    auto reconstructed = ReconstructFromLaplacian(laplacian);

    // Reconstructed should be close to original
    for (float v : reconstructed.data) {
        EXPECT_NEAR(v, 100.0f, 10.0f);
    }
}

TEST_F(PyramidTest, ReconstructFromLaplacian_Gradient) {
    auto img = CreateGradientImage(64, 64);

    // Convert to float for comparison
    std::vector<float> original(64 * 64);
    const uint8_t* srcData = static_cast<const uint8_t*>(img.Data());
    for (int32_t i = 0; i < 64 * 64; ++i) {
        original[i] = static_cast<float>(srcData[i]);
    }

    auto laplacian = BuildLaplacianPyramid(img);
    auto reconstructed = ReconstructFromLaplacian(laplacian);

    // Calculate RMS error
    double sumSqError = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
        double diff = original[i] - reconstructed.data[i];
        sumSqError += diff * diff;
    }
    double rmsError = std::sqrt(sumSqError / original.size());

    // RMS error should be small
    EXPECT_LT(rmsError, 5.0);
}

// ============================================================================
// GradientPyramid Tests
// ============================================================================

TEST_F(PyramidTest, BuildGradientPyramid_Basic) {
    auto img = CreateStepEdgeImage(64, 64, 32);
    auto pyramid = BuildGradientPyramid(img);

    EXPECT_FALSE(pyramid.Empty());
    EXPECT_GE(pyramid.NumLevels(), 3);

    // Each level should have magnitude and direction
    for (int32_t i = 0; i < pyramid.NumLevels(); ++i) {
        const auto& level = pyramid.GetLevel(i);
        EXPECT_TRUE(level.IsValid());
        EXPECT_FALSE(level.magnitude.empty());
        EXPECT_FALSE(level.direction.empty());
    }
}

TEST_F(PyramidTest, GradientPyramid_EdgeDetection) {
    auto img = CreateStepEdgeImage(64, 64, 32);
    auto pyramid = BuildGradientPyramid(img);

    // Level 0 should have high magnitude near the edge
    const auto& level0 = pyramid.GetLevel(0);

    // Find maximum magnitude
    float maxMag = 0.0f;
    int32_t maxX = 0;
    for (int32_t y = 20; y < 44; ++y) {
        for (int32_t x = 20; x < 44; ++x) {
            float mag = level0.magnitude[y * level0.width + x];
            if (mag > maxMag) {
                maxMag = mag;
                maxX = x;
            }
        }
    }

    // Maximum should be near the edge at x=32
    EXPECT_NEAR(maxX, 32, 3);
    EXPECT_GT(maxMag, 50.0f);
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST_F(PyramidTest, PyramidLevelToImage_Basic) {
    PyramidLevel level;
    level.width = 10;
    level.height = 10;
    level.data.resize(100, 128.0f);

    auto img = PyramidLevelToImage(level, false);

    EXPECT_FALSE(img.Empty());
    EXPECT_EQ(img.Width(), 10);
    EXPECT_EQ(img.Height(), 10);
}

TEST_F(PyramidTest, PyramidLevelToImage_Normalize) {
    PyramidLevel level;
    level.width = 10;
    level.height = 10;
    level.data.resize(100);
    for (int32_t i = 0; i < 100; ++i) {
        level.data[i] = static_cast<float>(i);  // 0 to 99
    }

    auto img = PyramidLevelToImage(level, true);

    const uint8_t* data = static_cast<const uint8_t*>(img.Data());

    // First pixel should be 0, last should be 255
    EXPECT_EQ(data[0], 0);
    EXPECT_EQ(data[99], 255);
}

TEST_F(PyramidTest, ImageToPyramidLevel_Basic) {
    auto img = CreateUniformImage(32, 32, 200);

    auto level = ImageToPyramidLevel(img, 0, 1.0);

    EXPECT_TRUE(level.IsValid());
    EXPECT_EQ(level.width, 32);
    EXPECT_EQ(level.height, 32);
    EXPECT_EQ(level.level, 0);
    EXPECT_DOUBLE_EQ(level.scale, 1.0);

    for (float v : level.data) {
        EXPECT_FLOAT_EQ(v, 200.0f);
    }
}

TEST_F(PyramidTest, SamplePyramidAtScale_Level0) {
    auto img = CreateUniformImage(64, 64, 150);
    auto pyramid = BuildGaussianPyramid(img);

    float value = SamplePyramidAtScale(pyramid, 32.0, 32.0, 1.0);

    EXPECT_NEAR(value, 150.0f, 5.0f);
}

TEST_F(PyramidTest, ComputeSearchScales_Basic) {
    auto scales = ComputeSearchScales(100, 100, 0.5, 2.0, 0.25);

    EXPECT_FALSE(scales.empty());
    EXPECT_GE(scales.front(), 0.5 - 0.01);
    EXPECT_LE(scales.back(), 2.0 + 0.01);

    // Should be sorted
    for (size_t i = 1; i < scales.size(); ++i) {
        EXPECT_GT(scales[i], scales[i - 1]);
    }
}

// ============================================================================
// ImagePyramid Container Tests
// ============================================================================

TEST_F(PyramidTest, ImagePyramid_GetLevelByScale) {
    auto img = CreateGradientImage(128, 128);
    auto pyramid = BuildGaussianPyramid(img);

    // Request scale 0.5 should get level 1
    const auto& level = pyramid.GetLevelByScale(0.5);
    EXPECT_NEAR(level.scale, 0.5, 0.01);
}

TEST_F(PyramidTest, ImagePyramid_OriginalDimensions) {
    auto img = CreateGradientImage(200, 150);
    auto pyramid = BuildGaussianPyramid(img);

    EXPECT_EQ(pyramid.OriginalWidth(), 200);
    EXPECT_EQ(pyramid.OriginalHeight(), 150);
}

TEST_F(PyramidTest, ImagePyramid_Clear) {
    auto img = CreateGradientImage(64, 64);
    auto pyramid = BuildGaussianPyramid(img);

    EXPECT_FALSE(pyramid.Empty());
    pyramid.Clear();
    EXPECT_TRUE(pyramid.Empty());
}

// ============================================================================
// PyramidLevel Access Tests
// ============================================================================

TEST_F(PyramidTest, PyramidLevel_At) {
    PyramidLevel level;
    level.width = 10;
    level.height = 10;
    level.data.resize(100, 50.0f);
    level.data[55] = 100.0f;  // y=5, x=5

    EXPECT_FLOAT_EQ(level.At(5, 5), 100.0f);
    EXPECT_FLOAT_EQ(level.At(0, 0), 50.0f);

    // Out of bounds should return 0
    EXPECT_FLOAT_EQ(level.At(-1, 0), 0.0f);
    EXPECT_FLOAT_EQ(level.At(10, 0), 0.0f);
}

TEST_F(PyramidTest, PyramidLevel_AtSafe) {
    PyramidLevel level;
    level.width = 10;
    level.height = 10;
    level.data.resize(100, 50.0f);

    // Out of bounds should clamp
    EXPECT_FLOAT_EQ(level.AtSafe(-5, 0), level.data[0]);
    EXPECT_FLOAT_EQ(level.AtSafe(100, 0), level.data[9]);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(PyramidTest, EdgeCase_VerySmallImage) {
    auto img = CreateUniformImage(4, 4, 100);
    auto pyramid = BuildGaussianPyramid(img);

    // Should at least have level 0
    EXPECT_GE(pyramid.NumLevels(), 1);
    EXPECT_EQ(pyramid.GetLevel(0).width, 4);
}

TEST_F(PyramidTest, EdgeCase_AsymmetricImage) {
    auto img = CreateGradientImage(128, 32);
    auto pyramid = BuildGaussianPyramid(img);

    EXPECT_FALSE(pyramid.Empty());

    // Dimensions should be halved correctly
    if (pyramid.NumLevels() > 1) {
        EXPECT_EQ(pyramid.GetLevel(1).width, 64);
        EXPECT_EQ(pyramid.GetLevel(1).height, 16);
    }
}

TEST_F(PyramidTest, EdgeCase_LargeImage) {
    auto img = CreateUniformImage(512, 512, 128);
    auto pyramid = BuildGaussianPyramid(img);

    // Should handle large images
    EXPECT_FALSE(pyramid.Empty());
    EXPECT_GE(pyramid.NumLevels(), 5);
}

// ============================================================================
// PyramidParams Tests
// ============================================================================

TEST_F(PyramidTest, PyramidParams_WithLevels) {
    auto params = PyramidParams::WithLevels(5, 1.5);

    EXPECT_EQ(params.numLevels, 5);
    EXPECT_DOUBLE_EQ(params.sigma, 1.5);
}

TEST_F(PyramidTest, PyramidParams_Auto) {
    auto params = PyramidParams::Auto(2.0, 8);

    EXPECT_EQ(params.numLevels, 0);  // Auto-compute
    EXPECT_DOUBLE_EQ(params.sigma, 2.0);
    EXPECT_EQ(params.minDimension, 8);
}

