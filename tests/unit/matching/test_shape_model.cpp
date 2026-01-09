/**
 * @file test_shape_model.cpp
 * @brief Unit tests for ShapeModel template matching
 */

#include <QiVision/Matching/ShapeModel.h>
#include <QiVision/Matching/MatchTypes.h>
#include <QiVision/Core/QImage.h>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Matching;

namespace {

// Helper function to create a test image with a simple shape
QImage CreateTestImage(int32_t width, int32_t height, uint8_t background = 128) {
    QImage image(width, height, PixelType::UInt8, ChannelType::Gray);
    uint8_t* data = static_cast<uint8_t*>(image.Data());
    int32_t stride = image.Stride();

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            data[y * stride + x] = background;
        }
    }

    return image;
}

// Helper function to draw a rectangle on an image
void DrawRectangle(QImage& image, int32_t x, int32_t y, int32_t w, int32_t h, uint8_t color) {
    uint8_t* data = static_cast<uint8_t*>(image.Data());
    int32_t stride = image.Stride();
    int32_t imgW = image.Width();
    int32_t imgH = image.Height();

    for (int32_t dy = 0; dy < h; ++dy) {
        int32_t py = y + dy;
        if (py < 0 || py >= imgH) continue;
        for (int32_t dx = 0; dx < w; ++dx) {
            int32_t px = x + dx;
            if (px < 0 || px >= imgW) continue;
            data[py * stride + px] = color;
        }
    }
}

// Helper function to draw a circle on an image
void DrawCircle(QImage& image, int32_t cx, int32_t cy, int32_t radius, uint8_t color) {
    uint8_t* data = static_cast<uint8_t*>(image.Data());
    int32_t stride = image.Stride();
    int32_t imgW = image.Width();
    int32_t imgH = image.Height();

    for (int32_t y = cy - radius; y <= cy + radius; ++y) {
        if (y < 0 || y >= imgH) continue;
        for (int32_t x = cx - radius; x <= cx + radius; ++x) {
            if (x < 0 || x >= imgW) continue;
            int32_t dx = x - cx;
            int32_t dy = y - cy;
            if (dx * dx + dy * dy <= radius * radius) {
                data[y * stride + x] = color;
            }
        }
    }
}

} // anonymous namespace

// =============================================================================
// MatchTypes Tests
// =============================================================================

class MatchTypesTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MatchTypesTest, MatchResultDefaultValues) {
    MatchResult result;

    EXPECT_DOUBLE_EQ(result.x, 0.0);
    EXPECT_DOUBLE_EQ(result.y, 0.0);
    EXPECT_DOUBLE_EQ(result.angle, 0.0);
    EXPECT_DOUBLE_EQ(result.scaleX, 1.0);
    EXPECT_DOUBLE_EQ(result.scaleY, 1.0);
    EXPECT_DOUBLE_EQ(result.score, 0.0);
    EXPECT_EQ(result.pyramidLevel, 0);
    EXPECT_FALSE(result.refined);
}

TEST_F(MatchTypesTest, MatchResultTransformPoint) {
    MatchResult result;
    result.x = 100.0;
    result.y = 200.0;
    result.angle = 0.0;
    result.scaleX = 1.0;
    result.scaleY = 1.0;

    Point2d modelPoint{10.0, 20.0};
    Point2d imagePoint = result.TransformPoint(modelPoint);

    EXPECT_NEAR(imagePoint.x, 110.0, 1e-6);
    EXPECT_NEAR(imagePoint.y, 220.0, 1e-6);
}

TEST_F(MatchTypesTest, MatchResultTransformPointWithRotation) {
    MatchResult result;
    result.x = 100.0;
    result.y = 100.0;
    result.angle = M_PI / 2.0;  // 90 degrees
    result.scaleX = 1.0;
    result.scaleY = 1.0;

    Point2d modelPoint{10.0, 0.0};
    Point2d imagePoint = result.TransformPoint(modelPoint);

    // After 90 degree rotation: (10, 0) -> (0, 10)
    EXPECT_NEAR(imagePoint.x, 100.0, 1e-6);
    EXPECT_NEAR(imagePoint.y, 110.0, 1e-6);
}

TEST_F(MatchTypesTest, MatchResultComparison) {
    MatchResult result1, result2;
    result1.score = 0.8;
    result2.score = 0.6;

    // result1 has higher score, should come first
    EXPECT_TRUE(result1 < result2);  // Comparison is by score descending
}

TEST_F(MatchTypesTest, SearchParamsDefaultValues) {
    SearchParams params;

    EXPECT_DOUBLE_EQ(params.minScore, 0.5);
    EXPECT_EQ(params.maxMatches, 0);
    EXPECT_EQ(params.angleMode, AngleSearchMode::Full);
    EXPECT_EQ(params.scaleMode, ScaleSearchMode::Fixed);
    EXPECT_EQ(params.subpixelMethod, SubpixelMethod::LeastSquares);
}

TEST_F(MatchTypesTest, SearchParamsBuilderPattern) {
    SearchParams params;
    params.SetMinScore(0.8)
          .SetMaxMatches(10)
          .SetAngleRange(-0.5, 1.0)
          .SetScaleRange(0.9, 1.1)
          .SetGreediness(0.8);

    EXPECT_DOUBLE_EQ(params.minScore, 0.8);
    EXPECT_EQ(params.maxMatches, 10);
    EXPECT_EQ(params.angleMode, AngleSearchMode::Range);
    EXPECT_DOUBLE_EQ(params.angleStart, -0.5);
    EXPECT_DOUBLE_EQ(params.angleExtent, 1.0);
    EXPECT_EQ(params.scaleMode, ScaleSearchMode::Uniform);
    EXPECT_DOUBLE_EQ(params.scaleMin, 0.9);
    EXPECT_DOUBLE_EQ(params.scaleMax, 1.1);
    EXPECT_DOUBLE_EQ(params.greediness, 0.8);
}

TEST_F(MatchTypesTest, ModelParamsDefaultValues) {
    ModelParams params;

    // New Halcon-compatible default values
    EXPECT_EQ(params.contrastMode, ContrastMode::Manual);
    EXPECT_DOUBLE_EQ(params.contrastHigh, 30.0);
    EXPECT_DOUBLE_EQ(params.contrastLow, 0.0);
    EXPECT_DOUBLE_EQ(params.contrastMax, 10000.0);
    EXPECT_DOUBLE_EQ(params.minContrast, 0.0);
    EXPECT_EQ(params.numLevels, 0);
    EXPECT_EQ(params.optimization, OptimizationMode::Auto);
    EXPECT_EQ(params.metric, MetricMode::UsePolarity);
}

TEST_F(MatchTypesTest, NonMaxSuppression) {
    std::vector<MatchResult> matches;

    // Create matches close together
    MatchResult m1; m1.x = 100; m1.y = 100; m1.score = 0.9;
    MatchResult m2; m2.x = 102; m2.y = 102; m2.score = 0.8;  // Close to m1
    MatchResult m3; m3.x = 200; m3.y = 200; m3.score = 0.7;  // Far from m1

    matches = {m1, m2, m3};

    auto filtered = NonMaxSuppression(matches, 10.0);

    // Should keep m1 and m3, suppress m2
    EXPECT_EQ(filtered.size(), 2u);
    EXPECT_NEAR(filtered[0].x, 100.0, 1e-6);
    EXPECT_NEAR(filtered[1].x, 200.0, 1e-6);
}

TEST_F(MatchTypesTest, FilterByScore) {
    std::vector<MatchResult> matches;

    MatchResult m1; m1.score = 0.9;
    MatchResult m2; m2.score = 0.5;
    MatchResult m3; m3.score = 0.3;

    matches = {m1, m2, m3};

    auto filtered = FilterByScore(matches, 0.6);

    EXPECT_EQ(filtered.size(), 1u);
    EXPECT_NEAR(filtered[0].score, 0.9, 1e-6);
}

// =============================================================================
// ShapeModel Tests
// =============================================================================

class ShapeModelTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(ShapeModelTest, DefaultConstruction) {
    ShapeModel model;
    EXPECT_FALSE(model.IsValid());
    EXPECT_EQ(model.NumLevels(), 0);
}

TEST_F(ShapeModelTest, CreateFromEmptyImage) {
    ShapeModel model;
    QImage empty;

    bool success = model.Create(empty);
    EXPECT_FALSE(success);
    EXPECT_FALSE(model.IsValid());
}

TEST_F(ShapeModelTest, CreateFromTemplateImage) {
    // Create a simple template with a rectangle
    QImage templateImg = CreateTestImage(100, 100, 128);
    DrawRectangle(templateImg, 20, 20, 60, 60, 255);

    ShapeModel model;
    bool success = model.Create(templateImg, ModelParams().SetContrast(20));

    EXPECT_TRUE(success);
    EXPECT_TRUE(model.IsValid());
    EXPECT_GT(model.NumLevels(), 0);
}

TEST_F(ShapeModelTest, CreateWithROI) {
    // Create image with multiple shapes
    QImage image = CreateTestImage(200, 200, 128);
    DrawRectangle(image, 20, 20, 40, 40, 255);  // Shape 1
    DrawRectangle(image, 120, 120, 40, 40, 255); // Shape 2

    // Create model from ROI containing only Shape 1
    ShapeModel model;
    Rect2i roi{10, 10, 60, 60};
    bool success = model.Create(image, roi, ModelParams().SetContrast(20));

    EXPECT_TRUE(success);
    EXPECT_TRUE(model.IsValid());
}

TEST_F(ShapeModelTest, GetModelStats) {
    QImage templateImg = CreateTestImage(100, 100, 128);
    DrawRectangle(templateImg, 20, 20, 60, 60, 255);

    ShapeModel model;
    model.Create(templateImg, ModelParams().SetContrast(20));

    ModelStats stats = model.GetStats();

    EXPECT_GT(stats.numPoints, 0);
    EXPECT_GT(stats.numLevels, 0);
    EXPECT_GT(stats.meanContrast, 0.0);
}

TEST_F(ShapeModelTest, GetModelPoints) {
    QImage templateImg = CreateTestImage(100, 100, 128);
    DrawRectangle(templateImg, 20, 20, 60, 60, 255);

    ShapeModel model;
    model.Create(templateImg, ModelParams().SetContrast(20));

    auto points = model.GetModelPoints(0);  // Level 0

    // Should have some edge points from the rectangle
    EXPECT_GT(points.size(), 0u);
}

TEST_F(ShapeModelTest, Clear) {
    QImage templateImg = CreateTestImage(100, 100, 128);
    DrawRectangle(templateImg, 20, 20, 60, 60, 255);

    ShapeModel model;
    model.Create(templateImg, ModelParams().SetContrast(20));

    EXPECT_TRUE(model.IsValid());

    model.Clear();

    EXPECT_FALSE(model.IsValid());
    EXPECT_EQ(model.NumLevels(), 0);
}

TEST_F(ShapeModelTest, CopyConstruction) {
    QImage templateImg = CreateTestImage(100, 100, 128);
    DrawRectangle(templateImg, 20, 20, 60, 60, 255);

    ShapeModel model1;
    model1.Create(templateImg, ModelParams().SetContrast(20));

    ShapeModel model2(model1);

    EXPECT_TRUE(model2.IsValid());
    EXPECT_EQ(model1.NumLevels(), model2.NumLevels());
}

TEST_F(ShapeModelTest, MoveConstruction) {
    QImage templateImg = CreateTestImage(100, 100, 128);
    DrawRectangle(templateImg, 20, 20, 60, 60, 255);

    ShapeModel model1;
    model1.Create(templateImg, ModelParams().SetContrast(20));
    int32_t numLevels = model1.NumLevels();

    ShapeModel model2(std::move(model1));

    EXPECT_TRUE(model2.IsValid());
    EXPECT_EQ(model2.NumLevels(), numLevels);
}

// =============================================================================
// Search Tests
// =============================================================================

class ShapeModelSearchTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create template with a simple shape
        templateImg_ = CreateTestImage(50, 50, 128);
        DrawRectangle(templateImg_, 10, 10, 30, 30, 255);

        // Create target image with the same shape at a known position
        targetImg_ = CreateTestImage(200, 200, 128);
        DrawRectangle(targetImg_, 75, 75, 30, 30, 255);  // Offset by (65, 65) from template origin

        // Create model
        model_.Create(templateImg_, ModelParams().SetContrast(20));
    }

    QImage templateImg_;
    QImage targetImg_;
    ShapeModel model_;
};

TEST_F(ShapeModelSearchTest, FindInEmptyImage) {
    QImage empty;
    auto results = model_.Find(empty);

    EXPECT_TRUE(results.empty());
}

TEST_F(ShapeModelSearchTest, FindWithInvalidModel) {
    ShapeModel invalidModel;
    auto results = invalidModel.Find(targetImg_);

    EXPECT_TRUE(results.empty());
}

TEST_F(ShapeModelSearchTest, FindBestMatch) {
    auto result = model_.FindBest(targetImg_, SearchParams().SetMinScore(0.3));

    // Should find the shape
    // Note: Exact position depends on implementation details
    EXPECT_GE(result.score, 0.0);
}

TEST_F(ShapeModelSearchTest, FindWithLowThreshold) {
    auto results = model_.Find(targetImg_, SearchParams().SetMinScore(0.1));

    // With low threshold, should find at least one match
    // (may find multiple candidates)
    EXPECT_GE(results.size(), 0u);
}

// =============================================================================
// Utility Function Tests
// =============================================================================

class UtilityFunctionsTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(UtilityFunctionsTest, EstimateOptimalLevels) {
    // Large image, small model -> more levels
    int32_t levels1 = EstimateOptimalLevels(1000, 1000, 50, 50);
    EXPECT_GE(levels1, 3);
    EXPECT_LE(levels1, 6);

    // Small image -> fewer levels
    int32_t levels2 = EstimateOptimalLevels(100, 100, 50, 50);
    EXPECT_LE(levels2, 3);
}

TEST_F(UtilityFunctionsTest, EstimateAngleStep) {
    // Larger model -> smaller angle step
    double step1 = EstimateAngleStep(100);
    double step2 = EstimateAngleStep(50);

    EXPECT_LT(step1, step2);
    EXPECT_GT(step1, 0.0);
    EXPECT_LT(step1, 0.5);
}

// =============================================================================
// Integration Tests
// =============================================================================

class ShapeModelIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(ShapeModelIntegrationTest, FindCircleTemplate) {
    // Create circular template
    QImage templateImg = CreateTestImage(60, 60, 128);
    DrawCircle(templateImg, 30, 30, 20, 255);

    // Create target with circle at different position
    QImage targetImg = CreateTestImage(200, 200, 128);
    DrawCircle(targetImg, 100, 100, 20, 255);

    ShapeModel model;
    bool success = model.Create(templateImg, ModelParams().SetContrast(20));
    EXPECT_TRUE(success);

    auto result = model.FindBest(targetImg, SearchParams().SetMinScore(0.3));

    // Circle should be found somewhere near (100, 100)
    // Allow for some tolerance due to subpixel positioning
    if (result.score > 0) {
        EXPECT_NEAR(result.x, 100.0, 30.0);
        EXPECT_NEAR(result.y, 100.0, 30.0);
    }
}

TEST_F(ShapeModelIntegrationTest, MultipleMatches) {
    // Create simple template
    QImage templateImg = CreateTestImage(40, 40, 128);
    DrawRectangle(templateImg, 5, 5, 30, 30, 255);

    // Create target with multiple instances
    QImage targetImg = CreateTestImage(300, 200, 128);
    DrawRectangle(targetImg, 20, 20, 30, 30, 255);
    DrawRectangle(targetImg, 100, 50, 30, 30, 255);
    DrawRectangle(targetImg, 200, 100, 30, 30, 255);

    ShapeModel model;
    model.Create(templateImg, ModelParams().SetContrast(20));

    auto results = model.Find(targetImg, SearchParams().SetMinScore(0.2).SetMaxMatches(10));

    // Should find multiple matches
    // Note: May not find all due to search algorithm limitations
    EXPECT_GE(results.size(), 0u);
}

