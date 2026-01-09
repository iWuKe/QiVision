/**
 * @file test_caliper_array.cpp
 * @brief Unit tests for CaliperArray module
 */

#include <gtest/gtest.h>
#include <QiVision/Measure/CaliperArray.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QContour.h>
#include <QiVision/Core/Constants.h>
#include <cmath>

using namespace Qi::Vision;
using namespace Qi::Vision::Measure;

namespace {

// Use Qi::Vision::PI from Constants.h
constexpr double TEST_EPSILON = 1e-6;

// Helper: Create a test image with a vertical line
QImage CreateVerticalLineImage(int width, int height, int lineX, int lineWidth = 1) {
    QImage image(width, height, PixelType::UInt8, ChannelType::Gray);
    auto* data = static_cast<uint8_t*>(image.Data());
    int stride = image.Stride();

    // Fill with background (50)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            data[y * stride + x] = 50;
        }
    }

    // Draw vertical line (200)
    int halfWidth = lineWidth / 2;
    for (int y = 0; y < height; ++y) {
        for (int dx = -halfWidth; dx <= halfWidth; ++dx) {
            int x = lineX + dx;
            if (x >= 0 && x < width) {
                data[y * stride + x] = 200;
            }
        }
    }

    return image;
}

// Helper: Create a test image with a horizontal line
QImage CreateHorizontalLineImage(int width, int height, int lineY, int lineWidth = 1) {
    QImage image(width, height, PixelType::UInt8, ChannelType::Gray);
    auto* data = static_cast<uint8_t*>(image.Data());
    int stride = image.Stride();

    // Fill with background (50)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            data[y * stride + x] = 50;
        }
    }

    // Draw horizontal line (200)
    int halfWidth = lineWidth / 2;
    for (int dy = -halfWidth; dy <= halfWidth; ++dy) {
        int y = lineY + dy;
        if (y >= 0 && y < height) {
            for (int x = 0; x < width; ++x) {
                data[y * stride + x] = 200;
            }
        }
    }

    return image;
}

// Helper: Create a test image with a circle
QImage CreateCircleImage(int width, int height, double cx, double cy, double radius, int ringWidth = 1) {
    QImage image(width, height, PixelType::UInt8, ChannelType::Gray);
    auto* data = static_cast<uint8_t*>(image.Data());
    int stride = image.Stride();

    // Fill with background (50)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            data[y * stride + x] = 50;
        }
    }

    // Draw circle ring (200)
    double halfWidth = ringWidth / 2.0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double dist = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
            if (std::abs(dist - radius) <= halfWidth) {
                data[y * stride + x] = 200;
            }
        }
    }

    return image;
}

} // namespace

// =============================================================================
// Array Creation Tests
// =============================================================================

class CaliperArrayCreationTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(CaliperArrayCreationTest, CreateAlongLine_Basic) {
    CaliperArray array;

    Point2d p1{50, 100};
    Point2d p2{450, 100};

    CaliperArrayParams params;
    params.caliperCount = 10;
    params.profileLength = 30.0;
    params.handleWidth = 10.0;

    EXPECT_TRUE(array.CreateAlongLine(p1, p2, params));
    EXPECT_TRUE(array.IsValid());
    EXPECT_EQ(array.Size(), 10);
    EXPECT_EQ(array.GetPathType(), PathType::Line);
    EXPECT_NEAR(array.GetPathLength(), 400.0, TEST_EPSILON);
}

TEST_F(CaliperArrayCreationTest, CreateAlongLine_Segment) {
    CaliperArray array;

    Segment2d segment{{100, 100}, {100, 500}};

    CaliperArrayParams params;
    params.caliperCount = 8;

    EXPECT_TRUE(array.CreateAlongLine(segment, params));
    EXPECT_EQ(array.Size(), 8);
}

TEST_F(CaliperArrayCreationTest, CreateAlongArc_Basic) {
    CaliperArray array;

    Point2d center{250, 250};
    double radius = 100.0;
    double startAngle = 0.0;
    double sweepAngle = PI / 2.0;  // 90 degrees

    CaliperArrayParams params;
    params.caliperCount = 10;
    params.profileLength = 30.0;

    EXPECT_TRUE(array.CreateAlongArc(center, radius, startAngle, sweepAngle, params));
    EXPECT_TRUE(array.IsValid());
    EXPECT_EQ(array.Size(), 10);
    EXPECT_EQ(array.GetPathType(), PathType::Arc);
}

TEST_F(CaliperArrayCreationTest, CreateAlongArc_Arc2d) {
    CaliperArray array;

    Arc2d arc(Point2d{250, 250}, 100.0, 0.0, PI);

    CaliperArrayParams params;
    params.caliperCount = 12;

    EXPECT_TRUE(array.CreateAlongArc(arc, params));
    EXPECT_EQ(array.Size(), 12);
}

TEST_F(CaliperArrayCreationTest, CreateAlongCircle_Basic) {
    CaliperArray array;

    Point2d center{250, 250};
    double radius = 100.0;

    CaliperArrayParams params;
    params.caliperCount = 24;
    params.profileLength = 30.0;

    EXPECT_TRUE(array.CreateAlongCircle(center, radius, params));
    EXPECT_TRUE(array.IsValid());
    EXPECT_EQ(array.Size(), 24);
    EXPECT_EQ(array.GetPathType(), PathType::Circle);
}

TEST_F(CaliperArrayCreationTest, CreateAlongCircle_Circle2d) {
    CaliperArray array;

    Circle2d circle{{200, 200}, 80.0};

    CaliperArrayParams params;
    params.caliperCount = 16;

    EXPECT_TRUE(array.CreateAlongCircle(circle, params));
    EXPECT_EQ(array.Size(), 16);
}

TEST_F(CaliperArrayCreationTest, CreateAlongContour_Basic) {
    // Create a simple rectangular contour
    QContour contour;
    contour.AddPoint(Point2d{100, 100});
    contour.AddPoint(Point2d{200, 100});
    contour.AddPoint(Point2d{200, 200});
    contour.AddPoint(Point2d{100, 200});
    contour.SetClosed(true);

    CaliperArray array;
    CaliperArrayParams params;
    params.caliperCount = 8;

    EXPECT_TRUE(array.CreateAlongContour(contour, params));
    EXPECT_TRUE(array.IsValid());
    EXPECT_EQ(array.Size(), 8);
    EXPECT_EQ(array.GetPathType(), PathType::Contour);
}

TEST_F(CaliperArrayCreationTest, Clear) {
    CaliperArray array;
    array.CreateAlongLine({0, 0}, {100, 0}, CaliperArrayParams().SetCaliperCount(5));
    EXPECT_TRUE(array.IsValid());
    EXPECT_EQ(array.Size(), 5);

    array.Clear();
    EXPECT_FALSE(array.IsValid());
    EXPECT_EQ(array.Size(), 0);
}

TEST_F(CaliperArrayCreationTest, CopyConstructor) {
    CaliperArray array1;
    array1.CreateAlongLine({0, 0}, {100, 0}, CaliperArrayParams().SetCaliperCount(5));

    CaliperArray array2(array1);
    EXPECT_TRUE(array2.IsValid());
    EXPECT_EQ(array2.Size(), 5);
}

TEST_F(CaliperArrayCreationTest, MoveConstructor) {
    CaliperArray array1;
    array1.CreateAlongLine({0, 0}, {100, 0}, CaliperArrayParams().SetCaliperCount(5));

    CaliperArray array2(std::move(array1));
    EXPECT_TRUE(array2.IsValid());
    EXPECT_EQ(array2.Size(), 5);
}

TEST_F(CaliperArrayCreationTest, CopyAssignment) {
    CaliperArray array1;
    array1.CreateAlongLine({0, 0}, {100, 0}, CaliperArrayParams().SetCaliperCount(5));

    CaliperArray array2;
    array2 = array1;
    EXPECT_TRUE(array2.IsValid());
    EXPECT_EQ(array2.Size(), 5);
}

TEST_F(CaliperArrayCreationTest, MoveAssignment) {
    CaliperArray array1;
    array1.CreateAlongLine({0, 0}, {100, 0}, CaliperArrayParams().SetCaliperCount(5));

    CaliperArray array2;
    array2 = std::move(array1);
    EXPECT_TRUE(array2.IsValid());
    EXPECT_EQ(array2.Size(), 5);
}

// =============================================================================
// Factory Functions Tests
// =============================================================================

class CaliperArrayFactoryTest : public ::testing::Test {};

TEST_F(CaliperArrayFactoryTest, CreateCaliperArrayLine_Points) {
    auto array = CreateCaliperArrayLine({50, 100}, {450, 100}, 10, 30.0, 10.0);
    EXPECT_TRUE(array.IsValid());
    EXPECT_EQ(array.Size(), 10);
    EXPECT_EQ(array.GetPathType(), PathType::Line);
}

TEST_F(CaliperArrayFactoryTest, CreateCaliperArrayLine_Segment) {
    Segment2d segment{{100, 50}, {100, 450}};
    auto array = CreateCaliperArrayLine(segment, 8);
    EXPECT_TRUE(array.IsValid());
    EXPECT_EQ(array.Size(), 8);
}

TEST_F(CaliperArrayFactoryTest, CreateCaliperArrayArc_Points) {
    auto array = CreateCaliperArrayArc({250, 250}, 100.0, 0.0, PI, 12, 30.0, 10.0);
    EXPECT_TRUE(array.IsValid());
    EXPECT_EQ(array.Size(), 12);
    EXPECT_EQ(array.GetPathType(), PathType::Arc);
}

TEST_F(CaliperArrayFactoryTest, CreateCaliperArrayArc_Arc2d) {
    Arc2d arc(Point2d{250, 250}, 100.0, 0.0, PI / 2.0);
    auto array = CreateCaliperArrayArc(arc, 8);
    EXPECT_TRUE(array.IsValid());
    EXPECT_EQ(array.Size(), 8);
}

TEST_F(CaliperArrayFactoryTest, CreateCaliperArrayCircle_Points) {
    auto array = CreateCaliperArrayCircle({250, 250}, 100.0, 24, 30.0, 10.0);
    EXPECT_TRUE(array.IsValid());
    EXPECT_EQ(array.Size(), 24);
    EXPECT_EQ(array.GetPathType(), PathType::Circle);
}

TEST_F(CaliperArrayFactoryTest, CreateCaliperArrayCircle_Circle2d) {
    Circle2d circle{{200, 200}, 80.0};
    auto array = CreateCaliperArrayCircle(circle, 16);
    EXPECT_TRUE(array.IsValid());
    EXPECT_EQ(array.Size(), 16);
}

TEST_F(CaliperArrayFactoryTest, CreateCaliperArrayContour) {
    QContour contour;
    contour.AddPoint(Point2d{100, 100});
    contour.AddPoint(Point2d{200, 100});
    contour.AddPoint(Point2d{200, 200});

    auto array = CreateCaliperArrayContour(contour, 6);
    EXPECT_TRUE(array.IsValid());
    EXPECT_EQ(array.Size(), 6);
    EXPECT_EQ(array.GetPathType(), PathType::Contour);
}

// =============================================================================
// Property Access Tests
// =============================================================================

class CaliperArrayPropertyTest : public ::testing::Test {
protected:
    CaliperArray array_;

    void SetUp() override {
        CaliperArrayParams params;
        params.caliperCount = 10;
        params.profileLength = 30.0;
        params.handleWidth = 10.0;
        array_.CreateAlongLine({50, 100}, {450, 100}, params);
    }
};

TEST_F(CaliperArrayPropertyTest, GetHandle) {
    // First caliper should be near start point
    const auto& handle0 = array_.GetHandle(0);
    EXPECT_NEAR(handle0.CenterRow(), 100.0, 1.0);

    // Last caliper should be near end point
    const auto& handle9 = array_.GetHandle(9);
    EXPECT_NEAR(handle9.CenterRow(), 100.0, 1.0);
}

TEST_F(CaliperArrayPropertyTest, GetHandle_OutOfRange) {
    EXPECT_THROW(array_.GetHandle(-1), std::out_of_range);
    EXPECT_THROW(array_.GetHandle(10), std::out_of_range);
}

TEST_F(CaliperArrayPropertyTest, GetHandles) {
    const auto& handles = array_.GetHandles();
    EXPECT_EQ(handles.size(), 10);
}

TEST_F(CaliperArrayPropertyTest, GetPathPosition) {
    // First caliper at start
    double pos0 = array_.GetPathPosition(0);
    // Positions should increase along path
    double pos5 = array_.GetPathPosition(5);
    EXPECT_GT(pos5, pos0);
}

TEST_F(CaliperArrayPropertyTest, GetPathRatio) {
    // First caliper near 0
    double ratio0 = array_.GetPathRatio(0);
    EXPECT_GE(ratio0, 0.0);
    EXPECT_LT(ratio0, 0.5);

    // Last caliper near 1
    double ratio9 = array_.GetPathRatio(9);
    EXPECT_GT(ratio9, 0.5);
    EXPECT_LE(ratio9, 1.0);
}

TEST_F(CaliperArrayPropertyTest, GetPathPoint) {
    Point2d p0 = array_.GetPathPoint(0);
    // Should be near start of line
    EXPECT_NEAR(p0.y, 100.0, 1.0);
    EXPECT_GT(p0.x, 40.0);
    EXPECT_LT(p0.x, 100.0);
}

// =============================================================================
// Measurement Tests
// =============================================================================

class CaliperArrayMeasureTest : public ::testing::Test {
protected:
    QImage lineImage_;
    QImage circleImage_;

    void SetUp() override {
        // Create image with vertical line at x=200
        lineImage_ = CreateVerticalLineImage(500, 200, 200, 1);
        // Create image with circle
        circleImage_ = CreateCircleImage(500, 500, 250.0, 250.0, 100.0, 3);
    }
};

TEST_F(CaliperArrayMeasureTest, MeasurePos_Line) {
    // Create array perpendicular to vertical line
    CaliperArray array;
    CaliperArrayParams params;
    params.caliperCount = 10;
    params.profileLength = 50.0;
    params.handleWidth = 20.0;

    // Create array along horizontal line crossing vertical edge at x=200
    array.CreateAlongLine({200, 20}, {200, 180}, params);

    MeasureParams measureParams;
    measureParams.sigma = 1.0;
    measureParams.minAmplitude = 20.0;

    auto result = array.MeasurePos(lineImage_, measureParams);

    // Should find edges in most calipers
    EXPECT_GE(result.numValid, 5);
    EXPECT_LE(result.numInvalid, 5);
    EXPECT_GT(result.validRatio, 0.5);
}

TEST_F(CaliperArrayMeasureTest, MeasurePos_Circle) {
    // Create array along circle
    CaliperArray array;
    CaliperArrayParams params;
    params.caliperCount = 24;
    params.profileLength = 40.0;
    params.handleWidth = 10.0;

    array.CreateAlongCircle({250, 250}, 100.0, params);

    MeasureParams measureParams;
    measureParams.sigma = 1.0;
    measureParams.minAmplitude = 20.0;

    auto result = array.MeasurePos(circleImage_, measureParams);

    // Should find edges in most calipers
    EXPECT_GE(result.numValid, 12);  // At least half
}

TEST_F(CaliperArrayMeasureTest, FuzzyMeasurePos_Line) {
    CaliperArray array;
    CaliperArrayParams params;
    params.caliperCount = 8;
    params.profileLength = 50.0;
    params.handleWidth = 20.0;

    array.CreateAlongLine({200, 20}, {200, 180}, params);

    FuzzyParams fuzzyParams;
    fuzzyParams.sigma = 1.0;
    fuzzyParams.minAmplitude = 20.0;

    auto result = array.FuzzyMeasurePos(lineImage_, fuzzyParams);

    EXPECT_GE(result.numValid, 4);
}

TEST_F(CaliperArrayMeasureTest, MeasureFirstEdges) {
    CaliperArray array;
    array.CreateAlongLine({200, 20}, {200, 180},
                          CaliperArrayParams().SetCaliperCount(8).SetProfileLength(50.0));

    auto points = array.MeasureFirstEdges(lineImage_, MeasureParams().SetSigma(1.0).SetMinAmplitude(20.0));

    // Should return some edge points
    EXPECT_FALSE(points.empty());
}

TEST_F(CaliperArrayMeasureTest, MeasurePairs_Stripe) {
    // Create image with stripe (two edges)
    QImage stripeImage(500, 200, PixelType::UInt8, ChannelType::Gray);
    auto* data = static_cast<uint8_t*>(stripeImage.Data());
    int stride = stripeImage.Stride();

    for (int y = 0; y < 200; ++y) {
        for (int x = 0; x < 500; ++x) {
            // Background = 50, Stripe from x=180 to x=220 = 200
            if (x >= 180 && x <= 220) {
                data[y * stride + x] = 200;
            } else {
                data[y * stride + x] = 50;
            }
        }
    }

    CaliperArray array;
    CaliperArrayParams params;
    params.caliperCount = 8;
    params.profileLength = 80.0;  // Wide enough to capture both edges
    params.handleWidth = 20.0;

    // Create array along horizontal line crossing stripe
    array.CreateAlongLine({200, 20}, {200, 180}, params);

    PairParams pairParams;
    pairParams.sigma = 1.0;
    pairParams.minAmplitude = 20.0;
    pairParams.SetWidthRange(20.0, 60.0);  // Expected width ~40px

    auto result = array.MeasurePairs(stripeImage, pairParams);

    // Should find pairs in most calipers
    EXPECT_GE(result.numValid, 4);

    // Width should be around 40 pixels
    if (result.numValid > 0) {
        EXPECT_NEAR(result.meanWidth, 40.0, 10.0);
    }
}

TEST_F(CaliperArrayMeasureTest, FuzzyMeasurePairs) {
    // Create stripe image
    QImage stripeImage(500, 200, PixelType::UInt8, ChannelType::Gray);
    auto* data = static_cast<uint8_t*>(stripeImage.Data());
    int stride = stripeImage.Stride();

    for (int y = 0; y < 200; ++y) {
        for (int x = 0; x < 500; ++x) {
            data[y * stride + x] = (x >= 180 && x <= 220) ? 200 : 50;
        }
    }

    CaliperArray array;
    array.CreateAlongLine({200, 20}, {200, 180},
                          CaliperArrayParams().SetCaliperCount(6).SetProfileLength(80.0));

    FuzzyParams fuzzyParams;
    fuzzyParams.sigma = 1.0;
    fuzzyParams.minAmplitude = 20.0;
    // Note: FuzzyParams doesn't have width filtering

    auto result = array.FuzzyMeasurePairs(stripeImage, fuzzyParams);

    EXPECT_GE(result.numValid, 3);
}

TEST_F(CaliperArrayMeasureTest, MeasurePairCenters) {
    QImage stripeImage(500, 200, PixelType::UInt8, ChannelType::Gray);
    auto* data = static_cast<uint8_t*>(stripeImage.Data());
    int stride = stripeImage.Stride();

    for (int y = 0; y < 200; ++y) {
        for (int x = 0; x < 500; ++x) {
            data[y * stride + x] = (x >= 180 && x <= 220) ? 200 : 50;
        }
    }

    CaliperArray array;
    array.CreateAlongLine({200, 20}, {200, 180},
                          CaliperArrayParams().SetCaliperCount(6).SetProfileLength(80.0));

    PairParams pairParams;
    pairParams.sigma = 1.0;
    pairParams.minAmplitude = 20.0;
    pairParams.SetWidthRange(20.0, 60.0);

    auto centers = array.MeasurePairCenters(stripeImage, pairParams);

    // Should return some center points
    EXPECT_FALSE(centers.empty());

    // Centers should be around x=200
    for (const auto& pt : centers) {
        EXPECT_NEAR(pt.x, 200.0, 5.0);
    }
}

TEST_F(CaliperArrayMeasureTest, MeasureWidths) {
    QImage stripeImage(500, 200, PixelType::UInt8, ChannelType::Gray);
    auto* data = static_cast<uint8_t*>(stripeImage.Data());
    int stride = stripeImage.Stride();

    for (int y = 0; y < 200; ++y) {
        for (int x = 0; x < 500; ++x) {
            data[y * stride + x] = (x >= 180 && x <= 220) ? 200 : 50;
        }
    }

    CaliperArray array;
    array.CreateAlongLine({200, 20}, {200, 180},
                          CaliperArrayParams().SetCaliperCount(6).SetProfileLength(80.0));

    PairParams pairParams;
    pairParams.sigma = 1.0;
    pairParams.minAmplitude = 20.0;
    pairParams.SetWidthRange(20.0, 60.0);

    double meanWidth, stdWidth;
    auto widths = array.MeasureWidths(stripeImage, pairParams, meanWidth, stdWidth);

    if (!widths.empty()) {
        // Width should be around 40 pixels
        EXPECT_NEAR(meanWidth, 40.0, 10.0);
        // Width should be consistent
        EXPECT_LT(stdWidth, 5.0);
    }
}

// =============================================================================
// Visualization Tests
// =============================================================================

class CaliperArrayVisualizationTest : public ::testing::Test {
protected:
    CaliperArray array_;

    void SetUp() override {
        array_.CreateAlongLine({50, 100}, {450, 100},
                               CaliperArrayParams().SetCaliperCount(10).SetProfileLength(30.0));
    }
};

TEST_F(CaliperArrayVisualizationTest, GetHandleRects) {
    auto rects = array_.GetHandleRects();
    EXPECT_EQ(rects.size(), 10);

    for (const auto& rect : rects) {
        EXPECT_GT(rect.width, 0);
        EXPECT_GT(rect.height, 0);
    }
}

TEST_F(CaliperArrayVisualizationTest, GetPathPoints_Default) {
    auto points = array_.GetPathPoints();
    EXPECT_GE(points.size(), 10);
}

TEST_F(CaliperArrayVisualizationTest, GetPathPoints_Specific) {
    auto points = array_.GetPathPoints(50);
    EXPECT_EQ(points.size(), 50);
}

TEST_F(CaliperArrayVisualizationTest, GetCaliperCenters) {
    auto centers = array_.GetCaliperCenters();
    EXPECT_EQ(centers.size(), 10);

    // Centers should be along the line (y=100)
    for (const auto& pt : centers) {
        EXPECT_NEAR(pt.y, 100.0, 1.0);
    }
}

// =============================================================================
// Result Structure Tests
// =============================================================================

class CaliperArrayResultTest : public ::testing::Test {};

TEST_F(CaliperArrayResultTest, CanFitLine) {
    CaliperArrayResult result;
    result.numValid = 1;
    EXPECT_FALSE(result.CanFitLine());

    result.numValid = 2;
    EXPECT_TRUE(result.CanFitLine());
}

TEST_F(CaliperArrayResultTest, CanFitCircle) {
    CaliperArrayResult result;
    result.numValid = 2;
    EXPECT_FALSE(result.CanFitCircle());

    result.numValid = 3;
    EXPECT_TRUE(result.CanFitCircle());
}

TEST_F(CaliperArrayResultTest, CanFitEllipse) {
    CaliperArrayResult result;
    result.numValid = 4;
    EXPECT_FALSE(result.CanFitEllipse());

    result.numValid = 5;
    EXPECT_TRUE(result.CanFitEllipse());
}

// =============================================================================
// Edge Cases Tests
// =============================================================================

class CaliperArrayEdgeCaseTest : public ::testing::Test {};

TEST_F(CaliperArrayEdgeCaseTest, ZeroLengthLine) {
    CaliperArray array;
    // Degenerate case: same start and end point
    bool result = array.CreateAlongLine({100, 100}, {100, 100}, CaliperArrayParams().SetCaliperCount(5));
    // Should handle gracefully (either return false or create at single point)
}

TEST_F(CaliperArrayEdgeCaseTest, MinCaliperCount) {
    CaliperArray array;
    CaliperArrayParams params;
    params.caliperCount = 2;  // Minimum

    EXPECT_TRUE(array.CreateAlongLine({0, 0}, {100, 0}, params));
    EXPECT_EQ(array.Size(), 2);
}

TEST_F(CaliperArrayEdgeCaseTest, SingleCaliper) {
    CaliperArray array;
    CaliperArrayParams params;
    params.caliperCount = 1;  // Below minimum

    // Should clamp to minimum or handle gracefully
    array.CreateAlongLine({0, 0}, {100, 0}, params);
}

TEST_F(CaliperArrayEdgeCaseTest, EmptyImage) {
    QImage emptyImage(100, 100, PixelType::UInt8, ChannelType::Gray);
    auto* data = static_cast<uint8_t*>(emptyImage.Data());
    int stride = emptyImage.Stride();

    // Fill with constant value (no edges)
    for (int y = 0; y < 100; ++y) {
        for (int x = 0; x < 100; ++x) {
            data[y * stride + x] = 128;
        }
    }

    CaliperArray array;
    array.CreateAlongLine({50, 10}, {50, 90}, CaliperArrayParams().SetCaliperCount(5));

    auto result = array.MeasurePos(emptyImage, MeasureParams());

    // Should find no edges
    EXPECT_EQ(result.numValid, 0);
}

TEST_F(CaliperArrayEdgeCaseTest, ArrayNotInitialized) {
    CaliperArray array;

    EXPECT_FALSE(array.IsValid());
    EXPECT_EQ(array.Size(), 0);

    QImage image(100, 100, PixelType::UInt8, ChannelType::Gray);
    auto result = array.MeasurePos(image, MeasureParams());

    // Should return empty result
    EXPECT_EQ(result.numCalipers, 0);
    EXPECT_EQ(result.numValid, 0);
}

// =============================================================================
// Params Builder Tests
// =============================================================================

class CaliperArrayParamsTest : public ::testing::Test {};

TEST_F(CaliperArrayParamsTest, DefaultValues) {
    CaliperArrayParams params;

    EXPECT_EQ(params.caliperCount, DEFAULT_CALIPER_COUNT);
    EXPECT_DOUBLE_EQ(params.profileLength, 50.0);
    EXPECT_DOUBLE_EQ(params.handleWidth, 10.0);
    EXPECT_TRUE(params.profilePerpendicular);
}

TEST_F(CaliperArrayParamsTest, BuilderPattern) {
    CaliperArrayParams params;
    params.SetCaliperCount(20)
          .SetProfileLength(40.0)
          .SetHandleWidth(15.0)
          .SetProfilePerpendicular(false)
          .SetPathOffset(5.0)
          .SetCoverage(0.1, 0.9);

    EXPECT_EQ(params.caliperCount, 20);
    EXPECT_DOUBLE_EQ(params.profileLength, 40.0);
    EXPECT_DOUBLE_EQ(params.handleWidth, 15.0);
    EXPECT_FALSE(params.profilePerpendicular);
    EXPECT_DOUBLE_EQ(params.pathOffset, 5.0);
    EXPECT_DOUBLE_EQ(params.startRatio, 0.1);
    EXPECT_DOUBLE_EQ(params.endRatio, 0.9);
}

TEST_F(CaliperArrayParamsTest, SetCaliperSpacing) {
    CaliperArrayParams params;
    params.SetCaliperSpacing(20.0);

    EXPECT_DOUBLE_EQ(params.caliperSpacing, 20.0);
    EXPECT_EQ(params.caliperCount, 0);  // Should be reset when using spacing
}
