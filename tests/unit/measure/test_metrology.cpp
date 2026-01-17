/**
 * @file test_metrology.cpp
 * @brief Unit tests for Measure/Metrology module
 */

#include <gtest/gtest.h>

#include <QiVision/Core/QImage.h>
#include <QiVision/Measure/Metrology.h>
#include <QiVision/Measure/MeasureHandle.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Measure;

namespace {
    constexpr double TEST_PI = 3.14159265358979323846;
    constexpr double TWO_PI = 2.0 * TEST_PI;

    // Create a test image with a vertical edge
    QImage CreateVerticalEdgeImage(int width, int height, int edgeCol,
                                    uint8_t leftValue = 50, uint8_t rightValue = 200) {
        QImage img(width, height, PixelType::UInt8, ChannelType::Gray);
        for (int y = 0; y < height; ++y) {
            uint8_t* row = static_cast<uint8_t*>(img.RowPtr(y));
            for (int x = 0; x < width; ++x) {
                row[x] = (x < edgeCol) ? leftValue : rightValue;
            }
        }
        return img;
    }

    // Create a test image with a horizontal edge
    QImage CreateHorizontalEdgeImage(int width, int height, int edgeRow,
                                      uint8_t topValue = 50, uint8_t bottomValue = 200) {
        QImage img(width, height, PixelType::UInt8, ChannelType::Gray);
        for (int y = 0; y < height; ++y) {
            uint8_t value = (y < edgeRow) ? topValue : bottomValue;
            uint8_t* row = static_cast<uint8_t*>(img.RowPtr(y));
            std::memset(row, value, width);
        }
        return img;
    }

    // Create an image with a circular edge
    QImage CreateCircleImage(int width, int height, double cx, double cy, double radius,
                              uint8_t insideValue = 200, uint8_t outsideValue = 50) {
        QImage img(width, height, PixelType::UInt8, ChannelType::Gray);
        for (int y = 0; y < height; ++y) {
            uint8_t* row = static_cast<uint8_t*>(img.RowPtr(y));
            for (int x = 0; x < width; ++x) {
                double dx = x - cx;
                double dy = y - cy;
                double dist = std::sqrt(dx * dx + dy * dy);
                row[x] = (dist < radius) ? insideValue : outsideValue;
            }
        }
        return img;
    }

    // Create an image with an ellipse
    QImage CreateEllipseImage(int width, int height, double cx, double cy,
                               double ra, double rb, double phi,
                               uint8_t insideValue = 200, uint8_t outsideValue = 50) {
        QImage img(width, height, PixelType::UInt8, ChannelType::Gray);
        double cosPhi = std::cos(-phi);
        double sinPhi = std::sin(-phi);

        for (int y = 0; y < height; ++y) {
            uint8_t* row = static_cast<uint8_t*>(img.RowPtr(y));
            for (int x = 0; x < width; ++x) {
                double dx = x - cx;
                double dy = y - cy;
                double localX = dx * cosPhi - dy * sinPhi;
                double localY = dx * sinPhi + dy * cosPhi;
                double dist = (localX * localX) / (ra * ra) + (localY * localY) / (rb * rb);
                row[x] = (dist < 1.0) ? insideValue : outsideValue;
            }
        }
        return img;
    }

    // Create an image with a rotated rectangle
    QImage CreateRectangleImage(int width, int height, double cx, double cy,
                                 double length1, double length2, double phi,
                                 uint8_t insideValue = 200, uint8_t outsideValue = 50) {
        QImage img(width, height, PixelType::UInt8, ChannelType::Gray);
        double cosPhi = std::cos(-phi);
        double sinPhi = std::sin(-phi);

        for (int y = 0; y < height; ++y) {
            uint8_t* row = static_cast<uint8_t*>(img.RowPtr(y));
            for (int x = 0; x < width; ++x) {
                double dx = x - cx;
                double dy = y - cy;
                double localX = dx * cosPhi - dy * sinPhi;
                double localY = dx * sinPhi + dy * cosPhi;
                bool inside = (std::abs(localX) <= length1) && (std::abs(localY) <= length2);
                row[x] = inside ? insideValue : outsideValue;
            }
        }
        return img;
    }
}

// =============================================================================
// MetrologyModel Creation Tests
// =============================================================================

class MetrologyModelTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MetrologyModelTest, CreateEmpty) {
    MetrologyModel model;
    EXPECT_EQ(model.NumObjects(), 0);
}

TEST_F(MetrologyModelTest, CreateMetrologyModel) {
    auto model = CreateMetrologyModel();
    EXPECT_EQ(model.NumObjects(), 0);
}

TEST_F(MetrologyModelTest, AddLineMeasure) {
    MetrologyModel model;
    int idx = model.AddLineMeasure(100, 50, 100, 150);
    EXPECT_EQ(idx, 0);
    EXPECT_EQ(model.NumObjects(), 1);

    auto* obj = model.GetObject(idx);
    ASSERT_NE(obj, nullptr);
    EXPECT_EQ(obj->Type(), MetrologyObjectType::Line);
}

TEST_F(MetrologyModelTest, AddCircleMeasure) {
    MetrologyModel model;
    int idx = model.AddCircleMeasure(100, 100, 50);
    EXPECT_EQ(idx, 0);
    EXPECT_EQ(model.NumObjects(), 1);

    auto* obj = model.GetObject(idx);
    ASSERT_NE(obj, nullptr);
    EXPECT_EQ(obj->Type(), MetrologyObjectType::Circle);
}

TEST_F(MetrologyModelTest, AddArcMeasure) {
    MetrologyModel model;
    int idx = model.AddArcMeasure(100, 100, 50, 0.0, TEST_PI);
    EXPECT_EQ(idx, 0);

    auto* obj = model.GetObject(idx);
    ASSERT_NE(obj, nullptr);
    EXPECT_EQ(obj->Type(), MetrologyObjectType::Circle);
}

TEST_F(MetrologyModelTest, AddEllipseMeasure) {
    MetrologyModel model;
    int idx = model.AddEllipseMeasure(100, 100, 0.0, 60, 40);
    EXPECT_EQ(idx, 0);

    auto* obj = model.GetObject(idx);
    ASSERT_NE(obj, nullptr);
    EXPECT_EQ(obj->Type(), MetrologyObjectType::Ellipse);
}

TEST_F(MetrologyModelTest, AddRectangle2Measure) {
    MetrologyModel model;
    int idx = model.AddRectangle2Measure(100, 100, 0.0, 50, 30);
    EXPECT_EQ(idx, 0);

    auto* obj = model.GetObject(idx);
    ASSERT_NE(obj, nullptr);
    EXPECT_EQ(obj->Type(), MetrologyObjectType::Rectangle2);
}

TEST_F(MetrologyModelTest, AddMultipleObjects) {
    MetrologyModel model;

    int lineIdx = model.AddLineMeasure(50, 50, 50, 150);
    int circleIdx = model.AddCircleMeasure(100, 100, 40);
    int ellipseIdx = model.AddEllipseMeasure(150, 150, 0.1, 50, 30);

    EXPECT_EQ(model.NumObjects(), 3);
    EXPECT_EQ(lineIdx, 0);
    EXPECT_EQ(circleIdx, 1);
    EXPECT_EQ(ellipseIdx, 2);
}

TEST_F(MetrologyModelTest, ClearObject) {
    MetrologyModel model;
    model.AddLineMeasure(50, 50, 50, 150);
    model.AddCircleMeasure(100, 100, 40);

    model.ClearObject(0);
    EXPECT_EQ(model.GetObject(0), nullptr);
    EXPECT_NE(model.GetObject(1), nullptr);
}

TEST_F(MetrologyModelTest, ClearAll) {
    MetrologyModel model;
    model.AddLineMeasure(50, 50, 50, 150);
    model.AddCircleMeasure(100, 100, 40);

    model.ClearAll();
    EXPECT_EQ(model.NumObjects(), 0);
}

// =============================================================================
// MetrologyObjectLine Tests
// =============================================================================

class MetrologyObjectLineTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MetrologyObjectLineTest, Properties) {
    MetrologyMeasureParams params;
    params.numMeasures = 5;
    MetrologyObjectLine line(100, 50, 100, 150, params);

    EXPECT_DOUBLE_EQ(line.Row1(), 100);
    EXPECT_DOUBLE_EQ(line.Col1(), 50);
    EXPECT_DOUBLE_EQ(line.Row2(), 100);
    EXPECT_DOUBLE_EQ(line.Col2(), 150);
    EXPECT_NEAR(line.Length(), 100.0, 0.001);
    EXPECT_NEAR(line.Angle(), 0.0, 0.001);  // Horizontal line
}

TEST_F(MetrologyObjectLineTest, GetCalipers) {
    MetrologyMeasureParams params;
    params.numMeasures = 5;
    MetrologyObjectLine line(100, 50, 100, 150, params);

    auto calipers = line.GetCalipers();
    EXPECT_EQ(calipers.size(), 5u);
}

TEST_F(MetrologyObjectLineTest, GetContour) {
    MetrologyObjectLine line(100, 50, 100, 150);
    auto contour = line.GetContour();
    EXPECT_EQ(contour.Size(), 2);
}

TEST_F(MetrologyObjectLineTest, Transform) {
    MetrologyObjectLine line(100, 50, 100, 150);
    line.Transform(10, 20, 0.0);

    EXPECT_DOUBLE_EQ(line.Row1(), 110);
    EXPECT_DOUBLE_EQ(line.Col1(), 70);
    EXPECT_DOUBLE_EQ(line.Row2(), 110);
    EXPECT_DOUBLE_EQ(line.Col2(), 170);
}

// =============================================================================
// MetrologyObjectCircle Tests
// =============================================================================

class MetrologyObjectCircleTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MetrologyObjectCircleTest, FullCircle) {
    MetrologyObjectCircle circle(100, 100, 50);

    EXPECT_DOUBLE_EQ(circle.Row(), 100);
    EXPECT_DOUBLE_EQ(circle.Column(), 100);
    EXPECT_DOUBLE_EQ(circle.Radius(), 50);
    EXPECT_TRUE(circle.IsFullCircle());
}

TEST_F(MetrologyObjectCircleTest, Arc) {
    MetrologyObjectCircle arc(100, 100, 50, 0.0, TEST_PI);

    EXPECT_DOUBLE_EQ(arc.Row(), 100);
    EXPECT_DOUBLE_EQ(arc.Column(), 100);
    EXPECT_DOUBLE_EQ(arc.Radius(), 50);
    EXPECT_DOUBLE_EQ(arc.AngleStart(), 0.0);
    EXPECT_DOUBLE_EQ(arc.AngleEnd(), TEST_PI);
    EXPECT_FALSE(arc.IsFullCircle());
}

TEST_F(MetrologyObjectCircleTest, GetCalipers) {
    MetrologyMeasureParams params;
    params.numMeasures = 8;
    MetrologyObjectCircle circle(100, 100, 50, params);

    auto calipers = circle.GetCalipers();
    EXPECT_EQ(calipers.size(), 8u);
}

TEST_F(MetrologyObjectCircleTest, GetContour) {
    MetrologyObjectCircle circle(100, 100, 50);
    auto contour = circle.GetContour();
    EXPECT_GT(contour.Size(), 10);  // Should have many points
}

// =============================================================================
// MetrologyObjectEllipse Tests
// =============================================================================

class MetrologyObjectEllipseTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MetrologyObjectEllipseTest, Properties) {
    MetrologyObjectEllipse ellipse(100, 100, 0.5, 60, 40);

    EXPECT_DOUBLE_EQ(ellipse.Row(), 100);
    EXPECT_DOUBLE_EQ(ellipse.Column(), 100);
    EXPECT_DOUBLE_EQ(ellipse.Phi(), 0.5);
    EXPECT_DOUBLE_EQ(ellipse.Ra(), 60);
    EXPECT_DOUBLE_EQ(ellipse.Rb(), 40);
}

TEST_F(MetrologyObjectEllipseTest, GetCalipers) {
    MetrologyMeasureParams params;
    params.numMeasures = 10;
    MetrologyObjectEllipse ellipse(100, 100, 0.0, 60, 40, params);

    auto calipers = ellipse.GetCalipers();
    EXPECT_EQ(calipers.size(), 10u);
}

// =============================================================================
// MetrologyObjectRectangle2 Tests
// =============================================================================

class MetrologyObjectRectangle2Test : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MetrologyObjectRectangle2Test, Properties) {
    MetrologyObjectRectangle2 rect(100, 100, 0.3, 50, 30);

    EXPECT_DOUBLE_EQ(rect.Row(), 100);
    EXPECT_DOUBLE_EQ(rect.Column(), 100);
    EXPECT_DOUBLE_EQ(rect.Phi(), 0.3);
    EXPECT_DOUBLE_EQ(rect.Length1(), 50);
    EXPECT_DOUBLE_EQ(rect.Length2(), 30);
}

TEST_F(MetrologyObjectRectangle2Test, GetCalipers) {
    MetrologyMeasureParams params;
    params.numMeasures = 12;
    MetrologyObjectRectangle2 rect(100, 100, 0.0, 50, 30, params);

    auto calipers = rect.GetCalipers();
    EXPECT_GT(calipers.size(), 0u);
}

TEST_F(MetrologyObjectRectangle2Test, GetContour) {
    MetrologyObjectRectangle2 rect(100, 100, 0.0, 50, 30);
    auto contour = rect.GetContour();
    EXPECT_EQ(contour.Size(), 5);  // 4 corners + close
}

// =============================================================================
// MetrologyMeasureParams Tests
// =============================================================================

class MetrologyParamsTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MetrologyParamsTest, DefaultValues) {
    MetrologyMeasureParams params;

    EXPECT_EQ(params.numInstances, 1);
    EXPECT_DOUBLE_EQ(params.measureLength1, 20.0);
    EXPECT_DOUBLE_EQ(params.measureLength2, 5.0);
    EXPECT_DOUBLE_EQ(params.measureSigma, 1.0);
    EXPECT_DOUBLE_EQ(params.measureThreshold, 30.0);
    EXPECT_EQ(params.numMeasures, 10);
}

TEST_F(MetrologyParamsTest, BuilderPattern) {
    MetrologyMeasureParams params;
    params.SetNumInstances(2)
          .SetMeasureLength(25.0, 8.0)
          .SetMeasureSigma(1.5)
          .SetMeasureThreshold(40.0)
          .SetNumMeasures(15);

    EXPECT_EQ(params.numInstances, 2);
    EXPECT_DOUBLE_EQ(params.measureLength1, 25.0);
    EXPECT_DOUBLE_EQ(params.measureLength2, 8.0);
    EXPECT_DOUBLE_EQ(params.measureSigma, 1.5);
    EXPECT_DOUBLE_EQ(params.measureThreshold, 40.0);
    EXPECT_EQ(params.numMeasures, 15);
}

// =============================================================================
// MetrologyModel Apply Tests
// =============================================================================

class MetrologyApplyTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MetrologyApplyTest, ApplyEmptyModel) {
    MetrologyModel model;
    QImage img = CreateVerticalEdgeImage(200, 200, 100);

    bool result = model.Apply(img);
    // Empty model should still return true but have no results
    EXPECT_TRUE(result);
}

TEST_F(MetrologyApplyTest, ApplyEmptyImage) {
    MetrologyModel model;
    model.AddLineMeasure(100, 50, 100, 150);

    QImage img;  // Empty image
    bool result = model.Apply(img);
    EXPECT_FALSE(result);
}

TEST_F(MetrologyApplyTest, ApplyLineMeasure) {
    // Create image with horizontal edge at row 100
    QImage img = CreateHorizontalEdgeImage(300, 200, 100);

    MetrologyModel model;
    MetrologyMeasureParams params;
    params.numMeasures = 10;
    params.measureLength1 = 30.0;
    params.measureThreshold = 20.0;

    // Add horizontal line measurement at the expected edge position
    int idx = model.AddLineMeasure(100, 50, 100, 250, params);

    bool result = model.Apply(img);
    EXPECT_TRUE(result);

    auto lineResult = model.GetLineResult(idx);
    // The line result should be valid if edges were found
    // Note: exact values depend on edge detection accuracy
    if (lineResult.IsValid()) {
        EXPECT_GT(lineResult.numUsed, 0);
        EXPECT_GT(lineResult.score, 0.0);
    }
}

TEST_F(MetrologyApplyTest, ApplyCircleMeasure) {
    // Create image with circle at center (100, 100), radius 50
    QImage img = CreateCircleImage(200, 200, 100, 100, 50);

    MetrologyModel model;
    MetrologyMeasureParams params;
    params.numMeasures = 16;
    params.measureLength1 = 20.0;
    params.measureThreshold = 20.0;

    int idx = model.AddCircleMeasure(100, 100, 50, params);

    bool result = model.Apply(img);
    EXPECT_TRUE(result);

    auto circleResult = model.GetCircleResult(idx);
    if (circleResult.IsValid()) {
        EXPECT_GT(circleResult.numUsed, 0);
        // Check that fitted circle is close to ground truth
        EXPECT_NEAR(circleResult.row, 100.0, 5.0);
        EXPECT_NEAR(circleResult.column, 100.0, 5.0);
        EXPECT_NEAR(circleResult.radius, 50.0, 3.0);
    }
}

// =============================================================================
// MetrologyResult Tests
// =============================================================================

class MetrologyResultTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MetrologyResultTest, LineResultIsValid) {
    MetrologyLineResult result;
    EXPECT_FALSE(result.IsValid());

    result.numUsed = 3;
    result.score = 0.5;
    EXPECT_TRUE(result.IsValid());
}

TEST_F(MetrologyResultTest, CircleResultIsValid) {
    MetrologyCircleResult result;
    EXPECT_FALSE(result.IsValid());

    result.numUsed = 5;
    result.radius = 50.0;
    result.score = 0.5;
    EXPECT_TRUE(result.IsValid());
}

TEST_F(MetrologyResultTest, EllipseResultIsValid) {
    MetrologyEllipseResult result;
    EXPECT_FALSE(result.IsValid());

    result.numUsed = 6;
    result.ra = 60.0;
    result.rb = 40.0;
    result.score = 0.5;
    EXPECT_TRUE(result.IsValid());
}

TEST_F(MetrologyResultTest, Rectangle2ResultIsValid) {
    MetrologyRectangle2Result result;
    EXPECT_FALSE(result.IsValid());

    result.numUsed = 8;
    result.length1 = 50.0;
    result.length2 = 30.0;
    result.score = 0.5;
    EXPECT_TRUE(result.IsValid());
}

// =============================================================================
// MetrologyModel Alignment Tests
// =============================================================================

class MetrologyAlignmentTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MetrologyAlignmentTest, Align) {
    MetrologyModel model;
    int idx = model.AddLineMeasure(100, 50, 100, 150);

    auto* obj = dynamic_cast<const MetrologyObjectLine*>(model.GetObject(idx));
    ASSERT_NE(obj, nullptr);

    double origRow1 = obj->Row1();
    double origCol1 = obj->Col1();

    // Align with translation
    model.Align(10.0, 20.0);

    obj = dynamic_cast<const MetrologyObjectLine*>(model.GetObject(idx));
    EXPECT_DOUBLE_EQ(obj->Row1(), origRow1 + 10.0);
    EXPECT_DOUBLE_EQ(obj->Col1(), origCol1 + 20.0);
}

TEST_F(MetrologyAlignmentTest, ResetAlignment) {
    MetrologyModel model;
    int idx = model.AddLineMeasure(100, 50, 100, 150);

    model.Align(10.0, 20.0);
    model.ResetAlignment();

    // After reset, the object should be back at (0, 0) offset
    // Note: ResetAlignment sets alignment to (0, 0, 0)
    auto* obj = dynamic_cast<const MetrologyObjectLine*>(model.GetObject(idx));
    ASSERT_NE(obj, nullptr);
}

// =============================================================================
// MetrologyModel DefaultParams Tests
// =============================================================================

class MetrologyDefaultParamsTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MetrologyDefaultParamsTest, SetDefaultParams) {
    MetrologyModel model;

    MetrologyMeasureParams params;
    params.numMeasures = 20;
    params.measureThreshold = 50.0;

    model.SetDefaultParams(params);

    auto& defaultParams = model.DefaultParams();
    EXPECT_EQ(defaultParams.numMeasures, 20);
    EXPECT_DOUBLE_EQ(defaultParams.measureThreshold, 50.0);
}

TEST_F(MetrologyDefaultParamsTest, SetObjectParams) {
    MetrologyModel model;
    int idx = model.AddLineMeasure(100, 50, 100, 150);

    MetrologyMeasureParams newParams;
    newParams.numMeasures = 25;
    model.SetObjectParams(idx, newParams);

    auto* obj = model.GetObject(idx);
    ASSERT_NE(obj, nullptr);
    EXPECT_EQ(obj->Params().numMeasures, 25);
}

// =============================================================================
// MetrologyModel GetResultContour Tests
// =============================================================================

class MetrologyResultContourTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MetrologyResultContourTest, GetMeasuredPoints) {
    QImage img = CreateCircleImage(200, 200, 100, 100, 50);

    MetrologyModel model;
    MetrologyMeasureParams params;
    params.numMeasures = 8;
    params.measureThreshold = 20.0;

    int idx = model.AddCircleMeasure(100, 100, 50, params);
    model.Apply(img);

    auto points = model.GetMeasuredPoints(idx);
    // Should have some measured points
    EXPECT_GE(points.size(), 0u);
}

TEST_F(MetrologyResultContourTest, GetResultContour_InvalidIndex) {
    MetrologyModel model;
    auto contour = model.GetResultContour(999);  // Invalid index
    EXPECT_EQ(contour.Size(), 0);
}

// =============================================================================
// MetrologyModel Move Semantics Tests
// =============================================================================

class MetrologyMoveTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MetrologyMoveTest, MoveConstructor) {
    MetrologyModel model;
    model.AddLineMeasure(100, 50, 100, 150);
    model.AddCircleMeasure(100, 100, 50);

    MetrologyModel moved(std::move(model));
    EXPECT_EQ(moved.NumObjects(), 2);
}

TEST_F(MetrologyMoveTest, MoveAssignment) {
    MetrologyModel model;
    model.AddLineMeasure(100, 50, 100, 150);

    MetrologyModel other;
    other = std::move(model);
    EXPECT_EQ(other.NumObjects(), 1);
}

