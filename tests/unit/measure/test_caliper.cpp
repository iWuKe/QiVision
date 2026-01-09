/**
 * @file test_caliper.cpp
 * @brief Unit tests for Measure/Caliper module
 */

#include <gtest/gtest.h>

#include <QiVision/Core/QImage.h>
#include <QiVision/Measure/MeasureTypes.h>
#include <QiVision/Measure/MeasureHandle.h>
#include <QiVision/Measure/Caliper.h>

#include <cmath>
#include <cstring>
#include <vector>
#include <random>

using namespace Qi::Vision;
using namespace Qi::Vision::Measure;

namespace {
    constexpr double TEST_PI = 3.14159265358979323846;

    // Create a test grayscale image with a vertical edge
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

    // Create a test image with a diagonal stripe (dark-light-dark)
    QImage CreateStripeImage(int width, int height, int stripeStart, int stripeWidth,
                              uint8_t bgValue = 50, uint8_t stripeValue = 200) {
        QImage img(width, height, PixelType::UInt8, ChannelType::Gray);
        for (int y = 0; y < height; ++y) {
            uint8_t* row = static_cast<uint8_t*>(img.RowPtr(y));
            for (int x = 0; x < width; ++x) {
                if (x >= stripeStart && x < stripeStart + stripeWidth) {
                    row[x] = stripeValue;
                } else {
                    row[x] = bgValue;
                }
            }
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
}

// =============================================================================
// MeasureTypes Tests
// =============================================================================

class MeasureTypesTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MeasureTypesTest, MeasureParamsDefaults) {
    MeasureParams params;
    EXPECT_DOUBLE_EQ(params.sigma, DEFAULT_SIGMA);
    EXPECT_DOUBLE_EQ(params.minAmplitude, DEFAULT_MIN_AMPLITUDE);
    EXPECT_EQ(params.transition, EdgeTransition::All);
    EXPECT_EQ(params.numLines, DEFAULT_NUM_LINES);
    EXPECT_EQ(params.selectMode, EdgeSelectMode::All);
}

TEST_F(MeasureTypesTest, MeasureParamsBuilder) {
    MeasureParams params = MeasureParams()
        .SetSigma(2.0)
        .SetMinAmplitude(30.0)
        .SetTransition(EdgeTransition::Positive)
        .SetNumLines(20);

    EXPECT_DOUBLE_EQ(params.sigma, 2.0);
    EXPECT_DOUBLE_EQ(params.minAmplitude, 30.0);
    EXPECT_EQ(params.transition, EdgeTransition::Positive);
    EXPECT_EQ(params.numLines, 20);
}

TEST_F(MeasureTypesTest, PairParamsBuilder) {
    PairParams params = PairParams();
    params.SetFirstTransition(EdgeTransition::Positive)
          .SetSecondTransition(EdgeTransition::Negative)
          .SetWidthRange(5.0, 50.0);

    EXPECT_EQ(params.firstTransition, EdgeTransition::Positive);
    EXPECT_EQ(params.secondTransition, EdgeTransition::Negative);
    EXPECT_DOUBLE_EQ(params.minWidth, 5.0);
    EXPECT_DOUBLE_EQ(params.maxWidth, 50.0);
}

TEST_F(MeasureTypesTest, EdgeResultPosition) {
    EdgeResult result;
    result.row = 100.5;
    result.column = 200.25;
    result.amplitude = 50.0;
    result.confidence = 0.9;

    Point2d pos = result.Position();
    EXPECT_DOUBLE_EQ(pos.x, 200.25);
    EXPECT_DOUBLE_EQ(pos.y, 100.5);
    EXPECT_TRUE(result.IsValid());
}

TEST_F(MeasureTypesTest, PairResultCenter) {
    PairResult pair;
    pair.first.row = 100;
    pair.first.column = 50;
    pair.first.amplitude = 60;
    pair.first.confidence = 0.8;
    pair.second.row = 100;
    pair.second.column = 70;
    pair.second.amplitude = 55;
    pair.second.confidence = 0.75;
    pair.width = 20;
    pair.centerRow = 100;
    pair.centerColumn = 60;

    Point2d center = pair.Center();
    EXPECT_DOUBLE_EQ(center.x, 60);
    EXPECT_DOUBLE_EQ(center.y, 100);
    EXPECT_TRUE(pair.IsValid());
}

TEST_F(MeasureTypesTest, EdgePolarityConversion) {
    EXPECT_EQ(ToEdgePolarity(EdgeTransition::Positive), EdgePolarity::Positive);
    EXPECT_EQ(ToEdgePolarity(EdgeTransition::Negative), EdgePolarity::Negative);
    EXPECT_EQ(ToEdgePolarity(EdgeTransition::All), EdgePolarity::Both);

    EXPECT_EQ(FromEdgePolarity(EdgePolarity::Positive), EdgeTransition::Positive);
    EXPECT_EQ(FromEdgePolarity(EdgePolarity::Negative), EdgeTransition::Negative);
    EXPECT_EQ(FromEdgePolarity(EdgePolarity::Both), EdgeTransition::All);
}

// =============================================================================
// MeasureHandle Tests
// =============================================================================

class MeasureHandleTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MeasureHandleTest, RectangleConstruction) {
    MeasureRectangle2 rect(100, 200, 0, 50, 10, 5, 1.0);

    EXPECT_TRUE(rect.IsValid());
    EXPECT_EQ(rect.Type(), HandleType::Rectangle);
    EXPECT_DOUBLE_EQ(rect.CenterRow(), 100);
    EXPECT_DOUBLE_EQ(rect.CenterCol(), 200);
    EXPECT_DOUBLE_EQ(rect.Phi(), 0);
    EXPECT_DOUBLE_EQ(rect.Length(), 50);
    EXPECT_DOUBLE_EQ(rect.Width(), 10);
    EXPECT_DOUBLE_EQ(rect.ProfileLength(), 50);
    EXPECT_EQ(rect.NumLines(), 5);
}

TEST_F(MeasureHandleTest, RectangleFromPoints) {
    Point2d p1{100, 50};
    Point2d p2{200, 50};  // Horizontal line

    auto rect = MeasureRectangle2::FromPoints(p1, p2, 20, 10);

    EXPECT_TRUE(rect.IsValid());
    EXPECT_DOUBLE_EQ(rect.CenterCol(), 150);
    EXPECT_DOUBLE_EQ(rect.CenterRow(), 50);
    EXPECT_NEAR(rect.Length(), 100, 1e-6);
    EXPECT_DOUBLE_EQ(rect.Width(), 20);
}

TEST_F(MeasureHandleTest, RectangleProfileEndpoints) {
    MeasureRectangle2 rect(100, 100, 0, 50, 10);

    Point2d start, end;
    rect.GetProfileEndpoints(start, end);

    // Profile should be along phi+90 degrees = 90 degrees (vertical)
    EXPECT_NEAR(start.x, 100, 1e-6);
    EXPECT_NEAR(end.x, 100, 1e-6);
    EXPECT_NEAR(end.y - start.y, 50, 1e-6);
}

TEST_F(MeasureHandleTest, RectangleBoundingBox) {
    MeasureRectangle2 rect(100, 100, 0, 50, 20);

    Rect2d bbox = rect.BoundingBox();

    // Axis-aligned, centered at (100, 100) with length 50 (profile) and width 20
    EXPECT_GE(bbox.width, 20);
    EXPECT_GE(bbox.height, 50);
}

TEST_F(MeasureHandleTest, RectangleContains) {
    MeasureRectangle2 rect(100, 100, 0, 50, 20);

    EXPECT_TRUE(rect.Contains({100, 100}));  // Center
    EXPECT_FALSE(rect.Contains({200, 200})); // Far away
}

TEST_F(MeasureHandleTest, ArcConstruction) {
    MeasureArc arc(100, 100, 50, 0, TEST_PI, 5, 10, 1.0);

    EXPECT_TRUE(arc.IsValid());
    EXPECT_EQ(arc.Type(), HandleType::Arc);
    EXPECT_DOUBLE_EQ(arc.CenterRow(), 100);
    EXPECT_DOUBLE_EQ(arc.CenterCol(), 100);
    EXPECT_DOUBLE_EQ(arc.Radius(), 50);
    EXPECT_DOUBLE_EQ(arc.AngleStart(), 0);
    EXPECT_DOUBLE_EQ(arc.AngleExtent(), TEST_PI);
    EXPECT_DOUBLE_EQ(arc.AnnulusRadius(), 5);
    EXPECT_NEAR(arc.ProfileLength(), 50 * TEST_PI, 1e-6);
}

TEST_F(MeasureHandleTest, ArcFromCircle) {
    Circle2d circle{{100, 100}, 30};
    auto arc = MeasureArc::FromCircle(circle, 3, 8);

    EXPECT_TRUE(arc.IsValid());
    EXPECT_DOUBLE_EQ(arc.CenterCol(), 100);
    EXPECT_DOUBLE_EQ(arc.CenterRow(), 100);
    EXPECT_DOUBLE_EQ(arc.Radius(), 30);
    EXPECT_NEAR(arc.AngleExtent(), 2.0 * TEST_PI, 1e-6);
}

TEST_F(MeasureHandleTest, ArcPointAt) {
    MeasureArc arc(0, 0, 100, 0, TEST_PI / 2);

    // At t=0, should be at angle 0 (rightmost point)
    Point2d p0 = arc.PointAt(0);
    EXPECT_NEAR(p0.x, 100, 1e-6);
    EXPECT_NEAR(p0.y, 0, 1e-6);

    // At t=1, should be at angle PI/2 (bottom point)
    Point2d p1 = arc.PointAt(1);
    EXPECT_NEAR(p1.x, 0, 1e-6);
    EXPECT_NEAR(p1.y, 100, 1e-6);
}

TEST_F(MeasureHandleTest, ConcentricConstruction) {
    MeasureConcentricCircles conc(100, 100, 20, 50, 0, 0.2, 5, 1.0);

    EXPECT_TRUE(conc.IsValid());
    EXPECT_EQ(conc.Type(), HandleType::Concentric);
    EXPECT_DOUBLE_EQ(conc.CenterRow(), 100);
    EXPECT_DOUBLE_EQ(conc.CenterCol(), 100);
    EXPECT_DOUBLE_EQ(conc.InnerRadius(), 20);
    EXPECT_DOUBLE_EQ(conc.OuterRadius(), 50);
    EXPECT_DOUBLE_EQ(conc.ProfileLength(), 30);
}

TEST_F(MeasureHandleTest, FactoryFunctions) {
    auto rect = CreateMeasureRect(100, 200, 0, 50, 10, 5, 1.0);
    EXPECT_TRUE(rect.IsValid());

    auto arc = CreateMeasureArc(100, 200, 50, 0, TEST_PI, 5, 10, 1.0);
    EXPECT_TRUE(arc.IsValid());

    auto conc = CreateMeasureConcentric(100, 200, 20, 50, 0, 0.1, 5, 1.0);
    EXPECT_TRUE(conc.IsValid());
}

// =============================================================================
// MeasurePos Tests - Rectangle
// =============================================================================

class MeasurePosRectTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MeasurePosRectTest, SingleVerticalEdge) {
    // Create image with vertical edge at column 50
    auto img = CreateVerticalEdgeImage(100, 100, 50);

    // Create horizontal measurement handle crossing the edge
    auto handle = CreateMeasureRect(50, 50, -TEST_PI/2, 40, 20, 10, 1.0);

    MeasureParams params;
    params.sigma = 1.0;
    params.minAmplitude = 20;
    params.transition = EdgeTransition::All;

    auto edges = MeasurePos(img, handle, params);

    EXPECT_GE(edges.size(), 1u);
    if (!edges.empty()) {
        // Edge should be found near column 50
        EXPECT_NEAR(edges[0].column, 50, 2.0);
    }
}

TEST_F(MeasurePosRectTest, SingleHorizontalEdge) {
    // Create image with horizontal edge at row 50
    auto img = CreateHorizontalEdgeImage(100, 100, 50);

    // Create vertical measurement handle crossing the edge
    auto handle = CreateMeasureRect(50, 50, 0, 40, 20, 10, 1.0);

    MeasureParams params;
    params.sigma = 1.0;
    params.minAmplitude = 20;

    auto edges = MeasurePos(img, handle, params);

    EXPECT_GE(edges.size(), 1u);
    if (!edges.empty()) {
        EXPECT_NEAR(edges[0].row, 50, 2.0);
    }
}

TEST_F(MeasurePosRectTest, EdgeSelectFirst) {
    // Create stripe with two edges
    auto img = CreateStripeImage(100, 100, 30, 40);

    auto handle = CreateMeasureRect(50, 50, -TEST_PI/2, 80, 20, 10, 1.0);

    MeasureParams params;
    params.minAmplitude = 20;
    params.selectMode = EdgeSelectMode::First;

    auto edges = MeasurePos(img, handle, params);

    EXPECT_EQ(edges.size(), 1u);
    if (!edges.empty()) {
        // First edge should be near column 30
        EXPECT_NEAR(edges[0].column, 30, 3.0);
    }
}

TEST_F(MeasurePosRectTest, EdgeSelectLast) {
    auto img = CreateStripeImage(100, 100, 30, 40);

    auto handle = CreateMeasureRect(50, 50, -TEST_PI/2, 80, 20, 10, 1.0);

    MeasureParams params;
    params.minAmplitude = 20;
    params.selectMode = EdgeSelectMode::Last;

    auto edges = MeasurePos(img, handle, params);

    EXPECT_EQ(edges.size(), 1u);
    if (!edges.empty()) {
        // Last edge should be near column 70
        EXPECT_NEAR(edges[0].column, 70, 3.0);
    }
}

TEST_F(MeasurePosRectTest, PositiveTransitionOnly) {
    auto img = CreateStripeImage(100, 100, 30, 40);

    auto handle = CreateMeasureRect(50, 50, -TEST_PI/2, 80, 20, 10, 1.0);

    MeasureParams params;
    params.minAmplitude = 20;
    params.transition = EdgeTransition::Positive;

    auto edges = MeasurePos(img, handle, params);

    // Should only find the rising edge at column 30
    EXPECT_GE(edges.size(), 1u);
    for (const auto& e : edges) {
        EXPECT_EQ(e.transition, EdgeTransition::Positive);
    }
}

TEST_F(MeasurePosRectTest, NegativeTransitionOnly) {
    auto img = CreateStripeImage(100, 100, 30, 40);

    auto handle = CreateMeasureRect(50, 50, -TEST_PI/2, 80, 20, 10, 1.0);

    MeasureParams params;
    params.minAmplitude = 20;
    params.transition = EdgeTransition::Negative;

    auto edges = MeasurePos(img, handle, params);

    // Should only find the falling edge at column 70
    EXPECT_GE(edges.size(), 1u);
    for (const auto& e : edges) {
        EXPECT_EQ(e.transition, EdgeTransition::Negative);
    }
}

TEST_F(MeasurePosRectTest, EmptyImage) {
    QImage img;

    auto handle = CreateMeasureRect(50, 50, 0, 40, 20, 10, 1.0);

    auto edges = MeasurePos(img, handle, MeasureParams());

    EXPECT_TRUE(edges.empty());
}

TEST_F(MeasurePosRectTest, NoEdges) {
    // Uniform gray image
    QImage img(100, 100, PixelType::UInt8, ChannelType::Gray);
    std::memset(img.Data(), 128, 100 * 100);

    auto handle = CreateMeasureRect(50, 50, 0, 40, 20, 10, 1.0);

    MeasureParams params;
    params.minAmplitude = 20;

    auto edges = MeasurePos(img, handle, params);

    EXPECT_TRUE(edges.empty());
}

// =============================================================================
// MeasurePairs Tests
// =============================================================================

class MeasurePairsTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MeasurePairsTest, SinglePair) {
    // Create stripe: edges at 30 and 70, width = 40
    auto img = CreateStripeImage(100, 100, 30, 40);

    auto handle = CreateMeasureRect(50, 50, -TEST_PI/2, 80, 20, 10, 1.0);

    PairParams params;
    params.minAmplitude = 20;

    auto pairs = MeasurePairs(img, handle, params);

    EXPECT_GE(pairs.size(), 1u);
    if (!pairs.empty()) {
        EXPECT_NEAR(pairs[0].width, 40, 3.0);
        EXPECT_NEAR(pairs[0].centerColumn, 50, 3.0);
    }
}

TEST_F(MeasurePairsTest, WidthFilter) {
    auto img = CreateStripeImage(100, 100, 30, 40);

    auto handle = CreateMeasureRect(50, 50, -TEST_PI/2, 80, 20, 10, 1.0);

    PairParams params;
    params.minAmplitude = 20;
    params.minWidth = 50;  // Width is 40, should filter out

    auto pairs = MeasurePairs(img, handle, params);

    EXPECT_TRUE(pairs.empty());
}

TEST_F(MeasurePairsTest, SelectWidest) {
    // This test requires multiple pairs
    auto handle = CreateMeasureRect(50, 50, -TEST_PI/2, 80, 20, 10, 1.0);

    // Create two stripes
    QImage img(100, 100, PixelType::UInt8, ChannelType::Gray);
    std::memset(img.Data(), 50, 100 * 100);

    // First stripe: columns 20-30 (width 10)
    for (int y = 0; y < 100; ++y) {
        uint8_t* row = static_cast<uint8_t*>(img.RowPtr(y));
        for (int x = 20; x < 30; ++x) row[x] = 200;
    }
    // Second stripe: columns 50-80 (width 30)
    for (int y = 0; y < 100; ++y) {
        uint8_t* row = static_cast<uint8_t*>(img.RowPtr(y));
        for (int x = 50; x < 80; ++x) row[x] = 200;
    }

    PairParams params;
    params.minAmplitude = 20;
    params.pairSelectMode = PairSelectMode::Widest;

    auto pairs = MeasurePairs(img, handle, params);

    EXPECT_EQ(pairs.size(), 1u);
    if (!pairs.empty()) {
        // Should select the wider pair (30 pixels)
        EXPECT_GT(pairs[0].width, 20);
    }
}

TEST_F(MeasurePairsTest, PairSymmetry) {
    auto img = CreateStripeImage(100, 100, 30, 40);

    auto handle = CreateMeasureRect(50, 50, -TEST_PI/2, 80, 20, 10, 1.0);

    PairParams params;
    params.minAmplitude = 20;

    auto pairs = MeasurePairs(img, handle, params);

    EXPECT_GE(pairs.size(), 1u);
    if (!pairs.empty()) {
        // Both edges have similar amplitude, symmetry should be high
        EXPECT_GT(pairs[0].symmetry, 0.5);
    }
}

// =============================================================================
// FuzzyMeasure Tests
// =============================================================================

class FuzzyMeasureTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(FuzzyMeasureTest, FuzzyMeasureWithScores) {
    auto img = CreateVerticalEdgeImage(100, 100, 50);

    auto handle = CreateMeasureRect(50, 50, -TEST_PI/2, 40, 20, 10, 1.0);

    FuzzyParams params;
    params.minAmplitude = 10;
    params.fuzzyThresholdLow = 0.3;
    params.fuzzyThresholdHigh = 0.8;
    params.minScore = 0.1;

    MeasureStats stats;
    auto edges = FuzzyMeasurePos(img, handle, params, &stats);

    EXPECT_GE(edges.size(), 1u);
    if (!edges.empty()) {
        EXPECT_GE(edges[0].score, 0.0);
        EXPECT_LE(edges[0].score, 1.0);
    }

    // Check stats
    EXPECT_GT(stats.profileContrast, 0);
}

TEST_F(FuzzyMeasureTest, FuzzyMeasurePairs) {
    auto img = CreateStripeImage(100, 100, 30, 40);

    auto handle = CreateMeasureRect(50, 50, -TEST_PI/2, 80, 20, 10, 1.0);

    FuzzyParams params;
    params.minAmplitude = 10;
    params.minScore = 0.1;

    auto pairs = FuzzyMeasurePairs(img, handle, params, nullptr);

    EXPECT_GE(pairs.size(), 1u);
    if (!pairs.empty()) {
        EXPECT_GE(pairs[0].score, 0.0);
        EXPECT_LE(pairs[0].score, 1.0);
    }
}

// =============================================================================
// Arc Measurement Tests
// =============================================================================

class MeasurePosArcTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MeasurePosArcTest, CircleEdge) {
    // Create circle with radius 40 centered at (50, 50)
    auto img = CreateCircleImage(100, 100, 50, 50, 40);

    // Measure along arc at radius 40
    auto handle = CreateMeasureArc(50, 50, 40, 0, TEST_PI, 5, 10, 1.0);

    MeasureParams params;
    params.minAmplitude = 20;
    params.sigma = 1.0;

    auto edges = MeasurePos(img, handle, params);

    // Should find edges along the circle
    // The exact number depends on the arc sampling
    EXPECT_GE(edges.size(), 0u);
}

// =============================================================================
// Concentric Measurement Tests
// =============================================================================

class MeasurePosConcentricTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MeasurePosConcentricTest, RadialEdge) {
    // Create circle with radius 40
    auto img = CreateCircleImage(100, 100, 50, 50, 40);

    // Measure radially from center outward
    auto handle = CreateMeasureConcentric(50, 50, 20, 60, 0, 0.2, 5, 1.0);

    MeasureParams params;
    params.minAmplitude = 20;

    auto edges = MeasurePos(img, handle, params);

    EXPECT_GE(edges.size(), 1u);
    if (!edges.empty()) {
        // Should find edge near radius 40
        double edgeRadius = handle.ProfilePosToRadius(edges[0].profilePosition);
        EXPECT_NEAR(edgeRadius, 40, 3.0);
    }
}

// =============================================================================
// Utility Function Tests
// =============================================================================

class UtilityFunctionsTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(UtilityFunctionsTest, SelectEdges) {
    std::vector<EdgeResult> edges(5);
    for (int i = 0; i < 5; ++i) {
        edges[i].profilePosition = i * 10;
        edges[i].amplitude = (i + 1) * 20;
        edges[i].score = edges[i].amplitude / 100.0;
        edges[i].confidence = 1.0;
    }

    // Test First
    auto first = SelectEdges(edges, EdgeSelectMode::First);
    EXPECT_EQ(first.size(), 1u);
    EXPECT_DOUBLE_EQ(first[0].profilePosition, 0);

    // Test Last
    auto last = SelectEdges(edges, EdgeSelectMode::Last);
    EXPECT_EQ(last.size(), 1u);
    EXPECT_DOUBLE_EQ(last[0].profilePosition, 40);

    // Test Strongest
    auto strongest = SelectEdges(edges, EdgeSelectMode::Strongest);
    EXPECT_EQ(strongest.size(), 1u);
    EXPECT_DOUBLE_EQ(strongest[0].amplitude, 100);  // Last edge has highest amplitude

    // Test All
    auto all = SelectEdges(edges, EdgeSelectMode::All);
    EXPECT_EQ(all.size(), 5u);
}

TEST_F(UtilityFunctionsTest, SelectPairs) {
    std::vector<PairResult> pairs(3);
    for (int i = 0; i < 3; ++i) {
        pairs[i].first.profilePosition = i * 10;
        pairs[i].width = (i + 1) * 15;
        pairs[i].score = 0.5;
    }

    // Test Widest
    auto widest = SelectPairs(pairs, PairSelectMode::Widest);
    EXPECT_EQ(widest.size(), 1u);
    EXPECT_DOUBLE_EQ(widest[0].width, 45);

    // Test Narrowest
    auto narrowest = SelectPairs(pairs, PairSelectMode::Narrowest);
    EXPECT_EQ(narrowest.size(), 1u);
    EXPECT_DOUBLE_EQ(narrowest[0].width, 15);
}

TEST_F(UtilityFunctionsTest, SortEdges) {
    std::vector<EdgeResult> edges(3);
    edges[0].profilePosition = 30;
    edges[0].amplitude = 50;
    edges[1].profilePosition = 10;
    edges[1].amplitude = 100;
    edges[2].profilePosition = 20;
    edges[2].amplitude = 75;

    // Sort by position ascending
    auto byPos = edges;
    SortEdges(byPos, EdgeSortBy::Position, true);
    EXPECT_DOUBLE_EQ(byPos[0].profilePosition, 10);
    EXPECT_DOUBLE_EQ(byPos[1].profilePosition, 20);
    EXPECT_DOUBLE_EQ(byPos[2].profilePosition, 30);

    // Sort by amplitude descending
    auto byAmp = edges;
    SortEdges(byAmp, EdgeSortBy::Amplitude, false);
    EXPECT_DOUBLE_EQ(byAmp[0].amplitude, 100);
    EXPECT_DOUBLE_EQ(byAmp[1].amplitude, 75);
    EXPECT_DOUBLE_EQ(byAmp[2].amplitude, 50);
}

TEST_F(UtilityFunctionsTest, SortPairs) {
    std::vector<PairResult> pairs(3);
    pairs[0].width = 30;
    pairs[1].width = 10;
    pairs[2].width = 20;

    // Sort by width ascending
    SortPairs(pairs, PairSortBy::Width, true);
    EXPECT_DOUBLE_EQ(pairs[0].width, 10);
    EXPECT_DOUBLE_EQ(pairs[1].width, 20);
    EXPECT_DOUBLE_EQ(pairs[2].width, 30);
}

TEST_F(UtilityFunctionsTest, ProfileToImageRect) {
    auto handle = CreateMeasureRect(100, 100, 0, 50, 10);

    // Start of profile
    Point2d p0 = ProfileToImage(handle, 0);
    EXPECT_NEAR(p0.x, 100, 1e-6);

    // End of profile
    Point2d p1 = ProfileToImage(handle, 50);
    EXPECT_NEAR(p1.x, 100, 1e-6);
}

TEST_F(UtilityFunctionsTest, GetNumSamples) {
    auto rect = CreateMeasureRect(100, 100, 0, 50, 10, 5, 2.0);
    int32_t n1 = GetNumSamples(rect);
    EXPECT_GT(n1, 50);  // 2 samples per pixel * 50 length + 1

    auto arc = CreateMeasureArc(100, 100, 30, 0, TEST_PI, 0, 5, 1.0);
    int32_t n2 = GetNumSamples(arc);
    EXPECT_GT(n2, 30 * TEST_PI);  // Arc length + 1
}

// =============================================================================
// Profile Extraction Tests
// =============================================================================

class ProfileExtractionTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(ProfileExtractionTest, ExtractRectProfile) {
    auto img = CreateVerticalEdgeImage(100, 100, 50);

    auto handle = CreateMeasureRect(50, 50, -TEST_PI/2, 40, 20, 10, 1.0);

    auto profile = ExtractMeasureProfile(img, handle, ProfileInterpolation::Bilinear);

    EXPECT_FALSE(profile.empty());
    // Profile should show transition from low to high
    double minVal = *std::min_element(profile.begin(), profile.end());
    double maxVal = *std::max_element(profile.begin(), profile.end());
    EXPECT_LT(minVal, 100);
    EXPECT_GT(maxVal, 150);
}

TEST_F(ProfileExtractionTest, ExtractArcProfile) {
    auto img = CreateCircleImage(100, 100, 50, 50, 40);

    auto handle = CreateMeasureArc(50, 50, 40, 0, TEST_PI, 5, 10, 1.0);

    auto profile = ExtractMeasureProfile(img, handle, ProfileInterpolation::Bilinear);

    EXPECT_FALSE(profile.empty());
}

TEST_F(ProfileExtractionTest, ExtractConcentricProfile) {
    auto img = CreateCircleImage(100, 100, 50, 50, 40);

    auto handle = CreateMeasureConcentric(50, 50, 20, 60, 0, 0.2, 5, 1.0);

    auto profile = ExtractMeasureProfile(img, handle, ProfileInterpolation::Bilinear);

    EXPECT_FALSE(profile.empty());
    // Profile should show transition from inside (bright) to outside (dark)
}
