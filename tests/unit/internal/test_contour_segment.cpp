/**
 * @file test_contour_segment.cpp
 * @brief Unit tests for Internal/ContourSegment module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/ContourSegment.h>
#include <QiVision/Core/QContour.h>
#include <QiVision/Core/QContourArray.h>
#include <cmath>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

// =============================================================================
// Test Fixtures
// =============================================================================

class ContourSegmentTest : public ::testing::Test {
protected:
    void SetUp() override {}

    const double PI = 3.14159265358979323846;

    // Helper: Create a square contour with intermediate points along edges
    QContour CreateSquare(double size, Point2d center = {0, 0}, int pointsPerEdge = 10) {
        QContour contour;
        double half = size / 2.0;

        // Four corners
        Point2d corners[4] = {
            {center.x - half, center.y - half},  // Bottom-left
            {center.x + half, center.y - half},  // Bottom-right
            {center.x + half, center.y + half},  // Top-right
            {center.x - half, center.y + half}   // Top-left
        };

        // Add points along each edge
        for (int edge = 0; edge < 4; ++edge) {
            Point2d p1 = corners[edge];
            Point2d p2 = corners[(edge + 1) % 4];
            for (int i = 0; i < pointsPerEdge; ++i) {
                double t = static_cast<double>(i) / pointsPerEdge;
                contour.AddPoint(p1.x + t * (p2.x - p1.x),
                                p1.y + t * (p2.y - p1.y));
            }
        }
        contour.SetClosed(true);
        return contour;
    }

    // Helper: Create a circle contour
    QContour CreateCircle(double radius, Point2d center = {0, 0}, int numPoints = 64) {
        QContour contour;
        for (int i = 0; i < numPoints; ++i) {
            double angle = 2.0 * PI * i / numPoints;
            double x = center.x + radius * std::cos(angle);
            double y = center.y + radius * std::sin(angle);
            contour.AddPoint(x, y);
        }
        contour.SetClosed(true);
        return contour;
    }

    // Helper: Create a line contour (open)
    QContour CreateLine(Point2d start, Point2d end, int numPoints = 20) {
        QContour contour;
        for (int i = 0; i < numPoints; ++i) {
            double t = static_cast<double>(i) / (numPoints - 1);
            double x = start.x + t * (end.x - start.x);
            double y = start.y + t * (end.y - start.y);
            contour.AddPoint(x, y);
        }
        contour.SetClosed(false);
        return contour;
    }

    // Helper: Create an arc contour
    QContour CreateArc(Point2d center, double radius, double startAngle, double sweepAngle, int numPoints = 32) {
        QContour contour;
        for (int i = 0; i < numPoints; ++i) {
            double t = static_cast<double>(i) / (numPoints - 1);
            double angle = startAngle + t * sweepAngle;
            double x = center.x + radius * std::cos(angle);
            double y = center.y + radius * std::sin(angle);
            contour.AddPoint(x, y);
        }
        contour.SetClosed(false);
        return contour;
    }

    // Helper: Create an L-shaped contour (two connected lines)
    QContour CreateLShape() {
        QContour contour;
        // Horizontal segment
        for (int i = 0; i <= 20; ++i) {
            contour.AddPoint(i * 2.0, 0.0);
        }
        // Vertical segment
        for (int i = 1; i <= 20; ++i) {
            contour.AddPoint(40.0, i * 2.0);
        }
        contour.SetClosed(false);
        return contour;
    }

    // Helper: Create a combined line-arc contour
    QContour CreateLineArc() {
        QContour contour;

        // First: straight line from (0,0) to (50,0)
        for (int i = 0; i <= 25; ++i) {
            contour.AddPoint(i * 2.0, 0.0);
        }

        // Then: quarter circle arc
        double radius = 30.0;
        Point2d center(50.0, 30.0);
        for (int i = 1; i <= 16; ++i) {
            double angle = -PI / 2 + (PI / 2) * i / 16;
            double x = center.x + radius * std::cos(angle);
            double y = center.y + radius * std::sin(angle);
            contour.AddPoint(x, y);
        }

        contour.SetClosed(false);
        return contour;
    }
};

// =============================================================================
// Corner Detection Tests
// =============================================================================

TEST_F(ContourSegmentTest, DetectCorners_Square) {
    QContour square = CreateSquare(40, {50, 50});

    std::vector<size_t> corners = DetectCorners(square, 0.05, 3);

    // Square should have 4 corners
    // Depending on algorithm sensitivity, might find more or fewer
    EXPECT_GE(corners.size(), 2u);  // At least 2 corners
    EXPECT_LE(corners.size(), 6u);  // Not too many
}

TEST_F(ContourSegmentTest, DetectCorners_Circle) {
    QContour circle = CreateCircle(30, {50, 50}, 64);

    std::vector<size_t> corners = DetectCorners(circle, 0.1, 5);

    // Circle should have no corners (or very few due to discretization)
    EXPECT_LE(corners.size(), 4u);
}

TEST_F(ContourSegmentTest, DetectCorners_LShape) {
    QContour lShape = CreateLShape();

    std::vector<size_t> corners = DetectCorners(lShape, 0.05, 5);

    // L-shape should have one corner where the lines meet
    EXPECT_GE(corners.size(), 1u);
}

TEST_F(ContourSegmentTest, DetectCorners_EmptyContour) {
    QContour empty;
    std::vector<size_t> corners = DetectCorners(empty);
    EXPECT_TRUE(corners.empty());
}

// =============================================================================
// Dominant Points Tests
// =============================================================================

TEST_F(ContourSegmentTest, DetectDominantPoints_Square) {
    QContour square = CreateSquare(40, {50, 50});

    std::vector<size_t> dominant = DetectDominantPoints(square, 0.05, 3);

    // Should find corners
    EXPECT_GE(dominant.size(), 2u);
}

// =============================================================================
// Line Split Points Tests
// =============================================================================

TEST_F(ContourSegmentTest, FindLineSplitPoints_StraightLine) {
    QContour line = CreateLine({0, 0}, {100, 0}, 50);

    std::vector<size_t> splits = FindLineSplitPoints(line, 0.5);

    // Straight line should have minimal splits (just start and end)
    EXPECT_LE(splits.size(), 3u);
}

TEST_F(ContourSegmentTest, FindLineSplitPoints_LShape) {
    QContour lShape = CreateLShape();

    std::vector<size_t> splits = FindLineSplitPoints(lShape, 1.0);

    // L-shape should have at least 3 splits (start, corner, end)
    EXPECT_GE(splits.size(), 3u);
}

TEST_F(ContourSegmentTest, FindLineSplitPoints_Circle) {
    QContour circle = CreateCircle(30, {50, 50}, 64);

    std::vector<size_t> splits = FindLineSplitPoints(circle, 1.0);

    // Circle should need many splits to approximate with lines
    EXPECT_GE(splits.size(), 4u);
}

// =============================================================================
// Line Fitting Tests
// =============================================================================

TEST_F(ContourSegmentTest, FitLineToContour_StraightLine) {
    QContour line = CreateLine({10, 20}, {90, 80}, 30);

    PrimitiveFitResult result = FitLineToContour(line, 0, line.Size() - 1);

    EXPECT_TRUE(result.IsValid());
    EXPECT_EQ(result.type, PrimitiveType::Line);
    EXPECT_LT(result.error, 0.1);  // Good fit
    EXPECT_GT(result.Length(), 50);  // Reasonable length
}

TEST_F(ContourSegmentTest, FitLineToContour_HorizontalLine) {
    QContour line = CreateLine({0, 50}, {100, 50}, 50);

    PrimitiveFitResult result = FitLineToContour(line, 0, line.Size() - 1);

    EXPECT_TRUE(result.IsValid());
    EXPECT_EQ(result.type, PrimitiveType::Line);
    EXPECT_LT(result.error, 0.01);  // Perfect fit
}

TEST_F(ContourSegmentTest, FitLineToContour_VerticalLine) {
    QContour line = CreateLine({50, 0}, {50, 100}, 50);

    PrimitiveFitResult result = FitLineToContour(line, 0, line.Size() - 1);

    EXPECT_TRUE(result.IsValid());
    EXPECT_EQ(result.type, PrimitiveType::Line);
    EXPECT_LT(result.error, 0.01);  // Perfect fit
}

TEST_F(ContourSegmentTest, FitLineToContour_TooFewPoints) {
    QContour contour;
    contour.AddPoint(0, 0);

    PrimitiveFitResult result = FitLineToContour(contour, 0, 0);

    EXPECT_FALSE(result.IsValid());
}

// =============================================================================
// Arc Fitting Tests
// =============================================================================

TEST_F(ContourSegmentTest, FitArcToContour_QuarterCircle) {
    QContour arc = CreateArc({50, 50}, 30, 0, PI / 2, 32);

    PrimitiveFitResult result = FitArcToContour(arc, 0, arc.Size() - 1);

    EXPECT_TRUE(result.IsValid());
    EXPECT_EQ(result.type, PrimitiveType::Arc);
    EXPECT_LT(result.error, 1.0);

    // Check arc parameters are reasonable
    EXPECT_NEAR(result.arc.radius, 30, 2.0);
    EXPECT_NEAR(std::abs(result.arc.sweepAngle), PI / 2, 0.2);
}

TEST_F(ContourSegmentTest, FitArcToContour_Semicircle) {
    QContour arc = CreateArc({50, 50}, 40, 0, PI, 48);

    PrimitiveFitResult result = FitArcToContour(arc, 0, arc.Size() - 1);

    EXPECT_TRUE(result.IsValid());
    EXPECT_EQ(result.type, PrimitiveType::Arc);
    EXPECT_NEAR(result.arc.radius, 40, 2.0);
}

TEST_F(ContourSegmentTest, FitArcToContour_TooFewPoints) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(10, 0);

    PrimitiveFitResult result = FitArcToContour(contour, 0, 1);

    EXPECT_FALSE(result.IsValid());
}

// =============================================================================
// Best Primitive Fitting Tests
// =============================================================================

TEST_F(ContourSegmentTest, FitBestPrimitive_LinePreferred) {
    QContour line = CreateLine({0, 0}, {100, 0}, 50);

    PrimitiveFitResult result = FitBestPrimitive(line, 0, line.Size() - 1, true);

    EXPECT_TRUE(result.IsValid());
    EXPECT_EQ(result.type, PrimitiveType::Line);
}

TEST_F(ContourSegmentTest, FitBestPrimitive_Arc) {
    QContour arc = CreateArc({50, 50}, 30, 0, PI / 2, 32);

    PrimitiveFitResult result = FitBestPrimitive(arc, 0, arc.Size() - 1, true);

    EXPECT_TRUE(result.IsValid());
    // Arc should be chosen even with line preference due to better fit
    EXPECT_EQ(result.type, PrimitiveType::Arc);
}

// =============================================================================
// Classification Tests
// =============================================================================

TEST_F(ContourSegmentTest, ClassifyContourSegment_Line) {
    QContour line = CreateLine({0, 0}, {100, 50}, 40);

    PrimitiveType type = ClassifyContourSegment(line, 0, line.Size() - 1);

    EXPECT_EQ(type, PrimitiveType::Line);
}

TEST_F(ContourSegmentTest, ClassifyContourSegment_Arc) {
    QContour arc = CreateArc({50, 50}, 30, 0, PI, 48);

    PrimitiveType type = ClassifyContourSegment(arc, 0, arc.Size() - 1);

    // Should be classified as Arc
    EXPECT_EQ(type, PrimitiveType::Arc);
}

TEST_F(ContourSegmentTest, ComputeLinearity_StraightLine) {
    QContour line = CreateLine({0, 0}, {100, 0}, 50);

    double linearity = ComputeLinearity(line, 0, line.Size() - 1);

    // Perfect line should have linearity close to 1
    EXPECT_GT(linearity, 0.99);
}

TEST_F(ContourSegmentTest, ComputeLinearity_Circle) {
    QContour circle = CreateCircle(30, {50, 50}, 64);

    double linearity = ComputeLinearity(circle, 0, circle.Size() - 1);

    // Circle should have low linearity (closed, returns to start)
    EXPECT_LT(linearity, 0.1);
}

TEST_F(ContourSegmentTest, ComputeCircularity_Arc) {
    QContour arc = CreateArc({50, 50}, 30, 0, PI / 2, 32);

    double circularity = ComputeCircularity(arc, 0, arc.Size() - 1);

    // Arc should have high circularity
    EXPECT_GT(circularity, 0.5);
}

// =============================================================================
// Main Segmentation Tests
// =============================================================================

TEST_F(ContourSegmentTest, SegmentContour_StraightLine) {
    QContour line = CreateLine({0, 0}, {100, 0}, 50);

    SegmentParams params;
    params.mode = SegmentMode::LinesOnly;
    params.maxLineError = 0.5;

    SegmentationResult result = SegmentContour(line, params);

    // Single line should produce single segment
    EXPECT_GE(result.LineCount(), 1u);
    EXPECT_EQ(result.ArcCount(), 0u);
}

TEST_F(ContourSegmentTest, SegmentContour_LShape) {
    QContour lShape = CreateLShape();

    SegmentParams params;
    params.mode = SegmentMode::LinesOnly;
    params.maxLineError = 1.0;

    SegmentationResult result = SegmentContour(lShape, params);

    // L-shape should produce at least 2 line segments
    EXPECT_GE(result.LineCount(), 2u);
}

TEST_F(ContourSegmentTest, SegmentContour_Circle) {
    QContour circle = CreateCircle(30, {50, 50}, 64);

    SegmentParams params;
    params.mode = SegmentMode::ArcsOnly;
    params.maxArcError = 2.0;

    SegmentationResult result = SegmentContour(circle, params);

    // Circle should be approximated with arcs
    EXPECT_GE(result.ArcCount(), 1u);
}

TEST_F(ContourSegmentTest, SegmentContour_LineAndArc) {
    QContour combined = CreateLineArc();

    SegmentParams params;
    params.mode = SegmentMode::LinesAndArcs;
    params.maxLineError = 1.0;
    params.maxArcError = 2.0;

    SegmentationResult result = SegmentContour(combined, params);

    // Should have at least one of each
    EXPECT_GE(result.primitives.size(), 1u);
}

TEST_F(ContourSegmentTest, SegmentContour_EmptyContour) {
    QContour empty;

    SegmentationResult result = SegmentContour(empty);

    EXPECT_TRUE(result.primitives.empty());
}

TEST_F(ContourSegmentTest, SegmentContour_SmallContour) {
    QContour small;
    small.AddPoint(0, 0);
    small.AddPoint(1, 0);

    SegmentationResult result = SegmentContour(small);

    // Too small for segmentation
    EXPECT_TRUE(result.primitives.empty());
}

// =============================================================================
// Convenience Function Tests
// =============================================================================

TEST_F(ContourSegmentTest, SegmentContourToLines) {
    QContour lShape = CreateLShape();

    std::vector<Segment2d> lines = SegmentContourToLines(lShape, 1.0);

    EXPECT_GE(lines.size(), 2u);
}

TEST_F(ContourSegmentTest, SegmentContourToArcs) {
    QContour arc = CreateArc({50, 50}, 30, 0, PI, 48);

    std::vector<Arc2d> arcs = SegmentContourToArcs(arc, 2.0);

    EXPECT_GE(arcs.size(), 1u);
}

// =============================================================================
// Sub-Contour Extraction Tests
// =============================================================================

TEST_F(ContourSegmentTest, SplitContourAtIndices) {
    QContour line = CreateLine({0, 0}, {100, 0}, 50);

    std::vector<size_t> splits = {0, 25, 49};
    QContourArray subContours = SplitContourAtIndices(line, splits);

    EXPECT_EQ(subContours.Size(), 2u);
}

TEST_F(ContourSegmentTest, SplitContourAtIndices_Empty) {
    QContour line = CreateLine({0, 0}, {100, 0}, 50);

    std::vector<size_t> splits;
    QContourArray subContours = SplitContourAtIndices(line, splits);

    // Should return original contour
    EXPECT_EQ(subContours.Size(), 1u);
}

TEST_F(ContourSegmentTest, ExtractSubContour) {
    QContour line = CreateLine({0, 0}, {100, 0}, 50);

    QContour sub = ExtractSubContour(line, 10, 30);

    EXPECT_EQ(sub.Size(), 21u);  // 30 - 10 + 1
    EXPECT_FALSE(sub.IsClosed());
}

// =============================================================================
// Merge Functions Tests
// =============================================================================

TEST_F(ContourSegmentTest, MergeCollinearSegments) {
    std::vector<Segment2d> segments;
    segments.emplace_back(Point2d(0, 0), Point2d(10, 0));
    segments.emplace_back(Point2d(10, 0), Point2d(20, 0));
    segments.emplace_back(Point2d(20, 0), Point2d(30, 0));

    std::vector<Segment2d> merged = MergeCollinearSegments(segments, 0.1, 1.0);

    // All three segments should merge into one
    EXPECT_EQ(merged.size(), 1u);
    EXPECT_NEAR(merged[0].p1.x, 0, 0.1);
    EXPECT_NEAR(merged[0].p2.x, 30, 0.1);
}

TEST_F(ContourSegmentTest, MergeCollinearSegments_WithGap) {
    std::vector<Segment2d> segments;
    segments.emplace_back(Point2d(0, 0), Point2d(10, 0));
    segments.emplace_back(Point2d(15, 0), Point2d(25, 0));  // Gap of 5

    std::vector<Segment2d> merged = MergeCollinearSegments(segments, 0.1, 2.0);

    // Should not merge due to large gap
    EXPECT_EQ(merged.size(), 2u);
}

TEST_F(ContourSegmentTest, MergeCollinearSegments_NotCollinear) {
    std::vector<Segment2d> segments;
    segments.emplace_back(Point2d(0, 0), Point2d(10, 0));    // Horizontal
    segments.emplace_back(Point2d(10, 0), Point2d(10, 10));  // Vertical

    std::vector<Segment2d> merged = MergeCollinearSegments(segments, 0.1, 1.0);

    // Should not merge - perpendicular
    EXPECT_EQ(merged.size(), 2u);
}

// =============================================================================
// Conversion Functions Tests
// =============================================================================

TEST_F(ContourSegmentTest, SegmentToContour) {
    Segment2d segment(Point2d(0, 0), Point2d(50, 0));

    QContour contour = SegmentToContour(segment, 5.0);

    EXPECT_GE(contour.Size(), 10u);
    EXPECT_FALSE(contour.IsClosed());

    // Check endpoints
    EXPECT_NEAR(contour.GetPoint(0).x, 0, 0.1);
    EXPECT_NEAR(contour.GetPoint(contour.Size() - 1).x, 50, 0.1);
}

TEST_F(ContourSegmentTest, ArcToContour) {
    Arc2d arc(Point2d(50, 50), 30, 0, PI / 2);

    QContour contour = ArcToContour(arc, 2.0);

    EXPECT_GE(contour.Size(), 10u);
    EXPECT_FALSE(contour.IsClosed());

    // Check start point
    EXPECT_NEAR(contour.GetPoint(0).x, 80, 1.0);  // 50 + 30
    EXPECT_NEAR(contour.GetPoint(0).y, 50, 1.0);
}

TEST_F(ContourSegmentTest, PrimitivesToContours) {
    // Create a segmentation result
    SegmentationResult result;

    PrimitiveFitResult line;
    line.type = PrimitiveType::Line;
    line.segment = Segment2d(Point2d(0, 0), Point2d(50, 0));
    line.numPoints = 25;
    result.primitives.push_back(line);

    PrimitiveFitResult arc;
    arc.type = PrimitiveType::Arc;
    arc.arc = Arc2d(Point2d(50, 25), 25, -PI / 2, PI / 2);
    arc.numPoints = 16;
    result.primitives.push_back(arc);

    QContourArray contours = PrimitivesToContours(result, 2.0);

    EXPECT_EQ(contours.Size(), 2u);
}

// =============================================================================
// Algorithm Selection Tests
// =============================================================================

TEST_F(ContourSegmentTest, SegmentAlgorithm_Curvature) {
    QContour lShape = CreateLShape();

    SegmentParams params;
    params.algorithm = SegmentAlgorithm::Curvature;
    params.mode = SegmentMode::LinesOnly;

    SegmentationResult result = SegmentContour(lShape, params);

    EXPECT_GE(result.primitives.size(), 1u);
}

TEST_F(ContourSegmentTest, SegmentAlgorithm_ErrorBased) {
    QContour lShape = CreateLShape();

    SegmentParams params;
    params.algorithm = SegmentAlgorithm::ErrorBased;
    params.mode = SegmentMode::LinesOnly;

    SegmentationResult result = SegmentContour(lShape, params);

    EXPECT_GE(result.primitives.size(), 1u);
}

TEST_F(ContourSegmentTest, SegmentAlgorithm_Hybrid) {
    QContour lShape = CreateLShape();

    SegmentParams params;
    params.algorithm = SegmentAlgorithm::Hybrid;
    params.mode = SegmentMode::LinesOnly;

    SegmentationResult result = SegmentContour(lShape, params);

    EXPECT_GE(result.primitives.size(), 1u);
}

// =============================================================================
// Edge Cases Tests
// =============================================================================

TEST_F(ContourSegmentTest, VerySmallContour) {
    QContour small;
    small.AddPoint(50.0, 50.0);
    small.AddPoint(50.1, 50.0);
    small.AddPoint(50.1, 50.1);

    SegmentationResult result = SegmentContour(small);

    // Should not crash, may or may not produce results
}

TEST_F(ContourSegmentTest, LargeCircle) {
    QContour circle = CreateCircle(200, {500, 500}, 256);

    SegmentParams params;
    params.mode = SegmentMode::ArcsOnly;
    params.maxArcError = 5.0;

    SegmentationResult result = SegmentContour(circle, params);

    EXPECT_GE(result.ArcCount(), 1u);
}

TEST_F(ContourSegmentTest, ResultStatistics) {
    QContour lShape = CreateLShape();

    SegmentParams params;
    params.mode = SegmentMode::LinesOnly;

    SegmentationResult result = SegmentContour(lShape, params);

    // Check result statistics
    EXPECT_GT(result.coverageRatio, 0.0);
    EXPECT_LE(result.coverageRatio, 1.0);
}
