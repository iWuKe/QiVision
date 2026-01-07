#include <gtest/gtest.h>
#include <QiVision/Core/QContour.h>
#include <QiVision/Core/QMatrix.h>
#include <QiVision/Core/Constants.h>
#include <cmath>

using namespace Qi::Vision;

// =============================================================================
// Constructor Tests
// =============================================================================

TEST(QContourTest, DefaultConstructor) {
    QContour contour;
    EXPECT_TRUE(contour.Empty());
    EXPECT_EQ(contour.Size(), 0u);
    EXPECT_FALSE(contour.IsClosed());
}

TEST(QContourTest, ConstructFromPoint2dVector) {
    std::vector<Point2d> points = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    QContour contour(points, true);

    EXPECT_EQ(contour.Size(), 4u);
    EXPECT_TRUE(contour.IsClosed());
    EXPECT_DOUBLE_EQ(contour.GetPoint(0).x, 0.0);
    EXPECT_DOUBLE_EQ(contour.GetPoint(0).y, 0.0);
}

TEST(QContourTest, ConstructFromContourPointVector) {
    std::vector<ContourPoint> points = {
        {0, 0, 100, 0.0},
        {1, 0, 100, 0.0},
        {1, 1, 100, HALF_PI}
    };
    QContour contour(points, false);

    EXPECT_EQ(contour.Size(), 3u);
    EXPECT_FALSE(contour.IsClosed());
    EXPECT_DOUBLE_EQ(contour.GetAmplitude(0), 100.0);
}

// =============================================================================
// Point Access Tests
// =============================================================================

TEST(QContourTest, PointAccess) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(1, 1);
    contour.AddPoint(2, 2);

    EXPECT_DOUBLE_EQ(contour[0].x, 0.0);
    EXPECT_DOUBLE_EQ(contour[1].x, 1.0);
    EXPECT_DOUBLE_EQ(contour.At(2).x, 2.0);
}

TEST(QContourTest, PointAccessOutOfRange) {
    QContour contour;
    contour.AddPoint(0, 0);

    EXPECT_THROW(contour.At(5), std::out_of_range);
}

TEST(QContourTest, GetPoints) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(1, 0);
    contour.AddPoint(1, 1);

    std::vector<Point2d> points = contour.GetPoints();
    EXPECT_EQ(points.size(), 3u);
    EXPECT_DOUBLE_EQ(points[0].x, 0.0);
    EXPECT_DOUBLE_EQ(points[1].x, 1.0);
}

// =============================================================================
// Point Modification Tests
// =============================================================================

TEST(QContourTest, AddPoints) {
    QContour contour;
    contour.AddPoint(Point2d{0, 0});
    contour.AddPoint(1.0, 2.0);
    contour.AddPoint(ContourPoint{3, 4, 100, 0.5});

    EXPECT_EQ(contour.Size(), 3u);
    EXPECT_DOUBLE_EQ(contour[2].x, 3.0);
    EXPECT_DOUBLE_EQ(contour[2].amplitude, 100.0);
}

TEST(QContourTest, InsertRemovePoint) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(2, 2);

    contour.InsertPoint(1, ContourPoint{1, 1});
    EXPECT_EQ(contour.Size(), 3u);
    EXPECT_DOUBLE_EQ(contour[1].x, 1.0);

    contour.RemovePoint(1);
    EXPECT_EQ(contour.Size(), 2u);
    EXPECT_DOUBLE_EQ(contour[1].x, 2.0);
}

TEST(QContourTest, Clear) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(1, 1);
    contour.Clear();

    EXPECT_TRUE(contour.Empty());
}

// =============================================================================
// Attributes Tests
// =============================================================================

TEST(QContourTest, Attributes) {
    QContour contour;
    contour.AddPoint(ContourPoint{0, 0, 100, 0.5, 0.01});
    contour.AddPoint(ContourPoint{1, 1, 200, 1.0, 0.02});

    EXPECT_DOUBLE_EQ(contour.GetAmplitude(0), 100.0);
    EXPECT_DOUBLE_EQ(contour.GetDirection(1), 1.0);
    EXPECT_DOUBLE_EQ(contour.GetCurvature(1), 0.02);

    contour.SetAmplitude(0, 150);
    EXPECT_DOUBLE_EQ(contour.GetAmplitude(0), 150.0);
}

TEST(QContourTest, GetAttributeArrays) {
    QContour contour;
    contour.AddPoint(ContourPoint{0, 0, 100, 0.0, 0.01});
    contour.AddPoint(ContourPoint{1, 1, 200, 0.5, 0.02});

    auto amps = contour.GetAmplitudes();
    EXPECT_EQ(amps.size(), 2u);
    EXPECT_DOUBLE_EQ(amps[0], 100.0);
    EXPECT_DOUBLE_EQ(amps[1], 200.0);
}

// =============================================================================
// Hierarchy Tests
// =============================================================================

TEST(QContourTest, Hierarchy) {
    QContour contour;
    EXPECT_FALSE(contour.HasParent());
    EXPECT_FALSE(contour.HasChildren());

    contour.SetParent(0);
    EXPECT_TRUE(contour.HasParent());
    EXPECT_EQ(contour.GetParent(), 0);

    contour.AddChild(1);
    contour.AddChild(2);
    EXPECT_TRUE(contour.HasChildren());
    EXPECT_EQ(contour.GetChildren().size(), 2u);

    contour.RemoveChild(1);
    EXPECT_EQ(contour.GetChildren().size(), 1u);

    contour.ClearChildren();
    EXPECT_FALSE(contour.HasChildren());
}

// =============================================================================
// Geometric Properties Tests
// =============================================================================

TEST(QContourTest, Length_OpenContour) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(3, 0);
    contour.AddPoint(3, 4);

    // Length = 3 + 4 = 7
    EXPECT_DOUBLE_EQ(contour.Length(), 7.0);
}

TEST(QContourTest, Length_ClosedContour) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(3, 0);
    contour.AddPoint(3, 4);
    contour.SetClosed(true);

    // Length = 3 + 4 + 5 = 12
    EXPECT_DOUBLE_EQ(contour.Length(), 12.0);
}

TEST(QContourTest, Area_Square) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(10, 0);
    contour.AddPoint(10, 10);
    contour.AddPoint(0, 10);
    contour.SetClosed(true);

    EXPECT_DOUBLE_EQ(contour.Area(), 100.0);
}

TEST(QContourTest, SignedArea) {
    // Counter-clockwise square (positive area)
    QContour ccw;
    ccw.AddPoint(0, 0);
    ccw.AddPoint(10, 0);
    ccw.AddPoint(10, 10);
    ccw.AddPoint(0, 10);
    ccw.SetClosed(true);

    EXPECT_GT(ccw.SignedArea(), 0.0);
    EXPECT_TRUE(ccw.IsCounterClockwise());

    // Clockwise square (negative area)
    QContour cw;
    cw.AddPoint(0, 0);
    cw.AddPoint(0, 10);
    cw.AddPoint(10, 10);
    cw.AddPoint(10, 0);
    cw.SetClosed(true);

    EXPECT_LT(cw.SignedArea(), 0.0);
    EXPECT_FALSE(cw.IsCounterClockwise());
}

TEST(QContourTest, Centroid) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(10, 0);
    contour.AddPoint(10, 10);
    contour.AddPoint(0, 10);
    contour.SetClosed(true);

    Point2d center = contour.Centroid();
    EXPECT_NEAR(center.x, 5.0, 1e-10);
    EXPECT_NEAR(center.y, 5.0, 1e-10);
}

TEST(QContourTest, BoundingBox) {
    QContour contour;
    contour.AddPoint(5, 10);
    contour.AddPoint(15, 20);
    contour.AddPoint(25, 30);

    Rect2d bbox = contour.BoundingBox();
    EXPECT_DOUBLE_EQ(bbox.x, 5.0);
    EXPECT_DOUBLE_EQ(bbox.y, 10.0);
    EXPECT_DOUBLE_EQ(bbox.width, 20.0);
    EXPECT_DOUBLE_EQ(bbox.height, 20.0);
}

TEST(QContourTest, Circularity) {
    // Circle should have circularity close to 1
    QContour circle = QContour::FromCircle({0, 0, 10}, 100);
    double circ = circle.Circularity();
    EXPECT_NEAR(circ, 1.0, 0.02);  // Allow small error due to discretization
}

TEST(QContourTest, Reverse) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(1, 0);
    contour.AddPoint(2, 0);

    contour.Reverse();

    EXPECT_DOUBLE_EQ(contour[0].x, 2.0);
    EXPECT_DOUBLE_EQ(contour[1].x, 1.0);
    EXPECT_DOUBLE_EQ(contour[2].x, 0.0);
}

// =============================================================================
// Point Query Tests
// =============================================================================

TEST(QContourTest, PointAt) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(10, 0);

    Point2d p0 = contour.PointAt(0.0);
    EXPECT_DOUBLE_EQ(p0.x, 0.0);
    EXPECT_DOUBLE_EQ(p0.y, 0.0);

    Point2d p1 = contour.PointAt(1.0);
    EXPECT_DOUBLE_EQ(p1.x, 10.0);
    EXPECT_DOUBLE_EQ(p1.y, 0.0);

    Point2d mid = contour.PointAt(0.5);
    EXPECT_DOUBLE_EQ(mid.x, 5.0);
    EXPECT_DOUBLE_EQ(mid.y, 0.0);
}

TEST(QContourTest, TangentAt) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(10, 0);

    double tangent = contour.TangentAt(0.5);
    EXPECT_NEAR(tangent, 0.0, 1e-10);  // Horizontal line, angle = 0
}

TEST(QContourTest, DistanceToPoint) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(10, 0);

    // Point 5 units above the line
    double dist = contour.DistanceToPoint({5, 5});
    EXPECT_DOUBLE_EQ(dist, 5.0);

    // Point beyond the end
    dist = contour.DistanceToPoint({15, 0});
    EXPECT_DOUBLE_EQ(dist, 5.0);
}

TEST(QContourTest, Contains) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(10, 0);
    contour.AddPoint(10, 10);
    contour.AddPoint(0, 10);
    contour.SetClosed(true);

    EXPECT_TRUE(contour.Contains({5, 5}));
    EXPECT_TRUE(contour.Contains({1, 1}));
    EXPECT_FALSE(contour.Contains({15, 5}));
    EXPECT_FALSE(contour.Contains({5, 15}));
}

// =============================================================================
// Transformation Tests
// =============================================================================

TEST(QContourTest, Translate) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(10, 0);

    QContour translated = contour.Translate(5, 10);

    EXPECT_DOUBLE_EQ(translated[0].x, 5.0);
    EXPECT_DOUBLE_EQ(translated[0].y, 10.0);
    EXPECT_DOUBLE_EQ(translated[1].x, 15.0);
    EXPECT_DOUBLE_EQ(translated[1].y, 10.0);
}

TEST(QContourTest, Scale) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(10, 0);
    contour.AddPoint(10, 10);
    contour.AddPoint(0, 10);

    QContour scaled = contour.Scale(2.0, 2.0, {0, 0});

    EXPECT_DOUBLE_EQ(scaled[0].x, 0.0);
    EXPECT_DOUBLE_EQ(scaled[0].y, 0.0);
    EXPECT_DOUBLE_EQ(scaled[2].x, 20.0);
    EXPECT_DOUBLE_EQ(scaled[2].y, 20.0);
}

TEST(QContourTest, Rotate) {
    QContour contour;
    contour.AddPoint(10, 0);

    QContour rotated = contour.Rotate(HALF_PI, {0, 0});

    EXPECT_NEAR(rotated[0].x, 0.0, 1e-10);
    EXPECT_NEAR(rotated[0].y, 10.0, 1e-10);
}

TEST(QContourTest, Transform) {
    QContour contour;
    contour.AddPoint(1, 0);
    contour.AddPoint(0, 1);

    QMatrix m = QMatrix::Translation(10, 20);
    QContour transformed = contour.Transform(m);

    EXPECT_DOUBLE_EQ(transformed[0].x, 11.0);
    EXPECT_DOUBLE_EQ(transformed[0].y, 20.0);
    EXPECT_DOUBLE_EQ(transformed[1].x, 10.0);
    EXPECT_DOUBLE_EQ(transformed[1].y, 21.0);
}

TEST(QContourTest, Clone) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(1, 1);
    contour.SetClosed(true);

    QContour clone = contour.Clone();
    clone.AddPoint(2, 2);

    EXPECT_EQ(contour.Size(), 2u);  // Original unchanged
    EXPECT_EQ(clone.Size(), 3u);
}

// =============================================================================
// Processing Tests
// =============================================================================

TEST(QContourTest, Smooth) {
    QContour contour;
    for (int i = 0; i < 10; ++i) {
        // Add some noise
        double noise = (i % 2 == 0) ? 0.5 : -0.5;
        contour.AddPoint(static_cast<double>(i), noise);
    }

    QContour smoothed = contour.Smooth(1.0);

    EXPECT_EQ(smoothed.Size(), contour.Size());
    // Smoothed contour should have smaller variance
}

TEST(QContourTest, Simplify) {
    // Create a contour with many points on a line
    QContour contour;
    for (int i = 0; i <= 100; ++i) {
        contour.AddPoint(static_cast<double>(i), 0.0);
    }

    QContour simplified = contour.Simplify(0.1);

    // Should be simplified to just 2 points (start and end)
    EXPECT_EQ(simplified.Size(), 2u);
    EXPECT_DOUBLE_EQ(simplified[0].x, 0.0);
    EXPECT_DOUBLE_EQ(simplified[1].x, 100.0);
}

TEST(QContourTest, Resample) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(10, 0);

    QContour resampled = contour.Resample(2.0);

    // Length = 10, interval = 2, so 6 points
    EXPECT_EQ(resampled.Size(), 6u);
}

TEST(QContourTest, ResampleCount) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(10, 0);

    QContour resampled = contour.ResampleCount(11);

    EXPECT_EQ(resampled.Size(), 11u);
    EXPECT_DOUBLE_EQ(resampled[0].x, 0.0);
    EXPECT_DOUBLE_EQ(resampled[5].x, 5.0);
    EXPECT_DOUBLE_EQ(resampled[10].x, 10.0);
}

TEST(QContourTest, ComputeCurvature) {
    // Straight line should have zero curvature
    QContour line;
    for (int i = 0; i < 10; ++i) {
        line.AddPoint(static_cast<double>(i), 0.0);
    }
    line.ComputeCurvature(3);

    for (size_t i = 1; i < line.Size() - 1; ++i) {
        EXPECT_NEAR(line.GetCurvature(i), 0.0, 1e-10);
    }
}

TEST(QContourTest, CloseOpen) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(1, 1);

    EXPECT_FALSE(contour.IsClosed());

    contour.Close();
    EXPECT_TRUE(contour.IsClosed());

    contour.Open();
    EXPECT_FALSE(contour.IsClosed());
}

// =============================================================================
// Factory Method Tests
// =============================================================================

TEST(QContourTest, FromSegment) {
    Segment2d seg({0, 0}, {10, 0});
    QContour contour = QContour::FromSegment(seg, 2.0);

    EXPECT_GE(contour.Size(), 5u);
    EXPECT_DOUBLE_EQ(contour[0].x, 0.0);
    EXPECT_DOUBLE_EQ(contour.GetPoints().back().x, 10.0);
}

TEST(QContourTest, FromCircle) {
    Circle2d circle(50, 50, 25);
    QContour contour = QContour::FromCircle(circle, 64);

    EXPECT_EQ(contour.Size(), 64u);
    EXPECT_TRUE(contour.IsClosed());

    // Check that points are on the circle
    for (size_t i = 0; i < contour.Size(); ++i) {
        double dx = contour[i].x - 50.0;
        double dy = contour[i].y - 50.0;
        double dist = std::sqrt(dx * dx + dy * dy);
        EXPECT_NEAR(dist, 25.0, 1e-10);
    }
}

TEST(QContourTest, FromEllipse) {
    Ellipse2d ellipse({0, 0}, 10, 5, 0);
    QContour contour = QContour::FromEllipse(ellipse, 32);

    EXPECT_EQ(contour.Size(), 32u);
    EXPECT_TRUE(contour.IsClosed());

    // Check that first point is on the major axis
    EXPECT_NEAR(contour[0].x, 10.0, 1e-10);
    EXPECT_NEAR(contour[0].y, 0.0, 1e-10);
}

TEST(QContourTest, FromRectangle) {
    Rect2d rect(0, 0, 10, 20);
    QContour contour = QContour::FromRectangle(rect);

    EXPECT_EQ(contour.Size(), 4u);
    EXPECT_TRUE(contour.IsClosed());
    EXPECT_DOUBLE_EQ(contour.Area(), 200.0);
}

TEST(QContourTest, FromRotatedRect) {
    RotatedRect2d rect({5, 5}, 10, 20, 0);
    QContour contour = QContour::FromRotatedRect(rect);

    EXPECT_EQ(contour.Size(), 4u);
    EXPECT_TRUE(contour.IsClosed());
    EXPECT_DOUBLE_EQ(contour.Area(), 200.0);
}

TEST(QContourTest, FromPolygon) {
    std::vector<Point2d> vertices = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    QContour contour = QContour::FromPolygon(vertices, true);

    EXPECT_EQ(contour.Size(), 4u);
    EXPECT_TRUE(contour.IsClosed());
}

// =============================================================================
// Alias Test
// =============================================================================

TEST(QContourTest, QXldAlias) {
    QXld xld;
    xld.AddPoint(0, 0);
    xld.AddPoint(1, 1);

    EXPECT_EQ(xld.Size(), 2u);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(QContourTest, EmptyContour) {
    QContour contour;

    EXPECT_DOUBLE_EQ(contour.Length(), 0.0);
    EXPECT_DOUBLE_EQ(contour.Area(), 0.0);
    EXPECT_FALSE(contour.Contains(0, 0));

    Point2d p = contour.PointAt(0.5);
    EXPECT_DOUBLE_EQ(p.x, 0.0);
    EXPECT_DOUBLE_EQ(p.y, 0.0);
}

TEST(QContourTest, SinglePoint) {
    QContour contour;
    contour.AddPoint(5, 5);

    EXPECT_DOUBLE_EQ(contour.Length(), 0.0);

    Point2d center = contour.Centroid();
    EXPECT_DOUBLE_EQ(center.x, 5.0);
    EXPECT_DOUBLE_EQ(center.y, 5.0);
}

TEST(QContourTest, TwoPoints) {
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(10, 0);

    EXPECT_DOUBLE_EQ(contour.Length(), 10.0);

    Point2d center = contour.Centroid();
    EXPECT_DOUBLE_EQ(center.x, 5.0);
    EXPECT_DOUBLE_EQ(center.y, 0.0);
}
