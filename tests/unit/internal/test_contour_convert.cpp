/**
 * @file test_contour_convert.cpp
 * @brief Unit tests for Internal/ContourConvert module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/ContourConvert.h>
#include <QiVision/Internal/ContourProcess.h>
#include <QiVision/Core/QContour.h>
#include <QiVision/Core/QContourArray.h>
#include <QiVision/Core/QRegion.h>
#include <cmath>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

// =============================================================================
// Test Fixtures
// =============================================================================

class ContourConvertTest : public ::testing::Test {
protected:
    void SetUp() override {}

    // Helper: Create a square contour
    QContour CreateSquare(double size, Point2d center = {0, 0}) {
        QContour contour;
        double half = size / 2.0;
        contour.AddPoint(center.x - half, center.y - half);
        contour.AddPoint(center.x + half, center.y - half);
        contour.AddPoint(center.x + half, center.y + half);
        contour.AddPoint(center.x - half, center.y + half);
        contour.SetClosed(true);
        return contour;
    }

    // Helper: Create a rectangle contour
    QContour CreateRectangle(double width, double height, Point2d center = {0, 0}) {
        QContour contour;
        double hw = width / 2.0;
        double hh = height / 2.0;
        contour.AddPoint(center.x - hw, center.y - hh);
        contour.AddPoint(center.x + hw, center.y - hh);
        contour.AddPoint(center.x + hw, center.y + hh);
        contour.AddPoint(center.x - hw, center.y + hh);
        contour.SetClosed(true);
        return contour;
    }

    // Helper: Create a circle contour
    QContour CreateCircle(double radius, Point2d center = {0, 0}, int numPoints = 64) {
        QContour contour;
        const double PI = 3.14159265358979323846;
        for (int i = 0; i < numPoints; ++i) {
            double angle = 2.0 * PI * i / numPoints;
            double x = center.x + radius * std::cos(angle);
            double y = center.y + radius * std::sin(angle);
            contour.AddPoint(x, y);
        }
        contour.SetClosed(true);
        return contour;
    }

    // Helper: Create a triangle contour
    QContour CreateTriangle(Point2d p1, Point2d p2, Point2d p3) {
        QContour contour;
        contour.AddPoint(p1.x, p1.y);
        contour.AddPoint(p2.x, p2.y);
        contour.AddPoint(p3.x, p3.y);
        contour.SetClosed(true);
        return contour;
    }

    // Helper: Create an L-shaped contour
    QContour CreateLShape() {
        QContour contour;
        contour.AddPoint(10.0, 10.0);
        contour.AddPoint(40.0, 10.0);
        contour.AddPoint(40.0, 20.0);
        contour.AddPoint(20.0, 20.0);
        contour.AddPoint(20.0, 40.0);
        contour.AddPoint(10.0, 40.0);
        contour.SetClosed(true);
        return contour;
    }

    // Helper: Create a line contour (open)
    QContour CreateLine(Point2d start, Point2d end, int numPoints = 10) {
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
};

// =============================================================================
// IsPointInsideContour Tests
// =============================================================================

TEST_F(ContourConvertTest, PointInsideSquare) {
    QContour square = CreateSquare(20, {50, 50});

    // Inside
    EXPECT_TRUE(IsPointInsideContour(square, {50, 50}));
    EXPECT_TRUE(IsPointInsideContour(square, {45, 45}));
    EXPECT_TRUE(IsPointInsideContour(square, {55, 55}));

    // Outside
    EXPECT_FALSE(IsPointInsideContour(square, {30, 30}));
    EXPECT_FALSE(IsPointInsideContour(square, {70, 70}));
    EXPECT_FALSE(IsPointInsideContour(square, {0, 0}));
}

TEST_F(ContourConvertTest, PointInsideCircle) {
    QContour circle = CreateCircle(20, {50, 50});

    // Inside
    EXPECT_TRUE(IsPointInsideContour(circle, {50, 50}));  // Center
    EXPECT_TRUE(IsPointInsideContour(circle, {55, 50}));  // Near center
    EXPECT_TRUE(IsPointInsideContour(circle, {50, 60}));

    // Outside
    EXPECT_FALSE(IsPointInsideContour(circle, {50, 75}));
    EXPECT_FALSE(IsPointInsideContour(circle, {100, 100}));
}

TEST_F(ContourConvertTest, PointInsideTriangle) {
    QContour triangle = CreateTriangle({50, 10}, {90, 90}, {10, 90});

    // Inside
    EXPECT_TRUE(IsPointInsideContour(triangle, {50, 50}));

    // Outside
    EXPECT_FALSE(IsPointInsideContour(triangle, {10, 10}));
    EXPECT_FALSE(IsPointInsideContour(triangle, {90, 10}));
}

TEST_F(ContourConvertTest, PointInsideLShape) {
    QContour lShape = CreateLShape();

    // Inside the L
    EXPECT_TRUE(IsPointInsideContour(lShape, {15, 15}));
    EXPECT_TRUE(IsPointInsideContour(lShape, {35, 15}));
    EXPECT_TRUE(IsPointInsideContour(lShape, {15, 35}));

    // Outside (in the concave part)
    EXPECT_FALSE(IsPointInsideContour(lShape, {30, 30}));

    // Far outside
    EXPECT_FALSE(IsPointInsideContour(lShape, {0, 0}));
}

TEST_F(ContourConvertTest, PointInsideEmptyContour) {
    QContour empty;
    EXPECT_FALSE(IsPointInsideContour(empty, {0, 0}));

    QContour twoPoints;
    twoPoints.AddPoint(0, 0);
    twoPoints.AddPoint(10, 10);
    EXPECT_FALSE(IsPointInsideContour(twoPoints, {5, 5}));
}

// =============================================================================
// WindingNumber Tests
// =============================================================================

TEST_F(ContourConvertTest, WindingNumberInsideSquare) {
    QContour square = CreateSquare(20, {50, 50});

    // Inside: winding number should be non-zero
    EXPECT_NE(ContourWindingNumber(square, {50, 50}), 0);
    EXPECT_NE(ContourWindingNumber(square, {45, 45}), 0);

    // Outside: winding number should be zero
    EXPECT_EQ(ContourWindingNumber(square, {30, 30}), 0);
    EXPECT_EQ(ContourWindingNumber(square, {100, 100}), 0);
}

// =============================================================================
// IsContourCCW / ReverseContour Tests
// =============================================================================

TEST_F(ContourConvertTest, ContourCCW) {
    // CCW square
    QContour ccwSquare;
    ccwSquare.AddPoint(0, 0);
    ccwSquare.AddPoint(10, 0);
    ccwSquare.AddPoint(10, 10);
    ccwSquare.AddPoint(0, 10);
    ccwSquare.SetClosed(true);

    EXPECT_TRUE(IsContourCCW(ccwSquare));

    // CW square
    QContour cwSquare;
    cwSquare.AddPoint(0, 0);
    cwSquare.AddPoint(0, 10);
    cwSquare.AddPoint(10, 10);
    cwSquare.AddPoint(10, 0);
    cwSquare.SetClosed(true);

    EXPECT_FALSE(IsContourCCW(cwSquare));
}

TEST_F(ContourConvertTest, ReverseContourPreservesShape) {
    QContour square = CreateSquare(20, {50, 50});
    bool originalCCW = IsContourCCW(square);

    QContour reversed = ReverseContour(square);
    bool reversedCCW = IsContourCCW(reversed);

    // Direction should be opposite
    EXPECT_NE(originalCCW, reversedCCW);

    // Size should be same
    EXPECT_EQ(square.Size(), reversed.Size());
}

// =============================================================================
// ContourToRegion Tests
// =============================================================================

TEST_F(ContourConvertTest, SquareContourToRegion) {
    QContour square = CreateSquare(20, {50, 50});
    QRegion region = ContourToRegion(square);

    EXPECT_FALSE(region.Empty());

    // Check region contains expected points
    EXPECT_TRUE(region.Contains(50, 50));  // Center
    EXPECT_TRUE(region.Contains(45, 45));
    EXPECT_TRUE(region.Contains(55, 55));

    // Check region excludes outside points
    EXPECT_FALSE(region.Contains(30, 30));
    EXPECT_FALSE(region.Contains(70, 70));

    // Area should be approximately 20*20 = 400
    int64_t area = region.Area();
    EXPECT_GT(area, 350);
    EXPECT_LT(area, 450);
}

TEST_F(ContourConvertTest, RectangleContourToRegion) {
    QContour rect = CreateRectangle(30, 20, {50, 50});
    QRegion region = ContourToRegion(rect);

    EXPECT_FALSE(region.Empty());

    // Area should be approximately 30*20 = 600
    int64_t area = region.Area();
    EXPECT_GT(area, 550);
    EXPECT_LT(area, 650);
}

TEST_F(ContourConvertTest, CircleContourToRegion) {
    QContour circle = CreateCircle(20, {50, 50}, 100);
    QRegion region = ContourToRegion(circle);

    EXPECT_FALSE(region.Empty());

    // Area should be approximately pi*20^2 = 1256.6
    const double PI = 3.14159265358979323846;
    double expectedArea = PI * 20 * 20;
    int64_t area = region.Area();

    EXPECT_GT(area, expectedArea * 0.9);
    EXPECT_LT(area, expectedArea * 1.1);
}

TEST_F(ContourConvertTest, TriangleContourToRegion) {
    QContour triangle = CreateTriangle({10, 10}, {50, 10}, {30, 50});
    QRegion region = ContourToRegion(triangle);

    EXPECT_FALSE(region.Empty());

    // Area should be approximately 0.5 * base * height = 0.5 * 40 * 40 = 800
    int64_t area = region.Area();
    EXPECT_GT(area, 700);
    EXPECT_LT(area, 900);
}

TEST_F(ContourConvertTest, LShapeContourToRegion) {
    QContour lShape = CreateLShape();
    QRegion region = ContourToRegion(lShape);

    EXPECT_FALSE(region.Empty());

    // Check inside points
    EXPECT_TRUE(region.Contains(15, 15));
    EXPECT_TRUE(region.Contains(35, 15));
    EXPECT_TRUE(region.Contains(15, 35));

    // Check concave region is NOT filled
    EXPECT_FALSE(region.Contains(30, 30));
}

TEST_F(ContourConvertTest, EmptyContourToRegion) {
    QContour empty;
    QRegion region = ContourToRegion(empty);
    EXPECT_TRUE(region.Empty());

    QContour twoPoints;
    twoPoints.AddPoint(0, 0);
    twoPoints.AddPoint(10, 10);
    region = ContourToRegion(twoPoints);
    EXPECT_TRUE(region.Empty());
}

TEST_F(ContourConvertTest, ContourToRegionMarginMode) {
    QContour square = CreateSquare(20, {50, 50});
    QRegion region = ContourToRegion(square, ContourFillMode::Margin);

    EXPECT_FALSE(region.Empty());

    // Margin should be smaller than filled
    QRegion filledRegion = ContourToRegion(square, ContourFillMode::Filled);
    EXPECT_LT(region.Area(), filledRegion.Area());
}

// =============================================================================
// ContoursToRegion Tests
// =============================================================================

TEST_F(ContourConvertTest, MultipleContoursToRegion) {
    QContourArray contours;
    contours.Add(CreateSquare(10, {20, 20}));
    contours.Add(CreateSquare(10, {50, 50}));
    contours.Add(CreateSquare(10, {80, 80}));

    QRegion region = ContoursToRegion(contours);

    EXPECT_FALSE(region.Empty());

    // Should contain all three squares
    EXPECT_TRUE(region.Contains(20, 20));
    EXPECT_TRUE(region.Contains(50, 50));
    EXPECT_TRUE(region.Contains(80, 80));

    // Should not contain gaps
    EXPECT_FALSE(region.Contains(35, 35));
}

TEST_F(ContourConvertTest, EmptyContoursToRegion) {
    QContourArray empty;
    QRegion region = ContoursToRegion(empty);
    EXPECT_TRUE(region.Empty());
}

// =============================================================================
// ContourWithHolesToRegion Tests
// =============================================================================

TEST_F(ContourConvertTest, ContourWithSingleHole) {
    QContour outer = CreateSquare(40, {50, 50});
    QContourArray holes;
    holes.Add(CreateSquare(10, {50, 50}));

    QRegion region = ContourWithHolesToRegion(outer, holes);

    EXPECT_FALSE(region.Empty());

    // Outer region pixels
    EXPECT_TRUE(region.Contains(35, 35));
    EXPECT_TRUE(region.Contains(65, 65));

    // Hole should not be filled
    EXPECT_FALSE(region.Contains(50, 50));
}

TEST_F(ContourConvertTest, ContourWithMultipleHoles) {
    QContour outer = CreateSquare(60, {50, 50});
    QContourArray holes;
    holes.Add(CreateSquare(8, {35, 35}));
    holes.Add(CreateSquare(8, {65, 65}));

    QRegion region = ContourWithHolesToRegion(outer, holes);

    EXPECT_FALSE(region.Empty());

    // Outer region
    EXPECT_TRUE(region.Contains(50, 50));

    // Holes should be empty
    EXPECT_FALSE(region.Contains(35, 35));
    EXPECT_FALSE(region.Contains(65, 65));
}

TEST_F(ContourConvertTest, ContourWithNoHoles) {
    QContour outer = CreateSquare(20, {50, 50});
    QContourArray noHoles;

    QRegion region = ContourWithHolesToRegion(outer, noHoles);
    QRegion simpleRegion = ContourToRegion(outer);

    // Should be equivalent
    EXPECT_EQ(region.Area(), simpleRegion.Area());
}

// =============================================================================
// RegionToContours Tests
// =============================================================================

TEST_F(ContourConvertTest, SimpleRegionToContours) {
    QRegion region = QRegion::Rectangle(10, 10, 20, 20);
    QContourArray contours = RegionToContours(region);

    EXPECT_FALSE(contours.Empty());
    EXPECT_GE(contours.Size(), 1u);

    // First contour should be closed
    EXPECT_TRUE(contours[0].IsClosed());
    EXPECT_GE(contours[0].Size(), 4u);
}

TEST_F(ContourConvertTest, CircleRegionToContours) {
    QRegion region = QRegion::Circle(50, 50, 20);
    QContourArray contours = RegionToContours(region);

    EXPECT_FALSE(contours.Empty());
    EXPECT_GE(contours.Size(), 1u);

    // Contour should roughly trace the circle
    EXPECT_TRUE(contours[0].IsClosed());
}

TEST_F(ContourConvertTest, EmptyRegionToContours) {
    QRegion empty;
    QContourArray contours = RegionToContours(empty);
    EXPECT_TRUE(contours.Empty());
}

TEST_F(ContourConvertTest, RegionToContour_SingleBoundary) {
    QRegion region = QRegion::Rectangle(10, 10, 30, 30);
    QContour contour = RegionToContour(region);

    EXPECT_FALSE(contour.Empty());
    EXPECT_TRUE(contour.IsClosed());
    EXPECT_GE(contour.Size(), 4u);
}

TEST_F(ContourConvertTest, BoundaryMode_OuterOnly) {
    QRegion region = QRegion::Rectangle(10, 10, 30, 30);
    QContourArray contours = RegionToContours(region, BoundaryMode::Outer);

    EXPECT_GE(contours.Size(), 1u);
}

TEST_F(ContourConvertTest, BoundaryConnectivity_Four) {
    QRegion region = QRegion::Rectangle(10, 10, 20, 20);
    QContourArray contours4 = RegionToContours(region, BoundaryMode::Outer,
                                                BoundaryConnectivity::FourConnected);
    QContourArray contours8 = RegionToContours(region, BoundaryMode::Outer,
                                                BoundaryConnectivity::EightConnected);

    EXPECT_FALSE(contours4.Empty());
    EXPECT_FALSE(contours8.Empty());

    // 4-connected typically has more points
    EXPECT_GE(contours4[0].Size(), contours8[0].Size());
}

// =============================================================================
// RegionToSubpixelContours Tests
// =============================================================================

TEST_F(ContourConvertTest, SubpixelContours) {
    QRegion region = QRegion::Circle(50, 50, 20);
    QContourArray subpixelContours = RegionToSubpixelContours(region);

    EXPECT_FALSE(subpixelContours.Empty());

    // Subpixel contours should be smoother
    EXPECT_TRUE(subpixelContours[0].IsClosed());
}

// =============================================================================
// ContourLineToRegion Tests
// =============================================================================

TEST_F(ContourConvertTest, LineToRegion) {
    QContour line = CreateLine({10, 10}, {50, 50}, 20);
    QRegion region = ContourLineToRegion(line);

    EXPECT_FALSE(region.Empty());

    // Line should create a thin region
    EXPECT_GT(region.Area(), 0);
    EXPECT_LT(region.Area(), 100);  // Should be thin
}

TEST_F(ContourConvertTest, ClosedContourLineToRegion) {
    QContour square = CreateSquare(20, {50, 50});
    QRegion region = ContourLineToRegion(square);

    EXPECT_FALSE(region.Empty());

    // Should trace the boundary only
    QRegion filledRegion = ContourToRegion(square);
    EXPECT_LT(region.Area(), filledRegion.Area());
}

TEST_F(ContourConvertTest, HorizontalLineToRegion) {
    QContour hLine;
    hLine.AddPoint(10, 50);
    hLine.AddPoint(90, 50);
    hLine.SetClosed(false);

    QRegion region = ContourLineToRegion(hLine);

    EXPECT_FALSE(region.Empty());
    EXPECT_TRUE(region.Contains(50, 50));
}

TEST_F(ContourConvertTest, VerticalLineToRegion) {
    QContour vLine;
    vLine.AddPoint(50, 10);
    vLine.AddPoint(50, 90);
    vLine.SetClosed(false);

    QRegion region = ContourLineToRegion(vLine);

    EXPECT_FALSE(region.Empty());
    EXPECT_TRUE(region.Contains(50, 50));
}

TEST_F(ContourConvertTest, SinglePointLineToRegion) {
    QContour singlePoint;
    singlePoint.AddPoint(50, 50);
    singlePoint.SetClosed(false);

    QRegion region = ContourLineToRegion(singlePoint);

    EXPECT_FALSE(region.Empty());
    EXPECT_EQ(region.Area(), 1);
    EXPECT_TRUE(region.Contains(50, 50));
}

TEST_F(ContourConvertTest, EmptyContourLineToRegion) {
    QContour empty;
    QRegion region = ContourLineToRegion(empty);
    EXPECT_TRUE(region.Empty());
}

// =============================================================================
// ContourToThickLineRegion Tests
// =============================================================================

TEST_F(ContourConvertTest, ThickLineRegion) {
    QContour line = CreateLine({10, 50}, {90, 50}, 10);
    QRegion thinRegion = ContourLineToRegion(line);
    QRegion thickRegion = ContourToThickLineRegion(line, 5.0);

    EXPECT_FALSE(thickRegion.Empty());

    // Thick region should be larger
    EXPECT_GT(thickRegion.Area(), thinRegion.Area());
}

TEST_F(ContourConvertTest, ThickLineWithZeroThickness) {
    QContour line = CreateLine({10, 50}, {90, 50}, 10);
    QRegion thinRegion = ContourLineToRegion(line);
    QRegion thickRegion = ContourToThickLineRegion(line, 0.5);

    // Should be similar to thin line
    EXPECT_FALSE(thickRegion.Empty());
}

// =============================================================================
// ContourToPolygon Tests
// =============================================================================

TEST_F(ContourConvertTest, CircleToPolygon) {
    QContour circle = CreateCircle(20, {50, 50}, 100);
    QContour polygon = ContourToPolygon(circle, 2.0);

    EXPECT_FALSE(polygon.Empty());

    // Polygon should have fewer points than original
    EXPECT_LT(polygon.Size(), circle.Size());

    // But still be a reasonable approximation
    EXPECT_GE(polygon.Size(), 8u);
}

TEST_F(ContourConvertTest, SquareToPolygon) {
    QContour square = CreateSquare(20, {50, 50});
    QContour polygon = ContourToPolygon(square, 0.5);

    // Square should remain as 4 points
    EXPECT_EQ(polygon.Size(), 4u);
}

TEST_F(ContourConvertTest, PolygonPreservesClosedState) {
    QContour closedCircle = CreateCircle(20, {50, 50});
    closedCircle.SetClosed(true);

    QContour polygon = ContourToPolygon(closedCircle, 2.0);
    EXPECT_TRUE(polygon.IsClosed());
}

// =============================================================================
// RegionToPolygon Tests
// =============================================================================

TEST_F(ContourConvertTest, RectRegionToPolygon) {
    QRegion region = QRegion::Rectangle(10, 10, 30, 30);
    QContour polygon = RegionToPolygon(region, 1.0);

    EXPECT_FALSE(polygon.Empty());
    EXPECT_TRUE(polygon.IsClosed());
}

TEST_F(ContourConvertTest, CircleRegionToPolygon) {
    QRegion region = QRegion::Circle(50, 50, 20);
    QContour polygon = RegionToPolygon(region, 2.0);

    EXPECT_FALSE(polygon.Empty());
    EXPECT_TRUE(polygon.IsClosed());
}

// =============================================================================
// ContourPointsToRegion Tests
// =============================================================================

TEST_F(ContourConvertTest, ContourPointsToRegion) {
    QContour contour;
    contour.AddPoint(10, 10);
    contour.AddPoint(20, 20);
    contour.AddPoint(30, 30);
    contour.AddPoint(40, 40);

    QRegion region = ContourPointsToRegion(contour);

    // Each point should become one pixel
    EXPECT_EQ(region.Area(), 4);
    EXPECT_TRUE(region.Contains(10, 10));
    EXPECT_TRUE(region.Contains(20, 20));
    EXPECT_TRUE(region.Contains(30, 30));
    EXPECT_TRUE(region.Contains(40, 40));
}

TEST_F(ContourConvertTest, EmptyContourPointsToRegion) {
    QContour empty;
    QRegion region = ContourPointsToRegion(empty);
    EXPECT_TRUE(region.Empty());
}

TEST_F(ContourConvertTest, DuplicatePointsToRegion) {
    QContour contour;
    contour.AddPoint(50, 50);
    contour.AddPoint(50, 50);
    contour.AddPoint(50, 50);

    QRegion region = ContourPointsToRegion(contour);

    // Duplicates should merge
    EXPECT_EQ(region.Area(), 1);
}

// =============================================================================
// RegionPixelsToContour Tests
// =============================================================================

TEST_F(ContourConvertTest, RegionPixelsToContour) {
    QRegion region = QRegion::Rectangle(10, 10, 5, 3);
    QContour contour = RegionPixelsToContour(region);

    // 5 * 3 = 15 pixels
    EXPECT_EQ(contour.Size(), 15u);
    EXPECT_FALSE(contour.IsClosed());
}

TEST_F(ContourConvertTest, EmptyRegionPixelsToContour) {
    QRegion empty;
    QContour contour = RegionPixelsToContour(empty);
    EXPECT_TRUE(contour.Empty());
}

TEST_F(ContourConvertTest, SinglePixelRegionToContour) {
    std::vector<QRegion::Run> runs;
    runs.emplace_back(50, 50, 51);
    QRegion region(runs);

    QContour contour = RegionPixelsToContour(region);

    EXPECT_EQ(contour.Size(), 1u);
}

// =============================================================================
// Round-trip Tests (Contour -> Region -> Contour)
// =============================================================================

TEST_F(ContourConvertTest, RoundTripSquare) {
    QContour original = CreateSquare(30, {50, 50});
    QRegion region = ContourToRegion(original);
    QContour recovered = RegionToContour(region);

    EXPECT_FALSE(recovered.Empty());
    EXPECT_TRUE(recovered.IsClosed());

    // Area should be similar (within discretization error)
    double originalArea = std::abs(original.Area());
    double recoveredArea = std::abs(recovered.Area());
    EXPECT_NEAR(originalArea, recoveredArea, originalArea * 0.1);
}

TEST_F(ContourConvertTest, RoundTripCircle) {
    QContour original = CreateCircle(20, {50, 50}, 100);
    QRegion region = ContourToRegion(original);
    QContour recovered = RegionToContour(region);

    EXPECT_FALSE(recovered.Empty());
    EXPECT_TRUE(recovered.IsClosed());

    // Area should be similar
    double originalArea = std::abs(original.Area());
    double recoveredArea = std::abs(recovered.Area());
    EXPECT_NEAR(originalArea, recoveredArea, originalArea * 0.15);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(ContourConvertTest, VerySmallContour) {
    QContour small;
    small.AddPoint(50, 50);
    small.AddPoint(51, 50);
    small.AddPoint(50.5, 51);
    small.SetClosed(true);

    QRegion region = ContourToRegion(small);
    // May or may not be empty depending on rounding
    // Just ensure no crash
}

TEST_F(ContourConvertTest, LargeContour) {
    QContour large = CreateCircle(200, {500, 500}, 360);
    QRegion region = ContourToRegion(large);

    EXPECT_FALSE(region.Empty());

    // Area should be approximately pi * 200^2 = 125663.7
    const double PI = 3.14159265358979323846;
    double expectedArea = PI * 200 * 200;
    EXPECT_NEAR(static_cast<double>(region.Area()), expectedArea, expectedArea * 0.05);
}

TEST_F(ContourConvertTest, SelfIntersectingContour) {
    // Figure-8 shape
    QContour figure8;
    figure8.AddPoint(50, 30);
    figure8.AddPoint(70, 50);
    figure8.AddPoint(50, 70);
    figure8.AddPoint(30, 50);
    figure8.AddPoint(50, 70);  // Cross back
    figure8.AddPoint(70, 50);
    figure8.AddPoint(50, 30);
    figure8.SetClosed(true);

    // Should not crash
    QRegion region = ContourToRegion(figure8);
    // Even-odd fill rule applies
}

TEST_F(ContourConvertTest, ConcaveContour) {
    // Star shape
    QContour star;
    const double PI = 3.14159265358979323846;
    for (int i = 0; i < 10; ++i) {
        double angle = PI * i / 5 - PI / 2;
        double r = (i % 2 == 0) ? 30 : 15;
        star.AddPoint(50 + r * std::cos(angle), 50 + r * std::sin(angle));
    }
    star.SetClosed(true);

    QRegion region = ContourToRegion(star);
    EXPECT_FALSE(region.Empty());

    // Center should be filled
    EXPECT_TRUE(region.Contains(50, 50));
}
