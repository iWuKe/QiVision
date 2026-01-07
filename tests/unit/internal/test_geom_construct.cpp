/**
 * @file test_geom_construct.cpp
 * @brief Unit tests for Internal/GeomConstruct module
 */

#include <QiVision/Internal/GeomConstruct.h>
#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace Qi::Vision::Internal {
namespace {

constexpr double kTol = 1e-9;

bool NearEqual(double a, double b, double tol = kTol) {
    return std::abs(a - b) < tol;
}

bool PointNearEqual(const Point2d& a, const Point2d& b, double tol = kTol) {
    return NearEqual(a.x, b.x, tol) && NearEqual(a.y, b.y, tol);
}

// =============================================================================
// Perpendicular Line Tests
// =============================================================================

class PerpendicularLineTest : public ::testing::Test {};

TEST_F(PerpendicularLineTest, PerpendicularToHorizontalLine) {
    Line2d horizontal = Line2d::FromPoints({0, 0}, {10, 0});
    Point2d point(5, 3);

    Line2d perp = PerpendicularLine(horizontal, point);

    // Should pass through point
    double dist = std::abs(perp.a * point.x + perp.b * point.y + perp.c) /
                  std::sqrt(perp.a * perp.a + perp.b * perp.b);
    EXPECT_NEAR(dist, 0.0, kTol);

    // Should be perpendicular (dot product of normals = 0)
    double dot = horizontal.a * perp.a + horizontal.b * perp.b;
    EXPECT_NEAR(dot, 0.0, kTol);
}

TEST_F(PerpendicularLineTest, PerpendicularToVerticalLine) {
    Line2d vertical = Line2d::FromPoints({0, 0}, {0, 10});
    Point2d point(3, 5);

    Line2d perp = PerpendicularLine(vertical, point);

    double dot = vertical.a * perp.a + vertical.b * perp.b;
    EXPECT_NEAR(dot, 0.0, kTol);
}

TEST_F(PerpendicularLineTest, PerpendicularToDiagonalLine) {
    Line2d diagonal = Line2d::FromPoints({0, 0}, {10, 10});
    Point2d point(5, 0);

    Line2d perp = PerpendicularLine(diagonal, point);

    double dot = diagonal.a * perp.a + diagonal.b * perp.b;
    EXPECT_NEAR(dot, 0.0, kTol);
}

TEST_F(PerpendicularLineTest, PerpendicularFromPointWithFoot) {
    Line2d line = Line2d::FromPoints({0, 0}, {10, 0});
    Point2d point(5, 3);
    Point2d foot;

    Line2d perp = PerpendicularFromPoint(line, point, foot);

    // Foot should be on original line
    double distToLine = std::abs(line.a * foot.x + line.b * foot.y + line.c) /
                        std::sqrt(line.a * line.a + line.b * line.b);
    EXPECT_NEAR(distToLine, 0.0, kTol);

    // Foot should be at (5, 0) for horizontal line
    EXPECT_NEAR(foot.x, 5.0, kTol);
    EXPECT_NEAR(foot.y, 0.0, kTol);
}

TEST_F(PerpendicularLineTest, PerpendicularSegmentFromLine) {
    Line2d line = Line2d::FromPoints({0, 0}, {10, 0});
    Point2d point(5, 3);

    Segment2d perpSeg = PerpendicularSegment(line, point);

    // One endpoint should be the point
    bool hasPoint = PointNearEqual(perpSeg.p1, point) || PointNearEqual(perpSeg.p2, point);
    EXPECT_TRUE(hasPoint);

    // Other endpoint should be on line
    Point2d other = PointNearEqual(perpSeg.p1, point) ? perpSeg.p2 : perpSeg.p1;
    double distToLine = std::abs(line.a * other.x + line.b * other.y + line.c) /
                        std::sqrt(line.a * line.a + line.b * line.b);
    EXPECT_NEAR(distToLine, 0.0, kTol);
}

// =============================================================================
// Parallel Line Tests
// =============================================================================

class ParallelLineTest : public ::testing::Test {};

TEST_F(ParallelLineTest, ParallelThroughPoint) {
    Line2d line = Line2d::FromPoints({0, 0}, {10, 0});
    Point2d point(5, 3);

    Line2d parallel = ParallelLine(line, point);

    // Should pass through point
    double dist = std::abs(parallel.a * point.x + parallel.b * point.y + parallel.c) /
                  std::sqrt(parallel.a * parallel.a + parallel.b * parallel.b);
    EXPECT_NEAR(dist, 0.0, kTol);

    // Should be parallel (same direction)
    double cross = line.a * parallel.b - line.b * parallel.a;
    EXPECT_NEAR(cross, 0.0, kTol);
}

TEST_F(ParallelLineTest, ParallelAtDistance) {
    Line2d line = Line2d::FromPoints({0, 0}, {10, 0});
    double distance = 5.0;

    Line2d parallel = ParallelLineAtDistance(line, distance);

    // Should be parallel
    double cross = line.a * parallel.b - line.b * parallel.a;
    EXPECT_NEAR(cross, 0.0, kTol);

    // Distance should be correct (using c difference for normalized lines)
    double lineNorm = std::sqrt(line.a * line.a + line.b * line.b);
    double parallelNorm = std::sqrt(parallel.a * parallel.a + parallel.b * parallel.b);
    double actualDist = std::abs(parallel.c / parallelNorm - line.c / lineNorm);
    EXPECT_NEAR(actualDist, distance, kTol);
}

TEST_F(ParallelLineTest, ParallelLinesAtDistanceBothSides) {
    Line2d line = Line2d::FromPoints({0, 0}, {10, 0});
    double distance = 3.0;

    auto [line1, line2] = ParallelLinesAtDistance(line, distance);

    // Both should be parallel to original
    double cross1 = line.a * line1.b - line.b * line1.a;
    double cross2 = line.a * line2.b - line.b * line2.a;
    EXPECT_NEAR(cross1, 0.0, kTol);
    EXPECT_NEAR(cross2, 0.0, kTol);
}

TEST_F(ParallelLineTest, ParallelSegmentAtDistance) {
    Segment2d segment({0, 0}, {10, 0});
    double distance = 2.0;

    Segment2d parallel = ParallelSegmentAtDistance(segment, distance);

    // Length should be preserved
    EXPECT_NEAR(parallel.Length(), segment.Length(), kTol);
}

// =============================================================================
// Angle Bisector Tests
// =============================================================================

class AngleBisectorTest : public ::testing::Test {};

TEST_F(AngleBisectorTest, BisectorOfPerpendicularLines) {
    Line2d line1 = Line2d::FromPoints({0, 0}, {10, 0});
    Line2d line2 = Line2d::FromPoints({0, 0}, {0, 10});

    auto result = AngleBisector(line1, line2);

    ASSERT_TRUE(result.has_value());

    // Bisector of perpendicular lines should be at 45 degrees
    Line2d bisector = result.value();
    double angle = std::atan2(-bisector.a, bisector.b);
    EXPECT_NEAR(std::abs(angle), M_PI / 4, kTol);
}

TEST_F(AngleBisectorTest, BothBisectors) {
    Line2d line1 = Line2d::FromPoints({0, 0}, {10, 0});
    Line2d line2 = Line2d::FromPoints({0, 0}, {0, 10});

    auto result = AngleBisectors(line1, line2);

    ASSERT_TRUE(result.has_value());

    auto [bisector1, bisector2] = result.value();

    // Two bisectors should be perpendicular to each other
    double dot = bisector1.a * bisector2.a + bisector1.b * bisector2.b;
    EXPECT_NEAR(dot, 0.0, kTol);
}

TEST_F(AngleBisectorTest, BisectorFromThreePoints) {
    Point2d p1(10, 0);
    Point2d vertex(0, 0);
    Point2d p3(0, 10);

    Line2d bisector = AngleBisectorFromPoints(p1, vertex, p3);

    // Should pass through vertex
    double dist = std::abs(bisector.a * vertex.x + bisector.b * vertex.y + bisector.c) /
                  std::sqrt(bisector.a * bisector.a + bisector.b * bisector.b);
    EXPECT_NEAR(dist, 0.0, kTol);
}

// =============================================================================
// Tangent to Circle Tests
// =============================================================================

class TangentToCircleTest : public ::testing::Test {};

TEST_F(TangentToCircleTest, TangentLinesFromExternalPoint) {
    Circle2d circle({0, 0}, 5);
    Point2d point(10, 0);  // External point on x-axis

    auto tangents = TangentLinesToCircle(circle, point);

    EXPECT_EQ(tangents.size(), 2);

    for (const auto& tangent : tangents) {
        // Should pass through external point
        double distToPoint = std::abs(tangent.a * point.x + tangent.b * point.y + tangent.c) /
                             std::sqrt(tangent.a * tangent.a + tangent.b * tangent.b);
        EXPECT_NEAR(distToPoint, 0.0, kTol);

        // Distance from center to tangent should equal radius
        double distToCenter = std::abs(tangent.a * circle.center.x + tangent.b * circle.center.y + tangent.c) /
                              std::sqrt(tangent.a * tangent.a + tangent.b * tangent.b);
        EXPECT_NEAR(distToCenter, circle.radius, kTol);
    }
}

TEST_F(TangentToCircleTest, TangentFromPointOnCircle) {
    Circle2d circle({0, 0}, 5);
    Point2d point(5, 0);  // Point on circle

    auto tangents = TangentLinesToCircle(circle, point);

    // Should have exactly one tangent
    EXPECT_EQ(tangents.size(), 1);
}

TEST_F(TangentToCircleTest, TangentFromInsideCircle) {
    Circle2d circle({0, 0}, 5);
    Point2d point(2, 0);  // Point inside circle

    auto tangents = TangentLinesToCircle(circle, point);

    // No tangent from inside
    EXPECT_EQ(tangents.size(), 0);
}

TEST_F(TangentToCircleTest, TangentPointsFromExternalPoint) {
    Circle2d circle({0, 0}, 5);
    Point2d point(10, 0);

    auto tangentPoints = TangentPointsToCircle(circle, point);

    EXPECT_EQ(tangentPoints.size(), 2);

    for (const auto& tp : tangentPoints) {
        // Should be on circle
        double dist = tp.DistanceTo(circle.center);
        EXPECT_NEAR(dist, circle.radius, kTol);
    }
}

TEST_F(TangentToCircleTest, TangentLineAtAngle) {
    Circle2d circle({0, 0}, 5);
    double angle = M_PI / 4;  // 45 degrees

    Line2d tangent = TangentLineAtAngle(circle, angle);

    // Distance from center to tangent should equal radius
    double dist = std::abs(tangent.a * circle.center.x + tangent.b * circle.center.y + tangent.c) /
                  std::sqrt(tangent.a * tangent.a + tangent.b * tangent.b);
    EXPECT_NEAR(dist, circle.radius, kTol);
}

// =============================================================================
// Common Tangents Tests
// =============================================================================

class CommonTangentsTest : public ::testing::Test {};

TEST_F(CommonTangentsTest, ExternalTangentsOfSeparatedCircles) {
    Circle2d circle1({0, 0}, 3);
    Circle2d circle2({10, 0}, 2);

    auto tangents = ExternalCommonTangents(circle1, circle2);

    EXPECT_EQ(tangents.size(), 2);

    for (const auto& tangent : tangents) {
        // Distance from center1 to tangent should equal radius1
        double dist1 = std::abs(tangent.a * circle1.center.x + tangent.b * circle1.center.y + tangent.c) /
                       std::sqrt(tangent.a * tangent.a + tangent.b * tangent.b);
        EXPECT_NEAR(dist1, circle1.radius, kTol);

        // Distance from center2 to tangent should equal radius2
        double dist2 = std::abs(tangent.a * circle2.center.x + tangent.b * circle2.center.y + tangent.c) /
                       std::sqrt(tangent.a * tangent.a + tangent.b * tangent.b);
        EXPECT_NEAR(dist2, circle2.radius, kTol);
    }
}

TEST_F(CommonTangentsTest, InternalTangentsOfSeparatedCircles) {
    Circle2d circle1({0, 0}, 3);
    Circle2d circle2({10, 0}, 2);

    auto tangents = InternalCommonTangents(circle1, circle2);

    EXPECT_EQ(tangents.size(), 2);
}

TEST_F(CommonTangentsTest, AllCommonTangents) {
    Circle2d circle1({0, 0}, 3);
    Circle2d circle2({10, 0}, 2);

    auto result = CommonTangents(circle1, circle2);

    EXPECT_EQ(result.external.size(), 2);
    EXPECT_EQ(result.internal.size(), 2);
    EXPECT_EQ(result.TotalCount(), 4);
}

TEST_F(CommonTangentsTest, TangentsOfOverlappingCircles) {
    Circle2d circle1({0, 0}, 5);
    Circle2d circle2({3, 0}, 5);  // Overlapping

    auto result = CommonTangents(circle1, circle2);

    // Overlapping circles have 2 external tangents, 0 internal
    EXPECT_EQ(result.external.size(), 2);
    EXPECT_EQ(result.internal.size(), 0);
}

// =============================================================================
// Circumscribed Circle Tests
// =============================================================================

class CircumscribedCircleTest : public ::testing::Test {};

TEST_F(CircumscribedCircleTest, CircumscribedCircleOfRightTriangle) {
    Point2d p1(0, 0);
    Point2d p2(6, 0);
    Point2d p3(0, 8);

    auto result = CircumscribedCircle(p1, p2, p3);

    ASSERT_TRUE(result.has_value());
    Circle2d circle = result.value();

    // All three points should be on circle
    EXPECT_NEAR(p1.DistanceTo(circle.center), circle.radius, kTol);
    EXPECT_NEAR(p2.DistanceTo(circle.center), circle.radius, kTol);
    EXPECT_NEAR(p3.DistanceTo(circle.center), circle.radius, kTol);

    // For right triangle, hypotenuse is diameter
    double hypotenuse = p2.DistanceTo(p3);
    EXPECT_NEAR(circle.radius * 2, hypotenuse, kTol);
}

TEST_F(CircumscribedCircleTest, CircumscribedCircleOfEquilateralTriangle) {
    double side = 10.0;
    Point2d p1(0, 0);
    Point2d p2(side, 0);
    Point2d p3(side / 2, side * std::sqrt(3) / 2);

    auto result = CircumscribedCircle(p1, p2, p3);

    ASSERT_TRUE(result.has_value());
    Circle2d circle = result.value();

    // For equilateral triangle, R = side / sqrt(3)
    double expectedRadius = side / std::sqrt(3);
    EXPECT_NEAR(circle.radius, expectedRadius, kTol);
}

TEST_F(CircumscribedCircleTest, CollinearPointsNoCircle) {
    Point2d p1(0, 0);
    Point2d p2(5, 0);
    Point2d p3(10, 0);  // Collinear

    auto result = CircumscribedCircle(p1, p2, p3);

    EXPECT_FALSE(result.has_value());
}

// =============================================================================
// Inscribed Circle Tests
// =============================================================================

class InscribedCircleTest : public ::testing::Test {};

TEST_F(InscribedCircleTest, InscribedCircleOfRightTriangle) {
    Point2d p1(0, 0);
    Point2d p2(6, 0);
    Point2d p3(0, 8);

    auto result = InscribedCircle(p1, p2, p3);

    ASSERT_TRUE(result.has_value());
    Circle2d circle = result.value();

    // For 3-4-5 scaled right triangle (6-8-10)
    // Inradius = (a + b - c) / 2 = (6 + 8 - 10) / 2 = 2
    EXPECT_NEAR(circle.radius, 2.0, kTol);
}

TEST_F(InscribedCircleTest, InscribedCircleOfEquilateralTriangle) {
    double side = 10.0;
    Point2d p1(0, 0);
    Point2d p2(side, 0);
    Point2d p3(side / 2, side * std::sqrt(3) / 2);

    auto result = InscribedCircle(p1, p2, p3);

    ASSERT_TRUE(result.has_value());
    Circle2d circle = result.value();

    // For equilateral triangle, r = side / (2 * sqrt(3))
    double expectedRadius = side / (2 * std::sqrt(3));
    EXPECT_NEAR(circle.radius, expectedRadius, kTol);
}

// =============================================================================
// Minimum Enclosing Circle Tests
// =============================================================================

class MinEnclosingCircleTest : public ::testing::Test {};

TEST_F(MinEnclosingCircleTest, TwoPoints) {
    std::vector<Point2d> points = {{0, 0}, {10, 0}};

    auto result = MinEnclosingCircle(points);

    ASSERT_TRUE(result.has_value());
    Circle2d circle = result.value();

    EXPECT_NEAR(circle.center.x, 5.0, kTol);
    EXPECT_NEAR(circle.center.y, 0.0, kTol);
    EXPECT_NEAR(circle.radius, 5.0, kTol);
}

TEST_F(MinEnclosingCircleTest, ThreePointsOnCircle) {
    std::vector<Point2d> points = {{0, 0}, {10, 0}, {5, 5}};

    auto result = MinEnclosingCircle(points);

    ASSERT_TRUE(result.has_value());
    Circle2d circle = result.value();

    // All points should be on or inside circle
    for (const auto& p : points) {
        EXPECT_LE(p.DistanceTo(circle.center), circle.radius + kTol);
    }
}

TEST_F(MinEnclosingCircleTest, SquarePoints) {
    std::vector<Point2d> points = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};

    auto result = MinEnclosingCircle(points);

    ASSERT_TRUE(result.has_value());
    Circle2d circle = result.value();

    // Center should be at (5, 5)
    EXPECT_NEAR(circle.center.x, 5.0, kTol);
    EXPECT_NEAR(circle.center.y, 5.0, kTol);

    // Radius should be half diagonal = 5 * sqrt(2)
    EXPECT_NEAR(circle.radius, 5 * std::sqrt(2), kTol);
}

TEST_F(MinEnclosingCircleTest, SinglePoint) {
    std::vector<Point2d> points = {{5, 5}};

    auto result = MinEnclosingCircle(points);

    ASSERT_TRUE(result.has_value());
    Circle2d circle = result.value();

    EXPECT_NEAR(circle.center.x, 5.0, kTol);
    EXPECT_NEAR(circle.center.y, 5.0, kTol);
    EXPECT_NEAR(circle.radius, 0.0, kTol);
}

// =============================================================================
// Minimum Area Rectangle Tests
// =============================================================================

class MinAreaRectTest : public ::testing::Test {};

TEST_F(MinAreaRectTest, SquarePoints) {
    std::vector<Point2d> points = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};

    auto result = MinAreaRect(points);

    ASSERT_TRUE(result.has_value());
    RotatedRect2d rect = result.value();

    EXPECT_NEAR(rect.width * rect.height, 100.0, 1.0);  // Area = 100
}

TEST_F(MinAreaRectTest, RotatedSquarePoints) {
    // 45-degree rotated square
    double d = 5 * std::sqrt(2);
    std::vector<Point2d> points = {{0, 5}, {5, 0}, {10, 5}, {5, 10}};

    auto result = MinAreaRect(points);

    ASSERT_TRUE(result.has_value());
    RotatedRect2d rect = result.value();

    // Width and height should be equal (square)
    EXPECT_NEAR(rect.width, rect.height, 1.0);
}

TEST_F(MinAreaRectTest, MinBoundingRectAxisAligned) {
    std::vector<Point2d> points = {{1, 2}, {5, 8}, {3, 4}};

    auto result = MinBoundingRect(points);

    ASSERT_TRUE(result.has_value());
    Rect2d rect = result.value();

    // Should contain all points
    for (const auto& p : points) {
        EXPECT_GE(p.x, rect.x);
        EXPECT_LE(p.x, rect.x + rect.width);
        EXPECT_GE(p.y, rect.y);
        EXPECT_LE(p.y, rect.y + rect.height);
    }
}

// =============================================================================
// Convex Hull Tests
// =============================================================================

class ConvexHullTest : public ::testing::Test {};

TEST_F(ConvexHullTest, SquareWithInteriorPoint) {
    std::vector<Point2d> points = {{0, 0}, {10, 0}, {10, 10}, {0, 10}, {5, 5}};

    auto hull = ConvexHull(points);

    // Hull should have 4 points (the square corners)
    EXPECT_EQ(hull.size(), 4);
}

TEST_F(ConvexHullTest, TrianglePoints) {
    std::vector<Point2d> points = {{0, 0}, {10, 0}, {5, 10}};

    auto hull = ConvexHull(points);

    EXPECT_EQ(hull.size(), 3);
}

TEST_F(ConvexHullTest, CollinearPoints) {
    std::vector<Point2d> points = {{0, 0}, {5, 0}, {10, 0}};

    auto hull = ConvexHull(points);

    // Collinear points should give 2 endpoints
    EXPECT_EQ(hull.size(), 2);
}

TEST_F(ConvexHullTest, IsConvexTrue) {
    std::vector<Point2d> polygon = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};

    EXPECT_TRUE(IsConvex(polygon));
}

TEST_F(ConvexHullTest, IsConvexFalse) {
    // Concave polygon (star-like indentation)
    std::vector<Point2d> polygon = {{0, 0}, {5, 2}, {10, 0}, {10, 10}, {0, 10}};

    EXPECT_FALSE(IsConvex(polygon));
}

// =============================================================================
// Polygon Area Tests
// =============================================================================

class PolygonAreaTest : public ::testing::Test {};

TEST_F(PolygonAreaTest, SquareArea) {
    std::vector<Point2d> square = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};

    double area = PolygonArea(square);

    EXPECT_NEAR(area, 100.0, kTol);
}

TEST_F(PolygonAreaTest, TriangleArea) {
    std::vector<Point2d> triangle = {{0, 0}, {10, 0}, {5, 10}};

    double area = PolygonArea(triangle);

    EXPECT_NEAR(area, 50.0, kTol);
}

TEST_F(PolygonAreaTest, SignedAreaCounterClockwise) {
    // Counter-clockwise should be positive
    std::vector<Point2d> ccw = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};

    double signedArea = SignedPolygonArea(ccw);

    EXPECT_GT(signedArea, 0);
}

TEST_F(PolygonAreaTest, SignedAreaClockwise) {
    // Clockwise should be negative
    std::vector<Point2d> cw = {{0, 0}, {0, 10}, {10, 10}, {10, 0}};

    double signedArea = SignedPolygonArea(cw);

    EXPECT_LT(signedArea, 0);
}

// =============================================================================
// Polygon Centroid and Perimeter Tests
// =============================================================================

class PolygonPropertiesTest : public ::testing::Test {};

TEST_F(PolygonPropertiesTest, SquareCentroid) {
    std::vector<Point2d> square = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};

    Point2d centroid = PolygonCentroid(square);

    EXPECT_NEAR(centroid.x, 5.0, kTol);
    EXPECT_NEAR(centroid.y, 5.0, kTol);
}

TEST_F(PolygonPropertiesTest, TriangleCentroid) {
    std::vector<Point2d> triangle = {{0, 0}, {9, 0}, {0, 9}};

    Point2d centroid = PolygonCentroid(triangle);

    EXPECT_NEAR(centroid.x, 3.0, kTol);
    EXPECT_NEAR(centroid.y, 3.0, kTol);
}

TEST_F(PolygonPropertiesTest, SquarePerimeter) {
    std::vector<Point2d> square = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};

    double perimeter = PolygonPerimeter(square, true);

    EXPECT_NEAR(perimeter, 40.0, kTol);
}

TEST_F(PolygonPropertiesTest, OpenPolylinePerimeter) {
    std::vector<Point2d> polyline = {{0, 0}, {10, 0}, {10, 10}};

    double perimeter = PolygonPerimeter(polyline, false);

    EXPECT_NEAR(perimeter, 20.0, kTol);
}

// =============================================================================
// Perpendicular Bisector Tests
// =============================================================================

class PerpendicularBisectorTest : public ::testing::Test {};

TEST_F(PerpendicularBisectorTest, HorizontalSegment) {
    Segment2d segment({0, 0}, {10, 0});

    Line2d bisector = PerpendicularBisector(segment);

    // Should pass through midpoint
    Point2d midpoint(5, 0);
    double dist = std::abs(bisector.a * midpoint.x + bisector.b * midpoint.y + bisector.c) /
                  std::sqrt(bisector.a * bisector.a + bisector.b * bisector.b);
    EXPECT_NEAR(dist, 0.0, kTol);

    // Should be vertical (perpendicular to horizontal)
    Line2d segmentLine = Line2d::FromPoints(segment.p1, segment.p2);
    double dot = segmentLine.a * bisector.a + segmentLine.b * bisector.b;
    EXPECT_NEAR(dot, 0.0, kTol);
}

TEST_F(PerpendicularBisectorTest, DiagonalSegment) {
    Segment2d segment({0, 0}, {10, 10});

    Line2d bisector = PerpendicularBisector(segment);

    // Should pass through midpoint
    Point2d midpoint(5, 5);
    double dist = std::abs(bisector.a * midpoint.x + bisector.b * midpoint.y + bisector.c) /
                  std::sqrt(bisector.a * bisector.a + bisector.b * bisector.b);
    EXPECT_NEAR(dist, 0.0, kTol);
}

// =============================================================================
// Circle Construction Tests
// =============================================================================

class CircleConstructionTest : public ::testing::Test {};

TEST_F(CircleConstructionTest, CircleFromDiameter) {
    Point2d p1(0, 0);
    Point2d p2(10, 0);

    Circle2d circle = CircleFromDiameter(p1, p2);

    EXPECT_NEAR(circle.center.x, 5.0, kTol);
    EXPECT_NEAR(circle.center.y, 0.0, kTol);
    EXPECT_NEAR(circle.radius, 5.0, kTol);
}

TEST_F(CircleConstructionTest, CircleFromCenterAndPoint) {
    Point2d center(0, 0);
    Point2d pointOnCircle(3, 4);

    Circle2d circle = CircleFromCenterAndPoint(center, pointOnCircle);

    EXPECT_NEAR(circle.center.x, 0.0, kTol);
    EXPECT_NEAR(circle.center.y, 0.0, kTol);
    EXPECT_NEAR(circle.radius, 5.0, kTol);  // 3-4-5 triangle
}

// =============================================================================
// Edge Cases and Degenerate Input Tests
// =============================================================================

class GeomConstructDegenerateTest : public ::testing::Test {};

TEST_F(GeomConstructDegenerateTest, EmptyPointsConvexHull) {
    std::vector<Point2d> points;

    auto hull = ConvexHull(points);

    EXPECT_TRUE(hull.empty());
}

TEST_F(GeomConstructDegenerateTest, SinglePointConvexHull) {
    std::vector<Point2d> points = {{5, 5}};

    auto hull = ConvexHull(points);

    EXPECT_EQ(hull.size(), 1);
}

TEST_F(GeomConstructDegenerateTest, TwoPointsConvexHull) {
    std::vector<Point2d> points = {{0, 0}, {10, 10}};

    auto hull = ConvexHull(points);

    EXPECT_EQ(hull.size(), 2);
}

TEST_F(GeomConstructDegenerateTest, EmptyPointsMinEnclosingCircle) {
    std::vector<Point2d> points;

    auto result = MinEnclosingCircle(points);

    EXPECT_FALSE(result.has_value());
}

TEST_F(GeomConstructDegenerateTest, ZeroLengthSegmentPerpendicularBisector) {
    Segment2d segment({5, 5}, {5, 5});  // Zero length

    // Should handle gracefully (may return any line through the point)
    Line2d bisector = PerpendicularBisector(segment);

    // At minimum, should not crash
    EXPECT_TRUE(std::isfinite(bisector.a) || std::isfinite(bisector.b));
}

} // namespace
} // namespace Qi::Vision::Internal
