/**
 * @file test_distance.cpp
 * @brief Unit tests for Internal/Distance module
 *
 * Tests cover:
 * - Point-to-Point distance
 * - Point-to-Line distance (unsigned and signed)
 * - Point-to-Segment distance (unsigned and signed)
 * - Point-to-Circle distance (unsigned and signed)
 * - Point-to-Ellipse distance (unsigned and signed)
 * - Point-to-Arc distance
 * - Point-to-RotatedRect distance (unsigned and signed)
 * - Line-to-Line distance
 * - Segment-to-Segment distance
 * - Circle-to-Circle distance
 * - Point-to-Contour distance (unsigned and signed)
 * - Batch distance functions
 * - Utility functions (NearestPointOnContour, HausdorffDistance, PointInsidePolygon)
 */

#include <QiVision/Internal/Distance.h>
#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>
#include <random>

namespace Qi::Vision::Internal {
namespace {

// =============================================================================
// Test Utilities
// =============================================================================

/// Check if two doubles are approximately equal
bool NearEqual(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

/// Check if two points are approximately equal
bool PointNearEqual(const Point2d& a, const Point2d& b, double tol = 1e-9) {
    return NearEqual(a.x, b.x, tol) && NearEqual(a.y, b.y, tol);
}

// =============================================================================
// Point-to-Point Distance Tests
// =============================================================================

class PointToPointDistanceTest : public ::testing::Test {};

TEST_F(PointToPointDistanceTest, SamePoint) {
    Point2d p1(5.0, 10.0);
    Point2d p2(5.0, 10.0);

    double dist = DistancePointToPoint(p1, p2);
    EXPECT_NEAR(dist, 0.0, 1e-14);
}

TEST_F(PointToPointDistanceTest, HorizontalDistance) {
    Point2d p1(0.0, 0.0);
    Point2d p2(10.0, 0.0);

    double dist = DistancePointToPoint(p1, p2);
    EXPECT_NEAR(dist, 10.0, 1e-14);
}

TEST_F(PointToPointDistanceTest, VerticalDistance) {
    Point2d p1(0.0, 0.0);
    Point2d p2(0.0, -15.0);

    double dist = DistancePointToPoint(p1, p2);
    EXPECT_NEAR(dist, 15.0, 1e-14);
}

TEST_F(PointToPointDistanceTest, DiagonalDistance) {
    Point2d p1(0.0, 0.0);
    Point2d p2(3.0, 4.0);

    double dist = DistancePointToPoint(p1, p2);
    EXPECT_NEAR(dist, 5.0, 1e-14);
}

TEST_F(PointToPointDistanceTest, SquaredDistance) {
    Point2d p1(0.0, 0.0);
    Point2d p2(3.0, 4.0);

    double distSq = DistancePointToPointSquared(p1, p2);
    EXPECT_NEAR(distSq, 25.0, 1e-14);
}

TEST_F(PointToPointDistanceTest, LargeCoordinates) {
    Point2d p1(1e6, 1e6);
    Point2d p2(1e6 + 3.0, 1e6 + 4.0);

    double dist = DistancePointToPoint(p1, p2);
    EXPECT_NEAR(dist, 5.0, 1e-9);  // Allow slightly larger tolerance for large coordinates
}

// =============================================================================
// Point-to-Line Distance Tests
// =============================================================================

class PointToLineDistanceTest : public ::testing::Test {};

TEST_F(PointToLineDistanceTest, PointOnLine) {
    Line2d line = Line2d::FromPoints({0.0, 0.0}, {10.0, 0.0});  // X-axis
    Point2d point(5.0, 0.0);

    DistanceResult result = DistancePointToLine(point, line);

    EXPECT_NEAR(result.distance, 0.0, 1e-14);
    EXPECT_TRUE(PointNearEqual(result.closestPoint, point, 1e-12));
}

TEST_F(PointToLineDistanceTest, PointAboveLine) {
    Line2d line = Line2d::FromPoints({0.0, 0.0}, {10.0, 0.0});  // X-axis
    Point2d point(5.0, 3.0);

    DistanceResult result = DistancePointToLine(point, line);

    EXPECT_NEAR(result.distance, 3.0, 1e-14);
    EXPECT_TRUE(PointNearEqual(result.closestPoint, Point2d(5.0, 0.0), 1e-12));
}

TEST_F(PointToLineDistanceTest, PointBelowLine) {
    Line2d line = Line2d::FromPoints({0.0, 0.0}, {10.0, 0.0});  // X-axis
    Point2d point(5.0, -7.0);

    DistanceResult result = DistancePointToLine(point, line);

    EXPECT_NEAR(result.distance, 7.0, 1e-14);
    EXPECT_TRUE(PointNearEqual(result.closestPoint, Point2d(5.0, 0.0), 1e-12));
}

TEST_F(PointToLineDistanceTest, DiagonalLine) {
    // Line y = x (45 degrees)
    Line2d line = Line2d::FromPoints({0.0, 0.0}, {1.0, 1.0});
    Point2d point(0.0, 2.0);  // Point perpendicular distance = sqrt(2)

    DistanceResult result = DistancePointToLine(point, line);

    EXPECT_NEAR(result.distance, std::sqrt(2.0), 1e-12);
    EXPECT_TRUE(PointNearEqual(result.closestPoint, Point2d(1.0, 1.0), 1e-12));
}

TEST_F(PointToLineDistanceTest, SignedDistancePositive) {
    Line2d line = Line2d::FromPoints({0.0, 0.0}, {10.0, 0.0});  // X-axis
    Point2d point(5.0, 3.0);

    SignedDistanceResult result = SignedDistancePointToLine(point, line);

    // Sign depends on normal direction, just check absolute value
    EXPECT_NEAR(result.Distance(), 3.0, 1e-14);
}

TEST_F(PointToLineDistanceTest, SignedDistanceNegative) {
    Line2d line = Line2d::FromPoints({0.0, 0.0}, {10.0, 0.0});  // X-axis
    Point2d point(5.0, -3.0);

    SignedDistanceResult result = SignedDistancePointToLine(point, line);

    // Check absolute value is correct
    EXPECT_NEAR(result.Distance(), 3.0, 1e-14);

    // Opposite sign from point above line
    SignedDistanceResult resultAbove = SignedDistancePointToLine(Point2d(5.0, 3.0), line);
    EXPECT_TRUE((result.signedDistance > 0) != (resultAbove.signedDistance > 0) ||
                std::abs(result.signedDistance) < 1e-12 ||
                std::abs(resultAbove.signedDistance) < 1e-12);
}

// =============================================================================
// Point-to-Segment Distance Tests
// =============================================================================

class PointToSegmentDistanceTest : public ::testing::Test {};

TEST_F(PointToSegmentDistanceTest, PointOnSegment) {
    Segment2d seg({0.0, 0.0}, {10.0, 0.0});
    Point2d point(5.0, 0.0);

    DistanceResult result = DistancePointToSegment(point, seg);

    EXPECT_NEAR(result.distance, 0.0, 1e-14);
    EXPECT_TRUE(PointNearEqual(result.closestPoint, point, 1e-12));
    EXPECT_NEAR(result.parameter, 0.5, 1e-12);
}

TEST_F(PointToSegmentDistanceTest, ClosestPointInterior) {
    Segment2d seg({0.0, 0.0}, {10.0, 0.0});
    Point2d point(5.0, 4.0);

    DistanceResult result = DistancePointToSegment(point, seg);

    EXPECT_NEAR(result.distance, 4.0, 1e-14);
    EXPECT_TRUE(PointNearEqual(result.closestPoint, Point2d(5.0, 0.0), 1e-12));
    EXPECT_NEAR(result.parameter, 0.5, 1e-12);
}

TEST_F(PointToSegmentDistanceTest, ClosestPointP1) {
    Segment2d seg({0.0, 0.0}, {10.0, 0.0});
    Point2d point(-5.0, 3.0);  // Beyond p1

    DistanceResult result = DistancePointToSegment(point, seg);

    double expectedDist = std::sqrt(25.0 + 9.0);  // sqrt(34)
    EXPECT_NEAR(result.distance, expectedDist, 1e-12);
    EXPECT_TRUE(PointNearEqual(result.closestPoint, Point2d(0.0, 0.0), 1e-12));
    EXPECT_NEAR(result.parameter, 0.0, 1e-12);
}

TEST_F(PointToSegmentDistanceTest, ClosestPointP2) {
    Segment2d seg({0.0, 0.0}, {10.0, 0.0});
    Point2d point(15.0, 4.0);  // Beyond p2

    DistanceResult result = DistancePointToSegment(point, seg);

    double expectedDist = std::sqrt(25.0 + 16.0);  // sqrt(41)
    EXPECT_NEAR(result.distance, expectedDist, 1e-12);
    EXPECT_TRUE(PointNearEqual(result.closestPoint, Point2d(10.0, 0.0), 1e-12));
    EXPECT_NEAR(result.parameter, 1.0, 1e-12);
}

TEST_F(PointToSegmentDistanceTest, DegenerateSegment) {
    Segment2d seg({5.0, 5.0}, {5.0, 5.0});  // Zero length
    Point2d point(8.0, 9.0);

    DistanceResult result = DistancePointToSegment(point, seg);

    EXPECT_NEAR(result.distance, 5.0, 1e-12);  // sqrt(9 + 16)
    EXPECT_TRUE(PointNearEqual(result.closestPoint, Point2d(5.0, 5.0), 1e-12));
}

TEST_F(PointToSegmentDistanceTest, SignedDistanceLeftSide) {
    Segment2d seg({0.0, 0.0}, {10.0, 0.0});
    Point2d pointAbove(5.0, 3.0);
    Point2d pointBelow(5.0, -3.0);

    SignedDistanceResult resultAbove = SignedDistancePointToSegment(pointAbove, seg);
    SignedDistanceResult resultBelow = SignedDistancePointToSegment(pointBelow, seg);

    // Points on opposite sides should have opposite signs
    EXPECT_TRUE((resultAbove.signedDistance > 0) != (resultBelow.signedDistance > 0));
    EXPECT_NEAR(resultAbove.Distance(), 3.0, 1e-14);
    EXPECT_NEAR(resultBelow.Distance(), 3.0, 1e-14);
}

// =============================================================================
// Point-to-Circle Distance Tests
// =============================================================================

class PointToCircleDistanceTest : public ::testing::Test {};

TEST_F(PointToCircleDistanceTest, PointOnCircle) {
    Circle2d circle({0.0, 0.0}, 10.0);
    Point2d point(10.0, 0.0);

    DistanceResult result = DistancePointToCircle(point, circle);

    EXPECT_NEAR(result.distance, 0.0, 1e-14);
    EXPECT_TRUE(PointNearEqual(result.closestPoint, point, 1e-12));
}

TEST_F(PointToCircleDistanceTest, PointOutsideCircle) {
    Circle2d circle({0.0, 0.0}, 10.0);
    Point2d point(15.0, 0.0);

    DistanceResult result = DistancePointToCircle(point, circle);

    EXPECT_NEAR(result.distance, 5.0, 1e-14);
    EXPECT_TRUE(PointNearEqual(result.closestPoint, Point2d(10.0, 0.0), 1e-12));
}

TEST_F(PointToCircleDistanceTest, PointInsideCircle) {
    Circle2d circle({0.0, 0.0}, 10.0);
    Point2d point(3.0, 0.0);

    DistanceResult result = DistancePointToCircle(point, circle);

    EXPECT_NEAR(result.distance, 7.0, 1e-14);
    EXPECT_TRUE(PointNearEqual(result.closestPoint, Point2d(10.0, 0.0), 1e-12));
}

TEST_F(PointToCircleDistanceTest, PointAtCenter) {
    Circle2d circle({0.0, 0.0}, 10.0);
    Point2d point(0.0, 0.0);

    DistanceResult result = DistancePointToCircle(point, circle);

    EXPECT_NEAR(result.distance, 10.0, 1e-14);
    // Closest point can be any point on circle (implementation chooses +X direction)
    EXPECT_TRUE(PointNearEqual(result.closestPoint, Point2d(10.0, 0.0), 1e-12));
}

TEST_F(PointToCircleDistanceTest, DiagonalPoint) {
    Circle2d circle({0.0, 0.0}, 10.0);
    Point2d point(20.0, 20.0);  // Distance to center = 20*sqrt(2) ~ 28.28

    DistanceResult result = DistancePointToCircle(point, circle);

    double distToCenter = 20.0 * std::sqrt(2.0);
    EXPECT_NEAR(result.distance, distToCenter - 10.0, 1e-10);

    // Closest point at 45 degrees
    double angle = PI / 4.0;
    Point2d expectedClosest(10.0 * std::cos(angle), 10.0 * std::sin(angle));
    EXPECT_TRUE(PointNearEqual(result.closestPoint, expectedClosest, 1e-10));
}

TEST_F(PointToCircleDistanceTest, SignedDistanceOutside) {
    Circle2d circle({0.0, 0.0}, 10.0);
    Point2d point(15.0, 0.0);

    SignedDistanceResult result = SignedDistancePointToCircle(point, circle);

    EXPECT_GT(result.signedDistance, 0.0);  // Outside = positive
    EXPECT_NEAR(result.signedDistance, 5.0, 1e-14);
}

TEST_F(PointToCircleDistanceTest, SignedDistanceInside) {
    Circle2d circle({0.0, 0.0}, 10.0);
    Point2d point(3.0, 0.0);

    SignedDistanceResult result = SignedDistancePointToCircle(point, circle);

    EXPECT_LT(result.signedDistance, 0.0);  // Inside = negative
    EXPECT_NEAR(result.signedDistance, -7.0, 1e-14);
}

TEST_F(PointToCircleDistanceTest, ZeroRadiusCircle) {
    Circle2d circle({5.0, 5.0}, 0.0);
    Point2d point(8.0, 9.0);

    DistanceResult result = DistancePointToCircle(point, circle);

    EXPECT_NEAR(result.distance, 5.0, 1e-12);  // sqrt(9 + 16)
}

// =============================================================================
// Point-to-Ellipse Distance Tests
// =============================================================================

class PointToEllipseDistanceTest : public ::testing::Test {};

TEST_F(PointToEllipseDistanceTest, CircularEllipse) {
    // Ellipse with a == b is a circle
    Ellipse2d ellipse({0.0, 0.0}, 10.0, 10.0, 0.0);
    Point2d point(15.0, 0.0);

    DistanceResult result = DistancePointToEllipse(point, ellipse);

    EXPECT_NEAR(result.distance, 5.0, 1e-10);
}

TEST_F(PointToEllipseDistanceTest, PointOnMajorAxis) {
    Ellipse2d ellipse({0.0, 0.0}, 100.0, 50.0, 0.0);  // a=100, b=50
    Point2d point(150.0, 0.0);

    DistanceResult result = DistancePointToEllipse(point, ellipse);

    EXPECT_NEAR(result.distance, 50.0, 1e-10);
    EXPECT_NEAR(result.closestPoint.x, 100.0, 1e-8);
    EXPECT_NEAR(result.closestPoint.y, 0.0, 1e-8);
}

TEST_F(PointToEllipseDistanceTest, PointOnMinorAxis) {
    Ellipse2d ellipse({0.0, 0.0}, 100.0, 50.0, 0.0);  // a=100, b=50
    Point2d point(0.0, 80.0);

    DistanceResult result = DistancePointToEllipse(point, ellipse);

    EXPECT_NEAR(result.distance, 30.0, 1e-10);
    EXPECT_NEAR(result.closestPoint.x, 0.0, 1e-8);
    EXPECT_NEAR(result.closestPoint.y, 50.0, 1e-8);
}

TEST_F(PointToEllipseDistanceTest, PointOnEllipse) {
    Ellipse2d ellipse({0.0, 0.0}, 100.0, 50.0, 0.0);
    // Point on ellipse at theta = PI/4 (approximately)
    Point2d point(100.0 * std::cos(PI / 4), 50.0 * std::sin(PI / 4));

    DistanceResult result = DistancePointToEllipse(point, ellipse);

    EXPECT_LT(result.distance, 0.1);  // Should be very close to 0
}

TEST_F(PointToEllipseDistanceTest, RotatedEllipse) {
    // Ellipse rotated 45 degrees
    Ellipse2d ellipse({0.0, 0.0}, 100.0, 50.0, PI / 4);

    // Point along original major axis (now rotated)
    double angle = PI / 4;
    Point2d point(150.0 * std::cos(angle), 150.0 * std::sin(angle));

    DistanceResult result = DistancePointToEllipse(point, ellipse);

    EXPECT_NEAR(result.distance, 50.0, 0.1);  // ~50 pixels from ellipse edge
}

TEST_F(PointToEllipseDistanceTest, SignedDistanceOutside) {
    Ellipse2d ellipse({0.0, 0.0}, 100.0, 50.0, 0.0);
    Point2d point(150.0, 0.0);  // Outside on major axis

    SignedDistanceResult result = SignedDistancePointToEllipse(point, ellipse);

    EXPECT_GT(result.signedDistance, 0.0);  // Outside = positive
}

TEST_F(PointToEllipseDistanceTest, SignedDistanceInside) {
    Ellipse2d ellipse({0.0, 0.0}, 100.0, 50.0, 0.0);
    Point2d point(50.0, 0.0);  // Inside on major axis

    SignedDistanceResult result = SignedDistancePointToEllipse(point, ellipse);

    EXPECT_LT(result.signedDistance, 0.0);  // Inside = negative
}

// =============================================================================
// Point-to-Arc Distance Tests
// =============================================================================

class PointToArcDistanceTest : public ::testing::Test {};

TEST_F(PointToArcDistanceTest, PointOnArc) {
    // Arc from 0 to 90 degrees
    Arc2d arc({0.0, 0.0}, 10.0, 0.0, PI / 2);
    Point2d point(10.0, 0.0);  // At start point

    DistanceResult result = DistancePointToArc(point, arc);

    EXPECT_NEAR(result.distance, 0.0, 1e-12);
}

TEST_F(PointToArcDistanceTest, ClosestPointOnArc) {
    // Arc from 0 to 90 degrees
    Arc2d arc({0.0, 0.0}, 10.0, 0.0, PI / 2);
    Point2d point(15.0, 0.0);  // Along 0-degree direction, outside arc

    DistanceResult result = DistancePointToArc(point, arc);

    EXPECT_NEAR(result.distance, 5.0, 1e-12);
    EXPECT_TRUE(PointNearEqual(result.closestPoint, Point2d(10.0, 0.0), 1e-10));
}

TEST_F(PointToArcDistanceTest, ClosestPointAtStartEndpoint) {
    // Arc from 0 to 90 degrees
    Arc2d arc({0.0, 0.0}, 10.0, 0.0, PI / 2);
    Point2d point(15.0, -10.0);  // Below arc, closest to start

    DistanceResult result = DistancePointToArc(point, arc);

    // Distance to start point (10, 0)
    double expectedDist = std::sqrt(25.0 + 100.0);
    EXPECT_NEAR(result.distance, expectedDist, 1e-10);
    EXPECT_NEAR(result.parameter, -1.0, 1e-12);  // -1 = start endpoint
}

TEST_F(PointToArcDistanceTest, ClosestPointAtEndEndpoint) {
    // Arc from 0 to 90 degrees
    Arc2d arc({0.0, 0.0}, 10.0, 0.0, PI / 2);
    Point2d point(-10.0, 15.0);  // Beyond end, closest to end

    DistanceResult result = DistancePointToArc(point, arc);

    // Distance to end point (0, 10)
    double expectedDist = std::sqrt(100.0 + 25.0);
    EXPECT_NEAR(result.distance, expectedDist, 1e-10);
    EXPECT_NEAR(result.parameter, -2.0, 1e-12);  // -2 = end endpoint
}

TEST_F(PointToArcDistanceTest, PointAtCenter) {
    Arc2d arc({0.0, 0.0}, 10.0, 0.0, PI / 2);
    Point2d point(0.0, 0.0);

    DistanceResult result = DistancePointToArc(point, arc);

    // Distance should be radius (to closest point on arc)
    EXPECT_NEAR(result.distance, 10.0, 1e-10);
}

TEST_F(PointToArcDistanceTest, ZeroSweepArc) {
    Arc2d arc({0.0, 0.0}, 10.0, PI / 4, 0.0);  // Zero sweep
    Point2d point(10.0, 10.0);

    DistanceResult result = DistancePointToArc(point, arc);

    // Should be treated as single point at start
    Point2d startPt = arc.StartPoint();
    double expectedDist = point.DistanceTo(startPt);
    EXPECT_NEAR(result.distance, expectedDist, 1e-10);
}

// =============================================================================
// Point-to-RotatedRect Distance Tests
// =============================================================================

class PointToRotatedRectDistanceTest : public ::testing::Test {};

TEST_F(PointToRotatedRectDistanceTest, AxisAlignedRectInterior) {
    RotatedRect2d rect({50.0, 50.0}, 40.0, 20.0, 0.0);  // Center at (50,50), 40x20
    Point2d point(50.0, 50.0);  // At center (inside)

    DistanceResult result = DistancePointToRotatedRect(point, rect);

    // Distance to nearest edge (half-height = 10)
    EXPECT_NEAR(result.distance, 10.0, 1e-10);
}

TEST_F(PointToRotatedRectDistanceTest, AxisAlignedRectExterior) {
    RotatedRect2d rect({50.0, 50.0}, 40.0, 20.0, 0.0);
    Point2d point(100.0, 50.0);  // Outside to the right

    DistanceResult result = DistancePointToRotatedRect(point, rect);

    // Right edge at x = 70, distance = 30
    EXPECT_NEAR(result.distance, 30.0, 1e-10);
}

TEST_F(PointToRotatedRectDistanceTest, PointOnEdge) {
    RotatedRect2d rect({50.0, 50.0}, 40.0, 20.0, 0.0);
    Point2d point(70.0, 50.0);  // On right edge

    DistanceResult result = DistancePointToRotatedRect(point, rect);

    EXPECT_NEAR(result.distance, 0.0, 1e-10);
}

TEST_F(PointToRotatedRectDistanceTest, RotatedRect) {
    // Rectangle rotated 45 degrees
    RotatedRect2d rect({0.0, 0.0}, 20.0, 10.0, PI / 4);
    Point2d point(0.0, 0.0);  // At center

    DistanceResult result = DistancePointToRotatedRect(point, rect);

    // Distance to nearest edge (half of smaller dimension = 5)
    EXPECT_NEAR(result.distance, 5.0, 1e-10);
}

TEST_F(PointToRotatedRectDistanceTest, SignedDistanceInside) {
    RotatedRect2d rect({50.0, 50.0}, 40.0, 20.0, 0.0);
    Point2d point(50.0, 50.0);  // At center (inside)

    SignedDistanceResult result = SignedDistancePointToRotatedRect(point, rect);

    EXPECT_LT(result.signedDistance, 0.0);  // Inside = negative
    EXPECT_NEAR(result.Distance(), 10.0, 1e-10);
}

TEST_F(PointToRotatedRectDistanceTest, SignedDistanceOutside) {
    RotatedRect2d rect({50.0, 50.0}, 40.0, 20.0, 0.0);
    Point2d point(100.0, 50.0);  // Outside

    SignedDistanceResult result = SignedDistancePointToRotatedRect(point, rect);

    EXPECT_GT(result.signedDistance, 0.0);  // Outside = positive
    EXPECT_NEAR(result.signedDistance, 30.0, 1e-10);
}

// =============================================================================
// Line-to-Line Distance Tests
// =============================================================================

class LineToLineDistanceTest : public ::testing::Test {};

TEST_F(LineToLineDistanceTest, ParallelLines) {
    Line2d line1 = Line2d::FromPoints({0.0, 0.0}, {10.0, 0.0});  // Y = 0
    Line2d line2 = Line2d::FromPoints({0.0, 5.0}, {10.0, 5.0});  // Y = 5

    auto result = DistanceLineToLine(line1, line2);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result.value(), 5.0, 1e-10);
}

TEST_F(LineToLineDistanceTest, IntersectingLines) {
    Line2d line1 = Line2d::FromPoints({0.0, 0.0}, {10.0, 10.0});  // Y = X
    Line2d line2 = Line2d::FromPoints({0.0, 10.0}, {10.0, 0.0}); // Y = 10 - X

    auto result = DistanceLineToLine(line1, line2);

    EXPECT_FALSE(result.has_value());  // Intersecting lines
}

TEST_F(LineToLineDistanceTest, SameLines) {
    Line2d line1 = Line2d::FromPoints({0.0, 5.0}, {10.0, 5.0});
    Line2d line2 = Line2d::FromPoints({5.0, 5.0}, {15.0, 5.0});

    auto result = DistanceLineToLine(line1, line2);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result.value(), 0.0, 1e-10);
}

TEST_F(LineToLineDistanceTest, SignedDistanceParallel) {
    Line2d line1 = Line2d::FromPoints({0.0, 0.0}, {10.0, 0.0});
    Line2d line2 = Line2d::FromPoints({0.0, 5.0}, {10.0, 5.0});

    auto result = SignedDistanceLineToLine(line1, line2);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(std::abs(result.value()), 5.0, 1e-10);
}

// =============================================================================
// Segment-to-Segment Distance Tests
// =============================================================================

class SegmentToSegmentDistanceTest : public ::testing::Test {};

TEST_F(SegmentToSegmentDistanceTest, IntersectingSegments) {
    Segment2d seg1({0.0, 0.0}, {10.0, 10.0});
    Segment2d seg2({0.0, 10.0}, {10.0, 0.0});

    SegmentDistanceResult result = DistanceSegmentToSegment(seg1, seg2);

    EXPECT_NEAR(result.distance, 0.0, 1e-10);
    EXPECT_TRUE(result.Intersects());
}

TEST_F(SegmentToSegmentDistanceTest, ParallelNonIntersecting) {
    Segment2d seg1({0.0, 0.0}, {10.0, 0.0});
    Segment2d seg2({0.0, 5.0}, {10.0, 5.0});

    SegmentDistanceResult result = DistanceSegmentToSegment(seg1, seg2);

    EXPECT_NEAR(result.distance, 5.0, 1e-10);
}

TEST_F(SegmentToSegmentDistanceTest, EndpointClosest) {
    Segment2d seg1({0.0, 0.0}, {5.0, 0.0});
    Segment2d seg2({10.0, 0.0}, {15.0, 0.0});  // Gap between segments

    SegmentDistanceResult result = DistanceSegmentToSegment(seg1, seg2);

    EXPECT_NEAR(result.distance, 5.0, 1e-10);
    EXPECT_TRUE(PointNearEqual(result.closestPoint1, Point2d(5.0, 0.0), 1e-10));
    EXPECT_TRUE(PointNearEqual(result.closestPoint2, Point2d(10.0, 0.0), 1e-10));
}

TEST_F(SegmentToSegmentDistanceTest, PerpendicularSegments) {
    Segment2d seg1({0.0, 0.0}, {10.0, 0.0});
    Segment2d seg2({5.0, 3.0}, {5.0, 10.0});

    SegmentDistanceResult result = DistanceSegmentToSegment(seg1, seg2);

    EXPECT_NEAR(result.distance, 3.0, 1e-10);
}

TEST_F(SegmentToSegmentDistanceTest, CollinearOverlapping) {
    Segment2d seg1({0.0, 0.0}, {10.0, 0.0});
    Segment2d seg2({5.0, 0.0}, {15.0, 0.0});

    SegmentDistanceResult result = DistanceSegmentToSegment(seg1, seg2);

    EXPECT_NEAR(result.distance, 0.0, 1e-10);
}

// =============================================================================
// Circle-to-Circle Distance Tests
// =============================================================================

class CircleToCircleDistanceTest : public ::testing::Test {};

TEST_F(CircleToCircleDistanceTest, ExternallySeparated) {
    Circle2d c1({0.0, 0.0}, 10.0);
    Circle2d c2({30.0, 0.0}, 10.0);

    CircleDistanceResult result = DistanceCircleToCircle(c1, c2);

    EXPECT_NEAR(result.distance, 10.0, 1e-10);
    EXPECT_TRUE(result.AreSeparated());
    EXPECT_FALSE(result.OneContainsOther());
}

TEST_F(CircleToCircleDistanceTest, ExternallyTangent) {
    Circle2d c1({0.0, 0.0}, 10.0);
    Circle2d c2({20.0, 0.0}, 10.0);

    CircleDistanceResult result = DistanceCircleToCircle(c1, c2);

    EXPECT_NEAR(result.distance, 0.0, 1e-10);
    EXPECT_TRUE(result.AreExternallyTangent());
}

TEST_F(CircleToCircleDistanceTest, Overlapping) {
    Circle2d c1({0.0, 0.0}, 10.0);
    Circle2d c2({15.0, 0.0}, 10.0);

    CircleDistanceResult result = DistanceCircleToCircle(c1, c2);

    EXPECT_NEAR(result.distance, -5.0, 1e-10);  // Overlap by 5
    EXPECT_FALSE(result.AreSeparated());
}

TEST_F(CircleToCircleDistanceTest, OneContainsOther) {
    Circle2d c1({0.0, 0.0}, 50.0);
    Circle2d c2({10.0, 0.0}, 10.0);

    CircleDistanceResult result = DistanceCircleToCircle(c1, c2);

    EXPECT_LT(result.distance, 0.0);  // Negative (contained)
    EXPECT_TRUE(result.OneContainsOther());
}

TEST_F(CircleToCircleDistanceTest, ConcentricCircles) {
    Circle2d c1({0.0, 0.0}, 20.0);
    Circle2d c2({0.0, 0.0}, 10.0);

    CircleDistanceResult result = DistanceCircleToCircle(c1, c2);

    EXPECT_NEAR(result.distance, -10.0, 1e-10);
}

// =============================================================================
// Point-to-Contour Distance Tests
// =============================================================================

class PointToContourDistanceTest : public ::testing::Test {};

TEST_F(PointToContourDistanceTest, SimpleSquare) {
    // Square with corners at (0,0), (10,0), (10,10), (0,10)
    std::vector<Point2d> contour = {
        {0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}
    };

    Point2d point(5.0, -3.0);  // Outside bottom edge

    ContourDistanceResult result = DistancePointToContour(point, contour, true);

    EXPECT_NEAR(result.distance, 3.0, 1e-10);
    EXPECT_TRUE(result.IsValid());
}

TEST_F(PointToContourDistanceTest, OpenContour) {
    std::vector<Point2d> contour = {
        {0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}
    };

    Point2d point(5.0, 5.0);

    ContourDistanceResult result = DistancePointToContour(point, contour, false);

    EXPECT_TRUE(result.IsValid());
    EXPECT_GT(result.distance, 0.0);
}

TEST_F(PointToContourDistanceTest, EmptyContour) {
    std::vector<Point2d> contour;
    Point2d point(5.0, 5.0);

    ContourDistanceResult result = DistancePointToContour(point, contour, false);

    EXPECT_FALSE(result.IsValid());
}

TEST_F(PointToContourDistanceTest, SinglePointContour) {
    std::vector<Point2d> contour = {{5.0, 5.0}};
    Point2d point(8.0, 9.0);

    ContourDistanceResult result = DistancePointToContour(point, contour, false);

    EXPECT_TRUE(result.IsValid());
    EXPECT_NEAR(result.distance, 5.0, 1e-10);  // sqrt(9 + 16)
}

TEST_F(PointToContourDistanceTest, SignedDistanceInside) {
    // Square with corners at (0,0), (10,0), (10,10), (0,10)
    std::vector<Point2d> contour = {
        {0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}
    };

    Point2d point(5.0, 5.0);  // Center, inside

    SignedDistanceResult result = SignedDistancePointToContour(point, contour);

    EXPECT_LT(result.signedDistance, 0.0);  // Inside = negative
}

TEST_F(PointToContourDistanceTest, SignedDistanceOutside) {
    std::vector<Point2d> contour = {
        {0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}
    };

    Point2d point(5.0, -5.0);  // Outside below

    SignedDistanceResult result = SignedDistancePointToContour(point, contour);

    EXPECT_GT(result.signedDistance, 0.0);  // Outside = positive
}

// =============================================================================
// Batch Distance Function Tests
// =============================================================================

class BatchDistanceTest : public ::testing::Test {};

TEST_F(BatchDistanceTest, DistancePointsToLine) {
    Line2d line = Line2d::FromPoints({0.0, 0.0}, {10.0, 0.0});  // X-axis
    std::vector<Point2d> points = {
        {0.0, 1.0}, {5.0, 2.0}, {10.0, 3.0}
    };

    std::vector<double> distances = DistancePointsToLine(points, line);

    ASSERT_EQ(distances.size(), 3u);
    EXPECT_NEAR(distances[0], 1.0, 1e-10);
    EXPECT_NEAR(distances[1], 2.0, 1e-10);
    EXPECT_NEAR(distances[2], 3.0, 1e-10);
}

TEST_F(BatchDistanceTest, SignedDistancePointsToLine) {
    Line2d line = Line2d::FromPoints({0.0, 0.0}, {10.0, 0.0});
    std::vector<Point2d> points = {
        {0.0, 1.0}, {5.0, -2.0}
    };

    std::vector<double> distances = SignedDistancePointsToLine(points, line);

    ASSERT_EQ(distances.size(), 2u);
    EXPECT_NEAR(std::abs(distances[0]), 1.0, 1e-10);
    EXPECT_NEAR(std::abs(distances[1]), 2.0, 1e-10);
    // Opposite signs
    EXPECT_TRUE((distances[0] > 0) != (distances[1] > 0));
}

TEST_F(BatchDistanceTest, DistancePointsToCircle) {
    Circle2d circle({0.0, 0.0}, 10.0);
    std::vector<Point2d> points = {
        {15.0, 0.0}, {5.0, 0.0}, {0.0, 20.0}
    };

    std::vector<double> distances = DistancePointsToCircle(points, circle);

    ASSERT_EQ(distances.size(), 3u);
    EXPECT_NEAR(distances[0], 5.0, 1e-10);
    EXPECT_NEAR(distances[1], 5.0, 1e-10);
    EXPECT_NEAR(distances[2], 10.0, 1e-10);
}

TEST_F(BatchDistanceTest, SignedDistancePointsToCircle) {
    Circle2d circle({0.0, 0.0}, 10.0);
    std::vector<Point2d> points = {
        {15.0, 0.0}, {5.0, 0.0}
    };

    std::vector<double> distances = SignedDistancePointsToCircle(points, circle);

    ASSERT_EQ(distances.size(), 2u);
    EXPECT_NEAR(distances[0], 5.0, 1e-10);   // Outside = positive
    EXPECT_NEAR(distances[1], -5.0, 1e-10);  // Inside = negative
}

TEST_F(BatchDistanceTest, DistancePointsToContour) {
    std::vector<Point2d> contour = {
        {0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}
    };

    std::vector<Point2d> points = {
        {5.0, -3.0}, {-2.0, 5.0}
    };

    std::vector<double> distances = DistancePointsToContour(points, contour, true);

    ASSERT_EQ(distances.size(), 2u);
    EXPECT_NEAR(distances[0], 3.0, 1e-10);
    EXPECT_NEAR(distances[1], 2.0, 1e-10);
}

// =============================================================================
// Utility Function Tests
// =============================================================================

class DistanceUtilityTest : public ::testing::Test {};

TEST_F(DistanceUtilityTest, NearestPointOnContour) {
    std::vector<Point2d> contour = {
        {0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}
    };

    Point2d point(5.0, 5.0);

    auto result = NearestPointOnContour(point, contour, false);

    ASSERT_TRUE(result.has_value());
}

TEST_F(DistanceUtilityTest, NearestPointOnEmptyContour) {
    std::vector<Point2d> contour;
    Point2d point(5.0, 5.0);

    auto result = NearestPointOnContour(point, contour, false);

    EXPECT_FALSE(result.has_value());
}

TEST_F(DistanceUtilityTest, PointInsidePolygonSquare) {
    std::vector<Point2d> polygon = {
        {0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}
    };

    EXPECT_TRUE(PointInsidePolygon({5.0, 5.0}, polygon));
    EXPECT_TRUE(PointInsidePolygon({1.0, 1.0}, polygon));
    EXPECT_FALSE(PointInsidePolygon({-1.0, 5.0}, polygon));
    EXPECT_FALSE(PointInsidePolygon({11.0, 5.0}, polygon));
    EXPECT_FALSE(PointInsidePolygon({5.0, -1.0}, polygon));
    EXPECT_FALSE(PointInsidePolygon({5.0, 11.0}, polygon));
}

TEST_F(DistanceUtilityTest, PointInsidePolygonTriangle) {
    std::vector<Point2d> polygon = {
        {0.0, 0.0}, {10.0, 0.0}, {5.0, 10.0}
    };

    EXPECT_TRUE(PointInsidePolygon({5.0, 3.0}, polygon));
    EXPECT_FALSE(PointInsidePolygon({0.0, 10.0}, polygon));
}

TEST_F(DistanceUtilityTest, PointInsidePolygonConcave) {
    // L-shaped polygon
    std::vector<Point2d> polygon = {
        {0.0, 0.0}, {10.0, 0.0}, {10.0, 5.0}, {5.0, 5.0}, {5.0, 10.0}, {0.0, 10.0}
    };

    EXPECT_TRUE(PointInsidePolygon({2.0, 2.0}, polygon));  // In bottom-left
    EXPECT_TRUE(PointInsidePolygon({7.0, 2.0}, polygon));  // In bottom-right
    EXPECT_TRUE(PointInsidePolygon({2.0, 7.0}, polygon));  // In top-left
    EXPECT_FALSE(PointInsidePolygon({7.0, 7.0}, polygon)); // In concave part (outside)
}

TEST_F(DistanceUtilityTest, HausdorffDistanceSameContour) {
    std::vector<Point2d> contour = {
        {0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}
    };

    double dist = HausdorffDistance(contour, contour, true, true);

    EXPECT_NEAR(dist, 0.0, 1e-10);
}

TEST_F(DistanceUtilityTest, HausdorffDistanceTranslatedContour) {
    std::vector<Point2d> contour1 = {
        {0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}
    };

    std::vector<Point2d> contour2 = {
        {5.0, 0.0}, {15.0, 0.0}, {15.0, 10.0}, {5.0, 10.0}
    };

    double dist = HausdorffDistance(contour1, contour2, true, true);

    EXPECT_NEAR(dist, 5.0, 1e-10);  // Maximum distance from any point
}

TEST_F(DistanceUtilityTest, AverageDistanceContourToContour) {
    std::vector<Point2d> contour1 = {{0.0, 0.0}, {10.0, 0.0}};
    std::vector<Point2d> contour2 = {{0.0, 5.0}, {10.0, 5.0}};

    double avgDist = AverageDistanceContourToContour(contour1, contour2, false);

    EXPECT_NEAR(avgDist, 5.0, 1e-10);  // All points at distance 5
}

// =============================================================================
// Precision Tests
// =============================================================================

class DistancePrecisionTest : public ::testing::Test {};

TEST_F(DistancePrecisionTest, PointToLinePrecision) {
    // Test with known analytical solution
    Line2d line = Line2d::FromPoints({0.0, 0.0}, {100.0, 0.0});
    Point2d point(50.0, 10.0);

    DistanceResult result = DistancePointToLine(point, line);

    // Expect machine precision for simple cases
    EXPECT_NEAR(result.distance, 10.0, 1e-14);
    EXPECT_NEAR(result.closestPoint.x, 50.0, 1e-14);
    EXPECT_NEAR(result.closestPoint.y, 0.0, 1e-14);
}

TEST_F(DistancePrecisionTest, PointToSegmentPrecision) {
    Segment2d seg({0.0, 0.0}, {100.0, 0.0});
    Point2d point(50.0, 10.0);

    DistanceResult result = DistancePointToSegment(point, seg);

    EXPECT_NEAR(result.distance, 10.0, 1e-14);
    EXPECT_NEAR(result.parameter, 0.5, 1e-14);
}

TEST_F(DistancePrecisionTest, PointToCirclePrecision) {
    Circle2d circle({0.0, 0.0}, 100.0);
    Point2d point(150.0, 0.0);

    DistanceResult result = DistancePointToCircle(point, circle);

    EXPECT_NEAR(result.distance, 50.0, 1e-14);
    EXPECT_NEAR(result.closestPoint.x, 100.0, 1e-14);
    EXPECT_NEAR(result.closestPoint.y, 0.0, 1e-14);
}

TEST_F(DistancePrecisionTest, PointToEllipsePrecision) {
    Ellipse2d ellipse({0.0, 0.0}, 100.0, 50.0, 0.0);
    Point2d point(150.0, 0.0);

    DistanceResult result = DistancePointToEllipse(point, ellipse);

    // Ellipse distance uses Newton iteration, expect 1e-10 precision
    EXPECT_NEAR(result.distance, 50.0, 1e-10);
    EXPECT_NEAR(result.closestPoint.x, 100.0, 1e-8);
    EXPECT_NEAR(result.closestPoint.y, 0.0, 1e-8);
}

// =============================================================================
// Edge Cases and Degenerate Conditions
// =============================================================================

class DegenerateConditionsTest : public ::testing::Test {};

TEST_F(DegenerateConditionsTest, ZeroLengthSegment) {
    Segment2d seg({5.0, 5.0}, {5.0, 5.0});
    Point2d point(8.0, 9.0);

    DistanceResult result = DistancePointToSegment(point, seg);

    EXPECT_NEAR(result.distance, 5.0, 1e-10);
    EXPECT_TRUE(PointNearEqual(result.closestPoint, Point2d(5.0, 5.0), 1e-10));
}

TEST_F(DegenerateConditionsTest, ZeroRadiusCircle) {
    Circle2d circle({5.0, 5.0}, 0.0);
    Point2d point(8.0, 9.0);

    DistanceResult result = DistancePointToCircle(point, circle);

    EXPECT_NEAR(result.distance, 5.0, 1e-10);
}

TEST_F(DegenerateConditionsTest, DegenerateEllipse) {
    // Very small ellipse (nearly a point)
    Ellipse2d ellipse({5.0, 5.0}, 1e-15, 1e-15, 0.0);
    Point2d point(8.0, 9.0);

    DistanceResult result = DistancePointToEllipse(point, ellipse);

    // Should return distance to center
    EXPECT_NEAR(result.distance, 5.0, 1e-8);
}

TEST_F(DegenerateConditionsTest, ZeroSweepArc) {
    Arc2d arc({0.0, 0.0}, 10.0, PI / 4, 0.0);
    Point2d point(10.0, 10.0);

    DistanceResult result = DistancePointToArc(point, arc);

    // Distance to single point at start
    Point2d startPt = arc.StartPoint();
    double expectedDist = point.DistanceTo(startPt);
    EXPECT_NEAR(result.distance, expectedDist, 1e-10);
}

TEST_F(DegenerateConditionsTest, TwoPointContour) {
    std::vector<Point2d> contour = {{0.0, 0.0}, {10.0, 0.0}};
    Point2d point(5.0, 5.0);

    ContourDistanceResult result = DistancePointToContour(point, contour, false);

    EXPECT_TRUE(result.IsValid());
    EXPECT_NEAR(result.distance, 5.0, 1e-10);
}

} // anonymous namespace
} // namespace Qi::Vision::Internal
