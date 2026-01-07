/**
 * @file test_geom_relation.cpp
 * @brief Unit tests for GeomRelation module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/GeomRelation.h>
#include <QiVision/Internal/Intersection.h>
#include <QiVision/Core/Types.h>
#include <QiVision/Core/QContour.h>
#include <cmath>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

// =============================================================================
// Point-Circle Containment Tests
// =============================================================================

class PointCircleTest : public ::testing::Test {
protected:
    Circle2d circle{Point2d(0, 0), 10.0};
};

TEST_F(PointCircleTest, PointInsideCircle) {
    EXPECT_TRUE(PointInsideCircle(Point2d(0, 0), circle));
    EXPECT_TRUE(PointInsideCircle(Point2d(5, 0), circle));
    EXPECT_TRUE(PointInsideCircle(Point2d(0, 5), circle));
    EXPECT_TRUE(PointInsideCircle(Point2d(3, 4), circle));  // Distance = 5
}

TEST_F(PointCircleTest, PointOutsideCircle) {
    EXPECT_FALSE(PointInsideCircle(Point2d(10, 0), circle));  // On boundary
    EXPECT_FALSE(PointInsideCircle(Point2d(11, 0), circle));
    EXPECT_FALSE(PointInsideCircle(Point2d(8, 8), circle));   // Distance ~11.3
}

TEST_F(PointCircleTest, PointInsideOrOnCircle) {
    EXPECT_TRUE(PointInsideOrOnCircle(Point2d(0, 0), circle));
    EXPECT_TRUE(PointInsideOrOnCircle(Point2d(10, 0), circle));  // On boundary
    EXPECT_TRUE(PointInsideOrOnCircle(Point2d(0, 10), circle));  // On boundary
    EXPECT_FALSE(PointInsideOrOnCircle(Point2d(11, 0), circle));
}

// =============================================================================
// Point-Rectangle Containment Tests
// =============================================================================

class PointRectTest : public ::testing::Test {
protected:
    Rect2d rect{0, 0, 100, 50};  // x, y, width, height
};

TEST_F(PointRectTest, PointInsideRect) {
    EXPECT_TRUE(PointInsideRect(Point2d(50, 25), rect));  // Center
    EXPECT_TRUE(PointInsideRect(Point2d(1, 1), rect));
    EXPECT_TRUE(PointInsideRect(Point2d(99, 49), rect));
}

TEST_F(PointRectTest, PointOnRectBoundary) {
    EXPECT_FALSE(PointInsideRect(Point2d(0, 25), rect));   // On left edge
    EXPECT_FALSE(PointInsideRect(Point2d(100, 25), rect)); // On right edge
    EXPECT_FALSE(PointInsideRect(Point2d(50, 0), rect));   // On top edge
    EXPECT_FALSE(PointInsideRect(Point2d(50, 50), rect));  // On bottom edge
}

TEST_F(PointRectTest, PointOutsideRect) {
    EXPECT_FALSE(PointInsideRect(Point2d(-1, 25), rect));
    EXPECT_FALSE(PointInsideRect(Point2d(101, 25), rect));
    EXPECT_FALSE(PointInsideRect(Point2d(50, -1), rect));
    EXPECT_FALSE(PointInsideRect(Point2d(50, 51), rect));
}

TEST_F(PointRectTest, PointInsideOrOnRect) {
    EXPECT_TRUE(PointInsideOrOnRect(Point2d(50, 25), rect));   // Inside
    EXPECT_TRUE(PointInsideOrOnRect(Point2d(0, 0), rect));     // Corner
    EXPECT_TRUE(PointInsideOrOnRect(Point2d(100, 50), rect));  // Corner
    EXPECT_FALSE(PointInsideOrOnRect(Point2d(-1, 0), rect));   // Outside
}

// =============================================================================
// Point-RotatedRect Containment Tests
// =============================================================================

TEST(PointRotatedRectTest, AxisAlignedRect) {
    RotatedRect2d rect{Point2d(50, 25), 100, 50, 0};  // No rotation
    EXPECT_TRUE(PointInsideRotatedRect(Point2d(50, 25), rect));  // Center
    EXPECT_TRUE(PointInsideRotatedRect(Point2d(10, 10), rect));
    EXPECT_FALSE(PointInsideRotatedRect(Point2d(-10, 25), rect));
}

TEST(PointRotatedRectTest, Rotated45Degrees) {
    RotatedRect2d rect{Point2d(0, 0), 20, 10, M_PI/4};  // 45 degrees
    EXPECT_TRUE(PointInsideRotatedRect(Point2d(0, 0), rect));  // Center
    // Point along rotated major axis
    double d = 5 * std::cos(M_PI/4);
    EXPECT_TRUE(PointInsideRotatedRect(Point2d(d, d), rect));
}

// =============================================================================
// Point-Ellipse Containment Tests
// =============================================================================

TEST(PointEllipseTest, AxisAlignedEllipse) {
    Ellipse2d ellipse{Point2d(0, 0), 20, 10, 0};  // Semi-axes: 20, 10
    EXPECT_TRUE(PointInsideEllipse(Point2d(0, 0), ellipse));    // Center
    EXPECT_TRUE(PointInsideEllipse(Point2d(10, 0), ellipse));   // Inside (x/a=0.5 < 1)
    EXPECT_TRUE(PointInsideEllipse(Point2d(0, 5), ellipse));    // Inside (y/b=0.5 < 1)
    EXPECT_TRUE(PointInsideEllipse(Point2d(19, 0), ellipse));   // Inside (19/20=0.95, 0.95^2=0.9025 < 1)
    EXPECT_FALSE(PointInsideEllipse(Point2d(21, 0), ellipse));  // Outside (21/20 > 1)
    EXPECT_FALSE(PointInsideEllipse(Point2d(0, 11), ellipse));  // Outside (11/10 > 1)
}

TEST(PointEllipseTest, RotatedEllipse) {
    Ellipse2d ellipse{Point2d(0, 0), 20, 10, M_PI/2};  // Rotated 90 degrees
    EXPECT_TRUE(PointInsideEllipse(Point2d(0, 0), ellipse));
    EXPECT_TRUE(PointInsideEllipse(Point2d(0, 10), ellipse));
    EXPECT_FALSE(PointInsideEllipse(Point2d(15, 0), ellipse));  // Outside after rotation
}

// =============================================================================
// Point-Polygon Containment Tests
// =============================================================================

class PointPolygonTest : public ::testing::Test {
protected:
    // Square polygon: (0,0), (10,0), (10,10), (0,10)
    std::vector<Point2d> square{{0, 0}, {10, 0}, {10, 10}, {0, 10}};

    // Triangle: (0,0), (10,0), (5,10)
    std::vector<Point2d> triangle{{0, 0}, {10, 0}, {5, 10}};

    // Concave polygon (L-shape)
    std::vector<Point2d> lShape{{0, 0}, {10, 0}, {10, 5}, {5, 5}, {5, 10}, {0, 10}};
};

TEST_F(PointPolygonTest, PointInsideConvexPolygon) {
    EXPECT_TRUE(PointInsideConvexPolygon(Point2d(5, 5), square));
    EXPECT_TRUE(PointInsideConvexPolygon(Point2d(1, 1), square));
    EXPECT_FALSE(PointInsideConvexPolygon(Point2d(-1, 5), square));
    EXPECT_FALSE(PointInsideConvexPolygon(Point2d(11, 5), square));
}

TEST_F(PointPolygonTest, PointInPolygonInside) {
    auto result = PointInPolygon(Point2d(5, 5), square);
    EXPECT_EQ(result, PointPolygonRelation::Inside);

    result = PointInPolygon(Point2d(5, 3), triangle);
    EXPECT_EQ(result, PointPolygonRelation::Inside);
}

TEST_F(PointPolygonTest, PointInPolygonOutside) {
    auto result = PointInPolygon(Point2d(-1, 5), square);
    EXPECT_EQ(result, PointPolygonRelation::Outside);

    result = PointInPolygon(Point2d(5, 15), triangle);
    EXPECT_EQ(result, PointPolygonRelation::Outside);
}

TEST_F(PointPolygonTest, PointInPolygonOnBoundary) {
    auto result = PointInPolygon(Point2d(5, 0), square);
    EXPECT_EQ(result, PointPolygonRelation::OnBoundary);

    result = PointInPolygon(Point2d(0, 5), square);
    EXPECT_EQ(result, PointPolygonRelation::OnBoundary);
}

TEST_F(PointPolygonTest, PointInConcavePolygon) {
    // Inside L-shape
    auto result = PointInPolygon(Point2d(2, 2), lShape);
    EXPECT_EQ(result, PointPolygonRelation::Inside);

    // In the "cut-out" part - should be outside
    result = PointInPolygon(Point2d(7, 7), lShape);
    EXPECT_EQ(result, PointPolygonRelation::Outside);
}

TEST_F(PointPolygonTest, PointOnPolygonBoundary) {
    EXPECT_TRUE(PointOnPolygonBoundary(Point2d(5, 0), square));
    EXPECT_TRUE(PointOnPolygonBoundary(Point2d(0, 0), square));
    EXPECT_FALSE(PointOnPolygonBoundary(Point2d(5, 5), square));
}

// =============================================================================
// Circle-Circle Relationship Tests
// =============================================================================

class CircleCircleRelationTest : public ::testing::Test {
protected:
    Circle2d c1{Point2d(0, 0), 10};
};

TEST_F(CircleCircleRelationTest, SeparateCircles) {
    Circle2d c2{Point2d(25, 0), 5};  // Far apart
    EXPECT_EQ(GetCircleCircleRelation(c1, c2), CircleCircleRelation::Separate);
}

TEST_F(CircleCircleRelationTest, ExternalTangent) {
    Circle2d c2{Point2d(15, 0), 5};  // Distance = 15, radii sum = 15
    EXPECT_EQ(GetCircleCircleRelation(c1, c2, TANGENT_TOLERANCE), CircleCircleRelation::ExternalTangent);
}

TEST_F(CircleCircleRelationTest, Intersecting) {
    Circle2d c2{Point2d(12, 0), 5};  // Distance = 12, radii sum = 15, diff = 5
    EXPECT_EQ(GetCircleCircleRelation(c1, c2), CircleCircleRelation::Intersecting);
}

TEST_F(CircleCircleRelationTest, InternalTangent) {
    Circle2d c2{Point2d(5, 0), 5};  // Distance = 5, r1 - r2 = 5
    EXPECT_EQ(GetCircleCircleRelation(c1, c2, TANGENT_TOLERANCE), CircleCircleRelation::InternalTangent);
}

TEST_F(CircleCircleRelationTest, Containing) {
    Circle2d c2{Point2d(2, 0), 3};  // Completely inside c1
    EXPECT_EQ(GetCircleCircleRelation(c1, c2), CircleCircleRelation::Containing);
}

TEST_F(CircleCircleRelationTest, Coincident) {
    Circle2d c2{Point2d(0, 0), 10};  // Same circle
    EXPECT_EQ(GetCircleCircleRelation(c1, c2), CircleCircleRelation::Coincident);
}

TEST_F(CircleCircleRelationTest, CirclesAreTangent) {
    Circle2d external{Point2d(15, 0), 5};
    Circle2d internal{Point2d(5, 0), 5};
    Circle2d intersecting{Point2d(12, 0), 5};

    EXPECT_TRUE(CirclesAreTangent(c1, external));
    EXPECT_TRUE(CirclesAreTangent(c1, internal));
    EXPECT_FALSE(CirclesAreTangent(c1, intersecting));
}

TEST_F(CircleCircleRelationTest, CircleContainsCircle) {
    Circle2d inner{Point2d(2, 0), 3};
    Circle2d notContained{Point2d(8, 0), 5};

    EXPECT_TRUE(CircleContainsCircle(c1, inner));
    EXPECT_FALSE(CircleContainsCircle(c1, notContained));
}

// =============================================================================
// Line-Circle Relationship Tests
// =============================================================================

class LineCircleRelationTest : public ::testing::Test {
protected:
    Circle2d circle{Point2d(0, 0), 10};
};

TEST_F(LineCircleRelationTest, Secant) {
    Line2d line = Line2d::FromPoints(Point2d(-15, 0), Point2d(15, 0));  // Passes through center
    EXPECT_EQ(GetLineCircleRelation(line, circle), LineCircleRelation::Secant);
}

TEST_F(LineCircleRelationTest, Tangent) {
    Line2d line = Line2d::FromPoints(Point2d(-15, 10), Point2d(15, 10));  // Touches at (0, 10)
    EXPECT_EQ(GetLineCircleRelation(line, circle, TANGENT_TOLERANCE), LineCircleRelation::Tangent);
}

TEST_F(LineCircleRelationTest, Disjoint) {
    Line2d line = Line2d::FromPoints(Point2d(-15, 15), Point2d(15, 15));  // Above circle
    EXPECT_EQ(GetLineCircleRelation(line, circle), LineCircleRelation::Disjoint);
}

TEST_F(LineCircleRelationTest, LineIsTangentToCircle) {
    Line2d tangent = Line2d::FromPoints(Point2d(-15, 10), Point2d(15, 10));
    Line2d secant = Line2d::FromPoints(Point2d(-15, 0), Point2d(15, 0));

    EXPECT_TRUE(LineIsTangentToCircle(tangent, circle));
    EXPECT_FALSE(LineIsTangentToCircle(secant, circle));
}

TEST_F(LineCircleRelationTest, SegmentIsTangentToCircle) {
    Segment2d tangent{Point2d(-5, 10), Point2d(5, 10)};  // Short tangent segment
    Segment2d secant{Point2d(-15, 0), Point2d(15, 0)};

    EXPECT_TRUE(SegmentIsTangentToCircle(tangent, circle));
    EXPECT_FALSE(SegmentIsTangentToCircle(secant, circle));
}

// =============================================================================
// Segment-Segment Relationship Tests
// =============================================================================

class SegmentRelationTest : public ::testing::Test {
protected:
    Segment2d s1{Point2d(0, 0), Point2d(10, 0)};  // Horizontal segment
};

TEST_F(SegmentRelationTest, Parallel) {
    Segment2d s2{Point2d(0, 5), Point2d(10, 5)};  // Parallel, above
    EXPECT_EQ(GetSegmentRelation(s1, s2), SegmentRelation::Parallel);
}

TEST_F(SegmentRelationTest, Collinear) {
    Segment2d s2{Point2d(15, 0), Point2d(25, 0)};  // Same line, no overlap
    EXPECT_EQ(GetSegmentRelation(s1, s2), SegmentRelation::Collinear);
}

TEST_F(SegmentRelationTest, Overlapping) {
    Segment2d s2{Point2d(5, 0), Point2d(15, 0)};  // Overlaps with s1
    EXPECT_EQ(GetSegmentRelation(s1, s2), SegmentRelation::Overlapping);
}

TEST_F(SegmentRelationTest, Intersecting) {
    Segment2d s2{Point2d(5, -5), Point2d(5, 5)};  // Crosses at (5, 0)
    EXPECT_EQ(GetSegmentRelation(s1, s2), SegmentRelation::Intersecting);
}

TEST_F(SegmentRelationTest, Disjoint) {
    Segment2d s2{Point2d(5, 5), Point2d(5, 10)};  // Same direction, no intersection
    EXPECT_EQ(GetSegmentRelation(s1, s2), SegmentRelation::Disjoint);
}

TEST_F(SegmentRelationTest, SegmentsOverlap) {
    Segment2d overlapping{Point2d(5, 0), Point2d(15, 0)};
    Segment2d separate{Point2d(15, 0), Point2d(25, 0)};

    EXPECT_TRUE(SegmentsOverlap(s1, overlapping));
    EXPECT_FALSE(SegmentsOverlap(s1, separate));
}

TEST_F(SegmentRelationTest, SegmentsConnected) {
    Segment2d connected{Point2d(10, 0), Point2d(20, 0)};  // Shares endpoint
    Segment2d notConnected{Point2d(15, 0), Point2d(25, 0)};

    EXPECT_TRUE(SegmentsConnected(s1, connected));
    EXPECT_FALSE(SegmentsConnected(s1, notConnected));
}

TEST_F(SegmentRelationTest, SegmentsEqual) {
    Segment2d same{Point2d(0, 0), Point2d(10, 0)};
    Segment2d reversed{Point2d(10, 0), Point2d(0, 0)};
    Segment2d different{Point2d(0, 0), Point2d(10, 1)};

    EXPECT_TRUE(SegmentsEqual(s1, same));
    EXPECT_TRUE(SegmentsEqual(s1, reversed));
    EXPECT_FALSE(SegmentsEqual(s1, different));
}

// =============================================================================
// Rectangle Relationship Tests
// =============================================================================

class RectRelationTest : public ::testing::Test {
protected:
    Rect2d r1{0, 0, 100, 50};  // x, y, width, height
};

TEST_F(RectRelationTest, RectsOverlap) {
    Rect2d overlapping{50, 25, 100, 50};  // Overlaps
    Rect2d separate{200, 0, 50, 50};       // Far away
    Rect2d adjacent{100, 0, 50, 50};       // Touches edge

    EXPECT_TRUE(RectsOverlap(r1, overlapping));
    EXPECT_FALSE(RectsOverlap(r1, separate));
    // Note: Adjacent (touching) rectangles are considered overlapping by this implementation
    EXPECT_TRUE(RectsOverlap(r1, adjacent));
}

TEST_F(RectRelationTest, RectContainsRect) {
    Rect2d inner{10, 10, 30, 20};  // Inside r1
    Rect2d partial{50, 25, 100, 50};  // Partially inside

    EXPECT_TRUE(RectContainsRect(r1, inner));
    EXPECT_FALSE(RectContainsRect(r1, partial));
}

TEST_F(RectRelationTest, RectContainsPoint) {
    EXPECT_TRUE(RectContainsPoint(r1, Point2d(50, 25)));
    EXPECT_FALSE(RectContainsPoint(r1, Point2d(-1, 25)));
}

// =============================================================================
// Rotated Rectangle Relationship Tests
// =============================================================================

TEST(RotatedRectRelationTest, AxisAlignedOverlap) {
    RotatedRect2d r1{Point2d(50, 25), 100, 50, 0};
    RotatedRect2d r2{Point2d(100, 50), 100, 50, 0};
    RotatedRect2d r3{Point2d(200, 25), 50, 50, 0};

    EXPECT_TRUE(RotatedRectsOverlap(r1, r2));
    EXPECT_FALSE(RotatedRectsOverlap(r1, r3));
}

TEST(RotatedRectRelationTest, RotatedOverlap) {
    RotatedRect2d r1{Point2d(0, 0), 20, 10, 0};
    RotatedRect2d r2{Point2d(0, 0), 20, 10, M_PI/4};  // Same center, rotated

    EXPECT_TRUE(RotatedRectsOverlap(r1, r2));  // Should overlap at center
}

// =============================================================================
// Polygon Property Tests
// =============================================================================

class PolygonPropertyTest : public ::testing::Test {
protected:
    std::vector<Point2d> convexCCW{{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    std::vector<Point2d> convexCW{{0, 0}, {0, 10}, {10, 10}, {10, 0}};
    std::vector<Point2d> concave{{0, 0}, {10, 0}, {10, 10}, {5, 5}, {0, 10}};
};

TEST_F(PolygonPropertyTest, IsPolygonConvex) {
    EXPECT_TRUE(IsPolygonConvex(convexCCW));
    EXPECT_TRUE(IsPolygonConvex(convexCW));
    EXPECT_FALSE(IsPolygonConvex(concave));
}

TEST_F(PolygonPropertyTest, IsPolygonCCW) {
    EXPECT_TRUE(IsPolygonCCW(convexCCW));
    EXPECT_FALSE(IsPolygonCCW(convexCW));
}

// =============================================================================
// Polygon Relationship Tests
// =============================================================================

class PolygonRelationTest : public ::testing::Test {
protected:
    std::vector<Point2d> outer{{0, 0}, {100, 0}, {100, 100}, {0, 100}};
    std::vector<Point2d> inner{{20, 20}, {80, 20}, {80, 80}, {20, 80}};
    std::vector<Point2d> overlapping{{50, 50}, {150, 50}, {150, 150}, {50, 150}};
    std::vector<Point2d> separate{{200, 200}, {300, 200}, {300, 300}, {200, 300}};
};

TEST_F(PolygonRelationTest, PolygonsOverlap) {
    EXPECT_TRUE(PolygonsOverlap(outer, inner));
    EXPECT_TRUE(PolygonsOverlap(outer, overlapping));
    EXPECT_FALSE(PolygonsOverlap(outer, separate));
}

TEST_F(PolygonRelationTest, PolygonContainsPolygon) {
    EXPECT_TRUE(PolygonContainsPolygon(outer, inner));
    EXPECT_FALSE(PolygonContainsPolygon(outer, overlapping));
    EXPECT_FALSE(PolygonContainsPolygon(inner, outer));  // Reversed
}

// =============================================================================
// Angle Relationship Tests
// =============================================================================

TEST(AngleRelationTest, AnglesEqual) {
    EXPECT_TRUE(AnglesEqual(0, 0));
    EXPECT_TRUE(AnglesEqual(M_PI, M_PI));
    EXPECT_TRUE(AnglesEqual(0, 2*M_PI, 1e-6));  // Wrap-around
    EXPECT_TRUE(AnglesEqual(-M_PI, M_PI, 1e-6));  // -180 == 180
    EXPECT_FALSE(AnglesEqual(0, M_PI/2));
}

TEST(AngleRelationTest, AngleInRange) {
    // Range from 0 to PI/2 (CCW)
    EXPECT_TRUE(AngleInRange(M_PI/4, 0, M_PI/2));
    EXPECT_FALSE(AngleInRange(M_PI, 0, M_PI/2));

    // Range crossing 0 (from -PI/4 to PI/4)
    EXPECT_TRUE(AngleInRange(0, -M_PI/4, M_PI/4));
    EXPECT_TRUE(AngleInRange(M_PI/8, -M_PI/4, M_PI/4));
}

TEST(AngleRelationTest, AngleDifference) {
    // AngleDifference returns signed shortest path
    EXPECT_NEAR(AngleDifference(0, M_PI/2), -M_PI/2, 1e-10);  // 0 to PI/2 is -PI/2
    EXPECT_NEAR(AngleDifference(M_PI/2, 0), M_PI/2, 1e-10);   // PI/2 to 0 is +PI/2
    EXPECT_NEAR(std::abs(AngleDifference(0, M_PI)), M_PI, 1e-10);  // Both +/-PI are valid
    EXPECT_NEAR(AngleDifference(0.1, -0.1), 0.2, 1e-10);
    // Shortest path across 0
    EXPECT_NEAR(AngleDifference(-M_PI + 0.1, M_PI - 0.1), 0.2, 1e-10);
}

// =============================================================================
// Collinearity and Concyclicity Tests
// =============================================================================

TEST(AlignmentTest, PointsAreCollinear) {
    std::vector<Point2d> collinear{{0, 0}, {5, 5}, {10, 10}};
    std::vector<Point2d> notCollinear{{0, 0}, {5, 5}, {10, 11}};

    EXPECT_TRUE(PointsAreCollinear(collinear));
    EXPECT_FALSE(PointsAreCollinear(notCollinear));
}

TEST(AlignmentTest, PointsAreCollinearHorizontal) {
    std::vector<Point2d> horizontal{{0, 5}, {10, 5}, {20, 5}, {30, 5}};
    EXPECT_TRUE(PointsAreCollinear(horizontal));
}

TEST(AlignmentTest, PointsAreCollinearVertical) {
    std::vector<Point2d> vertical{{5, 0}, {5, 10}, {5, 20}};
    EXPECT_TRUE(PointsAreCollinear(vertical));
}

TEST(AlignmentTest, PointsAreConcyclic) {
    // Points on a circle of radius 10 centered at origin
    std::vector<Point2d> onCircle{
        {10, 0}, {0, 10}, {-10, 0}, {0, -10}
    };
    EXPECT_TRUE(PointsAreConcyclic(onCircle));

    // Points not on a circle
    std::vector<Point2d> notOnCircle{
        {10, 0}, {0, 10}, {-10, 0}, {1, -10}
    };
    EXPECT_FALSE(PointsAreConcyclic(notOnCircle));
}

TEST(AlignmentTest, ThreePointsAlwaysConcyclic) {
    // Any 3 non-collinear points are concyclic
    std::vector<Point2d> threePoints{{0, 0}, {10, 0}, {5, 5}};
    EXPECT_TRUE(PointsAreConcyclic(threePoints));
}

// =============================================================================
// Triangle Validity Tests
// =============================================================================

TEST(TriangleTest, ValidTriangle) {
    EXPECT_TRUE(IsValidTriangle(Point2d(0, 0), Point2d(10, 0), Point2d(5, 10)));
    EXPECT_TRUE(IsValidTriangle(Point2d(0, 0), Point2d(1, 0), Point2d(0.5, 0.1)));
}

TEST(TriangleTest, DegenerateTriangle) {
    // Collinear points - degenerate triangle
    EXPECT_FALSE(IsValidTriangle(Point2d(0, 0), Point2d(5, 0), Point2d(10, 0)));
    // Coincident points
    EXPECT_FALSE(IsValidTriangle(Point2d(0, 0), Point2d(0, 0), Point2d(10, 0)));
}

// =============================================================================
// Contour Tests
// =============================================================================

TEST(ContourTest, IsContourClosed) {
    std::vector<Point2d> closedPts = {{0, 0}, {10, 0}, {10, 10}, {0, 10}, {0, 0}};
    QContour closed(closedPts, true);
    EXPECT_TRUE(IsContourClosed(closed));

    std::vector<Point2d> openPts = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    QContour open(openPts, false);
    EXPECT_FALSE(IsContourClosed(open));
}

TEST(ContourTest, PointInsideContour) {
    std::vector<Point2d> pts = {{0, 0}, {10, 0}, {10, 10}, {0, 10}, {0, 0}};
    QContour contour(pts, true);

    EXPECT_TRUE(PointInsideContour(Point2d(5, 5), contour));
    EXPECT_FALSE(PointInsideContour(Point2d(-1, 5), contour));
    EXPECT_FALSE(PointInsideContour(Point2d(15, 5), contour));
}

// =============================================================================
// Edge Cases and Boundary Conditions
// =============================================================================

TEST(GeomRelationEdgeCaseTest, EmptyPolygon) {
    std::vector<Point2d> empty;
    EXPECT_FALSE(IsPolygonConvex(empty));
    // Empty set is trivially collinear (< 3 points returns true)
    EXPECT_TRUE(PointsAreCollinear(empty));
}

TEST(GeomRelationEdgeCaseTest, SinglePointPolygon) {
    std::vector<Point2d> single{{0, 0}};
    // Single point is trivially collinear
    EXPECT_TRUE(PointsAreCollinear(single));
}

TEST(GeomRelationEdgeCaseTest, TwoPointPolygon) {
    std::vector<Point2d> twoPoints{{0, 0}, {10, 10}};
    EXPECT_TRUE(PointsAreCollinear(twoPoints));
}

TEST(GeomRelationEdgeCaseTest, ZeroRadiusCircle) {
    Circle2d zeroCircle{Point2d(5, 5), 0};
    EXPECT_FALSE(PointInsideCircle(Point2d(5, 5), zeroCircle));
    EXPECT_TRUE(PointInsideOrOnCircle(Point2d(5, 5), zeroCircle));
}

TEST(GeomRelationEdgeCaseTest, CoincidentSegments) {
    Segment2d s1{Point2d(0, 0), Point2d(10, 10)};
    Segment2d s2{Point2d(0, 0), Point2d(10, 10)};
    EXPECT_TRUE(SegmentsEqual(s1, s2));
    EXPECT_EQ(GetSegmentRelation(s1, s2), SegmentRelation::Overlapping);
}
