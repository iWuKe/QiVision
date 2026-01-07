/**
 * @file test_intersection.cpp
 * @brief Unit tests for Internal/Intersection module
 */

#include <QiVision/Internal/Intersection.h>
#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace Qi::Vision::Internal {
namespace {

// =============================================================================
// Test Utilities
// =============================================================================

bool NearEqual(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

bool PointNearEqual(const Point2d& a, const Point2d& b, double tol = 1e-9) {
    return NearEqual(a.x, b.x, tol) && NearEqual(a.y, b.y, tol);
}

// =============================================================================
// Line-Line Intersection Tests
// =============================================================================

class LineLineIntersectionTest : public ::testing::Test {};

TEST_F(LineLineIntersectionTest, PerpendicularLines) {
    Line2d line1 = Line2d::FromPoints({0.0, 0.0}, {10.0, 0.0});
    Line2d line2 = Line2d::FromPoints({0.0, 0.0}, {0.0, 10.0});

    IntersectionResult result = IntersectLineLine(line1, line2);

    ASSERT_TRUE(result.exists);
    EXPECT_NEAR(result.point.x, 0.0, 1e-10);
    EXPECT_NEAR(result.point.y, 0.0, 1e-10);
}

TEST_F(LineLineIntersectionTest, DiagonalLines) {
    Line2d line1 = Line2d::FromPoints({0.0, 0.0}, {10.0, 10.0});
    Line2d line2 = Line2d::FromPoints({0.0, 10.0}, {10.0, 0.0});

    IntersectionResult result = IntersectLineLine(line1, line2);

    ASSERT_TRUE(result.exists);
    EXPECT_NEAR(result.point.x, 5.0, 1e-10);
    EXPECT_NEAR(result.point.y, 5.0, 1e-10);
}

TEST_F(LineLineIntersectionTest, ParallelLines_NoIntersection) {
    Line2d line1 = Line2d::FromPoints({0.0, 0.0}, {10.0, 0.0});
    Line2d line2 = Line2d::FromPoints({0.0, 5.0}, {10.0, 5.0});

    IntersectionResult result = IntersectLineLine(line1, line2);

    EXPECT_FALSE(result.exists);
}

TEST_F(LineLineIntersectionTest, CoincidentLines_NoIntersection) {
    Line2d line1 = Line2d::FromPoints({0.0, 0.0}, {10.0, 10.0});
    Line2d line2 = Line2d::FromPoints({5.0, 5.0}, {15.0, 15.0});

    IntersectionResult result = IntersectLineLine(line1, line2);

    EXPECT_FALSE(result.exists);
}

TEST_F(LineLineIntersectionTest, AreLinesCoincident_True) {
    Line2d line1 = Line2d::FromPoints({0.0, 0.0}, {10.0, 10.0});
    Line2d line2 = Line2d::FromPoints({5.0, 5.0}, {20.0, 20.0});

    EXPECT_TRUE(AreLinesCoincident(line1, line2));
}

TEST_F(LineLineIntersectionTest, AreLinesCoincident_False) {
    Line2d line1 = Line2d::FromPoints({0.0, 0.0}, {10.0, 10.0});
    Line2d line2 = Line2d::FromPoints({0.0, 1.0}, {10.0, 11.0});

    EXPECT_FALSE(AreLinesCoincident(line1, line2));
}

// =============================================================================
// Line-Segment Intersection Tests
// =============================================================================

class LineSegmentIntersectionTest : public ::testing::Test {};

TEST_F(LineSegmentIntersectionTest, IntersectsMiddle) {
    Line2d line = Line2d::FromPoints({0.0, 5.0}, {10.0, 5.0});
    Segment2d seg({5.0, 0.0}, {5.0, 10.0});

    IntersectionResult result = IntersectLineSegment(line, seg);

    ASSERT_TRUE(result.exists);
    EXPECT_NEAR(result.point.x, 5.0, 1e-10);
    EXPECT_NEAR(result.point.y, 5.0, 1e-10);
    EXPECT_NEAR(result.param2, 0.5, 1e-10);
}

TEST_F(LineSegmentIntersectionTest, NoIntersection_Parallel) {
    Line2d line = Line2d::FromPoints({0.0, 5.0}, {10.0, 5.0});
    Segment2d seg({0.0, 0.0}, {10.0, 0.0});

    IntersectionResult result = IntersectLineSegment(line, seg);

    EXPECT_FALSE(result.exists);
}

TEST_F(LineSegmentIntersectionTest, NoIntersection_MissesSegment) {
    Line2d line = Line2d::FromPoints({0.0, 15.0}, {10.0, 15.0});
    Segment2d seg({5.0, 0.0}, {5.0, 10.0});

    IntersectionResult result = IntersectLineSegment(line, seg);

    EXPECT_FALSE(result.exists);
}

// =============================================================================
// Segment-Segment Intersection Tests
// =============================================================================

class SegmentSegmentIntersectionTest : public ::testing::Test {};

TEST_F(SegmentSegmentIntersectionTest, CrossInMiddle) {
    Segment2d seg1({0.0, 0.0}, {10.0, 10.0});
    Segment2d seg2({0.0, 10.0}, {10.0, 0.0});

    IntersectionResult result = IntersectSegmentSegment(seg1, seg2);

    ASSERT_TRUE(result.exists);
    EXPECT_NEAR(result.point.x, 5.0, 1e-10);
    EXPECT_NEAR(result.point.y, 5.0, 1e-10);
}

TEST_F(SegmentSegmentIntersectionTest, TouchAtEndpoint) {
    Segment2d seg1({0.0, 0.0}, {5.0, 5.0});
    Segment2d seg2({5.0, 5.0}, {10.0, 0.0});

    IntersectionResult result = IntersectSegmentSegment(seg1, seg2);

    ASSERT_TRUE(result.exists);
    EXPECT_NEAR(result.point.x, 5.0, 1e-10);
    EXPECT_NEAR(result.point.y, 5.0, 1e-10);
}

TEST_F(SegmentSegmentIntersectionTest, ParallelNoIntersection) {
    Segment2d seg1({0.0, 0.0}, {10.0, 0.0});
    Segment2d seg2({0.0, 5.0}, {10.0, 5.0});

    IntersectionResult result = IntersectSegmentSegment(seg1, seg2);

    EXPECT_FALSE(result.exists);
}

TEST_F(SegmentSegmentIntersectionTest, CollinearNoOverlap) {
    Segment2d seg1({0.0, 0.0}, {5.0, 0.0});
    Segment2d seg2({6.0, 0.0}, {10.0, 0.0});

    IntersectionResult result = IntersectSegmentSegment(seg1, seg2);

    EXPECT_FALSE(result.exists);
}

// =============================================================================
// Line-Circle Intersection Tests
// =============================================================================

class LineCircleIntersectionTest : public ::testing::Test {};

TEST_F(LineCircleIntersectionTest, LineThroughCenter) {
    Line2d line = Line2d::FromPoints({-20.0, 0.0}, {20.0, 0.0});
    Circle2d circle({0.0, 0.0}, 10.0);

    IntersectionResult2 result = IntersectLineCircle(line, circle);

    EXPECT_TRUE(result.HasTwoIntersections());
    EXPECT_EQ(result.count, 2);
}

TEST_F(LineCircleIntersectionTest, LineTangent) {
    Line2d line = Line2d::FromPoints({-10.0, 10.0}, {10.0, 10.0});
    Circle2d circle({0.0, 0.0}, 10.0);

    IntersectionResult2 result = IntersectLineCircle(line, circle);

    EXPECT_TRUE(result.HasOneIntersection());
    EXPECT_EQ(result.count, 1);
    EXPECT_NEAR(result.point1.x, 0.0, 1e-10);
    EXPECT_NEAR(result.point1.y, 10.0, 1e-10);
}

TEST_F(LineCircleIntersectionTest, LineNoIntersection) {
    Line2d line = Line2d::FromPoints({-10.0, 15.0}, {10.0, 15.0});
    Circle2d circle({0.0, 0.0}, 10.0);

    IntersectionResult2 result = IntersectLineCircle(line, circle);

    EXPECT_FALSE(result.HasIntersection());
    EXPECT_EQ(result.count, 0);
}

// =============================================================================
// Segment-Circle Intersection Tests
// =============================================================================

class SegmentCircleIntersectionTest : public ::testing::Test {};

TEST_F(SegmentCircleIntersectionTest, SegmentThroughCenter) {
    Segment2d seg({-20.0, 0.0}, {20.0, 0.0});
    Circle2d circle({0.0, 0.0}, 10.0);

    IntersectionResult2 result = IntersectSegmentCircle(seg, circle);

    EXPECT_TRUE(result.HasTwoIntersections());
}

TEST_F(SegmentCircleIntersectionTest, SegmentPartiallyInside) {
    Segment2d seg({-5.0, 0.0}, {20.0, 0.0});
    Circle2d circle({0.0, 0.0}, 10.0);

    IntersectionResult2 result = IntersectSegmentCircle(seg, circle);

    EXPECT_EQ(result.count, 1);
    EXPECT_NEAR(result.point1.x, 10.0, 1e-10);
    EXPECT_NEAR(result.point1.y, 0.0, 1e-10);
}

TEST_F(SegmentCircleIntersectionTest, SegmentFullyInside) {
    Segment2d seg({-3.0, 0.0}, {3.0, 0.0});
    Circle2d circle({0.0, 0.0}, 10.0);

    IntersectionResult2 result = IntersectSegmentCircle(seg, circle);

    EXPECT_FALSE(result.HasIntersection());
}

TEST_F(SegmentCircleIntersectionTest, SegmentFullyOutside) {
    Segment2d seg({15.0, 0.0}, {20.0, 0.0});
    Circle2d circle({0.0, 0.0}, 10.0);

    IntersectionResult2 result = IntersectSegmentCircle(seg, circle);

    EXPECT_FALSE(result.HasIntersection());
}

// =============================================================================
// Line-Ellipse Intersection Tests
// =============================================================================

class LineEllipseIntersectionTest : public ::testing::Test {};

TEST_F(LineEllipseIntersectionTest, LineThroughCenter_MajorAxis) {
    Line2d line = Line2d::FromPoints({-50.0, 0.0}, {50.0, 0.0});
    Ellipse2d ellipse({0.0, 0.0}, 30.0, 20.0, 0.0);

    IntersectionResult2 result = IntersectLineEllipse(line, ellipse);

    EXPECT_TRUE(result.HasTwoIntersections());
}

TEST_F(LineEllipseIntersectionTest, LineNoIntersection) {
    Line2d line = Line2d::FromPoints({40.0, -10.0}, {40.0, 10.0});
    Ellipse2d ellipse({0.0, 0.0}, 30.0, 20.0, 0.0);

    IntersectionResult2 result = IntersectLineEllipse(line, ellipse);

    EXPECT_FALSE(result.HasIntersection());
}

// =============================================================================
// Circle-Circle Intersection Tests
// =============================================================================

class CircleCircleIntersectionTest : public ::testing::Test {};

TEST_F(CircleCircleIntersectionTest, TwoIntersections) {
    Circle2d c1({0.0, 0.0}, 10.0);
    Circle2d c2({15.0, 0.0}, 10.0);

    IntersectionResult2 result = IntersectCircleCircle(c1, c2);

    EXPECT_TRUE(result.HasTwoIntersections());

    double dist1_c1 = result.point1.DistanceTo(c1.center);
    double dist1_c2 = result.point1.DistanceTo(c2.center);
    EXPECT_NEAR(dist1_c1, 10.0, 1e-10);
    EXPECT_NEAR(dist1_c2, 10.0, 1e-10);
}

TEST_F(CircleCircleIntersectionTest, ExternallyTangent) {
    Circle2d c1({0.0, 0.0}, 10.0);
    Circle2d c2({20.0, 0.0}, 10.0);

    IntersectionResult2 result = IntersectCircleCircle(c1, c2);

    EXPECT_TRUE(result.HasOneIntersection());
    EXPECT_NEAR(result.point1.x, 10.0, 1e-10);
    EXPECT_NEAR(result.point1.y, 0.0, 1e-10);
}

TEST_F(CircleCircleIntersectionTest, NoIntersection_Separate) {
    Circle2d c1({0.0, 0.0}, 10.0);
    Circle2d c2({30.0, 0.0}, 10.0);

    IntersectionResult2 result = IntersectCircleCircle(c1, c2);

    EXPECT_FALSE(result.HasIntersection());
}

TEST_F(CircleCircleIntersectionTest, CircleRelation_AllCases) {
    Circle2d c1({0.0, 0.0}, 10.0);

    EXPECT_EQ(CircleRelation(c1, Circle2d({30.0, 0.0}, 10.0)), 0);  // Separate
    EXPECT_EQ(CircleRelation(c1, Circle2d({20.0, 0.0}, 10.0)), 1);  // External tangent
    EXPECT_EQ(CircleRelation(c1, Circle2d({15.0, 0.0}, 10.0)), 2);  // 2 intersections
    EXPECT_EQ(CircleRelation(c1, Circle2d({5.0, 0.0}, 5.0)), 3);    // Internal tangent
    EXPECT_EQ(CircleRelation(Circle2d({0.0, 0.0}, 50.0), Circle2d({0.0, 0.0}, 10.0)), -1); // contains
    EXPECT_EQ(CircleRelation(Circle2d({0.0, 0.0}, 10.0), Circle2d({0.0, 0.0}, 50.0)), -2); // contained
    EXPECT_EQ(CircleRelation(c1, Circle2d({0.0, 0.0}, 10.0)), 4);   // Coincident
}

// =============================================================================
// Line-Arc Intersection Tests
// =============================================================================

class LineArcIntersectionTest : public ::testing::Test {};

TEST_F(LineArcIntersectionTest, LineThroughArc) {
    Line2d line = Line2d::FromPoints({-20.0, 0.0}, {20.0, 0.0});
    Arc2d arc({0.0, 0.0}, 10.0, 0.0, PI);

    IntersectionResult2 result = IntersectLineArc(line, arc);

    EXPECT_TRUE(result.HasTwoIntersections());
}

TEST_F(LineArcIntersectionTest, LineNoIntersection_MissesArc) {
    Line2d line = Line2d::FromPoints({-20.0, -5.0}, {20.0, -5.0});
    Arc2d arc({0.0, 0.0}, 10.0, 0.0, PI);

    IntersectionResult2 result = IntersectLineArc(line, arc);

    EXPECT_FALSE(result.HasIntersection());
}

// =============================================================================
// Line-RotatedRect Intersection Tests
// =============================================================================

class LineRotatedRectIntersectionTest : public ::testing::Test {};

TEST_F(LineRotatedRectIntersectionTest, LineThroughCenter_Horizontal) {
    Line2d line = Line2d::FromPoints({-50.0, 0.0}, {50.0, 0.0});
    RotatedRect2d rect({0.0, 0.0}, 40.0, 20.0, 0.0);

    IntersectionResult2 result = IntersectLineRotatedRect(line, rect);

    EXPECT_TRUE(result.HasTwoIntersections());
}

TEST_F(LineRotatedRectIntersectionTest, LineNoIntersection) {
    Line2d line = Line2d::FromPoints({-50.0, 20.0}, {50.0, 20.0});
    RotatedRect2d rect({0.0, 0.0}, 40.0, 20.0, 0.0);

    IntersectionResult2 result = IntersectLineRotatedRect(line, rect);

    EXPECT_FALSE(result.HasIntersection());
}

// =============================================================================
// Segment Clipping Tests
// =============================================================================

class SegmentClippingTest : public ::testing::Test {};

TEST_F(SegmentClippingTest, ClipSegmentToRect_ThroughRect) {
    Segment2d seg({-50.0, 50.0}, {150.0, 50.0});
    Rect2d rect(0, 0, 100, 100);

    auto clipped = ClipSegmentToRect(seg, rect);

    ASSERT_TRUE(clipped.has_value());
    EXPECT_NEAR(clipped->p1.x, 0.0, 1e-8);
    EXPECT_NEAR(clipped->p2.x, 100.0, 1e-8);
}

TEST_F(SegmentClippingTest, ClipSegmentToRect_FullyOutside) {
    Segment2d seg({150.0, 50.0}, {200.0, 50.0});
    Rect2d rect(0, 0, 100, 100);

    auto clipped = ClipSegmentToRect(seg, rect);

    EXPECT_FALSE(clipped.has_value());
}

// =============================================================================
// Batch Intersection Tests
// =============================================================================

class BatchIntersectionTest : public ::testing::Test {};

TEST_F(BatchIntersectionTest, IntersectLineWithSegments) {
    Line2d line = Line2d::FromPoints({0.0, 5.0}, {10.0, 5.0});

    std::vector<Segment2d> segments = {
        {{0.0, 0.0}, {0.0, 10.0}},
        {{5.0, 0.0}, {5.0, 10.0}},
        {{10.0, 0.0}, {10.0, 10.0}},
        {{15.0, 0.0}, {15.0, 3.0}},
    };

    auto results = IntersectLineWithSegments(line, segments);

    ASSERT_EQ(results.size(), 4u);
    EXPECT_TRUE(results[0].exists);
    EXPECT_TRUE(results[1].exists);
    EXPECT_TRUE(results[2].exists);
    EXPECT_FALSE(results[3].exists);
}

// =============================================================================
// Ray-Contour Intersection Tests
// =============================================================================

class RayContourIntersectionTest : public ::testing::Test {};

TEST_F(RayContourIntersectionTest, PointInsideSquare) {
    std::vector<Point2d> contour = {
        {0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}
    };

    Point2d inside(5.0, 5.0);
    int count = CountRayContourIntersections(inside, contour);

    EXPECT_EQ(count % 2, 1);  // Odd = inside
}

TEST_F(RayContourIntersectionTest, PointOutsideSquare) {
    std::vector<Point2d> contour = {
        {0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}
    };

    Point2d outside(15.0, 5.0);
    int count = CountRayContourIntersections(outside, contour);

    EXPECT_EQ(count % 2, 0);  // Even = outside
}

// =============================================================================
// Precision Tests
// =============================================================================

class IntersectionPrecisionTest : public ::testing::Test {};

TEST_F(IntersectionPrecisionTest, LineLinePrecision) {
    Line2d line1 = Line2d::FromPoints({0.0, 0.0}, {100.0, 100.0});
    Line2d line2 = Line2d::FromPoints({0.0, 100.0}, {100.0, 0.0});

    IntersectionResult result = IntersectLineLine(line1, line2);

    ASSERT_TRUE(result.exists);
    EXPECT_NEAR(result.point.x, 50.0, 1e-12);
    EXPECT_NEAR(result.point.y, 50.0, 1e-12);
}

TEST_F(IntersectionPrecisionTest, CircleCirclePrecision) {
    Circle2d c1({0.0, 0.0}, 100.0);
    Circle2d c2({150.0, 0.0}, 100.0);

    IntersectionResult2 result = IntersectCircleCircle(c1, c2);

    ASSERT_TRUE(result.HasTwoIntersections());

    double dist1_c1 = result.point1.DistanceTo(c1.center);
    double dist1_c2 = result.point1.DistanceTo(c2.center);
    EXPECT_NEAR(dist1_c1, 100.0, 1e-10);
    EXPECT_NEAR(dist1_c2, 100.0, 1e-10);
}

} // anonymous namespace
} // namespace Qi::Vision::Internal
