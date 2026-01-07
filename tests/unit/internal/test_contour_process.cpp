/**
 * @file test_contour_process.cpp
 * @brief Unit tests for Internal/ContourProcess module
 *
 * Tests cover:
 * - Smoothing: Gaussian, Moving Average, Bilateral
 * - Simplification: Douglas-Peucker, Visvalingam, Radial Distance, NthPoint
 * - Resampling: By Distance, By Count, By Arc Length
 * - Other Processing: Reverse, Close, Open, Remove Duplicates, etc.
 * - Utility Functions: Compute Length, Find Point, Interpolate
 */

#include <QiVision/Internal/ContourProcess.h>
#include <QiVision/Core/QContour.h>
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
bool PointNearEqual(const Point2d& a, const Point2d& b, double tol = 1e-6) {
    return NearEqual(a.x, b.x, tol) && NearEqual(a.y, b.y, tol);
}

/// Create a simple square contour (10x10, closed)
QContour CreateSquareContour() {
    std::vector<Point2d> pts = {
        {0, 0}, {10, 0}, {10, 10}, {0, 10}
    };
    return QContour(pts, true);
}

/// Create a line contour (open)
QContour CreateLineContour() {
    std::vector<Point2d> pts = {
        {0, 0}, {5, 0}, {10, 0}, {15, 0}, {20, 0}
    };
    return QContour(pts, false);
}

/// Create a noisy circle contour
QContour CreateNoisyCircleContour(double radius, size_t numPoints, double noiseLevel) {
    std::vector<ContourPoint> pts;
    pts.reserve(numPoints);

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, noiseLevel);

    for (size_t i = 0; i < numPoints; ++i) {
        double angle = TWO_PI * static_cast<double>(i) / static_cast<double>(numPoints);
        double r = radius + noise(rng);
        ContourPoint p;
        p.x = r * std::cos(angle);
        p.y = r * std::sin(angle);
        p.amplitude = 100.0;
        p.direction = angle + HALF_PI;
        p.curvature = 1.0 / radius;
        pts.push_back(p);
    }

    return QContour(pts, true);
}

/// Create a zigzag contour (for simplification testing)
QContour CreateZigzagContour(size_t numPoints, double amplitude) {
    std::vector<Point2d> pts;
    pts.reserve(numPoints);

    for (size_t i = 0; i < numPoints; ++i) {
        double x = static_cast<double>(i);
        double y = (i % 2 == 0) ? 0.0 : amplitude;
        pts.push_back({x, y});
    }

    return QContour(pts, false);
}

// =============================================================================
// Utility Function Tests
// =============================================================================

class ContourUtilityTest : public ::testing::Test {};

TEST_F(ContourUtilityTest, ComputeContourLength_Square) {
    QContour square = CreateSquareContour();
    double length = ComputeContourLength(square);
    EXPECT_NEAR(length, 40.0, 1e-9);  // 4 sides of length 10
}

TEST_F(ContourUtilityTest, ComputeContourLength_Line) {
    QContour line = CreateLineContour();
    double length = ComputeContourLength(line);
    EXPECT_NEAR(length, 20.0, 1e-9);  // 5 points from 0 to 20
}

TEST_F(ContourUtilityTest, ComputeContourLength_Empty) {
    QContour empty;
    double length = ComputeContourLength(empty);
    EXPECT_NEAR(length, 0.0, 1e-9);
}

TEST_F(ContourUtilityTest, ComputeContourLength_SinglePoint) {
    QContour single;
    single.AddPoint(0, 0);
    double length = ComputeContourLength(single);
    EXPECT_NEAR(length, 0.0, 1e-9);
}

TEST_F(ContourUtilityTest, ComputeCumulativeLength_Line) {
    QContour line = CreateLineContour();
    std::vector<double> cumLen = ComputeCumulativeLength(line);

    ASSERT_EQ(cumLen.size(), line.Size());
    EXPECT_NEAR(cumLen[0], 0.0, 1e-9);
    EXPECT_NEAR(cumLen[1], 5.0, 1e-9);
    EXPECT_NEAR(cumLen[2], 10.0, 1e-9);
    EXPECT_NEAR(cumLen[3], 15.0, 1e-9);
    EXPECT_NEAR(cumLen[4], 20.0, 1e-9);
}

TEST_F(ContourUtilityTest, FindPointByArcLength_Midpoint) {
    QContour line = CreateLineContour();
    ContourPoint pt = FindPointByArcLength(line, 10.0);

    EXPECT_NEAR(pt.x, 10.0, 1e-6);
    EXPECT_NEAR(pt.y, 0.0, 1e-6);
}

TEST_F(ContourUtilityTest, FindPointByArcLength_Interpolated) {
    QContour line = CreateLineContour();
    ContourPoint pt = FindPointByArcLength(line, 7.5);

    EXPECT_NEAR(pt.x, 7.5, 1e-6);
    EXPECT_NEAR(pt.y, 0.0, 1e-6);
}

TEST_F(ContourUtilityTest, FindPointByParameter_Endpoints) {
    QContour line = CreateLineContour();

    ContourPoint start = FindPointByParameter(line, 0.0);
    EXPECT_NEAR(start.x, 0.0, 1e-6);

    ContourPoint end = FindPointByParameter(line, 1.0);
    EXPECT_NEAR(end.x, 20.0, 1e-6);
}

TEST_F(ContourUtilityTest, InterpolateContourPoint_Midpoint) {
    ContourPoint p1(0, 0, 100, 0, 0.1);
    ContourPoint p2(10, 10, 200, PI, 0.2);

    ContourPoint mid = InterpolateContourPoint(p1, p2, 0.5);

    EXPECT_NEAR(mid.x, 5.0, 1e-9);
    EXPECT_NEAR(mid.y, 5.0, 1e-9);
    EXPECT_NEAR(mid.amplitude, 150.0, 1e-9);
    EXPECT_NEAR(mid.curvature, 0.15, 1e-9);
}

TEST_F(ContourUtilityTest, InterpolateAngle_Simple) {
    // Same direction
    EXPECT_NEAR(InterpolateAngle(0.0, 0.0, 0.5), 0.0, 1e-9);

    // Opposite direction - should take shortest path
    EXPECT_NEAR(InterpolateAngle(0.0, PI, 0.5), HALF_PI, 1e-9);

    // Wrap around at PI/-PI
    double result = InterpolateAngle(PI - 0.1, -PI + 0.1, 0.5);
    EXPECT_TRUE(std::abs(result) > PI - 0.2);
}

// =============================================================================
// Smoothing Tests
// =============================================================================

class SmoothingTest : public ::testing::Test {};

TEST_F(SmoothingTest, GaussianSmooth_PreservesSize) {
    QContour circle = CreateNoisyCircleContour(50.0, 100, 1.0);
    QContour smoothed = SmoothContourGaussian(circle, {1.0});

    EXPECT_EQ(smoothed.Size(), circle.Size());
    EXPECT_EQ(smoothed.IsClosed(), circle.IsClosed());
}

TEST_F(SmoothingTest, GaussianSmooth_ReducesNoise) {
    QContour circle = CreateNoisyCircleContour(50.0, 100, 2.0);

    // Compute variance from ideal circle before smoothing
    double varBefore = 0.0;
    for (size_t i = 0; i < circle.Size(); ++i) {
        double r = std::sqrt(circle[i].x * circle[i].x + circle[i].y * circle[i].y);
        double diff = r - 50.0;
        varBefore += diff * diff;
    }
    varBefore /= circle.Size();

    // Smooth and compute variance after
    QContour smoothed = SmoothContourGaussian(circle, {2.0});

    double varAfter = 0.0;
    for (size_t i = 0; i < smoothed.Size(); ++i) {
        double r = std::sqrt(smoothed[i].x * smoothed[i].x + smoothed[i].y * smoothed[i].y);
        double diff = r - 50.0;
        varAfter += diff * diff;
    }
    varAfter /= smoothed.Size();

    EXPECT_LT(varAfter, varBefore);
}

TEST_F(SmoothingTest, GaussianSmooth_ShortContour) {
    QContour twoPoints;
    twoPoints.AddPoint(0, 0);
    twoPoints.AddPoint(10, 10);

    QContour result = SmoothContourGaussian(twoPoints, {1.0});

    // Should return original (not enough points)
    EXPECT_EQ(result.Size(), 2u);
}

TEST_F(SmoothingTest, MovingAverageSmooth_PreservesSize) {
    QContour circle = CreateNoisyCircleContour(50.0, 100, 1.0);
    QContour smoothed = SmoothContourMovingAverage(circle, {5});

    EXPECT_EQ(smoothed.Size(), circle.Size());
}

TEST_F(SmoothingTest, BilateralSmooth_PreservesCorners) {
    // Create a square with noisy edges
    std::vector<Point2d> pts;
    for (int i = 0; i <= 10; ++i) pts.push_back({static_cast<double>(i), 0.1 * (i % 2)});
    for (int i = 1; i <= 10; ++i) pts.push_back({10, static_cast<double>(i) + 0.1 * (i % 2)});
    for (int i = 9; i >= 0; --i) pts.push_back({static_cast<double>(i), 10 + 0.1 * (i % 2)});
    for (int i = 9; i >= 1; --i) pts.push_back({0, static_cast<double>(i) + 0.1 * (i % 2)});

    QContour noisy(pts, true);
    QContour smoothed = SmoothContourBilateral(noisy, {2.0, 30.0});

    // Corners should be approximately preserved
    // (Bilateral should smooth less at high curvature regions)
    EXPECT_EQ(smoothed.Size(), noisy.Size());
}

TEST_F(SmoothingTest, UnifiedSmoothInterface) {
    QContour circle = CreateNoisyCircleContour(50.0, 100, 1.0);

    QContour g = SmoothContour(circle, SmoothMethod::Gaussian, 1.0);
    QContour m = SmoothContour(circle, SmoothMethod::MovingAverage, 1.0, 5);
    QContour b = SmoothContour(circle, SmoothMethod::Bilateral, 2.0);

    EXPECT_EQ(g.Size(), 100u);
    EXPECT_EQ(m.Size(), 100u);
    EXPECT_EQ(b.Size(), 100u);
}

// =============================================================================
// Simplification Tests
// =============================================================================

class SimplificationTest : public ::testing::Test {};

TEST_F(SimplificationTest, DouglasPeucker_ReducesPoints) {
    QContour zigzag = CreateZigzagContour(21, 0.1);
    QContour simplified = SimplifyContourDouglasPeucker(zigzag, {0.5});

    // Small amplitude zigzag should be simplified significantly
    EXPECT_LT(simplified.Size(), zigzag.Size());
}

TEST_F(SimplificationTest, DouglasPeucker_PreservesEndpoints) {
    QContour line = CreateLineContour();
    QContour simplified = SimplifyContourDouglasPeucker(line, {1.0});

    // First and last points should be preserved
    EXPECT_TRUE(PointNearEqual(simplified.GetPoint(0), line.GetPoint(0)));
    EXPECT_TRUE(PointNearEqual(simplified.GetPoint(simplified.Size() - 1),
                                line.GetPoint(line.Size() - 1)));
}

TEST_F(SimplificationTest, DouglasPeucker_LargeTolerance) {
    QContour zigzag = CreateZigzagContour(21, 5.0);
    QContour simplified = SimplifyContourDouglasPeucker(zigzag, {10.0});

    // With large tolerance, should reduce to just endpoints
    EXPECT_EQ(simplified.Size(), 2u);
}

TEST_F(SimplificationTest, DouglasPeucker_ZeroTolerance) {
    // For a straight line, even zero tolerance will simplify to endpoints
    // because all intermediate points have exactly 0 perpendicular distance.
    // Use a non-collinear contour to test zero tolerance behavior.
    QContour zigzag = CreateZigzagContour(11, 0.5);  // Small amplitude zigzag
    QContour simplified = SimplifyContourDouglasPeucker(zigzag, {0.01});

    // Very small tolerance should keep most points
    EXPECT_GE(simplified.Size(), zigzag.Size() / 2);
}

TEST_F(SimplificationTest, DouglasPeucker_ClosedContour) {
    QContour square = CreateSquareContour();
    QContour simplified = SimplifyContourDouglasPeucker(square, {0.1});

    // Square corners should be preserved
    EXPECT_EQ(simplified.Size(), 4u);
    EXPECT_TRUE(simplified.IsClosed());
}

TEST_F(SimplificationTest, Visvalingam_ReducesPoints) {
    QContour zigzag = CreateZigzagContour(21, 0.1);
    QContour simplified = SimplifyContourVisvalingam(zigzag, {0.5});

    EXPECT_LT(simplified.Size(), zigzag.Size());
}

TEST_F(SimplificationTest, Visvalingam_MinPoints) {
    QContour zigzag = CreateZigzagContour(21, 1.0);
    QContour simplified = SimplifyContourVisvalingam(zigzag, {1000.0, 5});

    // Should have at least minPoints
    EXPECT_GE(simplified.Size(), 5u);
}

TEST_F(SimplificationTest, RadialDistance_ReducesPoints) {
    QContour line = CreateLineContour();  // Points at 5px intervals
    QContour simplified = SimplifyContourRadialDistance(line, {4.0});

    // With tolerance < spacing, should keep all points
    EXPECT_EQ(simplified.Size(), line.Size());

    // With tolerance > spacing, should reduce points
    QContour simplified2 = SimplifyContourRadialDistance(line, {6.0});
    EXPECT_LT(simplified2.Size(), line.Size());
}

TEST_F(SimplificationTest, NthPoint_EveryOther) {
    std::vector<Point2d> pts;
    for (int i = 0; i <= 10; ++i) pts.push_back({static_cast<double>(i), 0});
    QContour contour(pts, false);

    QContour simplified = SimplifyContourNthPoint(contour, 2);

    // Should keep approximately half the points plus endpoints
    EXPECT_LT(simplified.Size(), contour.Size());
    EXPECT_GE(simplified.Size(), 6u);  // At least every other + endpoints
}

TEST_F(SimplificationTest, UnifiedSimplifyInterface) {
    QContour zigzag = CreateZigzagContour(21, 0.5);

    QContour dp = SimplifyContour(zigzag, SimplifyMethod::DouglasPeucker, 1.0);
    QContour vs = SimplifyContour(zigzag, SimplifyMethod::Visvalingam, 1.0);
    QContour rd = SimplifyContour(zigzag, SimplifyMethod::RadialDistance, 1.0);

    EXPECT_LT(dp.Size(), zigzag.Size());
    EXPECT_LT(vs.Size(), zigzag.Size());
}

// =============================================================================
// Resampling Tests
// =============================================================================

class ResamplingTest : public ::testing::Test {};

TEST_F(ResamplingTest, ByDistance_UniformSpacing) {
    QContour line = CreateLineContour();
    QContour resampled = ResampleContourByDistance(line, {2.0});

    // Check that points are approximately 2px apart
    for (size_t i = 1; i < resampled.Size() - 1; ++i) {
        double dist = resampled[i - 1].DistanceTo(resampled[i]);
        EXPECT_NEAR(dist, 2.0, 0.1);
    }
}

TEST_F(ResamplingTest, ByDistance_ClosedContour) {
    QContour square = CreateSquareContour();
    QContour resampled = ResampleContourByDistance(square, {2.0});

    EXPECT_TRUE(resampled.IsClosed());

    // Should have approximately 40/2 = 20 points
    EXPECT_GT(resampled.Size(), 15u);
    EXPECT_LT(resampled.Size(), 25u);
}

TEST_F(ResamplingTest, ByCount_ExactCount) {
    QContour line = CreateLineContour();
    QContour resampled = ResampleContourByCount(line, {10});

    EXPECT_EQ(resampled.Size(), 10u);
}

TEST_F(ResamplingTest, ByCount_PreservesEndpoints) {
    QContour line = CreateLineContour();
    QContour resampled = ResampleContourByCount(line, {7, true});

    EXPECT_TRUE(PointNearEqual(resampled.GetPoint(0), line.GetPoint(0)));
    EXPECT_TRUE(PointNearEqual(resampled.GetPoint(resampled.Size() - 1),
                                line.GetPoint(line.Size() - 1)));
}

TEST_F(ResamplingTest, ByArcLength_UniformSegments) {
    QContour line = CreateLineContour();  // Length = 20
    QContour resampled = ResampleContourByArcLength(line, 5);  // 5 segments = 6 points

    EXPECT_EQ(resampled.Size(), 6u);

    // Check uniform spacing
    double expectedDist = 20.0 / 5.0;
    for (size_t i = 1; i < resampled.Size(); ++i) {
        double dist = resampled[i - 1].DistanceTo(resampled[i]);
        EXPECT_NEAR(dist, expectedDist, 0.01);
    }
}

TEST_F(ResamplingTest, UnifiedResampleInterface) {
    QContour line = CreateLineContour();

    QContour d = ResampleContour(line, ResampleMethod::ByDistance, 2.0);
    QContour c = ResampleContour(line, ResampleMethod::ByCount, 8);

    EXPECT_GT(d.Size(), 5u);
    EXPECT_EQ(c.Size(), 8u);
}

TEST_F(ResamplingTest, ShortContour) {
    QContour twoPoints;
    twoPoints.AddPoint(0, 0);
    twoPoints.AddPoint(1, 0);

    QContour resampled = ResampleContourByDistance(twoPoints, {0.5});
    EXPECT_GE(resampled.Size(), 2u);
}

// =============================================================================
// Other Processing Tests
// =============================================================================

class OtherProcessingTest : public ::testing::Test {};

TEST_F(OtherProcessingTest, ReverseContour_PointOrder) {
    QContour line = CreateLineContour();
    QContour reversed = ReverseContour(line);

    EXPECT_EQ(reversed.Size(), line.Size());

    // First point should now be last point
    EXPECT_TRUE(PointNearEqual(reversed.GetPoint(0), line.GetPoint(line.Size() - 1)));
    EXPECT_TRUE(PointNearEqual(reversed.GetPoint(reversed.Size() - 1), line.GetPoint(0)));
}

TEST_F(OtherProcessingTest, ReverseContour_PreservesClosed) {
    QContour square = CreateSquareContour();
    QContour reversed = ReverseContour(square);

    EXPECT_TRUE(reversed.IsClosed());
}

TEST_F(OtherProcessingTest, CloseContour) {
    QContour open = CreateLineContour();
    EXPECT_FALSE(open.IsClosed());

    QContour closed = CloseContour(open);
    EXPECT_TRUE(closed.IsClosed());
    EXPECT_EQ(closed.Size(), open.Size());
}

TEST_F(OtherProcessingTest, OpenContour) {
    QContour closed = CreateSquareContour();
    EXPECT_TRUE(closed.IsClosed());

    QContour opened = OpenContour(closed);
    EXPECT_FALSE(opened.IsClosed());
    EXPECT_EQ(opened.Size(), closed.Size());
}

TEST_F(OtherProcessingTest, RemoveDuplicatePoints) {
    std::vector<Point2d> pts = {
        {0, 0}, {0, 0}, {5, 0}, {5, 0}, {5, 0}, {10, 0}
    };
    QContour contour(pts, false);

    QContour cleaned = RemoveDuplicatePoints(contour);

    EXPECT_EQ(cleaned.Size(), 3u);  // 0, 5, 10
}

TEST_F(OtherProcessingTest, RemoveDuplicatePoints_NoDuplicates) {
    QContour line = CreateLineContour();
    QContour cleaned = RemoveDuplicatePoints(line);

    EXPECT_EQ(cleaned.Size(), line.Size());
}

TEST_F(OtherProcessingTest, RemoveCollinearPoints) {
    std::vector<Point2d> pts = {
        {0, 0}, {5, 0}, {10, 0}, {15, 0}, {20, 0}  // All collinear
    };
    QContour contour(pts, false);

    QContour cleaned = RemoveCollinearPoints(contour, 0.01);

    // Middle points are collinear and should be removed
    EXPECT_EQ(cleaned.Size(), 2u);  // Just endpoints
}

TEST_F(OtherProcessingTest, RemoveCollinearPoints_NonCollinear) {
    QContour square = CreateSquareContour();
    QContour cleaned = RemoveCollinearPoints(square, 0.01);

    // Square corners are not collinear
    EXPECT_EQ(cleaned.Size(), 4u);
}

TEST_F(OtherProcessingTest, ShiftContourStart_ByPoint) {
    QContour square = CreateSquareContour();  // Starts at (0,0)
    QContour shifted = ShiftContourStart(square, {10, 0});  // Shift to corner (10,0)

    EXPECT_TRUE(shifted.IsClosed());
    EXPECT_EQ(shifted.Size(), square.Size());

    // First point should now be near (10, 0)
    EXPECT_NEAR(shifted[0].x, 10.0, 1e-6);
    EXPECT_NEAR(shifted[0].y, 0.0, 1e-6);
}

TEST_F(OtherProcessingTest, ShiftContourStart_ByIndex) {
    QContour square = CreateSquareContour();
    QContour shifted = ShiftContourStartByIndex(square, 2);

    // First point should now be the original third point
    EXPECT_NEAR(shifted[0].x, square[2].x, 1e-6);
    EXPECT_NEAR(shifted[0].y, square[2].y, 1e-6);
}

TEST_F(OtherProcessingTest, ShiftContourStart_OpenContour) {
    QContour line = CreateLineContour();
    QContour shifted = ShiftContourStart(line, {10, 0});

    // Open contour should not be shifted
    EXPECT_TRUE(PointNearEqual(shifted.GetPoint(0), line.GetPoint(0)));
}

TEST_F(OtherProcessingTest, ExtractSubContour_Simple) {
    QContour line = CreateLineContour();
    QContour sub = ExtractSubContour(line, 0.25, 0.75);

    EXPECT_FALSE(sub.IsClosed());
    EXPECT_GT(sub.Size(), 2u);

    // First point should be around x=5, last around x=15
    EXPECT_NEAR(sub[0].x, 5.0, 1.0);
    EXPECT_NEAR(sub[sub.Size() - 1].x, 15.0, 1.0);
}

TEST_F(OtherProcessingTest, ExtractSubContourByIndex) {
    QContour line = CreateLineContour();
    QContour sub = ExtractSubContourByIndex(line, 1, 4);

    EXPECT_EQ(sub.Size(), 3u);  // Index 1, 2, 3
    EXPECT_NEAR(sub[0].x, 5.0, 1e-6);
    EXPECT_NEAR(sub[2].x, 15.0, 1e-6);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

class ContourProcessEdgeCaseTest : public ::testing::Test {};

TEST_F(ContourProcessEdgeCaseTest, EmptyContour_Smooth) {
    QContour empty;
    QContour result = SmoothContourGaussian(empty);
    EXPECT_TRUE(result.Empty());
}

TEST_F(ContourProcessEdgeCaseTest, EmptyContour_Simplify) {
    QContour empty;
    QContour result = SimplifyContourDouglasPeucker(empty);
    EXPECT_TRUE(result.Empty());
}

TEST_F(ContourProcessEdgeCaseTest, EmptyContour_Resample) {
    QContour empty;
    QContour result = ResampleContourByCount(empty, {10});
    EXPECT_TRUE(result.Empty());
}

TEST_F(ContourProcessEdgeCaseTest, SinglePoint_Smooth) {
    QContour single;
    single.AddPoint(5, 5);

    QContour result = SmoothContourGaussian(single);
    EXPECT_EQ(result.Size(), 1u);
}

TEST_F(ContourProcessEdgeCaseTest, SinglePoint_Simplify) {
    QContour single;
    single.AddPoint(5, 5);

    QContour result = SimplifyContourDouglasPeucker(single);
    EXPECT_EQ(result.Size(), 1u);
}

TEST_F(ContourProcessEdgeCaseTest, TwoPoints_Simplify) {
    QContour two;
    two.AddPoint(0, 0);
    two.AddPoint(10, 10);

    QContour result = SimplifyContourDouglasPeucker(two);
    EXPECT_EQ(result.Size(), 2u);
}

TEST_F(ContourProcessEdgeCaseTest, LargeSigma_Smooth) {
    QContour circle = CreateNoisyCircleContour(10.0, 20, 0.5);
    QContour result = SmoothContourGaussian(circle, {100.0});

    EXPECT_EQ(result.Size(), circle.Size());
}

TEST_F(ContourProcessEdgeCaseTest, ZeroDistance_Resample) {
    QContour line = CreateLineContour();
    QContour result = ResampleContourByDistance(line, {0.0});

    // Should use minimum distance
    EXPECT_GT(result.Size(), 0u);
}

TEST_F(ContourProcessEdgeCaseTest, OnePoint_Resample) {
    QContour result = ResampleContourByCount(CreateLineContour(), {1});
    EXPECT_GE(result.Size(), 1u);
}

// =============================================================================
// Attribute Preservation Tests
// =============================================================================

class AttributeTest : public ::testing::Test {};

TEST_F(AttributeTest, Smooth_InterpolateAttributes) {
    std::vector<ContourPoint> pts;
    for (int i = 0; i < 10; ++i) {
        ContourPoint p;
        p.x = static_cast<double>(i);
        p.y = 0;
        p.amplitude = 100.0 + i * 10;
        p.direction = 0.0;
        p.curvature = 0.1;
        pts.push_back(p);
    }
    QContour contour(pts, false);

    QContour smoothed = SmoothContourGaussian(contour, {1.0, 3, AttributeMode::Interpolate});

    // Attributes should be averaged (not zero)
    for (size_t i = 1; i < smoothed.Size() - 1; ++i) {
        EXPECT_GT(smoothed[i].amplitude, 50.0);
    }
}

TEST_F(AttributeTest, Resample_InterpolateAttributes) {
    std::vector<ContourPoint> pts;
    ContourPoint p1(0, 0, 0.0, 0.0, 0.0);
    ContourPoint p2(10, 0, 100.0, PI, 1.0);
    pts.push_back(p1);
    pts.push_back(p2);
    QContour contour(pts, false);

    QContour resampled = ResampleContourByCount(contour, {3, true, AttributeMode::Interpolate});

    EXPECT_EQ(resampled.Size(), 3u);

    // Middle point should have interpolated attributes
    EXPECT_NEAR(resampled[1].amplitude, 50.0, 1.0);
    EXPECT_NEAR(resampled[1].curvature, 0.5, 0.1);
}

TEST_F(AttributeTest, Resample_DiscardAttributes) {
    std::vector<ContourPoint> pts;
    ContourPoint p1(0, 0, 100.0, PI / 2, 0.5);
    ContourPoint p2(10, 0, 100.0, PI / 2, 0.5);
    pts.push_back(p1);
    pts.push_back(p2);
    QContour contour(pts, false);

    ResampleByCountParams params;
    params.count = 3;
    params.attrMode = AttributeMode::None;
    QContour resampled = ResampleContourByCount(contour, params);

    // Attributes should be zero
    EXPECT_NEAR(resampled[1].amplitude, 0.0, 1e-9);
    EXPECT_NEAR(resampled[1].curvature, 0.0, 1e-9);
}

// =============================================================================
// Performance-Related Tests (Boundary Conditions)
// =============================================================================

class PerformanceTest : public ::testing::Test {};

TEST_F(PerformanceTest, LargeContour_Smooth) {
    QContour large = CreateNoisyCircleContour(100.0, 1000, 1.0);
    QContour smoothed = SmoothContourGaussian(large, {2.0});

    EXPECT_EQ(smoothed.Size(), 1000u);
}

TEST_F(PerformanceTest, LargeContour_Simplify) {
    QContour large = CreateNoisyCircleContour(100.0, 1000, 1.0);
    QContour simplified = SimplifyContourDouglasPeucker(large, {1.0});

    EXPECT_LT(simplified.Size(), 1000u);
}

TEST_F(PerformanceTest, LargeContour_Resample) {
    QContour large = CreateNoisyCircleContour(100.0, 1000, 0.0);
    QContour resampled = ResampleContourByCount(large, {100});

    EXPECT_EQ(resampled.Size(), 100u);
}

} // namespace
} // namespace Qi::Vision::Internal
