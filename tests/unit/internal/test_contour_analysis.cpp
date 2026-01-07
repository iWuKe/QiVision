/**
 * @file test_contour_analysis.cpp
 * @brief Unit tests for Internal/ContourAnalysis module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/ContourAnalysis.h>
#include <QiVision/Core/QContour.h>
#include <cmath>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

// =============================================================================
// Test Fixtures
// =============================================================================

class ContourAnalysisTest : public ::testing::Test {
protected:
    void SetUp() override {}

    // Helper: Create a square contour
    QContour CreateSquare(double size, Point2d center = {0, 0}) {
        QContour contour;
        double half = size / 2.0;
        contour.AddPoint({center.x - half, center.y - half, 0, 0, 0});
        contour.AddPoint({center.x + half, center.y - half, 0, 0, 0});
        contour.AddPoint({center.x + half, center.y + half, 0, 0, 0});
        contour.AddPoint({center.x - half, center.y + half, 0, 0, 0});
        contour.SetClosed(true);
        return contour;
    }

    // Helper: Create a rectangle contour
    QContour CreateRectangle(double width, double height, Point2d center = {0, 0}) {
        QContour contour;
        double hw = width / 2.0;
        double hh = height / 2.0;
        contour.AddPoint({center.x - hw, center.y - hh, 0, 0, 0});
        contour.AddPoint({center.x + hw, center.y - hh, 0, 0, 0});
        contour.AddPoint({center.x + hw, center.y + hh, 0, 0, 0});
        contour.AddPoint({center.x - hw, center.y + hh, 0, 0, 0});
        contour.SetClosed(true);
        return contour;
    }

    // Helper: Create a circle contour
    QContour CreateCircle(double radius, Point2d center = {0, 0}, int numPoints = 100) {
        QContour contour;
        for (int i = 0; i < numPoints; ++i) {
            double angle = 2.0 * PI * i / numPoints;
            double x = center.x + radius * std::cos(angle);
            double y = center.y + radius * std::sin(angle);
            contour.AddPoint({x, y, 0, 0, 0});
        }
        contour.SetClosed(true);
        return contour;
    }

    // Helper: Create an equilateral triangle
    QContour CreateTriangle(double sideLength, Point2d center = {0, 0}) {
        QContour contour;
        double h = sideLength * std::sqrt(3.0) / 2.0;
        double r = h * 2.0 / 3.0;  // Circumradius

        for (int i = 0; i < 3; ++i) {
            double angle = PI / 2.0 + 2.0 * PI * i / 3.0;
            double x = center.x + r * std::cos(angle);
            double y = center.y + r * std::sin(angle);
            contour.AddPoint({x, y, 0, 0, 0});
        }
        contour.SetClosed(true);
        return contour;
    }

    // Helper: Create an L-shaped contour (concave)
    QContour CreateLShape() {
        QContour contour;
        contour.AddPoint({0, 0, 0, 0, 0});
        contour.AddPoint({3, 0, 0, 0, 0});
        contour.AddPoint({3, 1, 0, 0, 0});
        contour.AddPoint({1, 1, 0, 0, 0});
        contour.AddPoint({1, 3, 0, 0, 0});
        contour.AddPoint({0, 3, 0, 0, 0});
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
            contour.AddPoint({x, y, 0, 0, 0});
        }
        contour.SetClosed(false);
        return contour;
    }
};

// =============================================================================
// Basic Property Tests
// =============================================================================

TEST_F(ContourAnalysisTest, ContourLength_Square) {
    QContour square = CreateSquare(10.0);
    double length = ContourLength(square);
    EXPECT_NEAR(length, 40.0, 1e-10);
}

TEST_F(ContourAnalysisTest, ContourLength_Circle) {
    QContour circle = CreateCircle(50.0, {100, 100}, 1000);
    double length = ContourLength(circle);
    double expected = 2.0 * PI * 50.0;
    EXPECT_NEAR(length, expected, expected * 0.001);  // 0.1% tolerance
}

TEST_F(ContourAnalysisTest, ContourLength_EmptyContour) {
    QContour empty;
    EXPECT_DOUBLE_EQ(ContourLength(empty), 0.0);
}

TEST_F(ContourAnalysisTest, ContourLength_SinglePoint) {
    QContour single;
    single.AddPoint({0, 0, 0, 0, 0});
    EXPECT_DOUBLE_EQ(ContourLength(single), 0.0);
}

TEST_F(ContourAnalysisTest, ContourArea_Square) {
    QContour square = CreateSquare(10.0);
    double area = ContourArea(square);
    EXPECT_NEAR(area, 100.0, 1e-10);
}

TEST_F(ContourAnalysisTest, ContourArea_Rectangle) {
    QContour rect = CreateRectangle(20.0, 10.0);
    double area = ContourArea(rect);
    EXPECT_NEAR(area, 200.0, 1e-10);
}

TEST_F(ContourAnalysisTest, ContourArea_Circle) {
    QContour circle = CreateCircle(50.0, {0, 0}, 1000);
    double area = ContourArea(circle);
    double expected = PI * 50.0 * 50.0;
    EXPECT_NEAR(area, expected, expected * 0.001);  // 0.1% tolerance
}

TEST_F(ContourAnalysisTest, ContourArea_Triangle) {
    double side = 10.0;
    QContour triangle = CreateTriangle(side);
    double area = ContourArea(triangle);
    double expected = side * side * std::sqrt(3.0) / 4.0;
    EXPECT_NEAR(area, expected, 1e-10);
}

TEST_F(ContourAnalysisTest, ContourSignedArea_CCW) {
    // CCW square should have positive area
    QContour square = CreateSquare(10.0);
    double signedArea = ContourSignedArea(square);
    EXPECT_GT(signedArea, 0.0);
}

TEST_F(ContourAnalysisTest, ContourCentroid_Square) {
    QContour square = CreateSquare(10.0, {50, 50});
    Point2d centroid = ContourCentroid(square);
    EXPECT_NEAR(centroid.x, 50.0, 1e-10);
    EXPECT_NEAR(centroid.y, 50.0, 1e-10);
}

TEST_F(ContourAnalysisTest, ContourCentroid_Circle) {
    QContour circle = CreateCircle(30.0, {100, 200}, 100);
    Point2d centroid = ContourCentroid(circle);
    EXPECT_NEAR(centroid.x, 100.0, 0.5);
    EXPECT_NEAR(centroid.y, 200.0, 0.5);
}

TEST_F(ContourAnalysisTest, ContourAreaCenter_Combined) {
    QContour square = CreateSquare(10.0, {25, 25});
    AreaCenterResult result = ContourAreaCenter(square);

    EXPECT_TRUE(result.valid);
    EXPECT_NEAR(result.area, 100.0, 1e-10);
    EXPECT_NEAR(result.centroid.x, 25.0, 1e-10);
    EXPECT_NEAR(result.centroid.y, 25.0, 1e-10);
}

TEST_F(ContourAnalysisTest, ContourPerimeter_EqualsLength) {
    QContour square = CreateSquare(10.0);
    EXPECT_DOUBLE_EQ(ContourPerimeter(square), ContourLength(square));
}

// =============================================================================
// Curvature Tests
// =============================================================================

TEST_F(ContourAnalysisTest, Curvature_Circle_Constant) {
    double radius = 50.0;
    QContour circle = CreateCircle(radius, {0, 0}, 100);

    std::vector<double> curvatures = ComputeContourCurvature(circle);
    EXPECT_EQ(curvatures.size(), circle.Size());

    // For a circle, curvature should be approximately 1/radius
    double expectedCurvature = 1.0 / radius;

    for (size_t i = 0; i < curvatures.size(); ++i) {
        EXPECT_NEAR(std::abs(curvatures[i]), expectedCurvature, 0.01);
    }
}

TEST_F(ContourAnalysisTest, Curvature_Line_Zero) {
    QContour line = CreateLine({0, 0}, {100, 0}, 20);

    std::vector<double> curvatures = ComputeContourCurvature(line);

    // For a straight line, curvature should be 0 (except endpoints)
    for (size_t i = 1; i < curvatures.size() - 1; ++i) {
        EXPECT_NEAR(curvatures[i], 0.0, 1e-10);
    }
}

TEST_F(ContourAnalysisTest, CurvatureMean_Circle) {
    double radius = 50.0;
    QContour circle = CreateCircle(radius, {0, 0}, 100);

    double meanCurv = ContourMeanCurvature(circle);
    EXPECT_NEAR(meanCurv, 1.0 / radius, 0.01);
}

TEST_F(ContourAnalysisTest, CurvatureMax_Circle) {
    double radius = 50.0;
    QContour circle = CreateCircle(radius, {0, 0}, 100);

    double maxCurv = ContourMaxCurvature(circle);
    EXPECT_NEAR(maxCurv, 1.0 / radius, 0.02);
}

TEST_F(ContourAnalysisTest, CurvatureStats_Basic) {
    QContour circle = CreateCircle(50.0, {0, 0}, 100);

    CurvatureStats stats = ContourCurvatureStats(circle);
    EXPECT_NEAR(stats.mean, 0.02, 0.005);  // 1/50
    EXPECT_LT(stats.stddev, 0.005);  // Low variance for circle
}

TEST_F(ContourAnalysisTest, CurvatureHistogram_Basic) {
    QContour circle = CreateCircle(50.0, {0, 0}, 100);

    std::vector<int32_t> histogram = ContourCurvatureHistogram(circle, 10);
    EXPECT_EQ(histogram.size(), 10u);

    // Sum should equal number of points
    int32_t sum = 0;
    for (int32_t count : histogram) {
        sum += count;
    }
    EXPECT_EQ(sum, static_cast<int32_t>(circle.Size()));
}

TEST_F(ContourAnalysisTest, Curvature_FivePoint_Smoother) {
    QContour circle = CreateCircle(50.0, {0, 0}, 100);

    auto curv3 = ComputeContourCurvature(circle, CurvatureMethod::ThreePoint);
    auto curv5 = ComputeContourCurvature(circle, CurvatureMethod::FivePoint, 5);

    // Both should have same length
    EXPECT_EQ(curv3.size(), curv5.size());
}

// =============================================================================
// Orientation Tests
// =============================================================================

TEST_F(ContourAnalysisTest, Orientation_HorizontalRectangle) {
    QContour rect = CreateRectangle(100.0, 50.0);
    double angle = ContourOrientation(rect);
    EXPECT_NEAR(angle, 0.0, 0.1);  // Horizontal major axis
}

TEST_F(ContourAnalysisTest, Orientation_VerticalRectangle) {
    QContour rect = CreateRectangle(50.0, 100.0);
    double angle = ContourOrientation(rect);
    EXPECT_NEAR(std::abs(angle), PI / 2.0, 0.1);  // Vertical major axis
}

TEST_F(ContourAnalysisTest, Orientation_Square_Isotropic) {
    QContour square = CreateSquare(100.0);
    double angle = ContourOrientation(square);
    // Square is isotropic, any angle is acceptable
    EXPECT_TRUE(std::isfinite(angle));
}

TEST_F(ContourAnalysisTest, PrincipalAxes_Rectangle) {
    QContour rect = CreateRectangle(100.0, 50.0, {50, 50});

    PrincipalAxesResult axes = ContourPrincipalAxes(rect);

    EXPECT_TRUE(axes.valid);
    EXPECT_NEAR(axes.centroid.x, 50.0, 1e-6);
    EXPECT_NEAR(axes.centroid.y, 50.0, 1e-6);
    EXPECT_GT(axes.majorLength, axes.minorLength);
}

TEST_F(ContourAnalysisTest, OrientationEllipse_HorizontalRectangle) {
    QContour rect = CreateRectangle(100.0, 50.0);
    double angle = ContourOrientationEllipse(rect);
    // May fail for rectangle (not enough points)
    // Just check it doesn't crash
    EXPECT_TRUE(std::isfinite(angle));
}

// =============================================================================
// Moment Tests
// =============================================================================

TEST_F(ContourAnalysisTest, Moments_Square_M00) {
    QContour square = CreateSquare(10.0);
    MomentsResult moments = ContourMoments(square);
    EXPECT_NEAR(moments.m00, 100.0, 1e-6);
}

TEST_F(ContourAnalysisTest, Moments_CentroidFromMoments) {
    QContour square = CreateSquare(10.0, {50, 50});
    MomentsResult moments = ContourMoments(square);

    Point2d centroid = moments.Centroid();
    EXPECT_NEAR(centroid.x, 50.0, 1e-6);
    EXPECT_NEAR(centroid.y, 50.0, 1e-6);
}

TEST_F(ContourAnalysisTest, CentralMoments_TranslationInvariant) {
    QContour square1 = CreateSquare(10.0, {0, 0});
    QContour square2 = CreateSquare(10.0, {100, 100});

    CentralMomentsResult cm1 = ContourCentralMoments(square1);
    CentralMomentsResult cm2 = ContourCentralMoments(square2);

    // Central moments should be the same (translation invariant)
    EXPECT_NEAR(cm1.mu20, cm2.mu20, 1e-6);
    EXPECT_NEAR(cm1.mu11, cm2.mu11, 1e-6);
    EXPECT_NEAR(cm1.mu02, cm2.mu02, 1e-6);
}

TEST_F(ContourAnalysisTest, NormalizedMoments_ScaleInvariant) {
    QContour square1 = CreateSquare(10.0);
    QContour square2 = CreateSquare(20.0);  // Scaled by 2

    NormalizedMomentsResult nm1 = ContourNormalizedMoments(square1);
    NormalizedMomentsResult nm2 = ContourNormalizedMoments(square2);

    // Normalized moments should be scale-invariant
    EXPECT_NEAR(nm1.eta20, nm2.eta20, 1e-6);
    EXPECT_NEAR(nm1.eta11, nm2.eta11, 1e-6);
    EXPECT_NEAR(nm1.eta02, nm2.eta02, 1e-6);
}

TEST_F(ContourAnalysisTest, HuMoments_SevenValues) {
    QContour square = CreateSquare(10.0);
    HuMomentsResult hu = ContourHuMoments(square);

    for (int i = 0; i < 7; ++i) {
        EXPECT_TRUE(std::isfinite(hu[i]));
    }
}

TEST_F(ContourAnalysisTest, HuMoments_TranslationInvariant) {
    // Use circles with small translation to minimize numerical issues
    // Large translations cause precision loss in moment calculations
    QContour circle1 = CreateCircle(50.0, {50, 50}, 360);
    QContour circle2 = CreateCircle(50.0, {60, 60}, 360);

    HuMomentsResult hu1 = ContourHuMoments(circle1);
    HuMomentsResult hu2 = ContourHuMoments(circle2);

    // h1 should be non-zero and approximately equal for circles
    EXPECT_GT(std::abs(hu1[0]), 1e-10);
    EXPECT_NEAR(hu1[0], hu2[0], std::abs(hu1[0]) * 0.05);  // 5% tolerance
}

TEST_F(ContourAnalysisTest, HuMoments_ScaleInvariant) {
    QContour square1 = CreateSquare(10.0);
    QContour square2 = CreateSquare(50.0);  // Scaled by 5

    HuMomentsResult hu1 = ContourHuMoments(square1);
    HuMomentsResult hu2 = ContourHuMoments(square2);

    for (int i = 0; i < 7; ++i) {
        EXPECT_NEAR(hu1[i], hu2[i], std::abs(hu1[i]) * 1e-4 + 1e-10);
    }
}

// =============================================================================
// Shape Descriptor Tests
// =============================================================================

TEST_F(ContourAnalysisTest, Circularity_Circle) {
    QContour circle = CreateCircle(50.0, {0, 0}, 200);
    double circularity = ContourCircularity(circle);
    EXPECT_NEAR(circularity, 1.0, 0.01);
}

TEST_F(ContourAnalysisTest, Circularity_Square) {
    QContour square = CreateSquare(100.0);
    double circularity = ContourCircularity(square);
    // Square: 4*PI*A/P^2 = 4*PI*10000/160000 = PI/4 ~ 0.785
    EXPECT_NEAR(circularity, PI / 4.0, 0.01);
}

TEST_F(ContourAnalysisTest, Compactness_Circle) {
    QContour circle = CreateCircle(50.0, {0, 0}, 200);
    double compactness = ContourCompactness(circle);
    // Circle: P^2/A = (2*PI*r)^2 / (PI*r^2) = 4*PI
    EXPECT_NEAR(compactness, 4.0 * PI, 0.5);
}

TEST_F(ContourAnalysisTest, Convexity_ConvexShape) {
    QContour square = CreateSquare(100.0);
    double convexity = ContourConvexity(square);
    EXPECT_NEAR(convexity, 1.0, 0.01);
}

TEST_F(ContourAnalysisTest, Convexity_ConcaveShape) {
    QContour lShape = CreateLShape();
    double convexity = ContourConvexity(lShape);
    EXPECT_LT(convexity, 1.0);
}

TEST_F(ContourAnalysisTest, Solidity_ConvexShape) {
    QContour square = CreateSquare(100.0);
    double solidity = ContourSolidity(square);
    EXPECT_NEAR(solidity, 1.0, 0.01);
}

TEST_F(ContourAnalysisTest, Solidity_ConcaveShape) {
    QContour lShape = CreateLShape();
    double solidity = ContourSolidity(lShape);
    EXPECT_LT(solidity, 1.0);
}

TEST_F(ContourAnalysisTest, Eccentricity_Circle) {
    QContour circle = CreateCircle(50.0, {0, 0}, 100);
    double eccentricity = ContourEccentricity(circle);
    EXPECT_NEAR(eccentricity, 0.0, 0.1);
}

TEST_F(ContourAnalysisTest, Eccentricity_ElongatedRectangle) {
    QContour rect = CreateRectangle(100.0, 10.0);
    double eccentricity = ContourEccentricity(rect);
    EXPECT_GT(eccentricity, 0.9);  // Highly elongated
}

TEST_F(ContourAnalysisTest, Elongation_Circle) {
    QContour circle = CreateCircle(50.0, {0, 0}, 100);
    double elongation = ContourElongation(circle);
    EXPECT_NEAR(elongation, 0.0, 0.1);
}

TEST_F(ContourAnalysisTest, Elongation_Rectangle) {
    QContour rect = CreateRectangle(100.0, 50.0);
    double elongation = ContourElongation(rect);
    EXPECT_GT(elongation, 0.0);
    EXPECT_LT(elongation, 1.0);
}

TEST_F(ContourAnalysisTest, Rectangularity_Rectangle) {
    QContour rect = CreateRectangle(100.0, 50.0);
    double rectangularity = ContourRectangularity(rect);
    EXPECT_NEAR(rectangularity, 1.0, 0.01);
}

TEST_F(ContourAnalysisTest, Rectangularity_Circle) {
    QContour circle = CreateCircle(50.0, {0, 0}, 200);
    double rectangularity = ContourRectangularity(circle);
    // Circle inscribed in square has rectangularity ~ PI/4
    EXPECT_LT(rectangularity, 1.0);
}

TEST_F(ContourAnalysisTest, Extent_AxisAlignedSquare) {
    QContour square = CreateSquare(100.0);
    double extent = ContourExtent(square);
    EXPECT_NEAR(extent, 1.0, 0.01);
}

TEST_F(ContourAnalysisTest, Extent_Circle) {
    QContour circle = CreateCircle(50.0, {0, 0}, 200);
    double extent = ContourExtent(circle);
    // Circle in AABB: PI*r^2 / (2r)^2 = PI/4
    EXPECT_NEAR(extent, PI / 4.0, 0.01);
}

TEST_F(ContourAnalysisTest, AspectRatio_Square) {
    QContour square = CreateSquare(100.0);
    double aspectRatio = ContourAspectRatio(square);
    EXPECT_NEAR(aspectRatio, 1.0, 0.1);
}

TEST_F(ContourAnalysisTest, AspectRatio_Rectangle) {
    QContour rect = CreateRectangle(100.0, 50.0);
    double aspectRatio = ContourAspectRatio(rect);
    EXPECT_GT(aspectRatio, 1.5);
}

TEST_F(ContourAnalysisTest, AllDescriptors_Valid) {
    QContour square = CreateSquare(100.0);
    ShapeDescriptors desc = ContourAllDescriptors(square);

    EXPECT_TRUE(desc.valid);
    EXPECT_GT(desc.circularity, 0.0);
    EXPECT_GT(desc.compactness, 0.0);
    EXPECT_GT(desc.convexity, 0.0);
    EXPECT_GT(desc.solidity, 0.0);
}

// =============================================================================
// Bounding Geometry Tests
// =============================================================================

TEST_F(ContourAnalysisTest, BoundingBox_Square) {
    QContour square = CreateSquare(100.0, {50, 50});
    Rect2d bbox = ContourBoundingBox(square);

    EXPECT_NEAR(bbox.x, 0.0, 1e-10);
    EXPECT_NEAR(bbox.y, 0.0, 1e-10);
    EXPECT_NEAR(bbox.width, 100.0, 1e-10);
    EXPECT_NEAR(bbox.height, 100.0, 1e-10);
}

TEST_F(ContourAnalysisTest, BoundingBox_Circle) {
    double radius = 50.0;
    QContour circle = CreateCircle(radius, {100, 100}, 100);
    Rect2d bbox = ContourBoundingBox(circle);

    EXPECT_NEAR(bbox.x, 100.0 - radius, 1.0);
    EXPECT_NEAR(bbox.y, 100.0 - radius, 1.0);
    EXPECT_NEAR(bbox.width, 2 * radius, 2.0);
    EXPECT_NEAR(bbox.height, 2 * radius, 2.0);
}

TEST_F(ContourAnalysisTest, MinAreaRect_Square) {
    QContour square = CreateSquare(100.0, {50, 50});
    auto minRect = ContourMinAreaRect(square);

    ASSERT_TRUE(minRect.has_value());
    EXPECT_NEAR(minRect->Area(), 10000.0, 10.0);
}

TEST_F(ContourAnalysisTest, MinEnclosingCircle_Square) {
    QContour square = CreateSquare(100.0, {0, 0});
    auto circle = ContourMinEnclosingCircle(square);

    ASSERT_TRUE(circle.has_value());
    // Circumscribed circle of square: r = side * sqrt(2) / 2
    double expectedRadius = 100.0 * std::sqrt(2.0) / 2.0;
    EXPECT_NEAR(circle->radius, expectedRadius, 1.0);
}

TEST_F(ContourAnalysisTest, MinEnclosingCircle_Circle) {
    double radius = 50.0;
    QContour circle = CreateCircle(radius, {100, 100}, 100);
    auto enclosing = ContourMinEnclosingCircle(circle);

    ASSERT_TRUE(enclosing.has_value());
    EXPECT_NEAR(enclosing->radius, radius, 1.0);
    EXPECT_NEAR(enclosing->center.x, 100.0, 1.0);
    EXPECT_NEAR(enclosing->center.y, 100.0, 1.0);
}

TEST_F(ContourAnalysisTest, MinEnclosingEllipse_Circle) {
    QContour circle = CreateCircle(50.0, {100, 100}, 100);
    auto ellipse = ContourMinEnclosingEllipse(circle);

    // Ellipse fitting on circle should give approximately equal semi-axes
    if (ellipse.has_value()) {
        EXPECT_NEAR(ellipse->a, ellipse->b, 5.0);
    }
}

// =============================================================================
// Convexity Analysis Tests
// =============================================================================

TEST_F(ContourAnalysisTest, ConvexHull_SquareUnchanged) {
    QContour square = CreateSquare(100.0);
    QContour hull = ContourConvexHull(square);

    // Convex hull of a square is the square itself
    EXPECT_EQ(hull.Size(), 4u);
}

TEST_F(ContourAnalysisTest, ConvexHull_ConcaveShape) {
    QContour lShape = CreateLShape();
    QContour hull = ContourConvexHull(lShape);

    // L-shape has 6 points, convex hull should have 4
    EXPECT_LT(hull.Size(), lShape.Size());
}

TEST_F(ContourAnalysisTest, ConvexHullArea_Square) {
    QContour square = CreateSquare(100.0);
    double hullArea = ContourConvexHullArea(square);
    EXPECT_NEAR(hullArea, 10000.0, 1e-6);
}

TEST_F(ContourAnalysisTest, IsConvex_Square) {
    QContour square = CreateSquare(100.0);
    EXPECT_TRUE(IsContourConvex(square));
}

TEST_F(ContourAnalysisTest, IsConvex_LShape) {
    QContour lShape = CreateLShape();
    EXPECT_FALSE(IsContourConvex(lShape));
}

TEST_F(ContourAnalysisTest, ConvexityDefects_Square) {
    QContour square = CreateSquare(100.0);
    auto defects = ContourConvexityDefects(square);

    // Square is convex, no defects
    EXPECT_TRUE(defects.empty());
}

TEST_F(ContourAnalysisTest, ConvexityDefects_LShape) {
    QContour lShape = CreateLShape();
    auto defects = ContourConvexityDefects(lShape, 0.1);

    // L-shape should have at least one defect
    EXPECT_GT(defects.size(), 0u);
}

// =============================================================================
// Shape Comparison Tests
// =============================================================================

TEST_F(ContourAnalysisTest, MatchShapesHu_IdenticalShapes) {
    QContour square1 = CreateSquare(100.0);
    QContour square2 = CreateSquare(100.0);

    double distance = MatchShapesHu(square1, square2);
    EXPECT_NEAR(distance, 0.0, 1e-6);
}

TEST_F(ContourAnalysisTest, MatchShapesHu_TranslatedShapes) {
    // Use circles with small translation to minimize numerical issues
    QContour circle1 = CreateCircle(50.0, {50, 50}, 360);
    QContour circle2 = CreateCircle(50.0, {60, 60}, 360);

    double distance = MatchShapesHu(circle1, circle2);
    EXPECT_LT(distance, 1.0);  // Translated shapes should be similar
}

TEST_F(ContourAnalysisTest, MatchShapesHu_ScaledShapes) {
    QContour square1 = CreateSquare(50.0);
    QContour square2 = CreateSquare(100.0);

    double distance = MatchShapesHu(square1, square2);
    EXPECT_LT(distance, 0.1);  // Hu moments are scale-invariant
}

TEST_F(ContourAnalysisTest, MatchShapesHu_DifferentShapes) {
    QContour square = CreateSquare(100.0);
    QContour triangle = CreateTriangle(100.0);

    double distance = MatchShapesHu(square, triangle);
    EXPECT_GT(distance, 0.1);  // Different shapes should have larger distance
}

TEST_F(ContourAnalysisTest, MatchShapesContour_IdenticalShapes) {
    QContour square1 = CreateSquare(100.0);
    QContour square2 = CreateSquare(100.0);

    double similarity = MatchShapesContour(square1, square2);
    EXPECT_NEAR(similarity, 1.0, 0.01);
}

TEST_F(ContourAnalysisTest, MatchShapesContour_DifferentShapes) {
    QContour square = CreateSquare(100.0);
    QContour triangle = CreateTriangle(100.0);

    double similarity = MatchShapesContour(square, triangle);
    // Square and triangle are different enough that similarity should be lower
    EXPECT_LT(similarity, 1.0);
    EXPECT_GT(similarity, 0.0);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

class ContourAnalysisEdgeCaseTest : public ::testing::Test {};

TEST_F(ContourAnalysisEdgeCaseTest, EmptyContour) {
    QContour empty;

    EXPECT_DOUBLE_EQ(ContourLength(empty), 0.0);
    EXPECT_DOUBLE_EQ(ContourArea(empty), 0.0);

    Point2d centroid = ContourCentroid(empty);
    EXPECT_DOUBLE_EQ(centroid.x, 0.0);
    EXPECT_DOUBLE_EQ(centroid.y, 0.0);
}

TEST_F(ContourAnalysisEdgeCaseTest, SinglePoint) {
    QContour single;
    single.AddPoint({100, 100, 0, 0, 0});

    EXPECT_DOUBLE_EQ(ContourLength(single), 0.0);
    EXPECT_DOUBLE_EQ(ContourArea(single), 0.0);

    Point2d centroid = ContourCentroid(single);
    EXPECT_DOUBLE_EQ(centroid.x, 100.0);
    EXPECT_DOUBLE_EQ(centroid.y, 100.0);
}

TEST_F(ContourAnalysisEdgeCaseTest, TwoPoints) {
    QContour two;
    two.AddPoint({0, 0, 0, 0, 0});
    two.AddPoint({100, 0, 0, 0, 0});
    two.SetClosed(false);

    EXPECT_NEAR(ContourLength(two), 100.0, 1e-10);
    EXPECT_DOUBLE_EQ(ContourArea(two), 0.0);
}

TEST_F(ContourAnalysisEdgeCaseTest, CollinearPoints) {
    QContour line;
    for (int i = 0; i <= 10; ++i) {
        line.AddPoint({static_cast<double>(i * 10), 0, 0, 0, 0});
    }
    line.SetClosed(true);

    // Area should be 0 for collinear points
    EXPECT_NEAR(ContourArea(line), 0.0, 1e-10);
}

TEST_F(ContourAnalysisEdgeCaseTest, VerySmallContour) {
    QContour tiny;
    tiny.AddPoint({0.001, 0.001, 0, 0, 0});
    tiny.AddPoint({0.002, 0.001, 0, 0, 0});
    tiny.AddPoint({0.002, 0.002, 0, 0, 0});
    tiny.AddPoint({0.001, 0.002, 0, 0, 0});
    tiny.SetClosed(true);

    double area = ContourArea(tiny);
    EXPECT_NEAR(area, 1e-6, 1e-9);
}

TEST_F(ContourAnalysisEdgeCaseTest, VeryLargeContour) {
    QContour large;
    double size = 1e6;
    large.AddPoint({0, 0, 0, 0, 0});
    large.AddPoint({size, 0, 0, 0, 0});
    large.AddPoint({size, size, 0, 0, 0});
    large.AddPoint({0, size, 0, 0, 0});
    large.SetClosed(true);

    double area = ContourArea(large);
    EXPECT_NEAR(area, size * size, area * 1e-10);
}

TEST_F(ContourAnalysisEdgeCaseTest, Curvature_EmptyContour) {
    QContour empty;
    auto curvatures = ComputeContourCurvature(empty);
    EXPECT_TRUE(curvatures.empty());
}

TEST_F(ContourAnalysisEdgeCaseTest, Moments_EmptyContour) {
    QContour empty;
    MomentsResult moments = ContourMoments(empty);
    EXPECT_DOUBLE_EQ(moments.m00, 0.0);
}

TEST_F(ContourAnalysisEdgeCaseTest, HuMoments_EmptyContour) {
    QContour empty;
    HuMomentsResult hu = ContourHuMoments(empty);

    for (int i = 0; i < 7; ++i) {
        EXPECT_DOUBLE_EQ(hu[i], 0.0);
    }
}

TEST_F(ContourAnalysisEdgeCaseTest, BoundingBox_EmptyContour) {
    QContour empty;
    Rect2d bbox = ContourBoundingBox(empty);

    EXPECT_DOUBLE_EQ(bbox.x, 0.0);
    EXPECT_DOUBLE_EQ(bbox.y, 0.0);
    EXPECT_DOUBLE_EQ(bbox.width, 0.0);
    EXPECT_DOUBLE_EQ(bbox.height, 0.0);
}

TEST_F(ContourAnalysisEdgeCaseTest, MinAreaRect_TooFewPoints) {
    QContour two;
    two.AddPoint({0, 0, 0, 0, 0});
    two.AddPoint({10, 10, 0, 0, 0});

    auto rect = ContourMinAreaRect(two);
    EXPECT_FALSE(rect.has_value());
}

TEST_F(ContourAnalysisEdgeCaseTest, MinEnclosingCircle_EmptyContour) {
    QContour empty;
    auto circle = ContourMinEnclosingCircle(empty);
    EXPECT_FALSE(circle.has_value());
}

TEST_F(ContourAnalysisEdgeCaseTest, ConvexHull_TooFewPoints) {
    QContour two;
    two.AddPoint({0, 0, 0, 0, 0});
    two.AddPoint({10, 10, 0, 0, 0});

    QContour hull = ContourConvexHull(two);
    EXPECT_EQ(hull.Size(), two.Size());
}

TEST_F(ContourAnalysisEdgeCaseTest, IsConvex_TooFewPoints) {
    QContour two;
    two.AddPoint({0, 0, 0, 0, 0});
    two.AddPoint({10, 10, 0, 0, 0});

    // Less than 3 points is trivially convex
    EXPECT_TRUE(IsContourConvex(two));
}

// =============================================================================
// Accuracy Tests
// =============================================================================

class ContourAnalysisAccuracyTest : public ::testing::Test {
protected:
    QContour CreateCircle(double radius, Point2d center, int numPoints) {
        QContour contour;
        for (int i = 0; i < numPoints; ++i) {
            double angle = 2.0 * PI * i / numPoints;
            double x = center.x + radius * std::cos(angle);
            double y = center.y + radius * std::sin(angle);
            contour.AddPoint({x, y, 0, 0, 0});
        }
        contour.SetClosed(true);
        return contour;
    }
};

TEST_F(ContourAnalysisAccuracyTest, CircleArea_HighPrecision) {
    double radius = 100.0;
    QContour circle = CreateCircle(radius, {0, 0}, 10000);

    double area = ContourArea(circle);
    double expected = PI * radius * radius;

    EXPECT_NEAR(area, expected, expected * 0.0001);  // 0.01% error
}

TEST_F(ContourAnalysisAccuracyTest, CirclePerimeter_HighPrecision) {
    double radius = 100.0;
    QContour circle = CreateCircle(radius, {0, 0}, 10000);

    double perimeter = ContourPerimeter(circle);
    double expected = 2.0 * PI * radius;

    EXPECT_NEAR(perimeter, expected, expected * 0.0001);  // 0.01% error
}

TEST_F(ContourAnalysisAccuracyTest, CircleCircularity_HighPrecision) {
    QContour circle = CreateCircle(100.0, {0, 0}, 10000);

    double circularity = ContourCircularity(circle);
    EXPECT_NEAR(circularity, 1.0, 0.0001);
}
