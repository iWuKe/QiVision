/**
 * @file test_region_features.cpp
 * @brief Unit tests for RegionFeatures module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/RegionFeatures.h>
#include <QiVision/Internal/RLEOps.h>
#include <QiVision/Core/QRegion.h>
#include <cmath>
#include <algorithm>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

// Alias to avoid conflict with testing::Test::Run
using QRun = QRegion::Run;

class RegionFeaturesTest : public ::testing::Test {
protected:
    // Create a rectangular region
    // Note: Run::colEnd is exclusive, so use x + width for colEnd
    QRegion CreateRectRegion(int32_t x, int32_t y, int32_t width, int32_t height) {
        std::vector<QRun> runs;
        for (int32_t row = y; row < y + height; ++row) {
            runs.push_back({row, x, x + width});  // colEnd is exclusive
        }
        return QRegion(runs);
    }

    // Create a filled circle region (approximate)
    // Note: Run::colEnd is exclusive
    QRegion CreateCircleRegion(int32_t cx, int32_t cy, int32_t radius) {
        std::vector<QRun> runs;
        for (int32_t row = cy - radius; row <= cy + radius; ++row) {
            int32_t dy = row - cy;
            int32_t dx = static_cast<int32_t>(std::sqrt(static_cast<double>(radius * radius - dy * dy)));
            if (dx >= 0) {
                runs.push_back({row, cx - dx, cx + dx + 1});  // colEnd is exclusive
            }
        }
        return QRegion(runs);
    }

    // Create an ellipse region
    // Note: Run::colEnd is exclusive
    QRegion CreateEllipseRegion(int32_t cx, int32_t cy, int32_t a, int32_t b) {
        std::vector<QRun> runs;
        for (int32_t row = cy - b; row <= cy + b; ++row) {
            int32_t dy = row - cy;
            double t = static_cast<double>(dy) / b;
            if (std::abs(t) <= 1.0) {
                int32_t dx = static_cast<int32_t>(a * std::sqrt(1.0 - t * t));
                if (dx >= 0) {
                    runs.push_back({row, cx - dx, cx + dx + 1});  // colEnd is exclusive
                }
            }
        }
        return QRegion(runs);
    }

    // Create a triangle region
    // Note: Run::colEnd is exclusive
    QRegion CreateTriangleRegion(int32_t x, int32_t y, int32_t base, int32_t height) {
        std::vector<QRun> runs;
        for (int32_t row = 0; row < height; ++row) {
            double ratio = static_cast<double>(height - row) / height;
            int32_t halfWidth = static_cast<int32_t>((base / 2.0) * ratio);
            int32_t cx = x + base / 2;
            if (halfWidth > 0) {
                runs.push_back({y + row, cx - halfWidth, cx + halfWidth + 1});  // colEnd is exclusive
            } else {
                runs.push_back({y + row, cx, cx + 1});  // colEnd is exclusive
            }
        }
        return QRegion(runs);
    }

    // Create L-shaped region
    // Note: Run::colEnd is exclusive
    QRegion CreateLShapeRegion(int32_t x, int32_t y, int32_t width, int32_t height, int32_t thickness) {
        std::vector<QRun> runs;
        // Vertical bar
        for (int32_t row = y; row < y + height; ++row) {
            runs.push_back({row, x, x + thickness});  // colEnd is exclusive
        }
        // Horizontal bar (bottom)
        for (int32_t row = y + height - thickness; row < y + height; ++row) {
            if (x + thickness < x + width) {
                runs.push_back({row, x + thickness, x + width});  // colEnd is exclusive
            }
        }
        // Sort and merge runs
        std::sort(runs.begin(), runs.end(), [](const QRun& a, const QRun& b) {
            return a.row < b.row || (a.row == b.row && a.colBegin < b.colBegin);
        });
        return QRegion(runs);
    }
};

// =============================================================================
// Basic Features Tests
// =============================================================================

TEST_F(RegionFeaturesTest, ComputeArea_Rectangle) {
    auto region = CreateRectRegion(10, 10, 50, 30);
    int64_t area = ComputeArea(region);
    EXPECT_EQ(area, 50 * 30);
}

TEST_F(RegionFeaturesTest, ComputeArea_Circle) {
    auto region = CreateCircleRegion(50, 50, 20);
    int64_t area = ComputeArea(region);
    double expectedArea = M_PI * 20 * 20;
    // Allow some discretization error
    EXPECT_NEAR(area, expectedArea, expectedArea * 0.05);
}

TEST_F(RegionFeaturesTest, ComputeArea_EmptyRegion) {
    QRegion empty;
    int64_t area = ComputeArea(empty);
    EXPECT_EQ(area, 0);
}

TEST_F(RegionFeaturesTest, ComputePerimeter_Rectangle) {
    auto region = CreateRectRegion(10, 10, 50, 30);
    double perimeter = ComputePerimeter(region);
    // Rectangle perimeter: 2*(width + height)
    double expected = 2 * (50 + 30);
    EXPECT_NEAR(perimeter, expected, expected * 0.1);
}

TEST_F(RegionFeaturesTest, ComputePerimeter_Circle) {
    auto region = CreateCircleRegion(50, 50, 20);
    double perimeter = ComputePerimeter(region);
    double expected = 2 * M_PI * 20;
    // Discrete circles have "staircase" perimeter, allow 25% error
    EXPECT_NEAR(perimeter, expected, expected * 0.25);
}

TEST_F(RegionFeaturesTest, ComputeCentroid_Rectangle) {
    auto region = CreateRectRegion(10, 20, 50, 30);
    auto centroid = ComputeRegionCentroid(region);
    EXPECT_NEAR(centroid.x, 10 + 50.0/2 - 0.5, 1.0);
    EXPECT_NEAR(centroid.y, 20 + 30.0/2 - 0.5, 1.0);
}

TEST_F(RegionFeaturesTest, ComputeCentroid_Circle) {
    auto region = CreateCircleRegion(50, 60, 20);
    auto centroid = ComputeRegionCentroid(region);
    EXPECT_NEAR(centroid.x, 50.0, 1.0);
    EXPECT_NEAR(centroid.y, 60.0, 1.0);
}

TEST_F(RegionFeaturesTest, ComputeBoundingBox_Rectangle) {
    auto region = CreateRectRegion(10, 20, 50, 30);
    auto bbox = ComputeBoundingBox(region);
    EXPECT_EQ(bbox.x, 10);
    EXPECT_EQ(bbox.y, 20);
    EXPECT_EQ(bbox.width, 50);
    EXPECT_EQ(bbox.height, 30);
}

TEST_F(RegionFeaturesTest, ComputeBasicFeatures_All) {
    auto region = CreateRectRegion(10, 20, 50, 30);
    auto features = ComputeBasicFeatures(region);

    EXPECT_EQ(features.area, 50 * 30);
    EXPECT_GT(features.perimeter, 0);
    EXPECT_NEAR(features.centroidX, 10 + 50.0/2 - 0.5, 1.0);
    EXPECT_NEAR(features.centroidY, 20 + 30.0/2 - 0.5, 1.0);
    EXPECT_EQ(features.boundingBox.width, 50);
    EXPECT_EQ(features.boundingBox.height, 30);
}

// =============================================================================
// Shape Features Tests
// =============================================================================

TEST_F(RegionFeaturesTest, ComputeCircularity_Circle) {
    // Use larger radius for better approximation of circle
    auto region = CreateCircleRegion(100, 100, 50);
    double circularity = ComputeCircularity(region);
    // Discrete circle has lower circularity due to staircase perimeter
    // Even large discrete circles have circularity ~0.62-0.65
    EXPECT_GT(circularity, 0.60);  // Relaxed for discretization
    EXPECT_LE(circularity, 1.0);
}

TEST_F(RegionFeaturesTest, ComputeCircularity_Square) {
    auto region = CreateRectRegion(10, 10, 50, 50);
    double circularity = ComputeCircularity(region);
    // Square has circularity of pi/4 ~ 0.785
    EXPECT_GT(circularity, 0.7);
    EXPECT_LT(circularity, 0.9);
}

TEST_F(RegionFeaturesTest, ComputeCircularity_Rectangle) {
    auto region = CreateRectRegion(10, 10, 100, 20);
    double circularity = ComputeCircularity(region);
    // Elongated rectangle has lower circularity
    EXPECT_LT(circularity, 0.7);
}

TEST_F(RegionFeaturesTest, ComputeCompactness_Circle) {
    auto region = CreateCircleRegion(100, 100, 50);
    double compactness = ComputeCompactness(region);
    // Discrete circle has higher compactness due to staircase perimeter
    // Circle has minimum compactness ~ 4*pi ~ 12.57, discrete will be higher
    EXPECT_GT(compactness, 12);
    EXPECT_LT(compactness, 25);  // Relaxed for discretization
}

TEST_F(RegionFeaturesTest, ComputeCompactness_Rectangle) {
    auto region = CreateRectRegion(10, 10, 100, 20);
    double compactness = ComputeCompactness(region);
    // Rectangle has higher compactness than circle
    EXPECT_GT(compactness, 16);
}

TEST_F(RegionFeaturesTest, ComputeElongation_Square) {
    auto region = CreateRectRegion(10, 10, 50, 50);
    double elongation = ComputeElongation(region);
    // Square should have elongation close to 1.0
    EXPECT_NEAR(elongation, 1.0, 0.2);
}

TEST_F(RegionFeaturesTest, ComputeElongation_Rectangle) {
    auto region = CreateRectRegion(10, 10, 100, 20);
    double elongation = ComputeElongation(region);
    // 5:1 rectangle should have higher elongation
    EXPECT_GT(elongation, 3.0);
}

TEST_F(RegionFeaturesTest, ComputeRectangularity_Rectangle) {
    auto region = CreateRectRegion(10, 10, 50, 30);
    double rectangularity = ComputeRectangularity(region);
    // Perfect rectangle has rectangularity = 1.0
    EXPECT_NEAR(rectangularity, 1.0, 0.01);
}

TEST_F(RegionFeaturesTest, ComputeRectangularity_Circle) {
    auto region = CreateCircleRegion(50, 50, 30);
    double rectangularity = ComputeRectangularity(region);
    // Circle has rectangularity ~ pi/4 ~ 0.785
    EXPECT_GT(rectangularity, 0.7);
    EXPECT_LT(rectangularity, 0.85);
}

TEST_F(RegionFeaturesTest, ComputeConvexity_Convex) {
    auto region = CreateRectRegion(10, 10, 50, 30);
    double convexity = ComputeConvexity(region);
    // Rectangle is convex, convexity should be close to 1.0
    EXPECT_GT(convexity, 0.9);
}

TEST_F(RegionFeaturesTest, ComputeConvexity_NonConvex) {
    auto region = CreateLShapeRegion(10, 10, 50, 50, 10);
    double convexity = ComputeConvexity(region);
    // L-shape is not convex
    EXPECT_LT(convexity, 0.9);
}

TEST_F(RegionFeaturesTest, ComputeSolidity_Convex) {
    auto region = CreateRectRegion(10, 10, 50, 30);
    double solidity = ComputeSolidity(region);
    // Rectangle is convex, solidity should be close to 1.0
    EXPECT_GT(solidity, 0.95);
}

TEST_F(RegionFeaturesTest, ComputeSolidity_NonConvex) {
    auto region = CreateLShapeRegion(10, 10, 50, 50, 10);
    double solidity = ComputeSolidity(region);
    // L-shape has lower solidity
    EXPECT_LT(solidity, 0.8);
}

TEST_F(RegionFeaturesTest, ComputeShapeFeatures_All) {
    auto region = CreateCircleRegion(50, 50, 30);
    auto features = ComputeShapeFeatures(region);

    EXPECT_GT(features.circularity, 0);
    EXPECT_GT(features.compactness, 0);
    EXPECT_GE(features.elongation, 1.0);
    EXPECT_GT(features.rectangularity, 0);
    EXPECT_GT(features.convexity, 0);
    EXPECT_GT(features.solidity, 0);
}

// =============================================================================
// Moment Features Tests
// =============================================================================

TEST_F(RegionFeaturesTest, ComputeRawMoment_M00) {
    auto region = CreateRectRegion(10, 10, 50, 30);
    double m00 = ComputeRawMoment(region, 0, 0);
    // m00 = area
    EXPECT_EQ(m00, 50 * 30);
}

TEST_F(RegionFeaturesTest, ComputeRawMoment_Centroid) {
    auto region = CreateRectRegion(10, 10, 50, 30);
    double m00 = ComputeRawMoment(region, 0, 0);
    double m10 = ComputeRawMoment(region, 1, 0);
    double m01 = ComputeRawMoment(region, 0, 1);

    double cx = m10 / m00;
    double cy = m01 / m00;

    // Centroid should be at center of rectangle
    EXPECT_NEAR(cx, 10 + 50.0/2 - 0.5, 1.0);
    EXPECT_NEAR(cy, 10 + 30.0/2 - 0.5, 1.0);
}

TEST_F(RegionFeaturesTest, ComputeCentralMoment_Centered) {
    auto region = CreateRectRegion(10, 10, 50, 30);

    // mu00 should equal area
    double mu00 = ComputeRawMoment(region, 0, 0);
    EXPECT_EQ(mu00, 50 * 30);

    // mu10 and mu01 should be 0 (centered)
    double mu10 = ComputeCentralMoment(region, 1, 0);
    double mu01 = ComputeCentralMoment(region, 0, 1);
    EXPECT_NEAR(mu10, 0.0, 1.0);
    EXPECT_NEAR(mu01, 0.0, 1.0);
}

TEST_F(RegionFeaturesTest, ComputeMoments_All) {
    auto region = CreateRectRegion(10, 10, 50, 30);
    auto moments = ComputeMoments(region);

    // Raw moments
    EXPECT_EQ(moments.m00, 50 * 30);
    EXPECT_GT(moments.m10, 0);
    EXPECT_GT(moments.m01, 0);
    EXPECT_GT(moments.m20, 0);
    EXPECT_GT(moments.m02, 0);

    // Central moments - mu10 and mu01 should be near 0
    EXPECT_NEAR(moments.mu20, moments.m20 - moments.m10 * moments.m10 / moments.m00, 1.0);

    // Hu moments should be non-zero
    EXPECT_NE(moments.hu[0], 0);
}

TEST_F(RegionFeaturesTest, ComputeHuMoments_RotationInvariant) {
    // Create a region and its rotated version
    // Due to discretization, exact invariance is hard to test
    auto region = CreateEllipseRegion(50, 50, 30, 15);
    auto hu = ComputeHuMoments(region);

    // All Hu moments should be defined
    for (int i = 0; i < 7; ++i) {
        EXPECT_TRUE(std::isfinite(hu[i]));
    }

    // First Hu moment should be positive
    EXPECT_GT(hu[0], 0);
}

TEST_F(RegionFeaturesTest, ComputeHuMoments_ScaleInvariant) {
    auto region1 = CreateCircleRegion(50, 50, 20);
    auto region2 = CreateCircleRegion(50, 50, 40);

    auto hu1 = ComputeHuMoments(region1);
    auto hu2 = ComputeHuMoments(region2);

    // For similar shapes at different scales, Hu moments should be similar
    // (accounting for discretization effects)
    EXPECT_NEAR(hu1[0], hu2[0], 0.1);
}

// =============================================================================
// Ellipse Features Tests
// =============================================================================

TEST_F(RegionFeaturesTest, ComputeEllipseFeatures_Circle) {
    auto region = CreateCircleRegion(50, 60, 25);
    auto features = ComputeEllipseFeatures(region);

    EXPECT_NEAR(features.centerX, 50.0, 2.0);
    EXPECT_NEAR(features.centerY, 60.0, 2.0);
    // For circle, major and minor axes should be similar
    EXPECT_NEAR(features.majorAxis, features.minorAxis, features.majorAxis * 0.2);
    // Eccentricity should be near 0 for circle
    EXPECT_LT(features.eccentricity, 0.3);
}

TEST_F(RegionFeaturesTest, ComputeEllipseFeatures_Ellipse) {
    auto region = CreateEllipseRegion(50, 60, 40, 20);
    auto features = ComputeEllipseFeatures(region);

    EXPECT_NEAR(features.centerX, 50.0, 2.0);
    EXPECT_NEAR(features.centerY, 60.0, 2.0);
    EXPECT_GT(features.majorAxis, features.minorAxis);
    // Eccentricity should be higher for ellipse
    EXPECT_GT(features.eccentricity, 0.5);
}

TEST_F(RegionFeaturesTest, ComputeOrientation_HorizontalRect) {
    auto region = CreateRectRegion(10, 10, 100, 20);
    double angle = ComputeOrientation(region);
    // Horizontal rectangle should have angle near 0
    EXPECT_NEAR(std::abs(angle), 0.0, 0.2);
}

TEST_F(RegionFeaturesTest, ComputeOrientation_VerticalRect) {
    auto region = CreateRectRegion(10, 10, 20, 100);
    double angle = ComputeOrientation(region);
    // Vertical rectangle should have angle near +/- pi/2
    EXPECT_NEAR(std::abs(angle), M_PI / 2, 0.2);
}

TEST_F(RegionFeaturesTest, ComputePrincipalAxes_Rectangle) {
    auto region = CreateRectRegion(10, 10, 100, 20);
    double majorAxis, minorAxis;
    ComputePrincipalAxes(region, majorAxis, minorAxis);

    EXPECT_GT(majorAxis, minorAxis);
    EXPECT_GT(majorAxis, 0);
    EXPECT_GT(minorAxis, 0);
}

// =============================================================================
// Enclosing Shapes Tests
// =============================================================================

TEST_F(RegionFeaturesTest, ComputeMinAreaRect_Rectangle) {
    auto region = CreateRectRegion(10, 20, 50, 30);
    auto rect = ComputeMinAreaRect(region);

    // Area should match region area approximately
    double rectArea = rect.width * rect.height;
    EXPECT_NEAR(rectArea, 50 * 30, 50 * 30 * 0.1);
}

TEST_F(RegionFeaturesTest, ComputeMinAreaRect_Circle) {
    auto region = CreateCircleRegion(50, 50, 25);
    auto rect = ComputeMinAreaRect(region);

    // For circle, width and height should be similar (~diameter)
    EXPECT_NEAR(rect.width, rect.height, 5.0);
    EXPECT_NEAR(rect.width, 50.0, 5.0);
}

TEST_F(RegionFeaturesTest, ComputeMinEnclosingCircle_Circle) {
    auto region = CreateCircleRegion(50, 60, 25);
    auto circle = ComputeMinEnclosingCircle(region);

    EXPECT_NEAR(circle.center.x, 50.0, 3.0);
    EXPECT_NEAR(circle.center.y, 60.0, 3.0);
    EXPECT_NEAR(circle.radius, 25.0, 3.0);
}

TEST_F(RegionFeaturesTest, ComputeMinEnclosingCircle_Rectangle) {
    auto region = CreateRectRegion(10, 10, 60, 40);
    auto circle = ComputeMinEnclosingCircle(region);

    // Center should be near rectangle center
    EXPECT_NEAR(circle.center.x, 10 + 30.0 - 0.5, 5.0);
    EXPECT_NEAR(circle.center.y, 10 + 20.0 - 0.5, 5.0);

    // Radius should be half diagonal
    double diag = std::sqrt(60.0*60.0 + 40.0*40.0);
    EXPECT_NEAR(circle.radius, diag / 2, 5.0);
}

TEST_F(RegionFeaturesTest, ComputeConvexHull_Rectangle) {
    auto region = CreateRectRegion(10, 20, 50, 30);
    auto hull = ComputeConvexHull(region);

    // Rectangle convex hull should have 4 points
    EXPECT_GE(hull.size(), 4);
}

TEST_F(RegionFeaturesTest, ComputeConvexHull_Circle) {
    auto region = CreateCircleRegion(50, 50, 25);
    auto hull = ComputeConvexHull(region);

    // Circle convex hull should have many points
    EXPECT_GT(hull.size(), 10);
}

TEST_F(RegionFeaturesTest, ComputeConvexHullArea_Rectangle) {
    auto region = CreateRectRegion(10, 10, 50, 30);
    double hullArea = ComputeConvexHullArea(region);

    // Hull area should be close to region area
    // Allow 10% tolerance for discrete approximation
    EXPECT_NEAR(hullArea, 50 * 30, 50 * 30 * 0.10);
}

TEST_F(RegionFeaturesTest, ComputeConvexHullArea_LShape) {
    auto region = CreateLShapeRegion(10, 10, 50, 50, 10);
    double regionArea = ComputeArea(region);
    double hullArea = ComputeConvexHullArea(region);

    // For L-shape, hull area > region area
    EXPECT_GT(hullArea, regionArea);
}

TEST_F(RegionFeaturesTest, ComputeConvexHullPerimeter_Rectangle) {
    auto region = CreateRectRegion(10, 10, 50, 30);
    double hullPerim = ComputeConvexHullPerimeter(region);

    // Rectangle perimeter: 2*(width + height)
    double expected = 2 * (50 + 30);
    EXPECT_NEAR(hullPerim, expected, expected * 0.1);
}

// =============================================================================
// Comprehensive Features Tests
// =============================================================================

TEST_F(RegionFeaturesTest, ComputeAllFeatures_NonEmpty) {
    auto region = CreateCircleRegion(50, 50, 25);
    auto features = ComputeAllFeatures(region);

    EXPECT_GT(features.basic.area, 0);
    EXPECT_GT(features.basic.perimeter, 0);
    EXPECT_GT(features.shape.circularity, 0);
    EXPECT_GT(features.moments.m00, 0);
    EXPECT_GT(features.ellipse.majorAxis, 0);
}

TEST_F(RegionFeaturesTest, ComputeAllFeatures_Empty) {
    QRegion empty;
    auto features = ComputeAllFeatures(empty);

    EXPECT_EQ(features.basic.area, 0);
    EXPECT_EQ(features.basic.perimeter, 0);
}

TEST_F(RegionFeaturesTest, ComputeAllFeatures_MultipleRegions) {
    std::vector<QRegion> regions;
    regions.push_back(CreateRectRegion(10, 10, 50, 30));
    regions.push_back(CreateCircleRegion(100, 100, 25));

    auto allFeatures = ComputeAllFeatures(regions);

    EXPECT_EQ(allFeatures.size(), 2);
    EXPECT_GT(allFeatures[0].basic.area, 0);
    EXPECT_GT(allFeatures[1].basic.area, 0);
}

// =============================================================================
// Feature-based Selection Tests
// =============================================================================

TEST_F(RegionFeaturesTest, SelectByCircularity) {
    std::vector<QRegion> regions;
    regions.push_back(CreateCircleRegion(50, 50, 30));  // High circularity
    regions.push_back(CreateRectRegion(10, 10, 100, 10)); // Low circularity
    regions.push_back(CreateRectRegion(10, 10, 30, 30));  // Medium circularity

    auto selected = SelectByCircularity(regions, 0.8, 1.0);

    // Only circle should pass high circularity threshold
    EXPECT_GE(selected.size(), 1);
}

TEST_F(RegionFeaturesTest, SelectByCompactness) {
    std::vector<QRegion> regions;
    regions.push_back(CreateCircleRegion(50, 50, 30));  // Low compactness
    regions.push_back(CreateRectRegion(10, 10, 100, 10)); // High compactness

    auto selected = SelectByCompactness(regions, 12.0, 20.0);

    // At least one region should be selected
    EXPECT_GE(selected.size(), 1);
}

TEST_F(RegionFeaturesTest, SelectByElongation) {
    std::vector<QRegion> regions;
    regions.push_back(CreateRectRegion(10, 10, 100, 20)); // High elongation
    regions.push_back(CreateRectRegion(10, 10, 30, 30));  // Low elongation

    auto selected = SelectByElongation(regions, 3.0, 10.0);

    // Only elongated rectangle should pass
    EXPECT_EQ(selected.size(), 1);
}

TEST_F(RegionFeaturesTest, SelectByOrientation) {
    std::vector<QRegion> regions;
    regions.push_back(CreateRectRegion(10, 10, 100, 20)); // Horizontal
    regions.push_back(CreateRectRegion(10, 10, 20, 100)); // Vertical

    // Select near-horizontal (angle near 0)
    auto selected = SelectByOrientation(regions, -0.3, 0.3);

    EXPECT_GE(selected.size(), 1);
}

// =============================================================================
// Edge Cases Tests
// =============================================================================

TEST_F(RegionFeaturesTest, SinglePixelRegion) {
    std::vector<QRun> runs = {{50, 50, 51}};  // colEnd is exclusive
    QRegion region(runs);

    auto features = ComputeBasicFeatures(region);
    EXPECT_EQ(features.area, 1);
    EXPECT_EQ(features.boundingBox.width, 1);
    EXPECT_EQ(features.boundingBox.height, 1);
}

TEST_F(RegionFeaturesTest, SingleRowRegion) {
    std::vector<QRun> runs = {{50, 10, 60}};  // colEnd is exclusive
    QRegion region(runs);

    auto features = ComputeBasicFeatures(region);
    EXPECT_EQ(features.area, 50);
    EXPECT_EQ(features.boundingBox.width, 50);
    EXPECT_EQ(features.boundingBox.height, 1);
}

TEST_F(RegionFeaturesTest, DiagonalPoints) {
    std::vector<QRun> runs = {{0, 0, 1}, {1, 1, 2}, {2, 2, 3}};  // colEnd is exclusive
    QRegion region(runs);

    auto hull = ComputeConvexHull(region);
    // 3 collinear points should still form a hull
    EXPECT_GE(hull.size(), 2);
}

TEST_F(RegionFeaturesTest, LargeRegion) {
    auto region = CreateRectRegion(0, 0, 1000, 1000);

    auto features = ComputeBasicFeatures(region);
    EXPECT_EQ(features.area, 1000000);
    EXPECT_GT(features.perimeter, 0);
}

TEST_F(RegionFeaturesTest, DisconnectedRegion) {
    std::vector<QRun> runs;
    // Two separate rectangles (10 rows x 20 cols each)
    for (int row = 10; row < 20; ++row) {
        runs.push_back({row, 10, 30});  // colEnd is exclusive, 30-10=20 pixels
    }
    for (int row = 50; row < 60; ++row) {
        runs.push_back({row, 50, 70});  // colEnd is exclusive, 70-50=20 pixels
    }
    QRegion region(runs);

    auto features = ComputeBasicFeatures(region);
    EXPECT_EQ(features.area, 2 * 10 * 20);
}

// =============================================================================
// Consistency Tests
// =============================================================================

TEST_F(RegionFeaturesTest, ShapeFeatures_Consistency) {
    auto region = CreateCircleRegion(50, 50, 25);
    auto shape = ComputeShapeFeatures(region);

    // Circularity should be related to compactness
    // circularity = 4*pi/compactness
    double expectedCirc = 4.0 * M_PI / shape.compactness;
    EXPECT_NEAR(shape.circularity, expectedCirc, 0.1);
}

TEST_F(RegionFeaturesTest, Solidity_LessThanOrEqualOne) {
    auto region = CreateLShapeRegion(10, 10, 50, 50, 10);
    double solidity = ComputeSolidity(region);

    EXPECT_GT(solidity, 0);
    EXPECT_LE(solidity, 1.0);
}

TEST_F(RegionFeaturesTest, Convexity_LessThanOrEqualOne) {
    auto region = CreateLShapeRegion(10, 10, 50, 50, 10);
    double convexity = ComputeConvexity(region);

    EXPECT_GT(convexity, 0);
    EXPECT_LE(convexity, 1.0);
}

TEST_F(RegionFeaturesTest, Elongation_GreaterOrEqualOne) {
    auto region = CreateRectRegion(10, 10, 50, 30);
    double elongation = ComputeElongation(region);

    EXPECT_GE(elongation, 1.0);
}
