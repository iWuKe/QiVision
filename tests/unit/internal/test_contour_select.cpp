/**
 * @file test_contour_select.cpp
 * @brief Unit tests for ContourSelect module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/ContourSelect.h>
#include <QiVision/Internal/ContourAnalysis.h>
#include <cmath>

namespace Qi::Vision::Internal {
namespace {

// =============================================================================
// Test Fixture
// =============================================================================

class ContourSelectTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test contours
        CreateTestContours();
    }

    void CreateTestContours() {
        // Contour 0: Small square (10x10, area=100, length=40)
        QContour smallSquare;
        smallSquare.SetPoints(std::vector<Point2d>{{0, 0}, {10, 0}, {10, 10}, {0, 10}});
        smallSquare.SetClosed(true);
        testContours_.Add(smallSquare);

        // Contour 1: Large square (100x100, area=10000, length=400)
        QContour largeSquare;
        largeSquare.SetPoints(std::vector<Point2d>{{0, 0}, {100, 0}, {100, 100}, {0, 100}});
        largeSquare.SetClosed(true);
        testContours_.Add(largeSquare);

        // Contour 2: Rectangle (20x10, area=200, length=60)
        QContour rectangle;
        rectangle.SetPoints(std::vector<Point2d>{{0, 0}, {20, 0}, {20, 10}, {0, 10}});
        rectangle.SetClosed(true);
        testContours_.Add(rectangle);

        // Contour 3: Circle approximation (radius=50, area~7854, length~314)
        QContour circle;
        std::vector<Point2d> circlePoints;
        const int numPoints = 100;
        const double radius = 50.0;
        for (int i = 0; i < numPoints; ++i) {
            double angle = 2.0 * M_PI * i / numPoints;
            circlePoints.push_back({
                200 + radius * std::cos(angle),
                200 + radius * std::sin(angle)
            });
        }
        circle.SetPoints(circlePoints);
        circle.SetClosed(true);
        testContours_.Add(circle);

        // Contour 4: Long thin rectangle (100x5, elongated)
        QContour thinRect;
        thinRect.SetPoints(std::vector<Point2d>{{0, 0}, {100, 0}, {100, 5}, {0, 5}});
        thinRect.SetClosed(true);
        testContours_.Add(thinRect);

        // Contour 5: Open polyline (not closed, 3 points)
        QContour openLine;
        openLine.SetPoints(std::vector<Point2d>{{0, 0}, {50, 50}, {100, 0}});
        openLine.SetClosed(false);
        testContours_.Add(openLine);

        // Contour 6: L-shape (non-convex)
        QContour lShape;
        lShape.SetPoints(std::vector<Point2d>{{0, 0}, {30, 0}, {30, 10}, {10, 10}, {10, 30}, {0, 30}});
        lShape.SetClosed(true);
        testContours_.Add(lShape);
    }

    QContourArray testContours_;
};

// =============================================================================
// Feature Name Conversion Tests
// =============================================================================

TEST(ContourFeatureTest, FeatureToString) {
    EXPECT_EQ(ContourFeatureToString(ContourFeature::Length), "length");
    EXPECT_EQ(ContourFeatureToString(ContourFeature::Area), "area");
    EXPECT_EQ(ContourFeatureToString(ContourFeature::Circularity), "circularity");
    EXPECT_EQ(ContourFeatureToString(ContourFeature::AspectRatio), "aspect_ratio");
}

TEST(ContourFeatureTest, StringToFeature) {
    EXPECT_EQ(StringToContourFeature("length"), ContourFeature::Length);
    EXPECT_EQ(StringToContourFeature("AREA"), ContourFeature::Area);
    EXPECT_EQ(StringToContourFeature("Circularity"), ContourFeature::Circularity);
    EXPECT_EQ(StringToContourFeature("aspect_ratio"), ContourFeature::AspectRatio);
    EXPECT_EQ(StringToContourFeature("aspectratio"), ContourFeature::AspectRatio);
}

TEST(ContourFeatureTest, StringToFeature_Aliases) {
    EXPECT_EQ(StringToContourFeature("row"), ContourFeature::CentroidRow);
    EXPECT_EQ(StringToContourFeature("col"), ContourFeature::CentroidCol);
    EXPECT_EQ(StringToContourFeature("column"), ContourFeature::CentroidCol);
    EXPECT_EQ(StringToContourFeature("width"), ContourFeature::BoundingBoxWidth);
    EXPECT_EQ(StringToContourFeature("height"), ContourFeature::BoundingBoxHeight);
    EXPECT_EQ(StringToContourFeature("angle"), ContourFeature::Orientation);
}

// =============================================================================
// Feature Computation Tests
// =============================================================================

TEST_F(ContourSelectTest, ComputeContourFeature_Length) {
    // Small square: 4 * 10 = 40
    double length0 = ComputeContourFeature(testContours_[0], ContourFeature::Length);
    EXPECT_NEAR(length0, 40.0, 0.1);

    // Large square: 4 * 100 = 400
    double length1 = ComputeContourFeature(testContours_[1], ContourFeature::Length);
    EXPECT_NEAR(length1, 400.0, 0.1);
}

TEST_F(ContourSelectTest, ComputeContourFeature_Area) {
    // Small square: 10 * 10 = 100
    double area0 = ComputeContourFeature(testContours_[0], ContourFeature::Area);
    EXPECT_NEAR(area0, 100.0, 0.1);

    // Large square: 100 * 100 = 10000
    double area1 = ComputeContourFeature(testContours_[1], ContourFeature::Area);
    EXPECT_NEAR(area1, 10000.0, 0.1);

    // Circle: pi * 50^2 ~ 7854
    double area3 = ComputeContourFeature(testContours_[3], ContourFeature::Area);
    EXPECT_NEAR(area3, M_PI * 50.0 * 50.0, 100.0);
}

TEST_F(ContourSelectTest, ComputeContourFeature_NumPoints) {
    EXPECT_EQ(ComputeContourFeature(testContours_[0], ContourFeature::NumPoints), 4.0);
    EXPECT_EQ(ComputeContourFeature(testContours_[3], ContourFeature::NumPoints), 100.0);
}

TEST_F(ContourSelectTest, ComputeContourFeature_Circularity) {
    // Square has circularity ~ 0.785
    double circSquare = ComputeContourFeature(testContours_[0], ContourFeature::Circularity);
    EXPECT_NEAR(circSquare, M_PI / 4.0, 0.05);

    // Circle has circularity ~ 1.0
    double circCircle = ComputeContourFeature(testContours_[3], ContourFeature::Circularity);
    EXPECT_NEAR(circCircle, 1.0, 0.05);
}

TEST_F(ContourSelectTest, ComputeContourFeature_Centroid) {
    // Small square centroid at (5, 5)
    double row = ComputeContourFeature(testContours_[0], ContourFeature::CentroidRow);
    double col = ComputeContourFeature(testContours_[0], ContourFeature::CentroidCol);
    EXPECT_NEAR(row, 5.0, 0.1);
    EXPECT_NEAR(col, 5.0, 0.1);

    // Circle centroid at (200, 200)
    double circleRow = ComputeContourFeature(testContours_[3], ContourFeature::CentroidRow);
    double circleCol = ComputeContourFeature(testContours_[3], ContourFeature::CentroidCol);
    EXPECT_NEAR(circleRow, 200.0, 1.0);
    EXPECT_NEAR(circleCol, 200.0, 1.0);
}

TEST_F(ContourSelectTest, ComputeContourFeature_BoundingBox) {
    // Small square: 10x10
    double width0 = ComputeContourFeature(testContours_[0], ContourFeature::BoundingBoxWidth);
    double height0 = ComputeContourFeature(testContours_[0], ContourFeature::BoundingBoxHeight);
    EXPECT_NEAR(width0, 10.0, 0.1);
    EXPECT_NEAR(height0, 10.0, 0.1);

    // Rectangle: 20x10
    double width2 = ComputeContourFeature(testContours_[2], ContourFeature::BoundingBoxWidth);
    double height2 = ComputeContourFeature(testContours_[2], ContourFeature::BoundingBoxHeight);
    EXPECT_NEAR(width2, 20.0, 0.1);
    EXPECT_NEAR(height2, 10.0, 0.1);
}

TEST_F(ContourSelectTest, ComputeContourFeatures_Multiple) {
    std::vector<ContourFeature> features = {
        ContourFeature::Length,
        ContourFeature::Area,
        ContourFeature::NumPoints
    };

    auto values = ComputeContourFeatures(testContours_[0], features);

    ASSERT_EQ(values.size(), 3);
    EXPECT_NEAR(values[0], 40.0, 0.1);   // Length
    EXPECT_NEAR(values[1], 100.0, 0.1);  // Area
    EXPECT_EQ(values[2], 4.0);           // NumPoints
}

// =============================================================================
// Single-Feature Selection Tests
// =============================================================================

TEST_F(ContourSelectTest, SelectContoursByLength_All) {
    // Select all (no constraint)
    auto result = SelectContoursByLength(testContours_);
    EXPECT_EQ(result.Size(), testContours_.Size());
}

TEST_F(ContourSelectTest, SelectContoursByLength_Range) {
    // Select contours with length between 50 and 100
    auto result = SelectContoursByLength(testContours_, 50.0, 100.0);

    // Should include: rectangle (60)
    EXPECT_GE(result.Size(), 1);

    for (size_t i = 0; i < result.Size(); ++i) {
        double len = ContourLength(result[i]);
        EXPECT_GE(len, 50.0);
        EXPECT_LE(len, 100.0);
    }
}

TEST_F(ContourSelectTest, SelectContoursByArea_Range) {
    // Select contours with area between 50 and 500
    auto result = SelectContoursByArea(testContours_, 50.0, 500.0);

    for (size_t i = 0; i < result.Size(); ++i) {
        double area = ContourArea(result[i]);
        EXPECT_GE(area, 50.0);
        EXPECT_LE(area, 500.0);
    }
}

TEST_F(ContourSelectTest, SelectContoursByNumPoints) {
    // Select contours with 4-10 points
    auto result = SelectContoursByNumPoints(testContours_, 4, 10);

    for (size_t i = 0; i < result.Size(); ++i) {
        EXPECT_GE(result[i].Size(), 4);
        EXPECT_LE(result[i].Size(), 10);
    }
}

TEST_F(ContourSelectTest, SelectContoursByCircularity) {
    // Select nearly circular contours (circularity > 0.9)
    auto result = SelectContoursByCircularity(testContours_, 0.9, 1.0);

    // Should include the circle approximation
    EXPECT_GE(result.Size(), 1);

    for (size_t i = 0; i < result.Size(); ++i) {
        double circ = ContourCircularity(result[i]);
        EXPECT_GE(circ, 0.9);
    }
}

TEST_F(ContourSelectTest, SelectContoursByConvexity) {
    // Select convex contours (convexity > 0.99)
    auto result = SelectContoursByConvexity(testContours_, 0.99, 1.0);

    // Should include squares, rectangle, circle but not L-shape
    for (size_t i = 0; i < result.Size(); ++i) {
        double conv = ContourConvexity(result[i]);
        EXPECT_GE(conv, 0.99);
    }
}

TEST_F(ContourSelectTest, SelectContoursByElongation) {
    // Select elongated contours (elongation > 0.5)
    auto result = SelectContoursByElongation(testContours_, 0.5, 1.0);

    // Should include thin rectangle
    EXPECT_GE(result.Size(), 1);
}

// =============================================================================
// Generic Selection Tests
// =============================================================================

TEST_F(ContourSelectTest, SelectContoursByFeature) {
    // Test generic function
    auto result = SelectContoursByFeature(testContours_, ContourFeature::Area, 100.0, 1000.0);

    for (size_t i = 0; i < result.Size(); ++i) {
        double area = ContourArea(result[i]);
        EXPECT_GE(area, 100.0);
        EXPECT_LE(area, 1000.0);
    }
}

TEST_F(ContourSelectTest, SelectContoursByCriteria_And) {
    // Select contours that satisfy both: area > 50 AND length < 100
    std::vector<SelectionCriterion> criteria = {
        {ContourFeature::Area, 50.0, SELECT_MAX_DEFAULT},
        {ContourFeature::Length, SELECT_MIN_DEFAULT, 100.0}
    };

    auto result = SelectContoursByCriteria(testContours_, criteria, SelectionLogic::And);

    for (size_t i = 0; i < result.Size(); ++i) {
        double area = ContourArea(result[i]);
        double length = ContourLength(result[i]);
        EXPECT_GE(area, 50.0);
        EXPECT_LE(length, 100.0);
    }
}

TEST_F(ContourSelectTest, SelectContoursByCriteria_Or) {
    // Select contours that satisfy: area > 5000 OR length > 300
    std::vector<SelectionCriterion> criteria = {
        {ContourFeature::Area, 5000.0, SELECT_MAX_DEFAULT},
        {ContourFeature::Length, 300.0, SELECT_MAX_DEFAULT}
    };

    auto result = SelectContoursByCriteria(testContours_, criteria, SelectionLogic::Or);

    for (size_t i = 0; i < result.Size(); ++i) {
        double area = ContourArea(result[i]);
        double length = ContourLength(result[i]);
        EXPECT_TRUE(area >= 5000.0 || length >= 300.0);
    }
}

TEST_F(ContourSelectTest, SelectContoursByCriteria_Empty) {
    // No criteria should return all
    std::vector<SelectionCriterion> criteria;
    auto result = SelectContoursByCriteria(testContours_, criteria);
    EXPECT_EQ(result.Size(), testContours_.Size());
}

TEST_F(ContourSelectTest, SelectContoursIf_Lambda) {
    // Select using custom predicate
    auto result = SelectContoursIf(testContours_,
        [](const QContour& c) {
            return c.Size() == 4;
        });

    for (size_t i = 0; i < result.Size(); ++i) {
        EXPECT_EQ(result[i].Size(), 4);
    }
}

// =============================================================================
// Index-Based Selection Tests
// =============================================================================

TEST_F(ContourSelectTest, SelectContoursByIndex) {
    std::vector<size_t> indices = {0, 2, 4};
    auto result = SelectContoursByIndex(testContours_, indices);

    EXPECT_EQ(result.Size(), 3);
}

TEST_F(ContourSelectTest, SelectContoursByIndex_InvalidIndices) {
    std::vector<size_t> indices = {0, 100, 200};  // 100 and 200 are invalid
    auto result = SelectContoursByIndex(testContours_, indices);

    EXPECT_EQ(result.Size(), 1);  // Only index 0 is valid
}

TEST_F(ContourSelectTest, SelectContourRange) {
    auto result = SelectContourRange(testContours_, 1, 4);

    EXPECT_EQ(result.Size(), 3);  // indices 1, 2, 3
}

TEST_F(ContourSelectTest, SelectFirstContours) {
    auto result = SelectFirstContours(testContours_, 3);
    EXPECT_EQ(result.Size(), 3);
}

TEST_F(ContourSelectTest, SelectFirstContours_MoreThanSize) {
    auto result = SelectFirstContours(testContours_, 100);
    EXPECT_EQ(result.Size(), testContours_.Size());
}

TEST_F(ContourSelectTest, SelectLastContours) {
    auto result = SelectLastContours(testContours_, 2);
    EXPECT_EQ(result.Size(), 2);
}

// =============================================================================
// Sorting and Ranking Tests
// =============================================================================

TEST_F(ContourSelectTest, SortContoursByFeature_Ascending) {
    auto sorted = SortContoursByFeature(testContours_, ContourFeature::Area, true);

    EXPECT_EQ(sorted.Size(), testContours_.Size());

    // Verify ascending order
    for (size_t i = 1; i < sorted.Size(); ++i) {
        double prev = ContourArea(sorted[i - 1]);
        double curr = ContourArea(sorted[i]);
        EXPECT_LE(prev, curr);
    }
}

TEST_F(ContourSelectTest, SortContoursByFeature_Descending) {
    auto sorted = SortContoursByFeature(testContours_, ContourFeature::Length, false);

    // Verify descending order
    for (size_t i = 1; i < sorted.Size(); ++i) {
        double prev = ContourLength(sorted[i - 1]);
        double curr = ContourLength(sorted[i]);
        EXPECT_GE(prev, curr);
    }
}

TEST_F(ContourSelectTest, SelectTopContoursByFeature_Largest) {
    auto top3 = SelectTopContoursByFeature(testContours_, ContourFeature::Area, 3, true);

    EXPECT_EQ(top3.Size(), 3);

    // Verify these are the 3 largest
    double minTopArea = ContourArea(top3[0]);
    for (size_t i = 1; i < top3.Size(); ++i) {
        minTopArea = std::min(minTopArea, ContourArea(top3[i]));
    }

    // Count how many contours in original have area >= minTopArea
    int countLarger = 0;
    for (size_t i = 0; i < testContours_.Size(); ++i) {
        if (ContourArea(testContours_[i]) >= minTopArea) {
            countLarger++;
        }
    }
    EXPECT_GE(countLarger, 3);
}

TEST_F(ContourSelectTest, SelectTopContoursByFeature_Smallest) {
    auto smallest2 = SelectTopContoursByFeature(testContours_, ContourFeature::Area, 2, false);

    EXPECT_EQ(smallest2.Size(), 2);
}

// =============================================================================
// Spatial Selection Tests
// =============================================================================

TEST_F(ContourSelectTest, SelectContoursInRect) {
    // Select contours with centroid in rect [0, 50] x [0, 50]
    auto result = SelectContoursInRect(testContours_, 0, 50, 0, 50);

    for (size_t i = 0; i < result.Size(); ++i) {
        Point2d centroid = ContourCentroid(result[i]);
        EXPECT_GE(centroid.y, 0);
        EXPECT_LE(centroid.y, 50);
        EXPECT_GE(centroid.x, 0);
        EXPECT_LE(centroid.x, 50);
    }
}

TEST_F(ContourSelectTest, SelectContoursInCircle) {
    // Select contours with centroid within 30 pixels of origin
    auto result = SelectContoursInCircle(testContours_, 0, 0, 30);

    for (size_t i = 0; i < result.Size(); ++i) {
        Point2d centroid = ContourCentroid(result[i]);
        double dist = std::hypot(centroid.x, centroid.y);
        EXPECT_LE(dist, 30);
    }
}

TEST_F(ContourSelectTest, SelectContoursInCircle_CircleContour) {
    // Circle contour centroid at (200, 200)
    // Select with center at (200, 200) and radius 10
    auto result = SelectContoursInCircle(testContours_, 200, 200, 10);

    EXPECT_GE(result.Size(), 1);
}

// =============================================================================
// Closed/Open Selection Tests
// =============================================================================

TEST_F(ContourSelectTest, SelectClosedContours) {
    auto closed = SelectClosedContours(testContours_);

    for (size_t i = 0; i < closed.Size(); ++i) {
        EXPECT_TRUE(closed[i].IsClosed());
    }
}

TEST_F(ContourSelectTest, SelectOpenContours) {
    auto open = SelectOpenContours(testContours_);

    for (size_t i = 0; i < open.Size(); ++i) {
        EXPECT_FALSE(open[i].IsClosed());
    }

    // We have one open contour in test set
    EXPECT_GE(open.Size(), 1);
}

// =============================================================================
// Utility Function Tests
// =============================================================================

TEST_F(ContourSelectTest, GetContourIndicesByFeature) {
    auto indices = GetContourIndicesByFeature(testContours_, ContourFeature::Area, 100.0, 1000.0);

    for (size_t idx : indices) {
        EXPECT_LT(idx, testContours_.Size());
        double area = ContourArea(testContours_[idx]);
        EXPECT_GE(area, 100.0);
        EXPECT_LE(area, 1000.0);
    }
}

TEST_F(ContourSelectTest, PartitionContoursByFeature) {
    QContourArray below, aboveOrEqual;
    PartitionContoursByFeature(testContours_, ContourFeature::Area, 500.0, below, aboveOrEqual);

    for (size_t i = 0; i < below.Size(); ++i) {
        EXPECT_LT(ContourArea(below[i]), 500.0);
    }

    for (size_t i = 0; i < aboveOrEqual.Size(); ++i) {
        EXPECT_GE(ContourArea(aboveOrEqual[i]), 500.0);
    }

    // Total should equal original
    EXPECT_EQ(below.Size() + aboveOrEqual.Size(), testContours_.Size());
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST_F(ContourSelectTest, EmptyContourArray) {
    QContourArray empty;

    auto result = SelectContoursByLength(empty, 0, 100);
    EXPECT_EQ(result.Size(), 0);

    result = SelectContoursByFeature(empty, ContourFeature::Area, 0, 100);
    EXPECT_EQ(result.Size(), 0);

    result = SelectContoursInRect(empty, 0, 100, 0, 100);
    EXPECT_EQ(result.Size(), 0);
}

TEST_F(ContourSelectTest, SingleContour) {
    QContourArray single;
    single.Add(testContours_[0]);

    auto result = SelectContoursByLength(single, 0, 100);
    EXPECT_EQ(result.Size(), 1);

    result = SelectContoursByLength(single, 100, 200);  // Out of range
    EXPECT_EQ(result.Size(), 0);
}

TEST_F(ContourSelectTest, NoMatches) {
    // Select contours with impossible criteria
    auto result = SelectContoursByArea(testContours_, 1e10, 1e11);
    EXPECT_EQ(result.Size(), 0);
}

TEST_F(ContourSelectTest, SelectionCriterion_Passes) {
    SelectionCriterion crit(ContourFeature::Area, 10.0, 100.0);

    EXPECT_TRUE(crit.Passes(50.0));
    EXPECT_TRUE(crit.Passes(10.0));   // Inclusive
    EXPECT_TRUE(crit.Passes(100.0));  // Inclusive
    EXPECT_FALSE(crit.Passes(5.0));
    EXPECT_FALSE(crit.Passes(200.0));
}

// =============================================================================
// Performance Tests (basic)
// =============================================================================

TEST_F(ContourSelectTest, ManyContours) {
    // Create array with many contours
    QContourArray many;
    for (int i = 0; i < 1000; ++i) {
        QContour c;
        double size = 10.0 + i * 0.1;
        c.SetPoints(std::vector<Point2d>{{0, 0}, {size, 0}, {size, size}, {0, size}});
        c.SetClosed(true);
        many.Add(c);
    }

    // Should complete quickly
    auto result = SelectContoursByArea(many, 500.0, 1000.0);
    EXPECT_GT(result.Size(), 0);

    // Sort should complete
    auto sorted = SortContoursByFeature(many, ContourFeature::Area, true);
    EXPECT_EQ(sorted.Size(), many.Size());

    // Multi-criteria
    std::vector<SelectionCriterion> criteria = {
        {ContourFeature::Area, 500.0, 2000.0},
        {ContourFeature::Length, 100.0, 200.0}
    };
    auto filtered = SelectContoursByCriteria(many, criteria, SelectionLogic::And);
    EXPECT_GE(filtered.Size(), 0);
}

} // namespace
} // namespace Qi::Vision::Internal
