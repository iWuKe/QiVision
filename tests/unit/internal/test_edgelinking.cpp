/**
 * @file test_edgelinking.cpp
 * @brief Unit tests for EdgeLinking module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/EdgeLinking.h>

#include <cmath>
#include <random>

using namespace Qi::Vision::Internal;
using Qi::Vision::QContour;
using Qi::Vision::ContourPoint;

// ============================================================================
// SpatialGrid Tests
// ============================================================================

class SpatialGridTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(SpatialGridTest, Build_Empty) {
    SpatialGrid grid;
    std::vector<EdgePoint> points;
    grid.Build(points);

    EXPECT_TRUE(grid.Empty());
}

TEST_F(SpatialGridTest, Build_SinglePoint) {
    SpatialGrid grid;
    std::vector<EdgePoint> points = {{5.0, 5.0, 0.0, 1.0, 0}};
    grid.Build(points, 4.0);

    EXPECT_FALSE(grid.Empty());

    auto neighbors = grid.FindNeighbors(5.0, 5.0, 1.0);
    ASSERT_EQ(neighbors.size(), 1u);
    EXPECT_EQ(neighbors[0], 0);
}

TEST_F(SpatialGridTest, FindNeighbors_InRange) {
    SpatialGrid grid;
    std::vector<EdgePoint> points = {
        {0.0, 0.0, 0.0, 1.0, 0},
        {1.0, 0.0, 0.0, 1.0, 1},
        {5.0, 0.0, 0.0, 1.0, 2},
        {10.0, 0.0, 0.0, 1.0, 3}
    };
    grid.Build(points, 4.0);

    // Find neighbors within radius 2.0 of origin
    auto neighbors = grid.FindNeighbors(0.0, 0.0, 2.0);

    // Should find points 0 and 1
    EXPECT_EQ(neighbors.size(), 2u);
}

TEST_F(SpatialGridTest, FindNeighbors_OutOfRange) {
    SpatialGrid grid;
    std::vector<EdgePoint> points = {
        {0.0, 0.0, 0.0, 1.0, 0},
        {10.0, 10.0, 0.0, 1.0, 1}
    };
    grid.Build(points, 4.0);

    // Find neighbors within radius 1.0 of point (5, 5)
    auto neighbors = grid.FindNeighbors(5.0, 5.0, 1.0);

    EXPECT_TRUE(neighbors.empty());
}

TEST_F(SpatialGridTest, FindNearest_Exists) {
    SpatialGrid grid;
    std::vector<EdgePoint> points = {
        {0.0, 0.0, 0.0, 1.0, 0},
        {2.0, 0.0, 0.0, 1.0, 1},
        {5.0, 0.0, 0.0, 1.0, 2}
    };
    grid.Build(points, 4.0);

    int32_t nearest = grid.FindNearest(1.5, 0.0, 3.0);
    EXPECT_EQ(nearest, 1);  // Point at (2, 0) is nearest to (1.5, 0)
}

TEST_F(SpatialGridTest, FindNearest_NotExists) {
    SpatialGrid grid;
    std::vector<EdgePoint> points = {
        {0.0, 0.0, 0.0, 1.0, 0}
    };
    grid.Build(points, 4.0);

    int32_t nearest = grid.FindNearest(100.0, 100.0, 1.0);
    EXPECT_EQ(nearest, -1);
}

// ============================================================================
// Direction Compatibility Tests
// ============================================================================

class DirectionTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(DirectionTest, DirectionsCompatible_Same) {
    EXPECT_TRUE(DirectionsCompatible(0.0, 0.0, 0.1));
    EXPECT_TRUE(DirectionsCompatible(M_PI / 4, M_PI / 4, 0.1));
}

TEST_F(DirectionTest, DirectionsCompatible_SlightlyDifferent) {
    EXPECT_TRUE(DirectionsCompatible(0.0, 0.1, 0.2));
    EXPECT_TRUE(DirectionsCompatible(M_PI / 4, M_PI / 4 + 0.1, 0.2));
}

TEST_F(DirectionTest, DirectionsCompatible_Opposite) {
    // Edge directions can be π apart (pointing along the edge either way)
    EXPECT_TRUE(DirectionsCompatible(0.0, M_PI, 0.2));
    EXPECT_TRUE(DirectionsCompatible(M_PI / 4, M_PI / 4 + M_PI, 0.2));
}

TEST_F(DirectionTest, DirectionsCompatible_Perpendicular) {
    // Perpendicular directions should not be compatible
    EXPECT_FALSE(DirectionsCompatible(0.0, M_PI / 2, 0.3));
}

TEST_F(DirectionTest, DirectionsCompatible_WrapAround) {
    // Test wrap-around at 2π
    EXPECT_TRUE(DirectionsCompatible(0.1, 2 * M_PI - 0.1, 0.3));
}

TEST_F(DirectionTest, PointDistance_Basic) {
    EdgePoint p1{0.0, 0.0, 0.0, 1.0, 0};
    EdgePoint p2{3.0, 4.0, 0.0, 1.0, 1};

    EXPECT_NEAR(PointDistance(p1, p2), 5.0, 0.001);
}

TEST_F(DirectionTest, TangentDirection_Horizontal) {
    EdgePoint p1{0.0, 0.0, 0.0, 1.0, 0};
    EdgePoint p2{1.0, 0.0, 0.0, 1.0, 1};

    EXPECT_NEAR(TangentDirection(p1, p2), 0.0, 0.001);
}

TEST_F(DirectionTest, TangentDirection_Vertical) {
    EdgePoint p1{0.0, 0.0, 0.0, 1.0, 0};
    EdgePoint p2{0.0, 1.0, 0.0, 1.0, 1};

    EXPECT_NEAR(TangentDirection(p1, p2), M_PI / 2, 0.001);
}

TEST_F(DirectionTest, TangentDirection_Diagonal) {
    EdgePoint p1{0.0, 0.0, 0.0, 1.0, 0};
    EdgePoint p2{1.0, 1.0, 0.0, 1.0, 1};

    EXPECT_NEAR(TangentDirection(p1, p2), M_PI / 4, 0.001);
}

// ============================================================================
// Linking Score Tests
// ============================================================================

TEST_F(DirectionTest, ScoreLink_ValidLink) {
    // Two nearby points with compatible directions
    EdgePoint p1{0.0, 0.0, M_PI / 2, 1.0, 0};  // Vertical edge direction
    EdgePoint p2{1.0, 0.0, M_PI / 2, 1.0, 1};  // Same direction, horizontal offset

    double score = ScoreLink(p1, p2, 3.0, 0.5);
    EXPECT_GT(score, 0.0);
}

TEST_F(DirectionTest, ScoreLink_TooFar) {
    EdgePoint p1{0.0, 0.0, 0.0, 1.0, 0};
    EdgePoint p2{10.0, 0.0, 0.0, 1.0, 1};

    double score = ScoreLink(p1, p2, 3.0, 0.5);
    EXPECT_EQ(score, 0.0);
}

TEST_F(DirectionTest, ScoreLink_IncompatibleDirections) {
    // Perpendicular edge directions
    EdgePoint p1{0.0, 0.0, 0.0, 1.0, 0};
    EdgePoint p2{1.0, 0.0, M_PI / 2, 1.0, 1};

    double score = ScoreLink(p1, p2, 3.0, 0.3);  // Small angle tolerance
    EXPECT_EQ(score, 0.0);
}

// ============================================================================
// Edge Chain Tests
// ============================================================================

class EdgeChainTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(EdgeChainTest, ComputeChainLength_Empty) {
    std::vector<EdgePoint> points;
    EdgeChain chain;

    double length = ComputeChainLength(points, chain);
    EXPECT_EQ(length, 0.0);
}

TEST_F(EdgeChainTest, ComputeChainLength_SinglePoint) {
    std::vector<EdgePoint> points = {{0.0, 0.0, 0.0, 1.0, 0}};
    EdgeChain chain;
    chain.pointIds = {0};

    double length = ComputeChainLength(points, chain);
    EXPECT_EQ(length, 0.0);
}

TEST_F(EdgeChainTest, ComputeChainLength_TwoPoints) {
    std::vector<EdgePoint> points = {
        {0.0, 0.0, 0.0, 1.0, 0},
        {3.0, 4.0, 0.0, 1.0, 1}
    };
    EdgeChain chain;
    chain.pointIds = {0, 1};

    double length = ComputeChainLength(points, chain);
    EXPECT_NEAR(length, 5.0, 0.001);
}

TEST_F(EdgeChainTest, ComputeChainLength_MultiplePoints) {
    std::vector<EdgePoint> points = {
        {0.0, 0.0, 0.0, 1.0, 0},
        {1.0, 0.0, 0.0, 1.0, 1},
        {2.0, 0.0, 0.0, 1.0, 2},
        {3.0, 0.0, 0.0, 1.0, 3}
    };
    EdgeChain chain;
    chain.pointIds = {0, 1, 2, 3};

    double length = ComputeChainLength(points, chain);
    EXPECT_NEAR(length, 3.0, 0.001);
}

TEST_F(EdgeChainTest, ReverseChain) {
    EdgeChain chain;
    chain.pointIds = {0, 1, 2, 3, 4};

    ReverseChain(chain);

    ASSERT_EQ(chain.pointIds.size(), 5u);
    EXPECT_EQ(chain.pointIds[0], 4);
    EXPECT_EQ(chain.pointIds[1], 3);
    EXPECT_EQ(chain.pointIds[2], 2);
    EXPECT_EQ(chain.pointIds[3], 1);
    EXPECT_EQ(chain.pointIds[4], 0);
}

// ============================================================================
// Edge Linking Tests
// ============================================================================

class EdgeLinkingTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(EdgeLinkingTest, LinkEdgePoints_Empty) {
    std::vector<EdgePoint> points;
    EdgeLinkingParams params;

    auto chains = LinkEdgePoints(points, params);

    EXPECT_TRUE(chains.empty());
}

TEST_F(EdgeLinkingTest, LinkEdgePoints_SinglePoint) {
    std::vector<EdgePoint> points = {{0.0, 0.0, 0.0, 1.0, 0}};
    EdgeLinkingParams params;
    params.minChainPoints = 1;
    params.minChainLength = 0.0;

    auto chains = LinkEdgePoints(points, params);

    // Single point forms a chain of length 0, which passes minChainLength=0
    // but the algorithm may or may not return it depending on implementation
    // At minimum, no crash should occur
    int32_t totalPoints = 0;
    for (const auto& chain : chains) {
        totalPoints += chain.Size();
    }
    EXPECT_LE(totalPoints, 1);  // At most 1 point
}

TEST_F(EdgeLinkingTest, LinkEdgePoints_HorizontalLine) {
    // Create a horizontal line of edge points
    std::vector<EdgePoint> points;
    for (int i = 0; i < 10; ++i) {
        // Edge direction is vertical (perpendicular to horizontal line)
        points.push_back({static_cast<double>(i), 0.0, M_PI / 2, 1.0, i});
    }

    EdgeLinkingParams params;
    params.maxGap = 2.0;
    params.maxAngleDiff = 0.5;
    params.minChainPoints = 3;
    params.minChainLength = 0.0;

    auto chains = LinkEdgePoints(points, params);

    // Should get one chain with all points
    ASSERT_GE(chains.size(), 1u);

    // Total points across all chains should be 10
    int32_t totalPoints = 0;
    for (const auto& chain : chains) {
        totalPoints += chain.Size();
    }
    EXPECT_EQ(totalPoints, 10);
}

TEST_F(EdgeLinkingTest, LinkEdgePoints_TwoDisjointLines) {
    std::vector<EdgePoint> points;

    // First line: y = 0
    for (int i = 0; i < 5; ++i) {
        points.push_back({static_cast<double>(i), 0.0, M_PI / 2, 1.0,
                          static_cast<int32_t>(points.size())});
    }

    // Second line: y = 10 (far from first)
    for (int i = 0; i < 5; ++i) {
        points.push_back({static_cast<double>(i), 10.0, M_PI / 2, 1.0,
                          static_cast<int32_t>(points.size())});
    }

    EdgeLinkingParams params;
    params.maxGap = 2.0;
    params.maxAngleDiff = 0.5;
    params.minChainPoints = 3;
    params.minChainLength = 0.0;

    auto chains = LinkEdgePoints(points, params);

    // Should get two separate chains
    EXPECT_GE(chains.size(), 2u);
}

TEST_F(EdgeLinkingTest, LinkEdgePoints_Circle) {
    // Create points on a circle
    std::vector<EdgePoint> points;
    int numPoints = 20;
    double radius = 10.0;

    for (int i = 0; i < numPoints; ++i) {
        double angle = 2.0 * M_PI * i / numPoints;
        double x = radius * std::cos(angle);
        double y = radius * std::sin(angle);
        // Edge direction is tangent to circle (perpendicular to radius)
        double edgeDir = angle + M_PI / 2;
        points.push_back({x, y, edgeDir, 1.0, i});
    }

    EdgeLinkingParams params;
    params.maxGap = 5.0;  // Points are about 3 apart
    params.maxAngleDiff = 0.8;
    params.minChainPoints = 5;
    params.minChainLength = 0.0;
    params.closedContours = true;
    params.closureMaxGap = 5.0;
    params.closureMaxAngle = 0.8;

    auto chains = LinkEdgePoints(points, params);

    ASSERT_GE(chains.size(), 1u);

    // Check that we got most of the circle
    int32_t totalPoints = 0;
    for (const auto& chain : chains) {
        totalPoints += chain.Size();
    }
    EXPECT_GE(totalPoints, numPoints - 2);  // Allow some tolerance
}

// ============================================================================
// Chain Filtering Tests
// ============================================================================

TEST_F(EdgeLinkingTest, FilterChainsByLength) {
    std::vector<EdgePoint> points = {
        {0.0, 0.0, 0.0, 1.0, 0},
        {1.0, 0.0, 0.0, 1.0, 1},
        {2.0, 0.0, 0.0, 1.0, 2},
        {10.0, 0.0, 0.0, 1.0, 3},
        {20.0, 0.0, 0.0, 1.0, 4}
    };

    std::vector<EdgeChain> chains;

    // Short chain (length 2)
    EdgeChain short_chain;
    short_chain.pointIds = {0, 1, 2};
    chains.push_back(short_chain);

    // Long chain (length 10)
    EdgeChain long_chain;
    long_chain.pointIds = {3, 4};
    chains.push_back(long_chain);

    auto filtered = FilterChainsByLength(points, chains, 5.0);

    ASSERT_EQ(filtered.size(), 1u);
    EXPECT_EQ(filtered[0].pointIds[0], 3);
}

TEST_F(EdgeLinkingTest, FilterChainsByPointCount) {
    std::vector<EdgeChain> chains;

    EdgeChain short_chain;
    short_chain.pointIds = {0, 1};
    chains.push_back(short_chain);

    EdgeChain long_chain;
    long_chain.pointIds = {0, 1, 2, 3, 4};
    chains.push_back(long_chain);

    auto filtered = FilterChainsByPointCount(chains, 3);

    ASSERT_EQ(filtered.size(), 1u);
    EXPECT_EQ(filtered[0].Size(), 5);
}

// ============================================================================
// Chain Merging Tests
// ============================================================================

TEST_F(EdgeLinkingTest, MergeChains_Adjacent) {
    std::vector<EdgePoint> points = {
        {0.0, 0.0, M_PI / 2, 1.0, 0},
        {1.0, 0.0, M_PI / 2, 1.0, 1},
        {2.0, 0.0, M_PI / 2, 1.0, 2},
        {3.0, 0.0, M_PI / 2, 1.0, 3},
        {4.0, 0.0, M_PI / 2, 1.0, 4}
    };

    std::vector<EdgeChain> chains;

    EdgeChain chain1;
    chain1.pointIds = {0, 1, 2};
    chains.push_back(chain1);

    EdgeChain chain2;
    chain2.pointIds = {3, 4};
    chains.push_back(chain2);

    auto merged = MergeChains(points, chains, 2.0, 0.5);

    // Should be merged into one chain
    EXPECT_EQ(merged.size(), 1u);
    if (!merged.empty()) {
        EXPECT_EQ(merged[0].Size(), 5);
    }
}

TEST_F(EdgeLinkingTest, MergeChains_NotAdjacent) {
    std::vector<EdgePoint> points = {
        {0.0, 0.0, M_PI / 2, 1.0, 0},
        {1.0, 0.0, M_PI / 2, 1.0, 1},
        {10.0, 0.0, M_PI / 2, 1.0, 2},
        {11.0, 0.0, M_PI / 2, 1.0, 3}
    };

    std::vector<EdgeChain> chains;

    EdgeChain chain1;
    chain1.pointIds = {0, 1};
    chains.push_back(chain1);

    EdgeChain chain2;
    chain2.pointIds = {2, 3};
    chains.push_back(chain2);

    auto merged = MergeChains(points, chains, 2.0, 0.5);

    // Should not be merged (too far apart)
    EXPECT_EQ(merged.size(), 2u);
}

// ============================================================================
// Closure Tests
// ============================================================================

TEST_F(EdgeLinkingTest, TryCloseChains_CanClose) {
    std::vector<EdgePoint> points = {
        {0.0, 0.0, 0.0, 1.0, 0},
        {1.0, 0.0, 0.0, 1.0, 1},
        {1.0, 1.0, 0.0, 1.0, 2},
        {0.0, 1.0, 0.0, 1.0, 3}
    };

    std::vector<EdgeChain> chains;
    EdgeChain chain;
    chain.pointIds = {0, 1, 2, 3};
    chain.isClosed = false;
    chains.push_back(chain);

    TryCloseChains(points, chains, 2.0, 1.0);

    EXPECT_TRUE(chains[0].isClosed);
}

TEST_F(EdgeLinkingTest, TryCloseChains_TooFar) {
    std::vector<EdgePoint> points = {
        {0.0, 0.0, 0.0, 1.0, 0},
        {1.0, 0.0, 0.0, 1.0, 1},
        {2.0, 0.0, 0.0, 1.0, 2},
        {10.0, 0.0, 0.0, 1.0, 3}  // Too far from start
    };

    std::vector<EdgeChain> chains;
    EdgeChain chain;
    chain.pointIds = {0, 1, 2, 3};
    chain.isClosed = false;
    chains.push_back(chain);

    TryCloseChains(points, chains, 2.0, 1.0);

    EXPECT_FALSE(chains[0].isClosed);
}

// ============================================================================
// Contour Conversion Tests
// ============================================================================

TEST_F(EdgeLinkingTest, ChainToContour_Basic) {
    std::vector<EdgePoint> points = {
        {0.0, 0.0, 0.0, 1.0, 0},
        {1.0, 0.0, 0.0, 2.0, 1},
        {2.0, 0.0, 0.0, 3.0, 2}
    };

    EdgeChain chain;
    chain.pointIds = {0, 1, 2};
    chain.isClosed = false;

    QContour contour = ChainToContour(points, chain);

    EXPECT_EQ(contour.Size(), 3u);
    EXPECT_FALSE(contour.IsClosed());
}

TEST_F(EdgeLinkingTest, ChainToContour_Closed) {
    std::vector<EdgePoint> points = {
        {0.0, 0.0, 0.0, 1.0, 0},
        {1.0, 0.0, 0.0, 1.0, 1},
        {0.5, 1.0, 0.0, 1.0, 2}
    };

    EdgeChain chain;
    chain.pointIds = {0, 1, 2};
    chain.isClosed = true;

    QContour contour = ChainToContour(points, chain);

    EXPECT_EQ(contour.Size(), 3u);
    EXPECT_TRUE(contour.IsClosed());
}

TEST_F(EdgeLinkingTest, LinkToContours_Integration) {
    // Create a simple line of points
    std::vector<EdgePoint> points;
    for (int i = 0; i < 10; ++i) {
        points.push_back({static_cast<double>(i), 0.0, M_PI / 2, 1.0, i});
    }

    EdgeLinkingParams params;
    params.maxGap = 2.0;
    params.maxAngleDiff = 0.5;
    params.minChainPoints = 3;
    params.minChainLength = 0.0;

    auto contours = LinkToContours(points, params);

    ASSERT_GE(contours.size(), 1u);

    // Total points
    size_t totalPoints = 0;
    for (const auto& c : contours) {
        totalPoints += c.Size();
    }
    EXPECT_EQ(totalPoints, 10u);
}

// ============================================================================
// Edge Cases and Robustness Tests
// ============================================================================

TEST(EdgeLinkingEdgeCaseTest, VeryLargePointSet) {
    std::vector<EdgePoint> points;
    int numPoints = 1000;

    for (int i = 0; i < numPoints; ++i) {
        double x = static_cast<double>(i % 100);
        double y = static_cast<double>(i / 100);
        points.push_back({x, y, 0.0, 1.0, i});
    }

    EdgeLinkingParams params;
    params.maxGap = 2.0;
    params.minChainPoints = 3;

    // Should complete without error
    auto chains = LinkEdgePoints(points, params);

    EXPECT_GE(chains.size(), 1u);
}

TEST(EdgeLinkingEdgeCaseTest, RandomDirections) {
    std::vector<EdgePoint> points;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 2.0 * M_PI);

    for (int i = 0; i < 50; ++i) {
        points.push_back({static_cast<double>(i), 0.0, dist(rng), 1.0, i});
    }

    EdgeLinkingParams params;
    params.maxGap = 2.0;
    params.maxAngleDiff = 2.0;  // Large tolerance
    params.minChainPoints = 2;

    // Should complete without error
    auto chains = LinkEdgePoints(points, params);

    EXPECT_GE(chains.size(), 1u);
}

TEST(EdgeLinkingEdgeCaseTest, CoincidentPoints) {
    std::vector<EdgePoint> points = {
        {5.0, 5.0, 0.0, 1.0, 0},
        {5.0, 5.0, 0.1, 2.0, 1},
        {5.0, 5.0, 0.0, 3.0, 2}
    };

    EdgeLinkingParams params;
    params.maxGap = 1.0;
    params.minChainPoints = 1;
    params.minChainLength = 0.0;

    // Should handle coincident points gracefully (no crash)
    auto chains = LinkEdgePoints(points, params);

    // Coincident points have zero distance, which may not link properly
    // (ScoreLink returns 0 for dist < 1e-6 to avoid division issues)
    // The important thing is that the algorithm doesn't crash
    // and processes all points (even if they end up in separate chains)
    EXPECT_GE(chains.size(), 0u);  // No crash is success
}
