/**
 * @file test_morph_binary.cpp
 * @brief Unit tests for MorphBinary module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/MorphBinary.h>
#include <QiVision/Internal/StructElement.h>
#include <QiVision/Core/QRegion.h>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

// Create a small rectangular region for testing
QRegion CreateRectRegion(int32_t x, int32_t y, int32_t width, int32_t height) {
    std::vector<QRegion::Run> runs;
    for (int32_t r = y; r < y + height; ++r) {
        runs.emplace_back(r, x, x + width);
    }
    return QRegion(runs);
}

// Create a plus-shaped region (cross)
QRegion CreateCrossRegion(int32_t cx, int32_t cy, int32_t armLength) {
    std::vector<QRegion::Run> runs;
    // Vertical arm
    for (int32_t r = cy - armLength; r <= cy + armLength; ++r) {
        runs.emplace_back(r, cx, cx + 1);
    }
    // Horizontal arm (except center which is already added)
    runs.emplace_back(cy, cx - armLength, cx + armLength + 1);
    return QRegion(runs);
}

// Check if a point is in region
bool RegionContains(const QRegion& region, int32_t row, int32_t col) {
    for (const auto& run : region.Runs()) {
        if (run.row == row && col >= run.colBegin && col < run.colEnd) {
            return true;
        }
    }
    return false;
}

} // anonymous namespace

// =============================================================================
// Basic Dilation Tests
// =============================================================================

TEST(MorphBinaryTest, Dilate_Empty) {
    QRegion empty;
    auto se = StructElement::Square(3);
    QRegion result = Dilate(empty, se);
    EXPECT_TRUE(result.Empty());
}

TEST(MorphBinaryTest, Dilate_SinglePixel_Square3) {
    // Single pixel at (5, 5)
    std::vector<QRegion::Run> runs = {{5, 5, 6}};
    QRegion region(runs);

    auto se = StructElement::Square(3);
    QRegion result = Dilate(region, se);

    // Should expand to 3x3 region around (5, 5)
    EXPECT_EQ(result.Area(), 9);

    // Check corners
    EXPECT_TRUE(RegionContains(result, 4, 4));
    EXPECT_TRUE(RegionContains(result, 4, 6));
    EXPECT_TRUE(RegionContains(result, 6, 4));
    EXPECT_TRUE(RegionContains(result, 6, 6));
}

TEST(MorphBinaryTest, Dilate_Rectangle_Square3) {
    QRegion region = CreateRectRegion(10, 10, 5, 3);
    EXPECT_EQ(region.Area(), 15);

    auto se = StructElement::Square(3);
    QRegion result = Dilate(region, se);

    // Should expand by 1 pixel on all sides
    // Original: 5x3, becomes: 7x5 = 35
    EXPECT_EQ(result.Area(), 35);
}

TEST(MorphBinaryTest, DilateRect_Basic) {
    // Single pixel
    std::vector<QRegion::Run> runs = {{10, 10, 11}};
    QRegion region(runs);

    QRegion result = DilateRect(region, 5, 3);

    // Should expand to 5x3
    EXPECT_EQ(result.Area(), 15);
}

TEST(MorphBinaryTest, DilateCircle_Basic) {
    std::vector<QRegion::Run> runs = {{10, 10, 11}};
    QRegion region(runs);

    QRegion result = DilateCircle(region, 2);

    // Circle radius 2 has about 13 pixels
    EXPECT_GT(result.Area(), 10);
}

// =============================================================================
// Basic Erosion Tests
// =============================================================================

TEST(MorphBinaryTest, Erode_Empty) {
    QRegion empty;
    auto se = StructElement::Square(3);
    QRegion result = Erode(empty, se);
    EXPECT_TRUE(result.Empty());
}

TEST(MorphBinaryTest, Erode_SinglePixel) {
    // Single pixel - erosion with 3x3 should produce empty
    std::vector<QRegion::Run> runs = {{5, 5, 6}};
    QRegion region(runs);

    auto se = StructElement::Square(3);
    QRegion result = Erode(region, se);

    EXPECT_TRUE(result.Empty());
}

TEST(MorphBinaryTest, Erode_Rectangle_Square3) {
    // 5x5 rectangle
    QRegion region = CreateRectRegion(10, 10, 5, 5);
    EXPECT_EQ(region.Area(), 25);

    auto se = StructElement::Square(3);
    QRegion result = Erode(region, se);

    // Should shrink by 1 pixel on all sides
    // Original: 5x5, becomes: 3x3 = 9
    EXPECT_EQ(result.Area(), 9);
}

TEST(MorphBinaryTest, Erode_SmallRegion_LargeSE) {
    // 3x3 region eroded with 5x5 SE -> empty
    QRegion region = CreateRectRegion(10, 10, 3, 3);

    auto se = StructElement::Square(5);
    QRegion result = Erode(region, se);

    EXPECT_TRUE(result.Empty());
}

TEST(MorphBinaryTest, ErodeRect_Basic) {
    QRegion region = CreateRectRegion(10, 10, 7, 5);

    QRegion result = ErodeRect(region, 3, 3);

    // 7x5 eroded by 3x3 -> 5x3 = 15
    EXPECT_EQ(result.Area(), 15);
}

// =============================================================================
// Opening Tests
// =============================================================================

TEST(MorphBinaryTest, Opening_Empty) {
    QRegion empty;
    auto se = StructElement::Square(3);
    QRegion result = Opening(empty, se);
    EXPECT_TRUE(result.Empty());
}

TEST(MorphBinaryTest, Opening_RemovesSmallProtrusions) {
    // Rectangle with a single pixel protrusion
    std::vector<QRegion::Run> runs;
    // Main body: 5x5 at (10, 10)
    for (int32_t r = 10; r < 15; ++r) {
        runs.emplace_back(r, 10, 15);
    }
    // Protrusion: single pixel at (9, 12)
    runs.emplace_back(9, 12, 13);
    QRegion region(runs);

    auto se = StructElement::Square(3);
    QRegion result = Opening(region, se);

    // Opening should remove the protrusion
    EXPECT_FALSE(RegionContains(result, 9, 12));
    // Main body should remain (eroded then dilated)
    EXPECT_GT(result.Area(), 0);
}

TEST(MorphBinaryTest, Opening_PreservesLargeRegion) {
    // Large rectangle should mostly survive opening
    QRegion region = CreateRectRegion(10, 10, 10, 10);

    auto se = StructElement::Square(3);
    QRegion result = Opening(region, se);

    // Opening of rectangle is smaller but significant
    EXPECT_GT(result.Area(), 50);
}

TEST(MorphBinaryTest, OpeningRect_Basic) {
    QRegion region = CreateRectRegion(10, 10, 10, 10);

    QRegion result = OpeningRect(region, 3, 3);

    // Should preserve most of the rectangle
    EXPECT_GT(result.Area(), 50);
}

// =============================================================================
// Closing Tests
// =============================================================================

TEST(MorphBinaryTest, Closing_Empty) {
    QRegion empty;
    auto se = StructElement::Square(3);
    QRegion result = Closing(empty, se);
    EXPECT_TRUE(result.Empty());
}

TEST(MorphBinaryTest, Closing_FillsSmallGaps) {
    // Two rectangles with a 1-pixel gap
    std::vector<QRegion::Run> runs;
    // Left part: 3x5 at (10, 10)
    for (int32_t r = 10; r < 15; ++r) {
        runs.emplace_back(r, 10, 13);
    }
    // Right part: 3x5 at (14, 10) - gap at column 13
    for (int32_t r = 10; r < 15; ++r) {
        runs.emplace_back(r, 14, 17);
    }
    QRegion region(runs);

    auto se = StructElement::Square(3);
    QRegion result = Closing(region, se);

    // Closing should fill the 1-pixel gap
    EXPECT_TRUE(RegionContains(result, 12, 13));
}

TEST(MorphBinaryTest, ClosingRect_Basic) {
    QRegion region = CreateRectRegion(10, 10, 10, 10);

    QRegion result = ClosingRect(region, 3, 3);

    // Closing of a rectangle should be at least as large
    EXPECT_GE(result.Area(), region.Area());
}

// =============================================================================
// Gradient Tests
// =============================================================================

TEST(MorphBinaryTest, MorphGradient_Rectangle) {
    QRegion region = CreateRectRegion(10, 10, 10, 10);

    auto se = StructElement::Square(3);
    QRegion gradient = MorphGradient(region, se);

    // Gradient is boundary-like
    // Should be non-empty and smaller than dilated region
    EXPECT_GT(gradient.Area(), 0);
    EXPECT_LT(gradient.Area(), region.Area() * 2);
}

TEST(MorphBinaryTest, InternalGradient_Basic) {
    QRegion region = CreateRectRegion(10, 10, 10, 10);

    auto se = StructElement::Square(3);
    QRegion internal = InternalGradient(region, se);

    // Internal gradient is inside the original region
    EXPECT_GT(internal.Area(), 0);
    EXPECT_LT(internal.Area(), region.Area());
}

TEST(MorphBinaryTest, ExternalGradient_Basic) {
    QRegion region = CreateRectRegion(10, 10, 10, 10);

    auto se = StructElement::Square(3);
    QRegion external = ExternalGradient(region, se);

    // External gradient is outside the original region
    EXPECT_GT(external.Area(), 0);
}

// =============================================================================
// TopHat/BlackHat Tests
// =============================================================================

TEST(MorphBinaryTest, TopHat_Empty) {
    QRegion empty;
    auto se = StructElement::Square(3);
    QRegion result = TopHat(empty, se);
    EXPECT_TRUE(result.Empty());
}

TEST(MorphBinaryTest, TopHat_ExtractsSmallBrightRegions) {
    // Small region should be mostly preserved by tophat
    std::vector<QRegion::Run> runs = {{10, 10, 12}};  // 2-pixel wide
    QRegion region(runs);

    auto se = StructElement::Square(5);  // Larger SE
    QRegion result = TopHat(region, se);

    // Small region should appear in tophat (since opening removes it)
    EXPECT_EQ(result.Area(), region.Area());
}

TEST(MorphBinaryTest, BlackHat_Empty) {
    QRegion empty;
    auto se = StructElement::Square(3);
    QRegion result = BlackHat(empty, se);
    EXPECT_TRUE(result.Empty());
}

// =============================================================================
// Hit-or-Miss Tests
// =============================================================================

TEST(MorphBinaryTest, HitOrMiss_Empty) {
    QRegion empty;
    auto hit = StructElement::Square(1);
    auto miss = StructElement::Square(1);
    Rect2i bounds(0, 0, 100, 100);

    QRegion result = HitOrMiss(empty, hit, miss, bounds);
    EXPECT_TRUE(result.Empty());
}

TEST(MorphBinaryTest, HitOrMiss_FindsPattern) {
    // Create region with isolated pixels
    std::vector<QRegion::Run> runs = {{5, 5, 6}, {10, 10, 11}, {15, 15, 16}};
    QRegion region(runs);

    // Hit: single pixel, Miss: 8-neighbors
    auto hit = StructElement::FromCoordinates({{0, 0}});
    auto miss = StructElement::FromCoordinates({{-1, -1}, {-1, 0}, {-1, 1},
                                                 {0, -1}, {0, 1},
                                                 {1, -1}, {1, 0}, {1, 1}});

    Rect2i bounds(0, 0, 20, 20);
    QRegion result = HitOrMiss(region, hit, miss, bounds);

    // Should find all 3 isolated pixels
    EXPECT_EQ(result.Area(), 3);
}

TEST(MorphBinaryTest, HitOrMiss_WithPair) {
    std::vector<QRegion::Run> runs = {{5, 5, 6}};
    QRegion region(runs);

    auto hit = StructElement::FromCoordinates({{0, 0}});
    auto miss = StructElement::FromCoordinates({{-1, 0}, {1, 0}, {0, -1}, {0, 1}});

    Rect2i bounds(0, 0, 20, 20);
    auto pair = std::make_pair(hit, miss);
    QRegion result = HitOrMiss(region, pair, bounds);

    // Should find the isolated pixel
    EXPECT_EQ(result.Area(), 1);
}

// =============================================================================
// Thinning Tests
// =============================================================================

TEST(MorphBinaryTest, ThinOnce_Basic) {
    QRegion region = CreateRectRegion(10, 10, 5, 5);

    auto hit = StructElement::FromCoordinates({{0, 0}, {-1, 0}});
    auto miss = StructElement::FromCoordinates({{1, 0}});

    Rect2i bounds(0, 0, 30, 30);
    QRegion result = ThinOnce(region, hit, miss, bounds);

    // Should remove some pixels
    EXPECT_LT(result.Area(), region.Area());
}

TEST(MorphBinaryTest, Thin_Converges) {
    QRegion region = CreateRectRegion(10, 10, 7, 7);

    QRegion result = Thin(region, 0);  // Until stable

    // Thinned region should be smaller
    EXPECT_LT(result.Area(), region.Area());
    EXPECT_GT(result.Area(), 0);
}

TEST(MorphBinaryTest, Thin_LimitedIterations) {
    QRegion region = CreateRectRegion(10, 10, 7, 7);

    QRegion result1 = Thin(region, 1);
    QRegion resultFull = Thin(region, 0);

    // Limited iterations should have more pixels
    EXPECT_GE(result1.Area(), resultFull.Area());
}

TEST(MorphBinaryTest, Skeleton_Basic) {
    QRegion region = CreateRectRegion(10, 10, 7, 7);

    QRegion skeleton = Skeleton(region);

    // Skeleton should be thinner than original
    EXPECT_LT(skeleton.Area(), region.Area());
    EXPECT_GT(skeleton.Area(), 0);
}

TEST(MorphBinaryTest, PruneSkeleton_Basic) {
    // Create a line-like region
    std::vector<QRegion::Run> runs;
    for (int32_t r = 10; r < 20; ++r) {
        runs.emplace_back(r, 15, 16);
    }
    QRegion line(runs);

    QRegion skeleton = Skeleton(line);
    QRegion pruned = PruneSkeleton(skeleton, 2);

    // Pruning may or may not reduce size depending on skeleton structure
    EXPECT_GE(skeleton.Area(), pruned.Area());
}

// =============================================================================
// Iterative Operations Tests
// =============================================================================

TEST(MorphBinaryTest, DilateN_Basic) {
    std::vector<QRegion::Run> runs = {{10, 10, 11}};
    QRegion region(runs);

    auto se = StructElement::Square(3);
    QRegion result = DilateN(region, se, 2);

    // Two dilations should expand more
    QRegion single = Dilate(region, se);
    EXPECT_GT(result.Area(), single.Area());
}

TEST(MorphBinaryTest, ErodeN_Basic) {
    QRegion region = CreateRectRegion(10, 10, 10, 10);

    auto se = StructElement::Square(3);
    QRegion result = ErodeN(region, se, 2);

    // Two erosions should shrink more
    QRegion single = Erode(region, se);
    EXPECT_LT(result.Area(), single.Area());
}

TEST(MorphBinaryTest, OpeningN_Basic) {
    QRegion region = CreateRectRegion(10, 10, 10, 10);

    auto se = StructElement::Square(3);
    QRegion result = OpeningN(region, se, 2);

    EXPECT_GT(result.Area(), 0);
}

TEST(MorphBinaryTest, ClosingN_Basic) {
    QRegion region = CreateRectRegion(10, 10, 10, 10);

    auto se = StructElement::Square(3);
    QRegion result = ClosingN(region, se, 2);

    EXPECT_GE(result.Area(), region.Area());
}

// =============================================================================
// Geodesic Operations Tests
// =============================================================================

TEST(MorphBinaryTest, GeodesicDilate_Basic) {
    // Marker: small region inside mask
    std::vector<QRegion::Run> markerRuns = {{12, 12, 13}};
    QRegion marker(markerRuns);

    // Mask: larger containing region
    QRegion mask = CreateRectRegion(10, 10, 10, 10);

    auto se = StructElement::Square(3);
    QRegion result = GeodesicDilate(marker, mask, se);

    // Should grow within mask bounds
    EXPECT_GT(result.Area(), marker.Area());
    EXPECT_LE(result.Area(), mask.Area());
}

TEST(MorphBinaryTest, ReconstructByDilation_Basic) {
    // Marker: seed inside mask
    std::vector<QRegion::Run> markerRuns = {{12, 12, 13}};
    QRegion marker(markerRuns);

    // Mask: region that marker should fill
    QRegion mask = CreateRectRegion(10, 10, 5, 5);

    QRegion result = ReconstructByDilation(marker, mask);

    // Should reconstruct to fill mask
    EXPECT_EQ(result.Area(), mask.Area());
}

TEST(MorphBinaryTest, ReconstructByDilation_Disconnected) {
    // Marker in one component
    std::vector<QRegion::Run> markerRuns = {{12, 12, 13}};
    QRegion marker(markerRuns);

    // Mask with two disconnected components
    std::vector<QRegion::Run> maskRuns;
    for (int32_t r = 10; r < 15; ++r) {
        maskRuns.emplace_back(r, 10, 15);  // Component containing marker
    }
    for (int32_t r = 10; r < 15; ++r) {
        maskRuns.emplace_back(r, 20, 25);  // Disconnected component
    }
    QRegion mask(maskRuns);

    QRegion result = ReconstructByDilation(marker, mask);

    // Should only fill the connected component
    EXPECT_EQ(result.Area(), 25);  // First component only
}

TEST(MorphBinaryTest, FillHolesByReconstruction_Basic) {
    // Create region with a hole
    std::vector<QRegion::Run> runs;
    for (int32_t r = 10; r < 20; ++r) {
        if (r == 15) {
            // Row with hole: gap at 15
            runs.emplace_back(r, 10, 15);
            runs.emplace_back(r, 16, 20);
        } else {
            runs.emplace_back(r, 10, 20);
        }
    }
    QRegion region(runs);

    QRegion filled = FillHolesByReconstruction(region);

    // Should fill the hole
    EXPECT_GE(filled.Area(), region.Area());
}

TEST(MorphBinaryTest, ClearBorder_Basic) {
    // Region touching left border
    std::vector<QRegion::Run> runs;
    for (int32_t r = 5; r < 10; ++r) {
        runs.emplace_back(r, 0, 5);  // Touches x=0 border
    }
    // Another region not touching border
    for (int32_t r = 15; r < 20; ++r) {
        runs.emplace_back(r, 10, 15);
    }
    QRegion region(runs);

    Rect2i bounds(0, 0, 30, 30);
    QRegion result = ClearBorder(region, bounds);

    // Should remove border-touching component
    // Keep the interior component
    EXPECT_LT(result.Area(), region.Area());
    EXPECT_GT(result.Area(), 0);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(MorphBinaryTest, Dilate_EmptySE) {
    QRegion region = CreateRectRegion(10, 10, 5, 5);
    StructElement emptySE;

    QRegion result = Dilate(region, emptySE);

    // Empty SE returns original
    EXPECT_EQ(result.Area(), region.Area());
}

TEST(MorphBinaryTest, Erode_EmptySE) {
    QRegion region = CreateRectRegion(10, 10, 5, 5);
    StructElement emptySE;

    QRegion result = Erode(region, emptySE);

    // Empty SE returns empty
    EXPECT_TRUE(result.Empty());
}

TEST(MorphBinaryTest, DilateRect_ZeroSize) {
    QRegion region = CreateRectRegion(10, 10, 5, 5);

    QRegion result = DilateRect(region, 0, 0);

    // Zero size returns original
    EXPECT_EQ(result.Area(), region.Area());
}

TEST(MorphBinaryTest, ThickenOnce_Basic) {
    // Small region
    std::vector<QRegion::Run> runs = {{10, 10, 11}};
    QRegion region(runs);

    auto hit = StructElement::FromCoordinates({{0, 0}});
    auto miss = StructElement::FromCoordinates({{1, 0}});

    Rect2i bounds(0, 0, 20, 20);
    QRegion result = ThickenOnce(region, hit, miss, bounds);

    // Thickening should expand
    EXPECT_GE(result.Area(), region.Area());
}

