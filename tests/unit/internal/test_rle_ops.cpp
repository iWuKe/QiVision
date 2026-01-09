/**
 * @file test_rle_ops.cpp
 * @brief Unit tests for RLEOps module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/RLEOps.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>
#include <cmath>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

// =============================================================================
// Test Fixtures
// =============================================================================

class RLEOpsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Create test image with a white rectangle
    QImage CreateRectImage(int32_t width, int32_t height,
                           int32_t rectX, int32_t rectY,
                           int32_t rectW, int32_t rectH,
                           uint8_t bgVal = 0, uint8_t fgVal = 255) {
        QImage img(width, height, PixelType::UInt8, ChannelType::Gray);
        for (int32_t y = 0; y < height; ++y) {
            auto* row = static_cast<uint8_t*>(img.RowPtr(y));
            for (int32_t x = 0; x < width; ++x) {
                if (x >= rectX && x < rectX + rectW &&
                    y >= rectY && y < rectY + rectH) {
                    row[x] = fgVal;
                } else {
                    row[x] = bgVal;
                }
            }
        }
        return img;
    }

    // Create test image with a circle
    QImage CreateCircleImage(int32_t size, int32_t cx, int32_t cy, int32_t radius,
                              uint8_t bgVal = 0, uint8_t fgVal = 255) {
        QImage img(size, size, PixelType::UInt8, ChannelType::Gray);
        for (int32_t y = 0; y < size; ++y) {
            auto* row = static_cast<uint8_t*>(img.RowPtr(y));
            for (int32_t x = 0; x < size; ++x) {
                double dx = x - cx;
                double dy = y - cy;
                if (dx * dx + dy * dy <= radius * radius) {
                    row[x] = fgVal;
                } else {
                    row[x] = bgVal;
                }
            }
        }
        return img;
    }

    // Create rectangular region
    QRegion CreateRectRegion(int32_t x, int32_t y, int32_t w, int32_t h) {
        std::vector<QRegion::Run> runs;
        for (int32_t row = y; row < y + h; ++row) {
            runs.emplace_back(row, x, x + w);  // colEnd is exclusive
        }
        return QRegion(runs);
    }
};

// =============================================================================
// RLE Utility Tests
// =============================================================================

TEST_F(RLEOpsTest, SortRuns_UnsortedInput) {
    RunVector runs = {
        {5, 10, 20},
        {3, 5, 15},
        {3, 20, 30},
        {1, 0, 10}
    };

    SortRuns(runs);

    ASSERT_EQ(runs.size(), 4u);
    EXPECT_EQ(runs[0].row, 1);
    EXPECT_EQ(runs[1].row, 3);
    EXPECT_EQ(runs[1].colBegin, 5);
    EXPECT_EQ(runs[2].row, 3);
    EXPECT_EQ(runs[2].colBegin, 20);
    EXPECT_EQ(runs[3].row, 5);
}

TEST_F(RLEOpsTest, MergeRuns_OverlappingRuns) {
    RunVector runs = {
        {1, 0, 10},
        {1, 5, 15},   // Overlaps with previous
        {1, 20, 30},  // Gap, separate
        {2, 0, 5}
    };

    SortRuns(runs);
    MergeRuns(runs);

    ASSERT_EQ(runs.size(), 3u);
    EXPECT_EQ(runs[0].row, 1);
    EXPECT_EQ(runs[0].colBegin, 0);
    EXPECT_EQ(runs[0].colEnd, 15);  // Merged
    EXPECT_EQ(runs[1].row, 1);
    EXPECT_EQ(runs[1].colBegin, 20);
    EXPECT_EQ(runs[2].row, 2);
}

TEST_F(RLEOpsTest, MergeRuns_AdjacentRuns) {
    RunVector runs = {
        {1, 0, 10},
        {1, 10, 20},  // Adjacent (colBegin == prev.colEnd)
    };

    SortRuns(runs);
    MergeRuns(runs);

    ASSERT_EQ(runs.size(), 1u);
    EXPECT_EQ(runs[0].colBegin, 0);
    EXPECT_EQ(runs[0].colEnd, 20);
}

TEST_F(RLEOpsTest, NormalizeRuns_Full) {
    RunVector runs = {
        {3, 20, 30},
        {1, 5, 15},
        {1, 0, 10},
        {3, 5, 15},
    };

    NormalizeRuns(runs);

    ASSERT_EQ(runs.size(), 3u);
    EXPECT_EQ(runs[0].row, 1);
    EXPECT_EQ(runs[0].colBegin, 0);
    EXPECT_EQ(runs[0].colEnd, 15);  // Merged
}

TEST_F(RLEOpsTest, ValidateRuns_ValidInput) {
    RunVector runs = {
        {1, 0, 10},
        {1, 20, 30},
        {2, 0, 5}
    };

    EXPECT_TRUE(ValidateRuns(runs));
}

TEST_F(RLEOpsTest, ValidateRuns_InvalidOverlap) {
    RunVector runs = {
        {1, 0, 15},
        {1, 10, 20},  // Overlaps
    };

    EXPECT_FALSE(ValidateRuns(runs));
}

TEST_F(RLEOpsTest, TranslateRuns_PositiveOffset) {
    RunVector runs = {
        {0, 10, 20},
        {1, 5, 15}
    };

    auto translated = TranslateRuns(runs, 5, 3);

    ASSERT_EQ(translated.size(), 2u);
    EXPECT_EQ(translated[0].row, 3);
    EXPECT_EQ(translated[0].colBegin, 15);
    EXPECT_EQ(translated[0].colEnd, 25);
    EXPECT_EQ(translated[1].row, 4);
}

TEST_F(RLEOpsTest, ClipRuns_PartialClip) {
    RunVector runs = {
        {0, 0, 100},
        {5, 0, 100},
        {10, 0, 100}  // Will be clipped out
    };

    Rect2i bounds = {10, 2, 50, 6};  // x=10, y=2, w=50, h=6 (rows 2-7)
    auto clipped = ClipRuns(runs, bounds);

    // Only row 5 should remain, clipped to x=10-59 (colEnd=60 exclusive)
    ASSERT_EQ(clipped.size(), 1u);
    EXPECT_EQ(clipped[0].row, 5);
    EXPECT_EQ(clipped[0].colBegin, 10);
    EXPECT_EQ(clipped[0].colEnd, 60);  // exclusive: covers cols 10-59
}

TEST_F(RLEOpsTest, GetRunsForRow_ValidRow) {
    RunVector runs = {
        {1, 0, 10},
        {2, 5, 15},
        {2, 20, 30},
        {3, 0, 5}
    };

    auto row2Runs = GetRunsForRow(runs, 2);

    ASSERT_EQ(row2Runs.size(), 2u);
    EXPECT_EQ(row2Runs[0].colBegin, 5);
    EXPECT_EQ(row2Runs[1].colBegin, 20);
}

TEST_F(RLEOpsTest, GetRowRange_Basic) {
    RunVector runs = {
        {5, 0, 10},
        {10, 5, 15},
        {3, 20, 30}
    };

    int32_t minRow, maxRow;
    GetRowRange(runs, minRow, maxRow);

    EXPECT_EQ(minRow, 3);
    EXPECT_EQ(maxRow, 10);
}

// =============================================================================
// Analysis Operation Tests
// =============================================================================

TEST_F(RLEOpsTest, ComputeArea_Rectangle) {
    RunVector runs = {
        {0, 0, 10},  // 10 pixels (cols 0-9)
        {1, 0, 10},  // 10 pixels (cols 0-9)
        {2, 0, 10}   // 10 pixels (cols 0-9)
    };

    EXPECT_EQ(ComputeArea(runs), 30);
}

TEST_F(RLEOpsTest, ComputeBoundingBox_Basic) {
    RunVector runs = {
        {2, 5, 15},   // cols 5-14
        {5, 10, 25},  // cols 10-24
        {8, 3, 20}    // cols 3-19
    };

    auto bbox = ComputeBoundingBox(runs);

    EXPECT_EQ(bbox.x, 3);
    EXPECT_EQ(bbox.y, 2);
    EXPECT_EQ(bbox.width, 22);  // 25 - 3 = 22 (exclusive colEnd)
    EXPECT_EQ(bbox.height, 7);  // 8 - 2 + 1
}

TEST_F(RLEOpsTest, ComputeCentroid_SymmetricRegion) {
    // Create 3x3 square centered at (5, 5)
    // With exclusive colEnd, {row, 4, 7} covers cols 4, 5, 6 (3 pixels)
    RunVector runs = {
        {4, 4, 7},  // cols 4, 5, 6
        {5, 4, 7},  // cols 4, 5, 6
        {6, 4, 7}   // cols 4, 5, 6
    };

    auto centroid = ComputeCentroid(runs);

    EXPECT_NEAR(centroid.x, 5.0, 0.001);  // center of cols 4, 5, 6
    EXPECT_NEAR(centroid.y, 5.0, 0.001);  // center of rows 4, 5, 6
}

TEST_F(RLEOpsTest, ComputeRectangularity_FullRectangle) {
    auto region = CreateRectRegion(0, 0, 10, 10);
    double rect = ComputeRectangularity(region);

    EXPECT_NEAR(rect, 1.0, 0.001);
}

// =============================================================================
// Set Operation Tests
// =============================================================================

TEST_F(RLEOpsTest, UnionRuns_NonOverlapping) {
    RunVector runs1 = {{0, 0, 10}};
    RunVector runs2 = {{0, 20, 30}};

    auto result = UnionRuns(runs1, runs2);

    ASSERT_EQ(result.size(), 2u);
}

TEST_F(RLEOpsTest, UnionRuns_Overlapping) {
    RunVector runs1 = {{0, 0, 15}};
    RunVector runs2 = {{0, 10, 25}};

    auto result = UnionRuns(runs1, runs2);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].colBegin, 0);
    EXPECT_EQ(result[0].colEnd, 25);
}

TEST_F(RLEOpsTest, IntersectRuns_Overlapping) {
    RunVector runs1 = {{0, 0, 20}};
    RunVector runs2 = {{0, 10, 30}};

    auto result = IntersectRuns(runs1, runs2);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].colBegin, 10);
    EXPECT_EQ(result[0].colEnd, 20);
}

TEST_F(RLEOpsTest, IntersectRuns_NonOverlapping) {
    RunVector runs1 = {{0, 0, 10}};
    RunVector runs2 = {{0, 20, 30}};

    auto result = IntersectRuns(runs1, runs2);

    EXPECT_TRUE(result.empty());
}

TEST_F(RLEOpsTest, DifferenceRuns_PartialOverlap) {
    // runs1: cols 0-29 (exclusive colEnd=30)
    // runs2: cols 10-19 (exclusive colEnd=20)
    // Result: cols 0-9 and cols 20-29
    RunVector runs1 = {{0, 0, 30}};
    RunVector runs2 = {{0, 10, 20}};

    auto result = DifferenceRuns(runs1, runs2);

    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].colBegin, 0);
    EXPECT_EQ(result[0].colEnd, 10);  // exclusive: cols 0-9
    EXPECT_EQ(result[1].colBegin, 20);
    EXPECT_EQ(result[1].colEnd, 30);  // exclusive: cols 20-29
}

TEST_F(RLEOpsTest, ComplementRuns_Basic) {
    // runs: cols 10-19 (exclusive colEnd=20)
    // bounds: x=0, y=0, w=40, h=1 (cols 0-39)
    // Complement: cols 0-9 and cols 20-39
    RunVector runs = {{0, 10, 20}};
    Rect2i bounds = {0, 0, 40, 1};

    auto result = ComplementRuns(runs, bounds);

    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].colBegin, 0);
    EXPECT_EQ(result[0].colEnd, 10);  // exclusive: cols 0-9
    EXPECT_EQ(result[1].colBegin, 20);
    EXPECT_EQ(result[1].colEnd, 40);  // exclusive: cols 20-39
}

TEST_F(RLEOpsTest, SymmetricDifferenceRuns_Basic) {
    RunVector runs1 = {{0, 0, 20}};
    RunVector runs2 = {{0, 10, 30}};

    auto result = SymmetricDifferenceRuns(runs1, runs2);

    // Should have [0-9] and [21-30]
    ASSERT_EQ(result.size(), 2u);
}

// =============================================================================
// Image to Region Conversion Tests
// =============================================================================

TEST_F(RLEOpsTest, ThresholdToRegion_BinaryMode) {
    auto img = CreateRectImage(100, 100, 20, 20, 50, 50, 0, 200);

    auto region = ThresholdToRegion(img, 100, 255, ThresholdMode::Range);

    EXPECT_EQ(region.Area(), 50 * 50);
}

TEST_F(RLEOpsTest, ThresholdToRegion_RangeMode) {
    QImage img(100, 100, PixelType::UInt8, ChannelType::Gray);
    for (int32_t y = 0; y < 100; ++y) {
        auto* row = static_cast<uint8_t*>(img.RowPtr(y));
        for (int32_t x = 0; x < 100; ++x) {
            row[x] = static_cast<uint8_t>(x);  // Gradient 0-99
        }
    }

    auto region = ThresholdToRegion(img, 30, 60, ThresholdMode::Range);

    // Columns 30-60 = 31 columns, 100 rows
    EXPECT_EQ(region.Area(), 31 * 100);
}

TEST_F(RLEOpsTest, NonZeroToRegion_Basic) {
    QImage img(50, 50, PixelType::UInt8, ChannelType::Gray);
    for (int32_t y = 0; y < 50; ++y) {
        auto* row = static_cast<uint8_t*>(img.RowPtr(y));
        for (int32_t x = 0; x < 50; ++x) {
            row[x] = (x >= 10 && x < 30 && y >= 10 && y < 30) ? 100 : 0;
        }
    }

    auto region = NonZeroToRegion(img);

    EXPECT_EQ(region.Area(), 20 * 20);
}

// =============================================================================
// Region to Image Conversion Tests
// =============================================================================

TEST_F(RLEOpsTest, PaintRegion_Basic) {
    QImage img(100, 100, PixelType::UInt8, ChannelType::Gray);
    for (int32_t y = 0; y < 100; ++y) {
        auto* row = static_cast<uint8_t*>(img.RowPtr(y));
        for (int32_t x = 0; x < 100; ++x) {
            row[x] = 0;
        }
    }

    auto region = CreateRectRegion(10, 10, 20, 20);
    PaintRegion(region, img, 128);

    auto* testRow = static_cast<uint8_t*>(img.RowPtr(15));
    EXPECT_EQ(testRow[5], 0);      // Outside
    EXPECT_EQ(testRow[15], 128);   // Inside
    EXPECT_EQ(testRow[35], 0);     // Outside
}

TEST_F(RLEOpsTest, RegionToMask_Basic) {
    auto region = CreateRectRegion(10, 10, 30, 30);

    auto mask = RegionToMask(region, 100, 100);

    EXPECT_EQ(mask.Width(), 100);
    EXPECT_EQ(mask.Height(), 100);

    auto* row = static_cast<uint8_t*>(mask.RowPtr(15));
    EXPECT_EQ(row[5], 0);     // Outside
    EXPECT_EQ(row[15], 255);  // Inside
}

// =============================================================================
// Fill Operation Tests
// =============================================================================

TEST_F(RLEOpsTest, FillHorizontalGaps_SmallGap) {
    std::vector<QRegion::Run> runs = {
        {0, 0, 11},   // columns 0-10
        {0, 15, 26}   // columns 15-25, Gap of 4 (11-14)
    };
    QRegion region(runs);

    auto filled = FillHorizontalGaps(region, 5);

    EXPECT_EQ(filled.Area(), 26);  // All filled
}

TEST_F(RLEOpsTest, FillHorizontalGaps_LargeGap) {
    std::vector<QRegion::Run> runs = {
        {0, 0, 11},   // columns 0-10
        {0, 50, 61}   // columns 50-60, Gap of 39 (11-49)
    };
    QRegion region(runs);

    auto filled = FillHorizontalGaps(region, 5);

    EXPECT_EQ(filled.Area(), 22);  // Not filled, stays separate
}

TEST_F(RLEOpsTest, FillHoles_DonutShape) {
    // Create a donut (filled circle with hole)
    std::vector<QRegion::Run> runs;
    int cx = 50, cy = 50, outerR = 30, innerR = 10;

    for (int y = 0; y < 100; ++y) {
        for (int x = 0; x < 100; ++x) {
            double dx = x - cx;
            double dy = y - cy;
            double dist2 = dx * dx + dy * dy;
            if (dist2 <= outerR * outerR && dist2 > innerR * innerR) {
                runs.emplace_back(y, x, x + 1);
            }
        }
    }
    QRegion region(runs);

    int64_t areaWithHole = region.Area();
    auto filled = FillHoles(region);
    int64_t areaFilled = filled.Area();

    // Filled area should be larger
    EXPECT_GT(areaFilled, areaWithHole);
}

// =============================================================================
// Connected Component Tests
// =============================================================================

TEST_F(RLEOpsTest, SplitConnectedComponents_TwoSeparateRects) {
    std::vector<QRegion::Run> runs;
    // First rectangle
    for (int y = 0; y < 10; ++y) {
        runs.emplace_back(y, 0, 10);  // columns 0-9
    }
    // Second rectangle (separated)
    for (int y = 0; y < 10; ++y) {
        runs.emplace_back(y, 50, 60);  // columns 50-59
    }
    QRegion region(runs);

    auto components = SplitConnectedComponents(region, Connectivity::Eight);

    EXPECT_EQ(components.size(), 2u);
}

TEST_F(RLEOpsTest, SplitConnectedComponents_SingleRect) {
    auto region = CreateRectRegion(0, 0, 50, 50);

    auto components = SplitConnectedComponents(region, Connectivity::Eight);

    EXPECT_EQ(components.size(), 1u);
    EXPECT_EQ(components[0].Area(), 50 * 50);
}

TEST_F(RLEOpsTest, IsConnected_SingleComponent) {
    auto region = CreateRectRegion(10, 10, 30, 30);

    EXPECT_TRUE(IsConnected(region, Connectivity::Eight));
}

TEST_F(RLEOpsTest, IsConnected_TwoComponents) {
    std::vector<QRegion::Run> runs = {
        {0, 0, 11},   // columns 0-10
        {0, 50, 61}   // columns 50-60, Separated on same row
    };
    QRegion region(runs);

    EXPECT_FALSE(IsConnected(region, Connectivity::Eight));
}

TEST_F(RLEOpsTest, CountConnectedComponents_Multiple) {
    std::vector<QRegion::Run> runs = {
        {0, 0, 1},    // single pixel at (0, 0)
        {0, 50, 51},  // single pixel at (0, 50)
        {50, 25, 26}  // single pixel at (50, 25)
    };
    QRegion region(runs);

    EXPECT_EQ(CountConnectedComponents(region, Connectivity::Eight), 3u);
}

// =============================================================================
// Boundary Tests
// =============================================================================

TEST_F(RLEOpsTest, ExtractBoundary_FilledRectangle) {
    auto region = CreateRectRegion(10, 10, 20, 20);

    auto boundary = ExtractBoundary(region, Connectivity::Eight);

    // Boundary should be smaller than original
    EXPECT_LT(boundary.Area(), region.Area());
    // Boundary should form a frame around the rectangle
    EXPECT_GT(boundary.Area(), 0);
}

TEST_F(RLEOpsTest, InnerBoundary_BasicRect) {
    auto region = CreateRectRegion(10, 10, 30, 30);

    auto inner = InnerBoundary(region);

    // Inner boundary should exist and be smaller
    EXPECT_GT(inner.Area(), 0);
    EXPECT_LT(inner.Area(), region.Area());
}

// =============================================================================
// Perimeter and Shape Features Tests
// =============================================================================

TEST_F(RLEOpsTest, ComputePerimeter_Square) {
    auto region = CreateRectRegion(0, 0, 10, 10);

    double perimeter = ComputePerimeter(region, Connectivity::Four);

    // 4-connected perimeter of 10x10 should be approximately 4*10 = 40
    EXPECT_NEAR(perimeter, 40.0, 5.0);
}

TEST_F(RLEOpsTest, ComputeCircularity_Circle) {
    // Create a circular region
    std::vector<QRegion::Run> runs;
    int cx = 50, cy = 50, radius = 20;
    for (int y = 0; y < 100; ++y) {
        for (int x = 0; x < 100; ++x) {
            double dx = x - cx;
            double dy = y - cy;
            if (dx * dx + dy * dy <= radius * radius) {
                runs.emplace_back(y, x, x + 1);
            }
        }
    }
    QRegion region(runs);

    double circularity = ComputeCircularity(region);

    // Discrete circles have rough boundaries, so circularity is lower
    // than ideal (1.0). We use boundary pixel count as perimeter,
    // which is larger than the true perimeter.
    EXPECT_GT(circularity, 0.5);   // Discrete circles have lower circularity
    EXPECT_LT(circularity, 1.0);
}

TEST_F(RLEOpsTest, ComputeCompactness_Square) {
    auto region = CreateRectRegion(0, 0, 20, 20);

    double compactness = ComputeCompactness(region);

    // Square has compactness = perimeter^2 / area = 80^2 / 400 = 16
    // With some boundary effects, should be reasonably close
    EXPECT_GT(compactness, 10.0);
    EXPECT_LT(compactness, 25.0);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(RLEOpsTest, EmptyRegion_Analysis) {
    QRegion empty;

    EXPECT_EQ(ComputeArea(empty.Runs()), 0);
    EXPECT_TRUE(IsConnected(empty, Connectivity::Eight));
    EXPECT_EQ(CountConnectedComponents(empty, Connectivity::Eight), 0u);
}

TEST_F(RLEOpsTest, SinglePixel_Analysis) {
    std::vector<QRegion::Run> runs = {{50, 50, 51}};  // single pixel at (50, 50)
    QRegion region(runs);

    EXPECT_EQ(region.Area(), 1);
    EXPECT_TRUE(IsConnected(region, Connectivity::Eight));

    auto centroid = ComputeCentroid(region.Runs());
    EXPECT_NEAR(centroid.x, 50.0, 0.001);
    EXPECT_NEAR(centroid.y, 50.0, 0.001);
}

TEST_F(RLEOpsTest, UnionRuns_EmptyInputs) {
    RunVector empty;
    RunVector runs = {{0, 0, 10}};

    auto result1 = UnionRuns(empty, runs);
    EXPECT_EQ(result1.size(), 1u);

    auto result2 = UnionRuns(runs, empty);
    EXPECT_EQ(result2.size(), 1u);

    auto result3 = UnionRuns(empty, empty);
    EXPECT_TRUE(result3.empty());
}

TEST_F(RLEOpsTest, IntersectRuns_EmptyInputs) {
    RunVector empty;
    RunVector runs = {{0, 0, 10}};

    auto result = IntersectRuns(empty, runs);
    EXPECT_TRUE(result.empty());
}

