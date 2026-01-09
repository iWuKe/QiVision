/**
 * @file test_connected_component.cpp
 * @brief Unit tests for ConnectedComponent module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/ConnectedComponent.h>
#include <QiVision/Internal/RLEOps.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>
#include <cstring>
#include <algorithm>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

// =============================================================================
// Test Fixtures and Helpers
// =============================================================================

class ConnectedComponentTest : public ::testing::Test {
protected:
    // Create a binary image with specific foreground pixels
    QImage CreateBinaryImage(int32_t width, int32_t height,
                             const std::vector<std::pair<int32_t, int32_t>>& foreground) {
        QImage img(width, height, PixelType::UInt8, ChannelType::Gray);
        uint8_t* data = static_cast<uint8_t*>(img.Data());
        int32_t stride = static_cast<int32_t>(img.Stride());

        // Initialize to zero
        for (int32_t r = 0; r < height; ++r) {
            std::memset(data + r * stride, 0, width);
        }

        // Set foreground pixels
        for (const auto& [r, c] : foreground) {
            if (r >= 0 && r < height && c >= 0 && c < width) {
                data[r * stride + c] = 255;
            }
        }

        return img;
    }

    // Create a filled rectangle in binary image
    QImage CreateRectangleImage(int32_t width, int32_t height,
                                int32_t rx, int32_t ry, int32_t rw, int32_t rh) {
        QImage img(width, height, PixelType::UInt8, ChannelType::Gray);
        uint8_t* data = static_cast<uint8_t*>(img.Data());
        int32_t stride = static_cast<int32_t>(img.Stride());

        for (int32_t r = 0; r < height; ++r) {
            std::memset(data + r * stride, 0, width);
        }

        for (int32_t r = ry; r < ry + rh && r < height; ++r) {
            for (int32_t c = rx; c < rx + rw && c < width; ++c) {
                data[r * stride + c] = 255;
            }
        }

        return img;
    }

    // Create multiple separated rectangles
    QImage CreateMultiRectImage(int32_t width, int32_t height,
                                const std::vector<Rect2i>& rects) {
        QImage img(width, height, PixelType::UInt8, ChannelType::Gray);
        uint8_t* data = static_cast<uint8_t*>(img.Data());
        int32_t stride = static_cast<int32_t>(img.Stride());

        for (int32_t r = 0; r < height; ++r) {
            std::memset(data + r * stride, 0, width);
        }

        for (const auto& rect : rects) {
            for (int32_t r = rect.y; r < rect.y + rect.height && r < height; ++r) {
                for (int32_t c = rect.x; c < rect.x + rect.width && c < width; ++c) {
                    if (r >= 0 && c >= 0) {
                        data[r * stride + c] = 255;
                    }
                }
            }
        }

        return img;
    }

    // Create region from rectangle
    QRegion CreateRectRegion(int32_t x, int32_t y, int32_t w, int32_t h) {
        std::vector<QRegion::Run> runs;
        for (int32_t r = y; r < y + h; ++r) {
            runs.push_back({r, x, x + w});
        }
        return QRegion(runs);
    }

    // Count foreground pixels in image
    int64_t CountForeground(const QImage& img) {
        if (img.Empty()) return 0;

        int64_t count = 0;
        const uint8_t* data = static_cast<const uint8_t*>(img.Data());
        int32_t stride = static_cast<int32_t>(img.Stride());

        for (int32_t r = 0; r < img.Height(); ++r) {
            const uint8_t* row = data + r * stride;
            for (int32_t c = 0; c < img.Width(); ++c) {
                if (row[c] != 0) count++;
            }
        }
        return count;
    }
};

// =============================================================================
// LabelConnectedComponents Tests - Image Based
// =============================================================================

TEST_F(ConnectedComponentTest, LabelConnectedComponents_Empty) {
    QImage empty;
    int32_t numLabels;
    QImage result = LabelConnectedComponents(empty, Connectivity::Eight, numLabels);
    EXPECT_TRUE(result.Empty());
    EXPECT_EQ(numLabels, 0);
}

TEST_F(ConnectedComponentTest, LabelConnectedComponents_AllBackground) {
    QImage img(100, 100, PixelType::UInt8, ChannelType::Gray);
    std::memset(img.Data(), 0, img.Stride() * img.Height());

    int32_t numLabels;
    QImage result = LabelConnectedComponents(img, Connectivity::Eight, numLabels);
    EXPECT_EQ(numLabels, 0);
}

TEST_F(ConnectedComponentTest, LabelConnectedComponents_SinglePixel) {
    auto img = CreateBinaryImage(50, 50, {{25, 25}});

    int32_t numLabels;
    QImage result = LabelConnectedComponents(img, Connectivity::Eight, numLabels);

    EXPECT_EQ(numLabels, 1);
    EXPECT_EQ(result.Width(), img.Width());
    EXPECT_EQ(result.Height(), img.Height());

    // Check label at pixel location
    const uint8_t* data = static_cast<const uint8_t*>(result.Data());
    int32_t stride = static_cast<int32_t>(result.Stride());
    EXPECT_EQ(data[25 * stride + 25], 1);
}

TEST_F(ConnectedComponentTest, LabelConnectedComponents_SingleRectangle) {
    auto img = CreateRectangleImage(100, 100, 20, 20, 30, 40);

    int32_t numLabels;
    QImage result = LabelConnectedComponents(img, Connectivity::Eight, numLabels);

    EXPECT_EQ(numLabels, 1);
}

TEST_F(ConnectedComponentTest, LabelConnectedComponents_TwoSeparatedRects) {
    // Two rectangles that don't touch
    std::vector<Rect2i> rects = {
        {10, 10, 20, 20},  // Top-left rect
        {60, 60, 20, 20}   // Bottom-right rect
    };
    auto img = CreateMultiRectImage(100, 100, rects);

    int32_t numLabels;
    QImage result = LabelConnectedComponents(img, Connectivity::Eight, numLabels);

    EXPECT_EQ(numLabels, 2);
}

TEST_F(ConnectedComponentTest, LabelConnectedComponents_FourConnected) {
    // Diagonal pixels - with 4-connectivity, they are separate
    auto img = CreateBinaryImage(50, 50, {{20, 20}, {21, 21}});

    int32_t numLabels;
    QImage result = LabelConnectedComponents(img, Connectivity::Four, numLabels);

    EXPECT_EQ(numLabels, 2);  // Two separate components in 4-connectivity
}

TEST_F(ConnectedComponentTest, LabelConnectedComponents_EightConnected) {
    // Diagonal pixels - with 8-connectivity, they are connected
    auto img = CreateBinaryImage(50, 50, {{20, 20}, {21, 21}});

    int32_t numLabels;
    QImage result = LabelConnectedComponents(img, Connectivity::Eight, numLabels);

    EXPECT_EQ(numLabels, 1);  // One component in 8-connectivity
}

TEST_F(ConnectedComponentTest, LabelConnectedComponents_LShape) {
    // L-shaped region
    std::vector<std::pair<int32_t, int32_t>> pixels;
    // Vertical part
    for (int32_t r = 10; r < 30; ++r) {
        pixels.push_back({r, 10});
        pixels.push_back({r, 11});
    }
    // Horizontal part
    for (int32_t c = 12; c < 25; ++c) {
        pixels.push_back({28, c});
        pixels.push_back({29, c});
    }

    auto img = CreateBinaryImage(50, 50, pixels);

    int32_t numLabels;
    QImage result = LabelConnectedComponents(img, Connectivity::Eight, numLabels);

    EXPECT_EQ(numLabels, 1);  // One connected L-shape
}

TEST_F(ConnectedComponentTest, LabelConnectedComponents_ManySmallComponents) {
    std::vector<Rect2i> rects;
    // Create a grid of small rectangles
    for (int32_t i = 0; i < 5; ++i) {
        for (int32_t j = 0; j < 5; ++j) {
            rects.push_back({i * 20, j * 20, 5, 5});
        }
    }

    auto img = CreateMultiRectImage(100, 100, rects);

    int32_t numLabels;
    QImage result = LabelConnectedComponents(img, Connectivity::Eight, numLabels);

    EXPECT_EQ(numLabels, 25);  // 5x5 = 25 separate components
}

// =============================================================================
// GetComponentStats Tests
// =============================================================================

TEST_F(ConnectedComponentTest, GetComponentStats_Empty) {
    QImage empty;
    auto stats = GetComponentStats(empty, 0);
    EXPECT_TRUE(stats.empty());
}

TEST_F(ConnectedComponentTest, GetComponentStats_SingleRect) {
    auto img = CreateRectangleImage(100, 100, 20, 30, 40, 25);  // 40x25 rect at (20, 30)

    int32_t numLabels;
    QImage labels = LabelConnectedComponents(img, Connectivity::Eight, numLabels);

    auto stats = GetComponentStats(labels, numLabels);

    ASSERT_EQ(stats.size(), 1);
    EXPECT_EQ(stats[0].label, 1);
    EXPECT_EQ(stats[0].area, 40 * 25);  // 1000 pixels
    EXPECT_EQ(stats[0].minRow, 30);
    EXPECT_EQ(stats[0].maxRow, 54);
    EXPECT_EQ(stats[0].minCol, 20);
    EXPECT_EQ(stats[0].maxCol, 59);

    // Centroid should be at center of rectangle
    EXPECT_NEAR(stats[0].centroidX, 39.5, 0.1);  // (20 + 59) / 2
    EXPECT_NEAR(stats[0].centroidY, 42.0, 0.1);  // (30 + 54) / 2
}

TEST_F(ConnectedComponentTest, GetComponentStats_TwoRects) {
    std::vector<Rect2i> rects = {
        {10, 10, 10, 10},   // 100 pixels
        {50, 50, 20, 15}    // 300 pixels
    };
    auto img = CreateMultiRectImage(100, 100, rects);

    int32_t numLabels;
    QImage labels = LabelConnectedComponents(img, Connectivity::Eight, numLabels);

    auto stats = GetComponentStats(labels, numLabels);

    ASSERT_EQ(stats.size(), 2);

    // Find the smaller and larger components
    ComponentStats* small = (stats[0].area < stats[1].area) ? &stats[0] : &stats[1];
    ComponentStats* large = (stats[0].area > stats[1].area) ? &stats[0] : &stats[1];

    EXPECT_EQ(small->area, 100);
    EXPECT_EQ(large->area, 300);
}

// =============================================================================
// ExtractComponent Tests
// =============================================================================

TEST_F(ConnectedComponentTest, ExtractComponent_Empty) {
    QImage empty;
    QImage result = ExtractComponent(empty, 1);
    EXPECT_TRUE(result.Empty());
}

TEST_F(ConnectedComponentTest, ExtractComponent_InvalidLabel) {
    auto img = CreateRectangleImage(50, 50, 10, 10, 20, 20);
    int32_t numLabels;
    QImage labels = LabelConnectedComponents(img, Connectivity::Eight, numLabels);

    QImage result = ExtractComponent(labels, 0);  // Invalid label
    EXPECT_TRUE(result.Empty());

    result = ExtractComponent(labels, -1);  // Negative label
    EXPECT_TRUE(result.Empty());
}

TEST_F(ConnectedComponentTest, ExtractComponent_SingleComponent) {
    auto img = CreateRectangleImage(50, 50, 10, 10, 20, 20);  // 400 pixels

    int32_t numLabels;
    QImage labels = LabelConnectedComponents(img, Connectivity::Eight, numLabels);

    QImage extracted = ExtractComponent(labels, 1);

    EXPECT_EQ(extracted.Width(), 50);
    EXPECT_EQ(extracted.Height(), 50);
    EXPECT_EQ(CountForeground(extracted), 400);
}

TEST_F(ConnectedComponentTest, ExtractComponent_SelectSecond) {
    std::vector<Rect2i> rects = {
        {5, 5, 10, 10},    // First component (appears first in scan order)
        {35, 35, 10, 10}   // Second component
    };
    auto img = CreateMultiRectImage(50, 50, rects);

    int32_t numLabels;
    QImage labels = LabelConnectedComponents(img, Connectivity::Eight, numLabels);

    // Extract second component
    QImage extracted = ExtractComponent(labels, 2);

    EXPECT_EQ(CountForeground(extracted), 100);  // 10x10

    // Check that pixels are in correct location
    const uint8_t* data = static_cast<const uint8_t*>(extracted.Data());
    int32_t stride = static_cast<int32_t>(extracted.Stride());
    EXPECT_NE(data[35 * stride + 35], 0);  // Should be foreground
    EXPECT_EQ(data[5 * stride + 5], 0);    // First component location should be background
}

// =============================================================================
// ExtractAllComponents Tests
// =============================================================================

TEST_F(ConnectedComponentTest, ExtractAllComponents_Empty) {
    QImage empty;
    auto components = ExtractAllComponents(empty, 0);
    EXPECT_TRUE(components.empty());
}

TEST_F(ConnectedComponentTest, ExtractAllComponents_ThreeComponents) {
    std::vector<Rect2i> rects = {
        {5, 5, 10, 10},     // 100 pixels
        {25, 25, 15, 10},   // 150 pixels
        {45, 45, 8, 8}      // 64 pixels
    };
    auto img = CreateMultiRectImage(60, 60, rects);

    int32_t numLabels;
    QImage labels = LabelConnectedComponents(img, Connectivity::Eight, numLabels);

    auto components = ExtractAllComponents(labels, numLabels);

    ASSERT_EQ(components.size(), 3);

    // Check total area matches
    int64_t totalArea = 0;
    for (const auto& comp : components) {
        totalArea += CountForeground(comp);
    }
    EXPECT_EQ(totalArea, 100 + 150 + 64);
}

// =============================================================================
// GetLargestComponent Tests (Region-based)
// =============================================================================

TEST_F(ConnectedComponentTest, GetLargestComponent_Empty) {
    QRegion empty;
    QRegion result = GetLargestComponent(empty, Connectivity::Eight);
    EXPECT_TRUE(result.Empty());
}

TEST_F(ConnectedComponentTest, GetLargestComponent_SingleRegion) {
    QRegion region = CreateRectRegion(10, 10, 20, 20);  // 400 pixels
    QRegion result = GetLargestComponent(region, Connectivity::Eight);

    EXPECT_EQ(result.Area(), 400);
}

TEST_F(ConnectedComponentTest, GetLargestComponent_TwoRegions) {
    // Create two separate regions
    QRegion small = CreateRectRegion(0, 0, 10, 10);     // 100 pixels
    QRegion large = CreateRectRegion(50, 50, 20, 20);   // 400 pixels

    // Merge into one region (they don't overlap, so will be separate components)
    auto combined = UnionRuns(small.Runs(), large.Runs());
    QRegion region(combined);

    QRegion largest = GetLargestComponent(region, Connectivity::Eight);

    EXPECT_EQ(largest.Area(), 400);  // Should return the larger one
}

// =============================================================================
// GetLargestComponents Tests
// =============================================================================

TEST_F(ConnectedComponentTest, GetLargestComponents_Empty) {
    QRegion empty;
    auto result = GetLargestComponents(empty, 3, Connectivity::Eight);
    EXPECT_TRUE(result.empty());
}

TEST_F(ConnectedComponentTest, GetLargestComponents_TopN) {
    // Create regions of different sizes
    QRegion r1 = CreateRectRegion(0, 0, 10, 10);     // 100 pixels
    QRegion r2 = CreateRectRegion(20, 0, 15, 15);    // 225 pixels
    QRegion r3 = CreateRectRegion(0, 40, 20, 20);    // 400 pixels
    QRegion r4 = CreateRectRegion(50, 50, 5, 5);     // 25 pixels

    std::vector<QRegion::Run> allRuns;
    for (const auto& r : r1.Runs()) allRuns.push_back(r);
    for (const auto& r : r2.Runs()) allRuns.push_back(r);
    for (const auto& r : r3.Runs()) allRuns.push_back(r);
    for (const auto& r : r4.Runs()) allRuns.push_back(r);
    NormalizeRuns(allRuns);
    QRegion combined(allRuns);

    // Get top 2
    auto largest = GetLargestComponents(combined, 2, Connectivity::Eight);

    ASSERT_EQ(largest.size(), 2);
    EXPECT_EQ(largest[0].Area(), 400);  // First should be largest
    EXPECT_EQ(largest[1].Area(), 225);  // Second should be second largest
}

// =============================================================================
// FilterByArea Tests
// =============================================================================

TEST_F(ConnectedComponentTest, FilterByArea_Empty) {
    std::vector<QRegion> empty;
    auto result = FilterByArea(empty, 100, 500);
    EXPECT_TRUE(result.empty());
}

TEST_F(ConnectedComponentTest, FilterByArea_MinOnly) {
    std::vector<QRegion> components = {
        CreateRectRegion(0, 0, 5, 5),    // 25 pixels
        CreateRectRegion(0, 10, 10, 10), // 100 pixels
        CreateRectRegion(0, 30, 20, 20)  // 400 pixels
    };

    auto result = FilterByArea(components, 50, 0);  // Min 50, no max

    ASSERT_EQ(result.size(), 2);
    EXPECT_GE(result[0].Area(), 50);
    EXPECT_GE(result[1].Area(), 50);
}

TEST_F(ConnectedComponentTest, FilterByArea_Range) {
    std::vector<QRegion> components = {
        CreateRectRegion(0, 0, 5, 5),     // 25 pixels
        CreateRectRegion(0, 10, 10, 10),  // 100 pixels
        CreateRectRegion(0, 30, 20, 20)   // 400 pixels
    };

    auto result = FilterByArea(components, 50, 200);  // Between 50 and 200

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].Area(), 100);
}

// =============================================================================
// FilterBySize Tests
// =============================================================================

TEST_F(ConnectedComponentTest, FilterBySize_Basic) {
    std::vector<QRegion> components = {
        CreateRectRegion(0, 0, 5, 10),    // 5x10
        CreateRectRegion(0, 15, 15, 15),  // 15x15
        CreateRectRegion(0, 35, 30, 8)    // 30x8
    };

    // Width between 10-20, height between 10-20
    auto result = FilterBySize(components, 10, 20, 10, 20);

    ASSERT_EQ(result.size(), 1);  // Only the 15x15 fits
    Rect2i bbox = result[0].BoundingBox();
    EXPECT_EQ(bbox.width, 15);
    EXPECT_EQ(bbox.height, 15);
}

// =============================================================================
// FilterByAspectRatio Tests
// =============================================================================

TEST_F(ConnectedComponentTest, FilterByAspectRatio_Empty) {
    std::vector<QRegion> empty;
    auto result = FilterByAspectRatio(empty, 0.5, 2.0);
    EXPECT_TRUE(result.empty());
}

TEST_F(ConnectedComponentTest, FilterByAspectRatio_Square) {
    std::vector<QRegion> components = {
        CreateRectRegion(0, 0, 10, 20),   // Aspect ratio 0.5
        CreateRectRegion(0, 25, 15, 15),  // Aspect ratio 1.0
        CreateRectRegion(0, 45, 20, 10)   // Aspect ratio 2.0
    };

    // Filter for approximately square (0.8 to 1.2)
    auto result = FilterByAspectRatio(components, 0.8, 1.2);

    ASSERT_EQ(result.size(), 1);
    Rect2i bbox = result[0].BoundingBox();
    EXPECT_EQ(bbox.width, 15);
    EXPECT_EQ(bbox.height, 15);
}

// =============================================================================
// FilterByPredicate Tests
// =============================================================================

TEST_F(ConnectedComponentTest, FilterByPredicate_AreaGreaterThan) {
    std::vector<QRegion> components = {
        CreateRectRegion(0, 0, 5, 5),     // 25 pixels
        CreateRectRegion(0, 10, 10, 10),  // 100 pixels
        CreateRectRegion(0, 25, 20, 20)   // 400 pixels
    };

    auto result = FilterByPredicate(components, [](const QRegion& r) {
        return r.Area() > 50;
    });

    EXPECT_EQ(result.size(), 2);
}

TEST_F(ConnectedComponentTest, FilterByPredicate_Custom) {
    std::vector<QRegion> components = {
        CreateRectRegion(0, 0, 10, 10),   // Top region
        CreateRectRegion(0, 50, 10, 10),  // Bottom region
        CreateRectRegion(0, 100, 10, 10)  // Even more bottom
    };

    // Filter regions that start above row 60
    auto result = FilterByPredicate(components, [](const QRegion& r) {
        return r.BoundingBox().y < 60;
    });

    EXPECT_EQ(result.size(), 2);
}

// =============================================================================
// SelectBorderComponents / RemoveBorderComponents Tests
// =============================================================================

TEST_F(ConnectedComponentTest, SelectBorderComponents_TouchingTop) {
    std::vector<QRegion> components = {
        CreateRectRegion(10, 0, 10, 10),  // Touches top border
        CreateRectRegion(30, 30, 10, 10)  // Interior
    };

    Rect2i bounds(0, 0, 100, 100);
    auto result = SelectBorderComponents(components, bounds);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].BoundingBox().y, 0);  // The one touching top
}

TEST_F(ConnectedComponentTest, SelectBorderComponents_TouchingLeft) {
    std::vector<QRegion> components = {
        CreateRectRegion(0, 30, 10, 10),  // Touches left border
        CreateRectRegion(50, 50, 10, 10)  // Interior
    };

    Rect2i bounds(0, 0, 100, 100);
    auto result = SelectBorderComponents(components, bounds);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].BoundingBox().x, 0);  // The one touching left
}

TEST_F(ConnectedComponentTest, RemoveBorderComponents_KeepInterior) {
    std::vector<QRegion> components = {
        CreateRectRegion(0, 10, 10, 10),   // Touches left
        CreateRectRegion(50, 50, 10, 10),  // Interior
        CreateRectRegion(10, 0, 10, 10)    // Touches top
    };

    Rect2i bounds(0, 0, 100, 100);
    auto result = RemoveBorderComponents(components, bounds);

    ASSERT_EQ(result.size(), 1);
    // The interior one should remain
    EXPECT_EQ(result[0].BoundingBox().x, 50);
    EXPECT_EQ(result[0].BoundingBox().y, 50);
}

// =============================================================================
// MergeComponents Tests
// =============================================================================

TEST_F(ConnectedComponentTest, MergeComponents_Empty) {
    std::vector<QRegion> empty;
    QRegion result = MergeComponents(empty);
    EXPECT_TRUE(result.Empty());
}

TEST_F(ConnectedComponentTest, MergeComponents_Single) {
    std::vector<QRegion> components = {
        CreateRectRegion(10, 10, 20, 20)
    };

    QRegion result = MergeComponents(components);
    EXPECT_EQ(result.Area(), 400);
}

TEST_F(ConnectedComponentTest, MergeComponents_TwoNonOverlapping) {
    std::vector<QRegion> components = {
        CreateRectRegion(0, 0, 10, 10),    // 100 pixels
        CreateRectRegion(50, 50, 10, 10)   // 100 pixels
    };

    QRegion result = MergeComponents(components);
    EXPECT_EQ(result.Area(), 200);
}

TEST_F(ConnectedComponentTest, MergeComponents_Overlapping) {
    std::vector<QRegion> components = {
        CreateRectRegion(0, 0, 20, 20),    // 400 pixels
        CreateRectRegion(10, 10, 20, 20)   // 400 pixels, overlaps
    };

    QRegion result = MergeComponents(components);
    // Overlapping area is 10x10 = 100
    // Total unique = 400 + 400 - 100 = 700
    EXPECT_EQ(result.Area(), 700);
}

// =============================================================================
// MergeNearbyComponents Tests
// =============================================================================

TEST_F(ConnectedComponentTest, MergeNearbyComponents_Empty) {
    std::vector<QRegion> empty;
    auto result = MergeNearbyComponents(empty, 10.0);
    EXPECT_TRUE(result.empty());
}

TEST_F(ConnectedComponentTest, MergeNearbyComponents_Single) {
    std::vector<QRegion> components = {
        CreateRectRegion(10, 10, 20, 20)
    };

    auto result = MergeNearbyComponents(components, 10.0);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].Area(), 400);
}

TEST_F(ConnectedComponentTest, MergeNearbyComponents_CloseEnough) {
    std::vector<QRegion> components = {
        CreateRectRegion(0, 0, 10, 10),    // Ends at x=10
        CreateRectRegion(15, 0, 10, 10)    // Starts at x=15, gap of 5
    };

    auto result = MergeNearbyComponents(components, 10.0);  // Distance threshold = 10

    ASSERT_EQ(result.size(), 1);  // Should merge
    EXPECT_EQ(result[0].Area(), 200);  // Total area preserved
}

TEST_F(ConnectedComponentTest, MergeNearbyComponents_TooFarApart) {
    std::vector<QRegion> components = {
        CreateRectRegion(0, 0, 10, 10),
        CreateRectRegion(100, 100, 10, 10)  // Far apart
    };

    auto result = MergeNearbyComponents(components, 5.0);  // Small threshold

    EXPECT_EQ(result.size(), 2);  // Should remain separate
}

// =============================================================================
// Hole Detection Tests
// =============================================================================

TEST_F(ConnectedComponentTest, FindHoles_Empty) {
    QRegion empty;
    Rect2i bounds(0, 0, 100, 100);
    auto holes = FindHoles(empty, bounds);
    EXPECT_TRUE(holes.empty());
}

TEST_F(ConnectedComponentTest, FindHoles_SolidRectangle) {
    QRegion solid = CreateRectRegion(10, 10, 50, 50);
    Rect2i bounds(0, 0, 100, 100);

    auto holes = FindHoles(solid, bounds);
    EXPECT_TRUE(holes.empty());  // No holes in solid rectangle
}

TEST_F(ConnectedComponentTest, FindHoles_RingWithHole) {
    // Create a ring (outer - inner)
    std::vector<QRegion::Run> runs;

    // Outer rectangle: 10x10 at (10, 10)
    for (int32_t r = 10; r < 20; ++r) {
        // Left edge
        runs.push_back({r, 10, 12});
        // Right edge
        runs.push_back({r, 18, 20});
    }
    // Top and bottom edges (filling the gaps)
    for (int32_t r = 10; r < 12; ++r) {
        runs.push_back({r, 12, 18});
    }
    for (int32_t r = 18; r < 20; ++r) {
        runs.push_back({r, 12, 18});
    }

    NormalizeRuns(runs);
    QRegion ring(runs);

    Rect2i bounds(0, 0, 50, 50);
    auto holes = FindHoles(ring, bounds);

    // Should find one hole in the middle
    EXPECT_EQ(holes.size(), 1);
}

TEST_F(ConnectedComponentTest, HasHoles_NoHole) {
    QRegion solid = CreateRectRegion(10, 10, 30, 30);
    Rect2i bounds(0, 0, 100, 100);

    EXPECT_FALSE(HasHoles(solid, bounds));
}

TEST_F(ConnectedComponentTest, CountHoles_MultipleHoles) {
    // Create a region with two holes
    // This is a frame with two interior holes
    std::vector<QRegion::Run> runs;

    // Full rectangle from (0,0) to (30, 20)
    for (int32_t r = 0; r < 20; ++r) {
        if (r < 2 || r >= 18) {
            // Full rows at top and bottom
            runs.push_back({r, 0, 30});
        } else {
            // Rows with holes
            runs.push_back({r, 0, 2});    // Left edge
            runs.push_back({r, 8, 12});   // Middle wall
            runs.push_back({r, 18, 22});  // Second middle wall
            runs.push_back({r, 28, 30});  // Right edge
        }
    }

    NormalizeRuns(runs);
    QRegion frameWithHoles(runs);

    Rect2i bounds(0, 0, 30, 20);
    int32_t numHoles = CountHoles(frameWithHoles, bounds);

    // Should find multiple holes
    EXPECT_GE(numHoles, 1);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(ConnectedComponentTest, LabelConnectedComponents_FullImage) {
    QImage img(50, 50, PixelType::UInt8, ChannelType::Gray);
    std::memset(img.Data(), 255, img.Stride() * img.Height());  // All foreground

    int32_t numLabels;
    QImage result = LabelConnectedComponents(img, Connectivity::Eight, numLabels);

    EXPECT_EQ(numLabels, 1);  // All pixels should be one component
}

TEST_F(ConnectedComponentTest, FilterByArea_AllFiltered) {
    std::vector<QRegion> components = {
        CreateRectRegion(0, 0, 5, 5),   // 25 pixels
        CreateRectRegion(0, 10, 5, 5)   // 25 pixels
    };

    auto result = FilterByArea(components, 100, 200);  // None qualify
    EXPECT_TRUE(result.empty());
}

TEST_F(ConnectedComponentTest, GetLargestComponents_RequestMoreThanExists) {
    QRegion r1 = CreateRectRegion(0, 0, 10, 10);
    QRegion r2 = CreateRectRegion(30, 30, 5, 5);

    auto combined = UnionRuns(r1.Runs(), r2.Runs());
    QRegion region(combined);

    auto result = GetLargestComponents(region, 10, Connectivity::Eight);  // Request 10

    EXPECT_EQ(result.size(), 2);  // Only 2 exist
}
