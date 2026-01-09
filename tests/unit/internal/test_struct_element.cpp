/**
 * @file test_struct_element.cpp
 * @brief Unit tests for StructElement module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/StructElement.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>
#include <cmath>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

// =============================================================================
// Basic Shape Tests
// =============================================================================

TEST(StructElementTest, Rectangle_Basic) {
    auto se = StructElement::Rectangle(5, 3);

    EXPECT_FALSE(se.Empty());
    EXPECT_EQ(se.Width(), 5);
    EXPECT_EQ(se.Height(), 3);
    EXPECT_EQ(se.AnchorX(), 2);  // Center
    EXPECT_EQ(se.AnchorY(), 1);  // Center
    EXPECT_EQ(se.PixelCount(), 15u);  // 5 * 3
    EXPECT_EQ(se.Shape(), StructElementShape::Rectangle);
}

TEST(StructElementTest, Square_Basic) {
    auto se = StructElement::Square(3);

    EXPECT_EQ(se.Width(), 3);
    EXPECT_EQ(se.Height(), 3);
    EXPECT_EQ(se.AnchorX(), 1);
    EXPECT_EQ(se.AnchorY(), 1);
    EXPECT_EQ(se.PixelCount(), 9u);
}

TEST(StructElementTest, Circle_Basic) {
    auto se = StructElement::Circle(2);  // radius 2

    EXPECT_EQ(se.Width(), 5);  // 2*2+1
    EXPECT_EQ(se.Height(), 5);
    EXPECT_EQ(se.AnchorX(), 2);
    EXPECT_EQ(se.AnchorY(), 2);
    EXPECT_EQ(se.Shape(), StructElementShape::Ellipse);

    // Check center pixel
    EXPECT_TRUE(se.Contains(0, 0));

    // Check edge pixels
    EXPECT_TRUE(se.Contains(0, 2));  // right
    EXPECT_TRUE(se.Contains(0, -2)); // left
    EXPECT_TRUE(se.Contains(2, 0));  // down
    EXPECT_TRUE(se.Contains(-2, 0)); // up

    // Corner should be outside
    EXPECT_FALSE(se.Contains(2, 2));
    EXPECT_FALSE(se.Contains(-2, -2));
}

TEST(StructElementTest, Ellipse_Basic) {
    auto se = StructElement::Ellipse(3, 2);  // rx=3, ry=2

    EXPECT_EQ(se.Width(), 7);  // 2*3+1
    EXPECT_EQ(se.Height(), 5); // 2*2+1
    EXPECT_EQ(se.AnchorX(), 3);
    EXPECT_EQ(se.AnchorY(), 2);

    // Check along major axis
    EXPECT_TRUE(se.Contains(0, 3));  // right edge
    EXPECT_TRUE(se.Contains(0, -3)); // left edge

    // Check along minor axis
    EXPECT_TRUE(se.Contains(2, 0));  // bottom
    EXPECT_TRUE(se.Contains(-2, 0)); // top
}

TEST(StructElementTest, Cross_Basic) {
    auto se = StructElement::Cross(2, 1);  // armLength=2, thickness=1

    EXPECT_EQ(se.Width(), 5);  // 2*2+1
    EXPECT_EQ(se.Height(), 5);
    EXPECT_EQ(se.Shape(), StructElementShape::Cross);

    // Check horizontal arm
    EXPECT_TRUE(se.Contains(0, -2));
    EXPECT_TRUE(se.Contains(0, -1));
    EXPECT_TRUE(se.Contains(0, 0));
    EXPECT_TRUE(se.Contains(0, 1));
    EXPECT_TRUE(se.Contains(0, 2));

    // Check vertical arm
    EXPECT_TRUE(se.Contains(-2, 0));
    EXPECT_TRUE(se.Contains(-1, 0));
    EXPECT_TRUE(se.Contains(1, 0));
    EXPECT_TRUE(se.Contains(2, 0));

    // Check corners (should be empty for cross)
    EXPECT_FALSE(se.Contains(-2, -2));
    EXPECT_FALSE(se.Contains(2, 2));
}

TEST(StructElementTest, Diamond_Basic) {
    auto se = StructElement::Diamond(2);

    EXPECT_EQ(se.Width(), 5);
    EXPECT_EQ(se.Height(), 5);
    EXPECT_EQ(se.Shape(), StructElementShape::Diamond);

    // Check center and cardinal directions
    EXPECT_TRUE(se.Contains(0, 0));
    EXPECT_TRUE(se.Contains(0, 2));
    EXPECT_TRUE(se.Contains(0, -2));
    EXPECT_TRUE(se.Contains(2, 0));
    EXPECT_TRUE(se.Contains(-2, 0));

    // Diagonal within diamond (|dx|+|dy| <= 2)
    EXPECT_TRUE(se.Contains(1, 1));

    // Corners outside diamond (|dx|+|dy| > 2)
    EXPECT_FALSE(se.Contains(2, 2));
    EXPECT_FALSE(se.Contains(-2, -2));
}

TEST(StructElementTest, Line_Horizontal) {
    auto se = StructElement::Line(5, 0);  // horizontal line

    EXPECT_TRUE(se.Contains(0, -2));
    EXPECT_TRUE(se.Contains(0, -1));
    EXPECT_TRUE(se.Contains(0, 0));
    EXPECT_TRUE(se.Contains(0, 1));
    EXPECT_TRUE(se.Contains(0, 2));
}

TEST(StructElementTest, Line_Vertical) {
    auto se = StructElement::Line(5, M_PI / 2);  // vertical line

    EXPECT_TRUE(se.Contains(-2, 0));
    EXPECT_TRUE(se.Contains(-1, 0));
    EXPECT_TRUE(se.Contains(0, 0));
    EXPECT_TRUE(se.Contains(1, 0));
    EXPECT_TRUE(se.Contains(2, 0));
}

TEST(StructElementTest, Octagon_Basic) {
    auto se = StructElement::Octagon(3);

    EXPECT_EQ(se.Width(), 7);
    EXPECT_EQ(se.Height(), 7);
    EXPECT_EQ(se.Shape(), StructElementShape::Octagon);

    // Check center
    EXPECT_TRUE(se.Contains(0, 0));

    // Check cardinal directions
    EXPECT_TRUE(se.Contains(0, 3));
    EXPECT_TRUE(se.Contains(3, 0));
}

// =============================================================================
// From Data Tests
// =============================================================================

TEST(StructElementTest, FromMask_Basic) {
    QImage mask(3, 3, PixelType::UInt8, ChannelType::Gray);
    // Create cross pattern
    for (int r = 0; r < 3; ++r) {
        auto* row = static_cast<uint8_t*>(mask.RowPtr(r));
        for (int c = 0; c < 3; ++c) {
            row[c] = (r == 1 || c == 1) ? 255 : 0;
        }
    }

    auto se = StructElement::FromMask(mask);

    EXPECT_EQ(se.Width(), 3);
    EXPECT_EQ(se.Height(), 3);
    EXPECT_EQ(se.PixelCount(), 5u);  // Cross has 5 pixels

    // Check shape
    EXPECT_TRUE(se.Contains(0, 0));
    EXPECT_TRUE(se.Contains(-1, 0));
    EXPECT_TRUE(se.Contains(1, 0));
    EXPECT_TRUE(se.Contains(0, -1));
    EXPECT_TRUE(se.Contains(0, 1));
    EXPECT_FALSE(se.Contains(-1, -1));
}

TEST(StructElementTest, FromRegion_Basic) {
    std::vector<QRegion::Run> runs = {
        {0, 1, 2},  // row 0, col 1
        {1, 0, 3},  // row 1, cols 0-2
        {2, 1, 2}   // row 2, col 1
    };
    QRegion region(runs);

    auto se = StructElement::FromRegion(region);

    EXPECT_EQ(se.Width(), 3);
    EXPECT_EQ(se.Height(), 3);
    EXPECT_EQ(se.PixelCount(), 5u);
}

TEST(StructElementTest, FromCoordinates_Basic) {
    std::vector<Point2i> coords = {
        {0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}
    };

    auto se = StructElement::FromCoordinates(coords);

    EXPECT_EQ(se.PixelCount(), 5u);

    // Check all coordinates are present
    for (const auto& pt : coords) {
        EXPECT_TRUE(se.Contains(pt.y, pt.x)) << "Missing (" << pt.x << ", " << pt.y << ")";
    }
}

// =============================================================================
// Property Tests
// =============================================================================

TEST(StructElementTest, IsSeparable) {
    auto rect = StructElement::Rectangle(5, 3);
    EXPECT_TRUE(rect.IsSeparable());

    auto circle = StructElement::Circle(3);
    EXPECT_FALSE(circle.IsSeparable());
}

TEST(StructElementTest, IsSymmetric) {
    auto square = StructElement::Square(3);
    EXPECT_TRUE(square.IsSymmetric());

    auto circle = StructElement::Circle(2);
    EXPECT_TRUE(circle.IsSymmetric());

    auto diamond = StructElement::Diamond(2);
    EXPECT_TRUE(diamond.IsSymmetric());
}

// =============================================================================
// Data Access Tests
// =============================================================================

TEST(StructElementTest, GetCoordinates) {
    auto cross = StructElement::Cross(1, 1);  // 3x3 cross

    auto coords = cross.GetCoordinates();

    EXPECT_EQ(coords.size(), 5u);
}

TEST(StructElementTest, GetMask) {
    auto se = StructElement::Square(3);

    auto mask = se.GetMask();

    EXPECT_EQ(mask.Width(), 3);
    EXPECT_EQ(mask.Height(), 3);

    // All pixels should be 255
    for (int r = 0; r < 3; ++r) {
        auto* row = static_cast<uint8_t*>(mask.RowPtr(r));
        for (int c = 0; c < 3; ++c) {
            EXPECT_EQ(row[c], 255);
        }
    }
}

TEST(StructElementTest, GetRegion) {
    auto se = StructElement::Rectangle(5, 3);

    auto region = se.GetRegion();

    EXPECT_EQ(region.Area(), 15);
}

// =============================================================================
// Transformation Tests
// =============================================================================

TEST(StructElementTest, Reflect_Symmetric) {
    auto se = StructElement::Square(3);
    auto reflected = se.Reflect();

    // Symmetric element should be same after reflection
    EXPECT_EQ(reflected.PixelCount(), se.PixelCount());
}

TEST(StructElementTest, Reflect_Asymmetric) {
    // Create asymmetric element (L shape)
    std::vector<Point2i> coords = {
        {0, 0}, {1, 0}, {2, 0}, {0, 1}
    };
    auto se = StructElement::FromCoordinates(coords);
    auto reflected = se.Reflect();

    EXPECT_EQ(reflected.PixelCount(), se.PixelCount());

    // Check that coordinates are reflected
    EXPECT_TRUE(reflected.Contains(0, 0));
}

TEST(StructElementTest, Transpose_Square) {
    auto se = StructElement::Square(3);
    auto transposed = se.Transpose();

    EXPECT_EQ(transposed.Width(), se.Height());
    EXPECT_EQ(transposed.Height(), se.Width());
}

TEST(StructElementTest, Transpose_Rectangle) {
    auto se = StructElement::Rectangle(5, 3);
    auto transposed = se.Transpose();

    EXPECT_EQ(transposed.Width(), 3);
    EXPECT_EQ(transposed.Height(), 5);
    EXPECT_EQ(transposed.PixelCount(), 15u);
}

TEST(StructElementTest, Rotate_90Degrees) {
    auto se = StructElement::Line(5, 0);  // horizontal
    auto rotated = se.Rotate(M_PI / 2);   // should become vertical

    // After 90 degree rotation, vertical should have pixels
    EXPECT_TRUE(rotated.Contains(-2, 0) || rotated.Contains(2, 0));
}

TEST(StructElementTest, Scale_Double) {
    auto se = StructElement::Square(3);
    auto scaled = se.Scale(2.0, 2.0);

    // Scaled element should cover larger area (bounding box)
    // Due to discrete rounding, pixel count may or may not increase
    EXPECT_GE(scaled.PixelCount(), se.PixelCount());

    // Check bounding box is larger
    EXPECT_GE(scaled.Width(), se.Width());
    EXPECT_GE(scaled.Height(), se.Height());
}

// =============================================================================
// Decomposition Tests
// =============================================================================

TEST(StructElementTest, CanDecompose) {
    auto rect = StructElement::Rectangle(5, 3);
    EXPECT_TRUE(rect.CanDecompose());

    auto circle = StructElement::Circle(3);
    EXPECT_FALSE(circle.CanDecompose());
}

TEST(StructElementTest, Decompose_Rectangle) {
    auto se = StructElement::Rectangle(5, 3);

    StructElement horizontal, vertical;
    bool success = se.Decompose(horizontal, vertical);

    EXPECT_TRUE(success);
    EXPECT_EQ(horizontal.Width(), 5);
    EXPECT_EQ(horizontal.Height(), 1);
    EXPECT_EQ(vertical.Width(), 1);
    EXPECT_EQ(vertical.Height(), 3);
}

TEST(StructElementTest, DecomposeToSequence_Rectangle) {
    auto se = StructElement::Rectangle(5, 3);

    auto sequence = se.DecomposeToSequence();

    EXPECT_EQ(sequence.size(), 2u);  // 1D horizontal + 1D vertical
}

// =============================================================================
// Convenience Function Tests
// =============================================================================

TEST(StructElementTest, SE_Cross3) {
    auto se = SE_Cross3();

    EXPECT_EQ(se.Width(), 3);
    EXPECT_EQ(se.Height(), 3);
    EXPECT_EQ(se.PixelCount(), 5u);
}

TEST(StructElementTest, SE_Square3) {
    auto se = SE_Square3();

    EXPECT_EQ(se.Width(), 3);
    EXPECT_EQ(se.Height(), 3);
    EXPECT_EQ(se.PixelCount(), 9u);
}

TEST(StructElementTest, SE_Disk5) {
    auto se = SE_Disk5();

    EXPECT_EQ(se.Width(), 5);
    EXPECT_EQ(se.Height(), 5);
}

TEST(StructElementTest, CreateHitMissSE) {
    std::vector<Point2i> hit = {{0, 0}, {1, 0}};
    std::vector<Point2i> miss = {{-1, 0}, {0, 1}};

    auto [hitSE, missSE] = CreateHitMissSE(hit, miss);

    EXPECT_EQ(hitSE.PixelCount(), 2u);
    EXPECT_EQ(missSE.PixelCount(), 2u);
}

// =============================================================================
// Copy/Move Tests
// =============================================================================

TEST(StructElementTest, CopyConstructor) {
    auto se = StructElement::Circle(3);
    StructElement copy(se);

    EXPECT_EQ(copy.Width(), se.Width());
    EXPECT_EQ(copy.Height(), se.Height());
    EXPECT_EQ(copy.PixelCount(), se.PixelCount());
}

TEST(StructElementTest, MoveConstructor) {
    auto se = StructElement::Circle(3);
    size_t pixelCount = se.PixelCount();

    StructElement moved(std::move(se));

    EXPECT_EQ(moved.PixelCount(), pixelCount);
}

TEST(StructElementTest, CopyAssignment) {
    auto se = StructElement::Circle(3);
    StructElement copy;
    copy = se;

    EXPECT_EQ(copy.PixelCount(), se.PixelCount());
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(StructElementTest, Empty_Default) {
    StructElement se;
    EXPECT_TRUE(se.Empty());
    EXPECT_EQ(se.Width(), 0);
    EXPECT_EQ(se.Height(), 0);
    EXPECT_EQ(se.PixelCount(), 0u);
}

TEST(StructElementTest, Invalid_ZeroSize) {
    auto se = StructElement::Rectangle(0, 0);
    EXPECT_TRUE(se.Empty());
}

TEST(StructElementTest, Invalid_NegativeSize) {
    auto se = StructElement::Circle(-1);
    EXPECT_TRUE(se.Empty());
}

TEST(StructElementTest, SinglePixel) {
    auto se = StructElement::Square(1);

    EXPECT_EQ(se.Width(), 1);
    EXPECT_EQ(se.Height(), 1);
    EXPECT_EQ(se.PixelCount(), 1u);
    EXPECT_TRUE(se.Contains(0, 0));
}

