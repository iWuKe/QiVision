/**
 * @file test_steger.cpp
 * @brief Unit tests for Internal/Steger.h
 */

#include <QiVision/Internal/Steger.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

class StegerTest : public ::testing::Test {
protected:
    // Test image size
    int32_t width_ = 64;
    int32_t height_ = 64;

    // Create a horizontal ridge line image (Gaussian profile)
    std::vector<float> CreateHorizontalRidgeLine(double y0, double sigma, double amplitude) {
        std::vector<float> image(static_cast<size_t>(width_ * height_));
        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                double dy = y - y0;
                double value = amplitude * std::exp(-dy * dy / (2.0 * sigma * sigma));
                image[y * width_ + x] = static_cast<float>(value);
            }
        }
        return image;
    }

    // Create a vertical ridge line image
    std::vector<float> CreateVerticalRidgeLine(double x0, double sigma, double amplitude) {
        std::vector<float> image(static_cast<size_t>(width_ * height_));
        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                double dx = x - x0;
                double value = amplitude * std::exp(-dx * dx / (2.0 * sigma * sigma));
                image[y * width_ + x] = static_cast<float>(value);
            }
        }
        return image;
    }

    // Create a diagonal ridge line image
    std::vector<float> CreateDiagonalRidgeLine(double offset, double angle, double sigma, double amplitude) {
        std::vector<float> image(static_cast<size_t>(width_ * height_));
        double cosA = std::cos(angle);
        double sinA = std::sin(angle);

        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                // Distance from point to line
                double cx = x - width_ / 2.0;
                double cy = y - height_ / 2.0;
                double dist = std::abs(cosA * cy - sinA * cx - offset);
                double value = amplitude * std::exp(-dist * dist / (2.0 * sigma * sigma));
                image[y * width_ + x] = static_cast<float>(value);
            }
        }
        return image;
    }

    // Create a valley line image (inverted Gaussian)
    std::vector<float> CreateHorizontalValleyLine(double y0, double sigma, double amplitude) {
        std::vector<float> image(static_cast<size_t>(width_ * height_), static_cast<float>(amplitude));
        for (int32_t y = 0; y < height_; ++y) {
            for (int32_t x = 0; x < width_; ++x) {
                double dy = y - y0;
                double value = amplitude * (1.0 - std::exp(-dy * dy / (2.0 * sigma * sigma)));
                image[y * width_ + x] = static_cast<float>(value);
            }
        }
        return image;
    }
};

// ============================================================================
// StegerParams Tests
// ============================================================================

TEST_F(StegerTest, StegerParams_Defaults) {
    StegerParams params;
    EXPECT_DOUBLE_EQ(params.sigma, 1.0);
    EXPECT_DOUBLE_EQ(params.lowThreshold, 5.0);
    EXPECT_DOUBLE_EQ(params.highThreshold, 15.0);
    EXPECT_EQ(params.lineType, LineType::Both);
    EXPECT_TRUE(params.subPixelRefinement);
}

// ============================================================================
// StegerPoint Tests
// ============================================================================

TEST_F(StegerTest, StegerPoint_SubpixelOffset) {
    StegerPoint pt;
    pt.pixelX = 10;
    pt.pixelY = 20;
    pt.x = 10.3;
    pt.y = 20.7;

    auto offset = pt.SubpixelOffset();
    EXPECT_NEAR(offset.x, 0.3, 1e-10);
    EXPECT_NEAR(offset.y, 0.7, 1e-10);
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST_F(StegerTest, IsEdgeCandidate_Ridge) {
    // Ridge: λ1 < 0, |λ1| >> |λ2|
    EXPECT_TRUE(IsEdgeCandidate(-10.0, -1.0, 5.0, LineType::Ridge));
    EXPECT_TRUE(IsEdgeCandidate(-10.0, -1.0, 5.0, LineType::Both));
    EXPECT_FALSE(IsEdgeCandidate(-10.0, -1.0, 5.0, LineType::Valley));

    // Not a ridge (positive λ1)
    EXPECT_FALSE(IsEdgeCandidate(10.0, 1.0, 5.0, LineType::Ridge));
}

TEST_F(StegerTest, IsEdgeCandidate_Valley) {
    // Valley: λ1 > 0, |λ1| >> |λ2|
    EXPECT_TRUE(IsEdgeCandidate(10.0, 1.0, 5.0, LineType::Valley));
    EXPECT_TRUE(IsEdgeCandidate(10.0, 1.0, 5.0, LineType::Both));
    EXPECT_FALSE(IsEdgeCandidate(10.0, 1.0, 5.0, LineType::Ridge));
}

TEST_F(StegerTest, IsEdgeCandidate_BelowThreshold) {
    // Below threshold
    EXPECT_FALSE(IsEdgeCandidate(-2.0, -0.5, 5.0, LineType::Ridge));
}

TEST_F(StegerTest, IsEdgeCandidate_TooIsotropic) {
    // Too blob-like (|λ1| ≈ |λ2|)
    EXPECT_FALSE(IsEdgeCandidate(-10.0, -8.0, 5.0, LineType::Ridge));
}

TEST_F(StegerTest, TangentAngleDiff_Parallel) {
    // Parallel tangents (should be 0)
    double diff = TangentAngleDiff(1.0, 0.0, 1.0, 0.0);
    EXPECT_NEAR(diff, 0.0, 1e-10);
}

TEST_F(StegerTest, TangentAngleDiff_Antiparallel) {
    // Antiparallel tangents (should also be 0 since tangents are undirected)
    double diff = TangentAngleDiff(1.0, 0.0, -1.0, 0.0);
    EXPECT_NEAR(diff, 0.0, 1e-10);
}

TEST_F(StegerTest, TangentAngleDiff_Perpendicular) {
    // Perpendicular tangents (should be π/2)
    double diff = TangentAngleDiff(1.0, 0.0, 0.0, 1.0);
    EXPECT_NEAR(diff, M_PI / 2.0, 1e-10);
}

TEST_F(StegerTest, PointDistance) {
    StegerPoint p1, p2;
    p1.x = 0.0;
    p1.y = 0.0;
    p2.x = 3.0;
    p2.y = 4.0;

    EXPECT_DOUBLE_EQ(PointDistance(p1, p2), 5.0);
}

// ============================================================================
// Horizontal Ridge Detection
// ============================================================================

TEST_F(StegerTest, DetectStegerEdges_HorizontalRidge) {
    double trueY = 32.0;
    double lineSigma = 3.0;
    auto image = CreateHorizontalRidgeLine(trueY, lineSigma, 255.0);

    // Create QImage from data
    QImage qimage(width_, height_, PixelType::Float32);
    std::memcpy(qimage.Data(), image.data(), image.size() * sizeof(float));

    StegerParams params;
    params.sigma = lineSigma;
    params.lowThreshold = 1.0;
    params.highThreshold = 2.0;
    params.lineType = LineType::Ridge;
    params.minLength = 3.0;

    auto contours = DetectStegerEdges(qimage, params);

    // Should detect at least one contour
    EXPECT_GE(contours.size(), 1u);

    // Check that the contour is near the true position
    if (!contours.empty()) {
        bool foundNearTrueY = false;
        for (const auto& contour : contours) {
            for (const auto& pt : contour.GetPoints()) {
                if (std::abs(pt.y - trueY) < 1.0) {
                    foundNearTrueY = true;
                    break;
                }
            }
            if (foundNearTrueY) break;
        }
        EXPECT_TRUE(foundNearTrueY);
    }
}

TEST_F(StegerTest, DetectStegerEdges_VerticalRidge) {
    double trueX = 32.0;
    double lineSigma = 3.0;
    auto image = CreateVerticalRidgeLine(trueX, lineSigma, 255.0);

    QImage qimage(width_, height_, PixelType::Float32);
    std::memcpy(qimage.Data(), image.data(), image.size() * sizeof(float));

    StegerParams params;
    params.sigma = lineSigma;
    params.lowThreshold = 1.0;
    params.highThreshold = 2.0;
    params.lineType = LineType::Ridge;
    params.minLength = 3.0;

    auto contours = DetectStegerEdges(qimage, params);

    EXPECT_GE(contours.size(), 1u);
}

TEST_F(StegerTest, DetectStegerEdges_DiagonalRidge) {
    double lineSigma = 3.0;
    auto image = CreateDiagonalRidgeLine(0.0, M_PI / 4.0, lineSigma, 255.0);

    QImage qimage(width_, height_, PixelType::Float32);
    std::memcpy(qimage.Data(), image.data(), image.size() * sizeof(float));

    StegerParams params;
    params.sigma = lineSigma;
    params.lowThreshold = 1.0;
    params.highThreshold = 2.0;
    params.lineType = LineType::Ridge;
    params.minLength = 3.0;

    auto contours = DetectStegerEdges(qimage, params);

    EXPECT_GE(contours.size(), 1u);
}

// ============================================================================
// Valley Detection
// ============================================================================

TEST_F(StegerTest, DetectStegerEdges_Valley) {
    double trueY = 32.0;
    double lineSigma = 3.0;
    auto image = CreateHorizontalValleyLine(trueY, lineSigma, 255.0);

    QImage qimage(width_, height_, PixelType::Float32);
    std::memcpy(qimage.Data(), image.data(), image.size() * sizeof(float));

    StegerParams params;
    params.sigma = lineSigma;
    params.lowThreshold = 1.0;
    params.highThreshold = 2.0;
    params.lineType = LineType::Valley;
    params.minLength = 3.0;

    auto contours = DetectStegerEdges(qimage, params);

    EXPECT_GE(contours.size(), 1u);
}

// ============================================================================
// Full Result Tests
// ============================================================================

TEST_F(StegerTest, DetectStegerEdgesFull_PointCount) {
    double trueY = 32.0;
    double lineSigma = 3.0;
    auto image = CreateHorizontalRidgeLine(trueY, lineSigma, 255.0);

    QImage qimage(width_, height_, PixelType::Float32);
    std::memcpy(qimage.Data(), image.data(), image.size() * sizeof(float));

    StegerParams params;
    params.sigma = lineSigma;
    params.lowThreshold = 1.0;
    params.highThreshold = 2.0;
    params.lineType = LineType::Ridge;

    auto result = DetectStegerEdgesFull(qimage, params);

    // Should have some points
    EXPECT_GT(result.points.size(), 0u);

    // All points should be ridges
    EXPECT_GT(result.numRidgePoints, 0);
    EXPECT_EQ(result.numValleyPoints, 0);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(StegerTest, DetectStegerEdges_EmptyImage) {
    QImage empty;
    StegerParams params;

    auto contours = DetectStegerEdges(empty, params);
    EXPECT_TRUE(contours.empty());
}

TEST_F(StegerTest, DetectStegerEdges_ConstantImage) {
    // Constant image should have no edges
    std::vector<float> image(static_cast<size_t>(width_ * height_), 128.0f);

    QImage qimage(width_, height_, PixelType::Float32);
    std::memcpy(qimage.Data(), image.data(), image.size() * sizeof(float));

    StegerParams params;
    params.sigma = 2.0;
    params.lowThreshold = 0.1;
    params.highThreshold = 0.5;

    auto contours = DetectStegerEdges(qimage, params);

    // Should have no contours for constant image
    EXPECT_TRUE(contours.empty());
}

// ============================================================================
// Conversion Functions
// ============================================================================

TEST_F(StegerTest, ToContourPoints_Conversion) {
    std::vector<StegerPoint> stegerPoints;

    StegerPoint p1;
    p1.x = 10.5;
    p1.y = 20.3;
    p1.tx = 1.0;
    p1.ty = 0.0;
    p1.response = 50.0;
    stegerPoints.push_back(p1);

    auto contourPoints = ToContourPoints(stegerPoints);

    ASSERT_EQ(contourPoints.size(), 1u);
    EXPECT_DOUBLE_EQ(contourPoints[0].x, 10.5);
    EXPECT_DOUBLE_EQ(contourPoints[0].y, 20.3);
    EXPECT_DOUBLE_EQ(contourPoints[0].amplitude, 50.0);
}

TEST_F(StegerTest, CreateContour_FromPoints) {
    std::vector<StegerPoint> points;

    for (int i = 0; i < 10; ++i) {
        StegerPoint pt;
        pt.x = static_cast<double>(i);
        pt.y = 10.0;
        pt.tx = 1.0;
        pt.ty = 0.0;
        pt.response = 50.0;
        points.push_back(pt);
    }

    auto contour = CreateContour(points);

    EXPECT_EQ(contour.GetPoints().size(), 10u);
    EXPECT_GT(contour.Length(), 0.0);
}

// ============================================================================
// Edge Linking
// ============================================================================

TEST_F(StegerTest, LinkEdgePoints_SimpleChain) {
    std::vector<StegerPoint> points;

    // Create a horizontal chain of points
    for (int i = 0; i < 20; ++i) {
        StegerPoint pt;
        pt.x = static_cast<double>(i);
        pt.y = 32.0;
        pt.pixelX = i;
        pt.pixelY = 32;
        pt.tx = 1.0;  // Tangent along x
        pt.ty = 0.0;
        pt.nx = 0.0;
        pt.ny = 1.0;
        pt.response = 50.0;
        pt.isRidge = true;
        points.push_back(pt);
    }

    auto contours = LinkEdgePoints(points, 2.0, 0.5);

    // Should link into one or more contours
    EXPECT_GE(contours.size(), 1u);

    // Total points should match
    size_t totalPoints = 0;
    for (const auto& c : contours) {
        totalPoints += c.GetPoints().size();
    }
    EXPECT_EQ(totalPoints, 20u);
}

TEST_F(StegerTest, FilterByLength_Works) {
    std::vector<QContour> contours;

    // Create short contour
    QContour short1;
    short1.AddPoint(ContourPoint(0, 0));
    short1.AddPoint(ContourPoint(1, 0));
    contours.push_back(short1);

    // Create long contour
    QContour long1;
    for (int i = 0; i < 20; ++i) {
        long1.AddPoint(ContourPoint(static_cast<double>(i), 0));
    }
    contours.push_back(long1);

    auto filtered = FilterByLength(contours, 5.0);

    EXPECT_EQ(filtered.size(), 1u);
}
