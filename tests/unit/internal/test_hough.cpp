/**
 * @file test_hough.cpp
 * @brief Unit tests for Hough Transform module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/Hough.h>
#include <QiVision/Core/QImage.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

namespace {
    constexpr double TEST_PI = 3.14159265358979323846;
    constexpr double TOLERANCE = 1e-6;
    constexpr double ANGLE_TOLERANCE = 0.05;  // ~3 degrees
    constexpr double POSITION_TOLERANCE = 3.0;  // pixels
}

// =============================================================================
// HoughLine Structure Tests
// =============================================================================

class HoughLineTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(HoughLineTest, DefaultConstructor) {
    HoughLine line;
    EXPECT_DOUBLE_EQ(line.rho, 0.0);
    EXPECT_DOUBLE_EQ(line.theta, 0.0);
    EXPECT_DOUBLE_EQ(line.score, 0.0);
}

TEST_F(HoughLineTest, ParameterizedConstructor) {
    HoughLine line(50.0, TEST_PI / 4, 100.0);
    EXPECT_DOUBLE_EQ(line.rho, 50.0);
    EXPECT_DOUBLE_EQ(line.theta, TEST_PI / 4);
    EXPECT_DOUBLE_EQ(line.score, 100.0);
}

TEST_F(HoughLineTest, ToLine2d_HorizontalLine) {
    // Horizontal line at y = 50 (theta = TEST_PI/2, rho = 50)
    HoughLine hLine(50.0, TEST_PI / 2, 1.0);
    Line2d line = hLine.ToLine2d();

    // For horizontal line: 0*x + 1*y - 50 = 0
    EXPECT_NEAR(line.a, 0.0, TOLERANCE);
    EXPECT_NEAR(std::abs(line.b), 1.0, TOLERANCE);
    EXPECT_NEAR(line.c / line.b, -50.0, TOLERANCE);
}

TEST_F(HoughLineTest, ToLine2d_VerticalLine) {
    // Vertical line at x = 100 (theta = 0, rho = 100)
    HoughLine hLine(100.0, 0.0, 1.0);
    Line2d line = hLine.ToLine2d();

    // For vertical line: 1*x + 0*y - 100 = 0
    EXPECT_NEAR(std::abs(line.a), 1.0, TOLERANCE);
    EXPECT_NEAR(line.b, 0.0, TOLERANCE);
}

TEST_F(HoughLineTest, GetTwoPoints) {
    HoughLine hLine(50.0, TEST_PI / 4, 1.0);
    auto [p1, p2] = hLine.GetTwoPoints(100.0);

    // Points should be on the line
    double d1 = std::abs(p1.x * std::cos(TEST_PI/4) + p1.y * std::sin(TEST_PI/4) - 50.0);
    double d2 = std::abs(p2.x * std::cos(TEST_PI/4) + p2.y * std::sin(TEST_PI/4) - 50.0);

    EXPECT_NEAR(d1, 0.0, TOLERANCE);
    EXPECT_NEAR(d2, 0.0, TOLERANCE);

    // Distance between points should be approximately the length
    double dist = std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2));
    EXPECT_NEAR(dist, 100.0, TOLERANCE);
}

// =============================================================================
// HoughLineSegment Structure Tests
// =============================================================================

class HoughLineSegmentTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(HoughLineSegmentTest, DefaultConstructor) {
    HoughLineSegment seg;
    EXPECT_DOUBLE_EQ(seg.score, 0.0);
}

TEST_F(HoughLineSegmentTest, ParameterizedConstructor) {
    Point2d p1(0, 0), p2(100, 100);
    HoughLineSegment seg(p1, p2, 50.0);

    EXPECT_DOUBLE_EQ(seg.p1.x, 0.0);
    EXPECT_DOUBLE_EQ(seg.p1.y, 0.0);
    EXPECT_DOUBLE_EQ(seg.p2.x, 100.0);
    EXPECT_DOUBLE_EQ(seg.p2.y, 100.0);
    EXPECT_DOUBLE_EQ(seg.score, 50.0);
}

TEST_F(HoughLineSegmentTest, Length) {
    HoughLineSegment seg(Point2d(0, 0), Point2d(3, 4), 1.0);
    EXPECT_NEAR(seg.Length(), 5.0, TOLERANCE);
}

TEST_F(HoughLineSegmentTest, Angle) {
    HoughLineSegment seg(Point2d(0, 0), Point2d(10, 10), 1.0);
    EXPECT_NEAR(seg.Angle(), TEST_PI / 4, TOLERANCE);

    HoughLineSegment seg2(Point2d(0, 0), Point2d(10, 0), 1.0);
    EXPECT_NEAR(seg2.Angle(), 0.0, TOLERANCE);

    HoughLineSegment seg3(Point2d(0, 0), Point2d(0, 10), 1.0);
    EXPECT_NEAR(seg3.Angle(), TEST_PI / 2, TOLERANCE);
}

TEST_F(HoughLineSegmentTest, ToSegment2d) {
    Point2d p1(10, 20), p2(30, 40);
    HoughLineSegment seg(p1, p2, 1.0);
    Segment2d s = seg.ToSegment2d();

    EXPECT_DOUBLE_EQ(s.p1.x, 10.0);
    EXPECT_DOUBLE_EQ(s.p1.y, 20.0);
    EXPECT_DOUBLE_EQ(s.p2.x, 30.0);
    EXPECT_DOUBLE_EQ(s.p2.y, 40.0);
}

// =============================================================================
// HoughCircle Structure Tests
// =============================================================================

class HoughCircleTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(HoughCircleTest, DefaultConstructor) {
    HoughCircle circle;
    EXPECT_DOUBLE_EQ(circle.radius, 0.0);
    EXPECT_DOUBLE_EQ(circle.score, 0.0);
}

TEST_F(HoughCircleTest, ParameterizedConstructor) {
    HoughCircle circle(Point2d(50, 60), 30.0, 100.0);

    EXPECT_DOUBLE_EQ(circle.center.x, 50.0);
    EXPECT_DOUBLE_EQ(circle.center.y, 60.0);
    EXPECT_DOUBLE_EQ(circle.radius, 30.0);
    EXPECT_DOUBLE_EQ(circle.score, 100.0);
}

TEST_F(HoughCircleTest, ToCircle2d) {
    HoughCircle hc(Point2d(100, 200), 50.0, 1.0);
    Circle2d c = hc.ToCircle2d();

    EXPECT_DOUBLE_EQ(c.center.x, 100.0);
    EXPECT_DOUBLE_EQ(c.center.y, 200.0);
    EXPECT_DOUBLE_EQ(c.radius, 50.0);
}

// =============================================================================
// HoughAccumulator Tests
// =============================================================================

class HoughAccumulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        acc.rhoMin = -100.0;
        acc.rhoMax = 100.0;
        acc.rhoStep = 1.0;
        acc.thetaMin = 0.0;
        acc.thetaMax = TEST_PI;
        acc.thetaStep = TEST_PI / 180.0;
    }

    HoughAccumulator acc;
};

TEST_F(HoughAccumulatorTest, GetRho) {
    EXPECT_NEAR(acc.GetRho(0), -100.0, TOLERANCE);
    EXPECT_NEAR(acc.GetRho(100), 0.0, TOLERANCE);
    EXPECT_NEAR(acc.GetRho(200), 100.0, TOLERANCE);
}

TEST_F(HoughAccumulatorTest, GetTheta) {
    EXPECT_NEAR(acc.GetTheta(0), 0.0, TOLERANCE);
    EXPECT_NEAR(acc.GetTheta(90), TEST_PI / 2, TOLERANCE);
    EXPECT_NEAR(acc.GetTheta(180), TEST_PI, TOLERANCE);
}

TEST_F(HoughAccumulatorTest, GetRhoIndex) {
    EXPECT_EQ(acc.GetRhoIndex(-100.0), 0);
    EXPECT_EQ(acc.GetRhoIndex(0.0), 100);
    EXPECT_EQ(acc.GetRhoIndex(100.0), 200);
}

TEST_F(HoughAccumulatorTest, GetThetaIndex) {
    EXPECT_EQ(acc.GetThetaIndex(0.0), 0);
    EXPECT_EQ(acc.GetThetaIndex(TEST_PI / 2), 90);
    EXPECT_EQ(acc.GetThetaIndex(TEST_PI), 180);
}

// =============================================================================
// Standard Hough Line Detection Tests
// =============================================================================

class HoughLinesTest : public ::testing::Test {
protected:
    void SetUp() override {}

    // Helper: Create edge image with a line from (x1,y1) to (x2,y2)
    QImage CreateLineImage(int width, int height, int x1, int y1, int x2, int y2) {
        QImage img(width, height, PixelType::UInt8, ChannelType::Gray);
        std::memset(img.Data(), 0, width * height);

        // Bresenham's line algorithm
        int dx = std::abs(x2 - x1), sx = x1 < x2 ? 1 : -1;
        int dy = -std::abs(y2 - y1), sy = y1 < y2 ? 1 : -1;
        int err = dx + dy, e2;

        while (true) {
            if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
                static_cast<uint8_t*>(img.RowPtr(y1))[x1] = 255;
            }
            if (x1 == x2 && y1 == y2) break;
            e2 = 2 * err;
            if (e2 >= dy) { err += dy; x1 += sx; }
            if (e2 <= dx) { err += dx; y1 += sy; }
        }

        return img;
    }

    // Helper: Generate points on a line
    std::vector<Point2d> GenerateLinePoints(double rho, double theta,
                                             int count, double length) {
        std::vector<Point2d> points;
        double x0 = rho * std::cos(theta);
        double y0 = rho * std::sin(theta);
        double dx = -std::sin(theta);
        double dy = std::cos(theta);

        for (int i = 0; i < count; ++i) {
            double t = (i - count / 2.0) * length / count;
            points.emplace_back(x0 + t * dx, y0 + t * dy);
        }

        return points;
    }
};

TEST_F(HoughLinesTest, EmptyImage) {
    QImage emptyImg;
    auto lines = HoughLines(emptyImg);
    EXPECT_TRUE(lines.empty());
}

TEST_F(HoughLinesTest, EmptyPoints) {
    std::vector<Point2d> points;
    auto lines = HoughLines(points, 100, 100);
    EXPECT_TRUE(lines.empty());
}

TEST_F(HoughLinesTest, SingleHorizontalLine) {
    // Create a horizontal line at y = 50
    QImage img = CreateLineImage(200, 100, 10, 50, 190, 50);

    HoughLineParams params;
    params.threshold = 0.3;
    params.maxLines = 1;

    auto lines = HoughLines(img, params);

    ASSERT_GE(lines.size(), 1u);

    // Horizontal line: theta should be near TEST_PI/2, rho should be near 50
    EXPECT_NEAR(lines[0].theta, TEST_PI / 2, ANGLE_TOLERANCE);
    EXPECT_NEAR(lines[0].rho, 50.0, POSITION_TOLERANCE);
}

TEST_F(HoughLinesTest, SingleVerticalLine) {
    // Create a vertical line at x = 100 (200 pixels long for better detection)
    QImage img = CreateLineImage(250, 220, 100, 10, 100, 200);

    HoughLineParams params;
    params.threshold = 0.2;  // Lower threshold for more sensitive detection
    params.maxLines = 1;

    auto lines = HoughLines(img, params);

    ASSERT_GE(lines.size(), 1u);

    // Vertical line: theta should be near 0 or TEST_PI, rho should be near 100
    EXPECT_TRUE(lines[0].theta < ANGLE_TOLERANCE ||
                std::abs(lines[0].theta - TEST_PI) < ANGLE_TOLERANCE);
    EXPECT_NEAR(std::abs(lines[0].rho), 100.0, POSITION_TOLERANCE);
}

TEST_F(HoughLinesTest, DiagonalLine) {
    // Create a diagonal line from (0,0) to (100,100)
    QImage img = CreateLineImage(150, 150, 0, 0, 100, 100);

    HoughLineParams params;
    params.threshold = 0.3;
    params.maxLines = 1;

    auto lines = HoughLines(img, params);

    ASSERT_GE(lines.size(), 1u);

    // 45-degree line through origin: theta ~= TEST_PI/4 or 3*TEST_PI/4, rho ~= 0
    // The exact form depends on the line equation
    EXPECT_TRUE(std::abs(lines[0].theta - TEST_PI/4) < ANGLE_TOLERANCE ||
                std::abs(lines[0].theta - 3*TEST_PI/4) < ANGLE_TOLERANCE);
}

TEST_F(HoughLinesTest, MultipleLines) {
    QImage img(200, 200, PixelType::UInt8, ChannelType::Gray);
    std::memset(img.Data(), 0, 200 * 200);

    // Draw horizontal line
    for (int x = 20; x < 180; ++x) {
        static_cast<uint8_t*>(img.RowPtr(50))[x] = 255;
    }
    // Draw vertical line
    for (int y = 20; y < 180; ++y) {
        static_cast<uint8_t*>(img.RowPtr(y))[100] = 255;
    }

    HoughLineParams params;
    params.threshold = 0.2;
    params.maxLines = 5;
    params.suppressOverlapping = true;

    auto lines = HoughLines(img, params);

    EXPECT_GE(lines.size(), 2u);
}

TEST_F(HoughLinesTest, FromPointsList) {
    // Generate points on a horizontal line at y = 100
    auto points = GenerateLinePoints(100.0, TEST_PI/2, 50, 200.0);

    HoughLineParams params;
    params.threshold = 0.3;
    params.maxLines = 1;

    auto lines = HoughLines(points, 300, 200, params);

    ASSERT_GE(lines.size(), 1u);
    EXPECT_NEAR(lines[0].theta, TEST_PI / 2, ANGLE_TOLERANCE);
    EXPECT_NEAR(lines[0].rho, 100.0, POSITION_TOLERANCE);
}

// =============================================================================
// Hough Accumulator Tests
// =============================================================================

class HoughAccumulatorFuncTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(HoughAccumulatorFuncTest, BasicAccumulator) {
    std::vector<Point2d> points;
    // Add points on a line
    for (int i = 0; i < 100; ++i) {
        points.emplace_back(i, 50.0);  // Horizontal line at y=50
    }

    HoughLineParams params;
    auto acc = GetHoughAccumulator(points, 200, 100, params);

    EXPECT_GT(acc.data.Rows(), 0);
    EXPECT_GT(acc.data.Cols(), 0);

    // Find peak
    double maxVal = 0;
    int maxR = 0, maxT = 0;
    for (int r = 0; r < acc.data.Rows(); ++r) {
        for (int t = 0; t < acc.data.Cols(); ++t) {
            if (acc.data(r, t) > maxVal) {
                maxVal = acc.data(r, t);
                maxR = r;
                maxT = t;
            }
        }
    }

    // Peak should correspond to horizontal line at y=50
    double peakTheta = acc.GetTheta(maxT);
    double peakRho = acc.GetRho(maxR);

    EXPECT_NEAR(peakTheta, TEST_PI / 2, ANGLE_TOLERANCE);
    EXPECT_NEAR(peakRho, 50.0, POSITION_TOLERANCE);
}

TEST_F(HoughAccumulatorFuncTest, FindPeaks) {
    std::vector<Point2d> points;
    // Horizontal line
    for (int i = 0; i < 100; ++i) {
        points.emplace_back(i, 50.0);
    }

    HoughLineParams params;
    auto acc = GetHoughAccumulator(points, 200, 100, params);

    auto peaks = FindAccumulatorPeaks(acc, 0.3, 5, 10);

    ASSERT_GE(peaks.size(), 1u);
    EXPECT_NEAR(peaks[0].theta, TEST_PI / 2, ANGLE_TOLERANCE);
    EXPECT_NEAR(peaks[0].rho, 50.0, POSITION_TOLERANCE);
}

// =============================================================================
// Probabilistic Hough Transform Tests
// =============================================================================

class HoughLinesPTest : public ::testing::Test {
protected:
    QImage CreateLineImage(int width, int height, int x1, int y1, int x2, int y2) {
        QImage img(width, height, PixelType::UInt8, ChannelType::Gray);
        std::memset(img.Data(), 0, width * height);

        int dx = std::abs(x2 - x1), sx = x1 < x2 ? 1 : -1;
        int dy = -std::abs(y2 - y1), sy = y1 < y2 ? 1 : -1;
        int err = dx + dy, e2;

        while (true) {
            if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
                static_cast<uint8_t*>(img.RowPtr(y1))[x1] = 255;
            }
            if (x1 == x2 && y1 == y2) break;
            e2 = 2 * err;
            if (e2 >= dy) { err += dy; x1 += sx; }
            if (e2 <= dx) { err += dx; y1 += sy; }
        }

        return img;
    }
};

TEST_F(HoughLinesPTest, EmptyImage) {
    QImage emptyImg;
    auto segments = HoughLinesP(emptyImg);
    EXPECT_TRUE(segments.empty());
}

TEST_F(HoughLinesPTest, EmptyPoints) {
    std::vector<Point2d> points;
    auto segments = HoughLinesP(points, 100, 100);
    EXPECT_TRUE(segments.empty());
}

TEST_F(HoughLinesPTest, SingleSegment) {
    QImage img = CreateLineImage(200, 100, 20, 50, 180, 50);

    HoughLineProbParams params;
    params.threshold = 30;
    params.minLineLength = 50;
    params.maxLineGap = 10;

    auto segments = HoughLinesP(img, params);

    // Should detect at least one segment
    ASSERT_GE(segments.size(), 1u);

    // Segment endpoints should be roughly correct
    double len = segments[0].Length();
    EXPECT_GT(len, 100.0);  // Should be long enough
}

TEST_F(HoughLinesPTest, SegmentAngle) {
    // Create a 45-degree line segment
    QImage img = CreateLineImage(200, 200, 50, 50, 150, 150);

    HoughLineProbParams params;
    params.threshold = 20;
    params.minLineLength = 30;
    params.maxLineGap = 10;

    auto segments = HoughLinesP(img, params);

    if (segments.size() > 0) {
        double angle = segments[0].Angle();
        // Should be close to TEST_PI/4 or -3*TEST_PI/4
        EXPECT_TRUE(std::abs(angle - TEST_PI/4) < ANGLE_TOLERANCE ||
                    std::abs(angle + 3*TEST_PI/4) < ANGLE_TOLERANCE);
    }
}

// =============================================================================
// Hough Circle Detection Tests
// =============================================================================

class HoughCirclesTest : public ::testing::Test {
protected:
    // Helper: Create edge image with a circle
    QImage CreateCircleImage(int width, int height, int cx, int cy, int radius) {
        QImage img(width, height, PixelType::UInt8, ChannelType::Gray);
        std::memset(img.Data(), 0, width * height);

        // Bresenham's circle algorithm
        int x = 0;
        int y = radius;
        int d = 3 - 2 * radius;

        while (x <= y) {
            // 8 symmetric points
            int pts[8][2] = {
                {cx + x, cy + y}, {cx - x, cy + y},
                {cx + x, cy - y}, {cx - x, cy - y},
                {cx + y, cy + x}, {cx - y, cy + x},
                {cx + y, cy - x}, {cx - y, cy - x}
            };

            for (auto& p : pts) {
                if (p[0] >= 0 && p[0] < width && p[1] >= 0 && p[1] < height) {
                    static_cast<uint8_t*>(img.RowPtr(p[1]))[p[0]] = 255;
                }
            }

            if (d < 0) {
                d += 4 * x + 6;
            } else {
                d += 4 * (x - y) + 10;
                --y;
            }
            ++x;
        }

        return img;
    }

    // Generate circle points
    std::vector<Point2d> GenerateCirclePoints(double cx, double cy, double radius, int count) {
        std::vector<Point2d> points;
        for (int i = 0; i < count; ++i) {
            double angle = 2 * TEST_PI * i / count;
            points.emplace_back(cx + radius * std::cos(angle),
                               cy + radius * std::sin(angle));
        }
        return points;
    }
};

TEST_F(HoughCirclesTest, EmptyPoints) {
    std::vector<Point2d> points;
    auto circles = HoughCircles(points, 10, 50, 0.5, 5);
    EXPECT_TRUE(circles.empty());
}

TEST_F(HoughCirclesTest, InvalidRadiusRange) {
    std::vector<Point2d> points = GenerateCirclePoints(50, 50, 30, 100);
    // minRadius > maxRadius should return empty
    auto circles = HoughCircles(points, 50, 10, 0.5, 5);
    EXPECT_TRUE(circles.empty());
}

TEST_F(HoughCirclesTest, SingleCircle) {
    auto points = GenerateCirclePoints(100, 100, 40, 200);

    auto circles = HoughCircles(points, 30, 50, 0.3, 5);

    ASSERT_GE(circles.size(), 1u);
    EXPECT_NEAR(circles[0].center.x, 100.0, POSITION_TOLERANCE);
    EXPECT_NEAR(circles[0].center.y, 100.0, POSITION_TOLERANCE);
    EXPECT_NEAR(circles[0].radius, 40.0, POSITION_TOLERANCE);
}

TEST_F(HoughCirclesTest, CircleFromImage) {
    QImage img = CreateCircleImage(200, 200, 100, 100, 40);

    auto circles = HoughCirclesStandard(img, 30, 50, 0.3, 5);

    ASSERT_GE(circles.size(), 1u);
    EXPECT_NEAR(circles[0].center.x, 100.0, POSITION_TOLERANCE);
    EXPECT_NEAR(circles[0].center.y, 100.0, POSITION_TOLERANCE);
    EXPECT_NEAR(circles[0].radius, 40.0, POSITION_TOLERANCE);
}

TEST_F(HoughCirclesTest, EmptyImage) {
    QImage emptyImg;
    auto circles = HoughCirclesStandard(emptyImg, 10, 50, 0.5, 5);
    EXPECT_TRUE(circles.empty());
}

// =============================================================================
// Utility Function Tests
// =============================================================================

class HoughUtilityTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(HoughUtilityTest, CartesianToHoughLine) {
    // Horizontal line y = 50 -> a=0, b=1, c=-50
    Line2d line1(0, 1, -50);
    HoughLine hl1 = CartesianToHoughLine(line1);
    EXPECT_NEAR(hl1.theta, TEST_PI / 2, TOLERANCE);
    EXPECT_NEAR(hl1.rho, 50.0, TOLERANCE);

    // Vertical line x = 100 -> a=1, b=0, c=-100
    Line2d line2(1, 0, -100);
    HoughLine hl2 = CartesianToHoughLine(line2);
    EXPECT_TRUE(hl2.theta < TOLERANCE || std::abs(hl2.theta - TEST_PI) < TOLERANCE);
    EXPECT_NEAR(std::abs(hl2.rho), 100.0, TOLERANCE);
}

TEST_F(HoughUtilityTest, HoughLineToCartesian) {
    // Test roundtrip conversion
    HoughLine hl(50.0, TEST_PI / 4, 1.0);
    Line2d line = HoughLineToCartesian(hl);
    HoughLine hl2 = CartesianToHoughLine(line);

    EXPECT_NEAR(hl.rho, hl2.rho, TOLERANCE);
    EXPECT_NEAR(hl.theta, hl2.theta, TOLERANCE);
}

TEST_F(HoughUtilityTest, PointToHoughLineDistance) {
    // Line at rho=50, theta=TEST_PI/2 (horizontal at y=50)
    HoughLine line(50.0, TEST_PI / 2, 1.0);

    // Point on the line
    EXPECT_NEAR(PointToHoughLineDistance(Point2d(0, 50), line), 0.0, TOLERANCE);
    EXPECT_NEAR(PointToHoughLineDistance(Point2d(100, 50), line), 0.0, TOLERANCE);

    // Point above the line
    EXPECT_NEAR(PointToHoughLineDistance(Point2d(0, 60), line), 10.0, TOLERANCE);

    // Point below the line
    EXPECT_NEAR(PointToHoughLineDistance(Point2d(0, 40), line), -10.0, TOLERANCE);
}

TEST_F(HoughUtilityTest, AreHoughLinesParallel) {
    HoughLine line1(50.0, TEST_PI / 4, 1.0);
    HoughLine line2(100.0, TEST_PI / 4, 1.0);  // Same angle, different distance
    HoughLine line3(50.0, TEST_PI / 2, 1.0);   // Different angle

    EXPECT_TRUE(AreHoughLinesParallel(line1, line2, 0.1));
    EXPECT_FALSE(AreHoughLinesParallel(line1, line3, 0.1));
}

TEST_F(HoughUtilityTest, AreHoughLinesPerpendicular) {
    HoughLine line1(50.0, 0.0, 1.0);
    HoughLine line2(50.0, TEST_PI / 2, 1.0);
    HoughLine line3(50.0, TEST_PI / 4, 1.0);

    EXPECT_TRUE(AreHoughLinesPerpendicular(line1, line2, 0.1));
    EXPECT_FALSE(AreHoughLinesPerpendicular(line1, line3, 0.1));
}

TEST_F(HoughUtilityTest, HoughLinesIntersection) {
    // Horizontal line at y=50
    HoughLine line1(50.0, TEST_PI / 2, 1.0);
    // Vertical line at x=100
    HoughLine line2(100.0, 0.0, 1.0);

    Point2d intersection;
    bool found = HoughLinesIntersection(line1, line2, intersection);

    EXPECT_TRUE(found);
    EXPECT_NEAR(intersection.x, 100.0, TOLERANCE);
    EXPECT_NEAR(intersection.y, 50.0, TOLERANCE);
}

TEST_F(HoughUtilityTest, ParallelLinesNoIntersection) {
    HoughLine line1(50.0, TEST_PI / 2, 1.0);
    HoughLine line2(100.0, TEST_PI / 2, 1.0);

    Point2d intersection;
    bool found = HoughLinesIntersection(line1, line2, intersection);

    EXPECT_FALSE(found);
}

TEST_F(HoughUtilityTest, ClipHoughLineToImage) {
    // Horizontal line at y=50
    HoughLine line(50.0, TEST_PI / 2, 1.0);

    Segment2d seg = ClipHoughLineToImage(line, 200, 100);

    // Should be clipped to [0,50] - [200,50] approximately
    EXPECT_NEAR(seg.p1.y, 50.0, POSITION_TOLERANCE);
    EXPECT_NEAR(seg.p2.y, 50.0, POSITION_TOLERANCE);
    EXPECT_TRUE(seg.p1.x >= -1 && seg.p1.x <= 201);
    EXPECT_TRUE(seg.p2.x >= -1 && seg.p2.x <= 201);
}

// =============================================================================
// Merge Functions Tests
// =============================================================================

class HoughMergeTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(HoughMergeTest, MergeHoughLines_Empty) {
    std::vector<HoughLine> lines;
    auto merged = MergeHoughLines(lines, 10.0, 0.1);
    EXPECT_TRUE(merged.empty());
}

TEST_F(HoughMergeTest, MergeHoughLines_NoMerge) {
    std::vector<HoughLine> lines;
    lines.emplace_back(50.0, 0.0, 100.0);
    lines.emplace_back(50.0, TEST_PI / 2, 80.0);

    auto merged = MergeHoughLines(lines, 10.0, 0.1);

    EXPECT_EQ(merged.size(), 2u);
}

TEST_F(HoughMergeTest, MergeHoughLines_MergeSimilar) {
    std::vector<HoughLine> lines;
    lines.emplace_back(50.0, TEST_PI / 4, 100.0);
    lines.emplace_back(52.0, TEST_PI / 4 + 0.01, 80.0);
    lines.emplace_back(48.0, TEST_PI / 4 - 0.01, 60.0);

    auto merged = MergeHoughLines(lines, 10.0, 0.1);

    EXPECT_EQ(merged.size(), 1u);
    // Should be weighted average
    EXPECT_NEAR(merged[0].theta, TEST_PI / 4, 0.02);
}

TEST_F(HoughMergeTest, MergeHoughCircles_Empty) {
    std::vector<HoughCircle> circles;
    auto merged = MergeHoughCircles(circles, 10.0, 5.0);
    EXPECT_TRUE(merged.empty());
}

TEST_F(HoughMergeTest, MergeHoughCircles_NoMerge) {
    std::vector<HoughCircle> circles;
    circles.emplace_back(Point2d(50, 50), 30.0, 100.0);
    circles.emplace_back(Point2d(150, 150), 30.0, 80.0);

    auto merged = MergeHoughCircles(circles, 10.0, 5.0);

    EXPECT_EQ(merged.size(), 2u);
}

TEST_F(HoughMergeTest, MergeHoughCircles_MergeSimilar) {
    std::vector<HoughCircle> circles;
    circles.emplace_back(Point2d(50, 50), 30.0, 100.0);
    circles.emplace_back(Point2d(52, 48), 31.0, 80.0);
    circles.emplace_back(Point2d(48, 52), 29.0, 60.0);

    auto merged = MergeHoughCircles(circles, 10.0, 5.0);

    EXPECT_EQ(merged.size(), 1u);
    EXPECT_NEAR(merged[0].center.x, 50.0, 3.0);
    EXPECT_NEAR(merged[0].center.y, 50.0, 3.0);
    EXPECT_NEAR(merged[0].radius, 30.0, 2.0);
}

// =============================================================================
// Refinement Tests
// =============================================================================

class HoughRefineTest : public ::testing::Test {
protected:
    std::vector<Point2d> GenerateNoisyLinePoints(double rho, double theta,
                                                  int count, double length,
                                                  double noiseStd) {
        std::vector<Point2d> points;
        double x0 = rho * std::cos(theta);
        double y0 = rho * std::sin(theta);
        double dx = -std::sin(theta);
        double dy = std::cos(theta);
        double nx = std::cos(theta);  // Normal direction
        double ny = std::sin(theta);

        for (int i = 0; i < count; ++i) {
            double t = (i - count / 2.0) * length / count;
            // Add noise perpendicular to line
            double noise = ((rand() % 1000) / 500.0 - 1.0) * noiseStd;
            points.emplace_back(x0 + t * dx + noise * nx,
                               y0 + t * dy + noise * ny);
        }

        return points;
    }
};

TEST_F(HoughRefineTest, RefineHoughLine_BasicTest) {
    // Initial line estimate
    HoughLine initial(50.0, TEST_PI / 2, 100.0);

    // Points near the line
    std::vector<Point2d> points;
    for (int i = 0; i < 100; ++i) {
        points.emplace_back(i, 50.0);
    }

    HoughLine refined = RefineHoughLine(initial, points, 5.0);

    EXPECT_NEAR(refined.theta, TEST_PI / 2, ANGLE_TOLERANCE);
    EXPECT_NEAR(refined.rho, 50.0, POSITION_TOLERANCE);
}

TEST_F(HoughRefineTest, RefineHoughLine_NoisyPoints) {
    // Points with noise
    auto points = GenerateNoisyLinePoints(100.0, TEST_PI / 3, 100, 200.0, 2.0);

    // Initial estimate (slightly off)
    HoughLine initial(102.0, TEST_PI / 3 + 0.02, 50.0);

    HoughLine refined = RefineHoughLine(initial, points, 10.0);

    // Should improve the estimate
    EXPECT_NEAR(refined.rho, 100.0, 5.0);
    EXPECT_NEAR(refined.theta, TEST_PI / 3, 0.1);
}

TEST_F(HoughRefineTest, RefineHoughLine_TooFewPoints) {
    HoughLine initial(50.0, TEST_PI / 2, 100.0);
    std::vector<Point2d> points;  // Empty

    HoughLine refined = RefineHoughLine(initial, points, 5.0);

    // Should return original if not enough points
    EXPECT_DOUBLE_EQ(refined.rho, initial.rho);
    EXPECT_DOUBLE_EQ(refined.theta, initial.theta);
}

TEST_F(HoughRefineTest, RefineHoughCircle_BasicTest) {
    // Initial circle estimate
    HoughCircle initial(Point2d(100, 100), 40.0, 100.0);

    // Points on circle
    std::vector<Point2d> points;
    for (int i = 0; i < 100; ++i) {
        double angle = 2 * TEST_PI * i / 100;
        points.emplace_back(100 + 40 * std::cos(angle),
                           100 + 40 * std::sin(angle));
    }

    HoughCircle refined = RefineHoughCircle(initial, points, 5.0);

    EXPECT_NEAR(refined.center.x, 100.0, POSITION_TOLERANCE);
    EXPECT_NEAR(refined.center.y, 100.0, POSITION_TOLERANCE);
    EXPECT_NEAR(refined.radius, 40.0, POSITION_TOLERANCE);
}

TEST_F(HoughRefineTest, RefineHoughCircle_TooFewPoints) {
    HoughCircle initial(Point2d(100, 100), 40.0, 100.0);
    std::vector<Point2d> points;  // Empty

    HoughCircle refined = RefineHoughCircle(initial, points, 5.0);

    // Should return original if not enough points
    EXPECT_DOUBLE_EQ(refined.center.x, initial.center.x);
    EXPECT_DOUBLE_EQ(refined.center.y, initial.center.y);
    EXPECT_DOUBLE_EQ(refined.radius, initial.radius);
}

// =============================================================================
// Configuration Parameter Tests
// =============================================================================

class HoughParamsTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(HoughParamsTest, HoughLineParams_Defaults) {
    HoughLineParams params;

    EXPECT_DOUBLE_EQ(params.rhoResolution, HOUGH_DEFAULT_RHO_RESOLUTION);
    EXPECT_DOUBLE_EQ(params.thetaResolution, HOUGH_DEFAULT_THETA_RESOLUTION);
    EXPECT_DOUBLE_EQ(params.threshold, HOUGH_DEFAULT_THRESHOLD_RATIO);
    EXPECT_TRUE(params.thresholdIsRatio);
    EXPECT_EQ(params.maxLines, 0);
    EXPECT_DOUBLE_EQ(params.minDistance, 10.0);
    EXPECT_TRUE(params.suppressOverlapping);
}

TEST_F(HoughParamsTest, HoughLineProbParams_Defaults) {
    HoughLineProbParams params;

    EXPECT_DOUBLE_EQ(params.rhoResolution, HOUGH_DEFAULT_RHO_RESOLUTION);
    EXPECT_DOUBLE_EQ(params.thetaResolution, HOUGH_DEFAULT_THETA_RESOLUTION);
    EXPECT_DOUBLE_EQ(params.threshold, 50);
    EXPECT_DOUBLE_EQ(params.minLineLength, HOUGH_DEFAULT_MIN_LINE_LENGTH);
    EXPECT_DOUBLE_EQ(params.maxLineGap, HOUGH_DEFAULT_MAX_LINE_GAP);
    EXPECT_EQ(params.maxLines, 0);
}

TEST_F(HoughParamsTest, HoughCircleParams_Defaults) {
    HoughCircleParams params;

    EXPECT_DOUBLE_EQ(params.dp, 1.0);
    EXPECT_DOUBLE_EQ(params.minDist, 20.0);
    EXPECT_DOUBLE_EQ(params.param1, 100.0);
    EXPECT_DOUBLE_EQ(params.param2, 50.0);
    EXPECT_EQ(params.minRadius, 5);
    EXPECT_EQ(params.maxRadius, 0);
    EXPECT_EQ(params.maxCircles, 0);
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

class HoughEdgeCasesTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(HoughEdgeCasesTest, SinglePoint) {
    std::vector<Point2d> points;
    points.emplace_back(50, 50);

    // Should not crash, may return empty or single line
    auto lines = HoughLines(points, 100, 100);
    // No assertion on size, just check it doesn't crash
}

TEST_F(HoughEdgeCasesTest, TwoPoints) {
    std::vector<Point2d> points;
    points.emplace_back(0, 0);
    points.emplace_back(100, 100);

    auto lines = HoughLines(points, 200, 200);
    // Should work without crashing
}

TEST_F(HoughEdgeCasesTest, CollinearPoints) {
    std::vector<Point2d> points;
    // All points on same line
    for (int i = 0; i < 1000; ++i) {
        points.emplace_back(i * 0.1, i * 0.1);  // y = x
    }

    HoughLineParams params;
    params.threshold = 0.5;
    params.maxLines = 1;

    auto lines = HoughLines(points, 200, 200, params);

    ASSERT_GE(lines.size(), 1u);
}

TEST_F(HoughEdgeCasesTest, VerySmallImage) {
    QImage img(10, 10, PixelType::UInt8, ChannelType::Gray);
    std::memset(img.Data(), 0, 10 * 10);
    static_cast<uint8_t*>(img.RowPtr(5))[0] = 255;
    static_cast<uint8_t*>(img.RowPtr(5))[9] = 255;

    auto lines = HoughLines(img);
    // Should handle small images without issues
}

TEST_F(HoughEdgeCasesTest, HighResolutionParameters) {
    std::vector<Point2d> points;
    for (int i = 0; i < 100; ++i) {
        points.emplace_back(i, 50);
    }

    HoughLineParams params;
    params.rhoResolution = 0.1;  // High resolution
    params.thetaResolution = TEST_PI / 720;  // 0.25 degree

    auto lines = HoughLines(points, 200, 100, params);
    // Should work with high resolution
    EXPECT_GE(lines.size(), 0u);  // Just check it runs
}

TEST_F(HoughEdgeCasesTest, LowThreshold) {
    std::vector<Point2d> points;
    for (int i = 0; i < 10; ++i) {
        points.emplace_back(i * 10, 50);
    }

    HoughLineParams params;
    params.threshold = 0.1;  // Very low
    params.maxLines = 100;

    auto lines = HoughLines(points, 200, 100, params);
    // May return many lines
    EXPECT_GE(lines.size(), 0u);
}

