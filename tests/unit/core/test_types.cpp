#include <gtest/gtest.h>
#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>

using namespace Qi::Vision;

// =============================================================================
// Point2d Tests
// =============================================================================

TEST(Point2dTest, DefaultConstructor) {
    Point2d p;
    EXPECT_DOUBLE_EQ(p.x, 0.0);
    EXPECT_DOUBLE_EQ(p.y, 0.0);
}

TEST(Point2dTest, ParameterizedConstructor) {
    Point2d p(3.0, 4.0);
    EXPECT_DOUBLE_EQ(p.x, 3.0);
    EXPECT_DOUBLE_EQ(p.y, 4.0);
}

TEST(Point2dTest, Norm) {
    Point2d p(3.0, 4.0);
    EXPECT_DOUBLE_EQ(p.Norm(), 5.0);
}

TEST(Point2dTest, DotProduct) {
    Point2d a(1.0, 2.0);
    Point2d b(3.0, 4.0);
    EXPECT_DOUBLE_EQ(a.Dot(b), 11.0);
}

TEST(Point2dTest, CrossProduct) {
    Point2d a(1.0, 0.0);
    Point2d b(0.0, 1.0);
    EXPECT_DOUBLE_EQ(a.Cross(b), 1.0);
}

TEST(Point2dTest, Distance) {
    Point2d a(0.0, 0.0);
    Point2d b(3.0, 4.0);
    EXPECT_DOUBLE_EQ(a.DistanceTo(b), 5.0);
}

// =============================================================================
// Rect2i Tests
// =============================================================================

TEST(Rect2iTest, BasicProperties) {
    Rect2i r(10, 20, 100, 50);
    EXPECT_EQ(r.x, 10);
    EXPECT_EQ(r.y, 20);
    EXPECT_EQ(r.width, 100);
    EXPECT_EQ(r.height, 50);
    EXPECT_EQ(r.Right(), 110);
    EXPECT_EQ(r.Bottom(), 70);
    EXPECT_EQ(r.Area(), 5000);
}

TEST(Rect2iTest, Contains) {
    Rect2i r(10, 20, 100, 50);
    EXPECT_TRUE(r.Contains(50, 40));
    EXPECT_FALSE(r.Contains(5, 40));
    EXPECT_FALSE(r.Contains(50, 100));
}

// =============================================================================
// Constants Tests
// =============================================================================

TEST(ConstantsTest, MathConstants) {
    EXPECT_NEAR(PI, 3.14159265358979, 1e-10);
    EXPECT_NEAR(TWO_PI, 2.0 * PI, 1e-10);
    EXPECT_NEAR(DEG_TO_RAD * 180.0, PI, 1e-10);
}

TEST(ConstantsTest, ApproxEqual) {
    EXPECT_TRUE(ApproxEqual(1.0, 1.0 + 1e-10));
    EXPECT_FALSE(ApproxEqual(1.0, 1.001));
}

TEST(ConstantsTest, Clamp) {
    EXPECT_EQ(Clamp(5, 0, 10), 5);
    EXPECT_EQ(Clamp(-1, 0, 10), 0);
    EXPECT_EQ(Clamp(15, 0, 10), 10);
}

TEST(ConstantsTest, NormalizeAngle) {
    EXPECT_NEAR(NormalizeAngle(0.0), 0.0, 1e-10);
    EXPECT_NEAR(NormalizeAngle(PI), PI, 1e-10);
    EXPECT_NEAR(NormalizeAngle(3 * PI), PI, 1e-10);
    EXPECT_NEAR(NormalizeAngle(-3 * PI), -PI, 1e-10);
}

// =============================================================================
// QImage Tests
// =============================================================================

TEST(QImageTest, DefaultConstructor) {
    QImage img;
    EXPECT_TRUE(img.Empty());
    EXPECT_FALSE(img.IsValid());
}

TEST(QImageTest, CreateWithDimensions) {
    QImage img(640, 480);
    EXPECT_FALSE(img.Empty());
    EXPECT_TRUE(img.IsValid());
    EXPECT_EQ(img.Width(), 640);
    EXPECT_EQ(img.Height(), 480);
    EXPECT_EQ(img.Channels(), 1);
    EXPECT_EQ(img.Type(), PixelType::UInt8);
}

TEST(QImageTest, StrideAlignment) {
    QImage img(100, 100);
    // Stride should be aligned to 64 bytes
    EXPECT_EQ(img.Stride() % 64, 0u);
}

TEST(QImageTest, PixelAccess) {
    QImage img(100, 100);
    img.SetAt(50, 50, 128);
    EXPECT_EQ(img.At(50, 50), 128);
}

TEST(QImageTest, Clone) {
    QImage img(100, 100);
    img.SetAt(50, 50, 200);

    QImage clone = img.Clone();
    EXPECT_EQ(clone.Width(), 100);
    EXPECT_EQ(clone.Height(), 100);
    EXPECT_EQ(clone.At(50, 50), 200);

    // Modify original, clone should be unchanged
    img.SetAt(50, 50, 100);
    EXPECT_EQ(clone.At(50, 50), 200);
}

TEST(QImageTest, Domain) {
    QImage img(100, 100);
    EXPECT_TRUE(img.IsFullDomain());

    QRegion roi = QRegion::Rectangle(10, 10, 50, 50);
    img.SetDomain(roi);
    EXPECT_FALSE(img.IsFullDomain());

    img.ResetDomain();
    EXPECT_TRUE(img.IsFullDomain());
}

// =============================================================================
// QRegion Tests
// =============================================================================

TEST(QRegionTest, DefaultConstructor) {
    QRegion region;
    EXPECT_TRUE(region.Empty());
    EXPECT_EQ(region.Area(), 0);
}

TEST(QRegionTest, Rectangle) {
    QRegion region = QRegion::Rectangle(10, 20, 100, 50);
    EXPECT_FALSE(region.Empty());
    EXPECT_EQ(region.Area(), 5000);
    EXPECT_EQ(region.RunCount(), 50u);
}

TEST(QRegionTest, Circle) {
    QRegion region = QRegion::Circle(50, 50, 10);
    EXPECT_FALSE(region.Empty());
    // Circle area ≈ π * r²
    EXPECT_NEAR(region.Area(), PI * 100, 50);  // Allow some discretization error
}

TEST(QRegionTest, Contains) {
    QRegion region = QRegion::Rectangle(10, 10, 20, 20);
    EXPECT_TRUE(region.Contains(15, 15));
    EXPECT_TRUE(region.Contains(10, 10));
    EXPECT_FALSE(region.Contains(30, 30));
    EXPECT_FALSE(region.Contains(5, 15));
}

TEST(QRegionTest, Union) {
    QRegion r1 = QRegion::Rectangle(0, 0, 10, 10);
    QRegion r2 = QRegion::Rectangle(5, 5, 10, 10);
    QRegion u = r1.Union(r2);

    EXPECT_TRUE(u.Contains(0, 0));
    EXPECT_TRUE(u.Contains(14, 14));
    EXPECT_TRUE(u.Contains(7, 7));
}

TEST(QRegionTest, Intersection) {
    QRegion r1 = QRegion::Rectangle(0, 0, 10, 10);
    QRegion r2 = QRegion::Rectangle(5, 5, 10, 10);
    QRegion inter = r1.Intersection(r2);

    EXPECT_FALSE(inter.Contains(0, 0));
    EXPECT_FALSE(inter.Contains(14, 14));
    EXPECT_TRUE(inter.Contains(7, 7));
    EXPECT_EQ(inter.Area(), 25);  // 5x5 overlap
}

TEST(QRegionTest, Translate) {
    QRegion region = QRegion::Rectangle(0, 0, 10, 10);
    QRegion translated = region.Translate(5, 5);

    EXPECT_FALSE(translated.Contains(0, 0));
    EXPECT_TRUE(translated.Contains(5, 5));
    EXPECT_TRUE(translated.Contains(14, 14));
}

TEST(QRegionTest, BoundingBox) {
    QRegion region = QRegion::Rectangle(10, 20, 30, 40);
    Rect2i bbox = region.BoundingBox();

    EXPECT_EQ(bbox.x, 10);
    EXPECT_EQ(bbox.y, 20);
    EXPECT_EQ(bbox.width, 30);
    EXPECT_EQ(bbox.height, 40);
}

// =============================================================================
// Line2d Tests
// =============================================================================

TEST(Line2dTest, FromPoints) {
    Line2d line = Line2d::FromPoints({0, 0}, {1, 0});
    EXPECT_NEAR(line.Angle(), 0.0, 1e-10);
    EXPECT_NEAR(line.Distance({0.5, 1.0}), 1.0, 1e-10);
}

TEST(Line2dTest, FromPointAngle) {
    Line2d line = Line2d::FromPointAngle({0, 0}, PI / 4);
    EXPECT_NEAR(line.Angle(), PI / 4, 1e-10);
}

TEST(Line2dTest, SignedDistance) {
    Line2d line = Line2d::FromPoints({0, 0}, {1, 0});
    EXPECT_GT(line.SignedDistance({0, 1}), 0);
    EXPECT_LT(line.SignedDistance({0, -1}), 0);
}

// =============================================================================
// Segment2d Tests
// =============================================================================

TEST(Segment2dTest, Length) {
    Segment2d seg({0, 0}, {3, 4});
    EXPECT_DOUBLE_EQ(seg.Length(), 5.0);
}

TEST(Segment2dTest, Midpoint) {
    Segment2d seg({0, 0}, {4, 6});
    Point2d mid = seg.Midpoint();
    EXPECT_DOUBLE_EQ(mid.x, 2.0);
    EXPECT_DOUBLE_EQ(mid.y, 3.0);
}

TEST(Segment2dTest, DistanceToPoint) {
    Segment2d seg({0, 0}, {10, 0});
    // Point directly above midpoint
    EXPECT_DOUBLE_EQ(seg.DistanceToPoint({5, 5}), 5.0);
    // Point beyond endpoint
    EXPECT_DOUBLE_EQ(seg.DistanceToPoint({15, 0}), 5.0);
    // Point before start
    EXPECT_DOUBLE_EQ(seg.DistanceToPoint({-3, 4}), 5.0);
}

TEST(Segment2dTest, PointAt) {
    Segment2d seg({0, 0}, {10, 0});
    Point2d p = seg.PointAt(0.5);
    EXPECT_DOUBLE_EQ(p.x, 5.0);
    EXPECT_DOUBLE_EQ(p.y, 0.0);
}

// =============================================================================
// Circle2d Tests
// =============================================================================

TEST(Circle2dTest, AreaAndCircumference) {
    Circle2d circle(0, 0, 10);
    EXPECT_NEAR(circle.Area(), PI * 100, 1e-10);
    EXPECT_NEAR(circle.Circumference(), TWO_PI * 10, 1e-10);
}

TEST(Circle2dTest, Contains) {
    Circle2d circle(0, 0, 10);
    EXPECT_TRUE(circle.Contains({5, 5}));
    EXPECT_TRUE(circle.Contains({10, 0}));
    EXPECT_FALSE(circle.Contains({10, 10}));
}

// =============================================================================
// Ellipse2d Tests
// =============================================================================

TEST(Ellipse2dTest, Area) {
    Ellipse2d ellipse(0, 0, 10, 5);
    EXPECT_NEAR(ellipse.Area(), PI * 50, 1e-10);
}

TEST(Ellipse2dTest, Eccentricity) {
    // Circle has eccentricity 0
    Ellipse2d circle(0, 0, 10, 10);
    EXPECT_NEAR(circle.Eccentricity(), 0.0, 1e-10);

    // Flat ellipse has eccentricity close to 1
    Ellipse2d flat(0, 0, 10, 1);
    EXPECT_GT(flat.Eccentricity(), 0.9);
}

TEST(Ellipse2dTest, Contains) {
    Ellipse2d ellipse(0, 0, 10, 5);
    EXPECT_TRUE(ellipse.Contains({5, 2}));
    EXPECT_TRUE(ellipse.Contains({10, 0}));
    EXPECT_FALSE(ellipse.Contains({10, 5}));
}

TEST(Ellipse2dTest, PointAt) {
    Ellipse2d ellipse(0, 0, 10, 5);
    Point2d p0 = ellipse.PointAt(0);
    EXPECT_NEAR(p0.x, 10, 1e-10);
    EXPECT_NEAR(p0.y, 0, 1e-10);

    Point2d p90 = ellipse.PointAt(HALF_PI);
    EXPECT_NEAR(p90.x, 0, 1e-10);
    EXPECT_NEAR(p90.y, 5, 1e-10);
}

// =============================================================================
// Arc2d Tests
// =============================================================================

TEST(Arc2dTest, Length) {
    Arc2d arc(0, 0, 10, 0, HALF_PI);
    EXPECT_NEAR(arc.Length(), HALF_PI * 10, 1e-10);
}

TEST(Arc2dTest, StartEndPoints) {
    Arc2d arc(0, 0, 10, 0, HALF_PI);
    Point2d start = arc.StartPoint();
    Point2d end = arc.EndPoint();

    EXPECT_NEAR(start.x, 10, 1e-10);
    EXPECT_NEAR(start.y, 0, 1e-10);
    EXPECT_NEAR(end.x, 0, 1e-10);
    EXPECT_NEAR(end.y, 10, 1e-10);
}

TEST(Arc2dTest, Midpoint) {
    Arc2d arc(0, 0, 10, 0, HALF_PI);
    Point2d mid = arc.Midpoint();
    double expected = 10 * std::cos(PI / 4);
    EXPECT_NEAR(mid.x, expected, 1e-10);
    EXPECT_NEAR(mid.y, expected, 1e-10);
}

// =============================================================================
// RotatedRect2d Tests
// =============================================================================

TEST(RotatedRect2dTest, Area) {
    RotatedRect2d rect(0, 0, 10, 20);
    EXPECT_DOUBLE_EQ(rect.Area(), 200);
}

TEST(RotatedRect2dTest, Corners_NoRotation) {
    RotatedRect2d rect(0, 0, 10, 20, 0);
    Point2d corners[4];
    rect.GetCorners(corners);

    // Corners should be at (-5,-10), (5,-10), (5,10), (-5,10)
    EXPECT_NEAR(corners[0].x, -5, 1e-10);
    EXPECT_NEAR(corners[0].y, -10, 1e-10);
    EXPECT_NEAR(corners[2].x, 5, 1e-10);
    EXPECT_NEAR(corners[2].y, 10, 1e-10);
}

TEST(RotatedRect2dTest, Corners_Rotated90) {
    RotatedRect2d rect(0, 0, 10, 20, HALF_PI);
    Point2d corners[4];
    rect.GetCorners(corners);

    // After 90° rotation, corners swap roles
    EXPECT_NEAR(corners[0].x, 10, 1e-10);
    EXPECT_NEAR(corners[0].y, -5, 1e-10);
}

TEST(RotatedRect2dTest, BoundingBox) {
    RotatedRect2d rect(0, 0, 10, 10, PI / 4);
    Rect2d bbox = rect.BoundingBox();

    // 45° rotated square has larger bounding box
    double expected = 10 * std::sqrt(2) / 2;
    EXPECT_NEAR(bbox.width, expected * 2, 1e-10);
    EXPECT_NEAR(bbox.height, expected * 2, 1e-10);
}

TEST(RotatedRect2dTest, Contains_NoRotation) {
    RotatedRect2d rect(0, 0, 10, 20, 0);
    EXPECT_TRUE(rect.Contains({0, 0}));
    EXPECT_TRUE(rect.Contains({4, 9}));
    EXPECT_FALSE(rect.Contains({6, 0}));
    EXPECT_FALSE(rect.Contains({0, 11}));
}

TEST(RotatedRect2dTest, Contains_Rotated) {
    RotatedRect2d rect(0, 0, 10, 10, PI / 4);
    // Center should be inside
    EXPECT_TRUE(rect.Contains({0, 0}));
    // Point on axis at distance 4 from center should be inside
    EXPECT_TRUE(rect.Contains({2, 2}));
    // Point outside the rotated square
    EXPECT_FALSE(rect.Contains({5, 5}));
}
