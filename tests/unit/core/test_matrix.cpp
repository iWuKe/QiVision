#include <gtest/gtest.h>
#include <QiVision/Core/QMatrix.h>
#include <QiVision/Core/Constants.h>
#include <cmath>

using namespace Qi::Vision;

// =============================================================================
// Constructor Tests
// =============================================================================

TEST(QMatrixTest, DefaultConstructor_Identity) {
    QMatrix m;
    EXPECT_TRUE(m.IsIdentity());
    EXPECT_DOUBLE_EQ(m.M00(), 1.0);
    EXPECT_DOUBLE_EQ(m.M01(), 0.0);
    EXPECT_DOUBLE_EQ(m.M02(), 0.0);
    EXPECT_DOUBLE_EQ(m.M10(), 0.0);
    EXPECT_DOUBLE_EQ(m.M11(), 1.0);
    EXPECT_DOUBLE_EQ(m.M12(), 0.0);
}

TEST(QMatrixTest, ParameterizedConstructor) {
    QMatrix m(1, 2, 3, 4, 5, 6);
    EXPECT_DOUBLE_EQ(m.M00(), 1.0);
    EXPECT_DOUBLE_EQ(m.M01(), 2.0);
    EXPECT_DOUBLE_EQ(m.M02(), 3.0);
    EXPECT_DOUBLE_EQ(m.M10(), 4.0);
    EXPECT_DOUBLE_EQ(m.M11(), 5.0);
    EXPECT_DOUBLE_EQ(m.M12(), 6.0);
}

TEST(QMatrixTest, ArrayConstructor) {
    double elements[6] = {1, 2, 3, 4, 5, 6};
    QMatrix m(elements);
    EXPECT_DOUBLE_EQ(m.M00(), 1.0);
    EXPECT_DOUBLE_EQ(m.M01(), 2.0);
    EXPECT_DOUBLE_EQ(m.M02(), 3.0);
    EXPECT_DOUBLE_EQ(m.M10(), 4.0);
    EXPECT_DOUBLE_EQ(m.M11(), 5.0);
    EXPECT_DOUBLE_EQ(m.M12(), 6.0);
}

// =============================================================================
// Factory Method Tests
// =============================================================================

TEST(QMatrixTest, Identity) {
    QMatrix m = QMatrix::Identity();
    EXPECT_TRUE(m.IsIdentity());
}

TEST(QMatrixTest, Translation) {
    QMatrix m = QMatrix::Translation(10.0, 20.0);
    Point2d p = m.Transform({0, 0});
    EXPECT_DOUBLE_EQ(p.x, 10.0);
    EXPECT_DOUBLE_EQ(p.y, 20.0);

    p = m.Transform({5, 5});
    EXPECT_DOUBLE_EQ(p.x, 15.0);
    EXPECT_DOUBLE_EQ(p.y, 25.0);
}

TEST(QMatrixTest, Translation_Point2d) {
    QMatrix m = QMatrix::Translation(Point2d{10.0, 20.0});
    Point2d p = m.Transform({0, 0});
    EXPECT_DOUBLE_EQ(p.x, 10.0);
    EXPECT_DOUBLE_EQ(p.y, 20.0);
}

TEST(QMatrixTest, Rotation_90Degrees) {
    QMatrix m = QMatrix::Rotation(HALF_PI);
    Point2d p = m.Transform({1, 0});
    EXPECT_NEAR(p.x, 0.0, 1e-10);
    EXPECT_NEAR(p.y, 1.0, 1e-10);
}

TEST(QMatrixTest, Rotation_180Degrees) {
    QMatrix m = QMatrix::Rotation(PI);
    Point2d p = m.Transform({1, 0});
    EXPECT_NEAR(p.x, -1.0, 1e-10);
    EXPECT_NEAR(p.y, 0.0, 1e-10);
}

TEST(QMatrixTest, Rotation_AroundCenter) {
    QMatrix m = QMatrix::Rotation(HALF_PI, 5.0, 5.0);
    // Point at (6, 5) rotated 90° around (5, 5) should go to (5, 6)
    Point2d p = m.Transform({6, 5});
    EXPECT_NEAR(p.x, 5.0, 1e-10);
    EXPECT_NEAR(p.y, 6.0, 1e-10);
}

TEST(QMatrixTest, Rotation_AroundCenter_Point2d) {
    QMatrix m = QMatrix::Rotation(HALF_PI, Point2d{5.0, 5.0});
    Point2d p = m.Transform({6, 5});
    EXPECT_NEAR(p.x, 5.0, 1e-10);
    EXPECT_NEAR(p.y, 6.0, 1e-10);
}

TEST(QMatrixTest, Scaling_Uniform) {
    QMatrix m = QMatrix::Scaling(2.0);
    Point2d p = m.Transform({3, 4});
    EXPECT_DOUBLE_EQ(p.x, 6.0);
    EXPECT_DOUBLE_EQ(p.y, 8.0);
}

TEST(QMatrixTest, Scaling_NonUniform) {
    QMatrix m = QMatrix::Scaling(2.0, 3.0);
    Point2d p = m.Transform({3, 4});
    EXPECT_DOUBLE_EQ(p.x, 6.0);
    EXPECT_DOUBLE_EQ(p.y, 12.0);
}

TEST(QMatrixTest, Scaling_AroundCenter) {
    QMatrix m = QMatrix::Scaling(2.0, 2.0, Point2d{5.0, 5.0});
    // Point at (6, 6) scaled 2x around (5, 5) should go to (7, 7)
    Point2d p = m.Transform({6, 6});
    EXPECT_DOUBLE_EQ(p.x, 7.0);
    EXPECT_DOUBLE_EQ(p.y, 7.0);
}

TEST(QMatrixTest, Shearing) {
    QMatrix m = QMatrix::Shearing(1.0, 0.0);
    Point2d p = m.Transform({0, 1});
    EXPECT_DOUBLE_EQ(p.x, 1.0);
    EXPECT_DOUBLE_EQ(p.y, 1.0);
}

TEST(QMatrixTest, FromPoints) {
    Point2d src[3] = {{0, 0}, {1, 0}, {0, 1}};
    Point2d dst[3] = {{10, 10}, {11, 10}, {10, 11}};
    QMatrix m = QMatrix::FromPoints(src, dst);

    // Should be pure translation by (10, 10)
    Point2d p = m.Transform({0, 0});
    EXPECT_NEAR(p.x, 10.0, 1e-10);
    EXPECT_NEAR(p.y, 10.0, 1e-10);

    p = m.Transform({1, 0});
    EXPECT_NEAR(p.x, 11.0, 1e-10);
    EXPECT_NEAR(p.y, 10.0, 1e-10);
}

TEST(QMatrixTest, FromPoints_WithRotation) {
    Point2d src[3] = {{0, 0}, {1, 0}, {0, 1}};
    // Rotate 90° around origin and translate to (10, 10)
    Point2d dst[3] = {{10, 10}, {10, 11}, {9, 10}};
    QMatrix m = QMatrix::FromPoints(src, dst);

    Point2d p = m.Transform({0, 0});
    EXPECT_NEAR(p.x, 10.0, 1e-10);
    EXPECT_NEAR(p.y, 10.0, 1e-10);

    p = m.Transform({1, 0});
    EXPECT_NEAR(p.x, 10.0, 1e-10);
    EXPECT_NEAR(p.y, 11.0, 1e-10);
}

// =============================================================================
// Matrix Operation Tests
// =============================================================================

TEST(QMatrixTest, Multiplication_TranslationComposition) {
    QMatrix t1 = QMatrix::Translation(10, 0);
    QMatrix t2 = QMatrix::Translation(0, 20);
    QMatrix m = t1 * t2;

    Point2d p = m.Transform({0, 0});
    EXPECT_DOUBLE_EQ(p.x, 10.0);
    EXPECT_DOUBLE_EQ(p.y, 20.0);
}

TEST(QMatrixTest, Multiplication_RotationComposition) {
    QMatrix r1 = QMatrix::Rotation(HALF_PI);
    QMatrix r2 = QMatrix::Rotation(HALF_PI);
    QMatrix m = r1 * r2;

    // 90° + 90° = 180°
    Point2d p = m.Transform({1, 0});
    EXPECT_NEAR(p.x, -1.0, 1e-10);
    EXPECT_NEAR(p.y, 0.0, 1e-10);
}

TEST(QMatrixTest, CompoundAssignment) {
    QMatrix m = QMatrix::Translation(10, 0);
    m *= QMatrix::Translation(0, 20);

    Point2d p = m.Transform({0, 0});
    EXPECT_DOUBLE_EQ(p.x, 10.0);
    EXPECT_DOUBLE_EQ(p.y, 20.0);
}

TEST(QMatrixTest, Equality) {
    QMatrix m1(1, 2, 3, 4, 5, 6);
    QMatrix m2(1, 2, 3, 4, 5, 6);
    QMatrix m3(1, 2, 3, 4, 5, 7);

    EXPECT_TRUE(m1 == m2);
    EXPECT_FALSE(m1 == m3);
    EXPECT_FALSE(m1 != m2);
    EXPECT_TRUE(m1 != m3);
}

TEST(QMatrixTest, Inverse_Translation) {
    QMatrix m = QMatrix::Translation(10, 20);
    QMatrix inv = m.Inverse();
    QMatrix identity = m * inv;

    EXPECT_TRUE(identity.IsIdentity());
}

TEST(QMatrixTest, Inverse_Rotation) {
    QMatrix m = QMatrix::Rotation(0.5);
    QMatrix inv = m.Inverse();
    QMatrix identity = m * inv;

    EXPECT_TRUE(identity.IsIdentity());
}

TEST(QMatrixTest, Inverse_Scaling) {
    QMatrix m = QMatrix::Scaling(2.0, 3.0);
    QMatrix inv = m.Inverse();
    QMatrix identity = m * inv;

    EXPECT_TRUE(identity.IsIdentity());
}

TEST(QMatrixTest, Inverse_Complex) {
    // Rotation + Translation + Scaling
    QMatrix m = QMatrix::Translation(10, 20) * QMatrix::Rotation(0.3) * QMatrix::Scaling(2.0, 1.5);
    QMatrix inv = m.Inverse();
    QMatrix identity = m * inv;

    EXPECT_NEAR(identity.M00(), 1.0, 1e-10);
    EXPECT_NEAR(identity.M01(), 0.0, 1e-10);
    EXPECT_NEAR(identity.M02(), 0.0, 1e-10);
    EXPECT_NEAR(identity.M10(), 0.0, 1e-10);
    EXPECT_NEAR(identity.M11(), 1.0, 1e-10);
    EXPECT_NEAR(identity.M12(), 0.0, 1e-10);
}

TEST(QMatrixTest, IsInvertible) {
    EXPECT_TRUE(QMatrix::Translation(10, 20).IsInvertible());
    EXPECT_TRUE(QMatrix::Rotation(0.5).IsInvertible());
    EXPECT_TRUE(QMatrix::Scaling(2.0, 3.0).IsInvertible());

    // Singular matrix (scale to zero)
    QMatrix singular(0, 0, 0, 0, 0, 0);
    EXPECT_FALSE(singular.IsInvertible());
}

TEST(QMatrixTest, Determinant) {
    EXPECT_DOUBLE_EQ(QMatrix::Identity().Determinant(), 1.0);
    EXPECT_DOUBLE_EQ(QMatrix::Scaling(2.0, 3.0).Determinant(), 6.0);
    EXPECT_NEAR(QMatrix::Rotation(0.5).Determinant(), 1.0, 1e-10);
}

TEST(QMatrixTest, TransposeLinear) {
    QMatrix m(1, 2, 3, 4, 5, 6);
    QMatrix t = m.TransposeLinear();

    EXPECT_DOUBLE_EQ(t.M00(), 1.0);
    EXPECT_DOUBLE_EQ(t.M01(), 4.0);  // swapped
    EXPECT_DOUBLE_EQ(t.M02(), 3.0);  // unchanged
    EXPECT_DOUBLE_EQ(t.M10(), 2.0);  // swapped
    EXPECT_DOUBLE_EQ(t.M11(), 5.0);
    EXPECT_DOUBLE_EQ(t.M12(), 6.0);  // unchanged
}

// =============================================================================
// Element Access Tests
// =============================================================================

TEST(QMatrixTest, At) {
    QMatrix m(1, 2, 3, 4, 5, 6);
    EXPECT_DOUBLE_EQ(m.At(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m.At(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m.At(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(m.At(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(m.At(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(m.At(1, 2), 6.0);
    EXPECT_DOUBLE_EQ(m.At(2, 0), 0.0);
    EXPECT_DOUBLE_EQ(m.At(2, 1), 0.0);
    EXPECT_DOUBLE_EQ(m.At(2, 2), 1.0);
}

TEST(QMatrixTest, At_OutOfRange) {
    QMatrix m;
    EXPECT_THROW(m.At(3, 0), std::out_of_range);
    EXPECT_THROW(m.At(0, 3), std::out_of_range);
    EXPECT_THROW(m.At(-1, 0), std::out_of_range);
}

TEST(QMatrixTest, SetAt) {
    QMatrix m;
    m.SetAt(0, 0, 2.0);
    m.SetAt(0, 2, 10.0);
    m.SetAt(1, 2, 20.0);

    EXPECT_DOUBLE_EQ(m.M00(), 2.0);
    EXPECT_DOUBLE_EQ(m.M02(), 10.0);
    EXPECT_DOUBLE_EQ(m.M12(), 20.0);
}

TEST(QMatrixTest, SetAt_OutOfRange) {
    QMatrix m;
    EXPECT_THROW(m.SetAt(2, 0, 1.0), std::out_of_range);  // Third row is fixed
}

TEST(QMatrixTest, GetElements) {
    QMatrix m(1, 2, 3, 4, 5, 6);
    double elements[6];
    m.GetElements(elements);

    EXPECT_DOUBLE_EQ(elements[0], 1.0);
    EXPECT_DOUBLE_EQ(elements[1], 2.0);
    EXPECT_DOUBLE_EQ(elements[2], 3.0);
    EXPECT_DOUBLE_EQ(elements[3], 4.0);
    EXPECT_DOUBLE_EQ(elements[4], 5.0);
    EXPECT_DOUBLE_EQ(elements[5], 6.0);
}

TEST(QMatrixTest, GetMatrix3x3) {
    QMatrix m(1, 2, 3, 4, 5, 6);
    double matrix[9];
    m.GetMatrix3x3(matrix);

    EXPECT_DOUBLE_EQ(matrix[0], 1.0);
    EXPECT_DOUBLE_EQ(matrix[1], 2.0);
    EXPECT_DOUBLE_EQ(matrix[2], 3.0);
    EXPECT_DOUBLE_EQ(matrix[3], 4.0);
    EXPECT_DOUBLE_EQ(matrix[4], 5.0);
    EXPECT_DOUBLE_EQ(matrix[5], 6.0);
    EXPECT_DOUBLE_EQ(matrix[6], 0.0);
    EXPECT_DOUBLE_EQ(matrix[7], 0.0);
    EXPECT_DOUBLE_EQ(matrix[8], 1.0);
}

// =============================================================================
// Point Transformation Tests
// =============================================================================

TEST(QMatrixTest, Transform_Identity) {
    QMatrix m;
    Point2d p = m.Transform({5, 10});
    EXPECT_DOUBLE_EQ(p.x, 5.0);
    EXPECT_DOUBLE_EQ(p.y, 10.0);
}

TEST(QMatrixTest, Transform_Operator) {
    QMatrix m = QMatrix::Translation(10, 20);
    Point2d p = m * Point2d{5, 5};
    EXPECT_DOUBLE_EQ(p.x, 15.0);
    EXPECT_DOUBLE_EQ(p.y, 25.0);
}

TEST(QMatrixTest, TransformPoints_InPlace) {
    QMatrix m = QMatrix::Translation(10, 20);
    Point2d points[3] = {{0, 0}, {1, 1}, {2, 2}};
    m.TransformPoints(points, 3);

    EXPECT_DOUBLE_EQ(points[0].x, 10.0);
    EXPECT_DOUBLE_EQ(points[0].y, 20.0);
    EXPECT_DOUBLE_EQ(points[1].x, 11.0);
    EXPECT_DOUBLE_EQ(points[1].y, 21.0);
    EXPECT_DOUBLE_EQ(points[2].x, 12.0);
    EXPECT_DOUBLE_EQ(points[2].y, 22.0);
}

TEST(QMatrixTest, TransformPoints_SeparateArrays) {
    QMatrix m = QMatrix::Scaling(2.0);
    Point2d src[3] = {{1, 1}, {2, 2}, {3, 3}};
    Point2d dst[3];
    m.TransformPoints(src, dst, 3);

    EXPECT_DOUBLE_EQ(dst[0].x, 2.0);
    EXPECT_DOUBLE_EQ(dst[0].y, 2.0);
    EXPECT_DOUBLE_EQ(dst[1].x, 4.0);
    EXPECT_DOUBLE_EQ(dst[1].y, 4.0);
    EXPECT_DOUBLE_EQ(dst[2].x, 6.0);
    EXPECT_DOUBLE_EQ(dst[2].y, 6.0);
}

TEST(QMatrixTest, TransformVector) {
    QMatrix m = QMatrix::Translation(100, 200) * QMatrix::Scaling(2.0);

    // TransformVector should ignore translation
    Point2d v = m.TransformVector({1, 0});
    EXPECT_DOUBLE_EQ(v.x, 2.0);
    EXPECT_DOUBLE_EQ(v.y, 0.0);
}

TEST(QMatrixTest, TransformVector_Rotation) {
    QMatrix m = QMatrix::Translation(100, 200) * QMatrix::Rotation(HALF_PI);
    Point2d v = m.TransformVector({1, 0});
    EXPECT_NEAR(v.x, 0.0, 1e-10);
    EXPECT_NEAR(v.y, 1.0, 1e-10);
}

// =============================================================================
// Decomposition Tests
// =============================================================================

TEST(QMatrixTest, GetTranslation) {
    QMatrix m = QMatrix::Translation(10, 20);
    Point2d t = m.GetTranslation();
    EXPECT_DOUBLE_EQ(t.x, 10.0);
    EXPECT_DOUBLE_EQ(t.y, 20.0);
}

TEST(QMatrixTest, GetRotation) {
    QMatrix m = QMatrix::Rotation(0.5);
    EXPECT_NEAR(m.GetRotation(), 0.5, 1e-10);
}

TEST(QMatrixTest, GetRotation_Combined) {
    QMatrix m = QMatrix::Translation(10, 20) * QMatrix::Rotation(0.3);
    EXPECT_NEAR(m.GetRotation(), 0.3, 1e-10);
}

TEST(QMatrixTest, GetScale) {
    QMatrix m = QMatrix::Scaling(2.0, 3.0);
    double sx, sy;
    m.GetScale(sx, sy);
    EXPECT_NEAR(sx, 2.0, 1e-10);
    EXPECT_NEAR(sy, 3.0, 1e-10);
}

TEST(QMatrixTest, GetScale_WithRotation) {
    QMatrix m = QMatrix::Rotation(0.5) * QMatrix::Scaling(2.0, 3.0);
    double sx, sy;
    m.GetScale(sx, sy);
    EXPECT_NEAR(sx, 2.0, 1e-10);
    EXPECT_NEAR(sy, 3.0, 1e-10);
}

TEST(QMatrixTest, IsIdentity) {
    EXPECT_TRUE(QMatrix::Identity().IsIdentity());
    EXPECT_TRUE(QMatrix().IsIdentity());
    EXPECT_FALSE(QMatrix::Translation(1, 0).IsIdentity());
    EXPECT_FALSE(QMatrix::Rotation(0.1).IsIdentity());
}

TEST(QMatrixTest, IsTranslationOnly) {
    EXPECT_TRUE(QMatrix::Identity().IsTranslationOnly());
    EXPECT_TRUE(QMatrix::Translation(10, 20).IsTranslationOnly());
    EXPECT_FALSE(QMatrix::Rotation(0.1).IsTranslationOnly());
    EXPECT_FALSE(QMatrix::Scaling(2.0).IsTranslationOnly());
}

TEST(QMatrixTest, PreservesOrientation) {
    EXPECT_TRUE(QMatrix::Identity().PreservesOrientation());
    EXPECT_TRUE(QMatrix::Rotation(0.5).PreservesOrientation());
    EXPECT_TRUE(QMatrix::Translation(10, 20).PreservesOrientation());
    EXPECT_TRUE(QMatrix::Scaling(2.0, 3.0).PreservesOrientation());

    // Reflection (negative scale) does NOT preserve orientation
    EXPECT_FALSE(QMatrix::Scaling(-1.0, 1.0).PreservesOrientation());
}

// =============================================================================
// Alias Test
// =============================================================================

TEST(QMatrixTest, QHomMat2dAlias) {
    QHomMat2d m = QHomMat2d::Translation(10, 20);
    Point2d p = m.Transform({0, 0});
    EXPECT_DOUBLE_EQ(p.x, 10.0);
    EXPECT_DOUBLE_EQ(p.y, 20.0);
}
