/**
 * @file test_internal_matrix.cpp
 * @brief Unit tests for Internal/Matrix module
 */

#include <QiVision/Internal/Matrix.h>
#include <QiVision/Core/QMatrix.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

// =============================================================================
// Vec<N> Tests
// =============================================================================

class VecTest : public ::testing::Test {
protected:
    static constexpr double EPS = 1e-12;
};

TEST_F(VecTest, DefaultConstruction) {
    Vec2 v2;
    EXPECT_DOUBLE_EQ(v2[0], 0.0);
    EXPECT_DOUBLE_EQ(v2[1], 0.0);

    Vec3 v3;
    EXPECT_DOUBLE_EQ(v3[0], 0.0);
    EXPECT_DOUBLE_EQ(v3[1], 0.0);
    EXPECT_DOUBLE_EQ(v3[2], 0.0);

    Vec4 v4;
    EXPECT_DOUBLE_EQ(v4[0], 0.0);
    EXPECT_DOUBLE_EQ(v4[1], 0.0);
    EXPECT_DOUBLE_EQ(v4[2], 0.0);
    EXPECT_DOUBLE_EQ(v4[3], 0.0);
}

TEST_F(VecTest, InitializerListConstruction) {
    Vec2 v2{1.0, 2.0};
    EXPECT_DOUBLE_EQ(v2[0], 1.0);
    EXPECT_DOUBLE_EQ(v2[1], 2.0);

    Vec3 v3{1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(v3[0], 1.0);
    EXPECT_DOUBLE_EQ(v3[1], 2.0);
    EXPECT_DOUBLE_EQ(v3[2], 3.0);

    Vec4 v4{1.0, 2.0, 3.0, 4.0};
    EXPECT_DOUBLE_EQ(v4[0], 1.0);
    EXPECT_DOUBLE_EQ(v4[1], 2.0);
    EXPECT_DOUBLE_EQ(v4[2], 3.0);
    EXPECT_DOUBLE_EQ(v4[3], 4.0);
}

TEST_F(VecTest, FactoryMethods) {
    Vec3 zero = Vec3::Zero();
    EXPECT_DOUBLE_EQ(zero[0], 0.0);
    EXPECT_DOUBLE_EQ(zero[1], 0.0);
    EXPECT_DOUBLE_EQ(zero[2], 0.0);

    Vec3 ones = Vec3::Ones();
    EXPECT_DOUBLE_EQ(ones[0], 1.0);
    EXPECT_DOUBLE_EQ(ones[1], 1.0);
    EXPECT_DOUBLE_EQ(ones[2], 1.0);

    Vec3 unit0 = Vec3::Unit(0);
    EXPECT_DOUBLE_EQ(unit0[0], 1.0);
    EXPECT_DOUBLE_EQ(unit0[1], 0.0);
    EXPECT_DOUBLE_EQ(unit0[2], 0.0);

    Vec3 unit1 = Vec3::Unit(1);
    EXPECT_DOUBLE_EQ(unit1[0], 0.0);
    EXPECT_DOUBLE_EQ(unit1[1], 1.0);
    EXPECT_DOUBLE_EQ(unit1[2], 0.0);
}

TEST_F(VecTest, MakeVecFunctions) {
    Vec2 v2 = MakeVec2(3.0, 4.0);
    EXPECT_DOUBLE_EQ(v2[0], 3.0);
    EXPECT_DOUBLE_EQ(v2[1], 4.0);

    Vec3 v3 = MakeVec3(1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(v3[0], 1.0);
    EXPECT_DOUBLE_EQ(v3[1], 2.0);
    EXPECT_DOUBLE_EQ(v3[2], 3.0);

    Vec4 v4 = MakeVec4(1.0, 2.0, 3.0, 4.0);
    EXPECT_DOUBLE_EQ(v4[0], 1.0);
    EXPECT_DOUBLE_EQ(v4[1], 2.0);
    EXPECT_DOUBLE_EQ(v4[2], 3.0);
    EXPECT_DOUBLE_EQ(v4[3], 4.0);
}

TEST_F(VecTest, ElementAccess) {
    Vec3 v{1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v(1), 2.0);
    EXPECT_DOUBLE_EQ(v[2], 3.0);

    v[0] = 10.0;
    v(1) = 20.0;
    EXPECT_DOUBLE_EQ(v[0], 10.0);
    EXPECT_DOUBLE_EQ(v[1], 20.0);
}

TEST_F(VecTest, Arithmetic) {
    Vec3 a{1.0, 2.0, 3.0};
    Vec3 b{4.0, 5.0, 6.0};

    Vec3 sum = a + b;
    EXPECT_DOUBLE_EQ(sum[0], 5.0);
    EXPECT_DOUBLE_EQ(sum[1], 7.0);
    EXPECT_DOUBLE_EQ(sum[2], 9.0);

    Vec3 diff = b - a;
    EXPECT_DOUBLE_EQ(diff[0], 3.0);
    EXPECT_DOUBLE_EQ(diff[1], 3.0);
    EXPECT_DOUBLE_EQ(diff[2], 3.0);

    Vec3 scaled = a * 2.0;
    EXPECT_DOUBLE_EQ(scaled[0], 2.0);
    EXPECT_DOUBLE_EQ(scaled[1], 4.0);
    EXPECT_DOUBLE_EQ(scaled[2], 6.0);

    Vec3 scaled2 = 3.0 * a;
    EXPECT_DOUBLE_EQ(scaled2[0], 3.0);
    EXPECT_DOUBLE_EQ(scaled2[1], 6.0);
    EXPECT_DOUBLE_EQ(scaled2[2], 9.0);

    Vec3 divided = a / 2.0;
    EXPECT_DOUBLE_EQ(divided[0], 0.5);
    EXPECT_DOUBLE_EQ(divided[1], 1.0);
    EXPECT_DOUBLE_EQ(divided[2], 1.5);

    Vec3 negated = -a;
    EXPECT_DOUBLE_EQ(negated[0], -1.0);
    EXPECT_DOUBLE_EQ(negated[1], -2.0);
    EXPECT_DOUBLE_EQ(negated[2], -3.0);
}

TEST_F(VecTest, CompoundAssignment) {
    Vec3 a{1.0, 2.0, 3.0};
    Vec3 b{4.0, 5.0, 6.0};

    a += b;
    EXPECT_DOUBLE_EQ(a[0], 5.0);
    EXPECT_DOUBLE_EQ(a[1], 7.0);
    EXPECT_DOUBLE_EQ(a[2], 9.0);

    a -= b;
    EXPECT_DOUBLE_EQ(a[0], 1.0);
    EXPECT_DOUBLE_EQ(a[1], 2.0);
    EXPECT_DOUBLE_EQ(a[2], 3.0);

    a *= 2.0;
    EXPECT_DOUBLE_EQ(a[0], 2.0);
    EXPECT_DOUBLE_EQ(a[1], 4.0);
    EXPECT_DOUBLE_EQ(a[2], 6.0);

    a /= 2.0;
    EXPECT_DOUBLE_EQ(a[0], 1.0);
    EXPECT_DOUBLE_EQ(a[1], 2.0);
    EXPECT_DOUBLE_EQ(a[2], 3.0);
}

TEST_F(VecTest, DotProduct) {
    Vec3 a{1.0, 2.0, 3.0};
    Vec3 b{4.0, 5.0, 6.0};

    double dot = a.Dot(b);
    EXPECT_DOUBLE_EQ(dot, 1.0*4.0 + 2.0*5.0 + 3.0*6.0);  // 32
}

TEST_F(VecTest, CrossProduct) {
    Vec3 x = Vec3::Unit(0);
    Vec3 y = Vec3::Unit(1);
    Vec3 z = Vec3::Unit(2);

    Vec3 xy = Cross(x, y);
    EXPECT_NEAR(xy[0], z[0], EPS);
    EXPECT_NEAR(xy[1], z[1], EPS);
    EXPECT_NEAR(xy[2], z[2], EPS);

    Vec3 yz = Cross(y, z);
    EXPECT_NEAR(yz[0], x[0], EPS);
    EXPECT_NEAR(yz[1], x[1], EPS);
    EXPECT_NEAR(yz[2], x[2], EPS);

    Vec3 zx = Cross(z, x);
    EXPECT_NEAR(zx[0], y[0], EPS);
    EXPECT_NEAR(zx[1], y[1], EPS);
    EXPECT_NEAR(zx[2], y[2], EPS);
}

TEST_F(VecTest, Norms) {
    Vec3 v{3.0, 4.0, 0.0};

    EXPECT_DOUBLE_EQ(v.NormSquared(), 25.0);
    EXPECT_DOUBLE_EQ(v.Norm(), 5.0);
    EXPECT_DOUBLE_EQ(v.NormL1(), 7.0);
    EXPECT_DOUBLE_EQ(v.NormInf(), 4.0);
}

TEST_F(VecTest, Normalization) {
    Vec3 v{3.0, 4.0, 0.0};

    Vec3 normalized = v.Normalized();
    EXPECT_NEAR(normalized[0], 0.6, EPS);
    EXPECT_NEAR(normalized[1], 0.8, EPS);
    EXPECT_NEAR(normalized[2], 0.0, EPS);
    EXPECT_NEAR(normalized.Norm(), 1.0, EPS);

    Vec3 vCopy = v;
    vCopy.Normalize();
    EXPECT_NEAR(vCopy[0], 0.6, EPS);
    EXPECT_NEAR(vCopy[1], 0.8, EPS);
}

TEST_F(VecTest, ZeroVectorNormalization) {
    Vec3 zero = Vec3::Zero();
    Vec3 normalized = zero.Normalized();
    EXPECT_DOUBLE_EQ(normalized[0], 0.0);
    EXPECT_DOUBLE_EQ(normalized[1], 0.0);
    EXPECT_DOUBLE_EQ(normalized[2], 0.0);
}

TEST_F(VecTest, PointConversion) {
    Point2d p2d(3.5, 4.5);
    Vec2 v2 = ToVec(p2d);
    EXPECT_DOUBLE_EQ(v2[0], 3.5);
    EXPECT_DOUBLE_EQ(v2[1], 4.5);

    Point2d p2dBack = ToPoint2d(v2);
    EXPECT_DOUBLE_EQ(p2dBack.x, 3.5);
    EXPECT_DOUBLE_EQ(p2dBack.y, 4.5);

    Point3d p3d(1.0, 2.0, 3.0);
    Vec3 v3 = ToVec(p3d);
    EXPECT_DOUBLE_EQ(v3[0], 1.0);
    EXPECT_DOUBLE_EQ(v3[1], 2.0);
    EXPECT_DOUBLE_EQ(v3[2], 3.0);

    Point3d p3dBack = ToPoint3d(v3);
    EXPECT_DOUBLE_EQ(p3dBack.x, 1.0);
    EXPECT_DOUBLE_EQ(p3dBack.y, 2.0);
    EXPECT_DOUBLE_EQ(p3dBack.z, 3.0);
}

// =============================================================================
// Mat<M,N> Tests
// =============================================================================

class MatTest : public ::testing::Test {
protected:
    static constexpr double EPS = 1e-12;
};

TEST_F(MatTest, DefaultConstruction) {
    Mat22 m22;
    EXPECT_DOUBLE_EQ(m22(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(m22(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(m22(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(m22(1, 1), 0.0);

    Mat33 m33;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_DOUBLE_EQ(m33(i, j), 0.0);
}

TEST_F(MatTest, InitializerListConstruction) {
    Mat22 m{1.0, 2.0, 3.0, 4.0};
    EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 4.0);
}

TEST_F(MatTest, IdentityMatrix) {
    Mat33 id = Mat33::Identity();
    EXPECT_DOUBLE_EQ(id(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(id(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(id(2, 2), 1.0);
    EXPECT_DOUBLE_EQ(id(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(id(1, 2), 0.0);
}

TEST_F(MatTest, DiagonalMatrix) {
    Vec3 diag{2.0, 3.0, 4.0};
    Mat33 m = Mat33::Diagonal(diag);
    EXPECT_DOUBLE_EQ(m(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 3.0);
    EXPECT_DOUBLE_EQ(m(2, 2), 4.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(m(1, 0), 0.0);
}

TEST_F(MatTest, RowColAccess) {
    Mat33 m{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };

    Vec3 row1 = m.Row(1);
    EXPECT_DOUBLE_EQ(row1[0], 4.0);
    EXPECT_DOUBLE_EQ(row1[1], 5.0);
    EXPECT_DOUBLE_EQ(row1[2], 6.0);

    Vec3 col0 = m.Col(0);
    EXPECT_DOUBLE_EQ(col0[0], 1.0);
    EXPECT_DOUBLE_EQ(col0[1], 4.0);
    EXPECT_DOUBLE_EQ(col0[2], 7.0);

    m.SetRow(0, Vec3{10.0, 11.0, 12.0});
    EXPECT_DOUBLE_EQ(m(0, 0), 10.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 11.0);
    EXPECT_DOUBLE_EQ(m(0, 2), 12.0);

    m.SetCol(2, Vec3{20.0, 21.0, 22.0});
    EXPECT_DOUBLE_EQ(m(0, 2), 20.0);
    EXPECT_DOUBLE_EQ(m(1, 2), 21.0);
    EXPECT_DOUBLE_EQ(m(2, 2), 22.0);
}

TEST_F(MatTest, MatrixArithmetic) {
    Mat22 a{1.0, 2.0, 3.0, 4.0};
    Mat22 b{5.0, 6.0, 7.0, 8.0};

    Mat22 sum = a + b;
    EXPECT_DOUBLE_EQ(sum(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(sum(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(sum(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(sum(1, 1), 12.0);

    Mat22 diff = b - a;
    EXPECT_DOUBLE_EQ(diff(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(diff(0, 1), 4.0);

    Mat22 scaled = a * 2.0;
    EXPECT_DOUBLE_EQ(scaled(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(scaled(1, 1), 8.0);

    Mat22 scaled2 = 3.0 * a;
    EXPECT_DOUBLE_EQ(scaled2(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(scaled2(1, 1), 12.0);
}

TEST_F(MatTest, MatrixMultiplication) {
    Mat22 a{1.0, 2.0, 3.0, 4.0};
    Mat22 b{5.0, 6.0, 7.0, 8.0};

    Mat22 c = a * b;
    // [1 2] * [5 6] = [1*5+2*7  1*6+2*8]   = [19 22]
    // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]     [43 50]
    EXPECT_DOUBLE_EQ(c(0, 0), 19.0);
    EXPECT_DOUBLE_EQ(c(0, 1), 22.0);
    EXPECT_DOUBLE_EQ(c(1, 0), 43.0);
    EXPECT_DOUBLE_EQ(c(1, 1), 50.0);
}

TEST_F(MatTest, MatrixVectorMultiplication) {
    Mat22 m{1.0, 2.0, 3.0, 4.0};
    Vec2 v{5.0, 6.0};

    Vec2 result = m * v;
    EXPECT_DOUBLE_EQ(result[0], 1.0*5.0 + 2.0*6.0);  // 17
    EXPECT_DOUBLE_EQ(result[1], 3.0*5.0 + 4.0*6.0);  // 39
}

TEST_F(MatTest, Transpose) {
    Mat23 m{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };

    Mat<3, 2> mt = m.Transpose();
    EXPECT_DOUBLE_EQ(mt(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(mt(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(mt(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(mt(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(mt(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(mt(2, 1), 6.0);
}

TEST_F(MatTest, Trace) {
    Mat33 m{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };
    EXPECT_DOUBLE_EQ(m.Trace(), 15.0);  // 1 + 5 + 9
}

TEST_F(MatTest, Determinant2x2) {
    Mat22 m{1.0, 2.0, 3.0, 4.0};
    double det = m.Determinant();
    EXPECT_DOUBLE_EQ(det, 1.0*4.0 - 2.0*3.0);  // -2
}

TEST_F(MatTest, Determinant3x3) {
    Mat33 m{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0  // Changed from 9 to make it non-singular
    };
    double det = m.Determinant();
    // Expected: 1*(5*10-6*8) - 2*(4*10-6*7) + 3*(4*8-5*7)
    //         = 1*(50-48) - 2*(40-42) + 3*(32-35)
    //         = 2 + 4 - 9 = -3
    EXPECT_NEAR(det, -3.0, EPS);
}

TEST_F(MatTest, Determinant4x4) {
    Mat44 m{
        1.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 3.0, 0.0,
        0.0, 0.0, 0.0, 4.0
    };
    double det = m.Determinant();
    EXPECT_NEAR(det, 24.0, EPS);  // 1 * 2 * 3 * 4
}

TEST_F(MatTest, Inverse2x2) {
    Mat22 m{4.0, 7.0, 2.0, 6.0};
    Mat22 inv = m.Inverse();
    Mat22 id = m * inv;

    EXPECT_NEAR(id(0, 0), 1.0, EPS);
    EXPECT_NEAR(id(0, 1), 0.0, EPS);
    EXPECT_NEAR(id(1, 0), 0.0, EPS);
    EXPECT_NEAR(id(1, 1), 1.0, EPS);
}

TEST_F(MatTest, Inverse3x3) {
    Mat33 m{
        1.0, 2.0, 3.0,
        0.0, 1.0, 4.0,
        5.0, 6.0, 0.0
    };
    Mat33 inv = m.Inverse();
    Mat33 id = m * inv;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(id(i, j), (i == j) ? 1.0 : 0.0, EPS);
        }
    }
}

TEST_F(MatTest, Inverse4x4) {
    Mat44 m{
        1.0, 0.0, 0.0, 1.0,
        0.0, 2.0, 0.0, 2.0,
        0.0, 0.0, 3.0, 3.0,
        0.0, 0.0, 0.0, 1.0
    };
    Mat44 inv = m.Inverse();
    Mat44 id = m * inv;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(id(i, j), (i == j) ? 1.0 : 0.0, EPS);
        }
    }
}

TEST_F(MatTest, SingularMatrixInverse) {
    Mat22 singular{1.0, 2.0, 2.0, 4.0};  // Rows are linearly dependent
    Mat22 inv = singular.Inverse();
    // Should return zero matrix
    EXPECT_DOUBLE_EQ(inv(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(inv(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(inv(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(inv(1, 1), 0.0);
}

TEST_F(MatTest, IsInvertible) {
    Mat22 invertible{4.0, 7.0, 2.0, 6.0};
    Mat22 singular{1.0, 2.0, 2.0, 4.0};

    EXPECT_TRUE(invertible.IsInvertible());
    EXPECT_FALSE(singular.IsInvertible());
}

TEST_F(MatTest, Norms) {
    Mat22 m{1.0, 2.0, 3.0, 4.0};

    double frob = m.NormFrobenius();
    EXPECT_NEAR(frob, std::sqrt(1.0 + 4.0 + 9.0 + 16.0), EPS);

    double l1 = m.NormL1();  // Max column sum
    EXPECT_DOUBLE_EQ(l1, 6.0);  // max(1+3, 2+4) = max(4, 6)

    double inf = m.NormInf();  // Max row sum
    EXPECT_DOUBLE_EQ(inf, 7.0);  // max(1+2, 3+4) = max(3, 7)
}

// =============================================================================
// VecX Tests
// =============================================================================

class VecXTest : public ::testing::Test {
protected:
    static constexpr double EPS = 1e-12;
};

TEST_F(VecXTest, DefaultConstruction) {
    VecX v;
    EXPECT_EQ(v.Size(), 0);
}

TEST_F(VecXTest, SizeConstruction) {
    VecX v(5);
    EXPECT_EQ(v.Size(), 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(v[i], 0.0);
    }
}

TEST_F(VecXTest, ValueConstruction) {
    VecX v(4, 3.14);
    EXPECT_EQ(v.Size(), 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(v[i], 3.14);
    }
}

TEST_F(VecXTest, InitializerListConstruction) {
    VecX v{1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_EQ(v.Size(), 5);
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[4], 5.0);
}

TEST_F(VecXTest, CopyConstruction) {
    VecX original{1.0, 2.0, 3.0};
    VecX copy(original);

    EXPECT_EQ(copy.Size(), 3);
    EXPECT_DOUBLE_EQ(copy[0], 1.0);
    EXPECT_DOUBLE_EQ(copy[1], 2.0);
    EXPECT_DOUBLE_EQ(copy[2], 3.0);

    // Modify original, copy should be unchanged
    original[0] = 100.0;
    EXPECT_DOUBLE_EQ(copy[0], 1.0);
}

TEST_F(VecXTest, MoveConstruction) {
    VecX original{1.0, 2.0, 3.0};
    VecX moved(std::move(original));

    EXPECT_EQ(moved.Size(), 3);
    EXPECT_DOUBLE_EQ(moved[0], 1.0);
    EXPECT_EQ(original.Size(), 0);
}

TEST_F(VecXTest, LargeVectorHeapAllocation) {
    // Size > VECX_STACK_SIZE should use heap
    VecX large(100);
    EXPECT_EQ(large.Size(), 100);
    for (int i = 0; i < 100; ++i) {
        large[i] = static_cast<double>(i);
    }
    for (int i = 0; i < 100; ++i) {
        EXPECT_DOUBLE_EQ(large[i], static_cast<double>(i));
    }
}

TEST_F(VecXTest, Arithmetic) {
    VecX a{1.0, 2.0, 3.0};
    VecX b{4.0, 5.0, 6.0};

    VecX sum = a + b;
    EXPECT_DOUBLE_EQ(sum[0], 5.0);
    EXPECT_DOUBLE_EQ(sum[1], 7.0);
    EXPECT_DOUBLE_EQ(sum[2], 9.0);

    VecX diff = b - a;
    EXPECT_DOUBLE_EQ(diff[0], 3.0);

    VecX scaled = a * 2.0;
    EXPECT_DOUBLE_EQ(scaled[0], 2.0);

    VecX scaled2 = 2.0 * a;
    EXPECT_DOUBLE_EQ(scaled2[0], 2.0);
}

TEST_F(VecXTest, DotProduct) {
    VecX a{1.0, 2.0, 3.0};
    VecX b{4.0, 5.0, 6.0};

    double dot = a.Dot(b);
    EXPECT_DOUBLE_EQ(dot, 32.0);
}

TEST_F(VecXTest, Norm) {
    VecX v{3.0, 4.0};
    EXPECT_DOUBLE_EQ(v.Norm(), 5.0);
    EXPECT_DOUBLE_EQ(v.NormSquared(), 25.0);
}

TEST_F(VecXTest, FactoryMethods) {
    VecX zero = VecX::Zero(5);
    EXPECT_EQ(zero.Size(), 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(zero[i], 0.0);
    }

    VecX ones = VecX::Ones(5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(ones[i], 1.0);
    }

    VecX lin = VecX::LinSpace(0.0, 4.0, 5);
    EXPECT_DOUBLE_EQ(lin[0], 0.0);
    EXPECT_DOUBLE_EQ(lin[1], 1.0);
    EXPECT_DOUBLE_EQ(lin[2], 2.0);
    EXPECT_DOUBLE_EQ(lin[3], 3.0);
    EXPECT_DOUBLE_EQ(lin[4], 4.0);
}

TEST_F(VecXTest, SegmentOperations) {
    VecX v{1.0, 2.0, 3.0, 4.0, 5.0};

    VecX seg = v.Segment(1, 3);
    EXPECT_EQ(seg.Size(), 3);
    EXPECT_DOUBLE_EQ(seg[0], 2.0);
    EXPECT_DOUBLE_EQ(seg[1], 3.0);
    EXPECT_DOUBLE_EQ(seg[2], 4.0);

    VecX replacement{10.0, 20.0};
    v.SetSegment(2, replacement);
    EXPECT_DOUBLE_EQ(v[2], 10.0);
    EXPECT_DOUBLE_EQ(v[3], 20.0);
}

TEST_F(VecXTest, FixedSizeConversion) {
    Vec3 fixed{1.0, 2.0, 3.0};
    VecX dynamic(fixed);
    EXPECT_EQ(dynamic.Size(), 3);
    EXPECT_DOUBLE_EQ(dynamic[0], 1.0);
    EXPECT_DOUBLE_EQ(dynamic[1], 2.0);
    EXPECT_DOUBLE_EQ(dynamic[2], 3.0);

    Vec3 backToFixed = dynamic.ToFixed<3>();
    EXPECT_DOUBLE_EQ(backToFixed[0], 1.0);
    EXPECT_DOUBLE_EQ(backToFixed[1], 2.0);
    EXPECT_DOUBLE_EQ(backToFixed[2], 3.0);
}

// =============================================================================
// MatX Tests
// =============================================================================

class MatXTest : public ::testing::Test {
protected:
    static constexpr double EPS = 1e-12;
};

TEST_F(MatXTest, DefaultConstruction) {
    MatX m;
    EXPECT_EQ(m.Rows(), 0);
    EXPECT_EQ(m.Cols(), 0);
}

TEST_F(MatXTest, SizeConstruction) {
    MatX m(3, 4);
    EXPECT_EQ(m.Rows(), 3);
    EXPECT_EQ(m.Cols(), 4);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_DOUBLE_EQ(m(i, j), 0.0);
        }
    }
}

TEST_F(MatXTest, ValueConstruction) {
    MatX m(2, 3, 5.0);
    EXPECT_EQ(m.Rows(), 2);
    EXPECT_EQ(m.Cols(), 3);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(m(i, j), 5.0);
        }
    }
}

TEST_F(MatXTest, CopyAndMove) {
    MatX original(2, 3, 1.0);
    MatX copy(original);
    EXPECT_EQ(copy.Rows(), 2);
    EXPECT_EQ(copy.Cols(), 3);
    EXPECT_DOUBLE_EQ(copy(0, 0), 1.0);

    MatX moved(std::move(original));
    EXPECT_EQ(moved.Rows(), 2);
    EXPECT_EQ(moved.Cols(), 3);
    EXPECT_EQ(original.Rows(), 0);
}

TEST_F(MatXTest, Arithmetic) {
    MatX a(2, 2, 1.0);
    MatX b(2, 2, 2.0);

    MatX sum = a + b;
    EXPECT_DOUBLE_EQ(sum(0, 0), 3.0);

    MatX diff = b - a;
    EXPECT_DOUBLE_EQ(diff(0, 0), 1.0);

    MatX scaled = a * 3.0;
    EXPECT_DOUBLE_EQ(scaled(0, 0), 3.0);
}

TEST_F(MatXTest, MatrixMultiplication) {
    MatX a(2, 3);
    a(0, 0) = 1.0; a(0, 1) = 2.0; a(0, 2) = 3.0;
    a(1, 0) = 4.0; a(1, 1) = 5.0; a(1, 2) = 6.0;

    MatX b(3, 2);
    b(0, 0) = 7.0;  b(0, 1) = 8.0;
    b(1, 0) = 9.0;  b(1, 1) = 10.0;
    b(2, 0) = 11.0; b(2, 1) = 12.0;

    MatX c = a * b;
    EXPECT_EQ(c.Rows(), 2);
    EXPECT_EQ(c.Cols(), 2);

    // [1 2 3] * [7  8 ]   = [1*7+2*9+3*11   1*8+2*10+3*12]   = [58  64 ]
    // [4 5 6]   [9  10]     [4*7+5*9+6*11   4*8+5*10+6*12]     [139 154]
    //           [11 12]
    EXPECT_NEAR(c(0, 0), 58.0, EPS);
    EXPECT_NEAR(c(0, 1), 64.0, EPS);
    EXPECT_NEAR(c(1, 0), 139.0, EPS);
    EXPECT_NEAR(c(1, 1), 154.0, EPS);
}

TEST_F(MatXTest, MatrixVectorMultiplication) {
    MatX m(2, 3);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 6.0;

    VecX v{7.0, 8.0, 9.0};
    VecX result = m * v;

    EXPECT_EQ(result.Size(), 2);
    EXPECT_NEAR(result[0], 1*7 + 2*8 + 3*9, EPS);  // 50
    EXPECT_NEAR(result[1], 4*7 + 5*8 + 6*9, EPS);  // 122
}

TEST_F(MatXTest, Transpose) {
    MatX m(2, 3);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 6.0;

    MatX mt = m.Transpose();
    EXPECT_EQ(mt.Rows(), 3);
    EXPECT_EQ(mt.Cols(), 2);
    EXPECT_DOUBLE_EQ(mt(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(mt(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(mt(2, 1), 6.0);
}

TEST_F(MatXTest, FactoryMethods) {
    MatX zero = MatX::Zero(3, 4);
    EXPECT_EQ(zero.Rows(), 3);
    EXPECT_EQ(zero.Cols(), 4);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_DOUBLE_EQ(zero(i, j), 0.0);
        }
    }

    MatX id = MatX::Identity(3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(id(i, j), (i == j) ? 1.0 : 0.0);
        }
    }

    VecX diag{2.0, 3.0, 4.0};
    MatX diagMat = MatX::Diagonal(diag);
    EXPECT_DOUBLE_EQ(diagMat(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(diagMat(1, 1), 3.0);
    EXPECT_DOUBLE_EQ(diagMat(2, 2), 4.0);
}

TEST_F(MatXTest, BlockOperations) {
    MatX m(4, 4);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            m(i, j) = i * 4 + j;
        }
    }

    MatX block = m.Block(1, 1, 2, 2);
    EXPECT_EQ(block.Rows(), 2);
    EXPECT_EQ(block.Cols(), 2);
    EXPECT_DOUBLE_EQ(block(0, 0), 5.0);  // m(1,1)
    EXPECT_DOUBLE_EQ(block(0, 1), 6.0);  // m(1,2)
    EXPECT_DOUBLE_EQ(block(1, 0), 9.0);  // m(2,1)
    EXPECT_DOUBLE_EQ(block(1, 1), 10.0); // m(2,2)

    MatX newBlock(2, 2, 100.0);
    m.SetBlock(0, 0, newBlock);
    EXPECT_DOUBLE_EQ(m(0, 0), 100.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 100.0);
    EXPECT_DOUBLE_EQ(m(1, 0), 100.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 100.0);
}

TEST_F(MatXTest, DeterminantAndInverse) {
    MatX m(3, 3);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 0.0; m(1, 1) = 1.0; m(1, 2) = 4.0;
    m(2, 0) = 5.0; m(2, 1) = 6.0; m(2, 2) = 0.0;

    double det = m.Determinant();
    EXPECT_NE(det, 0.0);

    MatX inv = m.Inverse();
    MatX id = m * inv;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(id(i, j), (i == j) ? 1.0 : 0.0, EPS);
        }
    }
}

TEST_F(MatXTest, FixedSizeConversion) {
    Mat33 fixed{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };

    MatX dynamic(fixed);
    EXPECT_EQ(dynamic.Rows(), 3);
    EXPECT_EQ(dynamic.Cols(), 3);
    EXPECT_DOUBLE_EQ(dynamic(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(dynamic(2, 2), 9.0);

    Mat33 backToFixed = dynamic.ToFixed<3, 3>();
    EXPECT_DOUBLE_EQ(backToFixed(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(backToFixed(2, 2), 9.0);
}

// =============================================================================
// 2D Transformation Tests
// =============================================================================

class Transform2DTest : public ::testing::Test {
protected:
    static constexpr double EPS = 1e-10;
};

TEST_F(Transform2DTest, Rotation2D) {
    double angle = PI / 4.0;  // 45 degrees
    Mat33 R = Rotation2D(angle);

    // Rotate point (1, 0)
    Vec3 p{1.0, 0.0, 1.0};
    Vec3 rotated = R * p;

    double c = std::cos(angle);
    double s = std::sin(angle);
    EXPECT_NEAR(rotated[0], c, EPS);
    EXPECT_NEAR(rotated[1], s, EPS);
    EXPECT_DOUBLE_EQ(rotated[2], 1.0);
}

TEST_F(Transform2DTest, Translation2D) {
    Mat33 T = Translation2D(10.0, 20.0);

    Vec3 p{5.0, 3.0, 1.0};
    Vec3 translated = T * p;

    EXPECT_NEAR(translated[0], 15.0, EPS);
    EXPECT_NEAR(translated[1], 23.0, EPS);
    EXPECT_DOUBLE_EQ(translated[2], 1.0);
}

TEST_F(Transform2DTest, Scaling2D) {
    Mat33 S = Scaling2D(2.0, 3.0);

    Vec3 p{5.0, 4.0, 1.0};
    Vec3 scaled = S * p;

    EXPECT_NEAR(scaled[0], 10.0, EPS);
    EXPECT_NEAR(scaled[1], 12.0, EPS);
}

TEST_F(Transform2DTest, UniformScaling2D) {
    Mat33 S = Scaling2D(2.0);

    Vec3 p{5.0, 4.0, 1.0};
    Vec3 scaled = S * p;

    EXPECT_NEAR(scaled[0], 10.0, EPS);
    EXPECT_NEAR(scaled[1], 8.0, EPS);
}

TEST_F(Transform2DTest, Affine2D) {
    // Scale by 2, rotate by 45 degrees, translate by (10, 20)
    double angle = PI / 4.0;
    Mat33 A = Affine2D(10.0, 20.0, angle, 2.0, 2.0);

    Vec3 p{1.0, 0.0, 1.0};
    Vec3 result = A * p;

    // Expected: scale (1,0) by 2 -> (2, 0)
    //           rotate by 45 -> (sqrt(2), sqrt(2))
    //           translate -> (sqrt(2)+10, sqrt(2)+20)
    double expected_x = 2.0 * std::cos(angle) + 10.0;
    double expected_y = 2.0 * std::sin(angle) + 20.0;

    EXPECT_NEAR(result[0], expected_x, EPS);
    EXPECT_NEAR(result[1], expected_y, EPS);
}

// =============================================================================
// 3D Transformation Tests
// =============================================================================

class Transform3DTest : public ::testing::Test {
protected:
    static constexpr double EPS = 1e-10;
};

TEST_F(Transform3DTest, RotationX) {
    double angle = PI / 2.0;  // 90 degrees
    Mat44 R = RotationX(angle);

    Vec4 p{0.0, 1.0, 0.0, 1.0};
    Vec4 rotated = R * p;

    EXPECT_NEAR(rotated[0], 0.0, EPS);
    EXPECT_NEAR(rotated[1], 0.0, EPS);
    EXPECT_NEAR(rotated[2], 1.0, EPS);
}

TEST_F(Transform3DTest, RotationY) {
    double angle = PI / 2.0;
    Mat44 R = RotationY(angle);

    Vec4 p{1.0, 0.0, 0.0, 1.0};
    Vec4 rotated = R * p;

    EXPECT_NEAR(rotated[0], 0.0, EPS);
    EXPECT_NEAR(rotated[1], 0.0, EPS);
    EXPECT_NEAR(rotated[2], -1.0, EPS);
}

TEST_F(Transform3DTest, RotationZ) {
    double angle = PI / 2.0;
    Mat44 R = RotationZ(angle);

    Vec4 p{1.0, 0.0, 0.0, 1.0};
    Vec4 rotated = R * p;

    EXPECT_NEAR(rotated[0], 0.0, EPS);
    EXPECT_NEAR(rotated[1], 1.0, EPS);
    EXPECT_NEAR(rotated[2], 0.0, EPS);
}

TEST_F(Transform3DTest, RotationAxisAngle) {
    // Rotate around Z axis by 90 degrees
    Vec3 axis{0.0, 0.0, 1.0};
    Mat44 R = RotationAxisAngle(axis, PI / 2.0);

    Vec4 p{1.0, 0.0, 0.0, 1.0};
    Vec4 rotated = R * p;

    EXPECT_NEAR(rotated[0], 0.0, EPS);
    EXPECT_NEAR(rotated[1], 1.0, EPS);
    EXPECT_NEAR(rotated[2], 0.0, EPS);
}

TEST_F(Transform3DTest, Translation3D) {
    Mat44 T = Translation3D(1.0, 2.0, 3.0);

    Vec4 p{5.0, 6.0, 7.0, 1.0};
    Vec4 translated = T * p;

    EXPECT_NEAR(translated[0], 6.0, EPS);
    EXPECT_NEAR(translated[1], 8.0, EPS);
    EXPECT_NEAR(translated[2], 10.0, EPS);
}

TEST_F(Transform3DTest, Scaling3D) {
    Mat44 S = Scaling3D(2.0, 3.0, 4.0);

    Vec4 p{1.0, 1.0, 1.0, 1.0};
    Vec4 scaled = S * p;

    EXPECT_NEAR(scaled[0], 2.0, EPS);
    EXPECT_NEAR(scaled[1], 3.0, EPS);
    EXPECT_NEAR(scaled[2], 4.0, EPS);
}

TEST_F(Transform3DTest, EulerAnglesZYX) {
    // Test with simple angles
    Mat44 R = RotationEulerZYX(0.0, 0.0, PI / 2.0);  // 90 degree yaw

    Vec4 p{1.0, 0.0, 0.0, 1.0};
    Vec4 rotated = R * p;

    EXPECT_NEAR(rotated[0], 0.0, EPS);
    EXPECT_NEAR(rotated[1], 1.0, EPS);
    EXPECT_NEAR(rotated[2], 0.0, EPS);
}

TEST_F(Transform3DTest, Rotation3x3EulerZYX) {
    Mat33 R = Rotation3x3EulerZYX(0.1, 0.2, 0.3);

    // Check orthogonality: R * R^T = I
    Mat33 RRt = R * R.Transpose();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(RRt(i, j), (i == j) ? 1.0 : 0.0, EPS);
        }
    }

    // Check determinant = 1 (proper rotation)
    EXPECT_NEAR(R.Determinant(), 1.0, EPS);
}

TEST_F(Transform3DTest, ExtractEulerZYX) {
    double roll = 0.1, pitch = 0.2, yaw = 0.3;
    Mat33 R = Rotation3x3EulerZYX(roll, pitch, yaw);
    Vec3 euler = ExtractEulerZYX(R);

    EXPECT_NEAR(euler[0], roll, EPS);
    EXPECT_NEAR(euler[1], pitch, EPS);
    EXPECT_NEAR(euler[2], yaw, EPS);
}

TEST_F(Transform3DTest, ExtractAxisAngle) {
    Vec3 axis{1.0, 1.0, 1.0};
    axis.Normalize();
    double angle = 0.5;

    Mat33 R = Rotation3x3AxisAngle(axis, angle);
    auto [extractedAxis, extractedAngle] = ExtractAxisAngle(R);

    EXPECT_NEAR(extractedAngle, angle, EPS);
    EXPECT_NEAR(extractedAxis[0], axis[0], EPS);
    EXPECT_NEAR(extractedAxis[1], axis[1], EPS);
    EXPECT_NEAR(extractedAxis[2], axis[2], EPS);
}

// =============================================================================
// Camera Matrix Tests
// =============================================================================

class CameraMatrixTest : public ::testing::Test {
protected:
    static constexpr double EPS = 1e-10;
};

TEST_F(CameraMatrixTest, CameraIntrinsic) {
    Mat33 K = CameraIntrinsic(1000.0, 1000.0, 320.0, 240.0);

    EXPECT_DOUBLE_EQ(K(0, 0), 1000.0);  // fx
    EXPECT_DOUBLE_EQ(K(1, 1), 1000.0);  // fy
    EXPECT_DOUBLE_EQ(K(0, 2), 320.0);   // cx
    EXPECT_DOUBLE_EQ(K(1, 2), 240.0);   // cy
    EXPECT_DOUBLE_EQ(K(2, 2), 1.0);
    EXPECT_DOUBLE_EQ(K(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(K(1, 0), 0.0);
}

TEST_F(CameraMatrixTest, ProjectionMatrix) {
    Mat33 K = CameraIntrinsic(1000.0, 1000.0, 320.0, 240.0);
    Mat33 R = Mat33::Identity();
    Vec3 t{0.0, 0.0, 100.0};

    Mat34 P = ProjectionMatrix(K, R, t);

    // Project a 3D point
    Vec4 X{0.0, 0.0, 100.0, 1.0};  // Point at (0,0,100)
    Vec3 x = P * X;

    // Normalize by z
    double u = x[0] / x[2];
    double v = x[1] / x[2];

    // Should project to principal point (320, 240)
    EXPECT_NEAR(u, 320.0, EPS);
    EXPECT_NEAR(v, 240.0, EPS);
}

// =============================================================================
// QMatrix Conversion Tests
// =============================================================================

class QMatrixConversionTest : public ::testing::Test {
protected:
    static constexpr double EPS = 1e-12;
};

TEST_F(QMatrixConversionTest, FromQMatrix) {
    QMatrix qm = QMatrix::Rotation(0.5);
    Mat33 m = FromQMatrix(qm);

    EXPECT_NEAR(m(0, 0), qm.M00(), EPS);
    EXPECT_NEAR(m(0, 1), qm.M01(), EPS);
    EXPECT_NEAR(m(0, 2), qm.M02(), EPS);
    EXPECT_NEAR(m(1, 0), qm.M10(), EPS);
    EXPECT_NEAR(m(1, 1), qm.M11(), EPS);
    EXPECT_NEAR(m(1, 2), qm.M12(), EPS);
    EXPECT_DOUBLE_EQ(m(2, 0), 0.0);
    EXPECT_DOUBLE_EQ(m(2, 1), 0.0);
    EXPECT_DOUBLE_EQ(m(2, 2), 1.0);
}

TEST_F(QMatrixConversionTest, ToQMatrix) {
    Mat33 m = Rotation2D(0.5);
    QMatrix qm = ToQMatrix(m);

    EXPECT_NEAR(qm.M00(), m(0, 0), EPS);
    EXPECT_NEAR(qm.M01(), m(0, 1), EPS);
    EXPECT_NEAR(qm.M02(), m(0, 2), EPS);
    EXPECT_NEAR(qm.M10(), m(1, 0), EPS);
    EXPECT_NEAR(qm.M11(), m(1, 1), EPS);
    EXPECT_NEAR(qm.M12(), m(1, 2), EPS);
}

TEST_F(QMatrixConversionTest, RoundTrip) {
    QMatrix original = QMatrix::Rotation(0.3) * QMatrix::Translation(10.0, 20.0);
    Mat33 m = FromQMatrix(original);
    QMatrix back = ToQMatrix(m);

    double elements1[6], elements2[6];
    original.GetElements(elements1);
    back.GetElements(elements2);

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(elements1[i], elements2[i], EPS);
    }
}

// =============================================================================
// Decomposition Result Tests
// =============================================================================

class DecompositionTest : public ::testing::Test {
protected:
    static constexpr double EPS = 1e-10;
};

TEST_F(DecompositionTest, LUResultDeterminant) {
    LUResult lu;
    lu.valid = true;
    lu.sign = 1;
    lu.U = MatX(3, 3);
    lu.U(0, 0) = 2.0;
    lu.U(1, 1) = 3.0;
    lu.U(2, 2) = 4.0;

    double det = lu.Determinant();
    EXPECT_NEAR(det, 24.0, EPS);  // 2 * 3 * 4
}

TEST_F(DecompositionTest, SVDResultCondition) {
    SVDResult svd;
    svd.valid = true;
    svd.rank = 3;
    svd.S = VecX{10.0, 5.0, 1.0};

    double cond = svd.Condition();
    EXPECT_NEAR(cond, 10.0, EPS);  // 10 / 1
}

// =============================================================================
// Precision Tests
// =============================================================================

class PrecisionTest : public ::testing::Test {
protected:
    static constexpr double EPS = 1e-10;
};

TEST_F(PrecisionTest, MatrixInversePrecision) {
    // Test with random-ish well-conditioned matrix
    Mat33 A{
        1.2,  0.3,  0.1,
        0.2,  1.5,  0.2,
        0.1,  0.2,  1.3
    };

    Mat33 Ainv = A.Inverse();
    Mat33 I = A * Ainv;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(I(i, j), (i == j) ? 1.0 : 0.0, EPS);
        }
    }
}

TEST_F(PrecisionTest, RotationMatrixOrthogonality) {
    for (double angle = 0.0; angle < TWO_PI; angle += 0.1) {
        Mat33 R = Rotation3x3EulerZYX(angle, angle * 0.5, angle * 0.3);

        // R * R^T should be identity
        Mat33 RRt = R * R.Transpose();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                EXPECT_NEAR(RRt(i, j), (i == j) ? 1.0 : 0.0, EPS);
            }
        }

        // det(R) should be 1
        EXPECT_NEAR(R.Determinant(), 1.0, EPS);
    }
}

TEST_F(PrecisionTest, EulerAngleRoundTrip) {
    // Test many combinations of Euler angles
    for (double roll = -PI; roll < PI; roll += 0.5) {
        for (double pitch = -HALF_PI + 0.1; pitch < HALF_PI - 0.1; pitch += 0.5) {
            for (double yaw = -PI; yaw < PI; yaw += 0.5) {
                Mat33 R = Rotation3x3EulerZYX(roll, pitch, yaw);
                Vec3 euler = ExtractEulerZYX(R);

                EXPECT_NEAR(euler[0], roll, EPS);
                EXPECT_NEAR(euler[1], pitch, EPS);
                EXPECT_NEAR(euler[2], yaw, EPS);
            }
        }
    }
}

TEST_F(PrecisionTest, LargeDynamicMatrixOperations) {
    int n = 50;
    MatX A = MatX::Identity(n);
    MatX B = MatX::Identity(n);

    // A * B should be identity
    MatX C = A * B;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(C(i, j), (i == j) ? 1.0 : 0.0, EPS);
        }
    }
}
