/**
 * @file test_solver.cpp
 * @brief Unit tests for Internal/Solver module
 */

#include <QiVision/Internal/Solver.h>
#include <QiVision/Internal/Matrix.h>
#include <QiVision/Core/Exception.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>

namespace Qi::Vision::Internal {
namespace {

// =============================================================================
// Test Utilities
// =============================================================================

/// Create a random matrix for testing
MatX RandomMatrix(int rows, int cols, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    MatX A(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            A(i, j) = dist(rng);
        }
    }
    return A;
}

/// Create a random symmetric positive definite matrix
MatX RandomSPD(int n, std::mt19937& rng) {
    MatX A = RandomMatrix(n, n, rng);
    // A^T * A is symmetric positive semi-definite
    // Add n*I to make it strictly positive definite
    MatX ATA = A.Transpose() * A;
    for (int i = 0; i < n; ++i) {
        ATA(i, i) += n;
    }
    return ATA;
}

/// Create a random vector
VecX RandomVector(int n, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    VecX v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

/// Apply permutation to vector
VecX ApplyPermutation(const VecX& v, const std::vector<int>& P) {
    VecX pv(v.Size());
    for (int i = 0; i < v.Size(); ++i) {
        pv[i] = v[P[i]];
    }
    return pv;
}

/// Apply permutation to matrix rows
MatX ApplyRowPermutation(const MatX& A, const std::vector<int>& P) {
    MatX PA(A.Rows(), A.Cols());
    for (int i = 0; i < A.Rows(); ++i) {
        for (int j = 0; j < A.Cols(); ++j) {
            PA(i, j) = A(P[i], j);
        }
    }
    return PA;
}

// =============================================================================
// Small Matrix Solver Tests (2x2, 3x3, 4x4)
// =============================================================================

class SmallMatrixSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(42);
    }
    std::mt19937 rng_;
};

TEST_F(SmallMatrixSolverTest, Solve2x2_Simple) {
    // Simple 2x2 system
    // 2x + y = 5
    // x + 3y = 10
    // Solution: x = 1, y = 3
    Mat22 A{2, 1, 1, 3};
    Vec2 b{5, 10};

    Vec2 x = Solve2x2(A, b);

    EXPECT_NEAR(x[0], 1.0, 1e-12);
    EXPECT_NEAR(x[1], 3.0, 1e-12);
}

TEST_F(SmallMatrixSolverTest, Solve2x2_Singular) {
    // Singular matrix (parallel lines)
    Mat22 A{1, 2, 2, 4};
    Vec2 b{3, 6};

    Vec2 x = Solve2x2(A, b);

    EXPECT_NEAR(x[0], 0.0, 1e-12);
    EXPECT_NEAR(x[1], 0.0, 1e-12);
}

TEST_F(SmallMatrixSolverTest, Solve2x2_Random) {
    for (int trial = 0; trial < 10; ++trial) {
        std::uniform_real_distribution<double> dist(-10.0, 10.0);
        Mat22 A{dist(rng_), dist(rng_), dist(rng_), dist(rng_)};
        Vec2 xTrue{dist(rng_), dist(rng_)};
        Vec2 b = A * xTrue;

        if (IsSolvable2x2(A)) {
            Vec2 x = Solve2x2(A, b);
            EXPECT_NEAR(x[0], xTrue[0], 1e-10);
            EXPECT_NEAR(x[1], xTrue[1], 1e-10);
        }
    }
}

TEST_F(SmallMatrixSolverTest, Solve3x3_Simple) {
    // x + 2y + 3z = 14
    // 4x + 5y + 6z = 32
    // 7x + 8y + 10z = 53
    // Solution: x = 1, y = 2, z = 3
    Mat33 A{1, 2, 3, 4, 5, 6, 7, 8, 10};
    Vec3 b{14, 32, 53};

    Vec3 x = Solve3x3(A, b);

    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 2.0, 1e-10);
    EXPECT_NEAR(x[2], 3.0, 1e-10);
}

TEST_F(SmallMatrixSolverTest, Solve3x3_Singular) {
    // Singular matrix (linearly dependent rows)
    Mat33 A{1, 2, 3, 2, 4, 6, 1, 1, 1};
    Vec3 b{1, 2, 1};

    Vec3 x = Solve3x3(A, b);

    // Should return zero for singular matrix
    EXPECT_NEAR(x[0], 0.0, 1e-10);
    EXPECT_NEAR(x[1], 0.0, 1e-10);
    EXPECT_NEAR(x[2], 0.0, 1e-10);
}

TEST_F(SmallMatrixSolverTest, Solve3x3_Random) {
    for (int trial = 0; trial < 10; ++trial) {
        std::uniform_real_distribution<double> dist(-10.0, 10.0);
        Mat33 A;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                A(i, j) = dist(rng_);

        Vec3 xTrue{dist(rng_), dist(rng_), dist(rng_)};
        Vec3 b = A * xTrue;

        if (IsSolvable3x3(A)) {
            Vec3 x = Solve3x3(A, b);
            EXPECT_NEAR(x[0], xTrue[0], 1e-8);
            EXPECT_NEAR(x[1], xTrue[1], 1e-8);
            EXPECT_NEAR(x[2], xTrue[2], 1e-8);
        }
    }
}

TEST_F(SmallMatrixSolverTest, Solve4x4_Simple) {
    // Identity system
    Mat44 A = Mat44::Identity();
    Vec4 b{1, 2, 3, 4};

    Vec4 x = Solve4x4(A, b);

    EXPECT_NEAR(x[0], 1.0, 1e-12);
    EXPECT_NEAR(x[1], 2.0, 1e-12);
    EXPECT_NEAR(x[2], 3.0, 1e-12);
    EXPECT_NEAR(x[3], 4.0, 1e-12);
}

TEST_F(SmallMatrixSolverTest, Solve4x4_Random) {
    for (int trial = 0; trial < 10; ++trial) {
        std::uniform_real_distribution<double> dist(-10.0, 10.0);
        Mat44 A;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                A(i, j) = dist(rng_);

        Vec4 xTrue{dist(rng_), dist(rng_), dist(rng_), dist(rng_)};
        Vec4 b = A * xTrue;

        if (IsSolvable4x4(A)) {
            Vec4 x = Solve4x4(A, b);
            EXPECT_NEAR(x[0], xTrue[0], 1e-8);
            EXPECT_NEAR(x[1], xTrue[1], 1e-8);
            EXPECT_NEAR(x[2], xTrue[2], 1e-8);
            EXPECT_NEAR(x[3], xTrue[3], 1e-8);
        }
    }
}

TEST_F(SmallMatrixSolverTest, IsSolvable_Checks) {
    Mat22 A2_good{1, 0, 0, 1};
    Mat22 A2_bad{1, 1, 1, 1};
    EXPECT_TRUE(IsSolvable2x2(A2_good));
    EXPECT_FALSE(IsSolvable2x2(A2_bad));

    Mat33 A3_good = Mat33::Identity();
    Mat33 A3_bad{1, 2, 3, 2, 4, 6, 3, 6, 9};  // Rank 1
    EXPECT_TRUE(IsSolvable3x3(A3_good));
    EXPECT_FALSE(IsSolvable3x3(A3_bad));

    Mat44 A4_good = Mat44::Identity();
    EXPECT_TRUE(IsSolvable4x4(A4_good));
}

// =============================================================================
// Triangular Solver Tests
// =============================================================================

class TriangularSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(123);
    }
    std::mt19937 rng_;
};

TEST_F(TriangularSolverTest, LowerTriangular_Simple) {
    // L = [1 0 0; 2 3 0; 4 5 6]
    MatX L(3, 3);
    L(0, 0) = 1; L(0, 1) = 0; L(0, 2) = 0;
    L(1, 0) = 2; L(1, 1) = 3; L(1, 2) = 0;
    L(2, 0) = 4; L(2, 1) = 5; L(2, 2) = 6;

    VecX b{1, 8, 32};
    VecX x = SolveLowerTriangular(L, b);

    // Verify L*x = b
    VecX Lx = L * x;
    EXPECT_NEAR(Lx[0], b[0], 1e-12);
    EXPECT_NEAR(Lx[1], b[1], 1e-12);
    EXPECT_NEAR(Lx[2], b[2], 1e-12);
}

TEST_F(TriangularSolverTest, LowerTriangular_UnitDiagonal) {
    MatX L(3, 3);
    L(0, 0) = 1; L(0, 1) = 0; L(0, 2) = 0;
    L(1, 0) = 2; L(1, 1) = 1; L(1, 2) = 0;
    L(2, 0) = 3; L(2, 1) = 4; L(2, 2) = 1;

    VecX b{1, 5, 14};
    VecX x = SolveLowerTriangular(L, b, true);

    VecX Lx = L * x;
    EXPECT_NEAR(Lx[0], b[0], 1e-12);
    EXPECT_NEAR(Lx[1], b[1], 1e-12);
    EXPECT_NEAR(Lx[2], b[2], 1e-12);
}

TEST_F(TriangularSolverTest, UpperTriangular_Simple) {
    // U = [1 2 3; 0 4 5; 0 0 6]
    MatX U(3, 3);
    U(0, 0) = 1; U(0, 1) = 2; U(0, 2) = 3;
    U(1, 0) = 0; U(1, 1) = 4; U(1, 2) = 5;
    U(2, 0) = 0; U(2, 1) = 0; U(2, 2) = 6;

    VecX b{14, 23, 18};
    VecX x = SolveUpperTriangular(U, b);

    VecX Ux = U * x;
    EXPECT_NEAR(Ux[0], b[0], 1e-12);
    EXPECT_NEAR(Ux[1], b[1], 1e-12);
    EXPECT_NEAR(Ux[2], b[2], 1e-12);
}

TEST_F(TriangularSolverTest, UpperTriangular_Random) {
    int n = 5;
    MatX U(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            std::uniform_real_distribution<double> dist(0.1, 10.0);
            U(i, j) = dist(rng_);
        }
    }

    VecX xTrue = RandomVector(n, rng_);
    VecX b = U * xTrue;
    VecX x = SolveUpperTriangular(U, b);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], xTrue[i], 1e-10);
    }
}

// =============================================================================
// Tridiagonal Solver Tests
// =============================================================================

TEST(TridiagonalSolverTest, Simple) {
    // [2  1  0  0] [x0]   [1]
    // [1  2  1  0] [x1] = [2]
    // [0  1  2  1] [x2]   [3]
    // [0  0  1  2] [x3]   [4]

    VecX lower{1, 1, 1};
    VecX diag{2, 2, 2, 2};
    VecX upper{1, 1, 1};
    VecX b{1, 2, 3, 4};

    VecX x = SolveTridiagonal(lower, diag, upper, b);

    // Verify solution by computing Ax
    VecX Ax(4);
    Ax[0] = diag[0] * x[0] + upper[0] * x[1];
    Ax[1] = lower[0] * x[0] + diag[1] * x[1] + upper[1] * x[2];
    Ax[2] = lower[1] * x[1] + diag[2] * x[2] + upper[2] * x[3];
    Ax[3] = lower[2] * x[2] + diag[3] * x[3];

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(Ax[i], b[i], 1e-12);
    }
}

TEST(TridiagonalSolverTest, DiagonallyDominant) {
    int n = 10;
    VecX lower(n - 1);
    VecX diag(n);
    VecX upper(n - 1);
    VecX b(n);

    std::mt19937 rng(789);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < n - 1; ++i) {
        lower[i] = dist(rng);
        upper[i] = dist(rng);
    }
    for (int i = 0; i < n; ++i) {
        // Diagonally dominant
        diag[i] = 10.0 + dist(rng);
        b[i] = dist(rng);
    }

    VecX x = SolveTridiagonal(lower, diag, upper, b);

    // Verify
    VecX Ax(n);
    Ax[0] = diag[0] * x[0] + upper[0] * x[1];
    for (int i = 1; i < n - 1; ++i) {
        Ax[i] = lower[i - 1] * x[i - 1] + diag[i] * x[i] + upper[i] * x[i + 1];
    }
    Ax[n - 1] = lower[n - 2] * x[n - 2] + diag[n - 1] * x[n - 1];

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(Ax[i], b[i], 1e-10);
    }
}

// =============================================================================
// LU Decomposition Tests
// =============================================================================

class LUDecomposeTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(456);
    }
    std::mt19937 rng_;
};

TEST_F(LUDecomposeTest, Identity) {
    MatX A = MatX::Identity(3);
    LUResult lu = LU_Decompose(A);

    EXPECT_TRUE(lu.valid);

    // L should be identity
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(lu.L(i, j), expected, 1e-12);
        }
    }

    // U should be identity
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(lu.U(i, j), expected, 1e-12);
        }
    }
}

TEST_F(LUDecomposeTest, Simple) {
    // A = [2 1 1; 4 3 3; 8 7 9]
    MatX A(3, 3);
    A(0, 0) = 2; A(0, 1) = 1; A(0, 2) = 1;
    A(1, 0) = 4; A(1, 1) = 3; A(1, 2) = 3;
    A(2, 0) = 8; A(2, 1) = 7; A(2, 2) = 9;

    LUResult lu = LU_Decompose(A);

    EXPECT_TRUE(lu.valid);

    // Verify PA = LU
    MatX PA = ApplyRowPermutation(A, lu.P);
    MatX LU = lu.L * lu.U;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(PA(i, j), LU(i, j), 1e-12);
        }
    }
}

TEST_F(LUDecomposeTest, Random) {
    for (int n = 2; n <= 10; ++n) {
        MatX A = RandomMatrix(n, n, rng_);
        LUResult lu = LU_Decompose(A);

        if (lu.valid) {
            // Verify PA = LU
            MatX PA = ApplyRowPermutation(A, lu.P);
            MatX LU = lu.L * lu.U;

            double error = (PA - LU).NormFrobenius() / A.NormFrobenius();
            EXPECT_LT(error, 1e-12) << "Failed for n=" << n;
        }
    }
}

TEST_F(LUDecomposeTest, Singular) {
    // Singular matrix
    MatX A(3, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 2; A(1, 1) = 4; A(1, 2) = 6;  // Row 1 = 2 * Row 0
    A(2, 0) = 1; A(2, 1) = 1; A(2, 2) = 1;

    LUResult lu = LU_Decompose(A);

    EXPECT_FALSE(lu.valid);
}

TEST_F(LUDecomposeTest, Determinant) {
    MatX A(3, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;
    A(2, 0) = 7; A(2, 1) = 8; A(2, 2) = 10;

    LUResult lu = LU_Decompose(A);
    EXPECT_TRUE(lu.valid);

    double det = lu.Determinant();

    // Manually computed determinant = -3
    EXPECT_NEAR(det, -3.0, 1e-10);
}

TEST_F(LUDecomposeTest, SolveFromLU_Single) {
    int n = 5;
    MatX A = RandomMatrix(n, n, rng_);
    // Make well-conditioned
    for (int i = 0; i < n; ++i) {
        A(i, i) += 5.0;
    }

    VecX xTrue = RandomVector(n, rng_);
    VecX b = A * xTrue;

    LUResult lu = LU_Decompose(A);
    EXPECT_TRUE(lu.valid);

    VecX x = SolveFromLU(lu, b);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], xTrue[i], 1e-10);
    }
}

TEST_F(LUDecomposeTest, SolveFromLU_Multiple) {
    int n = 5;
    int k = 3;

    MatX A = RandomMatrix(n, n, rng_);
    for (int i = 0; i < n; ++i) {
        A(i, i) += 5.0;
    }

    MatX XTrue = RandomMatrix(n, k, rng_);
    MatX B = A * XTrue;

    LUResult lu = LU_Decompose(A);
    EXPECT_TRUE(lu.valid);

    MatX X = SolveFromLU(lu, B);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            EXPECT_NEAR(X(i, j), XTrue(i, j), 1e-10);
        }
    }
}

TEST_F(LUDecomposeTest, SolveLU_Direct) {
    int n = 5;
    MatX A = RandomMatrix(n, n, rng_);
    for (int i = 0; i < n; ++i) {
        A(i, i) += 5.0;
    }

    VecX xTrue = RandomVector(n, rng_);
    VecX b = A * xTrue;

    VecX x = SolveLU(A, b);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], xTrue[i], 1e-10);
    }
}

// =============================================================================
// Cholesky Decomposition Tests
// =============================================================================

class CholeskyTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(789);
    }
    std::mt19937 rng_;
};

TEST_F(CholeskyTest, Identity) {
    MatX A = MatX::Identity(3);
    CholeskyResult chol = Cholesky_Decompose(A);

    EXPECT_TRUE(chol.valid);

    // L should be identity
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(chol.L(i, j), expected, 1e-12);
        }
    }
}

TEST_F(CholeskyTest, Simple) {
    // A = [4 2; 2 5] is SPD
    MatX A(2, 2);
    A(0, 0) = 4; A(0, 1) = 2;
    A(1, 0) = 2; A(1, 1) = 5;

    CholeskyResult chol = Cholesky_Decompose(A);

    EXPECT_TRUE(chol.valid);

    // Verify A = L * L^T
    MatX LLT = chol.L * chol.L.Transpose();

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(A(i, j), LLT(i, j), 1e-12);
        }
    }
}

TEST_F(CholeskyTest, RandomSPD) {
    for (int n = 2; n <= 8; ++n) {
        MatX A = RandomSPD(n, rng_);
        CholeskyResult chol = Cholesky_Decompose(A);

        EXPECT_TRUE(chol.valid) << "Failed for n=" << n;

        // Verify A = L * L^T
        MatX LLT = chol.L * chol.L.Transpose();
        double error = (A - LLT).NormFrobenius() / A.NormFrobenius();
        EXPECT_LT(error, 1e-12) << "Reconstruction error for n=" << n;
    }
}

TEST_F(CholeskyTest, NotPositiveDefinite) {
    // A = [1 2; 2 1] has eigenvalues 3 and -1, not SPD
    MatX A(2, 2);
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 2; A(1, 1) = 1;

    CholeskyResult chol = Cholesky_Decompose(A);

    EXPECT_FALSE(chol.valid);
}

TEST_F(CholeskyTest, SolveCholesky) {
    int n = 5;
    MatX A = RandomSPD(n, rng_);

    VecX xTrue = RandomVector(n, rng_);
    VecX b = A * xTrue;

    VecX x = SolveCholesky(A, b);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], xTrue[i], 1e-10);
    }
}

TEST_F(CholeskyTest, SolveFromCholesky) {
    int n = 5;
    MatX A = RandomSPD(n, rng_);
    CholeskyResult chol = Cholesky_Decompose(A);

    VecX xTrue = RandomVector(n, rng_);
    VecX b = A * xTrue;

    VecX x = SolveFromCholesky(chol, b);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], xTrue[i], 1e-10);
    }
}

// =============================================================================
// QR Decomposition Tests
// =============================================================================

class QRDecomposeTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(321);
    }
    std::mt19937 rng_;
};

TEST_F(QRDecomposeTest, Identity) {
    MatX A = MatX::Identity(3);
    QRResult qr = QR_Decompose(A);

    EXPECT_TRUE(qr.valid);

    // Q should be identity (or sign permutation)
    MatX QTQ = qr.Q.Transpose() * qr.Q;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(QTQ(i, j), expected, 1e-12);
        }
    }
}

TEST_F(QRDecomposeTest, Square) {
    MatX A(3, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;
    A(2, 0) = 7; A(2, 1) = 8; A(2, 2) = 10;

    QRResult qr = QR_Decompose(A);

    EXPECT_TRUE(qr.valid);

    // Verify A = Q * R
    MatX QR = qr.Q * qr.R;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(A(i, j), QR(i, j), 1e-12);
        }
    }

    // Verify Q^T * Q = I
    MatX QTQ = qr.Q.Transpose() * qr.Q;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(QTQ(i, j), expected, 1e-12);
        }
    }
}

TEST_F(QRDecomposeTest, Overdetermined) {
    // m > n
    int m = 6, n = 3;
    MatX A = RandomMatrix(m, n, rng_);

    QRResult qr = QR_Decompose(A);

    EXPECT_TRUE(qr.valid);

    // Verify A = Q * R
    MatX QR = qr.Q * qr.R;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(A(i, j), QR(i, j), 1e-12);
        }
    }

    // Verify Q is orthogonal
    MatX QTQ = qr.Q.Transpose() * qr.Q;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(QTQ(i, j), expected, 1e-12);
        }
    }
}

TEST_F(QRDecomposeTest, ThinQR) {
    int m = 8, n = 3;
    MatX A = RandomMatrix(m, n, rng_);

    QRResult qr = QR_DecomposeThin(A);

    EXPECT_TRUE(qr.valid);
    EXPECT_TRUE(qr.thinQR);
    EXPECT_EQ(qr.Q.Rows(), m);
    EXPECT_EQ(qr.Q.Cols(), n);
    EXPECT_EQ(qr.R.Rows(), n);
    EXPECT_EQ(qr.R.Cols(), n);

    // Verify A = Q * R
    MatX QR = qr.Q * qr.R;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(A(i, j), QR(i, j), 1e-12);
        }
    }

    // Verify Q^T * Q = I (n x n)
    MatX QTQ = qr.Q.Transpose() * qr.Q;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(QTQ(i, j), expected, 1e-12);
        }
    }
}

TEST_F(QRDecomposeTest, SolveLeastSquares) {
    // Overdetermined system: fit y = ax + b to noisy data
    int m = 10;
    int n = 2;

    MatX A(m, n);
    VecX b(m);

    // True parameters: a = 2, b = 3
    double aTrue = 2.0, bTrue = 3.0;

    std::uniform_real_distribution<double> noiseDist(-0.01, 0.01);
    for (int i = 0; i < m; ++i) {
        double x = static_cast<double>(i);
        A(i, 0) = x;
        A(i, 1) = 1.0;
        b[i] = aTrue * x + bTrue + noiseDist(rng_);
    }

    VecX x = SolveLeastSquares(A, b);

    EXPECT_NEAR(x[0], aTrue, 0.1);
    EXPECT_NEAR(x[1], bTrue, 0.1);
}

TEST_F(QRDecomposeTest, SolveFromQR) {
    int m = 8, n = 4;
    MatX A = RandomMatrix(m, n, rng_);

    VecX xTrue = RandomVector(n, rng_);
    VecX b = A * xTrue;  // Exact, no noise

    QRResult qr = QR_DecomposeThin(A);
    VecX x = SolveFromQR(qr, b);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], xTrue[i], 1e-10);
    }
}

// =============================================================================
// SVD Decomposition Tests
// =============================================================================

class SVDDecomposeTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(654);
    }
    std::mt19937 rng_;
};

TEST_F(SVDDecomposeTest, Identity) {
    MatX A = MatX::Identity(3);
    SVDResult svd = SVD_Decompose(A);

    EXPECT_TRUE(svd.valid);
    EXPECT_EQ(svd.rank, 3);

    // All singular values should be 1
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(svd.S[i], 1.0, 1e-10);
    }
}

TEST_F(SVDDecomposeTest, DiagonalMatrix) {
    MatX A(3, 3);
    A(0, 0) = 5; A(0, 1) = 0; A(0, 2) = 0;
    A(1, 0) = 0; A(1, 1) = 3; A(1, 2) = 0;
    A(2, 0) = 0; A(2, 1) = 0; A(2, 2) = 1;

    SVDResult svd = SVD_Decompose(A);

    EXPECT_TRUE(svd.valid);
    EXPECT_EQ(svd.rank, 3);

    // Singular values should be 5, 3, 1 (sorted descending)
    EXPECT_NEAR(svd.S[0], 5.0, 1e-10);
    EXPECT_NEAR(svd.S[1], 3.0, 1e-10);
    EXPECT_NEAR(svd.S[2], 1.0, 1e-10);
}

TEST_F(SVDDecomposeTest, SquareMatrix) {
    MatX A(3, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;
    A(2, 0) = 7; A(2, 1) = 8; A(2, 2) = 10;

    SVDResult svd = SVD_Decompose(A);

    EXPECT_TRUE(svd.valid);

    // Verify A = U * S * V^T
    MatX S = MatX::Diagonal(svd.S);
    MatX USVt = svd.U * S * svd.V.Transpose();

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(A(i, j), USVt(i, j), 1e-10);
        }
    }

    // Verify U^T * U = I
    MatX UTU = svd.U.Transpose() * svd.U;
    for (int i = 0; i < svd.U.Cols(); ++i) {
        for (int j = 0; j < svd.U.Cols(); ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(UTU(i, j), expected, 1e-10);
        }
    }

    // Verify V^T * V = I
    MatX VTV = svd.V.Transpose() * svd.V;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(VTV(i, j), expected, 1e-10);
        }
    }
}

TEST_F(SVDDecomposeTest, Overdetermined) {
    int m = 6, n = 3;
    MatX A = RandomMatrix(m, n, rng_);

    SVDResult svd = SVD_Decompose(A);

    EXPECT_TRUE(svd.valid);
    EXPECT_EQ(svd.S.Size(), n);

    // Verify reconstruction
    MatX S = MatX::Zero(svd.U.Cols(), svd.V.Cols());
    for (int i = 0; i < svd.S.Size(); ++i) {
        S(i, i) = svd.S[i];
    }
    MatX USVt = svd.U * S * svd.V.Transpose();

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(A(i, j), USVt(i, j), 1e-10);
        }
    }
}

TEST_F(SVDDecomposeTest, Underdetermined) {
    int m = 3, n = 6;
    MatX A = RandomMatrix(m, n, rng_);

    SVDResult svd = SVD_Decompose(A);

    EXPECT_TRUE(svd.valid);
    EXPECT_EQ(svd.S.Size(), m);

    // Verify reconstruction
    MatX S = MatX::Zero(svd.U.Cols(), svd.V.Cols());
    for (int i = 0; i < svd.S.Size(); ++i) {
        S(i, i) = svd.S[i];
    }
    MatX USVt = svd.U * S * svd.V.Transpose();

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(A(i, j), USVt(i, j), 1e-10);
        }
    }
}

TEST_F(SVDDecomposeTest, RankDeficient) {
    // Rank 2 matrix in 3x3
    MatX A(3, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 2; A(1, 1) = 4; A(1, 2) = 6;  // 2 * row 0
    A(2, 0) = 1; A(2, 1) = 0; A(2, 2) = 1;

    SVDResult svd = SVD_Decompose(A);

    EXPECT_TRUE(svd.valid);
    EXPECT_EQ(svd.rank, 2);

    // Third singular value should be near zero
    EXPECT_LT(svd.S[2], 1e-10);
}

TEST_F(SVDDecomposeTest, SolveSVD) {
    int n = 5;
    MatX A = RandomMatrix(n, n, rng_);
    for (int i = 0; i < n; ++i) {
        A(i, i) += 5.0;
    }

    VecX xTrue = RandomVector(n, rng_);
    VecX b = A * xTrue;

    VecX x = SolveSVD(A, b);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], xTrue[i], 1e-8);
    }
}

TEST_F(SVDDecomposeTest, SolveHomogeneous) {
    // 2x + y - z = 0
    // x + 2y + z = 0
    // 3x + 3y + 0 = 0
    MatX A(3, 3);
    A(0, 0) = 2; A(0, 1) = 1; A(0, 2) = -1;
    A(1, 0) = 1; A(1, 1) = 2; A(1, 2) = 1;
    A(2, 0) = 3; A(2, 1) = 3; A(2, 2) = 0;

    VecX x = SolveHomogeneous(A);

    EXPECT_EQ(x.Size(), 3);

    // x should be unit vector
    EXPECT_NEAR(x.Norm(), 1.0, 1e-10);

    // Ax should be close to zero
    VecX Ax = A * x;
    EXPECT_LT(Ax.Norm(), 1e-8);
}

// =============================================================================
// Least Squares via Normal Equations Tests
// =============================================================================

class LeastSquaresNormalTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(987);
    }
    std::mt19937 rng_;
};

TEST_F(LeastSquaresNormalTest, Basic) {
    int m = 10, n = 3;
    MatX A = RandomMatrix(m, n, rng_);

    VecX xTrue = RandomVector(n, rng_);
    VecX b = A * xTrue;  // Exact solution

    VecX x = SolveLeastSquaresNormal(A, b);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], xTrue[i], 1e-8);
    }
}

TEST_F(LeastSquaresNormalTest, CompareWithQR) {
    int m = 20, n = 5;
    MatX A = RandomMatrix(m, n, rng_);
    VecX b = RandomVector(m, rng_);

    VecX xQR = SolveLeastSquares(A, b);
    VecX xNormal = SolveLeastSquaresNormal(A, b);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(xQR[i], xNormal[i], 1e-6);
    }
}

// =============================================================================
// Utility Functions Tests
// =============================================================================

class UtilityFunctionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(111);
    }
    std::mt19937 rng_;
};

TEST_F(UtilityFunctionsTest, ComputeConditionNumber_Identity) {
    MatX A = MatX::Identity(5);
    double cond = ComputeConditionNumber(A);
    EXPECT_NEAR(cond, 1.0, 1e-10);
}

TEST_F(UtilityFunctionsTest, ComputeConditionNumber_WellConditioned) {
    MatX A(3, 3);
    A(0, 0) = 10; A(0, 1) = 1; A(0, 2) = 1;
    A(1, 0) = 1; A(1, 1) = 10; A(1, 2) = 1;
    A(2, 0) = 1; A(2, 1) = 1; A(2, 2) = 10;

    double cond = ComputeConditionNumber(A);
    EXPECT_LT(cond, 3.0);  // Well-conditioned, diagonally dominant
}

TEST_F(UtilityFunctionsTest, ComputeConditionNumber_IllConditioned) {
    MatX A(2, 2);
    A(0, 0) = 1; A(0, 1) = 1;
    A(1, 0) = 1; A(1, 1) = 1.0001;

    double cond = ComputeConditionNumber(A);
    EXPECT_GT(cond, 1000.0);  // Ill-conditioned
}

TEST_F(UtilityFunctionsTest, EstimateConditionNumber) {
    MatX A = RandomMatrix(5, 5, rng_);
    for (int i = 0; i < 5; ++i) {
        A(i, i) += 5.0;
    }

    double condExact = ComputeConditionNumber(A);
    double condEst = EstimateConditionNumber(A);

    // Estimate should be within an order of magnitude
    EXPECT_GT(condEst, condExact * 0.1);
    EXPECT_LT(condEst, condExact * 10.0);
}

TEST_F(UtilityFunctionsTest, ComputeRank_FullRank) {
    MatX A = MatX::Identity(5);
    int rank = ComputeRank(A);
    EXPECT_EQ(rank, 5);
}

TEST_F(UtilityFunctionsTest, ComputeRank_RankDeficient) {
    MatX A(3, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 2; A(1, 1) = 4; A(1, 2) = 6;  // 2 * row 0
    A(2, 0) = 3; A(2, 1) = 6; A(2, 2) = 9;  // 3 * row 0

    int rank = ComputeRank(A);
    EXPECT_EQ(rank, 1);
}

TEST_F(UtilityFunctionsTest, PseudoInverse_Square) {
    MatX A = MatX::Identity(3);
    MatX Ainv = PseudoInverse(A);

    // For identity, pseudoinverse is also identity
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(Ainv(i, j), expected, 1e-12);
        }
    }
}

TEST_F(UtilityFunctionsTest, PseudoInverse_Properties) {
    int m = 5, n = 3;
    MatX A = RandomMatrix(m, n, rng_);
    MatX Ainv = PseudoInverse(A);

    // A * A^+ * A = A
    MatX AAinvA = A * Ainv * A;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(AAinvA(i, j), A(i, j), 1e-10);
        }
    }

    // A^+ * A * A^+ = A^+
    MatX AinvAAinv = Ainv * A * Ainv;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            EXPECT_NEAR(AinvAAinv(i, j), Ainv(i, j), 1e-10);
        }
    }
}

TEST_F(UtilityFunctionsTest, ComputeResidual) {
    MatX A = MatX::Identity(3);
    VecX x{1, 2, 3};
    VecX b{1.1, 2.1, 3.1};

    double res = ComputeResidual(A, x, b);

    // Expected: sqrt(0.1^2 + 0.1^2 + 0.1^2) = sqrt(0.03)
    EXPECT_NEAR(res, std::sqrt(0.03), 1e-12);
}

TEST_F(UtilityFunctionsTest, ComputeRelativeResidual) {
    MatX A = MatX::Identity(3);
    VecX x{1, 2, 3};
    VecX b{1, 2, 3};  // Exact solution

    double relRes = ComputeRelativeResidual(A, x, b);
    EXPECT_NEAR(relRes, 0.0, 1e-12);
}

TEST_F(UtilityFunctionsTest, ComputeNullSpace_FullRank) {
    MatX A = MatX::Identity(3);
    MatX nullSpace = ComputeNullSpace(A);

    // Full rank matrix has empty null space
    EXPECT_EQ(nullSpace.Rows(), 0);
}

TEST_F(UtilityFunctionsTest, ComputeNullSpace_RankDeficient) {
    MatX A(2, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 2; A(1, 1) = 4; A(1, 2) = 6;  // 2 * row 0

    MatX nullSpace = ComputeNullSpace(A);

    // Nullity should be 3 - 1 = 2
    EXPECT_EQ(nullSpace.Cols(), 2);
    EXPECT_EQ(nullSpace.Rows(), 3);

    // Each null space vector should satisfy Ax = 0
    for (int i = 0; i < nullSpace.Cols(); ++i) {
        VecX v = nullSpace.Col(i);
        VecX Av = A * v;
        EXPECT_LT(Av.Norm(), 1e-10);
    }
}

TEST_F(UtilityFunctionsTest, ComputeNullity) {
    MatX A(2, 4);
    A(0, 0) = 1; A(0, 1) = 0; A(0, 2) = 2; A(0, 3) = 1;
    A(1, 0) = 0; A(1, 1) = 1; A(1, 2) = 1; A(1, 3) = 2;

    int nullity = ComputeNullity(A);
    // Rank is 2, so nullity is 4 - 2 = 2
    EXPECT_EQ(nullity, 2);
}

// =============================================================================
// Precision Tests
// =============================================================================

class SolverPrecisionTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(222);
    }
    std::mt19937 rng_;
};

TEST_F(SolverPrecisionTest, LU_Precision) {
    for (int n = 5; n <= 20; n += 5) {
        MatX A = RandomMatrix(n, n, rng_);
        for (int i = 0; i < n; ++i) {
            A(i, i) += 5.0;  // Make well-conditioned
        }

        LUResult lu = LU_Decompose(A);
        ASSERT_TRUE(lu.valid);

        MatX PA = ApplyRowPermutation(A, lu.P);
        MatX LU = lu.L * lu.U;

        double error = (PA - LU).NormFrobenius() / A.NormFrobenius();
        EXPECT_LT(error, 1e-12) << "LU precision failed for n=" << n;
    }
}

TEST_F(SolverPrecisionTest, Cholesky_Precision) {
    for (int n = 5; n <= 20; n += 5) {
        MatX A = RandomSPD(n, rng_);

        CholeskyResult chol = Cholesky_Decompose(A);
        ASSERT_TRUE(chol.valid);

        MatX LLT = chol.L * chol.L.Transpose();

        double error = (A - LLT).NormFrobenius() / A.NormFrobenius();
        EXPECT_LT(error, 1e-12) << "Cholesky precision failed for n=" << n;
    }
}

TEST_F(SolverPrecisionTest, QR_Precision) {
    for (int n = 5; n <= 20; n += 5) {
        int m = n + 5;
        MatX A = RandomMatrix(m, n, rng_);

        QRResult qr = QR_Decompose(A);
        ASSERT_TRUE(qr.valid);

        MatX QR = qr.Q * qr.R;

        double error = (A - QR).NormFrobenius() / A.NormFrobenius();
        EXPECT_LT(error, 1e-12) << "QR precision failed for m=" << m << ", n=" << n;

        // Orthogonality
        MatX QTQ = qr.Q.Transpose() * qr.Q;
        MatX I = MatX::Identity(m);
        double orthError = (QTQ - I).NormFrobenius();
        EXPECT_LT(orthError, 1e-12) << "QR orthogonality failed for m=" << m;
    }
}

TEST_F(SolverPrecisionTest, SVD_Precision) {
    for (int n = 3; n <= 10; n += 2) {
        MatX A = RandomMatrix(n, n, rng_);

        SVDResult svd = SVD_Decompose(A);
        ASSERT_TRUE(svd.valid);

        // Reconstruct
        MatX S = MatX::Diagonal(svd.S);
        MatX USVt = svd.U * S * svd.V.Transpose();

        double error = (A - USVt).NormFrobenius() / A.NormFrobenius();
        EXPECT_LT(error, 1e-10) << "SVD precision failed for n=" << n;
    }
}

TEST_F(SolverPrecisionTest, SolveAccuracy) {
    for (int n = 5; n <= 15; n += 5) {
        MatX A = RandomMatrix(n, n, rng_);
        for (int i = 0; i < n; ++i) {
            A(i, i) += 5.0;
        }

        VecX xTrue = RandomVector(n, rng_);
        VecX b = A * xTrue;

        // Test all solvers
        VecX xLU = SolveLU(A, b);
        VecX xSVD = SolveSVD(A, b);

        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(xLU[i], xTrue[i], 1e-10) << "LU solve failed for n=" << n;
            EXPECT_NEAR(xSVD[i], xTrue[i], 1e-8) << "SVD solve failed for n=" << n;
        }
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

class EdgeCasesTest : public ::testing::Test {};

TEST_F(EdgeCasesTest, EmptyMatrix) {
    MatX A(0, 0);
    VecX b;

    LUResult lu = LU_Decompose(A);
    EXPECT_TRUE(lu.valid);

    CholeskyResult chol = Cholesky_Decompose(A);
    EXPECT_TRUE(chol.valid);

    QRResult qr = QR_Decompose(A);
    EXPECT_TRUE(qr.valid);

    SVDResult svd = SVD_Decompose(A);
    EXPECT_TRUE(svd.valid);
}

TEST_F(EdgeCasesTest, Size1Matrix) {
    MatX A(1, 1);
    A(0, 0) = 5.0;
    VecX b{10.0};

    VecX xLU = SolveLU(A, b);
    EXPECT_NEAR(xLU[0], 2.0, 1e-12);

    VecX xChol = SolveCholesky(A, b);
    EXPECT_NEAR(xChol[0], 2.0, 1e-12);

    VecX xSVD = SolveSVD(A, b);
    EXPECT_NEAR(xSVD[0], 2.0, 1e-12);
}

TEST_F(EdgeCasesTest, ZeroMatrix) {
    MatX A = MatX::Zero(3, 3);
    VecX b{1, 2, 3};

    LUResult lu = LU_Decompose(A);
    EXPECT_FALSE(lu.valid);

    CholeskyResult chol = Cholesky_Decompose(A);
    EXPECT_FALSE(chol.valid);
}

TEST_F(EdgeCasesTest, NearSingularMatrix) {
    MatX A(2, 2);
    A(0, 0) = 1; A(0, 1) = 1;
    A(1, 0) = 1; A(1, 1) = 1 + 1e-14;

    double cond = ComputeConditionNumber(A);
    EXPECT_GT(cond, 1e10);
}

TEST_F(EdgeCasesTest, DimensionMismatch) {
    MatX A(3, 3);
    VecX b(5);  // Wrong size

    EXPECT_THROW(SolveLowerTriangular(A, b), InvalidArgumentException);
    EXPECT_THROW(SolveUpperTriangular(A, b), InvalidArgumentException);
}

} // anonymous namespace
} // namespace Qi::Vision::Internal
