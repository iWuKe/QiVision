/**
 * @file test_eigen.cpp
 * @brief Unit tests for Eigen.h eigenvalue decomposition module
 */

#include <gtest/gtest.h>
#include <QiVision/Internal/Eigen.h>
#include <cmath>

using namespace Qi::Vision::Internal;

namespace {
    constexpr double TOLERANCE = 1e-9;
    constexpr double LOOSE_TOLERANCE = 1e-6;

    // Helper: Check if A * v = lambda * v
    bool CheckEigenpair(const MatX& A, double lambda, const VecX& v, double tol = TOLERANCE) {
        int n = A.Rows();
        VecX Av(n);
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                sum += A(i, j) * v[j];
            }
            Av[i] = sum;
        }

        for (int i = 0; i < n; ++i) {
            if (std::abs(Av[i] - lambda * v[i]) > tol) {
                return false;
            }
        }
        return true;
    }

    // Helper: Check orthogonality of eigenvectors
    bool CheckOrthogonal(const MatX& V, double tol = TOLERANCE) {
        int n = V.Cols();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double dot = 0.0;
                for (int k = 0; k < V.Rows(); ++k) {
                    dot += V(k, i) * V(k, j);
                }
                if (std::abs(dot) > tol) {
                    return false;
                }
            }
        }
        return true;
    }

    // Helper: Check unit norm
    bool CheckUnitNorm(const VecX& v, double tol = TOLERANCE) {
        double norm = 0.0;
        for (int i = 0; i < v.Size(); ++i) {
            norm += v[i] * v[i];
        }
        return std::abs(std::sqrt(norm) - 1.0) < tol;
    }
}

// =============================================================================
// 2x2 Symmetric Eigenvalue Tests
// =============================================================================

class Eigen2x2SymmetricTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(Eigen2x2SymmetricTest, DiagonalMatrix) {
    Mat22 A;
    A(0, 0) = 5.0; A(0, 1) = 0.0;
    A(1, 0) = 0.0; A(1, 1) = 3.0;

    auto result = EigenSymmetric2x2(A);

    ASSERT_TRUE(result.valid);
    EXPECT_TRUE(result.isReal);
    EXPECT_NEAR(result.lambda1, 5.0, TOLERANCE);
    EXPECT_NEAR(result.lambda2, 3.0, TOLERANCE);
}

TEST_F(Eigen2x2SymmetricTest, SymmetricMatrix) {
    Mat22 A;
    A(0, 0) = 4.0; A(0, 1) = 2.0;
    A(1, 0) = 2.0; A(1, 1) = 4.0;

    auto result = EigenSymmetric2x2(A);

    ASSERT_TRUE(result.valid);
    EXPECT_TRUE(result.isReal);
    // Eigenvalues: 6 and 2
    EXPECT_NEAR(std::abs(result.lambda1), 6.0, TOLERANCE);
    EXPECT_NEAR(std::abs(result.lambda2), 2.0, TOLERANCE);
}

TEST_F(Eigen2x2SymmetricTest, IdentityMatrix) {
    Mat22 A = Mat22::Identity();

    auto result = EigenSymmetric2x2(A);

    ASSERT_TRUE(result.valid);
    EXPECT_NEAR(result.lambda1, 1.0, TOLERANCE);
    EXPECT_NEAR(result.lambda2, 1.0, TOLERANCE);
}

TEST_F(Eigen2x2SymmetricTest, ZeroMatrix) {
    Mat22 A;
    A(0, 0) = 0; A(0, 1) = 0;
    A(1, 0) = 0; A(1, 1) = 0;

    auto result = EigenSymmetric2x2(A);

    ASSERT_TRUE(result.valid);
    EXPECT_NEAR(result.lambda1, 0.0, TOLERANCE);
    EXPECT_NEAR(result.lambda2, 0.0, TOLERANCE);
}

TEST_F(Eigen2x2SymmetricTest, NegativeEigenvalues) {
    Mat22 A;
    A(0, 0) = -2.0; A(0, 1) = 1.0;
    A(1, 0) = 1.0; A(1, 1) = -2.0;

    auto result = EigenSymmetric2x2(A);

    ASSERT_TRUE(result.valid);
    // Eigenvalues: -1 and -3
    double max = std::max(std::abs(result.lambda1), std::abs(result.lambda2));
    double min = std::min(std::abs(result.lambda1), std::abs(result.lambda2));
    EXPECT_NEAR(max, 3.0, TOLERANCE);
    EXPECT_NEAR(min, 1.0, TOLERANCE);
}

TEST_F(Eigen2x2SymmetricTest, EigenvectorsOrthogonal) {
    Mat22 A;
    A(0, 0) = 3.0; A(0, 1) = 1.0;
    A(1, 0) = 1.0; A(1, 1) = 3.0;

    auto result = EigenSymmetric2x2(A);

    ASSERT_TRUE(result.valid);

    // Check orthogonality
    double dot = result.v1[0] * result.v2[0] + result.v1[1] * result.v2[1];
    EXPECT_NEAR(dot, 0.0, TOLERANCE);

    // Check unit norm
    double norm1 = std::sqrt(result.v1[0] * result.v1[0] + result.v1[1] * result.v1[1]);
    double norm2 = std::sqrt(result.v2[0] * result.v2[0] + result.v2[1] * result.v2[1]);
    EXPECT_NEAR(norm1, 1.0, TOLERANCE);
    EXPECT_NEAR(norm2, 1.0, TOLERANCE);
}

// =============================================================================
// 2x2 General Eigenvalue Tests
// =============================================================================

class Eigen2x2GeneralTest : public ::testing::Test {};

TEST_F(Eigen2x2GeneralTest, RealEigenvalues) {
    Mat22 A;
    A(0, 0) = 4.0; A(0, 1) = 1.0;
    A(1, 0) = 2.0; A(1, 1) = 3.0;

    auto result = EigenGeneral2x2(A);

    ASSERT_TRUE(result.valid);
    EXPECT_TRUE(result.isReal);

    // trace = 7, det = 10, eigenvalues = (7 ± sqrt(9))/2 = 5, 2
    double sum = result.lambda1 + result.lambda2;
    double prod = result.lambda1 * result.lambda2;
    EXPECT_NEAR(sum, 7.0, TOLERANCE);
    EXPECT_NEAR(prod, 10.0, TOLERANCE);
}

TEST_F(Eigen2x2GeneralTest, ComplexEigenvalues) {
    Mat22 A;
    A(0, 0) = 0.0; A(0, 1) = -1.0;
    A(1, 0) = 1.0; A(1, 1) = 0.0;

    auto result = EigenGeneral2x2(A);

    ASSERT_TRUE(result.valid);
    EXPECT_FALSE(result.isReal);

    // Eigenvalues: ±i
    EXPECT_NEAR(result.lambda1, 0.0, TOLERANCE);
    EXPECT_NEAR(result.imagPart, 1.0, TOLERANCE);
}

TEST_F(Eigen2x2GeneralTest, RepeatedEigenvalues) {
    Mat22 A;
    A(0, 0) = 2.0; A(0, 1) = 0.0;
    A(1, 0) = 0.0; A(1, 1) = 2.0;

    auto result = EigenGeneral2x2(A);

    ASSERT_TRUE(result.valid);
    EXPECT_TRUE(result.isReal);
    EXPECT_NEAR(result.lambda1, 2.0, TOLERANCE);
    EXPECT_NEAR(result.lambda2, 2.0, TOLERANCE);
}

// =============================================================================
// 3x3 Symmetric Eigenvalue Tests
// =============================================================================

class Eigen3x3SymmetricTest : public ::testing::Test {};

TEST_F(Eigen3x3SymmetricTest, DiagonalMatrix) {
    Mat33 A;
    A(0, 0) = 5.0; A(0, 1) = 0.0; A(0, 2) = 0.0;
    A(1, 0) = 0.0; A(1, 1) = 3.0; A(1, 2) = 0.0;
    A(2, 0) = 0.0; A(2, 1) = 0.0; A(2, 2) = 1.0;

    auto result = EigenSymmetric3x3(A);

    ASSERT_TRUE(result.valid);
    EXPECT_TRUE(result.allReal);

    std::vector<double> eigs = {result.lambda1, result.lambda2, result.lambda3};
    std::sort(eigs.begin(), eigs.end(), std::greater<double>());

    EXPECT_NEAR(eigs[0], 5.0, TOLERANCE);
    EXPECT_NEAR(eigs[1], 3.0, TOLERANCE);
    EXPECT_NEAR(eigs[2], 1.0, TOLERANCE);
}

TEST_F(Eigen3x3SymmetricTest, SymmetricMatrix) {
    Mat33 A;
    A(0, 0) = 2.0; A(0, 1) = 1.0; A(0, 2) = 0.0;
    A(1, 0) = 1.0; A(1, 1) = 2.0; A(1, 2) = 1.0;
    A(2, 0) = 0.0; A(2, 1) = 1.0; A(2, 2) = 2.0;

    auto result = EigenSymmetric3x3(A);

    ASSERT_TRUE(result.valid);
    EXPECT_TRUE(result.allReal);

    // Sum of eigenvalues = trace = 6
    double sum = result.lambda1 + result.lambda2 + result.lambda3;
    EXPECT_NEAR(sum, 6.0, LOOSE_TOLERANCE);
}

TEST_F(Eigen3x3SymmetricTest, IdentityMatrix) {
    Mat33 A = Mat33::Identity();

    auto result = EigenSymmetric3x3(A);

    ASSERT_TRUE(result.valid);
    EXPECT_NEAR(result.lambda1, 1.0, TOLERANCE);
    EXPECT_NEAR(result.lambda2, 1.0, TOLERANCE);
    EXPECT_NEAR(result.lambda3, 1.0, TOLERANCE);
}

TEST_F(Eigen3x3SymmetricTest, CovarianceMatrix) {
    // Typical covariance matrix (positive semi-definite)
    Mat33 A;
    A(0, 0) = 4.0; A(0, 1) = 2.0; A(0, 2) = 1.0;
    A(1, 0) = 2.0; A(1, 1) = 5.0; A(1, 2) = 2.0;
    A(2, 0) = 1.0; A(2, 1) = 2.0; A(2, 2) = 3.0;

    auto result = EigenSymmetric3x3(A);

    ASSERT_TRUE(result.valid);

    // All eigenvalues should be positive for positive definite matrix
    EXPECT_GT(result.lambda1, 0);
    EXPECT_GT(result.lambda2, 0);
    EXPECT_GT(result.lambda3, 0);
}

TEST_F(Eigen3x3SymmetricTest, EigenvectorsOrthogonal) {
    Mat33 A;
    A(0, 0) = 3.0; A(0, 1) = 1.0; A(0, 2) = 0.5;
    A(1, 0) = 1.0; A(1, 1) = 4.0; A(1, 2) = 1.0;
    A(2, 0) = 0.5; A(2, 1) = 1.0; A(2, 2) = 2.0;

    auto result = EigenSymmetric3x3(A);

    ASSERT_TRUE(result.valid);

    // Check orthogonality
    double dot12 = result.v1.Dot(result.v2);
    double dot13 = result.v1.Dot(result.v3);
    double dot23 = result.v2.Dot(result.v3);

    EXPECT_NEAR(dot12, 0.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(dot13, 0.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(dot23, 0.0, LOOSE_TOLERANCE);

    // Check unit norm
    EXPECT_NEAR(result.v1.Norm(), 1.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(result.v2.Norm(), 1.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(result.v3.Norm(), 1.0, LOOSE_TOLERANCE);
}

// =============================================================================
// General NxN Jacobi Method Tests
// =============================================================================

class EigenSymmetricTest : public ::testing::Test {};

TEST_F(EigenSymmetricTest, SmallMatrix2x2) {
    MatX A(2, 2);
    A(0, 0) = 4.0; A(0, 1) = 2.0;
    A(1, 0) = 2.0; A(1, 1) = 4.0;

    auto result = EigenSymmetric(A);

    ASSERT_TRUE(result.valid);
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.eigenvalues.Size(), 2);

    // Eigenvalues: 6 and 2
    std::vector<double> eigs(2);
    for (int i = 0; i < 2; ++i) eigs[i] = result.eigenvalues[i];
    std::sort(eigs.begin(), eigs.end(), [](double a, double b) { return std::abs(a) > std::abs(b); });

    EXPECT_NEAR(eigs[0], 6.0, TOLERANCE);
    EXPECT_NEAR(eigs[1], 2.0, TOLERANCE);
}

TEST_F(EigenSymmetricTest, SmallMatrix3x3) {
    MatX A(3, 3);
    A(0, 0) = 2.0; A(0, 1) = 1.0; A(0, 2) = 0.0;
    A(1, 0) = 1.0; A(1, 1) = 2.0; A(1, 2) = 1.0;
    A(2, 0) = 0.0; A(2, 1) = 1.0; A(2, 2) = 2.0;

    auto result = EigenSymmetric(A);

    ASSERT_TRUE(result.valid);
    EXPECT_TRUE(result.converged);

    // Sum of eigenvalues = trace = 6
    double sum = 0.0;
    for (int i = 0; i < result.eigenvalues.Size(); ++i) {
        sum += result.eigenvalues[i];
    }
    EXPECT_NEAR(sum, 6.0, LOOSE_TOLERANCE);
}

TEST_F(EigenSymmetricTest, Matrix4x4) {
    MatX A(4, 4);
    // Symmetric positive definite
    A(0, 0) = 4; A(0, 1) = 1; A(0, 2) = 0; A(0, 3) = 0;
    A(1, 0) = 1; A(1, 1) = 4; A(1, 2) = 1; A(1, 3) = 0;
    A(2, 0) = 0; A(2, 1) = 1; A(2, 2) = 4; A(2, 3) = 1;
    A(3, 0) = 0; A(3, 1) = 0; A(3, 2) = 1; A(3, 3) = 4;

    auto result = EigenSymmetric(A);

    ASSERT_TRUE(result.valid);
    EXPECT_TRUE(result.converged);

    // All eigenvalues should be positive
    for (int i = 0; i < result.eigenvalues.Size(); ++i) {
        EXPECT_GT(result.eigenvalues[i], 0);
    }

    // Check eigenpairs
    for (int i = 0; i < result.eigenvalues.Size(); ++i) {
        VecX v(4);
        for (int j = 0; j < 4; ++j) {
            v[j] = result.eigenvectors(j, i);
        }
        EXPECT_TRUE(CheckEigenpair(A, result.eigenvalues[i], v, LOOSE_TOLERANCE));
    }
}

TEST_F(EigenSymmetricTest, Matrix5x5) {
    MatX A(5, 5);
    // Create symmetric matrix
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j <= i; ++j) {
            double val = (i == j) ? 5.0 + i : 1.0 / (1.0 + std::abs(i - j));
            A(i, j) = val;
            A(j, i) = val;
        }
    }

    auto result = EigenSymmetric(A);

    ASSERT_TRUE(result.valid);
    EXPECT_TRUE(result.converged);

    // Verify eigenvalue equation A*v = lambda*v
    for (int i = 0; i < result.eigenvalues.Size(); ++i) {
        VecX v(5);
        for (int j = 0; j < 5; ++j) {
            v[j] = result.eigenvectors(j, i);
        }
        EXPECT_TRUE(CheckEigenpair(A, result.eigenvalues[i], v, LOOSE_TOLERANCE));
    }
}

TEST_F(EigenSymmetricTest, EigenvectorsOrthonormal) {
    MatX A(4, 4);
    A(0, 0) = 5; A(0, 1) = 2; A(0, 2) = 1; A(0, 3) = 0;
    A(1, 0) = 2; A(1, 1) = 6; A(1, 2) = 2; A(1, 3) = 1;
    A(2, 0) = 1; A(2, 1) = 2; A(2, 2) = 7; A(2, 3) = 2;
    A(3, 0) = 0; A(3, 1) = 1; A(3, 2) = 2; A(3, 3) = 8;

    auto result = EigenSymmetric(A);

    ASSERT_TRUE(result.valid);

    // Check orthogonality
    EXPECT_TRUE(CheckOrthogonal(result.eigenvectors, LOOSE_TOLERANCE));

    // Check unit norm
    for (int i = 0; i < result.eigenvectors.Cols(); ++i) {
        double norm = 0.0;
        for (int j = 0; j < result.eigenvectors.Rows(); ++j) {
            norm += result.eigenvectors(j, i) * result.eigenvectors(j, i);
        }
        EXPECT_NEAR(std::sqrt(norm), 1.0, LOOSE_TOLERANCE);
    }
}

// =============================================================================
// Power Iteration Tests
// =============================================================================

class PowerIterationTest : public ::testing::Test {};

TEST_F(PowerIterationTest, DominantEigenvalue) {
    MatX A(3, 3);
    A(0, 0) = 4; A(0, 1) = 1; A(0, 2) = 0;
    A(1, 0) = 1; A(1, 1) = 3; A(1, 2) = 1;
    A(2, 0) = 0; A(2, 1) = 1; A(2, 2) = 2;

    auto [lambda, v] = PowerIteration(A);

    EXPECT_GT(v.Size(), 0);

    // Verify this is an eigenpair
    EXPECT_TRUE(CheckEigenpair(A, lambda, v, LOOSE_TOLERANCE));

    // Compare with full eigendecomposition
    auto full = EigenSymmetric(A);
    double maxEig = full.eigenvalues[0];
    for (int i = 1; i < full.eigenvalues.Size(); ++i) {
        if (std::abs(full.eigenvalues[i]) > std::abs(maxEig)) {
            maxEig = full.eigenvalues[i];
        }
    }
    EXPECT_NEAR(lambda, maxEig, LOOSE_TOLERANCE);
}

TEST_F(PowerIterationTest, WithInitialGuess) {
    MatX A(2, 2);
    A(0, 0) = 3; A(0, 1) = 1;
    A(1, 0) = 1; A(1, 1) = 3;

    VecX guess(2);
    guess[0] = 1.0;
    guess[1] = 1.0;

    auto [lambda, v] = PowerIteration(A, EIGEN_TOLERANCE, EIGEN_MAX_ITERATIONS, guess);

    EXPECT_NEAR(lambda, 4.0, LOOSE_TOLERANCE);
}

TEST_F(PowerIterationTest, InversePower) {
    MatX A(3, 3);
    A(0, 0) = 4; A(0, 1) = 1; A(0, 2) = 0;
    A(1, 0) = 1; A(1, 1) = 3; A(1, 2) = 1;
    A(2, 0) = 0; A(2, 1) = 1; A(2, 2) = 2;

    auto [lambda, v] = InversePowerIteration(A);

    EXPECT_GT(v.Size(), 0);

    // Should find smallest eigenvalue
    auto full = EigenSymmetric(A);
    double minEig = full.eigenvalues[0];
    for (int i = 1; i < full.eigenvalues.Size(); ++i) {
        if (std::abs(full.eigenvalues[i]) < std::abs(minEig)) {
            minEig = full.eigenvalues[i];
        }
    }
    EXPECT_NEAR(std::abs(lambda), std::abs(minEig), LOOSE_TOLERANCE);
}

TEST_F(PowerIterationTest, ShiftedInversePower) {
    MatX A(3, 3);
    A(0, 0) = 4; A(0, 1) = 0; A(0, 2) = 0;
    A(1, 0) = 0; A(1, 1) = 3; A(1, 2) = 0;
    A(2, 0) = 0; A(2, 1) = 0; A(2, 2) = 1;

    // Shift close to eigenvalue 3
    auto [lambda, v] = ShiftedInversePowerIteration(A, 2.9);

    EXPECT_NEAR(lambda, 3.0, LOOSE_TOLERANCE);
}

// =============================================================================
// Rayleigh Quotient Tests
// =============================================================================

class RayleighQuotientTest : public ::testing::Test {};

TEST_F(RayleighQuotientTest, ForEigenvector) {
    MatX A(2, 2);
    A(0, 0) = 3; A(0, 1) = 1;
    A(1, 0) = 1; A(1, 1) = 3;

    // Eigenvector for eigenvalue 4: [1, 1]/sqrt(2)
    VecX v(2);
    v[0] = 1.0 / std::sqrt(2.0);
    v[1] = 1.0 / std::sqrt(2.0);

    double rq = RayleighQuotient(A, v);
    EXPECT_NEAR(rq, 4.0, TOLERANCE);
}

TEST_F(RayleighQuotientTest, IterationConvergence) {
    MatX A(3, 3);
    A(0, 0) = 5; A(0, 1) = 1; A(0, 2) = 0;
    A(1, 0) = 1; A(1, 1) = 4; A(1, 2) = 1;
    A(2, 0) = 0; A(2, 1) = 1; A(2, 2) = 3;

    VecX guess(3);
    guess[0] = 1.0;
    guess[1] = 0.5;
    guess[2] = 0.3;

    auto [lambda, v] = RayleighQuotientIteration(A, guess);

    EXPECT_TRUE(CheckEigenpair(A, lambda, v, LOOSE_TOLERANCE));
}

// =============================================================================
// Utility Function Tests
// =============================================================================

class EigenUtilityTest : public ::testing::Test {};

TEST_F(EigenUtilityTest, IsPositiveDefinite) {
    MatX A(3, 3);
    A(0, 0) = 4; A(0, 1) = 1; A(0, 2) = 0;
    A(1, 0) = 1; A(1, 1) = 5; A(1, 2) = 1;
    A(2, 0) = 0; A(2, 1) = 1; A(2, 2) = 6;

    EXPECT_TRUE(IsPositiveDefinite(A));
}

TEST_F(EigenUtilityTest, IsNotPositiveDefinite) {
    MatX A(2, 2);
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 2; A(1, 1) = 1;

    // det = 1 - 4 = -3 < 0, so not PD
    EXPECT_FALSE(IsPositiveDefinite(A));
}

TEST_F(EigenUtilityTest, IsPositiveSemiDefinite) {
    MatX A(2, 2);
    A(0, 0) = 1; A(0, 1) = 1;
    A(1, 0) = 1; A(1, 1) = 1;

    // Eigenvalues: 2 and 0
    EXPECT_TRUE(IsPositiveSemiDefinite(A));
}

TEST_F(EigenUtilityTest, MatrixSquareRoot) {
    MatX A(2, 2);
    A(0, 0) = 4; A(0, 1) = 0;
    A(1, 0) = 0; A(1, 1) = 9;

    MatX sqrtA = MatrixSquareRoot(A);

    ASSERT_GT(sqrtA.Rows(), 0);

    // Check sqrtA * sqrtA ≈ A
    MatX check = sqrtA * sqrtA;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(check(i, j), A(i, j), LOOSE_TOLERANCE);
        }
    }
}

TEST_F(EigenUtilityTest, SortByMagnitude) {
    VecX eigenvalues(4);
    eigenvalues[0] = 1.0;
    eigenvalues[1] = -5.0;
    eigenvalues[2] = 3.0;
    eigenvalues[3] = -2.0;

    MatX eigenvectors = MatX::Identity(4);

    SortEigenByMagnitude(eigenvalues, eigenvectors);

    // Should be sorted by magnitude: -5, 3, -2, 1
    EXPECT_NEAR(std::abs(eigenvalues[0]), 5.0, TOLERANCE);
    EXPECT_NEAR(std::abs(eigenvalues[1]), 3.0, TOLERANCE);
    EXPECT_NEAR(std::abs(eigenvalues[2]), 2.0, TOLERANCE);
    EXPECT_NEAR(std::abs(eigenvalues[3]), 1.0, TOLERANCE);
}

TEST_F(EigenUtilityTest, SortByValue) {
    VecX eigenvalues(4);
    eigenvalues[0] = 1.0;
    eigenvalues[1] = -5.0;
    eigenvalues[2] = 3.0;
    eigenvalues[3] = -2.0;

    MatX eigenvectors = MatX::Identity(4);

    SortEigenByValue(eigenvalues, eigenvectors);

    // Should be sorted by value: 3, 1, -2, -5
    EXPECT_NEAR(eigenvalues[0], 3.0, TOLERANCE);
    EXPECT_NEAR(eigenvalues[1], 1.0, TOLERANCE);
    EXPECT_NEAR(eigenvalues[2], -2.0, TOLERANCE);
    EXPECT_NEAR(eigenvalues[3], -5.0, TOLERANCE);
}

// =============================================================================
// Tridiagonalization Tests
// =============================================================================

class TridiagonalizeTest : public ::testing::Test {};

TEST_F(TridiagonalizeTest, PreservesEigenvalues) {
    MatX A(4, 4);
    A(0, 0) = 4; A(0, 1) = 1; A(0, 2) = 0; A(0, 3) = 0;
    A(1, 0) = 1; A(1, 1) = 4; A(1, 2) = 1; A(1, 3) = 0;
    A(2, 0) = 0; A(2, 1) = 1; A(2, 2) = 4; A(2, 3) = 1;
    A(3, 0) = 0; A(3, 1) = 0; A(3, 2) = 1; A(3, 3) = 4;

    auto [T, Q] = Tridiagonalize(A);

    // Eigenvalues of T should match A
    VecX eigsA = EigenvaluesSymmetric(A);
    VecX eigsT = EigenvaluesSymmetric(T);

    std::vector<double> sortedA(eigsA.Size()), sortedT(eigsT.Size());
    for (int i = 0; i < eigsA.Size(); ++i) sortedA[i] = eigsA[i];
    for (int i = 0; i < eigsT.Size(); ++i) sortedT[i] = eigsT[i];

    std::sort(sortedA.begin(), sortedA.end());
    std::sort(sortedT.begin(), sortedT.end());

    for (size_t i = 0; i < sortedA.size() && i < sortedT.size(); ++i) {
        EXPECT_NEAR(sortedA[i], sortedT[i], LOOSE_TOLERANCE);
    }
}

// =============================================================================
// General Eigenvalue Tests
// =============================================================================

class EigenGeneralTest : public ::testing::Test {};

TEST_F(EigenGeneralTest, SmallMatrix2x2Real) {
    MatX A(2, 2);
    A(0, 0) = 4; A(0, 1) = 1;
    A(1, 0) = 2; A(1, 1) = 3;

    auto result = EigenGeneral(A);

    ASSERT_TRUE(result.valid);

    // Eigenvalues: 5 and 2
    double sum = result.eigenvalues[0] + result.eigenvalues[1];
    double prod = result.eigenvalues[0] * result.eigenvalues[1];
    EXPECT_NEAR(sum, 7.0, LOOSE_TOLERANCE);  // trace
    EXPECT_NEAR(prod, 10.0, LOOSE_TOLERANCE); // determinant
}

TEST_F(EigenGeneralTest, SmallMatrix3x3) {
    MatX A(3, 3);
    A(0, 0) = 2; A(0, 1) = 1; A(0, 2) = 0;
    A(1, 0) = 0; A(1, 1) = 3; A(1, 2) = 1;
    A(2, 0) = 0; A(2, 1) = 0; A(2, 2) = 4;

    auto result = EigenGeneral(A);

    ASSERT_TRUE(result.valid);

    // Upper triangular: eigenvalues are diagonal elements
    std::vector<double> eigs(result.eigenvalues.Size());
    for (int i = 0; i < result.eigenvalues.Size(); ++i) {
        eigs[i] = result.eigenvalues[i];
    }
    std::sort(eigs.begin(), eigs.end());

    EXPECT_NEAR(eigs[0], 2.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(eigs[1], 3.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(eigs[2], 4.0, LOOSE_TOLERANCE);
}

// =============================================================================
// Generalized Eigenvalue Tests
// =============================================================================

class GeneralizedEigenTest : public ::testing::Test {};

TEST_F(GeneralizedEigenTest, Simple2x2) {
    Mat22 A, B;
    A(0, 0) = 2; A(0, 1) = 0;
    A(1, 0) = 0; A(1, 1) = 3;

    B(0, 0) = 1; B(0, 1) = 0;
    B(1, 0) = 0; B(1, 1) = 1;

    auto result = GeneralizedEigen2x2(A, B);

    ASSERT_TRUE(result.valid);
    EXPECT_TRUE(result.isReal);

    // With B = I, should get eigenvalues of A
    std::vector<double> eigs = {result.lambda1, result.lambda2};
    std::sort(eigs.begin(), eigs.end());
    EXPECT_NEAR(eigs[0], 2.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(eigs[1], 3.0, LOOSE_TOLERANCE);
}

TEST_F(GeneralizedEigenTest, Simple3x3) {
    Mat33 A, B;
    A(0, 0) = 2; A(0, 1) = 0; A(0, 2) = 0;
    A(1, 0) = 0; A(1, 1) = 4; A(1, 2) = 0;
    A(2, 0) = 0; A(2, 1) = 0; A(2, 2) = 6;

    B = Mat33::Identity();

    auto result = GeneralizedEigen3x3(A, B);

    ASSERT_TRUE(result.valid);
    EXPECT_TRUE(result.allReal);

    std::vector<double> eigs = {result.lambda1, result.lambda2, result.lambda3};
    std::sort(eigs.begin(), eigs.end());
    EXPECT_NEAR(eigs[0], 2.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(eigs[1], 4.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(eigs[2], 6.0, LOOSE_TOLERANCE);
}

// =============================================================================
// Matrix Functions Tests
// =============================================================================

class MatrixFunctionsTest : public ::testing::Test {};

TEST_F(MatrixFunctionsTest, Exponential) {
    MatX A(2, 2);
    A(0, 0) = 0; A(0, 1) = 0;
    A(1, 0) = 0; A(1, 1) = 0;

    MatX expA = MatrixExponential(A);

    // exp(0) = I
    ASSERT_GT(expA.Rows(), 0);
    EXPECT_NEAR(expA(0, 0), 1.0, TOLERANCE);
    EXPECT_NEAR(expA(0, 1), 0.0, TOLERANCE);
    EXPECT_NEAR(expA(1, 0), 0.0, TOLERANCE);
    EXPECT_NEAR(expA(1, 1), 1.0, TOLERANCE);
}

TEST_F(MatrixFunctionsTest, LogarithmOfIdentity) {
    MatX A = MatX::Identity(3);

    MatX logA = MatrixLogarithm(A);

    // log(I) = 0
    ASSERT_GT(logA.Rows(), 0);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(logA(i, j), 0.0, TOLERANCE);
        }
    }
}

TEST_F(MatrixFunctionsTest, SquareRootOfDiagonal) {
    MatX A(2, 2);
    A(0, 0) = 4; A(0, 1) = 0;
    A(1, 0) = 0; A(1, 1) = 16;

    MatX sqrtA = MatrixSquareRoot(A);

    ASSERT_GT(sqrtA.Rows(), 0);
    EXPECT_NEAR(sqrtA(0, 0), 2.0, TOLERANCE);
    EXPECT_NEAR(sqrtA(1, 1), 4.0, TOLERANCE);
}

// =============================================================================
// Edge Cases Tests
// =============================================================================

class EigenEdgeCaseTest : public ::testing::Test {};

TEST_F(EigenEdgeCaseTest, SingleElementMatrix) {
    MatX A(1, 1);
    A(0, 0) = 5.0;

    auto result = EigenSymmetric(A);

    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.eigenvalues.Size(), 1);
    EXPECT_NEAR(result.eigenvalues[0], 5.0, TOLERANCE);
}

TEST_F(EigenEdgeCaseTest, EmptyMatrix) {
    MatX A;

    auto result = EigenSymmetric(A);

    EXPECT_FALSE(result.valid);
}

TEST_F(EigenEdgeCaseTest, VerySmallOffDiagonal) {
    Mat22 A;
    A(0, 0) = 5.0; A(0, 1) = 1e-15;
    A(1, 0) = 1e-15; A(1, 1) = 3.0;

    auto result = EigenSymmetric2x2(A);

    ASSERT_TRUE(result.valid);
    EXPECT_NEAR(result.lambda1, 5.0, TOLERANCE);
    EXPECT_NEAR(result.lambda2, 3.0, TOLERANCE);
}

TEST_F(EigenEdgeCaseTest, LargeValues) {
    Mat22 A;
    A(0, 0) = 1e10; A(0, 1) = 1e5;
    A(1, 0) = 1e5; A(1, 1) = 1e10;

    auto result = EigenSymmetric2x2(A);

    ASSERT_TRUE(result.valid);
    // Should handle large values correctly
    EXPECT_GT(std::abs(result.lambda1), 1e9);
}

TEST_F(EigenEdgeCaseTest, SmallValues) {
    Mat22 A;
    A(0, 0) = 1e-10; A(0, 1) = 1e-15;
    A(1, 0) = 1e-15; A(1, 1) = 1e-10;

    auto result = EigenSymmetric2x2(A);

    ASSERT_TRUE(result.valid);
    // Should handle small values correctly
    EXPECT_LT(std::abs(result.lambda1), 1e-9);
}

// =============================================================================
// Convergence Tests
// =============================================================================

class ConvergenceTest : public ::testing::Test {};

TEST_F(ConvergenceTest, JacobiIterationCount) {
    MatX A(4, 4);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j <= i; ++j) {
            double val = (i == j) ? 4.0 : 0.5;
            A(i, j) = val;
            A(j, i) = val;
        }
    }

    auto result = EigenSymmetric(A);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.iterations, EIGEN_MAX_ITERATIONS);
}

TEST_F(ConvergenceTest, PowerIterationConvergence) {
    MatX A(3, 3);
    A(0, 0) = 10; A(0, 1) = 1; A(0, 2) = 0;
    A(1, 0) = 1; A(1, 1) = 5; A(1, 2) = 1;
    A(2, 0) = 0; A(2, 1) = 1; A(2, 2) = 1;

    auto [lambda, v] = PowerIteration(A, 1e-10, 100);

    EXPECT_GT(v.Size(), 0);
    // Use more relaxed tolerance for power iteration (iterative method)
    EXPECT_TRUE(CheckEigenpair(A, lambda, v, 1e-4));
}
