#pragma once

/**
 * @file Solver.h
 * @brief Linear equation solvers for QiVision
 *
 * This module provides:
 * - Matrix decompositions: LU, Cholesky, QR, SVD
 * - Linear system solvers for various matrix types
 * - Utility functions: condition number, rank, pseudo-inverse
 * - Specialized solvers for small matrices (2x2, 3x3, 4x4)
 *
 * Used by:
 * - Fitting.h (geometric fitting)
 * - Homography.h (perspective transforms)
 * - Calib module (camera calibration)
 * - Eigen.h (eigenvalue decomposition)
 *
 * Design principles:
 * - Numerical stability via pivoting and Householder reflections
 * - Graceful degradation for singular/rank-deficient matrices
 * - Pre-computed decomposition reuse for multiple right-hand sides
 * - Small matrix specialization to avoid overhead
 */

#include <QiVision/Internal/Matrix.h>

#include <vector>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Default tolerance for rank determination
constexpr double SOLVER_RANK_TOLERANCE = 1e-10;

/// Default tolerance for singularity detection
constexpr double SOLVER_SINGULAR_THRESHOLD = 1e-12;

/// Maximum iterations for SVD convergence
constexpr int SVD_MAX_ITERATIONS = 50;

/// Maximum iterations for iterative refinement
constexpr int SOLVER_MAX_REFINEMENT_ITERATIONS = 3;

// =============================================================================
// Matrix Decomposition Functions
// =============================================================================

/**
 * @brief LU decomposition with partial pivoting
 * Computes PA = LU where P is permutation, L lower triangular (diagonal=1), U upper triangular
 *
 * @param A Input square matrix (n x n)
 * @return LUResult containing L, U, P, and validity flag
 *
 * Complexity: O(n^3)
 */
LUResult LU_Decompose(const MatX& A);

/**
 * @brief In-place LU decomposition
 * A is overwritten with L (lower, unit diagonal) and U (upper)
 *
 * @param A Input/output matrix, overwritten with LU factors
 * @param P Output permutation vector
 * @return sign of permutation (+1 or -1), 0 if singular
 */
int LU_DecomposeInPlace(MatX& A, std::vector<int>& P);

/**
 * @brief Cholesky decomposition for symmetric positive definite matrices
 * Computes A = L * L^T where L is lower triangular
 *
 * @param A Input symmetric positive definite matrix
 * @return CholeskyResult containing L and validity flag
 *
 * Complexity: O(n^3/3), ~2x faster than LU
 */
CholeskyResult Cholesky_Decompose(const MatX& A);

/**
 * @brief QR decomposition using Householder reflections
 * Computes A = Q * R where Q is orthogonal (m x m), R is upper triangular (m x n)
 *
 * @param A Input matrix (m x n), m >= n for least squares
 * @return QRResult containing Q, R, and validity flag
 *
 * Complexity: O(2mn^2 - 2n^3/3)
 */
QRResult QR_Decompose(const MatX& A);

/**
 * @brief Thin QR decomposition (economy size)
 * Computes A = Q_thin * R_thin where Q_thin is (m x n), R_thin is (n x n)
 * More memory efficient for m >> n
 *
 * @param A Input matrix (m x n), requires m >= n
 * @return QRResult with thinQR=true
 */
QRResult QR_DecomposeThin(const MatX& A);

/**
 * @brief Singular Value Decomposition
 * Computes A = U * S * V^T where U, V are orthogonal, S is diagonal
 *
 * @param A Input matrix (m x n)
 * @param fullMatrices If true, compute full U (m x m) and V (n x n);
 *                     if false, compute thin U (m x k) and V (n x k) where k = min(m,n)
 * @return SVDResult containing U, S (as vector), V, rank
 *
 * Complexity: O(min(mn^2, m^2n))
 */
SVDResult SVD_Decompose(const MatX& A, bool fullMatrices = false);

// =============================================================================
// Linear System Solvers - Direct
// =============================================================================

/**
 * @brief Solve Ax = b using LU decomposition
 * For square systems with unique solution
 *
 * @param A Coefficient matrix (n x n)
 * @param b Right-hand side vector (n)
 * @return Solution x, or zero vector if singular
 *
 * @throws InvalidArgumentException if dimensions don't match
 */
VecX SolveLU(const MatX& A, const VecX& b);

/**
 * @brief Solve Ax = b using Cholesky decomposition
 * For symmetric positive definite systems, ~2x faster than LU
 *
 * @param A Symmetric positive definite matrix (n x n)
 * @param b Right-hand side vector (n)
 * @return Solution x, or zero vector if not positive definite
 */
VecX SolveCholesky(const MatX& A, const VecX& b);

/**
 * @brief Solve overdetermined Ax = b in least squares sense using QR
 * Minimizes ||Ax - b||^2
 *
 * @param A Coefficient matrix (m x n), m >= n
 * @param b Right-hand side vector (m)
 * @return Least squares solution x (n)
 */
VecX SolveLeastSquares(const MatX& A, const VecX& b);

/**
 * @brief Solve least squares via normal equations
 * Solves (A^T A) x = A^T b using Cholesky
 * Faster but less stable than QR for ill-conditioned problems
 *
 * @param A Coefficient matrix (m x n)
 * @param b Right-hand side vector (m)
 * @return Least squares solution x (n)
 */
VecX SolveLeastSquaresNormal(const MatX& A, const VecX& b);

/**
 * @brief Solve using SVD (most robust)
 * Handles rank-deficient matrices, returns minimum-norm solution
 *
 * @param A Coefficient matrix (m x n)
 * @param b Right-hand side vector (m)
 * @param tolerance Singular values below this are treated as zero
 * @return Minimum-norm least squares solution
 */
VecX SolveSVD(const MatX& A, const VecX& b, double tolerance = SOLVER_RANK_TOLERANCE);

/**
 * @brief Solve homogeneous system Ax = 0
 * Returns unit vector in null space (right singular vector for smallest singular value)
 *
 * @param A Coefficient matrix (m x n), typically m >= n
 * @return Unit null space vector x such that ||Ax|| is minimized
 */
VecX SolveHomogeneous(const MatX& A);

// =============================================================================
// Solve from Pre-computed Decomposition
// =============================================================================

/**
 * @brief Solve from pre-computed LU decomposition
 * Useful when solving multiple systems with same A
 *
 * @param lu LU decomposition result
 * @param b Right-hand side vector
 * @return Solution x
 */
VecX SolveFromLU(const LUResult& lu, const VecX& b);

/**
 * @brief Solve multiple right-hand sides from LU
 * @param lu LU decomposition result
 * @param B Multiple right-hand sides (n x k)
 * @return Solutions X (n x k)
 */
MatX SolveFromLU(const LUResult& lu, const MatX& B);

/**
 * @brief Solve from pre-computed Cholesky decomposition
 */
VecX SolveFromCholesky(const CholeskyResult& chol, const VecX& b);

/**
 * @brief Solve from pre-computed QR decomposition (least squares)
 */
VecX SolveFromQR(const QRResult& qr, const VecX& b);

/**
 * @brief Solve from pre-computed SVD
 * @param svd SVD decomposition result
 * @param b Right-hand side vector
 * @param tolerance Singular value tolerance
 * @return Minimum-norm least squares solution
 */
VecX SolveFromSVD(const SVDResult& svd, const VecX& b, double tolerance = SOLVER_RANK_TOLERANCE);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Compute condition number of matrix
 * Ratio of largest to smallest singular value
 *
 * @param A Input matrix
 * @return Condition number (infinity if singular)
 */
double ComputeConditionNumber(const MatX& A);

/**
 * @brief Estimate condition number using LU (faster, approximate)
 * Uses 1-norm estimation technique
 */
double EstimateConditionNumber(const MatX& A);

/**
 * @brief Compute numerical rank of matrix
 * Count of singular values above tolerance
 *
 * @param A Input matrix
 * @param tolerance Threshold for zero singular values
 * @return Numerical rank
 */
int ComputeRank(const MatX& A, double tolerance = SOLVER_RANK_TOLERANCE);

/**
 * @brief Compute Moore-Penrose pseudo-inverse
 * A^+ such that A * A^+ * A = A
 *
 * @param A Input matrix (m x n)
 * @param tolerance Singular value tolerance
 * @return Pseudo-inverse A^+ (n x m)
 */
MatX PseudoInverse(const MatX& A, double tolerance = SOLVER_RANK_TOLERANCE);

/**
 * @brief Compute residual ||Ax - b||
 * @return L2 norm of residual
 */
double ComputeResidual(const MatX& A, const VecX& x, const VecX& b);

/**
 * @brief Compute relative residual ||Ax - b|| / ||b||
 */
double ComputeRelativeResidual(const MatX& A, const VecX& x, const VecX& b);

/**
 * @brief Compute null space basis
 * Returns orthonormal basis for null space of A
 *
 * @param A Input matrix
 * @param tolerance Singular value tolerance
 * @return Matrix whose columns span null(A)
 */
MatX ComputeNullSpace(const MatX& A, double tolerance = SOLVER_RANK_TOLERANCE);

/**
 * @brief Compute null space dimension
 */
int ComputeNullity(const MatX& A, double tolerance = SOLVER_RANK_TOLERANCE);

// =============================================================================
// Specialized Solvers
// =============================================================================

/**
 * @brief Solve triangular system Lx = b (lower triangular)
 * @param L Lower triangular matrix
 * @param b Right-hand side
 * @param unitDiagonal If true, assume diagonal elements are 1
 * @return Solution x
 */
VecX SolveLowerTriangular(const MatX& L, const VecX& b, bool unitDiagonal = false);

/**
 * @brief Solve triangular system Ux = b (upper triangular)
 */
VecX SolveUpperTriangular(const MatX& U, const VecX& b);

/**
 * @brief Solve tridiagonal system
 * For banded matrices common in spline fitting
 *
 * @param lower Lower diagonal (n-1)
 * @param diag Main diagonal (n)
 * @param upper Upper diagonal (n-1)
 * @param b Right-hand side (n)
 * @return Solution x
 *
 * Complexity: O(n)
 */
VecX SolveTridiagonal(const VecX& lower, const VecX& diag, const VecX& upper, const VecX& b);

// =============================================================================
// Small Matrix Specializations (inline, no heap allocation)
// =============================================================================

/**
 * @brief Solve 2x2 system directly
 * Uses Cramer's rule, avoids decomposition overhead
 * @return Solution, or zero vector if singular
 */
Vec2 Solve2x2(const Mat22& A, const Vec2& b);

/**
 * @brief Solve 3x3 system directly
 * Uses Cramer's rule
 * @return Solution, or zero vector if singular
 */
Vec3 Solve3x3(const Mat33& A, const Vec3& b);

/**
 * @brief Solve 4x4 system directly
 * Uses matrix inverse
 * @return Solution, or zero vector if singular
 */
Vec4 Solve4x4(const Mat44& A, const Vec4& b);

/**
 * @brief Check if 2x2 system has unique solution
 */
bool IsSolvable2x2(const Mat22& A, double tolerance = SOLVER_SINGULAR_THRESHOLD);

/**
 * @brief Check if 3x3 system has unique solution
 */
bool IsSolvable3x3(const Mat33& A, double tolerance = SOLVER_SINGULAR_THRESHOLD);

/**
 * @brief Check if 4x4 system has unique solution
 */
bool IsSolvable4x4(const Mat44& A, double tolerance = SOLVER_SINGULAR_THRESHOLD);

} // namespace Qi::Vision::Internal
