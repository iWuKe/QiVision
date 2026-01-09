#pragma once

/**
 * @file Eigen.h
 * @brief Eigenvalue decomposition for QiVision
 *
 * This module provides:
 * - Symmetric eigenvalue decomposition (Jacobi, QR iteration)
 * - General eigenvalue decomposition
 * - Power iteration for dominant eigenvalue
 * - Specialized 2x2, 3x3 closed-form solutions
 *
 * Used by:
 * - Calib module (camera calibration, homography decomposition)
 * - PCA operations (covariance eigenanalysis)
 * - Fitting.h (ellipse fitting via conic eigenvalue problem)
 * - Shape analysis (moment of inertia, principal axes)
 *
 * Design principles:
 * - Numerical stability via Jacobi rotations for symmetric matrices
 * - Eigenvalues sorted in descending order by default
 * - Eigenvectors normalized and orthogonal
 * - Specialized small matrix routines avoid iteration overhead
 */

#include <QiVision/Internal/Matrix.h>

#include <vector>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Default tolerance for eigenvalue convergence
constexpr double EIGEN_TOLERANCE = 1e-12;

/// Maximum iterations for Jacobi/QR methods
constexpr int EIGEN_MAX_ITERATIONS = 100;

/// Tolerance for considering eigenvalues as equal (for multiplicity)
constexpr double EIGEN_MULTIPLICITY_TOLERANCE = 1e-10;

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief Result of eigenvalue decomposition
 *
 * For matrix A, computes A = V * D * V^T (symmetric) or A * V = V * D (general)
 * where D is diagonal matrix of eigenvalues
 */
struct EigenResult {
    VecX eigenvalues;      ///< Eigenvalues (real parts), sorted descending by magnitude
    MatX eigenvectors;     ///< Eigenvectors as columns, V(:,i) corresponds to eigenvalues[i]
    VecX imaginaryParts;   ///< Imaginary parts of eigenvalues (empty for symmetric)
    int iterations;        ///< Number of iterations used
    bool converged;        ///< True if algorithm converged
    bool valid;            ///< True if decomposition succeeded

    EigenResult() : iterations(0), converged(false), valid(false) {}
};

/**
 * @brief Result for 2x2 eigenvalue decomposition
 */
struct Eigen2x2Result {
    double lambda1;        ///< First eigenvalue (larger magnitude)
    double lambda2;        ///< Second eigenvalue
    Vec2 v1;              ///< First eigenvector
    Vec2 v2;              ///< Second eigenvector
    bool isReal;          ///< True if eigenvalues are real (always true for symmetric)
    double imagPart;      ///< Imaginary part if complex (lambda = lambda1 ± i*imagPart)
    bool valid;

    Eigen2x2Result() : lambda1(0), lambda2(0), isReal(true), imagPart(0), valid(false) {}
};

/**
 * @brief Result for 3x3 eigenvalue decomposition
 */
struct Eigen3x3Result {
    double lambda1;        ///< First eigenvalue (largest magnitude)
    double lambda2;        ///< Second eigenvalue
    double lambda3;        ///< Third eigenvalue
    Vec3 v1;              ///< First eigenvector
    Vec3 v2;              ///< Second eigenvector
    Vec3 v3;              ///< Third eigenvector
    bool allReal;         ///< True if all eigenvalues are real
    bool valid;

    Eigen3x3Result() : lambda1(0), lambda2(0), lambda3(0), allReal(true), valid(false) {}
};

// =============================================================================
// Symmetric Eigenvalue Decomposition
// =============================================================================

/**
 * @brief Eigenvalue decomposition for symmetric matrices using Jacobi method
 *
 * Computes A = V * D * V^T where:
 * - D is diagonal with eigenvalues
 * - V is orthogonal with eigenvectors as columns
 *
 * @param A Symmetric matrix (n x n)
 * @param tolerance Convergence tolerance for off-diagonal elements
 * @param maxIterations Maximum number of sweeps
 * @return EigenResult with real eigenvalues sorted by magnitude (descending)
 *
 * Complexity: O(n^3) per sweep, typically converges in O(log n) sweeps
 *
 * @note Jacobi method is very stable and preferred for symmetric matrices
 * @note Input matrix is NOT checked for symmetry - caller must ensure this
 */
EigenResult EigenSymmetric(const MatX& A,
                           double tolerance = EIGEN_TOLERANCE,
                           int maxIterations = EIGEN_MAX_ITERATIONS);

/**
 * @brief Eigenvalue decomposition using QR iteration with shifts
 *
 * More efficient than Jacobi for larger matrices but requires tridiagonalization
 *
 * @param A Symmetric matrix (n x n)
 * @param tolerance Convergence tolerance
 * @param maxIterations Maximum iterations
 * @return EigenResult with eigenvalues and eigenvectors
 */
EigenResult EigenSymmetricQR(const MatX& A,
                             double tolerance = EIGEN_TOLERANCE,
                             int maxIterations = EIGEN_MAX_ITERATIONS);

/**
 * @brief Compute only eigenvalues of symmetric matrix (faster, no eigenvectors)
 *
 * @param A Symmetric matrix
 * @return Vector of eigenvalues sorted descending by magnitude
 */
VecX EigenvaluesSymmetric(const MatX& A);

// =============================================================================
// General Eigenvalue Decomposition
// =============================================================================

/**
 * @brief Eigenvalue decomposition for general (non-symmetric) matrices
 *
 * Uses QR iteration with Hessenberg reduction
 * May produce complex eigenvalues (returned as conjugate pairs)
 *
 * @param A Square matrix (n x n)
 * @param tolerance Convergence tolerance
 * @param maxIterations Maximum iterations
 * @return EigenResult with possibly complex eigenvalues
 *
 * @note For complex eigenvalues, imaginaryParts contains the imaginary components
 * @note Eigenvectors for complex eigenvalues are also complex (stored as pairs)
 */
EigenResult EigenGeneral(const MatX& A,
                         double tolerance = EIGEN_TOLERANCE,
                         int maxIterations = EIGEN_MAX_ITERATIONS);

/**
 * @brief Compute only eigenvalues of general matrix
 *
 * @param A Square matrix
 * @return Vector of real parts of eigenvalues, imaginaryParts as second return
 */
std::pair<VecX, VecX> EigenvaluesGeneral(const MatX& A);

// =============================================================================
// Power Iteration Methods
// =============================================================================

/**
 * @brief Power iteration for dominant eigenvalue
 *
 * Fast method to find largest magnitude eigenvalue and its eigenvector
 *
 * @param A Square matrix
 * @param tolerance Convergence tolerance
 * @param maxIterations Maximum iterations
 * @param initialGuess Optional starting vector (random if not provided)
 * @return Pair of (eigenvalue, eigenvector), eigenvector is unit norm
 */
std::pair<double, VecX> PowerIteration(const MatX& A,
                                        double tolerance = EIGEN_TOLERANCE,
                                        int maxIterations = EIGEN_MAX_ITERATIONS,
                                        const VecX& initialGuess = VecX());

/**
 * @brief Inverse power iteration for smallest eigenvalue
 *
 * Finds eigenvalue closest to zero (smallest magnitude)
 *
 * @param A Square non-singular matrix
 * @param tolerance Convergence tolerance
 * @param maxIterations Maximum iterations
 * @return Pair of (eigenvalue, eigenvector)
 */
std::pair<double, VecX> InversePowerIteration(const MatX& A,
                                               double tolerance = EIGEN_TOLERANCE,
                                               int maxIterations = EIGEN_MAX_ITERATIONS);

/**
 * @brief Shifted inverse power iteration
 *
 * Finds eigenvalue closest to given shift value
 *
 * @param A Square matrix
 * @param shift Value to shift by (finds eigenvalue closest to this)
 * @param tolerance Convergence tolerance
 * @param maxIterations Maximum iterations
 * @return Pair of (eigenvalue, eigenvector)
 */
std::pair<double, VecX> ShiftedInversePowerIteration(const MatX& A,
                                                      double shift,
                                                      double tolerance = EIGEN_TOLERANCE,
                                                      int maxIterations = EIGEN_MAX_ITERATIONS);

/**
 * @brief Rayleigh quotient iteration
 *
 * Fast convergence (cubic) when initial guess is close to eigenvector
 *
 * @param A Symmetric matrix
 * @param initialGuess Starting vector
 * @param tolerance Convergence tolerance
 * @param maxIterations Maximum iterations
 * @return Pair of (eigenvalue, eigenvector)
 */
std::pair<double, VecX> RayleighQuotientIteration(const MatX& A,
                                                   const VecX& initialGuess,
                                                   double tolerance = EIGEN_TOLERANCE,
                                                   int maxIterations = EIGEN_MAX_ITERATIONS);

// =============================================================================
// Specialized 2x2 Eigenvalue (Closed-form)
// =============================================================================

/**
 * @brief 2x2 symmetric eigenvalue decomposition (analytical)
 *
 * Direct formula, no iteration needed
 * A = [a b; b c]
 *
 * @param A 2x2 symmetric matrix
 * @return Eigen2x2Result with eigenvalues and orthogonal eigenvectors
 */
Eigen2x2Result EigenSymmetric2x2(const Mat22& A);

/**
 * @brief 2x2 general eigenvalue decomposition (analytical)
 *
 * Handles both real and complex eigenvalue cases
 * A = [a b; c d]
 *
 * @param A 2x2 matrix
 * @return Eigen2x2Result (check isReal flag for complex case)
 */
Eigen2x2Result EigenGeneral2x2(const Mat22& A);

/**
 * @brief 2x2 eigenvalues only (fastest)
 *
 * @param A 2x2 matrix
 * @return Pair of eigenvalues (real parts), second pair element is imaginary if complex
 */
std::pair<double, double> Eigenvalues2x2(const Mat22& A);

// =============================================================================
// Specialized 3x3 Eigenvalue
// =============================================================================

/**
 * @brief 3x3 symmetric eigenvalue decomposition (analytical)
 *
 * Uses Cardano's formula for cubic roots
 * Very efficient for covariance matrices, moment of inertia, etc.
 *
 * @param A 3x3 symmetric matrix
 * @return Eigen3x3Result with three real eigenvalues and orthonormal eigenvectors
 */
Eigen3x3Result EigenSymmetric3x3(const Mat33& A);

/**
 * @brief 3x3 general eigenvalue decomposition
 *
 * @param A 3x3 matrix
 * @return Eigen3x3Result (may have complex eigenvalues)
 */
Eigen3x3Result EigenGeneral3x3(const Mat33& A);

/**
 * @brief 3x3 eigenvalues only
 *
 * @param A 3x3 matrix
 * @return Array of three eigenvalues (real parts)
 */
std::array<double, 3> Eigenvalues3x3(const Mat33& A);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Sort eigenvalues and eigenvectors by eigenvalue magnitude (descending)
 *
 * @param eigenvalues Eigenvalues to sort
 * @param eigenvectors Corresponding eigenvectors (columns reordered)
 */
void SortEigenByMagnitude(VecX& eigenvalues, MatX& eigenvectors);

/**
 * @brief Sort eigenvalues and eigenvectors by eigenvalue (descending, algebraic)
 */
void SortEigenByValue(VecX& eigenvalues, MatX& eigenvectors);

/**
 * @brief Compute Rayleigh quotient: (x^T * A * x) / (x^T * x)
 *
 * Gives best eigenvalue estimate for given vector
 */
double RayleighQuotient(const MatX& A, const VecX& x);

/**
 * @brief Check if matrix is positive definite via eigenvalues
 *
 * @param A Symmetric matrix
 * @param tolerance Minimum eigenvalue threshold
 * @return True if all eigenvalues > tolerance
 */
bool IsPositiveDefinite(const MatX& A, double tolerance = 0.0);

/**
 * @brief Check if matrix is positive semi-definite
 *
 * @param A Symmetric matrix
 * @param tolerance Eigenvalue threshold
 * @return True if all eigenvalues >= -tolerance
 */
bool IsPositiveSemiDefinite(const MatX& A, double tolerance = EIGEN_TOLERANCE);

/**
 * @brief Compute matrix square root for positive semi-definite matrix
 *
 * Returns B such that B * B = A (within tolerance)
 *
 * @param A Symmetric positive semi-definite matrix
 * @return Matrix square root, or empty matrix if not PSD
 */
MatX MatrixSquareRoot(const MatX& A);

/**
 * @brief Compute matrix exponential approximation via eigendecomposition
 *
 * exp(A) = V * exp(D) * V^T for symmetric A
 *
 * @param A Symmetric matrix
 * @return Matrix exponential
 */
MatX MatrixExponential(const MatX& A);

/**
 * @brief Compute matrix logarithm for positive definite matrix
 *
 * log(A) = V * log(D) * V^T
 *
 * @param A Symmetric positive definite matrix
 * @return Matrix logarithm, or empty matrix if not PD
 */
MatX MatrixLogarithm(const MatX& A);

/**
 * @brief Compute eigenvalue condition numbers
 *
 * Condition number for eigenvalue lambda_i indicates sensitivity to perturbations
 *
 * @param A Square matrix
 * @return Vector of condition numbers for each eigenvalue
 */
VecX EigenvalueConditionNumbers(const MatX& A);

/**
 * @brief Tridiagonalize symmetric matrix (Householder reduction)
 *
 * Preprocessing step for QR iteration
 * A = Q * T * Q^T where T is tridiagonal
 *
 * @param A Symmetric matrix
 * @return Pair of (T, Q)
 */
std::pair<MatX, MatX> Tridiagonalize(const MatX& A);

/**
 * @brief Reduce to upper Hessenberg form
 *
 * Preprocessing for general eigenvalue problem
 * A = Q * H * Q^T where H is upper Hessenberg
 *
 * @param A Square matrix
 * @return Pair of (H, Q)
 */
std::pair<MatX, MatX> HessenbergReduce(const MatX& A);

// =============================================================================
// Generalized Eigenvalue Problem
// =============================================================================

/**
 * @brief Solve generalized eigenvalue problem A * x = lambda * B * x
 *
 * For symmetric A, positive definite B
 * Used in constrained optimization, vibration analysis
 *
 * @param A Symmetric matrix
 * @param B Symmetric positive definite matrix
 * @return EigenResult for the generalized problem
 */
EigenResult GeneralizedEigen(const MatX& A, const MatX& B);

/**
 * @brief Solve generalized eigenvalue for 2x2 system
 */
Eigen2x2Result GeneralizedEigen2x2(const Mat22& A, const Mat22& B);

/**
 * @brief Solve generalized eigenvalue for 3x3 system
 */
Eigen3x3Result GeneralizedEigen3x3(const Mat33& A, const Mat33& B);

} // namespace Qi::Vision::Internal
