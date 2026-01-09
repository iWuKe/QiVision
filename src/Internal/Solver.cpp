/**
 * @file Solver.cpp
 * @brief Implementation of linear equation solvers
 */

#include <QiVision/Internal/Solver.h>
#include <QiVision/Core/Exception.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace Qi::Vision::Internal {

// =============================================================================
// Helper Functions (Forward Declarations)
// =============================================================================

namespace {

/// Sign function (returns +1 for x >= 0, -1 for x < 0)
inline double Sign(double x) {
    return (x >= 0.0) ? 1.0 : -1.0;
}


} // anonymous namespace

// =============================================================================
// Small Matrix Specializations
// =============================================================================

Vec2 Solve2x2(const Mat22& A, const Vec2& b) {
    double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    if (std::abs(det) < SOLVER_SINGULAR_THRESHOLD) {
        return Vec2::Zero();
    }
    double invDet = 1.0 / det;
    return Vec2{
        (A(1, 1) * b[0] - A(0, 1) * b[1]) * invDet,
        (A(0, 0) * b[1] - A(1, 0) * b[0]) * invDet
    };
}

Vec3 Solve3x3(const Mat33& A, const Vec3& b) {
    // Use matrix inverse (already properly implemented in Matrix.h)
    if (!A.IsInvertible(SOLVER_SINGULAR_THRESHOLD)) {
        return Vec3::Zero();
    }
    Mat33 Ainv = A.Inverse();
    return Ainv * b;
}

Vec4 Solve4x4(const Mat44& A, const Vec4& b) {
    Mat44 Ainv = A.Inverse();
    if (!A.IsInvertible(SOLVER_SINGULAR_THRESHOLD)) {
        return Vec4::Zero();
    }
    return Ainv * b;
}

bool IsSolvable2x2(const Mat22& A, double tolerance) {
    double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    return std::abs(det) > tolerance;
}

bool IsSolvable3x3(const Mat33& A, double tolerance) {
    return std::abs(A.Determinant()) > tolerance;
}

bool IsSolvable4x4(const Mat44& A, double tolerance) {
    return std::abs(A.Determinant()) > tolerance;
}

// =============================================================================
// Triangular Solvers
// =============================================================================

VecX SolveLowerTriangular(const MatX& L, const VecX& b, bool unitDiagonal) {
    int n = L.Rows();
    if (n != L.Cols() || n != b.Size()) {
        throw InvalidArgumentException("SolveLowerTriangular: dimension mismatch");
    }

    VecX x(n);
    for (int i = 0; i < n; ++i) {
        double sum = b[i];
        for (int j = 0; j < i; ++j) {
            sum -= L(i, j) * x[j];
        }
        if (unitDiagonal) {
            x[i] = sum;
        } else {
            if (std::abs(L(i, i)) < SOLVER_SINGULAR_THRESHOLD) {
                x[i] = 0.0;  // Singular, set to zero
            } else {
                x[i] = sum / L(i, i);
            }
        }
    }
    return x;
}

VecX SolveUpperTriangular(const MatX& U, const VecX& b) {
    int n = U.Rows();
    if (n != U.Cols() || n != b.Size()) {
        throw InvalidArgumentException("SolveUpperTriangular: dimension mismatch");
    }

    VecX x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= U(i, j) * x[j];
        }
        if (std::abs(U(i, i)) < SOLVER_SINGULAR_THRESHOLD) {
            x[i] = 0.0;  // Singular, set to zero
        } else {
            x[i] = sum / U(i, i);
        }
    }
    return x;
}

VecX SolveTridiagonal(const VecX& lower, const VecX& diag, const VecX& upper, const VecX& b) {
    int n = diag.Size();
    if (n < 1) {
        return VecX();
    }
    if (lower.Size() != n - 1 || upper.Size() != n - 1 || b.Size() != n) {
        throw InvalidArgumentException("SolveTridiagonal: dimension mismatch");
    }

    // Thomas algorithm
    VecX c_prime(n);
    VecX d_prime(n);

    // Forward elimination
    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = b[0] / diag[0];

    for (int i = 1; i < n; ++i) {
        double m = diag[i] - lower[i - 1] * c_prime[i - 1];
        if (std::abs(m) < SOLVER_SINGULAR_THRESHOLD) {
            // Nearly singular
            return VecX::Zero(n);
        }
        if (i < n - 1) {
            c_prime[i] = upper[i] / m;
        }
        d_prime[i] = (b[i] - lower[i - 1] * d_prime[i - 1]) / m;
    }

    // Back substitution
    VecX x(n);
    x[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    return x;
}

// =============================================================================
// LU Decomposition
// =============================================================================

int LU_DecomposeInPlace(MatX& A, std::vector<int>& P) {
    int n = A.Rows();
    if (n != A.Cols()) {
        throw InvalidArgumentException("LU_DecomposeInPlace: matrix must be square");
    }

    P.resize(n);
    for (int i = 0; i < n; ++i) {
        P[i] = i;
    }

    int sign = 1;

    for (int k = 0; k < n; ++k) {
        // Find pivot
        double maxVal = 0.0;
        int maxIdx = k;
        for (int i = k; i < n; ++i) {
            double absVal = std::abs(A(i, k));
            if (absVal > maxVal) {
                maxVal = absVal;
                maxIdx = i;
            }
        }

        // Check for singularity
        if (maxVal < SOLVER_SINGULAR_THRESHOLD) {
            return 0;  // Singular matrix
        }

        // Swap rows if needed
        if (maxIdx != k) {
            std::swap(P[k], P[maxIdx]);
            for (int j = 0; j < n; ++j) {
                std::swap(A(k, j), A(maxIdx, j));
            }
            sign = -sign;
        }

        // Elimination
        for (int i = k + 1; i < n; ++i) {
            A(i, k) /= A(k, k);
            for (int j = k + 1; j < n; ++j) {
                A(i, j) -= A(i, k) * A(k, j);
            }
        }
    }

    return sign;
}

LUResult LU_Decompose(const MatX& A) {
    LUResult result;
    int n = A.Rows();

    if (n != A.Cols()) {
        result.valid = false;
        return result;
    }

    if (n == 0) {
        result.valid = true;
        result.sign = 1;
        return result;
    }

    // Copy A to working matrix
    MatX work = A;
    result.sign = LU_DecomposeInPlace(work, result.P);

    if (result.sign == 0) {
        result.valid = false;
        return result;
    }

    // Extract L and U from packed form
    result.L = MatX::Identity(n);
    result.U = MatX::Zero(n, n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i > j) {
                result.L(i, j) = work(i, j);
            } else {
                result.U(i, j) = work(i, j);
            }
        }
    }

    result.valid = true;
    return result;
}

VecX SolveFromLU(const LUResult& lu, const VecX& b) {
    if (!lu.valid) {
        return VecX::Zero(b.Size());
    }

    int n = lu.L.Rows();
    if (b.Size() != n) {
        throw InvalidArgumentException("SolveFromLU: dimension mismatch");
    }

    // Apply permutation to b
    VecX pb(n);
    for (int i = 0; i < n; ++i) {
        pb[i] = b[lu.P[i]];
    }

    // Solve Ly = Pb (forward substitution)
    VecX y = SolveLowerTriangular(lu.L, pb, true);

    // Solve Ux = y (back substitution)
    return SolveUpperTriangular(lu.U, y);
}

MatX SolveFromLU(const LUResult& lu, const MatX& B) {
    if (!lu.valid) {
        return MatX::Zero(B.Rows(), B.Cols());
    }

    int n = lu.L.Rows();
    int k = B.Cols();

    if (B.Rows() != n) {
        throw InvalidArgumentException("SolveFromLU: dimension mismatch");
    }

    MatX X(n, k);
    for (int j = 0; j < k; ++j) {
        VecX x = SolveFromLU(lu, B.Col(j));
        X.SetCol(j, x);
    }
    return X;
}

VecX SolveLU(const MatX& A, const VecX& b) {
    LUResult lu = LU_Decompose(A);
    return SolveFromLU(lu, b);
}

// =============================================================================
// Cholesky Decomposition
// =============================================================================

CholeskyResult Cholesky_Decompose(const MatX& A) {
    CholeskyResult result;
    int n = A.Rows();

    if (n != A.Cols()) {
        result.valid = false;
        return result;
    }

    if (n == 0) {
        result.valid = true;
        return result;
    }

    result.L = MatX::Zero(n, n);

    for (int j = 0; j < n; ++j) {
        // Diagonal element
        double sum = A(j, j);
        for (int k = 0; k < j; ++k) {
            sum -= result.L(j, k) * result.L(j, k);
        }

        if (sum <= 0.0) {
            // Not positive definite
            result.valid = false;
            return result;
        }

        result.L(j, j) = std::sqrt(sum);

        // Off-diagonal elements
        for (int i = j + 1; i < n; ++i) {
            sum = A(i, j);
            for (int k = 0; k < j; ++k) {
                sum -= result.L(i, k) * result.L(j, k);
            }
            result.L(i, j) = sum / result.L(j, j);
        }
    }

    result.valid = true;
    return result;
}

VecX SolveFromCholesky(const CholeskyResult& chol, const VecX& b) {
    if (!chol.valid) {
        return VecX::Zero(b.Size());
    }

    int n = chol.L.Rows();
    if (b.Size() != n) {
        throw InvalidArgumentException("SolveFromCholesky: dimension mismatch");
    }

    // Solve Ly = b (forward substitution)
    VecX y = SolveLowerTriangular(chol.L, b);

    // Solve L^T x = y (back substitution)
    MatX LT = chol.L.Transpose();
    return SolveUpperTriangular(LT, y);
}

VecX SolveCholesky(const MatX& A, const VecX& b) {
    CholeskyResult chol = Cholesky_Decompose(A);
    return SolveFromCholesky(chol, b);
}

// =============================================================================
// QR Decomposition (Householder)
// =============================================================================

QRResult QR_Decompose(const MatX& A) {
    QRResult result;
    int m = A.Rows();
    int n = A.Cols();

    if (m == 0 || n == 0) {
        result.valid = true;
        result.thinQR = false;
        return result;
    }

    // Copy A to R (will be modified)
    MatX R = A;

    // Storage for Householder vectors
    std::vector<VecX> householder;
    householder.reserve(std::min(m, n));

    int minMN = std::min(m, n);

    for (int k = 0; k < minMN; ++k) {
        // Extract column k below diagonal
        VecX v(m - k);
        for (int i = k; i < m; ++i) {
            v[i - k] = R(i, k);
        }

        // Compute Householder vector
        double norm = v.Norm();
        if (norm > MATRIX_EPSILON) {
            // Choose sign to avoid cancellation
            double alpha = -Sign(v[0]) * norm;
            v[0] -= alpha;
            v.Normalize();

            // Apply Householder transformation to R
            // R = R - 2 * v * (v^T * R)
            for (int j = k; j < n; ++j) {
                double dot = 0.0;
                for (int i = k; i < m; ++i) {
                    dot += v[i - k] * R(i, j);
                }
                for (int i = k; i < m; ++i) {
                    R(i, j) -= 2.0 * v[i - k] * dot;
                }
            }
        }

        householder.push_back(v);
    }

    // Build Q by applying Householder transformations in reverse
    result.Q = MatX::Identity(m);
    for (int k = minMN - 1; k >= 0; --k) {
        const VecX& v = householder[k];
        // Apply H = I - 2*v*v^T to Q
        for (int j = 0; j < m; ++j) {
            double dot = 0.0;
            for (int i = k; i < m; ++i) {
                dot += v[i - k] * result.Q(i, j);
            }
            for (int i = k; i < m; ++i) {
                result.Q(i, j) -= 2.0 * v[i - k] * dot;
            }
        }
    }

    result.R = R;
    result.valid = true;
    result.thinQR = false;

    return result;
}

QRResult QR_DecomposeThin(const MatX& A) {
    QRResult result;
    int m = A.Rows();
    int n = A.Cols();

    if (m < n) {
        throw InvalidArgumentException("QR_DecomposeThin: requires m >= n");
    }

    if (m == 0 || n == 0) {
        result.valid = true;
        result.thinQR = true;
        return result;
    }

    // Copy A to R (will be modified)
    MatX R = A;

    // Storage for Householder vectors
    std::vector<VecX> householder;
    householder.reserve(n);

    for (int k = 0; k < n; ++k) {
        // Extract column k below diagonal
        VecX v(m - k);
        for (int i = k; i < m; ++i) {
            v[i - k] = R(i, k);
        }

        // Compute Householder vector
        double norm = v.Norm();
        if (norm > MATRIX_EPSILON) {
            double alpha = -Sign(v[0]) * norm;
            v[0] -= alpha;
            v.Normalize();

            // Apply to R
            for (int j = k; j < n; ++j) {
                double dot = 0.0;
                for (int i = k; i < m; ++i) {
                    dot += v[i - k] * R(i, j);
                }
                for (int i = k; i < m; ++i) {
                    R(i, j) -= 2.0 * v[i - k] * dot;
                }
            }
        }

        householder.push_back(v);
    }

    // Build Q_thin (m x n)
    result.Q = MatX::Zero(m, n);
    for (int j = 0; j < n; ++j) {
        result.Q(j, j) = 1.0;
    }

    for (int k = n - 1; k >= 0; --k) {
        const VecX& v = householder[k];
        for (int j = 0; j < n; ++j) {
            double dot = 0.0;
            for (int i = k; i < m; ++i) {
                dot += v[i - k] * result.Q(i, j);
            }
            for (int i = k; i < m; ++i) {
                result.Q(i, j) -= 2.0 * v[i - k] * dot;
            }
        }
    }

    // Extract upper part of R (n x n)
    result.R = R.Block(0, 0, n, n);
    result.valid = true;
    result.thinQR = true;

    return result;
}

VecX SolveFromQR(const QRResult& qr, const VecX& b) {
    if (!qr.valid) {
        return VecX::Zero(b.Size());
    }

    // Compute Q^T * b
    MatX QT = qr.Q.Transpose();
    VecX y = QT * b;

    // Solve R * x = y (back substitution)
    int n = qr.R.Cols();
    if (qr.thinQR) {
        return SolveUpperTriangular(qr.R, y.Segment(0, n));
    } else {
        return SolveUpperTriangular(qr.R.Block(0, 0, n, n), y.Segment(0, n));
    }
}

VecX SolveLeastSquares(const MatX& A, const VecX& b) {
    if (A.Rows() < A.Cols()) {
        // Underdetermined system - use SVD
        return SolveSVD(A, b);
    }

    QRResult qr = QR_DecomposeThin(A);
    return SolveFromQR(qr, b);
}

VecX SolveLeastSquaresNormal(const MatX& A, const VecX& b) {
    // Solve (A^T * A) * x = A^T * b
    MatX ATA = A.Transpose() * A;
    VecX ATb = A.Transpose() * b;
    return SolveCholesky(ATA, ATb);
}

// =============================================================================
// SVD Decomposition
// =============================================================================

// One-sided Jacobi SVD algorithm - applies rotations to columns of A
// Works directly with A to preserve sign information
namespace {

/// Apply Jacobi rotation to columns p and q of matrix M (from right)
void ApplyJacobiRight(MatX& M, int p, int q, double c, double s) {
    int m = M.Rows();
    for (int i = 0; i < m; ++i) {
        double mp = M(i, p);
        double mq = M(i, q);
        M(i, p) = c * mp + s * mq;
        M(i, q) = -s * mp + c * mq;
    }
}

} // anonymous namespace

SVDResult SVD_Decompose(const MatX& A, bool fullMatrices) {
    (void)fullMatrices;  // Not implemented yet

    SVDResult result;
    int m = A.Rows();
    int n = A.Cols();

    if (m == 0 || n == 0) {
        result.valid = true;
        result.rank = 0;
        return result;
    }

    // For wide matrices, transpose and swap U/V at the end
    bool transpose = m < n;
    MatX B = transpose ? A.Transpose() : A;

    if (transpose) {
        std::swap(m, n);
    }

    // Now B is m x n with m >= n
    // We will apply Jacobi rotations to columns of B
    MatX V = MatX::Identity(n);

    // One-sided Jacobi: rotate columns i and j of B to make them orthogonal
    // Use tighter tolerance and more sweeps for better convergence
    const int maxSweeps = 100;
    const double tol = 1e-14;

    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        double maxOffDiag = 0.0;
        double maxDiag = 0.0;

        // Sweep through all column pairs
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                // Compute elements of B^T B for columns i and j
                double bii = 0.0, bij = 0.0, bjj = 0.0;
                for (int k = 0; k < m; ++k) {
                    bii += B(k, i) * B(k, i);
                    bij += B(k, i) * B(k, j);
                    bjj += B(k, j) * B(k, j);
                }

                maxDiag = std::max(maxDiag, std::max(bii, bjj));

                // Track maximum off-diagonal for convergence check
                if (std::abs(bij) > maxOffDiag) {
                    maxOffDiag = std::abs(bij);
                }

                // Skip if already orthogonal (relative tolerance)
                if (std::abs(bij) < tol * std::sqrt(bii * bjj + 1e-300)) {
                    continue;
                }

                // Compute Jacobi rotation angle
                // tan(2*theta) = 2*bij / (bii - bjj)
                // tau = cot(2*theta) = (bii - bjj) / (2*bij)
                double tau = (bii - bjj) / (2.0 * bij);
                double t;
                // t = tan(theta), choose the smaller root for numerical stability
                if (tau >= 0.0) {
                    t = 1.0 / (tau + std::sqrt(1.0 + tau * tau));
                } else {
                    t = 1.0 / (tau - std::sqrt(1.0 + tau * tau));
                }
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;

                // Apply rotation to B and V
                ApplyJacobiRight(B, i, j, c, s);
                ApplyJacobiRight(V, i, j, c, s);
            }
        }

        // Check convergence (relative to diagonal)
        if (maxOffDiag < tol * maxDiag || maxOffDiag < 1e-300) {
            break;
        }
    }

    // Extract singular values and normalize U columns
    result.S = VecX(n);
    MatX U(m, n);

    // First pass: handle non-zero columns
    std::vector<int> zeroCols;
    for (int j = 0; j < n; ++j) {
        double norm = 0.0;
        for (int i = 0; i < m; ++i) {
            norm += B(i, j) * B(i, j);
        }
        norm = std::sqrt(norm);
        result.S[j] = norm;

        if (norm > MATRIX_EPSILON) {
            for (int i = 0; i < m; ++i) {
                U(i, j) = B(i, j) / norm;
            }
        } else {
            // Mark as zero column, will fill in second pass
            zeroCols.push_back(j);
            for (int i = 0; i < m; ++i) {
                U(i, j) = 0.0;
            }
        }
    }

    // Second pass: generate orthonormal vectors for zero columns
    // Use Gram-Schmidt on standard basis vectors
    int basisIdx = 0;
    for (int zeroCol : zeroCols) {
        // Try standard basis vectors until we find one that's not in the span
        while (basisIdx < m) {
            // Start with e_{basisIdx}
            VecX v = VecX::Zero(m);
            v[basisIdx] = 1.0;
            ++basisIdx;

            // Orthogonalize against all existing U columns
            for (int k = 0; k < n; ++k) {
                if (k == zeroCol) continue;
                double dot = 0.0;
                for (int i = 0; i < m; ++i) {
                    dot += v[i] * U(i, k);
                }
                for (int i = 0; i < m; ++i) {
                    v[i] -= dot * U(i, k);
                }
            }

            // Also orthogonalize against earlier zero columns we've filled
            for (int k : zeroCols) {
                if (k >= zeroCol) break;
                double dot = 0.0;
                for (int i = 0; i < m; ++i) {
                    dot += v[i] * U(i, k);
                }
                for (int i = 0; i < m; ++i) {
                    v[i] -= dot * U(i, k);
                }
            }

            double norm = v.Norm();
            if (norm > MATRIX_EPSILON) {
                // Found a valid orthogonal vector
                for (int i = 0; i < m; ++i) {
                    U(i, zeroCol) = v[i] / norm;
                }
                break;
            }
            // This basis vector was in the span, try next one
        }
    }

    // Sort singular values (descending) and reorder U, V
    for (int i = 0; i < n - 1; ++i) {
        int maxIdx = i;
        for (int j = i + 1; j < n; ++j) {
            if (result.S[j] > result.S[maxIdx]) {
                maxIdx = j;
            }
        }
        if (maxIdx != i) {
            std::swap(result.S[i], result.S[maxIdx]);
            for (int k = 0; k < m; ++k) {
                std::swap(U(k, i), U(k, maxIdx));
            }
            for (int k = 0; k < n; ++k) {
                std::swap(V(k, i), V(k, maxIdx));
            }
        }
    }

    // Handle transpose: if original A was wide (m < n), swap U and V
    if (transpose) {
        result.U = V;
        result.V = U;
        std::swap(m, n);
    } else {
        result.U = U;
        result.V = V;
    }

    // Compute rank
    result.rank = 0;
    for (int i = 0; i < result.S.Size(); ++i) {
        if (result.S[i] > SOLVER_RANK_TOLERANCE) {
            ++result.rank;
        }
    }

    result.valid = true;
    return result;
}

VecX SolveFromSVD(const SVDResult& svd, const VecX& b, double tolerance) {
    if (!svd.valid) {
        return VecX::Zero(b.Size());
    }

    int m = svd.U.Rows();
    int n = svd.V.Rows();
    int k = svd.S.Size();

    if (b.Size() != m) {
        throw InvalidArgumentException("SolveFromSVD: dimension mismatch");
    }

    // x = V * S^{-1} * U^T * b
    // For singular values below tolerance, set inverse to 0

    // Compute U^T * b
    VecX UTb(k);
    for (int i = 0; i < k; ++i) {
        double sum = 0.0;
        for (int j = 0; j < m; ++j) {
            sum += svd.U(j, i) * b[j];
        }
        UTb[i] = sum;
    }

    // Apply S^{-1}
    VecX Sinv_UTb(k);
    for (int i = 0; i < k; ++i) {
        if (svd.S[i] > tolerance) {
            Sinv_UTb[i] = UTb[i] / svd.S[i];
        } else {
            Sinv_UTb[i] = 0.0;
        }
    }

    // Compute V * Sinv_UTb
    VecX x(n);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < k; ++j) {
            sum += svd.V(i, j) * Sinv_UTb[j];
        }
        x[i] = sum;
    }

    return x;
}

VecX SolveSVD(const MatX& A, const VecX& b, double tolerance) {
    SVDResult svd = SVD_Decompose(A);
    return SolveFromSVD(svd, b, tolerance);
}

VecX SolveHomogeneous(const MatX& A) {
    int m = A.Rows();
    int n = A.Cols();

    if (m == 0 || n == 0) {
        return VecX();
    }

    // For wide matrices (m < n), the thin SVD doesn't give us the full null space.
    // We need to handle this specially by computing A^T * A and finding its
    // smallest eigenvector, or using the full null space computation.
    if (m < n) {
        // Compute A^T * A (n x n) and find its smallest eigenvector
        // This is equivalent to finding the right singular vector for the smallest
        // singular value in the full SVD.
        MatX ATA = A.Transpose() * A;

        // Use SVD on ATA to find its eigenvectors
        // The smallest eigenvalue's eigenvector is the null space of A
        SVDResult svd = SVD_Decompose(ATA);
        if (!svd.valid || svd.V.Cols() == 0) {
            return VecX();
        }

        // For a symmetric matrix like A^T*A, eigenvectors are in V
        // The last column corresponds to the smallest eigenvalue
        int lastCol = svd.V.Cols() - 1;
        VecX x(n);
        for (int i = 0; i < n; ++i) {
            x[i] = svd.V(i, lastCol);
        }
        return x;
    }

    // For tall or square matrices (m >= n), thin SVD gives us enough columns of V
    SVDResult svd = SVD_Decompose(A);
    if (!svd.valid || svd.V.Cols() == 0) {
        return VecX();
    }

    // Return last column of V (smallest singular value)
    int lastCol = svd.V.Cols() - 1;

    VecX x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = svd.V(i, lastCol);
    }

    return x;
}

// =============================================================================
// Utility Functions
// =============================================================================

double ComputeConditionNumber(const MatX& A) {
    SVDResult svd = SVD_Decompose(A);
    if (!svd.valid || svd.S.Size() == 0) {
        return std::numeric_limits<double>::infinity();
    }

    // Find largest and smallest non-zero singular values
    double sMax = svd.S[0];
    double sMin = svd.S[svd.S.Size() - 1];

    if (sMin < SOLVER_RANK_TOLERANCE) {
        return std::numeric_limits<double>::infinity();
    }

    return sMax / sMin;
}

double EstimateConditionNumber(const MatX& A) {
    // 1-norm condition number estimation
    // Faster than full SVD but approximate
    int n = A.Rows();
    if (n != A.Cols()) {
        throw InvalidArgumentException("EstimateConditionNumber: requires square matrix");
    }

    // Compute 1-norm of A
    double normA = 0.0;
    for (int j = 0; j < n; ++j) {
        double colSum = 0.0;
        for (int i = 0; i < n; ++i) {
            colSum += std::abs(A(i, j));
        }
        normA = std::max(normA, colSum);
    }

    // LU decomposition
    LUResult lu = LU_Decompose(A);
    if (!lu.valid) {
        return std::numeric_limits<double>::infinity();
    }

    // Estimate 1-norm of A^{-1} using iterative method
    VecX x(n, 1.0 / n);
    VecX y(n);

    for (int iter = 0; iter < 5; ++iter) {
        // y = A^{-1} * x
        y = SolveFromLU(lu, x);

        // Normalize
        double norm = 0.0;
        for (int i = 0; i < n; ++i) {
            norm += std::abs(y[i]);
        }
        if (norm < MATRIX_EPSILON) break;

        for (int i = 0; i < n; ++i) {
            x[i] = Sign(y[i]);
        }
    }

    double normAinv = 0.0;
    for (int i = 0; i < n; ++i) {
        normAinv = std::max(normAinv, std::abs(y[i]));
    }

    return normA * normAinv;
}

int ComputeRank(const MatX& A, double tolerance) {
    SVDResult svd = SVD_Decompose(A);
    if (!svd.valid) {
        return 0;
    }

    int rank = 0;
    for (int i = 0; i < svd.S.Size(); ++i) {
        if (svd.S[i] > tolerance) {
            ++rank;
        }
    }
    return rank;
}

MatX PseudoInverse(const MatX& A, double tolerance) {
    SVDResult svd = SVD_Decompose(A);
    if (!svd.valid) {
        return MatX::Zero(A.Cols(), A.Rows());
    }

    int m = svd.U.Rows();
    int n = svd.V.Rows();
    int k = svd.S.Size();

    // A^+ = V * S^+ * U^T
    // S^+[i,i] = 1/S[i] if S[i] > tolerance, else 0

    // Compute S^+ * U^T
    MatX SinvUT(k, m);
    for (int i = 0; i < k; ++i) {
        double sinv = (svd.S[i] > tolerance) ? (1.0 / svd.S[i]) : 0.0;
        for (int j = 0; j < m; ++j) {
            SinvUT(i, j) = sinv * svd.U(j, i);
        }
    }

    // Compute V * SinvUT
    MatX Ainv(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            double sum = 0.0;
            for (int l = 0; l < k; ++l) {
                sum += svd.V(i, l) * SinvUT(l, j);
            }
            Ainv(i, j) = sum;
        }
    }

    return Ainv;
}

double ComputeResidual(const MatX& A, const VecX& x, const VecX& b) {
    VecX Ax = A * x;
    VecX r = Ax - b;
    return r.Norm();
}

double ComputeRelativeResidual(const MatX& A, const VecX& x, const VecX& b) {
    double bNorm = b.Norm();
    if (bNorm < MATRIX_EPSILON) {
        return 0.0;
    }
    return ComputeResidual(A, x, b) / bNorm;
}

MatX ComputeNullSpace(const MatX& A, double tolerance) {
    int m = A.Rows();
    int n = A.Cols();

    SVDResult svd = SVD_Decompose(A);
    if (!svd.valid) {
        return MatX();
    }

    // Count non-zero singular values (rank)
    int rank = 0;
    for (int i = 0; i < svd.S.Size(); ++i) {
        if (svd.S[i] > tolerance) {
            ++rank;
        }
    }

    // Nullity = n - rank
    int nullity = n - rank;

    if (nullity == 0) {
        return MatX();  // Empty null space
    }

    // For wide matrices (m < n), V is n x m (thin SVD)
    // The last (n - rank) columns of the full V are the null space
    // But we only have the first m columns of V
    // We need to construct the remaining n - m basis vectors orthogonal to the range

    if (svd.V.Cols() >= n) {
        // Full V available, extract last nullity columns
        MatX nullSpace(n, nullity);
        for (int i = 0; i < nullity; ++i) {
            int col = n - nullity + i;
            for (int j = 0; j < n; ++j) {
                nullSpace(j, i) = svd.V(j, col);
            }
        }
        return nullSpace;
    } else {
        // Thin SVD: V is n x min(m,n)
        // Need to extend V to a full orthonormal basis
        // The null space consists of:
        // 1. Columns of V corresponding to zero singular values
        // 2. Any vectors orthogonal to all columns of V

        int numVCols = svd.V.Cols();

        // Collect null space vectors from V (zero singular values)
        int nullFromS = 0;
        for (int i = 0; i < svd.S.Size(); ++i) {
            if (svd.S[i] <= tolerance) {
                ++nullFromS;
            }
        }

        // Additional null space dimension from missing columns
        int nullFromMissing = n - numVCols;

        MatX nullSpace(n, nullity);
        int outCol = 0;

        // Copy columns corresponding to zero singular values
        for (int i = svd.S.Size() - nullFromS; i < svd.S.Size(); ++i) {
            for (int j = 0; j < n; ++j) {
                nullSpace(j, outCol) = svd.V(j, i);
            }
            ++outCol;
        }

        // Construct orthogonal complement for missing dimensions
        // Use Gram-Schmidt on standard basis vectors
        for (int trial = 0; trial < n && outCol < nullity; ++trial) {
            VecX v = VecX::Zero(n);
            v[trial] = 1.0;

            // Orthogonalize against columns of V
            for (int k = 0; k < numVCols; ++k) {
                double dot = 0.0;
                for (int i = 0; i < n; ++i) {
                    dot += v[i] * svd.V(i, k);
                }
                for (int i = 0; i < n; ++i) {
                    v[i] -= dot * svd.V(i, k);
                }
            }

            // Orthogonalize against already found null space vectors
            for (int k = 0; k < outCol; ++k) {
                double dot = 0.0;
                for (int i = 0; i < n; ++i) {
                    dot += v[i] * nullSpace(i, k);
                }
                for (int i = 0; i < n; ++i) {
                    v[i] -= dot * nullSpace(i, k);
                }
            }

            double norm = v.Norm();
            if (norm > tolerance) {
                for (int i = 0; i < n; ++i) {
                    nullSpace(i, outCol) = v[i] / norm;
                }
                ++outCol;
            }
        }

        return nullSpace;
    }
}

int ComputeNullity(const MatX& A, double tolerance) {
    int n = A.Cols();
    int rank = ComputeRank(A, tolerance);
    return n - rank;
}

} // namespace Qi::Vision::Internal
