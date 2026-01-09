/**
 * @file Eigen.cpp
 * @brief Eigenvalue decomposition implementation
 */

#include <QiVision/Internal/Eigen.h>
#include <QiVision/Internal/Solver.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

namespace Qi::Vision::Internal {

namespace {
    constexpr double EPSILON = 1e-15;
    constexpr double PI = 3.14159265358979323846;

    // Helper: Compute off-diagonal norm for Jacobi convergence check
    double OffDiagonalNorm(const MatX& A) {
        double sum = 0.0;
        int n = A.Rows();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                sum += A(i, j) * A(i, j);
            }
        }
        return std::sqrt(2.0 * sum);
    }

    // Helper: Apply Jacobi rotation to matrix A
    // Zeroes out A(p,q) and A(q,p) for symmetric A
    void JacobiRotation(MatX& A, MatX& V, int p, int q) {
        int n = A.Rows();

        double app = A(p, p);
        double aqq = A(q, q);
        double apq = A(p, q);

        if (std::abs(apq) < EPSILON) {
            return;
        }

        // Compute rotation angle
        double tau = (aqq - app) / (2.0 * apq);
        double t;
        if (tau >= 0) {
            t = 1.0 / (tau + std::sqrt(1.0 + tau * tau));
        } else {
            t = -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
        }

        double c = 1.0 / std::sqrt(1.0 + t * t);
        double s = t * c;

        // Update A
        A(p, p) = app - t * apq;
        A(q, q) = aqq + t * apq;
        A(p, q) = 0.0;
        A(q, p) = 0.0;

        for (int i = 0; i < n; ++i) {
            if (i != p && i != q) {
                double aip = A(i, p);
                double aiq = A(i, q);
                A(i, p) = c * aip - s * aiq;
                A(p, i) = A(i, p);
                A(i, q) = s * aip + c * aiq;
                A(q, i) = A(i, q);
            }
        }

        // Update eigenvector matrix V
        for (int i = 0; i < n; ++i) {
            double vip = V(i, p);
            double viq = V(i, q);
            V(i, p) = c * vip - s * viq;
            V(i, q) = s * vip + c * viq;
        }
    }

    // Helper: Find largest off-diagonal element
    void FindMaxOffDiagonal(const MatX& A, int& p, int& q, double& maxVal) {
        int n = A.Rows();
        maxVal = 0.0;
        p = 0;
        q = 1;

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double val = std::abs(A(i, j));
                if (val > maxVal) {
                    maxVal = val;
                    p = i;
                    q = j;
                }
            }
        }
    }

    // Helper: Cubic root (handles negative numbers)
    double CubeRoot(double x) {
        if (x >= 0) {
            return std::pow(x, 1.0 / 3.0);
        } else {
            return -std::pow(-x, 1.0 / 3.0);
        }
    }

    // Helper: Solve cubic equation x^3 + a*x^2 + b*x + c = 0
    // Returns real roots sorted descending
    std::vector<double> SolveCubic(double a, double b, double c) {
        std::vector<double> roots;

        // Convert to depressed cubic t^3 + pt + q = 0 with x = t - a/3
        double p = b - a * a / 3.0;
        double q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c;

        // Discriminant
        double D = q * q / 4.0 + p * p * p / 27.0;
        double shift = -a / 3.0;

        if (D > EPSILON) {
            // One real root
            double sqrtD = std::sqrt(D);
            double u = CubeRoot(-q / 2.0 + sqrtD);
            double v = CubeRoot(-q / 2.0 - sqrtD);
            roots.push_back(u + v + shift);
        } else if (D < -EPSILON) {
            // Three distinct real roots (Cardano's formula with trigonometric method)
            double r = std::sqrt(-p * p * p / 27.0);
            double theta = std::acos(-q / (2.0 * r)) / 3.0;
            double m = 2.0 * CubeRoot(r);

            roots.push_back(m * std::cos(theta) + shift);
            roots.push_back(m * std::cos(theta + 2.0 * PI / 3.0) + shift);
            roots.push_back(m * std::cos(theta + 4.0 * PI / 3.0) + shift);
        } else {
            // Multiple root
            double u = CubeRoot(-q / 2.0);
            roots.push_back(2.0 * u + shift);
            roots.push_back(-u + shift);
        }

        std::sort(roots.begin(), roots.end(), std::greater<double>());
        return roots;
    }

    // Helper: Normalize vector
    VecX NormalizeVec(const VecX& v) {
        double norm = 0.0;
        for (int i = 0; i < v.Size(); ++i) {
            norm += v[i] * v[i];
        }
        norm = std::sqrt(norm);

        if (norm < EPSILON) {
            return v;
        }

        VecX result(v.Size());
        for (int i = 0; i < v.Size(); ++i) {
            result[i] = v[i] / norm;
        }
        return result;
    }

    // Helper: Vector norm
    double VecNorm(const VecX& v) {
        double sum = 0.0;
        for (int i = 0; i < v.Size(); ++i) {
            sum += v[i] * v[i];
        }
        return std::sqrt(sum);
    }
}

// =============================================================================
// 2x2 Eigenvalue Decomposition (Closed-form)
// =============================================================================

Eigen2x2Result EigenSymmetric2x2(const Mat22& A) {
    Eigen2x2Result result;
    result.valid = true;
    result.isReal = true;

    double a = A(0, 0);
    double b = A(0, 1);  // = A(1, 0) for symmetric
    double c = A(1, 1);

    // Eigenvalues from characteristic polynomial: lambda^2 - (a+c)*lambda + (ac - b^2) = 0
    double trace = a + c;
    double det = a * c - b * b;

    double discriminant = trace * trace - 4.0 * det;

    if (discriminant < 0) {
        // Should not happen for symmetric matrix
        result.valid = false;
        return result;
    }

    double sqrtD = std::sqrt(discriminant);
    result.lambda1 = (trace + sqrtD) / 2.0;
    result.lambda2 = (trace - sqrtD) / 2.0;

    // Ensure lambda1 has larger magnitude
    if (std::abs(result.lambda2) > std::abs(result.lambda1)) {
        std::swap(result.lambda1, result.lambda2);
    }

    // Compute eigenvectors
    // For (A - lambda*I) * v = 0
    if (std::abs(b) > EPSILON) {
        result.v1[0] = b;
        result.v1[1] = result.lambda1 - a;
        result.v2[0] = b;
        result.v2[1] = result.lambda2 - a;
    } else if (std::abs(a - c) > EPSILON) {
        if (a > c) {
            result.v1[0] = 1; result.v1[1] = 0;
            result.v2[0] = 0; result.v2[1] = 1;
        } else {
            result.v1[0] = 0; result.v1[1] = 1;
            result.v2[0] = 1; result.v2[1] = 0;
        }
    } else {
        // a == c and b == 0, any orthonormal pair works
        result.v1[0] = 1; result.v1[1] = 0;
        result.v2[0] = 0; result.v2[1] = 1;
    }

    // Normalize eigenvectors
    double norm1 = std::sqrt(result.v1[0] * result.v1[0] + result.v1[1] * result.v1[1]);
    double norm2 = std::sqrt(result.v2[0] * result.v2[0] + result.v2[1] * result.v2[1]);

    if (norm1 > EPSILON) {
        result.v1[0] /= norm1;
        result.v1[1] /= norm1;
    }
    if (norm2 > EPSILON) {
        result.v2[0] /= norm2;
        result.v2[1] /= norm2;
    }

    return result;
}

Eigen2x2Result EigenGeneral2x2(const Mat22& A) {
    Eigen2x2Result result;
    result.valid = true;

    double a = A(0, 0);
    double b = A(0, 1);
    double c = A(1, 0);
    double d = A(1, 1);

    // Eigenvalues from characteristic polynomial: lambda^2 - (a+d)*lambda + (ad - bc) = 0
    double trace = a + d;
    double det = a * d - b * c;

    double discriminant = trace * trace - 4.0 * det;

    if (discriminant >= 0) {
        // Two real eigenvalues
        result.isReal = true;
        double sqrtD = std::sqrt(discriminant);
        result.lambda1 = (trace + sqrtD) / 2.0;
        result.lambda2 = (trace - sqrtD) / 2.0;
        result.imagPart = 0.0;

        // Ensure lambda1 has larger magnitude
        if (std::abs(result.lambda2) > std::abs(result.lambda1)) {
            std::swap(result.lambda1, result.lambda2);
        }

        // Compute eigenvectors: (A - lambda*I) * v = 0
        // v is in null space of [a-lambda, b; c, d-lambda]
        for (int k = 0; k < 2; ++k) {
            double lambda = (k == 0) ? result.lambda1 : result.lambda2;
            Vec2& v = (k == 0) ? result.v1 : result.v2;

            double m11 = a - lambda;
            double m12 = b;
            double m21 = c;
            double m22 = d - lambda;

            // Find non-zero row and use it
            if (std::abs(m11) > EPSILON || std::abs(m12) > EPSILON) {
                if (std::abs(m12) > std::abs(m11)) {
                    v[0] = 1.0; v[1] = -m11 / m12;
                } else if (std::abs(m11) > EPSILON) {
                    v[0] = -m12 / m11; v[1] = 1.0;
                } else {
                    v[0] = 1.0; v[1] = 0.0;
                }
            } else if (std::abs(m21) > EPSILON || std::abs(m22) > EPSILON) {
                if (std::abs(m22) > std::abs(m21)) {
                    v[0] = 1.0; v[1] = -m21 / m22;
                } else {
                    v[0] = -m22 / m21; v[1] = 1.0;
                }
            } else {
                v[0] = 1.0; v[1] = 0.0;
            }

            // Normalize
            double norm = std::sqrt(v[0] * v[0] + v[1] * v[1]);
            if (norm > EPSILON) {
                v[0] /= norm;
                v[1] /= norm;
            }
        }
    } else {
        // Complex conjugate eigenvalues
        result.isReal = false;
        result.lambda1 = trace / 2.0;
        result.lambda2 = trace / 2.0;
        result.imagPart = std::sqrt(-discriminant) / 2.0;

        // For complex eigenvalues, eigenvectors are also complex
        // We store a real basis for the 2D invariant subspace
        result.v1[0] = 1; result.v1[1] = 0;
        result.v2[0] = 0; result.v2[1] = 1;
    }

    return result;
}

std::pair<double, double> Eigenvalues2x2(const Mat22& A) {
    double trace = A(0, 0) + A(1, 1);
    double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    double discriminant = trace * trace - 4.0 * det;

    if (discriminant >= 0) {
        double sqrtD = std::sqrt(discriminant);
        return {(trace + sqrtD) / 2.0, (trace - sqrtD) / 2.0};
    } else {
        // Complex eigenvalues - return real part
        return {trace / 2.0, trace / 2.0};
    }
}

// =============================================================================
// 3x3 Eigenvalue Decomposition (Closed-form)
// =============================================================================

Eigen3x3Result EigenSymmetric3x3(const Mat33& A) {
    Eigen3x3Result result;
    result.valid = true;
    result.allReal = true;

    // For symmetric 3x3, use Cardano's formula
    // Characteristic polynomial: -lambda^3 + I1*lambda^2 - I2*lambda + I3 = 0
    // where I1 = trace, I2 = sum of 2x2 principal minors, I3 = det

    double a = A(0, 0), b = A(0, 1), c = A(0, 2);
    double d = A(1, 1), e = A(1, 2);
    double f = A(2, 2);

    // Invariants
    double I1 = a + d + f;  // trace
    double I2 = a*d + a*f + d*f - b*b - c*c - e*e;  // sum of 2x2 principal minors
    double I3 = a*d*f + 2*b*c*e - a*e*e - d*c*c - f*b*b;  // determinant

    // Solve cubic: lambda^3 - I1*lambda^2 + I2*lambda - I3 = 0
    std::vector<double> roots = SolveCubic(-I1, I2, -I3);

    if (roots.size() < 3) {
        // Handle degenerate case (repeated root)
        while (roots.size() < 3) {
            roots.push_back(roots.back());
        }
    }

    // Sort by magnitude (descending)
    std::sort(roots.begin(), roots.end(), [](double a, double b) {
        return std::abs(a) > std::abs(b);
    });

    result.lambda1 = roots[0];
    result.lambda2 = roots[1];
    result.lambda3 = roots[2];

    // Compute eigenvectors
    // For each eigenvalue, find null space of (A - lambda*I)
    auto computeEigenvector = [&](double lambda) -> Vec3 {
        Mat33 M;
        M(0, 0) = A(0, 0) - lambda; M(0, 1) = A(0, 1); M(0, 2) = A(0, 2);
        M(1, 0) = A(1, 0); M(1, 1) = A(1, 1) - lambda; M(1, 2) = A(1, 2);
        M(2, 0) = A(2, 0); M(2, 1) = A(2, 1); M(2, 2) = A(2, 2) - lambda;

        // Find null vector using cross products of rows
        Vec3 r0{M(0, 0), M(0, 1), M(0, 2)};
        Vec3 r1{M(1, 0), M(1, 1), M(1, 2)};
        Vec3 r2{M(2, 0), M(2, 1), M(2, 2)};

        // Try cross products
        Vec3 v1 = Cross(r0, r1);
        Vec3 v2 = Cross(r0, r2);
        Vec3 v3 = Cross(r1, r2);

        double n1 = v1.Norm();
        double n2 = v2.Norm();
        double n3 = v3.Norm();

        Vec3 v;
        if (n1 >= n2 && n1 >= n3 && n1 > EPSILON) {
            v = v1 * (1.0 / n1);
        } else if (n2 >= n1 && n2 >= n3 && n2 > EPSILON) {
            v = v2 * (1.0 / n2);
        } else if (n3 > EPSILON) {
            v = v3 * (1.0 / n3);
        } else {
            // Degenerate case - return arbitrary unit vector
            v[0] = 1; v[1] = 0; v[2] = 0;
        }

        return v;
    };

    result.v1 = computeEigenvector(result.lambda1);
    result.v2 = computeEigenvector(result.lambda2);
    result.v3 = computeEigenvector(result.lambda3);

    // Ensure orthogonality for repeated eigenvalues
    // Gram-Schmidt orthogonalization
    double dot12 = result.v1.Dot(result.v2);
    if (std::abs(dot12) > 0.1) {
        result.v2 = result.v2 - result.v1 * dot12;
        double norm2 = result.v2.Norm();
        if (norm2 > EPSILON) {
            result.v2 = result.v2 * (1.0 / norm2);
        }
    }

    double dot13 = result.v1.Dot(result.v3);
    double dot23 = result.v2.Dot(result.v3);
    if (std::abs(dot13) > 0.1 || std::abs(dot23) > 0.1) {
        result.v3 = result.v3 - result.v1 * dot13 - result.v2 * dot23;
        double norm3 = result.v3.Norm();
        if (norm3 > EPSILON) {
            result.v3 = result.v3 * (1.0 / norm3);
        }
    }

    return result;
}

Eigen3x3Result EigenGeneral3x3(const Mat33& A) {
    // For general 3x3, we use the characteristic polynomial approach
    // but need to handle potential complex roots

    Eigen3x3Result result;
    result.valid = true;

    // Compute coefficients of characteristic polynomial det(A - lambda*I) = 0
    // -lambda^3 + trace*lambda^2 - (sum of 2x2 minors)*lambda + det = 0

    double trace = A(0, 0) + A(1, 1) + A(2, 2);

    double m00 = A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
    double m11 = A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0);
    double m22 = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    double sumMinors = m00 + m11 + m22;

    double det = A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1))
               - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0))
               + A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));

    // Solve: lambda^3 - trace*lambda^2 + sumMinors*lambda - det = 0
    std::vector<double> roots = SolveCubic(-trace, sumMinors, -det);

    if (roots.size() >= 3) {
        result.allReal = true;
        std::sort(roots.begin(), roots.end(), [](double a, double b) {
            return std::abs(a) > std::abs(b);
        });
        result.lambda1 = roots[0];
        result.lambda2 = roots[1];
        result.lambda3 = roots[2];
    } else if (roots.size() == 1) {
        result.allReal = false;
        result.lambda1 = roots[0];
        // Complex pair - use Vieta's formulas
        double realSum = trace - roots[0];
        result.lambda2 = realSum / 2.0;
        result.lambda3 = realSum / 2.0;
    }

    // Compute eigenvectors (simplified for real case)
    if (result.allReal) {
        auto computeEigenvector = [&](double lambda) -> Vec3 {
            Mat33 M;
            M(0, 0) = A(0, 0) - lambda; M(0, 1) = A(0, 1); M(0, 2) = A(0, 2);
            M(1, 0) = A(1, 0); M(1, 1) = A(1, 1) - lambda; M(1, 2) = A(1, 2);
            M(2, 0) = A(2, 0); M(2, 1) = A(2, 1); M(2, 2) = A(2, 2) - lambda;

            Vec3 r0{M(0, 0), M(0, 1), M(0, 2)};
            Vec3 r1{M(1, 0), M(1, 1), M(1, 2)};
            Vec3 r2{M(2, 0), M(2, 1), M(2, 2)};

            Vec3 v1 = Cross(r0, r1);
            Vec3 v2 = Cross(r0, r2);
            Vec3 v3 = Cross(r1, r2);

            double n1 = v1.Norm();
            double n2 = v2.Norm();
            double n3 = v3.Norm();

            if (n1 >= n2 && n1 >= n3 && n1 > EPSILON) {
                return v1 * (1.0 / n1);
            } else if (n2 >= n1 && n2 >= n3 && n2 > EPSILON) {
                return v2 * (1.0 / n2);
            } else if (n3 > EPSILON) {
                return v3 * (1.0 / n3);
            }
            Vec3 unit;
            unit[0] = 1; unit[1] = 0; unit[2] = 0;
            return unit;
        };

        result.v1 = computeEigenvector(result.lambda1);
        result.v2 = computeEigenvector(result.lambda2);
        result.v3 = computeEigenvector(result.lambda3);
    } else {
        // For complex eigenvalues, store basis vectors
        result.v1[0] = 1; result.v1[1] = 0; result.v1[2] = 0;
        result.v2[0] = 0; result.v2[1] = 1; result.v2[2] = 0;
        result.v3[0] = 0; result.v3[1] = 0; result.v3[2] = 1;
    }

    return result;
}

std::array<double, 3> Eigenvalues3x3(const Mat33& A) {
    double trace = A(0, 0) + A(1, 1) + A(2, 2);

    double m00 = A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
    double m11 = A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0);
    double m22 = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    double sumMinors = m00 + m11 + m22;

    double det = A(0, 0) * m00 - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0))
               + A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));

    std::vector<double> roots = SolveCubic(-trace, sumMinors, -det);

    std::array<double, 3> result = {0, 0, 0};
    for (size_t i = 0; i < roots.size() && i < 3; ++i) {
        result[i] = roots[i];
    }

    return result;
}

// =============================================================================
// Jacobi Method for Symmetric Matrices
// =============================================================================

EigenResult EigenSymmetric(const MatX& A, double tolerance, int maxIterations) {
    EigenResult result;

    int n = A.Rows();
    if (n == 0 || A.Cols() != n) {
        return result;
    }

    // Special cases
    if (n == 1) {
        result.eigenvalues = VecX(1);
        result.eigenvalues[0] = A(0, 0);
        result.eigenvectors = MatX(1, 1);
        result.eigenvectors(0, 0) = 1.0;
        result.converged = true;
        result.valid = true;
        result.iterations = 0;
        return result;
    }

    if (n == 2) {
        Mat22 A2;
        A2(0, 0) = A(0, 0); A2(0, 1) = A(0, 1);
        A2(1, 0) = A(1, 0); A2(1, 1) = A(1, 1);
        Eigen2x2Result r2 = EigenSymmetric2x2(A2);

        result.eigenvalues = VecX(2);
        result.eigenvalues[0] = r2.lambda1;
        result.eigenvalues[1] = r2.lambda2;
        result.eigenvectors = MatX(2, 2);
        result.eigenvectors(0, 0) = r2.v1[0]; result.eigenvectors(1, 0) = r2.v1[1];
        result.eigenvectors(0, 1) = r2.v2[0]; result.eigenvectors(1, 1) = r2.v2[1];
        result.converged = true;
        result.valid = r2.valid;
        result.iterations = 0;
        return result;
    }

    if (n == 3) {
        Mat33 A3;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                A3(i, j) = A(i, j);
            }
        }
        Eigen3x3Result r3 = EigenSymmetric3x3(A3);

        result.eigenvalues = VecX(3);
        result.eigenvalues[0] = r3.lambda1;
        result.eigenvalues[1] = r3.lambda2;
        result.eigenvalues[2] = r3.lambda3;
        result.eigenvectors = MatX(3, 3);
        result.eigenvectors(0, 0) = r3.v1[0]; result.eigenvectors(1, 0) = r3.v1[1]; result.eigenvectors(2, 0) = r3.v1[2];
        result.eigenvectors(0, 1) = r3.v2[0]; result.eigenvectors(1, 1) = r3.v2[1]; result.eigenvectors(2, 1) = r3.v2[2];
        result.eigenvectors(0, 2) = r3.v3[0]; result.eigenvectors(1, 2) = r3.v3[1]; result.eigenvectors(2, 2) = r3.v3[2];
        result.converged = true;
        result.valid = r3.valid;
        result.iterations = 0;
        return result;
    }

    // General Jacobi iteration
    MatX D = A;  // Working copy (will become diagonal)
    MatX V = MatX::Identity(n);  // Accumulate rotations

    result.iterations = 0;

    for (int iter = 0; iter < maxIterations; ++iter) {
        result.iterations = iter + 1;

        // Find maximum off-diagonal element
        int p, q;
        double maxOff;
        FindMaxOffDiagonal(D, p, q, maxOff);

        // Check convergence
        if (maxOff < tolerance) {
            result.converged = true;
            break;
        }

        // Apply Jacobi rotation
        JacobiRotation(D, V, p, q);
    }

    // Extract eigenvalues and sort
    result.eigenvalues = VecX(n);
    for (int i = 0; i < n; ++i) {
        result.eigenvalues[i] = D(i, i);
    }
    result.eigenvectors = V;

    // Sort by magnitude (descending)
    SortEigenByMagnitude(result.eigenvalues, result.eigenvectors);

    result.valid = true;
    return result;
}

VecX EigenvaluesSymmetric(const MatX& A) {
    EigenResult result = EigenSymmetric(A);
    return result.eigenvalues;
}

// =============================================================================
// QR Iteration for Symmetric Matrices
// =============================================================================

EigenResult EigenSymmetricQR(const MatX& A, double tolerance, int maxIterations) {
    // For now, delegate to Jacobi which is more robust for general symmetric
    // TODO: Implement tridiagonalization + QR with Wilkinson shift for better performance
    return EigenSymmetric(A, tolerance, maxIterations);
}

// =============================================================================
// General Eigenvalue Decomposition
// =============================================================================

EigenResult EigenGeneral(const MatX& A, double tolerance, int maxIterations) {
    EigenResult result;

    int n = A.Rows();
    if (n == 0 || A.Cols() != n) {
        return result;
    }

    // Special cases
    if (n == 1) {
        result.eigenvalues = VecX(1);
        result.eigenvalues[0] = A(0, 0);
        result.eigenvectors = MatX(1, 1);
        result.eigenvectors(0, 0) = 1.0;
        result.converged = true;
        result.valid = true;
        return result;
    }

    if (n == 2) {
        Mat22 A2;
        A2(0, 0) = A(0, 0); A2(0, 1) = A(0, 1);
        A2(1, 0) = A(1, 0); A2(1, 1) = A(1, 1);
        Eigen2x2Result r2 = EigenGeneral2x2(A2);

        result.eigenvalues = VecX(2);
        result.eigenvalues[0] = r2.lambda1;
        result.eigenvalues[1] = r2.lambda2;
        result.eigenvectors = MatX(2, 2);
        result.eigenvectors(0, 0) = r2.v1[0]; result.eigenvectors(1, 0) = r2.v1[1];
        result.eigenvectors(0, 1) = r2.v2[0]; result.eigenvectors(1, 1) = r2.v2[1];

        if (!r2.isReal) {
            result.imaginaryParts = VecX(2);
            result.imaginaryParts[0] = r2.imagPart;
            result.imaginaryParts[1] = -r2.imagPart;
        }

        result.converged = true;
        result.valid = r2.valid;
        return result;
    }

    if (n == 3) {
        Mat33 A3;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                A3(i, j) = A(i, j);
            }
        }
        Eigen3x3Result r3 = EigenGeneral3x3(A3);

        result.eigenvalues = VecX(3);
        result.eigenvalues[0] = r3.lambda1;
        result.eigenvalues[1] = r3.lambda2;
        result.eigenvalues[2] = r3.lambda3;
        result.eigenvectors = MatX(3, 3);
        result.eigenvectors(0, 0) = r3.v1[0]; result.eigenvectors(1, 0) = r3.v1[1]; result.eigenvectors(2, 0) = r3.v1[2];
        result.eigenvectors(0, 1) = r3.v2[0]; result.eigenvectors(1, 1) = r3.v2[1]; result.eigenvectors(2, 1) = r3.v2[2];
        result.eigenvectors(0, 2) = r3.v3[0]; result.eigenvectors(1, 2) = r3.v3[1]; result.eigenvectors(2, 2) = r3.v3[2];
        result.converged = true;
        result.valid = r3.valid;
        return result;
    }

    // For larger matrices, use QR iteration with Hessenberg reduction
    // First reduce to Hessenberg form
    auto [H, Q] = HessenbergReduce(A);

    // QR iteration on Hessenberg matrix
    MatX Hwork = H;
    MatX Vwork = Q;

    result.eigenvalues = VecX(n);
    result.imaginaryParts = VecX(n);

    int deflatedSize = n;
    result.iterations = 0;

    for (int iter = 0; iter < maxIterations && deflatedSize > 2; ++iter) {
        result.iterations = iter + 1;

        // Check for deflation (subdiagonal element near zero)
        bool deflated = false;
        for (int i = deflatedSize - 1; i > 0; --i) {
            if (std::abs(Hwork(i, i - 1)) < tolerance * (std::abs(Hwork(i - 1, i - 1)) + std::abs(Hwork(i, i)))) {
                Hwork(i, i - 1) = 0.0;

                // Extract eigenvalue
                result.eigenvalues[deflatedSize - 1] = Hwork(deflatedSize - 1, deflatedSize - 1);
                result.imaginaryParts[deflatedSize - 1] = 0.0;

                deflatedSize--;
                deflated = true;
                break;
            }
        }

        if (!deflated) {
            // Apply QR step with Wilkinson shift
            double a = Hwork(deflatedSize - 2, deflatedSize - 2);
            double b = Hwork(deflatedSize - 2, deflatedSize - 1);
            double c = Hwork(deflatedSize - 1, deflatedSize - 2);
            double d = Hwork(deflatedSize - 1, deflatedSize - 1);

            double trace = a + d;
            double det = a * d - b * c;
            double disc = trace * trace - 4.0 * det;

            double shift;
            if (disc >= 0) {
                double sqrtDisc = std::sqrt(disc);
                double lambda1 = (trace + sqrtDisc) / 2.0;
                double lambda2 = (trace - sqrtDisc) / 2.0;
                shift = (std::abs(lambda1 - d) < std::abs(lambda2 - d)) ? lambda1 : lambda2;
            } else {
                shift = d;  // Use bottom-right element
            }

            // Implicit QR step (simplified)
            for (int i = 0; i < deflatedSize; ++i) {
                Hwork(i, i) -= shift;
            }

            QRResult qr = QR_Decompose(Hwork);
            if (qr.valid) {
                Hwork = qr.R * qr.Q;
                Vwork = Vwork * qr.Q;
            }

            for (int i = 0; i < deflatedSize; ++i) {
                Hwork(i, i) += shift;
            }
        }
    }

    // Handle remaining 2x2 or 1x1 blocks
    for (int i = 0; i < deflatedSize; ++i) {
        result.eigenvalues[i] = Hwork(i, i);
        result.imaginaryParts[i] = 0.0;
    }

    result.eigenvectors = Vwork;
    result.converged = (result.iterations < maxIterations);
    result.valid = true;

    SortEigenByMagnitude(result.eigenvalues, result.eigenvectors);

    return result;
}

std::pair<VecX, VecX> EigenvaluesGeneral(const MatX& A) {
    EigenResult result = EigenGeneral(A);
    return {result.eigenvalues, result.imaginaryParts};
}

// =============================================================================
// Power Iteration Methods
// =============================================================================

std::pair<double, VecX> PowerIteration(const MatX& A, double tolerance,
                                        int maxIterations, const VecX& initialGuess) {
    int n = A.Rows();
    if (n == 0 || A.Cols() != n) {
        return {0.0, VecX()};
    }

    // Initialize
    VecX x(n);
    if (initialGuess.Size() == n) {
        x = initialGuess;
    } else {
        // Random-ish initial vector
        for (int i = 0; i < n; ++i) {
            x[i] = 1.0 + 0.1 * i;
        }
    }
    x = NormalizeVec(x);

    double lambda = 0.0;

    for (int iter = 0; iter < maxIterations; ++iter) {
        // y = A * x
        VecX y(n);
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                sum += A(i, j) * x[j];
            }
            y[i] = sum;
        }

        // Compute Rayleigh quotient for eigenvalue estimate
        double newLambda = 0.0;
        double xDotY = 0.0;
        double xDotX = 0.0;
        for (int i = 0; i < n; ++i) {
            xDotY += x[i] * y[i];
            xDotX += x[i] * x[i];
        }
        newLambda = xDotY / xDotX;

        // Normalize
        x = NormalizeVec(y);

        // Check convergence
        if (std::abs(newLambda - lambda) < tolerance * std::abs(newLambda)) {
            lambda = newLambda;
            break;
        }

        lambda = newLambda;
    }

    return {lambda, x};
}

std::pair<double, VecX> InversePowerIteration(const MatX& A, double tolerance,
                                               int maxIterations) {
    // Solve A*y = x at each step, finding smallest eigenvalue
    int n = A.Rows();
    if (n == 0 || A.Cols() != n) {
        return {0.0, VecX()};
    }

    // LU decomposition for repeated solves
    LUResult lu = LU_Decompose(A);
    if (!lu.valid) {
        return {0.0, VecX()};
    }

    VecX x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0;
    }
    x = NormalizeVec(x);

    double lambda = 0.0;

    for (int iter = 0; iter < maxIterations; ++iter) {
        // Solve A*y = x
        VecX y = SolveFromLU(lu, x);

        // Rayleigh quotient: lambda_inv = x^T * y / (x^T * x)
        double xDotY = 0.0;
        double xDotX = 0.0;
        for (int i = 0; i < n; ++i) {
            xDotY += x[i] * y[i];
            xDotX += x[i] * x[i];
        }
        double lambdaInv = xDotY / xDotX;
        double newLambda = (std::abs(lambdaInv) > EPSILON) ? 1.0 / lambdaInv : 0.0;

        x = NormalizeVec(y);

        if (std::abs(newLambda - lambda) < tolerance * std::abs(newLambda)) {
            lambda = newLambda;
            break;
        }

        lambda = newLambda;
    }

    return {lambda, x};
}

std::pair<double, VecX> ShiftedInversePowerIteration(const MatX& A, double shift,
                                                      double tolerance, int maxIterations) {
    int n = A.Rows();
    if (n == 0 || A.Cols() != n) {
        return {0.0, VecX()};
    }

    // Form A - shift*I
    MatX Ashifted = A;
    for (int i = 0; i < n; ++i) {
        Ashifted(i, i) -= shift;
    }

    // LU decomposition
    LUResult lu = LU_Decompose(Ashifted);
    if (!lu.valid) {
        return {shift, VecX()};  // shift might be an eigenvalue
    }

    VecX x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0;
    }
    x = NormalizeVec(x);

    double lambda = shift;

    for (int iter = 0; iter < maxIterations; ++iter) {
        VecX y = SolveFromLU(lu, x);

        double xDotY = 0.0;
        double xDotX = 0.0;
        for (int i = 0; i < n; ++i) {
            xDotY += x[i] * y[i];
            xDotX += x[i] * x[i];
        }
        double lambdaInv = xDotY / xDotX;
        double newLambda = (std::abs(lambdaInv) > EPSILON) ? shift + 1.0 / lambdaInv : shift;

        x = NormalizeVec(y);

        if (std::abs(newLambda - lambda) < tolerance * std::abs(newLambda)) {
            lambda = newLambda;
            break;
        }

        lambda = newLambda;
    }

    return {lambda, x};
}

std::pair<double, VecX> RayleighQuotientIteration(const MatX& A, const VecX& initialGuess,
                                                   double tolerance, int maxIterations) {
    int n = A.Rows();
    if (n == 0 || A.Cols() != n || initialGuess.Size() != n) {
        return {0.0, VecX()};
    }

    VecX x = NormalizeVec(initialGuess);
    double lambda = RayleighQuotient(A, x);

    for (int iter = 0; iter < maxIterations; ++iter) {
        // Form A - lambda*I
        MatX Ashifted = A;
        for (int i = 0; i < n; ++i) {
            Ashifted(i, i) -= lambda;
        }

        // Solve (A - lambda*I) * y = x
        VecX y = SolveLU(Ashifted, x);

        if (VecNorm(y) < EPSILON) {
            // Found exact eigenvalue
            break;
        }

        x = NormalizeVec(y);
        double newLambda = RayleighQuotient(A, x);

        if (std::abs(newLambda - lambda) < tolerance) {
            lambda = newLambda;
            break;
        }

        lambda = newLambda;
    }

    return {lambda, x};
}

// =============================================================================
// Utility Functions
// =============================================================================

void SortEigenByMagnitude(VecX& eigenvalues, MatX& eigenvectors) {
    int n = eigenvalues.Size();
    if (n == 0) return;

    // Create index array
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices by eigenvalue magnitude (descending)
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return std::abs(eigenvalues[a]) > std::abs(eigenvalues[b]);
    });

    // Apply permutation
    VecX sortedVals(n);
    MatX sortedVecs(eigenvectors.Rows(), eigenvectors.Cols());

    for (int i = 0; i < n; ++i) {
        sortedVals[i] = eigenvalues[indices[i]];
        for (int j = 0; j < eigenvectors.Rows(); ++j) {
            sortedVecs(j, i) = eigenvectors(j, indices[i]);
        }
    }

    eigenvalues = sortedVals;
    eigenvectors = sortedVecs;
}

void SortEigenByValue(VecX& eigenvalues, MatX& eigenvectors) {
    int n = eigenvalues.Size();
    if (n == 0) return;

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return eigenvalues[a] > eigenvalues[b];
    });

    VecX sortedVals(n);
    MatX sortedVecs(eigenvectors.Rows(), eigenvectors.Cols());

    for (int i = 0; i < n; ++i) {
        sortedVals[i] = eigenvalues[indices[i]];
        for (int j = 0; j < eigenvectors.Rows(); ++j) {
            sortedVecs(j, i) = eigenvectors(j, indices[i]);
        }
    }

    eigenvalues = sortedVals;
    eigenvectors = sortedVecs;
}

double RayleighQuotient(const MatX& A, const VecX& x) {
    int n = A.Rows();
    if (n == 0 || x.Size() != n) {
        return 0.0;
    }

    // Compute x^T * A * x
    double xAx = 0.0;
    double xTx = 0.0;

    for (int i = 0; i < n; ++i) {
        double Axi = 0.0;
        for (int j = 0; j < n; ++j) {
            Axi += A(i, j) * x[j];
        }
        xAx += x[i] * Axi;
        xTx += x[i] * x[i];
    }

    return (std::abs(xTx) > EPSILON) ? xAx / xTx : 0.0;
}

bool IsPositiveDefinite(const MatX& A, double tolerance) {
    VecX eigenvalues = EigenvaluesSymmetric(A);
    for (int i = 0; i < eigenvalues.Size(); ++i) {
        if (eigenvalues[i] <= tolerance) {
            return false;
        }
    }
    return true;
}

bool IsPositiveSemiDefinite(const MatX& A, double tolerance) {
    VecX eigenvalues = EigenvaluesSymmetric(A);
    for (int i = 0; i < eigenvalues.Size(); ++i) {
        if (eigenvalues[i] < -tolerance) {
            return false;
        }
    }
    return true;
}

MatX MatrixSquareRoot(const MatX& A) {
    EigenResult eig = EigenSymmetric(A);
    if (!eig.valid) {
        return MatX();
    }

    int n = eig.eigenvalues.Size();

    // Check all eigenvalues are non-negative
    for (int i = 0; i < n; ++i) {
        if (eig.eigenvalues[i] < -EIGEN_TOLERANCE) {
            return MatX();  // Not positive semi-definite
        }
    }

    // B = V * sqrt(D) * V^T
    MatX sqrtD = MatX::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        sqrtD(i, i) = std::sqrt(std::max(0.0, eig.eigenvalues[i]));
    }

    MatX V = eig.eigenvectors;
    MatX VT = V.Transpose();

    return V * sqrtD * VT;
}

MatX MatrixExponential(const MatX& A) {
    EigenResult eig = EigenSymmetric(A);
    if (!eig.valid) {
        return MatX();
    }

    int n = eig.eigenvalues.Size();

    // exp(A) = V * exp(D) * V^T
    MatX expD = MatX::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        expD(i, i) = std::exp(eig.eigenvalues[i]);
    }

    MatX V = eig.eigenvectors;
    MatX VT = V.Transpose();

    return V * expD * VT;
}

MatX MatrixLogarithm(const MatX& A) {
    EigenResult eig = EigenSymmetric(A);
    if (!eig.valid) {
        return MatX();
    }

    int n = eig.eigenvalues.Size();

    // Check all eigenvalues are positive
    for (int i = 0; i < n; ++i) {
        if (eig.eigenvalues[i] <= 0) {
            return MatX();  // Not positive definite
        }
    }

    // log(A) = V * log(D) * V^T
    MatX logD = MatX::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        logD(i, i) = std::log(eig.eigenvalues[i]);
    }

    MatX V = eig.eigenvectors;
    MatX VT = V.Transpose();

    return V * logD * VT;
}

VecX EigenvalueConditionNumbers(const MatX& A) {
    // Simplified: condition number for simple eigenvalue is 1/(|v^T * w|)
    // where v is left eigenvector, w is right eigenvector
    // For symmetric matrices, condition number is 1

    int n = A.Rows();
    VecX condNums(n);

    // Check if symmetric
    bool isSymmetric = true;
    for (int i = 0; i < n && isSymmetric; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (std::abs(A(i, j) - A(j, i)) > EPSILON) {
                isSymmetric = false;
                break;
            }
        }
    }

    if (isSymmetric) {
        for (int i = 0; i < n; ++i) {
            condNums[i] = 1.0;
        }
    } else {
        // For non-symmetric, would need both left and right eigenvectors
        // Simplified: return 1 for now
        for (int i = 0; i < n; ++i) {
            condNums[i] = 1.0;
        }
    }

    return condNums;
}

std::pair<MatX, MatX> Tridiagonalize(const MatX& A) {
    int n = A.Rows();
    if (n == 0 || A.Cols() != n) {
        return {MatX(), MatX()};
    }

    MatX T = A;
    MatX Q = MatX::Identity(n);

    // Householder reduction to tridiagonal form
    for (int k = 0; k < n - 2; ++k) {
        // Create Householder vector for column k below diagonal
        VecX x(n - k - 1);
        for (int i = 0; i < n - k - 1; ++i) {
            x[i] = T(k + 1 + i, k);
        }

        double norm = VecNorm(x);
        if (norm < EPSILON) continue;

        // Householder vector
        VecX v = x;
        v[0] += (x[0] >= 0 ? 1 : -1) * norm;
        double vNorm = VecNorm(v);
        if (vNorm < EPSILON) continue;

        for (int i = 0; i < v.Size(); ++i) {
            v[i] /= vNorm;
        }

        // Apply H = I - 2*v*v^T from left and right
        // T = H * T * H

        // First: T = T - 2*v*(v^T*T) for rows k+1:n
        // Then: T = T - 2*(T*v)*v^T for columns k+1:n

        // Compute v^T * T (for rows k+1:n)
        VecX vTT(n);
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int i = 0; i < n - k - 1; ++i) {
                sum += v[i] * T(k + 1 + i, j);
            }
            vTT[j] = sum;
        }

        // T = T - 2*v*vTT^T
        for (int i = 0; i < n - k - 1; ++i) {
            for (int j = 0; j < n; ++j) {
                T(k + 1 + i, j) -= 2.0 * v[i] * vTT[j];
            }
        }

        // Compute T * v (for columns k+1:n)
        VecX Tv(n);
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n - k - 1; ++j) {
                sum += T(i, k + 1 + j) * v[j];
            }
            Tv[i] = sum;
        }

        // T = T - 2*Tv*v^T
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n - k - 1; ++j) {
                T(i, k + 1 + j) -= 2.0 * Tv[i] * v[j];
            }
        }

        // Update Q: Q = Q * H
        VecX Qv(n);
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n - k - 1; ++j) {
                sum += Q(i, k + 1 + j) * v[j];
            }
            Qv[i] = sum;
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n - k - 1; ++j) {
                Q(i, k + 1 + j) -= 2.0 * Qv[i] * v[j];
            }
        }
    }

    return {T, Q};
}

std::pair<MatX, MatX> HessenbergReduce(const MatX& A) {
    int n = A.Rows();
    if (n == 0 || A.Cols() != n) {
        return {MatX(), MatX()};
    }

    MatX H = A;
    MatX Q = MatX::Identity(n);

    // Householder reduction to upper Hessenberg form
    for (int k = 0; k < n - 2; ++k) {
        // Create Householder vector for column k below subdiagonal
        VecX x(n - k - 1);
        for (int i = 0; i < n - k - 1; ++i) {
            x[i] = H(k + 1 + i, k);
        }

        double norm = VecNorm(x);
        if (norm < EPSILON) continue;

        VecX v = x;
        v[0] += (x[0] >= 0 ? 1 : -1) * norm;
        double vNorm = VecNorm(v);
        if (vNorm < EPSILON) continue;

        for (int i = 0; i < v.Size(); ++i) {
            v[i] /= vNorm;
        }

        // H = H - 2*v*(v^T*H) for rows k+1:n
        VecX vTH(n);
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int i = 0; i < n - k - 1; ++i) {
                sum += v[i] * H(k + 1 + i, j);
            }
            vTH[j] = sum;
        }

        for (int i = 0; i < n - k - 1; ++i) {
            for (int j = 0; j < n; ++j) {
                H(k + 1 + i, j) -= 2.0 * v[i] * vTH[j];
            }
        }

        // H = H - 2*(H*v)*v^T for columns k+1:n (only affects columns k+1:n)
        VecX Hv(n);
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n - k - 1; ++j) {
                sum += H(i, k + 1 + j) * v[j];
            }
            Hv[i] = sum;
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n - k - 1; ++j) {
                H(i, k + 1 + j) -= 2.0 * Hv[i] * v[j];
            }
        }

        // Update Q
        VecX Qv(n);
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n - k - 1; ++j) {
                sum += Q(i, k + 1 + j) * v[j];
            }
            Qv[i] = sum;
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n - k - 1; ++j) {
                Q(i, k + 1 + j) -= 2.0 * Qv[i] * v[j];
            }
        }
    }

    return {H, Q};
}

// =============================================================================
// Generalized Eigenvalue Problem
// =============================================================================

EigenResult GeneralizedEigen(const MatX& A, const MatX& B) {
    // Solve A*x = lambda*B*x for symmetric A and SPD B
    // Transform to standard problem: L^{-1}*A*L^{-T}*y = lambda*y
    // where B = L*L^T (Cholesky) and x = L^{-T}*y

    EigenResult result;

    int n = A.Rows();
    if (n == 0 || A.Cols() != n || B.Rows() != n || B.Cols() != n) {
        return result;
    }

    // Cholesky decomposition of B
    CholeskyResult chol = Cholesky_Decompose(B);
    if (!chol.valid) {
        return result;  // B is not positive definite
    }

    // Form C = L^{-1} * A * L^{-T}
    // First compute L^{-1} * A
    MatX LinvA(n, n);
    for (int j = 0; j < n; ++j) {
        VecX col(n);
        for (int i = 0; i < n; ++i) {
            col[i] = A(i, j);
        }
        VecX x = SolveFromCholesky(chol, col);
        // Actually need just forward substitution L*y = col
        // Simplified: use full solve for now
        for (int i = 0; i < n; ++i) {
            LinvA(i, j) = x[i];
        }
    }

    // Then compute (L^{-1}*A) * L^{-T}
    MatX C(n, n);
    MatX LT = chol.L.Transpose();

    // C = LinvA * L^{-T}
    // Solve LT * C^T = LinvA^T column by column
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                // Need L^{-T} which requires solving L^T * x = e_k
                // Simplified: direct computation
                sum += LinvA(i, k) * LinvA(j, k);  // This is approximate
            }
            C(i, j) = LinvA(i, j);  // Simplified
        }
    }

    // For proper implementation, we should solve the triangular systems
    // For now, use a simplified approach: compute standard eigenvalue of B^{-1}*A
    MatX Binv = PseudoInverse(B);
    MatX BinvA = Binv * A;

    result = EigenSymmetric(BinvA);

    // Transform eigenvectors back: x = L^{-T} * y
    // Simplified: eigenvectors need transformation

    return result;
}

Eigen2x2Result GeneralizedEigen2x2(const Mat22& A, const Mat22& B) {
    // Solve A*x = lambda*B*x
    // det(A - lambda*B) = 0

    Eigen2x2Result result;
    result.valid = true;

    // Compute det(A - lambda*B) = 0
    // (a11 - lambda*b11)(a22 - lambda*b22) - (a12 - lambda*b12)(a21 - lambda*b21) = 0

    double a11 = A(0, 0), a12 = A(0, 1), a21 = A(1, 0), a22 = A(1, 1);
    double b11 = B(0, 0), b12 = B(0, 1), b21 = B(1, 0), b22 = B(1, 1);

    // Coefficient of lambda^2
    double c2 = b11 * b22 - b12 * b21;
    // Coefficient of lambda
    double c1 = -(a11 * b22 + a22 * b11 - a12 * b21 - a21 * b12);
    // Constant term
    double c0 = a11 * a22 - a12 * a21;

    if (std::abs(c2) < EPSILON) {
        // Linear equation
        if (std::abs(c1) > EPSILON) {
            result.lambda1 = -c0 / c1;
            result.lambda2 = result.lambda1;
        }
        result.isReal = true;
    } else {
        double disc = c1 * c1 - 4 * c2 * c0;
        if (disc >= 0) {
            result.isReal = true;
            double sqrtDisc = std::sqrt(disc);
            result.lambda1 = (-c1 + sqrtDisc) / (2 * c2);
            result.lambda2 = (-c1 - sqrtDisc) / (2 * c2);
        } else {
            result.isReal = false;
            result.lambda1 = -c1 / (2 * c2);
            result.lambda2 = result.lambda1;
            result.imagPart = std::sqrt(-disc) / (2 * std::abs(c2));
        }
    }

    // Compute eigenvectors (simplified)
    result.v1[0] = 1; result.v1[1] = 0;
    result.v2[0] = 0; result.v2[1] = 1;

    return result;
}

Eigen3x3Result GeneralizedEigen3x3(const Mat33& A, const Mat33& B) {
    // For 3x3, convert to standard form using B^{-1}*A
    Eigen3x3Result result;
    result.valid = true;

    double detB = B.Determinant();
    if (std::abs(detB) < EPSILON) {
        result.valid = false;
        return result;
    }

    Mat33 Binv = B.Inverse();
    Mat33 BinvA;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += Binv(i, k) * A(k, j);
            }
            BinvA(i, j) = sum;
        }
    }

    return EigenGeneral3x3(BinvA);
}

} // namespace Qi::Vision::Internal
