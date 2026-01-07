# Internal/Solver 设计文档

## 1. 概述

### 1.1 功能描述
Solver 模块是 QiVision 的线性方程组求解库，提供多种矩阵分解算法和方程组求解方法。专为工业视觉中的几何拟合、相机标定和坐标变换等应用设计。

### 1.2 应用场景
- **几何拟合**: 最小二乘直线/圆/椭圆拟合 (A^T A x = A^T b)
- **相机标定**: Zhang 标定法中的单应性求解、畸变参数优化
- **坐标变换**: 仿射变换、透视变换参数估计
- **齐次方程**: 基础矩阵/本质矩阵求解 (Ax = 0)
- **鲁棒估计**: RANSAC 内点验证中的快速求解

### 1.3 参考
- Halcon: `solve_linear_ode`, `hom_mat2d_to_affine_par`
- LAPACK: dgesv, dposv, dgels, dgesvd
- Numerical Recipes in C++ - 矩阵分解章节
- Golub & Van Loan, "Matrix Computations" (第4版)

### 1.4 设计原则
1. **数值稳定**: 使用部分主元 LU、Householder QR、隐式 QR SVD
2. **适度优化**: 小矩阵直接公式，大矩阵标准算法
3. **完整分解**: 提供分解后复用能力 (多右端项求解)
4. **优雅退化**: 奇异矩阵、秩亏矩阵返回最小范数解
5. **与 Matrix.h 协调**: 使用已定义的 MatX, VecX, LUResult 等结构

---

## 2. 设计规则验证

### 2.1 坐标类型符合规则
- [x] 矩阵元素使用 `double` (符合亚像素坐标精度要求)
- [x] 索引使用 `int` (与 MatX/VecX 一致)
- [x] 返回值使用标准 C++ 类型

### 2.2 层级依赖正确
- [x] Solver.h 位于 Internal 层
- [x] 依赖 Internal/Matrix.h (MatX, VecX, 分解结果结构)
- [x] 不依赖 Feature 层
- [x] 不依赖 Platform 层 (通过 Matrix.h 间接依赖 Memory.h)

### 2.3 算法完整性
- [x] LU 分解 (方阵直接求解)
- [x] Cholesky 分解 (对称正定，2倍效率)
- [x] QR 分解 (超定系统最小二乘)
- [x] SVD 分解 (通用最小二乘、伪逆)
- [x] 齐次方程求解 (零空间)
- [x] 条件数计算
- [x] 秩计算
- [x] 伪逆

### 2.4 退化情况处理
- [x] 奇异矩阵: 返回最小范数解 (via SVD)
- [x] 近奇异: 提供条件数警告
- [x] 秩亏: 返回数值秩和对应解
- [x] 尺寸不匹配: 抛出 InvalidArgumentException

---

## 3. 依赖分析

### 3.1 依赖的 Internal 模块
| 模块 | 用途 | 状态 |
|------|------|------|
| Internal/Matrix.h | MatX, VecX, 分解结果结构 | ✅ 已完成 |

### 3.2 依赖的 Core 类型
| 类型 | 用途 |
|------|------|
| Core/Constants.h | 数值常量 (EPSILON) |
| Core/Exception.h | 异常类型 |

### 3.3 被依赖的模块
| 模块 | 用途 | 状态 |
|------|------|------|
| Internal/Eigen.h | 特征值分解 (可能复用 SVD) | 待设计 |
| Internal/Fitting.h | 直线/圆/椭圆拟合 | 待设计 |
| Internal/Homography.h | 单应性矩阵计算 | 待设计 |
| Calib/CameraCalib.h | 相机标定 | 待设计 |

---

## 4. 类设计

### 4.1 模块结构

```
Solver Module
├── Matrix Decompositions
│   ├── LU_Decompose()           - LU 分解 (方阵)
│   ├── LU_DecomposeInPlace()    - 原地 LU 分解
│   ├── Cholesky_Decompose()     - Cholesky 分解 (正定)
│   ├── QR_Decompose()           - QR 分解 (一般矩形)
│   ├── QR_DecomposeThin()       - 薄 QR 分解
│   └── SVD_Decompose()          - SVD 分解
│
├── Linear System Solvers
│   ├── SolveLU()                - 使用 LU 分解求解
│   ├── SolveCholesky()          - 使用 Cholesky 分解求解
│   ├── SolveLeastSquares()      - 超定系统 (QR)
│   ├── SolveLeastSquaresNormal() - 法方程法 (A^T A x = A^T b)
│   ├── SolveSVD()               - 通用最小二乘 (SVD)
│   └── SolveHomogeneous()       - 齐次系统 Ax=0
│
├── Solve with Pre-computed Decomposition
│   ├── SolveFromLU()            - 从 LU 结果求解
│   ├── SolveFromCholesky()      - 从 Cholesky 结果求解
│   ├── SolveFromQR()            - 从 QR 结果求解
│   └── SolveFromSVD()           - 从 SVD 结果求解
│
└── Utility Functions
    ├── ComputeConditionNumber() - 条件数
    ├── ComputeRank()            - 数值秩
    ├── PseudoInverse()          - Moore-Penrose 伪逆
    ├── ComputeResidual()        - 残差计算
    └── ComputeNullSpace()       - 零空间基
```

### 4.2 API 设计

```cpp
#pragma once

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
 * A is overwritten with L (lower) and U (upper), diagonal stores U
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
 */
Vec2 Solve2x2(const Mat22& A, const Vec2& b);

/**
 * @brief Solve 3x3 system directly
 */
Vec3 Solve3x3(const Mat33& A, const Vec3& b);

/**
 * @brief Solve 4x4 system directly
 */
Vec4 Solve4x4(const Mat44& A, const Vec4& b);

} // namespace Qi::Vision::Internal
```

---

## 5. 参数设计

### 5.1 精度控制参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| SOLVER_RANK_TOLERANCE | double | 1e-10 | [1e-15, 1e-6] | SVD 秩判定阈值 |
| SOLVER_SINGULAR_THRESHOLD | double | 1e-12 | [1e-15, 1e-8] | 奇异性检测阈值 |
| SOLVER_MAX_REFINEMENT_ITERATIONS | int | 3 | [0, 10] | 迭代精化次数 |

### 5.2 算法选择指南

| 系统类型 | 推荐方法 | 原因 |
|----------|----------|------|
| 方阵，适定 | SolveLU | 最快 |
| 对称正定 | SolveCholesky | 2倍快 |
| 超定 m > n | SolveLeastSquares (QR) | 稳定 |
| 超定，快速 | SolveLeastSquaresNormal | 更快但不稳定 |
| 秩亏 | SolveSVD | 最鲁棒 |
| 齐次 Ax=0 | SolveHomogeneous | SVD 最后列 |
| 多右端项 | Decompose + SolveFrom* | 复用分解 |
| 小矩阵 2x2~4x4 | Solve2x2/3x3/4x4 | 直接公式 |
| 三对角 | SolveTridiagonal | O(n) |

---

## 6. 精度规格

### 6.1 分解精度

| 分解 | 条件 | 残差要求 |
|------|------|----------|
| LU | cond(A) < 1e6 | ||PA - LU||_F / ||A||_F < 1e-12 |
| Cholesky | SPD, cond < 1e6 | ||A - LL^T||_F / ||A||_F < 1e-12 |
| QR | m x n | ||A - QR||_F / ||A||_F < 1e-12, ||Q^T Q - I||_F < 1e-12 |
| SVD | 任意 | ||A - USV^T||_F / ||A||_F < 1e-10 |

### 6.2 求解精度

| 方法 | 条件 | 相对误差要求 |
|------|------|-------------|
| SolveLU | cond(A) < 1e6 | ||x - x_true|| / ||x_true|| < cond(A) * 1e-14 |
| SolveCholesky | SPD, cond < 1e6 | ||x - x_true|| / ||x_true|| < cond(A) * 1e-14 |
| SolveLeastSquares | 满秩 | ||Ax - b|| 最小化 |
| SolveSVD | 秩亏 | 最小范数解 |

### 6.3 数值稳定性措施

1. **LU 分解**: 部分主元选择 (partial pivoting)
2. **QR 分解**: Householder 反射 (比 Gram-Schmidt 稳定)
3. **SVD 分解**: 隐式 QR 迭代，Golub-Kahan 双对角化
4. **Cholesky**: 检测非正定，返回 valid=false

---

## 7. 算法要点

### 7.1 LU 分解 (Doolittle with Partial Pivoting)

```cpp
// 伪代码
for k = 0 to n-1:
    // 选主元
    pivot_row = argmax(|A[i,k]| for i >= k)
    swap(A[k], A[pivot_row])
    P[k] = pivot_row
    
    // 消元
    for i = k+1 to n-1:
        A[i,k] /= A[k,k]  // L 的元素
        for j = k+1 to n-1:
            A[i,j] -= A[i,k] * A[k,j]
```

**复杂度**: O(2n^3/3)

### 7.2 Cholesky 分解

```cpp
// A = L * L^T
for j = 0 to n-1:
    // 对角元
    sum = A[j,j]
    for k = 0 to j-1:
        sum -= L[j,k]^2
    if sum <= 0: return not_positive_definite
    L[j,j] = sqrt(sum)
    
    // 非对角元
    for i = j+1 to n-1:
        sum = A[i,j]
        for k = 0 to j-1:
            sum -= L[i,k] * L[j,k]
        L[i,j] = sum / L[j,j]
```

**复杂度**: O(n^3/3)

### 7.3 QR 分解 (Householder)

```cpp
// A = Q * R
for k = 0 to min(m,n)-1:
    // 构造 Householder 向量
    x = A[k:m, k]
    alpha = -sign(x[0]) * ||x||
    v = x; v[0] -= alpha
    v /= ||v||
    
    // 应用 Householder 反射 H = I - 2*v*v^T
    A[k:m, k:n] = A[k:m, k:n] - 2 * v * (v^T * A[k:m, k:n])
    
    // 存储 v 用于构造 Q (optional)
```

**复杂度**: O(2mn^2 - 2n^3/3)

### 7.4 SVD 分解 (Golub-Kahan)

```
1. 双对角化: A = U1 * B * V1^T (Householder)
2. 隐式 QR 迭代求 B 的奇异值
3. 累积变换: U = U1 * U2, V = V1 * V2
```

**复杂度**: O(min(mn^2, m^2n)) + 迭代

### 7.5 齐次方程求解

```cpp
// Ax = 0 的非零解是 V 的最后一列 (对应最小奇异值)
VecX SolveHomogeneous(const MatX& A) {
    SVDResult svd = SVD_Decompose(A);
    int n = A.Cols();
    return svd.V.Col(n - 1);  // 最后一列
}
```

### 7.6 伪逆计算

```cpp
// A^+ = V * S^+ * U^T
// 其中 S^+[i,i] = 1/S[i] if S[i] > tol, else 0
MatX PseudoInverse(const MatX& A, double tol) {
    SVDResult svd = SVD_Decompose(A);
    MatX Sinv = MatX::Zero(A.Cols(), A.Rows());
    for (int i = 0; i < svd.rank; ++i) {
        if (svd.S[i] > tol) {
            Sinv(i, i) = 1.0 / svd.S[i];
        }
    }
    return svd.V * Sinv * svd.U.Transpose();
}
```

### 7.7 退化情况处理

| 情况 | 检测方法 | 处理方式 |
|------|----------|----------|
| 奇异矩阵 | LU 主元 < eps | 返回零向量，设置 valid=false |
| 近奇异 | cond > 1e10 | 警告，可能使用 SVD |
| 非正定 | Cholesky 对角 <= 0 | 返回 valid=false |
| 秩亏超定 | rank < n | SVD 最小范数解 |
| 尺寸不匹配 | rows != cols 等 | 抛出 InvalidArgumentException |

---

## 8. 小矩阵特化

### 8.1 2x2 系统直接求解 (Cramer's Rule)

```cpp
Vec2 Solve2x2(const Mat22& A, const Vec2& b) {
    double det = A(0,0)*A(1,1) - A(0,1)*A(1,0);
    if (std::abs(det) < SOLVER_SINGULAR_THRESHOLD) {
        return Vec2::Zero();
    }
    double invDet = 1.0 / det;
    return Vec2{
        (A(1,1)*b[0] - A(0,1)*b[1]) * invDet,
        (A(0,0)*b[1] - A(1,0)*b[0]) * invDet
    };
}
```

### 8.2 3x3 系统直接求解

```cpp
Vec3 Solve3x3(const Mat33& A, const Vec3& b) {
    Mat33 Ainv = A.Inverse();
    if (!A.IsInvertible()) {
        return Vec3::Zero();
    }
    return Ainv * b;
}
```

### 8.3 性能考虑

| 矩阵尺寸 | 推荐方法 | 原因 |
|----------|----------|------|
| 2x2 | 直接公式 | 无分支，最快 |
| 3x3 | 直接公式 | 无分支，避免分配 |
| 4x4 | 直接或 LU | 视情况 |
| >4x4 | LU/QR/SVD | 标准算法 |

---

## 9. 实现任务分解

| 任务 | 文件 | 预估时间 | 依赖 | 优先级 |
|------|------|----------|------|--------|
| 头文件 API 定义 | Solver.h | 1h | Matrix.h | P0 |
| 小矩阵直接求解 | Solver.cpp | 1h | - | P0 |
| 三角系统求解 | Solver.cpp | 1h | - | P0 |
| LU 分解 + 求解 | Solver.cpp | 3h | 三角求解 | P0 |
| Cholesky 分解 + 求解 | Solver.cpp | 2h | 三角求解 | P0 |
| QR 分解 (Householder) | Solver.cpp | 3h | - | P0 |
| QR 最小二乘求解 | Solver.cpp | 1h | QR 分解 | P0 |
| SVD 分解 | Solver.cpp | 4h | - | P1 |
| SVD 求解 + 伪逆 | Solver.cpp | 2h | SVD 分解 | P1 |
| 齐次方程求解 | Solver.cpp | 1h | SVD 分解 | P1 |
| 条件数/秩计算 | Solver.cpp | 1h | SVD 分解 | P1 |
| 三对角求解 | Solver.cpp | 1h | - | P2 |
| 单元测试 | SolverTest.cpp | 4h | 全部 | P0 |
| 精度测试 | SolverAccuracyTest.cpp | 2h | 全部 | P1 |

**总计**: 约 27 小时

**实现顺序建议**:
1. P0 阶段 (基础): LU, Cholesky, QR, 小矩阵特化 (~12h)
2. P1 阶段 (完整): SVD, 伪逆, 条件数 (~9h)
3. P2 阶段 (优化): 三对角, 迭代精化 (~2h)
4. 测试: 单元测试 + 精度测试 (~4h)

---

## 10. 测试要点

### 10.1 单元测试覆盖

1. **LU 分解测试**
   - 可逆方阵
   - 奇异矩阵
   - 主元交换验证
   - 多右端项求解

2. **Cholesky 测试**
   - 正定矩阵
   - 非正定检测
   - 对称性验证

3. **QR 分解测试**
   - 方阵
   - 超定矩阵 (m > n)
   - 正交性验证: Q^T Q = I
   - 薄 QR vs 全 QR

4. **SVD 测试**
   - 满秩矩阵
   - 秩亏矩阵
   - 奇异值顺序 (降序)
   - 重建精度: A = U S V^T

5. **求解器测试**
   - 已知解系统
   - 最小二乘解验证
   - 齐次解验证: ||Ax|| 接近零
   - 伪逆属性: A A^+ A = A

### 10.2 精度测试用例

```cpp
// 示例: LU 分解精度
TEST(SolverAccuracy, LU_Precision) {
    MatX A = RandomMatrix(10, 10);
    LUResult lu = LU_Decompose(A);
    
    // 重建验证
    MatX PA = ApplyPermutation(A, lu.P);
    MatX LU = lu.L * lu.U;
    double error = (PA - LU).NormFrobenius() / A.NormFrobenius();
    EXPECT_LT(error, 1e-12);
}

// 示例: 最小二乘精度
TEST(SolverAccuracy, LeastSquares_Precision) {
    MatX A = RandomMatrix(100, 10);  // 超定
    VecX b = RandomVector(100);
    VecX x = SolveLeastSquares(A, b);
    
    // 验证法方程
    VecX ATAx = A.Transpose() * (A * x);
    VecX ATb = A.Transpose() * b;
    double error = (ATAx - ATb).Norm() / ATb.Norm();
    EXPECT_LT(error, 1e-10);
}
```

### 10.3 边界条件测试

- 1x1 矩阵
- 空矩阵
- 近奇异矩阵 (条件数 1e10)
- 零矩阵
- 单位矩阵
- 尺寸不匹配

### 10.4 性能基准

```cpp
// 基准测试不同规模
for (int n : {10, 50, 100, 500}) {
    MatX A = RandomMatrix(n, n);
    VecX b = RandomVector(n);
    
    BENCHMARK("LU_" + std::to_string(n)) {
        SolveLU(A, b);
    };
}
```

---

## 11. 示例用法

### 11.1 直线拟合 (最小二乘)

```cpp
// 拟合 y = ax + b
// 设计矩阵 A = [x1 1; x2 1; ...; xn 1]
// 解 A * [a; b] = y

std::vector<Point2d> points = ...;
int n = points.size();

MatX A(n, 2);
VecX y(n);
for (int i = 0; i < n; ++i) {
    A(i, 0) = points[i].x;
    A(i, 1) = 1.0;
    y[i] = points[i].y;
}

VecX params = SolveLeastSquares(A, y);
double a = params[0];  // 斜率
double b = params[1];  // 截距
```

### 11.2 单应性矩阵求解

```cpp
// 4 对点对应求单应性 H (3x3, 8 DOF)
// 使用 DLT 算法，构建 Ax = 0

MatX A(8, 9);  // 8 个方程，9 个未知数
// ... 填充 A ...

VecX h = SolveHomogeneous(A);  // 9 维单位向量

Mat33 H;
for (int i = 0; i < 9; ++i) {
    H(i/3, i%3) = h[i];
}
H = H / H(2,2);  // 归一化
```

### 11.3 相机内参标定

```cpp
// Zhang 标定法中的内参求解
// V^T b = 0，b = [B11, B12, B22, B13, B23, B33]^T

MatX V = ...;  // 从单应性构建
VecX b = SolveHomogeneous(V);

// 从 b 恢复 K
Mat33 K = RecoverIntrinsicsFromB(b);
```

### 11.4 多右端项复用分解

```cpp
// 求解多个右端项 AX = B
MatX A = ...;
MatX B = ...;  // 每列一个右端项

LUResult lu = LU_Decompose(A);  // 只分解一次
MatX X = SolveFromLU(lu, B);    // 多次求解
```

---

## 12. 线程安全

### 12.1 线程安全保证

| 函数类型 | 线程安全性 |
|----------|------------|
| 分解函数 | 可重入 (输入只读) |
| 求解函数 | 可重入 (输入只读) |
| SolveFrom* | 可重入 (分解结果只读) |

### 12.2 注意事项

- 所有函数无全局状态
- 临时内存在栈或函数内分配
- 可安全在多线程中并行调用

---

## 13. 未来扩展

1. **迭代精化**: 提高病态系统精度
2. **稀疏矩阵**: 用于大规模 Bundle Adjustment
3. **SIMD 优化**: 矩阵乘法 AVX2 版本
4. **GPU 加速**: 大矩阵 CUDA SVD
5. **带状矩阵**: 更高效的三对角/五对角求解
6. **Schur 补**: 分块矩阵求解优化

---

## 附录 A: 与 Halcon 对应

| QiVision | Halcon |
|----------|--------|
| SolveLU | hom_vector_to_proj_hom_mat2d (内部) |
| SolveLeastSquares | vector_to_hom_mat2d (最小二乘) |
| SolveHomogeneous | DLT 类算法内部使用 |
| ComputeConditionNumber | - (无直接对应) |
| PseudoInverse | - (无直接对应) |

---

## 附录 B: API 快速参考

```cpp
// 分解
LUResult lu = LU_Decompose(A);
CholeskyResult chol = Cholesky_Decompose(A);
QRResult qr = QR_Decompose(A);
SVDResult svd = SVD_Decompose(A);

// 直接求解
VecX x = SolveLU(A, b);           // 方阵
VecX x = SolveCholesky(A, b);     // SPD
VecX x = SolveLeastSquares(A, b); // 超定
VecX x = SolveSVD(A, b);          // 通用
VecX x = SolveHomogeneous(A);     // Ax=0

// 从分解求解
VecX x = SolveFromLU(lu, b);
VecX x = SolveFromQR(qr, b);
VecX x = SolveFromSVD(svd, b);

// 工具
double cond = ComputeConditionNumber(A);
int rank = ComputeRank(A);
MatX Ainv = PseudoInverse(A);
double res = ComputeResidual(A, x, b);

// 小矩阵
Vec2 x = Solve2x2(A2, b2);
Vec3 x = Solve3x3(A3, b3);
```
