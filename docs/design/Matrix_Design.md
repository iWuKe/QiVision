# Internal/Matrix 设计文档

## 1. 概述

### 1.1 功能描述
Matrix 模块是 QiVision 的轻量级小矩阵运算库，专为工业视觉应用设计。提供固定大小和动态大小矩阵的基本运算、分解和特殊矩阵构造功能。

### 1.2 应用场景
- **几何拟合**: 直线、圆、椭圆拟合的法方程求解 (ATA, ATb)
- **相机标定**: 内外参矩阵计算、畸变校正
- **单应性变换**: 3x3 矩阵操作
- **仿射变换**: 2x3 矩阵操作
- **特征值分解**: 2x2, 3x3 对称矩阵 (PCA, 协方差分析)
- **坐标变换**: 旋转矩阵、刚体变换

### 1.3 参考
- Halcon: `tuple_matrix_*` 系列算子
- Eigen 库设计思路 (不依赖)
- LAPACK 数值精度标准

### 1.4 设计原则
1. **小矩阵优化**: 2x2, 3x3, 4x4 使用栈内存和编译期优化
2. **动态矩阵支持**: 支持拟合问题 (Nx3, Nx6 等)
3. **无外部依赖**: 完全自实现，不依赖 Eigen/LAPACK
4. **数值稳定**: 使用 double 精度，处理退化情况
5. **与现有类协调**: 补充而非替代 QMatrix (QHomMat2d)

---

## 2. 设计规则验证

### 2.1 坐标类型符合规则
- [x] 矩阵元素使用 `double` (符合亚像素坐标精度要求)
- [x] 索引使用 `int32_t` (支持大矩阵)
- [x] 尺寸使用 `size_t` (与 STL 一致)

### 2.2 层级依赖正确
- [x] Matrix.h 位于 Internal 层
- [x] 只依赖 Platform 层 (Memory.h)
- [x] 不依赖 Feature 层
- [x] 被 Solver.h, Eigen.h, Fitting.h, Homography.h 使用

### 2.3 算法完整性
- [x] 基本运算 (加减乘除、转置、逆)
- [x] 分解准备 (LU, Cholesky, QR, SVD 的数据结构)
- [x] 特殊矩阵构造 (单位、旋转、缩放)
- [x] 范数计算

---

## 3. 依赖分析

### 3.1 依赖的 Platform 模块
| 模块 | 用途 | 状态 |
|------|------|------|
| Platform/Memory.h | 大矩阵对齐内存分配 | ✅ |

### 3.2 依赖的 Core 类型
| 类型 | 用途 |
|------|------|
| Core/Constants.h | 数值常量 (EPSILON, PI) |
| Core/Types.h | Point2d, Point3d (转换) |

### 3.3 被依赖的模块
| 模块 | 用途 | 状态 |
|------|------|------|
| Internal/Solver.h | 线性方程组求解 | ⬜ |
| Internal/Eigen.h | 特征值分解 | ⬜ |
| Internal/Fitting.h | 几何拟合 | ⬜ |
| Internal/Homography.h | 单应性计算 | ⬜ |
| Calib/* | 相机标定 | ⬜ |

---

## 4. 类设计

### 4.1 类型层次结构

```
Matrix Module
├── Vec<N>           - 固定大小向量 (2, 3, 4)
├── Mat<M,N>         - 固定大小矩阵 (2x2, 3x3, 4x4, 3x4)
├── VecX             - 动态大小向量
├── MatX             - 动态大小矩阵
└── Decomposition Results
    ├── LUResult
    ├── CholeskyResult
    ├── QRResult
    └── SVDResult
```

### 4.2 固定大小向量 Vec<N>

```cpp
namespace Qi::Vision::Internal {

/**
 * @brief 固定大小向量
 * @tparam N 向量维度 (2, 3, 4)
 * 
 * 栈内存存储，编译期优化。
 */
template<int N>
class Vec {
public:
    // 元素访问
    double& operator[](int i);
    const double& operator[](int i) const;
    double& operator()(int i);
    const double& operator()(int i) const;
    
    // 构造
    Vec();  // 零向量
    Vec(std::initializer_list<double> init);
    
    // 特化构造 (Vec2, Vec3, Vec4)
    // Vec<2>: Vec(double x, double y);
    // Vec<3>: Vec(double x, double y, double z);
    // Vec<4>: Vec(double x, double y, double z, double w);
    
    // 向量运算
    Vec operator+(const Vec& v) const;
    Vec operator-(const Vec& v) const;
    Vec operator*(double s) const;
    Vec operator/(double s) const;
    Vec& operator+=(const Vec& v);
    Vec& operator-=(const Vec& v);
    Vec& operator*=(double s);
    Vec& operator/=(double s);
    Vec operator-() const;  // 取负
    
    // 向量属性
    double Dot(const Vec& v) const;          // 点积
    double Norm() const;                      // L2 范数
    double NormSquared() const;               // L2 范数平方
    double NormL1() const;                    // L1 范数
    double NormInf() const;                   // 无穷范数
    Vec Normalized() const;                   // 单位化
    void Normalize();                         // 原地单位化
    
    // Vec<3> 特有
    Vec<3> Cross(const Vec<3>& v) const;      // 叉积 (仅 N=3)
    
    // 数据访问
    double* Data() { return data_; }
    const double* Data() const { return data_; }
    static constexpr int Size() { return N; }
    
    // 特殊构造
    static Vec Zero();
    static Vec Ones();
    static Vec Unit(int axis);  // 单位向量 (axis=0,1,...)
    
private:
    double data_[N];
};

// 类型别名
using Vec2 = Vec<2>;
using Vec3 = Vec<3>;
using Vec4 = Vec<4>;

// 与 Point2d/Point3d 的转换
Vec2 ToVec(const Point2d& p);
Vec3 ToVec(const Point3d& p);
Point2d ToPoint2d(const Vec2& v);
Point3d ToPoint3d(const Vec3& v);

} // namespace Qi::Vision::Internal
```

### 4.3 固定大小矩阵 Mat<M,N>

```cpp
namespace Qi::Vision::Internal {

/**
 * @brief 固定大小矩阵
 * @tparam M 行数
 * @tparam N 列数
 * 
 * 行主序存储，栈内存。
 * 常用: Mat<2,2>, Mat<3,3>, Mat<4,4>, Mat<3,4>
 */
template<int M, int N>
class Mat {
public:
    // 元素访问 (0-indexed)
    double& operator()(int row, int col);
    const double& operator()(int row, int col) const;
    
    // 行列访问
    Vec<N> Row(int i) const;
    Vec<M> Col(int j) const;
    void SetRow(int i, const Vec<N>& row);
    void SetCol(int j, const Vec<M>& col);
    
    // 构造
    Mat();  // 零矩阵
    Mat(std::initializer_list<double> init);  // 行主序
    explicit Mat(const double* data);  // 从数组
    
    // 矩阵运算
    Mat operator+(const Mat& m) const;
    Mat operator-(const Mat& m) const;
    Mat operator*(double s) const;
    Mat operator/(double s) const;
    Mat& operator+=(const Mat& m);
    Mat& operator-=(const Mat& m);
    Mat& operator*=(double s);
    Mat operator-() const;
    
    // 矩阵乘法
    template<int P>
    Mat<M,P> operator*(const Mat<N,P>& m) const;
    
    // 矩阵-向量乘法
    Vec<M> operator*(const Vec<N>& v) const;
    
    // 转置
    Mat<N,M> Transpose() const;
    
    // 方阵特有 (M == N)
    double Trace() const;                     // 迹
    double Determinant() const;               // 行列式
    Mat Inverse() const;                      // 逆矩阵
    bool IsInvertible(double eps = 1e-12) const;
    
    // 范数
    double NormFrobenius() const;             // Frobenius 范数
    double NormL1() const;                    // 列和范数 (最大列绝对值和)
    double NormInf() const;                   // 行和范数 (最大行绝对值和)
    
    // 数据访问
    double* Data() { return data_; }
    const double* Data() const { return data_; }
    static constexpr int Rows() { return M; }
    static constexpr int Cols() { return N; }
    
    // 特殊矩阵
    static Mat Zero();
    static Mat Identity();  // 仅方阵
    static Mat Diagonal(const Vec<(M<N?M:N)>& diag);
    
private:
    double data_[M * N];  // 行主序
};

// 类型别名
using Mat22 = Mat<2,2>;
using Mat33 = Mat<3,3>;
using Mat44 = Mat<4,4>;
using Mat23 = Mat<2,3>;  // 2D 仿射
using Mat34 = Mat<3,4>;  // 3D 投影

} // namespace Qi::Vision::Internal
```

### 4.4 动态大小向量 VecX

```cpp
namespace Qi::Vision::Internal {

/**
 * @brief 动态大小向量
 * 
 * 用于拟合问题的中间计算。
 * 小向量 (<=16) 使用栈内存，大向量使用堆内存。
 */
class VecX {
public:
    // 构造
    VecX();
    explicit VecX(int size);
    VecX(int size, double value);
    VecX(std::initializer_list<double> init);
    VecX(const VecX& other);
    VecX(VecX&& other) noexcept;
    VecX& operator=(const VecX& other);
    VecX& operator=(VecX&& other) noexcept;
    ~VecX();
    
    // 与固定大小转换
    template<int N>
    explicit VecX(const Vec<N>& v);
    template<int N>
    Vec<N> ToFixed() const;  // 需要 size == N
    
    // 元素访问
    double& operator[](int i);
    const double& operator[](int i) const;
    double& operator()(int i);
    const double& operator()(int i) const;
    
    // 向量运算 (与 Vec<N> 相同的接口)
    VecX operator+(const VecX& v) const;
    VecX operator-(const VecX& v) const;
    VecX operator*(double s) const;
    VecX operator/(double s) const;
    VecX& operator+=(const VecX& v);
    VecX& operator-=(const VecX& v);
    VecX& operator*=(double s);
    VecX operator-() const;
    
    // 向量属性
    double Dot(const VecX& v) const;
    double Norm() const;
    double NormSquared() const;
    VecX Normalized() const;
    void Normalize();
    
    // 尺寸
    int Size() const { return size_; }
    void Resize(int newSize);
    void SetZero();
    void SetOnes();
    void SetConstant(double value);
    
    // 数据访问
    double* Data();
    const double* Data() const;
    
    // 子向量
    VecX Segment(int start, int length) const;
    void SetSegment(int start, const VecX& segment);
    
    // 特殊构造
    static VecX Zero(int size);
    static VecX Ones(int size);
    static VecX LinSpace(double start, double end, int count);
    
private:
    static constexpr int STACK_SIZE = 16;
    int size_ = 0;
    double stackData_[STACK_SIZE];
    double* heapData_ = nullptr;
    
    bool UseHeap() const { return size_ > STACK_SIZE; }
};

} // namespace Qi::Vision::Internal
```

### 4.5 动态大小矩阵 MatX

```cpp
namespace Qi::Vision::Internal {

/**
 * @brief 动态大小矩阵
 * 
 * 用于拟合问题 (Nx3, Nx6 等)。
 * 行主序存储，大矩阵使用对齐内存。
 */
class MatX {
public:
    // 构造
    MatX();
    MatX(int rows, int cols);
    MatX(int rows, int cols, double value);
    MatX(const MatX& other);
    MatX(MatX&& other) noexcept;
    MatX& operator=(const MatX& other);
    MatX& operator=(MatX&& other) noexcept;
    ~MatX();
    
    // 与固定大小转换
    template<int M, int N>
    explicit MatX(const Mat<M,N>& m);
    template<int M, int N>
    Mat<M,N> ToFixed() const;  // 需要尺寸匹配
    
    // 元素访问
    double& operator()(int row, int col);
    const double& operator()(int row, int col) const;
    
    // 行列访问
    VecX Row(int i) const;
    VecX Col(int j) const;
    void SetRow(int i, const VecX& row);
    void SetCol(int j, const VecX& col);
    
    // 矩阵运算
    MatX operator+(const MatX& m) const;
    MatX operator-(const MatX& m) const;
    MatX operator*(double s) const;
    MatX operator/(double s) const;
    MatX& operator+=(const MatX& m);
    MatX& operator-=(const MatX& m);
    MatX& operator*=(double s);
    MatX operator-() const;
    
    // 矩阵乘法
    MatX operator*(const MatX& m) const;
    VecX operator*(const VecX& v) const;
    
    // 转置
    MatX Transpose() const;
    
    // 方阵特有
    double Trace() const;
    double Determinant() const;  // 仅小方阵
    MatX Inverse() const;        // 仅小方阵
    
    // 范数
    double NormFrobenius() const;
    
    // 尺寸
    int Rows() const { return rows_; }
    int Cols() const { return cols_; }
    void Resize(int rows, int cols);
    void SetZero();
    void SetIdentity();
    
    // 数据访问
    double* Data();
    const double* Data() const;
    int Stride() const { return cols_; }  // 行主序，stride = cols
    
    // 块操作
    MatX Block(int startRow, int startCol, int blockRows, int blockCols) const;
    void SetBlock(int startRow, int startCol, const MatX& block);
    
    // 特殊构造
    static MatX Zero(int rows, int cols);
    static MatX Identity(int size);
    static MatX Diagonal(const VecX& diag);
    
private:
    int rows_ = 0;
    int cols_ = 0;
    double* data_ = nullptr;  // 对齐分配
};

} // namespace Qi::Vision::Internal
```

### 4.6 分解结果结构

```cpp
namespace Qi::Vision::Internal {

/**
 * @brief LU 分解结果
 * PA = LU, 其中 P 是置换矩阵
 */
struct LUResult {
    MatX L;              // 下三角，对角线为 1
    MatX U;              // 上三角
    std::vector<int> P;  // 置换向量 (行交换)
    int sign;            // det(P) 的符号 (+1 或 -1)
    bool valid;          // 分解是否成功
    
    double Determinant() const;  // 从 U 计算行列式
};

/**
 * @brief Cholesky 分解结果
 * A = L * L^T (下三角) 或 A = U^T * U (上三角)
 */
struct CholeskyResult {
    MatX L;              // 下三角矩阵
    bool valid;          // 分解是否成功（矩阵需正定）
};

/**
 * @brief QR 分解结果
 * A = Q * R, 其中 Q 正交, R 上三角
 */
struct QRResult {
    MatX Q;              // 正交矩阵 (m x m 或 m x n)
    MatX R;              // 上三角矩阵
    bool valid;
    bool thinQR;         // true: Q 是 m x n, R 是 n x n
};

/**
 * @brief SVD 分解结果
 * A = U * S * V^T
 */
struct SVDResult {
    MatX U;              // 左奇异向量 (m x k)
    VecX S;              // 奇异值 (降序排列)
    MatX V;              // 右奇异向量 (n x k)
    bool valid;
    int rank;            // 数值秩 (非零奇异值数量)
    
    double Condition() const;  // 条件数 = S[0] / S[rank-1]
};

} // namespace Qi::Vision::Internal
```

### 4.7 特殊矩阵工厂函数

```cpp
namespace Qi::Vision::Internal {

// =========================================================================
// 2D 变换矩阵 (3x3 齐次坐标)
// =========================================================================

/**
 * @brief 2D 旋转矩阵 (3x3)
 * @param angle 旋转角度 (弧度)，逆时针为正
 */
Mat33 Rotation2D(double angle);

/**
 * @brief 2D 平移矩阵 (3x3)
 */
Mat33 Translation2D(double tx, double ty);
Mat33 Translation2D(const Vec2& t);

/**
 * @brief 2D 缩放矩阵 (3x3)
 */
Mat33 Scaling2D(double sx, double sy);
Mat33 Scaling2D(double s);  // 均匀缩放

/**
 * @brief 2D 仿射变换矩阵 (3x3)
 * 组合: 先缩放，后旋转，最后平移
 */
Mat33 Affine2D(double tx, double ty, double angle, double sx, double sy);

/**
 * @brief 从 QMatrix 转换
 */
Mat33 FromQMatrix(const QMatrix& qmat);

/**
 * @brief 转换为 QMatrix
 */
QMatrix ToQMatrix(const Mat33& mat);

// =========================================================================
// 3D 变换矩阵 (4x4 齐次坐标)
// =========================================================================

/**
 * @brief 3D 绕 X 轴旋转
 */
Mat44 RotationX(double angle);

/**
 * @brief 3D 绕 Y 轴旋转
 */
Mat44 RotationY(double angle);

/**
 * @brief 3D 绕 Z 轴旋转
 */
Mat44 RotationZ(double angle);

/**
 * @brief 3D 欧拉角旋转 (ZYX 顺序)
 * 等价于: Rz(yaw) * Ry(pitch) * Rx(roll)
 */
Mat44 RotationEulerZYX(double roll, double pitch, double yaw);

/**
 * @brief 3D 轴角旋转 (Rodrigues)
 * @param axis 旋转轴 (单位向量)
 * @param angle 旋转角度 (弧度)
 */
Mat44 RotationAxisAngle(const Vec3& axis, double angle);

/**
 * @brief 3D 平移矩阵 (4x4)
 */
Mat44 Translation3D(double tx, double ty, double tz);
Mat44 Translation3D(const Vec3& t);

/**
 * @brief 3D 缩放矩阵 (4x4)
 */
Mat44 Scaling3D(double sx, double sy, double sz);
Mat44 Scaling3D(double s);

// =========================================================================
// 3x3 旋转矩阵 (无齐次坐标)
// =========================================================================

/**
 * @brief 3x3 旋转矩阵 (欧拉角 ZYX)
 */
Mat33 Rotation3x3EulerZYX(double roll, double pitch, double yaw);

/**
 * @brief 3x3 旋转矩阵 (轴角)
 */
Mat33 Rotation3x3AxisAngle(const Vec3& axis, double angle);

/**
 * @brief 从旋转矩阵提取欧拉角 (ZYX)
 * @return Vec3(roll, pitch, yaw)
 */
Vec3 ExtractEulerZYX(const Mat33& R);

/**
 * @brief 从旋转矩阵提取轴角
 * @return (axis, angle)
 */
std::pair<Vec3, double> ExtractAxisAngle(const Mat33& R);

// =========================================================================
// 相机矩阵
// =========================================================================

/**
 * @brief 相机内参矩阵 K
 * K = [fx  0  cx]
 *     [0  fy  cy]
 *     [0   0   1]
 */
Mat33 CameraIntrinsic(double fx, double fy, double cx, double cy);

/**
 * @brief 投影矩阵 P = K * [R|t]
 */
Mat34 ProjectionMatrix(const Mat33& K, const Mat33& R, const Vec3& t);

} // namespace Qi::Vision::Internal
```

---

## 5. 参数设计

### 5.1 数值精度参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| MATRIX_EPSILON | double | 1e-12 | 矩阵元素比较精度 |
| MATRIX_SINGULAR_THRESHOLD | double | 1e-10 | 奇异矩阵判断阈值 |
| SVD_MAX_ITERATIONS | int | 100 | SVD 最大迭代次数 |

### 5.2 内存参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| VecX STACK_SIZE | int | 16 | VecX 栈内存阈值 |
| MATRIX_ALIGNMENT | size_t | 64 | 大矩阵内存对齐 (AVX512) |

---

## 6. 精度规格

### 6.1 基本运算精度

| 运算 | 条件 | 精度要求 |
|------|------|----------|
| 矩阵乘法 | 元素 |A|, |B| < 1e6 | 相对误差 < 1e-14 |
| 矩阵求逆 | cond(A) < 1e6 | 相对误差 < 1e-10 |
| 行列式 | 3x3, 4x4 | 相对误差 < 1e-14 |

### 6.2 分解精度

| 分解 | 条件 | 精度要求 |
|------|------|----------|
| LU | 元素 < 1e6 | ||PA - LU|| / ||A|| < 1e-12 |
| Cholesky | 正定, cond < 1e6 | ||A - LL^T|| / ||A|| < 1e-12 |
| QR | m x n, m >= n | ||A - QR|| / ||A|| < 1e-12 |
| SVD | 任意矩阵 | ||A - USV^T|| / ||A|| < 1e-10 |

### 6.3 数值稳定性

- LU 分解使用部分主元选择
- QR 分解使用 Householder 反射
- SVD 使用隐式 QR 迭代
- Cholesky 检测非正定情况

---

## 7. 算法要点

### 7.1 固定大小矩阵优化

**2x2 矩阵求逆**:
```cpp
// 直接公式，避免除法优化
det = a*d - b*c;
inv = [d, -b; -c, a] / det;
```

**3x3 矩阵求逆**:
```cpp
// 伴随矩阵法
cofactor[3][3];
det = a*(ei-fh) - b*(di-fg) + c*(dh-eg);
inv = transpose(cofactor) / det;
```

**4x4 矩阵求逆**:
```cpp
// 分块求逆或 LU 分解
// 对于刚体变换可优化: [R|t; 0|1]^-1 = [R^T|-R^T*t; 0|1]
```

### 7.2 动态矩阵内存管理

```cpp
// 小矩阵使用栈内存
static constexpr size_t STACK_THRESHOLD = 256;  // 256 doubles = 2KB

// 大矩阵使用对齐分配
data_ = (double*)Platform::AlignedAlloc(rows * cols * sizeof(double), 64);
```

### 7.3 矩阵乘法优化

**小矩阵**: 直接展开循环

**大矩阵**: 分块乘法 + 缓存优化
```cpp
// 分块大小选择
constexpr int BLOCK_SIZE = 32;  // 适合 L1 缓存

// 可选: 为 SIMD 准备数据布局
```

### 7.4 退化情况处理

| 情况 | 处理方式 |
|------|----------|
| 奇异矩阵求逆 | 返回零矩阵，设置 valid=false |
| Cholesky 非正定 | 返回 valid=false |
| SVD 不收敛 | 设置 valid=false |
| 尺寸不匹配 | 抛出 InvalidArgumentException |

---

## 8. 与现有模块的关系

### 8.1 与 QMatrix (Core层) 的关系

| QMatrix | Mat<3,3> |
|---------|----------|
| 表示 2D 仿射变换 | 通用 3x3 矩阵 |
| 6 个自由度 (2x3) | 9 个元素 |
| 高层 API | 底层运算 |
| 用于变换点 | 用于数学计算 |

**协作方式**:
- `FromQMatrix()` / `ToQMatrix()` 转换函数
- 标定等需要数学运算时使用 Mat33
- 用户 API 返回 QMatrix

### 8.2 与 Hessian.h 的关系

Hessian.h 中已有 `EigenDecompose2x2()` 函数:
```cpp
void EigenDecompose2x2(double a, double b, double c,
                       double& lambda1, double& lambda2,
                       double& nx, double& ny);
```

**设计决策**:
- Matrix.h 提供通用 Mat22 和基础分解
- Eigen.h 负责特征值分解 (包装/扩展 Hessian 中的实现)
- 避免代码重复

### 8.3 依赖图

```
Platform/Memory.h
       |
       v
Internal/Matrix.h
       |
       +---> Internal/Solver.h (LU/QR 求解)
       +---> Internal/Eigen.h (特征值分解)
       +---> Internal/Fitting.h (几何拟合)
       +---> Internal/Homography.h (单应性)
       +---> Calib/CameraCalib.h (相机标定)
```

---

## 9. 实现任务分解

| 任务 | 文件 | 预估时间 | 依赖 | 优先级 |
|------|------|----------|------|--------|
| Vec<N> 实现 | Matrix.h/cpp | 2h | - | P0 |
| Mat<M,N> 实现 | Matrix.h/cpp | 3h | Vec<N> | P0 |
| 2x2/3x3/4x4 特化 | Matrix.h/cpp | 2h | Mat<M,N> | P0 |
| VecX 实现 | Matrix.h/cpp | 2h | - | P0 |
| MatX 实现 | Matrix.h/cpp | 3h | VecX | P0 |
| 分解结果结构 | Matrix.h | 1h | MatX | P1 |
| 特殊矩阵工厂 | Matrix.h/cpp | 2h | Mat<M,N> | P1 |
| QMatrix 转换 | Matrix.h/cpp | 1h | Mat33 | P1 |
| 单元测试 | MatrixTest.cpp | 4h | 全部 | P0 |

**总计**: 约 20 小时

---

## 10. 测试要点

### 10.1 单元测试覆盖

1. **向量运算**
   - 加减乘除、点积、叉积
   - 范数计算、单位化
   - 边界情况 (零向量)

2. **矩阵运算**
   - 矩阵乘法 (尺寸组合)
   - 转置、行列式、迹
   - 求逆 (可逆/奇异)

3. **特殊矩阵**
   - 单位矩阵属性
   - 旋转矩阵正交性
   - 变换组合

4. **动态矩阵**
   - 内存分配/释放
   - 尺寸调整
   - 块操作

### 10.2 精度测试

```cpp
// 示例: 矩阵求逆精度测试
TEST(MatrixAccuracy, InversePrecision) {
    Mat33 A = RandomMatrix33();
    Mat33 Ainv = A.Inverse();
    Mat33 I = A * Ainv;
    
    double error = (I - Mat33::Identity()).NormFrobenius();
    EXPECT_LT(error, 1e-12);
}
```

### 10.3 边界条件

- 空矩阵/向量
- 尺寸不匹配
- 接近奇异的矩阵
- 非常大/非常小的元素值

---

## 11. 示例用法

### 11.1 几何拟合

```cpp
// 最小二乘直线拟合: y = ax + b
// 构建设计矩阵 A = [x1 1; x2 1; ... ; xn 1]
// 解法方程: (A^T * A) * [a; b] = A^T * y

MatX A(n, 2);
VecX y(n);
for (int i = 0; i < n; ++i) {
    A(i, 0) = points[i].x;
    A(i, 1) = 1.0;
    y[i] = points[i].y;
}

MatX ATA = A.Transpose() * A;
VecX ATy = A.Transpose() * y;
// 使用 Solver.h 解 ATA * params = ATy
```

### 11.2 坐标变换

```cpp
// 点变换
Mat33 T = Translation2D(10, 20);
Mat33 R = Rotation2D(PI / 4);
Mat33 transform = T * R;  // 先旋转后平移

Vec3 p = {5, 3, 1};  // 齐次坐标
Vec3 p_transformed = transform * p;
Point2d result = {p_transformed[0], p_transformed[1]};
```

### 11.3 相机投影

```cpp
Mat33 K = CameraIntrinsic(1000, 1000, 320, 240);
Mat33 R = Rotation3x3EulerZYX(0, 0, 0);
Vec3 t = {0, 0, 100};
Mat34 P = ProjectionMatrix(K, R, t);

// 投影 3D 点
Vec4 X = {10, 20, 50, 1};  // 齐次坐标
Vec3 x = P * X;  // 图像齐次坐标
Point2d pixel = {x[0]/x[2], x[1]/x[2]};
```

---

## 12. 未来扩展

1. **SIMD 优化**: 矩阵乘法 AVX2/AVX512 版本
2. **稀疏矩阵**: 用于大规模拟合问题
3. **GPU 加速**: 大矩阵 CUDA 计算
4. **更多分解**: 特征值分解、Schur 分解
5. **表达式模板**: 延迟求值优化

---

## 附录: API 快速参考

```cpp
// 向量
Vec3 v1(1, 2, 3);
Vec3 v2 = v1 * 2.0;
double d = v1.Dot(v2);
Vec3 c = v1.Cross(v2);

// 固定矩阵
Mat33 M = Mat33::Identity();
Mat33 R = Rotation2D(0.5);
Mat33 inv = M.Inverse();
double det = M.Determinant();

// 动态矩阵
MatX A(100, 3);
A(0, 0) = 1.0;
MatX B = A.Transpose() * A;

// 变换
QMatrix qm = ToQMatrix(R);
Mat33 m = FromQMatrix(qm);
```
