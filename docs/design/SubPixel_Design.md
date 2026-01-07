# Internal/SubPixel 设计文档

## 1. 概述

### 1.1 功能描述

SubPixel 模块是 QiVision 的亚像素精化算法库，提供将整数像素位置精化到亚像素精度的核心算法。这是实现高精度机器视觉测量的关键模块，被 Matching、Measure、Calib 等 Feature 模块广泛依赖。

### 1.2 应用场景

- **模板匹配精化**: NCC/形状匹配响应曲面亚像素峰值定位
- **边缘检测**: 1D 边缘位置亚像素精化
- **角点检测**: Harris、Shi-Tomasi 角点亚像素精化
- **标定**: 标定板角点亚像素定位
- **测量**: Caliper 边缘位置亚像素精化

### 1.3 参考

**Halcon 算子**:
- `subpix_max_rect` - 矩形区域内亚像素极值定位
- `subpix_max_contour` - 轮廓上亚像素极值定位
- `edges_sub_pix` 内部的亚像素精化
- `find_shape_model` 的亚像素位置精化

**经典论文/方法**:
1. Parabolic (Quadratic) Fitting - 最常用的 1D/2D 亚像素方法
2. Gaussian Peak Fitting - 对高斯形状响应更准确
3. Center of Gravity (CoG/Centroid) - 快速、对称分布效果好
4. Taylor Expansion - 基于梯度的迭代精化
5. Equiangular Line Fitting - 边缘亚像素精化

### 1.4 设计原则

1. **高精度**: 亚像素定位精度 < 0.02px (标准条件)
2. **高效**: 避免不必要的内存分配，支持内联优化
3. **多方法**: 提供多种精化方法以适应不同场景
4. **鲁棒性**: 处理边界、退化情况
5. **复用已有模块**: 使用 Interpolate.h, Matrix.h, Solver.h

---

## 2. 设计规则验证

### 2.1 坐标类型符合规则

- [x] 整数像素坐标使用 `int32_t`
- [x] 亚像素坐标使用 `double`
- [x] 返回结构体使用 `double` 存储亚像素位置

### 2.2 层级依赖正确

- [x] SubPixel.h 位于 Internal 层
- [x] 依赖 Internal/Matrix.h (小矩阵运算)
- [x] 依赖 Internal/Solver.h (方程组求解)
- [x] 依赖 Internal/Interpolate.h (双线性/双三次插值)
- [x] 不依赖 Feature 层
- [x] 不跨层依赖 Platform 层

### 2.3 算法完整性

- [x] 1D 亚像素极值 (Parabolic, Gaussian, Centroid)
- [x] 2D 亚像素极值 (Quadratic surface fitting)
- [x] 亚像素边缘定位 (Gradient interpolation, Zero crossing)
- [x] 亚像素角点定位 (Quadratic surface, Gradient minimization)
- [x] 模板匹配精化 (Response surface fitting)
- [x] 质量评估 (精化置信度)

### 2.4 退化情况处理

- [x] 边界像素: 不进行精化或使用单侧拟合
- [x] 平坦区域: 返回原始位置，标记低置信度
- [x] 鞍点检测: 2D 精化检测是否为真正极值
- [x] 数值异常: 处理除零、矩阵奇异

---

## 3. 依赖分析

### 3.1 依赖的 Internal 模块

| 模块 | 用途 | 状态 |
|------|------|------|
| Internal/Interpolate.h | 双线性/双三次插值采样 | ✅ 已完成 |
| Internal/Matrix.h | Vec2, Vec3, Mat22, Mat33 小矩阵运算 | ✅ 已完成 |
| Internal/Solver.h | 2x2, 3x3 方程组求解 | ✅ 已完成 |

### 3.2 依赖的 Core 类型

| 类型 | 用途 |
|------|------|
| Core/Types.h | Point2d, Point2i |
| Core/Constants.h | 精度常量 EPSILON |

### 3.3 被依赖的模块

| 模块 | 用途 | 状态 |
|------|------|------|
| Matching/ShapeModel | 匹配结果亚像素精化 | 待设计 |
| Measure/Caliper | 边缘位置亚像素精化 | 待设计 |
| Edge/SubPixelEdge | 边缘亚像素精化 | 待设计 |
| Calib/* | 标定点亚像素定位 | 待设计 |
| Internal/NonMaxSuppression | 已有部分亚像素精化 | ✅ 已完成 |

---

## 4. 类设计

### 4.1 模块结构

```
SubPixel Module
├── Constants
│   ├── SUBPIXEL_PARABOLIC_CLAMP    - 抛物线拟合偏移限制
│   ├── SUBPIXEL_MIN_CURVATURE      - 最小曲率阈值
│   └── SUBPIXEL_GAUSSIAN_SIGMA     - 高斯拟合 sigma 范围
│
├── Enumerations
│   ├── SubPixelMethod1D            - 1D 精化方法
│   ├── SubPixelMethod2D            - 2D 精化方法
│   └── EdgeSubPixelMethod          - 边缘精化方法
│
├── Result Structures
│   ├── SubPixelResult1D            - 1D 精化结果
│   ├── SubPixelResult2D            - 2D 精化结果
│   └── SubPixelEdgeResult          - 边缘精化结果
│
├── 1D Subpixel Functions
│   ├── RefineSubPixel1D()          - 通用 1D 精化
│   ├── RefineParabolic1D()         - 抛物线拟合
│   ├── RefineGaussian1D()          - 高斯拟合
│   ├── RefineCentroid1D()          - 质心法
│   └── RefineQuartic1D()           - 四次多项式 (5点)
│
├── 2D Subpixel Functions
│   ├── RefineSubPixel2D()          - 通用 2D 精化
│   ├── RefineQuadratic2D()         - 二次曲面拟合 (3x3)
│   ├── RefineTaylor2D()            - Taylor 展开迭代
│   ├── RefineCentroid2D()          - 2D 质心
│   ├── RefineQuartic2D()           - 四次曲面 (5x5)
│   └── RefineCorner2D()            - 角点专用精化
│
├── Edge Subpixel Functions
│   ├── RefineEdgeSubPixel()        - 边缘位置精化
│   ├── RefineEdgeGradient()        - 梯度插值法
│   ├── RefineEdgeZeroCrossing()    - 二阶导零交叉
│   └── RefineEdgeParabolic()       - 梯度峰值抛物线
│
├── Match Refinement Functions
│   ├── RefineMatchSubPixel()       - 匹配结果精化
│   ├── RefineNCCSubPixel()         - NCC 响应精化
│   └── RefineShapeModelSubPixel()  - 形状匹配精化
│
└── Utility Functions
    ├── ComputeSubPixelConfidence() - 精化置信度评估
    ├── IsValidSubPixelResult()     - 结果有效性检查
    └── SubPixelResultToPoint2d()   - 结果转换
```

### 4.2 API 设计

```cpp
#pragma once

/**
 * @file SubPixel.h
 * @brief Subpixel refinement algorithms for QiVision
 *
 * This module provides:
 * - 1D subpixel peak/extremum localization
 * - 2D subpixel peak localization (response surfaces)
 * - Edge subpixel localization
 * - Corner subpixel refinement
 * - Template matching refinement
 *
 * Used by:
 * - Matching/ShapeModel: Match position refinement
 * - Measure/Caliper: Edge position refinement
 * - Edge/SubPixelEdge: Edge detection
 * - Calib/*: Calibration point localization
 *
 * Precision targets (standard conditions: contrast>=50, noise sigma<=5):
 * - 1D extremum: < 0.02 px (1 sigma)
 * - 2D peak: < 0.05 px (1 sigma)
 * - Edge position: < 0.02 px (1 sigma)
 *
 * Design principles:
 * - Pure functions, no global state
 * - Multiple methods for different scenarios
 * - Graceful degradation for edge cases
 * - Confidence estimation for quality assessment
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Internal/Matrix.h>
#include <QiVision/Internal/Interpolate.h>

#include <cmath>
#include <cstdint>
#include <vector>
#include <optional>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Maximum allowed subpixel offset from integer position (prevents runaway)
constexpr double SUBPIXEL_MAX_OFFSET = 0.5;

/// Minimum curvature for valid parabolic fit (prevents flat region false positives)
constexpr double SUBPIXEL_MIN_CURVATURE = 1e-6;

/// Default window half-size for centroid calculation
constexpr int32_t SUBPIXEL_CENTROID_HALF_WINDOW = 2;

/// Minimum contrast for subpixel edge refinement
constexpr double SUBPIXEL_EDGE_MIN_CONTRAST = 5.0;

/// Maximum iterations for iterative refinement methods
constexpr int32_t SUBPIXEL_MAX_ITERATIONS = 10;

/// Convergence tolerance for iterative methods
constexpr double SUBPIXEL_CONVERGENCE_TOLERANCE = 1e-6;

/// Gaussian fitting sigma lower bound
constexpr double GAUSSIAN_FIT_MIN_SIGMA = 0.5;

/// Gaussian fitting sigma upper bound
constexpr double GAUSSIAN_FIT_MAX_SIGMA = 10.0;

// =============================================================================
// Enumerations
// =============================================================================

/**
 * @brief 1D subpixel refinement methods
 */
enum class SubPixelMethod1D {
    Parabolic,      ///< Quadratic (parabolic) fitting [default, most common]
    Gaussian,       ///< Gaussian peak fitting (better for Gaussian-shaped peaks)
    Centroid,       ///< Center of gravity / centroid method (fast, symmetric)
    Quartic,        ///< 4th order polynomial fitting (5 points, higher accuracy)
    Linear          ///< Linear interpolation (for monotonic signals)
};

/**
 * @brief 2D subpixel refinement methods
 */
enum class SubPixelMethod2D {
    Quadratic,      ///< Quadratic surface (paraboloid) fitting [default]
    Taylor,         ///< Taylor expansion with gradient descent iteration
    Centroid,       ///< 2D center of gravity (fast)
    BiQuadratic,    ///< Bi-quadratic (4th order) surface fitting
    Gaussian2D      ///< 2D Gaussian fitting (expensive but accurate)
};

/**
 * @brief Edge subpixel refinement methods
 */
enum class EdgeSubPixelMethod {
    GradientInterp,     ///< Gradient interpolation (fast, robust)
    ZeroCrossing,       ///< Second derivative zero crossing (for step edges)
    ParabolicGradient,  ///< Parabolic fit on gradient profile [default]
    Moment              ///< First moment (centroid) of gradient
};

/**
 * @brief Corner subpixel refinement methods
 */
enum class CornerSubPixelMethod {
    GradientLeastSquares,   ///< Gradient-based least squares [default]
    TemplateMatching,       ///< Local template matching refinement
    QuadraticSurface        ///< Quadratic surface on corner response
};

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief 1D subpixel refinement result
 */
struct SubPixelResult1D {
    bool success = false;           ///< Whether refinement succeeded
    
    int32_t integerPosition = 0;    ///< Original integer position
    double subpixelPosition = 0.0;  ///< Refined subpixel position
    double offset = 0.0;            ///< Offset from integer position [-0.5, 0.5]
    
    double peakValue = 0.0;         ///< Interpolated value at subpixel position
    double curvature = 0.0;         ///< Local curvature (second derivative)
    double confidence = 0.0;        ///< Confidence score [0, 1]
    
    /// Get the subpixel position
    double Position() const { return subpixelPosition; }
    
    /// Check if result is valid and trustworthy
    bool IsValid(double minConfidence = 0.5) const {
        return success && confidence >= minConfidence && 
               std::abs(offset) <= SUBPIXEL_MAX_OFFSET;
    }
};

/**
 * @brief 2D subpixel refinement result
 */
struct SubPixelResult2D {
    bool success = false;           ///< Whether refinement succeeded
    
    int32_t integerX = 0;           ///< Original integer X
    int32_t integerY = 0;           ///< Original integer Y
    double subpixelX = 0.0;         ///< Refined subpixel X
    double subpixelY = 0.0;         ///< Refined subpixel Y
    double offsetX = 0.0;           ///< X offset from integer [-0.5, 0.5]
    double offsetY = 0.0;           ///< Y offset from integer [-0.5, 0.5]
    
    double peakValue = 0.0;         ///< Interpolated value at subpixel position
    double curvatureX = 0.0;        ///< Curvature in X direction
    double curvatureY = 0.0;        ///< Curvature in Y direction
    double curvatureMixed = 0.0;    ///< Mixed partial derivative (dxdy)
    double confidence = 0.0;        ///< Confidence score [0, 1]
    
    bool isSaddlePoint = false;     ///< True if detected as saddle point
    
    /// Get subpixel position as Point2d
    Point2d Position() const { return {subpixelX, subpixelY}; }
    
    /// Get offset as Point2d
    Point2d Offset() const { return {offsetX, offsetY}; }
    
    /// Check if result is valid and trustworthy
    bool IsValid(double minConfidence = 0.5) const {
        return success && !isSaddlePoint && confidence >= minConfidence &&
               std::abs(offsetX) <= SUBPIXEL_MAX_OFFSET &&
               std::abs(offsetY) <= SUBPIXEL_MAX_OFFSET;
    }
};

/**
 * @brief Edge subpixel refinement result
 */
struct SubPixelEdgeResult {
    bool success = false;           ///< Whether refinement succeeded
    
    double position = 0.0;          ///< Subpixel edge position (along profile)
    double gradient = 0.0;          ///< Gradient magnitude at edge
    double direction = 0.0;         ///< Gradient direction (radians)
    double amplitude = 0.0;         ///< Edge amplitude (intensity difference)
    double confidence = 0.0;        ///< Confidence score [0, 1]
    
    /// Check if result is valid
    bool IsValid(double minConfidence = 0.5) const {
        return success && confidence >= minConfidence;
    }
};

// =============================================================================
// 1D Subpixel Refinement Functions
// =============================================================================

/**
 * @brief Refine 1D extremum position using specified method
 *
 * @param signal Signal data array
 * @param size Signal length
 * @param index Integer position of extremum (local max/min)
 * @param method Refinement method
 * @param windowHalfSize Half window size for centroid method (default 2)
 * @return Subpixel refinement result
 *
 * @note For maximum detection, signal values should be positive peaks
 * @note For minimum detection, negate the signal first
 */
SubPixelResult1D RefineSubPixel1D(const double* signal, size_t size,
                                   int32_t index,
                                   SubPixelMethod1D method = SubPixelMethod1D::Parabolic,
                                   int32_t windowHalfSize = SUBPIXEL_CENTROID_HALF_WINDOW);

/**
 * @brief Refine 1D extremum using parabolic (quadratic) fit
 *
 * Fits a parabola to 3 points: (i-1, v0), (i, v1), (i+1, v2)
 * and finds the vertex.
 *
 * @param v0 Value at position i-1
 * @param v1 Value at position i (the peak)
 * @param v2 Value at position i+1
 * @return Subpixel offset from i (in range [-0.5, 0.5])
 *
 * Accuracy: < 0.02 px for symmetric peaks with reasonable SNR
 */
inline double RefineParabolic1D(double v0, double v1, double v2) {
    // Parabola: y = a*x^2 + b*x + c
    // At x=-1: v0 = a - b + c
    // At x=0:  v1 = c
    // At x=1:  v2 = a + b + c
    // => a = (v0 + v2)/2 - v1
    // => b = (v2 - v0)/2
    // Vertex at x = -b/(2a)
    
    double denom = 2.0 * (v0 - 2.0 * v1 + v2);
    if (std::abs(denom) < SUBPIXEL_MIN_CURVATURE) {
        return 0.0;  // Flat region, no refinement
    }
    
    double offset = (v0 - v2) / denom;
    
    // Clamp to prevent runaway extrapolation
    return std::clamp(offset, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);
}

/**
 * @brief Compute interpolated value at parabolic peak
 *
 * @param v0 Value at position i-1
 * @param v1 Value at position i
 * @param v2 Value at position i+1
 * @param offset Offset from RefineParabolic1D
 * @return Interpolated peak value
 */
inline double ParabolicPeakValue(double v0, double v1, double v2, double offset) {
    double a = (v0 + v2) * 0.5 - v1;
    double b = (v2 - v0) * 0.5;
    return a * offset * offset + b * offset + v1;
}

/**
 * @brief Compute local curvature (second derivative)
 *
 * @param v0 Value at position i-1
 * @param v1 Value at position i
 * @param v2 Value at position i+1
 * @return Curvature (negative for maximum, positive for minimum)
 */
inline double ComputeCurvature1D(double v0, double v1, double v2) {
    return v0 - 2.0 * v1 + v2;
}

/**
 * @brief Refine 1D extremum using Gaussian peak fitting
 *
 * Assumes peak has Gaussian shape: y = A * exp(-x^2 / (2*sigma^2))
 * Fits log(y) as a parabola.
 *
 * @param signal Signal data
 * @param size Signal length
 * @param index Peak position
 * @return SubPixelResult1D with sigma stored in curvature field
 */
SubPixelResult1D RefineGaussian1D(const double* signal, size_t size, int32_t index);

/**
 * @brief Refine 1D extremum using centroid (center of gravity)
 *
 * Computes weighted centroid of values in window around peak.
 * Fast and robust for symmetric, well-separated peaks.
 *
 * @param signal Signal data
 * @param size Signal length
 * @param index Peak position
 * @param halfWindow Half window size (full window = 2*halfWindow + 1)
 * @param useAbsValues If true, use absolute values as weights
 * @return SubPixelResult1D
 */
SubPixelResult1D RefineCentroid1D(const double* signal, size_t size,
                                   int32_t index, int32_t halfWindow = 2,
                                   bool useAbsValues = false);

/**
 * @brief Refine 1D extremum using quartic (4th order) polynomial
 *
 * Fits 5 points to 4th order polynomial for higher accuracy.
 * Requires index to be at least 2 away from boundaries.
 *
 * @param signal Signal data
 * @param size Signal length
 * @param index Peak position
 * @return SubPixelResult1D
 */
SubPixelResult1D RefineQuartic1D(const double* signal, size_t size, int32_t index);

// =============================================================================
// 2D Subpixel Refinement Functions
// =============================================================================

/**
 * @brief Refine 2D extremum position using specified method
 *
 * @param data 2D data array (row-major)
 * @param width Image width
 * @param height Image height
 * @param x Integer X position
 * @param y Integer Y position
 * @param method Refinement method
 * @return SubPixelResult2D
 */
SubPixelResult2D RefineSubPixel2D(const float* data, int32_t width, int32_t height,
                                   int32_t x, int32_t y,
                                   SubPixelMethod2D method = SubPixelMethod2D::Quadratic);

/**
 * @brief Refine 2D extremum using double data type
 */
SubPixelResult2D RefineSubPixel2D(const double* data, int32_t width, int32_t height,
                                   int32_t x, int32_t y,
                                   SubPixelMethod2D method = SubPixelMethod2D::Quadratic);

/**
 * @brief Refine 2D extremum using quadratic (paraboloid) surface fit
 *
 * Fits z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f to 3x3 neighborhood.
 * Finds the extremum by solving the gradient equation.
 *
 * @tparam T Pixel data type (float or double)
 * @param data Image data
 * @param width Image width
 * @param height Image height
 * @param x Center X
 * @param y Center Y
 * @return SubPixelResult2D
 *
 * Accuracy: < 0.05 px for smooth response surfaces
 */
template<typename T>
SubPixelResult2D RefineQuadratic2D(const T* data, int32_t width, int32_t height,
                                    int32_t x, int32_t y);

/**
 * @brief Refine 2D extremum using Taylor expansion iteration
 *
 * Iteratively refines position using gradient and Hessian.
 * More accurate for non-symmetric response surfaces.
 *
 * @tparam T Pixel data type
 * @param data Image data
 * @param width Image width
 * @param height Image height
 * @param x Initial X
 * @param y Initial Y
 * @param maxIterations Maximum iterations
 * @param tolerance Convergence tolerance
 * @return SubPixelResult2D
 */
template<typename T>
SubPixelResult2D RefineTaylor2D(const T* data, int32_t width, int32_t height,
                                 int32_t x, int32_t y,
                                 int32_t maxIterations = SUBPIXEL_MAX_ITERATIONS,
                                 double tolerance = SUBPIXEL_CONVERGENCE_TOLERANCE);

/**
 * @brief Refine 2D extremum using centroid
 *
 * @tparam T Pixel data type
 * @param data Image data
 * @param width Image width
 * @param height Image height
 * @param x Center X
 * @param y Center Y
 * @param halfWindow Half window size
 * @return SubPixelResult2D
 */
template<typename T>
SubPixelResult2D RefineCentroid2D(const T* data, int32_t width, int32_t height,
                                   int32_t x, int32_t y,
                                   int32_t halfWindow = SUBPIXEL_CENTROID_HALF_WINDOW);

/**
 * @brief Refine 2D corner position using gradient-based method
 *
 * Uses the structure tensor and gradient constraints to refine corner position.
 * Based on: sum_window (grad dot (p - corner)) = 0
 *
 * @tparam T Pixel data type
 * @param data Image data
 * @param width Image width
 * @param height Image height
 * @param x Corner X
 * @param y Corner Y
 * @param windowSize Window size for gradient accumulation
 * @param maxIterations Maximum iterations
 * @return SubPixelResult2D
 */
template<typename T>
SubPixelResult2D RefineCorner2D(const T* data, int32_t width, int32_t height,
                                 int32_t x, int32_t y,
                                 int32_t windowSize = 5,
                                 int32_t maxIterations = SUBPIXEL_MAX_ITERATIONS);

// =============================================================================
// Edge Subpixel Refinement Functions
// =============================================================================

/**
 * @brief Refine edge position in 1D profile
 *
 * @param profile 1D intensity profile perpendicular to edge
 * @param size Profile length
 * @param edgeIndex Approximate integer edge position
 * @param method Refinement method
 * @return SubPixelEdgeResult
 */
SubPixelEdgeResult RefineEdgeSubPixel(const double* profile, size_t size,
                                       int32_t edgeIndex,
                                       EdgeSubPixelMethod method = EdgeSubPixelMethod::ParabolicGradient);

/**
 * @brief Refine edge using gradient interpolation
 *
 * Finds subpixel position where gradient equals a specific value
 * between two integer positions.
 *
 * @param g0 Gradient at position i
 * @param g1 Gradient at position i+1
 * @param targetGradient Target gradient value (typically peak gradient)
 * @return Offset from position i [0, 1]
 */
inline double RefineEdgeGradient(double g0, double g1, double targetGradient) {
    double denom = g1 - g0;
    if (std::abs(denom) < 1e-10) {
        return 0.5;  // Linear interpolation midpoint
    }
    double offset = (targetGradient - g0) / denom;
    return std::clamp(offset, 0.0, 1.0);
}

/**
 * @brief Refine edge using second derivative zero crossing
 *
 * Finds subpixel position where second derivative crosses zero.
 * Optimal for ideal step edges.
 *
 * @param profile Intensity profile
 * @param size Profile length
 * @param edgeIndex Approximate edge position
 * @return SubPixelEdgeResult
 */
SubPixelEdgeResult RefineEdgeZeroCrossing(const double* profile, size_t size,
                                           int32_t edgeIndex);

/**
 * @brief Refine edge using parabolic fit on gradient peak
 *
 * Fits parabola to gradient profile around peak and finds maximum.
 *
 * @param gradient Gradient profile
 * @param size Profile length
 * @param peakIndex Gradient peak position
 * @return SubPixelEdgeResult
 */
SubPixelEdgeResult RefineEdgeParabolic(const double* gradient, size_t size,
                                        int32_t peakIndex);

// =============================================================================
// Template Matching Subpixel Refinement
// =============================================================================

/**
 * @brief Refine template match position
 *
 * Refines the position of a template match result using the response surface.
 *
 * @param response Match response/score image
 * @param width Response image width
 * @param height Response image height
 * @param x Match X position
 * @param y Match Y position
 * @param method 2D refinement method
 * @return SubPixelResult2D
 */
SubPixelResult2D RefineMatchSubPixel(const float* response, int32_t width, int32_t height,
                                      int32_t x, int32_t y,
                                      SubPixelMethod2D method = SubPixelMethod2D::Quadratic);

/**
 * @brief Refine NCC (Normalized Cross-Correlation) match position
 *
 * Specialized refinement for NCC response surfaces which have known properties.
 *
 * @param nccResponse NCC response image (values in [-1, 1])
 * @param width Image width
 * @param height Image height
 * @param x Match X
 * @param y Match Y
 * @return SubPixelResult2D
 */
SubPixelResult2D RefineNCCSubPixel(const float* nccResponse, int32_t width, int32_t height,
                                    int32_t x, int32_t y);

// =============================================================================
// Angle Subpixel Refinement
// =============================================================================

/**
 * @brief Refine angle in angle-response lookup
 *
 * For shape matching where response is stored for discrete angles.
 *
 * @param responses Response values for consecutive angles
 * @param numAngles Number of angles
 * @param angleStep Angle step in radians
 * @param bestIndex Index of best response
 * @return Refined angle in radians
 */
double RefineAngleSubPixel(const double* responses, size_t numAngles,
                           double angleStep, int32_t bestIndex);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Compute confidence score for 1D subpixel result
 *
 * Based on curvature, SNR, and offset magnitude.
 *
 * @param curvature Second derivative at peak
 * @param peakValue Peak value
 * @param backgroundValue Estimated background level
 * @param offset Subpixel offset
 * @return Confidence score [0, 1]
 */
double ComputeSubPixelConfidence1D(double curvature, double peakValue,
                                    double backgroundValue, double offset);

/**
 * @brief Compute confidence score for 2D subpixel result
 *
 * @param curvatureX X curvature
 * @param curvatureY Y curvature
 * @param curvatureMixed Mixed curvature
 * @param peakValue Peak value
 * @param offsetX X offset
 * @param offsetY Y offset
 * @return Confidence score [0, 1]
 */
double ComputeSubPixelConfidence2D(double curvatureX, double curvatureY,
                                    double curvatureMixed, double peakValue,
                                    double offsetX, double offsetY);

/**
 * @brief Check if 2D Hessian indicates a true maximum
 *
 * @param hxx Second derivative in X
 * @param hyy Second derivative in Y
 * @param hxy Mixed second derivative
 * @return true if local maximum, false if minimum, saddle, or degenerate
 */
inline bool IsLocalMaximum2D(double hxx, double hyy, double hxy) {
    double det = hxx * hyy - hxy * hxy;
    return det > 0 && hxx < 0;  // Negative definite Hessian
}

/**
 * @brief Check if 2D Hessian indicates a saddle point
 *
 * @param hxx Second derivative in X
 * @param hyy Second derivative in Y
 * @param hxy Mixed second derivative
 * @return true if saddle point
 */
inline bool IsSaddlePoint2D(double hxx, double hyy, double hxy) {
    double det = hxx * hyy - hxy * hxy;
    return det < 0;  // Indefinite Hessian
}

/**
 * @brief Sample 3x3 neighborhood from 2D array
 *
 * @tparam T Data type
 * @param data 2D data array
 * @param width Image width
 * @param height Image height
 * @param x Center X
 * @param y Center Y
 * @param values Output 9 values [NW, N, NE, W, C, E, SW, S, SE]
 * @return true if all values are valid (not at boundary)
 */
template<typename T>
bool Sample3x3(const T* data, int32_t width, int32_t height,
               int32_t x, int32_t y, double values[9]);

/**
 * @brief Compute 2D gradient at a point using central differences
 *
 * @tparam T Data type
 * @param data Image data
 * @param width Image width
 * @param height Image height
 * @param x Point X
 * @param y Point Y
 * @param dx Output gradient X
 * @param dy Output gradient Y
 */
template<typename T>
void ComputeGradient2D(const T* data, int32_t width, int32_t height,
                       int32_t x, int32_t y, double& dx, double& dy);

/**
 * @brief Compute 2D Hessian at a point
 *
 * @tparam T Data type
 * @param data Image data
 * @param width Image width
 * @param height Image height
 * @param x Point X
 * @param y Point Y
 * @param hxx Output second derivative XX
 * @param hyy Output second derivative YY
 * @param hxy Output mixed derivative XY
 */
template<typename T>
void ComputeHessian2D(const T* data, int32_t width, int32_t height,
                      int32_t x, int32_t y, double& hxx, double& hyy, double& hxy);

// =============================================================================
// Template Implementations
// =============================================================================

template<typename T>
bool Sample3x3(const T* data, int32_t width, int32_t height,
               int32_t x, int32_t y, double values[9]) {
    // Check bounds
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        return false;
    }
    
    int32_t idx = y * width + x;
    values[0] = static_cast<double>(data[idx - width - 1]);  // NW
    values[1] = static_cast<double>(data[idx - width]);      // N
    values[2] = static_cast<double>(data[idx - width + 1]);  // NE
    values[3] = static_cast<double>(data[idx - 1]);          // W
    values[4] = static_cast<double>(data[idx]);              // C
    values[5] = static_cast<double>(data[idx + 1]);          // E
    values[6] = static_cast<double>(data[idx + width - 1]);  // SW
    values[7] = static_cast<double>(data[idx + width]);      // S
    values[8] = static_cast<double>(data[idx + width + 1]);  // SE
    
    return true;
}

template<typename T>
void ComputeGradient2D(const T* data, int32_t width, int32_t height,
                       int32_t x, int32_t y, double& dx, double& dy) {
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        dx = dy = 0.0;
        return;
    }
    
    int32_t idx = y * width + x;
    dx = 0.5 * (static_cast<double>(data[idx + 1]) - static_cast<double>(data[idx - 1]));
    dy = 0.5 * (static_cast<double>(data[idx + width]) - static_cast<double>(data[idx - width]));
}

template<typename T>
void ComputeHessian2D(const T* data, int32_t width, int32_t height,
                      int32_t x, int32_t y, double& hxx, double& hyy, double& hxy) {
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        hxx = hyy = hxy = 0.0;
        return;
    }
    
    int32_t idx = y * width + x;
    double c = static_cast<double>(data[idx]);
    double w = static_cast<double>(data[idx - 1]);
    double e = static_cast<double>(data[idx + 1]);
    double n = static_cast<double>(data[idx - width]);
    double s = static_cast<double>(data[idx + width]);
    double nw = static_cast<double>(data[idx - width - 1]);
    double ne = static_cast<double>(data[idx - width + 1]);
    double sw = static_cast<double>(data[idx + width - 1]);
    double se = static_cast<double>(data[idx + width + 1]);
    
    hxx = w - 2.0 * c + e;
    hyy = n - 2.0 * c + s;
    hxy = 0.25 * (se - sw - ne + nw);
}

template<typename T>
SubPixelResult2D RefineQuadratic2D(const T* data, int32_t width, int32_t height,
                                    int32_t x, int32_t y) {
    SubPixelResult2D result;
    result.integerX = x;
    result.integerY = y;
    result.subpixelX = static_cast<double>(x);
    result.subpixelY = static_cast<double>(y);
    
    // Sample 3x3 neighborhood
    double v[9];
    if (!Sample3x3(data, width, height, x, y, v)) {
        result.success = false;
        result.confidence = 0.0;
        return result;
    }
    
    // Compute gradient and Hessian using 3x3 values
    // v[0]=NW v[1]=N  v[2]=NE
    // v[3]=W  v[4]=C  v[5]=E
    // v[6]=SW v[7]=S  v[8]=SE
    
    double dx = 0.5 * (v[5] - v[3]);  // (E - W) / 2
    double dy = 0.5 * (v[7] - v[1]);  // (S - N) / 2
    double hxx = v[3] - 2.0 * v[4] + v[5];  // W - 2C + E
    double hyy = v[1] - 2.0 * v[4] + v[7];  // N - 2C + S
    double hxy = 0.25 * (v[8] - v[6] - v[2] + v[0]);  // (SE - SW - NE + NW) / 4
    
    result.curvatureX = hxx;
    result.curvatureY = hyy;
    result.curvatureMixed = hxy;
    
    // Check for saddle point
    double det = hxx * hyy - hxy * hxy;
    if (det <= 0) {
        result.success = true;
        result.isSaddlePoint = (det < 0);
        result.confidence = 0.2;  // Low confidence for non-maximum
        result.peakValue = v[4];
        return result;
    }
    
    // Solve 2x2 system: H * offset = -gradient
    // [hxx hxy] [dx'] = [-dx]
    // [hxy hyy] [dy']   [-dy]
    double invDet = 1.0 / det;
    double offsetX = (hxy * dy - hyy * dx) * invDet;
    double offsetY = (hxy * dx - hxx * dy) * invDet;
    
    // Clamp offsets
    offsetX = std::clamp(offsetX, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);
    offsetY = std::clamp(offsetY, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);
    
    result.success = true;
    result.offsetX = offsetX;
    result.offsetY = offsetY;
    result.subpixelX = x + offsetX;
    result.subpixelY = y + offsetY;
    
    // Interpolate peak value
    result.peakValue = v[4] + 0.5 * (dx * offsetX + dy * offsetY);
    
    // Compute confidence based on curvature and offset
    result.confidence = ComputeSubPixelConfidence2D(hxx, hyy, hxy, v[4], offsetX, offsetY);
    
    return result;
}

template<typename T>
SubPixelResult2D RefineTaylor2D(const T* data, int32_t width, int32_t height,
                                 int32_t x, int32_t y,
                                 int32_t maxIterations, double tolerance) {
    SubPixelResult2D result;
    result.integerX = x;
    result.integerY = y;
    
    double currentX = static_cast<double>(x);
    double currentY = static_cast<double>(y);
    
    for (int32_t iter = 0; iter < maxIterations; ++iter) {
        // Get integer position for current estimate
        int32_t ix = static_cast<int32_t>(std::round(currentX));
        int32_t iy = static_cast<int32_t>(std::round(currentY));
        
        // Check bounds
        if (ix < 1 || ix >= width - 1 || iy < 1 || iy >= height - 1) {
            result.success = (iter > 0);
            result.subpixelX = currentX;
            result.subpixelY = currentY;
            return result;
        }
        
        // Compute gradient and Hessian at current position
        double dx, dy, hxx, hyy, hxy;
        ComputeGradient2D(data, width, height, ix, iy, dx, dy);
        ComputeHessian2D(data, width, height, ix, iy, hxx, hyy, hxy);
        
        // Newton step: delta = -H^(-1) * gradient
        double det = hxx * hyy - hxy * hxy;
        if (std::abs(det) < SUBPIXEL_MIN_CURVATURE) {
            break;  // Degenerate Hessian
        }
        
        double invDet = 1.0 / det;
        double deltaX = (hxy * dy - hyy * dx) * invDet;
        double deltaY = (hxy * dx - hxx * dy) * invDet;
        
        // Update position
        double newX = currentX + deltaX;
        double newY = currentY + deltaY;
        
        // Check convergence
        double change = std::sqrt(deltaX * deltaX + deltaY * deltaY);
        if (change < tolerance) {
            currentX = newX;
            currentY = newY;
            break;
        }
        
        currentX = newX;
        currentY = newY;
        
        // Check if we've moved too far from original position
        if (std::abs(currentX - x) > 1.0 || std::abs(currentY - y) > 1.0) {
            // Reset to quadratic result
            return RefineQuadratic2D(data, width, height, x, y);
        }
    }
    
    result.success = true;
    result.subpixelX = currentX;
    result.subpixelY = currentY;
    result.offsetX = currentX - x;
    result.offsetY = currentY - y;
    
    // Clamp final offsets
    if (std::abs(result.offsetX) > SUBPIXEL_MAX_OFFSET ||
        std::abs(result.offsetY) > SUBPIXEL_MAX_OFFSET) {
        return RefineQuadratic2D(data, width, height, x, y);
    }
    
    // Interpolate value at final position
    result.peakValue = InterpolateBilinear(data, width, height, currentX, currentY);
    
    // Get curvature at final integer position
    int32_t fx = static_cast<int32_t>(std::round(currentX));
    int32_t fy = static_cast<int32_t>(std::round(currentY));
    ComputeHessian2D(data, width, height, fx, fy,
                     result.curvatureX, result.curvatureY, result.curvatureMixed);
    
    double det = result.curvatureX * result.curvatureY - 
                 result.curvatureMixed * result.curvatureMixed;
    result.isSaddlePoint = (det < 0);
    
    result.confidence = ComputeSubPixelConfidence2D(
        result.curvatureX, result.curvatureY, result.curvatureMixed,
        result.peakValue, result.offsetX, result.offsetY);
    
    return result;
}

template<typename T>
SubPixelResult2D RefineCentroid2D(const T* data, int32_t width, int32_t height,
                                   int32_t x, int32_t y, int32_t halfWindow) {
    SubPixelResult2D result;
    result.integerX = x;
    result.integerY = y;
    
    // Check bounds for window
    if (x - halfWindow < 0 || x + halfWindow >= width ||
        y - halfWindow < 0 || y + halfWindow >= height) {
        result.success = false;
        result.subpixelX = static_cast<double>(x);
        result.subpixelY = static_cast<double>(y);
        result.confidence = 0.0;
        return result;
    }
    
    // Compute weighted centroid
    double sumX = 0.0, sumY = 0.0, sumW = 0.0;
    double centerValue = static_cast<double>(data[y * width + x]);
    
    for (int32_t dy = -halfWindow; dy <= halfWindow; ++dy) {
        for (int32_t dx = -halfWindow; dx <= halfWindow; ++dx) {
            double value = static_cast<double>(data[(y + dy) * width + (x + dx)]);
            // Use value relative to center as weight
            double weight = std::max(0.0, value);
            sumX += weight * (x + dx);
            sumY += weight * (y + dy);
            sumW += weight;
        }
    }
    
    if (sumW < 1e-10) {
        result.success = false;
        result.subpixelX = static_cast<double>(x);
        result.subpixelY = static_cast<double>(y);
        result.confidence = 0.0;
        return result;
    }
    
    result.success = true;
    result.subpixelX = sumX / sumW;
    result.subpixelY = sumY / sumW;
    result.offsetX = result.subpixelX - x;
    result.offsetY = result.subpixelY - y;
    result.peakValue = centerValue;
    
    // Clamp offsets
    if (std::abs(result.offsetX) > SUBPIXEL_MAX_OFFSET ||
        std::abs(result.offsetY) > SUBPIXEL_MAX_OFFSET) {
        result.offsetX = std::clamp(result.offsetX, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);
        result.offsetY = std::clamp(result.offsetY, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);
        result.subpixelX = x + result.offsetX;
        result.subpixelY = y + result.offsetY;
    }
    
    result.confidence = 0.7;  // Centroid typically has moderate confidence
    
    return result;
}

template<typename T>
SubPixelResult2D RefineCorner2D(const T* data, int32_t width, int32_t height,
                                 int32_t x, int32_t y,
                                 int32_t windowSize, int32_t maxIterations) {
    SubPixelResult2D result;
    result.integerX = x;
    result.integerY = y;
    
    int32_t halfWindow = windowSize / 2;
    
    // Check bounds
    if (x - halfWindow - 1 < 0 || x + halfWindow + 1 >= width ||
        y - halfWindow - 1 < 0 || y + halfWindow + 1 >= height) {
        result.success = false;
        result.subpixelX = static_cast<double>(x);
        result.subpixelY = static_cast<double>(y);
        return result;
    }
    
    double currentX = static_cast<double>(x);
    double currentY = static_cast<double>(y);
    
    for (int32_t iter = 0; iter < maxIterations; ++iter) {
        // Build structure tensor and gradient sum
        // Solve: sum_window (grad * grad^T) * corner = sum_window (grad * grad^T * p)
        double a11 = 0, a12 = 0, a22 = 0;
        double b1 = 0, b2 = 0;
        
        int32_t cx = static_cast<int32_t>(std::round(currentX));
        int32_t cy = static_cast<int32_t>(std::round(currentY));
        
        for (int32_t dy = -halfWindow; dy <= halfWindow; ++dy) {
            for (int32_t dx = -halfWindow; dx <= halfWindow; ++dx) {
                int32_t px = cx + dx;
                int32_t py = cy + dy;
                
                // Compute gradient at this point
                double gx, gy;
                ComputeGradient2D(data, width, height, px, py, gx, gy);
                
                // Accumulate structure tensor
                a11 += gx * gx;
                a12 += gx * gy;
                a22 += gy * gy;
                
                // Accumulate gradient-weighted position
                b1 += gx * gx * px + gx * gy * py;
                b2 += gx * gy * px + gy * gy * py;
            }
        }
        
        // Solve 2x2 system
        double det = a11 * a22 - a12 * a12;
        if (std::abs(det) < SUBPIXEL_MIN_CURVATURE) {
            break;  // Singular matrix, stop iteration
        }
        
        double invDet = 1.0 / det;
        double newX = (a22 * b1 - a12 * b2) * invDet;
        double newY = (a11 * b2 - a12 * b1) * invDet;
        
        // Check convergence
        double change = std::sqrt((newX - currentX) * (newX - currentX) +
                                  (newY - currentY) * (newY - currentY));
        currentX = newX;
        currentY = newY;
        
        if (change < SUBPIXEL_CONVERGENCE_TOLERANCE) {
            break;
        }
    }
    
    result.success = true;
    result.subpixelX = currentX;
    result.subpixelY = currentY;
    result.offsetX = currentX - x;
    result.offsetY = currentY - y;
    
    // Clamp if moved too far
    if (std::abs(result.offsetX) > 1.0 || std::abs(result.offsetY) > 1.0) {
        result.offsetX = std::clamp(result.offsetX, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);
        result.offsetY = std::clamp(result.offsetY, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);
        result.subpixelX = x + result.offsetX;
        result.subpixelY = y + result.offsetY;
        result.confidence = 0.5;
    } else {
        result.confidence = 0.9;
    }
    
    return result;
}

} // namespace Qi::Vision::Internal
```

---

## 5. 参数设计

### 5.1 精度控制常量

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| SUBPIXEL_MAX_OFFSET | double | 0.5 | [0.3, 1.0] | 最大偏移限制 |
| SUBPIXEL_MIN_CURVATURE | double | 1e-6 | [1e-8, 1e-4] | 最小曲率阈值 |
| SUBPIXEL_CENTROID_HALF_WINDOW | int32_t | 2 | [1, 5] | 质心窗口半径 |
| SUBPIXEL_EDGE_MIN_CONTRAST | double | 5.0 | [1.0, 20.0] | 边缘最小对比度 |
| SUBPIXEL_MAX_ITERATIONS | int32_t | 10 | [3, 20] | 最大迭代次数 |
| SUBPIXEL_CONVERGENCE_TOLERANCE | double | 1e-6 | [1e-8, 1e-4] | 收敛容差 |

### 5.2 方法选择指南

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 模板匹配响应峰值 | Quadratic 2D | 快速、稳定 |
| 高精度匹配 | Taylor 2D | 迭代精化更准确 |
| Harris 角点精化 | Corner 2D | 专用算法 |
| 边缘检测 | ParabolicGradient | 平衡精度和速度 |
| 步边缘 | ZeroCrossing | 理论最优 |
| 对称峰值 | Centroid 1D/2D | 快速、对称 |
| 高斯形状响应 | Gaussian 1D | 匹配峰值形状 |

---

## 6. 精度规格

### 6.1 标准条件下精度

| 方法 | 条件 | 精度要求 (1 sigma) |
|------|------|-------------------|
| Parabolic 1D | 对比度 >= 50, 噪声 sigma <= 5 | < 0.02 px |
| Gaussian 1D | 高斯形状峰值 | < 0.01 px |
| Centroid 1D | 对称峰值 | < 0.03 px |
| Quadratic 2D | 对比度 >= 50, 噪声 sigma <= 5 | < 0.05 px |
| Taylor 2D | 同上 | < 0.03 px |
| Edge Parabolic | 对比度 >= 50, 噪声 sigma <= 5 | < 0.02 px |

### 6.2 CLAUDE.md 精度验证

从 CLAUDE.md 精度规格:
- Edge1D Position: < 0.02px (1 sigma) - 对应 RefineEdgeParabolic
- ShapeModel Position: < 0.05px (1 sigma) - 对应 RefineQuadratic2D/RefineTaylor2D

### 6.3 精度影响因素

| 因素 | 影响 | 缓解措施 |
|------|------|----------|
| 噪声 | 增加定位误差 | 使用更大窗口、鲁棒方法 |
| 离散化 | 截断误差 | 使用高阶拟合 |
| 不对称峰值 | 系统偏差 | 使用 Taylor 迭代 |
| 边界效应 | 单侧拟合偏差 | 检测并标记低置信度 |

---

## 7. 算法要点

### 7.1 抛物线拟合 (1D)

```cpp
// 拟合 y = a*x^2 + b*x + c 到三点 (-1,v0), (0,v1), (1,v2)
// 解析解:
// c = v1
// a = (v0 + v2)/2 - v1
// b = (v2 - v0)/2
// 顶点 x = -b/(2a) = (v0 - v2) / (2*(v0 - 2*v1 + v2))
```

**精度分析**:
- 对于真正的抛物线形状，理论上精确
- 对于高斯形状，误差 < 0.01 px (对称峰)
- 对于非对称形状，有系统偏差

### 7.2 高斯拟合 (1D)

```cpp
// y = A * exp(-x^2 / (2*sigma^2))
// 取对数: ln(y) = ln(A) - x^2 / (2*sigma^2)
// 拟合 ln(y) = a*x^2 + c
// sigma = sqrt(-1/(2*a)), A = exp(c)

// 实际实现需要处理:
// - 负值 (取最大值归一化)
// - 极小值 (设置下限避免 log(0))
```

### 7.3 二次曲面拟合 (2D)

```cpp
// z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
// 使用 3x3 邻域求解

// 梯度: [dz/dx, dz/dy] = [2ax + cy + d, 2by + cx + e]
// 极值点: 梯度 = 0
// [2a  c ] [x]   [-d]
// [c   2b] [y] = [-e]

// 使用 3x3 样本计算:
// d = (E - W) / 2
// e = (S - N) / 2
// a = (W - 2C + E) / 2
// b = (N - 2C + S) / 2
// c = (SE - SW - NE + NW) / 4
```

### 7.4 Taylor 迭代 (2D)

```cpp
// Newton-Raphson 在图像上的应用
// x_{n+1} = x_n - H^(-1) * gradient
// 其中 H 是 Hessian 矩阵, gradient 是梯度

// 优点: 处理非对称响应更好
// 缺点: 需要多次迭代, 可能不收敛
```

### 7.5 边缘零交叉

```cpp
// 边缘位置在二阶导数零交叉处
// d2I/dx2 = I(x-1) - 2*I(x) + I(x+1)
// 零交叉点 = 线性插值两个符号相反的位置
```

### 7.6 角点精化

```cpp
// 基于梯度约束: (p - corner) dot gradient = 0
// 在窗口内累积:
// sum(gradient * gradient^T) * corner = sum(gradient * gradient^T * p)
// 即: A * corner = b, 其中 A 是结构张量
```

---

## 8. 与已有模块的关系

### 8.1 与 NonMaxSuppression.h 的关系

NonMaxSuppression.h 已包含部分亚像素精化功能:
- `RefineSubpixelParabolic()` - 1D 抛物线
- `RefineSubpixel2D()` - 2D 简化版
- `RefineSubpixel2DTaylor()` - 2D Taylor

SubPixel.h 提供:
- 更完整的方法集 (Gaussian, Centroid, Quartic)
- 更丰富的结果信息 (置信度, 曲率)
- 专门的边缘/角点/匹配精化
- 统一的接口和命名

**建议**: 保持 NonMaxSuppression.h 的简化内联函数用于性能关键路径，SubPixel.h 提供完整功能。

### 8.2 与 Interpolate.h 的关系

SubPixel.h 使用 Interpolate.h:
- 2D 位置的值插值 (Taylor 迭代)
- 梯度计算的平滑采样

### 8.3 与 Fitting.h 的关系

概念相似但应用不同:
- Fitting.h: 对点集拟合几何图形
- SubPixel.h: 对局部像素邻域精化位置

---

## 9. 实现任务分解

| 任务 | 文件 | 预估时间 | 依赖 | 优先级 |
|------|------|----------|------|--------|
| 头文件 API 定义 | SubPixel.h | 2h | Matrix.h, Interpolate.h | P0 |
| 1D 抛物线/高斯/质心 | SubPixel.cpp | 2h | - | P0 |
| 1D 四次多项式 | SubPixel.cpp | 1h | Solver.h | P1 |
| 2D 二次曲面拟合 | SubPixel.cpp | 2h | Matrix.h | P0 |
| 2D Taylor 迭代 | SubPixel.cpp | 2h | Interpolate.h | P0 |
| 2D 质心 | SubPixel.cpp | 1h | - | P0 |
| 2D 角点精化 | SubPixel.cpp | 2h | Gradient | P1 |
| 边缘精化方法 | SubPixel.cpp | 2h | - | P0 |
| 置信度计算 | SubPixel.cpp | 1h | - | P0 |
| 匹配精化函数 | SubPixel.cpp | 1h | 2D 精化 | P1 |
| 单元测试 | SubPixelTest.cpp | 4h | 全部 | P0 |
| 精度测试 | SubPixelAccuracyTest.cpp | 3h | 全部 | P0 |

**总计**: 约 23 小时

**实现顺序建议**:
1. P0 阶段 (核心): 头文件 + 1D 基础 + 2D 基础 + 边缘 (~14h)
2. P1 阶段 (完整): 高级方法 + 角点 + 匹配 (~5h)
3. 测试 (~7h)

---

## 10. 测试要点

### 10.1 单元测试覆盖

1. **1D 精化测试**
   - 合成抛物线峰值
   - 合成高斯峰值
   - 不同噪声水平
   - 边界情况

2. **2D 精化测试**
   - 合成二次曲面
   - 合成高斯斑
   - 鞍点检测
   - 不同邻域大小

3. **边缘精化测试**
   - 理想步边缘
   - 渐变边缘
   - 噪声边缘
   - 不同极性

4. **角点精化测试**
   - 合成棋盘格角点
   - 不同角度
   - 噪声环境

### 10.2 精度测试用例

```cpp
// 示例: 1D 抛物线精度
TEST(SubPixelAccuracy, Parabolic1D_IdealCondition) {
    // 生成精确已知位置的合成峰值
    std::vector<double> signal(100);
    double truePeak = 50.23;  // 真实峰值位置
    for (int i = 0; i < 100; ++i) {
        double x = i - truePeak;
        signal[i] = 100.0 - x * x;  // 下开抛物线
    }
    
    int peakIndex = static_cast<int>(std::round(truePeak));
    auto result = RefineSubPixel1D(signal.data(), signal.size(), 
                                    peakIndex, SubPixelMethod1D::Parabolic);
    
    EXPECT_TRUE(result.success);
    EXPECT_NEAR(result.subpixelPosition, truePeak, 0.001);  // 理想条件精度
}

// 示例: 2D 精化精度
TEST(SubPixelAccuracy, Quadratic2D_StandardCondition) {
    // 生成 50x50 高斯斑
    std::vector<float> image(50 * 50);
    Point2d trueCenter(25.37, 24.82);
    double sigma = 3.0;
    
    for (int y = 0; y < 50; ++y) {
        for (int x = 0; x < 50; ++x) {
            double dx = x - trueCenter.x;
            double dy = y - trueCenter.y;
            image[y * 50 + x] = 100.0f * std::exp(-(dx*dx + dy*dy) / (2*sigma*sigma));
        }
    }
    
    // 添加噪声
    AddGaussianNoise(image.data(), 50 * 50, 5.0);
    
    int peakX = 25, peakY = 25;
    auto result = RefineSubPixel2D(image.data(), 50, 50, peakX, peakY,
                                    SubPixelMethod2D::Quadratic);
    
    EXPECT_TRUE(result.success);
    double error = std::sqrt(std::pow(result.subpixelX - trueCenter.x, 2) +
                             std::pow(result.subpixelY - trueCenter.y, 2));
    EXPECT_LT(error, 0.05);  // < 0.05 px
}
```

### 10.3 边界条件测试

- 峰值在边界 (x=0 或 x=width-1)
- 平坦区域
- 单调信号
- 极端噪声
- 数值溢出

### 10.4 回归测试

- 保存标准测试数据集
- 比较不同版本结果
- 监控精度退化

---

## 11. 示例用法

### 11.1 模板匹配精化

```cpp
// 假设已找到匹配位置 (matchX, matchY)
float* nccResponse = ...;
int width = ..., height = ...;
int matchX = ..., matchY = ...;

auto result = RefineSubPixel2D(nccResponse, width, height, 
                                matchX, matchY,
                                SubPixelMethod2D::Quadratic);

if (result.IsValid()) {
    Point2d refinedPos = result.Position();
    double confidence = result.confidence;
    // 使用 refinedPos...
}
```

### 11.2 边缘位置精化

```cpp
// 1D profile 沿边缘法向采样
std::vector<double> profile = ExtractProfile(image, edgeLine);
std::vector<double> gradient = ComputeGradient(profile);

// 找到梯度峰值
int peakIdx = FindGradientPeak(gradient);

// 亚像素精化
auto result = RefineEdgeSubPixel(profile.data(), profile.size(),
                                  peakIdx, EdgeSubPixelMethod::ParabolicGradient);

if (result.IsValid()) {
    double edgePos = result.position;
    double edgeStrength = result.gradient;
    // 使用 edgePos...
}
```

### 11.3 角点精化

```cpp
// Harris 角点检测后
std::vector<Point2i> corners = DetectHarrisCorners(image);

std::vector<Point2d> refinedCorners;
for (const auto& corner : corners) {
    auto result = RefineCorner2D(image.Data(), image.Width(), image.Height(),
                                  corner.x, corner.y,
                                  5,   // window size
                                  10); // max iterations
    
    if (result.IsValid()) {
        refinedCorners.push_back(result.Position());
    }
}
```

---

## 12. 线程安全

### 12.1 线程安全保证

| 函数类型 | 线程安全性 |
|----------|------------|
| 所有精化函数 | 可重入 (输入只读) |
| 结果结构体 | 值类型，线程隔离 |

### 12.2 无全局状态

- 所有函数为纯函数
- 无静态变量
- 无缓存状态

---

## 13. 未来扩展

1. **SIMD 优化**: 批量精化的向量化
2. **GPU 加速**: 大量点的并行精化
3. **自适应方法选择**: 根据局部特性自动选择方法
4. **不确定度估计**: 基于 Fisher 信息矩阵
5. **亚像素角度精化**: 对旋转角度的精化
6. **各向异性缩放精化**: 对缩放参数的精化

---

## 附录 A: 与 Halcon 对应

| QiVision | Halcon |
|----------|--------|
| RefineQuadratic2D | subpix_max_rect (内部) |
| RefineEdgeParabolic | edges_sub_pix (内部) |
| RefineCorner2D | corners_sub_pix |
| RefineMatchSubPixel | find_shape_model (亚像素精化阶段) |

---

## 附录 B: API 快速参考

```cpp
// 1D 精化
SubPixelResult1D result = RefineSubPixel1D(signal, size, index, method);
double offset = RefineParabolic1D(v0, v1, v2);  // inline
double value = ParabolicPeakValue(v0, v1, v2, offset);  // inline

// 2D 精化
SubPixelResult2D result = RefineSubPixel2D(data, width, height, x, y, method);
SubPixelResult2D result = RefineQuadratic2D(data, width, height, x, y);
SubPixelResult2D result = RefineTaylor2D(data, width, height, x, y);
SubPixelResult2D result = RefineCentroid2D(data, width, height, x, y);
SubPixelResult2D result = RefineCorner2D(data, width, height, x, y);

// 边缘精化
SubPixelEdgeResult result = RefineEdgeSubPixel(profile, size, edgeIdx, method);
SubPixelEdgeResult result = RefineEdgeZeroCrossing(profile, size, edgeIdx);
SubPixelEdgeResult result = RefineEdgeParabolic(gradient, size, peakIdx);

// 匹配精化
SubPixelResult2D result = RefineMatchSubPixel(response, width, height, x, y);
SubPixelResult2D result = RefineNCCSubPixel(nccResponse, width, height, x, y);

// 工具函数
bool isMax = IsLocalMaximum2D(hxx, hyy, hxy);
bool isSaddle = IsSaddlePoint2D(hxx, hyy, hxy);
double confidence = ComputeSubPixelConfidence2D(...);
```
