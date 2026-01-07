# Internal/ContourAnalysis 设计文档

## 1. 概述

### 1.1 功能描述

ContourAnalysis 模块是 QiVision Internal 层的轮廓几何属性分析核心库，提供轮廓的各类几何属性计算功能。与 ContourProcess（处理/变换）形成互补，ContourAnalysis 专注于分析/计算属性。

主要功能：
1. **基础属性** - 长度、面积、质心、周长
2. **曲率分析** - 点曲率、平均曲率、最大曲率、曲率直方图
3. **方向分析** - 主轴方向、椭圆拟合方向
4. **矩分析** - 几何矩、中心矩、归一化矩、Hu矩（形状不变量）
5. **形状描述符** - 圆度、紧凑度、凸度、伸长率、矩形度
6. **边界框** - 轴对齐边界框、最小外接矩形、最小外接圆
7. **凸性分析** - 凸包面积、凸缺陷点检测

### 1.2 应用场景

- **形状匹配**: 基于 Hu 矩进行形状相似度比较
- **缺陷检测**: 分析轮廓圆度、凸度变化
- **测量计量**: 计算轮廓面积、周长、方向
- **分类筛选**: 按几何属性筛选轮廓
- **对象识别**: 基于形状特征进行分类

### 1.3 参考 Halcon 算子

| Halcon 算子 | 功能 | 对应函数 |
|-------------|------|----------|
| length_xld | 轮廓长度 | ContourLength |
| area_center_xld | 面积和质心 | ContourAreaCenter |
| moments_xld | 几何矩 | ContourMoments |
| moments_region_central_invar | Hu不变矩 | ContourHuMoments |
| circularity_xld | 圆度 | ContourCircularity |
| compactness_xld | 紧凑度 | ContourCompactness |
| convexity_xld | 凸度 | ContourConvexity |
| eccentricity_xld | 偏心率/伸长率 | ContourEccentricity |
| rectangularity_xld | 矩形度 | ContourRectangularity |
| smallest_circle_xld | 最小外接圆 | ContourMinEnclosingCircle |
| smallest_rectangle1_xld | 轴对齐边界框 | ContourBoundingBox |
| smallest_rectangle2_xld | 最小面积矩形 | ContourMinAreaRect |
| orientation_xld | 主轴方向 | ContourOrientation |
| curvature_xld | 曲率 | ContourCurvature |
| convex_hull_xld | 凸包 | ContourConvexHull |

### 1.4 设计原则

1. **纯函数**: 输入轮廓不被修改
2. **高精度**: 保持亚像素精度 (double 类型)
3. **闭合/开放支持**: 区分闭合轮廓和开放轮廓的处理
4. **与 ContourProcess 分工明确**: 本模块只分析不变换
5. **复用现有模块**: 充分利用 Fitting、GeomConstruct 等已有功能

---

## 2. 设计规则验证

### 2.1 坐标类型符合规则

- [x] 所有坐标使用 `double` 类型 (亚像素精度)
- [x] 轮廓点使用 `ContourPoint` 结构
- [x] 面积、长度等测量值使用 `double`
- [x] 角度使用弧度制 (double)

### 2.2 层级依赖正确

- [x] ContourAnalysis.h 位于 Internal 层
- [x] 依赖 Core/QContour.h (轮廓数据结构)
- [x] 依赖 Core/Types.h (基础类型)
- [x] 依赖 Internal/Fitting.h (椭圆拟合)
- [x] 依赖 Internal/GeomConstruct.h (凸包、最小包围)
- [x] 依赖 Internal/ContourProcess.h (工具函数)
- [x] 不依赖 Feature 层
- [x] 不跨层依赖 Platform 层

### 2.3 算法完整性

- [x] 长度/面积/质心: 基础几何属性
- [x] 曲率: 点曲率、统计量
- [x] 矩: 完整矩体系 (几何矩、中心矩、归一化矩、Hu矩)
- [x] 形状描述符: 圆度、紧凑度、凸度、伸长率、矩形度
- [x] 边界: 轴对齐、最小矩形、最小圆
- [x] 凸性: 凸包、凸缺陷

### 2.4 退化情况处理

- [x] 空轮廓: 返回 0 或 NaN (视属性而定)
- [x] 单点轮廓: 返回退化值 (长度=0, 面积=0)
- [x] 两点轮廓: 返回线段属性
- [x] 开放轮廓计算面积: 返回有符号面积或警告
- [x] 数值退化: 除零保护

---

## 3. 依赖分析

### 3.1 依赖的 Internal 模块

| 模块 | 用途 | 状态 |
|------|------|------|
| Internal/Fitting.h | 椭圆拟合 (方向计算) | ✅ 已完成 |
| Internal/GeomConstruct.h | 凸包、最小包围 | ✅ 已完成 |
| Internal/ContourProcess.h | 工具函数 (弧长等) | ✅ 已完成 |
| Internal/Geometry2d.h | 几何计算辅助 | ✅ 已完成 |

### 3.2 依赖的 Core 类型

| 类型 | 用途 |
|------|------|
| Core/QContour.h | QContour, ContourPoint |
| Core/Types.h | Point2d, Rect2d, Circle2d, Ellipse2d, RotatedRect2d |
| Core/Constants.h | PI, EPSILON |

### 3.3 被依赖的模块 (预期)

| 模块 | 用途 | 状态 |
|------|------|------|
| Internal/ContourSelect | 按属性筛选轮廓 | ⬜ 待设计 |
| Feature/Blob | Blob 分析中的轮廓属性 | ⬜ 待设计 |
| Feature/Edge | 边缘检测结果分析 | ⬜ 待设计 |

---

## 4. 类设计

### 4.1 模块结构

```
ContourAnalysis Module
├── Basic Properties
│   ├── ContourLength()               - 轮廓长度
│   ├── ContourArea()                 - 面积 (闭合轮廓)
│   ├── ContourSignedArea()           - 有符号面积
│   ├── ContourPerimeter()            - 周长
│   ├── ContourCentroid()             - 几何质心
│   └── ContourAreaCenter()           - 面积和质心 (组合)
│
├── Curvature Analysis
│   ├── ComputeContourCurvature()     - 计算各点曲率
│   ├── ContourMeanCurvature()        - 平均曲率
│   ├── ContourMaxCurvature()         - 最大曲率
│   ├── ContourMinCurvature()         - 最小曲率
│   ├── ContourCurvatureStats()       - 曲率统计量
│   └── ContourCurvatureHistogram()   - 曲率直方图
│
├── Orientation
│   ├── ContourOrientation()          - 主轴方向 (基于惯性矩)
│   ├── ContourOrientationEllipse()   - 椭圆拟合方向
│   └── ContourPrincipalAxes()        - 主轴方向和长度
│
├── Moments
│   ├── ContourMoments()              - 几何矩 m_pq
│   ├── ContourCentralMoments()       - 中心矩 mu_pq
│   ├── ContourNormalizedMoments()    - 归一化中心矩 eta_pq
│   └── ContourHuMoments()            - Hu 不变矩 (7个)
│
├── Shape Descriptors
│   ├── ContourCircularity()          - 圆度 (4*pi*A/P^2)
│   ├── ContourCompactness()          - 紧凑度 (P^2/A)
│   ├── ContourConvexity()            - 凸度 (凸包周长/轮廓周长)
│   ├── ContourEccentricity()         - 偏心率/伸长率
│   ├── ContourRectangularity()       - 矩形度 (A/最小矩形面积)
│   ├── ContourSolidity()             - 实心度 (A/凸包面积)
│   ├── ContourExtent()               - 范围度 (A/边界框面积)
│   └── ContourAllDescriptors()       - 所有描述符
│
├── Bounding Geometry
│   ├── ContourBoundingBox()          - 轴对齐边界框
│   ├── ContourMinAreaRect()          - 最小面积外接矩形
│   ├── ContourMinEnclosingCircle()   - 最小外接圆
│   └── ContourMinEnclosingEllipse()  - 最小外接椭圆
│
├── Convexity Analysis
│   ├── ContourConvexHull()           - 凸包
│   ├── ContourConvexHullArea()       - 凸包面积
│   ├── IsContourConvex()             - 是否凸
│   └── ContourConvexityDefects()     - 凸缺陷
│
└── Comparison
    ├── MatchShapesHu()               - Hu矩形状匹配
    └── MatchShapesContour()          - 轮廓形状匹配
```

### 4.2 API 设计

```cpp
#pragma once

/**
 * @file ContourAnalysis.h
 * @brief Contour geometric property analysis for QiVision
 *
 * This module provides:
 * - Basic properties: length, area, centroid, perimeter
 * - Curvature analysis: point curvature, statistics, histogram
 * - Orientation: principal axis direction, ellipse fitting direction
 * - Moments: geometric, central, normalized, Hu invariant moments
 * - Shape descriptors: circularity, compactness, convexity, eccentricity, etc.
 * - Bounding geometry: AABB, min area rect, min enclosing circle/ellipse
 * - Convexity analysis: convex hull, convexity defects
 *
 * Used by:
 * - Internal/ContourSelect: Filter contours by properties
 * - Feature/Blob: Blob analysis
 * - Feature/Edge: Edge analysis
 *
 * Design principles:
 * - All functions are pure (input not modified)
 * - All coordinates use double for subpixel precision
 * - Distinguish between open and closed contours
 * - Area-based properties require closed contours
 */

#include <QiVision/Core/QContour.h>
#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>

#include <array>
#include <optional>
#include <vector>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Minimum points for area calculation
constexpr size_t MIN_POINTS_FOR_AREA = 3;

/// Minimum points for curvature calculation
constexpr size_t MIN_POINTS_FOR_CURVATURE = 3;

/// Minimum points for moment calculation
constexpr size_t MIN_POINTS_FOR_MOMENTS = 3;

/// Minimum points for convex hull
constexpr size_t MIN_POINTS_FOR_CONVEX_HULL = 3;

/// Default window size for curvature calculation
constexpr int32_t DEFAULT_CURVATURE_WINDOW = 5;

/// Curvature computation tolerance
constexpr double CURVATURE_TOLERANCE = 1e-10;

/// Moment computation tolerance
constexpr double MOMENT_TOLERANCE = 1e-15;

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief Area and centroid result
 */
struct AreaCenterResult {
    double area = 0.0;          ///< Signed area (positive=CCW)
    Point2d centroid;           ///< Geometric centroid
    bool valid = false;         ///< Whether result is valid
};

/**
 * @brief Curvature statistics
 */
struct CurvatureStats {
    double mean = 0.0;          ///< Mean curvature
    double stddev = 0.0;        ///< Standard deviation
    double min = 0.0;           ///< Minimum curvature
    double max = 0.0;           ///< Maximum curvature
    double median = 0.0;        ///< Median curvature
    size_t minIndex = 0;        ///< Index of minimum curvature point
    size_t maxIndex = 0;        ///< Index of maximum curvature point
};

/**
 * @brief Principal axes result
 */
struct PrincipalAxesResult {
    Point2d centroid;           ///< Centroid
    double angle = 0.0;         ///< Principal axis angle (radians)
    double majorLength = 0.0;   ///< Length along major axis
    double minorLength = 0.0;   ///< Length along minor axis
    Point2d majorAxis;          ///< Unit vector of major axis
    Point2d minorAxis;          ///< Unit vector of minor axis
    bool valid = false;
};

/**
 * @brief Geometric moments result (up to order 3)
 */
struct MomentsResult {
    // Raw moments m_pq = sum(x^p * y^q)
    double m00 = 0.0;           ///< Zeroth moment (area)
    double m10 = 0.0, m01 = 0.0;          ///< First moments
    double m20 = 0.0, m11 = 0.0, m02 = 0.0;   ///< Second moments
    double m30 = 0.0, m21 = 0.0, m12 = 0.0, m03 = 0.0; ///< Third moments
    
    /// Get centroid from moments
    Point2d Centroid() const {
        if (m00 < MOMENT_TOLERANCE) return {0, 0};
        return {m10 / m00, m01 / m00};
    }
};

/**
 * @brief Central moments result
 */
struct CentralMomentsResult {
    // Central moments mu_pq = sum((x-cx)^p * (y-cy)^q)
    double mu00 = 0.0;          ///< = m00
    double mu20 = 0.0, mu11 = 0.0, mu02 = 0.0;   ///< Second central moments
    double mu30 = 0.0, mu21 = 0.0, mu12 = 0.0, mu03 = 0.0; ///< Third central moments
    
    Point2d centroid;           ///< Centroid used
};

/**
 * @brief Normalized central moments result
 */
struct NormalizedMomentsResult {
    // Normalized moments eta_pq = mu_pq / mu00^((p+q)/2 + 1)
    double eta20 = 0.0, eta11 = 0.0, eta02 = 0.0;
    double eta30 = 0.0, eta21 = 0.0, eta12 = 0.0, eta03 = 0.0;
};

/**
 * @brief Hu invariant moments (7 values)
 */
struct HuMomentsResult {
    std::array<double, 7> hu = {};  ///< Hu moments h1-h7
    
    double& operator[](size_t i) { return hu[i]; }
    const double& operator[](size_t i) const { return hu[i]; }
};

/**
 * @brief All shape descriptors
 */
struct ShapeDescriptors {
    double circularity = 0.0;       ///< 4*pi*A/P^2, 1.0 for circle
    double compactness = 0.0;       ///< P^2/A
    double convexity = 0.0;         ///< Convex hull perimeter / contour perimeter
    double solidity = 0.0;          ///< Area / convex hull area
    double eccentricity = 0.0;      ///< sqrt(1 - (b/a)^2), 0 for circle
    double elongation = 0.0;        ///< 1 - minorAxis/majorAxis
    double rectangularity = 0.0;    ///< Area / min bounding rect area
    double extent = 0.0;            ///< Area / AABB area
    double aspectRatio = 0.0;       ///< Major axis / minor axis
    
    bool valid = false;
};

/**
 * @brief Convexity defect
 */
struct ConvexityDefect {
    size_t startIndex = 0;      ///< Start point index on contour
    size_t endIndex = 0;        ///< End point index on contour
    size_t deepestIndex = 0;    ///< Deepest point index on contour
    Point2d startPoint;         ///< Start point (on convex hull)
    Point2d endPoint;           ///< End point (on convex hull)
    Point2d deepestPoint;       ///< Deepest defect point
    double depth = 0.0;         ///< Defect depth (perpendicular distance)
};

// =============================================================================
// Curvature Methods
// =============================================================================

/**
 * @brief Method for curvature calculation
 */
enum class CurvatureMethod {
    ThreePoint,         ///< 3-point circle fitting (default)
    FivePoint,          ///< 5-point circle fitting (smoother)
    Derivative,         ///< Based on derivatives (k = |x'y'' - x''y'| / (x'^2+y'^2)^1.5)
    Regression          ///< Local polynomial regression
};

// =============================================================================
// Basic Property Functions
// =============================================================================

/**
 * @brief Compute contour length (arc length)
 *
 * For closed contours, includes the closing segment.
 *
 * @param contour Input contour
 * @return Total arc length (0 if empty or single point)
 */
double ContourLength(const QContour& contour);

/**
 * @brief Compute signed area of a closed contour
 *
 * Uses the shoelace formula: A = 0.5 * sum(x_i * y_{i+1} - x_{i+1} * y_i)
 * Positive area indicates counter-clockwise orientation.
 *
 * @param contour Input contour (should be closed)
 * @return Signed area (positive=CCW, negative=CW)
 *
 * @note For open contours, treats as closed by connecting last to first.
 *       Returns 0 for contours with fewer than 3 points.
 */
double ContourSignedArea(const QContour& contour);

/**
 * @brief Compute absolute area of a closed contour
 *
 * @param contour Input contour
 * @return Absolute area (always non-negative)
 */
double ContourArea(const QContour& contour);

/**
 * @brief Compute perimeter of contour
 *
 * For closed contours, same as length.
 * For open contours, same as length.
 *
 * @param contour Input contour
 * @return Perimeter
 */
double ContourPerimeter(const QContour& contour);

/**
 * @brief Compute geometric centroid of contour
 *
 * For closed contours: uses area-weighted centroid formula.
 * For open contours: uses simple average of points.
 *
 * @param contour Input contour
 * @return Centroid point
 */
Point2d ContourCentroid(const QContour& contour);

/**
 * @brief Compute area and centroid together (more efficient)
 *
 * @param contour Input contour
 * @return AreaCenterResult with area, centroid, and validity flag
 */
AreaCenterResult ContourAreaCenter(const QContour& contour);

// =============================================================================
// Curvature Analysis Functions
// =============================================================================

/**
 * @brief Compute curvature at each point of the contour
 *
 * Curvature k = 1/R where R is the radius of the osculating circle.
 * Positive curvature = left turn, negative = right turn.
 *
 * @param contour Input contour
 * @param method Curvature calculation method
 * @param windowSize Window size for smoothing (used by some methods)
 * @return Vector of curvatures (same size as contour points)
 *
 * @note Window size affects smoothness: larger = smoother but less local.
 */
std::vector<double> ComputeContourCurvature(const QContour& contour,
                                             CurvatureMethod method = CurvatureMethod::ThreePoint,
                                             int32_t windowSize = DEFAULT_CURVATURE_WINDOW);

/**
 * @brief Compute mean curvature of contour
 *
 * @param contour Input contour
 * @param method Curvature method
 * @return Mean absolute curvature
 */
double ContourMeanCurvature(const QContour& contour,
                            CurvatureMethod method = CurvatureMethod::ThreePoint);

/**
 * @brief Compute maximum absolute curvature
 *
 * @param contour Input contour
 * @param method Curvature method
 * @return Maximum absolute curvature value
 */
double ContourMaxCurvature(const QContour& contour,
                           CurvatureMethod method = CurvatureMethod::ThreePoint);

/**
 * @brief Compute minimum absolute curvature
 *
 * @param contour Input contour
 * @param method Curvature method
 * @return Minimum absolute curvature value (often near 0 for straight segments)
 */
double ContourMinCurvature(const QContour& contour,
                           CurvatureMethod method = CurvatureMethod::ThreePoint);

/**
 * @brief Compute comprehensive curvature statistics
 *
 * @param contour Input contour
 * @param method Curvature method
 * @return CurvatureStats with mean, stddev, min, max, median
 */
CurvatureStats ContourCurvatureStats(const QContour& contour,
                                      CurvatureMethod method = CurvatureMethod::ThreePoint);

/**
 * @brief Compute curvature histogram
 *
 * @param contour Input contour
 * @param numBins Number of histogram bins
 * @param minCurvature Minimum curvature for histogram range (auto if >= maxCurvature)
 * @param maxCurvature Maximum curvature for histogram range
 * @param method Curvature method
 * @return Histogram as vector of counts
 */
std::vector<int32_t> ContourCurvatureHistogram(const QContour& contour,
                                                int32_t numBins = 32,
                                                double minCurvature = 0.0,
                                                double maxCurvature = 0.0,
                                                CurvatureMethod method = CurvatureMethod::ThreePoint);

// =============================================================================
// Orientation Functions
// =============================================================================

/**
 * @brief Compute principal axis orientation
 *
 * Based on second-order central moments (covariance matrix eigenanalysis).
 * Returns angle of major axis from positive X-axis.
 *
 * @param contour Input contour
 * @return Angle in radians [-PI/2, PI/2]
 */
double ContourOrientation(const QContour& contour);

/**
 * @brief Compute orientation using ellipse fitting
 *
 * Fits an ellipse to the contour points and returns its orientation.
 *
 * @param contour Input contour
 * @return Angle in radians [-PI/2, PI/2], or 0 if fitting fails
 */
double ContourOrientationEllipse(const QContour& contour);

/**
 * @brief Compute principal axes with full information
 *
 * @param contour Input contour
 * @return PrincipalAxesResult with centroid, angle, axis lengths
 */
PrincipalAxesResult ContourPrincipalAxes(const QContour& contour);

// =============================================================================
// Moment Functions
// =============================================================================

/**
 * @brief Compute raw geometric moments
 *
 * m_pq = sum(x^p * y^q) for all contour points.
 * For closed contours, uses Green's theorem for exact area integration.
 *
 * @param contour Input contour
 * @return MomentsResult with m00, m10, m01, m20, m11, m02, m30, m21, m12, m03
 */
MomentsResult ContourMoments(const QContour& contour);

/**
 * @brief Compute central moments
 *
 * mu_pq = sum((x-cx)^p * (y-cy)^q) where (cx,cy) is centroid.
 * Translation invariant.
 *
 * @param contour Input contour
 * @return CentralMomentsResult
 */
CentralMomentsResult ContourCentralMoments(const QContour& contour);

/**
 * @brief Compute normalized central moments
 *
 * eta_pq = mu_pq / mu00^((p+q)/2 + 1)
 * Translation and scale invariant.
 *
 * @param contour Input contour
 * @return NormalizedMomentsResult
 */
NormalizedMomentsResult ContourNormalizedMoments(const QContour& contour);

/**
 * @brief Compute Hu invariant moments
 *
 * Seven moments invariant to translation, scale, and rotation.
 * h7 also has sign flip invariance under reflection.
 *
 * @param contour Input contour
 * @return HuMomentsResult with 7 Hu moments
 *
 * @note Formula based on Hu (1962):
 * h1 = eta20 + eta02
 * h2 = (eta20 - eta02)^2 + 4*eta11^2
 * h3 = (eta30 - 3*eta12)^2 + (3*eta21 - eta03)^2
 * h4 = (eta30 + eta12)^2 + (eta21 + eta03)^2
 * h5 = (eta30 - 3*eta12)(eta30 + eta12)[(eta30 + eta12)^2 - 3(eta21 + eta03)^2]
 *      + (3*eta21 - eta03)(eta21 + eta03)[3(eta30 + eta12)^2 - (eta21 + eta03)^2]
 * h6 = (eta20 - eta02)[(eta30 + eta12)^2 - (eta21 + eta03)^2]
 *      + 4*eta11*(eta30 + eta12)(eta21 + eta03)
 * h7 = (3*eta21 - eta03)(eta30 + eta12)[(eta30 + eta12)^2 - 3(eta21 + eta03)^2]
 *      - (eta30 - 3*eta12)(eta21 + eta03)[3(eta30 + eta12)^2 - (eta21 + eta03)^2]
 */
HuMomentsResult ContourHuMoments(const QContour& contour);

// =============================================================================
// Shape Descriptor Functions
// =============================================================================

/**
 * @brief Compute circularity (isoperimetric quotient)
 *
 * Circularity = 4 * PI * Area / Perimeter^2
 * Equals 1.0 for a perfect circle, < 1.0 for other shapes.
 *
 * @param contour Input contour (should be closed)
 * @return Circularity in [0, 1], or 0 if invalid
 */
double ContourCircularity(const QContour& contour);

/**
 * @brief Compute compactness
 *
 * Compactness = Perimeter^2 / Area
 * Minimum for circle (4*PI), larger for elongated/irregular shapes.
 *
 * @param contour Input contour
 * @return Compactness (>= 4*PI)
 */
double ContourCompactness(const QContour& contour);

/**
 * @brief Compute convexity
 *
 * Convexity = Convex hull perimeter / Contour perimeter
 * Equals 1.0 for convex shapes, < 1.0 for concave shapes.
 *
 * @param contour Input contour
 * @return Convexity in [0, 1]
 */
double ContourConvexity(const QContour& contour);

/**
 * @brief Compute solidity
 *
 * Solidity = Contour area / Convex hull area
 * Equals 1.0 for convex shapes, < 1.0 for concave shapes.
 *
 * @param contour Input contour
 * @return Solidity in [0, 1]
 */
double ContourSolidity(const QContour& contour);

/**
 * @brief Compute eccentricity
 *
 * Eccentricity = sqrt(1 - (minorAxis/majorAxis)^2)
 * Equals 0 for circle, approaches 1 for elongated shapes.
 *
 * @param contour Input contour
 * @return Eccentricity in [0, 1)
 */
double ContourEccentricity(const QContour& contour);

/**
 * @brief Compute elongation
 *
 * Elongation = 1 - minorAxis / majorAxis
 * Equals 0 for circle, approaches 1 for line.
 *
 * @param contour Input contour
 * @return Elongation in [0, 1)
 */
double ContourElongation(const QContour& contour);

/**
 * @brief Compute rectangularity
 *
 * Rectangularity = Area / MinAreaRect.Area
 * Equals 1.0 for rectangle, < 1.0 for other shapes.
 *
 * @param contour Input contour
 * @return Rectangularity in [0, 1]
 */
double ContourRectangularity(const QContour& contour);

/**
 * @brief Compute extent
 *
 * Extent = Area / BoundingBox.Area
 * Equals PI/4 for circle in AABB, 1.0 for axis-aligned rectangle.
 *
 * @param contour Input contour
 * @return Extent in [0, 1]
 */
double ContourExtent(const QContour& contour);

/**
 * @brief Compute aspect ratio
 *
 * AspectRatio = majorAxisLength / minorAxisLength
 * Equals 1.0 for circle, > 1.0 for elongated shapes.
 *
 * @param contour Input contour
 * @return Aspect ratio (>= 1.0)
 */
double ContourAspectRatio(const QContour& contour);

/**
 * @brief Compute all shape descriptors at once (more efficient)
 *
 * @param contour Input contour
 * @return ShapeDescriptors with all values
 */
ShapeDescriptors ContourAllDescriptors(const QContour& contour);

// =============================================================================
// Bounding Geometry Functions
// =============================================================================

/**
 * @brief Compute axis-aligned bounding box
 *
 * @param contour Input contour
 * @return Rect2d bounding box
 */
Rect2d ContourBoundingBox(const QContour& contour);

/**
 * @brief Compute minimum area enclosing rectangle
 *
 * Uses rotating calipers algorithm on convex hull.
 *
 * @param contour Input contour
 * @return RotatedRect2d, or nullopt if fewer than 3 points
 */
std::optional<RotatedRect2d> ContourMinAreaRect(const QContour& contour);

/**
 * @brief Compute minimum enclosing circle
 *
 * Uses Welzl's algorithm (expected O(n) time).
 *
 * @param contour Input contour
 * @return Circle2d, or nullopt if empty
 */
std::optional<Circle2d> ContourMinEnclosingCircle(const QContour& contour);

/**
 * @brief Compute minimum enclosing ellipse (approximate)
 *
 * Fits ellipse to contour using least squares.
 *
 * @param contour Input contour
 * @return Ellipse2d, or nullopt if fewer than 5 points
 */
std::optional<Ellipse2d> ContourMinEnclosingEllipse(const QContour& contour);

// =============================================================================
// Convexity Analysis Functions
// =============================================================================

/**
 * @brief Compute convex hull of contour points
 *
 * @param contour Input contour
 * @return QContour representing the convex hull (closed)
 */
QContour ContourConvexHull(const QContour& contour);

/**
 * @brief Compute convex hull area
 *
 * @param contour Input contour
 * @return Convex hull area
 */
double ContourConvexHullArea(const QContour& contour);

/**
 * @brief Check if contour is convex
 *
 * @param contour Input contour
 * @return true if all points form a convex polygon
 */
bool IsContourConvex(const QContour& contour);

/**
 * @brief Find convexity defects
 *
 * Convexity defects are regions where the contour deviates from its convex hull.
 *
 * @param contour Input contour
 * @param minDepth Minimum defect depth to report (pixels)
 * @return Vector of ConvexityDefect
 */
std::vector<ConvexityDefect> ContourConvexityDefects(const QContour& contour,
                                                      double minDepth = 1.0);

// =============================================================================
// Shape Comparison Functions
// =============================================================================

/**
 * @brief Compare two shapes using Hu moments
 *
 * Lower values indicate more similar shapes.
 *
 * @param contour1 First contour
 * @param contour2 Second contour
 * @param method Comparison method (1, 2, or 3)
 *               1: sum(|1/m1_i - 1/m2_i|)
 *               2: sum(|m1_i - m2_i|)
 *               3: max(|m1_i - m2_i| / |m1_i|)
 * @return Similarity measure (lower = more similar)
 */
double MatchShapesHu(const QContour& contour1, const QContour& contour2, int method = 1);

/**
 * @brief Compare two contours using shape context or Fourier descriptors
 *
 * @param contour1 First contour
 * @param contour2 Second contour
 * @return Similarity score in [0, 1] (1 = identical)
 */
double MatchShapesContour(const QContour& contour1, const QContour& contour2);

} // namespace Qi::Vision::Internal
```

---

## 5. 参数设计

### 5.1 常量

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| MIN_POINTS_FOR_AREA | size_t | 3 | - | 面积计算最小点数 |
| MIN_POINTS_FOR_CURVATURE | size_t | 3 | - | 曲率计算最小点数 |
| MIN_POINTS_FOR_MOMENTS | size_t | 3 | - | 矩计算最小点数 |
| DEFAULT_CURVATURE_WINDOW | int32_t | 5 | [3, 21] | 曲率计算窗口大小 |
| CURVATURE_TOLERANCE | double | 1e-10 | - | 曲率数值容差 |
| MOMENT_TOLERANCE | double | 1e-15 | - | 矩计算数值容差 |

### 5.2 输出值范围

| 属性 | 范围 | 说明 |
|------|------|------|
| Circularity | [0, 1] | 1.0 = 圆形 |
| Compactness | [4*PI, +inf) | 4*PI = 圆形最小值 |
| Convexity | [0, 1] | 1.0 = 凸形 |
| Solidity | [0, 1] | 1.0 = 凸形 |
| Eccentricity | [0, 1) | 0 = 圆形 |
| Elongation | [0, 1) | 0 = 圆形 |
| Rectangularity | [0, 1] | 1.0 = 矩形 |
| Extent | [0, 1] | PI/4 = 圆形 |
| AspectRatio | [1, +inf) | 1.0 = 圆形 |

---

## 6. 精度规格

### 6.1 面积精度

| 条件 | 指标 | 要求 |
|------|------|------|
| 规则多边形 | 面积误差 | < 0.01% |
| 圆形轮廓 (100点) | 面积误差 | < 0.1% |
| 亚像素轮廓 | 相对误差 | < 1e-10 |

### 6.2 矩精度

| 条件 | 指标 | 要求 |
|------|------|------|
| 几何矩 | 数值稳定性 | 双精度范围内 |
| Hu矩 | 旋转不变性 | < 1e-6 相对变化 |
| Hu矩 | 缩放不变性 | < 1e-6 相对变化 |

### 6.3 曲率精度

| 条件 | 指标 | 要求 |
|------|------|------|
| 圆形轮廓 | 曲率一致性 | 变化 < 5% |
| 已知曲率 | 绝对误差 | < 0.01 /像素 |

---

## 7. 算法要点

### 7.1 面积计算 (Shoelace Formula)

```cpp
double ContourSignedArea(const QContour& contour) {
    if (contour.Size() < MIN_POINTS_FOR_AREA) {
        return 0.0;
    }
    
    double area = 0.0;
    size_t n = contour.Size();
    
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        const auto& pi = contour[i];
        const auto& pj = contour[j];
        area += pi.x * pj.y - pj.x * pi.y;
    }
    
    return area * 0.5;
}
```

### 7.2 质心计算 (Area-weighted)

```cpp
Point2d ContourCentroid(const QContour& contour) {
    if (contour.Empty()) return {0, 0};
    if (contour.Size() < 3) {
        // 简单平均
        double sumX = 0, sumY = 0;
        for (size_t i = 0; i < contour.Size(); ++i) {
            sumX += contour[i].x;
            sumY += contour[i].y;
        }
        return {sumX / contour.Size(), sumY / contour.Size()};
    }
    
    double area = 0.0;
    double cx = 0.0, cy = 0.0;
    size_t n = contour.Size();
    
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        const auto& pi = contour[i];
        const auto& pj = contour[j];
        double cross = pi.x * pj.y - pj.x * pi.y;
        area += cross;
        cx += (pi.x + pj.x) * cross;
        cy += (pi.y + pj.y) * cross;
    }
    
    if (std::abs(area) < MOMENT_TOLERANCE) {
        // 退化情况：使用简单平均
        return /* 简单平均 */;
    }
    
    area *= 0.5;
    cx /= (6.0 * area);
    cy /= (6.0 * area);
    
    return {cx, cy};
}
```

### 7.3 曲率计算 (三点法)

```cpp
double ComputeCurvatureAtPoint(const Point2d& p0, const Point2d& p1, const Point2d& p2) {
    // 通过三点拟合圆，曲率 = 1/R
    // 使用面积公式: 2*Area = |cross(p1-p0, p2-p0)|
    // R = |p0-p1| * |p1-p2| * |p2-p0| / (4*Area)
    
    double dx1 = p1.x - p0.x, dy1 = p1.y - p0.y;
    double dx2 = p2.x - p1.x, dy2 = p2.y - p1.y;
    double dx3 = p0.x - p2.x, dy3 = p0.y - p2.y;
    
    double cross = dx1 * dy2 - dy1 * dx2;  // 2 * signed area
    
    double a = std::sqrt(dx1*dx1 + dy1*dy1);
    double b = std::sqrt(dx2*dx2 + dy2*dy2);
    double c = std::sqrt(dx3*dx3 + dy3*dy3);
    
    double denom = a * b * c;
    if (denom < CURVATURE_TOLERANCE) return 0.0;
    
    // 曲率 = 2 * cross / denom (带符号)
    return 2.0 * cross / denom;
}
```

### 7.4 Hu不变矩

```cpp
HuMomentsResult ContourHuMoments(const QContour& contour) {
    NormalizedMomentsResult nm = ContourNormalizedMoments(contour);
    HuMomentsResult result;
    
    double eta20 = nm.eta20, eta02 = nm.eta02, eta11 = nm.eta11;
    double eta30 = nm.eta30, eta21 = nm.eta21, eta12 = nm.eta12, eta03 = nm.eta03;
    
    // h1 = eta20 + eta02
    result.hu[0] = eta20 + eta02;
    
    // h2 = (eta20 - eta02)^2 + 4*eta11^2
    double diff20_02 = eta20 - eta02;
    result.hu[1] = diff20_02 * diff20_02 + 4.0 * eta11 * eta11;
    
    // h3 = (eta30 - 3*eta12)^2 + (3*eta21 - eta03)^2
    double t1 = eta30 - 3.0 * eta12;
    double t2 = 3.0 * eta21 - eta03;
    result.hu[2] = t1 * t1 + t2 * t2;
    
    // h4 = (eta30 + eta12)^2 + (eta21 + eta03)^2
    double s1 = eta30 + eta12;
    double s2 = eta21 + eta03;
    result.hu[3] = s1 * s1 + s2 * s2;
    
    // h5, h6, h7 更复杂，按公式计算
    double s1_sq = s1 * s1;
    double s2_sq = s2 * s2;
    
    // h5
    result.hu[4] = t1 * s1 * (s1_sq - 3.0*s2_sq) + t2 * s2 * (3.0*s1_sq - s2_sq);
    
    // h6
    result.hu[5] = diff20_02 * (s1_sq - s2_sq) + 4.0 * eta11 * s1 * s2;
    
    // h7 (skew invariant)
    result.hu[6] = t2 * s1 * (s1_sq - 3.0*s2_sq) - t1 * s2 * (3.0*s1_sq - s2_sq);
    
    return result;
}
```

### 7.5 形状描述符计算

```cpp
ShapeDescriptors ContourAllDescriptors(const QContour& contour) {
    ShapeDescriptors desc;
    
    if (contour.Size() < MIN_POINTS_FOR_AREA) {
        return desc;
    }
    
    // 基础属性
    double area = ContourArea(contour);
    double perimeter = ContourPerimeter(contour);
    
    if (area < MOMENT_TOLERANCE || perimeter < MOMENT_TOLERANCE) {
        return desc;
    }
    
    // 圆度
    desc.circularity = 4.0 * PI * area / (perimeter * perimeter);
    
    // 紧凑度
    desc.compactness = perimeter * perimeter / area;
    
    // 凸包相关
    QContour hull = ContourConvexHull(contour);
    double hullPerimeter = ContourPerimeter(hull);
    double hullArea = ContourArea(hull);
    
    desc.convexity = (hullPerimeter > 0) ? (hullPerimeter / perimeter) : 0.0;
    desc.solidity = (hullArea > 0) ? (area / hullArea) : 0.0;
    
    // 主轴相关
    PrincipalAxesResult axes = ContourPrincipalAxes(contour);
    if (axes.valid && axes.minorLength > 0) {
        double ratio = axes.minorLength / axes.majorLength;
        desc.eccentricity = std::sqrt(1.0 - ratio * ratio);
        desc.elongation = 1.0 - ratio;
        desc.aspectRatio = axes.majorLength / axes.minorLength;
    }
    
    // 矩形度
    auto minRect = ContourMinAreaRect(contour);
    if (minRect.has_value()) {
        double rectArea = minRect->Area();
        desc.rectangularity = (rectArea > 0) ? (area / rectArea) : 0.0;
    }
    
    // 范围度
    Rect2d bbox = ContourBoundingBox(contour);
    double bboxArea = bbox.Area();
    desc.extent = (bboxArea > 0) ? (area / bboxArea) : 0.0;
    
    desc.valid = true;
    return desc;
}
```

### 7.6 凸缺陷检测

```cpp
std::vector<ConvexityDefect> ContourConvexityDefects(const QContour& contour,
                                                      double minDepth) {
    std::vector<ConvexityDefect> defects;
    
    if (contour.Size() < MIN_POINTS_FOR_CONVEX_HULL) {
        return defects;
    }
    
    // 获取凸包点索引
    std::vector<Point2d> points = contour.GetPoints();
    std::vector<size_t> hullIndices = ConvexHullIndices(points);
    
    if (hullIndices.size() < 3) {
        return defects;
    }
    
    // 对每对相邻凸包点，找中间最深的凹陷
    for (size_t i = 0; i < hullIndices.size(); ++i) {
        size_t startIdx = hullIndices[i];
        size_t endIdx = hullIndices[(i + 1) % hullIndices.size()];
        
        // 确定遍历范围
        size_t begin = startIdx;
        size_t end = endIdx;
        if (end <= begin) end += contour.Size();
        
        Point2d startPt = contour.GetPoint(startIdx);
        Point2d endPt = contour.GetPoint(endIdx % contour.Size());
        
        // 找最深点
        double maxDist = 0.0;
        size_t deepestIdx = begin;
        
        for (size_t j = begin + 1; j < end; ++j) {
            size_t idx = j % contour.Size();
            Point2d pt = contour.GetPoint(idx);
            
            // 计算到线段的距离
            double dist = PointToSegmentDistance(pt, startPt, endPt);
            
            if (dist > maxDist) {
                maxDist = dist;
                deepestIdx = idx;
            }
        }
        
        if (maxDist >= minDepth) {
            ConvexityDefect defect;
            defect.startIndex = startIdx;
            defect.endIndex = endIdx % contour.Size();
            defect.deepestIndex = deepestIdx;
            defect.startPoint = startPt;
            defect.endPoint = endPt;
            defect.deepestPoint = contour.GetPoint(deepestIdx);
            defect.depth = maxDist;
            defects.push_back(defect);
        }
    }
    
    return defects;
}
```

---

## 8. 与现有模块的关系

### 8.1 与 Core/QContour 的关系

QContour 已有一些成员函数可以委托给 ContourAnalysis:

```cpp
// QContour.cpp 中可改为:
double QContour::Length() const {
    return Internal::ContourLength(*this);
}

double QContour::Area() const {
    return Internal::ContourArea(*this);
}

Point2d QContour::Centroid() const {
    return Internal::ContourCentroid(*this);
}

double QContour::Circularity() const {
    return Internal::ContourCircularity(*this);
}
```

### 8.2 与 Internal/Fitting 的关系

用于椭圆拟合以计算方向:

```cpp
double ContourOrientationEllipse(const QContour& contour) {
    if (contour.Size() < 5) return 0.0;
    
    std::vector<Point2d> points = contour.GetPoints();
    EllipseFitResult result = FitEllipseFitzgibbon(points);
    
    if (!result.success) return 0.0;
    return result.ellipse.angle;
}
```

### 8.3 与 Internal/GeomConstruct 的关系

用于凸包、最小包围等:

```cpp
QContour ContourConvexHull(const QContour& contour) {
    std::vector<Point2d> points = contour.GetPoints();
    std::vector<Point2d> hull = ConvexHull(points);
    return QContour(hull, true);
}

std::optional<Circle2d> ContourMinEnclosingCircle(const QContour& contour) {
    std::vector<Point2d> points = contour.GetPoints();
    return MinEnclosingCircle(points);
}

std::optional<RotatedRect2d> ContourMinAreaRect(const QContour& contour) {
    std::vector<Point2d> points = contour.GetPoints();
    return MinAreaRect(points);
}
```

### 8.4 与 Internal/ContourProcess 的关系

可复用工具函数:

```cpp
double ContourLength(const QContour& contour) {
    return ComputeContourLength(contour);  // 已在 ContourProcess 中实现
}
```

---

## 9. 实现任务分解

| 任务 | 文件 | 预估时间 | 依赖 | 优先级 |
|------|------|----------|------|--------|
| 头文件 API 定义 | ContourAnalysis.h | 2h | Types.h, QContour.h | P0 |
| 基础属性 (长度/面积/质心) | ContourAnalysis.cpp | 2h | - | P0 |
| 曲率计算 | ContourAnalysis.cpp | 3h | - | P0 |
| 曲率统计量 | ContourAnalysis.cpp | 1h | 曲率计算 | P0 |
| 几何矩 | ContourAnalysis.cpp | 2h | - | P0 |
| 中心矩/归一化矩 | ContourAnalysis.cpp | 1h | 几何矩 | P0 |
| Hu不变矩 | ContourAnalysis.cpp | 1h | 归一化矩 | P0 |
| 主轴方向 | ContourAnalysis.cpp | 1h | 中心矩 | P0 |
| 形状描述符 | ContourAnalysis.cpp | 2h | 基础属性 + 凸包 | P0 |
| 边界框/最小矩形/最小圆 | ContourAnalysis.cpp | 2h | GeomConstruct | P0 |
| 凸包/凸缺陷 | ContourAnalysis.cpp | 2h | GeomConstruct | P1 |
| 形状匹配 | ContourAnalysis.cpp | 2h | Hu矩 | P1 |
| 单元测试 | test_contour_analysis.cpp | 4h | 全部 | P0 |
| 精度测试 | accuracy_contour_analysis.cpp | 2h | 全部 | P1 |

**总计**: 约 27 小时

**实现顺序建议**:
1. P0 阶段: 头文件 + 基础属性 + 矩 + 主轴 + 形状描述符 + 边界 (~18h)
2. P1 阶段: 凸缺陷 + 形状匹配 (~4h)
3. 测试 (~6h)

---

## 10. 测试要点

### 10.1 单元测试覆盖

1. **基础属性测试**
   - 矩形轮廓: 面积 = width * height
   - 正多边形: 已知面积公式
   - 圆形轮廓: 面积 ≈ PI * r^2
   - 质心位置验证

2. **曲率测试**
   - 圆形: 曲率 ≈ 1/radius (常数)
   - 直线: 曲率 ≈ 0
   - 椭圆: 曲率变化范围

3. **矩测试**
   - 平移不变性 (Hu矩)
   - 旋转不变性 (Hu矩)
   - 缩放不变性 (Hu矩)
   - 已知形状的矩值

4. **形状描述符测试**
   - 圆形: circularity ≈ 1
   - 正方形: rectangularity ≈ 1
   - 凸形: convexity = 1

5. **边界几何测试**
   - AABB 包含所有点
   - 最小外接圆包含所有点
   - 最小矩形面积最小

### 10.2 边界条件测试

- 空轮廓
- 单点轮廓
- 两点轮廓
- 共线点
- 数值精度极限

### 10.3 精度测试

```cpp
// 示例: Hu矩旋转不变性
TEST(ContourAnalysisAccuracy, HuMomentsRotationInvariance) {
    QContour original = CreateTestContour();
    HuMomentsResult hu1 = ContourHuMoments(original);
    
    // 旋转45度
    QContour rotated = original.Rotate(PI / 4);
    HuMomentsResult hu2 = ContourHuMoments(rotated);
    
    for (int i = 0; i < 7; ++i) {
        EXPECT_NEAR(hu1[i], hu2[i], std::abs(hu1[i]) * 1e-6);
    }
}

// 示例: 圆形圆度
TEST(ContourAnalysisAccuracy, CircleCircularity) {
    QContour circle = QContour::FromCircle(Circle2d({100, 100}, 50), 100);
    double circularity = ContourCircularity(circle);
    EXPECT_NEAR(circularity, 1.0, 0.01);  // 误差 < 1%
}

// 示例: 矩形面积
TEST(ContourAnalysisAccuracy, RectangleArea) {
    QContour rect = QContour::FromRectangle(Rect2d(0, 0, 100, 50));
    double area = ContourArea(rect);
    EXPECT_NEAR(area, 5000.0, 0.001);
}
```

---

## 11. 线程安全

### 11.1 线程安全保证

| 函数类型 | 线程安全性 |
|----------|------------|
| 所有分析函数 | 可重入 (输入只读) |
| 结果值类型 | 线程隔离 |

### 11.2 无全局状态

- 所有函数为纯函数
- 无静态变量
- 无缓存状态

---

## 12. 未来扩展

1. **更多形状描述符**: 傅里叶描述符、形状上下文
2. **轮廓骨架**: 中轴线提取
3. **曲率平滑**: 带权重的曲率平滑
4. **点云支持**: 扩展到无序点云的形状分析
5. **3D扩展**: 3D轮廓分析

---

## 附录 A: 与 Halcon 对应

| QiVision | Halcon |
|----------|--------|
| ContourLength | length_xld |
| ContourArea | area_center_xld (area) |
| ContourCentroid | area_center_xld (center) |
| ContourMoments | moments_xld |
| ContourHuMoments | moments_region_central_invar |
| ContourCircularity | circularity_xld |
| ContourCompactness | compactness_xld |
| ContourConvexity | convexity_xld |
| ContourEccentricity | eccentricity_xld |
| ContourRectangularity | rectangularity_xld |
| ContourOrientation | orientation_xld |
| ComputeContourCurvature | curvature_xld |
| ContourBoundingBox | smallest_rectangle1_xld |
| ContourMinAreaRect | smallest_rectangle2_xld |
| ContourMinEnclosingCircle | smallest_circle_xld |
| ContourConvexHull | convex_hull_xld |
| MatchShapesHu | match_contours_xld_proj_hom |

---

## 附录 B: API 快速参考

```cpp
// 基础属性
double len = ContourLength(contour);
double area = ContourArea(contour);
double perimeter = ContourPerimeter(contour);
Point2d centroid = ContourCentroid(contour);
AreaCenterResult ac = ContourAreaCenter(contour);

// 曲率
std::vector<double> curvatures = ComputeContourCurvature(contour);
double meanCurv = ContourMeanCurvature(contour);
double maxCurv = ContourMaxCurvature(contour);
CurvatureStats stats = ContourCurvatureStats(contour);

// 方向
double angle = ContourOrientation(contour);
PrincipalAxesResult axes = ContourPrincipalAxes(contour);

// 矩
MomentsResult moments = ContourMoments(contour);
CentralMomentsResult cmom = ContourCentralMoments(contour);
HuMomentsResult hu = ContourHuMoments(contour);

// 形状描述符
double circ = ContourCircularity(contour);
double compact = ContourCompactness(contour);
double convex = ContourConvexity(contour);
double solid = ContourSolidity(contour);
double ecc = ContourEccentricity(contour);
double rect = ContourRectangularity(contour);
ShapeDescriptors desc = ContourAllDescriptors(contour);

// 边界几何
Rect2d bbox = ContourBoundingBox(contour);
auto minRect = ContourMinAreaRect(contour);
auto minCircle = ContourMinEnclosingCircle(contour);

// 凸性
QContour hull = ContourConvexHull(contour);
bool isConvex = IsContourConvex(contour);
auto defects = ContourConvexityDefects(contour, 2.0);

// 形状匹配
double similarity = MatchShapesHu(contour1, contour2);
```

---

## 附录 C: 数学公式汇总

### C.1 面积 (Shoelace)
$$A = \frac{1}{2} \sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1} y_i)$$

### C.2 质心
$$C_x = \frac{1}{6A} \sum_{i=0}^{n-1} (x_i + x_{i+1})(x_i y_{i+1} - x_{i+1} y_i)$$
$$C_y = \frac{1}{6A} \sum_{i=0}^{n-1} (y_i + y_{i+1})(x_i y_{i+1} - x_{i+1} y_i)$$

### C.3 几何矩
$$m_{pq} = \sum_{i} x_i^p y_i^q$$

### C.4 中心矩
$$\mu_{pq} = \sum_{i} (x_i - \bar{x})^p (y_i - \bar{y})^q$$

### C.5 归一化矩
$$\eta_{pq} = \frac{\mu_{pq}}{\mu_{00}^{(p+q)/2 + 1}}$$

### C.6 圆度
$$Circularity = \frac{4\pi A}{P^2}$$

### C.7 三点曲率
$$\kappa = \frac{2 \cdot cross(P_1-P_0, P_2-P_1)}{|P_0-P_1| \cdot |P_1-P_2| \cdot |P_2-P_0|}$$
