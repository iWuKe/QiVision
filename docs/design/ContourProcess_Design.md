# Internal/ContourProcess 设计文档

## 1. 概述

### 1.1 功能描述

ContourProcess 模块是 QiVision Internal 层的轮廓处理核心库，提供轮廓的平滑、简化、重采样等基础操作。该模块为 Core/QContour 类的成员函数提供底层算法实现，同时也可被 Feature 层模块直接调用。

主要功能：
1. **平滑 (Smoothing)** - 去除轮廓噪声，保持轮廓形状
2. **简化 (Simplification)** - 减少轮廓点数，保留关键特征
3. **重采样 (Resampling)** - 均匀化轮廓点分布
4. **其他处理** - 反转方向、闭合轮廓、去除重复点

### 1.2 应用场景

- **模板匹配**: 模型轮廓预处理，提高匹配鲁棒性
- **边缘检测后处理**: 平滑边缘检测结果
- **几何测量**: 轮廓简化后进行拟合
- **显示优化**: 减少轮廓点数以加速渲染
- **轮廓分析**: 预处理后进行曲率/特征计算

### 1.3 参考 Halcon 算子

| Halcon 算子 | 功能 | 对应函数 |
|-------------|------|----------|
| smooth_contours_xld | 高斯平滑轮廓 | SmoothContourGaussian |
| smooth_contours_xld (moving average) | 移动平均平滑 | SmoothContourMovingAverage |
| approx_chain_simple | 简单链码近似 | SimplifyContourChain |
| gen_polygons_xld | Douglas-Peucker 简化 | SimplifyContourDouglasPeucker |
| gen_contour_region_xld (smooth) | Visvalingam-Whyatt 简化 | SimplifyContourVisvalingam |
| sample_contour_xld | 等距重采样 | ResampleContourByDistance |
| sample_contour_xld (num_points) | 按点数重采样 | ResampleContourByCount |

### 1.4 设计原则

1. **纯函数**: 输入轮廓不被修改，返回新轮廓
2. **高精度**: 保持亚像素精度 (double 类型)
3. **属性传播**: 可选择性保留或插值轮廓点属性
4. **开闭轮廓支持**: 统一处理开放和闭合轮廓
5. **边界处理**: 合理处理轮廓端点

---

## 2. 设计规则验证

### 2.1 坐标类型符合规则

- [x] 所有坐标使用 `double` 类型 (亚像素精度)
- [x] 轮廓点使用 `ContourPoint` 结构 (含 x, y, amplitude, direction, curvature)
- [x] 参数索引使用 `size_t` 或 `int32_t`

### 2.2 层级依赖正确

- [x] ContourProcess.h 位于 Internal 层
- [x] 依赖 Core/QContour.h (轮廓数据结构)
- [x] 依赖 Core/Types.h (基础类型)
- [x] 依赖 Core/Constants.h (数学常量)
- [x] 可选依赖 Internal/Gaussian.h (高斯核生成)
- [x] 不依赖 Feature 层
- [x] 不跨层依赖 Platform 层

### 2.3 算法完整性

- [x] 平滑: 高斯平滑、移动平均平滑
- [x] 简化: Douglas-Peucker、Visvalingam-Whyatt
- [x] 重采样: 等距重采样、按点数重采样
- [x] 其他: 反转、闭合、去重

### 2.4 退化情况处理

- [x] 空轮廓: 返回空轮廓
- [x] 单点轮廓: 返回原轮廓
- [x] 点数不足: 返回原轮廓或最小有效结果
- [x] 参数无效: 使用默认值或返回原轮廓

---

## 3. 依赖分析

### 3.1 依赖的 Internal 模块

| 模块 | 用途 | 状态 |
|------|------|------|
| Internal/Gaussian.h | 高斯核生成 (可选) | ✅ 已完成 |
| Internal/Geometry2d.h | 几何计算辅助 | ✅ 已完成 |

### 3.2 依赖的 Core 类型

| 类型 | 用途 |
|------|------|
| Core/QContour.h | QContour, ContourPoint |
| Core/Types.h | Point2d |
| Core/Constants.h | PI, EPSILON |

### 3.3 被依赖的模块

| 模块 | 用途 | 状态 |
|------|------|------|
| Core/QContour | 成员函数底层实现 | ✅ 已完成 (待替换) |
| Internal/ContourAnalysis | 轮廓分析前预处理 | ⬜ 待设计 |
| Internal/ContourSegment | 轮廓分割前预处理 | ⬜ 待设计 |
| Feature/Matching | 模型轮廓预处理 | ⬜ 待设计 |
| Feature/Edge | 边缘检测后处理 | ⬜ 待设计 |

---

## 4. 类设计

### 4.1 模块结构

```
ContourProcess Module
├── Constants
│   ├── DEFAULT_SMOOTH_SIGMA          - 默认平滑 sigma
│   ├── DEFAULT_SMOOTH_WINDOW         - 默认平滑窗口大小
│   ├── DEFAULT_SIMPLIFY_TOLERANCE    - 默认简化容差
│   ├── MIN_CONTOUR_POINTS            - 最小轮廓点数
│   └── MAX_CONTOUR_POINTS            - 最大轮廓点数
│
├── Enumerations
│   ├── SmoothMethod                  - 平滑方法 (Gaussian/MovingAverage/Bilateral)
│   ├── SimplifyMethod                - 简化方法 (DouglasPeucker/Visvalingam/RadialDistance)
│   ├── ResampleMethod                - 重采样方法 (ByDistance/ByCount/ByArcLength)
│   └── AttributeMode                 - 属性处理模式 (None/Interpolate/NearestNeighbor)
│
├── Smoothing Functions
│   ├── SmoothContourGaussian()       - 高斯平滑
│   ├── SmoothContourMovingAverage()  - 移动平均平滑
│   ├── SmoothContourBilateral()      - 双边滤波平滑 (保边)
│   └── SmoothContour()               - 统一接口
│
├── Simplification Functions
│   ├── SimplifyContourDouglasPeucker()    - Douglas-Peucker 算法
│   ├── SimplifyContourVisvalingam()       - Visvalingam-Whyatt 算法
│   ├── SimplifyContourRadialDistance()    - 径向距离简化
│   ├── SimplifyContourNthPoint()          - 每 N 点保留
│   └── SimplifyContour()                  - 统一接口
│
├── Resampling Functions
│   ├── ResampleContourByDistance()        - 等距重采样
│   ├── ResampleContourByCount()           - 按点数重采样
│   ├── ResampleContourByArcLength()       - 按弧长重采样
│   └── ResampleContour()                  - 统一接口
│
├── Other Processing Functions
│   ├── ReverseContour()              - 反转轮廓方向
│   ├── CloseContour()                - 闭合轮廓
│   ├── OpenContour()                 - 打开轮廓
│   ├── RemoveDuplicatePoints()       - 去除重复点
│   ├── RemoveCollinearPoints()       - 去除共线点
│   ├── ShiftContourStart()           - 移动轮廓起点 (闭合轮廓)
│   └── ExtractSubContour()           - 提取子轮廓
│
└── Utility Functions
    ├── ComputeContourLength()        - 计算轮廓长度
    ├── ComputeCumulativeLength()     - 计算累积弧长
    ├── FindPointByArcLength()        - 按弧长查找点
    └── InterpolateContourPoint()     - 插值轮廓点
```

### 4.2 API 设计

```cpp
#pragma once

/**
 * @file ContourProcess.h
 * @brief Contour processing operations for QiVision
 *
 * This module provides:
 * - Smoothing: Gaussian, Moving Average, Bilateral
 * - Simplification: Douglas-Peucker, Visvalingam-Whyatt, Radial Distance
 * - Resampling: By distance, by count, by arc length
 * - Other operations: Reverse, Close, Remove duplicates
 *
 * Used by:
 * - Core/QContour: Member function implementations
 * - Internal/ContourAnalysis: Pre-processing for analysis
 * - Feature/Matching: Model contour preparation
 * - Feature/Edge: Edge detection post-processing
 *
 * Design principles:
 * - All functions are pure (input not modified)
 * - All coordinates use double for subpixel precision
 * - Support both open and closed contours
 * - Optional attribute preservation/interpolation
 */

#include <QiVision/Core/QContour.h>
#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>

#include <vector>
#include <cstdint>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Default sigma for Gaussian smoothing (in pixels)
constexpr double DEFAULT_SMOOTH_SIGMA = 1.0;

/// Default window size for moving average smoothing
constexpr int32_t DEFAULT_SMOOTH_WINDOW = 5;

/// Minimum window size for smoothing
constexpr int32_t MIN_SMOOTH_WINDOW = 3;

/// Maximum window size for smoothing
constexpr int32_t MAX_SMOOTH_WINDOW = 101;

/// Default tolerance for Douglas-Peucker simplification (in pixels)
constexpr double DEFAULT_SIMPLIFY_TOLERANCE = 1.0;

/// Minimum tolerance for simplification
constexpr double MIN_SIMPLIFY_TOLERANCE = 0.01;

/// Minimum number of points for a valid contour (for operations)
constexpr size_t MIN_CONTOUR_POINTS_FOR_SMOOTH = 3;

/// Minimum number of points for simplification
constexpr size_t MIN_CONTOUR_POINTS_FOR_SIMPLIFY = 3;

/// Default resampling distance (in pixels)
constexpr double DEFAULT_RESAMPLE_DISTANCE = 1.0;

/// Minimum distance for resampling
constexpr double MIN_RESAMPLE_DISTANCE = 0.01;

/// Tolerance for duplicate point detection
constexpr double DUPLICATE_POINT_TOLERANCE = 1e-9;

// =============================================================================
// Enumerations
// =============================================================================

/**
 * @brief Smoothing method enumeration
 */
enum class SmoothMethod {
    Gaussian,       ///< Gaussian smoothing (default)
    MovingAverage,  ///< Moving average (box filter)
    Bilateral       ///< Bilateral filter (edge-preserving)
};

/**
 * @brief Simplification method enumeration
 */
enum class SimplifyMethod {
    DouglasPeucker,   ///< Douglas-Peucker algorithm (default)
    Visvalingam,      ///< Visvalingam-Whyatt algorithm
    RadialDistance,   ///< Radial distance algorithm
    NthPoint          ///< Keep every Nth point
};

/**
 * @brief Resampling method enumeration
 */
enum class ResampleMethod {
    ByDistance,     ///< Fixed distance between points (default)
    ByCount,        ///< Fixed number of points
    ByArcLength     ///< Equal arc length intervals
};

/**
 * @brief Attribute handling mode during processing
 */
enum class AttributeMode {
    None,           ///< Discard attributes
    Interpolate,    ///< Linearly interpolate attributes (default)
    NearestNeighbor ///< Use nearest neighbor attributes
};

// =============================================================================
// Smoothing Parameters
// =============================================================================

/**
 * @brief Parameters for Gaussian smoothing
 */
struct GaussianSmoothParams {
    double sigma = DEFAULT_SMOOTH_SIGMA;        ///< Standard deviation
    int32_t windowSize = 0;                     ///< Window size (0 = auto from sigma)
    AttributeMode attrMode = AttributeMode::Interpolate; ///< How to handle attributes
    
    static GaussianSmoothParams Default() { return {}; }
};

/**
 * @brief Parameters for moving average smoothing
 */
struct MovingAverageSmoothParams {
    int32_t windowSize = DEFAULT_SMOOTH_WINDOW; ///< Window size (must be odd)
    AttributeMode attrMode = AttributeMode::Interpolate; ///< How to handle attributes
    
    static MovingAverageSmoothParams Default() { return {}; }
};

/**
 * @brief Parameters for bilateral smoothing
 */
struct BilateralSmoothParams {
    double sigmaSpace = 2.0;    ///< Spatial sigma
    double sigmaRange = 30.0;   ///< Range (curvature) sigma
    int32_t windowSize = 0;     ///< Window size (0 = auto)
    AttributeMode attrMode = AttributeMode::Interpolate;
    
    static BilateralSmoothParams Default() { return {}; }
};

// =============================================================================
// Simplification Parameters
// =============================================================================

/**
 * @brief Parameters for Douglas-Peucker simplification
 */
struct DouglasPeuckerParams {
    double tolerance = DEFAULT_SIMPLIFY_TOLERANCE;  ///< Maximum perpendicular distance
    
    static DouglasPeuckerParams Default() { return {}; }
};

/**
 * @brief Parameters for Visvalingam-Whyatt simplification
 */
struct VisvalingamParams {
    double minArea = 1.0;       ///< Minimum triangle area to preserve
    size_t minPoints = 0;       ///< Minimum number of points (0 = use minArea)
    
    static VisvalingamParams Default() { return {}; }
};

/**
 * @brief Parameters for radial distance simplification
 */
struct RadialDistanceParams {
    double tolerance = DEFAULT_SIMPLIFY_TOLERANCE;  ///< Radial distance tolerance
    
    static RadialDistanceParams Default() { return {}; }
};

// =============================================================================
// Resampling Parameters
// =============================================================================

/**
 * @brief Parameters for distance-based resampling
 */
struct ResampleByDistanceParams {
    double distance = DEFAULT_RESAMPLE_DISTANCE;    ///< Target distance between points
    bool preserveEndpoints = true;                  ///< Always include first/last points
    AttributeMode attrMode = AttributeMode::Interpolate;
    
    static ResampleByDistanceParams Default() { return {}; }
};

/**
 * @brief Parameters for count-based resampling
 */
struct ResampleByCountParams {
    size_t count = 100;                             ///< Target number of points
    bool preserveEndpoints = true;                  ///< Always include first/last points
    AttributeMode attrMode = AttributeMode::Interpolate;
    
    static ResampleByCountParams Default() { return {}; }
};

// =============================================================================
// Smoothing Functions
// =============================================================================

/**
 * @brief Apply Gaussian smoothing to a contour
 *
 * Smooths the contour using a Gaussian kernel. For closed contours,
 * the smoothing wraps around. For open contours, edge handling uses
 * reflection (mirroring).
 *
 * @param contour Input contour
 * @param params Gaussian smoothing parameters
 * @return Smoothed contour
 *
 * @note If contour has fewer than MIN_CONTOUR_POINTS_FOR_SMOOTH points,
 *       returns the original contour unchanged.
 *
 * @par Example:
 * @code
 * QContour smoothed = SmoothContourGaussian(contour, {.sigma = 2.0});
 * @endcode
 */
QContour SmoothContourGaussian(const QContour& contour, const GaussianSmoothParams& params = {});

/**
 * @brief Apply moving average smoothing to a contour
 *
 * Smooths the contour using a simple moving average (box filter).
 * Faster than Gaussian but may produce less smooth results.
 *
 * @param contour Input contour
 * @param params Moving average parameters
 * @return Smoothed contour
 */
QContour SmoothContourMovingAverage(const QContour& contour, const MovingAverageSmoothParams& params = {});

/**
 * @brief Apply bilateral smoothing to a contour
 *
 * Edge-preserving smoothing that considers both spatial distance and
 * local curvature difference. Preserves corners while smoothing noise.
 *
 * @param contour Input contour
 * @param params Bilateral smoothing parameters
 * @return Smoothed contour
 */
QContour SmoothContourBilateral(const QContour& contour, const BilateralSmoothParams& params = {});

/**
 * @brief Unified smoothing interface
 *
 * @param contour Input contour
 * @param method Smoothing method to use
 * @param sigma Smoothing strength (interpreted based on method)
 * @param windowSize Window size (0 = auto)
 * @return Smoothed contour
 */
QContour SmoothContour(const QContour& contour, SmoothMethod method = SmoothMethod::Gaussian,
                       double sigma = DEFAULT_SMOOTH_SIGMA, int32_t windowSize = 0);

// =============================================================================
// Simplification Functions
// =============================================================================

/**
 * @brief Simplify contour using Douglas-Peucker algorithm
 *
 * Iteratively removes points that are within 'tolerance' distance from
 * the line connecting their neighbors. Preserves overall shape well.
 *
 * @param contour Input contour
 * @param params Douglas-Peucker parameters
 * @return Simplified contour
 *
 * @par Algorithm:
 * 1. Start with line from first to last point
 * 2. Find point with maximum perpendicular distance
 * 3. If distance > tolerance, split and recurse
 * 4. Otherwise, remove intermediate points
 *
 * @note Time complexity: O(n log n) average, O(n^2) worst case
 */
QContour SimplifyContourDouglasPeucker(const QContour& contour, const DouglasPeuckerParams& params = {});

/**
 * @brief Simplify contour using Visvalingam-Whyatt algorithm
 *
 * Iteratively removes points that form the smallest triangle area
 * with their neighbors. Good for preserving topological features.
 *
 * @param contour Input contour
 * @param params Visvalingam parameters
 * @return Simplified contour
 *
 * @par Algorithm:
 * 1. Compute effective area for each point (triangle with neighbors)
 * 2. Remove point with smallest area
 * 3. Recompute affected areas
 * 4. Repeat until minArea threshold reached or minPoints count
 *
 * @note Time complexity: O(n log n) using priority queue
 */
QContour SimplifyContourVisvalingam(const QContour& contour, const VisvalingamParams& params = {});

/**
 * @brief Simplify contour using radial distance algorithm
 *
 * Removes consecutive points that are within 'tolerance' radial
 * distance from each other. Simple and fast.
 *
 * @param contour Input contour
 * @param params Radial distance parameters
 * @return Simplified contour
 *
 * @note Time complexity: O(n)
 */
QContour SimplifyContourRadialDistance(const QContour& contour, const RadialDistanceParams& params = {});

/**
 * @brief Simplify contour by keeping every Nth point
 *
 * Simple decimation - keeps every Nth point, always including first and last.
 *
 * @param contour Input contour
 * @param n Keep every Nth point (n >= 2)
 * @return Simplified contour
 */
QContour SimplifyContourNthPoint(const QContour& contour, size_t n);

/**
 * @brief Unified simplification interface
 *
 * @param contour Input contour
 * @param method Simplification method to use
 * @param tolerance Tolerance parameter (interpreted based on method)
 * @return Simplified contour
 */
QContour SimplifyContour(const QContour& contour, SimplifyMethod method = SimplifyMethod::DouglasPeucker,
                         double tolerance = DEFAULT_SIMPLIFY_TOLERANCE);

// =============================================================================
// Resampling Functions
// =============================================================================

/**
 * @brief Resample contour with fixed distance between points
 *
 * Creates a new contour where consecutive points are approximately
 * 'distance' apart (measured along the contour).
 *
 * @param contour Input contour
 * @param params Distance-based resampling parameters
 * @return Resampled contour
 *
 * @note For closed contours, the last point may be slightly closer to
 *       the first point to ensure proper closure.
 */
QContour ResampleContourByDistance(const QContour& contour, const ResampleByDistanceParams& params = {});

/**
 * @brief Resample contour to have a fixed number of points
 *
 * Creates a new contour with exactly 'count' points, uniformly
 * distributed along the contour (by arc length).
 *
 * @param contour Input contour
 * @param params Count-based resampling parameters
 * @return Resampled contour
 *
 * @note For closed contours with preserveEndpoints=true, first point
 *       is included but last point is omitted (as it would duplicate first).
 */
QContour ResampleContourByCount(const QContour& contour, const ResampleByCountParams& params = {});

/**
 * @brief Resample contour with equal arc length intervals
 *
 * Similar to ResampleContourByDistance but guarantees exact equal
 * arc length between all consecutive points.
 *
 * @param contour Input contour
 * @param numSegments Number of segments (points = segments + 1 for open)
 * @param attrMode Attribute handling mode
 * @return Resampled contour
 */
QContour ResampleContourByArcLength(const QContour& contour, size_t numSegments,
                                     AttributeMode attrMode = AttributeMode::Interpolate);

/**
 * @brief Unified resampling interface
 *
 * @param contour Input contour
 * @param method Resampling method to use
 * @param param Method-specific parameter (distance or count)
 * @return Resampled contour
 */
QContour ResampleContour(const QContour& contour, ResampleMethod method = ResampleMethod::ByDistance,
                         double param = DEFAULT_RESAMPLE_DISTANCE);

// =============================================================================
// Other Processing Functions
// =============================================================================

/**
 * @brief Reverse the direction of a contour
 *
 * Reverses the point order and adjusts direction attributes.
 *
 * @param contour Input contour
 * @return Reversed contour
 *
 * @note Direction attributes are rotated by PI.
 */
QContour ReverseContour(const QContour& contour);

/**
 * @brief Close an open contour
 *
 * If the contour is not already closed, marks it as closed.
 * Does not add a duplicate point.
 *
 * @param contour Input contour
 * @return Closed contour
 */
QContour CloseContour(const QContour& contour);

/**
 * @brief Open a closed contour
 *
 * Marks a closed contour as open.
 *
 * @param contour Input contour
 * @return Open contour
 */
QContour OpenContour(const QContour& contour);

/**
 * @brief Remove duplicate consecutive points
 *
 * Removes points that are within 'tolerance' distance of their predecessor.
 *
 * @param contour Input contour
 * @param tolerance Distance tolerance for duplicate detection
 * @return Contour with duplicates removed
 */
QContour RemoveDuplicatePoints(const QContour& contour, double tolerance = DUPLICATE_POINT_TOLERANCE);

/**
 * @brief Remove collinear points
 *
 * Removes points that lie on the line between their neighbors
 * (within tolerance).
 *
 * @param contour Input contour
 * @param tolerance Perpendicular distance tolerance
 * @return Contour with collinear points removed
 *
 * @note This is similar to Douglas-Peucker with tolerance=tolerance,
 *       but operates locally rather than recursively.
 */
QContour RemoveCollinearPoints(const QContour& contour, double tolerance = 1e-6);

/**
 * @brief Shift the starting point of a closed contour
 *
 * Rotates the point sequence so that the point nearest to 'newStart'
 * becomes the first point.
 *
 * @param contour Input contour (must be closed)
 * @param newStart Point near desired new start
 * @return Contour with shifted start (or original if open)
 */
QContour ShiftContourStart(const QContour& contour, const Point2d& newStart);

/**
 * @brief Shift the starting point of a closed contour by index
 *
 * @param contour Input contour (must be closed)
 * @param startIndex New starting point index
 * @return Contour with shifted start (or original if open/invalid index)
 */
QContour ShiftContourStartByIndex(const QContour& contour, size_t startIndex);

/**
 * @brief Extract a sub-contour between two parameters
 *
 * Extracts the portion of the contour between parameter t1 and t2,
 * where t in [0, 1] represents position along the contour.
 *
 * @param contour Input contour
 * @param t1 Start parameter [0, 1]
 * @param t2 End parameter [0, 1]
 * @return Extracted sub-contour
 *
 * @note If t2 < t1 and contour is closed, wraps around.
 *       If t2 < t1 and contour is open, swaps t1 and t2.
 */
QContour ExtractSubContour(const QContour& contour, double t1, double t2);

/**
 * @brief Extract a sub-contour between two point indices
 *
 * @param contour Input contour
 * @param startIdx Start point index
 * @param endIdx End point index (exclusive)
 * @return Extracted sub-contour
 */
QContour ExtractSubContourByIndex(const QContour& contour, size_t startIdx, size_t endIdx);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Compute total length of a contour
 *
 * @param contour Input contour
 * @return Total arc length
 */
double ComputeContourLength(const QContour& contour);

/**
 * @brief Compute cumulative arc length at each point
 *
 * @param contour Input contour
 * @return Vector of cumulative lengths (same size as contour)
 *
 * @note First element is 0, last element is total length.
 */
std::vector<double> ComputeCumulativeLength(const QContour& contour);

/**
 * @brief Find point on contour at given arc length
 *
 * @param contour Input contour
 * @param arcLength Target arc length from start
 * @param attrMode Attribute interpolation mode
 * @return Interpolated contour point at the given arc length
 *
 * @note If arcLength exceeds total length, returns last point.
 *       If arcLength < 0, returns first point.
 */
ContourPoint FindPointByArcLength(const QContour& contour, double arcLength,
                                   AttributeMode attrMode = AttributeMode::Interpolate);

/**
 * @brief Find point on contour at given parameter t
 *
 * @param contour Input contour
 * @param t Parameter in [0, 1] (by arc length)
 * @param attrMode Attribute interpolation mode
 * @return Interpolated contour point
 */
ContourPoint FindPointByParameter(const QContour& contour, double t,
                                   AttributeMode attrMode = AttributeMode::Interpolate);

/**
 * @brief Interpolate a contour point between two existing points
 *
 * @param p1 First point
 * @param p2 Second point
 * @param t Interpolation parameter [0, 1] (0=p1, 1=p2)
 * @param attrMode Attribute interpolation mode
 * @return Interpolated contour point
 */
ContourPoint InterpolateContourPoint(const ContourPoint& p1, const ContourPoint& p2,
                                      double t, AttributeMode attrMode = AttributeMode::Interpolate);

/**
 * @brief Find the segment index containing a given arc length
 *
 * @param contour Input contour
 * @param arcLength Target arc length
 * @param localT Output: local parameter within segment [0, 1]
 * @return Segment index (or last segment if arcLength exceeds total)
 */
size_t FindSegmentByArcLength(const QContour& contour, double arcLength, double& localT);

} // namespace Qi::Vision::Internal
```

---

## 5. 参数设计

### 5.1 常量

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| DEFAULT_SMOOTH_SIGMA | double | 1.0 | [0.1, 10.0] | 默认高斯平滑 sigma |
| DEFAULT_SMOOTH_WINDOW | int32_t | 5 | [3, 101] | 默认平滑窗口大小 |
| DEFAULT_SIMPLIFY_TOLERANCE | double | 1.0 | [0.01, 100.0] | 默认简化容差 |
| DEFAULT_RESAMPLE_DISTANCE | double | 1.0 | [0.01, 100.0] | 默认重采样间距 |
| DUPLICATE_POINT_TOLERANCE | double | 1e-9 | - | 重复点检测容差 |
| MIN_CONTOUR_POINTS_FOR_SMOOTH | size_t | 3 | - | 平滑操作最小点数 |
| MIN_CONTOUR_POINTS_FOR_SIMPLIFY | size_t | 3 | - | 简化操作最小点数 |

### 5.2 参数选择指南

| 操作 | 参数 | 低值效果 | 高值效果 | 推荐值 |
|------|------|----------|----------|--------|
| Gaussian Smooth | sigma | 轻微平滑 | 强烈平滑 | 1.0-3.0 |
| Moving Average | windowSize | 轻微平滑 | 强烈平滑 | 5-11 |
| Douglas-Peucker | tolerance | 保留更多点 | 更强简化 | 0.5-2.0 |
| Visvalingam | minArea | 保留更多点 | 更强简化 | 0.5-5.0 |
| Resample Distance | distance | 更密集点 | 更稀疏点 | 0.5-2.0 |

---

## 6. 精度规格

### 6.1 平滑精度

| 条件 | 指标 | 要求 |
|------|------|------|
| 标准高斯平滑 | 位置偏移 | < 0.1*sigma 像素 |
| 闭合轮廓边界 | 首尾连续性 | 无可见跳变 |
| 属性插值 | 属性连续性 | 线性插值 |

### 6.2 简化精度

| 条件 | 指标 | 要求 |
|------|------|------|
| Douglas-Peucker | 最大偏差 | <= tolerance |
| Visvalingam | 最小三角形面积 | >= minArea |
| 端点保留 | 首尾点 | 精确保留 |

### 6.3 重采样精度

| 条件 | 指标 | 要求 |
|------|------|------|
| 距离重采样 | 点间距偏差 | < 1% 标称距离 |
| 点数重采样 | 实际点数 | 精确等于指定值 |
| 端点保留 | 首尾点位置 | 精确保留 |

---

## 7. 算法要点

### 7.1 高斯平滑

```cpp
QContour SmoothContourGaussian(const QContour& contour, const GaussianSmoothParams& params) {
    if (contour.Size() < MIN_CONTOUR_POINTS_FOR_SMOOTH) {
        return contour;  // 返回原轮廓
    }
    
    double sigma = std::max(params.sigma, 0.1);
    int32_t halfWindow = static_cast<int32_t>(std::ceil(3.0 * sigma));
    int32_t windowSize = params.windowSize > 0 ? params.windowSize : 2 * halfWindow + 1;
    halfWindow = windowSize / 2;
    
    // 生成高斯核
    std::vector<double> kernel(windowSize);
    double sum = 0.0;
    for (int i = 0; i < windowSize; ++i) {
        double x = static_cast<double>(i - halfWindow);
        kernel[i] = std::exp(-0.5 * x * x / (sigma * sigma));
        sum += kernel[i];
    }
    for (auto& k : kernel) k /= sum;  // 归一化
    
    QContour result;
    result.Reserve(contour.Size());
    size_t n = contour.Size();
    bool closed = contour.IsClosed();
    
    for (size_t i = 0; i < n; ++i) {
        double sumX = 0, sumY = 0;
        double sumAmp = 0, sumCurv = 0;
        double sumDirCos = 0, sumDirSin = 0;
        
        for (int j = -halfWindow; j <= halfWindow; ++j) {
            size_t idx;
            if (closed) {
                idx = (i + n + static_cast<size_t>(j)) % n;
            } else {
                // 反射边界处理
                int ii = static_cast<int>(i) + j;
                if (ii < 0) ii = -ii;
                if (ii >= static_cast<int>(n)) ii = 2 * static_cast<int>(n) - ii - 2;
                idx = static_cast<size_t>(std::clamp(ii, 0, static_cast<int>(n) - 1));
            }
            
            const auto& pt = contour[idx];
            double w = kernel[j + halfWindow];
            sumX += w * pt.x;
            sumY += w * pt.y;
            sumAmp += w * pt.amplitude;
            sumDirCos += w * std::cos(pt.direction);
            sumDirSin += w * std::sin(pt.direction);
            sumCurv += w * pt.curvature;
        }
        
        ContourPoint p;
        p.x = sumX;
        p.y = sumY;
        p.amplitude = sumAmp;
        p.direction = std::atan2(sumDirSin, sumDirCos);  // 圆周平均
        p.curvature = sumCurv;
        result.AddPoint(p);
    }
    
    result.SetClosed(closed);
    return result;
}
```

### 7.2 Douglas-Peucker 简化

```cpp
QContour SimplifyContourDouglasPeucker(const QContour& contour, const DouglasPeuckerParams& params) {
    if (contour.Size() < MIN_CONTOUR_POINTS_FOR_SIMPLIFY) {
        return contour;
    }
    
    double tolerance = std::max(params.tolerance, MIN_SIMPLIFY_TOLERANCE);
    std::vector<bool> keep(contour.Size(), false);
    keep[0] = true;
    keep[contour.Size() - 1] = true;
    
    // 递归分治
    std::function<void(size_t, size_t)> simplify = [&](size_t start, size_t end) {
        if (end <= start + 1) return;
        
        Point2d p1 = contour.GetPoint(start);
        Point2d p2 = contour.GetPoint(end);
        
        double maxDist = 0.0;
        size_t maxIdx = start;
        
        // 计算线段方程
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        double lenSq = dx * dx + dy * dy;
        
        for (size_t i = start + 1; i < end; ++i) {
            Point2d p = contour.GetPoint(i);
            double dist;
            
            if (lenSq < EPSILON * EPSILON) {
                dist = p.DistanceTo(p1);
            } else {
                // 垂直距离 = |cross| / |line|
                double cross = (p.x - p1.x) * dy - (p.y - p1.y) * dx;
                dist = std::abs(cross) / std::sqrt(lenSq);
            }
            
            if (dist > maxDist) {
                maxDist = dist;
                maxIdx = i;
            }
        }
        
        if (maxDist > tolerance) {
            keep[maxIdx] = true;
            simplify(start, maxIdx);
            simplify(maxIdx, end);
        }
    };
    
    // 对闭合轮廓的特殊处理
    if (contour.IsClosed()) {
        // 找到距离首尾连线最远的点作为分割点
        // ... (处理闭合轮廓的环形结构)
    }
    
    simplify(0, contour.Size() - 1);
    
    // 构建结果
    QContour result;
    result.SetClosed(contour.IsClosed());
    for (size_t i = 0; i < contour.Size(); ++i) {
        if (keep[i]) {
            result.AddPoint(contour[i]);
        }
    }
    
    return result;
}
```

### 7.3 Visvalingam-Whyatt 简化

```cpp
QContour SimplifyContourVisvalingam(const QContour& contour, const VisvalingamParams& params) {
    if (contour.Size() < MIN_CONTOUR_POINTS_FOR_SIMPLIFY) {
        return contour;
    }
    
    // 使用优先队列实现，按三角形面积排序
    struct PointInfo {
        size_t index;
        double area;
        bool removed = false;
        size_t prevValid;  // 前一个未删除点
        size_t nextValid;  // 后一个未删除点
    };
    
    size_t n = contour.Size();
    std::vector<PointInfo> points(n);
    
    // 初始化链表和面积
    auto computeArea = [&](size_t prev, size_t curr, size_t next) -> double {
        Point2d p0 = contour.GetPoint(prev);
        Point2d p1 = contour.GetPoint(curr);
        Point2d p2 = contour.GetPoint(next);
        // 三角形面积 = |cross| / 2
        return 0.5 * std::abs((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y));
    };
    
    // ... 优先队列迭代删除最小面积点 ...
    
    // 构建结果
    QContour result;
    result.SetClosed(contour.IsClosed());
    for (size_t i = 0; i < n; ++i) {
        if (!points[i].removed) {
            result.AddPoint(contour[i]);
        }
    }
    
    return result;
}
```

### 7.4 等距重采样

```cpp
QContour ResampleContourByDistance(const QContour& contour, const ResampleByDistanceParams& params) {
    if (contour.Size() < 2) {
        return contour;
    }
    
    double distance = std::max(params.distance, MIN_RESAMPLE_DISTANCE);
    double totalLength = ComputeContourLength(contour);
    
    if (totalLength < distance) {
        // 轮廓太短，返回端点
        QContour result;
        result.AddPoint(contour[0]);
        if (contour.Size() > 1) {
            result.AddPoint(contour[contour.Size() - 1]);
        }
        result.SetClosed(contour.IsClosed());
        return result;
    }
    
    // 计算累积弧长
    std::vector<double> cumLen = ComputeCumulativeLength(contour);
    
    QContour result;
    result.AddPoint(contour[0]);  // 第一个点
    
    double targetLen = distance;
    size_t segIdx = 0;
    
    while (targetLen < totalLength - EPSILON) {
        // 找到包含 targetLen 的段
        while (segIdx < contour.Size() - 1 && cumLen[segIdx + 1] < targetLen) {
            ++segIdx;
        }
        
        // 在段内插值
        double segStart = cumLen[segIdx];
        double segEnd = cumLen[segIdx + 1];
        double localT = (targetLen - segStart) / (segEnd - segStart);
        
        ContourPoint pt = InterpolateContourPoint(contour[segIdx], contour[segIdx + 1],
                                                   localT, params.attrMode);
        result.AddPoint(pt);
        
        targetLen += distance;
    }
    
    // 添加最后一个点
    if (params.preserveEndpoints && !contour.IsClosed()) {
        result.AddPoint(contour[contour.Size() - 1]);
    }
    
    result.SetClosed(contour.IsClosed());
    return result;
}
```

### 7.5 属性插值

```cpp
ContourPoint InterpolateContourPoint(const ContourPoint& p1, const ContourPoint& p2,
                                      double t, AttributeMode attrMode) {
    ContourPoint result;
    
    // 位置始终线性插值
    result.x = p1.x + t * (p2.x - p1.x);
    result.y = p1.y + t * (p2.y - p1.y);
    
    switch (attrMode) {
        case AttributeMode::None:
            // 属性置零
            result.amplitude = 0.0;
            result.direction = 0.0;
            result.curvature = 0.0;
            break;
            
        case AttributeMode::Interpolate:
            result.amplitude = p1.amplitude + t * (p2.amplitude - p1.amplitude);
            // 方向需要圆周插值
            result.direction = InterpolateAngle(p1.direction, p2.direction, t);
            result.curvature = p1.curvature + t * (p2.curvature - p1.curvature);
            break;
            
        case AttributeMode::NearestNeighbor:
            if (t < 0.5) {
                result.amplitude = p1.amplitude;
                result.direction = p1.direction;
                result.curvature = p1.curvature;
            } else {
                result.amplitude = p2.amplitude;
                result.direction = p2.direction;
                result.curvature = p2.curvature;
            }
            break;
    }
    
    return result;
}

// 角度圆周插值
double InterpolateAngle(double a1, double a2, double t) {
    // 确保最短路径插值
    double diff = a2 - a1;
    while (diff > PI) diff -= TWO_PI;
    while (diff < -PI) diff += TWO_PI;
    return NormalizeAngle(a1 + t * diff);
}
```

---

## 8. 与现有模块的关系

### 8.1 与 Core/QContour 的关系

QContour 已有的成员函数可以委托给 ContourProcess:

```cpp
// QContour.cpp 中可改为:
QContour QContour::Smooth(double sigma) const {
    return Internal::SmoothContourGaussian(*this, {.sigma = sigma});
}

QContour QContour::Simplify(double tolerance) const {
    return Internal::SimplifyContourDouglasPeucker(*this, {.tolerance = tolerance});
}

QContour QContour::Resample(double interval) const {
    return Internal::ResampleContourByDistance(*this, {.distance = interval});
}
```

### 8.2 与 Internal/Gaussian 的关系

可复用 Gaussian 模块的核生成:

```cpp
#include <QiVision/Internal/Gaussian.h>

// 在 SmoothContourGaussian 中:
std::vector<double> kernel = Gaussian::Kernel1D(sigma, windowSize, true);
```

### 8.3 与 Internal/Geometry2d 的关系

使用 Geometry2d 的角度归一化:

```cpp
#include <QiVision/Internal/Geometry2d.h>

// 角度插值后归一化
result.direction = NormalizeAngle(interpolatedAngle);
```

---

## 9. 实现任务分解

| 任务 | 文件 | 预估时间 | 依赖 | 优先级 |
|------|------|----------|------|--------|
| 头文件 API 定义 | ContourProcess.h | 2h | QContour.h | P0 |
| 高斯平滑 | ContourProcess.cpp | 2h | Gaussian.h | P0 |
| 移动平均平滑 | ContourProcess.cpp | 1h | - | P0 |
| 双边滤波平滑 | ContourProcess.cpp | 2h | - | P1 |
| Douglas-Peucker 简化 | ContourProcess.cpp | 2h | - | P0 |
| Visvalingam 简化 | ContourProcess.cpp | 2h | - | P1 |
| 径向距离简化 | ContourProcess.cpp | 1h | - | P0 |
| 等距重采样 | ContourProcess.cpp | 2h | - | P0 |
| 按点数重采样 | ContourProcess.cpp | 1h | - | P0 |
| 按弧长重采样 | ContourProcess.cpp | 1h | - | P1 |
| 其他处理函数 | ContourProcess.cpp | 2h | - | P0 |
| 工具函数 | ContourProcess.cpp | 1h | - | P0 |
| 单元测试 | test_contour_process.cpp | 4h | 全部 | P0 |
| 精度测试 | accuracy_contour_process.cpp | 2h | 全部 | P1 |

**总计**: 约 25 小时

**实现顺序建议**:
1. P0 阶段: 头文件 + 高斯平滑 + 移动平均 + Douglas-Peucker + 径向距离 + 等距重采样 + 按点数重采样 + 其他 + 工具 (~14h)
2. P1 阶段: 双边滤波 + Visvalingam + 按弧长重采样 (~5h)
3. 测试 (~6h)

---

## 10. 测试要点

### 10.1 单元测试覆盖

1. **平滑测试**
   - 高斯平滑 (不同 sigma)
   - 移动平均 (不同窗口大小)
   - 开放/闭合轮廓边界处理
   - 属性传播验证

2. **简化测试**
   - Douglas-Peucker (不同 tolerance)
   - Visvalingam (不同 minArea)
   - 端点保留验证
   - 简化前后形状一致性

3. **重采样测试**
   - 等距重采样 (点间距验证)
   - 按点数重采样 (点数精确性)
   - 端点保留验证
   - 属性插值验证

4. **其他处理测试**
   - 反转方向 (方向属性调整)
   - 去重复点
   - 去共线点
   - 起点移动 (闭合轮廓)
   - 子轮廓提取

### 10.2 边界条件测试

- 空轮廓
- 单点轮廓
- 两点轮廓
- 极小参数值
- 极大参数值
- 闭合轮廓首尾处理

### 10.3 精度测试

```cpp
// 示例: 平滑不改变质心
TEST(ContourProcessAccuracy, SmoothPreservesCentroid) {
    QContour original = CreateTestContour();
    QContour smoothed = SmoothContourGaussian(original, {.sigma = 2.0});
    
    Point2d origCentroid = original.Centroid();
    Point2d smoothCentroid = smoothed.Centroid();
    
    EXPECT_NEAR(smoothCentroid.x, origCentroid.x, 0.5);
    EXPECT_NEAR(smoothCentroid.y, origCentroid.y, 0.5);
}

// 示例: Douglas-Peucker 误差约束
TEST(ContourProcessAccuracy, DouglasPeuckerMaxError) {
    QContour original = CreateTestContour();
    double tolerance = 1.0;
    QContour simplified = SimplifyContourDouglasPeucker(original, {.tolerance = tolerance});
    
    // 验证每个原始点到简化轮廓的距离 <= tolerance
    for (size_t i = 0; i < original.Size(); ++i) {
        double dist = simplified.DistanceToPoint(original.GetPoint(i));
        EXPECT_LE(dist, tolerance + EPSILON);
    }
}

// 示例: 重采样均匀性
TEST(ContourProcessAccuracy, ResampleUniformDistance) {
    QContour original = CreateTestContour();
    double targetDist = 2.0;
    QContour resampled = ResampleContourByDistance(original, {.distance = targetDist});
    
    for (size_t i = 1; i < resampled.Size(); ++i) {
        double dist = resampled.GetPoint(i).DistanceTo(resampled.GetPoint(i - 1));
        EXPECT_NEAR(dist, targetDist, targetDist * 0.05);  // 5% 误差
    }
}
```

---

## 11. 线程安全

### 11.1 线程安全保证

| 函数类型 | 线程安全性 |
|----------|------------|
| 所有处理函数 | 可重入 (输入只读) |
| 结果值类型 | 线程隔离 |

### 11.2 无全局状态

- 所有函数为纯函数
- 无静态变量
- 无缓存状态

---

## 12. 未来扩展

1. **更多平滑方法**: Savitzky-Golay 滤波、小波去噪
2. **自适应简化**: 基于局部曲率的变步长简化
3. **并行处理**: 长轮廓的分段并行平滑
4. **样条拟合**: 用 B 样条拟合并重采样
5. **轮廓合并**: 合并相邻轮廓

---

## 附录 A: 与 Halcon 对应

| QiVision | Halcon |
|----------|--------|
| SmoothContourGaussian | smooth_contours_xld (Gaussian) |
| SmoothContourMovingAverage | smooth_contours_xld (moving average) |
| SimplifyContourDouglasPeucker | gen_polygons_xld |
| SimplifyContourVisvalingam | (无直接对应) |
| ResampleContourByDistance | sample_contour_xld (by distance) |
| ResampleContourByCount | sample_contour_xld (by num_points) |
| ReverseContour | reverse_contour_xld |

---

## 附录 B: API 快速参考

```cpp
// 平滑
QContour smoothed = SmoothContourGaussian(contour, {.sigma = 2.0});
QContour smoothed = SmoothContourMovingAverage(contour, {.windowSize = 7});
QContour smoothed = SmoothContour(contour, SmoothMethod::Gaussian, 2.0);

// 简化
QContour simplified = SimplifyContourDouglasPeucker(contour, {.tolerance = 1.0});
QContour simplified = SimplifyContourVisvalingam(contour, {.minArea = 1.0});
QContour simplified = SimplifyContour(contour, SimplifyMethod::DouglasPeucker, 1.0);

// 重采样
QContour resampled = ResampleContourByDistance(contour, {.distance = 2.0});
QContour resampled = ResampleContourByCount(contour, {.count = 100});
QContour resampled = ResampleContour(contour, ResampleMethod::ByDistance, 2.0);

// 其他处理
QContour reversed = ReverseContour(contour);
QContour closed = CloseContour(contour);
QContour cleaned = RemoveDuplicatePoints(contour);
QContour shifted = ShiftContourStart(contour, newStartPoint);
QContour sub = ExtractSubContour(contour, 0.2, 0.8);

// 工具函数
double len = ComputeContourLength(contour);
std::vector<double> cumLen = ComputeCumulativeLength(contour);
ContourPoint pt = FindPointByArcLength(contour, 50.0);
ContourPoint interp = InterpolateContourPoint(p1, p2, 0.5);
```
