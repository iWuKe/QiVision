# Internal/Distance 设计文档

## 1. 概述

### 1.1 功能描述

Distance 模块是 QiVision Internal 层的距离计算库,提供各种几何基元之间的距离计算功能。该模块是 Measure、Fitting、Matching、Metrology 等上层模块的基础依赖。

### 1.2 应用场景

- **测量计算**: 卡尺测量中计算边缘到几何基元的距离
- **几何拟合**: RANSAC 中计算点到模型的残差
- **模板匹配**: 轮廓匹配中的距离评估
- **碰撞检测**: 判断几何体之间的最近距离
- **轮廓分析**: 计算轮廓点到参考几何体的距离
- **缺陷检测**: 计算特征点到理想几何的偏差

### 1.3 参考

**Halcon 相关算子**:
- `distance_pp`: 点到点距离
- `distance_pl`: 点到直线距离
- `distance_ps`: 点到线段距离
- `distance_pc`: 点到圆距离
- `distance_lc`: 直线到圆距离
- `distance_ll`: 两直线距离
- `distance_ss`: 两线段距离
- `distance_contour_points_xld`: 轮廓点距离

**数学基础**:
- 解析几何: 点到直线/圆的距离公式
- 最优化: 点到椭圆/轮廓的最近点搜索
- 参数曲线: 弧上最近点的参数求解

### 1.4 设计原则

1. **纯函数**: 无全局状态,所有函数可重入
2. **高精度**: 使用 double 类型,返回亚像素精度结果
3. **完整结果**: 提供距离值、最近点、符号信息
4. **鲁棒性**: 处理退化情况(零长度、零半径、共点等)
5. **复用**: 复用 Geometry2d.h 中的基础操作

---

## 2. 设计规则验证

### 2.1 坐标类型符合规则

- [x] 所有坐标使用 `double` 类型 (亚像素精度)
- [x] 距离返回 `double` 类型
- [x] 最近点返回 `Point2d` 类型

### 2.2 层级依赖正确

- [x] Distance.h 位于 Internal 层
- [x] 依赖 Core/Types.h (几何基元定义)
- [x] 依赖 Core/Constants.h (数学常量、容差)
- [x] 依赖 Internal/Geometry2d.h (投影、归一化等操作)
- [x] 不依赖 Feature 层
- [x] 不跨层依赖 Platform 层

### 2.3 算法完整性

- [x] Point-Point: 基础距离
- [x] Point-Line: 有符号/无符号距离
- [x] Point-Segment: 包含端点处理
- [x] Point-Circle: 圆心与圆周的距离
- [x] Point-Ellipse: 迭代求解最近点
- [x] Point-Arc: 考虑角度范围限制
- [x] Point-RotatedRect: 到四边最近距离
- [x] Line-Line: 平行线距离
- [x] Segment-Segment: 包含端点配对
- [x] Circle-Circle: 外切/内切/相离
- [x] Point-Contour: 最近点搜索

### 2.4 退化情况处理

- [x] 点与点重合: 返回 0
- [x] 零长度线段: 作为点处理
- [x] 零半径圆: 作为点处理
- [x] 零扫掠角弧: 作为点处理
- [x] 直线重合: 返回 0
- [x] 点在几何体上: 返回 0
- [x] 空轮廓: 返回特殊值或 nullopt

---

## 3. 依赖分析

### 3.1 依赖的 Internal 模块

| 模块 | 用途 | 状态 |
|------|------|------|
| Internal/Geometry2d.h | ProjectPointOnLine/Segment/Circle, AngleInArcRange | ✅ 已完成 |
| Internal/Matrix.h | (间接依赖,通过 Geometry2d) | ✅ 已完成 |

### 3.2 依赖的 Core 类型

| 类型 | 用途 |
|------|------|
| Core/Types.h | Point2d, Line2d, Segment2d, Circle2d, Ellipse2d, Arc2d, RotatedRect2d |
| Core/Constants.h | EPSILON, PI, ApproxZero |

### 3.3 被依赖的模块

| 模块 | 用途 | 状态 |
|------|------|------|
| Internal/Fitting.h | RANSAC 残差计算 | ✅ 已完成 |
| Internal/Intersection.h | 距离为 0 时的交点判断 | ⬜ 待设计 |
| Feature/Measure/* | 卡尺测量、边缘距离 | ⬜ 待设计 |
| Feature/Matching/* | 轮廓匹配距离评估 | ⬜ 待设计 |
| Feature/Metrology/* | 几何测量 | ⬜ 待设计 |

---

## 4. 类设计

### 4.1 模块结构

```
Distance Module
├── Result Structures
│   ├── DistanceResult                - 距离结果 (距离 + 最近点)
│   └── SignedDistanceResult          - 有符号距离结果
│
├── Point-to-Primitive Functions
│   ├── DistancePointToPoint()        - 点到点
│   ├── DistancePointToLine()         - 点到直线 (无符号)
│   ├── SignedDistancePointToLine()   - 点到直线 (有符号)
│   ├── DistancePointToSegment()      - 点到线段
│   ├── DistancePointToCircle()       - 点到圆
│   ├── DistancePointToEllipse()      - 点到椭圆
│   ├── DistancePointToArc()          - 点到圆弧
│   └── DistancePointToRotatedRect()  - 点到旋转矩形
│
├── Primitive-to-Primitive Functions
│   ├── DistanceLineToLine()          - 直线到直线 (平行线)
│   ├── DistanceSegmentToSegment()    - 线段到线段
│   └── DistanceCircleToCircle()      - 圆到圆
│
├── Point-to-Contour Functions
│   ├── DistancePointToContour()      - 点到轮廓最近距离
│   ├── DistancePointToContourSigned() - 点到轮廓有符号距离
│   └── NearestPointOnContour()       - 轮廓上最近点
│
└── Batch Functions
    ├── DistancePointsToLine()        - 多点到直线距离
    ├── DistancePointsToCircle()      - 多点到圆距离
    └── DistancePointsToContour()     - 多点到轮廓距离
```

### 4.2 API 设计

```cpp
#pragma once

/**
 * @file Distance.h
 * @brief Distance calculation between geometric primitives
 *
 * This module provides:
 * - Point to primitive distance (line, segment, circle, ellipse, arc, rotatedRect)
 * - Primitive to primitive distance (line-line, segment-segment, circle-circle)
 * - Point to contour distance with nearest point
 *
 * Used by:
 * - Internal/Fitting: RANSAC residual calculation
 * - Feature/Measure: Caliper measurement
 * - Feature/Matching: Contour matching distance
 * - Feature/Metrology: Geometric measurement
 *
 * Design principles:
 * - All functions are pure (no global state)
 * - All coordinates use double for subpixel precision
 * - Returns both distance and closest point when applicable
 * - Graceful handling of degenerate cases
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Internal/Geometry2d.h>

#include <cmath>
#include <vector>
#include <optional>

namespace Qi::Vision::Internal {

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief Distance calculation result with closest point
 */
struct DistanceResult {
    double distance = 0.0;           ///< Unsigned distance
    Point2d closestPoint;            ///< Closest point on the target primitive
    double parameter = 0.0;          ///< Parameter on target (t for segment, theta for circle/arc)
    
    /// Check if result is valid
    bool IsValid() const { return distance >= 0.0; }
    
    /// Create invalid result
    static DistanceResult Invalid() {
        DistanceResult r;
        r.distance = -1.0;
        return r;
    }
};

/**
 * @brief Signed distance result (positive = left/outside, negative = right/inside)
 */
struct SignedDistanceResult {
    double signedDistance = 0.0;     ///< Signed distance (convention depends on function)
    Point2d closestPoint;            ///< Closest point on the target primitive
    double parameter = 0.0;          ///< Parameter on target
    
    /// Get unsigned distance
    double Distance() const { return std::abs(signedDistance); }
    
    /// Check if point is on positive side
    bool IsOnPositiveSide() const { return signedDistance > 0.0; }
    
    /// Check if point is on negative side
    bool IsOnNegativeSide() const { return signedDistance < 0.0; }
};

// =============================================================================
// Point-to-Point Distance
// =============================================================================

/**
 * @brief Compute distance between two points
 *
 * @param p1 First point
 * @param p2 Second point
 * @return Euclidean distance
 *
 * @note This is equivalent to p1.DistanceTo(p2) in Point2d
 */
inline double DistancePointToPoint(const Point2d& p1, const Point2d& p2) {
    return p1.DistanceTo(p2);
}

/**
 * @brief Compute squared distance between two points (faster, no sqrt)
 *
 * @param p1 First point
 * @param p2 Second point
 * @return Squared Euclidean distance
 */
inline double DistancePointToPointSquared(const Point2d& p1, const Point2d& p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    return dx * dx + dy * dy;
}

// =============================================================================
// Point-to-Line Distance
// =============================================================================

/**
 * @brief Compute unsigned distance from point to infinite line
 *
 * @param point Query point
 * @param line Target line (assumed normalized: a^2 + b^2 = 1)
 * @return Distance result with closest point on line
 *
 * @note If line is not normalized, it will be normalized internally
 */
DistanceResult DistancePointToLine(const Point2d& point, const Line2d& line);

/**
 * @brief Compute signed distance from point to infinite line
 *
 * Sign convention:
 * - Positive: point is on the side of the normal vector (a, b)
 * - Negative: point is on the opposite side
 *
 * @param point Query point
 * @param line Target line
 * @return Signed distance result
 */
SignedDistanceResult SignedDistancePointToLine(const Point2d& point, const Line2d& line);

// =============================================================================
// Point-to-Segment Distance
// =============================================================================

/**
 * @brief Compute distance from point to line segment
 *
 * Returns distance to the closest point on the segment,
 * which may be an endpoint or an interior point.
 *
 * @param point Query point
 * @param segment Target segment
 * @return Distance result with closest point and parameter t in [0,1]
 *
 * @note For degenerate segment (p1 == p2), returns distance to p1
 */
DistanceResult DistancePointToSegment(const Point2d& point, const Segment2d& segment);

/**
 * @brief Compute signed distance from point to line segment
 *
 * Sign is determined by the direction from p1 to p2:
 * - Positive: point is on the left side of the segment direction
 * - Negative: point is on the right side
 *
 * @param point Query point
 * @param segment Target segment
 * @return Signed distance result
 */
SignedDistanceResult SignedDistancePointToSegment(const Point2d& point, const Segment2d& segment);

// =============================================================================
// Point-to-Circle Distance
// =============================================================================

/**
 * @brief Compute distance from point to circle boundary
 *
 * @param point Query point
 * @param circle Target circle
 * @return Distance result with closest point on circle boundary
 *         - If point is at circle center, returns arbitrary point on boundary
 *
 * @note Distance is always to the circumference, not the disk
 */
DistanceResult DistancePointToCircle(const Point2d& point, const Circle2d& circle);

/**
 * @brief Compute signed distance from point to circle boundary
 *
 * Sign convention:
 * - Positive: point is outside the circle
 * - Negative: point is inside the circle
 * - Zero: point is on the circle boundary
 *
 * @param point Query point
 * @param circle Target circle
 * @return Signed distance result
 */
SignedDistanceResult SignedDistancePointToCircle(const Point2d& point, const Circle2d& circle);

// =============================================================================
// Point-to-Ellipse Distance
// =============================================================================

/**
 * @brief Compute distance from point to ellipse boundary
 *
 * Uses iterative Newton's method to find the closest point on ellipse.
 *
 * @param point Query point
 * @param ellipse Target ellipse
 * @param maxIterations Maximum Newton iterations (default: 10)
 * @param tolerance Convergence tolerance (default: 1e-10)
 * @return Distance result with closest point on ellipse boundary
 *
 * @note For ellipse with a == b (circle), use DistancePointToCircle for efficiency
 */
DistanceResult DistancePointToEllipse(const Point2d& point, const Ellipse2d& ellipse,
                                       int maxIterations = 10, double tolerance = 1e-10);

/**
 * @brief Compute signed distance from point to ellipse boundary
 *
 * Sign convention:
 * - Positive: point is outside the ellipse
 * - Negative: point is inside the ellipse
 *
 * @param point Query point
 * @param ellipse Target ellipse
 * @param maxIterations Maximum Newton iterations
 * @param tolerance Convergence tolerance
 * @return Signed distance result
 */
SignedDistanceResult SignedDistancePointToEllipse(const Point2d& point, const Ellipse2d& ellipse,
                                                   int maxIterations = 10, double tolerance = 1e-10);

// =============================================================================
// Point-to-Arc Distance
// =============================================================================

/**
 * @brief Compute distance from point to circular arc
 *
 * The closest point is either on the arc itself or at one of the endpoints.
 *
 * @param point Query point
 * @param arc Target arc
 * @return Distance result with closest point on arc
 *         - parameter: angle of closest point, or -1/-2 for start/end endpoint
 */
DistanceResult DistancePointToArc(const Point2d& point, const Arc2d& arc);

// =============================================================================
// Point-to-RotatedRect Distance
// =============================================================================

/**
 * @brief Compute distance from point to rotated rectangle boundary
 *
 * Returns the minimum distance to any of the four edges.
 *
 * @param point Query point
 * @param rect Target rotated rectangle
 * @return Distance result with closest point on rectangle boundary
 *         - parameter: edge index (0-3: top, right, bottom, left)
 */
DistanceResult DistancePointToRotatedRect(const Point2d& point, const RotatedRect2d& rect);

/**
 * @brief Compute signed distance from point to rotated rectangle
 *
 * Sign convention:
 * - Positive: point is outside the rectangle
 * - Negative: point is inside the rectangle
 *
 * @param point Query point
 * @param rect Target rotated rectangle
 * @return Signed distance result
 */
SignedDistanceResult SignedDistancePointToRotatedRect(const Point2d& point, const RotatedRect2d& rect);

// =============================================================================
// Line-to-Line Distance
// =============================================================================

/**
 * @brief Compute distance between two parallel lines
 *
 * @param line1 First line
 * @param line2 Second line
 * @return Distance between lines if parallel, nullopt if lines intersect
 *
 * @note For intersecting lines, distance is 0 at intersection point
 */
std::optional<double> DistanceLineToLine(const Line2d& line1, const Line2d& line2);

/**
 * @brief Compute signed distance between two parallel lines
 *
 * Returns the signed distance from line1 to line2, measured
 * along line1's normal direction.
 *
 * @param line1 First line (reference)
 * @param line2 Second line
 * @return Signed distance if parallel, nullopt if lines intersect
 */
std::optional<double> SignedDistanceLineToLine(const Line2d& line1, const Line2d& line2);

// =============================================================================
// Segment-to-Segment Distance
// =============================================================================

/**
 * @brief Result of segment-to-segment distance computation
 */
struct SegmentDistanceResult {
    double distance = 0.0;           ///< Minimum distance between segments
    Point2d closestPoint1;           ///< Closest point on segment 1
    Point2d closestPoint2;           ///< Closest point on segment 2
    double parameter1 = 0.0;         ///< Parameter t1 in [0,1] on segment 1
    double parameter2 = 0.0;         ///< Parameter t2 in [0,1] on segment 2
    
    /// Check if segments intersect (distance ≈ 0)
    bool Intersects(double tolerance = GEOM_TOLERANCE) const {
        return distance <= tolerance;
    }
};

/**
 * @brief Compute minimum distance between two line segments
 *
 * Finds the closest pair of points, one on each segment.
 *
 * @param seg1 First segment
 * @param seg2 Second segment
 * @return Segment distance result with closest points on both segments
 */
SegmentDistanceResult DistanceSegmentToSegment(const Segment2d& seg1, const Segment2d& seg2);

// =============================================================================
// Circle-to-Circle Distance
// =============================================================================

/**
 * @brief Result of circle-to-circle distance computation
 */
struct CircleDistanceResult {
    double distance = 0.0;           ///< Distance between circle boundaries (can be negative if overlapping)
    Point2d closestPoint1;           ///< Closest point on circle 1
    Point2d closestPoint2;           ///< Closest point on circle 2
    
    /// Check if circles are externally separated
    bool AreSeparated() const { return distance > 0.0; }
    
    /// Check if circles are externally tangent
    bool AreExternallyTangent(double tolerance = GEOM_TOLERANCE) const {
        return std::abs(distance) <= tolerance;
    }
    
    /// Check if one circle contains the other (distance < 0 and |d| < |r1-r2|)
    bool OneContainsOther() const { return distance < 0.0; }
};

/**
 * @brief Compute distance between two circle boundaries
 *
 * Distance conventions:
 * - Positive: circles are separated
 * - Zero: circles are tangent (internally or externally)
 * - Negative: circles overlap or one contains the other
 *
 * @param circle1 First circle
 * @param circle2 Second circle
 * @return Circle distance result
 */
CircleDistanceResult DistanceCircleToCircle(const Circle2d& circle1, const Circle2d& circle2);

// =============================================================================
// Point-to-Contour Distance
// =============================================================================

/**
 * @brief Result of point-to-contour distance computation
 */
struct ContourDistanceResult {
    double distance = 0.0;           ///< Minimum distance to contour
    Point2d closestPoint;            ///< Closest point on contour
    size_t segmentIndex = 0;         ///< Index of segment containing closest point
    double segmentParameter = 0.0;   ///< Parameter on segment [0,1]
    
    /// Check if result is valid
    bool IsValid() const { return distance >= 0.0; }
    
    /// Create invalid result
    static ContourDistanceResult Invalid() {
        ContourDistanceResult r;
        r.distance = -1.0;
        return r;
    }
};

/**
 * @brief Compute distance from point to contour (polyline)
 *
 * Finds the closest point on the contour, which may be on any segment.
 *
 * @param point Query point
 * @param contourPoints Points defining the contour
 * @param closed If true, contour is closed (last point connects to first)
 * @return Contour distance result
 *
 * @note Returns Invalid() if contour has less than 2 points
 */
ContourDistanceResult DistancePointToContour(const Point2d& point,
                                              const std::vector<Point2d>& contourPoints,
                                              bool closed = false);

/**
 * @brief Compute signed distance from point to closed contour
 *
 * Sign convention:
 * - Positive: point is outside the contour
 * - Negative: point is inside the contour
 *
 * Uses ray casting to determine inside/outside.
 *
 * @param point Query point
 * @param contourPoints Points defining the closed contour
 * @return Signed distance (positive = outside, negative = inside)
 *
 * @note Contour is assumed to be closed
 * @note Returns Invalid() if contour has less than 3 points
 */
SignedDistanceResult SignedDistancePointToContour(const Point2d& point,
                                                   const std::vector<Point2d>& contourPoints);

// =============================================================================
// Batch Distance Functions
// =============================================================================

/**
 * @brief Compute distances from multiple points to a line
 *
 * @param points Query points
 * @param line Target line
 * @return Vector of distance values
 */
std::vector<double> DistancePointsToLine(const std::vector<Point2d>& points, const Line2d& line);

/**
 * @brief Compute signed distances from multiple points to a line
 *
 * @param points Query points
 * @param line Target line
 * @return Vector of signed distance values
 */
std::vector<double> SignedDistancePointsToLine(const std::vector<Point2d>& points, const Line2d& line);

/**
 * @brief Compute distances from multiple points to a circle
 *
 * @param points Query points
 * @param circle Target circle
 * @return Vector of distance values
 */
std::vector<double> DistancePointsToCircle(const std::vector<Point2d>& points, const Circle2d& circle);

/**
 * @brief Compute signed distances from multiple points to a circle
 *
 * @param points Query points
 * @param circle Target circle
 * @return Vector of signed distance values (positive = outside)
 */
std::vector<double> SignedDistancePointsToCircle(const std::vector<Point2d>& points, const Circle2d& circle);

/**
 * @brief Compute distances from multiple points to an ellipse
 *
 * @param points Query points
 * @param ellipse Target ellipse
 * @return Vector of distance values
 */
std::vector<double> DistancePointsToEllipse(const std::vector<Point2d>& points, const Ellipse2d& ellipse);

/**
 * @brief Compute distances from multiple points to a contour
 *
 * @param points Query points
 * @param contourPoints Points defining the contour
 * @param closed If true, contour is closed
 * @return Vector of distance values
 */
std::vector<double> DistancePointsToContour(const std::vector<Point2d>& points,
                                             const std::vector<Point2d>& contourPoints,
                                             bool closed = false);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Find the closest point on a contour to a query point
 *
 * @param point Query point
 * @param contourPoints Points defining the contour
 * @param closed If true, contour is closed
 * @return Closest point on contour, or nullopt if contour is empty
 */
std::optional<Point2d> NearestPointOnContour(const Point2d& point,
                                              const std::vector<Point2d>& contourPoints,
                                              bool closed = false);

/**
 * @brief Compute Hausdorff distance between two contours
 *
 * The Hausdorff distance is the maximum of the minimum distances
 * from each point on one contour to the other contour.
 *
 * @param contour1 First contour points
 * @param contour2 Second contour points
 * @param closed1 If true, first contour is closed
 * @param closed2 If true, second contour is closed
 * @return Hausdorff distance
 */
double HausdorffDistance(const std::vector<Point2d>& contour1,
                         const std::vector<Point2d>& contour2,
                         bool closed1 = false, bool closed2 = false);

/**
 * @brief Compute average distance between two contours (one-directional)
 *
 * Computes the average of minimum distances from each point on contour1
 * to contour2.
 *
 * @param contour1 Source contour points
 * @param contour2 Target contour points
 * @param closed2 If true, target contour is closed
 * @return Average distance
 */
double AverageDistanceContourToContour(const std::vector<Point2d>& contour1,
                                        const std::vector<Point2d>& contour2,
                                        bool closed2 = false);

} // namespace Qi::Vision::Internal
```

---

## 5. 参数设计

### 5.1 常量

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| GEOM_TOLERANCE | double | 1e-9 | [1e-12, 1e-6] | 距离比较容差 (来自 Geometry2d.h) |
| ELLIPSE_MAX_ITERATIONS | int | 10 | [5, 50] | 椭圆距离计算最大迭代次数 |
| ELLIPSE_TOLERANCE | double | 1e-10 | [1e-15, 1e-6] | 椭圆距离计算收敛容差 |

### 5.2 符号约定

| 距离类型 | 正值含义 | 负值含义 |
|----------|----------|----------|
| Point-to-Line | 法向量方向 | 法向量反方向 |
| Point-to-Segment | 左侧 (p1→p2方向) | 右侧 |
| Point-to-Circle | 圆外 | 圆内 |
| Point-to-Ellipse | 椭圆外 | 椭圆内 |
| Point-to-Contour | 轮廓外 | 轮廓内 |
| Point-to-RotatedRect | 矩形外 | 矩形内 |

---

## 6. 精度规格

### 6.1 距离计算精度

| 距离类型 | 精度要求 | 条件 |
|----------|----------|------|
| Point-Point | 精确 (机器精度) | - |
| Point-Line | < 1e-14 相对误差 | 归一化直线 |
| Point-Segment | < 1e-14 相对误差 | 非退化线段 |
| Point-Circle | < 1e-14 相对误差 | 非零半径 |
| Point-Ellipse | < 1e-10 px | 10次迭代 |
| Point-Arc | < 1e-14 相对误差 | 非退化弧 |
| Segment-Segment | < 1e-14 相对误差 | 非退化线段 |
| Point-Contour | < 1e-14 相对误差 | 非空轮廓 |

### 6.2 最近点精度

| 类型 | 精度要求 | 备注 |
|------|----------|------|
| 线段上最近点 | 精确 | 解析解 |
| 圆上最近点 | 精确 | 解析解 |
| 椭圆上最近点 | < 1e-10 px | Newton迭代 |
| 轮廓上最近点 | 精确 | 逐段解析 |

---

## 7. 算法要点

### 7.1 点到直线距离

```cpp
DistanceResult DistancePointToLine(const Point2d& point, const Line2d& line) {
    // 确保直线已归一化 (a^2 + b^2 = 1)
    Line2d normalized = NormalizeLine(line);
    
    // 有符号距离: d = a*x + b*y + c
    double signedDist = normalized.SignedDistance(point);
    
    // 最近点: P' = P - d * n, 其中 n = (a, b)
    Point2d closestPoint = ProjectPointOnLine(point, normalized);
    
    return {std::abs(signedDist), closestPoint, 0.0};
}
```

### 7.2 点到线段距离

```cpp
DistanceResult DistancePointToSegment(const Point2d& point, const Segment2d& segment) {
    // 处理退化线段
    double segLength = segment.Length();
    if (segLength < MIN_SEGMENT_LENGTH) {
        return {point.DistanceTo(segment.p1), segment.p1, 0.0};
    }
    
    // 计算投影参数 t
    Point2d d = segment.Direction();
    double t = (point - segment.p1).Dot(d) / (segLength * segLength);
    
    // 限制 t 到 [0, 1]
    t = std::clamp(t, 0.0, 1.0);
    
    // 计算最近点
    Point2d closestPoint = segment.PointAt(t);
    double distance = point.DistanceTo(closestPoint);
    
    return {distance, closestPoint, t};
}
```

### 7.3 点到圆距离

```cpp
DistanceResult DistancePointToCircle(const Point2d& point, const Circle2d& circle) {
    double distToCenter = point.DistanceTo(circle.center);
    
    // 处理点在圆心的情况
    if (distToCenter < GEOM_TOLERANCE) {
        // 返回圆上任意点 (选择正X方向)
        Point2d closestPoint = {circle.center.x + circle.radius, circle.center.y};
        return {circle.radius, closestPoint, 0.0};
    }
    
    // 计算圆上最近点
    double scale = circle.radius / distToCenter;
    Point2d direction = (point - circle.center) * (1.0 / distToCenter);
    Point2d closestPoint = circle.center + direction * circle.radius;
    
    // 距离 = |到圆心距离 - 半径|
    double distance = std::abs(distToCenter - circle.radius);
    
    // 参数: 角度
    double theta = std::atan2(direction.y, direction.x);
    
    return {distance, closestPoint, theta};
}
```

### 7.4 点到椭圆距离 (Newton迭代)

```cpp
DistanceResult DistancePointToEllipse(const Point2d& point, const Ellipse2d& ellipse,
                                       int maxIterations, double tolerance) {
    // 1. 将点转换到椭圆局部坐标系
    Point2d local = TransformToLocal(point, ellipse);
    
    // 2. 处理圆形椭圆
    if (std::abs(ellipse.a - ellipse.b) < GEOM_TOLERANCE) {
        Circle2d circle(ellipse.center, ellipse.a);
        return DistancePointToCircle(point, circle);
    }
    
    // 3. Newton迭代求解最近点参数
    // 目标: 最小化 ||P - E(t)||^2
    // 其中 E(t) = (a*cos(t), b*sin(t))
    // 梯度为零条件: (P - E(t)) · E'(t) = 0
    
    double t = std::atan2(local.y * ellipse.a, local.x * ellipse.b);  // 初始估计
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        double cosT = std::cos(t);
        double sinT = std::sin(t);
        
        // E(t) 和 E'(t)
        double ex = ellipse.a * cosT;
        double ey = ellipse.b * sinT;
        double dex = -ellipse.a * sinT;
        double dey = ellipse.b * cosT;
        
        // 梯度 f(t) = (P - E) · E'
        double dx = local.x - ex;
        double dy = local.y - ey;
        double f = dx * dex + dy * dey;
        
        // 二阶导数 f'(t)
        double ddex = -ellipse.a * cosT;
        double ddey = -ellipse.b * sinT;
        double df = -dex * dex - dey * dey + dx * ddex + dy * ddey;
        
        // Newton更新
        if (std::abs(df) < tolerance) break;
        double dt = -f / df;
        t += dt;
        
        if (std::abs(dt) < tolerance) break;
    }
    
    // 4. 计算最近点 (世界坐标)
    Point2d localClosest = {ellipse.a * std::cos(t), ellipse.b * std::sin(t)};
    Point2d closestPoint = TransformToWorld(localClosest, ellipse);
    
    double distance = point.DistanceTo(closestPoint);
    return {distance, closestPoint, t};
}
```

### 7.5 点到圆弧距离

```cpp
DistanceResult DistancePointToArc(const Point2d& point, const Arc2d& arc) {
    // 1. 计算到完整圆的最近点
    Circle2d circle = arc.ToCircle();
    DistanceResult circleResult = DistancePointToCircle(point, circle);
    
    // 2. 检查最近点是否在弧的角度范围内
    double theta = circleResult.parameter;  // 最近点的角度
    if (AngleInArcRange(theta, arc)) {
        return circleResult;
    }
    
    // 3. 最近点不在弧上,比较到两个端点的距离
    Point2d startPoint = arc.StartPoint();
    Point2d endPoint = arc.EndPoint();
    
    double distToStart = point.DistanceTo(startPoint);
    double distToEnd = point.DistanceTo(endPoint);
    
    if (distToStart <= distToEnd) {
        return {distToStart, startPoint, -1.0};  // -1 表示起点
    } else {
        return {distToEnd, endPoint, -2.0};       // -2 表示终点
    }
}
```

### 7.6 线段到线段距离

```cpp
SegmentDistanceResult DistanceSegmentToSegment(const Segment2d& seg1, const Segment2d& seg2) {
    // 使用参数化方法:
    // P1(s) = seg1.p1 + s * d1, s in [0,1]
    // P2(t) = seg2.p1 + t * d2, t in [0,1]
    // 最小化 ||P1(s) - P2(t)||^2
    
    Point2d d1 = seg1.Direction();
    Point2d d2 = seg2.Direction();
    Point2d r = seg1.p1 - seg2.p1;
    
    double a = d1.Dot(d1);  // |d1|^2
    double b = d1.Dot(d2);  // d1 · d2
    double c = d2.Dot(d2);  // |d2|^2
    double d = d1.Dot(r);   // d1 · r
    double e = d2.Dot(r);   // d2 · r
    
    double denom = a * c - b * b;
    
    double s, t;
    
    if (std::abs(denom) < GEOM_TOLERANCE) {
        // 线段平行
        s = 0.0;
        t = (b > c ? d / b : e / c);
    } else {
        s = (b * e - c * d) / denom;
        t = (a * e - b * d) / denom;
    }
    
    // 限制到 [0, 1]
    s = std::clamp(s, 0.0, 1.0);
    t = std::clamp(t, 0.0, 1.0);
    
    // 如果限制后需要重新计算另一个参数
    // (这里简化处理,完整实现需要迭代或分区处理)
    
    Point2d closest1 = seg1.PointAt(s);
    Point2d closest2 = seg2.PointAt(t);
    double distance = closest1.DistanceTo(closest2);
    
    return {distance, closest1, closest2, s, t};
}
```

### 7.7 点到轮廓距离

```cpp
ContourDistanceResult DistancePointToContour(const Point2d& point,
                                              const std::vector<Point2d>& contourPoints,
                                              bool closed) {
    if (contourPoints.size() < 2) {
        return ContourDistanceResult::Invalid();
    }
    
    double minDist = std::numeric_limits<double>::max();
    Point2d minClosest;
    size_t minSegIdx = 0;
    double minParam = 0.0;
    
    size_t numSegs = closed ? contourPoints.size() : contourPoints.size() - 1;
    
    for (size_t i = 0; i < numSegs; ++i) {
        size_t j = (i + 1) % contourPoints.size();
        Segment2d seg(contourPoints[i], contourPoints[j]);
        
        DistanceResult result = DistancePointToSegment(point, seg);
        
        if (result.distance < minDist) {
            minDist = result.distance;
            minClosest = result.closestPoint;
            minSegIdx = i;
            minParam = result.parameter;
        }
    }
    
    return {minDist, minClosest, minSegIdx, minParam};
}
```

---

## 8. 与已有模块的关系

### 8.1 与 Core/Types.h 的关系

Types.h 已定义:
- Point2d::DistanceTo() - 点到点距离
- Line2d::SignedDistance(), Distance() - 点到直线距离
- Segment2d::DistanceToPoint() - 点到线段距离

Distance.h 补充:
- 统一的结果结构 (包含最近点)
- 更多基元间距离 (椭圆、弧、轮廓)
- 有符号距离版本
- 批量计算函数

### 8.2 与 Internal/Geometry2d.h 的关系

复用 Geometry2d.h 的函数:
- NormalizeLine() - 直线归一化
- ProjectPointOnLine/Segment/Circle() - 投影函数
- AngleInArcRange() - 弧角度范围检查
- RotatedRectCorners/Edges() - 旋转矩形分解

### 8.3 与 Internal/Fitting.h 的关系

Fitting.h 可使用 Distance.h:
- RANSAC 中计算点到模型的残差
- FitLineLeastSquares 中的误差计算
- FitCircleLeastSquares 中的残差计算

---

## 9. 实现任务分解

| 任务 | 文件 | 预估时间 | 依赖 | 优先级 |
|------|------|----------|------|--------|
| 头文件 API 定义 | Distance.h | 1.5h | Types.h, Geometry2d.h | P0 |
| 结果结构体 | Distance.h | 0.5h | - | P0 |
| Point-Point/Line/Segment | Distance.cpp | 1h | Geometry2d.h | P0 |
| Point-Circle | Distance.cpp | 0.5h | - | P0 |
| Point-Ellipse (Newton) | Distance.cpp | 2h | - | P1 |
| Point-Arc | Distance.cpp | 1h | Geometry2d.h | P0 |
| Point-RotatedRect | Distance.cpp | 1h | Geometry2d.h | P0 |
| Line-Line | Distance.cpp | 0.5h | - | P0 |
| Segment-Segment | Distance.cpp | 1.5h | - | P1 |
| Circle-Circle | Distance.cpp | 0.5h | - | P0 |
| Point-Contour | Distance.cpp | 1h | - | P0 |
| Signed Distance 版本 | Distance.cpp | 1h | 基础版本 | P1 |
| 批量计算函数 | Distance.cpp | 1h | 基础版本 | P1 |
| Hausdorff 距离 | Distance.cpp | 1h | Point-Contour | P2 |
| 单元测试 | test_distance.cpp | 3h | 全部 | P0 |

**总计**: 约 17 小时

**实现顺序建议**:
1. P0 阶段: 头文件 + 基础距离函数 (~8h)
2. P1 阶段: 椭圆/线段对/有符号/批量 (~5h)
3. P2 阶段: Hausdorff + 高级函数 (~1h)
4. 测试 (~3h)

---

## 10. 测试要点

### 10.1 单元测试覆盖

1. **Point-Point 测试**
   - 相同点 (距离 = 0)
   - 不同点 (正距离)
   - 大坐标值

2. **Point-Line 测试**
   - 点在线上 (距离 = 0)
   - 点在法向量方向
   - 点在法向量反方向
   - 有符号距离正负

3. **Point-Segment 测试**
   - 最近点在内部
   - 最近点在端点 p1
   - 最近点在端点 p2
   - 退化线段 (零长度)

4. **Point-Circle 测试**
   - 点在圆上 (距离 = 0)
   - 点在圆外
   - 点在圆内
   - 点在圆心

5. **Point-Ellipse 测试**
   - 圆形椭圆 (a == b)
   - 点在长轴方向
   - 点在短轴方向
   - 点在椭圆上
   - 旋转椭圆

6. **Point-Arc 测试**
   - 最近点在弧上
   - 最近点在起点
   - 最近点在终点
   - 零扫掠角弧

7. **Point-RotatedRect 测试**
   - 点在内部
   - 点在外部各方向
   - 点在边上
   - 点在角上

8. **Segment-Segment 测试**
   - 相交线段
   - 平行线段
   - 端点相交
   - 不相交

9. **Point-Contour 测试**
   - 开轮廓
   - 闭轮廓
   - 有符号距离 (内/外)

### 10.2 边界条件测试

- 零长度线段
- 零半径圆
- 退化椭圆 (a == b)
- 空轮廓
- 单点轮廓
- 共线点
- 极大/极小坐标

### 10.3 精度测试

```cpp
// 示例: Point-Line 精度
TEST(DistanceAccuracy, PointToLine) {
    Line2d line = Line2d::FromPoints({0, 0}, {100, 0});  // X轴
    Point2d point(50, 10);
    
    auto result = DistancePointToLine(point, line);
    
    EXPECT_NEAR(result.distance, 10.0, 1e-14);
    EXPECT_NEAR(result.closestPoint.x, 50.0, 1e-14);
    EXPECT_NEAR(result.closestPoint.y, 0.0, 1e-14);
}

// 示例: Point-Ellipse 精度
TEST(DistanceAccuracy, PointToEllipse) {
    Ellipse2d ellipse({0, 0}, 100, 50, 0);  // a=100, b=50
    Point2d point(150, 0);  // 在长轴方向
    
    auto result = DistancePointToEllipse(point, ellipse);
    
    EXPECT_NEAR(result.distance, 50.0, 1e-10);
    EXPECT_NEAR(result.closestPoint.x, 100.0, 1e-10);
    EXPECT_NEAR(result.closestPoint.y, 0.0, 1e-10);
}

// 示例: 有符号距离
TEST(DistanceAccuracy, SignedDistanceToCircle) {
    Circle2d circle({0, 0}, 100);
    
    // 外部点
    Point2d outside(150, 0);
    auto resultOut = SignedDistancePointToCircle(outside, circle);
    EXPECT_GT(resultOut.signedDistance, 0);  // 正值
    EXPECT_NEAR(resultOut.signedDistance, 50.0, 1e-14);
    
    // 内部点
    Point2d inside(50, 0);
    auto resultIn = SignedDistancePointToCircle(inside, circle);
    EXPECT_LT(resultIn.signedDistance, 0);   // 负值
    EXPECT_NEAR(resultIn.signedDistance, -50.0, 1e-14);
}
```

---

## 11. 线程安全

### 11.1 线程安全保证

| 函数类型 | 线程安全性 |
|----------|------------|
| 所有函数 | 可重入 (输入只读) |
| 结果值类型 | 线程隔离 |

### 11.2 无全局状态

- 所有函数为纯函数
- 无静态变量
- 无缓存状态
- 批量函数内部无共享状态

---

## 12. 未来扩展

1. **SIMD 优化**: 批量距离计算的向量化
2. **更多基元**: 多边形、贝塞尔曲线、样条
3. **空间索引**: 使用 KD-Tree 加速轮廓距离查询
4. **3D 扩展**: 点到平面、点到球面等
5. **GPU 加速**: 大规模点集的并行距离计算

---

## 附录 A: 与 Halcon 对应

| QiVision | Halcon |
|----------|--------|
| DistancePointToPoint | distance_pp |
| DistancePointToLine | distance_pl |
| DistancePointToSegment | distance_ps |
| DistancePointToCircle | distance_pc |
| DistanceLineToLine | distance_ll |
| DistanceSegmentToSegment | distance_ss |
| DistanceCircleToCircle | distance_cc |
| DistancePointToContour | distance_contour_points_xld |
| HausdorffDistance | hausdorff_distance_contours_xld |

---

## 附录 B: API 快速参考

```cpp
// 点到点
double d = DistancePointToPoint(p1, p2);
double d2 = DistancePointToPointSquared(p1, p2);  // 无 sqrt

// 点到直线
DistanceResult r = DistancePointToLine(point, line);
SignedDistanceResult sr = SignedDistancePointToLine(point, line);

// 点到线段
DistanceResult r = DistancePointToSegment(point, segment);
SignedDistanceResult sr = SignedDistancePointToSegment(point, segment);

// 点到圆
DistanceResult r = DistancePointToCircle(point, circle);
SignedDistanceResult sr = SignedDistancePointToCircle(point, circle);

// 点到椭圆
DistanceResult r = DistancePointToEllipse(point, ellipse, maxIter, tol);

// 点到弧
DistanceResult r = DistancePointToArc(point, arc);

// 点到旋转矩形
DistanceResult r = DistancePointToRotatedRect(point, rect);
SignedDistanceResult sr = SignedDistancePointToRotatedRect(point, rect);

// 直线到直线
std::optional<double> d = DistanceLineToLine(line1, line2);

// 线段到线段
SegmentDistanceResult r = DistanceSegmentToSegment(seg1, seg2);

// 圆到圆
CircleDistanceResult r = DistanceCircleToCircle(circle1, circle2);

// 点到轮廓
ContourDistanceResult r = DistancePointToContour(point, contourPoints, closed);
SignedDistanceResult sr = SignedDistancePointToContour(point, contourPoints);

// 批量计算
std::vector<double> dists = DistancePointsToLine(points, line);
std::vector<double> dists = SignedDistancePointsToCircle(points, circle);

// 轮廓距离
double hd = HausdorffDistance(contour1, contour2, closed1, closed2);
double avgD = AverageDistanceContourToContour(contour1, contour2, closed2);
```
