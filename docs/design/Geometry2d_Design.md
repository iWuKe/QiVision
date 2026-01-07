# Internal/Geometry2d 设计文档

## 1. 概述

### 1.1 功能描述

Geometry2d 模块是 QiVision Internal 层的几何基元操作库,为 Distance、Intersection、GeomRelation、GeomConstruct、Homography 等模块提供基础。该模块提供对 Core/Types.h 中定义的几何基元(Point2d、Line2d、Segment2d、Circle2d、Ellipse2d、Arc2d、RotatedRect2d)的规范化、变换、属性计算、采样和构造功能。

### 1.2 应用场景

- **几何构造**: 从点/线构造圆、切线、垂线等
- **测量计算**: 长度、面积、周长、边界框计算
- **坐标变换**: 平移、旋转、缩放几何基元
- **路径采样**: 将连续几何基元离散化为点序列
- **模板匹配**: 模型轮廓生成与变换
- **标定**: 几何基元的世界坐标变换
- **Metrology**: 测量对象的几何描述

### 1.3 参考

**Halcon 相关算子**:
- `gen_contour_*`: 从几何基元生成轮廓 (gen_contour_circle, gen_contour_ellipse_xld 等)
- `affine_trans_*`: 几何基元仿射变换
- `project_*`: 点投影到几何基元
- 几何属性算子: `contlength`, `area_center`, `circularity` 等

**数学基础**:
- 解析几何: 点、线、圆的标准方程
- 仿射变换: 2D 平移、旋转、缩放矩阵
- 参数曲线: 圆弧/椭圆弧的参数方程

### 1.4 设计原则

1. **纯函数**: 无全局状态,所有函数可重入
2. **高精度**: 使用 double 类型,避免精度损失
3. **鲁棒性**: 处理退化情况(零长度线段、零半径圆等)
4. **复用**: 与 Core/Types.h 中已有方法互补,不重复
5. **一致性**: 统一的命名规范和参数顺序

---

## 2. 设计规则验证

### 2.1 坐标类型符合规则

- [x] 所有坐标使用 `double` 类型 (亚像素精度)
- [x] 角度使用弧度制 (`double`)
- [x] 返回整数像素坐标时使用 `int32_t`

### 2.2 层级依赖正确

- [x] Geometry2d.h 位于 Internal 层
- [x] 依赖 Core/Types.h (几何基元定义)
- [x] 依赖 Core/Constants.h (数学常量)
- [x] 可选依赖 Internal/Matrix.h (变换矩阵)
- [x] 不依赖 Feature 层
- [x] 不跨层依赖 Platform 层

### 2.3 算法完整性

- [x] 规范化/标准化: 直线归一化、角度归一化
- [x] 变换: 平移、旋转、缩放、仿射变换
- [x] 属性计算: 长度、面积、周长、边界框、质心
- [x] 采样/离散化: 生成点序列
- [x] 构造辅助: 从点构造线、从三点构造圆等

### 2.4 退化情况处理

- [x] 零长度线段: 返回有效结果或标记
- [x] 零半径圆: 作为点处理
- [x] 共线点: 无法构造圆时返回 nullopt
- [x] 无效角度范围: 归一化处理

---

## 3. 依赖分析

### 3.1 依赖的 Internal 模块

| 模块 | 用途 | 状态 |
|------|------|------|
| Internal/Matrix.h | Mat33 仿射变换矩阵 | ✅ 已完成 |

### 3.2 依赖的 Core 类型

| 类型 | 用途 |
|------|------|
| Core/Types.h | Point2d, Line2d, Segment2d, Circle2d, Ellipse2d, Arc2d, RotatedRect2d, Rect2d |
| Core/Constants.h | PI, EPSILON, DEG_TO_RAD, RAD_TO_DEG |

### 3.3 被依赖的模块

| 模块 | 用途 | 状态 |
|------|------|------|
| Internal/Distance.h | 距离计算基础 | ⬜ 待设计 |
| Internal/Intersection.h | 交点计算基础 | ⬜ 待设计 |
| Internal/GeomRelation.h | 几何关系判断基础 | ⬜ 待设计 |
| Internal/GeomConstruct.h | 几何构造基础 | ⬜ 待设计 |
| Internal/Homography.h | 单应性变换基础 | ⬜ 待设计 |
| Internal/Fitting.h | 几何拟合辅助 | ✅ 已完成 |
| Feature/Measure/* | 测量对象几何 | ⬜ 待设计 |
| Feature/Matching/* | 模型轮廓生成 | ⬜ 待设计 |

---

## 4. 类设计

### 4.1 模块结构

```
Geometry2d Module
├── Constants
│   ├── GEOM_TOLERANCE              - 几何计算容差
│   ├── ANGLE_TOLERANCE             - 角度比较容差
│   ├── DEFAULT_SAMPLING_STEP       - 默认采样步长
│   └── MAX_SAMPLING_POINTS         - 最大采样点数
│
├── Enumerations
│   ├── ArcDirection                - 弧方向 (CW/CCW)
│   └── SamplingMode                - 采样模式 (ByStep/ByCount/Adaptive)
│
├── Normalization Functions
│   ├── NormalizeLine()             - 直线系数归一化
│   ├── NormalizeAngle()            - 角度归一化到 [-PI, PI)
│   ├── NormalizeAngle0To2PI()      - 角度归一化到 [0, 2PI)
│   └── NormalizeEllipse()          - 椭圆参数规范化 (a >= b)
│
├── Point Operations
│   ├── RotatePoint()               - 绕原点/指定点旋转
│   ├── ScalePoint()                - 缩放点
│   ├── TranslatePoint()            - 平移点
│   └── TransformPoint()            - 仿射变换点
│
├── Line/Segment Operations
│   ├── LinePerpendicular()         - 过点作垂线
│   ├── LineParallel()              - 过点作平行线
│   ├── LineFromPointAndAngle()     - 从点和角度构造线
│   ├── SegmentFromLine()           - 从直线截取线段
│   ├── TransformLine()             - 变换直线
│   ├── TransformSegment()          - 变换线段
│   └── ExtendSegment()             - 延长线段
│
├── Circle/Arc Operations
│   ├── CircleFrom3Points()         - 三点确定圆 (复用 Fitting)
│   ├── ArcFrom3Points()            - 三点确定弧
│   ├── ArcFromCenter()             - 圆心+起止角确定弧
│   ├── TransformCircle()           - 变换圆
│   ├── TransformArc()              - 变换弧
│   ├── ArcToChord()                - 弧对应的弦
│   ├── ArcToSector()               - 弧对应的扇形
│   └── SplitArc()                  - 分割弧
│
├── Ellipse Operations
│   ├── TransformEllipse()          - 变换椭圆
│   ├── EllipseRadiusAt()           - 椭圆在角度θ处的半径
│   ├── EllipsePointAt()            - 椭圆在角度θ处的点
│   ├── EllipseTangentAt()          - 椭圆在角度θ处的切线
│   └── EllipseArcLength()          - 椭圆弧长 (近似)
│
├── RotatedRect Operations
│   ├── TransformRotatedRect()      - 变换旋转矩形
│   ├── RotatedRectCorners()        - 获取四个角点
│   ├── RotatedRectEdges()          - 获取四条边
│   └── RotatedRectContains()       - 点是否在矩形内
│
├── Property Computation
│   ├── SegmentLength()             - 线段长度 (已有 Segment2d::Length)
│   ├── ArcLength()                 - 弧长
│   ├── EllipsePerimeter()          - 椭圆周长 (已有 Ellipse2d::Perimeter)
│   ├── CircleArea()                - 圆面积 (已有 Circle2d::Area)
│   ├── EllipseArea()               - 椭圆面积 (已有 Ellipse2d::Area)
│   ├── RotatedRectArea()           - 旋转矩形面积 (已有)
│   ├── ArcSectorArea()             - 扇形面积
│   ├── ArcSegmentArea()            - 弓形面积
│   └── ComputeBoundingBox()        - 计算边界框
│
├── Sampling/Discretization
│   ├── SampleLine()                - 采样直线 (指定范围)
│   ├── SampleSegment()             - 采样线段
│   ├── SampleCircle()              - 采样圆
│   ├── SampleArc()                 - 采样圆弧
│   ├── SampleEllipse()             - 采样椭圆
│   ├── SampleEllipseArc()          - 采样椭圆弧
│   ├── SampleRotatedRect()         - 采样旋转矩形边界
│   └── ComputeSamplingCount()      - 计算合适采样点数
│
└── Utility Functions
    ├── PointOnLine()               - 判断点是否在直线上
    ├── PointOnSegment()            - 判断点是否在线段上
    ├── PointOnCircle()             - 判断点是否在圆上
    ├── PointInCircle()             - 判断点是否在圆内 (已有 Circle2d::Contains)
    ├── PointInEllipse()            - 判断点是否在椭圆内 (已有 Ellipse2d::Contains)
    ├── PointInRotatedRect()        - 判断点是否在旋转矩形内 (已有)
    ├── AngleBetweenLines()         - 两线夹角
    ├── SignedAngle()               - 有符号角度 (考虑方向)
    └── AreParallel() / ArePerpendicular() - 平行/垂直判断
```

### 4.2 API 设计

```cpp
#pragma once

/**
 * @file Geometry2d.h
 * @brief 2D geometric primitive operations for QiVision
 *
 * This module provides:
 * - Normalization of geometric primitives
 * - Transformation (translate, rotate, scale, affine)
 * - Property computation (length, area, perimeter, bounding box)
 * - Sampling/discretization (generate point sequences)
 * - Construction helpers
 *
 * Used by:
 * - Internal/Distance: Distance calculations
 * - Internal/Intersection: Intersection calculations
 * - Internal/GeomRelation: Geometric relation tests
 * - Internal/GeomConstruct: Geometric construction
 * - Feature/Measure: Measurement objects
 * - Feature/Matching: Model contour generation
 *
 * Design principles:
 * - All functions are pure (no global state)
 * - All coordinates use double for subpixel precision
 * - Angles in radians
 * - Graceful handling of degenerate cases
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/Constants.h>
#include <QiVision/Internal/Matrix.h>

#include <cmath>
#include <vector>
#include <optional>
#include <array>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

/// Tolerance for geometric comparisons (distance)
constexpr double GEOM_TOLERANCE = 1e-9;

/// Tolerance for angle comparisons (radians)
constexpr double ANGLE_TOLERANCE = 1e-9;

/// Default sampling step in pixels
constexpr double DEFAULT_SAMPLING_STEP = 1.0;

/// Maximum number of sampling points to prevent memory issues
constexpr size_t MAX_SAMPLING_POINTS = 1000000;

/// Minimum valid segment length
constexpr double MIN_SEGMENT_LENGTH = 1e-12;

/// Minimum valid radius
constexpr double MIN_RADIUS = 1e-12;

// =============================================================================
// Enumerations
// =============================================================================

/**
 * @brief Arc direction enumeration
 */
enum class ArcDirection {
    CounterClockwise,   ///< Positive angle direction (CCW), default
    Clockwise           ///< Negative angle direction (CW)
};

/**
 * @brief Sampling mode for discretization
 */
enum class SamplingMode {
    ByStep,     ///< Fixed step size (pixels)
    ByCount,    ///< Fixed number of points
    Adaptive    ///< Adaptive based on curvature
};

// =============================================================================
// Normalization Functions
// =============================================================================

/**
 * @brief Normalize a line equation so that a^2 + b^2 = 1
 *
 * @param line Input line (may be unnormalized)
 * @return Normalized line
 *
 * @note If line is degenerate (a = b = 0), returns line unchanged
 */
Line2d NormalizeLine(const Line2d& line);

/**
 * @brief Normalize angle to range [-PI, PI)
 *
 * @param angle Input angle in radians
 * @return Normalized angle in [-PI, PI)
 */
inline double NormalizeAngle(double angle) {
    while (angle >= PI) angle -= TWO_PI;
    while (angle < -PI) angle += TWO_PI;
    return angle;
}

/**
 * @brief Normalize angle to range [0, 2*PI)
 *
 * @param angle Input angle in radians
 * @return Normalized angle in [0, 2*PI)
 */
inline double NormalizeAngle0To2PI(double angle) {
    while (angle >= TWO_PI) angle -= TWO_PI;
    while (angle < 0) angle += TWO_PI;
    return angle;
}

/**
 * @brief Normalize angle difference to range [-PI, PI]
 *
 * @param angleDiff Angle difference in radians
 * @return Normalized angle difference
 */
inline double NormalizeAngleDiff(double angleDiff) {
    while (angleDiff > PI) angleDiff -= TWO_PI;
    while (angleDiff < -PI) angleDiff += TWO_PI;
    return angleDiff;
}

/**
 * @brief Normalize ellipse parameters so that a >= b
 *
 * If a < b, swaps axes and adjusts angle by PI/2.
 *
 * @param ellipse Input ellipse
 * @return Normalized ellipse with a >= b
 */
Ellipse2d NormalizeEllipse(const Ellipse2d& ellipse);

/**
 * @brief Normalize arc parameters
 *
 * Ensures startAngle is in [0, 2*PI) and sweepAngle is in [-2*PI, 2*PI].
 *
 * @param arc Input arc
 * @return Normalized arc
 */
Arc2d NormalizeArc(const Arc2d& arc);

// =============================================================================
// Point Operations
// =============================================================================

/**
 * @brief Rotate point around origin
 *
 * @param point Input point
 * @param angle Rotation angle in radians (positive = CCW)
 * @return Rotated point
 */
inline Point2d RotatePoint(const Point2d& point, double angle) {
    double cosA = std::cos(angle);
    double sinA = std::sin(angle);
    return {
        point.x * cosA - point.y * sinA,
        point.x * sinA + point.y * cosA
    };
}

/**
 * @brief Rotate point around a specified center
 *
 * @param point Input point
 * @param center Rotation center
 * @param angle Rotation angle in radians
 * @return Rotated point
 */
inline Point2d RotatePointAround(const Point2d& point, const Point2d& center, double angle) {
    Point2d translated = point - center;
    Point2d rotated = RotatePoint(translated, angle);
    return rotated + center;
}

/**
 * @brief Scale point relative to origin
 *
 * @param point Input point
 * @param scaleX Scale factor in X
 * @param scaleY Scale factor in Y
 * @return Scaled point
 */
inline Point2d ScalePoint(const Point2d& point, double scaleX, double scaleY) {
    return {point.x * scaleX, point.y * scaleY};
}

/**
 * @brief Scale point uniformly relative to origin
 *
 * @param point Input point
 * @param scale Uniform scale factor
 * @return Scaled point
 */
inline Point2d ScalePoint(const Point2d& point, double scale) {
    return point * scale;
}

/**
 * @brief Scale point relative to a specified center
 *
 * @param point Input point
 * @param center Scale center
 * @param scaleX Scale factor in X
 * @param scaleY Scale factor in Y
 * @return Scaled point
 */
inline Point2d ScalePointAround(const Point2d& point, const Point2d& center,
                                 double scaleX, double scaleY) {
    return Point2d{
        center.x + (point.x - center.x) * scaleX,
        center.y + (point.y - center.y) * scaleY
    };
}

/**
 * @brief Translate point
 *
 * @param point Input point
 * @param dx Translation in X
 * @param dy Translation in Y
 * @return Translated point
 */
inline Point2d TranslatePoint(const Point2d& point, double dx, double dy) {
    return {point.x + dx, point.y + dy};
}

/**
 * @brief Transform point using affine matrix
 *
 * @param point Input point
 * @param matrix 3x3 affine transformation matrix
 * @return Transformed point
 */
Point2d TransformPoint(const Point2d& point, const Mat33& matrix);

/**
 * @brief Transform multiple points using affine matrix
 *
 * @param points Input points
 * @param matrix 3x3 affine transformation matrix
 * @return Transformed points
 */
std::vector<Point2d> TransformPoints(const std::vector<Point2d>& points, const Mat33& matrix);

// =============================================================================
// Line/Segment Operations
// =============================================================================

/**
 * @brief Create a line perpendicular to given line through a point
 *
 * @param line Reference line
 * @param point Point the perpendicular passes through
 * @return Perpendicular line
 */
Line2d LinePerpendicular(const Line2d& line, const Point2d& point);

/**
 * @brief Create a line parallel to given line through a point
 *
 * @param line Reference line
 * @param point Point the parallel passes through
 * @return Parallel line
 */
Line2d LineParallel(const Line2d& line, const Point2d& point);

/**
 * @brief Create a line from point and angle
 *
 * @param point Point on the line
 * @param angle Line angle in radians (direction)
 * @return Line passing through point at given angle
 */
Line2d LineFromPointAndAngle(const Point2d& point, double angle);

/**
 * @brief Transform line using affine matrix
 *
 * @param line Input line
 * @param matrix 3x3 affine transformation matrix
 * @return Transformed line
 */
Line2d TransformLine(const Line2d& line, const Mat33& matrix);

/**
 * @brief Transform segment using affine matrix
 *
 * @param segment Input segment
 * @param matrix 3x3 affine transformation matrix
 * @return Transformed segment
 */
Segment2d TransformSegment(const Segment2d& segment, const Mat33& matrix);

/**
 * @brief Translate segment
 *
 * @param segment Input segment
 * @param dx Translation in X
 * @param dy Translation in Y
 * @return Translated segment
 */
inline Segment2d TranslateSegment(const Segment2d& segment, double dx, double dy) {
    return Segment2d(
        TranslatePoint(segment.p1, dx, dy),
        TranslatePoint(segment.p2, dx, dy)
    );
}

/**
 * @brief Rotate segment around a center
 *
 * @param segment Input segment
 * @param center Rotation center
 * @param angle Rotation angle in radians
 * @return Rotated segment
 */
inline Segment2d RotateSegment(const Segment2d& segment, const Point2d& center, double angle) {
    return Segment2d(
        RotatePointAround(segment.p1, center, angle),
        RotatePointAround(segment.p2, center, angle)
    );
}

/**
 * @brief Extend segment by distances at both ends
 *
 * @param segment Input segment
 * @param extendStart Distance to extend at start (p1 side), can be negative
 * @param extendEnd Distance to extend at end (p2 side), can be negative
 * @return Extended segment
 */
Segment2d ExtendSegment(const Segment2d& segment, double extendStart, double extendEnd);

/**
 * @brief Clip a line to get a segment within bounds
 *
 * @param line Input line
 * @param bounds Clipping rectangle
 * @return Segment if line intersects bounds, nullopt otherwise
 */
std::optional<Segment2d> ClipLineToRect(const Line2d& line, const Rect2d& bounds);

/**
 * @brief Reverse segment direction (swap p1 and p2)
 *
 * @param segment Input segment
 * @return Reversed segment
 */
inline Segment2d ReverseSegment(const Segment2d& segment) {
    return Segment2d(segment.p2, segment.p1);
}

// =============================================================================
// Circle/Arc Operations
// =============================================================================

/**
 * @brief Translate circle
 *
 * @param circle Input circle
 * @param dx Translation in X
 * @param dy Translation in Y
 * @return Translated circle
 */
inline Circle2d TranslateCircle(const Circle2d& circle, double dx, double dy) {
    return Circle2d(TranslatePoint(circle.center, dx, dy), circle.radius);
}

/**
 * @brief Scale circle uniformly
 *
 * @param circle Input circle
 * @param scale Scale factor
 * @return Scaled circle (both center and radius)
 */
inline Circle2d ScaleCircle(const Circle2d& circle, double scale) {
    return Circle2d(ScalePoint(circle.center, scale), circle.radius * std::abs(scale));
}

/**
 * @brief Scale circle around a center
 *
 * @param circle Input circle
 * @param center Scale center
 * @param scale Scale factor
 * @return Scaled circle
 */
inline Circle2d ScaleCircleAround(const Circle2d& circle, const Point2d& center, double scale) {
    return Circle2d(
        ScalePointAround(circle.center, center, scale, scale),
        circle.radius * std::abs(scale)
    );
}

/**
 * @brief Transform circle using affine matrix
 *
 * @param circle Input circle
 * @param matrix 3x3 affine transformation matrix
 * @return Transformed ellipse (circle becomes ellipse under non-uniform scale/shear)
 *
 * @note If the transform has non-uniform scaling or shear, the result is an ellipse.
 */
Ellipse2d TransformCircle(const Circle2d& circle, const Mat33& matrix);

/**
 * @brief Create arc from three points on the arc
 *
 * @param p1 First point (start of arc)
 * @param p2 Second point (on arc, defines curvature)
 * @param p3 Third point (end of arc)
 * @return Arc passing through all three points, or nullopt if collinear
 */
std::optional<Arc2d> ArcFrom3Points(const Point2d& p1, const Point2d& p2, const Point2d& p3);

/**
 * @brief Create arc from center, radius, and angles
 *
 * @param center Arc center
 * @param radius Arc radius
 * @param startAngle Start angle in radians
 * @param endAngle End angle in radians
 * @param direction Arc direction (CCW or CW)
 * @return Arc
 */
Arc2d ArcFromAngles(const Point2d& center, double radius,
                    double startAngle, double endAngle,
                    ArcDirection direction = ArcDirection::CounterClockwise);

/**
 * @brief Translate arc
 *
 * @param arc Input arc
 * @param dx Translation in X
 * @param dy Translation in Y
 * @return Translated arc
 */
inline Arc2d TranslateArc(const Arc2d& arc, double dx, double dy) {
    return Arc2d(TranslatePoint(arc.center, dx, dy), arc.radius, arc.startAngle, arc.sweepAngle);
}

/**
 * @brief Scale arc uniformly around origin
 *
 * @param arc Input arc
 * @param scale Scale factor
 * @return Scaled arc
 */
inline Arc2d ScaleArc(const Arc2d& arc, double scale) {
    return Arc2d(ScalePoint(arc.center, scale), arc.radius * std::abs(scale),
                 arc.startAngle, arc.sweepAngle);
}

/**
 * @brief Transform arc using affine matrix
 *
 * @param arc Input arc
 * @param matrix 3x3 affine transformation matrix
 * @return Transformed arc (approximation if transform is not similarity)
 *
 * @note For non-uniform scaling or shear, this returns an approximate arc.
 *       For exact transformation, use ellipse arc.
 */
Arc2d TransformArc(const Arc2d& arc, const Mat33& matrix);

/**
 * @brief Get the chord (straight line segment) of an arc
 *
 * @param arc Input arc
 * @return Chord segment from start point to end point
 */
Segment2d ArcToChord(const Arc2d& arc);

/**
 * @brief Split arc at a parameter value
 *
 * @param arc Input arc
 * @param t Split parameter [0, 1], where 0 = start, 1 = end
 * @return Pair of arcs (first: start to t, second: t to end)
 */
std::pair<Arc2d, Arc2d> SplitArc(const Arc2d& arc, double t);

/**
 * @brief Reverse arc direction
 *
 * @param arc Input arc
 * @return Reversed arc (same geometry, opposite direction)
 */
inline Arc2d ReverseArc(const Arc2d& arc) {
    return Arc2d(arc.center, arc.radius, arc.EndAngle(), -arc.sweepAngle);
}

// =============================================================================
// Ellipse Operations
// =============================================================================

/**
 * @brief Translate ellipse
 *
 * @param ellipse Input ellipse
 * @param dx Translation in X
 * @param dy Translation in Y
 * @return Translated ellipse
 */
inline Ellipse2d TranslateEllipse(const Ellipse2d& ellipse, double dx, double dy) {
    return Ellipse2d(TranslatePoint(ellipse.center, dx, dy),
                     ellipse.a, ellipse.b, ellipse.angle);
}

/**
 * @brief Scale ellipse uniformly around origin
 *
 * @param ellipse Input ellipse
 * @param scale Scale factor
 * @return Scaled ellipse
 */
inline Ellipse2d ScaleEllipse(const Ellipse2d& ellipse, double scale) {
    return Ellipse2d(ScalePoint(ellipse.center, scale),
                     ellipse.a * std::abs(scale),
                     ellipse.b * std::abs(scale),
                     ellipse.angle);
}

/**
 * @brief Rotate ellipse around its center
 *
 * @param ellipse Input ellipse
 * @param angle Rotation angle in radians
 * @return Rotated ellipse
 */
inline Ellipse2d RotateEllipse(const Ellipse2d& ellipse, double angle) {
    return Ellipse2d(ellipse.center, ellipse.a, ellipse.b,
                     NormalizeAngle(ellipse.angle + angle));
}

/**
 * @brief Rotate ellipse around a specified center
 *
 * @param ellipse Input ellipse
 * @param center Rotation center
 * @param angle Rotation angle in radians
 * @return Rotated ellipse
 */
Ellipse2d RotateEllipseAround(const Ellipse2d& ellipse, const Point2d& center, double angle);

/**
 * @brief Transform ellipse using affine matrix
 *
 * @param ellipse Input ellipse
 * @param matrix 3x3 affine transformation matrix
 * @return Transformed ellipse
 */
Ellipse2d TransformEllipse(const Ellipse2d& ellipse, const Mat33& matrix);

/**
 * @brief Compute radius of ellipse at a given angle
 *
 * @param ellipse Input ellipse
 * @param theta Angle in ellipse local coordinates (radians)
 * @return Radius at angle theta
 */
double EllipseRadiusAt(const Ellipse2d& ellipse, double theta);

/**
 * @brief Get point on ellipse at a given angle (in ellipse local coordinates)
 *
 * Uses parametric form: x = a*cos(t), y = b*sin(t)
 *
 * @param ellipse Input ellipse
 * @param theta Parameter angle in radians
 * @return Point on ellipse
 */
Point2d EllipsePointAt(const Ellipse2d& ellipse, double theta);

/**
 * @brief Get tangent direction at a point on ellipse
 *
 * @param ellipse Input ellipse
 * @param theta Parameter angle in radians
 * @return Unit tangent vector at the point
 */
Point2d EllipseTangentAt(const Ellipse2d& ellipse, double theta);

/**
 * @brief Get normal direction at a point on ellipse
 *
 * @param ellipse Input ellipse
 * @param theta Parameter angle in radians
 * @return Unit normal vector (outward) at the point
 */
Point2d EllipseNormalAt(const Ellipse2d& ellipse, double theta);

/**
 * @brief Compute approximate arc length of ellipse segment
 *
 * Uses numerical integration (adaptive Simpson's rule).
 *
 * @param ellipse Input ellipse
 * @param thetaStart Start angle (parameter)
 * @param thetaEnd End angle (parameter)
 * @return Arc length
 */
double EllipseArcLength(const Ellipse2d& ellipse, double thetaStart, double thetaEnd);

// =============================================================================
// RotatedRect Operations
// =============================================================================

/**
 * @brief Translate rotated rectangle
 *
 * @param rect Input rectangle
 * @param dx Translation in X
 * @param dy Translation in Y
 * @return Translated rectangle
 */
inline RotatedRect2d TranslateRotatedRect(const RotatedRect2d& rect, double dx, double dy) {
    return RotatedRect2d(TranslatePoint(rect.center, dx, dy),
                         rect.width, rect.height, rect.angle);
}

/**
 * @brief Scale rotated rectangle uniformly around origin
 *
 * @param rect Input rectangle
 * @param scale Scale factor
 * @return Scaled rectangle
 */
inline RotatedRect2d ScaleRotatedRect(const RotatedRect2d& rect, double scale) {
    return RotatedRect2d(ScalePoint(rect.center, scale),
                         rect.width * std::abs(scale),
                         rect.height * std::abs(scale),
                         rect.angle);
}

/**
 * @brief Rotate rotated rectangle around its center
 *
 * @param rect Input rectangle
 * @param angle Rotation angle in radians
 * @return Rotated rectangle
 */
inline RotatedRect2d RotateRotatedRect(const RotatedRect2d& rect, double angle) {
    return RotatedRect2d(rect.center, rect.width, rect.height,
                         NormalizeAngle(rect.angle + angle));
}

/**
 * @brief Rotate rotated rectangle around a specified center
 *
 * @param rect Input rectangle
 * @param center Rotation center
 * @param angle Rotation angle in radians
 * @return Rotated rectangle
 */
RotatedRect2d RotateRotatedRectAround(const RotatedRect2d& rect, const Point2d& center, double angle);

/**
 * @brief Transform rotated rectangle using affine matrix
 *
 * @param rect Input rectangle
 * @param matrix 3x3 affine transformation matrix
 * @return Transformed rectangle (may not be exact if transform has shear)
 */
RotatedRect2d TransformRotatedRect(const RotatedRect2d& rect, const Mat33& matrix);

/**
 * @brief Get the four corners of a rotated rectangle
 *
 * @param rect Input rectangle
 * @return Array of 4 corners [topLeft, topRight, bottomRight, bottomLeft] in local coordinates
 */
std::array<Point2d, 4> RotatedRectCorners(const RotatedRect2d& rect);

/**
 * @brief Get the four edges of a rotated rectangle as segments
 *
 * @param rect Input rectangle
 * @return Array of 4 edges [top, right, bottom, left]
 */
std::array<Segment2d, 4> RotatedRectEdges(const RotatedRect2d& rect);

// =============================================================================
// Property Computation
// =============================================================================

/**
 * @brief Compute arc sector area
 *
 * @param arc Input arc
 * @return Area of the circular sector defined by the arc
 */
inline double ArcSectorArea(const Arc2d& arc) {
    return 0.5 * arc.radius * arc.radius * std::abs(arc.sweepAngle);
}

/**
 * @brief Compute arc segment (bow) area
 *
 * @param arc Input arc
 * @return Area of the circular segment (region between arc and chord)
 */
double ArcSegmentArea(const Arc2d& arc);

/**
 * @brief Compute bounding box of a circle
 *
 * @param circle Input circle
 * @return Axis-aligned bounding box
 */
inline Rect2d CircleBoundingBox(const Circle2d& circle) {
    return Rect2d(circle.center.x - circle.radius,
                  circle.center.y - circle.radius,
                  2.0 * circle.radius,
                  2.0 * circle.radius);
}

/**
 * @brief Compute bounding box of an arc
 *
 * @param arc Input arc
 * @return Axis-aligned bounding box
 */
Rect2d ArcBoundingBox(const Arc2d& arc);

/**
 * @brief Compute bounding box of an ellipse
 *
 * @param ellipse Input ellipse
 * @return Axis-aligned bounding box
 */
Rect2d EllipseBoundingBox(const Ellipse2d& ellipse);

/**
 * @brief Compute bounding box of a segment
 *
 * @param segment Input segment
 * @return Axis-aligned bounding box
 */
inline Rect2d SegmentBoundingBox(const Segment2d& segment) {
    double minX = std::min(segment.p1.x, segment.p2.x);
    double minY = std::min(segment.p1.y, segment.p2.y);
    double maxX = std::max(segment.p1.x, segment.p2.x);
    double maxY = std::max(segment.p1.y, segment.p2.y);
    return Rect2d(minX, minY, maxX - minX, maxY - minY);
}

/**
 * @brief Compute centroid of an arc (centroid of the arc curve, not sector)
 *
 * @param arc Input arc
 * @return Centroid point
 */
Point2d ArcCentroid(const Arc2d& arc);

/**
 * @brief Compute centroid of an arc sector
 *
 * @param arc Input arc
 * @return Centroid of the sector region
 */
Point2d ArcSectorCentroid(const Arc2d& arc);

// =============================================================================
// Sampling/Discretization
// =============================================================================

/**
 * @brief Sample points along a segment
 *
 * @param segment Input segment
 * @param step Sampling step (pixels)
 * @param includeEndpoints If true, always includes p1 and p2
 * @return Vector of sampled points
 */
std::vector<Point2d> SampleSegment(const Segment2d& segment, double step = DEFAULT_SAMPLING_STEP,
                                    bool includeEndpoints = true);

/**
 * @brief Sample fixed number of points along a segment
 *
 * @param segment Input segment
 * @param numPoints Number of points to sample (minimum 2)
 * @return Vector of sampled points
 */
std::vector<Point2d> SampleSegmentByCount(const Segment2d& segment, size_t numPoints);

/**
 * @brief Sample points along a circle
 *
 * @param circle Input circle
 * @param step Sampling step (arc length in pixels)
 * @return Vector of sampled points (closed, first point repeated at end)
 */
std::vector<Point2d> SampleCircle(const Circle2d& circle, double step = DEFAULT_SAMPLING_STEP);

/**
 * @brief Sample fixed number of points along a circle
 *
 * @param circle Input circle
 * @param numPoints Number of points to sample
 * @param closed If true, last point connects to first
 * @return Vector of sampled points
 */
std::vector<Point2d> SampleCircleByCount(const Circle2d& circle, size_t numPoints, bool closed = true);

/**
 * @brief Sample points along an arc
 *
 * @param arc Input arc
 * @param step Sampling step (arc length in pixels)
 * @param includeEndpoints If true, always includes start and end points
 * @return Vector of sampled points
 */
std::vector<Point2d> SampleArc(const Arc2d& arc, double step = DEFAULT_SAMPLING_STEP,
                                bool includeEndpoints = true);

/**
 * @brief Sample fixed number of points along an arc
 *
 * @param arc Input arc
 * @param numPoints Number of points to sample (minimum 2)
 * @return Vector of sampled points
 */
std::vector<Point2d> SampleArcByCount(const Arc2d& arc, size_t numPoints);

/**
 * @brief Sample points along an ellipse
 *
 * @param ellipse Input ellipse
 * @param step Approximate sampling step (arc length in pixels)
 * @return Vector of sampled points (closed)
 */
std::vector<Point2d> SampleEllipse(const Ellipse2d& ellipse, double step = DEFAULT_SAMPLING_STEP);

/**
 * @brief Sample fixed number of points along an ellipse
 *
 * @param ellipse Input ellipse
 * @param numPoints Number of points to sample
 * @param closed If true, last point connects to first
 * @return Vector of sampled points
 */
std::vector<Point2d> SampleEllipseByCount(const Ellipse2d& ellipse, size_t numPoints, bool closed = true);

/**
 * @brief Sample points along an ellipse arc
 *
 * @param ellipse Base ellipse
 * @param thetaStart Start angle (parameter)
 * @param thetaEnd End angle (parameter)
 * @param step Approximate sampling step
 * @param includeEndpoints If true, always includes start and end points
 * @return Vector of sampled points
 */
std::vector<Point2d> SampleEllipseArc(const Ellipse2d& ellipse,
                                       double thetaStart, double thetaEnd,
                                       double step = DEFAULT_SAMPLING_STEP,
                                       bool includeEndpoints = true);

/**
 * @brief Sample points along the boundary of a rotated rectangle
 *
 * @param rect Input rectangle
 * @param step Sampling step (pixels)
 * @param closed If true, includes closing segment back to first point
 * @return Vector of sampled points
 */
std::vector<Point2d> SampleRotatedRect(const RotatedRect2d& rect,
                                        double step = DEFAULT_SAMPLING_STEP,
                                        bool closed = true);

/**
 * @brief Compute recommended number of sampling points
 *
 * @param arcLength Total arc length
 * @param step Desired step size
 * @param minPoints Minimum number of points
 * @param maxPoints Maximum number of points
 * @return Recommended number of points
 */
inline size_t ComputeSamplingCount(double arcLength, double step,
                                    size_t minPoints = 2, size_t maxPoints = MAX_SAMPLING_POINTS) {
    if (arcLength <= 0 || step <= 0) return minPoints;
    size_t count = static_cast<size_t>(std::ceil(arcLength / step)) + 1;
    return std::clamp(count, minPoints, maxPoints);
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Check if a point lies on a line (within tolerance)
 *
 * @param point Point to check
 * @param line Line to check against
 * @param tolerance Distance tolerance
 * @return true if point is on line within tolerance
 */
inline bool PointOnLine(const Point2d& point, const Line2d& line, double tolerance = GEOM_TOLERANCE) {
    return std::abs(line.SignedDistance(point)) <= tolerance;
}

/**
 * @brief Check if a point lies on a segment (within tolerance)
 *
 * @param point Point to check
 * @param segment Segment to check against
 * @param tolerance Distance tolerance
 * @return true if point is on segment within tolerance
 */
bool PointOnSegment(const Point2d& point, const Segment2d& segment, double tolerance = GEOM_TOLERANCE);

/**
 * @brief Check if a point lies on a circle (within tolerance)
 *
 * @param point Point to check
 * @param circle Circle to check against
 * @param tolerance Distance tolerance
 * @return true if point is on circle circumference within tolerance
 */
inline bool PointOnCircle(const Point2d& point, const Circle2d& circle, double tolerance = GEOM_TOLERANCE) {
    double dist = point.DistanceTo(circle.center);
    return std::abs(dist - circle.radius) <= tolerance;
}

/**
 * @brief Check if a point lies on an arc (within tolerance)
 *
 * @param point Point to check
 * @param arc Arc to check against
 * @param tolerance Distance tolerance
 * @return true if point is on arc within tolerance
 */
bool PointOnArc(const Point2d& point, const Arc2d& arc, double tolerance = GEOM_TOLERANCE);

/**
 * @brief Check if a point lies on an ellipse (within tolerance)
 *
 * @param point Point to check
 * @param ellipse Ellipse to check against
 * @param tolerance Distance tolerance
 * @return true if point is on ellipse circumference within tolerance
 */
bool PointOnEllipse(const Point2d& point, const Ellipse2d& ellipse, double tolerance = GEOM_TOLERANCE);

/**
 * @brief Compute angle between two lines
 *
 * Returns the acute angle between the lines (always positive, <= PI/2).
 *
 * @param line1 First line
 * @param line2 Second line
 * @return Angle in radians [0, PI/2]
 */
double AngleBetweenLines(const Line2d& line1, const Line2d& line2);

/**
 * @brief Compute signed angle from vector1 to vector2
 *
 * @param v1 First vector (or direction)
 * @param v2 Second vector
 * @return Signed angle in radians [-PI, PI], positive = CCW
 */
inline double SignedAngle(const Point2d& v1, const Point2d& v2) {
    return std::atan2(v1.Cross(v2), v1.Dot(v2));
}

/**
 * @brief Check if two lines are parallel
 *
 * @param line1 First line
 * @param line2 Second line
 * @param tolerance Angular tolerance in radians
 * @return true if lines are parallel within tolerance
 */
bool AreParallel(const Line2d& line1, const Line2d& line2, double tolerance = ANGLE_TOLERANCE);

/**
 * @brief Check if two lines are perpendicular
 *
 * @param line1 First line
 * @param line2 Second line
 * @param tolerance Angular tolerance in radians
 * @return true if lines are perpendicular within tolerance
 */
bool ArePerpendicular(const Line2d& line1, const Line2d& line2, double tolerance = ANGLE_TOLERANCE);

/**
 * @brief Check if two segments are collinear
 *
 * @param seg1 First segment
 * @param seg2 Second segment
 * @param tolerance Distance tolerance
 * @return true if segments lie on the same line within tolerance
 */
bool AreCollinear(const Segment2d& seg1, const Segment2d& seg2, double tolerance = GEOM_TOLERANCE);

/**
 * @brief Project point onto line
 *
 * @param point Point to project
 * @param line Line to project onto
 * @return Projection of point onto line
 */
Point2d ProjectPointOnLine(const Point2d& point, const Line2d& line);

/**
 * @brief Project point onto segment (clamped to segment)
 *
 * @param point Point to project
 * @param segment Segment to project onto
 * @return Projection of point onto segment (clamped to endpoints)
 */
Point2d ProjectPointOnSegment(const Point2d& point, const Segment2d& segment);

/**
 * @brief Project point onto circle
 *
 * @param point Point to project
 * @param circle Circle to project onto
 * @return Closest point on circle to the given point
 */
Point2d ProjectPointOnCircle(const Point2d& point, const Circle2d& circle);

/**
 * @brief Compute foot of perpendicular from point to line
 *
 * Same as ProjectPointOnLine.
 *
 * @param point Point
 * @param line Line
 * @return Foot point
 */
inline Point2d FootOfPerpendicular(const Point2d& point, const Line2d& line) {
    return ProjectPointOnLine(point, line);
}

/**
 * @brief Reflect point across a line
 *
 * @param point Point to reflect
 * @param line Mirror line
 * @return Reflected point
 */
Point2d ReflectPointAcrossLine(const Point2d& point, const Line2d& line);

/**
 * @brief Compute midpoint of a segment
 *
 * Same as segment.Midpoint().
 *
 * @param segment Input segment
 * @return Midpoint
 */
inline Point2d SegmentMidpoint(const Segment2d& segment) {
    return segment.Midpoint();
}

/**
 * @brief Check if an angle is within an arc's angular range
 *
 * @param angle Angle to check (radians)
 * @param arc Arc defining the angular range
 * @return true if angle is within arc's sweep
 */
bool AngleInArcRange(double angle, const Arc2d& arc);

/**
 * @brief Convert angle parameter to point on arc
 *
 * Same as arc.PointAt().
 *
 * @param arc Input arc
 * @param t Parameter [0, 1]
 * @return Point on arc
 */
inline Point2d ArcPointAtParameter(const Arc2d& arc, double t) {
    return arc.PointAt(t);
}

} // namespace Qi::Vision::Internal
```

---

## 5. 参数设计

### 5.1 常量

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| GEOM_TOLERANCE | double | 1e-9 | [1e-12, 1e-6] | 距离比较容差 |
| ANGLE_TOLERANCE | double | 1e-9 | [1e-12, 1e-6] | 角度比较容差 (弧度) |
| DEFAULT_SAMPLING_STEP | double | 1.0 | [0.1, 10.0] | 默认采样步长 (像素) |
| MAX_SAMPLING_POINTS | size_t | 1000000 | [1000, 10000000] | 最大采样点数 |
| MIN_SEGMENT_LENGTH | double | 1e-12 | - | 最小有效线段长度 |
| MIN_RADIUS | double | 1e-12 | - | 最小有效半径 |

### 5.2 采样参数选择指南

| 应用场景 | 推荐步长 | 备注 |
|----------|----------|------|
| 轮廓显示 | 1.0 - 2.0 | 视觉平滑足够 |
| 模板匹配模型 | 0.5 - 1.0 | 需要足够密度 |
| 精密测量 | 0.1 - 0.5 | 高精度需求 |
| 粗略估计 | 2.0 - 5.0 | 性能优先 |

---

## 6. 精度规格

### 6.1 变换精度

| 变换类型 | 精度要求 |
|----------|----------|
| 点平移 | 精确 (无误差) |
| 点旋转 | < 1e-15 相对误差 (双精度极限) |
| 点缩放 | < 1e-15 相对误差 |
| 仿射变换 | < 1e-14 相对误差 |

### 6.2 几何构造精度

| 操作 | 精度要求 | 条件 |
|------|----------|------|
| 三点确定圆 | 中心 < 1e-10 px | 三点不共线 |
| 三点确定弧 | < 1e-10 px | 三点不共线 |
| 投影点计算 | < 1e-14 px | 标准情况 |

### 6.3 采样精度

| 类型 | 精度要求 | 备注 |
|------|----------|------|
| 线段采样 | 精确端点 | 均匀分布 |
| 圆弧采样 | 弧长误差 < 0.1% | 足够采样点时 |
| 椭圆采样 | 近似等弧长 | 参数空间均匀 |

---

## 7. 算法要点

### 7.1 直线归一化

```cpp
// 确保 a^2 + b^2 = 1
Line2d NormalizeLine(const Line2d& line) {
    double norm = std::sqrt(line.a * line.a + line.b * line.b);
    if (norm < GEOM_TOLERANCE) {
        return line;  // 退化情况
    }
    return Line2d(line.a / norm, line.b / norm, line.c / norm);
}
```

### 7.2 角度归一化

```cpp
// 高效角度归一化
inline double NormalizeAngle(double angle) {
    // 快速路径: 大多数情况角度在合理范围
    if (angle >= -PI && angle < PI) return angle;
    // 使用 fmod 避免多次循环
    angle = std::fmod(angle + PI, TWO_PI);
    if (angle < 0) angle += TWO_PI;
    return angle - PI;
}
```

### 7.3 点在线段上判断

```cpp
bool PointOnSegment(const Point2d& point, const Segment2d& segment, double tolerance) {
    // 1. 检查距离到直线
    Line2d line = segment.ToLine();
    if (std::abs(line.SignedDistance(point)) > tolerance) {
        return false;
    }
    
    // 2. 检查参数 t 在 [0, 1] 范围
    double t = segment.ProjectPoint(point);
    return t >= -tolerance && t <= 1.0 + tolerance;
}
```

### 7.4 弧采样

```cpp
std::vector<Point2d> SampleArc(const Arc2d& arc, double step, bool includeEndpoints) {
    double arcLength = arc.Length();
    size_t numPoints = ComputeSamplingCount(arcLength, step, 2, MAX_SAMPLING_POINTS);
    
    std::vector<Point2d> points;
    points.reserve(numPoints);
    
    for (size_t i = 0; i < numPoints; ++i) {
        double t = static_cast<double>(i) / (numPoints - 1);
        points.push_back(arc.PointAt(t));
    }
    
    // 确保端点精确
    if (includeEndpoints && !points.empty()) {
        points.front() = arc.StartPoint();
        points.back() = arc.EndPoint();
    }
    
    return points;
}
```

### 7.5 椭圆变换

椭圆在仿射变换下仍然是椭圆。变换后的椭圆参数计算:

```cpp
Ellipse2d TransformEllipse(const Ellipse2d& ellipse, const Mat33& matrix) {
    // 1. 变换中心点
    Point2d newCenter = TransformPoint(ellipse.center, matrix);
    
    // 2. 变换椭圆的协方差矩阵
    // 椭圆可表示为 x^T A x = 1, 其中 A 是 2x2 正定矩阵
    // 变换后 A' = M^(-T) A M^(-1)
    
    // 3. 从 A' 提取新的 a, b, angle
    // 使用特征值分解
    
    // ... 详细实现见代码 ...
}
```

### 7.6 弧边界框

需要考虑弧是否经过 0, 90, 180, 270 度:

```cpp
Rect2d ArcBoundingBox(const Arc2d& arc) {
    double minX = std::min(arc.StartPoint().x, arc.EndPoint().x);
    double maxX = std::max(arc.StartPoint().x, arc.EndPoint().x);
    double minY = std::min(arc.StartPoint().y, arc.EndPoint().y);
    double maxY = std::max(arc.StartPoint().y, arc.EndPoint().y);
    
    // 检查弧是否经过坐标轴极值点
    auto checkAngle = [&](double angle, double& minCoord, double& maxCoord, bool isX) {
        if (AngleInArcRange(angle, arc)) {
            Point2d p = arc.ToCircle().center;
            double coord = isX ? (p.x + arc.radius * std::cos(angle))
                              : (p.y + arc.radius * std::sin(angle));
            minCoord = std::min(minCoord, coord);
            maxCoord = std::max(maxCoord, coord);
        }
    };
    
    checkAngle(0, minX, maxX, true);           // 右
    checkAngle(PI, minX, maxX, true);          // 左
    checkAngle(PI / 2, minY, maxY, false);     // 下
    checkAngle(-PI / 2, minY, maxY, false);    // 上
    
    return Rect2d(minX, minY, maxX - minX, maxY - minY);
}
```

---

## 8. 与已有模块的关系

### 8.1 与 Core/Types.h 的关系

Core/Types.h 已定义:
- Point2d: 基础运算 (+, -, *, Norm, Dot, Cross, DistanceTo)
- Line2d: FromPoints, FromPointAngle, Angle, Direction, Normal, SignedDistance, Distance
- Segment2d: Length, Midpoint, Direction, UnitDirection, Angle, ToLine, DistanceToPoint, ProjectPoint, PointAt
- Circle2d: Area, Circumference, Contains
- Ellipse2d: Area, Perimeter, Eccentricity, Contains, PointAt
- Arc2d: EndAngle, Length, StartPoint, EndPoint, Midpoint, PointAt, ToCircle
- RotatedRect2d: Area, GetCorners, BoundingBox, Contains

Geometry2d.h 补充:
- 规范化函数 (NormalizeLine, NormalizeAngle, NormalizeEllipse)
- 变换函数 (Transform*, Rotate*, Scale*, Translate*)
- 构造函数 (LinePerpendicular, LineParallel, ArcFrom3Points)
- 采样函数 (Sample*)
- 更多工具函数 (PointOn*, AreParallel, ArePerpendicular, Project*)

### 8.2 与 Internal/Matrix.h 的关系

使用 Mat33 进行仿射变换计算:
- TransformPoint/Points
- TransformLine/Segment/Circle/Arc/Ellipse/RotatedRect

### 8.3 与 Internal/Fitting.h 的关系

CircleFrom3Points 可复用 Fitting.h 中的 FitCircleExact3Points:
```cpp
std::optional<Circle2d> CircleFrom3Points(const Point2d& p1, const Point2d& p2, const Point2d& p3) {
    return Internal::FitCircleExact3Points(p1, p2, p3);
}
```

---

## 9. 实现任务分解

| 任务 | 文件 | 预估时间 | 依赖 | 优先级 |
|------|------|----------|------|--------|
| 头文件 API 定义 | Geometry2d.h | 2h | Types.h, Matrix.h | P0 |
| 规范化函数 | Geometry2d.cpp | 1h | - | P0 |
| 点操作函数 | Geometry2d.cpp | 1h | Matrix.h | P0 |
| 线/线段操作 | Geometry2d.cpp | 2h | - | P0 |
| 圆/弧操作 | Geometry2d.cpp | 2h | Fitting.h | P0 |
| 椭圆操作 | Geometry2d.cpp | 2h | - | P1 |
| 旋转矩形操作 | Geometry2d.cpp | 1h | - | P1 |
| 属性计算 | Geometry2d.cpp | 1h | - | P0 |
| 采样函数 | Geometry2d.cpp | 3h | - | P0 |
| 工具函数 | Geometry2d.cpp | 2h | - | P0 |
| 单元测试 | test_geometry2d.cpp | 4h | 全部 | P0 |

**总计**: 约 21 小时

**实现顺序建议**:
1. P0 阶段: 头文件 + 规范化 + 点操作 + 线段 + 圆弧 + 属性 + 采样 + 工具 (~16h)
2. P1 阶段: 椭圆 + 旋转矩形 (~3h)
3. 测试 (~4h)

---

## 10. 测试要点

### 10.1 单元测试覆盖

1. **规范化测试**
   - 直线归一化 (已归一化/未归一化/退化)
   - 角度归一化 (各种范围)
   - 椭圆规范化 (a > b / a < b / a == b)

2. **变换测试**
   - 点变换 (平移/旋转/缩放/仿射)
   - 各几何体变换
   - 单位变换恢复原值
   - 逆变换组合

3. **构造测试**
   - 垂线/平行线构造
   - 三点确定圆/弧
   - 退化情况 (共线三点)

4. **属性计算测试**
   - 弧长/面积/边界框
   - 与已知值比较

5. **采样测试**
   - 采样点数正确性
   - 端点包含
   - 均匀分布

6. **工具函数测试**
   - PointOn* 系列
   - 平行/垂直判断
   - 投影计算

### 10.2 边界条件测试

- 零长度线段
- 零半径圆
- 零扫掠角弧
- 退化椭圆 (a == b, 变成圆)
- 极小/极大数值
- 角度边界 (-PI, PI, 0, 2*PI)

### 10.3 精度测试

```cpp
// 示例: 变换精度
TEST(Geometry2dAccuracy, TransformInverse) {
    Point2d original(123.456, 789.012);
    
    Mat33 transform = Mat33::RotationZ(0.5) * Mat33::Translation(100, 200);
    Mat33 inverse = transform.Inverse();
    
    Point2d transformed = TransformPoint(original, transform);
    Point2d recovered = TransformPoint(transformed, inverse);
    
    EXPECT_NEAR(recovered.x, original.x, 1e-12);
    EXPECT_NEAR(recovered.y, original.y, 1e-12);
}

// 示例: 采样精度
TEST(Geometry2dAccuracy, ArcSamplingUniform) {
    Arc2d arc({0, 0}, 100, 0, PI / 2);  // 90度弧
    auto points = SampleArcByCount(arc, 91);  // 91个点 = 90个间隔
    
    // 检查相邻点间距近似相等
    double expectedArcStep = arc.Length() / 90;
    for (size_t i = 1; i < points.size(); ++i) {
        double dist = points[i].DistanceTo(points[i-1]);
        // 弧长近似等于弦长 (小角度)
        EXPECT_NEAR(dist, expectedArcStep, expectedArcStep * 0.01);
    }
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

---

## 12. 未来扩展

1. **SIMD 优化**: 批量点变换的向量化
2. **更多几何体**: 多边形、样条曲线
3. **自适应采样**: 基于曲率的变步长采样
4. **参数空间操作**: 椭圆参数到角度的转换
5. **布尔运算支持**: 几何体的并/交/差运算

---

## 附录 A: 与 Halcon 对应

| QiVision | Halcon |
|----------|--------|
| SampleCircle | gen_circle_contour_xld |
| SampleEllipse | gen_ellipse_contour_xld |
| SampleArc | gen_contour_polygon_xld (圆弧) |
| TransformPoint | affine_trans_point_2d |
| TransformLine | affine_trans_line |
| RotatedRectCorners | smallest_rectangle2 (输出) |
| ArcFrom3Points | gen_circle_3points + 转换 |

---

## 附录 B: API 快速参考

```cpp
// 规范化
Line2d normed = NormalizeLine(line);
double angle = NormalizeAngle(radians);
Ellipse2d normed = NormalizeEllipse(ellipse);

// 点操作
Point2d p = RotatePoint(point, angle);
Point2d p = RotatePointAround(point, center, angle);
Point2d p = TransformPoint(point, matrix);
std::vector<Point2d> pts = TransformPoints(points, matrix);

// 线/线段
Line2d perp = LinePerpendicular(line, point);
Line2d para = LineParallel(line, point);
Segment2d extended = ExtendSegment(segment, 10, 20);
Line2d transformed = TransformLine(line, matrix);

// 圆/弧
std::optional<Arc2d> arc = ArcFrom3Points(p1, p2, p3);
Arc2d arc = ArcFromAngles(center, radius, start, end, dir);
Ellipse2d ellipse = TransformCircle(circle, matrix);
Segment2d chord = ArcToChord(arc);

// 椭圆
Point2d pt = EllipsePointAt(ellipse, theta);
Point2d tangent = EllipseTangentAt(ellipse, theta);
double length = EllipseArcLength(ellipse, theta1, theta2);

// 旋转矩形
std::array<Point2d, 4> corners = RotatedRectCorners(rect);
std::array<Segment2d, 4> edges = RotatedRectEdges(rect);

// 属性
Rect2d bbox = ArcBoundingBox(arc);
Rect2d bbox = EllipseBoundingBox(ellipse);
double area = ArcSectorArea(arc);

// 采样
std::vector<Point2d> pts = SampleSegment(segment, step);
std::vector<Point2d> pts = SampleCircle(circle, step);
std::vector<Point2d> pts = SampleArc(arc, step);
std::vector<Point2d> pts = SampleEllipse(ellipse, step);

// 工具
bool on = PointOnSegment(point, segment, tol);
bool on = PointOnArc(point, arc, tol);
double angle = AngleBetweenLines(line1, line2);
bool parallel = AreParallel(line1, line2, tol);
Point2d proj = ProjectPointOnLine(point, line);
Point2d refl = ReflectPointAcrossLine(point, line);
```
