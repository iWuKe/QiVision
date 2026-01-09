# Measure/Caliper 模块设计文档

## 1. 概述

### 1.1 功能描述

Caliper（卡尺）模块提供基于1D边缘检测的精密测量功能，是工业视觉中最常用的测量工具之一。通过在指定测量区域内提取灰度剖面并检测边缘，实现亚像素级精度的位置和宽度测量。

### 1.2 参考 Halcon 算子

| Halcon 算子 | 功能 | QiVision 对应 |
|-------------|------|---------------|
| `gen_measure_rectangle2` | 创建矩形测量句柄 | `CreateMeasureRect` |
| `gen_measure_arc` | 创建弧形测量句柄 | `CreateMeasureArc` |
| `measure_pos` | 测量边缘位置 | `MeasurePos` |
| `measure_pairs` | 测量边缘对（宽度） | `MeasurePairs` |
| `fuzzy_measure_pos` | 模糊边缘检测 | `FuzzyMeasurePos` |
| `fuzzy_measure_pairs` | 模糊边缘对检测 | `FuzzyMeasurePairs` |
| `close_measure` | 关闭测量句柄 | 析构函数自动处理 |

### 1.3 应用场景

- **尺寸测量**：零件宽度、间隙、直径
- **定位检测**：边缘位置、中心定位
- **缺陷检测**：毛刺、缺口、断裂
- **装配验证**：插入深度、对齐度

---

## 2. 设计规则验证

### 2.1 检查清单

- [x] **坐标类型符合规则**
  - 像素坐标：`int32_t`
  - 亚像素坐标：`double`
  - 测量结果位置：`double`（亚像素精度）

- [x] **层级依赖正确**
  - Feature (Measure) → Internal (Edge1D, Profiler, SubPixel, Fitting, Interpolate)
  - 无反向依赖
  - 无跨层依赖

- [x] **算法完整性满足**
  - 矩形/弧形/同心圆句柄 ✓
  - 边缘配对策略 ✓
  - 亚像素精化 ✓
  - 鲁棒评分 (Fuzzy) ✓

- [x] **Domain 规则**
  - 空 Domain → 返回空结果
  - Full Domain → 无特殊处理需求（Caliper 使用显式 ROI）

- [x] **结果返回规则**
  - 无边缘 → 返回空 vector
  - 多结果按位置/分数排序
  - 无 NMS 需求（边缘检测内部已处理）

---

## 3. 依赖分析

### 3.1 依赖的 Internal 模块

| 模块 | 用途 | 状态 |
|------|------|------|
| Internal/Edge1D.h | 1D边缘检测核心算法 | ✅ 完成 |
| Internal/Profiler.h | 1D剖面提取 | ✅ 完成 |
| Internal/SubPixel.h | 亚像素精化 | ✅ 完成 |
| Internal/Fitting.h | 可选：边缘点拟合 | ✅ 完成 |
| Internal/Interpolate.h | 亚像素插值 | ✅ 完成 |

### 3.2 依赖的 Core 类型

- `QImage` - 输入图像（需要 Domain 支持）
- `Point2d` - 亚像素坐标
- `Rect2d` - 边界框
- `RotatedRect2d` - 旋转矩形测量区域
- `Arc2d` - 弧形测量区域
- `EdgePolarity` - 边缘极性

---

## 4. 类设计

### 4.1 文件结构

```
include/QiVision/Measure/
├── MeasureTypes.h      # 参数和结果结构体
├── MeasureHandle.h     # 测量句柄类
├── Caliper.h           # 卡尺测量函数
└── CaliperArray.h      # 多卡尺阵列（Phase 2）

src/Measure/
├── MeasureHandle.cpp
├── Caliper.cpp
└── CaliperArray.cpp
```

### 4.2 MeasureTypes.h

```cpp
#pragma once

/**
 * @file MeasureTypes.h
 * @brief Measure module type definitions
 *
 * Provides:
 * - Edge transition types
 * - Measurement parameters
 * - Edge and pair result structures
 * - Score and quality metrics
 */

#include <QiVision/Core/Types.h>

#include <cstdint>
#include <vector>

namespace Qi::Vision::Measure {

// =============================================================================
// Constants
// =============================================================================

/// Default minimum edge amplitude
constexpr double DEFAULT_MIN_AMPLITUDE = 20.0;

/// Default Gaussian smoothing sigma
constexpr double DEFAULT_SIGMA = 1.0;

/// Default number of perpendicular lines for averaging
constexpr int32_t DEFAULT_NUM_LINES = 10;

/// Default interpolation samples per pixel
constexpr double DEFAULT_SAMPLES_PER_PIXEL = 1.0;

/// Maximum number of edges to return
constexpr int32_t MAX_EDGES = 1000;

// =============================================================================
// Enumerations
// =============================================================================

/**
 * @brief Edge transition type (polarity)
 */
enum class EdgeTransition {
    Positive,       ///< Dark to light (rising edge)
    Negative,       ///< Light to dark (falling edge)
    All             ///< Both transitions
};

/**
 * @brief Edge selection mode
 */
enum class EdgeSelectMode {
    All,            ///< Return all detected edges
    First,          ///< First edge only (along profile direction)
    Last,           ///< Last edge only
    Strongest,      ///< Edge with highest amplitude
    Weakest         ///< Edge with lowest amplitude (above threshold)
};

/**
 * @brief Edge pair selection mode
 */
enum class PairSelectMode {
    All,            ///< All valid pairs
    First,          ///< First pair only
    Last,           ///< Last pair only
    Strongest,      ///< Pair with highest combined amplitude
    Widest,         ///< Pair with largest distance
    Narrowest       ///< Pair with smallest distance
};

/**
 * @brief Interpolation method for profile extraction
 */
enum class ProfileInterpolation {
    Nearest,        ///< Nearest neighbor (fast)
    Bilinear,       ///< Bilinear (default, good balance)
    Bicubic         ///< Bicubic (highest accuracy)
};

/**
 * @brief Score computation method for fuzzy measurement
 */
enum class ScoreMethod {
    Amplitude,      ///< Based on edge amplitude only
    AmplitudeScore, ///< Amplitude normalized by max possible
    Contrast,       ///< Local contrast ratio
    FuzzyScore      ///< Combined fuzzy logic score
};

// =============================================================================
// Parameter Structures
// =============================================================================

/**
 * @brief Common measurement parameters
 */
struct MeasureParams {
    // Edge detection parameters
    double sigma = DEFAULT_SIGMA;                   ///< Gaussian smoothing sigma
    double minAmplitude = DEFAULT_MIN_AMPLITUDE;    ///< Minimum edge amplitude
    EdgeTransition transition = EdgeTransition::All; ///< Edge polarity filter
    
    // Profile parameters
    int32_t numLines = DEFAULT_NUM_LINES;           ///< Number of perpendicular lines
    double samplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL; ///< Sampling density
    ProfileInterpolation interp = ProfileInterpolation::Bilinear;
    
    // Selection parameters
    EdgeSelectMode selectMode = EdgeSelectMode::All;
    int32_t maxEdges = MAX_EDGES;                   ///< Maximum edges to return
    
    // Score parameters
    ScoreMethod scoreMethod = ScoreMethod::Amplitude;
    
    // Builder pattern for fluent configuration
    MeasureParams& SetSigma(double s) { sigma = s; return *this; }
    MeasureParams& SetMinAmplitude(double a) { minAmplitude = a; return *this; }
    MeasureParams& SetTransition(EdgeTransition t) { transition = t; return *this; }
    MeasureParams& SetNumLines(int32_t n) { numLines = n; return *this; }
    MeasureParams& SetSelectMode(EdgeSelectMode m) { selectMode = m; return *this; }
    MeasureParams& SetMaxEdges(int32_t n) { maxEdges = n; return *this; }
};

/**
 * @brief Parameters for edge pair (width) measurement
 */
struct PairParams : public MeasureParams {
    // Pair-specific parameters
    EdgeTransition firstTransition = EdgeTransition::Positive;   ///< First edge polarity
    EdgeTransition secondTransition = EdgeTransition::Negative;  ///< Second edge polarity
    
    double minWidth = 0.0;          ///< Minimum pair width (pixels)
    double maxWidth = 1e9;          ///< Maximum pair width (pixels)
    
    PairSelectMode pairSelectMode = PairSelectMode::All;
    int32_t maxPairs = MAX_EDGES;   ///< Maximum pairs to return
    
    PairParams& SetFirstTransition(EdgeTransition t) { firstTransition = t; return *this; }
    PairParams& SetSecondTransition(EdgeTransition t) { secondTransition = t; return *this; }
    PairParams& SetWidthRange(double minW, double maxW) { 
        minWidth = minW; maxWidth = maxW; return *this; 
    }
    PairParams& SetPairSelectMode(PairSelectMode m) { pairSelectMode = m; return *this; }
};

/**
 * @brief Fuzzy measurement parameters (extended)
 */
struct FuzzyParams : public MeasureParams {
    // Fuzzy-specific parameters
    double fuzzyThresholdLow = 0.3;     ///< Lower amplitude threshold ratio
    double fuzzyThresholdHigh = 0.8;    ///< Upper amplitude threshold ratio
    double minScore = 0.5;              ///< Minimum score threshold
    
    bool computeScore = true;           ///< Whether to compute detailed score
    bool useAdaptiveThreshold = false;  ///< Adapt threshold based on local contrast
    
    FuzzyParams& SetFuzzyThresholds(double low, double high) {
        fuzzyThresholdLow = low; fuzzyThresholdHigh = high; return *this;
    }
    FuzzyParams& SetMinScore(double s) { minScore = s; return *this; }
};

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief Single edge measurement result
 */
struct EdgeResult {
    // Position in image coordinates (subpixel)
    double row = 0.0;           ///< Y coordinate (row)
    double column = 0.0;        ///< X coordinate (column)
    
    // Position along profile
    double profilePosition = 0.0;   ///< Position along measurement profile [0, length]
    
    // Edge properties
    double amplitude = 0.0;     ///< Edge amplitude (gradient magnitude)
    EdgeTransition transition = EdgeTransition::Positive; ///< Edge polarity
    
    // Quality metrics
    double score = 0.0;         ///< Quality score [0, 1] (for fuzzy)
    double confidence = 0.0;    ///< Detection confidence [0, 1]
    
    // Optional: edge direction
    double angle = 0.0;         ///< Edge normal angle (radians)
    
    /// Check if result is valid
    bool IsValid() const { return amplitude > 0 && confidence > 0; }
    
    /// Get position as Point2d
    Point2d Position() const { return {column, row}; }
};

/**
 * @brief Edge pair (width) measurement result
 */
struct PairResult {
    EdgeResult first;           ///< First edge of pair
    EdgeResult second;          ///< Second edge of pair
    
    // Pair metrics
    double width = 0.0;         ///< Distance between edges (pixels)
    double centerRow = 0.0;     ///< Center Y coordinate
    double centerColumn = 0.0;  ///< Center X coordinate
    
    // Quality
    double score = 0.0;         ///< Combined pair score [0, 1]
    double symmetry = 0.0;      ///< Amplitude symmetry [0, 1]
    
    /// Check if result is valid
    bool IsValid() const { 
        return first.IsValid() && second.IsValid() && width > 0; 
    }
    
    /// Get center position
    Point2d Center() const { return {centerColumn, centerRow}; }
};

/**
 * @brief Measurement statistics
 */
struct MeasureStats {
    int32_t numEdgesFound = 0;      ///< Total edges detected
    int32_t numEdgesReturned = 0;   ///< Edges after filtering
    
    double meanAmplitude = 0.0;     ///< Mean edge amplitude
    double maxAmplitude = 0.0;      ///< Maximum amplitude
    double minAmplitude = 0.0;      ///< Minimum amplitude (of detected)
    
    double profileContrast = 0.0;   ///< Overall profile contrast
    double signalNoiseRatio = 0.0;  ///< Estimated SNR
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Convert EdgeTransition to Internal EdgePolarity
 */
inline Internal::EdgePolarity ToEdgePolarity(EdgeTransition t) {
    switch (t) {
        case EdgeTransition::Positive: return Internal::EdgePolarity::Positive;
        case EdgeTransition::Negative: return Internal::EdgePolarity::Negative;
        case EdgeTransition::All:      return Internal::EdgePolarity::Both;
    }
    return Internal::EdgePolarity::Both;
}

/**
 * @brief Convert Internal EdgePolarity to EdgeTransition
 */
inline EdgeTransition FromEdgePolarity(Internal::EdgePolarity p) {
    switch (p) {
        case Internal::EdgePolarity::Positive: return EdgeTransition::Positive;
        case Internal::EdgePolarity::Negative: return EdgeTransition::Negative;
        case Internal::EdgePolarity::Both:     return EdgeTransition::All;
    }
    return EdgeTransition::All;
}

} // namespace Qi::Vision::Measure
```

### 4.3 MeasureHandle.h

```cpp
#pragma once

/**
 * @file MeasureHandle.h
 * @brief Measurement region handles (Rectangle, Arc, Concentric)
 *
 * Provides:
 * - Pre-computed sampling geometry for efficient repeated measurements
 * - Rectangle, Arc, and Concentric Circle handles
 * - Handle serialization for persistence
 *
 * Design:
 * - Handles are lightweight and can be copied
 * - Sampling points are computed once at creation
 * - Thread-safe for concurrent reads (const methods)
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Measure/MeasureTypes.h>

#include <memory>
#include <vector>

namespace Qi::Vision::Measure {

// Forward declarations
class MeasureHandleImpl;

// =============================================================================
// MeasureHandle Base (Abstract)
// =============================================================================

/**
 * @brief Measure handle type enumeration
 */
enum class HandleType {
    Rectangle,      ///< Rectangular (linear) measurement
    Arc,            ///< Arc (curved) measurement
    Concentric      ///< Concentric circles measurement
};

/**
 * @brief Base class for measurement handles
 *
 * A handle encapsulates the measurement region geometry and pre-computed
 * sampling information. This allows efficient repeated measurements in
 * the same region.
 */
class MeasureHandle {
public:
    virtual ~MeasureHandle() = default;
    
    /// Get handle type
    virtual HandleType Type() const = 0;
    
    /// Check if handle is valid
    virtual bool IsValid() const = 0;
    
    /// Get profile length in pixels
    virtual double ProfileLength() const = 0;
    
    /// Get profile direction angle (radians)
    virtual double ProfileAngle() const = 0;
    
    /// Get bounding box of measurement region
    virtual Rect2d BoundingBox() const = 0;
    
    /// Check if point is inside measurement region
    virtual bool Contains(const Point2d& point) const = 0;
    
    /// Get sampling parameters
    int32_t NumLines() const { return numLines_; }
    double SamplesPerPixel() const { return samplesPerPixel_; }
    
protected:
    int32_t numLines_ = DEFAULT_NUM_LINES;
    double samplesPerPixel_ = DEFAULT_SAMPLES_PER_PIXEL;
};

// =============================================================================
// MeasureRectangle2
// =============================================================================

/**
 * @brief Rectangular (linear) measurement handle
 *
 * Defines a rotated rectangle for linear edge measurement.
 * Profile is extracted perpendicular to the rectangle's long axis.
 *
 * Geometry:
 * - center: Rectangle center (row, column)
 * - phi: Rotation angle of rectangle (radians, perpendicular to profile)
 * - length: Profile length (along measurement direction)
 * - width: Rectangle width (perpendicular extent for averaging)
 *
 * Coordinate convention:
 * - Profile direction: angle = phi + PI/2
 * - Perpendicular lines are spread across 'width'
 */
class MeasureRectangle2 : public MeasureHandle {
public:
    /**
     * @brief Construct rectangle handle
     * @param centerRow Center Y coordinate
     * @param centerCol Center X coordinate
     * @param phi Rectangle rotation angle (radians)
     * @param length Profile length (measurement direction)
     * @param width Rectangle width (averaging direction)
     * @param numLines Number of perpendicular lines for averaging
     * @param samplesPerPixel Sampling density along profile
     */
    MeasureRectangle2(double centerRow, double centerCol,
                      double phi, double length, double width,
                      int32_t numLines = DEFAULT_NUM_LINES,
                      double samplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL);
    
    /// Create from RotatedRect2d
    static MeasureRectangle2 FromRotatedRect(const RotatedRect2d& rect,
                                              int32_t numLines = DEFAULT_NUM_LINES);
    
    /// Create from two points (defining profile axis)
    static MeasureRectangle2 FromPoints(const Point2d& p1, const Point2d& p2,
                                         double width,
                                         int32_t numLines = DEFAULT_NUM_LINES);
    
    // MeasureHandle interface
    HandleType Type() const override { return HandleType::Rectangle; }
    bool IsValid() const override;
    double ProfileLength() const override { return length_; }
    double ProfileAngle() const override;
    Rect2d BoundingBox() const override;
    bool Contains(const Point2d& point) const override;
    
    // Rectangle-specific accessors
    double CenterRow() const { return centerRow_; }
    double CenterCol() const { return centerCol_; }
    double Phi() const { return phi_; }
    double Length() const { return length_; }
    double Width() const { return width_; }
    
    /// Get the rotated rectangle
    RotatedRect2d ToRotatedRect() const;
    
    /// Get profile start and end points
    void GetProfileEndpoints(Point2d& start, Point2d& end) const;
    
    /// Get all sampling lines (for visualization)
    std::vector<Segment2d> GetSamplingLines() const;
    
private:
    double centerRow_;
    double centerCol_;
    double phi_;        // Rectangle rotation (perpendicular to profile)
    double length_;     // Profile length
    double width_;      // Rectangle width
    
    // Pre-computed geometry
    void ComputeSamplingGeometry();
    std::vector<double> lineOffsets_;  // Perpendicular offsets for averaging
};

// =============================================================================
// MeasureArc
// =============================================================================

/**
 * @brief Arc measurement handle
 *
 * Defines a circular arc for curved edge measurement.
 * Profile is extracted along the arc, with optional radial averaging.
 *
 * Geometry:
 * - center: Arc center
 * - radius: Arc radius (for single arc) or center radius (for annular)
 * - angleStart: Start angle (radians)
 * - angleExtent: Arc extent (radians), positive = CCW
 * - annulusRadius: Half-width of annular region for averaging
 */
class MeasureArc : public MeasureHandle {
public:
    /**
     * @brief Construct arc handle
     * @param centerRow Center Y coordinate
     * @param centerCol Center X coordinate
     * @param radius Arc radius
     * @param angleStart Start angle (radians)
     * @param angleExtent Arc extent (radians)
     * @param annulusRadius Radial extent for averaging (0 = single line)
     * @param numLines Number of radial lines for averaging
     * @param samplesPerPixel Sampling density along arc
     */
    MeasureArc(double centerRow, double centerCol,
               double radius, double angleStart, double angleExtent,
               double annulusRadius = 0.0,
               int32_t numLines = DEFAULT_NUM_LINES,
               double samplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL);
    
    /// Create from Arc2d
    static MeasureArc FromArc(const Arc2d& arc,
                               double annulusRadius = 0.0,
                               int32_t numLines = DEFAULT_NUM_LINES);
    
    /// Create from Circle2d (full circle)
    static MeasureArc FromCircle(const Circle2d& circle,
                                  double annulusRadius = 0.0,
                                  int32_t numLines = DEFAULT_NUM_LINES);
    
    // MeasureHandle interface
    HandleType Type() const override { return HandleType::Arc; }
    bool IsValid() const override;
    double ProfileLength() const override;  // Arc length
    double ProfileAngle() const override { return angleStart_; }  // Tangent at start
    Rect2d BoundingBox() const override;
    bool Contains(const Point2d& point) const override;
    
    // Arc-specific accessors
    double CenterRow() const { return centerRow_; }
    double CenterCol() const { return centerCol_; }
    double Radius() const { return radius_; }
    double AngleStart() const { return angleStart_; }
    double AngleExtent() const { return angleExtent_; }
    double AngleEnd() const { return angleStart_ + angleExtent_; }
    double AnnulusRadius() const { return annulusRadius_; }
    
    /// Get the arc
    Arc2d ToArc() const;
    
    /// Get point on arc at parameter t [0, 1]
    Point2d PointAt(double t) const;
    
    /// Get tangent angle at parameter t [0, 1]
    double TangentAt(double t) const;
    
    /// Convert profile position to angle
    double ProfilePosToAngle(double pos) const;
    
    /// Convert angle to profile position
    double AngleToProfilePos(double angle) const;
    
private:
    double centerRow_;
    double centerCol_;
    double radius_;
    double angleStart_;
    double angleExtent_;
    double annulusRadius_;
    
    void ComputeSamplingGeometry();
    std::vector<double> radiusOffsets_;  // Radial offsets for averaging
};

// =============================================================================
// MeasureConcentricCircles
// =============================================================================

/**
 * @brief Concentric circles measurement handle
 *
 * Defines a radial measurement region for multi-ring analysis.
 * Profile is extracted from inner to outer radius.
 *
 * Geometry:
 * - center: Circle center
 * - innerRadius, outerRadius: Radial extent
 * - angle: Direction angle for radial profile
 * - angularWidth: Angular width for averaging
 */
class MeasureConcentricCircles : public MeasureHandle {
public:
    /**
     * @brief Construct concentric circles handle
     * @param centerRow Center Y coordinate
     * @param centerCol Center X coordinate
     * @param innerRadius Inner radius
     * @param outerRadius Outer radius
     * @param angle Direction angle for radial profile (radians)
     * @param angularWidth Angular width for averaging (radians)
     * @param numLines Number of angular lines for averaging
     * @param samplesPerPixel Sampling density along radius
     */
    MeasureConcentricCircles(double centerRow, double centerCol,
                              double innerRadius, double outerRadius,
                              double angle,
                              double angularWidth = 0.1,
                              int32_t numLines = DEFAULT_NUM_LINES,
                              double samplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL);
    
    // MeasureHandle interface
    HandleType Type() const override { return HandleType::Concentric; }
    bool IsValid() const override;
    double ProfileLength() const override { return outerRadius_ - innerRadius_; }
    double ProfileAngle() const override { return angle_; }
    Rect2d BoundingBox() const override;
    bool Contains(const Point2d& point) const override;
    
    // Concentric-specific accessors
    double CenterRow() const { return centerRow_; }
    double CenterCol() const { return centerCol_; }
    double InnerRadius() const { return innerRadius_; }
    double OuterRadius() const { return outerRadius_; }
    double Angle() const { return angle_; }
    double AngularWidth() const { return angularWidth_; }
    
    /// Get point at radius
    Point2d PointAtRadius(double radius) const;
    
    /// Convert profile position to radius
    double ProfilePosToRadius(double pos) const;
    
private:
    double centerRow_;
    double centerCol_;
    double innerRadius_;
    double outerRadius_;
    double angle_;
    double angularWidth_;
    
    void ComputeSamplingGeometry();
    std::vector<double> angleOffsets_;  // Angular offsets for averaging
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Create a measure handle covering a line segment
 *
 * @param p1 Start point
 * @param p2 End point
 * @param width Perpendicular width for averaging
 * @param numLines Number of averaging lines
 * @return MeasureRectangle2 handle
 */
MeasureRectangle2 CreateMeasureFromSegment(const Point2d& p1, const Point2d& p2,
                                            double width = 10.0,
                                            int32_t numLines = DEFAULT_NUM_LINES);

/**
 * @brief Create a measure handle covering a rotated rectangle
 *
 * @param rect Rotated rectangle
 * @param numLines Number of averaging lines
 * @return MeasureRectangle2 handle
 */
MeasureRectangle2 CreateMeasureFromRect(const RotatedRect2d& rect,
                                         int32_t numLines = DEFAULT_NUM_LINES);

} // namespace Qi::Vision::Measure
```

### 4.4 Caliper.h

```cpp
#pragma once

/**
 * @file Caliper.h
 * @brief Caliper measurement functions
 *
 * Provides:
 * - Edge position measurement (MeasurePos)
 * - Edge pair/width measurement (MeasurePairs)
 * - Fuzzy (robust) variants with scoring
 * - Profile extraction and analysis
 *
 * Precision targets (standard conditions: contrast>=50, noise sigma<=5):
 * - Position: < 0.03 px (1 sigma)
 * - Width: < 0.05 px (1 sigma)
 *
 * Thread safety:
 * - All functions are thread-safe for const handles
 * - Multiple threads can measure with same handle simultaneously
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Measure/MeasureTypes.h>
#include <QiVision/Measure/MeasureHandle.h>

#include <vector>

namespace Qi::Vision::Measure {

// =============================================================================
// Handle Creation Functions
// =============================================================================

/**
 * @brief Create rectangular measurement handle
 *
 * @param centerRow Center Y coordinate
 * @param centerCol Center X coordinate  
 * @param phi Rectangle rotation angle (radians)
 * @param length Profile length (along measurement direction)
 * @param width Rectangle width (perpendicular, for averaging)
 * @param numLines Number of lines for averaging (default 10)
 * @param samplesPerPixel Sampling density (default 1.0)
 * @return MeasureRectangle2 handle
 *
 * @note The profile direction is perpendicular to phi
 * @note Use more numLines for noisy images
 */
MeasureRectangle2 CreateMeasureRect(double centerRow, double centerCol,
                                     double phi, double length, double width,
                                     int32_t numLines = DEFAULT_NUM_LINES,
                                     double samplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL);

/**
 * @brief Create arc measurement handle
 *
 * @param centerRow Center Y coordinate
 * @param centerCol Center X coordinate
 * @param radius Arc radius
 * @param angleStart Start angle (radians)
 * @param angleExtent Arc extent (radians, positive = CCW)
 * @param annulusRadius Radial width for averaging (0 = single line)
 * @param numLines Number of radial lines for averaging
 * @param samplesPerPixel Sampling density
 * @return MeasureArc handle
 */
MeasureArc CreateMeasureArc(double centerRow, double centerCol,
                             double radius, double angleStart, double angleExtent,
                             double annulusRadius = 0.0,
                             int32_t numLines = DEFAULT_NUM_LINES,
                             double samplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL);

/**
 * @brief Create concentric circles measurement handle
 *
 * @param centerRow Center Y coordinate
 * @param centerCol Center X coordinate
 * @param innerRadius Inner radius
 * @param outerRadius Outer radius
 * @param angle Radial direction angle (radians)
 * @param angularWidth Angular width for averaging (radians)
 * @param numLines Number of angular lines for averaging
 * @param samplesPerPixel Sampling density
 * @return MeasureConcentricCircles handle
 */
MeasureConcentricCircles CreateMeasureConcentric(double centerRow, double centerCol,
                                                   double innerRadius, double outerRadius,
                                                   double angle,
                                                   double angularWidth = 0.1,
                                                   int32_t numLines = DEFAULT_NUM_LINES,
                                                   double samplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL);

// =============================================================================
// Edge Position Measurement
// =============================================================================

/**
 * @brief Measure edge positions in rectangular region
 *
 * Extracts profile along the measurement direction, detects edges using
 * 1D edge detection with optional Gaussian smoothing.
 *
 * @param image Input image (grayscale)
 * @param handle Measurement handle
 * @param params Measurement parameters
 * @return Vector of edge results (may be empty if no edges found)
 *
 * @note Results are sorted by profile position (ascending)
 * @note Use higher sigma for noisy images
 * @note Use higher numLines for averaging across texture
 *
 * Example:
 * @code
 * auto handle = CreateMeasureRect(100, 200, 0, 50, 10);
 * auto params = MeasureParams().SetSigma(1.5).SetMinAmplitude(30);
 * auto edges = MeasurePos(image, handle, params);
 * for (const auto& e : edges) {
 *     std::cout << "Edge at (" << e.column << ", " << e.row << ")\n";
 * }
 * @endcode
 */
std::vector<EdgeResult> MeasurePos(const QImage& image,
                                    const MeasureRectangle2& handle,
                                    const MeasureParams& params = MeasureParams());

/**
 * @brief Measure edge positions along arc
 */
std::vector<EdgeResult> MeasurePos(const QImage& image,
                                    const MeasureArc& handle,
                                    const MeasureParams& params = MeasureParams());

/**
 * @brief Measure edge positions along concentric circles (radial)
 */
std::vector<EdgeResult> MeasurePos(const QImage& image,
                                    const MeasureConcentricCircles& handle,
                                    const MeasureParams& params = MeasureParams());

/**
 * @brief Generic MeasurePos for any handle type
 */
std::vector<EdgeResult> MeasurePos(const QImage& image,
                                    const MeasureHandle& handle,
                                    const MeasureParams& params = MeasureParams());

// =============================================================================
// Edge Pair (Width) Measurement
// =============================================================================

/**
 * @brief Measure edge pairs (width) in rectangular region
 *
 * Detects pairs of edges for width measurement. By default, finds
 * positive (rising) followed by negative (falling) edges.
 *
 * @param image Input image (grayscale)
 * @param handle Measurement handle
 * @param params Pair measurement parameters
 * @return Vector of pair results (may be empty if no pairs found)
 *
 * @note firstTransition and secondTransition control pairing
 * @note Use minWidth/maxWidth to filter by expected width
 *
 * Example:
 * @code
 * auto handle = CreateMeasureRect(100, 200, 0, 100, 10);
 * auto params = PairParams()
 *     .SetMinAmplitude(25)
 *     .SetWidthRange(5, 50)
 *     .SetPairSelectMode(PairSelectMode::Strongest);
 * auto pairs = MeasurePairs(image, handle, params);
 * if (!pairs.empty()) {
 *     std::cout << "Width: " << pairs[0].width << " pixels\n";
 * }
 * @endcode
 */
std::vector<PairResult> MeasurePairs(const QImage& image,
                                      const MeasureRectangle2& handle,
                                      const PairParams& params = PairParams());

/**
 * @brief Measure edge pairs along arc
 */
std::vector<PairResult> MeasurePairs(const QImage& image,
                                      const MeasureArc& handle,
                                      const PairParams& params = PairParams());

/**
 * @brief Measure edge pairs along concentric circles
 */
std::vector<PairResult> MeasurePairs(const QImage& image,
                                      const MeasureConcentricCircles& handle,
                                      const PairParams& params = PairParams());

/**
 * @brief Generic MeasurePairs for any handle type
 */
std::vector<PairResult> MeasurePairs(const QImage& image,
                                      const MeasureHandle& handle,
                                      const PairParams& params = PairParams());

// =============================================================================
// Fuzzy (Robust) Measurement
// =============================================================================

/**
 * @brief Fuzzy edge position measurement with scoring
 *
 * Similar to MeasurePos but computes quality scores for each edge.
 * Scores are based on amplitude, local contrast, and consistency.
 * Better for uncertain edge conditions.
 *
 * @param image Input image
 * @param handle Measurement handle
 * @param params Fuzzy measurement parameters
 * @param stats Optional output statistics
 * @return Vector of edge results with scores
 *
 * @note Scores in [0, 1], higher is better
 * @note Low-score edges can be filtered by minScore
 */
std::vector<EdgeResult> FuzzyMeasurePos(const QImage& image,
                                         const MeasureRectangle2& handle,
                                         const FuzzyParams& params = FuzzyParams(),
                                         MeasureStats* stats = nullptr);

std::vector<EdgeResult> FuzzyMeasurePos(const QImage& image,
                                         const MeasureArc& handle,
                                         const FuzzyParams& params = FuzzyParams(),
                                         MeasureStats* stats = nullptr);

std::vector<EdgeResult> FuzzyMeasurePos(const QImage& image,
                                         const MeasureConcentricCircles& handle,
                                         const FuzzyParams& params = FuzzyParams(),
                                         MeasureStats* stats = nullptr);

/**
 * @brief Fuzzy edge pair measurement with scoring
 *
 * @param image Input image
 * @param handle Measurement handle
 * @param params Fuzzy measurement parameters (uses as PairParams base)
 * @param stats Optional output statistics
 * @return Vector of pair results with scores
 */
std::vector<PairResult> FuzzyMeasurePairs(const QImage& image,
                                           const MeasureRectangle2& handle,
                                           const FuzzyParams& params = FuzzyParams(),
                                           MeasureStats* stats = nullptr);

std::vector<PairResult> FuzzyMeasurePairs(const QImage& image,
                                           const MeasureArc& handle,
                                           const FuzzyParams& params = FuzzyParams(),
                                           MeasureStats* stats = nullptr);

std::vector<PairResult> FuzzyMeasurePairs(const QImage& image,
                                           const MeasureConcentricCircles& handle,
                                           const FuzzyParams& params = FuzzyParams(),
                                           MeasureStats* stats = nullptr);

// =============================================================================
// Profile Extraction (for debugging/visualization)
// =============================================================================

/**
 * @brief Extract measurement profile from image
 *
 * Useful for debugging and visualization.
 *
 * @param image Input image
 * @param handle Measurement handle
 * @param interp Interpolation method
 * @return Profile data (gray values along profile)
 */
std::vector<double> ExtractMeasureProfile(const QImage& image,
                                           const MeasureHandle& handle,
                                           ProfileInterpolation interp = ProfileInterpolation::Bilinear);

/**
 * @brief Get profile sample coordinates
 *
 * Returns the (x, y) coordinates of each profile sample point.
 *
 * @param handle Measurement handle
 * @return Vector of sample coordinates
 */
std::vector<Point2d> GetProfileCoordinates(const MeasureHandle& handle);

// =============================================================================
// Coordinate Transformation
// =============================================================================

/**
 * @brief Convert profile position to image coordinates
 *
 * @param handle Measurement handle
 * @param profilePos Position along profile [0, ProfileLength]
 * @return Image coordinates (row, column)
 */
Point2d ProfileToImage(const MeasureHandle& handle, double profilePos);

/**
 * @brief Convert image coordinates to profile position
 *
 * @param handle Measurement handle
 * @param imagePoint Image coordinates
 * @return Position along profile (may be outside [0, length] if point not on profile)
 */
double ImageToProfile(const MeasureHandle& handle, const Point2d& imagePoint);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Compute expected number of samples for handle
 */
int32_t GetNumSamples(const MeasureHandle& handle);

/**
 * @brief Estimate optimal sigma based on image noise
 *
 * @param image Input image
 * @param handle Measurement handle
 * @return Suggested sigma value
 */
double EstimateOptimalSigma(const QImage& image, const MeasureHandle& handle);

/**
 * @brief Filter edges by selection mode
 */
std::vector<EdgeResult> SelectEdges(const std::vector<EdgeResult>& edges,
                                     EdgeSelectMode mode,
                                     int32_t maxCount = MAX_EDGES);

/**
 * @brief Filter pairs by selection mode
 */
std::vector<PairResult> SelectPairs(const std::vector<PairResult>& pairs,
                                     PairSelectMode mode,
                                     int32_t maxCount = MAX_EDGES);

/**
 * @brief Sort edges by various criteria
 */
enum class EdgeSortBy { Position, Amplitude, Score };
void SortEdges(std::vector<EdgeResult>& edges, EdgeSortBy criterion, bool ascending = true);

/**
 * @brief Sort pairs by various criteria
 */
enum class PairSortBy { Position, Width, Score, Symmetry };
void SortPairs(std::vector<PairResult>& pairs, PairSortBy criterion, bool ascending = true);

} // namespace Qi::Vision::Measure
```

---

## 5. 参数设计

### 5.1 MeasureParams 参数表

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| sigma | double | 1.0 | [0.5, 10.0] | 高斯平滑 sigma |
| minAmplitude | double | 20.0 | [1, 255] | 最小边缘幅度 |
| transition | EdgeTransition | All | - | 边缘极性过滤 |
| numLines | int32_t | 10 | [1, 256] | 平均用的垂线数 |
| samplesPerPixel | double | 1.0 | [0.5, 4.0] | 采样密度 |
| interp | ProfileInterpolation | Bilinear | - | 插值方法 |
| selectMode | EdgeSelectMode | All | - | 边缘选择模式 |
| maxEdges | int32_t | 1000 | [1, 10000] | 最大返回边缘数 |

### 5.2 PairParams 额外参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| firstTransition | EdgeTransition | Positive | - | 第一边缘极性 |
| secondTransition | EdgeTransition | Negative | - | 第二边缘极性 |
| minWidth | double | 0.0 | [0, inf) | 最小宽度过滤 |
| maxWidth | double | 1e9 | [0, inf) | 最大宽度过滤 |
| pairSelectMode | PairSelectMode | All | - | 对选择模式 |

### 5.3 FuzzyParams 额外参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| fuzzyThresholdLow | double | 0.3 | [0, 1] | 模糊下阈值比例 |
| fuzzyThresholdHigh | double | 0.8 | [0, 1] | 模糊上阈值比例 |
| minScore | double | 0.5 | [0, 1] | 最小分数过滤 |
| useAdaptiveThreshold | bool | false | - | 自适应阈值 |

---

## 6. 精度规格

### 6.1 标准条件

| 条件 | 值 |
|------|-----|
| 对比度 | >= 50 灰度级 |
| 噪声 | sigma <= 5 |
| 边缘类型 | 理想阶跃边缘 |
| 采样 | 1.0 samples/pixel |

### 6.2 精度要求

| 测量类型 | 指标 | 标准条件 | 困难条件* |
|----------|------|----------|-----------|
| 边缘位置 | 位置误差 (1 sigma) | < 0.03 px | < 0.1 px |
| 边缘对宽度 | 宽度误差 (1 sigma) | < 0.05 px | < 0.15 px |
| 弧形边缘 | 角度误差 (1 sigma) | < 0.1 度 | < 0.3 度 |

*困难条件：对比度 20-50，噪声 sigma 5-15

### 6.3 精度测试方法

1. **合成图像测试**
   - 生成已知位置的边缘图像
   - 添加不同级别高斯噪声
   - 统计测量误差分布

2. **蒙特卡洛测试**
   - 随机生成边缘位置
   - 随机添加噪声
   - 计算 RMS 误差

3. **重复性测试**
   - 同一图像多次测量
   - 计算标准差

---

## 7. 算法要点

### 7.1 核心测量流程

```
MeasurePos(image, handle, params):
    1. 提取测量区域剖面
       - 使用 Profiler::ExtractRectProfile / ExtractArcProfile
       - 多线平均降噪
       
    2. 1D 边缘检测
       - 高斯平滑 (sigma)
       - 计算梯度
       - 查找梯度峰值
       - 极性过滤
       
    3. 亚像素精化
       - 使用 SubPixel::RefineParabolic1D
       - 计算精确位置
       
    4. 坐标转换
       - 剖面位置 → 图像坐标
       - 计算边缘法线方向
       
    5. 结果过滤和排序
       - 应用 selectMode
       - 按位置排序
       
    return edges
```

### 7.2 边缘配对算法

```
MeasurePairs(image, handle, params):
    1. 检测所有边缘
       edges = MeasurePos(image, handle, params)
       
    2. 分离正负边缘
       positiveEdges = filter(edges, Positive)
       negativeEdges = filter(edges, Negative)
       
    3. 配对匹配
       for each positive edge p:
           for each negative edge n (after p):
               width = n.position - p.position
               if minWidth <= width <= maxWidth:
                   pair = {p, n, width}
                   pairs.add(pair)
                   
    4. 选择最佳配对
       - 按 pairSelectMode 过滤
       - 计算对称性分数
       
    return pairs
```

### 7.3 Fuzzy 评分算法

```
ComputeFuzzyScore(edge, profile, params):
    // 幅度分数 (梯形隶属函数)
    if amplitude < minAmplitude * fuzzyLow:
        ampScore = 0
    elif amplitude < minAmplitude * fuzzyHigh:
        ampScore = linear_interp(fuzzyLow, fuzzyHigh)
    else:
        ampScore = 1
        
    // 对比度分数
    localContrast = compute_local_contrast(profile, edge.pos)
    contrastScore = localContrast / maxContrast
    
    // 一致性分数 (多线平均时)
    consistencyScore = 1 - stddev(amplitudes) / mean(amplitudes)
    
    // 综合分数
    score = 0.5 * ampScore + 0.3 * contrastScore + 0.2 * consistencyScore
    
    return score
```

### 7.4 性能考虑

| 优化点 | 方法 |
|--------|------|
| 剖面提取 | 使用 Bicubic 仅在需要高精度时 |
| 边缘检测 | 使用 Edge1D 已优化的实现 |
| 多线平均 | 并行提取，向量化平均 |
| 弧形采样 | 预计算 sin/cos 表 |

---

## 8. 实现任务分解

| 任务 | 文件 | 预估时间 | 依赖 | 优先级 |
|------|------|----------|------|--------|
| 1. MeasureTypes 定义 | MeasureTypes.h | 2h | 无 | P0 |
| 2. MeasureRectangle2 实现 | MeasureHandle.h/cpp | 3h | Task 1 | P0 |
| 3. MeasureArc 实现 | MeasureHandle.h/cpp | 3h | Task 1 | P0 |
| 4. MeasureConcentricCircles | MeasureHandle.h/cpp | 2h | Task 1 | P1 |
| 5. CreateMeasure* 工厂函数 | Caliper.cpp | 1h | Task 2,3 | P0 |
| 6. MeasurePos (矩形) | Caliper.cpp | 4h | Task 5 | P0 |
| 7. MeasurePos (弧形) | Caliper.cpp | 3h | Task 5 | P0 |
| 8. MeasurePairs 实现 | Caliper.cpp | 3h | Task 6 | P0 |
| 9. FuzzyMeasurePos | Caliper.cpp | 3h | Task 6 | P1 |
| 10. FuzzyMeasurePairs | Caliper.cpp | 2h | Task 8,9 | P1 |
| 11. 单元测试 | test_caliper.cpp | 4h | Task 6-8 | P0 |
| 12. 精度测试 | test_caliper_accuracy.cpp | 4h | Task 6-8 | P0 |

**总预估时间**：约 34 小时（约 4.5 工作日）

---

## 9. 测试要点

### 9.1 单元测试覆盖

1. **Handle 测试**
   - 创建各类型 Handle
   - 边界情况（零尺寸、负尺寸）
   - 坐标转换正确性
   - BoundingBox 计算

2. **MeasurePos 测试**
   - 单边缘检测
   - 多边缘检测
   - 不同极性过滤
   - 不同选择模式
   - 空图像处理

3. **MeasurePairs 测试**
   - 单对检测
   - 多对检测
   - 宽度过滤
   - 配对模式

4. **参数边界测试**
   - sigma 范围
   - minAmplitude 边界
   - numLines 极值

### 9.2 精度测试方法

```cpp
// 合成边缘图像测试
TEST(CaliperAccuracy, SubpixelPosition) {
    for (double truePos = 10.0; truePos < 11.0; truePos += 0.1) {
        auto image = GenerateSyntheticEdge(100, 100, truePos, contrast=100);
        auto handle = CreateMeasureRect(50, truePos, 0, 50, 10);
        auto edges = MeasurePos(image, handle, MeasureParams());
        
        ASSERT_EQ(edges.size(), 1);
        double error = std::abs(edges[0].column - truePos);
        EXPECT_LT(error, 0.03);  // 精度要求
    }
}

// 噪声鲁棒性测试
TEST(CaliperAccuracy, NoiseRobustness) {
    for (double noiseSigma : {0, 5, 10, 15}) {
        auto image = GenerateSyntheticEdge(100, 100, 50.5, contrast=50);
        AddGaussianNoise(image, noiseSigma);
        
        std::vector<double> errors;
        for (int i = 0; i < 100; ++i) {
            auto edges = MeasurePos(image, handle, params);
            errors.push_back(edges[0].column - 50.5);
        }
        
        double rms = ComputeRMS(errors);
        // 根据噪声水平调整期望
        EXPECT_LT(rms, 0.03 + 0.01 * noiseSigma);
    }
}
```

### 9.3 边界条件测试

- 边缘在测量区域边界
- 非常窄的边缘对
- 非常宽的边缘对
- 弱边缘（接近阈值）
- 弧形测量跨越 0/2pi

---

## 10. 附录

### 10.1 与 Halcon 接口对比

| Halcon | QiVision | 说明 |
|--------|----------|------|
| `gen_measure_rectangle2(Row, Col, Phi, Len1, Len2)` | `CreateMeasureRect(row, col, phi, length, width)` | 参数命名更清晰 |
| `measure_pos(Handle, Image, Sigma, Threshold, Trans, Select)` | `MeasurePos(image, handle, params)` | 参数封装为结构体 |
| RowEdge, ColEdge, Amp, Distance | EdgeResult.row/column/amplitude/profilePos | 结果结构化 |

### 10.2 坐标约定

```
图像坐标系:
  +--------> X (column)
  |
  |
  v Y (row)

剖面方向:
  Rectangle: phi + PI/2
  Arc: 沿弧切线方向
  Concentric: 径向向外
```

### 10.3 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 0.1 | 2026-01-08 | 初始设计 |
