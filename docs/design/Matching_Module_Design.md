# Matching 模块架构设计文档

## 1. 概述

### 1.1 功能描述

Matching（模板匹配）模块提供工业视觉中核心的目标检测与定位功能。通过在图像中搜索与预先训练的模板相似的区域，实现目标的精确定位、角度和尺度测量。

### 1.2 参考 Halcon 功能

| Halcon 功能类别 | 核心算子 | QiVision 对应 |
|----------------|----------|---------------|
| Shape-Based Matching | `create_shape_model`, `find_shape_model` | `ShapeModel` |
| NCC Matching | `create_ncc_model`, `find_ncc_model` | `NCCModel` |
| Component-Based | `create_component_model`, `find_component_model` | `ComponentModel` |
| Deformable | `create_local_deformable_model` | `DeformableModel` (P2) |

### 1.3 应用场景

- **定位检测**：零件位置、角度、尺度测量
- **装配验证**：元器件有无、正确性检测
- **缺陷检测**：通过匹配差异发现缺陷
- **分拣引导**：多目标识别与抓取定位

---

## 2. 设计规则验证

### 2.1 检查清单

- [x] **坐标类型符合规则**
  - 像素坐标：`int32_t`
  - 亚像素坐标：`double`
  - 匹配位置/角度/尺度：`double`

- [x] **层级依赖正确**
  - Feature (Matching) -> Internal (Gradient, Pyramid, SubPixel, Interpolate, NMS)
  - 无反向依赖
  - 无跨层依赖

- [x] **算法完整性满足**
  - 角度预计算 (AnglePyramid)
  - 各向异性缩放支持
  - 遮挡处理
  - NMS 去重

- [x] **Domain 规则**
  - 模型训练支持 Domain/Region 裁剪
  - 搜索支持指定 Domain (ROI)
  - 空 Domain -> 返回空结果

- [x] **结果返回规则**
  - 无匹配 -> 返回空 vector
  - 多结果按分数降序排列
  - 支持 NMS 去除重叠匹配

---

## 3. 计划支持的匹配方式

### 3.1 Shape-Based Matching (形状匹配) - P0 优先级

#### 适用场景
- **优势**：对光照变化鲁棒，适合工业环境
- **目标**：具有清晰边缘轮廓的刚性目标
- **应用**：PCB元器件、机械零件、包装检测

#### 核心算法原理
1. **模型创建**：
   - 从模板图像提取梯度方向特征（Sobel/Scharr）
   - 构建高斯金字塔，各层预计算梯度
   - 对每个角度预计算旋转后的梯度模型（AnglePyramid）

2. **匹配搜索**：
   - 从金字塔顶层（低分辨率）开始粗搜索
   - 使用梯度方向相似度作为匹配分数
   - 逐层精化候选位置，减少搜索空间
   - 最终层亚像素精化

3. **匹配分数计算**：
   ```
   Score = (1/N) * sum_i(cos(theta_model_i - theta_image_i))
   ```
   其中 theta 为梯度方向

#### 依赖的 Internal 模块

| 模块 | 用途 | 状态 |
|------|------|------|
| Gradient.h | Sobel/Scharr 梯度计算 | ✅ 完成 |
| Pyramid.h | 高斯/梯度金字塔 | ✅ 完成 |
| SubPixel.h | 位置/角度亚像素精化 | ✅ 完成 |
| Interpolate.h | 模型旋转插值 | ✅ 完成 |
| NonMaxSuppression.h | 去除重叠匹配 | ✅ 完成 |
| **AnglePyramid.h** | 角度预计算模型 | ⬜ 待实现 |

#### 精度和性能目标

| 指标 | 标准条件 | 困难条件 |
|------|----------|----------|
| 位置精度 (1 sigma) | < 0.05 px | < 0.1 px |
| 角度精度 (1 sigma) | < 0.05 deg | < 0.15 deg |
| 尺度精度 (1 sigma) | < 0.2% | < 0.5% |
| 搜索速度 | < 10ms (640x480) | < 30ms |
| 最小模型尺寸 | 20x20 px | - |
| 角度范围 | 0-360 deg | - |
| 尺度范围 | 0.8-1.2x | 0.5-2.0x |

---

### 3.2 NCC Matching (归一化互相关匹配) - P1 优先级

#### 适用场景
- **优势**：对几何形状不敏感，适合纹理目标
- **目标**：具有独特纹理或灰度模式的区域
- **应用**：芯片文字、纹理表面、印刷品检测

#### 核心算法原理
1. **模型创建**：
   - 存储模板灰度值（归一化）
   - 预计算模板均值和标准差
   - 可选：构建金字塔加速搜索

2. **匹配计算**：
   ```
   NCC = sum((I - I_mean) * (T - T_mean)) / (N * sigma_I * sigma_T)
   ```
   - 使用积分图加速局部统计计算
   - FFT 加速大模板匹配

3. **优化技术**：
   - 积分图预计算局部均值/方差
   - 阈值提前终止（early termination）
   - 金字塔粗到细搜索

#### 依赖的 Internal 模块

| 模块 | 用途 | 状态 |
|------|------|------|
| Pyramid.h | 高斯金字塔 | ✅ 完成 |
| Interpolate.h | 亚像素值采样 | ✅ 完成 |
| SubPixel.h | 位置精化 | ✅ 完成 |
| NonMaxSuppression.h | 去除重叠 | ✅ 完成 |
| **IntegralImage.h** | 快速局部统计 | ⬜ 待实现 |

#### 精度和性能目标

| 指标 | 标准条件 | 困难条件 |
|------|----------|----------|
| 位置精度 (1 sigma) | < 0.1 px | < 0.3 px |
| 匹配阈值 | [0.0, 1.0] | - |
| 搜索速度 | < 20ms (640x480) | < 50ms |
| 模板尺寸 | 10x10 - 200x200 px | - |

---

### 3.3 Component-Based Matching (组件匹配) - P1 优先级

#### 适用场景
- **优势**：处理柔性组合、遮挡和变形
- **目标**：由多个可独立移动部分组成的目标
- **应用**：柔性电缆、铰接部件、部分遮挡目标

#### 核心算法原理
1. **模型创建**：
   - 用户定义多个组件区域（Component Region）
   - 每个组件独立创建 ShapeModel
   - 定义组件间的空间关系约束（距离、角度范围）

2. **匹配搜索**：
   - 首先匹配锚点组件（最可靠的组件）
   - 基于锚点位置预测其他组件位置
   - 在预测区域搜索其他组件
   - 验证组件间关系约束

3. **分数计算**：
   ```
   Score = w_match * avg(component_scores) + w_relation * relation_score
   ```

#### 依赖的 Internal 模块

| 模块 | 用途 | 状态 |
|------|------|------|
| 所有 ShapeModel 依赖 | - | ✅ |
| Geometry2d.h | 组件关系约束 | ✅ 完成 |

#### 精度和性能目标

| 指标 | 值 |
|------|-----|
| 支持组件数 | 2-10 个 |
| 组件角度范围 | 各自独立 +/- 30 deg |
| 遮挡容忍 | 每组件最多 30% |

---

### 3.4 Deformable Matching (变形匹配) - P2 优先级

#### 适用场景
- **优势**：处理局部形变
- **目标**：可变形或透视畸变的目标
- **应用**：柔性材料、包装袋、带透视的表面

#### 核心算法原理
1. 将模型划分为网格
2. 允许各网格点独立小范围偏移
3. 优化整体匹配分数和变形能量

*详细设计将在 P2 阶段完成*

---

## 4. 模块划分和文件结构

### 4.1 目录结构

```
include/QiVision/Matching/
├── MatchTypes.h           # 公共类型、枚举、结果结构
├── ShapeModel.h           # 形状匹配类
├── NCCModel.h             # NCC 匹配类
├── ComponentModel.h       # 组件匹配类 (P1)
├── DeformableModel.h      # 变形匹配类 (P2)
└── MatchingUtils.h        # 工具函数

include/QiVision/Internal/
├── AnglePyramid.h         # 角度预计算模型 (新增)
└── IntegralImage.h        # 积分图 (新增)

src/Matching/
├── MatchTypes.cpp
├── ShapeModel.cpp
├── NCCModel.cpp
├── ComponentModel.cpp
└── MatchingUtils.cpp

src/Internal/
├── AnglePyramid.cpp
└── IntegralImage.cpp

tests/unit/matching/
├── test_shape_model.cpp
├── test_ncc_model.cpp
├── test_component_model.cpp
└── test_matching_utils.cpp

tests/accuracy/matching/
├── test_shape_model_accuracy.cpp
├── test_ncc_model_accuracy.cpp
└── test_matching_benchmark.cpp
```

### 4.2 模块依赖关系

```
                    ┌─────────────────┐
                    │ MatchTypes.h    │
                    │ (枚举、结果)     │
                    └────────┬────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
       v                     v                     v
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ ShapeModel   │     │ NCCModel     │     │ComponentModel│
│ (P0)         │     │ (P1)         │     │ (P1)         │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │              ┌─────┴─────┐              │
       │              │           │              │
       v              v           v              v
┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│AnglePyramid.h│ │IntegralImg│ │ Geometry2d│ │ShapeModel│
│ (新增)       │ │ (新增)    │ │          │ │  (复用)  │
└──────────────┘ └──────────┘ └──────────┘ └──────────┘
       │              │           │
       └──────────────┴───────────┘
                      │
       ┌──────────────┼──────────────┐
       │              │              │
       v              v              v
┌──────────┐   ┌──────────┐   ┌──────────┐
│ Gradient │   │ Pyramid  │   │ SubPixel │
│ (✅完成)  │   │ (✅完成)  │   │ (✅完成)  │
└──────────┘   └──────────┘   └──────────┘
```

---

## 5. 类设计

### 5.1 MatchTypes.h

```cpp
#pragma once

/**
 * @file MatchTypes.h
 * @brief Matching module common types and structures
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>

#include <cstdint>
#include <vector>
#include <string>

namespace Qi::Vision::Matching {

// =============================================================================
// Constants
// =============================================================================

/// Default minimum match score
constexpr double DEFAULT_MIN_SCORE = 0.5;

/// Default number of pyramid levels (0 = auto)
constexpr int32_t DEFAULT_NUM_LEVELS = 0;

/// Default angle step for precomputation (degrees)
constexpr double DEFAULT_ANGLE_STEP = 1.0;

/// Maximum number of matches to return
constexpr int32_t MAX_MATCHES = 1000;

/// Default NMS overlap threshold
constexpr double DEFAULT_NMS_OVERLAP = 0.5;

// =============================================================================
// Enumerations
// =============================================================================

/**
 * @brief Matching optimization mode
 */
enum class OptimizationMode {
    None,           ///< No precomputation (slow search, small memory)
    LowMemory,      ///< Partial precomputation
    Standard,       ///< Balanced precomputation (default)
    Fast            ///< Full precomputation (fast search, large memory)
};

/**
 * @brief Model contrast type
 */
enum class ContrastMode {
    Normal,         ///< Normal contrast (white on black or black on white)
    LowContrast,    ///< Low contrast mode (more sensitive)
    HighContrast    ///< High contrast mode (faster, less sensitive)
};

/**
 * @brief Subpixel refinement mode
 */
enum class SubPixelMode {
    None,           ///< Integer pixel only
    LeastSquares,   ///< Least squares refinement (default)
    Interpolation   ///< Response interpolation
};

/**
 * @brief Model polarity
 */
enum class ModelPolarity {
    SameAsModel,    ///< Same polarity as training image
    Inverted,       ///< Inverted polarity
    Any             ///< Either polarity (check both)
};

/**
 * @brief Metric for match comparison
 */
enum class MatchMetric {
    UsePolarity,    ///< Use gradient direction sign
    IgnorePolarity, ///< Ignore gradient direction sign
    IgnoreLocalPolarity ///< Per-point polarity invariance
};

// =============================================================================
// Parameter Structures
// =============================================================================

/**
 * @brief Parameters for model creation
 */
struct CreateModelParams {
    // Pyramid parameters
    int32_t numLevels = DEFAULT_NUM_LEVELS;     ///< 0 = auto compute
    double startAngle = 0.0;                     ///< Start angle (degrees)
    double angleExtent = 360.0;                  ///< Angle extent (degrees)
    double angleStep = DEFAULT_ANGLE_STEP;       ///< Angle step (degrees)
    double minScale = 1.0;                       ///< Minimum scale
    double maxScale = 1.0;                       ///< Maximum scale
    double scaleStep = 0.0;                      ///< Scale step (0 = no scale)
    
    // Optimization
    OptimizationMode optimization = OptimizationMode::Standard;
    ContrastMode contrast = ContrastMode::Normal;
    
    // Feature extraction
    double minContrast = 10.0;                   ///< Minimum gradient magnitude
    int32_t minModelPoints = 10;                 ///< Minimum model feature points
    
    // Builder pattern
    CreateModelParams& SetAngleRange(double start, double extent, double step) {
        startAngle = start; angleExtent = extent; angleStep = step;
        return *this;
    }
    CreateModelParams& SetScaleRange(double minS, double maxS, double step) {
        minScale = minS; maxScale = maxS; scaleStep = step;
        return *this;
    }
    CreateModelParams& SetNumLevels(int32_t n) { numLevels = n; return *this; }
    CreateModelParams& SetOptimization(OptimizationMode m) { optimization = m; return *this; }
    CreateModelParams& SetMinContrast(double c) { minContrast = c; return *this; }
};

/**
 * @brief Parameters for find operation
 */
struct FindModelParams {
    // Search parameters
    double minScore = DEFAULT_MIN_SCORE;         ///< Minimum match score [0, 1]
    int32_t maxMatches = MAX_MATCHES;            ///< Maximum matches to return
    double greediness = 0.9;                     ///< Search greediness [0, 1]
    
    // Angle/Scale override (empty = use model range)
    double searchAngleStart = 0.0;               ///< Search angle start (degrees)
    double searchAngleExtent = 0.0;              ///< 0 = use model range
    double searchScaleMin = 0.0;                 ///< 0 = use model range
    double searchScaleMax = 0.0;                 ///< 0 = use model range
    
    // Refinement
    SubPixelMode subPixel = SubPixelMode::LeastSquares;
    
    // Result processing
    double nmsOverlap = DEFAULT_NMS_OVERLAP;     ///< NMS overlap threshold
    bool sortByScore = true;                     ///< Sort results by score
    
    // Polarity
    ModelPolarity polarity = ModelPolarity::SameAsModel;
    MatchMetric metric = MatchMetric::UsePolarity;
    
    // Builder pattern
    FindModelParams& SetMinScore(double s) { minScore = s; return *this; }
    FindModelParams& SetMaxMatches(int32_t n) { maxMatches = n; return *this; }
    FindModelParams& SetGreediness(double g) { greediness = g; return *this; }
    FindModelParams& SetSubPixel(SubPixelMode m) { subPixel = m; return *this; }
    FindModelParams& SetSearchAngle(double start, double extent) {
        searchAngleStart = start; searchAngleExtent = extent;
        return *this;
    }
};

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief Single match result
 */
struct MatchResult {
    // Position (subpixel)
    double row = 0.0;           ///< Y coordinate (row)
    double column = 0.0;        ///< X coordinate (column)
    
    // Transformation
    double angle = 0.0;         ///< Rotation angle (degrees)
    double scaleRow = 1.0;      ///< Scale in row direction
    double scaleCol = 1.0;      ///< Scale in column direction
    
    // Quality
    double score = 0.0;         ///< Match score [0, 1]
    
    // Model reference
    int32_t modelId = 0;        ///< Model ID (for multi-model)
    int32_t pyramidLevel = 0;   ///< Final pyramid level
    
    /// Check if result is valid
    bool IsValid() const { return score > 0; }
    
    /// Get position as Point2d
    Point2d Position() const { return {column, row}; }
    
    /// Get uniform scale (average of row/col)
    double Scale() const { return (scaleRow + scaleCol) * 0.5; }
    
    /// Get 2D affine transform matrix
    // Returns 2x3 matrix [cos*s, -sin*s, tx; sin*s, cos*s, ty]
    std::array<double, 6> GetAffineMatrix() const;
    
    /// Transform a point from model coordinates to image coordinates
    Point2d TransformPoint(const Point2d& modelPoint) const;
};

/**
 * @brief Component match result (for ComponentModel)
 */
struct ComponentMatchResult : public MatchResult {
    std::vector<MatchResult> components;    ///< Individual component matches
    double relationScore = 0.0;             ///< Component relation score
    bool isComplete = true;                 ///< All components found
};

/**
 * @brief Model statistics
 */
struct ModelStats {
    int32_t numPoints = 0;          ///< Number of feature points
    int32_t numLevels = 0;          ///< Number of pyramid levels
    int32_t numAngles = 0;          ///< Number of precomputed angles
    int32_t numScales = 0;          ///< Number of precomputed scales
    
    double minAngle = 0.0;          ///< Minimum angle (degrees)
    double maxAngle = 0.0;          ///< Maximum angle (degrees)
    double minScale = 1.0;          ///< Minimum scale
    double maxScale = 1.0;          ///< Maximum scale
    
    size_t memoryBytes = 0;         ///< Model memory usage
    
    Rect2d boundingBox;             ///< Model bounding box
    Point2d origin;                 ///< Model origin (reference point)
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Compute optimal number of pyramid levels
 */
int32_t ComputeOptimalLevels(int32_t modelWidth, int32_t modelHeight,
                              int32_t minDimension = 8);

/**
 * @brief Compute number of angle steps
 */
int32_t ComputeNumAngles(double angleExtent, double angleStep);

/**
 * @brief Convert degrees to radians
 */
inline double DegToRad(double deg) { return deg * M_PI / 180.0; }

/**
 * @brief Convert radians to degrees
 */
inline double RadToDeg(double rad) { return rad * 180.0 / M_PI; }

} // namespace Qi::Vision::Matching
```

### 5.2 ShapeModel.h

```cpp
#pragma once

/**
 * @file ShapeModel.h
 * @brief Shape-based template matching
 *
 * Provides robust shape-based matching using gradient direction features.
 * Based on: Ulrich & Steger, "Robust Template Matching" (2003)
 *
 * Features:
 * - Rotation invariant (arbitrary angle range)
 * - Optional scale invariance
 * - Multi-scale pyramid search
 * - Partial occlusion handling
 * - Subpixel position/angle refinement
 *
 * Thread safety:
 * - Model creation: NOT thread-safe (one thread)
 * - Model search: Thread-safe (multiple threads can use same model)
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Core/QRegion.h>
#include <QiVision/Core/QContour.h>
#include <QiVision/Matching/MatchTypes.h>

#include <memory>
#include <vector>

namespace Qi::Vision::Matching {

// Forward declaration
class ShapeModelImpl;

/**
 * @brief Shape-based matching model
 *
 * Usage:
 * @code
 * // Create model from template image
 * ShapeModel model;
 * auto params = CreateModelParams()
 *     .SetAngleRange(0, 360, 1)    // Full rotation, 1 degree step
 *     .SetNumLevels(4);            // 4 pyramid levels
 * model.Create(templateImage, params);
 *
 * // Find matches in search image
 * auto findParams = FindModelParams()
 *     .SetMinScore(0.7)
 *     .SetMaxMatches(10);
 * auto matches = model.Find(searchImage, findParams);
 *
 * // Process results
 * for (const auto& m : matches) {
 *     std::cout << "Match at (" << m.column << ", " << m.row << ")"
 *               << " angle=" << m.angle << " score=" << m.score << "\n";
 * }
 * @endcode
 */
class ShapeModel {
public:
    ShapeModel();
    ~ShapeModel();
    
    // Move only (no copy)
    ShapeModel(ShapeModel&& other) noexcept;
    ShapeModel& operator=(ShapeModel&& other) noexcept;
    ShapeModel(const ShapeModel&) = delete;
    ShapeModel& operator=(const ShapeModel&) = delete;

    // =========================================================================
    // Model Creation
    // =========================================================================
    
    /**
     * @brief Create model from grayscale image
     *
     * @param templateImage Template image (grayscale, 8-bit or 16-bit)
     * @param params Creation parameters
     * @throws Exception if image is empty or invalid
     *
     * @note Uses full image domain by default
     * @note Larger angleStep = faster search, less memory, lower accuracy
     */
    void Create(const QImage& templateImage,
                const CreateModelParams& params = CreateModelParams());
    
    /**
     * @brief Create model from image with region mask
     *
     * @param templateImage Template image
     * @param region Region of interest (only these pixels contribute)
     * @param params Creation parameters
     */
    void Create(const QImage& templateImage,
                const QRegion& region,
                const CreateModelParams& params = CreateModelParams());
    
    /**
     * @brief Create model from XLD contour (shape outline)
     *
     * @param contour XLD contour defining the shape
     * @param params Creation parameters
     *
     * @note Contour should be well-sampled
     */
    void CreateFromContour(const QContour& contour,
                           const CreateModelParams& params = CreateModelParams());
    
    /**
     * @brief Create model from points and gradients (advanced)
     *
     * @param points Feature point coordinates (relative to origin)
     * @param gradients Gradient directions at each point (radians)
     * @param params Creation parameters
     */
    void CreateFromFeatures(const std::vector<Point2d>& points,
                            const std::vector<double>& gradients,
                            const CreateModelParams& params = CreateModelParams());

    // =========================================================================
    // Model Search
    // =========================================================================
    
    /**
     * @brief Find model instances in image
     *
     * @param image Search image (grayscale)
     * @param params Search parameters
     * @return Vector of match results (sorted by score if requested)
     *
     * @note Returns empty vector if no matches found (no exception)
     * @note Thread-safe: multiple threads can call simultaneously
     */
    std::vector<MatchResult> Find(const QImage& image,
                                   const FindModelParams& params = FindModelParams()) const;
    
    /**
     * @brief Find model in region of interest
     *
     * @param image Search image
     * @param searchRegion Region to search in
     * @param params Search parameters
     */
    std::vector<MatchResult> Find(const QImage& image,
                                   const QRegion& searchRegion,
                                   const FindModelParams& params = FindModelParams()) const;
    
    /**
     * @brief Find model at specific location (for refinement)
     *
     * Performs local search around given position/angle.
     *
     * @param image Search image
     * @param initialRow Approximate row
     * @param initialCol Approximate column
     * @param initialAngle Approximate angle (degrees)
     * @param searchRadius Position search radius (pixels)
     * @param angleRadius Angle search radius (degrees)
     * @return Best match in search area, or invalid if not found
     */
    MatchResult FindLocal(const QImage& image,
                          double initialRow, double initialCol,
                          double initialAngle,
                          double searchRadius = 5.0,
                          double angleRadius = 5.0) const;

    // =========================================================================
    // Model Information
    // =========================================================================
    
    /// Check if model is valid
    bool IsValid() const;
    
    /// Get model statistics
    ModelStats GetStats() const;
    
    /// Get model origin (reference point)
    Point2d GetOrigin() const;
    
    /// Set model origin (reference point)
    void SetOrigin(const Point2d& origin);
    
    /// Get model contour (outline at angle 0, scale 1)
    QContour GetContour() const;
    
    /// Get model feature points
    std::vector<Point2d> GetFeaturePoints() const;
    
    /// Get angle range
    void GetAngleRange(double& startAngle, double& angleExtent) const;
    
    /// Get scale range
    void GetScaleRange(double& minScale, double& maxScale) const;

    // =========================================================================
    // Serialization
    // =========================================================================
    
    /**
     * @brief Save model to file
     *
     * @param filename Output file path
     * @return true if successful
     *
     * Supports formats:
     * - .qsm: Binary format (compact, fast)
     * - .json: JSON format (readable, larger)
     */
    bool Save(const std::string& filename) const;
    
    /**
     * @brief Load model from file
     *
     * @param filename Input file path
     * @return true if successful
     */
    bool Load(const std::string& filename);
    
    /// Clear model data
    void Clear();

private:
    std::unique_ptr<ShapeModelImpl> impl_;
};

// =============================================================================
// Convenience Functions
// =============================================================================

/**
 * @brief Create shape model in one call
 */
ShapeModel CreateShapeModel(const QImage& templateImage,
                            const CreateModelParams& params = CreateModelParams());

/**
 * @brief Create shape model from region
 */
ShapeModel CreateShapeModel(const QImage& templateImage,
                            const QRegion& region,
                            const CreateModelParams& params = CreateModelParams());

/**
 * @brief Find shape model in image
 */
std::vector<MatchResult> FindShapeModel(const ShapeModel& model,
                                         const QImage& image,
                                         const FindModelParams& params = FindModelParams());

} // namespace Qi::Vision::Matching
```

### 5.3 Internal/AnglePyramid.h (新增)

```cpp
#pragma once

/**
 * @file AnglePyramid.h
 * @brief Precomputed angle models for shape matching
 *
 * Stores rotated versions of the model at discrete angles
 * for fast angle-invariant matching.
 *
 * Memory/Speed tradeoff:
 * - More angles = more memory, faster search
 * - Fewer angles = less memory, slower search (needs interpolation)
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Internal/Pyramid.h>

#include <vector>
#include <cstdint>

namespace Qi::Vision::Internal {

/**
 * @brief Feature point in model
 */
struct ModelPoint {
    double x = 0.0;             ///< X coordinate (relative to origin)
    double y = 0.0;             ///< Y coordinate (relative to origin)
    double gradient = 0.0;      ///< Gradient direction (radians)
    double magnitude = 0.0;     ///< Gradient magnitude (for weighting)
    int32_t level = 0;          ///< Pyramid level
};

/**
 * @brief Single angle model
 */
struct AngleModel {
    double angle = 0.0;                     ///< Angle (radians)
    std::vector<ModelPoint> points;         ///< Rotated model points
    Rect2d boundingBox;                     ///< Bounding box at this angle
};

/**
 * @brief Pyramid of angle models
 */
class AnglePyramid {
public:
    AnglePyramid() = default;
    
    /**
     * @brief Build angle pyramid from base model points
     *
     * @param basePoints Points at angle 0
     * @param numLevels Number of pyramid levels
     * @param startAngle Start angle (radians)
     * @param angleExtent Angle extent (radians)
     * @param angleStep Angle step (radians)
     */
    void Build(const std::vector<ModelPoint>& basePoints,
               int32_t numLevels,
               double startAngle, double angleExtent, double angleStep);
    
    /**
     * @brief Build with scale support
     */
    void Build(const std::vector<ModelPoint>& basePoints,
               int32_t numLevels,
               double startAngle, double angleExtent, double angleStep,
               double minScale, double maxScale, double scaleStep);
    
    /// Get number of angles
    int32_t NumAngles() const { return static_cast<int32_t>(angleModels_.size()); }
    
    /// Get number of scales
    int32_t NumScales() const { return numScales_; }
    
    /// Get number of pyramid levels
    int32_t NumLevels() const { return numLevels_; }
    
    /// Get angle model at index
    const AngleModel& GetAngleModel(int32_t angleIndex) const;
    
    /// Get angle model with scale
    const AngleModel& GetAngleModel(int32_t angleIndex, int32_t scaleIndex) const;
    
    /// Find nearest angle index
    int32_t FindNearestAngle(double angle) const;
    
    /// Get points at specific level, angle, and scale
    const std::vector<ModelPoint>& GetLevelPoints(int32_t level, 
                                                   int32_t angleIndex,
                                                   int32_t scaleIndex = 0) const;
    
    /// Get angle step (radians)
    double AngleStep() const { return angleStep_; }
    
    /// Get angle range
    void GetAngleRange(double& start, double& extent) const {
        start = startAngle_;
        extent = angleExtent_;
    }
    
    /// Get memory usage
    size_t MemoryBytes() const;
    
    /// Clear all data
    void Clear();

private:
    std::vector<AngleModel> angleModels_;
    std::vector<std::vector<std::vector<ModelPoint>>> levelPoints_;  // [level][angle][scale]
    
    double startAngle_ = 0.0;
    double angleExtent_ = 0.0;
    double angleStep_ = 0.0;
    
    double minScale_ = 1.0;
    double maxScale_ = 1.0;
    double scaleStep_ = 0.0;
    int32_t numScales_ = 1;
    
    int32_t numLevels_ = 0;
    
    void RotatePoints(const std::vector<ModelPoint>& src,
                      std::vector<ModelPoint>& dst,
                      double angle) const;
    
    void ScalePoints(const std::vector<ModelPoint>& src,
                     std::vector<ModelPoint>& dst,
                     double scale) const;
};

/**
 * @brief Compute gradient-based matching score
 *
 * @param modelPoints Rotated model points
 * @param imageGradX Image gradient X
 * @param imageGradY Image gradient Y
 * @param width Image width
 * @param height Image height
 * @param offsetX X offset in image
 * @param offsetY Y offset in image
 * @param greediness Early termination factor [0, 1]
 * @return Match score [0, 1]
 */
double ComputeMatchScore(const std::vector<ModelPoint>& modelPoints,
                          const float* imageGradX,
                          const float* imageGradY,
                          int32_t width, int32_t height,
                          double offsetX, double offsetY,
                          double greediness = 0.9);

/**
 * @brief Refine match position and angle
 *
 * Uses least-squares optimization to refine the match.
 *
 * @param modelPoints Rotated model points
 * @param imageGradX Image gradient X
 * @param imageGradY Image gradient Y
 * @param width Image width
 * @param height Image height
 * @param[in,out] row Match row (refined in place)
 * @param[in,out] col Match column (refined in place)
 * @param[in,out] angle Match angle (refined in place)
 * @param maxIterations Maximum refinement iterations
 * @return Final match score
 */
double RefineMatch(const std::vector<ModelPoint>& modelPoints,
                   const float* imageGradX,
                   const float* imageGradY,
                   int32_t width, int32_t height,
                   double& row, double& col, double& angle,
                   int32_t maxIterations = 10);

} // namespace Qi::Vision::Internal
```

---

## 6. 精度规格

### 6.1 测试条件定义

| 条件等级 | 对比度 | 噪声 sigma | 遮挡 |
|----------|--------|------------|------|
| 标准条件 | >= 50 | <= 5 | 0% |
| 困难条件 | 20-50 | 5-15 | <= 30% |
| 极端条件 | 10-20 | 15-25 | <= 50% |

### 6.2 ShapeModel 精度要求

| 测量项 | 标准条件 | 困难条件 | 测试方法 |
|--------|----------|----------|----------|
| 位置 X (1 sigma) | < 0.05 px | < 0.1 px | 合成旋转图像，统计误差 |
| 位置 Y (1 sigma) | < 0.05 px | < 0.1 px | 合成旋转图像，统计误差 |
| 角度 (1 sigma) | < 0.05 deg | < 0.15 deg | 已知角度旋转，统计误差 |
| 尺度 (1 sigma) | < 0.2% | < 0.5% | 已知缩放，统计误差 |
| 召回率 | > 99% | > 95% | 已知位置目标检出率 |
| 误检率 | < 0.1% | < 1% | 无目标图像误检率 |

### 6.3 NCCModel 精度要求

| 测量项 | 标准条件 | 困难条件 |
|--------|----------|----------|
| 位置 X (1 sigma) | < 0.1 px | < 0.3 px |
| 位置 Y (1 sigma) | < 0.1 px | < 0.3 px |
| 分数阈值对应 | 0.9 -> 高可信 | 0.7 -> 可用 |

---

## 7. 实现优先级排序

### Phase 1: 核心形状匹配 (P0) - 预估 8 工作日

| 任务 | 文件 | 预估时间 | 依赖 |
|------|------|----------|------|
| 1. MatchTypes 定义 | MatchTypes.h | 4h | 无 |
| 2. AnglePyramid 实现 | Internal/AnglePyramid.h/cpp | 12h | Pyramid.h |
| 3. ShapeModel 创建 | ShapeModel.cpp | 10h | AnglePyramid, Gradient |
| 4. 金字塔搜索 | ShapeModel.cpp | 8h | Task 3 |
| 5. 亚像素精化 | ShapeModel.cpp | 6h | Task 4, SubPixel |
| 6. NMS 集成 | ShapeModel.cpp | 4h | Task 5, NMS |
| 7. 序列化 | ShapeModel.cpp | 4h | Task 6 |
| 8. 单元测试 | test_shape_model.cpp | 8h | Task 6 |
| 9. 精度测试 | test_shape_model_accuracy.cpp | 8h | Task 6 |

### Phase 2: NCC 匹配 (P1) - 预估 5 工作日

| 任务 | 文件 | 预估时间 | 依赖 |
|------|------|----------|------|
| 1. IntegralImage 实现 | Internal/IntegralImage.h/cpp | 6h | 无 |
| 2. NCCModel 创建 | NCCModel.cpp | 6h | IntegralImage |
| 3. NCC 搜索 | NCCModel.cpp | 8h | Task 2 |
| 4. 亚像素精化 | NCCModel.cpp | 4h | Task 3 |
| 5. 单元测试 | test_ncc_model.cpp | 6h | Task 4 |
| 6. 精度测试 | test_ncc_model_accuracy.cpp | 6h | Task 4 |

### Phase 3: 组件匹配 (P1) - 预估 4 工作日

| 任务 | 文件 | 预估时间 | 依赖 |
|------|------|----------|------|
| 1. ComponentModel 设计 | ComponentModel.h | 4h | ShapeModel 完成 |
| 2. 组件管理 | ComponentModel.cpp | 8h | Task 1 |
| 3. 关系约束 | ComponentModel.cpp | 6h | Task 2, Geometry2d |
| 4. 搜索算法 | ComponentModel.cpp | 8h | Task 3 |
| 5. 测试 | test_component_model.cpp | 6h | Task 4 |

### 总预估时间

| Phase | 预估时间 | 累计 |
|-------|----------|------|
| Phase 1 (ShapeModel) | 64h (~8 天) | 8 天 |
| Phase 2 (NCCModel) | 36h (~5 天) | 13 天 |
| Phase 3 (ComponentModel) | 32h (~4 天) | 17 天 |

---

## 8. 测试要点

### 8.1 单元测试覆盖

1. **模型创建测试**
   - 从图像创建（各种尺寸）
   - 从 Region 创建
   - 从 Contour 创建
   - 无效输入处理

2. **搜索测试**
   - 单目标检测
   - 多目标检测
   - 不同角度
   - 不同尺度
   - ROI 搜索

3. **序列化测试**
   - 保存/加载正确性
   - 版本兼容性

4. **参数边界测试**
   - 极端角度步长
   - 极端尺度范围
   - 极端分数阈值

### 8.2 精度测试方法

```cpp
TEST(ShapeModelAccuracy, PositionPrecision) {
    // 创建测试模板
    auto templateImg = GenerateSyntheticShape(100, 100, ShapeType::Cross);
    ShapeModel model;
    model.Create(templateImg, CreateModelParams().SetAngleRange(0, 360, 1));
    
    // 多次测试不同位置
    std::vector<double> errorX, errorY;
    for (double trueX = 100; trueX < 500; trueX += 50) {
        for (double trueY = 100; trueY < 400; trueY += 50) {
            // 生成搜索图像（模板放置在已知位置）
            auto searchImg = GenerateTestImage(640, 480, templateImg, trueX, trueY);
            
            auto matches = model.Find(searchImg);
            ASSERT_GE(matches.size(), 1);
            
            errorX.push_back(matches[0].column - trueX);
            errorY.push_back(matches[0].row - trueY);
        }
    }
    
    // 验证精度
    double rmsX = ComputeRMS(errorX);
    double rmsY = ComputeRMS(errorY);
    EXPECT_LT(rmsX, 0.05);  // 要求 < 0.05 像素
    EXPECT_LT(rmsY, 0.05);
}
```

### 8.3 性能基准测试

```cpp
TEST(ShapeModelBenchmark, SearchSpeed) {
    // 640x480 图像，100x100 模板，360度搜索
    auto templateImg = LoadTestImage("template_100x100.png");
    auto searchImg = LoadTestImage("search_640x480.png");
    
    ShapeModel model;
    model.Create(templateImg, CreateModelParams().SetAngleRange(0, 360, 1));
    
    auto params = FindModelParams().SetMinScore(0.7).SetMaxMatches(10);
    
    // 预热
    model.Find(searchImg, params);
    
    // 计时
    auto start = HighResTimer::Now();
    for (int i = 0; i < 100; ++i) {
        auto matches = model.Find(searchImg, params);
    }
    auto elapsed = HighResTimer::ElapsedMs(start);
    
    std::cout << "Average search time: " << (elapsed / 100) << " ms\n";
    EXPECT_LT(elapsed / 100, 10.0);  // 要求 < 10ms
}
```

---

## 9. 与 Halcon 接口对比

| Halcon 算子 | QiVision 接口 | 说明 |
|-------------|---------------|------|
| `create_shape_model(Image, NumLevels, AngleStart, ...)` | `ShapeModel::Create(image, params)` | 参数封装为结构体 |
| `find_shape_model(Image, Model, ...)` | `model.Find(image, params)` | 面向对象调用 |
| `get_shape_model_params(Model, NumLevels, ...)` | `model.GetStats()` | 统一获取统计信息 |
| `write_shape_model(Model, File)` | `model.Save(filename)` | 文件操作 |
| `read_shape_model(File, Model)` | `model.Load(filename)` | 文件操作 |
| `clear_shape_model(Model)` | `model.Clear()` | 清理资源 |

---

## 10. 待确认问题

1. **各向异性尺度**：是否需要支持 ScaleRow != ScaleCol？
   - 建议 Phase 1 仅支持等比例缩放
   - Phase 2 扩展各向异性

2. **极性处理**：IgnoreLocalPolarity 是否必需？
   - 增加复杂度和计算量
   - 可在 Phase 2 评估需求后添加

3. **GPU 加速**：是否规划 GPU 版本？
   - 暂时不在 Matching 模块范围
   - 可后续通过 Platform/GPU.h 扩展

---

## 11. 变更历史

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| 0.1 | 2026-01-08 | 初始架构设计 |
