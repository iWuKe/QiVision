# QiVision 编码规范

## Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Namespace | PascalCase | `Qi::Vision::Matching` |
| Class | PascalCase, Q prefix for core | `QImage`, `ShapeModel` |
| Function/Method | PascalCase | `CreateShapeModel()`, `FindScaledShapeModel()` |
| Variable | camelCase | `imageWidth`, `modelPoints` |
| Private member | trailing underscore | `data_`, `threshold_` |
| Constant | UPPER_SNAKE_CASE | `MAX_PYRAMID_LEVELS` |
| Enum value | PascalCase | `EdgePolarity::Positive` |
| File name | PascalCase | `QRegion.h`, `ShapeModel.cpp` |

## API 命名规范 (参考 Halcon)

**命名格式**: `<动作><修饰词><模块名称>`

| 动作 | 含义 | 示例 |
|------|------|------|
| Create | 创建模型 | `CreateShapeModel()`, `CreateScaledShapeModel()` |
| Find | 查找/匹配 | `FindShapeModel()`, `FindScaledShapeModel()` |
| Read | 从文件读取 | `ReadShapeModel()` |
| Write | 写入文件 | `WriteShapeModel()` |
| Get | 获取属性 | `GetShapeModelParams()`, `GetShapeModelContours()` |
| Set | 设置属性 | `SetShapeModelOrigin()` |
| Clear | 清除/释放 | `ClearShapeModel()` |
| Determine | 自动确定参数 | `DetermineShapeModelParams()` |

### ShapeModel 完整 API 命名

| Halcon | QiVision | 说明 |
|--------|----------|------|
| `create_shape_model` | `CreateShapeModel()` | 只带旋转 |
| `create_scaled_shape_model` | `CreateScaledShapeModel()` | 带各向同性缩放 |
| `create_aniso_shape_model` | `CreateAnisoShapeModel()` | 带各向异性缩放 |
| `create_shape_model_xld` | `CreateShapeModelXld()` | 从轮廓创建 |
| `find_shape_model` | `FindShapeModel()` | 查找（只旋转） |
| `find_scaled_shape_model` | `FindScaledShapeModel()` | 查找（带缩放） |
| `find_aniso_shape_model` | `FindAnisoShapeModel()` | 查找（各向异性） |
| `find_shape_models` | `FindShapeModels()` | 多模型查找 |
| `read_shape_model` | `ReadShapeModel()` | 从文件读取 |
| `write_shape_model` | `WriteShapeModel()` | 写入文件 |
| `get_shape_model_params` | `GetShapeModelParams()` | 获取模型参数 |
| `get_shape_model_contours` | `GetShapeModelContours()` | 获取轮廓 |
| `set_shape_model_origin` | `SetShapeModelOrigin()` | 设置原点 |
| `clear_shape_model` | `ClearShapeModel()` | 释放模型 |

### 其他模块命名示例

| 模块 | Create | Find/Apply | 说明 |
|------|--------|------------|------|
| NCC模板 | `CreateNccModel()` | `FindNccModel()` | 灰度相关匹配 |
| 卡尺 | `CreateMetrologyModel()` | `ApplyMetrologyModel()` | 测量 |
| 标定 | `CreateCalibData()` | `CalibrateCamera()` | 相机标定 |
| Blob | - | `Connection()`, `SelectShape()` | 区域分析 |

## Header Include Order

```cpp
#pragma once

// 1. Project headers
#include <QiVision/Core/Types.h>

// 2. Third-party libraries

// 3. C++ standard library
#include <memory>
#include <vector>

// 4. C standard library
#include <cmath>
```

## Namespaces

```cpp
namespace Qi::Vision { }              // Top-level API
namespace Qi::Vision::Internal { }    // Internal (not exported)
namespace Qi::Vision::Platform { }    // Platform abstraction
namespace Qi::Vision::Matching { }    // Template matching
namespace Qi::Vision::Measure { }     // Measurement
namespace Qi::Vision::Calib { }       // Calibration
// ... and others per Feature module
```

## Git Workflow

### Branch Naming
```
feature/<module>-<brief>   # Feature development
fix/<issue>-<brief>        # Bug fixes
internal/<module>          # Internal module development
refactor/<scope>           # Refactoring
```

### Commit Format
```
<type>(<scope>): <subject>

type: feat, fix, refactor, test, docs, perf, chore
scope: Core, Internal, Measure, Matching, Platform, etc.

Examples:
feat(Measure): implement arc caliper
fix(ShapeModel): correct angle interpolation
perf(Internal): add AVX2 for bilinear interpolation
```

## Core Algorithm Papers

1. Steger - "An Unbiased Detector of Curvilinear Structures" (1998)
2. Ulrich, Steger - "Robust Template Matching" (2003)
3. Hinterstoisser - "LINEMOD: Gradient Response Maps" (2012)
4. Zhang - "A Flexible New Technique for Camera Calibration" (2000)
5. Fitzgibbon - "Direct Least Square Fitting of Ellipses" (1999)
