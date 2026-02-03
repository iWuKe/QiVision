# API Unification Plan (OpenCV-style)

Goal: expose a single, modern C++ API surface (OpenCV-like), remove Halcon-style handle/string entrances from the public SDK, and keep internal implementation details private.

## Principles
- One public way to do a thing.
- Use enums/structs instead of string parameters.
- Prefer class/namespace-based APIs over free-function handle patterns.
- Internal implementation must not be reachable from public headers.

## Module-by-Module Decisions

### Core / IO
**Keep (public):**
- `Qi::Vision::IO::ReadImage`, `Qi::Vision::IO::WriteImage` (I/O centralized)

**Deprecate/remove:**
- `QImage::FromFile`, `QImage::SaveToFile` (removed)

**Rationale:** avoid duplicated I/O entry points.

---

### Color
**Keep (public):**
- `Qi::Vision::Color::ConvertColorSpace`, `Decompose/Compose`, `AccessChannel`

**Deprecate/remove:**
- `QImage::ToGray` (removed)

**Rationale:** all color conversion in one module.

---

### Display / GUI / Draw
**Keep (public):**
- `Qi::Vision::GUI::Window` (interactive display)
- `Qi::Vision::Draw` primitives (single draw module)

**Deprecate/remove:**
- `Qi::Vision::DispImage`, `Display::Disp*` (duplicate rendering wrappers)
- `Core/Draw.h` compatibility header (removed)

**Rationale:** one interactive display class + one draw namespace.

---

### Measure (Caliper / Metrology)
**Keep (public):**
- `Qi::Vision::Measure::CaliperArray`
- `Qi::Vision::Measure::Metrology` (high-level geometric measurement)

**Deprecate/remove:**
- `GenMeasure*`, `MeasurePos`, `MeasurePairs` (removed)

**Rationale:** single engineering-oriented entry point with batch/fit results.

---

### Morphology / QRegion
**Keep (public):**
- `Qi::Vision::Morphology::*` operators

**Deprecate/remove:**
- `QRegion::Dilate/Erode/Opening/Closing` (duplicate morphology)

**Rationale:** avoid both class methods and free functions for same ops.

---

### Matching
**Keep (public):**
- `NCCModel`, `ShapeModel` classes + function APIs (but no internal impl exposure)

**Deprecate/remove:**
- `Impl()` accessors in public headers

**Rationale:** models can remain handle-style, but implementation stays private.

---

### Internal Headers
**Change:**
- Move `include/QiVision/Internal/*` out of public install/include path
- Remove any direct reference to `Qi::Vision::Internal` types from public headers

**Rationale:** prevent SDK users from depending on internal APIs.

## Parameter Modernization
- Replace string parameters (e.g., "all", "first") with enums or `struct Params`.
- Keep legacy string overloads only during migration; remove in final cleanup.

## Migration Stages
1. Document unified surface and mark deprecated APIs.
2. Update samples and docs to use unified APIs exclusively.
3. Hide/remove `Internal` headers from public includes.
4. Remove deprecated APIs after internal usage is gone.

## Unified Behavior Rules
- Invalid arguments: throw `InvalidArgumentException` with a stable, descriptive message.
- Unsupported types/parameters: throw `UnsupportedException` (never silently coerce).
- Empty inputs:
  - Empty image/region/contour: return empty outputs when a "no-op" makes sense.
  - If the operation requires data (fit, model creation), throw `InvalidArgumentException`.
- Numeric ranges:
  - Angles must be finite; scales > 0; thresholds >= 0; counts >= 1.
  - Reject NaN/Inf for all geometric inputs.
- Output clearing:
  - Functions with output vectors must clear them before filling.
- Consistent coordinate semantics:
  - `row = y`, `col = x`, and angles are in radians (CCW positive).

## Module-Specific Exceptions (Allowed)
These are intentional deviations from "empty input returns empty output" when a module
cannot meaningfully proceed without real data or initialization.

### OCR
- Not initialized / model missing: throw `InvalidArgumentException`.
- Empty image: return empty `OCRResult`.
- Unsupported type/channels: throw `UnsupportedException`.

### Barcode
- Empty image: return empty result vector.
- Invalid params (formats None, negative limits): throw `InvalidArgumentException`.
- Unsupported type/channels: throw `UnsupportedException`.

### Measure / Matching
- Empty image or insufficient data for fitting: throw `InvalidArgumentException`
  or `InsufficientDataException` (when applicable).

### Display / Draw
- Empty image: treat as no-op (return without modifying).
- Invalid parameters: throw `InvalidArgumentException`.

## Validation Strategy Summary

| Function type              | Empty input behavior                  | Validation function          |
|---------------------------|---------------------------------------|------------------------------|
| Model creation (Create*)  | throw `InvalidArgumentException`      | `RequireImageNonEmpty*`      |
| Fitting (Fit*, Measure+Fit)| throw `InvalidArgumentException`     | `RequireImageNonEmpty*`      |
| Search (Find*)            | return empty result                   | `RequireImageValid`          |
| Measure (MeasurePos/Pair) | return empty result                   | `RequireImageValid`          |
| Filter/Transform/Convert  | return empty result / empty image     | `RequireImageValid`          |

---

## 行为一致性回归检查清单

> 用于验证各模块是否遵循统一规则。更新日期: 2026-02-03

### 检查规则

| 场景 | 预期行为 | 错误消息格式 |
|------|---------|-------------|
| 空图像输入（搜索/滤波） | 返回空结果 | 不抛异常 |
| 空图像输入（建模/拟合） | 抛 `InvalidArgumentException` | `funcName: image is empty` |
| 无效图像（corrupted） | 抛 `InvalidArgumentException` | `funcName: image is invalid` |
| 类型不匹配 | 抛 `UnsupportedException` | `funcName: expected X image, got Y` |
| 通道数不匹配 | 抛 `UnsupportedException` | `funcName: expected N channel(s), got M` |
| 参数越界 | 抛 `InvalidArgumentException` | `funcName: paramName must be ...` |
| 模型未初始化 | 抛 `InvalidArgumentException` | `funcName: model not initialized` |

### 模块检查状态

| 模块 | 校验函数 | 空图行为 | 异常文案 | 状态 |
|------|---------|---------|---------|------|
| **Matching/ShapeModel** | `RequireImageNonEmpty` | Create抛/Find空返回 | ✅ | ✅ |
| **Matching/NCCModel** | `RequireImageNonEmpty` | Create抛/Find空返回 | ✅ | ✅ |
| **Measure/Caliper** | `RequireImageValid` | 空返回 | ✅ | ✅ |
| **Measure/Metrology** | `RequireImageNonEmpty` | 拟合抛/测量空返回 | ✅ | ✅ |
| **Filter** | `RequireImageValid` | 空返回 | ✅ | ✅ |
| **Morphology** | `RequireImageValid` | 空返回 | ✅ | ✅ |
| **Transform** | `RequireImageValid` | 空返回 | ✅ | ✅ |
| **Color** | `RequireImageValid` | 空返回 | ✅ | ✅ |
| **Edge** | `RequireImageU8Gray` | 空返回 | ✅ | ✅ |
| **Blob** | `RequireImageValid` | 空返回 | ✅ | ✅ |
| **Segment** | `RequireImageValid` | 空返回 | ✅ | ✅ |
| **Hough** | `RequireImageU8Gray` | 空返回 | ✅ | ✅ |
| **Texture** | `RequireImageU8Gray` | 空返回 | ✅ | ✅ |
| **Defect** | `RequireImageNonEmpty` | 训练抛/检测空返回 | ✅ | ✅ |
| **Calib** | `RequireImageNonEmpty` | 标定抛 | ✅ | ✅ |
| **OCR** | `RequireImageU8Channels` | 空返回 | ✅ | ✅ |
| **Barcode** | `RequireImageU8Channels` | 空返回 | ✅ | ✅ |

---

## OCR / Barcode 接口语义审核

> 审核日期: 2026-02-03

### OCR 模块

| 检查项 | 预期 | 实际 | 状态 |
|-------|------|------|------|
| 空图像 | 返回空 `OCRResult` | ✅ 返回空结果 | ✅ |
| 无效图像 | 抛异常 | ✅ `RequireImageValid` 检查 | ✅ |
| 非 UInt8 | 抛 `UnsupportedException` | ✅ 内部转换或抛异常 | ✅ |
| 模型未初始化 | 抛 `InvalidArgumentException` | ✅ "OCR not initialized" | ✅ |
| 模型路径无效 | Init 返回 false | ✅ 返回 false + 友好提示 | ✅ |
| debug 参数 | 打印统计 | ✅ `params.debug=true` | ✅ |

**特殊行为（符合预期）**：
- 检测到文字但识别为空 → 不加入结果（第639行 `if (!tb.text.empty())`）
- 自动 sigmoid 检测 → 兼容不同模型输出

### Barcode 模块

| 检查项 | 预期 | 实际 | 状态 |
|-------|------|------|------|
| 空图像 | 返回空 vector | ✅ 返回 `{}` | ✅ |
| 无效图像 | 抛异常 | ✅ `RequireImageValid` 检查 | ✅ |
| 非 UInt8 | 抛 `UnsupportedException` | ✅ 类型检查 | ✅ |
| formats=None | 返回空（无搜索） | ✅ 提前返回 | ✅ |
| maxResults<=0 | 使用默认值 | ✅ 内部处理 | ✅ |

**ZXing 封装一致性**：
- 参数映射完整 (tryHarder, tryRotate, tryInvert, etc.)
- 结果格式统一 (`BarcodeResult` 结构)
- 预设模式 (`Default`, `QR`, `DataMatrix`, `Linear`, `Robust`)

### 待改进项（低优先级）

1. **OCR**: 考虑添加 `Robust()` 预设（类似 Barcode）
2. **Barcode**: 考虑添加可选预处理（对比度拉伸）
3. **两者**: 考虑统一 `IsAvailable()` / `GetVersion()` 接口风格
