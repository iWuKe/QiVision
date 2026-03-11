# QiVision Samples

## 构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

可执行文件输出到 `build/bin/samples/`。所有示例按 `q` 键退出 GUI 窗口。

## 示例列表

### Matching - 模板匹配

| 可执行文件 | 源文件 | 说明 |
|-----------|--------|------|
| `matching_test_create_m1` | `matching/test_create_m1.cpp` | CreateShapeModel 建模 + 特征点可视化 |
| `matching_shape_interactive` | `matching/shape_match_interactive.cpp` | 交互式形状匹配：绘制 ROI，在文件夹内批量搜索 |
| `matching_shape_brightness` | `matching/shape_brightness.cpp` | 变光照条件下的形状匹配 (`ignore_global_polarity`) |
| `matching_shape_scaled_match` | `matching/shape_scaled_match.cpp` | 缩放匹配演示 (0.5x ~ 1.5x 尺度范围) |
| `matching_shape_scaled_rings` | `matching/shape_scaled_match_rings.cpp` | 交互式缩放匹配 (真实硬件图像) |
| `matching_shape_mask_search` | `matching/shape_mask_search.cpp` | 带搜索掩膜的 FindShapeModel |
| `matching_shape_multi` | `matching/shape_multi_match.cpp` | 多模型同时匹配：3 个模板在同一图像中搜索 |
| `matching_ncc_match` | `matching/ncc_match.cpp` | NCC 归一化互相关模板匹配 |
| `matching_component_model` | `matching/component_model_demo.cpp` | 分量模型匹配 (可变形/多部件物体) |

### Measure - 测量与计量

| 可执行文件 | 源文件 | 说明 |
|-----------|--------|------|
| `measure_circle_metrology` | `measure/circle_metrology.cpp` | Metrology 高级 API：圆弧 + 矩形边缘测量 |
| `measure_line_metrology` | `measure/line_metrology.cpp` | Metrology 高级 API：直线边缘测量与参数调优 |
| `measure_caliper_pairs` | `measure/caliper_pairs.cpp` | 卡尺边对测量：两条边之间的宽度 |
| `measure_caliper_circle_manual` | `measure/caliper_circle_manual.cpp` | 手动圆周卡尺分布 (低级 Caliper API) |
| `measure_ellipse_fit_synthetic` | `measure/ellipse_fit_synthetic.cpp` | 椭圆拟合：边缘提取 + 轮廓 + 拟合 |

### Blob - 连通分量分析

| 可执行文件 | 源文件 | 说明 |
|-----------|--------|------|
| `blob_analysis` | `blob/blob_analysis.cpp` | 连通分量分析与形状特征提取 (面积/圆度/方向) |

### Color - 颜色检测

| 可执行文件 | 源文件 | 说明 |
|-----------|--------|------|
| `color_detect_demo` | `color/color_detect_demo.cpp` | HSV 颜色空间检测 (预设颜色 + 可调参数) |

### Calib - 标定与变换

| 可执行文件 | 源文件 | 说明 |
|-----------|--------|------|
| `calib_polar_transform` | `calib/polar_transform_test.cpp` | 圆检测 + 极坐标变换 (环形展开) |

### Barcode - 条码识别 (需要 ZXing-cpp)

| 可执行文件 | 源文件 | 说明 |
|-----------|--------|------|
| `barcode_read` | `barcode/barcode_read.cpp` | 1D/2D 条码读取 |

构建时需启用: `cmake -DQIVISION_BUILD_BARCODE=ON`

### OCR - 文字识别 (需要 ONNXRuntime)

| 可执行文件 | 源文件 | 说明 |
|-----------|--------|------|
| `ocr_demo` | `ocr/ocr_demo.cpp` | PaddleOCR 文字识别 (中/英文) |
| `ocr_polar_circular_barcode` | `ocr/polar_ocr_circular_barcode.cpp` | 圆形条码极坐标展开 + OCR 识别 |

构建时需启用: `cmake -DQIVISION_BUILD_OCR=ON`

## 快速开始

```bash
# 形状匹配入门
./build/bin/samples/matching_test_create_m1

# 多模型匹配
./build/bin/samples/matching_shape_multi

# 圆弧测量
./build/bin/samples/measure_circle_metrology

# Blob 分析
./build/bin/samples/blob_analysis
```

## 测试数据

示例使用的测试图像位于 `tests/data/` 目录下。运行示例前请确保在项目根目录执行。
