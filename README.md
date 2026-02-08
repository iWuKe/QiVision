<p align="center">
  <h1 align="center">QiVision</h1>
  <p align="center">
    <strong>工业级机器视觉算法库（C++17）- 亚像素精度</strong>
  </p>
</p>

<p align="center">
    <a href="./README_EN.md">English</a> | 简体中文
</p>

<p align="center">
    <img src="https://img.shields.io/badge/C++-17-blue.svg" alt="C++17">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
    <img src="https://img.shields.io/badge/Platform-Windows%20|%20Linux-lightgrey.svg" alt="Platform">
    <img src="https://img.shields.io/badge/SIMD-AVX2%20|%20SSE4-orange.svg" alt="SIMD">
    <img src="https://img.shields.io/badge/Dependencies-stb__image%20only-brightgreen.svg" alt="Dependencies">
</p>

---

## 项目定位

QiVision 面向工业视觉场景，提供亚像素精度测量与高性能匹配能力。适用于产线定位、缺陷检测、几何测量、条码/OCR 等任务。

---

## 核心能力

- 模板匹配：ShapeModel（梯度形状匹配）、NCCModel（灰度相关），支持旋转/缩放
- 组件匹配：ComponentModel（多部件相对关系约束）
- 测量：卡尺/计量模型（直线、圆、椭圆、矩形）
  - 支持点级诊断查询（`GetPointDetails`）用于平台调试与追溯
- 形态学/分割/Blob：阈值、连通域、形状筛选
- 轮廓与几何：XLD、拟合、变换、Hough
- 标定与畸变：相机模型、畸变校正、鱼眼模型（部分实现）
- OCR/Barcode：可选模块（依赖 ONNXRuntime / ZXing）

---

## 性能与精度（简要）

- 亚像素测量精度：< 0.03 px（典型卡尺测量）
- 形状匹配：支持 0–360°，多层金字塔 + SIMD 优化
- 低依赖：仅 stb_image 负责图像读写

---

## 快速开始

### 构建（Linux）

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### 运行示例

```bash
./build/bin/samples/matching_shape_match
```

---

## 构建与运行配置

```bash
# 构建测试
cmake -B build -DQIVISION_BUILD_TESTS=ON

# 构建 samples
cmake -B build -DQIVISION_BUILD_SAMPLES=ON

# GUI 显示与交互窗口
cmake -B build -DQIVISION_BUILD_GUI=ON

# 可选模块
cmake -B build -DQIVISION_BUILD_OCR=ON -DQIVISION_BUILD_BARCODE=ON
```

运行提示：
- `samples/*` 默认输出到 `build/bin/samples/`
- OCR/Barcode 需要对应依赖库可用（详见各模块文档）

---

## 示例入口

- `samples/matching_shape_match`
- `samples/matching_ncc_match`
- `samples/matching_component_model`
- `samples/measure_circle_metrology`
- `samples/blob_analysis`

---

## 进度与详细文档

- 开发进度：[PROGRESS.md](PROGRESS.md)
- API 参考：[docs/API_Reference.md](docs/API_Reference.md)
- Measure SDK 参数规范：[docs/MEASURE_SDK_PARAM_SPEC.md](docs/MEASURE_SDK_PARAM_SPEC.md)
- 点级结果说明（Metrology）：见 API 文档 `GetPointDetails` 小节
- Troubleshooting：[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- 示例代码：[samples/](samples/)

---

## 许可证

MIT License
