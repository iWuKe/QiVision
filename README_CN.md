# QiVision

[English](README.md)

**QiVision** 是一个从零开始用 C++17 实现的工业机器视觉库，不依赖 OpenCV，目标是匹配 Halcon 的核心功能和精度。

## 特性

- **零依赖** - 仅使用 stb_image 进行文件读写
- **亚像素精度** - 边缘检测精度 < 0.02px
- **Halcon 概念** - Domain、XLD 轮廓、RLE 区域
- **现代 C++17** - 简洁 API，RAII 设计
- **SIMD 优化** - AVX2/SSE 加速

## 状态

| 层级 | 进度 | 描述 |
|------|------|------|
| Core 核心层 | 80% | QImage, QRegion, QContour, QMatrix |
| Platform 平台层 | 70% | Memory, SIMD, Thread, Timer |
| Internal 内部层 | 45% | Gaussian, Gradient, Edge, Fitting, Contour |
| Feature 功能层 | 0% | ShapeModel, Caliper, Blob, OCR (计划中) |

## 快速开始

### 环境要求

- C++17 编译器 (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.16+

### 构建

```bash
git clone https://github.com/userqz1/QiVision.git
cd QiVision
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### 运行测试

```bash
./build/bin/unit_test
```

### 运行示例

```bash
./build/bin/samples/01_basic_image
./build/bin/samples/06_contour_segment
```

## 在项目中使用

**CMake FetchContent:**

```cmake
include(FetchContent)
FetchContent_Declare(QiVision
    GIT_REPOSITORY https://github.com/userqz1/QiVision.git
    GIT_TAG main)
FetchContent_MakeAvailable(QiVision)
target_link_libraries(your_app PRIVATE QiVision)
```

**作为子目录:**

```cmake
add_subdirectory(QiVision)
target_link_libraries(your_app PRIVATE QiVision)
```

## 示例程序

完整示例见 [samples/](samples/) 文件夹:

- `01_basic_image.cpp` - 图像创建、像素访问、保存/加载
- `06_contour_segment.cpp` - 轮廓分割为直线和圆弧

### 基本图像示例

```cpp
#include <QiVision/QiVision.h>
using namespace Qi::Vision;

int main() {
    // 创建图像
    QImage img(640, 480, PixelType::UInt8, ChannelType::Gray);

    // 填充像素
    for (int32_t y = 0; y < img.Height(); ++y) {
        uint8_t* row = static_cast<uint8_t*>(img.RowPtr(y));
        for (int32_t x = 0; x < img.Width(); ++x) {
            row[x] = static_cast<uint8_t>((x + y) % 256);
        }
    }

    // 保存
    img.SaveToFile("output.png");

    // 加载
    QImage loaded = QImage::FromFile("input.png");

    return 0;
}
```

### 轮廓分割示例

```cpp
#include <QiVision/QiVision.h>
#include <QiVision/Internal/ContourSegment.h>
using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

int main() {
    // 创建 L 形轮廓
    QContour contour;
    for (int i = 0; i <= 50; ++i) contour.AddPoint(i, 0);
    for (int i = 1; i <= 50; ++i) contour.AddPoint(50, i);

    // 分割为基元
    SegmentParams params;
    params.mode = SegmentMode::LinesOnly;
    SegmentationResult result = SegmentContour(contour, params);

    printf("找到 %zu 条直线\n", result.LineCount());
    return 0;
}
```

## 架构

```
┌───────────────────────────────────────────────────┐
│ API: QImage, QRegion, QContour, QMatrix           │
├───────────────────────────────────────────────────┤
│ Feature: ShapeModel, Caliper, Blob, OCR (计划中)  │
├───────────────────────────────────────────────────┤
│ Internal: Gaussian, Gradient, Canny, Steger,      │
│           Fitting, ContourProcess, ContourSegment │
├───────────────────────────────────────────────────┤
│ Platform: Memory, SIMD, Thread, Timer, FileIO     │
└───────────────────────────────────────────────────┘
```

## 精度目标

| 模块 | 目标 |
|------|------|
| Edge1D | < 0.02 px |
| CircleFit | < 0.02 px |
| LineFit | < 0.005° |

## 文档

- [PROGRESS.md](PROGRESS.md) - 开发进度
- [samples/](samples/) - 示例程序

## 许可证

MIT License
