# QiVision

[English](README.md)

**QiVision** 是一个从零开始用现代 C++17 实现的工业机器视觉算法库，旨在匹配 Halcon 的核心功能和精度——完全不依赖 OpenCV。

## 特性亮点

- **零 OpenCV 依赖** - 完全从零实现，仅使用 stb_image 进行文件读写
- **工业级精度** - 亚像素精度 (< 0.02px) 的边缘检测和测量
- **Halcon 兼容概念** - Domain、XLD 轮廓、RLE 区域等概念对 Halcon 用户友好
- **现代 C++17** - 简洁的 API，RAII 设计，无需手动内存管理
- **跨平台** - 支持 Windows、Linux、macOS
- **SIMD 优化** - 关键操作使用 AVX2/SSE 加速

## 开发状态

> **开发中** - Core 和 Internal 层已基本可用，Feature 层正在开发中。

| 层级 | 进度 | 模块 |
|------|------|------|
| Core 核心层 | 80% | QImage, QRegion, QContour, QMatrix |
| Platform 平台层 | 70% | Memory, SIMD, Thread, Timer, FileIO |
| Internal 内部层 | 45% | Gaussian, Gradient, Edge, Fitting, Contour 操作 |
| Feature 功能层 | 0% | ShapeModel, Caliper, Blob, OCR (计划中) |

## 快速开始

### 环境要求

- **C++17** 编译器 (GCC 9+, Clang 10+, MSVC 2019+)
- **CMake 3.16+**
- 无外部依赖

### 从源码构建

```bash
# 克隆仓库
git clone https://github.com/userqz1/QiVision.git
cd QiVision

# 配置并构建
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# 运行测试验证安装
./build/bin/unit_test
```

### 安装到系统（可选）

```bash
# 安装到 /usr/local（可能需要 sudo）
cmake --install build --prefix /usr/local
```

### 在你的项目中使用

**方式一：CMake FetchContent（推荐）**

```cmake
include(FetchContent)
FetchContent_Declare(
    QiVision
    GIT_REPOSITORY https://github.com/userqz1/QiVision.git
    GIT_TAG main
)
FetchContent_MakeAvailable(QiVision)

target_link_libraries(your_app PRIVATE QiVision)
```

**方式二：作为子目录添加**

```cmake
add_subdirectory(path/to/QiVision)
target_link_libraries(your_app PRIVATE QiVision)
```

**方式三：查找已安装的包**

```cmake
find_package(QiVision REQUIRED)
target_link_libraries(your_app PRIVATE QiVision::QiVision)
```

## 使用示例

### 基本图像操作

```cpp
#include <QiVision/QiVision.h>

using namespace Qi::Vision;

int main() {
    // 加载图像
    QImage image = QImage::Load("input.png");

    // 获取图像属性
    int width = image.Width();
    int height = image.Height();
    int channels = image.Channels();

    // 访问像素值
    uint8_t pixel = image.At<uint8_t>(100, 100);

    // 创建灰度副本
    QImage gray = image.ToGray();

    // 保存结果
    gray.Save("output.png");

    return 0;
}
```

### 高斯滤波

```cpp
#include <QiVision/QiVision.h>
#include <QiVision/Internal/Gaussian.h>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

int main() {
    QImage image = QImage::Load("noisy.png");

    // 应用 sigma=1.5 的高斯模糊
    GaussianParams params;
    params.sigmaX = 1.5;
    params.sigmaY = 1.5;

    QImage smoothed = GaussianFilter(image, params);
    smoothed.Save("smoothed.png");

    return 0;
}
```

### 边缘检测（Canny）

```cpp
#include <QiVision/QiVision.h>
#include <QiVision/Internal/Canny.h>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

int main() {
    QImage image = QImage::Load("input.png").ToGray();

    // 配置 Canny 边缘检测器
    CannyParams params;
    params.lowThreshold = 50;
    params.highThreshold = 150;
    params.sigma = 1.0;

    // 检测边缘
    QImage edges = CannyEdgeDetector(image, params);
    edges.Save("edges.png");

    return 0;
}
```

### 亚像素边缘检测（Steger）

```cpp
#include <QiVision/QiVision.h>
#include <QiVision/Internal/Steger.h>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

int main() {
    QImage image = QImage::Load("line_image.png").ToGray();

    // 使用 Steger 方法检测亚像素边缘
    StegerParams params;
    params.sigma = 1.5;
    params.lowThreshold = 5.0;
    params.highThreshold = 10.0;

    std::vector<StegerPoint> points = StegerLineDetector(image, params);

    // 每个点都有亚像素坐标
    for (const auto& pt : points) {
        printf("边缘位置 (%.3f, %.3f), 方向: %.2f\n",
               pt.x, pt.y, pt.angle);
    }

    return 0;
}
```

### 轮廓处理

```cpp
#include <QiVision/QiVision.h>
#include <QiVision/Internal/ContourProcess.h>
#include <QiVision/Internal/ContourAnalysis.h>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

int main() {
    // 创建轮廓（例如从边缘检测结果）
    QContour contour;
    contour.AddPoint(0, 0);
    contour.AddPoint(100, 0);
    contour.AddPoint(100, 100);
    contour.AddPoint(0, 100);
    contour.SetClosed(true);

    // 计算轮廓属性
    double length = ComputeContourLength(contour);
    double area = ComputeContourArea(contour);
    Point2d centroid = ComputeContourCentroid(contour);

    printf("长度: %.2f, 面积: %.2f\n", length, area);
    printf("质心: (%.2f, %.2f)\n", centroid.x, centroid.y);

    // 平滑轮廓
    GaussianSmoothParams smoothParams;
    smoothParams.sigma = 2.0;
    QContour smoothed = SmoothContourGaussian(contour, smoothParams);

    // 简化轮廓（Douglas-Peucker 算法）
    QContour simplified = SimplifyContourDP(contour, 1.0);

    return 0;
}
```

### 几何拟合

```cpp
#include <QiVision/QiVision.h>
#include <QiVision/Internal/Fitting.h>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

int main() {
    // 用于直线拟合的点
    std::vector<Point2d> linePoints = {
        {0, 0.1}, {1, 1.0}, {2, 2.1}, {3, 2.9}, {4, 4.0}
    };

    // 拟合直线
    LineFitResult lineResult = FitLine(linePoints);
    if (lineResult.success) {
        printf("直线: y = %.3fx + %.3f (误差: %.4f)\n",
               lineResult.line.Slope(),
               lineResult.line.YIntercept(),
               lineResult.residualRMS);
    }

    // 用于圆拟合的点
    std::vector<Point2d> circlePoints;
    for (int i = 0; i < 36; ++i) {
        double angle = i * 10.0 * M_PI / 180.0;
        circlePoints.push_back({
            50.0 + 30.0 * cos(angle) + (rand() % 100 - 50) * 0.01,
            50.0 + 30.0 * sin(angle) + (rand() % 100 - 50) * 0.01
        });
    }

    // 拟合圆
    CircleFitResult circleResult = FitCircle(circlePoints);
    if (circleResult.success) {
        printf("圆: 圆心(%.2f, %.2f), 半径=%.2f\n",
               circleResult.circle.center.x,
               circleResult.circle.center.y,
               circleResult.circle.radius);
    }

    return 0;
}
```

### 轮廓分割

```cpp
#include <QiVision/QiVision.h>
#include <QiVision/Internal/ContourSegment.h>

using namespace Qi::Vision;
using namespace Qi::Vision::Internal;

int main() {
    // 创建 L 形轮廓
    QContour contour;
    // 水平部分
    for (int i = 0; i <= 50; ++i) contour.AddPoint(i, 0);
    // 垂直部分
    for (int i = 1; i <= 50; ++i) contour.AddPoint(50, i);

    // 分割为几何基元
    SegmentParams params;
    params.mode = SegmentMode::LinesAndArcs;
    params.maxLineError = 1.0;

    SegmentationResult result = SegmentContour(contour, params);

    printf("找到 %zu 条直线, %zu 个圆弧\n",
           result.LineCount(), result.ArcCount());

    // 获取拟合的线段
    for (const auto& primitive : result.primitives) {
        if (primitive.type == PrimitiveType::Line) {
            printf("线段: (%.1f,%.1f) -> (%.1f,%.1f)\n",
                   primitive.segment.p1.x, primitive.segment.p1.y,
                   primitive.segment.p2.x, primitive.segment.p2.y);
        }
    }

    return 0;
}
```

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│  API 层                                                          │
│  QImage (支持 Domain), QRegion (RLE), QContour (XLD), QMatrix    │
├─────────────────────────────────────────────────────────────────┤
│  Feature 功能层 (计划中)                                          │
│  ShapeModel, Caliper, Blob, OCR, Barcode, Calibration            │
├─────────────────────────────────────────────────────────────────┤
│  Internal 内部层 (算法实现)                                       │
│  Gaussian, Gradient, Canny, Steger, Hessian, Fitting,            │
│  ContourProcess, ContourAnalysis, ContourSegment, ...            │
├─────────────────────────────────────────────────────────────────┤
│  Platform 平台层                                                  │
│  Memory (对齐分配), SIMD (AVX2/SSE), Thread, Timer, FileIO       │
└─────────────────────────────────────────────────────────────────┘
```

## 精度规格

在标准条件下（对比度 >= 50，噪声 sigma <= 5）：

| 模块 | 指标 | 目标精度 |
|------|------|----------|
| Edge1D | 位置 | < 0.02 px (1sigma) |
| Caliper | 位置/宽度 | < 0.03 px / < 0.05 px |
| ShapeModel | 位置/角度 | < 0.05 px / < 0.05° |
| CircleFit | 圆心/半径 | < 0.02 px |
| LineFit | 角度 | < 0.005° |

## API 参考

### 核心类

| 类 | 描述 |
|----|------|
| `QImage` | 图像容器，支持 Domain，64 字节行对齐 |
| `QRegion` | RLE 编码的区域（二值掩码） |
| `QContour` | XLD 轮廓，亚像素坐标 |
| `QContourArray` | 轮廓集合，支持层级结构 |
| `QMatrix` | 2D 矩阵，用于数值计算 |

### Internal 模块（当前已实现）

| 模块 | 函数 |
|------|------|
| Gaussian | `GaussianFilter`, `GaussianKernel`, `SeparableGaussian` |
| Gradient | `SobelGradient`, `ScharrGradient`, `ComputeGradientMagnitude` |
| Canny | `CannyEdgeDetector` |
| Steger | `StegerLineDetector`, `StegerEdgeDetector` |
| Fitting | `FitLine`, `FitCircle`, `FitEllipse`, `FitPolynomial` |
| ContourProcess | `SmoothContour`, `SimplifyContour`, `ResampleContour` |
| ContourAnalysis | `ComputeContourLength`, `ComputeContourArea`, `ComputeContourCurvature` |
| ContourSegment | `SegmentContour`, `DetectCorners`, `FitLineToContour` |

## 运行测试

```bash
# 运行所有单元测试
./build/bin/unit_test

# 运行特定测试
./build/bin/unit_test --gtest_filter=*Gaussian*

# 运行精度测试
./build/bin/accuracy_test
```

## 目录结构

```
QiVision/
├── include/QiVision/          # 公共头文件
│   ├── Core/                  # 核心数据结构
│   ├── Platform/              # 平台抽象
│   └── Internal/              # 算法头文件
├── src/                       # 实现文件
│   ├── Core/
│   ├── Platform/
│   └── Internal/
├── tests/                     # 测试套件
│   ├── unit/                  # 单元测试
│   └── accuracy/              # 精度测试
├── third_party/               # 外部依赖 (stb_image)
├── docs/                      # 文档
├── PROGRESS.md                # 开发进度
└── CMakeLists.txt             # 构建配置
```

## 参与贡献

欢迎贡献！请查看 [PROGRESS.md](PROGRESS.md) 了解当前开发状态和计划中的模块。

### 开发指南

- 遵循 [CLAUDE.md](.claude/CLAUDE.md) 中的编码风格
- 所有新功能必须包含单元测试
- 精度关键算法需要精度测试
- 像素坐标使用 `int32_t`（支持大图像）
- 亚像素坐标使用 `double`

## 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE)

## 致谢

- **Halcon** - 工业视觉概念设计灵感
- **stb_image** - 图像文件读写
- **Google Test** - 测试框架

---

## 常见问题

### Q: 为什么不用 OpenCV？

A: QiVision 的目标是提供工业级精度（类似 Halcon），并且完全掌控算法实现。OpenCV 主要面向通用计算机视觉，而工业视觉有更高的精度要求和特定的概念（如 Domain、XLD）。

### Q: 目前可以用于生产环境吗？

A: 目前 Core 和 Internal 层基本可用，但 Feature 层（如模板匹配、卡尺测量）尚未实现。建议等待功能更完善后再用于生产环境。

### Q: 如何报告问题或贡献代码？

A: 请通过 GitHub Issues 报告问题，或提交 Pull Request 贡献代码。

### Q: 支持 GPU 加速吗？

A: 目前不支持 GPU 加速，但 SIMD（AVX2/SSE）优化已经实现。GPU 支持在未来路线图中。
