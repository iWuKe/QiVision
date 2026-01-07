# QiVision Samples

示例程序 / Sample Programs

## Build

```bash
cd QiVision
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target samples
```

## Samples

### 01_basic_image

基本图像操作 / Basic Image Operations

- 创建图像 / Create image
- 像素访问 / Pixel access
- 保存/加载 / Save/Load
- 图像克隆 / Clone

```bash
./build/bin/samples/01_basic_image
```

### 06_contour_segment

轮廓分割 / Contour Segmentation

- 创建轮廓 / Create contours
- 角点检测 / Corner detection
- 分割为直线和圆弧 / Segment into lines and arcs
- 不同分割算法 / Different segmentation algorithms

```bash
./build/bin/samples/06_contour_segment
```

## More Samples Coming

更多示例将随着 API 稳定化而添加：

- Gaussian filtering
- Edge detection (Canny, Steger)
- Geometric fitting (line, circle, ellipse)
- Contour analysis
