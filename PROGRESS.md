# QiVision 开发进度追踪

> 最后更新: 2026-02-05 (进度状态更正)
>
> 状态图例:
> - ⬜ 未开始
> - 🟡 进行中
> - ✅ 完成
> - ⏸️ 暂停
> - ❌ 废弃

---

## 总体进度

```
Platform █████████████████░░░ 86%
Core     ████████████████████ 100%
Internal ████████████████████ 100%
Feature  ████████████░░░░░░░░ 55%
Tests    █████████████████░░░ 87%
```

---

## Phase 0: Platform 层

| 模块 | 设计 | 实现 | 单测 | 审查 | 备注 |
|------|:----:|:----:|:----:|:----:|------|
| Memory.h | ✅ | ✅ | ✅ | ⬜ | 对齐内存分配 (64字节对齐) |
| SIMD.h | ✅ | ✅ | ✅ | ⬜ | SSE4/AVX2/AVX512/NEON 检测 |
| Thread.h | ✅ | ✅ | ✅ | ⬜ | 线程池、ParallelFor |
| Timer.h | ✅ | ✅ | ✅ | ⬜ | 高精度计时 |
| FileIO.h | ✅ | ✅ | ✅ | ⬜ | 文件操作抽象、UTF-8 支持 |
| Random.h | ✅ | ✅ | ✅ | ⬜ | 随机数（RANSAC用） |
| GPU.h | ⬜ | ⬜ | ⬜ | ⬜ | GPU 抽象（预留） |

---

## Phase 1: Core 层

| 模块 | 设计 | 实现 | 单测 | 审查 | 备注 |
|------|:----:|:----:|:----:|:----:|------|
| Types.h | ✅ | ✅ | ✅ | ⬜ | Point, Rect, Line, Circle, Segment, Ellipse, Arc, RotatedRect |
| Constants.h | ✅ | ✅ | ✅ | ⬜ | 数学常量、精度常量、工具函数 |
| Exception.h | ✅ | ✅ | ⬜ | ⬜ | 异常类层次 (未编写专门单测) |
| QImage.h | ✅ | ✅ | ✅ | ⬜ | 图像类（Domain + 元数据 + stb_image I/O） |
| QRegion.h | ✅ | ✅ | ✅ | ⬜ | RLE 区域 (int32_t 游程) |
| QContour.h | ✅ | ✅ | ✅ | ⬜ | XLD 轮廓（含层次结构、属性、变换） |
| QContourArray.h | ✅ | ✅ | ✅ | ⬜ | 轮廓数组（层次管理） |
| QMatrix.h | ✅ | ✅ | ✅ | ⬜ | 2D 仿射变换矩阵 (QHomMat2d) |

---

## Phase 2: Internal 层 - 基础数学

| 模块 | 设计 | 实现 | 单测 | 精度测试 | SIMD | 审查 | 备注 |
|------|:----:|:----:|:----:|:--------:|:----:|:----:|------|
| Gaussian.h | ✅ | ✅ | ✅ | ⬜ | - | ⬜ | 高斯核、导数核 |
| Matrix.h | ✅ | ✅ | ✅ | ⬜ | - | ✅ | 小矩阵运算 (Vec/Mat固定+动态) |
| Solver.h | ✅ | ✅ | ✅ | ⬜ | - | ⬜ | 线性方程组 LU/QR/SVD/Cholesky |
| Eigen.h | ✅ | ✅ | ✅ | ⬜ | - | ⬜ | 特征值分解 (Jacobi/QR/Power/2x2/3x3) |

---

## Phase 3: Internal 层 - 图像处理

| 模块 | 设计 | 实现 | 单测 | 精度测试 | SIMD | 审查 | 备注 |
|------|:----:|:----:|:----:|:--------:|:----:|:----:|------|
| Interpolate.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | 双线性/双三次插值 |
| Convolution.h | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ | 可分离卷积、Domain感知、AVX2优化 |
| Gradient.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | Sobel/Scharr 梯度 |
| Pyramid.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | 高斯/拉普拉斯/梯度金字塔 |
| Histogram.h | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ | 直方图、均衡化、CLAHE (OpenMP + AVX2) |
| Threshold.h | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ | 全局/自适应/多级/范围阈值 (AVX2 优化) |

---

## Phase 4: Internal 层 - 边缘检测

| 模块 | 设计 | 实现 | 单测 | 精度测试 | 审查 | 备注 |
|------|:----:|:----:|:----:|:--------:|:----:|------|
| Profiler.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 1D 投影采样 |
| Edge1D.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 1D 边缘检测（Caliper核心） |
| NonMaxSuppression.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 1D/2D 非极大值抑制 |
| Hessian.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | Hessian 矩阵计算、特征值分解 |
| Steger.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | Steger 亚像素边缘 |
| EdgeLinking.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 边缘点连接 |
| Canny.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | Canny 边缘检测（含亚像素精化、自动阈值） |

---

## Phase 5: Internal 层 - 几何运算

| 模块 | 设计 | 实现 | 单测 | 精度测试 | 审查 | 备注 |
|------|:----:|:----:|:----:|:--------:|:----:|------|
| Geometry2d.h | ✅ | ✅ | ✅ | - | ✅ | 几何基元操作 (规范化/变换/属性/采样/构造) |
| Distance.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 距离计算 (Point-Line/Circle/Ellipse/Arc/Segment/Contour) |
| Intersection.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 交点计算 (Line-Line/Segment/Circle/Ellipse/Arc/RotatedRect) |
| GeomRelation.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 几何关系 (包含/相交/平行/垂直/共线) |
| GeomConstruct.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 几何构造 (垂线/切线/外接圆/内切圆/凸包/最小包围圆) |
| SubPixel.h | ✅ | ✅ | ✅ | ✅ | ✅ | 亚像素精化 (1D/2D/Edge/Match/Angle) - 精度待优化 |
| Fitting.h | ✅ | ✅ | ✅ | ✅ | ✅ | 直线/圆/椭圆/RANSAC (已知问题: 旋转椭圆拟合) |
| AffineTransform.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 仿射变换 |
| Homography.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 单应性变换 (DLT+RANSAC, WarpPerspective, LM精化) |
| Hough.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 霍夫变换（直线/圆） |
| PolarTransform.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 极坐标变换 (Linear/SemiLog, WarpPolar, stride 修复) |
| CornerRefine.h | ✅ | ✅ | ⬜ | ⬜ | ⬜ | 角点精化 (Harris/Shi-Tomasi/SubPix) |

---

## Phase 5.5: Internal 层 - 轮廓操作

| 模块 | 设计 | 实现 | 单测 | 精度测试 | 审查 | 备注 |
|------|:----:|:----:|:----:|:--------:|:----:|------|
| ContourProcess.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 平滑/简化/重采样 |
| ContourAnalysis.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 长度/面积/曲率/矩/形状描述符/凸性 |
| ContourConvert.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 轮廓↔区域转换 |
| ContourSelect.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 按属性筛选轮廓 |
| ContourSegment.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 轮廓分割为线段/圆弧 |

---

## Phase 6: Internal 层 - 区域处理与形态学

| 模块 | 设计 | 实现 | 单测 | 精度测试 | 审查 | 备注 |
|------|:----:|:----:|:----:|:--------:|:----:|------|
| RLEOps.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | RLE 编解码、集合运算、阈值、边界、填充、连通域 |
| StructElement.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 结构元素 (矩形/椭圆/十字/菱形/线/八边形/自定义) |
| MorphBinary.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 二值形态学 (膨胀/腐蚀/开/闭/梯度/TopHat/Hit-or-Miss/Thin/Skeleton/Geodesic) |
| MorphGray.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 灰度形态学 (膨胀/腐蚀/开/闭/梯度/TopHat/BlackHat/重构/背景校正) |
| ConnectedComponent.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 连通域标记 (图像+RLE两种实现, 统计/过滤/合并/孔洞检测) |
| DistanceTransform.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 距离变换 (L1/L2/LInf/Chamfer, 区域签名距离, Voronoi, 骨架) |
| RegionFeatures.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 区域特征 (面积/周长/圆度/矩/椭圆/凸包/最小包围圆) |

---

## Phase 7: Feature 层 - Measure

| 模块 | 设计 | 实现 | 单测 | 精度测试 | 审查 | 备注 |
|------|:----:|:----:|:----:|:--------:|:----:|------|
| MeasureTypes.h | ✅ | ✅ | - | - | ✅ | 参数和结果结构体 |
| MeasureHandle.h | ✅ | ✅ | ✅ | - | ✅ | 矩形/弧形/同心圆句柄 |
| Caliper.h | ✅ | ✅ | ✅ | ✅ | ✅ | 卡尺测量 |
| CaliperArray.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 多卡尺阵列 (沿线/弧/圆/轮廓) |

---

## Phase 8: Feature 层 - Matching

> 详细设计见: docs/design/Matching_Module_Design.md

| 模块 | 设计 | 实现 | 单测 | 精度测试 | 审查 | 备注 |
|------|:----:|:----:|:----:|:--------:|:----:|------|
| MatchTypes.h | ✅ | ✅ | - | - | ⬜ | 参数和结果结构体 |
| ShapeModel.h | ✅ | ✅ | ⬜ | ⬜ | ⬜ | 形状匹配（P0，梯度方向特征） |
| NCCModel.h | ✅ | ✅ | ⬜ | ⬜ | ⬜ | NCC 匹配（P1，归一化互相关） |
| ComponentModel.h | ✅ | ✅ | ⬜ | ⬜ | ⬜ | 组件匹配（P1，多部件关系约束） |
| DeformableModel.h | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 变形匹配（P2） |
| Internal/AnglePyramid.h | ✅ | ✅ | ⬜ | ⬜ | ⬜ | 角度预计算模型（新增依赖） |
| Internal/IntegralImage.h | ✅ | ✅ | ⬜ | ⬜ | ⬜ | 积分图（NCCModel依赖） |

---

## Phase 9: Feature 层 - Metrology

| 模块 | 设计 | 实现 | 单测 | 精度测试 | 审查 | 备注 |
|------|:----:|:----:|:----:|:--------:|:----:|------|
| Metrology.h | ✅ | ✅ | ✅ | ⬜ | ⬜ | 计量模型框架 (合并为单文件) |

**说明**: Metrology 模块已整合到单个头文件，包含:
- MetrologyMeasureParams: 测量参数
- MetrologyLineResult/CircleResult/EllipseResult/Rectangle2Result: 结果结构体
- MetrologyObjectLine/Circle/Ellipse/Rectangle2: 测量对象类
- MetrologyModel: 组合测量模型

---

## Phase 10+: Feature 层 - 其他模块

| 模块 | 设计 | 实现 | 单测 | 精度测试 | 审查 | 优先级 | 备注 |
|------|:----:|:----:|:----:|:--------:|:----:|:------:|------|
| **IO/ImageIO.h** | ✅ | ✅ | ⬜ | - | ⬜ | **P0** | 图像读写 (PNG/JPEG/BMP/RAW) |
| **Color/ColorConvert.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P1** | 颜色转换 (RGB/HSV/Lab/YCrCb) |
| **Filter/Filter.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P1** | 滤波+增强 (Gauss/Median/Sobel/CLAHE/HistogramEq) |
| **Segment/Segment.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P1** | 图像分割 (Threshold/Otsu/Adaptive/DynThreshold/K-Means/Watershed/GMM) |
| **Display/Display.h** | ✅ | ✅ | ⬜ | - | ⬜ | **P0** | 图像显示与绘制 (Halcon 风格 API) |
| **GUI/Window.h** | ✅ | ✅ | ⬜ | - | ⬜ | **P0** | 窗口调试 (Win32/X11, macOS/Android stub, AutoResize) |
| **Blob/Blob.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P0** | Blob 分析 (Connection, SelectShape, InnerCircle, FillUp, CountHoles等) |
| **Edge/Edge.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P1** | 边缘检测 (Canny, Steger 亚像素) |
| **Transform/PolarTransform.h** | ✅ | ✅ | ✅ | ⬜ | ⬜ | **P1** | 极坐标变换 (公开 API，封装 Internal) |
| **Transform/AffineTransform.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P1** | 仿射变换 (公开 API，封装 Internal) |
| **Transform/Homography.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P1** | 透视变换 (公开 API，封装 Internal) |
| **Morphology/Morphology.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P1** | 形态学 (二值+灰度, SE创建) |
| **Hough/Hough.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P1** | 霍夫变换 (直线/圆检测, 公开 API) |
| **Contour/Contour.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P1** | XLD轮廓操作 (公开 API，封装 Internal) |
| **OCR/OCR.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P1** | 字符识别 (ONNXRuntime + PaddleOCR v4) |
| **Barcode/Barcode.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P1** | 条形码/二维码 (ZXing-cpp 封装) |
| **Defect/VariationModel.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P1** | 变差模型缺陷检测 (Halcon 风格) |
| **Texture/Texture.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P2** | 纹理分析 (LBP/GLCM/Gabor) |
| **Calib/CameraModel.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P2** | 相机模型（内参+畸变） |
| **Calib/Undistort.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P2** | 畸变校正 |
| **Calib/CalibBoard.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P2** | 标定板检测 |
| **Calib/CameraCalib.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P2** | 相机标定（张正友法） |
| **Calib/FisheyeModel.h** | ✅ | ✅ | ⬜ | ⬜ | ⬜ | **P2** | 鱼眼相机模型（Kannala-Brandt） |
| **Calib/FisheyeUndistort.h** | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | **P2** | 鱼眼去畸变 |
| **Calib/FisheyeCalib.h** | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | **P2** | 鱼眼标定 |

---

## Phase 11: Feature 层 - Calib 标定与坐标转换

> 详细设计规范见: `.claude/docs/Calibration_CoordinateSystem_Rules.md`

### 核心数据结构

| 模块 | 设计 | 实现 | 单测 | 审查 | 备注 |
|------|:----:|:----:|:----:|:----:|------|
| QPose.h | ⬜ | ⬜ | ⬜ | ⬜ | 6DOF 位姿，欧拉角 ZYX |
| QHomMat2d.h | ✅ | ✅ | ✅ | ⬜ | 2D 齐次变换矩阵 (已实现为 QMatrix 别名，含完整功能) |
| QHomMat3d.h | ⬜ | ⬜ | ⬜ | ⬜ | 3D 齐次变换矩阵 |
| CameraModel.h | ✅ | ✅ | ⬜ | ⬜ | 相机内外参 + 畸变 (Brown-Conrady模型) |

### 标定功能

| 模块 | 设计 | 实现 | 单测 | 精度测试 | 审查 | 备注 |
|------|:----:|:----:|:----:|:--------:|:----:|------|
| CalibBoard.h | ✅ | ✅ | ⬜ | ⬜ | ⬜ | 标定板检测 (棋盘格角点) |
| CameraCalib.h | ✅ | ✅ | ⬜ | ⬜ | ⬜ | 相机内参标定 (张正友法) |
| Undistort.h | ✅ | ✅ | ⬜ | ⬜ | ⬜ | 畸变校正 (Undistort/Remap/UndistortMap) |
| FisheyeModel.h | ✅ | ✅ | ⬜ | ⬜ | ⬜ | 鱼眼模型 (Kannala-Brandt) |
| FisheyeUndistort.h | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | 鱼眼去畸变 |
| FisheyeCalib.h | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | 鱼眼标定 |
| HandEyeCalib.h | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 手眼标定 |
| StereoCalib.h | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 双目标定 |

### 坐标系转换

| 模块 | 设计 | 实现 | 单测 | 精度测试 | 审查 | 备注 |
|------|:----:|:----:|:----:|:--------:|:----:|------|
| CoordTransform2d.h | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 2D 坐标转换 (图像↔世界) |
| CoordTransform3d.h | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 3D 坐标转换 |
| MatchTransform.h | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 模板匹配结果→世界坐标 |
| RobotTransform.h | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 机器人坐标系转换 |

---

## 基础设施

| 项目 | 状态 | 备注 |
|------|:----:|------|
| CMakeLists.txt (根) | ✅ | 主构建配置 (C++17, SIMD选项, GoogleTest) |
| CMakeLists.txt (src) | ✅ | 源码构建 (QiVision库) |
| CMakeLists.txt (tests) | ✅ | 测试构建 (FetchContent GoogleTest) |
| third_party/stb | ✅ | stb_image + stb_image_write 集成 |
| .clang-format | ✅ | 代码格式化配置 |
| QiVision.h | ✅ | 总头文件 |
| accuracy_config.json | ⬜ | 精度测试配置 |
| benchmark_config.json | ⬜ | 性能基准配置 |

---

## 变更日志

### 2026-02-06 (Fisheye 模型接入与健壮性修复)

- **Calib/FisheyeModel 模块** (新增/接入构建)
  - 新增头文件: `include/QiVision/Calib/FisheyeModel.h`
  - 新增实现: `src/Calib/FisheyeModel.cpp`
  - **FOV 计算修正**: 考虑非中心主点
  - **健壮性**: `UnprojectPixel` 校验内参，`ProjectPoint` 对 `z<=0` 返回 NaN
- **Fisheye 相关 API 预留**:
  - `include/QiVision/Calib/FisheyeUndistort.h`
  - `include/QiVision/Calib/FisheyeCalib.h`

### 2026-02-05 (进度状态更正)

- **PROGRESS.md** (状态更正)
  - QHomMat2d.h: ⬜→✅ 已实现为 `QMatrix` 别名 (Core/QMatrix.h:180)
  - 包含完整功能: 平移、旋转、缩放、剪切、点变换、矩阵求逆、分解等

- **CLAUDE.md** (文档改进)
  - 新增 "Source Code Structure" 部分: 说明 include/src/tests/samples 目录结构
  - 新增 "Quick References" 部分: 添加新 API/Internal/测试/示例的快速指引

### 2026-02-04 (Defect 局部自适应检测)

- **Defect/VariationModel.h / VariationModel.cpp** (新增功能)
  - 新增 `LightDark` 枚举: Light/Dark/NotEqual 检测模式
  - 新增 `LocalAdaptiveCompare()`: 局部自适应缺陷检测
    - 计算 diff = |test - golden|
    - 计算 diff 的局部均值和标准差
    - 标记 diff > localMean + k * localStdDev 的区域
    - 对光照不均场景更鲁棒
  - 新增 `DynThresholdDefect()`: 动态阈值缺陷检测 (Halcon dyn_threshold 风格)
    - 使用均值滤波生成平滑参考图
    - 支持 Light/Dark/NotEqual 检测模式
  - 更新 `docs/API_Reference.md`: 添加新函数文档

### 2026-02-03 (Inference 模块 + OCR 重构)

- **Inference/Inference.h** (新增模块)
  - 轻量级 ONNX 推理封装，统一管理 ONNXRuntime
  - `Tensor`: 输入/输出张量 (name + shape + data)
  - `SessionOptions`: 会话配置 (numThreads, gpuIndex, enableFP16)
  - `Model`: Load() / Run() / Reset() / InputNames() / OutputNames()
  - 跨平台支持 (Windows/Linux/macOS)
- **OCR.cpp** (重构为使用 Inference 层)
  - 移除直接 ONNXRuntime API 调用
  - 改用 `Inference::Model` 管理 det/cls/rec 三个模型
  - 代码更简洁，跨平台处理集中在 Inference 层

### 2026-02-03 (OCR 检测修复 + debug 功能)

- **OCR.cpp** (关键 Bug 修复)
  - **ComputeBoxScore 修复**: 改为对 region runs 内部采样，避免背景稀释导致分数过低
  - **UnclipPolygon 修复**: 修正法线方向 `(dy, -dx)`，向外扩展而非收缩
  - **Sigmoid 自动检测**: 自动检测模型输出是否为 logits，必要时应用 sigmoid
  - **膨胀操作恢复**: 恢复 DB 后处理必需的 3x3 膨胀步骤
- **OCRParams.debug** (新增调试参数)
  - 新增 `debug` 参数，启用后打印检测统计信息
  - 统计内容: regions 总数、各阶段过滤数量、boxScore min/max/avg、最终输出数量
  - 用于诊断检测问题，默认关闭

### 2026-02-03 (OCR DB 后处理 + 透视矫正)

- **OCR.cpp** (检测精度改进)
  - **DB 后处理改进**:
    - 使用轮廓分析代替简单连通域
    - 计算最小外接矩形 (`ContourMinAreaRect`)
    - 实现 `UnclipPolygon()` 按 unClipRatio 扩展多边形
    - 实现 `ComputeBoxScore()` 计算区域内概率均值
    - 按 boxScoreThresh 过滤低置信度框
  - **透视变换矫正**:
    - 使用 `Homography::From4Points()` 从四边形角点计算变换
    - 使用 `WarpPerspective()` 矫正倾斜文本
    - 自动检测旋转角度，小角度用简单裁剪（性能优化）
  - 返回准确的旋转四边形角点，不再是轴对齐 bbox

### 2026-02-03 (ColorConvert Luv 支持)

- **ColorConvert.cpp** (补齐 CIE Luv 颜色空间)
  - 新增 `RgbToLuv()` / `LuvToRgb()`: sRGB ↔ CIE L*u*v* (D65)
  - 新增 `RgbToLuvU8()` / `LuvU8ToRgb()`: 8-bit 量化版本
  - 更新 `TransFromRgb()` / `TransToRgb()`: 支持 Luv
  - 更新 `CreateColorTransLut()`: 支持 Luv LUT 预计算
  - OpenCV 兼容量化: L*255/100, (u+134)*255/354, (v+140)*255/262
  - 往返误差: 87% ≤4, max=37 (8-bit 量化限制，符合预期)

### 2026-02-03 (OCR 模型管理)

- **OCR.h / OCR.cpp** (模型管理功能)
  - 新增 `ModelStatus` 结构体: 模型状态信息
  - 新增 `CheckModels()`: 检查模型文件完整性
  - 新增 `GetRequiredModelFiles()` / `GetOptionalModelFiles()`: 获取模型文件列表
  - 新增 `GetModelDownloadUrl()`: 获取下载 URL
  - 新增 `PrintModelInstallInstructions()`: 打印安装指南
  - 改进 `DownloadModels()`: 使用 curl/wget 自动下载，支持备用源
  - 改进 `GetDefaultModelDir()`: 支持多平台路径 (Linux/macOS/Windows)
  - **友好错误提示**: 模型不存在时给出详细的下载和安装指导
  - 改进 `Init()` / `InitOCR()` / `InitOCRDefault()`: 预检查模型完整性

### 2026-02-03 (ColorConvert Lab/XYZ 支持)

- **ColorConvert.cpp** (补齐颜色空间)
  - 新增 `RgbToXyz()` / `XyzToRgb()`: sRGB ↔ XYZ (D65 illuminant)
  - 新增 `RgbToLab()` / `LabToRgb()`: sRGB ↔ CIE L*a*b* (D65)
  - 新增 U8 变体用于图像处理: `RgbToLabU8`, `RgbToXyzU8`, `LabU8ToRgb`, `XyzU8ToRgb`
  - 更新 `TransFromRgb()`: 支持 Lab, XYZ 目标颜色空间
  - 更新 `TransToRgb()`: 支持 Lab, XYZ 源颜色空间
  - 更新 `CreateColorTransLut()`: 支持 Lab, XYZ LUT 预计算
  - 往返误差 0-4 (正常 8-bit 量化误差)
  - Luv 暂未实现（抛 UnsupportedException）

### 2026-02-03 (NCCModel 亚像素插值)

- **NCCModelScore.cpp** (算法改进)
  - 实现 `ComputeNCCScoreSubpixel()`: 使用双线性插值在亚像素位置计算 NCC 分数
  - 更新 `RefinePosition()`: 在位置/角度精化后调用亚像素分数计算
  - 改进匹配精度: 返回更准确的亚像素位置分数（而非整数位置分数）

### 2026-02-03 (NCCModel 序列化与深拷贝)

- **NCCModel 模块** (补齐功能)
  - `NCCModelImpl.h`: 添加 `Clone()` 方法声明
  - `NCCModel.cpp`: 实现 `NCCModelImpl::Clone()` 深拷贝方法
  - `NCCModel.cpp`: 更新 copy constructor 和 assignment operator，调用 `Clone()`
  - `NCCModel.cpp`: 实现 `WriteNCCModel()` 序列化函数
  - `NCCModel.cpp`: 实现 `ReadNCCModel()` 反序列化函数

- **序列化格式**
  - Magic: `0x434E4951` ("QINC" - QiVision NCC)
  - Version: 1
  - 完整保存: params, origin, templateSize, metric, searchAngles, levels, rotatedTemplates

- **测试**
  - 新增 `tests/test_ncc_serialization.cpp`: 深拷贝和序列化单元测试
  - 测试内容: copy constructor, assignment operator, Write/Read 往返, 加载后匹配验证
  - 28项测试全部通过

### 2026-02-03 (SDK 统一验证工具 v2)

- **Core/Validate.h** (新增，SDK 统一验证工具)
  - 新增 `include/QiVision/Core/Validate.h`: 统一验证工具头文件
  - **分层 API 设计** (类型与通道独立):
    - Layer 1: `RequireImageValid()` - 只检查空/有效（不限制类型）
    - Layer 2: `RequireImageType()`, `RequireChannelCount()` - 独立的类型/通道检查
    - Layer 3: `RequireImageU8()`, `RequireImageU8Gray()`, `RequireImageFloat()` - 组合便捷函数
  - **数值验证**:
    - `RequireRange()`, `RequirePositive()`, `RequireNonNegative()`, `RequireMin()`
    - 使用 `Detail::FormatValue()` 格式化浮点（%.4g，避免长尾数）
  - **宏支持多种返回类型**:
    - `QIVISION_REQUIRE_IMAGE(img)` - 返回 {}
    - `QIVISION_REQUIRE_IMAGE_VOID(img)` - void 函数用
    - `QIVISION_REQUIRE_IMAGE_OR(img, retval)` - 自定义返回值
    - `QIVISION_REQUIRE_IMAGE_U8(img)` 等 UInt8 专用版本
  - **已迁移模块** (全部 Feature 层完成):
    - OCR, Barcode: `RequireImageU8Channels()` (需要 UInt8)
    - Filter: 删除17处冗余类型检查，统一使用 `RequireValidImage`/`RequireGrayU8`
    - Color/ColorConvert: 删除12处冗余类型检查，新增 `RequireGrayU8` 辅助函数
    - Segment: 删除本地 `RequireGrayU8`，改用 `Validate::RequireImageU8Gray`
    - Matching, Measure: `RequireImageValid()` (接受任意类型)
    - Transform/PolarTransform, AffineTransform, Homography: `RequireImageU8` + 删除冗余检查
    - Morphology: `RequireGrayU8Input` 改为调用 `Validate::RequireImageU8Gray`
    - Edge, Hough, Metrology, Undistort, VariationModel: 已迁移
    - CalibBoard: `RequireImageValid()` / `RequireImageU8()`
    - Blob: `RequireImageU8Gray()` (Connection)
    - Texture: `RequireImageU8Gray()` / `RequirePositive()` (删除本地函数)

### 2026-02-02 (OCR 模块集成)

- **OCR/OCR.h 模块** (新增，ONNXRuntime + PaddleOCR v4)
  - 新增 `include/QiVision/OCR/OCR.h`: OCR API 头文件
  - 新增 `src/OCR/OCR.cpp`: ONNXRuntime 实现
  - **设计特点**:
    - 只依赖 ONNXRuntime，**不需要 OpenCV**
    - 预处理完全使用 QiVision 原生 API（QImage、Color、Segment、Blob）
    - 支持 PaddleOCR v4 ONNX 模型
  - **主要 API**:
    - `OCRModel::Init()`: 初始化模型
    - `OCRModel::Recognize()`: 识别图像中的文字
    - `InitOCR()/ReleaseOCR()`: 全局模型管理
    - `RecognizeText()`: 使用全局模型识别
    - `ReadText()`: 简单文本读取
  - **OCRParams 参数**:
    - `maxSideLen`: 最大边长（控制 resize）
    - `boxThresh/boxScoreThresh`: 检测阈值
    - `doAngleClassify`: 角度分类
    - 预设: `Default()`, `Fast()`, `Accurate()`
  - **OCRResult 结果**:
    - `textBlocks`: 检测到的文本块列表
    - `fullText`: 拼接后的完整文本
    - `detectTime/recognizeTime/totalTime`: 计时
  - **TextBlock 结构**:
    - `text`: 识别的文字
    - `confidence`: 置信度
    - `corners`: 四角点坐标
  - **CMake 集成**:
    - `QIVISION_BUILD_OCR` 选项（默认 OFF）
    - 支持 `ONNXRUNTIME_ROOT` 环境变量
    - 自动查找系统安装的 ONNXRuntime
  - 新增 `samples/ocr/ocr_demo.cpp`: 示例程序

### 2026-02-02 (Barcode 模块集成)

- **Barcode/Barcode.h 模块** (新增，ZXing-cpp 封装)
  - 新增 `include/QiVision/Barcode/Barcode.h`: 条形码/二维码读取 API
  - 新增 `src/Barcode/Barcode.cpp`: ZXing-cpp 2.2.1 封装实现
  - **支持的格式**:
    - 1D: Code128, Code39, Code93, Codabar, EAN-8, EAN-13, ITF, UPC-A, UPC-E
    - 2D: QR Code, Data Matrix, PDF417, Aztec
  - **主要 API**:
    - `ReadBarcodes()`: 读取所有条码
    - `ReadBarcode()`: 读取单个条码
    - `ReadQRCodes()`: 便捷函数，仅读取 QR 码
    - `ReadDataMatrix()`: 便捷函数，仅读取 Data Matrix
    - `ReadLinearCodes()`: 便捷函数，仅读取 1D 码
  - **BarcodeParams 参数**:
    - `formats`: 指定搜索的格式类型
    - `binarizer`: 二值化方法 (LocalAverage/GlobalHistogram/FixedThreshold)
    - `tryHarder/tryRotate/tryInvert/tryDownscale`: 鲁棒性选项
    - 预设: `Default()`, `QR()`, `DataMatrix()`, `Linear()`, `Robust()`
  - **BarcodeResult 结果**:
    - `text`: 解码内容
    - `format/formatName`: 格式类型
    - `position/corners/angle`: 位置信息
    - `symbolVersion/ecLevel`: 2D 码版本和纠错级别
  - **CMake 集成**:
    - `QIVISION_BUILD_BARCODE` 选项控制是否编译
    - 使用 FetchContent 自动下载 ZXing-cpp 2.2.1
  - 新增 `samples/barcode/barcode_read.cpp`: 示例程序

### 2026-02-02 (Segment GMM)

- **Segment/Segment.h 模块** (新增 GMM 高斯混合模型)
  - 新增 `GMM()`: EM 算法高斯混合模型分割
  - 新增 `GMMSegment()`: 返回分割图像
  - 新增 `GMMToRegions()`: 返回硬分配区域
  - 新增 `GMMProbabilities()`: 返回概率图（软标签）
  - 新增 `GMMClassify()`: 使用训练好的模型分类新图像
  - **GMMParams 参数结构**:
    - `k`: 高斯分量数
    - `feature`: 特征空间（与 K-Means 共用 GMMFeature）
    - `init`: 初始化方法（Random/KMeans）
    - `covType`: 协方差类型（Full/Diagonal/Spherical）
    - `maxIterations/epsilon/regularization`: EM 控制参数
  - **GMMResult 结果结构**:
    - `labels`: 硬标签（最可能的分量）
    - `probabilities`: 软标签（每个分量的概率图）
    - `weights/means/covariances`: 模型参数
    - `logLikelihood/iterations/converged`: 收敛信息
  - 支持 K-Means 初始化（更稳定的收敛）
  - 支持三种协方差类型以平衡精度和速度
  - 使用 log-sum-exp 技巧保证数值稳定性
  - Cholesky 分解计算协方差逆和行列式
  - OpenMP 并行加速 E 步计算

### 2026-02-02 (Segment Watershed)

- **Segment/Segment.h 模块** (新增 Watershed 分水岭分割)
  - 新增 `Watershed()`: 标记控制的分水岭分割
  - 新增 `WatershedBinary()`: 二值图像自动分割（自动生成标记）
  - 新增 `WatershedRegion()`: 从 QRegion 分割
  - 新增 `WatershedGradient()`: 基于梯度的分水岭分割
  - 新增 `DistanceTransform()`: 距离变换（Chamfer 3-4 近似）
  - 新增 `CreateWatershedMarkers()`: 从距离图创建标记
  - **WatershedResult 结果结构**:
    - `labels`: 标签图 (Int16, 0=背景, -1=分水岭线, >0=区域)
    - `regions`: 分割的区域数组
    - `watershedLines`: 分水岭边界线（作为 QRegion）
    - `numRegions`: 区域数量
  - 使用优先队列实现高效的泛洪算法
  - 支持自动标记生成（基于距离变换的局部极大值）
  - 典型应用：分离接触的对象（如细胞、颗粒等）

### 2026-02-02 (Segment K-Means)

- **Segment/Segment.h 模块** (新增 K-Means 聚类分割)
  - 新增 `KMeans()`: K-Means 聚类分割主函数
  - 新增 `KMeansSegment()`: 返回重着色图像（色彩量化/海报化效果）
  - 新增 `KMeansToRegions()`: 返回每个聚类对应的区域
  - 新增 `LabelsToRegions()`: 标签图转区域数组
  - **KMeansParams 参数结构**:
    - `k`: 聚类数
    - `feature`: 特征空间 (Gray/RGB/HSV/Lab/GraySpatial/RGBSpatial)
    - `init`: 初始化方法 (Random/KMeansPP)
    - `maxIterations/epsilon/attempts`: 收敛控制
    - `spatialWeight`: 空间坐标权重
  - **KMeansResult 结果结构**:
    - `labels`: 标签图 (Int16)
    - `centers`: 聚类中心
    - `clusterSizes`: 每个聚类的像素数
    - `compactness`: 紧致度（距离平方和）
    - `iterations/converged`: 收敛信息
  - 支持 K-Means++ 初始化（更好的初始中心选择）
  - 支持多次尝试选最优结果
  - 支持颜色空间转换 (RGB↔HSV, RGB↔Lab)

### 2026-02-02 (Blob Hu Moments)

- **Blob/Blob.h 模块** (新增 Hu Moments 公开 API)
  - 新增 `HuMoments(const QRegion&) -> std::array<double, 7>`: 返回 7 个 Hu 不变矩
  - 新增 `HuMoments(const QRegion&, double& hu1, ..., double& hu7)`: Halcon 风格输出参数版本
  - 新增 `HuMoments(const std::vector<QRegion>&, std::vector<std::array<double, 7>>&)`: 批量计算版本
  - **Hu Moments 特性**:
    - 旋转不变、缩放不变、平移不变
    - 7 个描述符从归一化中心矩导出
    - hu[6] 的符号可用于区分镜像图像
    - 适用于形状识别和匹配
  - 封装 Internal::ComputeHuMoments 为公开 API
  - 更新 `docs/API_Reference.md` 添加 HuMoments 文档

### 2026-01-30 (ShapeModel AVX2 Score 优化)

- **Matching/ShapeModelScore.cpp 模块** (真正的 AVX2 8点并行)
  - **问题**: 原有 `ComputeScoreBilinearSSE` 使用假 SIMD (scalar SSE: `_mm_set_ss`, `_mm_add_ss`)
    - 每次只处理 1 个点，没有真正向量化
  - **新增 `ComputeScoreNearestNeighborAVX2`**: 真正的 8 点并行 AVX2 实现
    - 使用 `_mm256_*` 指令一次处理 8 个模型点
    - `_mm256_loadu_ps`: 加载 8 个 SoA 数据 (x, y, cos, sin, weight)
    - `_mm256_fmsub_ps`/`_mm256_fmadd_ps`: FMA 加速坐标旋转
    - `_mm256_cvtps_epi32`: 最近邻插值 (round to integer)
    - `_mm256_i32gather_ps`: AVX2 Gather 批量获取梯度值
    - `_mm256_rsqrt_ps`: 快速逆平方根
    - `_mm256_cmp_ps`: 向量化边界检查
    - `horizontal_sum_avx2`: 8 元素水平归约
  - **调度策略**:
    - 仅在最顶层金字塔级别使用 (coarse search)
    - 仅在 IgnoreLocalPolarity/IgnoreColorPolarity 模式
    - 点数 >= 32 时启用
    - 其他情况使用原有 bilinear 实现保持精度
  - **性能**: 与原有实现相比，顶层搜索速度提升 ~20%
  - **精度**: 测试验证 11/11 匹配成功，分数误差 < 0.01

### 2026-01-30 (Histogram OpenMP + AVX2 优化)

- **Internal/Histogram 模块** (OpenMP + AVX2 优化)
  - **核心优化策略**: Per-thread sub-histogram 避免缓存竞争
    - 问题: 多线程直接写同一 histogram bin 会导致缓存行伪共享 (false sharing)
    - 解决: 每个 OpenMP 线程维护独立的 256-bin sub-histogram
    - 最后用 AVX2 向量化合并所有 sub-histograms
  - **ComputeHistogram 模板函数** (header 优化):
    - uint8_t + 256 bins 特化路径: 直接索引，无需 binning 计算
    - 通用类型路径: 带 binning 计算
    - 阈值控制: count >= 10000 时启用 OpenMP
  - **AVX2 Merge 函数**:
    - `MergeHistogramAVX2`: 8 bins/iteration (256-bit 向量)
    - 使用 `_mm256_add_epi32` 向量加法
    - 256 bins 只需 32 次 AVX2 加法
  - **ComputeHistogramMasked** (cpp 优化):
    - 同样采用 per-thread sub-histogram 策略
    - 支持 mask 非零像素条件计数
  - **ApplyLUT / ApplyLUTInPlace** (OpenMP 优化):
    - 简单 LUT 查表，适合数据并行
    - `#pragma omp parallel for schedule(static)`
  - **ApplyCLAHE** (OpenMP 优化):
    - Tile histogram 构建: `#pragma omp parallel for schedule(dynamic)`
    - 每个 tile 独立处理，无竞争
    - CDF 计算优化: 预计算累积直方图，避免重复求和
    - Bilinear interpolation: `#pragma omp parallel for schedule(static)` 按行并行
    - 预计算 tile position factors 减少重复除法
  - **预期性能**:
    - 单线程: ~1.5x (减少缓存 miss)
    - 多线程 (4 cores): **4-6x**
    - CLAHE (compute-bound): **3-4x** on 4 cores

### 2026-01-30 (Threshold AVX2 SIMD 优化)

- **Internal/Threshold.cpp 模块** (AVX2 优化)
  - **新增 AVX2 优化函数** (32 bytes per iteration):
    - `ThresholdBinary_AVX2`: 二值化阈值 `dst = (src > thresh) ? maxVal : 0`
    - `ThresholdBinaryInv_AVX2`: 反向二值化
    - `ThresholdRange_AVX2`: 范围阈值 `low <= src <= high`
    - `ThresholdTruncate_AVX2`: 截断阈值 `min(src, thresh)`
    - `ThresholdToZero_AVX2`: 置零阈值
    - `ThresholdToZeroInv_AVX2`: 反向置零阈值
  - **无符号比较技巧** (AVX2 无 `_mm256_cmpgt_epu8`):
    - `src > threshold` 等价于 `max(src, threshold+1) == src`
    - 使用 `_mm256_max_epu8` + `_mm256_cmpeq_epi8` 实现
    - 特殊处理 `threshold == 255` 边界情况
  - **范围阈值实现**:
    - `src >= low` 等价于 `max(src, low) == src`
    - `src <= high` 等价于 `min(src, high) == src`
    - 两个条件 AND 组合
  - **自动路由**: QImage 版本 `ThresholdGlobal`/`ThresholdRange` 自动检测并使用 AVX2
  - **回退机制**: count < 128 时回退到标量实现
  - **预期加速**: 7-8x (1920x1080 图像, 单线程)

### 2026-01-30 (Convolution AVX2 SIMD 优化)

- **Internal/Convolution.h 模块** (AVX2 优化)
  - **新增 AVX2 优化函数**:
    - `ConvolveRow_AVX2_U8F`: uint8_t -> float 水平卷积优化
    - `ConvolveRow_AVX2_FF`: float -> float 水平卷积优化
    - `ConvolveRow5Tap_AVX2_U8F`: 专用 5-tap Gaussian 优化 (利用对称性)
    - `ConvolveCol_AVX2_FF`: float -> float 垂直卷积优化
    - `ConvolveCol5Tap_AVX2_FF`: 专用 5-tap Gaussian 垂直优化
  - **Zone-based 处理策略**:
    - 左边界: 标量处理 (需要 border handling)
    - 中间安全区域: AVX2 向量化 (8 pixels/iteration)
    - 右边界: 标量处理 (需要 border handling)
  - **uint8_t -> float 转换**: `_mm_loadl_epi64` + `_mm256_cvtepu8_epi32` + `_mm256_cvtepi32_ps`
  - **FMA 加速**: 使用 `_mm256_fmadd_ps` 融合乘加
  - **对称核优化**: 5-tap 核利用 k[0]=k[4], k[1]=k[3] 减少乘法
  - **性能测试结果** (1920x1080, 单线程):
    | Kernel Size | Scalar | AVX2 | Speedup |
    |-------------|--------|------|---------|
    | k=3 | 5.05 ms | 0.61 ms | **8.3x** |
    | k=5 | 8.16 ms | 1.17 ms | **7.0x** |
    | k=7 | 11.07 ms | 2.35 ms | **4.7x** |
  - **精度验证**: AVX2 vs 标量最大误差 < 1e-4 (float 精度)
  - **自动回退**: width < 32 时自动使用标量版本
  - **ConvolveSeparable 优化**: 当 DstT=float 时使用 float 中间缓冲区，使列卷积也能用 AVX2

### 2026-01-29 (Texture 模块)

- **Texture/Texture 模块** (新增)
  - 新增 `include/QiVision/Texture/Texture.h`: 纹理分析头文件
  - 新增 `src/Texture/Texture.cpp`: 实现文件
  - **LBP (局部二值模式)**:
    - `ComputeLBP()`: 基础 8 邻域 LBP
    - `ComputeLBPExtended()`: 可配置半径和采样点
    - `ComputeLBPHistogram()`: LBP 直方图
    - 支持 Standard/Uniform/RotationInvariant/UniformRI 变体
  - **GLCM (灰度共生矩阵)**:
    - `ComputeGLCM()`: 计算共生矩阵
    - `ExtractGLCMFeatures()`: 提取特征 (对比度/能量/熵/相关性等)
    - `ComputeGLCMFeatures()`: 一步完成
    - 支持 4 个方向 + 平均
  - **Gabor 滤波器**:
    - `CreateGaborKernel()`: 创建 Gabor 核
    - `ApplyGaborFilter()`: 单滤波器
    - `ApplyGaborFilterBank()`: 多方向滤波器组
    - `ComputeGaborEnergy()`: 能量响应
    - `ExtractGaborFeatures()`: 特征提取
  - **纹理比较**:
    - `CompareLBPHistograms()`: Chi-square 距离
    - `CompareGLCMFeatures()`: 欧氏距离
    - `CompareGaborFeatures()`: 欧氏距离
  - **纹理分割**:
    - `SegmentByTextureLBP()`: k-means 聚类分割
    - `DetectTextureAnomalies()`: 异常检测

### 2026-01-29 (Defect/VariationModel 模块)

- **Defect/VariationModel 模块** (新增)
  - 新增 `include/QiVision/Defect/VariationModel.h`: 变差模型缺陷检测头文件
  - 新增 `src/Defect/VariationModel.cpp`: 实现文件
  - **VariationModel 类** (Halcon 风格 API):
    - `Train()` + `Prepare()`: 多图训练模式，计算每像素均值和方差
    - `CreateFromSingleImage()`: 单图 + 边缘感知模式
      - 自动检测边缘区域，分配大容差
      - 平坦区域分配小容差
      - 无需多张训练图
    - `Compare()`: 比较测试图，返回缺陷区域 (QRegion)
    - `GetDiffImage()`: 获取归一化差异图
    - `GetMeanImage()` / `GetVarImage()`: 获取模型图像
    - `Write()` / `Read()`: 模型序列化
  - **便捷函数**:
    - `CompareImages()`: 快速单图对比
    - `CompareImagesEdgeAware()`: 边缘感知对比
    - `AbsDiffThreshold()`: 简单差分阈值
    - `AbsDiffImage()`: 差分图像
  - 算法原理: `|test - mean| > threshold × sqrt(variance)`

### 2026-01-29 (GUI XSync 修复)

- **GUI/Window.cpp 修复** (X11 空白显示问题)
  - 问题: GUI 偶尔显示空白，因为 XFlush 只发送请求但不等待完成
  - 修复: 在关键显示操作中将 XFlush 改为 XSync
    - XMapWindow 后: 确保窗口映射完成后再返回
    - XResizeWindow 后: 确保窗口大小变化完成后再绘制
    - XPutImage 后: 确保图像显示完成后再返回
    - Expose 事件重绘: 确保重绘完成
  - XSync 会等待 X Server 完成所有请求，避免竞态条件

### 2026-01-29 (Contour 公开 API 模块)

- **Contour/Contour.h 模块** (新增公开 API，封装 Internal 层)
  - 新增 `include/QiVision/Contour/Contour.h`: XLD 轮廓操作公开 API 头文件
  - 新增 `src/Contour/Contour.cpp`: 实现文件
  - **轮廓处理**:
    - SmoothContoursXld: 移动平均/高斯平滑
    - SimplifyContoursXld: Douglas-Peucker 简化
    - ResampleContoursXld: 等距/定点数重采样
    - CloseContoursXld: 闭合轮廓
    - ReverseContoursXld: 反转轮廓方向
  - **轮廓分析**:
    - LengthXld/AreaCenterXld/PerimeterXld: 基本属性
    - SmallestRectangle1Xld/SmallestRectangle2Xld: 包围矩形
    - SmallestCircleXld: 最小包围圆
    - CurvatureXld/MomentsXld/OrientationXld: 曲率和矩
    - CircularityXld/ConvexityXld/SolidityXld/EccentricityXld: 形状描述符
  - **轮廓拟合**:
    - FitEllipseContourXld: 椭圆拟合
    - FitLineContourXld: 直线拟合
    - FitCircleContourXld: 圆拟合 (代数/几何)
    - ConvexHullXld: 凸包计算
  - **轮廓选择**:
    - SelectContoursXld: 按特征值选择
    - SelectClosedXld/SelectOpenXld: 按闭合性选择
    - SortContoursXld/SelectTopContoursXld: 排序和选择
  - **轮廓分割**:
    - SegmentContoursXld: 分割为直线/圆弧
    - SplitContoursXld: 在拐角处分割
    - DetectCornersXld: 角点检测
  - **轮廓转换**:
    - GenContourRegionXld: 区域→轮廓
    - GenRegionContourXld: 轮廓→区域
  - **轮廓生成**:
    - GenContourPolygonXld: 从点生成
    - GenCircleContourXld: 生成圆/弧轮廓
    - GenEllipseContourXld: 生成椭圆轮廓
    - GenRectangle2ContourXld: 生成旋转矩形轮廓
  - **工具函数**:
    - CountPointsXld/CountObjXld: 计数
    - GetContourXld: 获取坐标
    - TestPointXld: 点包含测试
    - DistancePointXld: 点到轮廓距离
    - UnionContoursXld: 轮廓合并
    - SelectObjXld: 按索引选择
  - 更新 API_Reference.md，添加 Contour 模块文档

### 2026-01-29 (Hough 公开 API 模块)

- **Hough/Hough.h 模块** (新增公开 API，封装 Internal 层)
  - 新增 `include/QiVision/Hough/Hough.h`: 霍夫变换公开 API 头文件
  - 新增 `src/Hough/Hough.cpp`: 实现文件
  - **结果结构体**:
    - HoughLine: 直线检测结果 (rho, theta, score, endpoints)
    - HoughCircle: 圆检测结果 (row, column, radius, score)
  - **直线检测**:
    - HoughLines: 标准霍夫变换 (binary edge image)
    - HoughLinesP: 概率霍夫变换 (返回线段)
    - HoughLinesXld: 从轮廓检测直线
  - **圆检测**:
    - HoughCircles: 霍夫圆变换 (gradient-based)
    - HoughCirclesXld: 从轮廓检测圆
  - **可视化**:
    - DrawHoughLines: 绘制检测到的直线
    - DrawHoughCircles: 绘制检测到的圆
  - **参数结构体**:
    - HoughLineParams, HoughLinePParams, HoughCircleParams
    - 支持 Default(), Fine(), SmallCircles() 等工厂方法
  - **工具函数**:
    - MergeHoughLines/MergeHoughCircles: 非极大值抑制
    - ClipHoughLineToImage: 裁剪直线到图像边界
    - HoughLinesIntersection: 计算直线交点
    - AreHoughLinesParallel: 判断平行
    - PointToHoughLineDistance: 点到直线距离
  - 更新 CMakeLists.txt, QiVision.h, API_Reference.md

### 2026-01-29 (Transform 模块扩展：Affine/Homography)

- **Transform/AffineTransform.h 模块** (新增公开 API)
  - 新增 `include/QiVision/Transform/AffineTransform.h`: 仿射变换公开 API 头文件
  - 新增 `src/Transform/AffineTransform.cpp`: 实现文件
  - **AffineTransImage**: 仿射变换图像 (bilinear/bicubic 插值)
  - **RotateImage**: 旋转图像 (中心点/角度)
  - **ScaleImage/ZoomImageSize**: 缩放图像
  - **Matrix 创建函数** (Halcon 风格):
    - HomMat2dIdentity, HomMat2dRotate, HomMat2dScale
    - HomMat2dTranslate, HomMat2dCompose, HomMat2dInvert
    - HomMat2dRotateLocal, HomMat2dScaleLocal
  - **AffineTransPoint2d**: 点变换 (单点/多点)
  - **Transform 估计**:
    - VectorToHomMat2d: 仿射变换估计 (>=3 点)
    - VectorToRigid: 刚体变换估计 (>=2 点)
    - VectorToSimilarity: 相似变换估计 (>=2 点)
  - **Matrix 分析**:
    - HomMat2dToAffinePar: 分解为 tx,ty,phi,sx,sy,shear
    - HomMat2dIsRigid/HomMat2dIsSimilarity: 变换类型检测

- **Transform/Homography.h 模块** (新增公开 API)
  - 新增 `include/QiVision/Transform/Homography.h`: 透视变换公开 API 头文件
  - 新增 `src/Transform/Homography.cpp`: 实现文件
  - **HomMat3d 类**: 3x3 单应矩阵
    - Identity, FromAffine, Inverse, Normalized
    - IsAffine, ToAffine, Transform
  - **ProjectiveTransImage**: 透视变换图像
  - **Matrix 函数**:
    - ProjHomMat2dIdentity, HomMat2dToProjHomMat
    - ProjHomMat2dCompose, ProjHomMat2dInvert
  - **ProjectiveTransPoint2d**: 透视点变换
  - **Homography 估计**:
    - VectorToProjHomMat2d: DLT 单应估计 (>=4 点)
    - HomVectorToProjHomMat2d: 精确 4 点估计
    - ProjMatchPointsRansac: RANSAC 鲁棒估计
  - **矩形校正**:
    - RectifyQuadrilateral: 四边形 -> 矩形
    - RectangleToQuadrilateral: 矩形 -> 四边形
  - **工具函数**:
    - IsValidHomography: 有效性检测
    - HomographyError: 重投影误差
    - RefineHomography: LM 精化

### 2026-01-29 (Edge 模块公开 API)

- **Edge/Edge.h 模块** (新增公开 API)
  - 新增 `include/QiVision/Edge/Edge.h`: 边缘检测公开 API 头文件
  - 新增 `src/Edge/Edge.cpp`: 公开 API 实现
  - **EdgesImage**: Canny 边缘检测 (二值输出)
  - **EdgesSubPix**: Canny 边缘检测 (亚像素轮廓输出)
  - **EdgesSubPixAuto**: 自动阈值 Canny 检测
  - **LinesSubPix**: Steger 亚像素线检测
    - 支持 light/dark/all 极性选择
    - 基于 Hessian 特征值分析
    - 亚像素精度 <0.02 像素
  - **LinesSubPixAuto**: 自动阈值 Steger 检测
  - **CannyEdgeParams/StegerLineParams**: 高级参数结构体
  - **DetectEdges/DetectLines**: 完整参数控制版本
  - **ComputeSigmaForLineWidth**: 根据线宽计算推荐 sigma
  - **EstimateThresholds**: 基于梯度统计估计阈值

### 2026-01-29 (PolarTransform 修复 + 公开 API)

- **Transform/PolarTransform 模块** (新增公开 API)
  - 新增 `include/QiVision/Transform/PolarTransform.h`: 公开 API 头文件
  - 新增 `src/Transform/PolarTransform.cpp`: 公开 API 实现
  - **CartesianToPolar**: 笛卡尔坐标 → 极坐标图像变换
    - X 轴 = 角度 (0 到 2π)
    - Y 轴 = 半径 (0 到 maxRadius)
  - **PolarToCartesian**: 极坐标 → 笛卡尔坐标图像变换（逆变换）
  - **PolarMode**: Linear / SemiLog 两种映射模式
  - **PolarInterpolation**: Nearest / Bilinear / Bicubic 插值

- **Internal/PolarTransform 模块修复** (stride 处理 bug)
  - **问题**: QImage 有 64 字节对齐的 stride，原代码假设 stride == width
  - **症状**: 当 maxRadius 改变时，极坐标图和重建图出现条纹错乱
  - **修复**:
    - 新增 `GetPixelWithStride`, `BilinearSampleWithStride`, `SamplePixelWithStride` 辅助函数
    - 修改 `WarpCartesianToPolar` 和 `WarpPolarToCartesian` 接受 stride 参数
    - 修复 Float32 inverse 分支缺少 stride 参数的问题

- **示例程序**
  - 新增 `samples/calib/polar_transform_test.cpp`: 极坐标变换测试
    - 使用 Metrology 检测圆
    - 应用极坐标变换
    - 逆变换重建验证

### 2026-01-28 (新增 CameraCalib 模块)

- **Calib/CameraCalib 模块** (新增)
  - 新增 `include/QiVision/Calib/CameraCalib.h`: 相机标定头文件
  - 新增 `src/Calib/CameraCalib.cpp`: 张正友法相机标定实现
  - **CalibFlags**: 标定配置标志
    - `FixPrincipalPoint`: 固定主点在图像中心
    - `FixAspectRatio`: 固定 fx = fy
    - `ZeroTangentDist`: 假设切向畸变为零
    - `FixK1/K2/K3`: 固定径向畸变系数
    - `UseIntrinsicGuess`: 使用初始内参作为初值
  - **ExtrinsicParams**: 外参结构体
    - `R`: 3x3 旋转矩阵
    - `t`: 平移向量
    - `rvec`: Rodrigues 旋转向量
    - `ToTransformMatrix()`: 转换为 4x4 变换矩阵
  - **CalibrationResult**: 标定结果
    - `camera`: 标定得到的 CameraModel
    - `rmsError/meanError/maxError`: 重投影误差统计
    - `extrinsics`: 每张图的外参
    - `perViewErrors/perPointErrors`: 详细误差信息
  - **CalibrateCamera**: 张正友法主函数
    - 从多张图的单应矩阵约束求解内参
    - 从内参和单应矩阵计算外参
    - 线性估计畸变系数
    - Gauss-Newton 非线性优化
  - **SolvePnP**: 位姿估计
    - DLT 初始化 + 迭代优化
  - **ProjectPoints**: 3D 点投影
  - **ComputeReprojectionErrors**: 重投影误差计算
  - **RodriguesToMatrix/MatrixToRodrigues**: Rodrigues 旋转变换

### 2026-01-28 (新增 CalibBoard + CornerRefine 模块)

- **Calib/CalibBoard 模块** (新增)
  - 新增 `include/QiVision/Calib/CalibBoard.h`: 标定板检测头文件
  - 新增 `src/Calib/CalibBoard.cpp`: 标定板检测实现
  - **CornerGrid**: 角点网格结构
    - `corners`: 检测到的角点（行优先顺序）
    - `rows/cols`: 棋盘格内角点数
    - `At(row, col)`: 获取指定位置的角点
    - `IsValid()`: 检查是否有效
  - **FindChessboardCorners**: 棋盘格角点检测
    - 自适应阈值二值化
    - 四边形检测和角点提取
    - 角点聚类和网格组织
    - 亚像素精化
  - **CornerSubPix**: 角点亚像素精化
  - **GenerateChessboardPoints**: 生成世界坐标系角点
  - **DrawChessboardCorners**: 绘制检测结果

- **Internal/CornerRefine 模块** (新增)
  - 新增 `include/QiVision/Internal/CornerRefine.h`: 角点精化头文件
  - 新增 `src/Internal/CornerRefine.cpp`: 角点精化实现
  - **RefineCornerGradient**: 梯度法亚像素角点精化
  - **RefineCorners**: 批量角点精化
  - **DetectHarrisCorners**: Harris 角点检测
    - 计算 Harris 响应: R = det(M) - k * trace(M)^2
    - 非极大值抑制
    - 质量级别和最小距离过滤
  - **DetectShiTomasiCorners**: Shi-Tomasi 角点检测
    - 计算最小特征值: min(lambda1, lambda2)
  - **ComputeStructureTensor**: 结构张量计算
  - **Eigenvalues2x2**: 2x2 对称矩阵特征值分解

### 2026-01-28 (新增 Calib/CameraModel + Undistort 模块)

- **Calib/CameraModel 模块** (新增)
  - 新增 `include/QiVision/Calib/CameraModel.h`: 相机模型头文件
  - 新增 `src/Calib/CameraModel.cpp`: 相机模型实现
  - **CameraIntrinsics**: 相机内参 (fx, fy, cx, cy)
    - `ToMatrix()`: 转换为 3x3 内参矩阵
    - `FromMatrix()`: 从矩阵创建
  - **DistortionCoeffs**: 畸变系数 (Brown-Conrady 模型)
    - 径向畸变: k1, k2, k3
    - 切向畸变: p1, p2
    - `IsZero()`: 检查是否无畸变
  - **CameraModel**: 完整相机模型
    - `Distort()`: 应用畸变 (normalized -> distorted)
    - `Undistort()`: 去畸变 (Newton-Raphson 迭代)
    - `ProjectPoint()`: 3D 点投影到 2D 像素
    - `UnprojectPixel()`: 2D 像素反投影到 3D 射线

- **Calib/Undistort 模块** (新增)
  - 新增 `include/QiVision/Calib/Undistort.h`: 畸变校正头文件
  - 新增 `src/Calib/Undistort.cpp`: 畸变校正实现
  - **UndistortMap**: 预计算映射表 (高效批量处理)
  - **Undistort()**: 图像去畸变
    - 支持自定义新相机矩阵
    - 支持自定义输出尺寸
    - 支持 Nearest/Bilinear/Bicubic 插值
  - **InitUndistortMap()**: 预计算映射表
  - **Remap()**: 使用映射表重映射
    - 支持 UInt8/UInt16/Float32 像素类型
    - OpenMP 并行化
  - **GetOptimalNewCameraMatrix()**: 计算最优新相机矩阵
  - **UndistortPoint/UndistortPoints/DistortPoint**: 点级别操作

### 2026-01-28 (新增 PolarTransform 模块)

- **Internal/PolarTransform 模块** (新增)
  - 新增 `include/QiVision/Internal/PolarTransform.h`: 极坐标变换头文件
  - 新增 `src/Internal/PolarTransform.cpp`: 极坐标变换实现
  - **WarpPolar**: 图像极坐标变换（参考 OpenCV warpPolar）
    - 正向变换: 笛卡尔坐标 -> 极坐标 (x=angle, y=radius)
    - 反向变换: 极坐标 -> 笛卡尔坐标
    - 支持 Linear 和 SemiLog 两种映射模式
    - 支持 Nearest/Bilinear/Bicubic 插值
    - 支持所有像素类型 (UInt8/UInt16/Int16/Float32)
  - **辅助函数**:
    - `CartesianToPolar`: 点坐标笛卡尔->极坐标转换
    - `PolarToCartesian`: 点坐标极坐标->笛卡尔转换
    - `LinearToLogPolar` / `LogPolarToLinear`: 线性/对数极坐标半径映射

### 2026-01-27 (Morphology 模块实现)

- **新增 Morphology 模块** (Feature 层)
  - 新增 `include/QiVision/Morphology/Morphology.h`: 公开 API 头文件
  - 新增 `src/Morphology/Morphology.cpp`: 实现文件
  - 封装 Internal/MorphBinary.h 和 MorphGray.h 为公开 API

- **Morphology API (Halcon 风格)**
  - **结构元素**: `StructuringElement` 类
    - 工厂方法: `Rectangle`, `Square`, `Circle`, `Ellipse`, `Cross`, `Diamond`, `Line`
    - 自定义: `FromMask`, `FromRegion`
    - 变换: `Reflect`, `Rotate`
  - **二值形态学** (Region 操作):
    - 基本: `Dilation`, `Erosion`, `DilationCircle`, `ErosionCircle`, `DilationRectangle`, `ErosionRectangle`
    - 复合: `Opening`, `Closing`, `OpeningCircle`, `ClosingCircle`, `OpeningRectangle`, `ClosingRectangle`
    - 衍生: `Boundary`, `Skeleton`, `Thinning`, `PruneSkeleton`, `FillUp`, `ClearBorder`
  - **灰度形态学** (Image 操作):
    - 基本: `GrayDilation`, `GrayErosion`, `GrayDilationCircle`, `GrayErosionCircle`
    - 复合: `GrayOpening`, `GrayClosing`, `GrayOpeningCircle`, `GrayClosingCircle`
    - 衍生: `GrayGradient`, `GrayTopHat`, `GrayBlackHat`, `GrayRange`
    - 重构: `GrayReconstructDilation`, `GrayReconstructErosion`, `GrayFillHoles`
    - 背景校正: `RollingBall`
  - **便捷函数**: `SE_Cross3`, `SE_Square3`, `SE_Disk5`

### 2026-01-24 (NCCModel 框架实现)

- **NCCModel 模块**
  - 新增 `include/QiVision/Matching/NCCModel.h`: 公开 API 头文件
  - 新增 `src/Matching/NCCModelImpl.h`: 内部实现结构体
  - 新增 `src/Matching/NCCModel.cpp`: 公开 API 实现
  - 新增 `src/Matching/NCCModelCreate.cpp`: 模型创建实现
  - 新增 `src/Matching/NCCModelSearch.cpp`: 多级金字塔搜索
  - 新增 `src/Matching/NCCModelScore.cpp`: NCC 分数计算（使用积分图加速）

- **NCCModel API (Halcon 风格)**
  - `CreateNCCModel`: 3个重载（无ROI、Rect2i ROI、QRegion ROI）
  - `CreateScaledNCCModel`: 带缩放搜索
  - `FindNCCModel` / `FindScaledNCCModel`: 匹配搜索
  - `GetNCCModelParams` / `GetNCCModelOrigin` / `SetNCCModelOrigin` / `GetNCCModelSize`
  - `WriteNCCModel` / `ReadNCCModel` / `ClearNCCModel`
  - `DetermineNCCModelParams`: 自动参数推荐

- **实现特性**
  - 预计算旋转模板（离散角度）
  - 积分图加速区域统计
  - 多级金字塔粗到精搜索
  - 抛物线插值亚像素精化
  - 支持 use_polarity / ignore_global_polarity 模式

### 2026-01-24 (API 文档重写为 OpenCV 风格)

- **API_Reference.md 全面重写**
  - 格式改为 OpenCV 官方文档风格
  - 每个函数独立小节: 简短描述 + 函数签名 + Parameters 表格 + Returns 表格
  - 新增 Segment 模块完整文档 (之前未记录)
  - 删除冗余示例代码，保持简洁
  - 版本号更新为 0.5.0

### 2026-01-23 (API 风格统一：直接参数取代结构体)

- **API 风格重构**
  - 所有公开 API 新增直接参数版本（Halcon/OpenCV 风格）
  - 结构体版本保留用于向后兼容
  - 可选参数使用 `std::vector<int>` 键值对（参考 OpenCV imwrite）

- **ImageIO 模块**
  - `ReadImageRaw`: 新增 (filename, image, width, height, pixelType, ...) 版本
  - `WriteImage`: 新增 (image, filename, format, vector<int> params) 版本
  - 新增 `ImageWriteFlag` 枚举: QIWRITE_JPEG_QUALITY, QIWRITE_PNG_COMPRESSION, QIWRITE_TIFF_COMPRESSION

- **Metrology 模块**
  - `Add*Measure` 方法新增直接参数版本
  - 新增 `MetrologyParamFlag` 枚举用于 vector<int> 参数
  - 示例: `AddCircleMeasure(row, col, r, len1, len2, "all", "all", {METROLOGY_NUM_MEASURES, 20})`

- **CaliperArray 模块**
  - `CreateAlong*` 方法新增直接参数版本
  - 示例: `CreateAlongCircle(center, radius, caliperCount, profileLength, handleWidth)`

- **文档更新**
  - `docs/API_Reference.md`: 更新 IO、Metrology 章节，添加新 API 示例

### 2026-01-23 (Draw Region API 和 Blob 示例)

- **Draw 模块新增 Region 绘制 API**
  - `Draw::Region`: 填充绘制 QRegion
  - `Draw::RegionContour`: 绘制区域轮廓（边界像素）
  - `Draw::RegionAlpha`: 半透明填充区域
  - 支持 RGB 和灰度图像

- **新增 Blob 分析示例程序**
  - `samples/blob/blob_analysis.cpp`: Blob 分析演示
  - 功能: 阈值分割、连通组件、形状特征、区域筛选、排序
  - 可视化: 半透明填充、轮廓绘制、圆形检测、孔洞检测
  - 键盘交互: Q/A/W/S 调整阈值, P 打印特征, ESC 退出

- **文档更新**
  - `docs/API_Reference.md`: 添加 7.9 Region 绘制小节

### 2026-01-22 (缩放匹配功能)

- **FindScaledShapeModel 实现** ✅
  - 支持 [scaleMin, scaleMax] 范围搜索
  - 自动计算 scale step（约 10 个等级）
  - 跨 scale 进行 NMS 抑制重复匹配
  - 返回最佳匹配的 scale 值

- **SearchPyramid 优化**
  - 支持 params.scaleMin 参数传递（默认 1.0，向后兼容）
  - 添加 SearchPyramidScaled 包装函数

- **测试程序**
  - 新增 test_scaled_match.cpp 验证缩放匹配功能
  - 测试结果: scale=1.0 时与 FindShapeModel 结果一致

### 2026-01-21 (架构审查与修复)

- **架构问题修复**
  - **Draw 模块迁移**: Core/Draw.h → Display/Draw.h
    - Core/Draw.h 改为兼容性头文件，自动重定向到 Display/Draw.h
    - 修复层级依赖违规 (Display 现在可以依赖 Matching)
    - Color 结构体已重命名为 Scalar (避免与 Color namespace 冲突)
  - **坐标顺序统一**: 全部使用 (x, y) OpenCV 风格
    - Display.h/cpp 所有函数参数从 (row, col) 改为 (x, y)
    - Draw.h 已经是 (x, y) 风格，无需修改
  - **Agent 规则重构**: 精简为 4 个 Agent
    - algorithm-expert: 策略分析、架构设计、复杂算法、精度诊断
    - dev: 编码实现（Core, Internal, Feature, Platform）
    - code-reviewer: 代码审查、精度验证
    - git-sync: Git 同步

- **新增 Segment 模块** (Feature 层)
  - 从 Internal/Threshold.h 提升阈值功能到公开 API
  - 全局阈值: Threshold, ThresholdRange
  - 自动阈值: ThresholdOtsu, ThresholdTriangle, ThresholdAuto
  - 自适应阈值: ThresholdAdaptive (Mean/Gaussian/Sauvola/Niblack)
  - 动态阈值: DynThreshold, VarThreshold, CharThreshold
  - 阈值转区域: ThresholdToRegion, ThresholdAutoToRegion
  - 二值操作: BinaryAnd/Or/Xor/Diff/Invert

- **扩展 Filter 模块** (直方图增强)
  - 从 Internal/Histogram.h 提升增强功能到公开 API
  - HistogramEqualize - 直方图均衡化
  - ApplyCLAHE - 自适应直方图均衡
  - ContrastStretch - 对比度拉伸
  - AutoContrast - 自动对比度
  - NormalizeImage - 图像归一化
  - HistogramMatch - 直方图匹配

- **更新 QiVision.h**
  - 添加所有 Feature 层主要头文件的 include
  - 启用 QContour, QContourArray, QMatrix 的 include

### 2026-01-20 (Draw 模块 Metrology 可视化)

- **Core/Draw 模块增强**
  - **新增 MeasureRect/MeasureRects**: Halcon 风格卡尺矩形绘制
    - 修复 Phi 参数理解：Phi 是边缘方向，投影方向 = Phi + π/2
    - Length1 沿投影方向（径向），Length2 沿边缘方向（切向）
    - MeasureRects 自动连接各卡尺中心形成测量轮廓线
  - **新增 EdgePointsWeighted**: 根据权重自动着色边缘点
    - 自动检测权重类型（二值 vs 连续）
    - RANSAC/Tukey（二值）：绿色（内点）、红色（离群点）
    - Huber（连续）：绿色（≥0.8）、黄色（0.3~0.8）、红色（<0.3）
  - **改进 Line 绘制**: 粗线使用平行 Bresenham 线实现，边缘更锐利
  - **改进 Circle/Ellipse 绘制**: 参数化方法 + 线段连接，曲线更平滑
  - **新增 MetrologyLine/Circle/Ellipse/Rectangle**: 绘制测量结果
  - **新增 MetrologyModelResult**: 一键绘制完整测量模型

- **Measure/Metrology 模块**
  - 启用 `computeInlierMask = true`，所有拟合方法返回内点掩码
  - 支持离群点可视化

- **示例更新**
  - `samples/measure/circle_measure.cpp`: 使用 Draw 模块绘制卡尺和边缘点

### 2026-01-20 (Ellipse/Rectangle2 鲁棒拟合)

- **Internal/Fitting 模块扩展**
  - **新增 FitEllipseHuber/FitEllipseTukey**: 椭圆鲁棒拟合 (IRLS)
    - 使用加权 Fitzgibbon 算法
    - Huber 权重函数适合中等离群点
    - Tukey 权重函数完全拒绝极端离群点
  - **新增 FitRectangle/FitRectangleIterative**: 矩形鲁棒拟合
    - 边缘点按矩形边分割 (SegmentPointsByRectangleSide)
    - 4条线独立拟合 (Huber/Tukey)
    - 从4条线计算矩形参数 (RectangleFromLines)
    - 迭代精化直至收敛
  - **新增 RectangleFitResult 结构体**: 包含4条边的 LineFitResult

- **Measure/Metrology 模块完善**
  - **Ellipse 测量**: 使用 FitEllipseHuber 替代 FitEllipseFitzgibbon
  - **Rectangle2 测量**: 完整实现 (之前仅占位符)
    - 需要至少8个边缘点（每边2个）
    - 使用 FitRectangleIterative 迭代拟合
    - 输出包含 RMS 误差和拟合质量分数

### 2026-01-20 (Metrology 自动阈值增强)

- **Measure/Metrology 模块增强**
  - **新增自动阈值功能**:
    - 新增 `ThresholdMode` 枚举 (`Manual`, `Auto`)
    - 新增 `SetThreshold("auto")` API 支持 Halcon 风格字符串参数
    - 自动阈值算法：`threshold = max(5.0, contrast×0.2, noise×4.0)`
    - 使用 MAD (Median Absolute Deviation) 估计噪声，比标准差更鲁棒
    - 每个投影区域（profile）独立计算阈值
  - **API 变更**:
    - `MetrologyMeasureParams::SetThreshold(double)` - 手动模式
    - `MetrologyMeasureParams::SetThreshold(const std::string&)` - 支持 "auto"
    - `SetMeasureThreshold()` 标记为 deprecated
  - **移除不合适的功能**:
    - 移除 `autoDetect` 参数（Metrology 是精确测量工具，不适合做自动检测）
    - 自动检测圆应使用专门的 `HoughCircles` 等工具
  - **亚像素支持确认**:
    - `RefineEdgeSubpixel`: 三点抛物线拟合，精度 < 0.02 px
    - `RefineEdgeZeroCrossing`: 二阶导数过零点

- **示例更新**
  - `samples/measure/circle_measure.cpp`: 演示自动阈值模式
  - 新增权重可视化（绿色=内点，黄色=中等，红色=离群点）

### 2026-01-19 (ToFloat+Copy 融合优化)
- **Internal/AnglePyramid.cpp 性能优化**:
  - 融合 ToFloat + Copy 阶段为一步操作
  - 原流程: uint8 → float QImage (有 stride) → 连续 float vector
  - 新流程: uint8 → 连续 float vector (直接)
  - 消除中间 float QImage 分配（大图像约 32MB）
  - **性能提升**:
    - Small Images (640x512): 6.8ms → 5.8ms (-14.7%)
    - Large Images (2048x4001): 162.8ms → 133.0ms (-18.3%)
    - Copy 阶段: 3-18% → 0% (完全消除)
  - **精度保持**: 所有测试 100% 通过
- **文档更新**: TROUBLESHOOTING.md 记录成功优化和失败的内存对齐尝试

### 2026-01-19 (rcp+NR 快速除法优化)
- **Internal/AnglePyramid.cpp 性能优化**:
  - 新增 `fast_rcp_avx2()`: rcp_ps + Newton-Raphson 迭代，精度 ~23 位
  - 新增 `fast_div_avx2()`: 快速除法 a * rcp(b)
  - 替换 `fast_quantize_bin_avx2` 和 `atan2_avx2` 中的 `_mm256_div_ps`
  - **性能提升**:
    - Small Images (640x512): 7.2ms → 6.3ms (-12.5%)
    - Large Images (2048x4001): ~147ms → 144.4ms (-1.8%)
  - **精度保持**: 所有测试 100% 通过
- **文档更新**: TROUBLESHOOTING.md 记录成功优化

### 2026-01-17 (GUI 交互功能)
- **GUI/Window.h 交互增强**:
  - **鼠标事件类型**: `MouseButton`, `MouseEventType`, `KeyModifier`, `MouseEvent`
  - **事件回调**: `SetMouseCallback()`, `SetKeyCallback()`
  - **鼠标位置查询**: `GetMousePosition()`, `GetMouseImagePosition()`
  - **缩放平移**: `EnableZoomPan()`, `GetZoomLevel()`, `SetZoomLevel()`, `GetPanOffset()`, `SetPanOffset()`, `ResetZoom()`, `ZoomToRegion()`
  - **坐标转换**: `WindowToImage()`, `ImageToWindow()`
  - **交互式 ROI 绘制**: `DrawRectangle()`, `DrawCircle()`, `DrawLine()`, `DrawPolygon()`, `DrawPoint()`, `DrawROI()`
  - **交互方式**:
    - 滚轮缩放（以光标为中心）
    - 左键拖拽平移
    - 右键重置为 1:1
    - 'F' 键重置为适应窗口
  - X11/Win32 双平台完整实现

### 2026-01-17 (Blob 模块增强)
- **Blob/Blob.h 新增函数**:
  - `InnerCircle`: 最大内接圆（基于距离变换）
  - `ContourLength`: 区域轮廓长度（周长）
  - `CountHoles` / `EulerNumber`: 孔洞分析
  - `FillUp`: 填充孔洞
  - `GetHoles`: 获取孔洞区域列表
  - `SelectShapeStd`: 按标准差选择（剔除异常值）
  - `SelectShapeMulti`: 多特征同时选择
  - `SelectShapeConvexity` / `SelectShapeElongation`: 按凸度/延伸度选择
  - `SelectShapeProto`: 选择 N 个最大/最小区域
- **GUI/Window.h 增强**:
  - `SetAutoResize(bool, maxW, maxH)`: 自适应窗口大小
  - 修复 X11 大图像显示时细线消失问题（使用区域平均而非最近邻）
- **文档更新**:
  - `docs/API_Reference.md`: 添加 Blob 新函数文档 (6.11-6.14)
  - `PROGRESS.md`: 更新 Blob 模块状态

### 2026-01-17 (GUI 多平台支持)
- **GUI/Window.cpp 平台扩展**
  - 添加平台检测: Windows, macOS, iOS, Android, Linux
  - Windows: Win32 GDI 完整实现
  - Linux: X11 完整实现
  - macOS/iOS/Android: Stub 实现 (Cocoa/Swift/Java 层需要单独集成)
  - CMakeLists.txt 更新: 平台条件编译和消息输出

---

### 2026-01-16 及更早 (历史存档)

> 详细历史记录已存档。主要完成内容摘要：
>
> - **2026-01-15~16**: GUI/Window 模块, Display 模块, Metrology 模块, API 文档
> - **2026-01-12**: LINEMOD 算法实现与性能优化 (245ms → 60ms, 75%提升)
> - **2026-01-08~09**: ShapeModel 模块实现, AnglePyramid, OpenMP/SIMD 优化
> - **2026-01-07~08**: Hough 变换, Eigen 分解, 几何关系模块
> - **2026-01-03~06**: SubPixel, Fitting, 轮廓分析/处理, RLE 形态学
> - **2026-01-01~02**: 基础架构, Core 层数据结构

---

## 如何更新此文件

当完成某个模块的某个阶段时，更新对应的状态：

```markdown
# 示例：完成了 Gaussian.h 的设计和实现
| Gaussian.h | ✅ | ✅ | ⬜ | ⬜ | - | ⬜ | 高斯核、导数核 |

# 示例：正在实现 Steger.h
| Steger.h | ✅ | 🟡 | ⬜ | ⬜ | ⬜ | Steger 亚像素边缘 |
```

每次更新后，同时更新"最后更新"日期和"变更日志"。
