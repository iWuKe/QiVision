# QiVision 常见问题与解决方案

## 目录

1. [编译问题](#编译问题)
2. [运行时问题](#运行时问题)
3. [精度问题](#精度问题)
4. [性能问题](#性能问题)
5. [跨平台问题](#跨平台问题)
6. [算法问题](#算法问题)
7. [测试问题](#测试问题)

---

## 编译问题

### C001: 找不到头文件 `<QiVision/xxx.h>`

**症状**：
```
fatal error: QiVision/Core/Types.h: No such file or directory
```

**原因**：include 路径未正确设置

**解决**：
```cmake
# CMakeLists.txt 确保添加
target_include_directories(QiVision PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
```

---

### C002: 链接错误 undefined reference

**症状**：
```
undefined reference to `Qi::Vision::QImage::QImage(int, int)'
```

**原因**：
1. 源文件未添加到 CMakeLists.txt
2. 命名空间不匹配

**解决**：
1. 检查 src/CMakeLists.txt 是否包含对应 .cpp 文件
2. 检查 .cpp 文件中的命名空间是否与 .h 一致

---

### C003: SIMD 指令编译错误

**症状**：
```
error: inlining failed: target specific option mismatch
```

**原因**：编译器未启用对应 SIMD 指令集

**解决**：
```cmake
# GCC/Clang
target_compile_options(QiVision PRIVATE
    $<$<BOOL:${COMPILER_SUPPORTS_AVX2}>:-mavx2>
    $<$<BOOL:${COMPILER_SUPPORTS_SSE4}>:-msse4.1>
)

# MSVC
target_compile_options(QiVision PRIVATE
    $<$<BOOL:${COMPILER_SUPPORTS_AVX2}>:/arch:AVX2>
)
```

---

### C004: C++17 特性不支持

**症状**：
```
error: 'std::filesystem' has not been declared
```

**原因**：编译器版本过低或未启用 C++17

**解决**：
```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GCC 8 需要额外链接
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(QiVision PRIVATE stdc++fs)
endif()
```

---

### C005: Windows 下中文路径问题

**症状**：文件路径包含中文时无法打开

**原因**：Windows 默认使用 GBK 编码

**解决**：
```cpp
// 使用 std::filesystem 的 u8path（C++17）
auto path = std::filesystem::u8path(u8"中文路径/图像.png");

// 或设置 UTF-8 代码页（Windows 10 1903+）
// 项目属性 → C/C++ → 命令行 → /utf-8
```

---

## 运行时问题

### R001: 图像加载失败

**症状**：`QImage::FromFile()` 返回空图像

**排查步骤**：
1. 检查文件路径是否正确
2. 检查文件是否存在且可读
3. 检查图像格式是否支持（PNG、JPG、BMP）
4. 检查文件是否损坏

**解决**：
```cpp
QImage img = QImage::FromFile(path);
if (img.Empty()) {
    // 检查错误原因
    auto error = QImage::GetLastError();
    // FileNotFound, InvalidFormat, CorruptedFile, etc.
}
```

---

### R002: 内存不足

**症状**：处理大图像时程序崩溃或抛出 std::bad_alloc

**原因**：图像过大，内存不足

**解决**：
1. 使用金字塔处理降低分辨率
2. 使用分块处理
3. 检查是否有内存泄漏

```cpp
// 分块处理示例
void ProcessLargeImage(const QImage& image) {
    const int tileSize = 2048;
    for (int y = 0; y < image.Height(); y += tileSize) {
        for (int x = 0; x < image.Width(); x += tileSize) {
            auto tile = image.SubImage(x, y, 
                std::min(tileSize, image.Width() - x),
                std::min(tileSize, image.Height() - y));
            ProcessTile(tile);
        }
    }
}
```

---

### R003: Domain 被忽略

**症状**：设置了 Domain 但算法处理了整个图像

**原因**：算法实现未正确处理 Domain

**排查**：
1. 检查算法是否调用 `IsFullDomain()` 分支
2. 检查 Domain 是否正确传递

**正确实现模式**：
```cpp
void SomeAlgorithm(const QImage& image) {
    if (image.IsFullDomain()) {
        ProcessFullImage(image);  // 快速路径
    } else {
        ProcessWithDomain(image, image.GetDomain());  // Domain 感知
    }
}
```

---

### R004: 线程安全问题

**症状**：多线程执行时结果不稳定或崩溃

**原因**：
1. 全局状态被修改
2. 缓存未正确保护
3. 输出缓冲区冲突

**解决**：
```cpp
// 错误：使用 static 变量
void Process() {
    static std::vector<double> buffer;  // ❌ 线程不安全
}

// 正确：线程局部或传入缓冲区
void Process(std::vector<double>& buffer) {  // ✓
}

// 或使用 thread_local
void Process() {
    thread_local std::vector<double> buffer;  // ✓
}
```

---

## 精度问题

### A001: 边缘检测精度不达标

**症状**：Edge1D 或 Caliper 精度超出规格

**排查步骤**：
1. 检查输入图像对比度是否足够（≥30）
2. 检查噪声水平是否在规格内
3. 检查 sigma 参数是否合适
4. 检查边缘类型（阶跃/斜坡）

**调优建议**：
| 问题 | 调整 |
|------|------|
| 噪声大 | 增大 sigma |
| 边缘模糊 | 减小 sigma |
| 弱边缘丢失 | 降低 threshold |
| 假边缘多 | 提高 threshold |

---

### A002: ShapeModel 匹配位置偏差

**症状**：匹配位置与真实位置有系统性偏差

**可能原因**：
1. 模板中心定义不准确
2. 角度搜索步长过大
3. 金字塔层数过多导致信息丢失

**解决**：
1. 确保模板中心与期望中心对齐
2. 减小 angleStep 提高角度精度
3. 减少金字塔层数或降低最小层分辨率要求

---

### A003: 圆/直线拟合精度不足

**症状**：拟合结果偏差大于预期

**排查**：
1. 检查输入点数量是否足够（圆≥6点，直线≥4点）
2. 检查是否有离群点
3. 检查点分布是否均匀

**解决**：
```cpp
// 使用 RANSAC 排除离群点
auto circle = FitCircleRansac(points, 0.5, 100, &inliers);

// 检查内点比例
double inlierRatio = CountTrue(inliers) / points.size();
if (inlierRatio < 0.8) {
    // 警告：可能有较多离群点
}
```

---

### A004: 亚像素精化失败

**症状**：亚像素位置跳跃或不稳定

**原因**：
1. 局部区域信息不足
2. 极值点在边界
3. 曲线拟合失败

**解决**：
```cpp
// 检查亚像素精化是否有效
auto subpixel = RefineSubpixelParabola(y0, y1, y2);
if (std::abs(subpixel) > 0.5) {
    // 精化结果异常，使用整数位置
    subpixel = 0;
}
```

---

## 性能问题

### P001: 处理速度慢

**排查步骤**：
1. 使用 Timer 定位瓶颈
2. 检查是否启用了 Release 模式
3. 检查 SIMD 是否生效

```cpp
Platform::Timer timer;
timer.Start();
// ... 操作 ...
double elapsed = timer.ElapsedMs();
QI_LOG_DEBUG("Performance", "Operation took {} ms", elapsed);
```

---

### P002: SIMD 未生效

**症状**：性能与标量实现相当

**排查**：
```cpp
// 检查 SIMD 支持
if (Platform::HasAVX2()) {
    std::cout << "AVX2 supported" << std::endl;
}
if (Platform::HasSSE4()) {
    std::cout << "SSE4 supported" << std::endl;
}
```

**原因**：
1. CPU 不支持对应指令集
2. 编译时未启用 SIMD
3. 数据未对齐

---

### P003: 内存分配频繁

**症状**：性能分析显示大量时间在 malloc/free

**解决**：
```cpp
// 预分配缓冲区
class Caliper {
    std::vector<double> profileBuffer_;  // 复用缓冲区
    
    void MeasurePos(...) {
        profileBuffer_.resize(numSamples);  // 只在需要时扩展
        // ...
    }
};
```

---

## 跨平台问题

### X001: Windows/Linux 结果不一致

**可能原因**：
1. 浮点精度差异（x87 vs SSE）
2. 随机数生成器差异
3. 文件路径分隔符

**解决**：
```cpp
// 强制使用 SSE 浮点
#ifdef _MSC_VER
    _set_SSE2_enable(1);
#endif

// 使用固定种子的随机数
std::mt19937 rng(12345);

// 使用 std::filesystem 处理路径
namespace fs = std::filesystem;
auto path = fs::path(dir) / "subdir" / "file.png";
```

---

### X002: ARM 平台性能差

**原因**：未实现 NEON 优化

**解决**：
1. 实现 NEON 版本的关键算法
2. 或依赖编译器自动向量化

```cpp
#ifdef __ARM_NEON
void GaussianBlur_NEON(...) {
    // NEON 实现
}
#endif
```

---

## 算法问题

### AL001: ShapeModel 找不到目标

**排查**：
1. minScore 是否设置过高
2. 搜索角度/缩放范围是否覆盖目标
3. 对比度是否足够

**调试方法**：
```cpp
// 降低阈值查看候选
FindParams params;
params.minScore = 0.3;  // 降低阈值
params.maxMatches = 100;  // 增加返回数量

auto results = model.FindModel(image, params);
// 分析 results 的 score 分布
```

---

### AL002: Blob 分析连通域错误

**症状**：相邻区域未分开或意外合并

**原因**：
1. 阈值不合适
2. 连通性设置错误（4连通/8连通）

**解决**：
```cpp
// 使用 4 连通避免对角连接
auto regions = BlobAnalyzer::Connection(binaryImage, Connectivity::Four);
```

---

### AL003: 标定精度不足

**症状**：畸变校正后仍有残余畸变

**排查**：
1. 标定图像数量是否足够（≥10张）
2. 标定板是否覆盖视野各区域
3. 角点检测是否准确

**建议**：
- 标定图像覆盖视野 80% 以上
- 包含不同角度和距离
- 检查每张图像的重投影误差

---

## 测试问题

### T001: 精度测试不稳定

**症状**：相同测试多次运行结果不同

**原因**：
1. 随机数未固定种子
2. 浮点计算顺序不确定
3. 多线程执行顺序

**解决**：
```cpp
// 测试中使用固定种子
TEST(AccuracyTest, EdgeDetection) {
    std::mt19937 rng(42);  // 固定种子
    // ...
}

// 单线程执行精度测试
Platform::SetMaxThreads(1);
```

---

### T002: 测试图像找不到

**症状**：测试失败，提示找不到测试数据

**解决**：
```cmake
# 复制测试数据到构建目录
file(COPY ${CMAKE_SOURCE_DIR}/tests/data 
     DESTINATION ${CMAKE_BINARY_DIR}/tests)

# 或设置工作目录
set_tests_properties(MyTest PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
```

---

### T003: CI 测试失败但本地通过

**排查**：
1. 检查 CI 环境与本地环境差异
2. 检查是否有路径硬编码
3. 检查浮点精度差异

**建议**：
- 使用相对路径
- 精度比较使用 EXPECT_NEAR 而非 EXPECT_EQ
- 在 CI 配置中打印环境信息

---

## 快速诊断命令

```bash
# 检查编译配置
cmake -B build -L

# 检查 SIMD 支持
./build/tests/platform_test --gtest_filter=*SIMD*

# 运行精度测试并输出详细信息
./build/tests/accuracy_test --gtest_output=xml:report.xml

# 性能分析（Linux）
perf record ./build/tests/benchmark_test
perf report

# 内存检查（Linux）
valgrind --leak-check=full ./build/tests/unit_test
```

---

## 获取帮助

如果以上方案无法解决问题：

1. **检查日志** - 启用 DEBUG 级别日志
2. **最小复现** - 构造最小复现用例
3. **记录环境** - OS、编译器版本、CPU 型号
4. **提交 Issue** - 附带上述信息
