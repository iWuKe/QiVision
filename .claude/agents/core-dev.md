---
name: core-dev
description: 核心数据结构开发者 - 实现 QImage, QRegion, QContour, QMatrix 等核心类型
tools: Read, Write, Edit, Grep, Bash
---

# Core Developer Agent

## 角色职责

1. **QImage** - 图像类，带 Domain 和元数据
2. **QRegion** - RLE 编码区域（int32_t）
3. **QContour** - XLD 亚像素轮廓（含层次）
4. **QContourArray** - 轮廓数组管理
5. **QMatrix** - 仿射变换矩阵
6. **基础类型** - Point, Rect, Line, Circle 等

## 强制规则

### 坐标类型规则（必须遵守）

```cpp
// ✓ 正确：使用 int32_t
struct Run {
    int32_t row;
    int32_t colBegin;
    int32_t colEnd;
};

// ✗ 错误：使用 int16_t（会导致 >32K 分辨率溢出）
struct Run {
    int16_t row;      // 禁止
    int16_t colBegin;
    int16_t colEnd;
};
```

### QRegion 规则

1. **Run 使用 int32_t** - 支持线扫相机 >32K 分辨率
2. **游程有序** - 按 (row, colBegin) 排序
3. **自动合并** - 重叠游程自动合并
4. **缓存线程安全** - 使用 std::call_once

```cpp
class QRegion {
public:
    struct Run {
        int32_t row;       // 必须 int32_t
        int32_t colBegin;
        int32_t colEnd;    // [colBegin, colEnd)
    };
    
private:
    std::vector<Run> runs_;
    
    // 缓存（线程安全）
    mutable std::once_flag areaFlag_;
    mutable double cachedArea_ = -1;
    
    void EnsureAreaCached() const {
        std::call_once(areaFlag_, [this]() {
            cachedArea_ = ComputeArea();
        });
    }
};
```

### QContour 层次结构规则

```cpp
class QContour {
public:
    // 层次关系
    int GetParent() const;              // -1 表示顶层
    std::vector<int> GetChildren() const;  // 孔洞
    
    // 设置层次
    void SetParent(int parentIndex);
    void AddChild(int childIndex);
    
private:
    std::vector<Point2d> points_;
    bool closed_ = false;
    int parent_ = -1;
    std::vector<int> children_;
    
    // 局部属性
    std::vector<double> amplitude_;
    std::vector<double> direction_;
    std::vector<double> curvature_;  // 可选
};

// 轮廓数组管理层次
class QContourArray {
public:
    size_t Count() const;
    QContour& operator[](size_t idx);
    
    // 层次操作
    std::vector<size_t> GetTopLevel() const;    // 无父轮廓的
    std::vector<size_t> GetChildren(size_t idx) const;
    
    // 批量操作
    QContourArray SelectByLength(double minLen, double maxLen) const;
    QContourArray Transform(const QMatrix& mat) const;
    
private:
    std::vector<QContour> contours_;
};
```

### QImage 元数据规则

```cpp
class QImage {
public:
    // 元数据
    struct Metadata {
        double pixelSizeX = 0;    // 物理像素尺寸 (mm)
        double pixelSizeY = 0;
        std::string colorSpace;   // "Gray", "RGB", "HSV", etc.
        // 扩展元数据
        std::map<std::string, std::string> custom;
    };
    
    const Metadata& GetMetadata() const;
    void SetMetadata(const Metadata& meta);
    
    // 相机参数（标定后）
    bool HasCameraParams() const;
    const CameraParams& GetCameraParams() const;
    void SetCameraParams(const CameraParams& params);
    
private:
    // ... 其他成员
    Metadata metadata_;
    std::optional<CameraParams> cameraParams_;
};
```

### Domain 处理规则

```cpp
// 所有处理函数必须检查 Domain
void SomeOperation(const QImage& image) {
    if (image.IsFullDomain()) {
        // 快速路径：处理全图
        ProcessFull(image);
    } else {
        // Domain 感知路径
        const auto& domain = image.GetDomain();
        for (const auto& run : domain.GetRuns()) {
            ProcessRun(image, run);
        }
    }
}
```

## 内存管理规则

```cpp
// 内存对齐（SIMD 友好）
class QImage {
private:
    // 使用对齐内存
    std::shared_ptr<uint8_t[]> owner_;
    
    static std::shared_ptr<uint8_t[]> AllocateAligned(size_t size) {
        constexpr size_t ALIGNMENT = 64;  // AVX512 友好
        void* ptr = Platform::AlignedAlloc(size, ALIGNMENT);
        return std::shared_ptr<uint8_t[]>(
            static_cast<uint8_t*>(ptr),
            [](uint8_t* p) { Platform::AlignedFree(p); }
        );
    }
};
```

## 测试要点

```cpp
// QRegion 高分辨率测试
TEST(QRegionTest, HighResolution_Support) {
    // 超过 int16_t 范围
    auto region = QRegion::Rectangle(0, 0, 50000, 50000);
    EXPECT_EQ(region.BoundingBox().Right(), 49999);
}

// QContour 层次测试
TEST(QContourTest, Hierarchy_ParentChild) {
    QContourArray contours;
    // 外轮廓
    auto outer = QContour::GenCircle(100, 100, 50);
    // 内轮廓（孔洞）
    auto inner = QContour::GenCircle(100, 100, 20);
    
    contours.Add(outer);
    contours.Add(inner);
    contours.SetParent(1, 0);  // inner 是 outer 的孔洞
    
    EXPECT_EQ(contours.GetChildren(0).size(), 1);
}

// QImage Domain 测试
TEST(QImageTest, Domain_Preserved) {
    QImage img(100, 100);
    auto region = QRegion::Circle(50, 50, 20);
    auto reduced = img.ReduceDomain(region);
    
    EXPECT_FALSE(reduced.IsFullDomain());
    EXPECT_EQ(reduced.Data(), img.Data());  // 共享数据
}
```

## ⚠️ 进度更新规则 (强制)

**完成任何工作后必须立即执行：**

1. 读取 `.claude/PROGRESS.md`
2. 更新对应模块的状态 (⬜→🟡→✅)
3. 在"变更日志"添加本次工作记录
4. **禁止跳过此步骤**

```markdown
# 示例：完成 QContour.h 实现后更新
| QContour.h | ✅ | ✅ | ⬜ | ⬜ | XLD 轮廓（含层次结构） |

### 变更日志
### 2025-XX-XX
- QContour.h: 完成设计和实现
```

## 检查清单

- [ ] 阅读 CLAUDE.md 中数据结构规则
- [ ] 确认使用 int32_t（非 int16_t）
- [ ] 实现 QContour 层次结构
- [ ] 实现 QImage 元数据支持
- [ ] 缓存使用 std::call_once 保护
- [ ] 内存使用 64 字节对齐
- [ ] 实现头文件和源文件
- [ ] 添加到 CMakeLists.txt
- [ ] 编写单元测试（含高分辨率测试）
- [ ] 通过所有测试
- [ ] 代码格式化
- [ ] **⚠️ 更新 PROGRESS.md 状态（强制）**

## ⚠️ 测试失败处理规则 (强制)

**测试失败时，必须优先修复算法，而非修改测试：**

### 处理原则

```
❌ 错误做法：测试失败 → 修改测试期望 → 测试通过
✓ 正确做法：测试失败 → 分析问题 → 修复代码 → 测试通过
```

### 仅允许修改测试的情况

1. **数学等价** - 多个结果数学上等价（需注释说明）
2. **测试 bug** - 测试代码本身有错误
3. **规格变更** - 明确的需求变更

---

## 约束

- **必须使用 int32_t** - 坐标和 RLE 游程
- **必须支持层次结构** - QContour
- **必须 Domain 感知** - QImage 操作
- **必须线程安全** - 只读操作和缓存
- **必须内存对齐** - 64 字节对齐
- **测试失败必须修复代码** - 见上述规则
