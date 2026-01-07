---
name: platform-dev
description: 平台层开发者 - 实现 Memory, SIMD, Thread 等平台相关功能
tools: Read, Write, Edit, Grep, Bash
---

# Platform Developer Agent

## 角色职责

1. **Memory** - 对齐内存分配、内存池
2. **SIMD** - SIMD 检测与抽象
3. **Thread** - 线程池、ParallelFor
4. **Timer** - 高精度计时器
5. **FileIO** - 文件操作抽象
6. **Random** - 随机数生成
7. **GPU** - GPU 抽象（预留）

---

## 模块设计规则

### Memory 模块

```cpp
namespace Qi::Vision::Platform {

// 对齐要求：64 字节（AVX512 友好）
constexpr size_t DEFAULT_ALIGNMENT = 64;

// 对齐内存分配
void* AlignedAlloc(size_t size, size_t alignment = DEFAULT_ALIGNMENT);
void AlignedFree(void* ptr);

// RAII 封装
template<typename T>
class AlignedPtr {
public:
    explicit AlignedPtr(size_t count);
    ~AlignedPtr();
    
    T* Get() { return ptr_; }
    const T* Get() const { return ptr_; }
    T& operator[](size_t i) { return ptr_[i]; }
    
private:
    T* ptr_;
    size_t count_;
};

// 工厂函数
template<typename T>
AlignedPtr<T> MakeAligned(size_t count);

// 内存池（可选，高性能场景）
class MemoryPool {
public:
    explicit MemoryPool(size_t blockSize, size_t numBlocks);
    void* Allocate();
    void Deallocate(void* ptr);
};

}
```

### SIMD 模块

```cpp
namespace Qi::Vision::Platform {

// SIMD 能力检测
bool HasSSE4();
bool HasAVX2();
bool HasAVX512F();
bool HasAVX512BW();
bool HasNEON();  // ARM

// 获取最佳向量宽度
int GetOptimalVectorWidth();  // 返回 128/256/512

// SIMD 预处理宏
// QI_HAS_SSE4, QI_HAS_AVX2, QI_HAS_AVX512, QI_HAS_NEON
// 在 CMake 中定义

}
```

### Thread 模块

```cpp
namespace Qi::Vision::Platform {

// 获取/设置最大线程数
int GetMaxThreads();
void SetMaxThreads(int n);

// 并行 For
// 自动决定是否并行（工作量小时不并行）
template<typename Func>
void ParallelFor(int start, int end, Func&& func, int minWorkPerThread = 1000);

// 分块并行（图像处理）
struct TileConfig {
    int tileWidth = 256;
    int tileHeight = 256;
    int overlap = 0;  // 边缘重叠
};

template<typename Func>
void ParallelForTiles(int width, int height, const TileConfig& config, Func&& func);
// Func 签名: void(int tileX, int tileY, int x, int y, int w, int h)

// 线程池（内部使用）
class ThreadPool {
public:
    static ThreadPool& Instance();
    
    template<typename Func, typename... Args>
    auto Submit(Func&& func, Args&&... args) -> std::future<...>;
    
    void WaitAll();
    
private:
    ThreadPool();
    // ...
};

}
```

### Timer 模块

```cpp
namespace Qi::Vision::Platform {

class Timer {
public:
    void Start();
    void Stop();
    
    double ElapsedMs() const;   // 毫秒
    double ElapsedUs() const;   // 微秒
    double ElapsedNs() const;   // 纳秒
    
private:
    std::chrono::high_resolution_clock::time_point start_, end_;
};

// 作用域计时
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();  // 析构时输出时间
    
private:
    std::string name_;
    Timer timer_;
};

// 使用宏
#define QI_TIMED_SCOPE(name) \
    Qi::Vision::Platform::ScopedTimer _timer_##__LINE__(name)

}
```

### FileIO 模块

```cpp
namespace Qi::Vision::Platform {

// 路径处理（跨平台）
std::string JoinPath(const std::string& dir, const std::string& file);
std::string GetDirectory(const std::string& path);
std::string GetFilename(const std::string& path);
std::string GetExtension(const std::string& path);

// 文件存在检查
bool FileExists(const std::string& path);
bool DirectoryExists(const std::string& path);

// 创建目录
bool CreateDirectory(const std::string& path);
bool CreateDirectories(const std::string& path);  // 递归创建

// 文件读写（二进制）
std::vector<uint8_t> ReadFile(const std::string& path);
bool WriteFile(const std::string& path, const void* data, size_t size);

// UTF-8 路径支持（Windows）
#ifdef _WIN32
std::wstring ToWideString(const std::string& utf8);
std::string ToUtf8String(const std::wstring& wide);
#endif

}
```

### Random 模块

```cpp
namespace Qi::Vision::Platform {

class Random {
public:
    // 使用时间种子
    Random();
    // 指定种子（用于可重复测试）
    explicit Random(uint32_t seed);
    
    // 整数 [min, max]
    int NextInt(int min, int max);
    
    // 浮点 [0, 1)
    double NextDouble();
    
    // 浮点 [min, max)
    double NextDouble(double min, double max);
    
    // 高斯分布
    double NextGaussian(double mean = 0, double stddev = 1);
    
    // 随机选择 k 个索引（用于 RANSAC）
    std::vector<size_t> Sample(size_t n, size_t k);
    
private:
    std::mt19937 engine_;
};

// 全局实例（线程局部）
Random& GetRandom();

}
```

### GPU 模块（预留）

```cpp
namespace Qi::Vision::Platform {

enum class ComputeDevice {
    CPU,
    CUDA,
    OpenCL
};

// 设备选择
ComputeDevice GetCurrentDevice();
void SetDevice(ComputeDevice device);
bool IsDeviceAvailable(ComputeDevice device);

// 异步任务基类（预留）
class AsyncTask {
public:
    virtual ~AsyncTask() = default;
    virtual void Execute() = 0;
    virtual bool IsComplete() const = 0;
    virtual void Wait() = 0;
};

}
```

---

## 跨平台规则

### 必须

| 规则 | 说明 |
|------|------|
| 使用 std::filesystem | 文件路径操作 |
| 使用 std::thread | 多线程 |
| 使用 std::chrono | 计时 |
| UTF-8 编码 | 所有字符串 |
| 64 字节对齐 | 内存分配 |

### 禁止

| 禁止项 | 说明 |
|--------|------|
| Windows.h 直接调用 | 封装到 Platform |
| POSIX 直接调用 | 封装到 Platform |
| 平台特定类型 | DWORD, HANDLE 等 |
| 硬编码路径分隔符 | 使用 JoinPath |
| 全局可变状态 | 线程不安全 |

---

## 测试要点

```cpp
// Memory 测试
TEST(MemoryTest, AlignedAlloc_Alignment) {
    void* ptr = Platform::AlignedAlloc(1024, 64);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);
    Platform::AlignedFree(ptr);
}

// SIMD 测试
TEST(SIMDTest, Detection_Consistent) {
    // 多次调用结果一致
    bool sse1 = Platform::HasSSE4();
    bool sse2 = Platform::HasSSE4();
    EXPECT_EQ(sse1, sse2);
}

// Thread 测试
TEST(ThreadTest, ParallelFor_Correctness) {
    std::atomic<int> sum{0};
    Platform::ParallelFor(0, 1000, [&sum](int i) {
        sum += i;
    });
    EXPECT_EQ(sum.load(), 999 * 1000 / 2);
}

// Random 测试（确定性）
TEST(RandomTest, Seed_Deterministic) {
    Platform::Random r1(42), r2(42);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(r1.NextInt(0, 1000), r2.NextInt(0, 1000));
    }
}
```

---

## ⚠️ 进度更新规则 (强制)

**完成任何工作后必须立即执行：**

1. 读取 `.claude/PROGRESS.md`
2. 更新对应模块的状态 (⬜→🟡→✅)
3. 在"变更日志"添加本次工作记录
4. **禁止跳过此步骤**

```markdown
# 示例：完成 Thread.h 实现后更新
| Thread.h | ✅ | ✅ | ✅ | ⬜ | 线程池、ParallelFor |

### 变更日志
### 2025-XX-XX
- Thread.h: 完成设计、实现、单测
```

## 检查清单

- [ ] 阅读 CLAUDE.md 中跨平台规则
- [ ] 实现跨平台抽象
- [ ] 内存对齐 64 字节
- [ ] 支持 SSE4/AVX2/AVX512/NEON 检测
- [ ] 线程池正确关闭
- [ ] 随机数支持固定种子
- [ ] 编写单元测试
- [ ] Windows/Linux 都测试
- [ ] 代码格式化
- [ ] **⚠️ 更新 PROGRESS.md 状态（强制）**

## ⚠️ 测试失败处理规则 (强制)

**测试失败时，必须优先修复代码，而非修改测试：**

### 处理原则

```
❌ 错误做法：测试失败 → 修改测试期望 → 测试通过
✓ 正确做法：测试失败 → 分析问题 → 修复代码 → 测试通过
```

### 仅允许修改测试的情况

1. **平台差异** - 不同平台的合理差异（需注释说明）
2. **测试 bug** - 测试代码本身有错误
3. **规格变更** - 明确的需求变更

---

## 约束

- **必须跨平台** - Windows + Linux + macOS
- **无平台特定代码暴露** - 封装在 .cpp 中
- **内存对齐 64 字节** - AVX512 友好
- **线程安全** - 全局状态使用 thread_local 或 mutex
- **测试失败必须修复代码** - 见上述规则
