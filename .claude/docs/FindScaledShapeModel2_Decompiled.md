# find_scaled_shape_model_2 反编译分析

> 与 find_scaled_shape_model 和 find_shape_model_2 的对比分析，重点关注缩放+掩膜的组合处理。
> 基于用户提供的反编译代码片段，非完整 IDA 输出。

---

## 0. 函数签名与参数映射

### 0.1 find_scaled_shape_model_2 签名

```c
__int64 __fastcall find_scaled_shape_model_2(
    void *a1,       // 图像数据指针 (cv::Mat data)
    int   a2,       // 图像列数 (cols)
    int   a3,       // 图像行数 (rows)
    int   a4,       // 模型 ID (model handle)
    int   a5,       // [float] angleStart   -- vmovss xmm9
    int   a6,       // [float] angleExtent  -- vmovss xmm10
    int   a7,       // [float] scaleMin     -- vmovss xmm11
    int   a8,       // [float] scaleMax     -- vmovss xmm12
    int   a9,       // [float] 传给 sub_18007E200(v591) 和 dword_1800D4D50 角度步长乘积 -- 待确认: 可能是 angleStep 或 minScore
    int   a10,      // maxMatches (v542=a10, v541=a10, 后用于结果截断)
    int   a11,      // 传给 sub_18007E200(v608) -- 待确认用途 (可能是 maxOverlap 或 greediness)
    int   a12,      // subPixel + searchRadius*10 编码 (a12%10=subPixel, a12/10=searchRadiusBase)
    int  *a13,      // [numLevels, startLevel] -- 2-element array (v28=a13, v35=*v28, v39=v28[1])
    int   a14,      // [float] 传给 sub_18007E200(v608) -- 待确认 (可能是 greediness 或 maxOverlap)
    __int64 a15,    // *** mask 数据指针 *** (v29=a15, 0=无mask)
    __int64 *a16,   // 输出: 结果数组 (v545=a16)
    int  *a17       // 输出: 匹配数量 (v546=a17, *a17=0)
);
```

### 0.2 三函数参数对比

| 位置 | find_scaled_shape_model (16参) | find_shape_model_2 (15参) | find_scaled_shape_model_2 (17参) | 说明 |
|------|-------------------------------|---------------------------|----------------------------------|------|
| a1 | image data | image data | image data | 同 |
| a2 | cols | cols | cols | 同 |
| a3 | rows | rows | rows | 同 |
| a4 | model ID | model ID | model ID | 同 |
| a5 | [float] angleStart | [float] angleStart | [float] angleStart | 同 |
| a6 | [float] angleExtent | [float] angleExtent | [float] angleExtent | 同 |
| a7 | **[float] scaleMin** | [float] angleStep | **[float] scaleMin** | 缩放版共享 |
| a8 | **[float] scaleMax** | [float] minScore | **[float] scaleMax** | 缩放版共享 |
| a9 | [float] angleStep/minScore | maxMatches | [float] angleStep/minScore | 待确认 |
| a10 | maxMatches | subPixel+searchRadius | maxMatches | |
| a11 | maxOverlap/greediness | [numLevels, startLevel] | maxOverlap/greediness | 待确认 |
| a12 | subPixel+searchRadius | greediness | subPixel+searchRadius | 同编码方式 |
| a13 | [numLevels, startLevel] | **mask data ptr** | [numLevels, startLevel] | 同 |
| a14 | greediness | output results | **[float] greediness/maxOverlap** | 待确认 |
| a15 | output results | output count | **mask data ptr** | **新增** |
| a16 | output count | -- | output results | 位移 +1 |
| a17 | -- | -- | output count | 位移 +1 |

**核心结论**: `find_scaled_shape_model_2` = `find_scaled_shape_model` 的全部 16 个参数 + 在输出参数前插入 1 个 mask 指针 (a15)，共 17 个参数。

---

## 1. 掩膜处理流程

### 1.1 掩膜输入与 Mat 构建

```c
v29 = a15;   // 保存 mask 指针

// 如果有 mask，构建 cv::Mat
if ( v29 ) {
    LODWORD(Time2) = v642[1];   // rows
    HIDWORD(Time2) = *v642;     // cols
    v31 = cv::Mat::Mat(&v627, Time2, 0, v29, 0);  // type 0 = CV_8UC1
    cv::Mat::operator=(v646, v31);                  // deep copy to v646
    cv::Mat::~Mat(&v627);
}
```

**分析**:
- mask 是一个与搜索图像同尺寸的单通道图像
- 通过 `cv::Mat::Mat(rows, cols, type, data, step)` 构建
- 存储在局部变量 v646 中供后续使用
- **与 find_shape_model_2 完全相同的构建方式**

### 1.2 掩膜金字塔构建

```c
// 掩膜金字塔层数: min(numLevels, 2)
v46 = 2;
if ( v35 < 2 ) v46 = v35;  // v35 = numLevels

// 只对前 min(numLevels, 2) 层构建掩膜金字塔
if ( v46 > 0 && v29 ) {
    v47 = 0;
    do {
        if ( v47 > 0 ) {
            cv::pyrDown(&v535, &v527, &Time2, 4);  // 降采样 mask
        }
        v48 = sub_180008120(v604, v47);  // 获取 mask 层 slot
        cv::Mat::operator=(v48, v646);    // 存储到金字塔
        ++v47;
    } while ( v47 < v46 );
}
```

**关键发现**:
1. **掩膜金字塔最多 2 层** -- 不管搜索金字塔有多少层，mask 只降采样 1 次
2. **与 find_shape_model_2 完全相同**: `min(numLevels, 2)` 限制 + cv::pyrDown
3. 掩膜存储在 v604 管理的数组中，通过 sub_180008120 按 level 索引访问

### 1.3 掩膜在细化循环中的应用

```c
// 在 Sobel 梯度计算之后:
if ( _R11D > 0 )
    sub_180037E10(v289, v288, v287, xmm3, _R11D);  // Sobel with border
else
    sub_180036DC0(v289, v288, v287, xmm3);          // Standard Sobel

// *** mask 应用 (与 find_shape_model_2 完全相同的 sub_180038450) ***
if ( !cv::Mat::empty((cv::Mat *)(v604[0] + 96LL * (int)v555)) )
    sub_180038450(v604[0] + 96LL * (int)v555,      // mask[level]
                  96LL * (int)v555 + v580,           // cosGrad[level]
                  96LL * (int)v555 + v585,           // sinGrad[level]
                  v560);                              // searchRadius
```

**sub_180038450 -- MaskGradient (掩膜梯度清零)**:
- **地址**: `0x180038450` (与 find_shape_model_2 使用的是**同一个函数**)
- **作用**: 将 mask 中为 0 的区域对应的梯度清零
- **等效**: `cosGrad[mask==0] = 0; sinGrad[mask==0] = 0;`
- **参数**: (mask_mat, cosGrad_mat, sinGrad_mat, searchRadius)
- 详细分析见 `FindShapeModel2_Decompiled.md` 第 4.1 节

### 1.4 掩膜 level 索引映射

```c
// v555 = current level index for mask
// 由于 mask 只有 min(numLevels, 2) 层:
// - 当 level == 0 时: 使用 mask[0] (原始 mask)
// - 当 level == 1 时: 使用 mask[1] (降采样 mask)
// - 当 level >= 2 时: cv::Mat::empty 返回 true, 跳过 mask 应用
```

### 1.5 粗搜索阶段

**mask 不参与粗搜索**。粗搜索部分（parallel_for + 响应图 LUT + NMS）的代码中没有任何 mask 相关引用。这与 find_shape_model_2 的行为一致。

---

## 2. 与 find_scaled_shape_model 的差异

### 2.1 唯一差异: mask 步骤

| 方面 | find_scaled_shape_model | find_scaled_shape_model_2 | 差异 |
|------|------------------------|--------------------------|------|
| 参数数量 | 16 | 17 (插入 a15=mask) | 多 1 个 |
| mask 金字塔构建 | 无 | min(numLevels, 2) 层, cv::pyrDown | **新增** |
| 细化循环 | Sobel -> 打分 | Sobel -> **mask 清零** -> 打分 | **新增步骤** |
| 粗搜索 | 响应图 LUT | 响应图 LUT (同) | 相同 |
| 缩放搜索网格 | angle x scale parallel_for | angle x scale parallel_for (同) | 相同 |
| SubPixel | 4 种模式 | 4 种模式 (同) | 相同 |
| NMS/后处理 | SpatialNMSCluster | SpatialNMSCluster (同) | 相同 |
| per-level 配置 | dword_1800D4D50/CF0 | dword_1800D4D50/CF0 (同) | 相同 |
| 缩放范围生成 | sub_1800B8160 | sub_1800B8160 (同) | 相同 |

### 2.2 共享子函数 (全部复用)

以下子函数在 find_scaled_shape_model 中已有完整记录 (见 `FindScaledShapeModel_Decompiled.md`)，find_scaled_shape_model_2 全部复用:

| 地址 | 名称 | 用途 | 已对齐 |
|------|------|------|--------|
| sub_1800B7FB0 | GenerateAngleRange | 角度搜索范围 | YES |
| sub_1800B8160 | GenerateScaleRange | 缩放搜索范围 | YES |
| sub_1800B82C0 | CalcScaleStep | 缩放步长计算 | YES |
| sub_1800B72F0 | GetModelTransform (scaled) | 4-vector SoA 模板变换 | YES |
| sub_1800B68B0 | GetModelTransform (non-scaled) | 3-vector SoA 模板变换 | YES |
| sub_180039480 | BuildResponseMap | 响应图构建 (SIMD) | YES |
| sub_1800497F0 | CollectCandidatesNMS | 粗搜 NMS | YES |
| sub_18004C8C0 | GreedyNMS | 层内 NMS (最终层) | YES |
| sub_18004B100 | SpatialNMSCluster | 后处理 NMS (5 阶段) | YES |
| sub_180037E10 | ComputeGradientWithBorder | Sobel + 扩展边界 | YES |
| sub_180036DC0 | ComputeGradient | 标准 Sobel | YES |
| sub_18005F700 | CoarseSearchWorker (scaled) | 粗搜 worker | YES |
| sub_18005F960 | RefineWorker (scaled) | 细化 worker | YES |
| sub_18005B7E0 | SubPixelMode1Worker | SubPixel mode 1 | YES |
| sub_18005B950 | SubPixelMode2Worker | SubPixel mode 2 (Bresenham) | YES |
| sub_18005BE10 | SubPixelMode3Worker | SubPixel mode 3 (Jacobian) | YES |
| sub_1800B7780 | GenerateAngleScaleCandidates | 角度+缩放候选 (SubPixel mode 2/3) | YES |

### 2.3 仅新增的子函数

| 地址 | 名称 | 用途 | 来源 |
|------|------|------|------|
| sub_180038450 | MaskGradient | 掩膜梯度清零 | 与 find_shape_model_2 共享 |

---

## 3. 与 find_shape_model_2 的对比

### 3.1 mask 处理一致性

| mask 环节 | find_shape_model_2 | find_scaled_shape_model_2 | 一致性 |
|-----------|-------------------|--------------------------|--------|
| mask 指针参数 | a13 | a15 | 位置不同, 语义相同 |
| Mat 构建 | cv::Mat(rows, cols, 0, ptr, 0) | cv::Mat(rows, cols, 0, ptr, 0) | 完全相同 |
| 金字塔层数 | min(numLevels, 2) | min(numLevels, 2) | 完全相同 |
| 降采样方法 | cv::pyrDown | cv::pyrDown | 完全相同 |
| 清零函数 | sub_180038450 | sub_180038450 | **同一函数地址** |
| 应用时机 | Sobel 后, 打分前 | Sobel 后, 打分前 | 完全相同 |
| 粗搜索参与 | 不参与 | 不参与 | 完全相同 |

**结论**: mask 处理逻辑在两个函数中完全一致，是正交的功能模块。

### 3.2 搜索核心差异

| 方面 | find_shape_model_2 | find_scaled_shape_model_2 |
|------|-------------------|--------------------------|
| 缩放支持 | 无 | 有 (scaleMin/scaleMax 参数) |
| 模板变换 | 3-vector SoA (sub_1800B68B0) | 4-vector SoA (sub_1800B72F0, 缩放版) |
| 搜索网格 | angle only | angle x scale |
| 细化路径 | 非缩放 worker | 缩放 worker (RefineAtLevelScaled) |
| SubPixel 候选 | 角度候选 | 角度+缩放候选 (sub_1800B7780) |

---

## 4. 细化循环详细结构

从反编译代码直接提取的循环结构:

```
v555 = v35 - 2   (numLevels - 2, 起始层)
do {
    // 1. 获取 per-level 参数 (scale range, angle range, stride)

    // 2. 生成 scale 范围 (sub_1800B8160)
    //    v620-v625 存储 per-level scale 参数

    // 3. 生成 angle 范围 (sub_1800B7FB0)

    // 4. 构建 scale x angle 搜索网格

    // 5. 粗搜索 parallel_for (sub_18005F700)

    // 6. 收集结果 -> v563

    // 7. NMS 调度:
    //    level == startLevel -> sub_18004C8C0 (GreedyNMS)
    //    level > startLevel  -> sub_18004B100 (SpatialNMSCluster)

    // --- 层间细化 ---

    // 8. Sobel 梯度计算
    //    searchRadius > 0: sub_180037E10 (带 border)
    //    searchRadius == 0: sub_180036DC0 (标准)

    // 9. *** MASK 应用 *** (仅此处与 find_scaled_shape_model 不同)
    //    if (!cv::Mat::empty(mask[level]))
    //        sub_180038450(mask[level], cosGrad[level], sinGrad[level], searchRadius)

    // 10. 细化 parallel_for (sub_18005F960)

    // 11. 角度范围过滤

    // 12. NMS

    // 13. SubPixel 调度:
    //     mode 1: sub_18005B7E0
    //     mode 2: sub_18005B950
    //     mode 3: sub_18005BE10

    v555 = v555 - 1
} while ( v555 >= v558 );   // v558 = startLevel
```

**与 find_scaled_shape_model 的差异**: 仅步骤 9 (mask 应用) 是新增的，其余步骤完全一致。

---

## 5. 常量表

与 find_scaled_shape_model 和 find_shape_model_2 完全相同:

### 5.1 dword_1800D4D50 -- per-level angleStep 倍率表

**地址**: `0x1800D4D50`
**确认值** (float[16]):
```
[0]  = 0.8f
[1]  = 0.9f
[2..15] = 0.9f
```

### 5.2 dword_1800D4CF0 -- per-level angle subdivision 表

**地址**: `0x1800D4CF0`
**确认值** (int[16]):
```
[0]=2, [1]=3, [2]=3, [3]=4, [4]=4, [5]=4, [6]=5,
[7]=5, [8]=5, [9]=5, [10]=6, [11]=6, [12]=6, [13]=6, [14]=6, [15]=7
```

### 5.3 dword_1800D6B38 / dword_1800D6ADC -- 缩放相关常量

与 find_scaled_shape_model 相同，详见 `FindScaledShapeModel_Decompiled.md` 第 6 节。

> **待 IDA 确认**: 具体值尚未完全确认。

---

## 6. 整体流程架构

```
find_scaled_shape_model_2 流程:
  |
  +-- 1. 参数解析 (与 find_scaled_shape_model 相同)
  |   +-- a12 % 10 -> subPixel
  |   +-- a12 / 10 -> searchRadiusBase (clamp to 32)
  |   +-- *a13 -> numLevels, a13[1] -> startLevel
  |   +-- a5-a8 (float): angleStart, angleExtent, scaleMin, scaleMax
  |
  +-- 2. *** 掩膜构建 (与 find_shape_model_2 相同) ***
  |   +-- if a15 != 0: 构建 mask cv::Mat
  |   +-- 构建 mask 金字塔 (max 2 层, cv::pyrDown)
  |
  +-- 3. 金字塔构建 (cv::pyrDown, 同 find_scaled_shape_model)
  |
  +-- 4. per-level 配置数组 (同 find_scaled_shape_model)
  |   +-- per-level angleStep (x dword_1800D4D50 倍率表)
  |   +-- per-level stride (searchRadiusBase 逐层减半)
  |   +-- per-level angle subdivision (dword_1800D4CF0)
  |
  +-- 5. 构建 angle x scale 搜索网格 (同 find_scaled_shape_model)
  |   +-- sub_1800B8160: 生成 scale 范围
  |   +-- sub_1800B7FB0: 生成 angle 范围
  |
  +-- 6. 顶层粗搜索 (parallel_for, 同 find_scaled_shape_model)
  |   +-- 对每个 (angle, scale) 组合:
  |   |   +-- sub_1800B72F0: 生成变换后模板 (4-vector SoA)
  |   |   +-- sub_180039480: 构建响应图 (SIMD)
  |   |   +-- sub_1800497F0: NMS + 候选提取
  |   +-- 注: 粗搜索 **不使用 mask**
  |
  +-- 7. 金字塔逐级细化 (numLevels-2 -> startLevel)
  |   +-- Sobel 梯度计算:
  |   |   +-- searchRadius > 0: sub_180037E10 (带 border)
  |   |   +-- searchRadius == 0: sub_180036DC0 (标准)
  |   +-- *** 掩膜应用 (新增, 与 find_shape_model_2 相同) ***
  |   |   +-- if !mask[level].empty(): sub_180038450(mask, cosGrad, sinGrad, radius)
  |   +-- parallel_for 细化 (sub_18005F960, 缩放版 worker)
  |   +-- 角度范围检查 + 过滤
  |   +-- 层内 NMS:
  |   |   +-- level == startLevel -> sub_18004C8C0 (GreedyNMS)
  |   |   +-- level > startLevel  -> sub_18004B100 (SpatialNMSCluster)
  |   +-- SubPixel (mode 1/2/3, 含缩放候选 sub_1800B7780)
  |
  +-- 8. SpatialNMSCluster (sub_18004B100)
  |
  +-- 9. sort + truncate -> 输出
```

---

## 7. 总结

### 关键发现

1. **find_scaled_shape_model_2 = find_scaled_shape_model + mask** -- 缩放搜索算法完全相同，唯一区别是在细化阶段对梯度应用掩膜
2. **mask 处理与 find_shape_model_2 完全一致** -- 同一个 sub_180038450 函数、同样的金字塔层数限制 (max 2)、同样的应用时机
3. **mask 不参与粗搜索** -- 响应图 LUT 阶段不使用 mask
4. **mask 应用方式: 梯度清零** -- 预处理式地将 mask 外的梯度清零，不影响打分函数内部逻辑
5. **mask 与 scale 完全正交** -- mask 处理不依赖缩放参数，缩放搜索不依赖 mask

### 函数关系图

```
                    +-- mask -->  find_shape_model_2
                    |
find_shape_model ---+
                    |
                    +-- scale --> find_scaled_shape_model
                                          |
                                          +-- mask --> find_scaled_shape_model_2
```

每个箭头表示"增加一个正交功能"。find_scaled_shape_model_2 是 scale + mask 的组合。

---

## 8. QiVision 实现策略

### 8.1 现有实现状态

| 功能 | 状态 | 位置 |
|------|------|------|
| FindShapeModel (非缩放, 无 mask) | DONE | ShapeModelSearch.cpp |
| FindShapeModel + mask | DONE | DownsampleMask + MaskGradientLevel + ApplySearchMask |
| FindScaledShapeModel (缩放, 无 mask) | DONE | ShapeModelSearch.cpp (缩放路径) |
| FindScaledShapeModel + mask | **未实现** | 需要组合 |

### 8.2 实现方案

**由于 mask 处理与 scale 搜索完全正交，只需将现有 mask 机制复用到缩放搜索路径即可。**

具体步骤:

1. **无需新增函数** -- DownsampleMask、MaskGradientLevel、ApplySearchMask 已实现
2. **在缩放搜索路径的细化循环中插入 mask 步骤** -- 与非缩放路径的 mask 插入点相同:
   - 时机: Sobel 梯度计算后、打分 parallel_for 前
   - 调用: ApplySearchMask(maskPyramid, level, cosGrad, sinGrad, searchRadius)
3. **mask 金字塔构建** -- 已有的 DownsampleMask(mask, min(numLevels, 2)) 直接复用

### 8.3 预期工作量

| 项目 | 工作量 | 说明 |
|------|--------|------|
| 缩放路径插入 mask 调用 | 极小 (~5 行) | 复用现有 ApplySearchMask |
| 测试 | 小 | 复用现有 mask 测试 + 缩放参数 |
| 新增函数 | 0 | 全部复用 |

### 8.4 实现差异说明

| 项 | 反编译 | QiVision | 影响 |
|----|--------|----------|------|
| mask 降采样 | cv::pyrDown (5x5 Gaussian) | 2x2 block + 阈值 256 | mask 边界 ~1px 差异，可忽略 |
| mask 应用时机 | 每层 refinement 循环内 | Build() 之后统一应用 | 等价 (同一份梯度数据) |
| 梯度 searchRadius 偏移 | grad[r+R][c+R] (有边界填充) | grad[r][c] (无填充) | QiVision 无填充，正确 |

---

## 附录: 待确认参数

以下参数的具体语义尚未从反编译代码中完全确定:

| 参数 | 推测 | 依据 | 状态 |
|------|------|------|------|
| a9 | angleStep 或 minScore | 传给 dword_1800D4D50 角度步长乘积 | 待确认 |
| a11 | maxOverlap 或 greediness | 传给 sub_18007E200 | 待确认 |
| a14 | greediness 或 maxOverlap | 传给 sub_18007E200 | 待确认 |

> 注: a9/a11/a14 的确切映射需要 IDA 中 sub_18007E200 的完整分析，或与 Halcon 文档的参数顺序对照来确定。从 Halcon 文档 `find_scaled_shape_model` 的参数顺序推测: a9=minScore, a10=maxMatches, a11=maxOverlap, a14=greediness。但未经 IDA 验证，标注为待确认。
