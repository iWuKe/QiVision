# Measure SDK 参数规范（草案 v0.1）

本文档用于固定 `Caliper / CaliperArray / Metrology` 的对外参数契约，减少后期 SDK 封装返工。

## 1. 适用范围

- `Qi::Vision::Measure::MeasurePos / MeasurePairs / FuzzyMeasurePos / FuzzyMeasurePairs`
- `Qi::Vision::Measure::CaliperArray`
- `Qi::Vision::Measure::MetrologyModel + MetrologyMeasureParams`

## 2. 当前实现的默认值（已落地）

### 2.1 CaliperArray 常用默认值

| 参数 | 默认值 |
|---|---|
| `sigma` | `1.0` |
| `threshold` | `20.0` |
| `transition` | `"all"` |
| `select` | `"first"` |
| `fuzzyThresh` | `0.5` |

### 2.2 MetrologyMeasureParams 默认值

| 参数 | 默认值 |
|---|---|
| `numInstances` | `1` |
| `measureLength1` | `20.0` |
| `measureLength2` | `5.0` |
| `measureSigma` | `1.0` |
| `measureThreshold` | `30.0` |
| `thresholdMode` | `Manual` |
| `measureTransition` | `All` |
| `measureSelect` | `All` |
| `numMeasures` | `10` |
| `minScore` | `0.7` |
| `minCoverage` | `0.0`（关闭） |
| `maxRmsError` | `-1.0`（关闭） |
| `fitMethod` | `RANSAC` |
| `distanceThreshold` | `3.5` |
| `maxIterations` | `-1` |
| `randSeed` | `42` |
| `ignorePointCount` | `0` |
| `ignorePointPolicy` | `ByResidual` |

## 3. 输入参数约束（建议作为 SDK 硬约束）

### 3.1 数值范围

| 参数 | 约束 |
|---|---|
| `sigma`, `measureSigma` | `> 0` |
| `threshold`, `measureThreshold` | `>= 0` |
| `fuzzyThresh` | `>= 0`（建议 SDK 进一步限制为 `[0,1]`） |
| `measureLength1`, `measureLength2` | `> 0` |
| `numMeasures` | `> 0` |
| `numInstances` | `> 0` |
| `minScore` | `[0,1]` |
| `minCoverage` | `[0,1]` |
| `maxRmsError` | 有限数值；`<=0` 代表关闭，`>0` 代表启用 |
| `distanceThreshold` | `> 0` |
| `maxIterations` | `>= -1` |
| `ignorePointCount` | `>= 0` |

### 3.2 枚举/字符串取值

| 参数 | 合法值 |
|---|---|
| `transition` | `"positive"`, `"negative"`, `"all"` |
| `select`（edge） | `"first"`, `"last"`, `"all"`, `"strongest"`, `"best"`(`best->strongest`) |
| `pair select` | `"first"`, `"last"`, `"all"` |
| `thresholdMode` | `"manual"`, `"auto"` |
| `fitMethod` | `"ransac"`, `"huber"`, `"tukey"` |
| `ignorePointPolicy` | `"residual"`, `"score"` |

## 4. 输出契约（建议固定）

### 4.1 几何结果有效性判定

| 结果类型 | `IsValid()` 判定 |
|---|---|
| `MetrologyLineResult` | `numUsed >= 2 && score > 0` |
| `MetrologyCircleResult` | `numUsed >= 3 && radius > 0 && score > 0` |
| `MetrologyEllipseResult` | `numUsed >= 5 && ra > 0 && rb > 0 && score > 0` |
| `MetrologyRectangle2Result` | `numUsed >= 4 && length1 > 0 && length2 > 0 && score > 0` |

### 4.2 ignore-point 行为

- `ignorePointCount` 代表“先测量点集，再按策略剔除，再重拟合”。
- `ByResidual` 为优先推荐策略，按残差从大到小剔除。
- `ByScore` 按点质量分数从低到高剔除。
- 若剔除后剩余点不足最小拟合数，则不应用剔除。

### 4.3 点级可解释结果（已实现）

- `GetMeasuredPoints(index)`：对象测得的全部边缘点坐标。
- `GetPointWeights(index)`：用于可视化的点权重（常用于红/绿内外点显示）。
- `GetPointDetails(index)`：按点输出可追溯信息（用于平台调试页/诊断页）。

`MetrologyPointDetail` 字段建议作为 SDK 稳定输出：

| 字段 | 说明 |
|---|---|
| `pointIndex` | 在 `GetMeasuredPoints(index)` 中的索引 |
| `caliperIndex` | 来源卡尺序号（用于“各点对各边”诊断） |
| `instanceIndex` | 所属拟合实例序号 |
| `row`, `column` | 点坐标 |
| `amplitude` | 边缘幅值 |
| `residual` | 点到拟合几何的残差（像素） |
| `weight` | 点权重（0~1） |
| `isInlier` | 最终是否作为内点 |

## 5. 错误语义（建议 SDK 对外固定）

### 5.1 参数错误

- 异常类型：`InvalidArgumentException`
- 推荐格式：`<FuncName>: <param> must be <rule>`
- 示例：
- `MeasurePos: sigma must be > 0`
- `AddRectangle2Measure: lengths must be > 0`
- `AddCircleMeasure: radius must be > 0`

### 5.2 图像错误

- 无效图像输入时，测量接口返回空结果或 `false`（按现有实现）。
- SDK 包装层建议统一映射为状态码并附带 message，避免上层直接处理异常字符串。

## 6. 命名稳定建议（SDK 前必须冻结）

- 入参统一使用 `camelCase`。
- 几何参数统一使用 `row/column`（不要混用 `x/y` 作为外部 API 主命名）。
- 长度语义统一：
- `length1/length2` 表示半长。
- 文档明确 `profileLength` 在 CaliperArray 中映射半长（`Length1`）。
- 策略类参数使用显式字段：
- `fitMethod`
- `ignorePointCount`
- `ignorePointPolicy`

## 7. 仍建议补齐的两项（封装 SDK 前）

- 增加参数一致性单测：
- 覆盖所有非法输入分支，校验异常 message 前缀和关键字段名。
- 增加“默认参数回归单测”：
- 固定一组基准图像，确保默认配置升级后结果漂移可监控。

---

状态说明：本文是“可执行草案”。如果你确认采用，我建议下一步把它升级为 `v1.0` 并同步到 `docs/API_Reference.md` 的 Measure 章节。
