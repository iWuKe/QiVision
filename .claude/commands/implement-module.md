# /project:implement-module 命令

## 用法

```
/project:implement-module <ModulePath>

示例:
/project:implement-module Internal/Steger
/project:implement-module Measure/Caliper
```

## 执行流程

```
Step 1: 设计 (vision-architect)
    ↓
Step 2: 检查依赖
    ↓
Step 3: 实现 (core-dev/internal-dev/feature-dev)
    ↓
Step 4: 单元测试 (unit-tester)
    ↓
Step 5: 测试数据 (test-generator)
    ↓
Step 6: 精度测试 (accuracy-tester)
    ↓
Step 7: 代码审查 (code-reviewer)
    ↓
Step 8: 更新 PROGRESS.md
```

---

## Agent 分发

| 模块路径 | Agent |
|----------|-------|
| Platform/* | platform-dev |
| Core/* | core-dev |
| Internal/* | internal-dev |
| Measure/Matching/... | feature-dev |

---

## 依赖检查

实现前检查依赖模块状态，如有缺失需先实现依赖。

---

## 检查点恢复

```
/project:resume-module <ModulePath>
```

从最后成功的检查点继续。
