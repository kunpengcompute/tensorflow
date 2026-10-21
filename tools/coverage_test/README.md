# TensorFlow 增量覆盖率测试工具

对 TensorFlow 特性增量做单元测试 + 增量代码覆盖率分析。入口是 `run_coverage.py`，特性在 `features/<name>/config.json` 里声明。

## 快速开始

```bash
# 列出可用特性
python3 tools/coverage_test/run_coverage.py --list

# 跑某个特性的单元测试（不收集覆盖率，快）
python3 tools/coverage_test/run_coverage.py --feature annc

# 收集覆盖率并分析增量（<ref> 到 HEAD 的变更，含该 commit）
python3 tools/coverage_test/run_coverage.py --feature annc \
    --since 086c065e758 --enable-coverage
```

## 命令行选项

| 选项 | 说明 |
|---|---|
| `--feature <name>` | 只跑指定特性 |
| `--all` | 跑所有特性，汇总一份增量覆盖率 |
| `--list` | 列出所有特性 |
| `--enable-coverage` | 收集覆盖率（需要 lcov） |
| `--since <ref>` | 增量范围：`<ref>`（含）到 HEAD |
| `--commit <ref>` | 增量范围：单个 commit |
| `--patch <file>` | 增量范围：patch 文件 |
| `--base-ref <r> --target-ref <r>` | 增量范围：两个 ref 之间的 diff |

四种增量来源（`--since` / `--commit` / `--patch` / `--base-ref`+`--target-ref`）四选一。

## 输出

```
coverage_test_results/<feature>/
├── report.json                # 测试结果 + 增量覆盖率（机器可读）
├── incremental_coverage.csv   # 每一条变更行的覆盖状态
├── file_coverage.csv          # 按文件汇总
├── directory_coverage.csv     # 按目录汇总（含子目录，父子的数字重叠不可相加）
└── html/
    ├── index.html             # 目录树 + 全局增量覆盖率 + 测试结果 + 诊断
    ├── directories/           # 每个目录一页：面包屑导航、子目录与文件列表
    └── files/                 # 每个文件一页：变更行标记、命中次数、覆盖状态
```

增量覆盖率在 `report.json` 的 `incremental_coverage.summary`，核心字段 `coverage_rate`；
网页入口 `html/index.html` 浏览器直接打开，不需要启动服务。覆盖率 =
covered / (covered + uncovered)：绿色 = 新增行已覆盖，红色 = 新增行未覆盖，
黄色 = 未插桩（unknown，单列不进分母），灰色 = 空行/注释/预处理行。

## 新增一个特性

1. 建目录 `features/<name>/`，写 `config.json`：

```json
{
  "name": "<name>",
  "description": "...",
  "bazel_args": ["--config=v2"],
  "bazel_build_flags": ["-fexceptions"],
  "test_targets": [
    {"id": "xxx-ut", "target": "//tensorflow/core/kernels:xxx_test",
     "kind": "gtest", "gtest_filter": "*", "enabled": true}
  ]
}
```

2. 直接运行即可：`python3 tools/coverage_test/run_coverage.py --feature <name>`。

字段说明：
- `bazel_args` / `bazel_build_flags`：特性级 bazel 参数 / 编译选项（后者自动加 `--copt=`）。
- `test_targets[].env` / `bazel_build_flags`：目标级环境变量 / 编译选项。
- `kind` 目前只支持 `gtest`（benchmark 会跳过）。

## 常见问题

- **覆盖率 0%**：确认加了 `--enable-coverage`、`lcov` 已装、增量来源选对了。
- **KDNN 链接失败**：KDNN 库与编译选项不匹配（`third_party/KDNN/src/*.a` 需与 `--define=enable_kdnn=true` 构建配套）。
