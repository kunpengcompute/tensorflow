#!/usr/bin/env bash
# 修改 force_ref_impl 属性的脚本（embedding_lookup 通过函数参数控制，无需修改源码）
# 保留此脚本用于兼容 run_full_experiment 流程；实际上 embedding_lookup 的
# force_ref_impl 已在 benchmark 中通过函数参数传递，此步骤为 no-op。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

VALUE="${1:-false}"

if [ "$VALUE" != "true" ] && [ "$VALUE" != "false" ]; then
    echo "Usage: $0 [true|false]"
    echo "  true  - 使用原生实现 (未优化)"
    echo "  false - 使用 ARM 优化实现"
    exit 1
fi

# embedding_lookup 的 force_ref_impl 通过 benchmark 函数参数控制，无需修改文件
echo "embedding_lookup 的 force_ref_impl 通过 benchmark 函数参数控制，无需修改源码"
echo "当前设定: force_ref_impl=${VALUE}"
echo "✓ 完成"
