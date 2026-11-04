"""predictor_server 依赖边的宏注入。

bazel 的构建配置是全局的（且不允许对 --define 做 transition），"仅
predictor_server 这个目标开宏"用自定义 build setting 实现：

  gflags_switch_flag  字符串旗标，默认 "off"，由下方的依赖边 transition 打开
                      成 "on"（predictor_server 路径），也可用命令行
                      --//:gflags_switch_flag=on 整体打开（测试 gflag 语义用，
                      与库同配置）。
  gflags_switch_dep   透明转发 CcInfo/DefaultInfo，仅给依赖子树把旗标置 "on"。

各读点所在 BUILD（tensorflow/core、grappler/graph_optimizer、util）经
//:gflags_switch_on 的 select 变成 -DKDNN_SWITCH_GFLAGS 与 gflags 依赖。裸跑
`bazel build //:predictor_server` 即得 gflag 分支；so/wheel 等不经过此边的
构建旗标保持 "off"，默认环境变量分支，gflags 不进 wheel。
"""

BuildSettingInfo = provider(
    doc = "build setting 的当前值（命令行或上游 transition，否则为默认值）",
    fields = ["value"],
)

def _gflags_switch_flag_impl(ctx):
    return BuildSettingInfo(value = ctx.build_setting_value)

gflags_switch_flag = rule(
    implementation = _gflags_switch_flag_impl,
    build_setting = config.string(flag = True),
)

def _gflags_switch_transition_impl(settings, attr):
    return {"//:gflags_switch_flag": "on"}

gflags_switch_transition = transition(
    # 无条件置 "on"，不读任何输入（settings 恒为空）。
    implementation = _gflags_switch_transition_impl,
    inputs = [],
    outputs = ["//:gflags_switch_flag"],
)

def _gflags_switch_dep_impl(ctx):
    # 带 transition 的 label attr 以 list 形态出现（1:1 transition 单元素）。
    dep = ctx.attr.dep[0]
    return [
        dep[CcInfo],
        DefaultInfo(files = dep.files, runfiles = dep.default_runfiles),
    ]

gflags_switch_dep = rule(
    implementation = _gflags_switch_dep_impl,
    attrs = {
        "dep": attr.label(cfg = gflags_switch_transition),
    },
)
