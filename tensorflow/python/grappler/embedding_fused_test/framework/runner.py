import os
import gc
import time
import shutil
import glob
import json
import sys
import re
import struct
import tensorflow.compat.v1 as tf
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from tensorflow.python.client import timeline
import numpy as np
import multiprocessing as mp
import pickle
from tensorflow.python.client import timeline
from tensorflow.core.protobuf import rewriter_config_pb2

tf.disable_eager_execution()


@dataclass
class TestCase:
    name: str
    op_fn: Callable
    input_fn: Callable
    check_fn: Optional[Callable] = None
    fused_op_name: str = ""
    start_op_name: str = ""
    end_op_name: str = ""
    fused_start_op_name: str = "" # 融合后非单算子场景
    fused_end_op_name: str = ""   # 融合后非单算子场景
    is_fused: bool = True
    num_iters: int = 0
    optimize_percent: int = 0
    meta: Dict = field(default_factory=dict)
    fused_env: Dict[str, str] = field(default_factory=dict)
    unfused_env: Dict[str, str] = field(default_factory=dict)
    unfused_op_names: List[str] = field(default_factory=list) # 基线分支图校验期望算子
    init_fn: Optional[Callable] = None

class CheckFuncClass:
    def check_fn_default(A, B, meta):
        if (len(A) != len(B)):
            return False
        is_check_OK = True
        for i in range(len(A)):
            if (A[i].dtype == tf.float32):
                is_check_OK &= np.testing.assert_allclose(A[i], B[i], rtol=1e-3, atol=1e-4) == None
            else:
                #整数类型严格判断相等, 当前浮点类型仅有float32
                np.testing.assert_array_equal(A[i], B[i]) # 如果失败抛出异常, 如果成功无返回值,

        return is_check_OK

class UniversalOpBenchmark:
    def __init__(self, log_dir="timeline"):
        self.log_dir = log_dir
        self.results = []

    def generate_timeline(self, step_stats_list, timeline_file):
        if not os.path.exists("timeline"):
            os.makedirs("timeline")
        ctf_list = []
        for step_stats in step_stats_list:
            tl = timeline.Timeline(step_stats)
            ctf_list.append(json.loads(tl.generate_chrome_trace_format()))
        with open(f"timeline/{timeline_file}", "w") as f:
            json.dump(ctf_list, f, indent=2)

    def is_fused_op_exist(self, timeline_file, op_name):
        """从 timeline JSON 文件中检查指定算子(fusedOp)是否存在"""
        with open(f"timeline/{timeline_file}", "r") as f:
            trace_events = json.load(f)[0]["traceEvents"]  # timeline.json的格式
        op_exists = any(e.get("name") == op_name for e in trace_events if "dur" in e)
        return op_exists

    def check_branch_ops(self, timeline_file, branch):
        """Verify that a performance branch executed the intended graph."""
        with open(f"timeline/{timeline_file}", "r") as f:
            traces = json.load(f)
        executed_ops = {
            event.get("name")
            for trace in traces
            for event in trace["traceEvents"]
            if "dur" in event
        }
        expected = branch.get("expected_op_names", [])
        forbidden = branch.get("forbidden_op_names", [])
        missing = [name for name in expected if name not in executed_ops]
        unexpected = [name for name in forbidden if name in executed_ops]
        if missing or unexpected:
            raise AssertionError(
                f"性能分支 {branch.get('name', 'branch')} 图校验失败: "
                f"缺少算子{missing}, 非预期算子{unexpected}"
            )

    def extract_op_total_time(self, timeline_file, start_op, end_op, times):
        """计算从 start_op 到 end_op 的总耗时的平均值（包含调度空隙）"""
        with open(f"timeline/{timeline_file}", "r") as f:
            trace_events_list = json.load(f)
        time_list = []
        for trace_events in trace_events_list:
            start_event = next(e for e in trace_events["traceEvents"] if e.get("args", {}).get("name") == start_op)
            end_event = next(e for e in trace_events["traceEvents"] if e.get("args", {}).get("name") == end_op)
            start_time = start_event["ts"]
            end_time = end_event["ts"] + end_event["dur"]  # ts 是开始时间，dur是算子的持续时间
            total_time = end_time - start_time
            time_list.append(total_time)
        if len(time_list) != times:
            raise ValueError(f"Expected {times} total times for {start_op} to {end_op}, but got {len(time_list)}")
        return np.mean(time_list)

    def extract_ops_total_dur(self, timeline_file, op_names, times):
        """Average sum of durations for one or more fused op types."""
        if isinstance(op_names, str):
            op_names = [op_names]
        with open(f"timeline/{timeline_file}", "r") as f:
            trace_events_list = json.load(f)
        time_list = []
        for trace_events in trace_events_list:
            total = 0
            for op_name in op_names:
                durations = [
                    e["dur"] for e in trace_events["traceEvents"]
                    if e.get("name") == op_name and "dur" in e
                ]
                if not durations:
                    raise ValueError(f"Could not find fused op {op_name}")
                total += sum(durations)
            time_list.append(total)
        if len(time_list) != times:
            raise ValueError(
                f"Expected {times} durations, but got {len(time_list)}"
            )
        return np.mean(time_list)

    @staticmethod
    def _apply_env(values):
        for key, value in values.items():
            os.environ[key] = str(value)

    def _performance_branch_time(self, branch, raw_inputs, test_case):
        self._apply_env(branch.get("env", {}))
        tf.reset_default_graph()
        res_node, feed_dict = test_case.op_fn(raw_inputs, test_case.meta)
        config = tf.compat.v1.ConfigProto()
        config.inter_op_parallelism_threads = 16
        config.intra_op_parallelism_threads = 16
        if branch.get("rewrite_config", True):
            config.graph_options.rewrite_options.constant_folding = (
                rewriter_config_pb2.RewriterConfig.OFF
            )
            config.graph_options.rewrite_options.arithmetic_optimization = (
                rewriter_config_pb2.RewriterConfig.OFF
            )
            config.graph_options.rewrite_options.remapping = (
                rewriter_config_pb2.RewriterConfig.AGGRESSIVE
            )
        run_options = tf.compat.v1.RunOptions(
            trace_level=tf.compat.v1.RunOptions.FULL_TRACE
        )
        all_step_stats = []
        branch_name = branch.get("name", "branch")
        filename = (
            f"{test_case.name}_performance_{branch_name}.timeline.json"
        )

        with tf.Session(config=config) as sess:
            if test_case.init_fn is not None:
                sess.run(test_case.init_fn())
            time.sleep(1)
            for _ in range(10):
                sess.run(res_node, feed_dict=feed_dict)
            for _ in range(test_case.num_iters):
                run_metadata = tf.compat.v1.RunMetadata()
                sess.run(
                    res_node,
                    feed_dict=feed_dict,
                    options=run_options,
                    run_metadata=run_metadata,
                )
                all_step_stats.append(run_metadata.step_stats)
        self.generate_timeline(all_step_stats, filename)
        self.check_branch_ops(filename, branch)
        op_names = branch.get("op_names", branch.get("op_name"))
        if op_names is not None:
            return self.extract_ops_total_dur(
                filename, op_names, test_case.num_iters
            )
        return self.extract_op_total_time(
            filename,
            branch["start_op_name"],
            branch["end_op_name"],
            test_case.num_iters,
        )

    def run_performance_test(self, test_case: TestCase):
        if test_case.num_iters == 0:
            return True

        print(f"Performance testing: {test_case.name} ...")

        optimize_percent = test_case.optimize_percent
        raw_inputs = test_case.input_fn(test_case.meta)
        baseline_branch = {
            "name": "unfused",
            "env": {"ANNC_FUSED_ALL": "0", **test_case.unfused_env},
            "start_op_name": test_case.start_op_name,
            "end_op_name": test_case.end_op_name,
            "expected_op_names": test_case.unfused_op_names,
            "forbidden_op_names": [test_case.fused_op_name],
        }
        fused_branch = {
            "name": "fused",
            "env": {"ANNC_FUSED_ALL": "1", **test_case.fused_env},
            "expected_op_names": [test_case.fused_op_name],
        }
        if test_case.fused_start_op_name:
            fused_branch.update({
                "start_op_name": test_case.fused_start_op_name,
                "end_op_name": test_case.fused_end_op_name,
            })
        else:
            fused_branch["op_name"] = test_case.fused_op_name
            baseline_branch["rewrite_config"] = False
        branches = [baseline_branch, fused_branch]

        times = {}
        for branch in branches:
            times[branch["name"]] = self._performance_branch_time(
                branch, raw_inputs, test_case
            )
        if any(value <= 0.0 for value in times.values()):
            print("⚠️ 获取平均运行时长失败, 无法进行比较")
            return False
        baseline_time = times["unfused"]
        optimized_time = times["fused"]
        real_percent = 100 * (
            (baseline_time - optimized_time) / optimized_time
        )
        print(
            "性能对比 fused vs unfused: "
            f"提升{real_percent:.2f}%, "
            f"基线耗时{baseline_time:.2f}us, "
            f"优化耗时{optimized_time:.2f}us"
        )
        if real_percent < optimize_percent:
            raise Exception(
                f"性能提升{real_percent:.2f}%, 低于预期{optimize_percent}%"
            )
        del raw_inputs
        gc.collect()
        return True

    def run_function_test(self, test_case: TestCase):
        print(f"Function testing: {test_case.name} ...")

        raw_inputs = test_case.input_fn(test_case.meta)

        def execute_variant(enable_embedding_fused, raw_inputs, test_case: TestCase):
            os.environ["ANNC_FUSED_ALL"] = str(enable_embedding_fused)
            self._apply_env(
                test_case.fused_env
                if enable_embedding_fused
                else test_case.unfused_env
            )
            tf.reset_default_graph()

            placeholders = []
            res_node, feed_dict = test_case.op_fn(raw_inputs, test_case.meta)
            config = tf.ConfigProto(
                inter_op_parallelism_threads=16,
                intra_op_parallelism_threads=16
            )

            config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
            config.graph_options.rewrite_options.arithmetic_optimization = rewriter_config_pb2.RewriterConfig.OFF
            config.graph_options.rewrite_options.remapping = rewriter_config_pb2.RewriterConfig.AGGRESSIVE
            with tf.Session(config=config) as sess:
                if test_case.init_fn is not None:
                    sess.run(test_case.init_fn())
                is_fused = False
                if enable_embedding_fused == 0:
                    result = sess.run(res_node, feed_dict=feed_dict)
                else:
                    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                    run_metadata = tf.compat.v1.RunMetadata()
                    filename = f"{test_case.name}_func_{enable_embedding_fused}.timeline.json"
                    result = sess.run(res_node, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                    self.generate_timeline([run_metadata.step_stats], filename)
                    is_fused = self.is_fused_op_exist(filename, test_case.fused_op_name)
                return result, is_fused
            return

        embedding_data, is_fused = execute_variant(1, raw_inputs, test_case)
        if is_fused != test_case.is_fused:
            print(f"⚠️ {test_case.fused_op_name}算子未做融合")
            raise Exception(f"⚠️ {test_case.fused_op_name}算子未做融合")
            return False
        no_embedding_data, _ = execute_variant(0, raw_inputs, test_case)
        if test_case.check_fn is None:
            is_correct = CheckFuncClass.check_fn_default(no_embedding_data, embedding_data, test_case.meta)
        else:
            is_correct = test_case.check_fn(no_embedding_data, embedding_data, test_case.meta)

        if not is_correct:
            print(f"⚠️ 误差较大")
            raise Exception(f"⚠️ 误差较大")
            return False
        print("功能测试通过")
        del raw_inputs
        gc.collect()
        return True
