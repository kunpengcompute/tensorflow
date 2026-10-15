import os
import time
import shutil
import glob
import json
import sys
import re
import struct
import tensorflow.compat.v1 as tf
from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any, Union, Optional
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
    is_fused: bool = True
    num_iters: int = 0
    optimize_percent: int = 0
    meta: Dict = field(default_factory=dict)

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

    def extract_op_dur(self, timeline_file, op_name, times=1):
        """从 timeline JSON 文件中提取指定算子(fusedOp)的平均耗时（μs）"""
        with open(f"timeline/{timeline_file}", "r") as f:
            trace_events_list = json.load(f)  # timeline.json的格式
        durations_list = []
        for trace_events in trace_events_list:
            durations = [e["dur"] for e in trace_events["traceEvents"] if e.get("name") == op_name and "dur" in e]
            durations_list.append(durations[0])
        if len(durations_list) != times:
            raise ValueError(f"Expected {times} durations for {op_name}, but got {len(durations)}")
        return np.mean(durations_list)


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

    def parse_performance_time(self, embedding_fused_enable, raw_inputs, test_case: TestCase):
        print(f"Testing: {test_case.name} ...")
        os.environ["ANNC_FUSED_ALL"] = str(embedding_fused_enable)

        res_node, feed_dict = test_case.op_fn(raw_inputs, test_case.meta)
        config = tf.compat.v1.ConfigProto()
        config.inter_op_parallelism_threads = 16
        config.intra_op_parallelism_threads = 16

        run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()
        all_step_stats = []
        filename = f"{test_case.name}_performance_{embedding_fused_enable}.timeline.json"
        if embedding_fused_enable != 0:
            # 开启融合条件下关闭部分开关防止中间进行部分非预期优化
            config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
            config.graph_options.rewrite_options.arithmetic_optimization = rewriter_config_pb2.RewriterConfig.OFF
            config.graph_options.rewrite_options.remapping = rewriter_config_pb2.RewriterConfig.AGGRESSIVE
        with tf.Session(config=config) as sess:
            time.sleep(1)
            for _ in range(10):
                sess.run(res_node, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            for _ in range(test_case.num_iters):
                sess.run(res_node, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                all_step_stats.append(run_metadata.step_stats)
            self.generate_timeline(all_step_stats, filename)
            if embedding_fused_enable != 0:
                return self.extract_op_dur(filename, test_case.fused_op_name, test_case.num_iters)
            else:
                return self.extract_op_total_time(filename, test_case.start_op_name, test_case.end_op_name, test_case.num_iters)
    
    def run_performance_test(self, test_case: TestCase):
        if test_case.num_iters == 0:
            return True

        print(f"Performance testing: {test_case.name} ...")

        optimize_percent = test_case.optimize_percent
        operator_name = test_case.fused_op_name
        raw_inputs = test_case.input_fn(test_case.meta)

        no_fused_time = self.parse_performance_time(0, raw_inputs, test_case)
        fused_time = self.parse_performance_time(1, raw_inputs, test_case)
        if fused_time <= 0.0 or no_fused_time <= 0.0:
            print(f"⚠️ 获取平均运行时长失败, 无法进行比较")
            return False
        real_percent = 100 * (no_fused_time/fused_time - 1)
        if real_percent < optimize_percent:
            raise Exception(f"⚠️ 性能提升{real_percent:.2f}%, 低于预期{optimize_percent}%, embedding算子融合开启状态下耗时{fused_time:.2f}us, 关闭状态下耗时{no_fused_time:.2f}us")
            return False

        print(f"性能测试通过, 性能提升{real_percent:.2f}%, embedding算子融合开启状态下耗时{fused_time:.2f}us, 关闭状态下耗时{no_fused_time:.2f}us")
        return True

    def run_function_test(self, test_case: TestCase):
        print(f"Function testing: {test_case.name} ...")

        raw_inputs = test_case.input_fn(test_case.meta)

        def execute_variant(enable_embedding_fused, raw_inputs, test_case: TestCase):
            os.environ["ANNC_FUSED_ALL"] = str(enable_embedding_fused)
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
        return True
