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
from tensorflow.python.profiler.internal import _pywrap_profiler
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.profiler.profiler_v2 import ProfilerOptions
from tensorflow.core.profiler.protobuf.xplane_pb2 import XSpace
import numpy as np
import multiprocessing as mp
import pickle

tf.disable_eager_execution()


@dataclass
class TestCase:
    name: str
    op_fn: Callable
    input_fn: Callable
    check_fn: Optional[Callable] = None
    num_iters: int = 0
    optimize_percent: int = 0
    operator_name: str = ""
    meta: Dict = field(default_factory=dict)

class CheckFuncClass:
    def check_fn_float(A, B, meta):
        atol = meta.get('atol')
        rtol = meta.get('rtol')
        if rtol is None:
            rtol = 1e-3
        if atol is None:
            atol = 1e-4
        # ∣a−b∣≤ atol + rtol ×∣b∣
        return np.allclose(A, B, rtol=rtol, atol=atol)

    def check_fn_equal(A, B, meta):
        np.testing.assert_array_equal(A, B) # 如果失败抛出异常, 如果成功无返回值,
        return True

class UniversalOpBenchmark:
    def __init__(self, log_dir="bench_logs", intra_threads=16):
        self.log_dir = log_dir
        self.intra_threads = intra_threads
        self.results = []

    def _subprocess_worker(
        self,
        queue: mp.Queue,
        func: Callable,
        is_kdnn_enable: bool,
        raw_inputs: list,
        func_args: TestCase,
        temp_dir: str
    ) -> None:
        """
        子进程工作函数：创建临时目录，执行目标函数，将结果/异常放入队列
        """
        try:
            # 1. 自动创建临时目录（不存在则创建，存在则不报错）
            os.makedirs(temp_dir, exist_ok=True)
            
            # 2. 执行目标函数A
            result = func(is_kdnn_enable, raw_inputs, func_args)
            try:
                pickle.dumps(result)
            except Exception as e:
                raise Exception(f"返回结果无法序列化，无法传递给主进程：{str(e)}") from e
            # 3. 将执行结果放入队列（传递给主进程）
            try:
                queue.put(('success', result), block=True, timeout=10)
            except mp.Queue.Full:
                raise Exception("队列缓冲区已满，无法写入执行结果")
        
        except Exception as e:
            # 捕获所有异常，传递给主进程
            queue.put(('error', e))
            return

    def run_func_in_subprocess(
        self,
        func_A: Callable,
        is_kdnn_enable: bool,
        raw_inputs: list,
        func_args: TestCase,
        temp_dir: str = "./temp_workdir"
    ) -> str | np.ndarray:
        
        # 2. 创建进程间通信队列（用于传递子进程执行结果/异常）
        result_queue = mp.Queue()
        
        # 3. 构建子进程
        sub_process = mp.Process(
            target=self._subprocess_worker,
            args=(result_queue, func_A, is_kdnn_enable, raw_inputs, func_args, temp_dir)
        )
        
        # 4. 启动并等待子进程执行完成
        sub_process.start()
        
        # 5. 从队列中获取子进程执行结果
        process_status, result = result_queue.get()
        if process_status == 'error':
            raise Exception(f"子进程中函数执行失败：{str(result)}") from result

        sub_process.join()  # 阻塞主进程，等待子进程退出
        return result

    def _get_event_name_by_metadata_id(self, metadata_id, plane):
        if not hasattr(plane, 'event_metadata'):
            return f"unknown_operator_{metadata_id}"

        metadata=plane.event_metadata[metadata_id]
        if metadata.id == metadata_id:
            return metadata.name if (metadata.name and metadata.name.strip()) else f"unknown_operator_{metadata_id}"
        
        # 未匹配到返回默认名称
        return f"unknown_operator_{metadata_id}"

    def parse_xplane_file_optimized(self, dir_path):
        if not os.path.isdir(dir_path):
            print(f"dir is not exist: {dir_path}")
            return None

        xplane_files = []
        for file_name in os.listdir(dir_path):
            if file_name.endswith(".xplane.pb"):
                xplane_files.append(os.path.join(dir_path, file_name))

        if not xplane_files:
            print(f"file is not exist in {dir_path}")
            return None
        elif len(xplane_files) > 1:
            print(f"Found multiple.xplane.pb files: {xplane_files}; the first file will be used.")

        file_path = xplane_files[0]
        with open(file_path, 'rb') as f:
            content = f.read()
    
        xspace = XSpace()
        xspace.ParseFromString(content)
        if len(xspace.planes) > 0:
            return xspace

        print("parse_xplane_file_optimized failed")
        return None

    def get_average_wall_during(self, input_dir, operator_name, test_case: TestCase):
        # 1.提取数据
        xplane_data = self.parse_xplane_file_optimized(input_dir)
        if not xplane_data:
            return 0.0

        op_name_list = []
        # 2. 获取wall_duration
        wall_durations_us_list = []
        for plane in xplane_data.planes:
            if hasattr(plane, 'lines') and len(plane.lines) > 0:
                for line in plane.lines:
                    if hasattr(line, 'events') and len(line.events) > 0:
                        for event in line.events:
                            # 1. 获取事件时长（ps转换为us）
                            duration_ps = event.duration_ps if hasattr(event, 'duration_ps') else 0
                            if duration_ps <= 0:
                                continue  # 过滤无效时长事件
                            #转换成us
                            wall_duration_us = duration_ps / 1000000

                            # 2. 通过metadata_id获取算子名称
                            metadata_id = event.metadata_id if hasattr(event, 'metadata_id') else 0
                            op_name = self._get_event_name_by_metadata_id(metadata_id, plane)
                            if op_name not in op_name_list:
                                op_name_list.append(op_name)
                            if operator_name in op_name:
                                wall_durations_us_list.append(wall_duration_us)
        # print(f"解析到的算子名称列表:{op_name_list}") # 打印所有算子名称
        if test_case.num_iters != np.size(wall_durations_us_list):
            print(f"解析到有效数据{np.size(wall_durations_us_list)}条, 预期{test_case.num_iters}条")
            return 0.0

        return float(np.mean(wall_durations_us_list)), np.var(wall_durations_us_list, ddof=1)

    def parse_performance_data(self, kdnn_enable, raw_inputs, test_case: TestCase):
        print(f"Testing: {test_case.name} ...")
        os.environ['TF_ENABLE_KDNN_OPTS'] = str(kdnn_enable)
        
        placeholders = []
        feed_dict = {}
        
        for i, val in enumerate(raw_inputs):
            np_val = np.array(val)
            
            p = tf.placeholder(
                dtype=tf.as_dtype(np_val.dtype),
                shape=np_val.shape,
                name=f"input_{i}"
            )
            placeholders.append(p)
            feed_dict[p] = np_val

        res_node = test_case.op_fn(placeholders, test_case.meta)
        config = tf.ConfigProto(
            inter_op_parallelism_threads=16,
            intra_op_parallelism_threads=self.intra_threads
        )
        options = ProfilerOptions(
            host_tracer_level=2,
            device_tracer_level=1,
            python_tracer_level=0  # 建议保持 0 以减少对 Kernel 测量的干扰
        )
        with tf.Session(config=config) as sess:
            for _ in range(10):
                sess.run(res_node, feed_dict=feed_dict)

            profiler.start(self.log_dir, options)
            time.sleep(3)
            for _ in range(test_case.num_iters):
                sess.run(res_node, feed_dict=feed_dict)
            profiler.stop()

        try:
            kdnn_status = os.environ.get('TF_ENABLE_KDNN_OPTS', '1')
            tag = f"{test_case.name}_kdnn_{kdnn_status}"  # 结果为 kdnn_0 或 kdnn_1
            profile_root = os.path.join(self.log_dir, "plugins", "profile")
            timestamp_dirs = [d for d in os.listdir(profile_root) 
                             if os.path.isdir(os.path.join(profile_root, d)) and d.startswith("202")]
            
            if timestamp_dirs:
                src_dir = os.path.join(profile_root, timestamp_dirs[0])
                # 目标路径: bench_logs/MatMul_Test/plugins/profile/kdnn_1
                target_dir = os.path.join(profile_root, tag)
                
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                
                os.rename(src_dir, target_dir)
                print(f"✅ Profile 已固化至: {target_dir}")
                return target_dir

                # 可选：清理掉空的时间戳父目录或日志文件
                # (TensorFlow 有时会留下一些空的 event 文件，通常不影响解析)
        except Exception as e:
            print(f"⚠️ 路径固化失败: {e}")

    def run_performance_test(self, test_case: TestCase):
        if test_case.num_iters == 0:
            return True

        print(f"Performance testing: {test_case.name} ...")

        optimize_percent = test_case.optimize_percent
        operator_name = test_case.operator_name
        raw_inputs = test_case.input_fn()
        if not isinstance(raw_inputs, (list, tuple)):
            raw_inputs = [raw_inputs]

        no_kdnn_path = self.run_func_in_subprocess(self.parse_performance_data, 0, raw_inputs, test_case)
        no_kdnn_wall_during, _ = self.get_average_wall_during(no_kdnn_path, operator_name, test_case)
        kdnn_path = self.run_func_in_subprocess(self.parse_performance_data, 1, raw_inputs, test_case)
        kdnn_wall_during, _ = self.get_average_wall_during(kdnn_path, operator_name, test_case)
        if kdnn_wall_during <= 0.0 or no_kdnn_wall_during <= 0.0:
            print(f"⚠️ 获取平均运行时长失败, 无法进行比较")
            return False
        real_percent = 100 * (no_kdnn_wall_during/kdnn_wall_during - 1)
        if real_percent < optimize_percent:
            raise Exception(f"⚠️ 性能提升{real_percent:.2f}%, 低于预期{optimize_percent}%, KDNN开启状态下耗时{kdnn_wall_during:.2f}us, 关闭状态下耗时{no_kdnn_wall_during:.2f}us")
            return False

        print(f"性能测试通过, 性能提升{real_percent:.2f}%, KDNN开启状态下耗时{kdnn_wall_during:.2f}us, 关闭状态下耗时{no_kdnn_wall_during:.2f}us")
        return True

    def run_function_test(self, test_case: TestCase):
        print(f"Function testing: {test_case.name} ...")
        
        raw_inputs = test_case.input_fn()
        if not isinstance(raw_inputs, list):
            raw_inputs = [raw_inputs]

        def execute_variant(enable_kdnn, raw_inputs, test_case: TestCase):
            os.environ['TF_ENABLE_KDNN_OPTS'] = str(enable_kdnn)
            tf.reset_default_graph()

            placeholders = []
            feed_dict = {}

            for i, val in enumerate(raw_inputs):
                np_val = np.array(val)
                
                p = tf.placeholder(
                    dtype=tf.as_dtype(np_val.dtype),
                    shape=np_val.shape,
                    name=f"input_{i}"
                )
                placeholders.append(p)
                feed_dict[p] = np_val
            res_node = test_case.op_fn(placeholders, test_case.meta)
            config = tf.ConfigProto(
                inter_op_parallelism_threads=16,
                intra_op_parallelism_threads=self.intra_threads
            )
            options = ProfilerOptions(
                host_tracer_level=2,
                device_tracer_level=1,
                python_tracer_level=0  # 建议保持 0 以减少对 Kernel 测量的干扰
            )
            with tf.Session(config=config) as sess:
                return sess.run(res_node, feed_dict=feed_dict)
            return

        kdnn_data = self.run_func_in_subprocess(execute_variant, 1, raw_inputs, test_case)
        no_kdnn_data = self.run_func_in_subprocess(execute_variant, 0, raw_inputs, test_case)
        if test_case.check_fn is None:
            is_correct = CheckFuncClass.check_fn_float(no_kdnn_data, kdnn_data, test_case.meta)
        else:
            is_correct = test_case.check_fn(no_kdnn_data, kdnn_data, test_case.meta)

        if not is_correct:
            print(f"⚠️ 误差较大")
            raise Exception(f"⚠️ 误差较大")
            return False
        print("功能测试通过")
        return True
